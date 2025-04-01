#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional, Any
import numpy as np
import pandas as pd
from numba import jit, vectorize, njit, prange

from .indicator import Indicator
from .alpha_ma import AlphaMA
from .alpha_atr import AlphaATR
from .cycle_efficiency_ratio import CycleEfficiencyRatio


@dataclass
class AlphaBandResult:
    """AlphaBandの計算結果"""
    middle: np.ndarray        # 中心線（AlphaMA）
    upper: np.ndarray         # 上限バンド
    lower: np.ndarray         # 下限バンド
    cer: np.ndarray           # Cycle Efficiency Ratio
    dynamic_multiplier: np.ndarray  # 動的ATR乗数
    alpha_atr: np.ndarray     # AlphaATR値


@vectorize(['float64(float64, float64, float64)'], nopython=True, fastmath=True)
def calculate_dynamic_multiplier_vec(cer: float, max_mult: float, min_mult: float) -> float:
    """
    サイクル効率比に基づいて動的なATR乗数を計算する（ベクトル化版）
    
    Args:
        cer: サイクル効率比の値
        max_mult: 最大乗数
        min_mult: 最小乗数
    
    Returns:
        動的な乗数の値
    """
    # CERが高い（トレンドが強い）ほど乗数は小さく、
    # CERが低い（トレンドが弱い）ほど乗数は大きくなる
    if np.isnan(cer):
        return np.nan
    cer_abs = abs(cer)
    return max_mult - cer_abs * (max_mult - min_mult)


@njit(fastmath=True)
def calculate_dynamic_multiplier(cer: np.ndarray, max_mult: float, min_mult: float) -> np.ndarray:
    """
    サイクル効率比に基づいて動的なATR乗数を計算する（高速化版）
    
    Args:
        cer: サイクル効率比の配列
        max_mult: 最大乗数
        min_mult: 最小乗数
    
    Returns:
        動的な乗数の配列
    """
    # CERが高い（トレンドが強い）ほど乗数は小さく、
    # CERが低い（トレンドが弱い）ほど乗数は大きくなる
    cer_abs = np.abs(cer)
    multipliers = max_mult - cer_abs * (max_mult - min_mult)
    return multipliers


@njit(fastmath=True, parallel=True)
def calculate_alpha_band(
    alpha_ma: np.ndarray,
    alpha_atr: np.ndarray,
    dynamic_multiplier: np.ndarray,
    use_percent: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    アルファバンドを計算する（パラレル高速化版）
    
    Args:
        alpha_ma: AlphaMA値の配列
        alpha_atr: AlphaATR値の配列（金額ベースまたはパーセントベース）
        dynamic_multiplier: 動的乗数の配列
        use_percent: パーセントベースATRを使用するかどうか
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            (中心線, 上限バンド, 下限バンド)の配列
    """
    length = len(alpha_ma)
    middle = np.copy(alpha_ma)
    upper = np.full_like(alpha_ma, np.nan, dtype=np.float64)
    lower = np.full_like(alpha_ma, np.nan, dtype=np.float64)
    
    # 有効なデータ判別用マスク
    valid_mask = ~(np.isnan(alpha_ma) | np.isnan(alpha_atr) | np.isnan(dynamic_multiplier))
    
    # バンド幅計算用の一時配列
    band_width = np.zeros_like(alpha_ma, dtype=np.float64)
    
    # 並列計算のためバンド幅を先に計算
    for i in prange(length):
        if valid_mask[i]:
            if use_percent:
                # パーセントベースのATRを使用する場合、価格に対する比率
                band_width[i] = alpha_ma[i] * alpha_atr[i] * dynamic_multiplier[i]
            else:
                # 金額ベースのATRを使用する場合
                band_width[i] = alpha_atr[i] * dynamic_multiplier[i]
    
    # 上下バンドを計算
    for i in prange(length):
        if valid_mask[i]:
            upper[i] = middle[i] + band_width[i]
            lower[i] = middle[i] - band_width[i]
    
    return middle, upper, lower


class AlphaBand(Indicator):
    """
    アルファバンド（Alpha Band）インジケーター
    
    特徴:
    - 中心線にAlphaMA（動的適応型移動平均）を使用
    - バンド幅の計算にAlphaATR（動的適応型ATR）を使用
    - ATR乗数がサイクル効率比（CER）に基づいて動的に調整
    - RSXの3段階平滑化アルゴリズムをAlphaATRで使用
    
    市場状態に応じた最適な挙動:
    - トレンド強い（CER高い）:
      - 狭いバンド幅（小さい乗数）でトレンドをタイトに追従
    - トレンド弱い（CER低い）:
      - 広いバンド幅（大きい乗数）でレンジ相場の振れ幅を捉える
    """
    
    def __init__(
        self,
        cycle_detector_type: str = 'hody_dc',
        lp_period: int = 5,
        hp_period: int = 144,
        cycle_part: float = 0.5,
        max_kama_period: int = 55,
        min_kama_period: int = 8,
        max_atr_period: int = 55,
        min_atr_period: int = 8,
        max_multiplier: float = 3.0,  # ALMA版ATR用に調整
        min_multiplier: float = 1.5,  # ALMA版ATR用に調整
        smoother_type: str = 'alma'   # 平滑化アルゴリズム（'alma'または'hyper'）
    ):
        """
        コンストラクタ
        
        Args:
            cycle_detector_type: サイクル検出器の種類
                'dudi_dc' - 二重微分
                'hody_dc' - ホモダイン判別機（デフォルト）
                'phac_dc' - 位相累積
                'dudi_dce' - 拡張二重微分
                'hody_dce' - 拡張ホモダイン判別機
                'phac_dce' - 拡張位相累積
            lp_period: ローパスフィルターの期間（デフォルト: 5）
            hp_period: ハイパスフィルターの期間（デフォルト: 144）
            cycle_part: サイクル部分の倍率（デフォルト: 0.5）
            max_kama_period: AlphaMAのKAMA最大期間（デフォルト: 55）
            min_kama_period: AlphaMAのKAMA最小期間（デフォルト: 8）
            max_atr_period: AlphaATRの最大期間（デフォルト: 55）
            min_atr_period: AlphaATRの最小期間（デフォルト: 8）
            max_multiplier: ATR乗数の最大値（デフォルト: 3.0）
            min_multiplier: ATR乗数の最小値（デフォルト: 1.5）
            smoother_type: 平滑化アルゴリズムのタイプ（デフォルト: 'alma'）
                'alma' - ALMA（Arnaud Legoux Moving Average）
                'hyper' - ハイパースムーサー（3段階平滑化）
        """
        super().__init__(
            f"AlphaBand({cycle_detector_type}, {max_multiplier}, {min_multiplier}, {smoother_type})"
        )
        self.cycle_detector_type = cycle_detector_type
        self.lp_period = lp_period
        self.hp_period = hp_period
        self.cycle_part = cycle_part
        self.max_kama_period = max_kama_period
        self.min_kama_period = min_kama_period
        self.max_atr_period = max_atr_period
        self.min_atr_period = min_atr_period
        self.max_multiplier = max_multiplier
        self.min_multiplier = min_multiplier
        self.smoother_type = smoother_type
        
        # サイクル効率比（CER）のインスタンス化
        self.cycle_er = CycleEfficiencyRatio(
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part
        )
        
        # AlphaMAとAlphaATRのインスタンス化
        self.alpha_ma = AlphaMA(
            max_kama_period=max_kama_period,
            min_kama_period=min_kama_period,
            src_type = 'hlc3'
        )
        
        self.alpha_atr = AlphaATR(
            max_atr_period=max_atr_period,
            min_atr_period=min_atr_period,
            smoother_type=smoother_type  # 平滑化アルゴリズムの選択
        )
        
        self._result = None
        self._data_hash = None  # データキャッシュ用ハッシュ
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算してキャッシュに使用する"""
        if isinstance(data, pd.DataFrame):
            # DataFrameの場合は必要なカラムのみハッシュする
            cols = ['high', 'low', 'close']
            data_hash = hash(tuple(map(tuple, (data[col].values for col in cols if col in data.columns))))
        else:
            # NumPy配列の場合は全体をハッシュする
            data_hash = hash(tuple(map(tuple, data)))
        
        # パラメータ値を含めることで、同じデータでもパラメータが異なる場合に再計算する
        param_str = f"{self.cycle_detector_type}_{self.lp_period}_{self.hp_period}_{self.cycle_part}_{self.max_kama_period}_{self.min_kama_period}_{self.max_atr_period}_{self.min_atr_period}_{self.max_multiplier}_{self.min_multiplier}_{self.smoother_type}"
        return f"{data_hash}_{param_str}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        アルファバンドを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'high', 'low', 'close'カラムが必要
        
        Returns:
            中心線の値（AlphaMA）
        """
        try:
            # キャッシュチェック - 同じデータの場合は計算をスキップ
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._result is not None:
                return self._result.middle
            
            self._data_hash = data_hash  # 新しいハッシュを保存
            
            # データの検証と変換（必要最小限の処理）
            if isinstance(data, pd.DataFrame):
                if not all(col in data.columns for col in ['high', 'low', 'close']):
                    raise ValueError("DataFrameには'high', 'low', 'close'カラムが必要です")
            elif data.ndim != 2 or data.shape[1] < 4:
                raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")
            
            # サイクル効率比（CER）の計算
            cer = self.cycle_er.calculate(data)
            if cer is None or len(cer) == 0:
                raise ValueError("サイクル効率比（CER）の計算に失敗しました")
            
            # AlphaMAの計算 (サイクル効率比を使用)
            alpha_ma_values = self.alpha_ma.calculate(data, external_er=cer)
            if alpha_ma_values is None:
                raise ValueError("AlphaMAの計算に失敗しました")
            
            # AlphaATRの計算 (サイクル効率比を使用)
            # 注: calculate()メソッドは%ベースのATRを返します
            alpha_atr_values = self.alpha_atr.calculate(data, external_er=cer)
            if alpha_atr_values is None:
                raise ValueError("AlphaATRの計算に失敗しました")
            
            # 金額ベースのAlphaATRを取得（パーセントベースではなく）
            # 重要: バンド計算には金額ベース(絶対値)のATRを使用する必要があります
            alpha_atr_absolute = self.alpha_atr.get_absolute_atr()
            
            # 動的ATR乗数の計算（ベクトル化関数を使用）
            dynamic_multiplier = calculate_dynamic_multiplier_vec(
                cer,
                self.max_multiplier,
                self.min_multiplier
            )
            
            # アルファバンドの計算（パラレル高速版）
            # 金額ベースのATRを使用
            middle, upper, lower = calculate_alpha_band(
                alpha_ma_values,
                alpha_atr_absolute,  # 金額ベースのATRを使用
                dynamic_multiplier,
                use_percent=False    # 金額ベース計算を使用
            )
            
            # 結果の保存（参照コピーを避けるためnp.copyを使用）
            self._result = AlphaBandResult(
                middle=np.copy(middle),
                upper=np.copy(upper),
                lower=np.copy(lower),
                cer=np.copy(cer),
                dynamic_multiplier=np.copy(dynamic_multiplier),
                alpha_atr=np.copy(alpha_atr_absolute)  # 金額ベースのATRを保存
            )
            
            # 中心線を値として保存
            self._values = middle
            return middle
            
        except Exception as e:
            self.logger.error(f"AlphaBand計算中にエラー: {str(e)}")
            
            # エラー時は前回の結果を維持する（nullではなく）
            if self._result is None:
                # 初回エラー時は空の結果を作成
                empty_array = np.array([])
                self._result = AlphaBandResult(
                    middle=empty_array,
                    upper=empty_array,
                    lower=empty_array,
                    cer=empty_array,
                    dynamic_multiplier=empty_array,
                    alpha_atr=empty_array
                )
                self._values = empty_array
            
            return self._values
    
    def get_bands(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        アルファバンドのバンド値を取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                (中心線, 上限バンド, 下限バンド)の値
        """
        if self._result is None:
            # 結果がない場合は空の配列を返す
            empty = np.array([])
            return empty, empty, empty
        return self._result.middle, self._result.upper, self._result.lower
    
    def get_cycle_er(self) -> np.ndarray:
        """
        サイクル効率比（CER）の値を取得する
        
        Returns:
            np.ndarray: サイクル効率比の値
        """
        if self._result is None:
            # 結果がない場合は空の配列を返す
            return np.array([])
        return self._result.cer
    
    def get_dynamic_multiplier(self) -> np.ndarray:
        """
        動的ATR乗数の値を取得する
        
        Returns:
            np.ndarray: 動的ATR乗数の値
        """
        if self._result is None:
            # 結果がない場合は空の配列を返す
            return np.array([])
        return self._result.dynamic_multiplier
    
    def get_alpha_atr(self) -> np.ndarray:
        """
        AlphaATR値を取得する
        
        Returns:
            np.ndarray: AlphaATR値
        """
        if self._result is None:
            # 結果がない場合は空の配列を返す
            return np.array([])
        return self._result.alpha_atr
    
    def reset(self) -> None:
        """
        インジケーターの状態をリセットする
        """
        self._result = None
        self._data_hash = None
        self._values = None
        self.cycle_er.reset() if hasattr(self.cycle_er, 'reset') else None
        self.alpha_ma.reset() if hasattr(self.alpha_ma, 'reset') else None
        self.alpha_atr.reset() if hasattr(self.alpha_atr, 'reset') else None 