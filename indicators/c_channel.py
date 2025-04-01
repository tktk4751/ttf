#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional, Any
import numpy as np
import pandas as pd
from numba import jit, vectorize, njit, prange

from .indicator import Indicator
from .c_ma import CMA
from .c_atr import CATR
from .cycle_efficiency_ratio import CycleEfficiencyRatio


@dataclass
class CChannelResult:
    """Cチャネルの計算結果"""
    middle: np.ndarray        # 中心線（CMA）
    upper: np.ndarray         # 上限バンド
    lower: np.ndarray         # 下限バンド
    cer: np.ndarray           # Cycle Efficiency Ratio
    dynamic_multiplier: np.ndarray  # 動的ATR乗数
    c_atr: np.ndarray         # CATR値
    max_mult_values: np.ndarray  # 動的に計算されたmax_multiplier値
    min_mult_values: np.ndarray  # 動的に計算されたmin_multiplier値


@vectorize(['float64(float64, float64, float64)'], nopython=True, fastmath=True, cache=True)
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


@vectorize(['float64(float64, float64, float64)'], nopython=True, fastmath=True, cache=True)
def calculate_dynamic_max_multiplier(cer: float, max_max_mult: float, min_max_mult: float) -> float:
    """
    サイクル効率比に基づいて動的な最大ATR乗数を計算する（ベクトル化版）
    
    Args:
        cer: サイクル効率比の値
        max_max_mult: 最大乗数の最大値（例：8.0）
        min_max_mult: 最大乗数の最小値（例：3.0）
    
    Returns:
        動的な最大乗数の値
    """
    if np.isnan(cer):
        return np.nan
    cer_abs = abs(cer)
    # CERが低い（トレンドが弱い）ほど最大乗数は大きく、
    # CERが高い（トレンドが強い）ほど最大乗数は小さくなる
    return max_max_mult - cer_abs * (max_max_mult - min_max_mult)


@vectorize(['float64(float64, float64, float64)'], nopython=True, fastmath=True, cache=True)
def calculate_dynamic_min_multiplier(cer: float, max_min_mult: float, min_min_mult: float) -> float:
    """
    サイクル効率比に基づいて動的な最小ATR乗数を計算する（ベクトル化版）
    
    Args:
        cer: サイクル効率比の値
        max_min_mult: 最小乗数の最大値（例：1.5）
        min_min_mult: 最小乗数の最小値（例：0.5）
    
    Returns:
        動的な最小乗数の値
    """
    if np.isnan(cer):
        return np.nan
    cer_abs = abs(cer)
    # CERが低い（トレンドが弱い）ほど最小乗数は大きく、
    # CERが高い（トレンドが強い）ほど最小乗数は小さくなる
    return max_min_mult - cer_abs * (max_min_mult - min_min_mult)


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


@njit(fastmath=True, parallel=True, cache=True)
def calculate_c_channel(
    c_ma: np.ndarray,
    c_atr: np.ndarray,
    dynamic_multiplier: np.ndarray,
    use_percent: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cチャネルを計算する（パラレル高速化版）
    
    Args:
        c_ma: CMA値の配列
        c_atr: CATR値の配列（金額ベースまたはパーセントベース）
        dynamic_multiplier: 動的乗数の配列
        use_percent: パーセントベースATRを使用するかどうか
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            (中心線, 上限バンド, 下限バンド)の配列
    """
    length = len(c_ma)
    middle = np.copy(c_ma)
    upper = np.full_like(c_ma, np.nan, dtype=np.float64)
    lower = np.full_like(c_ma, np.nan, dtype=np.float64)
    
    # 有効なデータ判別用マスク
    valid_mask = ~(np.isnan(c_ma) | np.isnan(c_atr) | np.isnan(dynamic_multiplier))
    
    # バンド幅計算用の一時配列 - パラレル計算の準備
    band_width = np.zeros_like(c_ma, dtype=np.float64)
    
    # 並列計算で一度にバンド幅を計算
    for i in prange(length):
        if valid_mask[i]:
            if use_percent:
                # パーセントベースのATRを使用する場合、価格に対する比率
                band_width[i] = c_ma[i] * c_atr[i] * dynamic_multiplier[i]
            else:
                # 金額ベースのATRを使用する場合
                band_width[i] = c_atr[i] * dynamic_multiplier[i]
    
    # 一度にバンドを計算（並列処理）
    for i in prange(length):
        if valid_mask[i]:
            upper[i] = middle[i] + band_width[i]
            lower[i] = middle[i] - band_width[i]
    
    return middle, upper, lower


class CChannel(Indicator):
    """
    CChannel (Cycle Channel) インジケーター
    
    特徴:
    - CMA（Cycle Moving Average）を中心線として使用
    - CATR（Cycle Average True Range）をボラティリティとして使用
    - サイクル効率比（CER）に基づく動的乗数でチャネル幅を調整
    - トレンドの強さに応じて適応的にバンド幅を変化
    
    使用方法:
    - トレンドフォロー戦略におけるエントリー・エグジットポイントの決定
    - サポート・レジスタンスレベルの特定
    - 価格アクションの分析とボラティリティの視覚化
    - 相場の状態（トレンド/レンジ）の判断
    """
    
    def __init__(
        self,
        # 基本パラメータ
        detector_type: str = 'phac_e',
        cer_detector_type: str = None,  # CER用の検出器タイプ
        lp_period: int = 5,
        hp_period: int = 55,
        cycle_part: float = 0.7,
        max_multiplier: float = 7.0,  # 固定乗数を使用する場合
        min_multiplier: float = 1.0,  # 固定乗数を使用する場合
        # 動的乗数の範囲パラメータ（固定乗数の代わりに動的乗数を使用する場合）
        max_max_multiplier: float = 8.0,    # 最大乗数の最大値
        min_max_multiplier: float = 6.0,    # 最大乗数の最小値
        max_min_multiplier: float = 1.5,    # 最小乗数の最大値
        min_min_multiplier: float = 0.5,    # 最小乗数の最小値
        smoother_type: str = 'alma',  # 'alma' または 'hyper'
        src_type: str = 'hlc3',       # 'open', 'high', 'low', 'close', 'hl2', 'hlc3', 'ohlc4'
        
        # CMA用パラメータ
        cma_detector_type: str = 'hody_e',
        cma_cycle_part: float = 0.5,
        cma_lp_period: int = 5,
        cma_hp_period: int = 55,
        cma_max_cycle: int = 144,
        cma_min_cycle: int = 5,
        cma_max_output: int = 62,
        cma_min_output: int = 13,
        cma_fast_period: int = 2,
        cma_slow_period: int = 30,
        cma_src_type: str = 'hlc3',
        
        # CATR用パラメータ
        catr_detector_type: str = 'hody',
        catr_cycle_part: float = 0.5,
        catr_lp_period: int = 5,
        catr_hp_period: int = 55,
        catr_max_cycle: int = 55,
        catr_min_cycle: int = 5,
        catr_max_output: int = 34,
        catr_min_output: int = 5,
        catr_smoother_type: str = 'alma'
    ):
        """
        コンストラクタ
        
        Args:
            detector_type: 検出器タイプ
                - 'hody': ホモダイン判別機
                - 'phac': 位相累積
                - 'dudi': 二重微分
                - 'dudi_e': 拡張二重微分
                - 'hody_e': 拡張ホモダイン判別機
                - 'phac_e': 拡張位相累積
            cer_detector_type: CER用の検出器タイプ（Noneの場合はdetector_typeと同じ）
            lp_period: ローパスフィルター期間
            hp_period: ハイパスフィルター期間
            cycle_part: サイクル部分の倍率
            max_multiplier: 最大乗数（固定）
            min_multiplier: 最小乗数（固定）
            max_max_multiplier: 動的最大乗数の最大値
            min_max_multiplier: 動的最大乗数の最小値
            max_min_multiplier: 動的最小乗数の最大値
            min_min_multiplier: 動的最小乗数の最小値
            smoother_type: 平滑化タイプ ('alma' または 'hyper')
            src_type: ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4')
            
            # CMAパラメータ
            cma_detector_type: CMA用検出器タイプ
            cma_cycle_part: CMA用サイクル部分の倍率
            cma_lp_period: CMA用ローパスフィルター期間
            cma_hp_period: CMA用ハイパスフィルター期間
            cma_max_cycle: CMA用最大サイクル期間
            cma_min_cycle: CMA用最小サイクル期間
            cma_max_output: CMA用最大出力値
            cma_min_output: CMA用最小出力値
            cma_fast_period: CMA用速い移動平均の期間
            cma_slow_period: CMA用遅い移動平均の期間
            cma_src_type: CMA用ソースタイプ
            
            # CATRパラメータ
            catr_detector_type: CATR用検出器タイプ
            catr_cycle_part: CATR用サイクル部分の倍率
            catr_lp_period: CATR用ローパスフィルター期間
            catr_hp_period: CATR用ハイパスフィルター期間
            catr_max_cycle: CATR用最大サイクル期間
            catr_min_cycle: CATR用最小サイクル期間
            catr_max_output: CATR用最大出力値
            catr_min_output: CATR用最小出力値
            catr_smoother_type: CATR用平滑化タイプ
        """
        super().__init__(
            f"CChannel({detector_type}, {max_multiplier}, {min_multiplier}, {smoother_type})"
        )
        
        # CERの検出器タイプが指定されていない場合、detector_typeと同じ値を使用
        if cer_detector_type is None:
            cer_detector_type = detector_type
            
        self.detector_type = detector_type
        self.cer_detector_type = cer_detector_type
        self.lp_period = lp_period
        self.hp_period = hp_period
        self.cycle_part = cycle_part
        self.max_multiplier = max_multiplier
        self.min_multiplier = min_multiplier
        self.max_max_multiplier = max_max_multiplier
        self.min_max_multiplier = min_max_multiplier
        self.max_min_multiplier = max_min_multiplier
        self.min_min_multiplier = min_min_multiplier
        self.smoother_type = smoother_type
        self.src_type = src_type
        
        # CMA用パラメータ
        self.cma_detector_type = cma_detector_type
        self.cma_cycle_part = cma_cycle_part
        self.cma_lp_period = cma_lp_period
        self.cma_hp_period = cma_hp_period
        self.cma_max_cycle = cma_max_cycle
        self.cma_min_cycle = cma_min_cycle
        self.cma_max_output = cma_max_output
        self.cma_min_output = cma_min_output
        self.cma_fast_period = cma_fast_period
        self.cma_slow_period = cma_slow_period
        self.cma_src_type = cma_src_type
        
        # CATR用パラメータ
        self.catr_detector_type = catr_detector_type
        self.catr_cycle_part = catr_cycle_part
        self.catr_lp_period = catr_lp_period
        self.catr_hp_period = catr_hp_period
        self.catr_max_cycle = catr_max_cycle
        self.catr_min_cycle = catr_min_cycle
        self.catr_max_output = catr_max_output
        self.catr_min_output = catr_min_output
        self.catr_smoother_type = catr_smoother_type
        
        # サイクル効率比（CER）計算インスタンスの初期化
        self.cycle_er = CycleEfficiencyRatio(
            detector_type=cer_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            max_cycle=cma_max_cycle,  # CMAの最大サイクルを使用
            min_cycle=cma_min_cycle,  # CMAの最小サイクルを使用
            max_output=cma_max_output,  # CMAの最大出力を使用
            min_output=cma_min_output,  # CMAの最小出力を使用
            src_type=src_type
        )
        
        # CMAインスタンスの初期化
        self.cma = CMA(
            detector_type=cma_detector_type,
            cycle_part=cma_cycle_part,
            lp_period=cma_lp_period,
            hp_period=cma_hp_period,
            max_cycle=cma_max_cycle,
            min_cycle=cma_min_cycle,
            max_output=cma_max_output,
            min_output=cma_min_output,
            fast_period=cma_fast_period,
            slow_period=cma_slow_period,
            src_type=cma_src_type
        )
        
        # CATRインスタンスの初期化
        self.catr = CATR(
            detector_type=catr_detector_type,
            cycle_part=catr_cycle_part,
            lp_period=catr_lp_period,
            hp_period=catr_hp_period,
            max_cycle=catr_max_cycle,
            min_cycle=catr_min_cycle,
            max_output=catr_max_output,
            min_output=catr_min_output,
            smoother_type=catr_smoother_type
        )
        
        # キャッシュの初期化
        self._result = None
        self._data_hash = None
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算してキャッシュに使用する"""
        if isinstance(data, pd.DataFrame):
            # DataFrameの場合は必要なカラムのみハッシュする（高速化）
            cols = ['high', 'low', 'close']
            # NumPyでの高速ハッシュ計算
            data_values = np.vstack([data[col].values for col in cols if col in data.columns])
            data_hash = hash(data_values.tobytes())
        else:
            # NumPy配列の場合は全体をハッシュする（高速化）
            data_hash = hash(data.tobytes() if isinstance(data, np.ndarray) else str(data))
        
        # パラメータ値を含めることで、同じデータでもパラメータが異なる場合に再計算する
        param_str = (
            f"{self.detector_type}_{self.cer_detector_type}_{self.lp_period}_{self.hp_period}_{self.cycle_part}_"
            f"{self.max_multiplier}_{self.min_multiplier}_{self.smoother_type}_{self.src_type}_"
            f"{self.max_max_multiplier}_{self.min_max_multiplier}_{self.max_min_multiplier}_{self.min_min_multiplier}_"
            f"cma_{self.cma_detector_type}_{self.cma_cycle_part}_{self.cma_lp_period}_{self.cma_hp_period}_"
            f"{self.cma_max_cycle}_{self.cma_min_cycle}_{self.cma_max_output}_{self.cma_min_output}_"
            f"{self.cma_fast_period}_{self.cma_slow_period}_{self.cma_src_type}_"
            f"catr_{self.catr_detector_type}_{self.catr_cycle_part}_{self.catr_lp_period}_{self.catr_hp_period}_"
            f"{self.catr_max_cycle}_{self.catr_min_cycle}_{self.catr_max_output}_{self.catr_min_output}_"
            f"{self.catr_smoother_type}"
        )
        return f"{data_hash}_{param_str}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Cチャネルを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'high', 'low', 'close'カラムが必要
        
        Returns:
            中心線の値（CMA）
        """
        try:
            # キャッシュチェック - 同じデータの場合は計算をスキップ
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._result is not None:
                return self._result.middle
            
            # 新しいハッシュを保存
            self._data_hash = data_hash
            
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
            
            # CMAの計算（サイクル効率比を使用）
            c_ma_values = self.cma.calculate(data, external_er=cer)
            if c_ma_values is None:
                raise ValueError("CMAの計算に失敗しました")
            
            # CATRの計算（サイクル効率比を使用）
            # 注: calculate()メソッドは%ベースのATRを返します
            c_atr_values = self.catr.calculate(data, external_er=cer)
            if c_atr_values is None:
                raise ValueError("CATRの計算に失敗しました")
            
            # 金額ベースのCATRを取得（パーセントベースではなく）
            # 重要: バンド計算には金額ベース(絶対値)のATRを使用する必要があります
            c_atr_absolute = self.catr.get_absolute_atr()
            
            # データをNumPy配列に変換して計算を高速化
            cer_np = np.asarray(cer, dtype=np.float64)
            
            # 動的な最大・最小乗数の計算
            max_mult_values = calculate_dynamic_max_multiplier(
                cer_np,
                self.max_max_multiplier,
                self.min_max_multiplier
            )
            
            min_mult_values = calculate_dynamic_min_multiplier(
                cer_np,
                self.max_min_multiplier,
                self.min_min_multiplier
            )
            
            # 動的ATR乗数の計算（ベクトル化関数を使用）- 高速化
            dynamic_multiplier = calculate_dynamic_multiplier_vec(
                cer_np,
                max_mult_values,
                min_mult_values
            )
            
            # Cチャネルの計算（パラレル高速版）
            # 金額ベースのATRを使用
            middle, upper, lower = calculate_c_channel(
                np.asarray(c_ma_values, dtype=np.float64),
                np.asarray(c_atr_absolute, dtype=np.float64),
                np.asarray(dynamic_multiplier, dtype=np.float64),
                use_percent=False    # 金額ベース計算を使用
            )
            
            # 結果の保存（参照コピーを避けるためnp.copyを使用）
            result = CChannelResult(
                middle=np.copy(middle),
                upper=np.copy(upper),
                lower=np.copy(lower),
                cer=np.copy(cer),
                dynamic_multiplier=np.copy(dynamic_multiplier),
                c_atr=np.copy(c_atr_absolute),  # 金額ベースのATRを保存
                max_mult_values=np.copy(max_mult_values),  # 動的最大乗数を保存
                min_mult_values=np.copy(min_mult_values)   # 動的最小乗数を保存
            )
            
            # 結果をクラス変数に保存
            self._result = result
            
            # 中心線を値として保存
            return middle
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"CChannel計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は空の配列を返す
            return np.array([])
    
    def get_bands(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Cチャネルのバンド値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                (中心線, 上限バンド, 下限バンド)の値
        """
        if data is not None:
            # データが指定された場合は再計算
            self.calculate(data)
            
        # 結果確認
        if not hasattr(self, '_result') or self._result is None:
            # 結果がない場合は空の配列を返す
            empty = np.array([])
            return empty, empty, empty
            
        return self._result.middle, self._result.upper, self._result.lower
    
    def get_cycle_er(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        サイクル効率比（CER）の値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: サイクル効率比の値
        """
        if data is not None:
            # データが指定された場合は再計算
            self.calculate(data)
            
        # 結果確認
        if not hasattr(self, '_result') or self._result is None:
            # 結果がない場合は空の配列を返す
            return np.array([])
            
        return self._result.cer
    
    def get_dynamic_multiplier(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        動的ATR乗数の値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 動的ATR乗数の値
        """
        if data is not None:
            # データが指定された場合は再計算
            self.calculate(data)
            
        # 結果確認
        if not hasattr(self, '_result') or self._result is None:
            # 結果がない場合は空の配列を返す
            return np.array([])
            
        return self._result.dynamic_multiplier
    
    def get_c_atr(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        CATR値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: CATR値
        """
        if data is not None:
            # データが指定された場合は再計算
            self.calculate(data)
            
        # 結果確認
        if not hasattr(self, '_result') or self._result is None:
            # 結果がない場合は空の配列を返す
            return np.array([])
            
        return self._result.c_atr
    
    def get_dynamic_max_multiplier(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        動的な最大ATR乗数の値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 動的最大ATR乗数の値
        """
        if data is not None:
            # データが指定された場合は再計算
            self.calculate(data)
            
        # 結果確認
        if not hasattr(self, '_result') or self._result is None:
            # 結果がない場合は空の配列を返す
            return np.array([])
            
        return self._result.max_mult_values
    
    def get_dynamic_min_multiplier(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        動的な最小ATR乗数の値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 動的最小ATR乗数の値
        """
        if data is not None:
            # データが指定された場合は再計算
            self.calculate(data)
            
        # 結果確認
        if not hasattr(self, '_result') or self._result is None:
            # 結果がない場合は空の配列を返す
            return np.array([])
            
        return self._result.min_mult_values
    
    def reset(self) -> None:
        """
        インディケーターの状態をリセットする
        """
        super().reset()
        self._result = None
        self._data_hash = None
        self.cycle_er.reset() if hasattr(self.cycle_er, 'reset') else None
        self.cma.reset() if hasattr(self.cma, 'reset') else None
        self.catr.reset() if hasattr(self.catr, 'reset') else None 