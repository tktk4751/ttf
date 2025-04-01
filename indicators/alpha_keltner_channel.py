#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional, Any
import numpy as np
import pandas as pd
from numba import jit, vectorize

from .indicator import Indicator
from .alpha_ma import AlphaMA
from .alpha_atr import AlphaATR
from .efficiency_ratio import calculate_efficiency_ratio_for_period


@dataclass
class AlphaKeltnerChannelResult:
    """AlphaKeltnerChannelの計算結果"""
    middle: np.ndarray        # 中心線（AlphaMA）
    upper: np.ndarray         # 上限バンド
    lower: np.ndarray         # 下限バンド
    er: np.ndarray            # Efficiency Ratio
    dynamic_multiplier: np.ndarray  # 動的ATR乗数
    alpha_atr: np.ndarray     # AlphaATR値


@vectorize(['float64(float64, float64, float64)'], nopython=True)
def calculate_dynamic_multiplier_vec(er: float, max_mult: float, min_mult: float) -> float:
    """
    効率比に基づいて動的なATR乗数を計算する（ベクトル化版）
    
    Args:
        er: 効率比の値
        max_mult: 最大乗数
        min_mult: 最小乗数
    
    Returns:
        動的な乗数の値
    """
    # ERが高い（トレンドが強い）ほど乗数は小さく、
    # ERが低い（トレンドが弱い）ほど乗数は大きくなる
    if np.isnan(er):
        return np.nan
    return max_mult - er * (max_mult - min_mult)


@jit(nopython=True)
def calculate_dynamic_multiplier(er: np.ndarray, max_mult: float, min_mult: float) -> np.ndarray:
    """
    効率比に基づいて動的なATR乗数を計算する（高速化版）
    
    Args:
        er: 効率比の配列
        max_mult: 最大乗数
        min_mult: 最小乗数
    
    Returns:
        動的な乗数の配列
    """
    # ERが高い（トレンドが強い）ほど乗数は小さく、
    # ERが低い（トレンドが弱い）ほど乗数は大きくなる
    multipliers = max_mult - er * (max_mult - min_mult)
    return multipliers


@jit(nopython=True, parallel=True)
def calculate_alpha_keltner_channel(
    alpha_ma: np.ndarray,
    alpha_atr: np.ndarray,
    dynamic_multiplier: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    アルファケルトナーチャネルを計算する（パラレル高速化版）
    
    Args:
        alpha_ma: AlphaMA値の配列
        alpha_atr: AlphaATR値の配列
        dynamic_multiplier: 動的乗数の配列
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            (中心線, 上限バンド, 下限バンド)の配列
    """
    length = len(alpha_ma)
    middle = np.copy(alpha_ma)
    upper = np.full_like(alpha_ma, np.nan)
    lower = np.full_like(alpha_ma, np.nan)
    
    # ベクトル化された計算を使用
    valid_mask = ~(np.isnan(alpha_ma) | np.isnan(alpha_atr) | np.isnan(dynamic_multiplier))
    band_width = np.zeros_like(alpha_ma)
    
    # 有効な要素のみ計算
    if np.any(valid_mask):
        band_width[valid_mask] = alpha_atr[valid_mask] * dynamic_multiplier[valid_mask]
        upper[valid_mask] = middle[valid_mask] + band_width[valid_mask]
        lower[valid_mask] = middle[valid_mask] - band_width[valid_mask]
    
    return middle, upper, lower


class AlphaKeltnerChannel(Indicator):
    """
    アルファケルトナーチャネル（Alpha Keltner Channel）インジケーター
    
    特徴:
    - 中心線にAlphaMA（動的適応型移動平均）を使用
    - バンド幅の計算にAlphaATR（動的適応型ATR）を使用
    - ATR乗数が効率比（ER）に基づいて動的に調整
    - RSXの3段階平滑化アルゴリズムをAlphaATRで使用
    
    市場状態に応じた最適な挙動:
    - トレンド強い（ER高い）:
      - 狭いバンド幅（小さい乗数）でトレンドをタイトに追従
    - トレンド弱い（ER低い）:
      - 広いバンド幅（大きい乗数）でレンジ相場の振れ幅を捉える
    """
    
    def __init__(
        self,
        er_period: int = 21,
        max_kama_period: int = 55,
        min_kama_period: int = 8,
        max_atr_period: int = 55,
        min_atr_period: int = 8,
        max_multiplier: float = 3.0,
        min_multiplier: float = 1.5
    ):
        """
        コンストラクタ
        
        Args:
            er_period: 効率比の計算期間（デフォルト: 21）
            max_kama_period: AlphaMAのKAMA最大期間（デフォルト: 55）
            min_kama_period: AlphaMAのKAMA最小期間（デフォルト: 8）
            max_atr_period: AlphaATRの最大期間（デフォルト: 55）
            min_atr_period: AlphaATRの最小期間（デフォルト: 8）
            max_multiplier: ATR乗数の最大値（デフォルト: 3.0）
            min_multiplier: ATR乗数の最小値（デフォルト: 1.5）
        """
        super().__init__(
            f"AlphaKC({er_period}, {max_multiplier}, {min_multiplier})"
        )
        self.er_period = er_period
        self.max_kama_period = max_kama_period
        self.min_kama_period = min_kama_period
        self.max_atr_period = max_atr_period
        self.min_atr_period = min_atr_period
        self.max_multiplier = max_multiplier
        self.min_multiplier = min_multiplier
        
        # AlphaMAとAlphaATRのインスタンス化
        self.alpha_ma = AlphaMA(
            er_period=er_period,
            max_kama_period=max_kama_period,
            min_kama_period=min_kama_period
        )
        
        self.alpha_atr = AlphaATR(
            er_period=er_period,
            max_atr_period=max_atr_period,
            min_atr_period=min_atr_period
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
        param_str = f"{self.er_period}_{self.max_kama_period}_{self.min_kama_period}_{self.max_atr_period}_{self.min_atr_period}_{self.max_multiplier}_{self.min_multiplier}"
        return f"{data_hash}_{param_str}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        アルファケルトナーチャネルを計算する
        
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
            
            # AlphaMAの計算 (計算コストが高いため最初に実行)
            alpha_ma_values = self.alpha_ma.calculate(data)
            if alpha_ma_values is None:
                raise ValueError("AlphaMAの計算に失敗しました")
            
            # AlphaATRの計算 (計算コストが高いため最初に実行)
            alpha_atr_values = self.alpha_atr.calculate(data)
            if alpha_atr_values is None:
                raise ValueError("AlphaATRの計算に失敗しました")
            
            # 効率比（ER）の取得（AlphaMAからキャッシュ済みの値を取得）
            er = self.alpha_ma.get_efficiency_ratio()
            
            # 動的ATR乗数の計算（ベクトル化関数を使用）
            # ベクトル化関数はより高速に動作する
            dynamic_multiplier = calculate_dynamic_multiplier_vec(
                er,
                self.max_multiplier,
                self.min_multiplier
            )
            
            # アルファケルトナーチャネルの計算（パラレル高速版）
            middle, upper, lower = calculate_alpha_keltner_channel(
                alpha_ma_values,
                alpha_atr_values,
                dynamic_multiplier
            )
            
            # 結果の保存（参照コピーを避けるためnp.copyを使用）
            self._result = AlphaKeltnerChannelResult(
                middle=np.copy(middle),
                upper=np.copy(upper),
                lower=np.copy(lower),
                er=np.copy(er),
                dynamic_multiplier=np.copy(dynamic_multiplier),
                alpha_atr=np.copy(alpha_atr_values)
            )
            
            # 中心線を値として保存
            self._values = middle
            return middle
            
        except Exception as e:
            self.logger.error(f"AlphaKeltnerChannel計算中にエラー: {str(e)}")
            
            # エラー時は前回の結果を維持する（nullではなく）
            if self._result is None:
                # 初回エラー時は空の結果を作成
                empty_array = np.array([])
                self._result = AlphaKeltnerChannelResult(
                    middle=empty_array,
                    upper=empty_array,
                    lower=empty_array,
                    er=empty_array,
                    dynamic_multiplier=empty_array,
                    alpha_atr=empty_array
                )
                self._values = empty_array
            
            return self._values
    
    def get_bands(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        ケルトナーチャネルのバンド値を取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                (中心線, 上限バンド, 下限バンド)の値
        """
        if self._result is None:
            # 結果がない場合は空の配列を返す
            empty = np.array([])
            return empty, empty, empty
        return self._result.middle, self._result.upper, self._result.lower
    
    def get_efficiency_ratio(self) -> np.ndarray:
        """
        効率比（ER）の値を取得する
        
        Returns:
            np.ndarray: 効率比の値
        """
        if self._result is None:
            # 結果がない場合は空の配列を返す
            return np.array([])
        return self._result.er
    
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
        self.alpha_ma.reset() if hasattr(self.alpha_ma, 'reset') else None
        self.alpha_atr.reset() if hasattr(self.alpha_atr, 'reset') else None 