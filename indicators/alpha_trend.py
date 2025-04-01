#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple
import numpy as np
import pandas as pd
from numba import jit

from .indicator import Indicator
from .alpha_atr import AlphaATR
from .efficiency_ratio import calculate_efficiency_ratio_for_period


@dataclass
class AlphaTrendResult:
    """AlphaTrendの計算結果"""
    upper_band: np.ndarray  # 上側のバンド価格
    lower_band: np.ndarray  # 下側のバンド価格
    trend: np.ndarray       # トレンド方向（1=上昇トレンド、-1=下降トレンド）
    smooth_upper: np.ndarray  # 75パーセンタイルの平滑化値
    smooth_lower: np.ndarray  # 25パーセンタイルの平滑化値
    er: np.ndarray          # 効率比
    dynamic_multiplier: np.ndarray  # 動的乗数
    dynamic_percentile_length: np.ndarray  # 動的パーセンタイル期間
    alpha_atr: np.ndarray   # AlphaATR値


@jit(nopython=True)
def calculate_dynamic_multiplier(er: np.ndarray, max_mult: float, min_mult: float) -> np.ndarray:
    """
    効率比に基づいて動的な乗数を計算する（高速化版）
    
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


@jit(nopython=True)
def calculate_dynamic_percentile_length(
    er: np.ndarray,
    max_length: int,
    min_length: int
) -> np.ndarray:
    """
    動的なパーセンタイル計算期間を計算する（高速化版）
    
    Args:
        er: 効率比の配列
        max_length: 最大期間（ERが低い時）
        min_length: 最小期間（ERが高い時）
    
    Returns:
        動的な期間の配列
    """
    # ERが高い（トレンドが強い）ほど期間は短く、
    # ERが低い（トレンドが弱い）ほど期間は長くなる
    periods = min_length + (1.0 - er) * (max_length - min_length)
    return np.round(periods).astype(np.int32)


@jit(nopython=True)
def calculate_dynamic_percentiles(
    high: np.ndarray,
    low: np.ndarray,
    dynamic_length: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    動的なパーセンタイルを計算する（高速化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        dynamic_length: 動的な期間配列
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: 25パーセンタイル（下限）と75パーセンタイル（上限）の配列
    """
    length = len(high)
    smooth_lower = np.zeros_like(high)
    smooth_upper = np.zeros_like(high)
    
    for i in range(length):
        current_length = min(i + 1, int(dynamic_length[i]))
        if current_length < 2:
            current_length = 2  # 最低2データポイント必要
            
        start_idx = max(0, i - current_length + 1)
        high_window = high[start_idx:i+1]
        low_window = low[start_idx:i+1]
        
        sorted_high = np.sort(high_window)
        sorted_low = np.sort(low_window)
        
        k25 = max(0, min(len(sorted_low) - 1, int(np.ceil(25/100.0 * len(sorted_low))) - 1))
        k75 = max(0, min(len(sorted_high) - 1, int(np.ceil(75/100.0 * len(sorted_high))) - 1))
        
        smooth_lower[i] = sorted_low[k25]
        smooth_upper[i] = sorted_high[k75]
    
    return smooth_lower, smooth_upper


@jit(nopython=True)
def calculate_alpha_trend(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    smooth_upper: np.ndarray,
    smooth_lower: np.ndarray,
    alpha_atr: np.ndarray,
    dynamic_multiplier: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    アルファトレンドを計算する（高速化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
        smooth_upper: 75パーセンタイルの平滑化値
        smooth_lower: 25パーセンタイルの平滑化値
        alpha_atr: AlphaATRの配列
        dynamic_multiplier: 動的乗数の配列
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 上側バンド、下側バンド、トレンド方向の配列
    """
    length = len(close)
    
    # バンドの初期化
    upper_band = np.zeros(length, dtype=np.float64)
    lower_band = np.zeros(length, dtype=np.float64)
    trend = np.zeros(length, dtype=np.int8)
    
    # 最初の有効なインデックスを特定（NaNを避ける）
    start_idx = 0
    for i in range(length):
        if not np.isnan(alpha_atr[i]) and not np.isnan(smooth_upper[i]) and not np.isnan(smooth_lower[i]):
            start_idx = i
            break
    
    # 最初の値を設定
    if start_idx < length:
        upper_band[start_idx] = smooth_upper[start_idx] + dynamic_multiplier[start_idx] * alpha_atr[start_idx]
        lower_band[start_idx] = smooth_lower[start_idx] - dynamic_multiplier[start_idx] * alpha_atr[start_idx]
        trend[start_idx] = 1 if close[start_idx] > upper_band[start_idx] else -1
    
    # バンドとトレンドの計算
    for i in range(start_idx + 1, length):
        if np.isnan(alpha_atr[i]) or np.isnan(smooth_upper[i]) or np.isnan(smooth_lower[i]):
            upper_band[i] = upper_band[i-1]
            lower_band[i] = lower_band[i-1]
            trend[i] = trend[i-1]
            continue
        
        # 新しいバンドの計算
        new_upper = smooth_upper[i] + dynamic_multiplier[i] * alpha_atr[i]
        new_lower = smooth_lower[i] - dynamic_multiplier[i] * alpha_atr[i]
        
        # トレンドに基づいてバンドを更新
        if trend[i-1] == 1:  # 上昇トレンド
            lower_band[i] = max(new_lower, lower_band[i-1])
            upper_band[i] = new_upper
            if close[i] < lower_band[i]:
                trend[i] = -1
            else:
                trend[i] = 1
        else:  # 下降トレンド
            upper_band[i] = min(new_upper, upper_band[i-1])
            lower_band[i] = new_lower
            if close[i] > upper_band[i]:
                trend[i] = 1
            else:
                trend[i] = -1
    
    return upper_band, lower_band, trend


class AlphaTrend(Indicator):
    """
    アルファトレンド（AlphaTrend）インジケーター
    
    ハイパートレンドをベースに、以下の改良を加えたトレンドフォローインジケーター：
    - ATRをアルファATR（効率比に基づく動的ATR）に置き換え
    - 動的パラメータ調整：効率比（ER）に基づいて全てのパラメータを最適化
    
    特徴：
    - 25-75パーセンタイルによるレベル計算（動的期間）
    - AlphaATRによる洗練されたボラティリティ測定
    - 効率比（ER）に基づく全パラメータの動的最適化
    - トレンドの強さに自動適応する高度なトレンドフォロー性能
    """
    
    def __init__(
        self,
        er_period: int = 21,
        max_percentile_length: int = 55,
        min_percentile_length: int = 13,
        max_atr_period: int = 89,
        min_atr_period: int = 13,
        max_multiplier: float = 3.0,
        min_multiplier: float = 1.0,
        alma_offset: float = 0.85,
        alma_sigma: float = 6
    ):
        """
        コンストラクタ
        
        Args:
            er_period: 効率比の計算期間（デフォルト: 21）
            max_percentile_length: パーセンタイル計算の最大期間（デフォルト: 55）
            min_percentile_length: パーセンタイル計算の最小期間（デフォルト: 13）
            max_atr_period: AlphaATR期間の最大値（デフォルト: 89）
            min_atr_period: AlphaATR期間の最小値（デフォルト: 13）
            max_multiplier: ATR乗数の最大値（デフォルト: 3.0）
            min_multiplier: ATR乗数の最小値（デフォルト: 1.0）
            alma_offset: ALMAのオフセット（デフォルト: 0.85）
            alma_sigma: ALMAのシグマ（デフォルト: 6）
        """
        super().__init__(
            f"AlphaTrend({er_period}, {max_percentile_length}, {min_percentile_length}, "
            f"{max_atr_period}, {min_atr_period}, {max_multiplier}, {min_multiplier})"
        )
        
        self.er_period = er_period
        self.max_percentile_length = max_percentile_length
        self.min_percentile_length = min_percentile_length
        self.max_atr_period = max_atr_period
        self.min_atr_period = min_atr_period
        self.max_multiplier = max_multiplier
        self.min_multiplier = min_multiplier
        self.alma_offset = alma_offset
        self.alma_sigma = alma_sigma
        
        # AlphaATRインジケーターのインスタンス化
        self.alpha_atr = AlphaATR(
            max_atr_period=max_atr_period,
            min_atr_period=min_atr_period,
        )
        
        self._result = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        アルファトレンドを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'high', 'low', 'close'カラムが必要
        
        Returns:
            トレンド方向の配列（1=上昇、-1=下降）
        """
        try:
            # データの検証と変換
            if isinstance(data, pd.DataFrame):
                if not all(col in data.columns for col in ['high', 'low', 'close']):
                    raise ValueError("DataFrameには'high', 'low', 'close'カラムが必要です")
                high = data['high'].values
                low = data['low'].values
                close = data['close'].values
            else:
                if data.ndim == 2 and data.shape[1] >= 4:
                    high = data[:, 1]  # high
                    low = data[:, 2]   # low
                    close = data[:, 3] # close
                else:
                    raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")
            
            # データ長の検証
            data_length = len(high)
            self._validate_period(self.er_period, data_length)
            
            # AlphaATRの計算
            alpha_atr_values = self.alpha_atr.calculate(data)
            
            # 効率比（ER）の取得
            er = self.alpha_atr.get_efficiency_ratio()
            
            # 動的なパーセンタイル計算期間
            dynamic_percentile_length = calculate_dynamic_percentile_length(
                er,
                self.max_percentile_length,
                self.min_percentile_length
            )
            
            # 動的乗数の計算
            dynamic_multiplier = calculate_dynamic_multiplier(
                er,
                self.max_multiplier,
                self.min_multiplier
            )
            
            # 動的パーセンタイルの計算
            smooth_lower, smooth_upper = calculate_dynamic_percentiles(
                high, low, dynamic_percentile_length
            )
            
            # アルファトレンドの計算
            upper_band, lower_band, trend = calculate_alpha_trend(
                high, low, close,
                smooth_upper, smooth_lower,
                alpha_atr_values,
                dynamic_multiplier
            )
            
            # 結果の保存
            self._result = AlphaTrendResult(
                upper_band=upper_band,
                lower_band=lower_band,
                trend=trend,
                smooth_upper=smooth_upper,
                smooth_lower=smooth_lower,
                er=er,
                dynamic_multiplier=dynamic_multiplier,
                dynamic_percentile_length=dynamic_percentile_length,
                alpha_atr=alpha_atr_values
            )
            
            # 基底クラスの要件を満たすため
            self._values = trend
            return trend
            
        except Exception as e:
            self.logger.error(f"AlphaTrend計算中にエラー: {str(e)}")
            return None
    
    def get_bands(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        アルファトレンドのバンド値を取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (上限バンド, 下限バンド)の値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.upper_band, self._result.lower_band
    
    def get_trend(self) -> np.ndarray:
        """
        トレンド方向を取得する
        
        Returns:
            np.ndarray: トレンド方向の配列（1=上昇、-1=下降）
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.trend
    
    def get_percentiles(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        25パーセンタイルと75パーセンタイルの値を取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (25パーセンタイル, 75パーセンタイル)の値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.smooth_lower, self._result.smooth_upper
    
    def get_efficiency_ratio(self) -> np.ndarray:
        """
        効率比（ER）の値を取得する
        
        Returns:
            np.ndarray: 効率比の値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.er
    
    def get_dynamic_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        動的パラメータの値を取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (動的乗数, 動的パーセンタイル期間)の値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.dynamic_multiplier, self._result.dynamic_percentile_length
    
    def get_alpha_atr(self) -> np.ndarray:
        """
        AlphaATRの値を取得する
        
        Returns:
            np.ndarray: AlphaATRの値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.alpha_atr 