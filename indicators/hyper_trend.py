#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple
import numpy as np
import pandas as pd
from numba import jit

from .indicator import Indicator
from .atr import ATR
from .efficiency_ratio import calculate_efficiency_ratio_for_period
from .normalized_chop import calculate_normalized_chop
from .normalized_adx import calculate_normalized_adx
from .trend_quality import TrendQuality


@dataclass
class HyperTrendResult:
    """HyperTrendの計算結果"""
    upper_band: np.ndarray  # 上側のバンド価格
    lower_band: np.ndarray  # 下側のバンド価格
    trend: np.ndarray      # トレンド方向（1=上昇トレンド、-1=下降トレンド）
    smooth_upper: np.ndarray  # 75パーセンタイルの平滑化値
    smooth_lower: np.ndarray  # 25パーセンタイルの平滑化値
    er: np.ndarray         # 効率比
    dynamic_period: np.ndarray  # 動的ATR期間
    dynamic_multiplier: np.ndarray  # 動的乗数
    dynamic_length: np.ndarray  # 動的パーセンタイル期間


@jit(nopython=True)
def calculate_efficiency_ratio(close: np.ndarray, period: int) -> np.ndarray:
    """
    効率比（ER）を計算する（高速化版）
    
    Args:
        close: 終値の配列
        period: 計算期間
    
    Returns:
        効率比の配列
    """
    length = len(close)
    er = np.zeros(length)
    
    for i in range(period, length):
        direction = abs(close[i] - close[i-period])
        volatility = 0.0
        for j in range(i-period+1, i+1):
            volatility += abs(close[j] - close[j-1])
        
        er[i] = direction / volatility if volatility != 0 else 0.0
    
    return er


@jit(nopython=True)
def percentile_nearest_rank(data: np.ndarray, length: int, percentile: float) -> np.ndarray:
    """
    指定されたパーセンタイルを計算する（高速化版）
    
    Args:
        data: 入力データ配列
        length: ルックバック期間
        percentile: パーセンタイル値（0-100）
    
    Returns:
        パーセンタイル値の配列
    """
    result = np.zeros_like(data)
    n = len(data)
    
    for i in range(n):
        if i < length - 1:
            result[i] = data[i]
            continue
            
        window = data[max(0, i-length+1):i+1]
        sorted_window = np.sort(window)
        k = int(np.ceil(percentile/100.0 * len(window)))
        result[i] = sorted_window[k-1]
    
    return result


@jit(nopython=True)
def calculate_dynamic_parameters(
    er: np.ndarray,
    max_period: int,
    min_period: int,
    max_multiplier: float,
    min_multiplier: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    動的なATR期間と乗数を計算する（高速化版）
    
    Args:
        er: 効率比の配列
        max_period: ATR期間の最大値
        min_period: ATR期間の最小値
        max_multiplier: ATR乗数の最大値
        min_multiplier: ATR乗数の最小値
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: 動的ATR期間と動的乗数の配列
    """
    length = len(er)
    dynamic_period = np.zeros(length, dtype=np.int32)
    dynamic_multiplier = np.zeros(length)
    
    for i in range(length):
        # ERに基づいて期間と乗数を線形補間
        if er[i] <= 0:
            dynamic_period[i] = max_period
            dynamic_multiplier[i] = max_multiplier
        elif er[i] >= 1:
            dynamic_period[i] = min_period
            dynamic_multiplier[i] = min_multiplier
        else:
            dynamic_period[i] = int(max_period - (max_period - min_period) * er[i])
            dynamic_multiplier[i] = max_multiplier - (max_multiplier - min_multiplier) * er[i]
    
    return dynamic_period, dynamic_multiplier


@jit(nopython=True)
def calculate_hyper_trend(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    smooth_upper: np.ndarray,
    smooth_lower: np.ndarray,
    atr: np.ndarray,
    dynamic_multiplier: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    HyperTrendを計算する（高速化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
        smooth_upper: 75パーセンタイルの平滑化値
        smooth_lower: 25パーセンタイルの平滑化値
        atr: ATRの配列
        dynamic_multiplier: 動的乗数の配列
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 上側バンド、下側バンド、トレンド方向の配列
    """
    length = len(close)
    
    # バンドの初期化
    upper_band = np.zeros(length, dtype=np.float64)
    lower_band = np.zeros(length, dtype=np.float64)
    trend = np.zeros(length, dtype=np.int8)
    
    # 最初の値を設定
    upper_band[0] = smooth_upper[0] + dynamic_multiplier[0] * atr[0]
    lower_band[0] = smooth_lower[0] - dynamic_multiplier[0] * atr[0]
    trend[0] = 1 if close[0] > upper_band[0] else -1
    
    # バンドとトレンドの計算
    for i in range(1, length):
        # 新しいバンドの計算
        new_upper = smooth_upper[i] + dynamic_multiplier[i] * atr[i]
        new_lower = smooth_lower[i] - dynamic_multiplier[i] * atr[i]
        
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


@jit(nopython=True)
def calculate_dynamic_length(
    er: np.ndarray,
    max_length: int,
    min_length: int
) -> np.ndarray:
    """
    動的なパーセンタイルlengthを計算する（高速化版）
    
    Args:
        er: 効率比の配列
        max_length: 最大length（ERが低い時）
        min_length: 最小length（ERが高い時）
    
    Returns:
        動的なlength配列
    """
    length = len(er)
    dynamic_length = np.zeros(length, dtype=np.int32)
    
    for i in range(length):
        if er[i] <= 0:
            dynamic_length[i] = max_length
        elif er[i] >= 1:
            dynamic_length[i] = min_length
        else:
            # ERに基づいて線形補間
            dynamic_length[i] = int(max_length - (max_length - min_length) * er[i])
    
    return dynamic_length


@jit(nopython=True)
def calculate_dynamic_percentiles(
    high: np.ndarray,
    dynamic_length: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    動的なパーセンタイルを計算する（高速化版）
    
    Args:
        high: 高値の配列
        dynamic_length: 動的なlength配列
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: 25パーセンタイルと75パーセンタイルの配列
    """
    length = len(high)
    smooth_lower = np.zeros_like(high)
    smooth_upper = np.zeros_like(high)
    
    for i in range(length):
        current_length = min(i + 1, dynamic_length[i])
        start_idx = max(0, i - current_length + 1)
        window = high[start_idx:i+1]
        
        if len(window) >= 2:  # 最低2点必要
            sorted_window = np.sort(window)
            k25 = max(0, min(len(window) - 1, int(np.ceil(25/100.0 * len(window))) - 1))
            k75 = max(0, min(len(window) - 1, int(np.ceil(75/100.0 * len(window))) - 1))
            smooth_lower[i] = sorted_window[k25]
            smooth_upper[i] = sorted_window[k75]
        else:
            smooth_lower[i] = high[i]
            smooth_upper[i] = high[i]
    
    return smooth_lower, smooth_upper


class HyperTrend(Indicator):
    """
    HyperTrendインジケーター
    
    TrendAlphaの動的パラメータ調整とパーセンタイルスーパートレンドの25-75パーセンタイルを組み合わせた
    アダプティブなトレンドフォローインジケーター
    
    特徴：
    - 25-75パーセンタイルによるミッドライン計算（動的期間）
    - 効率比（ER）に基づく動的なATR期間と乗数の調整
    - トレンドの強さに応じて自動的にパラメータを最適化
    """
    
    def __init__(
        self,
        er_period: int = 10,
        max_percentile_length: int = 55,
        min_percentile_length: int = 14,
        max_atr_period: int = 120,
        min_atr_period: int = 5,
        max_multiplier: float = 3.0,
        min_multiplier: float = 1.0
    ):
        """
        コンストラクタ
        
        Args:
            er_period: 効率比の計算期間（デフォルト: 10）
            max_percentile_length: パーセンタイル計算の最大期間（デフォルト: 55）
            min_percentile_length: パーセンタイル計算の最小期間（デフォルト: 14）
            max_atr_period: ATR期間の最大値（デフォルト: 120）
            min_atr_period: ATR期間の最小値（デフォルト: 5）
            max_multiplier: ATR乗数の最大値（デフォルト: 3.0）
            min_multiplier: ATR乗数の最小値（デフォルト: 1.0）
        """
        super().__init__(
            f"HyperTrend({er_period}, {max_percentile_length}, {min_percentile_length}, "
            f"{max_atr_period}, {min_atr_period}, {max_multiplier}, {min_multiplier})"
        )
        
        self.er_period = er_period
        self.max_percentile_length = max_percentile_length
        self.min_percentile_length = min_percentile_length
        self.max_atr_period = max_atr_period
        self.min_atr_period = min_atr_period
        self.max_multiplier = max_multiplier
        self.min_multiplier = min_multiplier
        
        # ATRインジケーターは動的に期間を変更するため、最大期間で初期化
        self._atr = ATR(max_atr_period)
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> HyperTrendResult:
        """
        HyperTrendを計算する
        
        Args:
            data: 価格データ
        
        Returns:
            HyperTrendの計算結果
        """
        # データの準備
        if isinstance(data, pd.DataFrame):
            high = data['high'].to_numpy()
            low = data['low'].to_numpy()
            close = data['close'].to_numpy()
        else:
            high = data[:, 1]  # high
            low = data[:, 2]   # low
            close = data[:, 3] # close
        
        # 効率比の計算
        er = calculate_efficiency_ratio_for_period(close, self.er_period)
        
        # 動的パラメータの計算
        dynamic_period, dynamic_multiplier = calculate_dynamic_parameters(
            er,
            self.max_atr_period,
            self.min_atr_period,
            self.max_multiplier,
            self.min_multiplier
        )
        
        # 動的lengthの計算
        dynamic_length = calculate_dynamic_length(
            er,
            self.max_percentile_length,
            self.min_percentile_length
        )
        
        # 動的パーセンタイルの計算
        smooth_lower, smooth_upper = calculate_dynamic_percentiles(high, dynamic_length)
        
        # ATRの計算（最大期間で計算）
        atr = self._atr.calculate(data)
        
        # HyperTrendの計算
        upper_band, lower_band, trend = calculate_hyper_trend(
            high, low, close, smooth_upper, smooth_lower,
            atr, dynamic_multiplier
        )
        
        self._values = trend  # 基底クラスの要件を満たすため
        
        return HyperTrendResult(
            upper_band=upper_band,
            lower_band=lower_band,
            trend=trend,
            smooth_upper=smooth_upper,
            smooth_lower=smooth_lower,
            er=er,
            dynamic_period=dynamic_period,
            dynamic_multiplier=dynamic_multiplier,
            dynamic_length=dynamic_length
        ) 