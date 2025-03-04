#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple
import numpy as np
import pandas as pd
from numba import jit

from .indicator import Indicator
from .atr import ATR


@dataclass
class PercentileSupertrendResult:
    """25-75パーセンタイルスーパートレンドの計算結果"""
    upper_band: np.ndarray  # 上側のバンド価格
    lower_band: np.ndarray  # 下側のバンド価格
    trend: np.ndarray      # トレンド方向（1=上昇トレンド、-1=下降トレンド）
    smooth_upper: np.ndarray  # 75パーセンタイルの平滑化値
    smooth_lower: np.ndarray  # 25パーセンタイルの平滑化値


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
def calculate_percentile_supertrend(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                                  smooth_upper: np.ndarray, smooth_lower: np.ndarray,
                                  atr: np.ndarray, multiplier: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    25-75パーセンタイルスーパートレンドを計算する（高速化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
        smooth_upper: 75パーセンタイルの平滑化値
        smooth_lower: 25パーセンタイルの平滑化値
        atr: ATRの配列
        multiplier: ATRの乗数
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 上側バンド、下側バンド、トレンド方向の配列
    """
    length = len(close)
    
    # バンドの初期化
    upper_band = smooth_upper + multiplier * atr
    lower_band = smooth_lower - multiplier * atr
    
    final_upper_band = np.zeros(length, dtype=np.float64)
    final_lower_band = np.zeros(length, dtype=np.float64)
    trend = np.zeros(length, dtype=np.int8)
    
    # 最初の値を設定
    final_upper_band[0] = upper_band[0]
    final_lower_band[0] = lower_band[0]
    trend[0] = 1 if close[0] > upper_band[0] else -1
    
    # バンドとトレンドの計算
    for i in range(1, length):
        if close[i-1] > final_upper_band[i-1]:
            trend[i] = -1
        elif close[i-1] < final_lower_band[i-1]:
            trend[i] = 1
        else:
            trend[i] = trend[i-1]
            
        if trend[i] == 1:
            final_lower_band[i] = max(lower_band[i], final_lower_band[i-1])
            final_upper_band[i] = upper_band[i]
        else:
            final_upper_band[i] = min(upper_band[i], final_upper_band[i-1])
            final_lower_band[i] = lower_band[i]
    
    return final_upper_band, final_lower_band, trend


class PercentileSupertrend(Indicator):
    """
    25-75パーセンタイルスーパートレンドインジケーター
    """
    
    def __init__(self, subject: int = 14, multiplier: float = 1.0, percentile_length: int = 27):
        """
        コンストラクタ
        
        Args:
            subject: ATR期間
            multiplier: ATRの乗数
            percentile_length: パーセンタイル計算期間
        """
        super().__init__(f"PercentileSupertrend({subject}, {multiplier}, {percentile_length})")
        self.subject = subject
        self.multiplier = multiplier
        self.percentile_length = percentile_length
        self._atr = ATR(subject)
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> PercentileSupertrendResult:
        """
        25-75パーセンタイルスーパートレンドを計算する
        
        Args:
            data: 価格データ
        
        Returns:
            25-75パーセンタイルスーパートレンドの計算結果
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
        
        # パーセンタイルの計算
        smooth_lower = percentile_nearest_rank(high, self.percentile_length, 25)
        smooth_upper = percentile_nearest_rank(high, self.percentile_length, 75)
        
        # ATRの計算
        atr = self._atr.calculate(data)
        
        # スーパートレンドの計算
        upper_band, lower_band, trend = calculate_percentile_supertrend(
            high, low, close, smooth_upper, smooth_lower, atr, self.multiplier
        )
        
        self._values = trend  # 基底クラスの要件を満たすため
        
        return PercentileSupertrendResult(
            upper_band=upper_band,
            lower_band=lower_band,
            trend=trend,
            smooth_upper=smooth_upper,
            smooth_lower=smooth_lower
        ) 