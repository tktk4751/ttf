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
class SupertrendResult:
    """スーパートレンドの計算結果"""
    upper_band: np.ndarray  # 上側のバンド価格
    lower_band: np.ndarray  # 下側のバンド価格
    trend: np.ndarray      # トレンド方向（1=上昇トレンド、-1=下降トレンド）


@jit(nopython=True)
def calculate_supertrend(high: np.ndarray, low: np.ndarray, close: np.ndarray, atr: np.ndarray, multiplier: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    スーパートレンドを計算する（高速化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
        atr: ATRの配列
        multiplier: ATRの乗数
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 上側バンド、下側バンド、トレンド方向の配列
    """
    length = len(close)
    
    # 基準となるバンドの計算
    hl_avg = (high + low) / 2
    final_upper_band = hl_avg + multiplier * atr
    final_lower_band = hl_avg - multiplier * atr
    
    # トレンド方向の配列を初期化
    trend = np.zeros(length, dtype=np.int8)
    upper_band = np.zeros(length, dtype=np.float64)
    lower_band = np.zeros(length, dtype=np.float64)
    
    # 最初の値を設定
    trend[0] = 1 if close[0] > final_upper_band[0] else -1
    
    # バンドとトレンドの計算
    for i in range(1, length):
        if close[i] > final_upper_band[i-1]:
            trend[i] = 1
        elif close[i] < final_lower_band[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
            
            # バンドの調整
            if trend[i] == 1 and final_lower_band[i] < final_lower_band[i-1]:
                final_lower_band[i] = final_lower_band[i-1]
            elif trend[i] == -1 and final_upper_band[i] > final_upper_band[i-1]:
                final_upper_band[i] = final_upper_band[i-1]
        
        # トレンドに基づいてバンドを設定
        if trend[i] == 1:
            upper_band[i] = np.nan
            lower_band[i] = final_lower_band[i]
        else:
            upper_band[i] = final_upper_band[i]
            lower_band[i] = np.nan
    
    return upper_band, lower_band, trend


class Supertrend(Indicator):
    """
    スーパートレンドインジケーター（高速化版）
    ATRを使用して、トレンドの方向と強さを判断する
    """
    
    def __init__(self, period: int = 10, multiplier: float = 3.0):
        """
        コンストラクタ
        
        Args:
            period: ATR期間
            multiplier: ATRの乗数
        """
        super().__init__(f"Supertrend({period}, {multiplier})")
        self.period = period
        self.multiplier = multiplier
        self._atr = ATR(period)
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> SupertrendResult:
        """
        スーパートレンドを計算する
        
        Args:
            data: 価格データ
        
        Returns:
            スーパートレンドの計算結果
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
        
        # ATRの計算
        atr = self._atr.calculate(data)
        
        # スーパートレンドの計算（高速化版）
        upper_band, lower_band, trend = calculate_supertrend(
            high, low, close, atr, self.multiplier
        )
        
        self._values = trend  # 基底クラスの要件を満たすため
        
        return SupertrendResult(
            upper_band=upper_band,
            lower_band=lower_band,
            trend=trend
        ) 