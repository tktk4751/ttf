#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from typing import Union
from numba import jit

from .indicator import Indicator


@jit(nopython=True)
def calculate_true_range(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> np.ndarray:
    """
    True Range (TR)を計算する（高速化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
    
    Returns:
        True Range (TR)の配列
    """
    length = len(high)
    tr = np.zeros(length)
    
    # 最初の要素は単純なレンジ
    tr[0] = high[0] - low[0]
    
    # 2番目以降の要素はTRを計算
    for i in range(1, length):
        tr1 = high[i] - low[i]  # 当日の高値 - 当日の安値
        tr2 = abs(high[i] - close[i-1])  # |当日の高値 - 前日の終値|
        tr3 = abs(low[i] - close[i-1])  # |当日の安値 - 前日の終値|
        tr[i] = max(tr1, tr2, tr3)
    
    return tr


@jit(nopython=True)
def calculate_atr(tr: np.ndarray, period: int) -> np.ndarray:
    """
    ATR (Average True Range)を計算する（高速化版）
    Wilder's Smoothingを使用
    
    Args:
        tr: True Range (TR)の配列
        period: 期間
    
    Returns:
        ATR値の配列
    """
    length = len(tr)
    atr = np.zeros(length)
    
    # 最初のATRは単純移動平均で計算
    atr[period-1] = np.mean(tr[:period])
    
    # 2番目以降はWilder's Smoothingで計算
    # ATR(t) = ((period-1) * ATR(t-1) + TR(t)) / period
    for i in range(period, length):
        atr[i] = ((period - 1) * atr[i-1] + tr[i]) / period
    
    return atr


class ATR(Indicator):
    """
    ATR (Average True Range) インジケーター
    価格のボラティリティを測定する
    
    Numbaによる高速化を実装：
    - True Range (TR)の計算を最適化
    - Wilder's Smoothingの計算を最適化
    """
    
    def __init__(self, period: int = 14):
        """
        コンストラクタ
        
        Args:
            period: 期間（デフォルト: 14）
        """
        super().__init__(f"ATR({period})")
        self.period = period
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        ATRを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'high'と'low'と'close'カラムが必要
        
        Returns:
            ATR値の配列
        """
        try:
            # データの検証と変換
            if isinstance(data, pd.DataFrame):
                if not all(col in data.columns for col in ['high', 'low', 'close']):
                    raise ValueError("DataFrameには'high'と'low'と'close'カラムが必要です")
                high = data['high'].values
                low = data['low'].values
                close = data['close'].values
            else:
                if data.ndim != 2 or data.shape[1] < 4:
                    raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")
                high = data[:, 1]  # high
                low = data[:, 2]   # low
                close = data[:, 3] # close
            
            # True Range (TR)の計算（高速化版）
            tr = calculate_true_range(high, low, close)
            
            # ATRの計算（高速化版）
            atr = calculate_atr(tr, self.period)
            
            # 計算結果を保存
            self._values = atr
            return atr
            
        except Exception:
            return None 