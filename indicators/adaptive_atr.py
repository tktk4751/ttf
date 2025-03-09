#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from typing import Union
from numba import jit

from .indicator import Indicator


@jit(nopython=True)
def calculate_tr(high: np.ndarray, low: np.ndarray, prev_close: np.ndarray) -> np.ndarray:
    """
    True Rangeを計算する（高速化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        prev_close: 1期前の終値の配列
    
    Returns:
        True Range の配列
    """
    h_l = high - low
    h_pc = np.abs(high - prev_close)
    l_pc = np.abs(low - prev_close)
    
    tr = np.maximum(h_l, h_pc)
    tr = np.maximum(tr, l_pc)
    
    return tr


@jit(nopython=True)
def calculate_adaptive_period(
    src: np.ndarray,
    period: int,
    mfast: int,
    mslow: int
) -> np.ndarray:
    """
    アダプティブな期間を計算する（高速化版）
    
    Args:
        src: ソース価格の配列（通常はHL2）
        period: 基本期間
        mfast: 最小期間
        mslow: 最大期間
    
    Returns:
        適応的な期間の配列
    """
    length = len(src)
    diff = np.zeros(length)
    signal = np.zeros(length)
    noise = np.zeros(length)
    adaptive_period = np.full(length, period)
    
    # パラメーターの計算
    mper = max(period, 1)
    mper_diff = mslow - mfast
    
    # 差分の計算
    for i in range(1, length):
        diff[i] = abs(src[i] - src[i-1])
    
    # シグナルの計算
    for i in range(mper, length):
        signal[i] = abs(src[i] - src[i-mper])
    
    # ノイズの計算
    for i in range(mper, length):
        noise_sum = 0.0
        for j in range(mper):
            noise_sum += diff[i-j]
        noise[i] = noise[i-1] + diff[i] - diff[i-mper] if i > mper else noise_sum
    
    # 適応的な期間の計算（ゼロ除算対策）
    for i in range(mper, length):
        if noise[i] > 1e-10:  # 非常に小さい値でも除算を防ぐ
            er = min(signal[i] / noise[i], 1.0)
            adaptive_period[i] = max(int(er * mper_diff + mfast), 1)  # 最小期間を保証
    
    return adaptive_period


@jit(nopython=True)
def calculate_adaptive_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
    mfast: int,
    mslow: int
) -> np.ndarray:
    """
    アダプティブATRを計算する（高速化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
        period: 基本期間
        mfast: 最小期間
        mslow: 最大期間
    
    Returns:
        アダプティブATRの配列
    """
    length = len(high)
    atr = np.full(length, np.nan)
    src = (high + low) / 2
    
    # True Rangeの計算
    tr = np.zeros(length)
    tr[0] = high[0] - low[0]  # 初日のTRは単純なレンジ
    tr[1:] = calculate_tr(high[1:], low[1:], close[:-1])
    
    # 適応的な期間の計算
    adaptive_period = calculate_adaptive_period(src, period, mfast, mslow)
    
    # アダプティブATRの計算
    for i in range(1, length):
        if np.isnan(adaptive_period[i]):
            continue
            
        current_period = int(adaptive_period[i])
        if current_period < 1:
            current_period = 1
        
        if i >= current_period:
            # 指定期間のTRの平均を計算
            atr[i] = np.mean(tr[i-current_period+1:i+1])
    
    return atr


class AdaptiveATR(Indicator):
    """
    アダプティブATR (Adaptive Average True Range) インジケーター
    Efficiency Ratioを使用して適応的にATRを計算する
    
    特徴：
    - 市場の状態に応じて期間を動的に調整
    - より早い市場状態の変化の検出が可能
    - Numbaによる高速化を実装
    """
    
    def __init__(self, period: int = 14, mfast: int = 3, mslow: int = 120):
        """
        コンストラクタ
        
        Args:
            period: 基本期間（デフォルト: 14）
            mfast: 最小期間（デフォルト: 2）
            mslow: 最大期間（デフォルト: 30）
        """
        super().__init__(f"AdaptiveATR({period}, {mfast}, {mslow})")
        self.period = period
        self.mfast = mfast
        self.mslow = mslow
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        アダプティブATRを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'high'と'low'と'close'カラムが必要
        
        Returns:
            アダプティブATR値の配列
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
            
            # データ長の検証
            data_length = len(high)
            self._validate_period(self.period, data_length)
            
            # アダプティブATRの計算（高速化版）
            self._values = calculate_adaptive_atr(
                high, low, close, self.period, self.mfast, self.mslow
            )
            
            return self._values
            
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            return None 