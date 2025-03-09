#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union
import numpy as np
import pandas as pd
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
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    
    tr = np.maximum(tr1, tr2)
    tr = np.maximum(tr, tr3)
    
    return tr


@jit(nopython=True)
def calculate_dm(high: np.ndarray, low: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Directional Movement（+DM, -DM）を計算する（高速化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
    
    Returns:
        tuple[np.ndarray, np.ndarray]: (+DM, -DM)の配列
    """
    length = len(high)
    plus_dm = np.zeros(length)
    minus_dm = np.zeros(length)
    
    for i in range(1, length):
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        
        if (up > down) and (up > 0):
            plus_dm[i] = up
        else:
            plus_dm[i] = 0
        
        if (down > up) and (down > 0):
            minus_dm[i] = down
        else:
            minus_dm[i] = 0
    
    return plus_dm, minus_dm


@jit(nopython=True)
def calculate_normalized_adx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int
) -> np.ndarray:
    """
    正規化されたADXを計算する（高速化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
        period: 計算期間
    
    Returns:
        正規化されたADX値の配列（0-1の範囲）
    """
    length = len(high)
    tr = np.zeros(length)
    tr[0] = high[0] - low[0]
    tr[1:] = calculate_tr(high[1:], low[1:], close[:-1])
    
    # +DM, -DMの計算
    plus_dm, minus_dm = calculate_dm(high, low)
    
    # 指数移動平均の計算用の係数
    alpha = 2.0 / (period + 1.0)
    
    # TR, +DM, -DMの平滑化
    smoothed_tr = np.zeros(length)
    smoothed_plus_dm = np.zeros(length)
    smoothed_minus_dm = np.zeros(length)
    
    # 最初の値の初期化
    smoothed_tr[0] = tr[0]
    smoothed_plus_dm[0] = plus_dm[0]
    smoothed_minus_dm[0] = minus_dm[0]
    
    # 指数移動平均による平滑化
    for i in range(1, length):
        smoothed_tr[i] = (tr[i] * alpha) + (smoothed_tr[i-1] * (1 - alpha))
        smoothed_plus_dm[i] = (plus_dm[i] * alpha) + (smoothed_plus_dm[i-1] * (1 - alpha))
        smoothed_minus_dm[i] = (minus_dm[i] * alpha) + (smoothed_minus_dm[i-1] * (1 - alpha))
    
    # +DI, -DIの計算
    nonzero_tr = np.where(smoothed_tr == 0, 1e-10, smoothed_tr)
    plus_di = smoothed_plus_dm / nonzero_tr
    minus_di = smoothed_minus_dm / nonzero_tr
    
    # DXの計算
    dx_sum = plus_di + minus_di
    # ゼロ除算を防ぐ
    nonzero_sum = np.where(dx_sum == 0, 1e-10, dx_sum)
    dx = np.abs(plus_di - minus_di) / nonzero_sum
    
    # ADXの計算（DXの平滑化）
    adx = np.zeros(length)
    adx[0] = dx[0]
    
    for i in range(1, length):
        adx[i] = (dx[i] * alpha) + (adx[i-1] * (1 - alpha))
    
    return adx


class NormalizedADX(Indicator):
    """
    正規化ADX（Normalized Average Directional Index）インジケーター
    
    通常のADXを0-1の範囲に正規化したインジケーター
    トレンドの強さを測定する
    - 0.25以上: 強いトレンド
    - 0.20以下: トレンドなし
    """
    
    def __init__(self, period: int = 13):
        """
        コンストラクタ
        
        Args:
            period: 計算期間（デフォルト: 13）
        """
        super().__init__(f"NADX({period})")
        self.period = period
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        正規化ADXを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'high'と'low'と'close'カラムが必要
        
        Returns:
            正規化ADXの値（0-1の範囲）
        """
        # データの検証と変換
        if isinstance(data, pd.DataFrame):
            if not all(col in data.columns for col in ['high', 'low', 'close']):
                raise ValueError("DataFrameには'high'と'low'と'close'カラムが必要です")
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
        else:
            if data.ndim == 2:
                high = data[:, 1]  # high
                low = data[:, 2]   # low
                close = data[:, 3] # close
            else:
                raise ValueError("データは2次元配列である必要があります")
        
        # データ長の検証
        data_length = len(high)
        self._validate_period(self.period, data_length)
        
        # 正規化ADXの計算（高速化版）
        self._values = calculate_normalized_adx(
            high, low, close, self.period
        )
        
        return self._values 