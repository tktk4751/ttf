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
    h_l = high - low
    h_pc = np.abs(high - prev_close)
    l_pc = np.abs(low - prev_close)
    
    tr = np.maximum(h_l, h_pc)
    tr = np.maximum(tr, l_pc)
    
    return tr


@jit(nopython=True)
def calculate_normalized_chop(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """
    トレンド指数を計算する（高速化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
        period: 計算期間
    
    Returns:
        トレンド指数の配列（0-1の範囲で正規化）
        - 1に近いほどトレンド相場
        - 0に近いほどレンジ相場
    """
    length = len(high)
    result = np.full(length, np.nan)
    
    # True Rangeの計算
    tr = np.zeros(length)
    tr[0] = high[0] - low[0]  # 初日のTRは単純なレンジ
    tr[1:] = calculate_tr(high[1:], low[1:], close[:-1])
    
    # EMATRの計算
    alpha = 2.0 / (period + 1.0)
    ema_tr = np.zeros(length)
    ema_tr[0] = tr[0]
    
    for i in range(1, length):
        ema_tr[i] = (tr[i] * alpha) + (ema_tr[i-1] * (1 - alpha))
    
    atr_sum = ema_tr * period
    
    # 期間ごとの最大値と最小値の計算
    for i in range(period-1, length):
        period_high = np.max(high[i-period+1:i+1])
        period_low = np.min(low[i-period+1:i+1])
        range_sum = period_high - period_low
        
        if range_sum != 0:  # ゼロ除算を防ぐ
            # チョピネスの計算と正規化（0-1の範囲に）
            chop = np.log10(atr_sum[i] / range_sum) / np.log10(period)
            # 100を超えることはないが、念のため制限
            chop = min(max(chop, 0.0), 1.0)
            # トレンド指数として反転（1 - chop）
            result[i] = 1.0 - chop
    
    return result


class NormalizedChop(Indicator):
    """
    トレンド指数（Trend Index）
    
    チョピネス指数を反転させ、0-1の範囲で正規化したインジケーター
    - 1に近いほどトレンド相場（チョピネスが低い）
    - 0に近いほどレンジ相場（チョピネスが高い）
    
    使用方法：
    - 0.618以上: 強いトレンド相場
    - 0.382以下: 強いレンジ相場
    """
    
    def __init__(self, period: int = 14):
        """
        コンストラクタ
        
        Args:
            period: 計算期間（デフォルト: 14）
        """
        super().__init__(f"TREND({period})")
        self.period = period
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        トレンド指数を計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'high'と'low'と'close'カラムが必要
        
        Returns:
            トレンド指数の配列（0-1の範囲）
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
        
        # トレンド指数の計算（高速化版）
        self._values = calculate_normalized_chop(high, low, close, self.period)
        
        return self._values 