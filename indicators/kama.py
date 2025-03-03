#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union
import numpy as np
import pandas as pd
from numba import jit

from .indicator import Indicator


@jit(nopython=True)
def calculate_efficiency_ratio(change: np.ndarray, volatility: np.ndarray) -> np.ndarray:
    """
    効率比（Efficiency Ratio）を計算する（高速化版）
    
    Args:
        change: 価格変化（終値の差分）の配列
        volatility: ボラティリティ（価格変化の絶対値の合計）の配列
    
    Returns:
        効率比の配列
    """
    return np.abs(change) / (volatility + 1e-10)  # ゼロ除算を防ぐ


@jit(nopython=True)
def calculate_smoothing_constant(er: np.ndarray, fast: float, slow: float) -> np.ndarray:
    """
    スムージング定数を計算する（高速化版）
    
    Args:
        er: 効率比の配列
        fast: 速い移動平均の期間から計算した定数
        slow: 遅い移動平均の期間から計算した定数
    
    Returns:
        スムージング定数の配列
    """
    return (er * (fast - slow) + slow) ** 2


@jit(nopython=True)
def calculate_kama(close: np.ndarray, period: int, fast_period: int, slow_period: int) -> np.ndarray:
    """
    カウフマン適応移動平均線を計算する（高速化版）
    
    Args:
        close: 終値の配列
        period: 効率比の計算期間
        fast_period: 速い移動平均の期間
        slow_period: 遅い移動平均の期間
    
    Returns:
        KAMAの配列
    """
    length = len(close)
    kama = np.full(length, np.nan)
    
    # 最初のKAMAは単純な平均値
    kama[period-1] = np.mean(close[:period])
    
    # 定数の計算
    fast = 2.0 / (fast_period + 1.0)
    slow = 2.0 / (slow_period + 1.0)
    
    # 各時点でのKAMAを計算
    for i in range(period, length):
        # 価格変化とボラティリティの計算
        change = close[i] - close[i-period]
        volatility = np.sum(np.abs(np.diff(close[i-period:i+1])))
        
        # 効率比の計算
        er = abs(change) / (volatility + 1e-10)
        
        # スムージング定数の計算
        sc = (er * (fast - slow) + slow) ** 2
        
        # KAMAの計算
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama


class KaufmanAdaptiveMA(Indicator):
    """
    カウフマン適応移動平均線（KAMA）インジケーター
    
    価格のトレンドの効率性に基づいて、移動平均線の感度を自動的に調整する
    - トレンドが強い時：速い移動平均に近づく
    - トレンドが弱い時：遅い移動平均に近づく
    """
    
    def __init__(self, period: int = 10, fast_period: int = 2, slow_period: int = 30):
        """
        コンストラクタ
        
        Args:
            period: 効率比の計算期間（デフォルト: 10）
            fast_period: 速い移動平均の期間（デフォルト: 2）
            slow_period: 遅い移動平均の期間（デフォルト: 30）
        """
        super().__init__(f"KAMA({period}, {fast_period}, {slow_period})")
        self.period = period
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        KAMAを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'close'カラムが必要
        
        Returns:
            KAMAの値を返す
        """
        # データの検証と変換
        if isinstance(data, pd.DataFrame):
            if 'close' not in data.columns:
                raise ValueError("DataFrameには'close'カラムが必要です")
            close = data['close'].values
        else:
            if data.ndim == 2:
                close = data[:, 3]  # close
            else:
                close = data  # 1次元配列として扱う
        
        # データ長の検証
        data_length = len(close)
        self._validate_period(self.period, data_length)
        
        # KAMAの計算（高速化版）
        self._values = calculate_kama(
            close,
            self.period,
            self.fast_period,
            self.slow_period
        )
        
        return self._values 