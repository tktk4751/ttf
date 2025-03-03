#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from typing import Tuple
from numba import jit

from .indicator import Indicator
from .kama import KaufmanAdaptiveMA
from .atr import ATR


@jit(nopython=True)
def calculate_bollinger_bands(prices: np.ndarray, length: int, mult: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bollinger Bandsを計算（高速化版）
    
    Args:
        prices: 価格データ
        length: 期間
        mult: 乗数
        
    Returns:
        (upper, middle, lower) のタプル
    """
    # 中心線（単純移動平均）
    middle = np.full_like(prices, np.nan)
    for i in range(length - 1, len(prices)):
        middle[i] = np.mean(prices[i-length+1:i+1])
    
    # 標準偏差
    std = np.full_like(prices, np.nan)
    for i in range(length - 1, len(prices)):
        std[i] = np.std(prices[i-length+1:i+1])
    
    # 上下のバンド
    upper = middle + mult * std
    lower = middle - mult * std
    
    return upper, middle, lower


@jit(nopython=True)
def calculate_rolling_max_min(high: np.ndarray, low: np.ndarray, length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    期間内の最高値と最安値を計算（高速化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        length: 期間
        
    Returns:
        (highest_high, lowest_low) のタプル
    """
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    
    for i in range(length - 1, len(high)):
        highest_high[i] = np.max(high[i-length+1:i+1])
        lowest_low[i] = np.min(low[i-length+1:i+1])
    
    return highest_high, lowest_low


@jit(nopython=True)
def calculate_linear_regression(x: np.ndarray, length: int) -> np.ndarray:
    """
    線形回帰を計算（高速化版）
    
    Args:
        x: 入力配列
        length: 期間
        
    Returns:
        線形回帰の結果
    """
    result = np.full_like(x, np.nan)
    x_range = np.arange(length)
    
    for i in range(length - 1, len(x)):
        y = x[i-length+1:i+1]
        # 線形回帰の係数を計算
        x_mean = np.mean(x_range)
        y_mean = np.mean(y)
        numerator = np.sum((x_range - x_mean) * (y - y_mean))
        denominator = np.sum((x_range - x_mean) ** 2)
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        # 結果を計算
        result[i] = slope * (length - 1) + intercept
    
    return result


@jit(nopython=True)
def calculate_squeeze_states(
    bb_lower: np.ndarray,
    bb_upper: np.ndarray,
    kc_lower: np.ndarray,
    kc_upper: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    スクイーズ状態を計算（高速化版）
    
    Args:
        bb_lower: BBの下限
        bb_upper: BBの上限
        kc_lower: KCの下限
        kc_upper: KCの上限
        
    Returns:
        (sqz_on, sqz_off, no_sqz) のタプル
    """
    sqz_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    sqz_off = (bb_lower < kc_lower) & (bb_upper > kc_upper)
    no_sqz = (~sqz_on) & (~sqz_off)
    
    return sqz_on, sqz_off, no_sqz


class SqueezeMomentum(Indicator):
    """
    LazyBearのSqueeze Momentum Indicator（高速化版）
    
    Bollinger BandsとKeltner Channelsを使用して価格のスクイーズ状態を検出し、
    モメンタムの方向と強さを計算します。
    """
    
    def __init__(
        self,
        bb_length: int = 20,
        bb_mult: float = 2.0,
        kc_length: int = 20,
        kc_mult: float = 1.5,
    ):
        """
        コンストラクタ
        
        Args:
            bb_length: Bollinger Bands の期間
            bb_mult: Bollinger Bands の乗数
            kc_length: Keltner Channels の期間
            kc_mult: Keltner Channels の乗数
        """
        super().__init__("Squeeze Momentum")
        
        # パラメータの保存
        self.bb_length = bb_length
        self.bb_mult = bb_mult
        self.kc_length = kc_length
        self.kc_mult = kc_mult
        
        # インディケーターの初期化
        self.ma = KaufmanAdaptiveMA(period=bb_length)
        self.ma2 = KaufmanAdaptiveMA(period=kc_length)
        self.atr = ATR(period=kc_length)
        
        # 状態を保持する変数
        self._sqz_on: np.ndarray = None
        self._sqz_off: np.ndarray = None
        self._no_sqz: np.ndarray = None
    
    def calculate(self, df: pd.DataFrame) -> np.ndarray:
        """
        Squeeze Momentumインディケーターを計算（高速化版）
        
        Args:
            df: 価格データ（OHLC）を含むDataFrame
            
        Returns:
            モメンタム値の配列
        """
        # データの準備
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # Bollinger Bands の計算
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, self.bb_length, self.bb_mult)
        
        # Keltner Channels の計算
        kc_middle = self.ma2.calculate(close)
        atr = self.atr.calculate(df)
        kc_upper = kc_middle + self.kc_mult * atr
        kc_lower = kc_middle - self.kc_mult * atr
        
        # スクイーズ状態の判定
        self._sqz_on, self._sqz_off, self._no_sqz = calculate_squeeze_states(
            bb_lower, bb_upper, kc_lower, kc_upper
        )
        
        # モメンタム値の計算
        highest_high, lowest_low = calculate_rolling_max_min(high, low, self.kc_length)
        avg_hl = (highest_high + lowest_low) / 2
        avg_hlc = (avg_hl + kc_middle) / 2
        
        self._values = calculate_linear_regression(close - avg_hlc, self.kc_length)
        
        return self._values
    
    def get_squeeze_states(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        スクイーズ状態を取得
        
        Returns:
            (sqz_on, sqz_off, no_sqz) のタプル
            - sqz_on: スクイーズオン状態
            - sqz_off: スクイーズオフ状態
            - no_sqz: スクイーズなし状態
        """
        if self._values is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._sqz_on, self._sqz_off, self._no_sqz