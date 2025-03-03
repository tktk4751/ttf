#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union
import numpy as np
import pandas as pd
from numba import jit

from .indicator import Indicator


@dataclass
class MACDResult:
    """MACDの計算結果"""
    macd: np.ndarray       # MACD線
    signal: np.ndarray     # シグナル線
    histogram: np.ndarray  # ヒストグラム


@jit(nopython=True)
def calculate_ema(data: np.ndarray, period: int) -> np.ndarray:
    """
    指数移動平均を計算する（高速化版）
    
    Args:
        data: 価格データ
        period: 期間
    
    Returns:
        np.ndarray: EMAの配列
    """
    alpha = 2.0 / (period + 1)
    ema = np.zeros_like(data)
    ema[0] = data[0]  # 最初の値で初期化
    
    # EMAの計算
    for i in range(1, len(data)):
        ema[i] = data[i] * alpha + ema[i-1] * (1 - alpha)
    
    return ema


@jit(nopython=True)
def calculate_macd(
    close: np.ndarray,
    fast_period: int,
    slow_period: int,
    signal_period: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MACDを計算する（高速化版）
    
    Args:
        close: 終値の配列
        fast_period: 短期EMAの期間
        slow_period: 長期EMAの期間
        signal_period: シグナルEMAの期間
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: MACD線、シグナル線、ヒストグラムの配列
    """
    # 短期と長期のEMAを計算
    fast_ema = calculate_ema(close, fast_period)
    slow_ema = calculate_ema(close, slow_period)
    
    # MACD線を計算
    macd_line = fast_ema - slow_ema
    
    # シグナル線を計算
    signal_line = calculate_ema(macd_line, signal_period)
    
    # ヒストグラムを計算
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


class MACD(Indicator):
    """
    MACDインジケーター（高速化版）
    2つの指数移動平均線の差を使用して、トレンドの方向と強さを判断する
    """
    
    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ):
        """
        コンストラクタ
        
        Args:
            fast_period: 短期EMAの期間
            slow_period: 長期EMAの期間
            signal_period: シグナルEMAの期間
        """
        super().__init__(f"MACD({fast_period}, {slow_period}, {signal_period})")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> MACDResult:
        """
        MACDを計算する
        
        Args:
            data: 価格データ
        
        Returns:
            MACDの計算結果
        """
        # データの準備
        if isinstance(data, pd.DataFrame):
            close = data['close'].to_numpy()
        else:
            close = data[:, 3]  # close
        
        # MACDの計算（高速化版）
        macd_line, signal_line, histogram = calculate_macd(
            close,
            self.fast_period,
            self.slow_period,
            self.signal_period
        )
        
        self._values = histogram  # 基底クラスの要件を満たすため
        
        return MACDResult(
            macd=macd_line,
            signal=signal_line,
            histogram=histogram
        ) 