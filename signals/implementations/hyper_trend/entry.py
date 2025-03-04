#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd
from numba import jit

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.hyper_trend import HyperTrend


@jit(nopython=True)
def calculate_trend_change_signals(trend: np.ndarray) -> np.ndarray:
    """
    トレンド転換シグナルを計算する（高速化版）
    
    Args:
        trend: トレンド方向の配列（1: アップトレンド、-1: ダウントレンド）
    
    Returns:
        シグナルの配列（1: ロングエントリー、-1: ショートエントリー、0: シグナルなし）
    """
    length = len(trend)
    signals = np.zeros(length, dtype=np.int8)
    
    # 最初のバーはシグナルなし
    signals[0] = 0
    
    # トレンド転換の判定
    for i in range(1, length):
        # ダウントレンドからアップトレンドへの転換
        if trend[i-1] == -1 and trend[i] == 1:
            signals[i] = 1
        # アップトレンドからダウントレンドへの転換
        elif trend[i-1] == 1 and trend[i] == -1:
            signals[i] = -1
    
    return signals


class HyperTrendEntrySignal(BaseSignal, IEntrySignal):
    """
    HyperTrendのトレンド転換によるエントリーシグナル
    
    エントリー条件：
    - ダウントレンドからアップトレンドへの転換: ロングエントリー (1)
    - アップトレンドからダウントレンドへの転換: ショートエントリー (-1)
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
        params = {
            'er_period': er_period,
            'max_percentile_length': max_percentile_length,
            'min_percentile_length': min_percentile_length,
            'max_atr_period': max_atr_period,
            'min_atr_period': min_atr_period,
            'max_multiplier': max_multiplier,
            'min_multiplier': min_multiplier
        }
        super().__init__(
            f"HyperTrendEntry({er_period}, {max_percentile_length}, {min_percentile_length}, "
            f"{max_atr_period}, {min_atr_period}, {max_multiplier}, {min_multiplier})",
            params
        )
        self._hyper_trend = HyperTrend(
            er_period=er_period,
            max_percentile_length=max_percentile_length,
            min_percentile_length=min_percentile_length,
            max_atr_period=max_atr_period,
            min_atr_period=min_atr_period,
            max_multiplier=max_multiplier,
            min_multiplier=min_multiplier
        )
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロングエントリー, -1: ショートエントリー, 0: シグナルなし)
        """
        # HyperTrendの計算
        result = self._hyper_trend.calculate(data)
        
        # トレンド転換シグナルの計算（高速化版）
        return calculate_trend_change_signals(result.trend) 