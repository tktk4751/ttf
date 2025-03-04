#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd
from numba import jit

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.tfc import TFC


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


class TFCEntrySignal(BaseSignal, IEntrySignal):
    """
    KAMAケルトナーTFCのトレンド転換によるエントリーシグナル
    
    - ダウントレンドからアップトレンドへの転換: ロングエントリー (1)
    - アップトレンドからダウントレンドへの転換: ショートエントリー (-1)
    """
    
    def __init__(
        self,
        kama_period: int = 10,
        kama_fast: int = 2,
        kama_slow: int = 30,
        atr_period: int = 10,
        upper_multiplier: float = 2.0,
        lower_multiplier: float = 2.0
    ):
        """
        コンストラクタ
        
        Args:
            kama_period: KAMAの効率比の計算期間（デフォルト: 10）
            kama_fast: KAMAの速い移動平均の期間（デフォルト: 2）
            kama_slow: KAMAの遅い移動平均の期間（デフォルト: 30）
            atr_period: ATRの期間（デフォルト: 10）
            upper_multiplier: アッパーバンドのATR乗数（デフォルト: 2.0）
            lower_multiplier: ロワーバンドのATR乗数（デフォルト: 2.0）
        """
        params = {
            'kama_period': kama_period,
            'kama_fast': kama_fast,
            'kama_slow': kama_slow,
            'atr_period': atr_period,
            'upper_multiplier': upper_multiplier,
            'lower_multiplier': lower_multiplier
        }
        super().__init__(
            f"KAMAKeltnerTFCTrendChange({kama_period}, {kama_fast}, {kama_slow}, {atr_period}, {upper_multiplier}, {lower_multiplier})",
            params
        )
        self._tfc = TFC(
            kama_period=kama_period,
            kama_fast=kama_fast,
            kama_slow=kama_slow,
            atr_period=atr_period,
            upper_multiplier=upper_multiplier,
            lower_multiplier=lower_multiplier
        )
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロングエントリー, -1: ショートエントリー, 0: シグナルなし)
        """
        # TFCの計算
        self._tfc.calculate(data)
        
        # トレンド方向の取得
        trend = self._tfc.get_trend()
        
        # トレンド転換シグナルの計算（高速化版）
        return calculate_trend_change_signals(trend) 