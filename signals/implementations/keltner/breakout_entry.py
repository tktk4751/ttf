#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd
from numba import jit

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.keltner import KeltnerChannel


@jit(nopython=True)
def calculate_breakout_signals(close: np.ndarray, middle: np.ndarray, upper: np.ndarray, lower: np.ndarray, period: int) -> np.ndarray:
    """
    ブレイクアウトシグナルを計算する（高速化版）
    
    Args:
        close: 終値の配列
        middle: 中心線の配列
        upper: アッパーバンドの配列
        lower: ロワーバンドの配列
        period: 期間
    
    Returns:
        シグナルの配列
    """
    length = len(close)
    signals = np.zeros(length, dtype=np.int8)
    
    # 最初の期間はシグナルなし
    signals[:period] = 0
    
    # ブレイクアウトの判定
    for i in range(period, length):
        # ロングエントリー: 終値がアッパーバンドを上回る
        if close[i] > upper[i-1]:
            signals[i] = 1
        # ショートエントリー: 終値がロワーバンドを下回る
        elif close[i] < lower[i-1]:
            signals[i] = -1
    
    return signals


class KeltnerBreakoutEntrySignal(BaseSignal, IEntrySignal):
    """
    ケルトナーチャネルのブレイクアウトによるエントリーシグナル
    
    - 現在の終値が前回のアッパーバンドを上回った場合: ロングエントリー (1)
    - 現在の終値が前回のロワーバンドを下回った場合: ショートエントリー (-1)
    """
    
    def __init__(
        self,
        period: int = 20,
        atr_period: int = 10,
        upper_multiplier: float = 2.0,
        lower_multiplier: float = 2.0
    ):
        """
        コンストラクタ
        
        Args:
            period: EMAの期間（デフォルト: 20）
            atr_period: ATRの期間（デフォルト: 10）
            upper_multiplier: アッパーバンドのATR乗数（デフォルト: 2.0）
            lower_multiplier: ロワーバンドのATR乗数（デフォルト: 2.0）
        """
        params = {
            'period': period,
            'atr_period': atr_period,
            'upper_multiplier': upper_multiplier,
            'lower_multiplier': lower_multiplier
        }
        super().__init__(
            f"KeltnerBreakout({period}, {atr_period}, {upper_multiplier}, {lower_multiplier})",
            params
        )
        self._keltner = KeltnerChannel(period, atr_period, max(upper_multiplier, lower_multiplier))
        self.upper_multiplier = upper_multiplier
        self.lower_multiplier = lower_multiplier
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
        """
        # ケルトナーチャネルの計算
        result = self._keltner.calculate(data)
        if result is None:
            return np.zeros(len(data))
        
        # 終値の取得
        close = data['close'].values if isinstance(data, pd.DataFrame) else data[:, 3]
        
        # ATRの取得
        atr = self._keltner.atr.calculate(data)
        
        # 独自のバンドを計算
        upper = result.middle + (self.upper_multiplier * atr)
        lower = result.middle - (self.lower_multiplier * atr)
        
        # ブレイクアウトシグナルの計算（高速化版）
        return calculate_breakout_signals(
            close,
            result.middle,
            upper,
            lower,
            self._params['period']
        )