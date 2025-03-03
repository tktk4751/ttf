#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd
from numba import jit

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.alma_keltner import ALMAKeltnerChannel


@jit(nopython=True)
def calculate_breakout_signals(close: np.ndarray, upper: np.ndarray, lower: np.ndarray, period: int) -> np.ndarray:
    """
    ブレイクアウトシグナルを計算する（高速化版）
    
    Args:
        close: 終値の配列
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


class ALMAKeltnerBreakoutEntrySignal(BaseSignal, IEntrySignal):
    """
    ALMAケルトナーチャネルのブレイクアウトによるエントリーシグナル
    
    - 現在の終値が前回のアッパーバンドを上回った場合: ロングエントリー (1)
    - 現在の終値が前回のロワーバンドを下回った場合: ショートエントリー (-1)
    """
    
    def __init__(
        self,
        alma_period: int = 9,
        alma_offset: float = 0.85,
        alma_sigma: float = 6,
        atr_period: int = 10,
        upper_multiplier: float = 2.0,
        lower_multiplier: float = 2.0
    ):
        """
        コンストラクタ
        
        Args:
            alma_period: ALMAの期間（デフォルト: 9）
            alma_offset: ALMAのオフセット（デフォルト: 0.85）
            alma_sigma: ALMAのシグマ（デフォルト: 6）
            atr_period: ATRの期間（デフォルト: 10）
            upper_multiplier: アッパーバンドのATR乗数（デフォルト: 2.0）
            lower_multiplier: ロワーバンドのATR乗数（デフォルト: 2.0）
        """
        params = {
            'alma_period': alma_period,
            'alma_offset': alma_offset,
            'alma_sigma': alma_sigma,
            'atr_period': atr_period,
            'upper_multiplier': upper_multiplier,
            'lower_multiplier': lower_multiplier
        }
        super().__init__(
            f"ALMAKeltnerBreakout({alma_period}, {alma_offset}, {alma_sigma}, {atr_period}, {upper_multiplier}, {lower_multiplier})",
            params
        )
        self._alma_keltner = ALMAKeltnerChannel(
            alma_period=alma_period,
            alma_offset=alma_offset,
            alma_sigma=alma_sigma,
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
            シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
        """
        # ALMAケルトナーチャネルの計算
        result = self._alma_keltner.calculate(data)
        if result is None:
            return np.zeros(len(data))
        
        # 終値の取得
        close = data['close'].values if isinstance(data, pd.DataFrame) else data[:, 3]
        
        # バンドの取得
        _, upper, lower, _, _ = self._alma_keltner.get_bands()
        
        # ブレイクアウトシグナルの計算（高速化版）
        return calculate_breakout_signals(
            close,
            upper,
            lower,
            self._params['alma_period']
        ) 