#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any

import numpy as np
import pandas as pd

from .signal import Signal
from indicators.rsi import RSI

class RSIEntrySignal(Signal):
    """
    RSIを使用したエントリーシグナル
    - RSI <= rsi_long_entry: ロングエントリー (1)
    - RSI >= rsi_short_entry: ショートエントリー (-1)
    """
    
    def __init__(self, period: int = 2, solid: Dict[str, Any] = None):
        """
        コンストラクタ
        
        Args:
            period: RSIの期間
            params: パラメータ辞書
                - rsi_long_entry: ロングエントリーのRSIしきい値
                - rsi_short_entry: ショートエントリーのRSIしきい値
        """
        super().__init__(f"RSIEntry({period})")
        self.period = period
        self.solid = solid or {
            'rsi_long_entry': 20,
            'rsi_short_entry': 80
        }
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
        """
        rsi = RSI(self.period)
        rsi_values = rsi.calculate(data)
        
        # シグナルの初期化
        signals = np.zeros(len(rsi_values))
        
        # エントリーシグナルの生成
        signals = np.where(rsi_values <= self.solid['rsi_long_entry'], 1, signals)  # ロングエントリー
        signals = np.where(rsi_values >= self.solid['rsi_short_entry'], -1, signals)  # ショートエントリー
        
        return signals
