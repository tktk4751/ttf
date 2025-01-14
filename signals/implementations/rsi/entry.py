#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.rsi import RSI

class RSIEntrySignal(BaseSignal, IEntrySignal):
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
            solid: パラメータ辞書
                - rsi_long_entry: ロングエントリーのRSIしきい値
                - rsi_short_entry: ショートエントリーのRSIしきい値
        """
        params = {
            'period': period,
            'solid': solid or {
                'rsi_long_entry': 20,
                'rsi_short_entry': 80
            }
        }
        super().__init__(f"RSIEntry({period})", params)
        self._rsi = RSI(period)
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
        """
        rsi_values = self._rsi.calculate(data)
        
        # シグナルの初期化
        signals = np.zeros(len(rsi_values))
        
        # エントリーシグナルの生成
        solid = self._params['solid']
        signals = np.where(rsi_values <= solid['rsi_long_entry'], 1, signals)  # ロングエントリー
        signals = np.where(rsi_values >= solid['rsi_short_entry'], -1, signals)  # ショートエントリー
        
        return signals 