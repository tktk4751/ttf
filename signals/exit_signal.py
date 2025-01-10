#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any

import numpy as np
import pandas as pd

from .signal import Signal
from indicators.rsi import RSI

class RSIExitSignal(Signal):
    """
    RSIを使用したエグジットシグナル
    - ロングエグジット: RSI(t-1) >= exit_solid かつ RSI(t) <= exit_solid
    - ショートエグジット: RSI(t-1) <= exit_solid かつ RSI(t) >= exit_solid
    """
    
    def __init__(self, period: int = 14, solid: Dict[str, Any] = None):
        """
        コンストラクタ
        
        Args:
            period: RSIの期間
            params: パラメータ辞書
                - rsi_long_exit_solid: ロングエグジットのRSIしきい値
                - rsi_short_exit_solid: ショートエグジットのRSIしきい値
        """
        super().__init__(f"RSIExit({period})")
        self.period = period
        self.solid = solid or {
            'rsi_long_exit_solid': 70,
            'rsi_short_exit_solid': 30
        }
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロングエグジット, -1: ショートエグジット, 0: シグナルなし)
        """
        rsi = RSI(self.period)
        rsi_values = pd.Series(rsi.calculate(data))
        
        # シグナルの初期化
        signals = np.zeros(len(rsi_values))
        
        # エグジットシグナルの生成
        # ロングポジションのエグジット条件
        long_exit = (rsi_values.shift(1) >= self.solid['rsi_long_exit_solid']) & \
                   (rsi_values <= self.solid['rsi_long_exit_solid'])
        
        # ショートポジションのエグジット条件
        short_exit = (rsi_values.shift(1) <= self.solid['rsi_short_exit_solid']) & \
                    (rsi_values >= self.solid['rsi_short_exit_solid'])
        
        # シグナルの設定
        signals = np.where(long_exit, 1, signals)
        signals = np.where(short_exit, -1, signals)
        
        return signals
