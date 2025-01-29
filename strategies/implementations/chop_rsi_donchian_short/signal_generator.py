#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from numba import jit

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.chop.filter import ChopFilterSignal
from signals.implementations.rsi.entry import RSIEntrySignal
from signals.implementations.donchian.entry import DonchianBreakoutEntrySignal

@jit(nopython=True)
def calculate_entry_signals(chop_signals: np.ndarray, rsi_signals: np.ndarray, donchian_signals: np.ndarray) -> np.ndarray:
    """エントリーシグナルを一度に計算（高速化版）"""
    signals = np.zeros_like(chop_signals)
    
    for i in range(len(signals)):
        # CHOPフィルターがレンジ相場、RSIが売りシグナル、ドンチャンシグナルが0の場合
        if chop_signals[i] == -1 and rsi_signals[i] == -1 and donchian_signals[i] == 0:
            signals[i] = -1
    
    return signals.astype(np.int8)

class ChopRSIDonchianShortSignalGenerator(BaseSignalGenerator):
    """
    CHOP+RSI+ドンチャンのシグナル生成クラス（売り専用・高速化版）
    
    エントリー条件:
    - CHOPフィルターがレンジ相場を示している
    - RSIが売りシグナル
    - ドンチャンシグナルが0
    
    エグジット条件:
    - ドンチャンブレイクアウトの買いシグナル
    """
    
    def __init__(
        self,
        chop_period: int = 14,
        chop_threshold: float = 50.0,
        rsi_period: int = 14,
        donchian_period: int = 20,
    ):
        """初期化"""
        super().__init__("ChopRSIDonchianShortSignalGenerator")
        
        # シグナル生成器の初期化
        self.chop_signal = ChopFilterSignal(
            period=chop_period,
            solid={'chop_solid': chop_threshold}
        )
        self.rsi_signal = RSIEntrySignal(
            period=rsi_period,
        )
        self.donchian_signal = DonchianBreakoutEntrySignal(
            period=donchian_period
        )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._signals = None
        self._donchian_signals = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算（高速化版）"""
        current_len = len(data)
        
        # データ長が変わった場合のみ再計算
        if self._signals is None or current_len != self._data_len:
            # 各シグナルの計算
            chop_signals = self.chop_signal.generate(data)
            rsi_signals = self.rsi_signal.generate(data)
            donchian_signals = self.donchian_signal.generate(data)
            
            # エントリーシグナルの計算（ベクトル化）
            self._signals = calculate_entry_signals(chop_signals, rsi_signals, donchian_signals)
            
            # エグジット用にドンチャンシグナルを保存
            self._donchian_signals = donchian_signals
            
            self._data_len = current_len
    
    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エントリーシグナル取得（高速化版）"""
        if self._signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._signals
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """エグジットシグナル生成（高速化版）"""
        if position != -1:  # ショートポジションのみ
            return False
        
        if self._signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        
        if index == -1:
            index = len(data) - 1
        
        # ドンチャンの買いシグナルでエグジット
        return bool(self._donchian_signals[index] == 1) 