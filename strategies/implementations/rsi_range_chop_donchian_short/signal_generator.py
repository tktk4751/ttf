#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from numba import jit

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.rsi.filter import RSIRangeFilterSignal
from signals.implementations.chop.filter import ChopFilterSignal
from signals.implementations.donchian.entry import DonchianBreakoutEntrySignal
@jit(nopython=True)
def calculate_entry_signals(rsi_signals: np.ndarray, chop_signals: np.ndarray, donchian_signals: np.ndarray) -> np.ndarray:
    """エントリーシグナルを一度に計算（高速化版）"""
    signals = np.zeros_like(rsi_signals)
    
    for i in range(len(signals)):
        # RSIレンジフィルターが1、CHOPフィルターが-1、ドンチャンが1の場合
        if rsi_signals[i] == 1 and chop_signals[i] == -1 and donchian_signals[i] == 1:
            signals[i] = -1
    
    return signals.astype(np.int8)

class RSIRangeChopDonchianShortSignalGenerator(BaseSignalGenerator):
    """
    RSIレンジ+CHOP+ドンチャンのシグナル生成クラス（売り専用・高速化版）
    
    エントリー条件:
    - RSIレンジフィルターが1（レンジ相場）
    - CHOPフィルターが-1（トレンド相場）
    - ドンチャンシグナルが1
    
    エグジット条件:
    - ドンチャンシグナルが再度1
    もしくは
    - ドンチャンシグナルが-1
    """
    
    def __init__(
        self,
        rsi_period: int = 14,
        chop_period: int = 14,
        chop_threshold: float = 50.0,
        donchian_period: int = 20,
    ):
        """初期化"""
        super().__init__("RSIRangeChopDonchianShortSignalGenerator")
        
        # シグナル生成器の初期化
        self.rsi_signal = RSIRangeFilterSignal(
            period=rsi_period
        )
        self.chop_signal = ChopFilterSignal(
            period=chop_period,
            solid={'chop_solid': chop_threshold}
        )
        self.donchian_signal = DonchianBreakoutEntrySignal(
            period=donchian_period
        )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._signals = None
        self._rsi_signals = None
        self._chop_signals = None
        self._donchian_signals = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算（高速化版）"""
        current_len = len(data)
        
        # データ長が変わった場合のみ再計算
        if self._signals is None or current_len != self._data_len:
            # 各シグナルの計算
            rsi_signals = self.rsi_signal.generate(data)
            chop_signals = self.chop_signal.generate(data)
            donchian_signals = self.donchian_signal.generate(data)
            
            # エントリーシグナルの計算（ベクトル化）
            self._signals = calculate_entry_signals(rsi_signals, chop_signals, donchian_signals)
            
            # エグジット用にシグナルを保存
            self._rsi_signals = rsi_signals
            self._chop_signals = chop_signals
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
        
        # ドンチャンシグナルが再度1、もしくは-1になったらエグジット
        return bool(self._donchian_signals[index] == 1 or self._donchian_signals[index] == -1) 