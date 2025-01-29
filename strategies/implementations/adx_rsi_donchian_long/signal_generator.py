#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from numba import jit

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.adx.filter import ADXFilterSignal
from signals.implementations.rsi.entry import RSIEntrySignal
from signals.implementations.donchian.entry import DonchianBreakoutEntrySignal

@jit(nopython=True)
def calculate_entry_signals(adx_signals: np.ndarray, rsi_signals: np.ndarray, donchian_signals: np.ndarray) -> np.ndarray:
    """エントリーシグナルを一度に計算（高速化版）"""
    signals = np.zeros_like(adx_signals)
    
    for i in range(len(signals)):
        # ADXフィルターがトレンド相場、RSIが買いシグナル、ドンチャンシグナルが0の場合
        if adx_signals[i] == 1 and rsi_signals[i] == 1 and donchian_signals[i] == 0:
            signals[i] = 1
    
    return signals.astype(np.int8)

class ADXRSIDonchianLongSignalGenerator(BaseSignalGenerator):
    """
    ADX+RSI+ドンチャンのシグナル生成クラス（買い専用・高速化版）
    
    エントリー条件:
    - ADXフィルターがトレンド相場を示している
    - RSIが買いシグナル
    - ドンチャンシグナルが0
    
    エグジット条件:
    - ドンチャンブレイクアウトの売りシグナル
    """
    
    def __init__(
        self,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        rsi_period: int = 14,
        rsi_upper: float = 70.0,
        rsi_lower: float = 30.0,
        donchian_period: int = 20,
    ):
        """初期化"""
        super().__init__("ADXRSIDonchianLongSignalGenerator")
        
        # シグナル生成器の初期化
        self.adx_signal = ADXFilterSignal(
            period=adx_period,
            solid={'adx_solid': adx_threshold}
        )
        self.rsi_signal = RSIEntrySignal(
            period=rsi_period,
            upper=rsi_upper,
            lower=rsi_lower
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
            adx_signals = self.adx_signal.generate(data)
            rsi_signals = self.rsi_signal.generate(data)
            donchian_signals = self.donchian_signal.generate(data)
            
            # エントリーシグナルの計算（ベクトル化）
            self._signals = calculate_entry_signals(adx_signals, rsi_signals, donchian_signals)
            
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
        if position != 1:  # ロングポジションのみ
            return False
        
        if self._signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        
        if index == -1:
            index = len(data) - 1
        
        # ドンチャンの売りシグナルでエグジット
        return bool(self._donchian_signals[index] == -1) 