#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from numba import jit

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.supertrend.direction import SupertrendDirectionSignal
from signals.implementations.adx.filter import ADXFilterSignal
from signals.implementations.rsi.entry import RSIEntrySignal

@jit(nopython=True)
def calculate_entry_signals(supertrend_signals: np.ndarray, adx_signals: np.ndarray, rsi_signals: np.ndarray) -> np.ndarray:
    """エントリーシグナルを一度に計算（高速化版）"""
    signals = np.zeros_like(supertrend_signals)
    
    for i in range(len(signals)):
        # スーパートレンドが買い、ADXフィルターが有効、RSIが買いシグナルの場合
        if supertrend_signals[i] == 1 and adx_signals[i] == 1 and rsi_signals[i] == 1:
            signals[i] = 1
    
    return signals.astype(np.int8)

class SupertrendADXRSILongSignalGenerator(BaseSignalGenerator):
    """
    スーパートレンド+ADX+RSIのシグナル生成クラス（買い専用・高速化版）
    
    エントリー条件:
    - スーパートレンドが買いシグナル
    - ADXフィルターが有効
    - RSIが買いシグナル
    
    エグジット条件:
    - スーパートレンドが売りシグナル
    """
    
    def __init__(
        self,
        supertrend_period: int = 10,
        supertrend_multiplier: float = 3.0,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        rsi_period: int = 14,
        rsi_upper: float = 70.0,
        rsi_lower: float = 30.0,
    ):
        """初期化"""
        super().__init__("SupertrendADXRSILongSignalGenerator")
        
        # シグナル生成器の初期化
        self.supertrend_signal = SupertrendDirectionSignal(
            period=supertrend_period,
            multiplier=supertrend_multiplier
        )
        self.adx_signal = ADXFilterSignal(
            period=adx_period,
            threshold=adx_threshold
        )
        self.rsi_signal = RSIEntrySignal(
            period=rsi_period,

        )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._signals = None
        self._supertrend_signals = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算（高速化版）"""
        current_len = len(data)
        
        # データ長が変わった場合のみ再計算
        if self._signals is None or current_len != self._data_len:
            # 各シグナルの計算
            supertrend_signals = self.supertrend_signal.generate(data)
            adx_signals = self.adx_signal.generate(data)
            rsi_signals = self.rsi_signal.generate(data)
            
            # エントリーシグナルの計算（ベクトル化）
            self._signals = calculate_entry_signals(supertrend_signals, adx_signals, rsi_signals)
            
            # エグジット用にスーパートレンドシグナルを保存
            self._supertrend_signals = supertrend_signals
            
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
        
        # スーパートレンドの売りシグナルでエグジット
        return bool(self._supertrend_signals[index] == -1) 