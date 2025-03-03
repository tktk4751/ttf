#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from numba import jit

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.supertrend.direction import SupertrendDirectionSignal
from signals.implementations.donchian.entry import DonchianBreakoutEntrySignal

@jit(nopython=True)
def calculate_entry_signals(supertrend_signals: np.ndarray, donchian_signals: np.ndarray) -> np.ndarray:
    """エントリーシグナルを一度に計算（高速化版）"""
    signals = np.zeros_like(supertrend_signals)
    
    for i in range(len(signals)):
        # スーパートレンドが1（上昇トレンド）かつドンチャンが-1の場合
        if supertrend_signals[i] == 1 and donchian_signals[i] == -1:
            signals[i] = 1
    
    return signals.astype(np.int8)

class SupertrendDonchianLongSignalGenerator(BaseSignalGenerator):
    """
    スーパートレンド+ドンチャンのシグナル生成クラス（買い専用・高速化版）
    
    エントリー条件:
    - スーパートレンドが1（上昇トレンド）
    - ドンチャンシグナルが-1
    
    エグジット条件:
    - スーパートレンドが-1（下降トレンド）
    """
    
    def __init__(
        self,
        supertrend_period: int = 10,
        supertrend_multiplier: float = 3.0,
        donchian_period: int = 20,
    ):
        """初期化"""
        super().__init__("SupertrendDonchianLongSignalGenerator")
        
        # シグナル生成器の初期化
        self.supertrend_signal = SupertrendDirectionSignal(
            period=supertrend_period,
            multiplier=supertrend_multiplier
        )
        self.donchian_signal = DonchianBreakoutEntrySignal(
            period=donchian_period
        )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._signals = None
        self._supertrend_signals = None
        self._donchian_signals = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算（高速化版）"""
        current_len = len(data)
        
        # データ長が変わった場合のみ再計算
        if self._signals is None or current_len != self._data_len:
            # 各シグナルの計算
            supertrend_signals = self.supertrend_signal.generate(data)
            donchian_signals = self.donchian_signal.generate(data)
            
            # エントリーシグナルの計算（ベクトル化）
            self._signals = calculate_entry_signals(supertrend_signals, donchian_signals)
            
            # エグジット用にシグナルを保存
            self._supertrend_signals = supertrend_signals
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
        
        # スーパートレンドが-1になったらエグジット
        return bool(self._supertrend_signals[index] == -1) 