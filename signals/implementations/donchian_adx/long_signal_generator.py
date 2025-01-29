#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from numba import jit

from ...base_signal import BaseSignal
from ..donchian.entry import DonchianBreakoutEntrySignal
from ..adx.filter import ADXFilterSignal

@jit(nopython=True)
def calculate_entry_signals(donchian_signals: np.ndarray, adx_signals: np.ndarray) -> np.ndarray:
    """エントリーシグナルを一度に計算（高速化版）"""
    signals = np.zeros_like(donchian_signals)
    
    for i in range(len(signals)):
        # ドンチャンのロングシグナルかつADXフィルターが有効な場合
        if donchian_signals[i] == 1 and adx_signals[i] == 1:
            signals[i] = 1
    
    return signals.astype(np.int8)

class DonchianADXLongSignalGenerator(BaseSignal):
    """
    ドンチャン+ADXのロングシグナル生成器
    
    エントリー条件:
    - ドンチャンブレイクアウトの買いシグナル
    - ADXフィルターが有効
    
    エグジット条件:
    - ドンチャンブレイクアウトの売りシグナル
    """
    
    def __init__(
        self,
        donchian_period: int = 20,
        adx_period: int = 14,
        adx_threshold: float = 25.0
    ):
        """
        初期化
        
        Args:
            donchian_period: ドンチャンチャネルの期間
            adx_period: ADXの期間
            adx_threshold: ADXの閾値
        """
        super().__init__("DonchianADXLongSignalGenerator", {
            'donchian_period': donchian_period,
            'adx_period': adx_period,
            'adx_threshold': adx_threshold
        })
        
        # シグナル生成器の初期化
        self.donchian_signal = DonchianBreakoutEntrySignal(period=donchian_period)
        self.adx_signal = ADXFilterSignal(period=adx_period, threshold=adx_threshold)
        
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
            donchian_signals = self.donchian_signal.generate(data)
            adx_signals = self.adx_signal.generate(data)
            
            # エントリーシグナルの計算（ベクトル化）
            self._signals = calculate_entry_signals(donchian_signals, adx_signals)
            
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