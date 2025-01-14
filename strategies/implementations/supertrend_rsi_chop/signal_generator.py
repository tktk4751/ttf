#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd

from signals.implementations.supertrend.direction import SupertrendDirectionSignal
from signals.implementations.rsi.entry import RSIEntrySignal
from signals.implementations.rsi.exit import RSIExitSignal
from signals.implementations.chop.filter import ChopFilterSignal

class SupertrendRsiChopSignalGenerator:
    """スーパートレンド、RSI、Chopを組み合わせたシグナル生成器"""
    
    def __init__(
        self,
        supertrend_params: Dict[str, Any],
        rsi_entry_params: Dict[str, Any],
        rsi_exit_params: Dict[str, Any],
        chop_params: Dict[str, Any]
    ):
        """
        コンストラクタ
        
        Args:
            supertrend_params: スーパートレンドのパラメータ
            rsi_entry_params: RSIエントリーのパラメータ
            rsi_exit_params: RSIエグジットのパラメータ
            chop_params: チョピネスインデックスのパラメータ
        """
        # シグナルの初期化
        self.supertrend = SupertrendDirectionSignal(
            period=supertrend_params['period'],
            multiplier=supertrend_params['multiplier']
        )
        self.rsi_entry = RSIEntrySignal(
            period=rsi_entry_params['period'],
            solid=rsi_entry_params['solid']
        )
        self.rsi_exit = RSIExitSignal(
            period=rsi_exit_params['period'],
            solid=rsi_exit_params['solid']
        )
        self.chop = ChopFilterSignal(
            period=chop_params['period'],
            solid=chop_params['solid']
        )
        
        # シグナルのキャッシュ
        self._supertrend_signals = None
        self._rsi_exit_signals = None
        self._rsi_entry_signals = None
        self._chop_signals = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """全てのシグナルを計算してキャッシュする"""
        if self._supertrend_signals is None:
            self._supertrend_signals = self.supertrend.generate(data)
        if self._rsi_exit_signals is None:
            self._rsi_exit_signals = self.rsi_exit.generate(data)
        if self._rsi_entry_signals is None:
            self._rsi_entry_signals = self.rsi_entry.generate(data)
        if self._chop_signals is None:
            self._chop_signals = self.chop.generate(data)
    
    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エントリーシグナルを生成する"""
        self.calculate_signals(data)
        
        # シグナルの初期化
        signals = np.zeros(len(self._supertrend_signals))
        
        # ロングエントリー条件: 全てのシグナルが1
        long_condition = (
            (self._supertrend_signals == 1) &
            (self._rsi_entry_signals == 1) &
            (self._chop_signals == 1)
        )
        
        # ショートエントリー条件: スーパートレンドが-1、RSIエントリーが-1、チョピネスが1
        short_condition = (
            (self._supertrend_signals == -1) &
            (self._rsi_entry_signals == -1) &
            (self._chop_signals == 1)
        )
        
        # シグナルの生成
        signals = np.where(long_condition, 1, signals)
        signals = np.where(short_condition, -1, signals)
        
        return signals
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """エグジットシグナルを生成する"""
        if position == 0:
            return False
        
        self.calculate_signals(data)
        
        # 指定されたインデックスのシグナルを取得
        current_supertrend = self._supertrend_signals[index]
        current_rsi_exit = self._rsi_exit_signals[index]
        
        # ロングポジションのエグジット条件
        if position == 1:
            return (current_supertrend == -1) or (current_rsi_exit == 1)
        
        # ショートポジションのエグジット条件
        if position == -1:
            return (current_supertrend == 1) or (current_rsi_exit == -1)
        
        return False
    
    def clear_cache(self) -> None:
        """シグナルのキャッシュをクリアする"""
        self._supertrend_signals = None
        self._rsi_exit_signals = None
        self._rsi_entry_signals = None
        self._chop_signals = None 