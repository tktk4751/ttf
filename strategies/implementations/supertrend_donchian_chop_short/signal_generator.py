#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from numba import jit

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.supertrend.direction import SupertrendDirectionSignal
from signals.implementations.donchian.entry import DonchianBreakoutEntrySignal
from signals.implementations.chop.filter import ChopFilterSignal
from signals.implementations.rsi.exit import RSIExitSignal

@jit(nopython=True)
def calculate_entry_signals(
    supertrend_signals: np.ndarray,
    donchian_signals: np.ndarray,
    chop_signals: np.ndarray
) -> np.ndarray:
    """エントリーシグナルを計算"""
    signals = np.zeros(len(supertrend_signals))
    
    for i in range(len(signals)):
        if (supertrend_signals[i] == -1 and 
            donchian_signals[i] == -1 and 
            chop_signals[i] == 1):
            signals[i] = 1
    
    return signals

@jit(nopython=True)
def calculate_exit_signals(
    supertrend_signals: np.ndarray,
    donchian_signals: np.ndarray,
    rsi_signals: np.ndarray
) -> np.ndarray:
    """エグジットシグナルを計算"""
    signals = np.zeros(len(supertrend_signals))
    
    for i in range(len(signals)):
        if (supertrend_signals[i] == 1 or 
            donchian_signals[i] == 1 or 
            rsi_signals[i] == -1):
            signals[i] = 1
    
    return signals

class SupertrendDonchianChopShortSignalGenerator(BaseSignalGenerator):
    """スーパートレンド+ドンチャン+CHOPのショートシグナル生成器"""
    
    def __init__(
        self,
        supertrend_period: int = 10,
        supertrend_multiplier: float = 3.0,
        donchian_period: int = 20,
        chop_period: int = 14,
        chop_threshold: float = 50.0,
        rsi_period: int = 14,
        rsi_lower: float = 30.0,
    ):
        """
        初期化
        
        Args:
            supertrend_period: スーパートレンドの期間
            supertrend_multiplier: スーパートレンドの乗数
            donchian_period: ドンチャンの期間
            chop_period: CHOPの期間
            chop_threshold: CHOPのしきい値
            rsi_period: RSIの期間
            rsi_lower: RSIの下限しきい値
        """
        super().__init__("SupertrendDonchianChopShort")
        
        # シグナル生成器の初期化
        self._supertrend = SupertrendDirectionSignal(
            period=supertrend_period,
            multiplier=supertrend_multiplier
        )
        self._donchian = DonchianBreakoutEntrySignal(
            period=donchian_period
        )
        self._chop = ChopFilterSignal(
            period=chop_period,
            solid={'chop_solid': chop_threshold}
        )
        self._rsi_exit = RSIExitSignal(
            period=rsi_period,
            solid={'lower': rsi_lower}
        )
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナルを計算してキャッシュに保存"""
        # 各シグナルの計算
        supertrend_signals = self._supertrend.generate(data)
        donchian_signals = self._donchian.generate(data)
        chop_signals = self._chop.generate(data)
        rsi_signals = self._rsi_exit.generate(data)
        
        # エントリー/エグジットシグナルの計算
        entry_signals = calculate_entry_signals(
            supertrend_signals,
            donchian_signals,
            chop_signals
        )
        exit_signals = calculate_exit_signals(
            supertrend_signals,
            donchian_signals,
            rsi_signals
        )
        
        # キャッシュに保存
        self._signals_cache['entry'] = entry_signals
        self._signals_cache['exit'] = exit_signals
    
    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エントリーシグナルを取得"""
        cached = self._get_cached_signal('entry')
        if cached is None:
            self.calculate_signals(data)
            cached = self._get_cached_signal('entry')
        return cached
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """エグジットシグナルを取得"""
        cached = self._get_cached_signal('exit')
        if cached is None:
            self.calculate_signals(data)
            cached = self._get_cached_signal('exit')
        
        if index == -1:
            index = len(cached) - 1
            
        return bool(cached[index]) 