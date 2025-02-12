#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from numba import jit

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.supertrend.direction import SupertrendDirectionSignal
from signals.implementations.chop.filter import ChopFilterSignal
from signals.implementations.adx.filter import ADXFilterSignal

@jit(nopython=True)
def calculate_entry_signals(supertrend_signals: np.ndarray, chop_signals: np.ndarray, adx_signals: np.ndarray) -> np.ndarray:
    """エントリーシグナルを一度に計算（高速化版）"""
    signals = np.zeros_like(supertrend_signals)
    
    for i in range(len(signals)):
        # スーパートレンドが買い、CHOPフィルターがトレンド相場、ADXフィルターがトレンド相場の場合
        if supertrend_signals[i] == 1 and chop_signals[i] == 1 and adx_signals[i] == 1:
            signals[i] = 1
    
    return signals.astype(np.int8)

@jit(nopython=True)
def calculate_exit_signals(supertrend_signals: np.ndarray, chop_signals: np.ndarray, adx_signals: np.ndarray) -> np.ndarray:
    """エグジットシグナルを一度に計算（高速化版）"""
    signals = np.zeros_like(supertrend_signals)
    
    for i in range(len(signals)):
        # スーパートレンドが売り、もしくはCHOPとADXが両方レンジ相場の場合
        if supertrend_signals[i] == -1 or (chop_signals[i] == -1 and adx_signals[i] == -1):
            signals[i] = 1
    
    return signals.astype(np.int8)

class SupertrendChopADXLongSignalGenerator(BaseSignalGenerator):
    """
    スーパートレンド+CHOP+ADXのシグナル生成クラス（買い専用・高速化版）
    
    エントリー条件:
    - スーパートレンドが買いシグナル
    - CHOPフィルターがトレンド相場を示している
    - ADXフィルターがトレンド相場を示している
    
    エグジット条件:
    - スーパートレンドが売りシグナル
    もしくは
    - CHOPフィルターがレンジ相場を示している
    - ADXフィルターがレンジ相場を示している
    """
    
    def __init__(
        self,
        supertrend_period: int = 10,
        supertrend_multiplier: float = 3.0,
        chop_period: int = 14,
        chop_threshold: float = 50.0,
        adx_period: int = 14,
        adx_threshold: float = 30.0,
    ):
        """初期化"""
        super().__init__("SupertrendChopADXLongSignalGenerator")
        
        # シグナル生成器の初期化
        self.supertrend_signal = SupertrendDirectionSignal(
            period=supertrend_period,
            multiplier=supertrend_multiplier
        )
        self.chop_signal = ChopFilterSignal(
            period=chop_period,
            solid={'chop_solid': chop_threshold}
        )
        self.adx_signal = ADXFilterSignal(
            period=adx_period,
            threshold=adx_threshold
        )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._signals = None
        self._supertrend_signals = None
        self._chop_signals = None
        self._adx_signals = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算（高速化版）"""
        current_len = len(data)
        
        # データ長が変わった場合のみ再計算
        if self._signals is None or current_len != self._data_len:
            # 各シグナルの計算
            supertrend_signals = self.supertrend_signal.generate(data)
            chop_signals = self.chop_signal.generate(data)
            adx_signals = self.adx_signal.generate(data)
            
            # エントリーシグナルの計算（ベクトル化）
            self._signals = calculate_entry_signals(supertrend_signals, chop_signals, adx_signals)
            
            # エグジット用にシグナルを保存
            self._supertrend_signals = supertrend_signals
            self._chop_signals = chop_signals
            self._adx_signals = adx_signals
            
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
        
        # スーパートレンドの売りシグナル、もしくはCHOPとADXが両方レンジ相場でエグジット
        return bool(self._supertrend_signals[index] == -1 or 
                   (self._chop_signals[index] == -1 and self._adx_signals[index] == -1)) 