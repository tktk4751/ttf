#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from numba import jit

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.supertrend.direction import SupertrendDirectionSignal
from signals.implementations.rsi.entry import RSICounterTrendEntrySignal
from signals.implementations.rsi.exit import RSIExitSignal
from signals.implementations.chop.filter import ChopFilterSignal


@jit(nopython=True)
def calculate_entry_signals(supertrend: np.ndarray, rsi: np.ndarray, chop: np.ndarray) -> np.ndarray:
    """エントリーシグナルを計算（高速化版）"""
    return np.where((supertrend == 1) & (rsi == 1) & (chop == 1), 1, 0).astype(np.int8)


class SupertrendRsiChopSignalGenerator(BaseSignalGenerator):
    """
    スーパートレンド、RSI、チョピネスフィルターを組み合わせたシグナル生成クラス（買い専用・高速化版）
    
    エントリー条件:
    - スーパートレンドが上昇トレンドと判断
    - RSIが売られすぎ水準から上抜けた場合
    - チョピネスフィルターがトレンド相場と判断
    
    エグジット条件:
    - スーパートレンドが上昇トレンドでなくなった場合
    - または、チョピネスフィルターがレンジ相場と判断した場合
    """
    
    def __init__(
        self,
        period: int = 10,
        multiplier: float = 3.0,
        rsi_period: int = 14,
        rsi_entry_solid: float = 30.0,
        chop_period: int = 14,
        chop_threshold: float = 61.8,
    ):
        """初期化"""
        super().__init__("SupertrendRsiChopSignalGenerator")
        
        # シグナル生成器の初期化
        self.supertrend = SupertrendDirectionSignal(
            period=period,
            multiplier=multiplier
        )
        self.rsi = RSICounterTrendEntrySignal(
            period=rsi_period,
            solid={'rsi_long_entry_solid': rsi_entry_solid}
        )
        self.chop = ChopFilterSignal(
            period=chop_period,
            solid={'chop_solid': chop_threshold}
        )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._signals = None
        self._supertrend_signals = None
        self._rsi_signals = None
        self._chop_signals = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算（高速化版）"""
        current_len = len(data)
        
        if self._signals is None or current_len != self._data_len:
            # データフレームの作成
            df = data[['open', 'high', 'low', 'close']] if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
            
            # 各シグナルの計算
            self._supertrend_signals = self.supertrend.generate(df)
            self._rsi_signals = self.rsi.generate(df)
            self._chop_signals = self.chop.generate(df)
            
            # エントリーシグナルの計算
            self._signals = calculate_entry_signals(
                self._supertrend_signals,
                self._rsi_signals,
                self._chop_signals
            )
            
            self._data_len = current_len
    
    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エントリーシグナル取得（高速化版）"""
        if self._signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._signals
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """エグジットシグナル生成（高速化版）"""
        if position != 1:  # 買いポジションのみ
            return False
        
        if self._signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        
        if index == -1:
            index = len(data) - 1
        
        # スーパートレンドが上昇トレンドでない、またはチョピネスがレンジ相場
        return bool(self._supertrend_signals[index] != 1 or self._chop_signals[index] != 1)
    
    def clear_cache(self) -> None:
        """シグナルのキャッシュをクリアする"""
        self._signals = None
        self._supertrend_signals = None
        self._rsi_signals = None
        self._chop_signals = None 