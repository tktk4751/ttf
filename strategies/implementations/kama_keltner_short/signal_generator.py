#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from numba import jit

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.kama_keltner.breakout_entry import KAMAKeltnerBreakoutEntrySignal


@jit(nopython=True)
def calculate_entry_signals(entry_signals: np.ndarray) -> np.ndarray:
    """エントリーシグナルを一度に計算（高速化版）"""
    return np.where(entry_signals == -1, -1, 0).astype(np.int8)


class KAMAKeltnerShortSignalGenerator(BaseSignalGenerator):
    """
    KAMAケルトナーチャネルのシグナル生成クラス（売り専用・高速化版）
    
    エントリー条件:
    - エントリー用KAMAケルトナーチャネルのロワーブレイクアウトで売りシグナル
    
    エグジット条件:
    - エグジット用KAMAケルトナーチャネルの買いシグナル
    """
    
    def __init__(
        self,
        entry_kama_period: int = 10,
        entry_kama_fast: int = 2,
        entry_kama_slow: int = 30,
        entry_atr_period: int = 10,
        entry_multiplier: float = 2.0,
        exit_kama_period: int = 10,
        exit_kama_fast: int = 2,
        exit_kama_slow: int = 30,
        exit_atr_period: int = 10,
        exit_multiplier: float = 2.0,
    ):
        """初期化"""
        super().__init__("KAMAKeltnerShortSignalGenerator")
        
        # シグナル生成器の初期化
        self.entry_signal = KAMAKeltnerBreakoutEntrySignal(
            kama_period=entry_kama_period,
            kama_fast=entry_kama_fast,
            kama_slow=entry_kama_slow,
            atr_period=entry_atr_period,
            multiplier=entry_multiplier
        )
        self.exit_signal = KAMAKeltnerBreakoutEntrySignal(
            kama_period=exit_kama_period,
            kama_fast=exit_kama_fast,
            kama_slow=exit_kama_slow,
            atr_period=exit_atr_period,
            multiplier=exit_multiplier
        )
        
        # キャッシュ用の変数（最小限に抑える）
        self._data_len = 0
        self._signals = None
        self._kama_signals = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算（高速化版）"""
        current_len = len(data)
        
        # データ長が変わった場合のみ再計算
        if self._signals is None or current_len != self._data_len:
            # データフレームの作成（必要な列のみ）
            df = data[['open', 'high', 'low', 'close']] if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
            
            # 各シグナルの計算（一度に実行）
            entry_signals = self.entry_signal.generate(df)
            exit_signals = self.exit_signal.generate(df)
            
            # エントリーシグナルの計算（ベクトル化・高速化）
            self._signals = calculate_entry_signals(entry_signals)
            
            # エグジット用のシグナルを事前計算
            self._kama_signals = exit_signals
            
            self._data_len = current_len
    
    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エントリーシグナル取得（高速化版）"""
        if self._signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._signals
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """エグジットシグナル生成（高速化版）"""
        if position != -1:  # 売りポジションのみ
            return False
        
        if self._signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        
        if index == -1:
            index = len(data) - 1
        
        # キャッシュされたシグナルを使用
        return bool(self._kama_signals[index] == 1) 