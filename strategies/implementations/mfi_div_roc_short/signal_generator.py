#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from numba import jit

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.divergence.divergence_signal import DivergenceSignal
from signals.implementations.roc.entry import ROCEntrySignal
from signals.implementations.mfi.exit import MFIExitSignal
from indicators.mfi import MFI

@jit(nopython=True)
def calculate_entry_signals(div_signals: np.ndarray, roc_signals: np.ndarray, lookback: int) -> np.ndarray:
    """エントリーシグナルを一度に計算（高速化版）"""
    signals = np.zeros_like(div_signals)
    
    for i in range(lookback, len(div_signals)):
        # 過去lookback期間でダイバージェンスシグナルが出ているか確認
        if div_signals[i] == -1:
            # 現在のROCシグナルが売りシグナルか確認
            if roc_signals[i] == -1:
                signals[i] = -1
        elif div_signals[i-lookback:i].min() == -1 and roc_signals[i] == -1:
            signals[i] = -1
    
    return signals.astype(np.int8)

class MFIDivROCShortSignalGenerator(BaseSignalGenerator):
    """
    MFIダイバージェンス+ROCのシグナル生成クラス（売り専用・高速化版）
    
    エントリー条件:
    - MFIダイバージェンスが売りシグナルを出してから13期間以内
    - ROCが売りシグナル
    
    エグジット条件:
    - ROCが買いシグナル
    - MFIエグジットシグナルが発生
    """
    
    def __init__(
        self,
        div_lookback: int = 30,
        roc_period: int = 21,
        entry_mfi_period: int = 14,  # エントリー用MFI期間
        exit_mfi_period: int = 14,   # エグジット用MFI期間
        entry_lookback: int = 13,
    ):
        """初期化"""
        super().__init__("MFIDivROCShortSignalGenerator")
        
        # シグナル生成器の初期化
        self._entry_mfi = MFI(entry_mfi_period)  # エントリー用MFI
        self._exit_mfi = MFI(exit_mfi_period)    # エグジット用MFI
        self.div_signal = DivergenceSignal(lookback=div_lookback)
        self.roc_signal = ROCEntrySignal(period=roc_period)
        self.exit_signal = MFIExitSignal(period=exit_mfi_period)
        
        self.entry_lookback = entry_lookback
        
        # キャッシュ用の変数
        self._data_len = 0
        self._signals = None
        self._roc_signals = None
        self._exit_signals = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算（高速化版）"""
        current_len = len(data)
        
        # データ長が変わった場合のみ再計算
        if self._signals is None or current_len != self._data_len:
            # データフレームの作成（必要な列のみ）
            df = data[['open', 'high', 'low', 'close', 'volume']] if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
            
            # MFI値の計算（エントリー用とエグジット用）
            entry_mfi_values = self._entry_mfi.calculate(data)
            
            # 各シグナルの計算（一度に実行）
            div_signals = self.div_signal.generate(df, entry_mfi_values)  # エントリー用MFIを使用
            roc_signals = self.roc_signal.generate(df)
            
            # エントリーシグナルの計算（ベクトル化）
            self._signals = calculate_entry_signals(div_signals, roc_signals, self.entry_lookback)
            
            # エグジット用のシグナルを事前計算
            self._roc_signals = roc_signals
            self._exit_signals = self.exit_signal.generate(df)
            
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
        return bool(
            self._roc_signals[index] == 1 or
            self._exit_signals[index] == -1
        ) 