#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from numba import jit

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.donchian_atr.entry import DonchianATRBreakoutEntrySignal


@jit(nopython=True)
def calculate_entry_signals(entry_signals: np.ndarray) -> np.ndarray:
    """エントリーシグナルを一度に計算（高速化版）"""
    return np.where(entry_signals == -1, -1, 0).astype(np.int8)


class DonchianATRShortSignalGenerator(BaseSignalGenerator):
    """
    2つのドンチャンATRのシグナル生成クラス（売り専用・高速化版）
    
    エントリー条件:
    - エントリー用ドンチャンATRのロワーブレイクアウトで売りシグナル
    
    エグジット条件:
    - エグジット用ドンチャンATRの買いシグナル
    """
    
    def __init__(
        self,
        entry_period: int = 20,
        entry_atr_period: int = 10,
        entry_multiplier: float = 2.0,
        exit_period: int = 20,
        exit_atr_period: int = 10,
        exit_multiplier: float = 2.0,
    ):
        """初期化"""
        super().__init__("DonchianATRShortSignalGenerator")
        
        # シグナル生成器の初期化
        self.entry_signal = DonchianATRBreakoutEntrySignal(
            period=entry_period,
            atr_period=entry_atr_period,
            multiplier=entry_multiplier
        )
        self.exit_signal = DonchianATRBreakoutEntrySignal(
            period=exit_period,
            atr_period=exit_atr_period,
            multiplier=exit_multiplier
        )
        
        # キャッシュ用の変数（最小限に抑える）
        self._data_len = 0
        self._signals = None
        self._donchian_signals = None
    
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
            self._donchian_signals = exit_signals
            
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
        return bool(self._donchian_signals[index] == 1) 