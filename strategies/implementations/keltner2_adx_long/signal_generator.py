#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from numba import jit

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.keltner.breakout_entry import KeltnerBreakoutEntrySignal
from signals.implementations.adx.filter import ADXFilterSignal


@jit(nopython=True)
def calculate_entry_signals(keltner: np.ndarray, filter_signal: np.ndarray) -> np.ndarray:
    """エントリーシグナルを一度に計算（高速化版）"""
    return np.where(
        (keltner == 1) & (filter_signal == 1),
        1,
        0
    ).astype(np.int8)


class Keltner2ADXLongSignalGenerator(BaseSignalGenerator):
    """
    2つのケルトナー+ADXフィルターのシグナル生成クラス（買い専用・高速化版）
    
    エントリー条件:
    - エントリー用ケルトナーチャネルのアッパーブレイクアウトで買いシグナル
    - ADXがトレンド相場を示している
    
    エグジット条件:
    - エグジット用ケルトナーチャネルの売りシグナル
    """
    
    def __init__(
        self,
        entry_period: int = 20,
        entry_atr_period: int = 10,
        entry_multiplier: float = 2.0,
        exit_period: int = 20,
        exit_atr_period: int = 10,
        exit_multiplier: float = 2.0,
        adx_period: int = 14,
        adx_threshold: float = 30.0,
    ):
        """初期化"""
        super().__init__("Keltner2ADXLongSignalGenerator")
        
        # シグナル生成器の初期化
        self.entry_signal = KeltnerBreakoutEntrySignal(
            period=entry_period,
            atr_period=entry_atr_period,
            multiplier=entry_multiplier
        )
        self.exit_signal = KeltnerBreakoutEntrySignal(
            period=exit_period,
            atr_period=exit_atr_period,
            multiplier=exit_multiplier
        )
        self.filter_signal = ADXFilterSignal(
            period=adx_period,
            threshold=adx_threshold
        )
        
        # キャッシュ用の変数（最小限に抑える）
        self._data_len = 0
        self._signals = None
        self._keltner_signals = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算（高速化版）"""
        current_len = len(data)
        
        # データ長が変わった場合のみ再計算
        if self._signals is None or current_len != self._data_len:
            # データフレームの作成（必要な列のみ）
            df = data[['open', 'high', 'low', 'close']] if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
            
            # 各シグナルの計算（一度に実行）
            entry_signals = self.entry_signal.generate(df).astype(np.int8)
            filter_signals = self.filter_signal.generate(df).astype(np.int8)
            exit_signals = self.exit_signal.generate(df).astype(np.int8)
            
            # エントリーシグナルの計算（ベクトル化）
            self._signals = calculate_entry_signals(entry_signals, filter_signals)
            
            # エグジット用のシグナルを事前計算
            self._keltner_signals = exit_signals
            
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
        
        # キャッシュされたシグナルを使用
        return bool(self._keltner_signals[index] == -1) 