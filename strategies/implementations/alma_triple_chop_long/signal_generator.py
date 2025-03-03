#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from numba import jit

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.alma.direction import ALMATripleDirectionSignal
from signals.implementations.chop.filter import ChopFilterSignal


@jit(nopython=True)
def calculate_entry_signals(alma_signals: np.ndarray, chop_signals: np.ndarray) -> np.ndarray:
    """エントリーシグナルを計算（高速化版）"""
    return np.where((alma_signals == 1) & (chop_signals == 1), 1, 0).astype(np.int8)


class ALMATripleChopLongSignalGenerator(BaseSignalGenerator):
    """
    ALMAトリプルとチョピネスフィルターを組み合わせたシグナル生成クラス（買い専用・高速化版）
    
    エントリー条件:
    - ALMAトリプルが完全な上昇配列（短期 > 中期 > 長期）
    - チョピネスフィルターがトレンド相場と判断
    
    エグジット条件:
    - ALMAトリプルが完全な上昇配列でなくなった場合
    - または、チョピネスフィルターがレンジ相場と判断した場合
    """
    
    def __init__(
        self,
        short_period: int = 21,
        middle_period: int = 89,
        long_period: int = 233,
        chop_period: int = 55,
        chop_threshold: float = 50.0,
    ):
        """初期化"""
        super().__init__("ALMATripleChopLongSignalGenerator")
        
        # シグナル生成器の初期化
        self.alma_triple = ALMATripleDirectionSignal(
            short_period=short_period,
            middle_period=middle_period,
            long_period=long_period
        )
        self.chop = ChopFilterSignal(
            period=chop_period,
            solid={'chop_solid': chop_threshold}
        )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._signals = None
        self._alma_signals = None
        self._chop_signals = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算（高速化版）"""
        current_len = len(data)
        
        if self._signals is None or current_len != self._data_len:
            # データフレームの作成
            df = data[['open', 'high', 'low', 'close']] if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
            
            # 各シグナルの計算
            self._alma_signals = self.alma_triple.generate(df)
            self._chop_signals = self.chop.generate(df)
            
            # エントリーシグナルの計算
            self._signals = calculate_entry_signals(self._alma_signals, self._chop_signals)
            
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
        
        # ALMAトリプルが上昇配列でない、またはチョピネスがレンジ相場
        return bool(self._alma_signals[index] != 1 or self._chop_signals[index] != 1) 