#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from numba import jit

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.kama.direction import KAMACirculationSignal
from signals.implementations.chop.filter import ChopFilterSignal


@jit(nopython=True)
def calculate_entry_signals(kama_signals: np.ndarray, chop_signals: np.ndarray) -> np.ndarray:
    """エントリーシグナルを一度に計算（高速化版）"""
    return np.where(
        (kama_signals == -1) & (chop_signals == 1),
        -1,
        0
    ).astype(np.int8)


class KAMACirculationSingleShortSignalGenerator(BaseSignalGenerator):
    """
    KAMAサーキュレーションのシグナル生成クラス（単一チャネル・売り専用・高速化版）
    
    エントリー条件:
    - KAMAサーキュレーションが下降相場を示している
    - チョピネスフィルターがトレンド相場を示している
    
    エグジット条件:
    - KAMAサーキュレーションが上昇相場を示している
    """
    
    def __init__(
        self,
        short_period: int = 9,
        middle_period: int = 21,
        long_period: int = 55,
        chop_period: int = 14,
        chop_solid: float = 50.0,
    ):
        """初期化"""
        super().__init__("KAMACirculationSingleShortSignalGenerator")
        
        # シグナル生成器の初期化
        self.kama_signal = KAMACirculationSignal(
            params={
                'short_period': short_period,
                'middle_period': middle_period,
                'long_period': long_period
            }
        )
        
        self.chop_signal = ChopFilterSignal(
            period=chop_period,
            solid={'chop_solid': chop_solid}
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
            kama_signals = self.kama_signal.generate(df)
            chop_signals = self.chop_signal.generate(df)
            
            # エントリーシグナルの計算（ベクトル化・高速化）
            self._signals = calculate_entry_signals(kama_signals, chop_signals)
            
            # エグジット用のシグナルを事前計算
            self._kama_signals = kama_signals
            
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