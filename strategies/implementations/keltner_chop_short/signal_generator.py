#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from numba import jit

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.keltner.breakout_entry import KeltnerBreakoutEntrySignal
from signals.implementations.chop.filter import ChopFilterSignal


@jit(nopython=True)
def calculate_entry_signals(keltner: np.ndarray, filter_signal: np.ndarray) -> np.ndarray:
    """エントリーシグナルを一度に計算（高速化版）"""
    return np.where(
        (keltner == -1) & (filter_signal == 1),
        -1,
        0
    ).astype(np.int8)


class KeltnerChopShortSignalGenerator(BaseSignalGenerator):
    """
    ケルトナーチャネル+チョピネスフィルターのシグナル生成クラス（売り専用・高速化版）
    
    エントリー条件:
    - ケルトナーチャネルのロワーブレイクアウトで売りシグナル
    - チョピネスインデックスがトレンド相場を示している
    
    エグジット条件:
    - ケルトナーチャネルのアッパーバンドを上回る
    """
    
    def __init__(
        self,
        keltner_period: int = 20,
        atr_period: int = 10,
        multiplier: float = 2.0,
        chop_period: int = 14,
        chop_threshold: float = 50.0,
    ):
        """初期化"""
        super().__init__("KeltnerChopShortSignalGenerator")
        
        # シグナル生成器の初期化
        self.entry_signal = KeltnerBreakoutEntrySignal(
            period=keltner_period,
            atr_period=atr_period,
            multiplier=multiplier
        )
        self.filter_signal = ChopFilterSignal(
            period=chop_period,
            solid={'chop_solid': chop_threshold}
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
            keltner_signals = self.entry_signal.generate(df).astype(np.int8)
            filter_signals = self.filter_signal.generate(df).astype(np.int8)
            
            # エントリーシグナルの計算（ベクトル化）
            self._signals = calculate_entry_signals(keltner_signals, filter_signals)
            
            # エグジット用のシグナルを事前計算
            self._keltner_signals = keltner_signals
            
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
        return bool(self._keltner_signals[index] == 1) 