#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from numba import jit

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.kama_keltner.breakout_entry import KAMAKeltnerBreakoutEntrySignal
from signals.implementations.chop.filter import ChopFilterSignal


@jit(nopython=True)
def calculate_entry_signals(kama: np.ndarray, filter_signal: np.ndarray) -> np.ndarray:
    """エントリーシグナルを一度に計算（高速化版）"""
    signals = np.zeros_like(kama, dtype=np.int8)
    
    # ロングエントリー: KAMAケルトナーのアッパーブレイクアウト + チョピネスがトレンド
    long_condition = (kama == 1) & (filter_signal == 1)
    # ショートエントリー: KAMAケルトナーのロワーブレイクアウト + チョピネスがトレンド
    short_condition = (kama == -1) & (filter_signal == 1)
    
    signals[long_condition] = 1    # ロング
    signals[short_condition] = -1  # ショート
    
    return signals


class KAMAKeltnerChopLongShortSignalGenerator(BaseSignalGenerator):
    """
    KAMAケルトナーチャネル+チョピネスフィルターのシグナル生成クラス（ロング・ショート両対応・高速化版）
    
    エントリー条件:
    [ロング]
    - KAMAケルトナーチャネルのアッパーブレイクアウトで買いシグナル
    - チョピネスインデックスがトレンド相場を示している
    
    [ショート]
    - KAMAケルトナーチャネルのロワーブレイクアウトで売りシグナル
    - チョピネスインデックスがトレンド相場を示している
    
    エグジット条件:
    [ロング]
    - KAMAケルトナーチャネルの売りシグナル
    
    [ショート]
    - KAMAケルトナーチャネルの買いシグナル
    """
    
    def __init__(
        self,
        kama_period: int = 10,
        kama_fast: int = 2,
        kama_slow: int = 30,
        atr_period: int = 10,
        upper_multiplier: float = 2.0,
        lower_multiplier: float = 2.0,
        chop_period: int = 14,
        chop_threshold: float = 50.0,
    ):
        """初期化"""
        super().__init__("KAMAKeltnerChopLongShortSignalGenerator")
        
        # シグナル生成器の初期化
        self.signal = KAMAKeltnerBreakoutEntrySignal(
            kama_period=kama_period,
            kama_fast=kama_fast,
            kama_slow=kama_slow,
            atr_period=atr_period,
            upper_multiplier=upper_multiplier,
            lower_multiplier=lower_multiplier
        )
        self.filter_signal = ChopFilterSignal(
            period=chop_period,
            solid={'chop_solid': chop_threshold}
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
            kama_signals = self.signal.generate(df)
            filter_signals = self.filter_signal.generate(df)
            
            # エントリーシグナルの計算（ベクトル化・高速化）
            self._signals = calculate_entry_signals(kama_signals, filter_signals)
            
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
        if self._signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        
        if index == -1:
            index = len(data) - 1
        
        # キャッシュされたシグナルを使用
        if position == 1:  # ロングポジション
            return bool(self._kama_signals[index] == -1)
        elif position == -1:  # ショートポジション
            return bool(self._kama_signals[index] == 1)
        return False 