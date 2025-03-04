#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from numba import jit

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.tfc.entry import TFCEntrySignal
from signals.implementations.chop.filter import ChopFilterSignal


@jit(nopython=True)
def calculate_entry_signals(tfc: np.ndarray, filter_signal: np.ndarray) -> np.ndarray:
    """エントリーシグナルを一度に計算（高速化版）"""
    signals = np.zeros_like(tfc, dtype=np.int8)
    
    # ロングエントリー: TFCが1 + チョピネスがトレンド
    long_condition = (tfc == 1) & (filter_signal == -1)
    
    signals[long_condition] = 1  # ロング
    
    return signals


class TFCChopSingleLongSignalGenerator(BaseSignalGenerator):
    """
    TFC+チョピネスフィルターのシグナル生成クラス（ロング専用・高速化版）
    
    エントリー条件:
    [ロング]
    - TFCのアップトレンドへの転換で買いシグナル
    - チョピネスインデックスがトレンド相場を示している
    
    エグジット条件:
    [ロング]
    - TFCのダウントレンドへの転換
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
        super().__init__("TFCChopSingleLongSignalGenerator")
        
        # シグナル生成器の初期化
        self.signal = TFCEntrySignal(
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
        self._tfc_signals = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算（高速化版）"""
        current_len = len(data)
        
        # データ長が変わった場合のみ再計算
        if self._signals is None or current_len != self._data_len:
            # データフレームの作成（必要な列のみ）
            df = data[['open', 'high', 'low', 'close']] if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
            
            # 各シグナルの計算（一度に実行）
            tfc_signals = self.signal.generate(df)
            filter_signals = self.filter_signal.generate(df)
            
            # エントリーシグナルの計算（ベクトル化・高速化）
            self._signals = calculate_entry_signals(tfc_signals, filter_signals)
            
            # エグジット用のシグナルを事前計算
            self._tfc_signals = tfc_signals
            
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
            return bool(self._tfc_signals[index] == -1)
        return False 