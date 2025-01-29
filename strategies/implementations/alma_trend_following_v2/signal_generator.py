#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from numba import jit

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.alma.direction import ALMATrendFollowingStrategy
from signals.implementations.adx.filter import ADXFilterSignal
from signals.implementations.roc.entry import ROCEntrySignal


@jit(nopython=True)
def calculate_entry_signals(direction: np.ndarray, filter_signal: np.ndarray, entry: np.ndarray) -> np.ndarray:
    """エントリーシグナルを一度に計算（高速化版）"""
    return np.where(
        (direction == 1) & (filter_signal == 1) & (entry == 1), 1,
        np.where((direction == -1) & (filter_signal == 1) & (entry == -1), -1, 0)
    ).astype(np.int8)


class ALMATrendFollowingV2SignalGenerator(BaseSignalGenerator):
    """ALMAトレンドフォロー戦略V2のシグナル生成クラス（高速化版）"""
    
    def __init__(
        self,
        alma_short_period: int = 9,
        alma_long_period: int = 200,
        adx_period: int = 14,
        adx_threshold: float = 30,
        roc_period: int = 21,
    ):
        """初期化"""
        super().__init__("ALMATrendFollowingV2SignalGenerator")
        
        # シグナル生成器の初期化
        self.direction_signal = ALMATrendFollowingStrategy(
            short_period=alma_short_period,
            long_period=alma_long_period,
            sigma=6.0,
            offset=0.85
        )
        self.filter_signal = ADXFilterSignal(
            period=adx_period,
            solid={'adx_solid': adx_threshold}
        )
        self.entry_signal = ROCEntrySignal(period=roc_period)
        
        # キャッシュ用の変数（最小限に抑える）
        self._data_len = 0
        self._signals = None
        self._direction = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算（高速化版）"""
        current_len = len(data)
        
        # データ長が変わった場合のみ再計算
        if self._signals is None or current_len != self._data_len:
            # データフレームの作成（必要な列のみ）
            df = data[['open', 'high', 'low', 'close']] if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
            
            # 各シグナルの計算（一度に実行）
            direction = self.direction_signal.generate(df)
            filter_signal = self.filter_signal.generate(df)
            entry = self.entry_signal.generate(df)
            
            # エントリーシグナルの計算（ベクトル化）
            self._signals = calculate_entry_signals(
                direction.astype(np.int8),
                filter_signal.astype(np.int8),
                entry.astype(np.int8)
            )
            
            self._direction = direction
            self._data_len = current_len
    
    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エントリーシグナル取得（高速化版）"""
        if self._signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._signals
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """エグジットシグナル生成（高速化版）"""
        if position == 0:
            return False
        
        if self._signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        
        if index == -1:
            index = len(data) - 1
        
        # ALMAのクロスオーバーでエグジット
        # ロングポジション: デッドクロス（方向シグナルが-1に変化）
        # ショートポジション: ゴールデンクロス（方向シグナルが1に変化）
        return bool(
            (position == 1 and self._direction[index] == -1) or
            (position == -1 and self._direction[index] == 1)
        ) 