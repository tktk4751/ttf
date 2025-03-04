#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from numba import jit

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.hyper_trend.direction import HyperTrendDirectionSignal
from signals.implementations.chop.filter import ChopFilterSignal


@jit(nopython=True)
def calculate_entry_signals(direction: np.ndarray, filter_signal: np.ndarray) -> np.ndarray:
    """エントリーシグナルを一度に計算（高速化版）"""
    signals = np.zeros_like(direction, dtype=np.int8)
    
    # ロングエントリー: HyperTrendが上昇トレンド（1）+ チョピネスがトレンド（1）
    long_condition = (direction == 1) & (filter_signal == 1)
    
    # ショートエントリー: HyperTrendが下降トレンド（-1）+ チョピネスがトレンド（1）
    short_condition = (direction == -1) & (filter_signal == 1)
    
    signals[long_condition] = 1  # ロング
    signals[short_condition] = -1  # ショート
    
    return signals


class HyperTrendChopDualSignalGenerator(BaseSignalGenerator):
    """
    HyperTrend+チョピネスフィルターのシグナル生成クラス（両方向・高速化版）
    
    エントリー条件:
    [ロング]
    - HyperTrendが上昇トレンドを示している
    - チョピネスインデックスがトレンド相場を示している
    
    [ショート]
    - HyperTrendが下降トレンドを示している
    - チョピネスインデックスがトレンド相場を示している
    
    エグジット条件:
    [ロング]
    - HyperTrendが下降トレンドに転換
    
    [ショート]
    - HyperTrendが上昇トレンドに転換
    """
    
    def __init__(
        self,
        er_period: int = 10,
        max_percentile_length: int = 55,
        min_percentile_length: int = 14,
        max_atr_period: int = 120,
        min_atr_period: int = 5,
        max_multiplier: float = 3.0,
        min_multiplier: float = 1.0,
        chop_period: int = 14,
        chop_threshold: float = 50.0,
    ):
        """初期化"""
        super().__init__("HyperTrendChopDualSignalGenerator")
        
        # シグナル生成器の初期化
        self.signal = HyperTrendDirectionSignal(
            er_period=er_period,
            max_percentile_length=max_percentile_length,
            min_percentile_length=min_percentile_length,
            max_atr_period=max_atr_period,
            min_atr_period=min_atr_period,
            max_multiplier=max_multiplier,
            min_multiplier=min_multiplier
        )
        self.filter_signal = ChopFilterSignal(
            period=chop_period,
            solid={'chop_solid': chop_threshold}
        )
        
        # キャッシュ用の変数（最小限に抑える）
        self._data_len = 0
        self._signals = None
        self._direction_signals = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算（高速化版）"""
        current_len = len(data)
        
        # データ長が変わった場合のみ再計算
        if self._signals is None or current_len != self._data_len:
            # データフレームの作成（必要な列のみ）
            df = data[['open', 'high', 'low', 'close']] if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
            
            # 各シグナルの計算（一度に実行）
            direction_signals = self.signal.generate(df)
            filter_signals = self.filter_signal.generate(df)
            
            # エントリーシグナルの計算（ベクトル化・高速化）
            self._signals = calculate_entry_signals(direction_signals, filter_signals)
            
            # エグジット用のシグナルを事前計算
            self._direction_signals = direction_signals
            
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
            return bool(self._direction_signals[index] == -1)
        elif position == -1:  # ショートポジション
            return bool(self._direction_signals[index] == 1)
        return False 