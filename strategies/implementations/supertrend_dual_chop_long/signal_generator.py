#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from numba import jit

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.supertrend.direction import SupertrendDirectionSignal
from signals.implementations.chop.filter import ChopFilterSignal

@jit(nopython=True)
def calculate_entry_signals(slow_supertrend: np.ndarray, fast_supertrend: np.ndarray, filter_signal: np.ndarray) -> np.ndarray:
    """エントリーシグナルを一度に計算（高速化版）"""
    signals = np.zeros(len(slow_supertrend), dtype=np.int8)
    
    for i in range(1, len(slow_supertrend)):
        # スロウSTが1かつチョピネスフィルターが1
        condition1 = (slow_supertrend[i] == 1) and (filter_signal[i] == 1)
        # スロウSTが1のときに、ファストSTが-1から1に切り替わる
        condition2 = (slow_supertrend[i] == 1) and (fast_supertrend[i-1] == -1) and (fast_supertrend[i] == 1)
        
        if condition1 or condition2:
            signals[i] = 1
            
    return signals

@jit(nopython=True)
def calculate_exit_signals(slow_supertrend: np.ndarray, fast_supertrend: np.ndarray) -> np.ndarray:
    """エグジットシグナルを一度に計算（高速化版）"""
    signals = np.zeros(len(slow_supertrend), dtype=np.int8)
    
    for i in range(1, len(slow_supertrend)):
        # スロウSTが1から-1に切り替わる
        condition1 = (slow_supertrend[i-1] == 1) and (slow_supertrend[i] == -1)
        # スロウSTが1のときに、ファストSTが1から-1に切り替わる
        condition2 = (slow_supertrend[i] == 1) and (fast_supertrend[i-1] == 1) and (fast_supertrend[i] == -1)
        
        if condition1 or condition2:
            signals[i] = 1
            
    return signals

class SupertrendDualChopLongSignalGenerator(BaseSignalGenerator):
    """
    デュアルスーパートレンド+チョピネスフィルターのシグナル生成クラス（買い専用・高速化版）
    
    エントリー条件:
    - スロウSTが1かつチョピネスフィルターが1
    - スロウSTが1のときに、ファストSTが-1から1に切り替わる
    
    エグジット条件:
    - スロウSTが1から-1に切り替わる
    - スロウSTが1のときに、ファストSTが1から-1に切り替わる
    """
    
    def __init__(
        self,
        period: int = 10,
        fast_multiplier: float = 2.0,
        slow_multiplier: float = 4.0,
        chop_period: int = 14,
        chop_threshold: float = 50.0,
    ):
        """初期化"""
        super().__init__("SupertrendDualChopLongSignalGenerator")
        
        # シグナル生成器の初期化
        self.fast_direction_signal = SupertrendDirectionSignal(
            period=period,
            multiplier=fast_multiplier
        )
        self.slow_direction_signal = SupertrendDirectionSignal(
            period=period,
            multiplier=slow_multiplier
        )
        self.filter_signal = ChopFilterSignal(
            period=chop_period,
            solid={'chop_solid': chop_threshold}
        )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._entry_signals = None
        self._exit_signals = None
        self._slow_supertrend_signals = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算（高速化版）"""
        current_len = len(data)
        
        # データ長が変わった場合のみ再計算
        if self._entry_signals is None or current_len != self._data_len:
            # データフレームの作成（必要な列のみ）
            df = data[['open', 'high', 'low', 'close']] if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
            
            # 各シグナルの計算（一度に実行）
            fast_supertrend_signals = self.fast_direction_signal.generate(df).astype(np.int8)
            slow_supertrend_signals = self.slow_direction_signal.generate(df).astype(np.int8)
            filter_signals = self.filter_signal.generate(df).astype(np.int8)
            
            # エントリー/エグジットシグナルの計算（ベクトル化）
            self._entry_signals = calculate_entry_signals(slow_supertrend_signals, fast_supertrend_signals, filter_signals)
            self._exit_signals = calculate_exit_signals(slow_supertrend_signals, fast_supertrend_signals)
            
            # エグジット用のシグナルを事前計算
            self._slow_supertrend_signals = slow_supertrend_signals
            
            self._data_len = current_len
    
    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エントリーシグナル取得（高速化版）"""
        if self._entry_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._entry_signals
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """エグジットシグナル生成（高速化版）"""
        if position != 1:  # 買いポジションのみ
            return False
        
        if self._exit_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        
        if index == -1:
            index = len(data) - 1
        
        return bool(self._exit_signals[index])

    def get_stop_price(self, data: Union[pd.DataFrame, np.ndarray], index: int = -1) -> float:
        """
        ストップロス価格を取得する（スロウスーパートレンドの下限値）
        
        Args:
            data: 価格データ
            index: データのインデックス（デフォルト: -1 = 最新のデータ）
            
        Returns:
            float: ストップロス価格
        """
        if index == -1:
            index = len(data) - 1
            
        # スロウスーパートレンドの下限値を取得
        return float(self.slow_direction_signal.get_lower_band(data)[index]) 