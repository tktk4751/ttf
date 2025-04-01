#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from numba import jit

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.true_momentum import TrueMomentumEntrySignal, TrueMomentumDirectionSignal
from signals.implementations.guardian_angel.filter import GuardianAngelFilterSignal


@jit(nopython=True)
def calculate_entry_signals(true_momentum_entry: np.ndarray, filter_signals: np.ndarray) -> np.ndarray:
    """エントリーシグナルを計算する（Numba最適化版）
    
    Args:
        true_momentum_entry: トゥルーモメンタムエントリーシグナル配列
        filter_signals: フィルターシグナル配列
    
    Returns:
        最終的なエントリーシグナル配列
    """
    length = len(true_momentum_entry)
    result = np.zeros(length, dtype=np.int8)
    
    for i in range(length):
        # フィルターが有効な場合のみエントリーシグナルを許可
        if filter_signals[i] == 1 and true_momentum_entry[i] != 0:
            result[i] = true_momentum_entry[i]
    
    return result


class TrueMomentumSignalGenerator(BaseSignalGenerator):
    """
    トゥルーモメンタム+ガーディアンエンジェルフィルターのシグナル生成クラス（両方向・高速化版）
    
    エントリー条件:
    - ロング: トゥルーモメンタムの買いシグナル + ガーディアンエンジェルがトレンド相場
    - ショート: トゥルーモメンタムの売りシグナル + ガーディアンエンジェルがトレンド相場
    
    エグジット条件:
    - ロング: トゥルーモメンタムの方向シグナルが売り
    - ショート: トゥルーモメンタムの方向シグナルが買い
    """
    
    def __init__(
        self,
        period: int = 20,
        max_std_mult: float = 2.0,
        min_std_mult: float = 1.0,
        max_kama_slow: int = 55,
        min_kama_slow: int = 30,
        max_kama_fast: int = 13,
        min_kama_fast: int = 2,
        max_atr_period: int = 120,
        min_atr_period: int = 13,
        max_atr_mult: float = 3.0,
        min_atr_mult: float = 1.0,
        max_momentum_period: int = 100,
        min_momentum_period: int = 20,
        momentum_threshold: float = 0.0,
        max_ga_period: int = 100,
        min_ga_period: int = 20,
        max_ga_threshold: float = 61.8,
        min_ga_threshold: float = 38.2
    ):
        """初期化"""
        super().__init__("TrueMomentumSignalGenerator")
        
        # パラメータの設定
        self._params = {
            'period': period,
            'max_std_mult': max_std_mult,
            'min_std_mult': min_std_mult,
            'max_kama_slow': max_kama_slow,
            'min_kama_slow': min_kama_slow,
            'max_kama_fast': max_kama_fast,
            'min_kama_fast': min_kama_fast,
            'max_atr_period': max_atr_period,
            'min_atr_period': min_atr_period,
            'max_atr_mult': max_atr_mult,
            'min_atr_mult': min_atr_mult,
            'max_momentum_period': max_momentum_period,
            'min_momentum_period': min_momentum_period,
            'momentum_threshold': momentum_threshold,
            'max_ga_period': max_ga_period,
            'min_ga_period': min_ga_period,
            'max_ga_threshold': max_ga_threshold,
            'min_ga_threshold': min_ga_threshold
        }
        
        # シグナル生成器の初期化
        self.entry_signal = TrueMomentumEntrySignal(
            period=period,
            max_std_mult=max_std_mult,
            min_std_mult=min_std_mult,
            max_kama_slow=max_kama_slow,
            min_kama_slow=min_kama_slow,
            max_kama_fast=max_kama_fast,
            min_kama_fast=min_kama_fast,
            max_atr_period=max_atr_period,
            min_atr_period=min_atr_period,
            max_atr_mult=max_atr_mult,
            min_atr_mult=min_atr_mult,
            max_momentum_period=max_momentum_period,
            min_momentum_period=min_momentum_period,
            momentum_threshold=momentum_threshold
        )
        
        self.direction_signal = TrueMomentumDirectionSignal(
            period=period,
            max_std_mult=max_std_mult,
            min_std_mult=min_std_mult,
            max_kama_slow=max_kama_slow,
            min_kama_slow=min_kama_slow,
            max_kama_fast=max_kama_fast,
            min_kama_fast=min_kama_fast,
            max_atr_period=max_atr_period,
            min_atr_period=min_atr_period,
            max_atr_mult=max_atr_mult,
            min_atr_mult=min_atr_mult,
            max_momentum_period=max_momentum_period,
            min_momentum_period=min_momentum_period
        )
        
        self.filter_signal = GuardianAngelFilterSignal(
            er_period=period,
            max_period=max_ga_period,
            min_period=min_ga_period,
            solid={
                'max_threshold': max_ga_threshold,
                'min_threshold': min_ga_threshold
            }
        )
        
        # キャッシュ用の変数（最小限に抑える）
        self._data_len = 0
        self._entry_signals = None
        self._direction_signals = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算（高速化版）"""
        current_len = len(data)
        
        # データ長が変わった場合のみ再計算
        if self._entry_signals is None or current_len != self._data_len:
            # データフレームの作成（必要な列のみ）
            df = data[['open', 'high', 'low', 'close']] if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
            
            # 各シグナルの計算（一度に実行）
            filter_signals = self.filter_signal.generate(df)
            true_momentum_direction = self.direction_signal.generate(df)
            true_momentum_entry = self.entry_signal.generate(df)
            
            # エントリーシグナルの計算（Numba最適化）
            self._entry_signals = calculate_entry_signals(true_momentum_entry, filter_signals)
            
            # エグジット用のシグナルを事前計算
            self._direction_signals = true_momentum_direction
            
            self._data_len = current_len
    
    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エントリーシグナル取得（高速化版）"""
        if self._entry_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._entry_signals
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """エグジットシグナル生成（高速化版）"""
        if self._entry_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        
        if index == -1:
            index = len(data) - 1
        
        # キャッシュされたシグナルを使用
        if position == 1:  # ロングポジション
            return bool(self._direction_signals[index] == -1)
        elif position == -1:  # ショートポジション
            return bool(self._direction_signals[index] == 1)
        return False 