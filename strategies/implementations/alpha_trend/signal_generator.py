#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from numba import jit

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.alpha_trend.breakout_entry import AlphaTrendBreakoutEntrySignal
from signals.implementations.chop.adaptive_filter import AdaptiveChopFilterSignal


@jit(nopython=True)
def calculate_entry_signals(alpha_trend: np.ndarray, filter_signal: np.ndarray) -> np.ndarray:
    """エントリーシグナルを一度に計算（高速化版）"""
    signals = np.zeros_like(alpha_trend, dtype=np.int8)
    
    # ロングエントリー: アルファトレンドの買いシグナル + アダプティブチョピネスがトレンド相場
    long_condition = (alpha_trend == 1) & (filter_signal == 1)
    
    # ショートエントリー: アルファトレンドの売りシグナル + アダプティブチョピネスがトレンド相場
    short_condition = (alpha_trend == -1) & (filter_signal == 1)
    
    signals[long_condition] = 1
    signals[short_condition] = -1
    
    return signals


class AlphaTrendSignalGenerator(BaseSignalGenerator):
    """
    アルファトレンド+アダプティブチョピネスフィルターのシグナル生成クラス（両方向・高速化版）
    
    エントリー条件:
    - ロング: アルファトレンドのブレイクアウトで買いシグナル + アダプティブチョピネスがトレンド相場
    - ショート: アルファトレンドのブレイクアウトで売りシグナル + アダプティブチョピネスがトレンド相場
    
    エグジット条件:
    - ロング: アルファトレンドの売りシグナル
    - ショート: アルファトレンドの買いシグナル
    """
    
    def __init__(
        self,
        period: int = 13,
        max_kama_slow: int = 89,
        min_kama_slow: int = 30,
        max_kama_fast: int = 15,
        min_kama_fast: int = 2,
        max_atr_period: int = 120,
        min_atr_period: int = 13,
        max_multiplier: float = 3.0,
        min_multiplier: float = 0.5,
        chop_mfast: int = 13,
        chop_mslow: int = 120,
        max_threshold: float = 0.6,
        min_threshold: float = 0.4
    ):
        """初期化"""
        super().__init__("AlphaTrendSignalGenerator")
        
        # パラメータの設定
        self._params = {
            'period': period,
            'max_kama_slow': max_kama_slow,
            'min_kama_slow': min_kama_slow,
            'max_kama_fast': max_kama_fast,
            'min_kama_fast': min_kama_fast,
            'max_atr_period': max_atr_period,
            'min_atr_period': min_atr_period,
            'max_multiplier': max_multiplier,
            'min_multiplier': min_multiplier,
            'chop_mfast': chop_mfast,
            'chop_mslow': chop_mslow,
            'max_threshold': max_threshold,
            'min_threshold': min_threshold
        }
        
        # シグナル生成器の初期化
        self.signal = AlphaTrendBreakoutEntrySignal(
            period=period,
            max_kama_slow=max_kama_slow,
            min_kama_slow=min_kama_slow,
            max_kama_fast=max_kama_fast,
            min_kama_fast=min_kama_fast,
            max_atr_period=max_atr_period,
            min_atr_period=min_atr_period,
            max_multiplier=max_multiplier,
            min_multiplier=min_multiplier
        )
        self.filter_signal = AdaptiveChopFilterSignal(
            period=period,
            mfast=chop_mfast,
            mslow=chop_mslow,
            max_threshold=max_threshold,
            min_threshold=min_threshold
        )
        
        # キャッシュ用の変数（最小限に抑える）
        self._data_len = 0
        self._signals = None
        self._alpha_trend_signals = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算（高速化版）"""
        current_len = len(data)
        
        # データ長が変わった場合のみ再計算
        if self._signals is None or current_len != self._data_len:
            # データフレームの作成（必要な列のみ）
            df = data[['open', 'high', 'low', 'close']] if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
            
            # 各シグナルの計算（一度に実行）
            alpha_trend_signals = self.signal.generate(df)
            filter_signals = self.filter_signal.generate(df)
            
            # エントリーシグナルの計算（ベクトル化・高速化）
            self._signals = calculate_entry_signals(alpha_trend_signals, filter_signals)
            
            # エグジット用のシグナルを事前計算
            self._alpha_trend_signals = alpha_trend_signals
            
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
            return bool(self._alpha_trend_signals[index] == -1)
        elif position == -1:  # ショートポジション
            return bool(self._alpha_trend_signals[index] == 1)
        return False 