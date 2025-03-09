#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from numba import jit

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.trend_alpha.breakout_entry import TrendAlphaBreakoutEntrySignal
from signals.implementations.trend_quality.filter import TrendQualityFilterSignal


@jit(nopython=True)
def calculate_entry_signals(trend_alpha: np.ndarray, trend_quality_filter: np.ndarray) -> np.ndarray:
    """エントリーシグナルを一度に計算（高速化版）"""
    signals = np.zeros_like(trend_alpha, dtype=np.int8)
    
    # ロングエントリー: TrendAlphaの買いシグナルかつトレンドクオリティフィルターが1
    long_condition = (trend_alpha == 1) & (trend_quality_filter == 1)
    
    # ショートエントリー: TrendAlphaの売りシグナルかつトレンドクオリティフィルターが1
    short_condition = (trend_alpha == -1) & (trend_quality_filter == 1)
    
    signals[long_condition] = 1
    signals[short_condition] = -1
    
    return signals


class TrendAlphaV2SignalGenerator(BaseSignalGenerator):
    """
    TrendAlpha V2のシグナル生成クラス（両方向・高速化版）
    
    エントリー条件:
    - ロング: TrendAlphaのブレイクアウトで買いシグナル
    - ショート: TrendAlphaのブレイクアウトで売りシグナル
    
    エグジット条件:
    - ロング: TrendAlphaの売りシグナル
    - ショート: TrendAlphaの買いシグナル
    """
    
    def __init__(
        self,
        period: int = 175,
        max_kama_slow: int = 55,
        min_kama_slow: int = 30,
        max_kama_fast: int = 13,
        min_kama_fast: int = 2,
        max_atr_period: int = 120,
        min_atr_period: int = 5,
        max_multiplier: float = 3.0,
        min_multiplier: float = 1.0,
        trend_quality_max_period: int = 120,
        trend_quality_min_period: int = 10,
        trend_quality_threshold: float = 0.5
    ):
        """初期化"""
        super().__init__("TrendAlphaV2SignalGenerator")
        
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
            'trend_quality_max_period': trend_quality_max_period,
            'trend_quality_min_period': trend_quality_min_period,
            'trend_quality_threshold': trend_quality_threshold
        }
        
        # シグナル生成器の初期化
        self.signal = TrendAlphaBreakoutEntrySignal(
            kama_period=period,
            max_kama_slow=max_kama_slow,
            min_kama_slow=min_kama_slow,
            max_kama_fast=max_kama_fast,
            min_kama_fast=min_kama_fast,
            max_atr_period=max_atr_period,
            min_atr_period=min_atr_period,
            max_multiplier=max_multiplier,
            min_multiplier=min_multiplier
        )
        
        # トレンドクオリティフィルターの初期化
        self.trend_quality_filter = TrendQualityFilterSignal(
            er_period=period,
            max_period=trend_quality_max_period,
            min_period=trend_quality_min_period,
            solid={'trend_quality_threshold': trend_quality_threshold}
        )
        
        # キャッシュ用の変数（最小限に抑える）
        self._data_len = 0
        self._signals = None
        self._trend_alpha_signals = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算（高速化版）"""
        current_len = len(data)
        
        # データ長が変わった場合のみ再計算
        if self._signals is None or current_len != self._data_len:
            # データフレームの作成（必要な列のみ）
            df = data[['open', 'high', 'low', 'close']] if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
            
            # TrendAlphaシグナルの計算
            trend_alpha_signals = self.signal.generate(df)
            
            # トレンドクオリティフィルターの計算
            trend_quality_signals = self.trend_quality_filter.generate(df)
            
            # エントリーシグナルの計算（ベクトル化・高速化）
            self._signals = calculate_entry_signals(trend_alpha_signals, trend_quality_signals)
            
            # エグジット用のシグナルを事前計算
            self._trend_alpha_signals = trend_alpha_signals
            
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
            return bool(self._trend_alpha_signals[index] == -1)
        elif position == -1:  # ショートポジション
            return bool(self._trend_alpha_signals[index] == 1)
        return False 