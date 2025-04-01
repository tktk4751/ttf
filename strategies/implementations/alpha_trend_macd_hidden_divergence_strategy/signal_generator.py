#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
from numba import jit
import logging

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.alpha_trend.direction import AlphaTrendDirectionSignal
from signals.implementations.divergence.alpha_macd_hidden_divergence import AlphaMACDHiddenDivergenceSignal

# ロガーの設定
logger = logging.getLogger(__name__)

@jit(nopython=True)
def calculate_entry_signals(alpha_trend: np.ndarray, alpha_macd_hidden_div: np.ndarray) -> np.ndarray:
    """エントリーシグナルを一度に計算（高速化版）"""
    signals = np.zeros_like(alpha_trend, dtype=np.int8)
    
    # ロングエントリー: AlphaTrendDirectionSignalが1 + AlphaMACDHiddenDivergenceSignalが1
    long_condition = (alpha_trend == 1) & (alpha_macd_hidden_div == 1)
    
    # ショートエントリー: AlphaTrendDirectionSignalが-1 + AlphaMACDHiddenDivergenceSignalが-1
    short_condition = (alpha_trend == -1) & (alpha_macd_hidden_div == -1)
    
    signals[long_condition] = 1
    signals[short_condition] = -1
    
    return signals


class AlphaTrendMACDHiddenDivergenceSignalGenerator(BaseSignalGenerator):
    """
    AlphaTrend + AlphaMACDヒドゥンダイバージェンスのシグナル生成クラス（両方向・高速化版）
    
    エントリー条件:
    - ロング: AlphaTrendDirectionSignalが1 + AlphaMACDHiddenDivergenceSignalが1
    - ショート: AlphaTrendDirectionSignalが-1 + AlphaMACDHiddenDivergenceSignalが-1
    
    エグジット条件:
    - ロング: AlphaTrendDirectionSignalが-1
    - ショート: AlphaTrendDirectionSignalが1
    """
    
    def __init__(
        self,
        # 共通パラメータ
        er_period: int = 21,
        # AlphaTrend用パラメータ
        max_percentile_length: int = 55,
        min_percentile_length: int = 13,
        max_atr_period: int = 89,
        min_atr_period: int = 13,
        max_trend_multiplier: float = 3.0,
        min_trend_multiplier: float = 1.0,
        alma_offset: float = 0.85,
        alma_sigma: int = 6,
        # AlphaMACDHiddenDivergence用パラメータ
        fast_max_kama_period: int = 89,
        fast_min_kama_period: int = 8,
        slow_max_kama_period: int = 144,
        slow_min_kama_period: int = 21,
        signal_max_kama_period: int = 55,
        signal_min_kama_period: int = 5,
        max_slow_period: int = 89,
        min_slow_period: int = 30,
        max_fast_period: int = 13,
        min_fast_period: int = 2,
        lookback: int = 30
    ):
        """初期化"""
        super().__init__("AlphaTrendMACDHiddenDivergenceSignalGenerator")
        
        # ロガーの設定
        self.logger = logger
        
        # パラメータの設定
        self._params = {
            'er_period': er_period,
            'max_percentile_length': max_percentile_length,
            'min_percentile_length': min_percentile_length,
            'max_atr_period': max_atr_period,
            'min_atr_period': min_atr_period,
            'max_trend_multiplier': max_trend_multiplier,
            'min_trend_multiplier': min_trend_multiplier,
            'alma_offset': alma_offset,
            'alma_sigma': alma_sigma,
            'fast_max_kama_period': fast_max_kama_period,
            'fast_min_kama_period': fast_min_kama_period,
            'slow_max_kama_period': slow_max_kama_period,
            'slow_min_kama_period': slow_min_kama_period,
            'signal_max_kama_period': signal_max_kama_period,
            'signal_min_kama_period': signal_min_kama_period,
            'max_slow_period': max_slow_period,
            'min_slow_period': min_slow_period,
            'max_fast_period': max_fast_period,
            'min_fast_period': min_fast_period,
            'lookback': lookback
        }
        
        # AlphaTrend方向シグナルの初期化
        self.alpha_trend_signal = AlphaTrendDirectionSignal(
            er_period=er_period,
            max_percentile_length=max_percentile_length,
            min_percentile_length=min_percentile_length,
            max_atr_period=max_atr_period,
            min_atr_period=min_atr_period,
            max_multiplier=max_trend_multiplier,
            min_multiplier=min_trend_multiplier,
            alma_offset=alma_offset,
            alma_sigma=alma_sigma
        )
        
        # AlphaMACDヒドゥンダイバージェンスシグナルの初期化
        self.alpha_macd_hidden_div_signal = AlphaMACDHiddenDivergenceSignal(
            er_period=er_period,
            fast_max_kama_period=fast_max_kama_period,
            fast_min_kama_period=fast_min_kama_period,
            slow_max_kama_period=slow_max_kama_period,
            slow_min_kama_period=slow_min_kama_period,
            signal_max_kama_period=signal_max_kama_period,
            signal_min_kama_period=signal_min_kama_period,
            max_slow_period=max_slow_period,
            min_slow_period=min_slow_period,
            max_fast_period=max_fast_period,
            min_fast_period=min_fast_period,
            lookback=lookback
        )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._signals = None
        self._alpha_trend_signals = None
        self._alpha_macd_hidden_div_signals = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算（高速化版）"""
        try:
            current_len = len(data)
            
            # データ長が変わった場合のみ再計算
            if self._signals is None or current_len != self._data_len:
                # データフレームの作成（必要な列のみ）
                if isinstance(data, pd.DataFrame):
                    df = data
                else:
                    df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
                
                # 各シグナルの計算
                try:
                    alpha_trend_signals = self.alpha_trend_signal.generate(df)
                    alpha_macd_hidden_div_signals = self.alpha_macd_hidden_div_signal.generate(df)
                    
                    # エントリーシグナルの計算（ベクトル化・高速化）
                    self._signals = calculate_entry_signals(alpha_trend_signals, alpha_macd_hidden_div_signals)
                    
                    # エグジット用のシグナルを事前計算
                    self._alpha_trend_signals = alpha_trend_signals
                    self._alpha_macd_hidden_div_signals = alpha_macd_hidden_div_signals
                except Exception as e:
                    self.logger.error(f"シグナル計算中にエラー: {str(e)}")
                    # エラー時はゼロシグナルを設定
                    self._signals = np.zeros(current_len, dtype=np.int8)
                    self._alpha_trend_signals = np.zeros(current_len, dtype=np.int8)
                    self._alpha_macd_hidden_div_signals = np.zeros(current_len, dtype=np.int8)
                
                self._data_len = current_len
        except Exception as e:
            self.logger.error(f"calculate_signals全体でエラー: {str(e)}")
            # エラー時はゼロシグナルを設定
            if data is not None:
                self._signals = np.zeros(len(data), dtype=np.int8)
                self._alpha_trend_signals = np.zeros(len(data), dtype=np.int8)
                self._alpha_macd_hidden_div_signals = np.zeros(len(data), dtype=np.int8)
                self._data_len = len(data)
    
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
            # AlphaTrendDirectionSignalが-1になったらロング決済
            return bool(self._alpha_trend_signals[index] == -1)
        elif position == -1:  # ショートポジション
            # AlphaTrendDirectionSignalが1になったらショート決済
            return bool(self._alpha_trend_signals[index] == 1)
        return False
    
    def get_trend_signals(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        AlphaTrendシグナルを取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: AlphaTrendシグナル
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self._alpha_trend_signals
        except Exception as e:
            self.logger.error(f"トレンドシグナル取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_macd_hidden_div_signals(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        AlphaMACDヒドゥンダイバージェンスシグナルを取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: AlphaMACDヒドゥンダイバージェンスシグナル
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self._alpha_macd_hidden_div_signals
        except Exception as e:
            self.logger.error(f"MACDヒドゥンダイバージェンスシグナル取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_alpha_macd_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        AlphaMACDの値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Dict[str, np.ndarray]: AlphaMACDの値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.alpha_macd_hidden_div_signal.get_alpha_macd_values(data)
        except Exception as e:
            self.logger.error(f"AlphaMACD値取得中にエラー: {str(e)}")
            return {'macd': np.array([]), 'signal': np.array([]), 'histogram': np.array([])}
    
    def get_divergence_states(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        ヒドゥンダイバージェンスの状態を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Dict[str, np.ndarray]: ヒドゥンダイバージェンスの状態
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.alpha_macd_hidden_div_signal.get_divergence_states(data)
        except Exception as e:
            self.logger.error(f"ダイバージェンス状態取得中にエラー: {str(e)}")
            return {'bullish': np.array([]), 'bearish': np.array([])} 