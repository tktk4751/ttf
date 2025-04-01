#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
from numba import jit
import logging

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.alpha_ma.direction import AlphaMATrendFollowingSignal
from signals.implementations.alpha_filter.filter import AlphaFilterSignal

# ロガーの設定
logger = logging.getLogger(__name__)

@jit(nopython=True)
def calculate_entry_signals(alpha_ma_trend: np.ndarray, filter_signal: np.ndarray) -> np.ndarray:
    """エントリーシグナルを一度に計算（高速化版）"""
    signals = np.zeros_like(alpha_ma_trend, dtype=np.int8)
    
    # ロングエントリー: AlphaMAトレンドの買いシグナル(1) + アルファフィルターがトレンド相場(1)
    long_condition = (alpha_ma_trend == 1) & (filter_signal == 1)
    
    # ショートエントリー: AlphaMAトレンドの売りシグナル(-1) + アルファフィルターがトレンド相場(1)
    short_condition = (alpha_ma_trend == -1) & (filter_signal == 1)
    
    signals[long_condition] = 1
    signals[short_condition] = -1
    
    return signals


class AlphaMAFilterSignalGenerator(BaseSignalGenerator):
    """
    AlphaMA+アルファフィルターのシグナル生成クラス（両方向・高速化版）
    
    エントリー条件:
    - ロング: AlphaMATrendFollowingSignalが上昇トレンド(1) + アルファフィルターがトレンド相場(1)
    - ショート: AlphaMATrendFollowingSignalが下降トレンド(-1) + アルファフィルターがトレンド相場(1)
    
    エグジット条件:
    - ロング: AlphaMATrendFollowingSignalが下降トレンド(-1)
    - ショート: AlphaMATrendFollowingSignalが上昇トレンド(1)
    """
    
    def __init__(
        self,
        # 共通パラメータ
        er_period: int = 21,
        # AlphaMA用パラメータ

        short_max_kama_period: int = 89,
        short_min_kama_period: int = 8,
        long_max_kama_period: int = 144,
        long_min_kama_period: int = 21,
        max_slow_period: int = 89,
        min_slow_period: int = 30,
        max_fast_period: int = 13,
        min_fast_period: int = 2,
        # アルファフィルター用パラメータ
        max_chop_period: int = 55,
        min_chop_period: int = 8,
        max_adx_period: int = 21,
        min_adx_period: int = 5,
        alma_offset: float = 0.85,
        alma_sigma: float = 6,
        filter_threshold: float = 0.5
    ):
        """初期化"""
        super().__init__("AlphaMAFilterSignalGenerator")
        
        # ロガーの設定
        self.logger = logger
        
        # パラメータの設定
        self._params = {
            'er_period': er_period,
            'short_er_period': er_period,
            'long_er_period': er_period,
            'short_max_kama_period': short_max_kama_period,
            'short_min_kama_period': short_min_kama_period,
            'long_max_kama_period': long_max_kama_period,
            'long_min_kama_period': long_min_kama_period,
            'max_slow_period': max_slow_period,
            'min_slow_period': min_slow_period,
            'max_fast_period': max_fast_period,
            'min_fast_period': min_fast_period,
            'max_chop_period': max_chop_period,
            'min_chop_period': min_chop_period,
            'max_adx_period': max_adx_period,
            'min_adx_period': min_adx_period,
            'alma_offset': alma_offset,
            'alma_sigma': alma_sigma,
            'filter_threshold': filter_threshold
        }
        
        # AlphaMA方向シグナルの初期化
        self.alpha_ma_trend_signal = AlphaMATrendFollowingSignal(
            short_er_period=er_period,
            long_er_period=er_period,
            short_max_kama_period=short_max_kama_period,
            short_min_kama_period=short_min_kama_period,
            long_max_kama_period=long_max_kama_period,
            long_min_kama_period=long_min_kama_period,
            max_slow_period=max_slow_period,
            min_slow_period=min_slow_period,
            max_fast_period=max_fast_period,
            min_fast_period=min_fast_period
        )
        
        # アルファフィルターシグナルの初期化
        self.alpha_filter_signal = AlphaFilterSignal(
            er_period=er_period,
            max_chop_period=max_chop_period,
            min_chop_period=min_chop_period,
            max_adx_period=max_adx_period,
            min_adx_period=min_adx_period,
            alma_period=10,  # 固定値
            alma_offset=alma_offset,
            alma_sigma=alma_sigma,
            solid={
                'threshold': filter_threshold
            }
        )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._signals = None
        self._alpha_ma_trend_signals = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算（高速化版）"""
        try:
            current_len = len(data)
            
            # データ長が変わった場合のみ再計算
            if self._signals is None or current_len != self._data_len:
                # データフレームの作成（必要な列のみ）
                if isinstance(data, pd.DataFrame):
                    df = data[['high', 'low', 'close']]
                else:
                    df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
                
                # 各シグナルの計算
                try:
                    alpha_ma_trend_signals = self.alpha_ma_trend_signal.generate(df)
                    filter_signals = self.alpha_filter_signal.generate(df)
                    
                    # エントリーシグナルの計算（ベクトル化・高速化）
                    self._signals = calculate_entry_signals(alpha_ma_trend_signals, filter_signals)
                    
                    # エグジット用のシグナルを事前計算
                    self._alpha_ma_trend_signals = alpha_ma_trend_signals
                except Exception as e:
                    self.logger.error(f"シグナル計算中にエラー: {str(e)}")
                    # エラー時はゼロシグナルを設定
                    self._signals = np.zeros(current_len, dtype=np.int8)
                    self._alpha_ma_trend_signals = np.zeros(current_len, dtype=np.int8)
                
                self._data_len = current_len
        except Exception as e:
            self.logger.error(f"calculate_signals全体でエラー: {str(e)}")
            # エラー時はゼロシグナルを設定
            if data is not None:
                self._signals = np.zeros(len(data), dtype=np.int8)
                self._alpha_ma_trend_signals = np.zeros(len(data), dtype=np.int8)
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
            return bool(self._alpha_ma_trend_signals[index] == -1)
        elif position == -1:  # ショートポジション
            return bool(self._alpha_ma_trend_signals[index] == 1)
        return False
    
    def get_trend_signals(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        AlphaMAトレンドシグナルを取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: AlphaMAトレンドシグナル
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self._alpha_ma_trend_signals
        except Exception as e:
            self.logger.error(f"トレンドシグナル取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_filter_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        アルファフィルターの値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: フィルター値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.alpha_filter_signal.get_filter_values()
        except Exception as e:
            self.logger.error(f"フィルター値取得中にエラー: {str(e)}")
            return np.array([])
        
    def get_filter_components(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        アルファフィルターのコンポーネント値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Dict[str, np.ndarray]: コンポーネント値のディクショナリ
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.alpha_filter_signal.get_component_values()
        except Exception as e:
            self.logger.error(f"フィルターコンポーネント取得中にエラー: {str(e)}")
            # エラー時は空のディクショナリを返す
            return {'er': np.array([]), 'alpha_chop': np.array([]), 'alpha_adx': np.array([]), 'dynamic_period': np.array([])} 