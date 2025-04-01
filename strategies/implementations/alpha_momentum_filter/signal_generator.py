#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
from numba import jit

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.alpha_momentum.entry import AlphaMomentumEntrySignal
from signals.implementations.alpha_filter.filter import AlphaFilterSignal


@jit(nopython=True)
def calculate_entry_signals(alpha_momentum: np.ndarray, filter_signal: np.ndarray) -> np.ndarray:
    """エントリーシグナルを一度に計算（高速化版）"""
    signals = np.zeros_like(alpha_momentum, dtype=np.int8)
    
    # ロングエントリー: アルファモメンタムの買いシグナル + アルファフィルターがトレンド相場
    long_condition = (alpha_momentum == 1) & (filter_signal == 1)
    
    # ショートエントリー: アルファモメンタムの売りシグナル + アルファフィルターがトレンド相場
    short_condition = (alpha_momentum == -1) & (filter_signal == 1)
    
    signals[long_condition] = 1
    signals[short_condition] = -1
    
    return signals


class AlphaMomentumFilterSignalGenerator(BaseSignalGenerator):
    """
    アルファモメンタム+アルファフィルターのシグナル生成クラス（両方向・高速化版）
    
    エントリー条件:
    - ロング: アルファモメンタムが買いシグナル(1) + アルファフィルターがトレンド相場(1)
    - ショート: アルファモメンタムが売りシグナル(-1) + アルファフィルターがトレンド相場(1)
    
    エグジット条件:
    - ロング: アルファモメンタムが売りシグナル(-1)
    - ショート: アルファモメンタムが買いシグナル(1)
    """
    
    def __init__(
        self,
        er_period: int = 21,
        # アルファモメンタム用パラメータ
        bb_max_period: int = 55,
        bb_min_period: int = 13,
        kc_max_period: int = 55,
        kc_min_period: int = 13,
        bb_max_mult: float = 2.0,
        bb_min_mult: float = 1.0,
        kc_max_mult: float = 3.0,
        kc_min_mult: float = 1.0,
        max_length: int = 34,
        min_length: int = 8,
        momentum_threshold: float = 0.0,
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
        super().__init__("AlphaMomentumFilterSignalGenerator")
        
        # パラメータの設定
        self._params = {
            'er_period': er_period,
            'bb_max_period': bb_max_period,
            'bb_min_period': bb_min_period,
            'kc_max_period': kc_max_period,
            'kc_min_period': kc_min_period,
            'bb_max_mult': bb_max_mult,
            'bb_min_mult': bb_min_mult,
            'kc_max_mult': kc_max_mult,
            'kc_min_mult': kc_min_mult,
            'max_length': max_length,
            'min_length': min_length,
            'momentum_threshold': momentum_threshold,
            'max_chop_period': max_chop_period,
            'min_chop_period': min_chop_period,
            'max_adx_period': max_adx_period,
            'min_adx_period': min_adx_period,
            'alma_offset': alma_offset,
            'alma_sigma': alma_sigma,
            'filter_threshold': filter_threshold
        }
        
        # アルファモメンタムシグナルの初期化
        self.alpha_momentum_signal = AlphaMomentumEntrySignal(
            er_period=er_period,
            bb_max_period=bb_max_period,
            bb_min_period=bb_min_period,
            kc_max_period=kc_max_period,
            kc_min_period=kc_min_period,
            bb_max_mult=bb_max_mult,
            bb_min_mult=bb_min_mult,
            kc_max_mult=kc_max_mult,
            kc_min_mult=kc_min_mult,
            max_length=max_length,
            min_length=min_length,
            alma_offset=alma_offset,
            alma_sigma=alma_sigma,
            momentum_threshold=momentum_threshold
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
        self._alpha_momentum_signals = None
    
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
                    alpha_momentum_signals = self.alpha_momentum_signal.generate(df)
                    filter_signals = self.alpha_filter_signal.generate(df)
                    
                    # エントリーシグナルの計算（ベクトル化・高速化）
                    self._signals = calculate_entry_signals(alpha_momentum_signals, filter_signals)
                    
                    # エグジット用のシグナルを事前計算
                    self._alpha_momentum_signals = alpha_momentum_signals
                except Exception as e:
                    self.logger.error(f"シグナル計算中にエラー: {str(e)}")
                    # エラー時はゼロシグナルを設定
                    self._signals = np.zeros(current_len, dtype=np.int8)
                    self._alpha_momentum_signals = np.zeros(current_len, dtype=np.int8)
                
                self._data_len = current_len
        except Exception as e:
            self.logger.error(f"calculate_signals全体でエラー: {str(e)}")
            # エラー時はゼロシグナルを設定
            if data is not None:
                self._signals = np.zeros(len(data), dtype=np.int8)
                self._alpha_momentum_signals = np.zeros(len(data), dtype=np.int8)
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
            return bool(self._alpha_momentum_signals[index] == -1)
        elif position == -1:  # ショートポジション
            return bool(self._alpha_momentum_signals[index] == 1)
        return False
    
    def get_momentum_signals(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        アルファモメンタムシグナルを取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: アルファモメンタムシグナル
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self._alpha_momentum_signals
        except Exception as e:
            self.logger.error(f"モメンタムシグナル取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_momentum_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        アルファモメンタム値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: モメンタム値
        """
        try:
            return self.alpha_momentum_signal.get_momentum(data)
        except Exception as e:
            self.logger.error(f"モメンタム値取得中にエラー: {str(e)}")
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