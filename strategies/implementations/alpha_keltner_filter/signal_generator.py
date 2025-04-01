#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
from numba import jit

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.alpha_keltner.breakout_entry import AlphaKeltnerBreakoutEntrySignal
from signals.implementations.alpha_filter.filter import AlphaFilterSignal


# 一時的にjitを無効化
# @jit(nopython=True)

def calculate_entry_signals(alpha_keltner: np.ndarray, filter_signal: np.ndarray) -> np.ndarray:
    """エントリーシグナルを一度に計算（高速化版）"""
    signals = np.zeros_like(alpha_keltner, dtype=np.int8)
    
    # ロングエントリー: アルファケルトナーの買いシグナル + アルファフィルターがトレンド相場
    long_condition = (alpha_keltner == 1) & (filter_signal == 1)
    
    # ショートエントリー: アルファケルトナーの売りシグナル + アルファフィルターがトレンド相場
    short_condition = (alpha_keltner == -1) & (filter_signal == 1)
    
    signals[long_condition] = 1
    signals[short_condition] = -1
    
    return signals


class AlphaKeltnerFilterSignalGenerator(BaseSignalGenerator):
    """
    アルファケルトナーチャネル+アルファフィルターのシグナル生成クラス（両方向・高速化版）
    
    エントリー条件:
    - ロング: アルファケルトナーのブレイクアウトで買いシグナル + アルファフィルターがトレンド相場
    - ショート: アルファケルトナーのブレイクアウトで売りシグナル + アルファフィルターがトレンド相場
    
    エグジット条件:
    - ロング: アルファケルトナーの売りシグナル
    - ショート: アルファケルトナーの買いシグナル
    """
    
    def __init__(
        self,
        er_period: int = 21,
        max_kama_period: int = 89,
        min_kama_period: int = 13,
        max_atr_period: int = 89,
        min_atr_period: int = 13,
        max_keltner_multiplier: float = 3.0,
        min_keltner_multiplier: float = 0.5,
        lookback: int = 1,
        max_chop_period: int = 55,
        min_chop_period: int = 8,
        filter_threshold: float = 0.5
    ):
        """初期化"""
        super().__init__("AlphaKeltnerFilterSignalGenerator")
        
        # パラメータの設定
        self._params = {
            'er_period': er_period,
            'max_kama_period': max_kama_period,
            'min_kama_period': min_kama_period,
            'max_atr_period': max_atr_period,
            'min_atr_period': min_atr_period,
            'max_keltner_multiplier': max_keltner_multiplier,
            'min_keltner_multiplier': min_keltner_multiplier,
            'lookback': lookback,
            'max_chop_period': max_chop_period,
            'min_chop_period': min_chop_period,
            'filter_threshold': filter_threshold
        }
        
        # アルファケルトナーブレイクアウトシグナルの初期化
        self.alpha_keltner_signal = AlphaKeltnerBreakoutEntrySignal(
            er_period=er_period,
            max_kama_period=max_kama_period,
            min_kama_period=min_kama_period,
            max_atr_period=max_atr_period,
            min_atr_period=min_atr_period,
            max_multiplier=max_keltner_multiplier,
            min_multiplier=min_keltner_multiplier,
            lookback=lookback
        )
        
        # アルファフィルターシグナルの初期化
        self.alpha_filter_signal = AlphaFilterSignal(
            er_period=er_period,
            max_chop_period=max_chop_period,
            min_chop_period=min_chop_period,
            solid={
                'threshold': filter_threshold
            }
        )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._signals = None
        self._alpha_keltner_signals = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算（高速化版）"""
        try:
            current_len = len(data)
            
            # データ長が変わった場合のみ再計算
            if self._signals is None or current_len != self._data_len:
                # データフレームの作成（必要な列のみ）
                if isinstance(data, pd.DataFrame):
                    df = data[['open', 'high', 'low', 'close']]
                else:
                    df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
                
                # 各シグナルの計算
                try:
                    alpha_keltner_signals = self.alpha_keltner_signal.generate(df)
                    filter_signals = self.alpha_filter_signal.generate(df)
                    
                    # エントリーシグナルの計算（ベクトル化・高速化）
                    self._signals = calculate_entry_signals(alpha_keltner_signals, filter_signals)
                    
                    # エグジット用のシグナルを事前計算
                    self._alpha_keltner_signals = alpha_keltner_signals
                except Exception as e:
                    self.logger.error(f"シグナル計算中にエラー: {str(e)}")
                    # エラー時はゼロシグナルを設定
                    self._signals = np.zeros(current_len, dtype=np.int8)
                    self._alpha_keltner_signals = np.zeros(current_len, dtype=np.int8)
                
                self._data_len = current_len
        except Exception as e:
            self.logger.error(f"calculate_signals全体でエラー: {str(e)}")
            # エラー時はゼロシグナルを設定
            if data is not None:
                self._signals = np.zeros(len(data), dtype=np.int8)
                self._alpha_keltner_signals = np.zeros(len(data), dtype=np.int8)
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
            return bool(self._alpha_keltner_signals[index] == -1)
        elif position == -1:  # ショートポジション
            return bool(self._alpha_keltner_signals[index] == 1)
        return False
    
    def get_keltner_bands(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        ケルトナーチャネルのバンド値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 上部バンドと下部バンド
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.alpha_keltner_signal.get_band_values()
        except Exception as e:
            self.logger.error(f"ケルトナーバンド取得中にエラー: {str(e)}")
            # エラー時は空の配列を返す
            empty = np.array([])
            return empty, empty
    
    def get_efficiency_ratio(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        効率比（ER）の値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 効率比の値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.alpha_keltner_signal.get_efficiency_ratio()
        except Exception as e:
            self.logger.error(f"効率比取得中にエラー: {str(e)}")
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