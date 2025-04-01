#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
from numba import jit

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.kernel_ma.direction import KernelMADirectionSignal
from signals.implementations.alpha_filter.filter import AlphaFilterSignal


@jit(nopython=True)
def calculate_entry_signals(kernel_ma_direction: np.ndarray, filter_signal: np.ndarray) -> np.ndarray:
    """エントリーシグナルを一度に計算（高速化版）"""
    signals = np.zeros_like(kernel_ma_direction, dtype=np.int8)
    
    # ロングエントリー: カーネルMAの上昇トレンド + アルファフィルターがトレンド相場
    long_condition = (kernel_ma_direction == 1) & (filter_signal == 1)
    
    # ショートエントリー: カーネルMAの下降トレンド + アルファフィルターがトレンド相場
    short_condition = (kernel_ma_direction == -1) & (filter_signal == 1)
    
    signals[long_condition] = 1
    signals[short_condition] = -1
    
    return signals


class KernelMAFilterSignalGenerator(BaseSignalGenerator):
    """
    カーネルMA+アルファフィルターのシグナル生成クラス（両方向・高速化版）
    
    エントリー条件:
    - ロング: カーネルMAが上昇トレンド(1) + アルファフィルターがトレンド相場(1)
    - ショート: カーネルMAが下降トレンド(-1) + アルファフィルターがトレンド相場(1)
    
    エグジット条件:
    - ロング: カーネルMAが下降トレンド(-1)または横ばい(0)
    - ショート: カーネルMAが上昇トレンド(1)または横ばい(0)
    """
    
    def __init__(
        self,
        er_period: int = 21,
        # カーネルMA用パラメータ
        max_bandwidth: float = 10.0,
        min_bandwidth: float = 2.0,
        kernel_type: str = 'gaussian',
        confidence_level: float = 0.95,
        slope_period: int = 5,
        slope_threshold: float = 0.0001,
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
        super().__init__("KernelMAFilterSignalGenerator")
        
        # パラメータの設定
        self._params = {
            'er_period': er_period,
            'max_bandwidth': max_bandwidth,
            'min_bandwidth': min_bandwidth,
            'kernel_type': kernel_type,
            'confidence_level': confidence_level,
            'slope_period': slope_period,
            'slope_threshold': slope_threshold,
            'max_chop_period': max_chop_period,
            'min_chop_period': min_chop_period,
            'max_adx_period': max_adx_period,
            'min_adx_period': min_adx_period,
            'alma_offset': alma_offset,
            'alma_sigma': alma_sigma,
            'filter_threshold': filter_threshold
        }
        
        # カーネルMA方向シグナルの初期化
        self.kernel_ma_signal = KernelMADirectionSignal(
            er_period=er_period,
            max_bandwidth=max_bandwidth,
            min_bandwidth=min_bandwidth,
            kernel_type=kernel_type,
            confidence_level=confidence_level,
            slope_period=slope_period,
            slope_threshold=slope_threshold
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
        self._kernel_ma_direction = None
    
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
                    kernel_ma_direction = self.kernel_ma_signal.generate(df)
                    filter_signals = self.alpha_filter_signal.generate(df)
                    
                    # エントリーシグナルの計算（ベクトル化・高速化）
                    self._signals = calculate_entry_signals(kernel_ma_direction, filter_signals)
                    
                    # エグジット用のシグナルを事前計算
                    self._kernel_ma_direction = kernel_ma_direction
                except Exception as e:
                    self.logger.error(f"シグナル計算中にエラー: {str(e)}")
                    # エラー時はゼロシグナルを設定
                    self._signals = np.zeros(current_len, dtype=np.int8)
                    self._kernel_ma_direction = np.zeros(current_len, dtype=np.int8)
                
                self._data_len = current_len
        except Exception as e:
            self.logger.error(f"calculate_signals全体でエラー: {str(e)}")
            # エラー時はゼロシグナルを設定
            if data is not None:
                self._signals = np.zeros(len(data), dtype=np.int8)
                self._kernel_ma_direction = np.zeros(len(data), dtype=np.int8)
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
            # 下降トレンド(-1)または横ばい(0)でエグジット
            return bool(self._kernel_ma_direction[index] <= 0)
        elif position == -1:  # ショートポジション
            # 上昇トレンド(1)または横ばい(0)でエグジット
            return bool(self._kernel_ma_direction[index] >= 0)
        return False
    
    def get_direction_signals(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        カーネルMA方向シグナルを取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: カーネルMA方向シグナル
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self._kernel_ma_direction
        except Exception as e:
            self.logger.error(f"方向シグナル取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_ma_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        カーネルMA値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: カーネルMA値
        """
        try:
            return self.kernel_ma_signal.get_ma_values(data)
        except Exception as e:
            self.logger.error(f"MA値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_bands(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        カーネルMAのバンド値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (上側バンド, 下側バンド)
        """
        try:
            upper = self.kernel_ma_signal.get_upper_band(data)
            lower = self.kernel_ma_signal.get_lower_band(data)
            return upper, lower
        except Exception as e:
            self.logger.error(f"バンド値取得中にエラー: {str(e)}")
            return np.array([]), np.array([])
    
    def get_slope(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        カーネルMAの傾き値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 傾き値
        """
        try:
            return self.kernel_ma_signal.get_slope(data)
        except Exception as e:
            self.logger.error(f"傾き値取得中にエラー: {str(e)}")
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