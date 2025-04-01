#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
from numba import jit

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.alpha_band.breakout_entry import AlphaBandBreakoutEntrySignal
from signals.implementations.alpha_trend_filter.filter import AlphaTrendFilterSignal


@jit(nopython=True)
def calculate_entry_signals(alpha_band: np.ndarray, filter_signal: np.ndarray) -> np.ndarray:
    """エントリーシグナルを一度に計算（高速化版）"""
    signals = np.zeros_like(alpha_band, dtype=np.int8)
    
    # ロングエントリー: アルファバンドの買いシグナル + アルファトレンドフィルターがトレンド相場
    long_condition = (alpha_band == 1) & (filter_signal == 1)
    
    # ショートエントリー: アルファバンドの売りシグナル + アルファトレンドフィルターがトレンド相場
    short_condition = (alpha_band == -1) & (filter_signal == 1)
    
    signals[long_condition] = 1
    signals[short_condition] = -1
    
    return signals


class AlphaBandTrendFilterSignalGenerator(BaseSignalGenerator):
    """
    アルファバンド+アルファトレンドフィルターのシグナル生成クラス（両方向・高速化版）
    
    エントリー条件:
    - ロング: アルファバンドのブレイクアウトで買いシグナル + アルファトレンドフィルターがトレンド相場
    - ショート: アルファバンドのブレイクアウトで売りシグナル + アルファトレンドフィルターがトレンド相場
    
    エグジット条件:
    - ロング: アルファバンドの売りシグナル
    - ショート: アルファバンドの買いシグナル
    """
    
    def __init__(
        self,
        # アルファバンドのパラメータ
        cycle_detector_type: str = 'hody_dc',
        lp_period: int = 5,
        hp_period: int = 144,
        cycle_part: float = 0.5,
        max_kama_period: int = 55,
        min_kama_period: int = 8,
        max_atr_period: int = 55,
        min_atr_period: int = 8,
        max_multiplier: float = 3.0,
        min_multiplier: float = 1.5,
        smoother_type: str = 'alma',
        band_lookback: int = 1,
        # アルファトレンドフィルターのパラメータ
        max_chop_period: int = 89,
        min_chop_period: int = 21,
        max_filter_atr_period: int = 89,
        min_filter_atr_period: int = 13,
        max_stddev_period: int = 13,
        min_stddev_period: int = 5,
        max_lookback_period: int = 13,
        min_lookback_period: int = 5,
        max_rms_window: int = 13,
        min_rms_window: int = 5,
        max_threshold: float = 0.75,
        min_threshold: float = 0.55,
        combination_weight: float = 0.6,
        combination_method: str = "sigmoid"
    ):
        """初期化"""
        super().__init__("AlphaBandTrendFilterSignalGenerator")
        
        # パラメータの設定
        self._params = {
            'cycle_detector_type': cycle_detector_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            'max_kama_period': max_kama_period,
            'min_kama_period': min_kama_period,
            'max_atr_period': max_atr_period,
            'min_atr_period': min_atr_period,
            'max_multiplier': max_multiplier,
            'min_multiplier': min_multiplier,
            'smoother_type': smoother_type,
            'band_lookback': band_lookback,
            'max_chop_period': max_chop_period,
            'min_chop_period': min_chop_period,
            'max_filter_atr_period': max_filter_atr_period,
            'min_filter_atr_period': min_filter_atr_period,
            'max_stddev_period': max_stddev_period,
            'min_stddev_period': min_stddev_period,
            'max_lookback_period': max_lookback_period,
            'min_lookback_period': min_lookback_period,
            'max_rms_window': max_rms_window,
            'min_rms_window': min_rms_window,
            'max_threshold': max_threshold,
            'min_threshold': min_threshold,
            'combination_weight': combination_weight,
            'combination_method': combination_method
        }
        
        # アルファバンドブレイクアウトシグナルの初期化
        self.alpha_band_signal = AlphaBandBreakoutEntrySignal(
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            max_kama_period=max_kama_period,
            min_kama_period=min_kama_period,
            max_atr_period=max_atr_period,
            min_atr_period=min_atr_period,
            max_multiplier=max_multiplier,
            min_multiplier=min_multiplier,
            smoother_type=smoother_type,
            lookback=band_lookback
        )
        
        # アルファトレンドフィルターシグナルの初期化
        self.alpha_trend_filter_signal = AlphaTrendFilterSignal(
            max_chop_period=max_chop_period,
            min_chop_period=min_chop_period,
            max_atr_period=max_filter_atr_period,
            min_atr_period=min_filter_atr_period,
            max_stddev_period=max_stddev_period,
            min_stddev_period=min_stddev_period,
            max_lookback_period=max_lookback_period,
            min_lookback_period=min_lookback_period,
            max_rms_window=max_rms_window,
            min_rms_window=min_rms_window,
            max_threshold=max_threshold,
            min_threshold=min_threshold,
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            combination_weight=combination_weight,
            combination_method=combination_method
        )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._signals = None
        self._alpha_band_signals = None
    
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
                    alpha_band_signals = self.alpha_band_signal.generate(df)
                    filter_signals = self.alpha_trend_filter_signal.generate(df)
                    
                    # エントリーシグナルの計算（ベクトル化・高速化）
                    self._signals = calculate_entry_signals(alpha_band_signals, filter_signals)
                    
                    # エグジット用のシグナルを事前計算
                    self._alpha_band_signals = alpha_band_signals
                except Exception as e:
                    self.logger.error(f"シグナル計算中にエラー: {str(e)}")
                    # エラー時はゼロシグナルを設定
                    self._signals = np.zeros(current_len, dtype=np.int8)
                    self._alpha_band_signals = np.zeros(current_len, dtype=np.int8)
                
                self._data_len = current_len
        except Exception as e:
            self.logger.error(f"calculate_signals全体でエラー: {str(e)}")
            # エラー時はゼロシグナルを設定
            if data is not None:
                self._signals = np.zeros(len(data), dtype=np.int8)
                self._alpha_band_signals = np.zeros(len(data), dtype=np.int8)
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
            return bool(self._alpha_band_signals[index] == -1)
        elif position == -1:  # ショートポジション
            return bool(self._alpha_band_signals[index] == 1)
        return False
    
    def get_band_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        アルファバンドのバンド値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (中心線, 上限バンド, 下限バンド)のタプル
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.alpha_band_signal.get_band_values()
        except Exception as e:
            self.logger.error(f"バンド値取得中にエラー: {str(e)}")
            # エラー時は空の配列を返す
            empty = np.array([])
            return empty, empty, empty
    
    def get_efficiency_ratio(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        サイクル効率比（CER）の値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: サイクル効率比の値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.alpha_band_signal.get_cycle_efficiency_ratio()
        except Exception as e:
            self.logger.error(f"効率比取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_filter_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        アルファトレンドフィルターの値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: フィルター値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.alpha_trend_filter_signal.get_filter_values()
        except Exception as e:
            self.logger.error(f"フィルター値取得中にエラー: {str(e)}")
            return np.array([])
        
    def get_threshold_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        アルファトレンドフィルターの動的しきい値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 動的しきい値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.alpha_trend_filter_signal.get_threshold_values()
        except Exception as e:
            self.logger.error(f"しきい値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_trend_index(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        アルファトレンドインデックスの値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: トレンドインデックス値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.alpha_trend_filter_signal.get_trend_index()
        except Exception as e:
            self.logger.error(f"トレンドインデックス取得中にエラー: {str(e)}")
            return np.array([]) 