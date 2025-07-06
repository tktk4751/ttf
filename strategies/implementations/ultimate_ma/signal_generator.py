#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.ultimate_ma.ultimate_ma_entry import UltimateMAEntrySignal


@njit(fastmath=True, parallel=True)
def calculate_exit_signals_numba(realtime_trends: np.ndarray, trend_signals: np.ndarray, 
                                current_position: int, current_index: int) -> bool:
    """
    決済シグナル計算（Numba高速化版）
    
    Args:
        realtime_trends: リアルタイムトレンド信号の配列
        trend_signals: トレンドシグナルの配列
        current_position: 現在のポジション（1=ロング、-1=ショート）
        current_index: 現在のインデックス
    
    Returns:
        bool: 決済すべきかどうか
    """
    if current_index >= len(realtime_trends) or current_index >= len(trend_signals):
        return False
    
    # NaN値チェック
    if np.isnan(realtime_trends[current_index]) or np.isnan(trend_signals[current_index]):
        return False
    
    # ロングポジション決済: realtime_trends <= 0 または trend_signals == -1
    if current_position == 1:
        return realtime_trends[current_index] <= 0.0 or trend_signals[current_index] == -1
    # ショートポジション決済: realtime_trends >= 0 または trend_signals == 1
    elif current_position == -1:
        return realtime_trends[current_index] >= 0.0 or trend_signals[current_index] == 1
    
    return False


class UltimateMASignalGenerator(BaseSignalGenerator):
    """
    Ultimate MAのシグナル生成クラス（両方向・高速化版）
    
    エントリー条件:
    - ロング: realtime_trends > 0 かつ trend_signals == 1
    - ショート: realtime_trends < 0 かつ trend_signals == -1
    
    エグジット条件:
    - ロング決済: realtime_trends <= 0 または trend_signals == -1
    - ショート決済: realtime_trends >= 0 または trend_signals == 1
    """
    
    def __init__(
        self,
        # Ultimate MAパラメータ
        ultimate_smoother_period: int = 10,
        zero_lag_period: int = 21,
        realtime_window: int = 13,
        src_type: str = 'hlc3',
        slope_index: int = 1,
        range_threshold: float = 0.005,
        enable_exit_signals: bool = True,
        # 動的適応パラメータ
        zero_lag_period_mode: str = 'fixed',
        realtime_window_mode: str = 'fixed',
        # ゼロラグ用サイクル検出器パラメータ
        zl_cycle_detector_type: str = 'absolute_ultimate',
        zl_cycle_detector_cycle_part: float = 1.0,
        zl_cycle_detector_max_cycle: int = 120,
        zl_cycle_detector_min_cycle: int = 5,
        zl_cycle_period_multiplier: float = 1.0,
        # リアルタイムウィンドウ用サイクル検出器パラメータ
        rt_cycle_detector_type: str = 'phac_e',
        rt_cycle_detector_cycle_part: float = 1.0,
        rt_cycle_detector_max_cycle: int = 120,
        rt_cycle_detector_min_cycle: int = 5,
        rt_cycle_period_multiplier: float = 0.33,
        # period_rangeパラメータ
        zl_cycle_detector_period_range: Tuple[int, int] = (5, 120),
        rt_cycle_detector_period_range: Tuple[int, int] = (5, 120)
    ):
        """初期化"""
        super().__init__("UltimateMASignalGenerator")
        
        # パラメータの設定
        self._params = {
            'ultimate_smoother_period': ultimate_smoother_period,
            'zero_lag_period': zero_lag_period,
            'realtime_window': realtime_window,
            'src_type': src_type,
            'slope_index': slope_index,
            'range_threshold': range_threshold,
            'enable_exit_signals': enable_exit_signals,
            'zero_lag_period_mode': zero_lag_period_mode,
            'realtime_window_mode': realtime_window_mode,
            'zl_cycle_detector_type': zl_cycle_detector_type,
            'zl_cycle_detector_cycle_part': zl_cycle_detector_cycle_part,
            'zl_cycle_detector_max_cycle': zl_cycle_detector_max_cycle,
            'zl_cycle_detector_min_cycle': zl_cycle_detector_min_cycle,
            'zl_cycle_period_multiplier': zl_cycle_period_multiplier,
            'rt_cycle_detector_type': rt_cycle_detector_type,
            'rt_cycle_detector_cycle_part': rt_cycle_detector_cycle_part,
            'rt_cycle_detector_max_cycle': rt_cycle_detector_max_cycle,
            'rt_cycle_detector_min_cycle': rt_cycle_detector_min_cycle,
            'rt_cycle_period_multiplier': rt_cycle_period_multiplier,
            'zl_cycle_detector_period_range': zl_cycle_detector_period_range,
            'rt_cycle_detector_period_range': rt_cycle_detector_period_range
        }
        
        # Ultimate MAエントリーシグナルの初期化
        self.ultimate_ma_signal = UltimateMAEntrySignal(
            ultimate_smoother_period=ultimate_smoother_period,
            zero_lag_period=zero_lag_period,
            realtime_window=realtime_window,
            src_type=src_type,
            slope_index=slope_index,
            range_threshold=range_threshold,
            enable_exit_signals=enable_exit_signals,
            zero_lag_period_mode=zero_lag_period_mode,
            realtime_window_mode=realtime_window_mode,
            zl_cycle_detector_type=zl_cycle_detector_type,
            zl_cycle_detector_cycle_part=zl_cycle_detector_cycle_part,
            zl_cycle_detector_max_cycle=zl_cycle_detector_max_cycle,
            zl_cycle_detector_min_cycle=zl_cycle_detector_min_cycle,
            zl_cycle_period_multiplier=zl_cycle_period_multiplier,
            rt_cycle_detector_type=rt_cycle_detector_type,
            rt_cycle_detector_cycle_part=rt_cycle_detector_cycle_part,
            rt_cycle_detector_max_cycle=rt_cycle_detector_max_cycle,
            rt_cycle_detector_min_cycle=rt_cycle_detector_min_cycle,
            rt_cycle_period_multiplier=rt_cycle_period_multiplier,
            zl_cycle_detector_period_range=zl_cycle_detector_period_range,
            rt_cycle_detector_period_range=rt_cycle_detector_period_range
        )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._signals = None
        self._realtime_trends = None
        self._trend_signals = None
        self._ultimate_ma_result = None
    
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
                
                # Ultimate MAシグナルの計算
                try:
                    # エントリーシグナルの計算
                    ultimate_ma_signals = self.ultimate_ma_signal.generate(df)
                    
                    # Ultimate MAの結果を取得
                    self._ultimate_ma_result = self.ultimate_ma_signal.get_ultimate_ma_result(df)
                    
                    # リアルタイムトレンドとトレンドシグナルを取得
                    self._realtime_trends = self._ultimate_ma_result.realtime_trends
                    self._trend_signals = self._ultimate_ma_result.trend_signals
                    
                    # エントリーシグナルを設定
                    self._signals = ultimate_ma_signals
                    
                except Exception as e:
                    self.logger.error(f"Ultimate MAシグナル計算中にエラー: {str(e)}")
                    # エラー時はゼロシグナルを設定
                    self._signals = np.zeros(current_len, dtype=np.int8)
                    self._realtime_trends = np.zeros(current_len, dtype=np.float64)
                    self._trend_signals = np.zeros(current_len, dtype=np.int8)
                
                self._data_len = current_len
        except Exception as e:
            self.logger.error(f"calculate_signals全体でエラー: {str(e)}")
            # エラー時はゼロシグナルを設定
            if data is not None:
                self._signals = np.zeros(len(data), dtype=np.int8)
                self._realtime_trends = np.zeros(len(data), dtype=np.float64)
                self._trend_signals = np.zeros(len(data), dtype=np.int8)
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
        
        # Numba高速化された決済シグナル計算
        if self._realtime_trends is not None and self._trend_signals is not None:
            return calculate_exit_signals_numba(
                self._realtime_trends, 
                self._trend_signals, 
                position, 
                index
            )
        
        return False
    
    def get_ultimate_ma_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Ultimate MAの最終フィルター済み値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: Ultimate MAの最終フィルター済み値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            if self._ultimate_ma_result is not None:
                return self._ultimate_ma_result.values.copy()
            else:
                return np.array([])
        except Exception as e:
            self.logger.error(f"Ultimate MA値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_realtime_trends(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        リアルタイムトレンド信号を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: リアルタイムトレンド信号の配列
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            if self._realtime_trends is not None:
                return self._realtime_trends.copy()
            else:
                return np.array([])
        except Exception as e:
            self.logger.error(f"リアルタイムトレンド取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_trend_signals(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        トレンドシグナルを取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: トレンドシグナルの配列（1=上昇、-1=下降、0=range）
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            if self._trend_signals is not None:
                return self._trend_signals.copy()
            else:
                return np.array([])
        except Exception as e:
            self.logger.error(f"トレンドシグナル取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_noise_reduction_stats(self, data: Union[pd.DataFrame, np.ndarray] = None) -> dict:
        """
        ノイズ除去統計を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            dict: ノイズ除去統計
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            return self.ultimate_ma_signal.get_noise_reduction_stats(data)
        except Exception as e:
            self.logger.error(f"ノイズ除去統計取得中にエラー: {str(e)}")
            return {}
    
    def get_all_ultimate_ma_stages(self, data: Union[pd.DataFrame, np.ndarray] = None) -> dict:
        """
        Ultimate MAの全段階の結果を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            dict: 全段階の結果
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            if self._ultimate_ma_result is not None:
                return {
                    'raw_values': self._ultimate_ma_result.raw_values.copy(),
                    'kalman_values': self._ultimate_ma_result.kalman_values.copy(),
                    'super_smooth_values': self._ultimate_ma_result.super_smooth_values.copy(),
                    'zero_lag_values': self._ultimate_ma_result.zero_lag_values.copy(),
                    'final_values': self._ultimate_ma_result.values.copy(),
                    'amplitude': self._ultimate_ma_result.amplitude.copy(),
                    'phase': self._ultimate_ma_result.phase.copy(),
                    'realtime_trends': self._ultimate_ma_result.realtime_trends.copy(),
                    'trend_signals': self._ultimate_ma_result.trend_signals.copy(),
                    'current_trend': self._ultimate_ma_result.current_trend,
                    'current_trend_value': self._ultimate_ma_result.current_trend_value
                }
            else:
                return {}
        except Exception as e:
            self.logger.error(f"Ultimate MA全段階結果取得中にエラー: {str(e)}")
            return {} 