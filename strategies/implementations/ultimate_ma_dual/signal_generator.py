#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.ultimate_ma.ultimate_ma_dual_entry import UltimateMADualEntrySignal


@njit(fastmath=True, parallel=True)
def calculate_dual_exit_signals_numba(long_realtime_trends: np.ndarray, 
                                     long_trend_signals: np.ndarray,
                                     current_position: int, current_index: int) -> bool:
    """
    デュアルUltimate MAの決済シグナル計算（Numba高速化版）
    
    Args:
        long_realtime_trends: 長期Ultimate MAのrealtime_trendsの配列
        long_trend_signals: 長期Ultimate MAのトレンドシグナルの配列
        current_position: 現在のポジション（1=ロング、-1=ショート）
        current_index: 現在のインデックス
    
    Returns:
        bool: 決済すべきかどうか
    """
    if current_index >= len(long_realtime_trends) or current_index >= len(long_trend_signals):
        return False
    
    # NaN値チェック
    if (np.isnan(long_realtime_trends[current_index]) or 
        np.isnan(long_trend_signals[current_index])):
        return False
    
    # ロングポジション決済: 長期 realtime_trends <= 0 または 長期 trend_signals == -1
    if current_position == 1:
        return (long_realtime_trends[current_index] <= 0.0 or 
                long_trend_signals[current_index] == -1)
    # ショートポジション決済: 長期 realtime_trends >= 0 または 長期 trend_signals == 1
    elif current_position == -1:
        return (long_realtime_trends[current_index] >= 0.0 or 
                long_trend_signals[current_index] == 1)
    
    return False


class UltimateMADualSignalGenerator(BaseSignalGenerator):
    """
    Ultimate MAデュアルエントリーのシグナル生成クラス（高速化版）
    
    エントリー条件:
    - ロングエントリー: 短期 realtime_trends > 0 かつ 短期 trend_signals == 1 かつ 長期 trend_signals == 1
    - ショートエントリー: 短期 realtime_trends < 0 かつ 短期 trend_signals == -1 かつ 長期 trend_signals == -1
    
    エグジット条件:
    - ロング決済: 長期 realtime_trends <= 0 または 長期 trend_signals == -1
    - ショート決済: 長期 realtime_trends >= 0 または 長期 trend_signals == 1
    """
    
    def __init__(
        self,
        # 短期Ultimate MAパラメータ
        short_super_smooth_period: int = 10,
        short_zero_lag_period: int = 21,
        short_realtime_window: int = 13,
        short_src_type: str = 'hlc3',
        short_slope_index: int = 1,
        short_range_threshold: float = 0.005,
        short_zero_lag_period_mode: str = 'fixed',
        short_realtime_window_mode: str = 'fixed',
        short_zl_cycle_detector_type: str = 'absolute_ultimate',
        short_zl_cycle_detector_cycle_part: float = 1.0,
        short_zl_cycle_detector_max_cycle: int = 120,
        short_zl_cycle_detector_min_cycle: int = 5,
        short_zl_cycle_period_multiplier: float = 1.0,
        short_rt_cycle_detector_type: str = 'phac_e',
        short_rt_cycle_detector_cycle_part: float = 1.0,
        short_rt_cycle_detector_max_cycle: int = 120,
        short_rt_cycle_detector_min_cycle: int = 5,
        short_rt_cycle_period_multiplier: float = 0.33,
        short_zl_cycle_detector_period_range: Tuple[int, int] = (5, 120),
        short_rt_cycle_detector_period_range: Tuple[int, int] = (5, 120),
        
        # 長期Ultimate MAパラメータ
        long_super_smooth_period: int = 20,
        long_zero_lag_period: int = 42,
        long_realtime_window: int = 26,
        long_src_type: str = 'hlc3',
        long_slope_index: int = 1,
        long_range_threshold: float = 0.005,
        long_zero_lag_period_mode: str = 'fixed',
        long_realtime_window_mode: str = 'fixed',
        long_zl_cycle_detector_type: str = 'absolute_ultimate',
        long_zl_cycle_detector_cycle_part: float = 1.0,
        long_zl_cycle_detector_max_cycle: int = 240,
        long_zl_cycle_detector_min_cycle: int = 10,
        long_zl_cycle_period_multiplier: float = 1.0,
        long_rt_cycle_detector_type: str = 'phac_e',
        long_rt_cycle_detector_cycle_part: float = 1.0,
        long_rt_cycle_detector_max_cycle: int = 240,
        long_rt_cycle_detector_min_cycle: int = 10,
        long_rt_cycle_period_multiplier: float = 0.33,
        long_zl_cycle_detector_period_range: Tuple[int, int] = (10, 240),
        long_rt_cycle_detector_period_range: Tuple[int, int] = (10, 240),
        
        # 共通パラメータ
        enable_exit_signals: bool = True
    ):
        """初期化"""
        super().__init__("UltimateMADualSignalGenerator")
        
        # パラメータの設定
        self._params = {
            # 短期パラメータ
            'short_super_smooth_period': short_super_smooth_period,
            'short_zero_lag_period': short_zero_lag_period,
            'short_realtime_window': short_realtime_window,
            'short_src_type': short_src_type,
            'short_slope_index': short_slope_index,
            'short_range_threshold': short_range_threshold,
            'short_zero_lag_period_mode': short_zero_lag_period_mode,
            'short_realtime_window_mode': short_realtime_window_mode,
            'short_zl_cycle_detector_type': short_zl_cycle_detector_type,
            'short_zl_cycle_detector_cycle_part': short_zl_cycle_detector_cycle_part,
            'short_zl_cycle_detector_max_cycle': short_zl_cycle_detector_max_cycle,
            'short_zl_cycle_detector_min_cycle': short_zl_cycle_detector_min_cycle,
            'short_zl_cycle_period_multiplier': short_zl_cycle_period_multiplier,
            'short_rt_cycle_detector_type': short_rt_cycle_detector_type,
            'short_rt_cycle_detector_cycle_part': short_rt_cycle_detector_cycle_part,
            'short_rt_cycle_detector_max_cycle': short_rt_cycle_detector_max_cycle,
            'short_rt_cycle_detector_min_cycle': short_rt_cycle_detector_min_cycle,
            'short_rt_cycle_period_multiplier': short_rt_cycle_period_multiplier,
            'short_zl_cycle_detector_period_range': short_zl_cycle_detector_period_range,
            'short_rt_cycle_detector_period_range': short_rt_cycle_detector_period_range,
            
            # 長期パラメータ
            'long_super_smooth_period': long_super_smooth_period,
            'long_zero_lag_period': long_zero_lag_period,
            'long_realtime_window': long_realtime_window,
            'long_src_type': long_src_type,
            'long_slope_index': long_slope_index,
            'long_range_threshold': long_range_threshold,
            'long_zero_lag_period_mode': long_zero_lag_period_mode,
            'long_realtime_window_mode': long_realtime_window_mode,
            'long_zl_cycle_detector_type': long_zl_cycle_detector_type,
            'long_zl_cycle_detector_cycle_part': long_zl_cycle_detector_cycle_part,
            'long_zl_cycle_detector_max_cycle': long_zl_cycle_detector_max_cycle,
            'long_zl_cycle_detector_min_cycle': long_zl_cycle_detector_min_cycle,
            'long_zl_cycle_period_multiplier': long_zl_cycle_period_multiplier,
            'long_rt_cycle_detector_type': long_rt_cycle_detector_type,
            'long_rt_cycle_detector_cycle_part': long_rt_cycle_detector_cycle_part,
            'long_rt_cycle_detector_max_cycle': long_rt_cycle_detector_max_cycle,
            'long_rt_cycle_detector_min_cycle': long_rt_cycle_detector_min_cycle,
            'long_rt_cycle_period_multiplier': long_rt_cycle_period_multiplier,
            'long_zl_cycle_detector_period_range': long_zl_cycle_detector_period_range,
            'long_rt_cycle_detector_period_range': long_rt_cycle_detector_period_range,
            
            # 共通パラメータ
            'enable_exit_signals': enable_exit_signals
        }
        
        # Ultimate MAデュアルエントリーシグナルの初期化
        self.ultimate_ma_dual_signal = UltimateMADualEntrySignal(
            short_super_smooth_period=short_super_smooth_period,
            short_zero_lag_period=short_zero_lag_period,
            short_realtime_window=short_realtime_window,
            short_src_type=short_src_type,
            short_slope_index=short_slope_index,
            short_range_threshold=short_range_threshold,
            short_zero_lag_period_mode=short_zero_lag_period_mode,
            short_realtime_window_mode=short_realtime_window_mode,
            short_zl_cycle_detector_type=short_zl_cycle_detector_type,
            short_zl_cycle_detector_cycle_part=short_zl_cycle_detector_cycle_part,
            short_zl_cycle_detector_max_cycle=short_zl_cycle_detector_max_cycle,
            short_zl_cycle_detector_min_cycle=short_zl_cycle_detector_min_cycle,
            short_zl_cycle_period_multiplier=short_zl_cycle_period_multiplier,
            short_rt_cycle_detector_type=short_rt_cycle_detector_type,
            short_rt_cycle_detector_cycle_part=short_rt_cycle_detector_cycle_part,
            short_rt_cycle_detector_max_cycle=short_rt_cycle_detector_max_cycle,
            short_rt_cycle_detector_min_cycle=short_rt_cycle_detector_min_cycle,
            short_rt_cycle_period_multiplier=short_rt_cycle_period_multiplier,
            short_zl_cycle_detector_period_range=short_zl_cycle_detector_period_range,
            short_rt_cycle_detector_period_range=short_rt_cycle_detector_period_range,
            
            long_super_smooth_period=long_super_smooth_period,
            long_zero_lag_period=long_zero_lag_period,
            long_realtime_window=long_realtime_window,
            long_src_type=long_src_type,
            long_slope_index=long_slope_index,
            long_range_threshold=long_range_threshold,
            long_zero_lag_period_mode=long_zero_lag_period_mode,
            long_realtime_window_mode=long_realtime_window_mode,
            long_zl_cycle_detector_type=long_zl_cycle_detector_type,
            long_zl_cycle_detector_cycle_part=long_zl_cycle_detector_cycle_part,
            long_zl_cycle_detector_max_cycle=long_zl_cycle_detector_max_cycle,
            long_zl_cycle_detector_min_cycle=long_zl_cycle_detector_min_cycle,
            long_zl_cycle_period_multiplier=long_zl_cycle_period_multiplier,
            long_rt_cycle_detector_type=long_rt_cycle_detector_type,
            long_rt_cycle_detector_cycle_part=long_rt_cycle_detector_cycle_part,
            long_rt_cycle_detector_max_cycle=long_rt_cycle_detector_max_cycle,
            long_rt_cycle_detector_min_cycle=long_rt_cycle_detector_min_cycle,
            long_rt_cycle_period_multiplier=long_rt_cycle_period_multiplier,
            long_zl_cycle_detector_period_range=long_zl_cycle_detector_period_range,
            long_rt_cycle_detector_period_range=long_rt_cycle_detector_period_range,
            
            enable_exit_signals=enable_exit_signals
        )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._signals = None
        self._long_realtime_trends = None
        self._long_trend_signals = None
        self._short_result = None
        self._long_result = None
    
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
                
                # Ultimate MAデュアルエントリーシグナルの計算
                try:
                    # エントリーシグナルの計算
                    dual_entry_signals = self.ultimate_ma_dual_signal.generate(df)
                    
                    # Ultimate MAの結果を取得
                    self._short_result = self.ultimate_ma_dual_signal.get_short_ultimate_ma_result(df)
                    self._long_result = self.ultimate_ma_dual_signal.get_long_ultimate_ma_result(df)
                    
                    # 決済に必要な長期データを取得
                    self._long_realtime_trends = self._long_result.realtime_trends
                    self._long_trend_signals = self._long_result.trend_signals
                    
                    # エントリーシグナルを設定
                    self._signals = dual_entry_signals
                    
                except Exception as e:
                    self.logger.error(f"Ultimate MAデュアルエントリーシグナル計算中にエラー: {str(e)}")
                    # エラー時はゼロシグナルを設定
                    self._signals = np.zeros(current_len, dtype=np.int8)
                    self._long_realtime_trends = np.zeros(current_len, dtype=np.float64)
                    self._long_trend_signals = np.zeros(current_len, dtype=np.int8)
                
                self._data_len = current_len
        except Exception as e:
            self.logger.error(f"calculate_signals全体でエラー: {str(e)}")
            # エラー時はゼロシグナルを設定
            if data is not None:
                self._signals = np.zeros(len(data), dtype=np.int8)
                self._long_realtime_trends = np.zeros(len(data), dtype=np.float64)
                self._long_trend_signals = np.zeros(len(data), dtype=np.int8)
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
        if self._long_realtime_trends is not None and self._long_trend_signals is not None:
            return calculate_dual_exit_signals_numba(
                self._long_realtime_trends,
                self._long_trend_signals,
                position, 
                index
            )
        
        return False
    
    def get_short_ma_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        短期Ultimate MAの値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 短期Ultimate MAの値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            if self._short_result is not None:
                return self._short_result.values.copy()
            else:
                return np.array([])
        except Exception as e:
            self.logger.error(f"短期Ultimate MA値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_long_ma_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        長期Ultimate MAの値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 長期Ultimate MAの値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            if self._long_result is not None:
                return self._long_result.values.copy()
            else:
                return np.array([])
        except Exception as e:
            self.logger.error(f"長期Ultimate MA値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_short_realtime_trends(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        短期Ultimate MAのrealtime_trendsを取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 短期Ultimate MAのrealtime_trends
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            if self._short_result is not None:
                return self._short_result.realtime_trends.copy()
            else:
                return np.array([])
        except Exception as e:
            self.logger.error(f"短期realtime_trends取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_long_realtime_trends(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        長期Ultimate MAのrealtime_trendsを取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 長期Ultimate MAのrealtime_trends
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            if self._long_realtime_trends is not None:
                return self._long_realtime_trends.copy()
            else:
                return np.array([])
        except Exception as e:
            self.logger.error(f"長期realtime_trends取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_short_trend_signals(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        短期Ultimate MAのトレンドシグナルを取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 短期Ultimate MAのトレンドシグナル
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            if self._short_result is not None:
                return self._short_result.trend_signals.copy()
            else:
                return np.array([])
        except Exception as e:
            self.logger.error(f"短期トレンドシグナル取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_long_trend_signals(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        長期Ultimate MAのトレンドシグナルを取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 長期Ultimate MAのトレンドシグナル
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            if self._long_trend_signals is not None:
                return self._long_trend_signals.copy()
            else:
                return np.array([])
        except Exception as e:
            self.logger.error(f"長期トレンドシグナル取得中にエラー: {str(e)}")
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
            
            return self.ultimate_ma_dual_signal.get_noise_reduction_stats(data)
        except Exception as e:
            self.logger.error(f"ノイズ除去統計取得中にエラー: {str(e)}")
            return {}
    
    def get_all_ultimate_ma_stages(self, data: Union[pd.DataFrame, np.ndarray] = None) -> dict:
        """
        Ultimate MAの全段階の結果を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            dict: 全段階の結果（短期・長期それぞれの生値からフィルター適用後まで）
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            # 短期・長期それぞれの結果を取得
            results = {
                'short_result': self._short_result,
                'long_result': self._long_result
            }
            
            return results
        except Exception as e:
            self.logger.error(f"Ultimate MA全段階結果取得中にエラー: {str(e)}")
            return {} 