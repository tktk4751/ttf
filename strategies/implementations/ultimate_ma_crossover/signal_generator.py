#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.ultimate_ma.ultimate_ma_crossover_entry import UltimateMAXoverEntrySignal


@njit(fastmath=True, parallel=True)
def calculate_exit_signals_numba(short_ma: np.ndarray, long_ma: np.ndarray, 
                                long_realtime_trends: np.ndarray, 
                                current_position: int, current_index: int, use_filter: bool) -> bool:
    """
    決済シグナル計算（Numba高速化版）
    
    Args:
        short_ma: 短期Ultimate MAの配列
        long_ma: 長期Ultimate MAの配列
        long_realtime_trends: 長期Ultimate MAのrealtime_trendsの配列
        current_position: 現在のポジション（1=ロング、-1=ショート）
        current_index: 現在のインデックス
        use_filter: フィルターオプション（True=realtime_trendsでフィルタリング）
    
    Returns:
        bool: 決済すべきかどうか
    """
    if current_index >= len(short_ma) or current_index >= len(long_ma):
        return False
    
    # NaN値チェック
    if (np.isnan(short_ma[current_index]) or np.isnan(long_ma[current_index])):
        return False
    
    # 現在の関係
    curr_short = short_ma[current_index]
    curr_long = long_ma[current_index]
    
    if use_filter and long_realtime_trends is not None:
        if current_index >= len(long_realtime_trends) or np.isnan(long_realtime_trends[current_index]):
            return False
        
        # フィルターオプション有効時
        # ロングポジション決済: 短期 <= 長期 または realtime_trends <= 0
        if current_position == 1:
            return curr_short <= curr_long or long_realtime_trends[current_index] <= 0.0
        # ショートポジション決済: 短期 >= 長期 または realtime_trends >= 0
        elif current_position == -1:
            return curr_short >= curr_long or long_realtime_trends[current_index] >= 0.0
    else:
        # ベース決済（フィルターなし）
        # ロングポジション決済: 短期 <= 長期
        if current_position == 1:
            return curr_short <= curr_long
        # ショートポジション決済: 短期 >= 長期
        elif current_position == -1:
            return curr_short >= curr_long
    
    return False


class UltimateMAXoverSignalGenerator(BaseSignalGenerator):
    """
    Ultimate MAクロスオーバーのシグナル生成クラス（両方向・高速化版）
    
    エントリー条件:
    - ベース: ゴールデンクロスでロング、デッドクロスでショート
    - フィルター有効時: 上記 + 長期MAのrealtime_trendsが同方向
    
    エグジット条件:
    - ベース: クロス関係が逆転した時
    - フィルター有効時: 上記 + 長期MAのrealtime_trendsが反転
    """
    
    def __init__(
        self,
        # 短期Ultimate MAパラメータ
        short_super_smooth_period: int = 5,
        short_zero_lag_period: int = 10,
        short_realtime_window: int = 13,
        # 長期Ultimate MAパラメータ
        long_super_smooth_period: int = 10,
        long_zero_lag_period: int = 21,
        long_realtime_window: int = 34,
        # 共通パラメータ
        src_type: str = 'hlc3',
        slope_index: int = 1,
        range_threshold: float = 0.005,
        # シグナル生成パラメータ
        use_filter: bool = False,
        enable_exit_signals: bool = True,
        # Ultimate MAの動的適応パラメータ
        zero_lag_period_mode: str = 'dynamic',
        realtime_window_mode: str = 'dynamic',
        # 短期Ultimate MAのゼロラグ用サイクル検出器パラメータ
        short_zl_cycle_detector_type: str = 'absolute_ultimate',
        short_zl_cycle_detector_cycle_part: float = 1.0,
        short_zl_cycle_detector_max_cycle: int = 120,
        short_zl_cycle_detector_min_cycle: int = 5,
        short_zl_cycle_period_multiplier: float = 1.0,
        # 短期リアルタイムウィンドウ用サイクル検出器パラメータ
        short_rt_cycle_detector_type: str = 'absolute_ultimate',
        short_rt_cycle_detector_cycle_part: float = 1.0,
        short_rt_cycle_detector_max_cycle: int = 120,
        short_rt_cycle_detector_min_cycle: int = 5,
        short_rt_cycle_period_multiplier: float = 0.5,
        # 短期period_rangeパラメータ
        short_zl_cycle_detector_period_range: Tuple[int, int] = (5, 120),
        short_rt_cycle_detector_period_range: Tuple[int, int] = (5, 120),
        # 長期Ultimate MAのゼロラグ用サイクル検出器パラメータ
        long_zl_cycle_detector_type: str = 'absolute_ultimate',
        long_zl_cycle_detector_cycle_part: float = 1.0,
        long_zl_cycle_detector_max_cycle: int = 120,
        long_zl_cycle_detector_min_cycle: int = 5,
        long_zl_cycle_period_multiplier: float = 1.0,
        # 長期リアルタイムウィンドウ用サイクル検出器パラメータ
        long_rt_cycle_detector_type: str = 'absolute_ultimate',
        long_rt_cycle_detector_cycle_part: float = 1.0,
        long_rt_cycle_detector_max_cycle: int = 120,
        long_rt_cycle_detector_min_cycle: int = 5,
        long_rt_cycle_period_multiplier: float = 0.5,
        # 長期period_rangeパラメータ
        long_zl_cycle_detector_period_range: Tuple[int, int] = (5, 120),
        long_rt_cycle_detector_period_range: Tuple[int, int] = (5, 120)
    ):
        """初期化"""
        super().__init__("UltimateMAXoverSignalGenerator")
        
        # パラメータの設定
        self._params = {
            'short_super_smooth_period': short_super_smooth_period,
            'short_zero_lag_period': short_zero_lag_period,
            'short_realtime_window': short_realtime_window,
            'long_super_smooth_period': long_super_smooth_period,
            'long_zero_lag_period': long_zero_lag_period,
            'long_realtime_window': long_realtime_window,
            'src_type': src_type,
            'slope_index': slope_index,
            'range_threshold': range_threshold,
            'use_filter': use_filter,
            'enable_exit_signals': enable_exit_signals,
            'zero_lag_period_mode': zero_lag_period_mode,
            'realtime_window_mode': realtime_window_mode,
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
            'long_rt_cycle_detector_period_range': long_rt_cycle_detector_period_range
        }
        
        # Ultimate MAクロスオーバーエントリーシグナルの初期化
        self.ultimate_ma_xover_signal = UltimateMAXoverEntrySignal(
            short_super_smooth_period=short_super_smooth_period,
            short_zero_lag_period=short_zero_lag_period,
            short_realtime_window=short_realtime_window,
            long_super_smooth_period=long_super_smooth_period,
            long_zero_lag_period=long_zero_lag_period,
            long_realtime_window=long_realtime_window,
            src_type=src_type,
            slope_index=slope_index,
            range_threshold=range_threshold,
            use_filter=use_filter,
            zero_lag_period_mode=zero_lag_period_mode,
            realtime_window_mode=realtime_window_mode,
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
            long_rt_cycle_detector_period_range=long_rt_cycle_detector_period_range
        )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._signals = None
        self._short_ma = None
        self._long_ma = None
        self._long_realtime_trends = None
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
                
                # Ultimate MAクロスオーバーシグナルの計算
                try:
                    # エントリーシグナルの計算
                    ultimate_ma_xover_signals = self.ultimate_ma_xover_signal.generate(df)
                    
                    # Ultimate MAの結果を取得
                    self._short_result = self.ultimate_ma_xover_signal.get_short_ultimate_ma_result(df)
                    self._long_result = self.ultimate_ma_xover_signal.get_long_ultimate_ma_result(df)
                    
                    # MA値とrealtime_trendsを取得
                    self._short_ma = self._short_result.values
                    self._long_ma = self._long_result.values
                    self._long_realtime_trends = self._long_result.realtime_trends
                    
                    # エントリーシグナルを設定
                    self._signals = ultimate_ma_xover_signals
                    
                except Exception as e:
                    self.logger.error(f"Ultimate MAクロスオーバーシグナル計算中にエラー: {str(e)}")
                    # エラー時はゼロシグナルを設定
                    self._signals = np.zeros(current_len, dtype=np.int8)
                    self._short_ma = np.zeros(current_len, dtype=np.float64)
                    self._long_ma = np.zeros(current_len, dtype=np.float64)
                    self._long_realtime_trends = np.zeros(current_len, dtype=np.float64)
                
                self._data_len = current_len
        except Exception as e:
            self.logger.error(f"calculate_signals全体でエラー: {str(e)}")
            # エラー時はゼロシグナルを設定
            if data is not None:
                self._signals = np.zeros(len(data), dtype=np.int8)
                self._short_ma = np.zeros(len(data), dtype=np.float64)
                self._long_ma = np.zeros(len(data), dtype=np.float64)
                self._long_realtime_trends = np.zeros(len(data), dtype=np.float64)
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
        if self._short_ma is not None and self._long_ma is not None:
            return calculate_exit_signals_numba(
                self._short_ma, 
                self._long_ma,
                self._long_realtime_trends if self._params['use_filter'] else None,
                position, 
                index,
                self._params['use_filter']
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
            
            if self._short_ma is not None:
                return self._short_ma.copy()
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
            
            if self._long_ma is not None:
                return self._long_ma.copy()
            else:
                return np.array([])
        except Exception as e:
            self.logger.error(f"長期Ultimate MA値取得中にエラー: {str(e)}")
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
            
            if self._long_result is not None:
                return self._long_result.trend_signals.copy()
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
            
            return self.ultimate_ma_xover_signal.get_noise_reduction_stats(data)
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
            
            return self.ultimate_ma_xover_signal.get_all_ultimate_ma_stages(data)
        except Exception as e:
            self.logger.error(f"Ultimate MA全段階結果取得中にエラー: {str(e)}")
            return {} 