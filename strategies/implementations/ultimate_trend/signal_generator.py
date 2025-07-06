#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.ultimate_trend.ultimate_trend_entry import UltimateTrendEntrySignal


@njit(fastmath=True, parallel=True)
def calculate_exit_signals_numba(trend: np.ndarray, trend_signals: np.ndarray, 
                                current_position: int, current_index: int, use_filter: bool) -> bool:
    """
    決済シグナル計算（Numba高速化版）
    
    Args:
        trend: アルティメットトレンド方向の配列（1=上昇、-1=下降、0=なし）
        trend_signals: Ultimate MAのトレンドシグナル配列（1=上昇、-1=下降、0=range）
        current_position: 現在のポジション（1=ロング、-1=ショート）
        current_index: 現在のインデックス
        use_filter: フィルターオプション（True=trend_signalsでフィルタリング）
    
    Returns:
        bool: 決済すべきかどうか
    """
    if current_index >= len(trend):
        return False
    
    # NaN値チェック
    if np.isnan(trend[current_index]):
        return False
    
    if use_filter and trend_signals is not None:
        if current_index >= len(trend_signals) or np.isnan(trend_signals[current_index]):
            return False
        
        # フィルターオプション有効時
        # ロングポジション決済: trend != 1 または trend_signals != 1
        if current_position == 1:
            return trend[current_index] != 1 or trend_signals[current_index] != 1
        # ショートポジション決済: trend != -1 または trend_signals != -1
        elif current_position == -1:
            return trend[current_index] != -1 or trend_signals[current_index] != -1
    else:
        # ベース決済（フィルターなし）
        # ロングポジション決済: trend != 1
        if current_position == 1:
            return trend[current_index] != 1
        # ショートポジション決済: trend != -1
        elif current_position == -1:
            return trend[current_index] != -1
    
    return False


class UltimateTrendSignalGenerator(BaseSignalGenerator):
    """
    Ultimate Trendのシグナル生成クラス（両方向・高速化版）
    
    エントリー条件:
    - ベース: trend == 1 でロング、trend == -1 でショート
    - フィルター有効時: 上記 + Ultimate MAのtrend_signalsが同方向
    
    エグジット条件:
    - ベース: trendが反転または0になった時
    - フィルター有効時: 上記 + Ultimate MAのtrend_signalsが反転
    """
    
    def __init__(
        self,
        # Ultimate Trendパラメータ
        length: int = 13,
        multiplier: float = 3.0,
        ultimate_smoother_period: int = 10,
        zero_lag_period: int = 21,
        filtering_mode: int = 1,
        # シグナル生成パラメータ
        use_filter: bool = False,
        enable_exit_signals: bool = True,
        # Ultimate MAの動的適応パラメータ
        zero_lag_period_mode: str = 'dynamic',        # Ultimate MAのゼロラグ用サイクル検出器パラメータ
        zl_cycle_detector_type: str = 'absolute_ultimate',
        zl_cycle_detector_cycle_part: float = 1.0,
        zl_cycle_detector_max_cycle: int = 120,
        zl_cycle_detector_min_cycle: int = 5,
        zl_cycle_period_multiplier: float = 1.0,
        zl_cycle_detector_period_range: Tuple[int, int] = (5, 120)
    ):
        """初期化"""
        super().__init__("UltimateTrendSignalGenerator")
        
        # パラメータの設定
        self._params = {
            'length': length,
            'multiplier': multiplier,
            'ultimate_smoother_period': ultimate_smoother_period,
            'zero_lag_period': zero_lag_period,
            'filtering_mode': filtering_mode,
            'use_filter': use_filter,
            'enable_exit_signals': enable_exit_signals,
            'zero_lag_period_mode': zero_lag_period_mode,
            'zl_cycle_detector_type': zl_cycle_detector_type,
            'zl_cycle_detector_cycle_part': zl_cycle_detector_cycle_part,
            'zl_cycle_detector_max_cycle': zl_cycle_detector_max_cycle,
            'zl_cycle_detector_min_cycle': zl_cycle_detector_min_cycle,
            'zl_cycle_period_multiplier': zl_cycle_period_multiplier,
            'zl_cycle_detector_period_range': zl_cycle_detector_period_range
        }
        
        # Ultimate Trendエントリーシグナルの初期化
        self.ultimate_trend_signal = UltimateTrendEntrySignal(
            length=length,
            multiplier=multiplier,
            ultimate_smoother_period=ultimate_smoother_period,
            zero_lag_period=zero_lag_period,
            filtering_mode=filtering_mode,
            use_filter=use_filter,
            enable_exit_signals=enable_exit_signals,
            zero_lag_period_mode=zero_lag_period_mode,
            zl_cycle_detector_type=zl_cycle_detector_type,
            zl_cycle_detector_cycle_part=zl_cycle_detector_cycle_part,
            zl_cycle_detector_max_cycle=zl_cycle_detector_max_cycle,
            zl_cycle_detector_min_cycle=zl_cycle_detector_min_cycle,
            zl_cycle_period_multiplier=zl_cycle_period_multiplier,
            zl_cycle_detector_period_range=zl_cycle_detector_period_range
        )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._signals = None
        self._trend = None
        self._trend_signals = None
        self._ultimate_trend_result = None
    
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
                
                # Ultimate Trendシグナルの計算
                try:
                    # エントリーシグナルの計算
                    ultimate_trend_signals = self.ultimate_trend_signal.generate(df)
                    
                    # Ultimate Trendの結果を取得
                    self._ultimate_trend_result = self.ultimate_trend_signal.get_ultimate_trend_result(df)
                    
                    # トレンド方向とトレンドシグナルを取得
                    self._trend = self._ultimate_trend_result.trend
                    self._trend_signals = self._ultimate_trend_result.trend_signals
                    
                    # エントリーシグナルを設定
                    self._signals = ultimate_trend_signals
                    
                except Exception as e:
                    self.logger.error(f"Ultimate Trendシグナル計算中にエラー: {str(e)}")
                    # エラー時はゼロシグナルを設定
                    self._signals = np.zeros(current_len, dtype=np.int8)
                    self._trend = np.zeros(current_len, dtype=np.int8)
                    self._trend_signals = np.zeros(current_len, dtype=np.int8)
                
                self._data_len = current_len
        except Exception as e:
            self.logger.error(f"calculate_signals全体でエラー: {str(e)}")
            # エラー時はゼロシグナルを設定
            if data is not None:
                self._signals = np.zeros(len(data), dtype=np.int8)
                self._trend = np.zeros(len(data), dtype=np.int8)
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
        if self._trend is not None:
            return calculate_exit_signals_numba(
                self._trend, 
                self._trend_signals if self._params['use_filter'] else None,
                position, 
                index,
                self._params['use_filter']
            )
        
        return False
    
    def get_ultimate_trend_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Ultimate Trendラインの値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: Ultimate Trendラインの値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            if self._ultimate_trend_result is not None:
                return self._ultimate_trend_result.values.copy()
            else:
                return np.array([])
        except Exception as e:
            self.logger.error(f"Ultimate Trend値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_trend(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        アルティメットトレンド方向を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: トレンド方向の配列（1=上昇、-1=下降、0=なし）
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            if self._trend is not None:
                return self._trend.copy()
            else:
                return np.array([])
        except Exception as e:
            self.logger.error(f"トレンド方向取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_trend_signals(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Ultimate MAのトレンドシグナルを取得
        
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
    
    def get_upper_band(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        上側バンドを取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 上側バンドの配列
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            if self._ultimate_trend_result is not None:
                return self._ultimate_trend_result.upper_band.copy()
            else:
                return np.array([])
        except Exception as e:
            self.logger.error(f"上側バンド取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_lower_band(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        下側バンドを取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 下側バンドの配列
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            if self._ultimate_trend_result is not None:
                return self._ultimate_trend_result.lower_band.copy()
            else:
                return np.array([])
        except Exception as e:
            self.logger.error(f"下側バンド取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_final_upper_band(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        調整済み上側バンドを取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 調整済み上側バンドの配列
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            if self._ultimate_trend_result is not None:
                return self._ultimate_trend_result.final_upper_band.copy()
            else:
                return np.array([])
        except Exception as e:
            self.logger.error(f"調整済み上側バンド取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_final_lower_band(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        調整済み下側バンドを取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 調整済み下側バンドの配列
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            if self._ultimate_trend_result is not None:
                return self._ultimate_trend_result.final_lower_band.copy()
            else:
                return np.array([])
        except Exception as e:
            self.logger.error(f"調整済み下側バンド取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_filtered_midline(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Ultimate MAフィルタ済みミッドラインを取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: フィルタ済みミッドラインの配列
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            if self._ultimate_trend_result is not None:
                return self._ultimate_trend_result.filtered_midline.copy()
            else:
                return np.array([])
        except Exception as e:
            self.logger.error(f"フィルタ済みミッドライン取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_ukf_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        カルマンフィルター後の値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: カルマンフィルター後の値の配列
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            if self._ultimate_trend_result is not None:
                return self._ultimate_trend_result.ukf_values.copy()
            else:
                return np.array([])
        except Exception as e:
            self.logger.error(f"カルマンフィルター値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_filtering_stats(self, data: Union[pd.DataFrame, np.ndarray] = None) -> dict:
        """
        フィルタリング統計を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            dict: フィルタリング統計
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            return self.ultimate_trend_signal.get_filtering_stats(data)
        except Exception as e:
            self.logger.error(f"フィルタリング統計取得中にエラー: {str(e)}")
            return {}
    
    def get_all_ultimate_trend_stages(self, data: Union[pd.DataFrame, np.ndarray] = None) -> dict:
        """
        Ultimate Trendの全段階の結果を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            dict: 全段階の結果
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            if self._ultimate_trend_result is not None:
                return {
                    'values': self._ultimate_trend_result.values.copy(),
                    'upper_band': self._ultimate_trend_result.upper_band.copy(),
                    'lower_band': self._ultimate_trend_result.lower_band.copy(),
                    'final_upper_band': self._ultimate_trend_result.final_upper_band.copy(),
                    'final_lower_band': self._ultimate_trend_result.final_lower_band.copy(),
                    'trend': self._ultimate_trend_result.trend.copy(),
                    'atr_values': self._ultimate_trend_result.atr_values.copy(),
                    'filtered_midline': self._ultimate_trend_result.filtered_midline.copy(),
                    'raw_midline': self._ultimate_trend_result.raw_midline.copy(),
                    'ukf_values': self._ultimate_trend_result.ukf_values.copy(),
                    'ultimate_smooth_values': self._ultimate_trend_result.ultimate_smooth_values.copy(),
                    'zero_lag_values': self._ultimate_trend_result.zero_lag_values.copy(),
                    'amplitude': self._ultimate_trend_result.amplitude.copy(),
                    'phase': self._ultimate_trend_result.phase.copy(),
                    'filtering_mode': self._ultimate_trend_result.filtering_mode,
                    'trend_signals': self._ultimate_trend_result.trend_signals.copy(),
                    'current_trend': self._ultimate_trend_result.current_trend,
                    'current_trend_value': self._ultimate_trend_result.current_trend_value
                }
            else:
                return {}
        except Exception as e:
            self.logger.error(f"Ultimate Trend全段階結果取得中にエラー: {str(e)}")
            return {} 