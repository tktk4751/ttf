#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, Tuple
import numpy as np
import pandas as pd
from numba import jit, njit, prange

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.ultimate_ma import UltimateMA


@njit(fastmath=True, parallel=True)
def calculate_ultimate_ma_crossover_signals(short_ma: np.ndarray, long_ma: np.ndarray, 
                                          long_realtime_trends: np.ndarray = None, 
                                          use_filter: bool = False) -> np.ndarray:
    """
    Ultimate MAのクロスオーバーシグナルを計算する（高速化版）
    
    Args:
        short_ma: 短期Ultimate MAの配列
        long_ma: 長期Ultimate MAの配列
        long_realtime_trends: 長期Ultimate MAのrealtime_trendsの配列
        use_filter: フィルターオプション（True=realtime_trendsでフィルタリング）
    
    Returns:
        シグナルの配列（1=ロング、-1=ショート、0=シグナルなし）
    """
    length = len(short_ma)
    signals = np.zeros(length, dtype=np.int8)
    
    if length < 2:
        return signals
    
    # クロスオーバー検出（並列処理化）
    for i in prange(1, length):
        # NaN値チェック
        if (np.isnan(short_ma[i]) or np.isnan(long_ma[i]) or 
            np.isnan(short_ma[i-1]) or np.isnan(long_ma[i-1])):
            signals[i] = 0
            continue
        
        # 前期と当期の関係
        prev_short = short_ma[i-1]
        prev_long = long_ma[i-1]
        curr_short = short_ma[i]
        curr_long = long_ma[i]
        
        # ゴールデンクロス（短期が長期を上抜け）
        golden_cross = (prev_short <= prev_long) and (curr_short > curr_long)
        
        # デッドクロス（短期が長期を下抜け）
        dead_cross = (prev_short >= prev_long) and (curr_short < curr_long)
        
        if use_filter and long_realtime_trends is not None:
            # フィルターオプション有効時
            if i < len(long_realtime_trends) and not np.isnan(long_realtime_trends[i]):
                # ゴールデンクロス + realtime_trends > 0
                if golden_cross and long_realtime_trends[i] > 0.0:
                    signals[i] = 1
                # デッドクロス + realtime_trends < 0
                elif dead_cross and long_realtime_trends[i] < 0.0:
                    signals[i] = -1
                else:
                    signals[i] = 0
            else:
                signals[i] = 0
        else:
            # ベースシグナル（フィルターなし）
            if golden_cross:
                signals[i] = 1
            elif dead_cross:
                signals[i] = -1
            else:
                signals[i] = 0
    
    return signals


class UltimateMAXoverEntrySignal(BaseSignal, IEntrySignal):
    """
    Ultimate MAクロスオーバーによるエントリーシグナル
    
    特徴:
    - 短期Ultimate MAと長期Ultimate MAのクロスオーバー検出
    - ゴールデンクロス（短期 > 長期）でロングエントリー
    - デッドクロス（短期 < 長期）でショートエントリー
    - オプションで長期MAのrealtime_trendsによるフィルタリング
    - Numbaによる高速化処理
    
    エントリー条件:
    - ベース: ゴールデンクロスでロング、デッドクロスでショート
    - フィルター有効時: 上記 + 長期MAのrealtime_trendsが同方向
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
        """
        コンストラクタ
        
        Args:
            # 短期Ultimate MAパラメータ
            short_super_smooth_period: 短期スーパースムーザー期間（デフォルト: 5）
            short_zero_lag_period: 短期ゼロラグEMA期間（デフォルト: 10）
            short_realtime_window: 短期リアルタイムウィンドウ（デフォルト: 13）
            
            # 長期Ultimate MAパラメータ
            long_super_smooth_period: 長期スーパースムーザー期間（デフォルト: 10）
            long_zero_lag_period: 長期ゼロラグEMA期間（デフォルト: 21）
            long_realtime_window: 長期リアルタイムウィンドウ（デフォルト: 34）
            
            # 共通パラメータ
            src_type: 価格ソース（'close', 'hlc3', 'hl2', 'ohlc4'など）
            slope_index: トレンド判定期間（デフォルト: 1）
            range_threshold: range判定の基本閾値（デフォルト: 0.005 = 0.5%）
            
            # シグナル生成パラメータ
            use_filter: フィルターオプション（True=長期MAのrealtime_trendsでフィルタリング）
            
            # Ultimate MAの動的適応パラメータ
            zero_lag_period_mode: ゼロラグEMA期間モード（'dynamic' or 'fixed'）
            realtime_window_mode: リアルタイムウィンドウモード（'dynamic' or 'fixed'）
            
            # 短期Ultimate MAのサイクル検出器パラメータ
            short_zl_cycle_detector_type: 短期ゼロラグ用サイクル検出器タイプ
            short_zl_cycle_detector_cycle_part: 短期ゼロラグ用サイクル部分
            short_zl_cycle_detector_max_cycle: 短期ゼロラグ用最大サイクル
            short_zl_cycle_detector_min_cycle: 短期ゼロラグ用最小サイクル
            short_zl_cycle_period_multiplier: 短期ゼロラグ用サイクル期間乗数
            short_rt_cycle_detector_type: 短期リアルタイム用サイクル検出器タイプ
            short_rt_cycle_detector_cycle_part: 短期リアルタイム用サイクル部分
            short_rt_cycle_detector_max_cycle: 短期リアルタイム用最大サイクル
            short_rt_cycle_detector_min_cycle: 短期リアルタイム用最小サイクル
            short_rt_cycle_period_multiplier: 短期リアルタイム用サイクル期間乗数
            short_zl_cycle_detector_period_range: 短期ゼロラグ用period_rangeパラメータ
            short_rt_cycle_detector_period_range: 短期リアルタイム用period_rangeパラメータ
            
            # 長期Ultimate MAのサイクル検出器パラメータ
            long_zl_cycle_detector_type: 長期ゼロラグ用サイクル検出器タイプ
            long_zl_cycle_detector_cycle_part: 長期ゼロラグ用サイクル部分
            long_zl_cycle_detector_max_cycle: 長期ゼロラグ用最大サイクル
            long_zl_cycle_detector_min_cycle: 長期ゼロラグ用最小サイクル
            long_zl_cycle_period_multiplier: 長期ゼロラグ用サイクル期間乗数
            long_rt_cycle_detector_type: 長期リアルタイム用サイクル検出器タイプ
            long_rt_cycle_detector_cycle_part: 長期リアルタイム用サイクル部分
            long_rt_cycle_detector_max_cycle: 長期リアルタイム用最大サイクル
            long_rt_cycle_detector_min_cycle: 長期リアルタイム用最小サイクル
            long_rt_cycle_period_multiplier: 長期リアルタイム用サイクル期間乗数
            long_zl_cycle_detector_period_range: 長期ゼロラグ用period_rangeパラメータ
            long_rt_cycle_detector_period_range: 長期リアルタイム用period_rangeパラメータ
        """
        params = {
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
        
        filter_desc = "Filtered" if use_filter else "Base"
        super().__init__(
            f"UltimateMAXover(Short:{short_zero_lag_period}, Long:{long_zero_lag_period}, {filter_desc})",
            params
        )
        
        # Ultimate MAインジケーターの初期化
        self._short_ultimate_ma = UltimateMA(
            super_smooth_period=short_super_smooth_period,
            zero_lag_period=short_zero_lag_period,
            realtime_window=short_realtime_window,
            src_type=src_type,
            slope_index=slope_index,
            range_threshold=range_threshold,
            zero_lag_period_mode=zero_lag_period_mode,
            realtime_window_mode=realtime_window_mode,
            zl_cycle_detector_type=short_zl_cycle_detector_type,
            zl_cycle_detector_cycle_part=short_zl_cycle_detector_cycle_part,
            zl_cycle_detector_max_cycle=short_zl_cycle_detector_max_cycle,
            zl_cycle_detector_min_cycle=short_zl_cycle_detector_min_cycle,
            zl_cycle_period_multiplier=short_zl_cycle_period_multiplier,
            rt_cycle_detector_type=short_rt_cycle_detector_type,
            rt_cycle_detector_cycle_part=short_rt_cycle_detector_cycle_part,
            rt_cycle_detector_max_cycle=short_rt_cycle_detector_max_cycle,
            rt_cycle_detector_min_cycle=short_rt_cycle_detector_min_cycle,
            rt_cycle_period_multiplier=short_rt_cycle_period_multiplier,
            zl_cycle_detector_period_range=short_zl_cycle_detector_period_range,
            rt_cycle_detector_period_range=short_rt_cycle_detector_period_range
        )
        
        self._long_ultimate_ma = UltimateMA(
            super_smooth_period=long_super_smooth_period,
            zero_lag_period=long_zero_lag_period,
            realtime_window=long_realtime_window,
            src_type=src_type,
            slope_index=slope_index,
            range_threshold=range_threshold,
            zero_lag_period_mode=zero_lag_period_mode,
            realtime_window_mode=realtime_window_mode,
            zl_cycle_detector_type=long_zl_cycle_detector_type,
            zl_cycle_detector_cycle_part=long_zl_cycle_detector_cycle_part,
            zl_cycle_detector_max_cycle=long_zl_cycle_detector_max_cycle,
            zl_cycle_detector_min_cycle=long_zl_cycle_detector_min_cycle,
            zl_cycle_period_multiplier=long_zl_cycle_period_multiplier,
            rt_cycle_detector_type=long_rt_cycle_detector_type,
            rt_cycle_detector_cycle_part=long_rt_cycle_detector_cycle_part,
            rt_cycle_detector_max_cycle=long_rt_cycle_detector_max_cycle,
            rt_cycle_detector_min_cycle=long_rt_cycle_detector_min_cycle,
            rt_cycle_period_multiplier=long_rt_cycle_period_multiplier,
            zl_cycle_detector_period_range=long_zl_cycle_detector_period_range,
            rt_cycle_detector_period_range=long_rt_cycle_detector_period_range
        )
        
        # 結果キャッシュ
        self._signals = None
        self._data_hash = None
        self._short_result = None
        self._long_result = None
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算してキャッシュに使用する"""
        if isinstance(data, pd.DataFrame):
            # DataFrameの場合は必要なカラム（OHLC）のハッシュ
            required_cols = ['open', 'high', 'low', 'close']
            available_cols = [col for col in required_cols if col in data.columns]
            if available_cols:
                data_hash = hash(tuple(map(tuple, data[available_cols].values)))
            else:
                # フォールバック
                data_hash = hash(tuple(map(tuple, data.values)))
        else:
            # NumPy配列の場合
            if data.ndim == 2 and data.shape[1] >= 4:
                # OHLCデータの場合
                data_hash = hash(tuple(map(tuple, data)))
            else:
                # それ以外は全体をハッシュ
                data_hash = hash(tuple(map(tuple, data)) if data.ndim == 2 else tuple(data))
        
        return f"{data_hash}_{hash(frozenset(self._params.items()))}"
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナルを生成する
        
        Args:
            data: 価格データ（OHLC必須）
        
        Returns:
            シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
        """
        try:
            # キャッシュチェック - 同じデータの場合は計算をスキップ
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._signals is not None:
                return self._signals
                
            self._data_hash = data_hash
            
            # 短期・長期Ultimate MAの計算
            self._short_result = self._short_ultimate_ma.calculate(data)
            self._long_result = self._long_ultimate_ma.calculate(data)
            
            # 計算が失敗した場合はゼロシグナルを返す
            if self._short_result is None or self._long_result is None:
                self._signals = np.zeros(len(data), dtype=np.int8)
                return self._signals
            
            # Ultimate MAの値を取得
            short_ma = self._short_result.values
            long_ma = self._long_result.values
            long_realtime_trends = self._long_result.realtime_trends if self._params['use_filter'] else None
            
            # クロスオーバーシグナルの計算（高速化版）
            crossover_signals = calculate_ultimate_ma_crossover_signals(
                short_ma,
                long_ma,
                long_realtime_trends,
                self._params['use_filter']
            )
            
            # 結果をキャッシュ
            self._signals = crossover_signals
            
            return crossover_signals
            
        except Exception as e:
            # エラーが発生した場合は警告を出力し、ゼロシグナルを返す
            print(f"UltimateMAXoverEntrySignal計算中にエラー: {str(e)}")
            import traceback
            print(traceback.format_exc())
            self._signals = np.zeros(len(data), dtype=np.int8)
            return self._signals
    
    def get_short_ultimate_ma_result(self, data: Union[pd.DataFrame, np.ndarray] = None) -> object:
        """
        短期Ultimate MAの計算結果を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            UltimateMAResult: 短期Ultimate MAの計算結果
        """
        if data is not None:
            self.generate(data)
            
        return self._short_result
    
    def get_long_ultimate_ma_result(self, data: Union[pd.DataFrame, np.ndarray] = None) -> object:
        """
        長期Ultimate MAの計算結果を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            UltimateMAResult: 長期Ultimate MAの計算結果
        """
        if data is not None:
            self.generate(data)
            
        return self._long_result
    
    def get_short_ma_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        短期Ultimate MAの値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 短期Ultimate MAの値
        """
        if data is not None:
            self.generate(data)
        
        if self._short_result is not None:
            return self._short_result.values.copy()
        else:
            return np.array([])
    
    def get_long_ma_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        長期Ultimate MAの値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 長期Ultimate MAの値
        """
        if data is not None:
            self.generate(data)
        
        if self._long_result is not None:
            return self._long_result.values.copy()
        else:
            return np.array([])
    
    def get_long_realtime_trends(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        長期Ultimate MAのrealtime_trendsを取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 長期Ultimate MAのrealtime_trends
        """
        if data is not None:
            self.generate(data)
        
        if self._long_result is not None:
            return self._long_result.realtime_trends.copy()
        else:
            return np.array([])
    
    def get_short_trend_signals(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        短期Ultimate MAのトレンドシグナルを取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 短期Ultimate MAのトレンドシグナル
        """
        if data is not None:
            self.generate(data)
        
        if self._short_result is not None:
            return self._short_result.trend_signals.copy()
        else:
            return np.array([])
    
    def get_long_trend_signals(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        長期Ultimate MAのトレンドシグナルを取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 長期Ultimate MAのトレンドシグナル
        """
        if data is not None:
            self.generate(data)
        
        if self._long_result is not None:
            return self._long_result.trend_signals.copy()
        else:
            return np.array([])
    
    def get_noise_reduction_stats(self, data: Union[pd.DataFrame, np.ndarray] = None) -> dict:
        """
        ノイズ除去統計を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            dict: ノイズ除去統計（短期・長期両方）
        """
        if data is not None:
            self.generate(data)
        
        stats = {}
        
        if self._short_result is not None:
            stats['short_ma'] = self._short_ultimate_ma.get_noise_reduction_stats()
        
        if self._long_result is not None:
            stats['long_ma'] = self._long_ultimate_ma.get_noise_reduction_stats()
        
        return stats
    
    def get_all_ultimate_ma_stages(self, data: Union[pd.DataFrame, np.ndarray] = None) -> dict:
        """
        Ultimate MAの全段階の結果を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            dict: 全段階の結果（短期・長期両方）
        """
        if data is not None:
            self.generate(data)
        
        stages = {}
        
        if self._short_result is not None:
            stages['short_ma'] = {
                'values': self._short_result.values.copy(),
                'raw_values': self._short_result.raw_values.copy(),
                'kalman_values': self._short_result.kalman_values.copy(),
                'super_smooth_values': self._short_result.super_smooth_values.copy(),
                'zero_lag_values': self._short_result.zero_lag_values.copy(),
                'amplitude': self._short_result.amplitude.copy(),
                'phase': self._short_result.phase.copy(),
                'realtime_trends': self._short_result.realtime_trends.copy(),
                'trend_signals': self._short_result.trend_signals.copy(),
                'current_trend': self._short_result.current_trend,
                'current_trend_value': self._short_result.current_trend_value
            }
        
        if self._long_result is not None:
            stages['long_ma'] = {
                'values': self._long_result.values.copy(),
                'raw_values': self._long_result.raw_values.copy(),
                'kalman_values': self._long_result.kalman_values.copy(),
                'super_smooth_values': self._long_result.super_smooth_values.copy(),
                'zero_lag_values': self._long_result.zero_lag_values.copy(),
                'amplitude': self._long_result.amplitude.copy(),
                'phase': self._long_result.phase.copy(),
                'realtime_trends': self._long_result.realtime_trends.copy(),
                'trend_signals': self._long_result.trend_signals.copy(),
                'current_trend': self._long_result.current_trend,
                'current_trend_value': self._long_result.current_trend_value
            }
        
        return stages
        
    def reset(self) -> None:
        """
        シグナルの状態をリセットする
        """
        super().reset()
        self._short_ultimate_ma.reset() if hasattr(self._short_ultimate_ma, 'reset') else None
        self._long_ultimate_ma.reset() if hasattr(self._long_ultimate_ma, 'reset') else None
        self._signals = None
        self._short_result = None
        self._long_result = None
        self._data_hash = None 