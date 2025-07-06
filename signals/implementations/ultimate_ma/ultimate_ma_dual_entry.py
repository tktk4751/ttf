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
def calculate_dual_ultimate_ma_signals(
    short_realtime_trends: np.ndarray, 
    short_trend_signals: np.ndarray,
    long_trend_signals: np.ndarray
) -> np.ndarray:
    """
    デュアルUltimate MAのエントリーシグナルを計算する（高速化版）
    
    Args:
        short_realtime_trends: 短期リアルタイムトレンド信号の配列
        short_trend_signals: 短期トレンドシグナルの配列（1=上昇、-1=下降、0=range）
        long_trend_signals: 長期トレンドシグナルの配列（1=上昇、-1=下降、0=range）
    
    Returns:
        シグナルの配列（1=ロング、-1=ショート、0=シグナルなし）
    """
    length = len(short_realtime_trends)
    signals = np.zeros(length, dtype=np.int8)
    
    # エントリーシグナル判定（並列処理化）
    for i in prange(length):
        # NaN値チェック
        if (np.isnan(short_realtime_trends[i]) or 
            np.isnan(short_trend_signals[i]) or 
            np.isnan(long_trend_signals[i])):
            signals[i] = 0
            continue
            
        # ロングエントリー: 短期 realtime_trends > 0 かつ 短期 trend_signals == 1 かつ 長期 trend_signals == 1
        if (short_realtime_trends[i] > 0.0 and 
            short_trend_signals[i] == 1 and 
            long_trend_signals[i] == 1):
            signals[i] = 1
        # ショートエントリー: 短期 realtime_trends < 0 かつ 短期 trend_signals == -1 かつ 長期 trend_signals == -1
        elif (short_realtime_trends[i] < 0.0 and 
              short_trend_signals[i] == -1 and 
              long_trend_signals[i] == -1):
            signals[i] = -1
        else:
            signals[i] = 0
    
    return signals


@njit(fastmath=True, parallel=True)
def calculate_dual_ultimate_ma_exit_signals(
    long_realtime_trends: np.ndarray, 
    long_trend_signals: np.ndarray, 
    current_position: np.ndarray
) -> np.ndarray:
    """
    デュアルUltimate MAの決済シグナルを計算する（高速化版）
    
    Args:
        long_realtime_trends: 長期リアルタイムトレンド信号の配列
        long_trend_signals: 長期トレンドシグナルの配列（1=上昇、-1=下降、0=range）
        current_position: 現在のポジション（1=ロング、-1=ショート、0=ポジションなし）
    
    Returns:
        決済シグナルの配列（1=ロング決済、-1=ショート決済、0=決済なし）
    """
    length = len(long_realtime_trends)
    exit_signals = np.zeros(length, dtype=np.int8)
    
    # 決済シグナル判定（並列処理化）
    for i in prange(length):
        # NaN値チェック
        if np.isnan(long_realtime_trends[i]) or np.isnan(long_trend_signals[i]):
            continue
            
        # ロングポジション決済: 長期 realtime_trends <= 0 または 長期 trend_signals == -1
        if current_position[i] == 1:
            if long_realtime_trends[i] <= 0.0 or long_trend_signals[i] == -1:
                exit_signals[i] = 1  # ロング決済
        # ショートポジション決済: 長期 realtime_trends >= 0 または 長期 trend_signals == 1
        elif current_position[i] == -1:
            if long_realtime_trends[i] >= 0.0 or long_trend_signals[i] == 1:
                exit_signals[i] = -1  # ショート決済
    
    return exit_signals


class UltimateMADualEntrySignal(BaseSignal, IEntrySignal):
    """
    デュアルUltimate MAによるエントリーシグナル
    
    特徴:
    - 短期と長期の2本のUltimate MAを使用
    - 短期線でエントリータイミングを判定、長期線でトレンド方向を確認
    - 長期線で決済タイミングを判定
    - それぞれ独立したサイクル検出器とパラメータ
    
    エントリー条件:
    - ロングエントリー: 短期 realtime_trends > 0 かつ 短期 trend_signals == 1 かつ 長期 trend_signals == 1
    - ショートエントリー: 短期 realtime_trends < 0 かつ 短期 trend_signals == -1 かつ 長期 trend_signals == -1
    
    決済条件:
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
        """
        コンストラクタ
        
        Args:
            # 短期Ultimate MAパラメータ
            short_super_smooth_period: 短期スーパースムーザーフィルター期間
            short_zero_lag_period: 短期ゼロラグEMA期間
            short_realtime_window: 短期リアルタイムトレンド検出ウィンドウ
            short_src_type: 短期価格ソース
            short_slope_index: 短期トレンド判定期間
            short_range_threshold: 短期range判定の基本閾値
            short_zero_lag_period_mode: 短期ゼロラグ期間モード
            short_realtime_window_mode: 短期リアルタイムウィンドウモード
            short_zl_cycle_detector_type: 短期ゼロラグ用サイクル検出器タイプ
            short_zl_cycle_detector_cycle_part: 短期ゼロラグ用サイクル検出器のサイクル部分倍率
            short_zl_cycle_detector_max_cycle: 短期ゼロラグ用サイクル検出器の最大サイクル期間
            short_zl_cycle_detector_min_cycle: 短期ゼロラグ用サイクル検出器の最小サイクル期間
            short_zl_cycle_period_multiplier: 短期ゼロラグ用サイクル期間の乗数
            short_rt_cycle_detector_type: 短期リアルタイム用サイクル検出器タイプ
            short_rt_cycle_detector_cycle_part: 短期リアルタイム用サイクル検出器のサイクル部分倍率
            short_rt_cycle_detector_max_cycle: 短期リアルタイム用サイクル検出器の最大サイクル期間
            short_rt_cycle_detector_min_cycle: 短期リアルタイム用サイクル検出器の最小サイクル期間
            short_rt_cycle_period_multiplier: 短期リアルタイム用サイクル期間の乗数
            short_zl_cycle_detector_period_range: 短期ゼロラグ用サイクル検出器の周期範囲
            short_rt_cycle_detector_period_range: 短期リアルタイム用サイクル検出器の周期範囲
            
            # 長期Ultimate MAパラメータ（同様の説明で長期版）
            long_super_smooth_period: 長期スーパースムーザーフィルター期間
            long_zero_lag_period: 長期ゼロラグEMA期間
            long_realtime_window: 長期リアルタイムトレンド検出ウィンドウ
            long_src_type: 長期価格ソース
            long_slope_index: 長期トレンド判定期間
            long_range_threshold: 長期range判定の基本閾値
            long_zero_lag_period_mode: 長期ゼロラグ期間モード
            long_realtime_window_mode: 長期リアルタイムウィンドウモード
            long_zl_cycle_detector_type: 長期ゼロラグ用サイクル検出器タイプ
            long_zl_cycle_detector_cycle_part: 長期ゼロラグ用サイクル検出器のサイクル部分倍率
            long_zl_cycle_detector_max_cycle: 長期ゼロラグ用サイクル検出器の最大サイクル期間
            long_zl_cycle_detector_min_cycle: 長期ゼロラグ用サイクル検出器の最小サイクル期間
            long_zl_cycle_period_multiplier: 長期ゼロラグ用サイクル期間の乗数
            long_rt_cycle_detector_type: 長期リアルタイム用サイクル検出器タイプ
            long_rt_cycle_detector_cycle_part: 長期リアルタイム用サイクル検出器のサイクル部分倍率
            long_rt_cycle_detector_max_cycle: 長期リアルタイム用サイクル検出器の最大サイクル期間
            long_rt_cycle_detector_min_cycle: 長期リアルタイム用サイクル検出器の最小サイクル期間
            long_rt_cycle_period_multiplier: 長期リアルタイム用サイクル期間の乗数
            long_zl_cycle_detector_period_range: 長期ゼロラグ用サイクル検出器の周期範囲
            long_rt_cycle_detector_period_range: 長期リアルタイム用サイクル検出器の周期範囲
            
            # 共通パラメータ
            enable_exit_signals: 決済シグナルを有効にするか
        """
        params = {
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
        
        super().__init__(
            f"UltimateMADualEntry(short_ss={short_super_smooth_period}, short_zl={short_zero_lag_period}, short_rt={short_realtime_window}, long_ss={long_super_smooth_period}, long_zl={long_zero_lag_period}, long_rt={long_realtime_window})",
            params
        )
        
        # 短期Ultimate MAのインスタンス化
        self._short_ultimate_ma = UltimateMA(
            super_smooth_period=short_super_smooth_period,
            zero_lag_period=short_zero_lag_period,
            realtime_window=short_realtime_window,
            src_type=short_src_type,
            slope_index=short_slope_index,
            range_threshold=short_range_threshold,
            zero_lag_period_mode=short_zero_lag_period_mode,
            realtime_window_mode=short_realtime_window_mode,
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
        
        # 長期Ultimate MAのインスタンス化
        self._long_ultimate_ma = UltimateMA(
            super_smooth_period=long_super_smooth_period,
            zero_lag_period=long_zero_lag_period,
            realtime_window=long_realtime_window,
            src_type=long_src_type,
            slope_index=long_slope_index,
            range_threshold=long_range_threshold,
            zero_lag_period_mode=long_zero_lag_period_mode,
            realtime_window_mode=long_realtime_window_mode,
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
        self._entry_signals = None
        self._exit_signals = None
        self._data_hash = None
        self._current_position = None
        self._short_result = None
        self._long_result = None
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算してキャッシュに使用する"""
        if isinstance(data, pd.DataFrame):
            # DataFrameの場合は必要なカラムのみハッシュする
            if 'close' in data.columns:
                data_hash = hash(tuple(data['close'].values))
            else:
                # closeカラムがない場合は全カラムのハッシュ
                data_hash = hash(tuple(map(tuple, data.values)))
        else:
            # NumPy配列の場合
            if data.ndim == 2 and data.shape[1] >= 4:
                # OHLCデータの場合はcloseだけハッシュ
                data_hash = hash(tuple(data[:, 3]))
            else:
                # それ以外は全体をハッシュ
                data_hash = hash(tuple(map(tuple, data)) if data.ndim == 2 else tuple(data))
        
        return f"{data_hash}_{hash(frozenset(self._params.items()))}"
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        デュアルUltimate MAによるエントリーシグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
        """
        try:
            # キャッシュチェック - 同じデータの場合は計算をスキップ
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._entry_signals is not None:
                return self._entry_signals
                
            self._data_hash = data_hash
            
            # 短期Ultimate MAの計算
            short_result = self._short_ultimate_ma.calculate(data)
            if short_result is None:
                self._entry_signals = np.zeros(len(data), dtype=np.int8)
                return self._entry_signals
            
            # 長期Ultimate MAの計算
            long_result = self._long_ultimate_ma.calculate(data)
            if long_result is None:
                self._entry_signals = np.zeros(len(data), dtype=np.int8)
                return self._entry_signals
            
            # 結果をキャッシュ
            self._short_result = short_result
            self._long_result = long_result
            
            # 必要なシグナルの取得
            short_realtime_trends = short_result.realtime_trends
            short_trend_signals = short_result.trend_signals
            long_trend_signals = long_result.trend_signals
            
            # エントリーシグナルの計算（高速化版）
            entry_signals = calculate_dual_ultimate_ma_signals(
                short_realtime_trends,
                short_trend_signals,
                long_trend_signals
            )
            
            # 結果をキャッシュ
            self._entry_signals = entry_signals
            
            # 決済シグナルも計算（有効な場合）
            if self._params['enable_exit_signals']:
                # 簡易的なポジション追跡
                current_position = self._track_position(entry_signals)
                self._exit_signals = calculate_dual_ultimate_ma_exit_signals(
                    long_result.realtime_trends,
                    long_result.trend_signals,
                    current_position
                )
                self._current_position = current_position
            
            return entry_signals
            
        except Exception as e:
            # エラーが発生した場合は警告を出力し、ゼロシグナルを返す
            print(f"UltimateMADualEntrySignal計算中にエラー: {str(e)}")
            import traceback
            print(traceback.format_exc())
            self._entry_signals = np.zeros(len(data), dtype=np.int8)
            return self._entry_signals
    
    def _track_position(self, signals: np.ndarray) -> np.ndarray:
        """
        シグナルからポジションを追跡する
        
        Args:
            signals: エントリーシグナル（1=ロング、-1=ショート、0=なし）
        
        Returns:
            ポジション配列（1=ロング、-1=ショート、0=ポジションなし）
        """
        length = len(signals)
        position = np.zeros(length, dtype=np.int8)
        current_pos = 0
        
        for i in range(length):
            if signals[i] == 1:  # ロングエントリー
                current_pos = 1
            elif signals[i] == -1:  # ショートエントリー
                current_pos = -1
            # signals[i] == 0の場合は現在のポジションを維持
            
            position[i] = current_pos
        
        return position
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        決済シグナルを取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 決済シグナルの配列（1=ロング決済、-1=ショート決済、0=決済なし）
        """
        if data is not None:
            self.generate(data)
        
        if self._exit_signals is not None:
            return self._exit_signals.copy()
        else:
            return np.zeros(len(data) if data is not None else 0, dtype=np.int8)
    
    def get_current_position(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        現在のポジション状態を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: ポジション配列（1=ロング、-1=ショート、0=ポジションなし）
        """
        if data is not None:
            self.generate(data)
        
        if self._current_position is not None:
            return self._current_position.copy()
        else:
            return np.zeros(len(data) if data is not None else 0, dtype=np.int8)
    
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
    
    def get_short_realtime_trends(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        短期リアルタイムトレンド信号を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 短期リアルタイムトレンド信号の配列
        """
        if data is not None:
            self.generate(data)
        
        if self._short_result is not None:
            return self._short_result.realtime_trends.copy()
        else:
            return np.array([])
    
    def get_short_trend_signals(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        短期トレンドシグナルを取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 短期トレンドシグナルの配列（1=上昇、-1=下降、0=range）
        """
        if data is not None:
            self.generate(data)
        
        if self._short_result is not None:
            return self._short_result.trend_signals.copy()
        else:
            return np.array([])
    
    def get_long_realtime_trends(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        長期リアルタイムトレンド信号を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 長期リアルタイムトレンド信号の配列
        """
        if data is not None:
            self.generate(data)
        
        if self._long_result is not None:
            return self._long_result.realtime_trends.copy()
        else:
            return np.array([])
    
    def get_long_trend_signals(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        長期トレンドシグナルを取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 長期トレンドシグナルの配列（1=上昇、-1=下降、0=range）
        """
        if data is not None:
            self.generate(data)
        
        if self._long_result is not None:
            return self._long_result.trend_signals.copy()
        else:
            return np.array([])
    
    def get_short_ultimate_ma_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        短期Ultimate MAの最終フィルター済み値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 短期Ultimate MAの最終フィルター済み値
        """
        if data is not None:
            self.generate(data)
        
        if self._short_result is not None:
            return self._short_result.values.copy()
        else:
            return np.array([])
    
    def get_long_ultimate_ma_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        長期Ultimate MAの最終フィルター済み値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 長期Ultimate MAの最終フィルター済み値
        """
        if data is not None:
            self.generate(data)
        
        if self._long_result is not None:
            return self._long_result.values.copy()
        else:
            return np.array([])
    
    def get_noise_reduction_stats(self, data: Union[pd.DataFrame, np.ndarray] = None) -> dict:
        """
        短期・長期のノイズ除去統計を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            dict: 短期・長期のノイズ除去統計
        """
        if data is not None:
            self.generate(data)
        
        short_stats = self._short_ultimate_ma.get_noise_reduction_stats()
        long_stats = self._long_ultimate_ma.get_noise_reduction_stats()
        
        return {
            'short_ma_stats': short_stats,
            'long_ma_stats': long_stats
        }
        
    def reset(self) -> None:
        """
        シグナルの状態をリセットする
        """
        super().reset()
        self._short_ultimate_ma.reset() if hasattr(self._short_ultimate_ma, 'reset') else None
        self._long_ultimate_ma.reset() if hasattr(self._long_ultimate_ma, 'reset') else None
        self._entry_signals = None
        self._exit_signals = None
        self._current_position = None
        self._data_hash = None
        self._short_result = None
        self._long_result = None 