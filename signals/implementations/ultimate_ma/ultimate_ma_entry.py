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
def calculate_ultimate_ma_signals(realtime_trends: np.ndarray, trend_signals: np.ndarray) -> np.ndarray:
    """
    Ultimate MAのシグナルを計算する（高速化版）
    
    Args:
        realtime_trends: リアルタイムトレンド信号の配列
        trend_signals: トレンドシグナルの配列（1=上昇、-1=下降、0=range）
    
    Returns:
        シグナルの配列（1=ロング、-1=ショート、0=シグナルなし）
    """
    length = len(realtime_trends)
    signals = np.zeros(length, dtype=np.int8)
    
    # エントリーシグナル判定（並列処理化）
    for i in prange(length):
        # NaN値チェック
        if np.isnan(realtime_trends[i]) or np.isnan(trend_signals[i]):
            signals[i] = 0
            continue
            
        # ロングエントリー: realtime_trends > 0 かつ trend_signals == 1
        if realtime_trends[i] > 0.0 and trend_signals[i] == 1:
            signals[i] = 1
        # ショートエントリー: realtime_trends < 0 かつ trend_signals == -1
        elif realtime_trends[i] < 0.0 and trend_signals[i] == -1:
            signals[i] = -1
        else:
            signals[i] = 0
    
    return signals


@njit(fastmath=True, parallel=True)
def calculate_ultimate_ma_exit_signals(realtime_trends: np.ndarray, trend_signals: np.ndarray, 
                                      current_position: np.ndarray) -> np.ndarray:
    """
    Ultimate MAの決済シグナルを計算する（高速化版）
    
    Args:
        realtime_trends: リアルタイムトレンド信号の配列
        trend_signals: トレンドシグナルの配列（1=上昇、-1=下降、0=range）
        current_position: 現在のポジション（1=ロング、-1=ショート、0=ポジションなし）
    
    Returns:
        決済シグナルの配列（1=ロング決済、-1=ショート決済、0=決済なし）
    """
    length = len(realtime_trends)
    exit_signals = np.zeros(length, dtype=np.int8)
    
    # 決済シグナル判定（並列処理化）
    for i in prange(length):
        # NaN値チェック
        if np.isnan(realtime_trends[i]) or np.isnan(trend_signals[i]):
            continue
            
        # ロングポジション決済: realtime_trends <= 0 または trend_signals == -1
        if current_position[i] == 1:
            if realtime_trends[i] <= 0.0 or trend_signals[i] == -1:
                exit_signals[i] = 1  # ロング決済
        # ショートポジション決済: realtime_trends >= 0 または trend_signals == 1
        elif current_position[i] == -1:
            if realtime_trends[i] >= 0.0 or trend_signals[i] == 1:
                exit_signals[i] = -1  # ショート決済
    
    return exit_signals


class UltimateMAEntrySignal(BaseSignal, IEntrySignal):
    """
    Ultimate MAによるエントリーシグナル
    
    特徴:
    - 6段階革新的フィルタリングシステムを活用
    - リアルタイムトレンド検出と従来のトレンド判定の組み合わせ
    - ノイズが除去された高精度シグナル
    - 適応的カルマンフィルターによる動的調整
    - ヒルベルト変換による位相遅延ゼロ
    
    エントリー条件:
    - ロングエントリー: realtime_trends > 0 かつ trend_signals == 1
    - ショートエントリー: realtime_trends < 0 かつ trend_signals == -1
    
    決済条件:
    - ロング決済: realtime_trends <= 0 または trend_signals == -1
    - ショート決済: realtime_trends >= 0 または trend_signals == 1
    """
    
    def __init__(
        self,
        super_smooth_period: int = 10,
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
        """
        コンストラクタ
        
        Args:
            super_smooth_period: スーパースムーザーフィルター期間（デフォルト: 10）
            zero_lag_period: ゼロラグEMA期間（デフォルト: 21）
            realtime_window: リアルタイムトレンド検出ウィンドウ（デフォルト: 13）
            src_type: 価格ソース（'close', 'hlc3', 'hl2', 'ohlc4'など）
            slope_index: トレンド判定期間（デフォルト: 1）
            range_threshold: range判定の基本閾値（デフォルト: 0.005 = 0.5%）
            enable_exit_signals: 決済シグナルを有効にするか（デフォルト: True）
            
            # 動的適応パラメータ
            zero_lag_period_mode: ゼロラグ期間モード ('fixed' or 'dynamic')
            realtime_window_mode: リアルタイムウィンドウモード ('fixed' or 'dynamic')
            
            # ゼロラグ用サイクル検出器パラメータ
            zl_cycle_detector_type: ゼロラグ用サイクル検出器タイプ
            zl_cycle_detector_cycle_part: ゼロラグ用サイクル検出器のサイクル部分倍率
            zl_cycle_detector_max_cycle: ゼロラグ用サイクル検出器の最大サイクル期間
            zl_cycle_detector_min_cycle: ゼロラグ用サイクル検出器の最小サイクル期間
            zl_cycle_period_multiplier: ゼロラグ用サイクル期間の乗数
            
            # リアルタイムウィンドウ用サイクル検出器パラメータ
            rt_cycle_detector_type: リアルタイム用サイクル検出器タイプ
            rt_cycle_detector_cycle_part: リアルタイム用サイクル検出器のサイクル部分倍率
            rt_cycle_detector_max_cycle: リアルタイム用サイクル検出器の最大サイクル期間
            rt_cycle_detector_min_cycle: リアルタイム用サイクル検出器の最小サイクル期間
            rt_cycle_period_multiplier: リアルタイム用サイクル期間の乗数
            
            # period_rangeパラメータ
            zl_cycle_detector_period_range: ゼロラグ用サイクル検出器の周期範囲
            rt_cycle_detector_period_range: リアルタイム用サイクル検出器の周期範囲
        """
        params = {
            'super_smooth_period': super_smooth_period,
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
        super().__init__(
            f"UltimateMAEntry(ss={super_smooth_period}, zl={zero_lag_period}, rt={realtime_window}, slope={slope_index})",
            params
        )
        
        # Ultimate MAのインスタンス化
        self._ultimate_ma = UltimateMA(
            super_smooth_period=super_smooth_period,
            zero_lag_period=zero_lag_period,
            realtime_window=realtime_window,
            src_type=src_type,
            slope_index=slope_index,
            range_threshold=range_threshold,
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
        
        # 結果キャッシュ
        self._entry_signals = None
        self._exit_signals = None
        self._data_hash = None
        self._current_position = None
    
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
        エントリーシグナルを生成する
        
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
            
            # Ultimate MAの計算
            result = self._ultimate_ma.calculate(data)
            
            # 計算が失敗した場合はゼロシグナルを返す
            if result is None:
                self._entry_signals = np.zeros(len(data), dtype=np.int8)
                return self._entry_signals
            
            # realtime_trendsとtrend_signalsの取得
            realtime_trends = result.realtime_trends
            trend_signals = result.trend_signals
            
            # エントリーシグナルの計算（高速化版）
            entry_signals = calculate_ultimate_ma_signals(
                realtime_trends,
                trend_signals
            )
            
            # 結果をキャッシュ
            self._entry_signals = entry_signals
            
            # 決済シグナルも計算（有効な場合）
            if self._params['enable_exit_signals']:
                # 簡易的なポジション追跡
                current_position = self._track_position(entry_signals)
                self._exit_signals = calculate_ultimate_ma_exit_signals(
                    realtime_trends,
                    trend_signals,
                    current_position
                )
                self._current_position = current_position
            
            return entry_signals
            
        except Exception as e:
            # エラーが発生した場合は警告を出力し、ゼロシグナルを返す
            print(f"UltimateMAEntrySignal計算中にエラー: {str(e)}")
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
    
    def get_ultimate_ma_result(self, data: Union[pd.DataFrame, np.ndarray] = None) -> object:
        """
        Ultimate MAの計算結果を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            UltimateMAResult: Ultimate MAの計算結果
        """
        if data is not None:
            self.generate(data)
            
        return self._ultimate_ma.calculate(data)
    
    def get_realtime_trends(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        リアルタイムトレンド信号を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: リアルタイムトレンド信号の配列
        """
        if data is not None:
            self.generate(data)
        
        result = self._ultimate_ma.get_realtime_trends()
        return result if result is not None else np.array([])
    
    def get_trend_signals(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        トレンドシグナルを取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: トレンドシグナルの配列（1=上昇、-1=下降、0=range）
        """
        if data is not None:
            self.generate(data)
        
        result = self._ultimate_ma.get_trend_signals()
        return result if result is not None else np.array([])
    
    def get_ultimate_ma_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Ultimate MAの最終フィルター済み値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: Ultimate MAの最終フィルター済み値
        """
        if data is not None:
            self.generate(data)
        
        result = self._ultimate_ma.get_values()
        return result if result is not None else np.array([])
    
    def get_noise_reduction_stats(self, data: Union[pd.DataFrame, np.ndarray] = None) -> dict:
        """
        ノイズ除去統計を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            dict: ノイズ除去統計
        """
        if data is not None:
            self.generate(data)
        
        return self._ultimate_ma.get_noise_reduction_stats()
        
    def reset(self) -> None:
        """
        シグナルの状態をリセットする
        """
        super().reset()
        self._ultimate_ma.reset() if hasattr(self._ultimate_ma, 'reset') else None
        self._entry_signals = None
        self._exit_signals = None
        self._current_position = None
        self._data_hash = None 