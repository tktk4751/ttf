#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple, Optional
import numpy as np
import pandas as pd
from numba import njit, prange
from enum import Enum

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.x_mama.entry import XMAMACrossoverEntrySignal
from signals.implementations.phasor_trend_filter.filter import PhasorTrendFilterSignal
from signals.implementations.correlation_cycle.filter import CorrelationCycleFilterSignal
from signals.implementations.correlation_trend.filter import CorrelationTrendFilterSignal
from signals.implementations.unified_trend_cycle.filter import UnifiedTrendCycleFilterSignal
from signals.implementations.x_choppiness.filter import XChoppinessFilterSignal


class FilterType(Enum):
    """フィルタータイプ列挙"""
    NONE = "none"
    PHASOR_TREND = "phasor_trend"
    CORRELATION_CYCLE = "correlation_cycle"
    CORRELATION_TREND = "correlation_trend"
    UNIFIED_TREND_CYCLE = "unified_trend_cycle"
    X_CHOPPINESS = "x_choppiness"


@njit(fastmath=True, parallel=False)
def combine_signals_numba(x_mama_signals: np.ndarray, filter_signals: np.ndarray, filter_type: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    X_MAMAシグナルとフィルターシグナルを統合する（Numba最適化版）
    
    Args:
        x_mama_signals: X_MAMAシグナル配列
        filter_signals: フィルターシグナル配列
        filter_type: フィルタータイプ（0=None, 1=Phasor, 2=CorrelationCycle, 3=CorrelationTrend, 4=UnifiedTrendCycle, 5=XChoppiness）
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (ロングシグナル, ショートシグナル)
    """
    length = len(x_mama_signals)
    long_signals = np.zeros(length, dtype=np.int8)
    short_signals = np.zeros(length, dtype=np.int8)
    
    for i in prange(length):
        x_mama_signal = x_mama_signals[i]
        
        # フィルターなしの場合
        if filter_type == 0:
            if x_mama_signal == 1:
                long_signals[i] = 1
            elif x_mama_signal == -1:
                short_signals[i] = 1
        else:
            # フィルターありの場合
            if i < len(filter_signals):
                filter_signal = filter_signals[i]
                
                # ロングエントリー条件: X_MAMAシグナル=1 かつ フィルターシグナル=1
                if x_mama_signal == 1 and filter_signal == 1:
                    long_signals[i] = 1
                
                # ショートエントリー条件: X_MAMAシグナル=-1 かつ フィルターシグナル=-1
                elif x_mama_signal == -1 and filter_signal == -1:
                    short_signals[i] = 1
    
    return long_signals, short_signals


@njit(fastmath=True, parallel=False)
def check_exit_conditions_numba(x_mama_signals: np.ndarray, position: int, index: int) -> bool:
    """
    エグジット条件をチェックする（Numba最適化版）
    
    Args:
        x_mama_signals: X_MAMAシグナル配列
        position: 現在のポジション（1=ロング、-1=ショート）
        index: チェックするインデックス
    
    Returns:
        bool: エグジットすべきかどうか
    """
    if index < 0 or index >= len(x_mama_signals):
        return False
    
    x_mama_signal = x_mama_signals[index]
    
    # ロングポジション: X_MAMAシグナル=-1でエグジット
    if position == 1 and x_mama_signal == -1:
        return True
    
    # ショートポジション: X_MAMAシグナル=1でエグジット
    if position == -1 and x_mama_signal == 1:
        return True
    
    return False


class XMAMAEnhancedSignalGenerator(BaseSignalGenerator):
    """
    X_MAMA Enhanced シグナル生成クラス
    
    特徴:
    - X_MAMAシグナルをベースとした高度なエントリー・エグジット制御
    - 5つのフィルターから選択可能:
      1. Phasor Trend Filter - フェーザー分析による高精度トレンド判定
      2. Correlation Cycle Filter - 相関サイクル分析によるトレンド・サイクル判定
      3. Correlation Trend Filter - 相関トレンド分析による線形トレンド検出
      4. Unified Trend Cycle Filter - 3つのEhlersアルゴリズム統合による超高精度判定
      5. X-Choppiness Filter - STR基盤改良型チョピネス指標によるトレンド・レンジ判定
    - フィルターなしオプションも提供
    
    エントリー条件:
    - ロング: X_MAMAシグナル=1 かつ フィルターシグナル=1（フィルター有効時）
    - ショート: X_MAMAシグナル=-1 かつ フィルターシグナル=-1（フィルター有効時）
    - フィルターシグナル=0の場合はスルー
    
    エグジット条件:
    - ロング: X_MAMAシグナル=-1
    - ショート: X_MAMAシグナル=1
    """
    
    def __init__(
        self,
        # X_MAMAパラメータ
        fast_limit: float = 0.5,               # 高速制限値
        slow_limit: float = 0.05,              # 低速制限値
        src_type: str = 'hlc3',                # ソースタイプ
        # カルマンフィルターパラメータ（X_MAMA用）
        use_kalman_filter: bool = False,       # カルマンフィルターを使用するか
        kalman_filter_type: str = 'unscented', # カルマンフィルタータイプ
        kalman_process_noise: float = 0.01,    # プロセスノイズ
        kalman_observation_noise: float = 0.001, # 観測ノイズ
        # ゼロラグ処理パラメータ（X_MAMA用）
        use_zero_lag: bool = True,             # ゼロラグ処理を使用するか
        # シグナル設定（X_MAMA用）
        position_mode: bool = False,           # 位置関係シグナル(True)またはクロスオーバーシグナル(False)
        # フィルター選択
        filter_type: FilterType = FilterType.NONE,  # フィルタータイプ
        # Phasor Trend Filterパラメータ
        phasor_period: int = 20,               # フェーザー分析周期
        phasor_trend_threshold: float = 6.0,   # フェーザートレンド判定閾値
        phasor_use_kalman: bool = False,       # フェーザーフィルター用カルマンフィルター
        # Correlation Cycle Filterパラメータ
        correlation_cycle_period: int = 20,    # 相関サイクル計算期間
        correlation_cycle_threshold: float = 9.0, # サイクルトレンド判定閾値
        # Correlation Trend Filterパラメータ
        correlation_trend_length: int = 20,    # 相関トレンド計算期間
        correlation_trend_threshold: float = 0.3, # トレンド判定閾値
        correlation_trend_smoothing: bool = False, # 平滑化有効
        # Unified Trend Cycle Filterパラメータ
        unified_period: int = 20,              # 統合フィルター基本周期
        unified_trend_length: int = 20,        # 統合トレンド分析長
        unified_trend_threshold: float = 0.5,  # 統合トレンド判定閾値
        unified_adaptability: float = 0.7,     # 適応性係数
        unified_consensus: bool = True,         # コンセンサスフィルター有効
        # X-Choppiness Filterパラメータ
        x_choppiness_detector_type: str = 'hody_e',  # サイクル検出器タイプ
        x_choppiness_lp_period: int = 12,      # ローパスフィルター期間
        x_choppiness_hp_period: int = 124,     # ハイパスフィルター期間
        x_choppiness_cycle_part: float = 0.5,  # サイクル部分
        x_choppiness_max_cycle: int = 124,     # 最大サイクル期間
        x_choppiness_min_cycle: int = 12,      # 最小サイクル期間
        x_choppiness_max_output: int = 89,     # 最大出力値
        x_choppiness_min_output: int = 5       # 最小出力値
    ):
        """
        初期化
        
        Args:
            fast_limit: X_MAMA高速制限値（デフォルト: 0.5）
            slow_limit: X_MAMA低速制限値（デフォルト: 0.05）
            src_type: ソースタイプ（デフォルト: 'hlc3'）
            use_kalman_filter: X_MAMA用カルマンフィルター使用（デフォルト: False）
            kalman_filter_type: カルマンフィルタータイプ（デフォルト: 'unscented'）
            kalman_process_noise: プロセスノイズ（デフォルト: 0.01）
            kalman_observation_noise: 観測ノイズ（デフォルト: 0.001）
            use_zero_lag: ゼロラグ処理使用（デフォルト: True）
            position_mode: 位置関係シグナル(True)またはクロスオーバーシグナル(False)
            filter_type: フィルタータイプ（デフォルト: FilterType.NONE）
            その他: 各フィルターのパラメータ
            x_choppiness_*: X-Choppinessフィルターのパラメータ（サイクル検出器関連のみ）
        """
        filter_name = filter_type.value if isinstance(filter_type, FilterType) else str(filter_type)
        signal_type = "Position" if position_mode else "Crossover"
        kalman_str = f"_kalman({kalman_filter_type})" if use_kalman_filter else ""
        zero_lag_str = "_zero_lag" if use_zero_lag else ""
        
        super().__init__(f"X_MAMA_Enhanced_{signal_type}_{filter_name}{kalman_str}{zero_lag_str}")
        
        # パラメータの設定
        self._params = {
            # X_MAMAパラメータ
            'fast_limit': fast_limit,
            'slow_limit': slow_limit,
            'src_type': src_type,
            'use_kalman_filter': use_kalman_filter,
            'kalman_filter_type': kalman_filter_type,
            'kalman_process_noise': kalman_process_noise,
            'kalman_observation_noise': kalman_observation_noise,
            'use_zero_lag': use_zero_lag,
            'position_mode': position_mode,
            # フィルター設定
            'filter_type': filter_type,
            # Phasor Trend Filterパラメータ
            'phasor_period': phasor_period,
            'phasor_trend_threshold': phasor_trend_threshold,
            'phasor_use_kalman': phasor_use_kalman,
            # Correlation Cycle Filterパラメータ
            'correlation_cycle_period': correlation_cycle_period,
            'correlation_cycle_threshold': correlation_cycle_threshold,
            # Correlation Trend Filterパラメータ
            'correlation_trend_length': correlation_trend_length,
            'correlation_trend_threshold': correlation_trend_threshold,
            'correlation_trend_smoothing': correlation_trend_smoothing,
            # Unified Trend Cycle Filterパラメータ
            'unified_period': unified_period,
            'unified_trend_length': unified_trend_length,
            'unified_trend_threshold': unified_trend_threshold,
            'unified_adaptability': unified_adaptability,
            'unified_consensus': unified_consensus,
            # X-Choppiness Filterパラメータ
            'x_choppiness_detector_type': x_choppiness_detector_type,
            'x_choppiness_lp_period': x_choppiness_lp_period,
            'x_choppiness_hp_period': x_choppiness_hp_period,
            'x_choppiness_cycle_part': x_choppiness_cycle_part,
            'x_choppiness_max_cycle': x_choppiness_max_cycle,
            'x_choppiness_min_cycle': x_choppiness_min_cycle,
            'x_choppiness_max_output': x_choppiness_max_output,
            'x_choppiness_min_output': x_choppiness_min_output
        }
        
        self.filter_type = filter_type if isinstance(filter_type, FilterType) else FilterType(filter_type)
        self.position_mode = position_mode
        
        # X_MAMAエントリーシグナルの初期化
        self.x_mama_entry_signal = XMAMACrossoverEntrySignal(
            fast_limit=fast_limit,
            slow_limit=slow_limit,
            src_type=src_type,
            use_kalman_filter=use_kalman_filter,
            kalman_filter_type=kalman_filter_type,
            kalman_process_noise=kalman_process_noise,
            kalman_observation_noise=kalman_observation_noise,
            use_zero_lag=use_zero_lag,
            position_mode=position_mode
        )
        
        # フィルターシグナルの初期化
        self.filter_signal = None
        if self.filter_type == FilterType.PHASOR_TREND:
            self.filter_signal = PhasorTrendFilterSignal(
                period=phasor_period,
                trend_threshold=phasor_trend_threshold,
                src_type=src_type,
                use_kalman_filter=phasor_use_kalman,
                kalman_filter_type=kalman_filter_type,
                kalman_process_noise=kalman_process_noise,
                kalman_observation_noise=kalman_observation_noise
            )
        elif self.filter_type == FilterType.CORRELATION_CYCLE:
            self.filter_signal = CorrelationCycleFilterSignal(
                period=correlation_cycle_period,
                src_type=src_type,
                trend_threshold=correlation_cycle_threshold
            )
        elif self.filter_type == FilterType.CORRELATION_TREND:
            self.filter_signal = CorrelationTrendFilterSignal(
                length=correlation_trend_length,
                src_type=src_type,
                trend_threshold=correlation_trend_threshold,
                enable_smoothing=correlation_trend_smoothing
            )
        elif self.filter_type == FilterType.UNIFIED_TREND_CYCLE:
            self.filter_signal = UnifiedTrendCycleFilterSignal(
                period=unified_period,
                trend_length=unified_trend_length,
                trend_threshold=unified_trend_threshold,
                adaptability_factor=unified_adaptability,
                src_type=src_type,
                enable_consensus_filter=unified_consensus
            )
        elif self.filter_type == FilterType.X_CHOPPINESS:
            self.filter_signal = XChoppinessFilterSignal(
                use_dynamic_period=True,
                detector_type=x_choppiness_detector_type,
                lp_period=x_choppiness_lp_period,
                hp_period=x_choppiness_hp_period,
                cycle_part=x_choppiness_cycle_part,
                max_cycle=x_choppiness_max_cycle,
                min_cycle=x_choppiness_min_cycle,
                max_output=x_choppiness_max_output,
                min_output=x_choppiness_min_output
            )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._long_signals = None
        self._short_signals = None
        self._x_mama_signals = None
        self._filter_signals = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算"""
        try:
            current_len = len(data)
            
            # データ長が変わった場合のみ再計算
            if self._long_signals is None or current_len != self._data_len:
                # データフレームの作成（必要な列のみ）
                if isinstance(data, pd.DataFrame):
                    df = data[['open', 'high', 'low', 'close']]
                else:
                    df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
                
                try:
                    # X_MAMAシグナルの計算
                    x_mama_signals = self.x_mama_entry_signal.generate(df)
                    self._x_mama_signals = x_mama_signals
                    
                    # フィルターシグナルの計算
                    if self.filter_signal is not None:
                        filter_signals = self.filter_signal.generate(df)
                        self._filter_signals = filter_signals
                    else:
                        self._filter_signals = np.zeros(current_len, dtype=np.int8)
                    
                    # シグナルの統合
                    filter_type_int = list(FilterType).index(self.filter_type)
                    self._long_signals, self._short_signals = combine_signals_numba(
                        x_mama_signals, self._filter_signals, filter_type_int
                    )
                    
                except Exception as e:
                    self.logger.error(f"シグナル計算中にエラー: {str(e)}")
                    # エラー時はゼロシグナルを設定
                    self._long_signals = np.zeros(current_len, dtype=np.int8)
                    self._short_signals = np.zeros(current_len, dtype=np.int8)
                    self._x_mama_signals = np.zeros(current_len, dtype=np.int8)
                    self._filter_signals = np.zeros(current_len, dtype=np.int8)
                
                self._data_len = current_len
        except Exception as e:
            self.logger.error(f"calculate_signals全体でエラー: {str(e)}")
            # エラー時はゼロシグナルを設定
            if data is not None:
                self._long_signals = np.zeros(len(data), dtype=np.int8)
                self._short_signals = np.zeros(len(data), dtype=np.int8)
                self._x_mama_signals = np.zeros(len(data), dtype=np.int8)
                self._filter_signals = np.zeros(len(data), dtype=np.int8)
                self._data_len = len(data)
    
    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナル取得
        
        Returns:
            統合されたエントリーシグナル（ロング=1、ショート=-1、なし=0）
        """
        if self._long_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        
        # ロング・ショートシグナルを統合
        combined_signals = self._long_signals - self._short_signals
        return combined_signals
    
    def get_long_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """ロングエントリーシグナル取得"""
        if self._long_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._long_signals
    
    def get_short_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """ショートエントリーシグナル取得"""
        if self._short_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._short_signals
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """エグジットシグナル生成"""
        if self._x_mama_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        
        if index == -1:
            index = len(data) - 1
        
        # Numba最適化された関数を使用
        return check_exit_conditions_numba(self._x_mama_signals, position, index)
    
    def get_x_mama_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """X_MAMAシグナル取得"""
        if self._x_mama_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._x_mama_signals
    
    def get_filter_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """フィルターシグナル取得"""
        if self._filter_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._filter_signals
    
    def get_mama_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """X_MAMA値を取得"""
        try:
            if data is not None:
                self.calculate_signals(data)
            return self.x_mama_entry_signal.get_mama_values()
        except Exception as e:
            self.logger.error(f"X_MAMA値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_fama_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """X_FAMA値を取得"""
        try:
            if data is not None:
                self.calculate_signals(data)
            return self.x_mama_entry_signal.get_fama_values()
        except Exception as e:
            self.logger.error(f"X_FAMA値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_filter_details(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        フィルター詳細情報を取得
        
        Returns:
            フィルタータイプに応じた詳細データ
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            if self.filter_signal is None:
                return {}
            
            details = {}
            
            if self.filter_type == FilterType.PHASOR_TREND:
                details.update({
                    'trend_values': self.filter_signal.get_trend_values(),
                    'phase_angle': self.filter_signal.get_phase_angle(),
                    'state_values': self.filter_signal.get_state_values(),
                    'trend_strength': self.filter_signal.get_trend_strength(),
                    'cycle_confidence': self.filter_signal.get_cycle_confidence(),
                    'instantaneous_period': self.filter_signal.get_instantaneous_period()
                })
            elif self.filter_type == FilterType.CORRELATION_CYCLE:
                details.update({
                    'angle_values': self.filter_signal.get_angle_values(),
                    'real_component': self.filter_signal.get_real_component(),
                    'imag_component': self.filter_signal.get_imag_component(),
                    'state_values': self.filter_signal.get_state_values(),
                    'rate_of_change': self.filter_signal.get_rate_of_change(),
                    'cycle_mode': self.filter_signal.get_cycle_mode()
                })
            elif self.filter_type == FilterType.CORRELATION_TREND:
                details.update({
                    'correlation_values': self.filter_signal.get_correlation_values(),
                    'trend_signal_values': self.filter_signal.get_trend_signal_values(),
                    'trend_strength': self.filter_signal.get_trend_strength(),
                    'smoothed_values': self.filter_signal.get_smoothed_values()
                })
            elif self.filter_type == FilterType.UNIFIED_TREND_CYCLE:
                details.update({
                    'unified_trend_strength': self.filter_signal.get_unified_trend_strength(),
                    'unified_cycle_confidence': self.filter_signal.get_unified_cycle_confidence(),
                    'unified_state_values': self.filter_signal.get_unified_state_values(),
                    'unified_signal': self.filter_signal.get_unified_signal(),
                    'consensus_strength': self.filter_signal.get_consensus_strength(),
                    'overall_confidence': self.filter_signal.get_overall_confidence()
                })
            elif self.filter_type == FilterType.X_CHOPPINESS:
                details.update({
                    'x_choppiness_values': self.filter_signal.get_x_choppiness_values(),
                    'midline_values': self.filter_signal.get_midline_values(),
                    'trend_signal_values': self.filter_signal.get_trend_signal_values(),
                    'str_values': self.filter_signal.get_str_values(),
                    'trend_state': self.filter_signal.get_trend_state(),
                    'trend_intensity': self.filter_signal.get_trend_intensity()
                })
            
            return details
        except Exception as e:
            self.logger.error(f"フィルター詳細取得中にエラー: {str(e)}")
            return {}
    
    def get_advanced_metrics(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        全ての高度なメトリクスを取得
        
        Returns:
            X_MAMAとフィルターの全メトリクス
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            metrics = {
                # X_MAMAメトリクス
                'mama_values': self.get_mama_values(),
                'fama_values': self.get_fama_values(),
                'x_mama_signals': self.get_x_mama_signals(data),
                # シグナルメトリクス
                'long_signals': self.get_long_signals(data),
                'short_signals': self.get_short_signals(data),
                'filter_signals': self.get_filter_signals(data),
                # フィルター詳細
                'filter_type': self.filter_type.value
            }
            
            # フィルター固有のメトリクスを追加
            filter_details = self.get_filter_details()
            metrics.update(filter_details)
            
            return metrics
        except Exception as e:
            self.logger.error(f"高度なメトリクス取得中にエラー: {str(e)}")
            return {}
    
    def reset(self) -> None:
        """シグナルジェネレーターの状態をリセット"""
        super().reset()
        self._data_len = 0
        self._long_signals = None
        self._short_signals = None
        self._x_mama_signals = None
        self._filter_signals = None
        
        if hasattr(self.x_mama_entry_signal, 'reset'):
            self.x_mama_entry_signal.reset()
        if self.filter_signal is not None and hasattr(self.filter_signal, 'reset'):
            self.filter_signal.reset()