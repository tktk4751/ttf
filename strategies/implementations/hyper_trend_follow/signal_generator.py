#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HyperTrendFollow戦略用シグナルジェネレーター

HyperFRAMA位置関係とHyperFRAMAChannelブレイクアウトを組み合わせたトレンドフォローシグナル
"""

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from numba import njit

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.hyper_trend_follow.hyper_trend_follow_signal import HyperTrendFollowSignal


@njit(fastmath=True)
def check_exit_conditions_numba(trend_follow_signals: np.ndarray, position: int, index: int) -> bool:
    """
    エグジット条件をチェックする（Numba最適化版）
    
    Args:
        trend_follow_signals: HyperTrendFollowシグナル配列
        position: 現在のポジション（1=ロング、-1=ショート）
        index: チェックするインデックス
    
    Returns:
        bool: エグジットすべきかどうか
    """
    if index < 0 or index >= len(trend_follow_signals):
        return False
    
    trend_follow_signal = trend_follow_signals[index]
    
    # ロングポジション: シグナル=-1でエグジット
    if position == 1 and trend_follow_signal == -1:
        return True
    
    # ショートポジション: シグナル=1でエグジット
    if position == -1 and trend_follow_signal == 1:
        return True
    
    return False


class HyperTrendFollowSignalGenerator(BaseSignalGenerator):
    """
    HyperTrendFollowシグナル用のシグナルジェネレーター
    
    特徴:
    - HyperFRAMA位置関係シグナルとHyperFRAMAChannelブレイクアウトシグナルの組み合わせ
    - 包括的なパラメータサポート
    - Numba最適化による高速処理
    """
    
    def __init__(
        self,
        # === HyperFRAMA位置関係シグナルパラメータ ===
        # 基本パラメータ
        hyper_frama_period: int = 16,
        hyper_frama_src_type: str = 'hl2',
        hyper_frama_fc: int = 1,
        hyper_frama_sc: int = 198,
        hyper_frama_alpha_multiplier: float = 0.5,
        # 動的期間パラメータ
        hyper_frama_period_mode: str = 'fixed',
        hyper_frama_cycle_detector_type: str = 'hody_e',
        hyper_frama_lp_period: int = 13,
        hyper_frama_hp_period: int = 124,
        hyper_frama_cycle_part: float = 0.5,
        hyper_frama_max_cycle: int = 89,
        hyper_frama_min_cycle: int = 8,
        hyper_frama_max_output: int = 124,
        hyper_frama_min_output: int = 8,
        
        # === HyperFRAMAChannelブレイクアウトシグナルパラメータ ===
        # 基本パラメータ
        channel_band_lookback: int = 1,
        channel_exit_mode: int = 1,  # 1: 逆ブレイクアウト, 2: 中心線クロス
        channel_src_type: str = 'hlc3',
        
        # HyperFRAMAChannel 基本パラメータ
        channel_period: int = 14,
        channel_multiplier_mode: str = "dynamic",
        channel_fixed_multiplier: float = 2.0,
        channel_hyper_frama_channel_src_type: str = "hlc3",
        
        # HyperFRAMA パラメータ（チャネル用）
        channel_hyper_frama_period: int = 16,
        channel_hyper_frama_src_type: str = 'hl2',
        channel_hyper_frama_fc: int = 1,
        channel_hyper_frama_sc: int = 198,
        channel_hyper_frama_alpha_multiplier: float = 0.5,
        # 動的期間パラメータ（チャネル用）
        channel_hyper_frama_period_mode: str = 'fixed',
        channel_hyper_frama_cycle_detector_type: str = 'hody_e',
        channel_hyper_frama_lp_period: int = 13,
        channel_hyper_frama_hp_period: int = 124,
        channel_hyper_frama_cycle_part: float = 0.5,
        channel_hyper_frama_max_cycle: int = 89,
        channel_hyper_frama_min_cycle: int = 8,
        channel_hyper_frama_max_output: int = 124,
        channel_hyper_frama_min_output: int = 8,
        
        # X_ATR パラメータ（チャネル用）
        channel_x_atr_period: float = 12.0,
        channel_x_atr_tr_method: str = 'atr',
        channel_x_atr_smoother_type: str = 'frama',
        channel_x_atr_src_type: str = 'close',
        channel_x_atr_enable_kalman: bool = False,
        channel_x_atr_kalman_type: str = 'unscented',
        
        # HyperER パラメータ（チャネル用）
        channel_hyper_er_period: int = 8,
        channel_hyper_er_midline_period: int = 100,
        channel_hyper_er_er_period: int = 13,
        channel_hyper_er_er_src_type: str = 'oc2',
        channel_hyper_er_use_kalman_filter: bool = True,
        channel_hyper_er_kalman_filter_type: str = 'simple',
        channel_hyper_er_kalman_process_noise: float = 1e-5,
        channel_hyper_er_kalman_min_observation_noise: float = 1e-6,
        channel_hyper_er_kalman_adaptation_window: int = 5,
        channel_hyper_er_use_roofing_filter: bool = True,
        channel_hyper_er_roofing_hp_cutoff: float = 55.0,
        channel_hyper_er_roofing_ss_band_edge: float = 10.0,
        channel_hyper_er_use_smoothing: bool = True,
        channel_hyper_er_smoother_type: str = 'frama',
        channel_hyper_er_smoother_period: int = 16
    ):
        """
        初期化
        
        Args:
            hyper_frama_*: HyperFRAMA位置関係シグナル用パラメータ
            channel_*: HyperFRAMAChannelブレイクアウトシグナル用パラメータ
        """
        
        super().__init__(f"HyperTrendFollowSignalGenerator")
        
        # パラメータの設定
        self._params = {
            # HyperFRAMA位置関係シグナルパラメータ
            'hyper_frama_period': hyper_frama_period,
            'hyper_frama_src_type': hyper_frama_src_type,
            'hyper_frama_fc': hyper_frama_fc,
            'hyper_frama_sc': hyper_frama_sc,
            'hyper_frama_alpha_multiplier': hyper_frama_alpha_multiplier,
            'hyper_frama_period_mode': hyper_frama_period_mode,
            'hyper_frama_cycle_detector_type': hyper_frama_cycle_detector_type,
            'hyper_frama_lp_period': hyper_frama_lp_period,
            'hyper_frama_hp_period': hyper_frama_hp_period,
            'hyper_frama_cycle_part': hyper_frama_cycle_part,
            'hyper_frama_max_cycle': hyper_frama_max_cycle,
            'hyper_frama_min_cycle': hyper_frama_min_cycle,
            'hyper_frama_max_output': hyper_frama_max_output,
            'hyper_frama_min_output': hyper_frama_min_output,
            
            # HyperFRAMAChannelブレイクアウトシグナルパラメータ
            'channel_band_lookback': channel_band_lookback,
            'channel_exit_mode': channel_exit_mode,
            'channel_src_type': channel_src_type,
            'channel_period': channel_period,
            'channel_multiplier_mode': channel_multiplier_mode,
            'channel_fixed_multiplier': channel_fixed_multiplier,
            'channel_hyper_frama_channel_src_type': channel_hyper_frama_channel_src_type,
            'channel_hyper_frama_period': channel_hyper_frama_period,
            'channel_hyper_frama_src_type': channel_hyper_frama_src_type,
            'channel_hyper_frama_fc': channel_hyper_frama_fc,
            'channel_hyper_frama_sc': channel_hyper_frama_sc,
            'channel_hyper_frama_alpha_multiplier': channel_hyper_frama_alpha_multiplier,
            'channel_hyper_frama_period_mode': channel_hyper_frama_period_mode,
            'channel_hyper_frama_cycle_detector_type': channel_hyper_frama_cycle_detector_type,
            'channel_hyper_frama_lp_period': channel_hyper_frama_lp_period,
            'channel_hyper_frama_hp_period': channel_hyper_frama_hp_period,
            'channel_hyper_frama_cycle_part': channel_hyper_frama_cycle_part,
            'channel_hyper_frama_max_cycle': channel_hyper_frama_max_cycle,
            'channel_hyper_frama_min_cycle': channel_hyper_frama_min_cycle,
            'channel_hyper_frama_max_output': channel_hyper_frama_max_output,
            'channel_hyper_frama_min_output': channel_hyper_frama_min_output,
            
            # X_ATRパラメータ（チャネル用）
            'channel_x_atr_period': channel_x_atr_period,
            'channel_x_atr_tr_method': channel_x_atr_tr_method,
            'channel_x_atr_smoother_type': channel_x_atr_smoother_type,
            'channel_x_atr_src_type': channel_x_atr_src_type,
            'channel_x_atr_enable_kalman': channel_x_atr_enable_kalman,
            'channel_x_atr_kalman_type': channel_x_atr_kalman_type,
            
            # HyperERパラメータ（チャネル用）
            'channel_hyper_er_period': channel_hyper_er_period,
            'channel_hyper_er_midline_period': channel_hyper_er_midline_period,
            'channel_hyper_er_er_period': channel_hyper_er_er_period,
            'channel_hyper_er_er_src_type': channel_hyper_er_er_src_type,
            'channel_hyper_er_use_kalman_filter': channel_hyper_er_use_kalman_filter,
            'channel_hyper_er_kalman_filter_type': channel_hyper_er_kalman_filter_type,
            'channel_hyper_er_kalman_process_noise': channel_hyper_er_kalman_process_noise,
            'channel_hyper_er_kalman_min_observation_noise': channel_hyper_er_kalman_min_observation_noise,
            'channel_hyper_er_kalman_adaptation_window': channel_hyper_er_kalman_adaptation_window,
            'channel_hyper_er_use_roofing_filter': channel_hyper_er_use_roofing_filter,
            'channel_hyper_er_roofing_hp_cutoff': channel_hyper_er_roofing_hp_cutoff,
            'channel_hyper_er_roofing_ss_band_edge': channel_hyper_er_roofing_ss_band_edge,
            'channel_hyper_er_use_smoothing': channel_hyper_er_use_smoothing,
            'channel_hyper_er_smoother_type': channel_hyper_er_smoother_type,
            'channel_hyper_er_smoother_period': channel_hyper_er_smoother_period,
        }
        
        # HyperTrendFollowSignalの初期化
        self.trend_follow_signal = HyperTrendFollowSignal(**self._params)
        
        # キャッシュ用の変数
        self._data_len = 0
        self._trend_follow_signals = None
        
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算"""
        try:
            current_len = len(data)
            
            # データ長が変わった場合のみ再計算
            if self._trend_follow_signals is None or current_len != self._data_len:
                # データフレームの作成（必要な列のみ）
                if isinstance(data, pd.DataFrame):
                    df = data[['open', 'high', 'low', 'close']]
                    if 'volume' in data.columns:
                        df = data[['open', 'high', 'low', 'close', 'volume']]
                else:
                    if data.shape[1] >= 5:
                        df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close', 'volume'])
                    else:
                        df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
                
                try:
                    # HyperTrendFollowシグナルの計算
                    self._trend_follow_signals = self.trend_follow_signal.generate(df)
                    
                except Exception as e:
                    self.logger.error(f"シグナル計算中にエラー: {str(e)}")
                    # エラー時はゼロシグナルを設定
                    self._trend_follow_signals = np.zeros(current_len, dtype=np.int8)
                
                self._data_len = current_len
        except Exception as e:
            self.logger.error(f"calculate_signals全体でエラー: {str(e)}")
            # エラー時はゼロシグナルを設定
            if data is not None:
                self._trend_follow_signals = np.zeros(len(data), dtype=np.int8)
                self._data_len = len(data)

    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナル取得
        
        Returns:
            統合されたエントリーシグナル（ロング=1、ショート=-1、なし=0）
        """
        if self._trend_follow_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        
        return self._trend_follow_signals
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """エグジットシグナル生成"""
        if self._trend_follow_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        
        if index == -1:
            index = len(data) - 1
        
        # Numba最適化された関数を使用
        return check_exit_conditions_numba(self._trend_follow_signals, position, index)
    
    def get_trend_follow_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """HyperTrendFollowシグナル取得"""
        if self._trend_follow_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._trend_follow_signals
    
    def reset(self) -> None:
        """シグナルジェネレーターの状態をリセット"""
        super().reset()
        self._data_len = 0
        self._trend_follow_signals = None
        
        if hasattr(self.trend_follow_signal, 'reset'):
            self.trend_follow_signal.reset()

