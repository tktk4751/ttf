#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HyperFRAMAChannelブレイクアウトシグナル（シンプル版）
HyperFRAMAのトレンド判定を除き、チャネルのブレイクアウトのみを検出
"""

from typing import Union, Dict, Any, Optional
import numpy as np
import pandas as pd
from numba import jit, njit, prange, int8, int64, float64, boolean, optional

from signals.interfaces.entry import IEntrySignal
from signals.interfaces.exit import IExitSignal
from indicators.hyper_frama_channel import HyperFRAMAChannel
from indicators.price_source import PriceSource


@njit(int8[:](float64[:], float64[:], float64[:], int64), fastmath=True, cache=True)
def calculate_channel_breakout_entry_signals(
    close: np.ndarray, 
    upper: np.ndarray, 
    lower: np.ndarray,
    lookback: int
) -> np.ndarray:
    """
    HyperFRAMAチャネルブレイクアウト エントリーシグナルを計算する（シンプル版）
    
    Args:
        close: 終値の配列
        upper: HyperFRAMAチャネルアッパーバンドの配列
        lower: HyperFRAMAチャネルロワーバンドの配列
        lookback: 過去のバンドを参照する期間
    
    Returns:
        シグナルの配列 (1: ロングエントリー, -1: ショートエントリー, 0: エントリーなし)
    """
    length = len(close)
    signals = np.zeros(length, dtype=np.int8)
    
    # ブレイクアウトの判定
    for i in prange(lookback + 1, length):
        # すべての値が有効かチェック
        if (np.isnan(close[i]) or np.isnan(close[i-1]) or 
            np.isnan(upper[i]) or np.isnan(upper[i-1]) or 
            np.isnan(lower[i]) or np.isnan(lower[i-1])):
            signals[i] = 0
            continue
            
        # ロングエントリー: HyperFRAMAチャネル上方向ブレイクアウト
        if close[i-1] <= upper[i-1] and close[i] > upper[i]:
            signals[i] = 1
        # ショートエントリー: HyperFRAMAチャネル下方向ブレイクアウト
        elif close[i-1] >= lower[i-1] and close[i] < lower[i]:
            signals[i] = -1
        # 近似ブレイクアウト検出（より敏感なシグナル）
        elif lookback > 0 and i > lookback:
            # ロング近似ブレイクアウト
            if (close[i] > close[i-1] and 
                close[i-1] <= upper[i-1] and 
                close[i] >= upper[i] * 0.995 and 
                close[i-1] < upper[i-1] * 0.995):
                signals[i] = 1
            # ショート近似ブレイクアウト
            elif (close[i] < close[i-1] and 
                  close[i-1] >= lower[i-1] and 
                  close[i] <= lower[i] * 1.005 and 
                  close[i-1] > lower[i-1] * 1.005):
                signals[i] = -1
    
    return signals


@njit(int8[:](float64[:], float64[:], float64[:], int64), fastmath=True, cache=True)
def calculate_channel_breakout_exit_signals(
    close: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    exit_mode: int
) -> np.ndarray:
    """
    HyperFRAMAチャネル逆ブレイクアウトによる エグジットシグナルを計算する（シンプル版）
    
    Args:
        close: 終値の配列
        upper: HyperFRAMAチャネルアッパーバンドの配列
        lower: HyperFRAMAチャネルロワーバンドの配列
        exit_mode: エグジットモード (1: 逆ブレイクアウト, 2: 中心線クロス)
    
    Returns:
        エグジットシグナルの配列 (1: ロングエグジット, -1: ショートエグジット, 0: エグジットなし)
    """
    length = len(close)
    signals = np.zeros(length, dtype=np.int8)
    
    # 中心線を計算（上限と下限の平均）
    midline = (upper + lower) / 2.0
    
    for i in prange(1, length):
        # すべての値が有効かチェック
        if (np.isnan(close[i]) or np.isnan(close[i-1]) or
            np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(midline[i])):
            signals[i] = 0
            continue
        
        if exit_mode == 1:
            # モード1: 逆ブレイクアウトによるエグジット
            # ロングエグジット: 価格が下限バンドを下抜けた場合
            if close[i-1] >= lower[i-1] and close[i] < lower[i]:
                signals[i] = 1
            # ショートエグジット: 価格が上限バンドを上抜けた場合
            elif close[i-1] <= upper[i-1] and close[i] > upper[i]:
                signals[i] = -1
        
        elif exit_mode == 2:
            # モード2: 中心線クロスによるエグジット
            # ロングエグジット: 価格が中心線を下抜けた場合
            if close[i-1] >= midline[i-1] and close[i] < midline[i]:
                signals[i] = 1
            # ショートエグジット: 価格が中心線を上抜けた場合
            elif close[i-1] <= midline[i-1] and close[i] > midline[i]:
                signals[i] = -1
    
    return signals


class HyperFRAMAChannelBreakoutSignal(IEntrySignal, IExitSignal):
    """
    HyperFRAMAChannelブレイクアウトシグナル（シンプル版）
    
    HyperFRAMAのトレンド判定を除き、チャネルのブレイクアウトのみを検出
    
    エントリー条件:
    - ロングエントリー: HyperFRAMAチャネル上方向ブレイクアウト (1)
    - ショートエントリー: HyperFRAMAチャネル下方向ブレイクアウト (-1)
    
    エグジット条件:
    モード1 (逆ブレイクアウト):
    - ロングエグジット: 価格が下限バンドを下抜け (1)
    - ショートエグジット: 価格が上限バンドを上抜け (-1)
    
    モード2 (中心線クロス):
    - ロングエグジット: 価格が中心線を下抜け (1)
    - ショートエグジット: 価格が中心線を上抜け (-1)
    """
    
    def __init__(
        self,
        # 基本パラメータ
        band_lookback: int = 1,
        exit_mode: int = 1,  # 1: 逆ブレイクアウト, 2: 中心線クロス
        src_type: str = 'hlc3',
        
        # === HyperFRAMAChannel 基本パラメータ ===
        channel_period: int = 14,
        channel_multiplier_mode: str = "dynamic",
        channel_fixed_multiplier: float = 2.0,
        channel_src_type: str = "hlc3",
        
        # === HyperFRAMA パラメータ ===
        # 基本パラメータ
        channel_hyper_frama_period: int = 16,
        channel_hyper_frama_src_type: str = 'hl2',
        channel_hyper_frama_fc: int = 1,
        channel_hyper_frama_sc: int = 198,
        channel_hyper_frama_alpha_multiplier: float = 0.5,
        # 動的期間パラメータ
        channel_hyper_frama_period_mode: str = 'fixed',
        channel_hyper_frama_cycle_detector_type: str = 'hody_e',
        channel_hyper_frama_lp_period: int = 13,
        channel_hyper_frama_hp_period: int = 124,
        channel_hyper_frama_cycle_part: float = 0.5,
        channel_hyper_frama_max_cycle: int = 89,
        channel_hyper_frama_min_cycle: int = 8,
        channel_hyper_frama_max_output: int = 124,
        channel_hyper_frama_min_output: int = 8,
        # 動的適応パラメータ
        channel_hyper_frama_enable_indicator_adaptation: bool = True,
        channel_hyper_frama_adaptation_indicator: str = 'hyper_er',
        channel_hyper_frama_hyper_er_period: int = 14,
        channel_hyper_frama_hyper_er_midline_period: int = 100,
        channel_hyper_frama_hyper_adx_period: int = 14,
        channel_hyper_frama_hyper_adx_midline_period: int = 100,
        channel_hyper_frama_hyper_trend_index_period: int = 14,
        channel_hyper_frama_hyper_trend_index_midline_period: int = 100,
        channel_hyper_frama_fc_min: float = 1.0,
        channel_hyper_frama_fc_max: float = 8.0,
        channel_hyper_frama_sc_min: float = 50.0,
        channel_hyper_frama_sc_max: float = 250.0,
        channel_hyper_frama_period_min: int = 4,
        channel_hyper_frama_period_max: int = 44,
        
        # === X_ATR パラメータ ===
        channel_x_atr_period: float = 12.0,
        channel_x_atr_tr_method: str = 'atr',
        channel_x_atr_smoother_type: str = 'frama',
        channel_x_atr_src_type: str = 'close',
        channel_x_atr_enable_kalman: bool = False,
        channel_x_atr_kalman_type: str = 'unscented',
        # 動的適応パラメータ
        channel_x_atr_period_mode: str = 'dynamic',
        channel_x_atr_cycle_detector_type: str = 'practical',
        channel_x_atr_cycle_detector_cycle_part: float = 0.5,
        channel_x_atr_cycle_detector_max_cycle: int = 55,
        channel_x_atr_cycle_detector_min_cycle: int = 5,
        channel_x_atr_cycle_period_multiplier: float = 1.0,
        channel_x_atr_cycle_detector_period_range: tuple = (5, 120),
        # ミッドラインパラメータ
        channel_x_atr_midline_period: int = 100,
        # パーセンタイルベースボラティリティ分析パラメータ
        channel_x_atr_enable_percentile_analysis: bool = True,
        channel_x_atr_percentile_lookback_period: int = 50,
        channel_x_atr_percentile_low_threshold: float = 0.25,
        channel_x_atr_percentile_high_threshold: float = 0.75,
        # スムーサーパラメータ
        channel_x_atr_smoother_params: dict = None,
        # カルマンフィルターパラメータ
        channel_x_atr_kalman_params: dict = None,
        
        # === HyperER パラメータ ===
        channel_hyper_er_period: int = 8,
        channel_hyper_er_midline_period: int = 100,
        # ERパラメータ
        channel_hyper_er_er_period: int = 13,
        channel_hyper_er_er_src_type: str = 'oc2',
        # 統合カルマンフィルターパラメータ
        channel_hyper_er_use_kalman_filter: bool = True,
        channel_hyper_er_kalman_filter_type: str = 'simple',
        channel_hyper_er_kalman_process_noise: float = 1e-5,
        channel_hyper_er_kalman_min_observation_noise: float = 1e-6,
        channel_hyper_er_kalman_adaptation_window: int = 5,
        # ルーフィングフィルターパラメータ
        channel_hyper_er_use_roofing_filter: bool = True,
        channel_hyper_er_roofing_hp_cutoff: float = 55.0,
        channel_hyper_er_roofing_ss_band_edge: float = 10.0,
        # ラゲールフィルターパラメータ（後方互換性のため残す）
        channel_hyper_er_use_laguerre_filter: bool = False,
        channel_hyper_er_laguerre_gamma: float = 0.5,
        # 平滑化オプション
        channel_hyper_er_use_smoothing: bool = True,
        channel_hyper_er_smoother_type: str = 'frama',
        channel_hyper_er_smoother_period: int = 16,
        channel_hyper_er_smoother_src_type: str = 'close',
        # エラーズ統合サイクル検出器パラメータ
        channel_hyper_er_use_dynamic_period: bool = True,
        channel_hyper_er_detector_type: str = 'dft_dominant',
        channel_hyper_er_lp_period: int = 13,
        channel_hyper_er_hp_period: int = 124,
        channel_hyper_er_cycle_part: float = 0.4,
        channel_hyper_er_max_cycle: int = 124,
        channel_hyper_er_min_cycle: int = 13,
        channel_hyper_er_max_output: int = 89,
        channel_hyper_er_min_output: int = 5,
        # パーセンタイルベーストレンド分析パラメータ
        channel_hyper_er_enable_percentile_analysis: bool = True,
        channel_hyper_er_percentile_lookback_period: int = 50,
        channel_hyper_er_percentile_low_threshold: float = 0.25,
        channel_hyper_er_percentile_high_threshold: float = 0.75,
        
        # === HyperFRAMAチャネル独自パラメータ ===
        channel_enable_signals: bool = True,
        channel_enable_percentile: bool = True,
        channel_percentile_period: int = 100
    ):
        # 基本パラメータ格納
        self.band_lookback = band_lookback
        self.exit_mode = exit_mode
        self.src_type = src_type
        
        # チャネル基本パラメータ
        self.channel_period = channel_period
        self.channel_multiplier_mode = channel_multiplier_mode
        self.channel_fixed_multiplier = channel_fixed_multiplier
        self.channel_src_type = channel_src_type
        
        # HyperFRAMAパラメータ
        self.channel_hyper_frama_period = channel_hyper_frama_period
        self.channel_hyper_frama_src_type = channel_hyper_frama_src_type
        self.channel_hyper_frama_fc = channel_hyper_frama_fc
        self.channel_hyper_frama_sc = channel_hyper_frama_sc
        self.channel_hyper_frama_alpha_multiplier = channel_hyper_frama_alpha_multiplier
        self.channel_hyper_frama_period_mode = channel_hyper_frama_period_mode
        self.channel_hyper_frama_cycle_detector_type = channel_hyper_frama_cycle_detector_type
        self.channel_hyper_frama_lp_period = channel_hyper_frama_lp_period
        self.channel_hyper_frama_hp_period = channel_hyper_frama_hp_period
        self.channel_hyper_frama_cycle_part = channel_hyper_frama_cycle_part
        self.channel_hyper_frama_max_cycle = channel_hyper_frama_max_cycle
        self.channel_hyper_frama_min_cycle = channel_hyper_frama_min_cycle
        self.channel_hyper_frama_max_output = channel_hyper_frama_max_output
        self.channel_hyper_frama_min_output = channel_hyper_frama_min_output
        self.channel_hyper_frama_enable_indicator_adaptation = channel_hyper_frama_enable_indicator_adaptation
        self.channel_hyper_frama_adaptation_indicator = channel_hyper_frama_adaptation_indicator
        self.channel_hyper_frama_hyper_er_period = channel_hyper_frama_hyper_er_period
        self.channel_hyper_frama_hyper_er_midline_period = channel_hyper_frama_hyper_er_midline_period
        self.channel_hyper_frama_hyper_adx_period = channel_hyper_frama_hyper_adx_period
        self.channel_hyper_frama_hyper_adx_midline_period = channel_hyper_frama_hyper_adx_midline_period
        self.channel_hyper_frama_hyper_trend_index_period = channel_hyper_frama_hyper_trend_index_period
        self.channel_hyper_frama_hyper_trend_index_midline_period = channel_hyper_frama_hyper_trend_index_midline_period
        self.channel_hyper_frama_fc_min = channel_hyper_frama_fc_min
        self.channel_hyper_frama_fc_max = channel_hyper_frama_fc_max
        self.channel_hyper_frama_sc_min = channel_hyper_frama_sc_min
        self.channel_hyper_frama_sc_max = channel_hyper_frama_sc_max
        self.channel_hyper_frama_period_min = channel_hyper_frama_period_min
        self.channel_hyper_frama_period_max = channel_hyper_frama_period_max
        
        # X_ATRパラメータ
        self.channel_x_atr_period = channel_x_atr_period
        self.channel_x_atr_tr_method = channel_x_atr_tr_method
        self.channel_x_atr_smoother_type = channel_x_atr_smoother_type
        self.channel_x_atr_src_type = channel_x_atr_src_type
        self.channel_x_atr_enable_kalman = channel_x_atr_enable_kalman
        self.channel_x_atr_kalman_type = channel_x_atr_kalman_type
        self.channel_x_atr_period_mode = channel_x_atr_period_mode
        self.channel_x_atr_cycle_detector_type = channel_x_atr_cycle_detector_type
        self.channel_x_atr_cycle_detector_cycle_part = channel_x_atr_cycle_detector_cycle_part
        self.channel_x_atr_cycle_detector_max_cycle = channel_x_atr_cycle_detector_max_cycle
        self.channel_x_atr_cycle_detector_min_cycle = channel_x_atr_cycle_detector_min_cycle
        self.channel_x_atr_cycle_period_multiplier = channel_x_atr_cycle_period_multiplier
        self.channel_x_atr_cycle_detector_period_range = channel_x_atr_cycle_detector_period_range
        self.channel_x_atr_midline_period = channel_x_atr_midline_period
        self.channel_x_atr_enable_percentile_analysis = channel_x_atr_enable_percentile_analysis
        self.channel_x_atr_percentile_lookback_period = channel_x_atr_percentile_lookback_period
        self.channel_x_atr_percentile_low_threshold = channel_x_atr_percentile_low_threshold
        self.channel_x_atr_percentile_high_threshold = channel_x_atr_percentile_high_threshold
        self.channel_x_atr_smoother_params = channel_x_atr_smoother_params
        self.channel_x_atr_kalman_params = channel_x_atr_kalman_params
        
        # HyperERパラメータ
        self.channel_hyper_er_period = channel_hyper_er_period
        self.channel_hyper_er_midline_period = channel_hyper_er_midline_period
        self.channel_hyper_er_er_period = channel_hyper_er_er_period
        self.channel_hyper_er_er_src_type = channel_hyper_er_er_src_type
        self.channel_hyper_er_use_kalman_filter = channel_hyper_er_use_kalman_filter
        self.channel_hyper_er_kalman_filter_type = channel_hyper_er_kalman_filter_type
        self.channel_hyper_er_kalman_process_noise = channel_hyper_er_kalman_process_noise
        self.channel_hyper_er_kalman_min_observation_noise = channel_hyper_er_kalman_min_observation_noise
        self.channel_hyper_er_kalman_adaptation_window = channel_hyper_er_kalman_adaptation_window
        self.channel_hyper_er_use_roofing_filter = channel_hyper_er_use_roofing_filter
        self.channel_hyper_er_roofing_hp_cutoff = channel_hyper_er_roofing_hp_cutoff
        self.channel_hyper_er_roofing_ss_band_edge = channel_hyper_er_roofing_ss_band_edge
        self.channel_hyper_er_use_laguerre_filter = channel_hyper_er_use_laguerre_filter
        self.channel_hyper_er_laguerre_gamma = channel_hyper_er_laguerre_gamma
        self.channel_hyper_er_use_smoothing = channel_hyper_er_use_smoothing
        self.channel_hyper_er_smoother_type = channel_hyper_er_smoother_type
        self.channel_hyper_er_smoother_period = channel_hyper_er_smoother_period
        self.channel_hyper_er_smoother_src_type = channel_hyper_er_smoother_src_type
        self.channel_hyper_er_use_dynamic_period = channel_hyper_er_use_dynamic_period
        self.channel_hyper_er_detector_type = channel_hyper_er_detector_type
        self.channel_hyper_er_lp_period = channel_hyper_er_lp_period
        self.channel_hyper_er_hp_period = channel_hyper_er_hp_period
        self.channel_hyper_er_cycle_part = channel_hyper_er_cycle_part
        self.channel_hyper_er_max_cycle = channel_hyper_er_max_cycle
        self.channel_hyper_er_min_cycle = channel_hyper_er_min_cycle
        self.channel_hyper_er_max_output = channel_hyper_er_max_output
        self.channel_hyper_er_min_output = channel_hyper_er_min_output
        self.channel_hyper_er_enable_percentile_analysis = channel_hyper_er_enable_percentile_analysis
        self.channel_hyper_er_percentile_lookback_period = channel_hyper_er_percentile_lookback_period
        self.channel_hyper_er_percentile_low_threshold = channel_hyper_er_percentile_low_threshold
        self.channel_hyper_er_percentile_high_threshold = channel_hyper_er_percentile_high_threshold
        
        # HyperFRAMAチャネル独自パラメータ
        self.channel_enable_signals = channel_enable_signals
        self.channel_enable_percentile = channel_enable_percentile
        self.channel_percentile_period = channel_percentile_period
        
        # インジケーター初期化
        self._setup_indicators()
        
        # パラメータをまとめて格納
        self._params = {
            'band_lookback': band_lookback,
            'exit_mode': exit_mode,
            'src_type': src_type,
            'channel_period': channel_period,
            'channel_multiplier_mode': channel_multiplier_mode,
            'channel_fixed_multiplier': channel_fixed_multiplier
        }

    def _setup_indicators(self):
        """インジケーターのセットアップ（シンプル版）"""
        try:
            # HyperFRAMAChannel の設定（全パラメータ版）
            self.hyper_frama_channel = HyperFRAMAChannel(
                period=self.channel_period,
                multiplier_mode=self.channel_multiplier_mode,
                fixed_multiplier=self.channel_fixed_multiplier,
                src_type=self.channel_src_type,
                
                # HyperFRAMAパラメータ
                hyper_frama_period=self.channel_hyper_frama_period,
                hyper_frama_src_type=self.channel_hyper_frama_src_type,
                hyper_frama_fc=self.channel_hyper_frama_fc,
                hyper_frama_sc=self.channel_hyper_frama_sc,
                hyper_frama_alpha_multiplier=self.channel_hyper_frama_alpha_multiplier,
                hyper_frama_period_mode=self.channel_hyper_frama_period_mode,
                hyper_frama_cycle_detector_type=self.channel_hyper_frama_cycle_detector_type,
                hyper_frama_lp_period=self.channel_hyper_frama_lp_period,
                hyper_frama_hp_period=self.channel_hyper_frama_hp_period,
                hyper_frama_cycle_part=self.channel_hyper_frama_cycle_part,
                hyper_frama_max_cycle=self.channel_hyper_frama_max_cycle,
                hyper_frama_min_cycle=self.channel_hyper_frama_min_cycle,
                hyper_frama_max_output=self.channel_hyper_frama_max_output,
                hyper_frama_min_output=self.channel_hyper_frama_min_output,
                hyper_frama_enable_indicator_adaptation=self.channel_hyper_frama_enable_indicator_adaptation,
                hyper_frama_adaptation_indicator=self.channel_hyper_frama_adaptation_indicator,
                hyper_frama_hyper_er_period=self.channel_hyper_frama_hyper_er_period,
                hyper_frama_hyper_er_midline_period=self.channel_hyper_frama_hyper_er_midline_period,
                hyper_frama_hyper_adx_period=self.channel_hyper_frama_hyper_adx_period,
                hyper_frama_hyper_adx_midline_period=self.channel_hyper_frama_hyper_adx_midline_period,
                hyper_frama_hyper_trend_index_period=self.channel_hyper_frama_hyper_trend_index_period,
                hyper_frama_hyper_trend_index_midline_period=self.channel_hyper_frama_hyper_trend_index_midline_period,
                hyper_frama_fc_min=self.channel_hyper_frama_fc_min,
                hyper_frama_fc_max=self.channel_hyper_frama_fc_max,
                hyper_frama_sc_min=self.channel_hyper_frama_sc_min,
                hyper_frama_sc_max=self.channel_hyper_frama_sc_max,
                hyper_frama_period_min=self.channel_hyper_frama_period_min,
                hyper_frama_period_max=self.channel_hyper_frama_period_max,
                
                # X_ATRパラメータ
                x_atr_period=self.channel_x_atr_period,
                x_atr_tr_method=self.channel_x_atr_tr_method,
                x_atr_smoother_type=self.channel_x_atr_smoother_type,
                x_atr_src_type=self.channel_x_atr_src_type,
                x_atr_enable_kalman=self.channel_x_atr_enable_kalman,
                x_atr_kalman_type=self.channel_x_atr_kalman_type,
                x_atr_period_mode=self.channel_x_atr_period_mode,
                x_atr_cycle_detector_type=self.channel_x_atr_cycle_detector_type,
                x_atr_cycle_detector_cycle_part=self.channel_x_atr_cycle_detector_cycle_part,
                x_atr_cycle_detector_max_cycle=self.channel_x_atr_cycle_detector_max_cycle,
                x_atr_cycle_detector_min_cycle=self.channel_x_atr_cycle_detector_min_cycle,
                x_atr_cycle_period_multiplier=self.channel_x_atr_cycle_period_multiplier,
                x_atr_cycle_detector_period_range=self.channel_x_atr_cycle_detector_period_range,
                x_atr_midline_period=self.channel_x_atr_midline_period,
                x_atr_enable_percentile_analysis=self.channel_x_atr_enable_percentile_analysis,
                x_atr_percentile_lookback_period=self.channel_x_atr_percentile_lookback_period,
                x_atr_percentile_low_threshold=self.channel_x_atr_percentile_low_threshold,
                x_atr_percentile_high_threshold=self.channel_x_atr_percentile_high_threshold,
                x_atr_smoother_params=self.channel_x_atr_smoother_params,
                x_atr_kalman_params=self.channel_x_atr_kalman_params,
                
                # HyperERパラメータ
                hyper_er_period=self.channel_hyper_er_period,
                hyper_er_midline_period=self.channel_hyper_er_midline_period,
                hyper_er_er_period=self.channel_hyper_er_er_period,
                hyper_er_er_src_type=self.channel_hyper_er_er_src_type,
                hyper_er_use_kalman_filter=self.channel_hyper_er_use_kalman_filter,
                hyper_er_kalman_filter_type=self.channel_hyper_er_kalman_filter_type,
                hyper_er_kalman_process_noise=self.channel_hyper_er_kalman_process_noise,
                hyper_er_kalman_min_observation_noise=self.channel_hyper_er_kalman_min_observation_noise,
                hyper_er_kalman_adaptation_window=self.channel_hyper_er_kalman_adaptation_window,
                hyper_er_use_roofing_filter=self.channel_hyper_er_use_roofing_filter,
                hyper_er_roofing_hp_cutoff=self.channel_hyper_er_roofing_hp_cutoff,
                hyper_er_roofing_ss_band_edge=self.channel_hyper_er_roofing_ss_band_edge,
                hyper_er_use_laguerre_filter=self.channel_hyper_er_use_laguerre_filter,
                hyper_er_laguerre_gamma=self.channel_hyper_er_laguerre_gamma,
                hyper_er_use_smoothing=self.channel_hyper_er_use_smoothing,
                hyper_er_smoother_type=self.channel_hyper_er_smoother_type,
                hyper_er_smoother_period=self.channel_hyper_er_smoother_period,
                hyper_er_smoother_src_type=self.channel_hyper_er_smoother_src_type,
                hyper_er_use_dynamic_period=self.channel_hyper_er_use_dynamic_period,
                hyper_er_detector_type=self.channel_hyper_er_detector_type,
                hyper_er_lp_period=self.channel_hyper_er_lp_period,
                hyper_er_hp_period=self.channel_hyper_er_hp_period,
                hyper_er_cycle_part=self.channel_hyper_er_cycle_part,
                hyper_er_max_cycle=self.channel_hyper_er_max_cycle,
                hyper_er_min_cycle=self.channel_hyper_er_min_cycle,
                hyper_er_max_output=self.channel_hyper_er_max_output,
                hyper_er_min_output=self.channel_hyper_er_min_output,
                hyper_er_enable_percentile_analysis=self.channel_hyper_er_enable_percentile_analysis,
                hyper_er_percentile_lookback_period=self.channel_hyper_er_percentile_lookback_period,
                hyper_er_percentile_low_threshold=self.channel_hyper_er_percentile_low_threshold,
                hyper_er_percentile_high_threshold=self.channel_hyper_er_percentile_high_threshold,
                
                # HyperFRAMAチャネル独自パラメータ
                enable_signals=self.channel_enable_signals,
                enable_percentile=self.channel_enable_percentile,
                percentile_period=self.channel_percentile_period
            )
            
        except Exception as e:
            raise RuntimeError(f"シンプルシグナルの初期化に失敗しました: {e}")

    def _calculate_channel_indicators(self, data: Union[pd.DataFrame, np.ndarray]) -> tuple:
        """チャネルインジケーターの計算"""
        try:
            # チャネル計算
            channel_result = self.hyper_frama_channel.calculate(data)
            
            return (
                channel_result.midline,
                channel_result.upper_band,
                channel_result.lower_band
            )
            
        except Exception as e:
            n_points = len(data) if hasattr(data, '__len__') else 100
            return tuple(np.full(n_points, np.nan) for _ in range(3))

    def generate_entry(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エントリーシグナル生成"""
        try:
            # インジケーター計算
            midline, upper_band, lower_band = self._calculate_channel_indicators(data)
            
            # 価格データ取得
            if isinstance(data, pd.DataFrame):
                close = data['close'].values
            else:
                close = data[:, 3] if data.shape[1] > 3 else data[:, -1]
            
            # エントリーシグナル計算（Numba最適化）
            entry_signals = calculate_channel_breakout_entry_signals(
                close, upper_band, lower_band, self.band_lookback
            )
            
            return entry_signals
            
        except Exception as e:
            n_points = len(data) if hasattr(data, '__len__') else 100
            return np.zeros(n_points, dtype=np.int8)

    def generate_exit(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エグジットシグナル生成"""
        try:
            # インジケーター計算
            midline, upper_band, lower_band = self._calculate_channel_indicators(data)
            
            # 価格データ取得
            if isinstance(data, pd.DataFrame):
                close = data['close'].values
            else:
                close = data[:, 3] if data.shape[1] > 3 else data[:, -1]
            
            # エグジットシグナル計算（Numba最適化）
            exit_signals = calculate_channel_breakout_exit_signals(
                close, upper_band, lower_band, self.exit_mode
            )
            
            return exit_signals
            
        except Exception as e:
            n_points = len(data) if hasattr(data, '__len__') else 100
            return np.zeros(n_points, dtype=np.int8)

    # IEntrySignal インターフェース実装
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エントリーシグナル生成（IEntrySignal用）"""
        return self.generate_entry(data)

    # 追加メソッド（付加情報取得用）
    def get_channel_values(self, data: Union[pd.DataFrame, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """チャネル値を取得"""
        try:
            channel_result = self.hyper_frama_channel.calculate(data)
            return channel_result.midline, channel_result.upper_band, channel_result.lower_band
        except Exception as e:
            n_points = len(data) if hasattr(data, '__len__') else 100
            return tuple(np.full(n_points, np.nan) for _ in range(3))

    def get_source_price(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """ソース価格を取得"""
        try:
            if isinstance(data, pd.DataFrame):
                data_array = data.values
            else:
                data_array = data
            return PriceSource.calculate_source(data_array, self.src_type)
        except Exception as e:
            n_points = len(data) if hasattr(data, '__len__') else 100
            return np.full(n_points, np.nan)

    @property
    def name(self) -> str:
        """シグナル名を取得"""
        exit_mode_str = "逆ブレイクアウト" if self.exit_mode == 1 else "中心線クロス"
        return f"HyperFRAMAChannelBreakoutSignal(channel={self.channel_multiplier_mode}, exit={exit_mode_str}, lookback={self.band_lookback})"