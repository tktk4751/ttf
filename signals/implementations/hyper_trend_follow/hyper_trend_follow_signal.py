#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hyperトレンドフォローシグナル

HyperFRAMA位置関係シグナルとHyperFRAMAChannelブレイクアウトシグナルを組み合わせたシグナル

エントリー条件:
- ロング: 位置関係シグナル=1 かつ ブレイクアウトシグナル=1
- ショート: 位置関係シグナル=-1 かつ ブレイクアウトシグナル=-1

エグジット条件:
- ロングエグジット: ブレイクアウトシグナル=-1
- ショートエグジット: ブレイクアウトシグナル=1
"""

from typing import Union, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from ...interfaces.exit import IExitSignal
from signals.implementations.hyper_frama.entry import HyperFRAMAPositionEntrySignal
from signals.implementations.hyper_frama_channel.break_out_signal import HyperFRAMAChannelBreakoutSignal


@njit(fastmath=True, parallel=True)
def calculate_trend_follow_entry_signals(
    position_signals: np.ndarray,
    breakout_signals: np.ndarray
) -> np.ndarray:
    """
    トレンドフォローエントリーシグナルを計算する（高速化版）
    
    Args:
        position_signals: HyperFRAMA位置関係シグナル配列
        breakout_signals: HyperFRAMAチャネルブレイクアウトシグナル配列
    
    Returns:
        シグナルの配列（1: ロング, -1: ショート, 0: シグナルなし）
    """
    length = len(position_signals)
    signals = np.zeros(length, dtype=np.int8)
    
    # トレンドフォロー条件の判定
    for i in prange(length):
        # ロングエントリー: 位置関係シグナル=1 かつ ブレイクアウトシグナル=1
        if position_signals[i] == 1 and breakout_signals[i] == 1:
            signals[i] = 1
        # ショートエントリー: 位置関係シグナル=-1 かつ ブレイクアウトシグナル=-1
        elif position_signals[i] == -1 and breakout_signals[i] == -1:
            signals[i] = -1
    
    return signals


@njit(fastmath=True, parallel=True)
def calculate_trend_follow_exit_signals(
    breakout_signals: np.ndarray
) -> np.ndarray:
    """
    トレンドフォローエグジットシグナルを計算する（高速化版）
    
    Args:
        breakout_signals: HyperFRAMAチャネルブレイクアウトシグナル配列
    
    Returns:
        エグジットシグナルの配列（1: ロングエグジット, -1: ショートエグジット, 0: エグジットなし）
    """
    length = len(breakout_signals)
    signals = np.zeros(length, dtype=np.int8)
    
    # エグジット条件の判定
    for i in prange(length):
        # ロングエグジット: ブレイクアウトシグナル=-1
        if breakout_signals[i] == -1:
            signals[i] = 1
        # ショートエグジット: ブレイクアウトシグナル=1
        elif breakout_signals[i] == 1:
            signals[i] = -1
    
    return signals


class HyperTrendFollowSignal(BaseSignal, IEntrySignal, IExitSignal):
    """
    Hyperトレンドフォローシグナル
    
    特徴:
    - HyperFRAMA位置関係シグナルとHyperFRAMAChannelブレイクアウトシグナルの組み合わせ
    - トレンド方向確認後のブレイクアウトでエントリー
    - ブレイクアウト逆転でエグジット
    
    エントリー条件:
    - ロング: HyperFRAMA位置関係シグナル=1 かつ チャネルブレイクアウトシグナル=1
    - ショート: HyperFRAMA位置関係シグナル=-1 かつ チャネルブレイクアウトシグナル=-1
    
    エグジット条件:
    - ロングエグジット: チャネルブレイクアウトシグナル=-1（逆方向ブレイクアウト）
    - ショートエグジット: チャネルブレイクアウトシグナル=1（逆方向ブレイクアウト）
    """
    
    def __init__(
        self,
        # === HyperFRAMA位置関係シグナルパラメータ ===
        # HyperFRAMAパラメータ
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
        # 動的適応パラメータ（チャネル用）
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
        
        # X_ATR パラメータ（チャネル用）
        channel_x_atr_period: float = 12.0,
        channel_x_atr_tr_method: str = 'atr',
        channel_x_atr_smoother_type: str = 'frama',
        channel_x_atr_src_type: str = 'close',
        channel_x_atr_enable_kalman: bool = False,
        channel_x_atr_kalman_type: str = 'unscented',
        # 動的適応パラメータ（X_ATR用）
        channel_x_atr_period_mode: str = 'dynamic',
        channel_x_atr_cycle_detector_type: str = 'practical',
        channel_x_atr_cycle_detector_cycle_part: float = 0.5,
        channel_x_atr_cycle_detector_max_cycle: int = 55,
        channel_x_atr_cycle_detector_min_cycle: int = 5,
        channel_x_atr_cycle_period_multiplier: float = 1.0,
        channel_x_atr_cycle_detector_period_range: tuple = (5, 120),
        # ミッドラインパラメータ（X_ATR用）
        channel_x_atr_midline_period: int = 100,
        # パーセンタイルベースボラティリティ分析パラメータ（X_ATR用）
        channel_x_atr_enable_percentile_analysis: bool = True,
        channel_x_atr_percentile_lookback_period: int = 50,
        channel_x_atr_percentile_low_threshold: float = 0.25,
        channel_x_atr_percentile_high_threshold: float = 0.75,
        # スムーサーパラメータ（X_ATR用）
        channel_x_atr_smoother_params: dict = None,
        # カルマンフィルターパラメータ（X_ATR用）
        channel_x_atr_kalman_params: dict = None,
        
        # HyperER パラメータ（チャネル用）
        channel_hyper_er_period: int = 8,
        channel_hyper_er_midline_period: int = 100,
        # ERパラメータ（チャネル用）
        channel_hyper_er_er_period: int = 13,
        channel_hyper_er_er_src_type: str = 'oc2',
        # 統合カルマンフィルターパラメータ（チャネル用）
        channel_hyper_er_use_kalman_filter: bool = True,
        channel_hyper_er_kalman_filter_type: str = 'simple',
        channel_hyper_er_kalman_process_noise: float = 1e-5,
        channel_hyper_er_kalman_min_observation_noise: float = 1e-6,
        channel_hyper_er_kalman_adaptation_window: int = 5,
        # ルーフィングフィルターパラメータ（チャネル用）
        channel_hyper_er_use_roofing_filter: bool = True,
        channel_hyper_er_roofing_hp_cutoff: float = 55.0,
        channel_hyper_er_roofing_ss_band_edge: float = 10.0,
        # ラゲールフィルターパラメータ（チャネル用、後方互換性のため残す）
        channel_hyper_er_use_laguerre_filter: bool = False,
        channel_hyper_er_laguerre_gamma: float = 0.5,
        # 平滑化オプション（チャネル用）
        channel_hyper_er_use_smoothing: bool = True,
        channel_hyper_er_smoother_type: str = 'frama',
        channel_hyper_er_smoother_period: int = 16,
        channel_hyper_er_smoother_src_type: str = 'close',
        # エラーズ統合サイクル検出器パラメータ（チャネル用）
        channel_hyper_er_use_dynamic_period: bool = True,
        channel_hyper_er_detector_type: str = 'dft_dominant',
        channel_hyper_er_lp_period: int = 13,
        channel_hyper_er_hp_period: int = 124,
        channel_hyper_er_cycle_part: float = 0.4,
        channel_hyper_er_max_cycle: int = 124,
        channel_hyper_er_min_cycle: int = 13,
        channel_hyper_er_max_output: int = 89,
        channel_hyper_er_min_output: int = 5,
        # パーセンタイルベーストレンド分析パラメータ（チャネル用）
        channel_hyper_er_enable_percentile_analysis: bool = True,
        channel_hyper_er_percentile_lookback_period: int = 50,
        channel_hyper_er_percentile_low_threshold: float = 0.25,
        channel_hyper_er_percentile_high_threshold: float = 0.75,
        
        # HyperFRAMAチャネル独自パラメータ
        channel_enable_signals: bool = True,
        channel_enable_percentile: bool = True,
        channel_percentile_period: int = 100
    ):
        """
        初期化
        
        Args:
            hyper_frama_*: HyperFRAMA位置関係シグナル用パラメータ
            channel_*: HyperFRAMAChannelブレイクアウトシグナル用パラメータ
        """
        name = f"HyperTrendFollowSignal(hyper_frama_period={hyper_frama_period}, channel_period={channel_period})"
        
        # パラメータの保存
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
            
            # チャネルブレイクアウトシグナルパラメータ（基本分のみ記録）
            'channel_band_lookback': channel_band_lookback,
            'channel_exit_mode': channel_exit_mode,
            'channel_src_type': channel_src_type,
            'channel_period': channel_period,
            'channel_multiplier_mode': channel_multiplier_mode,
            'channel_fixed_multiplier': channel_fixed_multiplier
        }
        
        # HyperFRAMA位置関係シグナルの初期化
        self.position_signal = HyperFRAMAPositionEntrySignal(
            period=hyper_frama_period,
            src_type=hyper_frama_src_type,
            fc=hyper_frama_fc,
            sc=hyper_frama_sc,
            alpha_multiplier=hyper_frama_alpha_multiplier,
            period_mode=hyper_frama_period_mode,
            cycle_detector_type=hyper_frama_cycle_detector_type,
            lp_period=hyper_frama_lp_period,
            hp_period=hyper_frama_hp_period,
            cycle_part=hyper_frama_cycle_part,
            max_cycle=hyper_frama_max_cycle,
            min_cycle=hyper_frama_min_cycle,
            max_output=hyper_frama_max_output,
            min_output=hyper_frama_min_output
        )
        
        # HyperFRAMAChannelブレイクアウトシグナルの初期化
        self.breakout_signal = HyperFRAMAChannelBreakoutSignal(
            band_lookback=channel_band_lookback,
            exit_mode=channel_exit_mode,
            src_type=channel_src_type,
            
            # HyperFRAMAChannel 基本パラメータ
            channel_period=channel_period,
            channel_multiplier_mode=channel_multiplier_mode,
            channel_fixed_multiplier=channel_fixed_multiplier,
            channel_src_type=channel_hyper_frama_channel_src_type,
            
            # HyperFRAMA パラメータ（チャネル用）
            channel_hyper_frama_period=channel_hyper_frama_period,
            channel_hyper_frama_src_type=channel_hyper_frama_src_type,
            channel_hyper_frama_fc=channel_hyper_frama_fc,
            channel_hyper_frama_sc=channel_hyper_frama_sc,
            channel_hyper_frama_alpha_multiplier=channel_hyper_frama_alpha_multiplier,
            channel_hyper_frama_period_mode=channel_hyper_frama_period_mode,
            channel_hyper_frama_cycle_detector_type=channel_hyper_frama_cycle_detector_type,
            channel_hyper_frama_lp_period=channel_hyper_frama_lp_period,
            channel_hyper_frama_hp_period=channel_hyper_frama_hp_period,
            channel_hyper_frama_cycle_part=channel_hyper_frama_cycle_part,
            channel_hyper_frama_max_cycle=channel_hyper_frama_max_cycle,
            channel_hyper_frama_min_cycle=channel_hyper_frama_min_cycle,
            channel_hyper_frama_max_output=channel_hyper_frama_max_output,
            channel_hyper_frama_min_output=channel_hyper_frama_min_output,
            channel_hyper_frama_enable_indicator_adaptation=channel_hyper_frama_enable_indicator_adaptation,
            channel_hyper_frama_adaptation_indicator=channel_hyper_frama_adaptation_indicator,
            channel_hyper_frama_hyper_er_period=channel_hyper_frama_hyper_er_period,
            channel_hyper_frama_hyper_er_midline_period=channel_hyper_frama_hyper_er_midline_period,
            channel_hyper_frama_hyper_adx_period=channel_hyper_frama_hyper_adx_period,
            channel_hyper_frama_hyper_adx_midline_period=channel_hyper_frama_hyper_adx_midline_period,
            channel_hyper_frama_hyper_trend_index_period=channel_hyper_frama_hyper_trend_index_period,
            channel_hyper_frama_hyper_trend_index_midline_period=channel_hyper_frama_hyper_trend_index_midline_period,
            channel_hyper_frama_fc_min=channel_hyper_frama_fc_min,
            channel_hyper_frama_fc_max=channel_hyper_frama_fc_max,
            channel_hyper_frama_sc_min=channel_hyper_frama_sc_min,
            channel_hyper_frama_sc_max=channel_hyper_frama_sc_max,
            channel_hyper_frama_period_min=channel_hyper_frama_period_min,
            channel_hyper_frama_period_max=channel_hyper_frama_period_max,
            
            # X_ATR パラメータ（チャネル用）
            channel_x_atr_period=channel_x_atr_period,
            channel_x_atr_tr_method=channel_x_atr_tr_method,
            channel_x_atr_smoother_type=channel_x_atr_smoother_type,
            channel_x_atr_src_type=channel_x_atr_src_type,
            channel_x_atr_enable_kalman=channel_x_atr_enable_kalman,
            channel_x_atr_kalman_type=channel_x_atr_kalman_type,
            channel_x_atr_period_mode=channel_x_atr_period_mode,
            channel_x_atr_cycle_detector_type=channel_x_atr_cycle_detector_type,
            channel_x_atr_cycle_detector_cycle_part=channel_x_atr_cycle_detector_cycle_part,
            channel_x_atr_cycle_detector_max_cycle=channel_x_atr_cycle_detector_max_cycle,
            channel_x_atr_cycle_detector_min_cycle=channel_x_atr_cycle_detector_min_cycle,
            channel_x_atr_cycle_period_multiplier=channel_x_atr_cycle_period_multiplier,
            channel_x_atr_cycle_detector_period_range=channel_x_atr_cycle_detector_period_range,
            channel_x_atr_midline_period=channel_x_atr_midline_period,
            channel_x_atr_enable_percentile_analysis=channel_x_atr_enable_percentile_analysis,
            channel_x_atr_percentile_lookback_period=channel_x_atr_percentile_lookback_period,
            channel_x_atr_percentile_low_threshold=channel_x_atr_percentile_low_threshold,
            channel_x_atr_percentile_high_threshold=channel_x_atr_percentile_high_threshold,
            channel_x_atr_smoother_params=channel_x_atr_smoother_params,
            channel_x_atr_kalman_params=channel_x_atr_kalman_params,
            
            # HyperER パラメータ（チャネル用）
            channel_hyper_er_period=channel_hyper_er_period,
            channel_hyper_er_midline_period=channel_hyper_er_midline_period,
            channel_hyper_er_er_period=channel_hyper_er_er_period,
            channel_hyper_er_er_src_type=channel_hyper_er_er_src_type,
            channel_hyper_er_use_kalman_filter=channel_hyper_er_use_kalman_filter,
            channel_hyper_er_kalman_filter_type=channel_hyper_er_kalman_filter_type,
            channel_hyper_er_kalman_process_noise=channel_hyper_er_kalman_process_noise,
            channel_hyper_er_kalman_min_observation_noise=channel_hyper_er_kalman_min_observation_noise,
            channel_hyper_er_kalman_adaptation_window=channel_hyper_er_kalman_adaptation_window,
            channel_hyper_er_use_roofing_filter=channel_hyper_er_use_roofing_filter,
            channel_hyper_er_roofing_hp_cutoff=channel_hyper_er_roofing_hp_cutoff,
            channel_hyper_er_roofing_ss_band_edge=channel_hyper_er_roofing_ss_band_edge,
            channel_hyper_er_use_laguerre_filter=channel_hyper_er_use_laguerre_filter,
            channel_hyper_er_laguerre_gamma=channel_hyper_er_laguerre_gamma,
            channel_hyper_er_use_smoothing=channel_hyper_er_use_smoothing,
            channel_hyper_er_smoother_type=channel_hyper_er_smoother_type,
            channel_hyper_er_smoother_period=channel_hyper_er_smoother_period,
            channel_hyper_er_smoother_src_type=channel_hyper_er_smoother_src_type,
            channel_hyper_er_use_dynamic_period=channel_hyper_er_use_dynamic_period,
            channel_hyper_er_detector_type=channel_hyper_er_detector_type,
            channel_hyper_er_lp_period=channel_hyper_er_lp_period,
            channel_hyper_er_hp_period=channel_hyper_er_hp_period,
            channel_hyper_er_cycle_part=channel_hyper_er_cycle_part,
            channel_hyper_er_max_cycle=channel_hyper_er_max_cycle,
            channel_hyper_er_min_cycle=channel_hyper_er_min_cycle,
            channel_hyper_er_max_output=channel_hyper_er_max_output,
            channel_hyper_er_min_output=channel_hyper_er_min_output,
            channel_hyper_er_enable_percentile_analysis=channel_hyper_er_enable_percentile_analysis,
            channel_hyper_er_percentile_lookback_period=channel_hyper_er_percentile_lookback_period,
            channel_hyper_er_percentile_low_threshold=channel_hyper_er_percentile_low_threshold,
            channel_hyper_er_percentile_high_threshold=channel_hyper_er_percentile_high_threshold,
            
            # HyperFRAMAチャネル独自パラメータ
            channel_enable_signals=channel_enable_signals,
            channel_enable_percentile=channel_enable_percentile,
            channel_percentile_period=channel_percentile_period
        )
        
        # BaseSignalの初期化をスキップ（nameプロパティが競合するため）
        
        # キャッシュの初期化
        self._signals_cache = {}
    
    def _get_data_hash(self, ohlcv_data):
        """
        データハッシュを取得する
        
        Args:
            ohlcv_data: OHLCVデータ
            
        Returns:
            データのハッシュ値
        """
        # DataFrameの場合はNumpy配列に変換
        if isinstance(ohlcv_data, pd.DataFrame):
            # 必要なカラムがあれば抽出、なければそのまま変換
            if all(col in ohlcv_data.columns for col in ['open', 'high', 'low', 'close']):
                ohlcv_array = ohlcv_data[['open', 'high', 'low', 'close']].values
            else:
                ohlcv_array = ohlcv_data.values
        else:
            ohlcv_array = ohlcv_data
            
        # Numpy配列でない場合はエラー
        if not isinstance(ohlcv_array, np.ndarray):
            raise TypeError("ohlcv_data must be a numpy array or pandas DataFrame")
        
        # 配列のハッシュと設定パラメータのハッシュを組み合わせる
        return hash((ohlcv_array.tobytes(), *sorted(self._params.items())))
    
    def generate_entry(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            cache_key = f"entry_{data_hash}"
            if cache_key in self._signals_cache:
                return self._signals_cache[cache_key]
            
            # HyperFRAMA位置関係シグナルの計算
            position_signals = self.position_signal.generate(data)
            
            # HyperFRAMAChannelブレイクアウトシグナルの計算
            breakout_signals = self.breakout_signal.generate_entry(data)
            
            # トレンドフォローエントリーシグナルの計算（高速化版）
            entry_signals = calculate_trend_follow_entry_signals(
                position_signals,
                breakout_signals
            )
            
            # 結果をキャッシュ
            self._signals_cache[cache_key] = entry_signals
            return entry_signals
            
        except Exception as e:
            # エラーが発生した場合は警告を出力し、ゼロシグナルを返す
            print(f"HyperTrendFollowSignal エントリー計算中にエラー: {str(e)}")
            return np.zeros(len(data), dtype=np.int8)
    
    def generate_exit(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エグジットシグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            エグジットシグナルの配列 (1: ロングエグジット, -1: ショートエグジット, 0: エグジットなし)
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            cache_key = f"exit_{data_hash}"
            if cache_key in self._signals_cache:
                return self._signals_cache[cache_key]
            
            # HyperFRAMAChannelブレイクアウトシグナルの計算
            breakout_signals = self.breakout_signal.generate_entry(data)
            
            # トレンドフォローエグジットシグナルの計算（高速化版）
            exit_signals = calculate_trend_follow_exit_signals(breakout_signals)
            
            # 結果をキャッシュ
            self._signals_cache[cache_key] = exit_signals
            return exit_signals
            
        except Exception as e:
            # エラーが発生した場合は警告を出力し、ゼロシグナルを返す
            print(f"HyperTrendFollowSignal エグジット計算中にエラー: {str(e)}")
            return np.zeros(len(data), dtype=np.int8)
    
    # IEntrySignal インターフェース実装
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エントリーシグナル生成（IEntrySignal用）"""
        return self.generate_entry(data)
    
    # 追加メソッド（詳細情報取得用）
    def get_position_signals(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        HyperFRAMA位置関係シグナルを取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 位置関係シグナル値
        """
        if data is not None:
            return self.position_signal.generate(data)
        return np.array([])
    
    def get_breakout_signals(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        HyperFRAMAChannelブレイクアウトシグナルを取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: ブレイクアウトシグナル値
        """
        if data is not None:
            return self.breakout_signal.generate_entry(data)
        return np.array([])
    
    def get_frama_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        FRAMA値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: FRAMA値
        """
        return self.position_signal.get_frama_values(data)
    
    def get_adjusted_frama_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Adjusted FRAMA値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: Adjusted FRAMA値
        """
        return self.position_signal.get_adjusted_frama_values(data)
    
    def get_channel_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        HyperFRAMAChannelのバンド値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (中心線, 上限バンド, 下限バンド)のタプル
        """
        return self.breakout_signal.get_channel_values(data)
    
    def get_advanced_metrics(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        全ての高度なメトリクスを取得
        
        Returns:
            HyperFRAMAとチャネルの全メトリクス
        """
        try:
            if data is None:
                return {}
            
            metrics = {
                # HyperFRAMAメトリクス
                'frama_values': self.get_frama_values(data),
                'adjusted_frama_values': self.get_adjusted_frama_values(data),
                'position_signals': self.get_position_signals(data),
                'breakout_signals': self.get_breakout_signals(data),
                # トレンドフォローシグナル
                'entry_signals': self.generate_entry(data),
                'exit_signals': self.generate_exit(data)
            }
            
            # チャネル値を取得
            try:
                midline, upper_band, lower_band = self.get_channel_values(data)
                metrics.update({
                    'channel_midline': midline,
                    'channel_upper_band': upper_band,
                    'channel_lower_band': lower_band
                })
            except Exception:
                pass
            
            # フラクタル次元とアルファ値（利用可能な場合）
            try:
                metrics.update({
                    'fractal_dimension': self.position_signal.get_fractal_dimension(data),
                    'alpha_values': self.position_signal.get_alpha_values(data)
                })
            except Exception:
                pass
            
            return metrics
        except Exception as e:
            print(f"高度なメトリクス取得中にエラー: {str(e)}")
            return {}
    
    def reset(self) -> None:
        """シグナルの状態をリセットする"""
        super().reset()
        self.position_signal.reset() if hasattr(self.position_signal, 'reset') else None
        self.breakout_signal.reset() if hasattr(self.breakout_signal, 'reset') else None
        self._signals_cache = {}
    
    @property
    def name(self) -> str:
        """シグナル名を取得"""
        return f"HyperTrendFollowSignal(hyper_frama={self._params['hyper_frama_period']}, channel={self._params['channel_period']})"