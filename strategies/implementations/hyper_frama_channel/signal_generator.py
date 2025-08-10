#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
from numba import njit, prange
from enum import Enum

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.hyper_frama_channel.break_out_signal import HyperFRAMAChannelBreakoutSignal
from indicators.trend_filter.hyper_er import HyperER
from indicators.hyper_trend_index import HyperTrendIndex
from indicators.trend_filter.hyper_adx import HyperADX


class FilterType(Enum):
    """HyperFRAMAChannelストラテジー用のフィルタータイプ"""
    NONE = "none"
    HYPER_ER = "hyper_er"
    HYPER_TREND_INDEX = "hyper_trend_index" 
    HYPER_ADX = "hyper_adx"
    CONSENSUS = "consensus"  # 3つのうち2つが1の場合に1を出力


@njit(fastmath=True, parallel=False)
def combine_channel_signals_numba(channel_signals: np.ndarray, filter_signals: np.ndarray, filter_type: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    HyperFRAMAChannelシグナルとフィルターシグナルを統合する（Numba最適化版）
    
    Args:
        channel_signals: HyperFRAMAChannelシグナル配列
        filter_signals: フィルターシグナル配列
        filter_type: フィルタータイプ（0=None, 1=HyperER, 2=HyperTrendIndex, 3=HyperADX, 4=Consensus）
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (ロングシグナル, ショートシグナル)
    """
    length = len(channel_signals)
    long_signals = np.zeros(length, dtype=np.int8)
    short_signals = np.zeros(length, dtype=np.int8)
    
    for i in prange(length):
        channel_signal = channel_signals[i]
        
        # フィルターなしの場合
        if filter_type == 0:
            if channel_signal == 1:
                long_signals[i] = 1
            elif channel_signal == -1:
                short_signals[i] = 1
        else:
            # フィルターありの場合
            if i < len(filter_signals):
                filter_signal = filter_signals[i]
                
                # ロングエントリー条件: チャネルシグナル=1 かつ フィルターシグナル=1
                if channel_signal == 1 and filter_signal == 1:
                    long_signals[i] = 1
                
                # ショートエントリー条件: チャネルシグナル=-1 かつ フィルターシグナル=1 (ショートも同様に判定)
                elif channel_signal == -1 and filter_signal == 1:
                    short_signals[i] = 1
    
    return long_signals, short_signals


@njit(fastmath=True)
def consensus_filter_numba(
    hyper_er_signals: np.ndarray,
    trend_index_signals: np.ndarray,
    hyper_adx_signals: np.ndarray
) -> np.ndarray:
    """3つの指標のうち2つ以上が1の場合に1を出力"""
    n = len(hyper_er_signals)
    result = np.zeros(n)
    
    for i in range(n):
        count = 0
        if hyper_er_signals[i] == 1.0:
            count += 1
        if trend_index_signals[i] == 1.0:
            count += 1
        if hyper_adx_signals[i] == 1.0:
            count += 1
        
        # 2つ以上が1の場合に1を出力、それ以外は-1
        if count >= 2:
            result[i] = 1.0
        else:
            result[i] = -1.0
    
    return result


class HyperFRAMAChannelSignalGenerator(BaseSignalGenerator):
    """
    HyperFRAMAChannelのシグナル生成クラス（フィルタリング機能強化版）
    
    エントリー条件:
    - ロング: HyperFRAMAチャネル上方向ブレイクアウト かつ フィルターシグナル=1（フィルター有効時）
    - ショート: HyperFRAMAチャネル下方向ブレイクアウト かつ フィルターシグナル=1（フィルター有効時）
    
    エグジット条件:
    - モード1 (逆ブレイクアウト): 価格が逆側のバンドをブレイクしたらエグジット
    - モード2 (中心線クロス): 価格が中心線をクロスしたらエグジット
    
    フィルター機能:
    - HyperER: 効率性比率ベースの高精度トレンド判定
    - HyperTrendIndex: 高度なトレンドインデックスによる判定
    - HyperADX: 方向性移動インデックスによる判定
    - Consensus: 3つのフィルターの合意判定（3つのうち2つが1を出力）
    - None: フィルターなし
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
        channel_percentile_period: int = 100,
        
        # === フィルター設定 ===
        filter_type: FilterType = FilterType.NONE,  # フィルタータイプ
        # HyperER フィルターパラメータ
        filter_hyper_er_period: int = 14,
        filter_hyper_er_midline_period: int = 100,
        # HyperTrendIndex フィルターパラメータ
        filter_hyper_trend_index_period: int = 14,
        filter_hyper_trend_index_midline_period: int = 100,
        # HyperADX フィルターパラメータ
        filter_hyper_adx_period: int = 14,
        filter_hyper_adx_midline_period: int = 100
    ):
        """初期化"""
        filter_name = filter_type.value if isinstance(filter_type, FilterType) else str(filter_type)
        super().__init__(f"HyperFRAMAChannel_{filter_name}")
        
        # パラメータの設定（全パラメータ版）
        self._params = {
            # 基本パラメータ
            'band_lookback': band_lookback,
            'exit_mode': exit_mode,
            'src_type': src_type,
            
            # HyperFRAMAChannel 基本パラメータ
            'channel_period': channel_period,
            'channel_multiplier_mode': channel_multiplier_mode,
            'channel_fixed_multiplier': channel_fixed_multiplier,
            'channel_src_type': channel_src_type,
            
            # HyperFRAMAパラメータ
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
            'channel_hyper_frama_enable_indicator_adaptation': channel_hyper_frama_enable_indicator_adaptation,
            'channel_hyper_frama_adaptation_indicator': channel_hyper_frama_adaptation_indicator,
            'channel_hyper_frama_hyper_er_period': channel_hyper_frama_hyper_er_period,
            'channel_hyper_frama_hyper_er_midline_period': channel_hyper_frama_hyper_er_midline_period,
            'channel_hyper_frama_hyper_adx_period': channel_hyper_frama_hyper_adx_period,
            'channel_hyper_frama_hyper_adx_midline_period': channel_hyper_frama_hyper_adx_midline_period,
            'channel_hyper_frama_hyper_trend_index_period': channel_hyper_frama_hyper_trend_index_period,
            'channel_hyper_frama_hyper_trend_index_midline_period': channel_hyper_frama_hyper_trend_index_midline_period,
            'channel_hyper_frama_fc_min': channel_hyper_frama_fc_min,
            'channel_hyper_frama_fc_max': channel_hyper_frama_fc_max,
            'channel_hyper_frama_sc_min': channel_hyper_frama_sc_min,
            'channel_hyper_frama_sc_max': channel_hyper_frama_sc_max,
            'channel_hyper_frama_period_min': channel_hyper_frama_period_min,
            'channel_hyper_frama_period_max': channel_hyper_frama_period_max,
            
            # X_ATRパラメータ
            'channel_x_atr_period': channel_x_atr_period,
            'channel_x_atr_tr_method': channel_x_atr_tr_method,
            'channel_x_atr_smoother_type': channel_x_atr_smoother_type,
            'channel_x_atr_src_type': channel_x_atr_src_type,
            'channel_x_atr_enable_kalman': channel_x_atr_enable_kalman,
            'channel_x_atr_kalman_type': channel_x_atr_kalman_type,
            'channel_x_atr_period_mode': channel_x_atr_period_mode,
            'channel_x_atr_cycle_detector_type': channel_x_atr_cycle_detector_type,
            'channel_x_atr_cycle_detector_cycle_part': channel_x_atr_cycle_detector_cycle_part,
            'channel_x_atr_cycle_detector_max_cycle': channel_x_atr_cycle_detector_max_cycle,
            'channel_x_atr_cycle_detector_min_cycle': channel_x_atr_cycle_detector_min_cycle,
            'channel_x_atr_cycle_period_multiplier': channel_x_atr_cycle_period_multiplier,
            'channel_x_atr_cycle_detector_period_range': channel_x_atr_cycle_detector_period_range,
            'channel_x_atr_midline_period': channel_x_atr_midline_period,
            'channel_x_atr_enable_percentile_analysis': channel_x_atr_enable_percentile_analysis,
            'channel_x_atr_percentile_lookback_period': channel_x_atr_percentile_lookback_period,
            'channel_x_atr_percentile_low_threshold': channel_x_atr_percentile_low_threshold,
            'channel_x_atr_percentile_high_threshold': channel_x_atr_percentile_high_threshold,
            'channel_x_atr_smoother_params': channel_x_atr_smoother_params,
            'channel_x_atr_kalman_params': channel_x_atr_kalman_params,
            
            # HyperERパラメータ
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
            'channel_hyper_er_use_laguerre_filter': channel_hyper_er_use_laguerre_filter,
            'channel_hyper_er_laguerre_gamma': channel_hyper_er_laguerre_gamma,
            'channel_hyper_er_use_smoothing': channel_hyper_er_use_smoothing,
            'channel_hyper_er_smoother_type': channel_hyper_er_smoother_type,
            'channel_hyper_er_smoother_period': channel_hyper_er_smoother_period,
            'channel_hyper_er_smoother_src_type': channel_hyper_er_smoother_src_type,
            'channel_hyper_er_use_dynamic_period': channel_hyper_er_use_dynamic_period,
            'channel_hyper_er_detector_type': channel_hyper_er_detector_type,
            'channel_hyper_er_lp_period': channel_hyper_er_lp_period,
            'channel_hyper_er_hp_period': channel_hyper_er_hp_period,
            'channel_hyper_er_cycle_part': channel_hyper_er_cycle_part,
            'channel_hyper_er_max_cycle': channel_hyper_er_max_cycle,
            'channel_hyper_er_min_cycle': channel_hyper_er_min_cycle,
            'channel_hyper_er_max_output': channel_hyper_er_max_output,
            'channel_hyper_er_min_output': channel_hyper_er_min_output,
            'channel_hyper_er_enable_percentile_analysis': channel_hyper_er_enable_percentile_analysis,
            'channel_hyper_er_percentile_lookback_period': channel_hyper_er_percentile_lookback_period,
            'channel_hyper_er_percentile_low_threshold': channel_hyper_er_percentile_low_threshold,
            'channel_hyper_er_percentile_high_threshold': channel_hyper_er_percentile_high_threshold,
            
            # HyperFRAMAチャネル独自パラメータ
            'channel_enable_signals': channel_enable_signals,
            'channel_enable_percentile': channel_enable_percentile,
            'channel_percentile_period': channel_percentile_period,
            
            # フィルター設定
            'filter_type': filter_type,
            # HyperER フィルターパラメータ
            'filter_hyper_er_period': filter_hyper_er_period,
            'filter_hyper_er_midline_period': filter_hyper_er_midline_period,
            # HyperTrendIndex フィルターパラメータ
            'filter_hyper_trend_index_period': filter_hyper_trend_index_period,
            'filter_hyper_trend_index_midline_period': filter_hyper_trend_index_midline_period,
            # HyperADX フィルターパラメータ
            'filter_hyper_adx_period': filter_hyper_adx_period,
            'filter_hyper_adx_midline_period': filter_hyper_adx_midline_period
        }
        
        self.filter_type = filter_type if isinstance(filter_type, FilterType) else FilterType(filter_type)
        
        # HyperFRAMAChannelブレイクアウトシグナル用のパラメータを抽出（フィルターパラメータを除外）
        channel_params = {k: v for k, v in self._params.items() if not k.startswith('filter_')}
        
        # HyperFRAMAChannelブレイクアウトシグナルの初期化（全パラメータ版）
        self.hyper_frama_channel_signal = HyperFRAMAChannelBreakoutSignal(**channel_params)
        
        # フィルターインジケーター（必要に応じて初期化）
        self.hyper_er_filter = None
        self.hyper_trend_index_filter = None
        self.hyper_adx_filter = None
        
        if self.filter_type != FilterType.NONE:
            # フィルター用インジケーターを初期化
            self.hyper_er_filter = HyperER(
                period=filter_hyper_er_period,
                midline_period=filter_hyper_er_midline_period
            )
            
            self.hyper_trend_index_filter = HyperTrendIndex(
                period=filter_hyper_trend_index_period,
                midline_period=filter_hyper_trend_index_midline_period
            )
            
            self.hyper_adx_filter = HyperADX(
                period=filter_hyper_adx_period,
                midline_period=filter_hyper_adx_midline_period
            )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._entry_signals = None
        self._exit_signals = None
        self._channel_signals = None
        self._filter_signals = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算（高速化版）"""
        try:
            current_len = len(data)
            
            # データ長が変わった場合のみ再計算
            if self._entry_signals is None or current_len != self._data_len:
                # データフレームの作成（必要な列のみ）
                if isinstance(data, pd.DataFrame):
                    df = data[['open', 'high', 'low', 'close']]
                else:
                    df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
                
                # HyperFRAMAChannelシグナルの計算
                try:
                    channel_entry_signals = self.hyper_frama_channel_signal.generate_entry(df)
                    exit_signals = self.hyper_frama_channel_signal.generate_exit(df)
                    
                    self._channel_signals = channel_entry_signals
                    self._exit_signals = exit_signals
                    
                    # フィルターシグナルの計算
                    self._filter_signals = self._get_filter_signals(df)
                    
                    # シグナルの統合
                    filter_type_int = list(FilterType).index(self.filter_type)
                    long_signals, short_signals = combine_channel_signals_numba(
                        channel_entry_signals, self._filter_signals, filter_type_int
                    )
                    
                    # エントリーシグナルを統合（ロング=1、ショート=-1、なし=0）
                    self._entry_signals = long_signals - short_signals
                    
                except Exception as e:
                    self.logger.error(f"シグナル計算中にエラー: {str(e)}")
                    # エラー時はゼロシグナルを設定
                    self._entry_signals = np.zeros(current_len, dtype=np.int8)
                    self._exit_signals = np.zeros(current_len, dtype=np.int8)
                    self._channel_signals = np.zeros(current_len, dtype=np.int8)
                    self._filter_signals = np.zeros(current_len, dtype=np.int8)
                
                self._data_len = current_len
        except Exception as e:
            self.logger.error(f"calculate_signals全体でエラー: {str(e)}")
            # エラー時はゼロシグナルを設定
            if data is not None:
                self._entry_signals = np.zeros(len(data), dtype=np.int8)
                self._exit_signals = np.zeros(len(data), dtype=np.int8)
                self._channel_signals = np.zeros(len(data), dtype=np.int8)
                self._filter_signals = np.zeros(len(data), dtype=np.int8)
                self._data_len = len(data)
    
    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エントリーシグナル取得（高速化版）"""
        if self._entry_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._entry_signals
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """エグジットシグナル生成（高速化版）"""
        if self._exit_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        
        if index == -1:
            index = len(data) - 1
        
        # exit_modeに応じたシグナル処理
        if position == 1:  # ロングポジション
            return bool(self._exit_signals[index] == 1)  # ロングエグジットシグナル
        elif position == -1:  # ショートポジション
            return bool(self._exit_signals[index] == -1)  # ショートエグジットシグナル
        return False
    
    def get_channel_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        HyperFRAMAChannelのバンド値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (中心線, 上限バンド, 下限バンド)のタプル
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.hyper_frama_channel_signal.get_channel_values(data)
        except Exception as e:
            self.logger.error(f"チャネルバンド値取得中にエラー: {str(e)}")
            # エラー時は空の配列を返す
            empty = np.array([])
            return empty, empty, empty
    
    def get_source_price(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        ソース価格を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: ソース価格の値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.hyper_frama_channel_signal.get_source_price(data)
        except Exception as e:
            self.logger.error(f"ソース価格取得中にエラー: {str(e)}")
            return np.array([])
    
    def _get_filter_signals(self, data: pd.DataFrame) -> np.ndarray:
        """フィルターシグナルを取得"""
        if self.filter_type == FilterType.HYPER_ER:
            self.hyper_er_filter.calculate(data)
            return self.hyper_er_filter.get_trend_signal()
            
        elif self.filter_type == FilterType.HYPER_TREND_INDEX:
            self.hyper_trend_index_filter.calculate(data)
            return self.hyper_trend_index_filter.get_trend_signal()
            
        elif self.filter_type == FilterType.HYPER_ADX:
            self.hyper_adx_filter.calculate(data)
            return self.hyper_adx_filter.get_trend_signal()
            
        elif self.filter_type == FilterType.CONSENSUS:
            # 統合フィルター（3つのうち2つが1なら1）
            self.hyper_er_filter.calculate(data)
            hyper_er_signals = self.hyper_er_filter.get_trend_signal()
            
            self.hyper_trend_index_filter.calculate(data)
            trend_index_signals = self.hyper_trend_index_filter.get_trend_signal()
            
            self.hyper_adx_filter.calculate(data)
            hyper_adx_signals = self.hyper_adx_filter.get_trend_signal()
            
            return consensus_filter_numba(
                hyper_er_signals,
                trend_index_signals, 
                hyper_adx_signals
            )
        
        # デフォルト（フィルターなし）
        return np.ones(len(data))
    
    def get_channel_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """HyperFRAMAChannelシグナル取得"""
        if self._channel_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._channel_signals
    
    def get_filter_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """フィルターシグナル取得"""
        if self._filter_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._filter_signals
    
    def get_filter_details(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        フィルター詳細情報を取得
        
        Returns:
            フィルタータイプに応じた詳細データ
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            if self.filter_type == FilterType.NONE:
                return {}
            
            details = {}
            
            if self.filter_type == FilterType.HYPER_ER:
                details.update({
                    'hyper_er_values': self.hyper_er_filter.get_values() if hasattr(self.hyper_er_filter, 'get_values') else np.array([]),
                    'hyper_er_trend_signals': self.hyper_er_filter.get_trend_signal() if hasattr(self.hyper_er_filter, 'get_trend_signal') else np.array([])
                })
            elif self.filter_type == FilterType.HYPER_TREND_INDEX:
                details.update({
                    'hyper_trend_index_values': self.hyper_trend_index_filter.get_values() if hasattr(self.hyper_trend_index_filter, 'get_values') else np.array([]),
                    'hyper_trend_index_signals': self.hyper_trend_index_filter.get_trend_signal() if hasattr(self.hyper_trend_index_filter, 'get_trend_signal') else np.array([])
                })
            elif self.filter_type == FilterType.HYPER_ADX:
                details.update({
                    'hyper_adx_values': self.hyper_adx_filter.get_values() if hasattr(self.hyper_adx_filter, 'get_values') else np.array([]),
                    'hyper_adx_signals': self.hyper_adx_filter.get_trend_signal() if hasattr(self.hyper_adx_filter, 'get_trend_signal') else np.array([])
                })
            elif self.filter_type == FilterType.CONSENSUS:
                # 各インジケーターの個別シグナルも追加
                details.update({
                    'hyper_er_values': self.hyper_er_filter.get_values() if hasattr(self.hyper_er_filter, 'get_values') else np.array([]),
                    'hyper_er_signals': self.hyper_er_filter.get_trend_signal() if hasattr(self.hyper_er_filter, 'get_trend_signal') else np.array([]),
                    'hyper_trend_index_values': self.hyper_trend_index_filter.get_values() if hasattr(self.hyper_trend_index_filter, 'get_values') else np.array([]),
                    'hyper_trend_index_signals': self.hyper_trend_index_filter.get_trend_signal() if hasattr(self.hyper_trend_index_filter, 'get_trend_signal') else np.array([]),
                    'hyper_adx_values': self.hyper_adx_filter.get_values() if hasattr(self.hyper_adx_filter, 'get_values') else np.array([]),
                    'hyper_adx_signals': self.hyper_adx_filter.get_trend_signal() if hasattr(self.hyper_adx_filter, 'get_trend_signal') else np.array([])
                })
            
            return details
        except Exception as e:
            self.logger.error(f"フィルター詳細取得中にエラー: {str(e)}")
            return {}
    
    def get_advanced_metrics(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        全ての高度なメトリクスを取得
        
        Returns:
            HyperFRAMAChannelとフィルターの全メトリクス
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            metrics = {
                # HyperFRAMAChannelメトリクス
                'channel_signals': self.get_channel_signals(data),
                'entry_signals': self.get_entry_signals(data),
                'filter_signals': self.get_filter_signals(data),
                # フィルター詳細
                'filter_type': self.filter_type.value
            }
            
            # チャネル値を取得
            try:
                midline, upper_band, lower_band = self.get_channel_values(data)
                metrics.update({
                    'midline': midline,
                    'upper_band': upper_band,
                    'lower_band': lower_band
                })
            except Exception:
                pass
            
            # フィルター固有のメトリクスを追加
            filter_details = self.get_filter_details()
            metrics.update(filter_details)
            
            return metrics
        except Exception as e:
            self.logger.error(f"高度なメトリクス取得中にエラー: {str(e)}")
            return {}