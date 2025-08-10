#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple, Optional, List
import numpy as np
import pandas as pd
from numba import njit, prange
from enum import Enum

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.hyper_adaptive_channel.breakout_entry import HyperAdaptiveChannelBreakoutEntrySignal
from indicators.trend_filter.hyper_er import HyperER
from indicators.hyper_trend_index import HyperTrendIndex
from indicators.trend_filter.hyper_adx import HyperADX


class FilterType(Enum):
    """HyperAdaptiveChannelストラテジー用のフィルタータイプ"""
    NONE = "none"
    HYPER_ER = "hyper_er"
    HYPER_TREND_INDEX = "hyper_trend_index" 
    HYPER_ADX = "hyper_adx"
    CONSENSUS = "consensus"  # 3つのうち2つが1の場合に1を出力


@njit(fastmath=True, parallel=False)
def combine_signals_numba(hyper_adaptive_channel_signals: np.ndarray, filter_signals: np.ndarray, filter_type: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    HyperAdaptiveChannelシグナルとフィルターシグナルを統合する（Numba最適化版）
    
    Args:
        hyper_adaptive_channel_signals: HyperAdaptiveChannelシグナル配列
        filter_signals: フィルターシグナル配列
        filter_type: フィルタータイプ（0=None, 1=HyperER, 2=HyperTrendIndex, 3=HyperADX, 4=Consensus）
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (ロングシグナル, ショートシグナル)
    """
    length = len(hyper_adaptive_channel_signals)
    long_signals = np.zeros(length, dtype=np.int8)
    short_signals = np.zeros(length, dtype=np.int8)
    
    for i in prange(length):
        channel_signal = hyper_adaptive_channel_signals[i]
        
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
                
                # ショートエントリー条件: チャネルシグナル=-1 かつ フィルターシグナル=1
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


@njit(fastmath=True, parallel=False)
def check_exit_conditions_numba(hyper_adaptive_channel_signals: np.ndarray, position: int, index: int) -> bool:
    """
    エグジット条件をチェックする（Numba最適化版）
    
    Args:
        hyper_adaptive_channel_signals: HyperAdaptiveChannelシグナル配列
        position: 現在のポジション（1=ロング、-1=ショート）
        index: チェックするインデックス
    
    Returns:
        bool: エグジットすべきかどうか
    """
    if index < 0 or index >= len(hyper_adaptive_channel_signals):
        return False
    
    channel_signal = hyper_adaptive_channel_signals[index]
    
    # ロングポジション: チャネルシグナル=-1でエグジット
    if position == 1 and channel_signal == -1:
        return True
    
    # ショートポジション: チャネルシグナル=1でエグジット
    if position == -1 and channel_signal == 1:
        return True
    
    return False


class HyperAdaptiveChannelSignalGenerator(BaseSignalGenerator):
    """
    HyperAdaptiveChannelのシグナル生成クラス（両方向・高速化版）
    
    エントリー条件:
    - ロング: HyperAdaptiveChannelのブレイクアウトで買いシグナル
    - ショート: HyperAdaptiveChannelのブレイクアウトで売りシグナル
    
    エグジット条件:
    - ロング: HyperAdaptiveChannelの売りシグナル
    - ショート: HyperAdaptiveChannelの買いシグナル
    """
    
    def __init__(
        self,
        # 基本パラメータ
        band_lookback: int = 1,
        period: int = 14,
        midline_smoother: str = "hyper_frama",
        multiplier_mode: str = "dynamic",
        fixed_multiplier: float = 2.5,
        src_type: str = "hlc3",
        
        # === HyperFRAMA パラメータ ===
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
        # 動的適応パラメータ
        hyper_frama_enable_indicator_adaptation: bool = True,
        hyper_frama_adaptation_indicator: str = 'hyper_er',
        hyper_frama_hyper_er_period: int = 14,
        hyper_frama_hyper_er_midline_period: int = 100,
        hyper_frama_hyper_adx_period: int = 14,
        hyper_frama_hyper_adx_midline_period: int = 100,
        hyper_frama_hyper_trend_index_period: int = 14,
        hyper_frama_hyper_trend_index_midline_period: int = 100,
        hyper_frama_fc_min: float = 1.0,
        hyper_frama_fc_max: float = 8.0,
        hyper_frama_sc_min: float = 50.0,
        hyper_frama_sc_max: float = 250.0,
        hyper_frama_period_min: int = 4,
        hyper_frama_period_max: int = 88,
        
        # === UltimateMA パラメータ ===
        ultimate_ma_ultimate_smoother_period: float = 5.0,
        ultimate_ma_zero_lag_period: int = 21,
        ultimate_ma_realtime_window: int = 89,
        ultimate_ma_src_type: str = 'hlc3',
        ultimate_ma_slope_index: int = 1,
        ultimate_ma_range_threshold: float = 0.005,
        # 適応的カルマンフィルターパラメータ
        ultimate_ma_use_adaptive_kalman: bool = True,
        ultimate_ma_kalman_process_variance: float = 1e-5,
        ultimate_ma_kalman_measurement_variance: float = 0.01,
        ultimate_ma_kalman_volatility_window: int = 5,
        # 動的適応パラメータ
        ultimate_ma_zero_lag_period_mode: str = 'dynamic',
        ultimate_ma_realtime_window_mode: str = 'dynamic',
        # ゼロラグ用サイクル検出器パラメータ
        ultimate_ma_zl_cycle_detector_type: str = 'absolute_ultimate',
        ultimate_ma_zl_cycle_detector_cycle_part: float = 1.0,
        ultimate_ma_zl_cycle_detector_max_cycle: int = 120,
        ultimate_ma_zl_cycle_detector_min_cycle: int = 5,
        ultimate_ma_zl_cycle_period_multiplier: float = 1.0,
        # リアルタイムウィンドウ用サイクル検出器パラメータ
        ultimate_ma_rt_cycle_detector_type: str = 'absolute_ultimate',
        ultimate_ma_rt_cycle_detector_cycle_part: float = 0.5,
        ultimate_ma_rt_cycle_detector_max_cycle: int = 120,
        ultimate_ma_rt_cycle_detector_min_cycle: int = 5,
        ultimate_ma_rt_cycle_period_multiplier: float = 0.5,
        # period_rangeパラメータ
        ultimate_ma_zl_cycle_detector_period_range: Tuple[int, int] = (5, 120),
        ultimate_ma_rt_cycle_detector_period_range: Tuple[int, int] = (5, 120),
        
        # === LaguerreFilter パラメータ ===
        laguerre_gamma: float = 0.5,
        laguerre_order: int = 4,
        laguerre_coefficients: Optional[List[float]] = None,
        laguerre_src_type: str = 'close',
        laguerre_period: int = 4,
        laguerre_period_mode: str = 'fixed',
        laguerre_cycle_detector_type: str = 'hody_e',
        laguerre_cycle_part: float = 0.5,
        laguerre_max_cycle: int = 124,
        laguerre_min_cycle: int = 13,
        laguerre_max_output: int = 124,
        laguerre_min_output: int = 13,
        laguerre_lp_period: int = 13,
        laguerre_hp_period: int = 124,
        
        # === ZAdaptiveMA パラメータ ===
        z_adaptive_fast_period: int = 2,
        z_adaptive_slow_period: int = 120,
        z_adaptive_src_type: str = 'hlc3',
        z_adaptive_slope_index: int = 1,
        z_adaptive_range_threshold: float = 0.005,
        
        # === SuperSmoother パラメータ ===
        super_smoother_length: int = 15,
        super_smoother_num_poles: int = 2,
        super_smoother_src_type: str = 'cc2',
        # 動的期間パラメータ
        super_smoother_period_mode: str = 'fixed',
        super_smoother_cycle_detector_type: str = 'hody_e',
        super_smoother_lp_period: int = 13,
        super_smoother_hp_period: int = 124,
        super_smoother_cycle_part: float = 0.5,
        super_smoother_max_cycle: int = 124,
        super_smoother_min_cycle: int = 13,
        super_smoother_max_output: int = 124,
        super_smoother_min_output: int = 13,
        
        # === X_ATR パラメータ ===
        x_atr_period: float = 12.0,
        x_atr_tr_method: str = 'str',
        x_atr_smoother_type: str = 'frama',
        x_atr_src_type: str = 'close',
        x_atr_enable_kalman: bool = False,
        x_atr_kalman_type: str = 'unscented',
        # 動的適応パラメータ
        x_atr_period_mode: str = 'fixed',
        x_atr_cycle_detector_type: str = 'absolute_ultimate',
        x_atr_cycle_detector_cycle_part: float = 0.5,
        x_atr_cycle_detector_max_cycle: int = 55,
        x_atr_cycle_detector_min_cycle: int = 5,
        x_atr_cycle_period_multiplier: float = 1.0,
        x_atr_cycle_detector_period_range: Tuple[int, int] = (5, 120),
        # ミッドラインパラメータ
        x_atr_midline_period: int = 100,
        # パーセンタイルベースボラティリティ分析パラメータ
        x_atr_enable_percentile_analysis: bool = True,
        x_atr_percentile_lookback_period: int = 50,
        x_atr_percentile_low_threshold: float = 0.25,
        x_atr_percentile_high_threshold: float = 0.75,
        # スムーサーパラメータ
        x_atr_smoother_params: Optional[Dict[str, Any]] = None,
        # カルマンフィルターパラメータ
        x_atr_kalman_params: Optional[Dict[str, Any]] = None,
        
        # === HyperER パラメータ ===
        hyper_er_period: int = 8,
        hyper_er_midline_period: int = 100,
        # ERパラメータ
        hyper_er_er_period: int = 13,
        hyper_er_er_src_type: str = 'oc2',
        # 統合カルマンフィルターパラメータ
        hyper_er_use_kalman_filter: bool = True,
        hyper_er_kalman_filter_type: str = 'unscented',
        hyper_er_kalman_process_noise: float = 1e-5,
        hyper_er_kalman_min_observation_noise: float = 1e-6,
        hyper_er_kalman_adaptation_window: int = 5,
        # ルーフィングフィルターパラメータ
        hyper_er_use_roofing_filter: bool = True,
        hyper_er_roofing_hp_cutoff: float = 55.0,
        hyper_er_roofing_ss_band_edge: float = 10.0,
        # ラゲールフィルターパラメータ（後方互換性のため残す）
        hyper_er_use_laguerre_filter: bool = False,
        hyper_er_laguerre_gamma: float = 0.5,
        # 平滑化オプション
        hyper_er_use_smoothing: bool = True,
        hyper_er_smoother_type: str = 'laguerre',
        hyper_er_smoother_period: int = 12,
        hyper_er_smoother_src_type: str = 'close',
        # エラーズ統合サイクル検出器パラメータ
        hyper_er_use_dynamic_period: bool = False,
        hyper_er_detector_type: str = 'dft_dominant',
        hyper_er_lp_period: int = 13,
        hyper_er_hp_period: int = 124,
        hyper_er_cycle_part: float = 0.4,
        hyper_er_max_cycle: int = 124,
        hyper_er_min_cycle: int = 13,
        hyper_er_max_output: int = 89,
        hyper_er_min_output: int = 5,
        # パーセンタイルベーストレンド分析パラメータ
        hyper_er_enable_percentile_analysis: bool = True,
        hyper_er_percentile_lookback_period: int = 50,
        hyper_er_percentile_low_threshold: float = 0.25,
        hyper_er_percentile_high_threshold: float = 0.75,
        
        # === ハイパーアダプティブチャネル独自パラメータ ===
        enable_signals: bool = True,
        enable_percentile: bool = True,
        percentile_period: int = 100,
        
        # === フィルターパラメータ ===
        # フィルター選択
        filter_type: FilterType = FilterType.NONE,  # フィルタータイプ
        # HyperER フィルターパラメータ（追加）
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
        super().__init__(f"HyperAdaptiveChannelSignalGenerator_{filter_name}")
        
        # パラメータの設定（全ての引数をそのまま辞書にする）
        self._params = locals().copy()
        # 不要なキーを削除
        self._params.pop('self', None)
        self._params.pop('__class__', None)
        
        # フィルタータイプの設定
        self.filter_type = filter_type if isinstance(filter_type, FilterType) else FilterType(filter_type)
        
        # HyperAdaptiveChannelシグナル用のパラメータ（フィルターパラメータと内部変数を除外）
        channel_params = self._params.copy()
        filter_param_keys = [
            'filter_type', 'filter_hyper_er_period', 'filter_hyper_er_midline_period',
            'filter_hyper_trend_index_period', 'filter_hyper_trend_index_midline_period',
            'filter_hyper_adx_period', 'filter_hyper_adx_midline_period',
            'filter_name'  # 内部変数も除外
        ]
        for key in filter_param_keys:
            channel_params.pop(key, None)
        
        # HyperAdaptiveChannelブレイクアウトシグナルの初期化（フィルターパラメータ除外）
        self.hyper_adaptive_channel_signal = HyperAdaptiveChannelBreakoutEntrySignal(**channel_params)
        
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
        self._long_signals = None
        self._short_signals = None
        self._hyper_adaptive_channel_signals = None
        self._filter_signals = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算（フィルター統合版）"""
        try:
            current_len = len(data)
            
            # データ長が変わった場合のみ再計算
            if self._long_signals is None or current_len != self._data_len:
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
                    # HyperAdaptiveChannelシグナルの計算
                    hyper_adaptive_channel_signals = self.hyper_adaptive_channel_signal.generate(df)
                    self._hyper_adaptive_channel_signals = hyper_adaptive_channel_signals
                    
                    # フィルターシグナルの計算
                    self._filter_signals = self._get_filter_signals(df)
                    
                    # シグナルの統合
                    filter_type_int = list(FilterType).index(self.filter_type)
                    self._long_signals, self._short_signals = combine_signals_numba(
                        hyper_adaptive_channel_signals, self._filter_signals, filter_type_int
                    )
                    
                except Exception as e:
                    self.logger.error(f"シグナル計算中にエラー: {str(e)}")
                    # エラー時はゼロシグナルを設定
                    self._long_signals = np.zeros(current_len, dtype=np.int8)
                    self._short_signals = np.zeros(current_len, dtype=np.int8)
                    self._hyper_adaptive_channel_signals = np.zeros(current_len, dtype=np.int8)
                    self._filter_signals = np.zeros(current_len, dtype=np.int8)
                
                self._data_len = current_len
        except Exception as e:
            self.logger.error(f"calculate_signals全体でエラー: {str(e)}")
            # エラー時はゼロシグナルを設定
            if data is not None:
                self._long_signals = np.zeros(len(data), dtype=np.int8)
                self._short_signals = np.zeros(len(data), dtype=np.int8)
                self._hyper_adaptive_channel_signals = np.zeros(len(data), dtype=np.int8)
                self._filter_signals = np.zeros(len(data), dtype=np.int8)
                self._data_len = len(data)
    
    def _get_filter_signals(self, data: pd.DataFrame) -> np.ndarray:
        """フィルターシグナルを取得"""
        if self.filter_type == FilterType.HYPER_ER:
            hyper_er_result = self.hyper_er_filter.calculate(data)
            return self.hyper_er_filter.get_trend_signal()
            
        elif self.filter_type == FilterType.HYPER_TREND_INDEX:
            trend_index_result = self.hyper_trend_index_filter.calculate(data)
            return self.hyper_trend_index_filter.get_trend_signal()
            
        elif self.filter_type == FilterType.HYPER_ADX:
            hyper_adx_result = self.hyper_adx_filter.calculate(data)
            return self.hyper_adx_filter.get_trend_signal()
            
        elif self.filter_type == FilterType.CONSENSUS:
            # 統合フィルター（3つのうち2つが1なら1）
            hyper_er_result = self.hyper_er_filter.calculate(data)
            hyper_er_signals = self.hyper_er_filter.get_trend_signal()
            
            trend_index_result = self.hyper_trend_index_filter.calculate(data)
            trend_index_signals = self.hyper_trend_index_filter.get_trend_signal()
            
            hyper_adx_result = self.hyper_adx_filter.calculate(data)
            hyper_adx_signals = self.hyper_adx_filter.get_trend_signal()
            
            return consensus_filter_numba(
                hyper_er_signals,
                trend_index_signals, 
                hyper_adx_signals
            )
        
        # デフォルト（フィルターなし）
        return np.ones(len(data))
    
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
        if self._hyper_adaptive_channel_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        
        if index == -1:
            index = len(data) - 1
        
        # Numba最適化された関数を使用
        return check_exit_conditions_numba(self._hyper_adaptive_channel_signals, position, index)
    
    def get_band_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        HyperAdaptiveChannelのバンド値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (中心線, 上限バンド, 下限バンド)のタプル
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.hyper_adaptive_channel_signal.get_band_values()
        except Exception as e:
            self.logger.error(f"バンド値取得中にエラー: {str(e)}")
            # エラー時は空の配列を返す
            empty = np.array([])
            return empty, empty, empty
    
    def get_hyper_er(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        HyperER（ハイパー効率比）の値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: HyperERの値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.hyper_adaptive_channel_signal.get_hyper_er()
        except Exception as e:
            self.logger.error(f"HyperER取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_dynamic_multiplier(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        動的乗数の値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 動的乗数の値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.hyper_adaptive_channel_signal.get_dynamic_multiplier()
        except Exception as e:
            self.logger.error(f"動的乗数取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_x_atr(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        X_ATRの値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: X_ATRの値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.hyper_adaptive_channel_signal.get_x_atr()
        except Exception as e:
            self.logger.error(f"X_ATR取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_hyper_adaptive_channel_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """HyperAdaptiveChannelシグナル取得"""
        if self._hyper_adaptive_channel_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._hyper_adaptive_channel_signals
    
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
            HyperAdaptiveChannelとフィルターの全メトリクス
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            # チャネルバンド情報を取得
            midline, upper_band, lower_band = self.get_band_values()
            
            metrics = {
                # HyperAdaptiveChannelメトリクス
                'midline': midline,
                'upper_band': upper_band,
                'lower_band': lower_band,
                'hyper_er_values': self.get_hyper_er(),
                'dynamic_multiplier': self.get_dynamic_multiplier(),
                'x_atr_values': self.get_x_atr(),
                'channel_signals': self.get_hyper_adaptive_channel_signals(data),
                # シグナルメトリクス
                'long_signals': self.get_long_signals(data),
                'short_signals': self.get_short_signals(data),
                'filter_signals': self.get_filter_signals(data),
                # フィルター情報
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
        self._hyper_adaptive_channel_signals = None
        self._filter_signals = None
        
        if hasattr(self.hyper_adaptive_channel_signal, 'reset'):
            self.hyper_adaptive_channel_signal.reset()
        if self.hyper_er_filter is not None and hasattr(self.hyper_er_filter, 'reset'):
            self.hyper_er_filter.reset()
        if self.hyper_trend_index_filter is not None and hasattr(self.hyper_trend_index_filter, 'reset'):
            self.hyper_trend_index_filter.reset()
        if self.hyper_adx_filter is not None and hasattr(self.hyper_adx_filter, 'reset'):
            self.hyper_adx_filter.reset()