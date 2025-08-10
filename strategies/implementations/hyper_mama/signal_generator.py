#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple, Optional
import numpy as np
import pandas as pd
from numba import njit, prange
from enum import Enum

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.hyper_mama.entry import HyperMAMACrossoverEntrySignal
from indicators.trend_filter.hyper_er import HyperER
from indicators.hyper_trend_index import HyperTrendIndex
from indicators.trend_filter.hyper_adx import HyperADX


class FilterType(Enum):
    """HyperMAMAストラテジー用のフィルタータイプ"""
    NONE = "none"
    HYPER_ER = "hyper_er"
    HYPER_TREND_INDEX = "hyper_trend_index" 
    HYPER_ADX = "hyper_adx"
    CONSENSUS = "consensus"  # 3つのうち2つが1の場合に1を出力


@njit(fastmath=True, parallel=False)
def combine_signals_numba(hyper_mama_signals: np.ndarray, filter_signals: np.ndarray, filter_type: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    HyperMAMAシグナルとフィルターシグナルを統合する（Numba最適化版）
    
    Args:
        hyper_mama_signals: HyperMAMAシグナル配列
        filter_signals: フィルターシグナル配列
        filter_type: フィルタータイプ（0=None, 1=HyperER, 2=HyperTrendIndex, 3=HyperADX, 4=Consensus）
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (ロングシグナル, ショートシグナル)
    """
    length = len(hyper_mama_signals)
    long_signals = np.zeros(length, dtype=np.int8)
    short_signals = np.zeros(length, dtype=np.int8)
    
    for i in prange(length):
        hyper_mama_signal = hyper_mama_signals[i]
        
        # フィルターなしの場合
        if filter_type == 0:
            if hyper_mama_signal == 1:
                long_signals[i] = 1
            elif hyper_mama_signal == -1:
                short_signals[i] = 1
        else:
            # フィルターありの場合
            if i < len(filter_signals):
                filter_signal = filter_signals[i]
                
                # ロングエントリー条件: HyperMAMAシグナル=1 かつ フィルターシグナル=1
                if hyper_mama_signal == 1 and filter_signal == 1:
                    long_signals[i] = 1
                
                # ショートエントリー条件: HyperMAMAシグナル=-1 かつ フィルターシグナル=-1
                elif hyper_mama_signal == -1 and filter_signal == -1:
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
def check_exit_conditions_numba(hyper_mama_signals: np.ndarray, position: int, index: int) -> bool:
    """
    エグジット条件をチェックする（Numba最適化版）
    
    Args:
        hyper_mama_signals: HyperMAMAシグナル配列
        position: 現在のポジション（1=ロング、-1=ショート）
        index: チェックするインデックス
    
    Returns:
        bool: エグジットすべきかどうか
    """
    if index < 0 or index >= len(hyper_mama_signals):
        return False
    
    hyper_mama_signal = hyper_mama_signals[index]
    
    # ロングポジション: HyperMAMAシグナル=-1でエグジット
    if position == 1 and hyper_mama_signal == -1:
        return True
    
    # ショートポジション: HyperMAMAシグナル=1でエグジット
    if position == -1 and hyper_mama_signal == 1:
        return True
    
    return False


class HyperMAMAEnhancedSignalGenerator(BaseSignalGenerator):
    """
    HyperMAMA Enhanced シグナル生成クラス
    
    特徴:
    - HyperMAMAシグナルをベースとした高度なエントリー・エグジット制御
    - HyperER効率性による動的適応制御（fastlimit: 0.1-0.5, slowlimit: 0.01-0.05）
    - 4つのフィルターから選択可能（Phasor Trend, Correlation Cycle, Correlation Trend, Unified Trend Cycle）
    - フィルターなしオプションも提供
    
    エントリー条件:
    - ロング: HyperMAMAシグナル=1 かつ フィルターシグナル=1（フィルター有効時）
    - ショート: HyperMAMAシグナル=-1 かつ フィルターシグナル=-1（フィルター有効時）
    - フィルターシグナル=0の場合はスルー
    
    エグジット条件:
    - ロング: HyperMAMAシグナル=-1
    - ショート: HyperMAMAシグナル=1
    """
    
    def __init__(
        self,
        # 動的適応のトリガータイプ
        trigger_type: str = 'hyper_er',          # 'hyper_er' または 'hyper_trend_index'
        
        # HyperMAMAパラメータ
        hyper_er_period: int = 14,               # HyperER計算期間
        hyper_er_midline_period: int = 100,      # HyperERミッドライン計算期間
        # HyperERの詳細パラメータ
        hyper_er_er_period: int = 13,            # HyperER ER計算期間
        hyper_er_src_type: str = 'oc2',          # HyperER ソースタイプ
        # HyperER ルーフィングフィルターパラメータ
        hyper_er_use_roofing_filter: bool = True,  # ルーフィングフィルターを使用するか
        hyper_er_roofing_hp_cutoff: float = 48.0,  # ルーフィングフィルターのHighPassカットオフ
        hyper_er_roofing_ss_band_edge: float = 10.0,  # ルーフィングフィルターのSuperSmootherバンドエッジ
        # HyperER ラゲールフィルターパラメータ
        hyper_er_use_laguerre_filter: bool = False,  # ラゲールフィルターを使用するか
        hyper_er_laguerre_gamma: float = 0.8,  # ラゲールフィルターのガンマパラメータ
        # HyperER 平滑化オプション
        hyper_er_use_smoothing: bool = True,     # 平滑化を使用するか
        hyper_er_smoother_type: str = 'frama',   # 統合スムーサータイプ
        hyper_er_smoother_period: int = 12,      # スムーサー期間
        hyper_er_smoother_src_type: str = 'close',  # スムーサーソースタイプ
        # HyperER エラーズ統合サイクル検出器パラメータ
        hyper_er_use_dynamic_period: bool = True,   # 動的期間適応を使用するか
        hyper_er_detector_type: str = 'hody_e',     # サイクル検出器タイプ
        hyper_er_lp_period: int = 13,               # ローパスフィルター期間
        hyper_er_hp_period: int = 124,              # ハイパスフィルター期間
        hyper_er_cycle_part: float = 0.5,           # サイクル部分
        hyper_er_max_cycle: int = 124,              # 最大サイクル期間
        hyper_er_min_cycle: int = 13,               # 最小サイクル期間
        hyper_er_max_output: int = 89,              # 最大出力値
        hyper_er_min_output: int = 5,               # 最小出力値
        # HyperER パーセンタイルベーストレンド分析パラメータ
        hyper_er_enable_percentile_analysis: bool = True,  # パーセンタイル分析を有効にするか
        hyper_er_percentile_lookback_period: int = 50,     # パーセンタイル分析のルックバック期間
        hyper_er_percentile_low_threshold: float = 0.25,   # パーセンタイル分析の低閾値
        hyper_er_percentile_high_threshold: float = 0.75,  # パーセンタイル分析の高閾値
        
        # HyperTrendIndex関連パラメータ
        hyper_trend_period: int = 14,                    # HyperTrendIndex計算期間
        hyper_trend_midline_period: int = 100,           # HyperTrendIndexミッドライン計算期間
        hyper_trend_src_type: str = 'hlc3',              # HyperTrendIndexソースタイプ
        hyper_trend_use_kalman_filter: bool = True,      # HyperTrendIndexカルマンフィルターを使用するか
        hyper_trend_kalman_filter_type: str = 'simple',  # HyperTrendIndexカルマンフィルタータイプ
        hyper_trend_use_dynamic_period: bool = True,     # HyperTrendIndex動的期間適応を使用するか
        hyper_trend_detector_type: str = 'dft_dominant',  # HyperTrendIndexサイクル検出器タイプ
        hyper_trend_use_roofing_filter: bool = True,     # HyperTrendIndexルーフィングフィルターを使用するか
        hyper_trend_roofing_hp_cutoff: float = 55.0,     # HyperTrendIndexルーフィングフィルターHighPassカットオフ
        hyper_trend_roofing_ss_band_edge: float = 10.0,  # HyperTrendIndexルーフィングフィルターSuperSmootherバンドエッジ
        # 動的適応パラメータ
        fast_max: float = 0.5,                   # fastlimitの最大値
        fast_min: float = 0.1,                   # fastlimitの最小値
        slow_max: float = 0.05,                  # slowlimitの最大値
        slow_min: float = 0.01,                  # slowlimitの最小値
        er_high_threshold: float = 0.8,          # HyperERの高閾値
        er_low_threshold: float = 0.2,           # HyperERの低閾値
        src_type: str = 'hlc3',                  # ソースタイプ
        # カルマンフィルターパラメータ（HyperMAMA用）
        use_kalman_filter: bool = False,         # カルマンフィルターを使用するか
        kalman_filter_type: str = 'unscented',   # カルマンフィルタータイプ
        kalman_process_noise: float = 0.01,      # プロセスノイズ
        kalman_observation_noise: float = 0.001, # 観測ノイズ
        # ゼロラグ処理パラメータ（HyperMAMA用）
        use_zero_lag: bool = True,               # ゼロラグ処理を使用するか
        # シグナル設定（HyperMAMA用）
        position_mode: bool = False,             # 位置関係シグナル(True)またはクロスオーバーシグナル(False)
        # フィルター選択
        filter_type: FilterType = FilterType.NONE,  # フィルタータイプ
        # HyperER フィルターパラメータ（2つ目）
        filter_hyper_er_period: int = 14,
        filter_hyper_er_midline_period: int = 100,
        # HyperTrendIndex フィルターパラメータ
        filter_hyper_trend_index_period: int = 14,
        filter_hyper_trend_index_midline_period: int = 100,
        # HyperADX フィルターパラメータ
        filter_hyper_adx_period: int = 14,
        filter_hyper_adx_midline_period: int = 100
    ):
        """
        初期化
        
        Args:
            hyper_er_period: HyperER計算期間（デフォルト: 14）
            hyper_er_midline_period: HyperERミッドライン計算期間（デフォルト: 100）
            fast_max: fastlimitの最大値（デフォルト: 0.5）
            fast_min: fastlimitの最小値（デフォルト: 0.1）
            slow_max: slowlimitの最大値（デフォルト: 0.05）
            slow_min: slowlimitの最小値（デフォルト: 0.01）
            er_high_threshold: HyperERの高閾値（デフォルト: 0.8）
            er_low_threshold: HyperERの低閾値（デフォルト: 0.2）
            src_type: ソースタイプ（デフォルト: 'hlc3'）
            use_kalman_filter: HyperMAMA用カルマンフィルター使用（デフォルト: False）
            kalman_filter_type: カルマンフィルタータイプ（デフォルト: 'unscented'）
            kalman_process_noise: プロセスノイズ（デフォルト: 0.01）
            kalman_observation_noise: 観測ノイズ（デフォルト: 0.001）
            use_zero_lag: ゼロラグ処理使用（デフォルト: True）
            position_mode: 位置関係シグナル(True)またはクロスオーバーシグナル(False)
            filter_type: フィルタータイプ（デフォルト: FilterType.NONE）
            その他: 各フィルターのパラメータ
        """
        filter_name = filter_type.value if isinstance(filter_type, FilterType) else str(filter_type)
        signal_type = "Position" if position_mode else "Crossover"
        kalman_str = f"_kalman({kalman_filter_type})" if use_kalman_filter else ""
        zero_lag_str = "_zero_lag" if use_zero_lag else ""
        
        super().__init__(f"HyperMAMA_Enhanced_{signal_type}_{filter_name}{kalman_str}{zero_lag_str}")
        
        # パラメータの設定
        self._params = {
            # HyperMAMAパラメータ
            'hyper_er_period': hyper_er_period,
            'hyper_er_midline_period': hyper_er_midline_period,
            # HyperERの詳細パラメータ
            'hyper_er_er_period': hyper_er_er_period,
            'hyper_er_src_type': hyper_er_src_type,
            # HyperER ルーフィングフィルターパラメータ
            'hyper_er_use_roofing_filter': hyper_er_use_roofing_filter,
            'hyper_er_roofing_hp_cutoff': hyper_er_roofing_hp_cutoff,
            'hyper_er_roofing_ss_band_edge': hyper_er_roofing_ss_band_edge,
            # HyperER ラゲールフィルターパラメータ
            'hyper_er_use_laguerre_filter': hyper_er_use_laguerre_filter,
            'hyper_er_laguerre_gamma': hyper_er_laguerre_gamma,
            # HyperER 平滑化オプション
            'hyper_er_use_smoothing': hyper_er_use_smoothing,
            'hyper_er_smoother_type': hyper_er_smoother_type,
            'hyper_er_smoother_period': hyper_er_smoother_period,
            'hyper_er_smoother_src_type': hyper_er_smoother_src_type,
            # HyperER エラーズ統合サイクル検出器パラメータ
            'hyper_er_use_dynamic_period': hyper_er_use_dynamic_period,
            'hyper_er_detector_type': hyper_er_detector_type,
            'hyper_er_lp_period': hyper_er_lp_period,
            'hyper_er_hp_period': hyper_er_hp_period,
            'hyper_er_cycle_part': hyper_er_cycle_part,
            'hyper_er_max_cycle': hyper_er_max_cycle,
            'hyper_er_min_cycle': hyper_er_min_cycle,
            'hyper_er_max_output': hyper_er_max_output,
            'hyper_er_min_output': hyper_er_min_output,
            # HyperER パーセンタイルベーストレンド分析パラメータ
            'hyper_er_enable_percentile_analysis': hyper_er_enable_percentile_analysis,
            'hyper_er_percentile_lookback_period': hyper_er_percentile_lookback_period,
            'hyper_er_percentile_low_threshold': hyper_er_percentile_low_threshold,
            'hyper_er_percentile_high_threshold': hyper_er_percentile_high_threshold,
            # HyperTrendIndexパラメータ
            'trigger_type': trigger_type,
            'hyper_trend_period': hyper_trend_period,
            'hyper_trend_midline_period': hyper_trend_midline_period,
            'hyper_trend_src_type': hyper_trend_src_type,
            'hyper_trend_use_kalman_filter': hyper_trend_use_kalman_filter,
            'hyper_trend_kalman_filter_type': hyper_trend_kalman_filter_type,
            'hyper_trend_use_dynamic_period': hyper_trend_use_dynamic_period,
            'hyper_trend_detector_type': hyper_trend_detector_type,
            'hyper_trend_use_roofing_filter': hyper_trend_use_roofing_filter,
            'hyper_trend_roofing_hp_cutoff': hyper_trend_roofing_hp_cutoff,
            'hyper_trend_roofing_ss_band_edge': hyper_trend_roofing_ss_band_edge,
            # 動的適応パラメータ
            'fast_max': fast_max,
            'fast_min': fast_min,
            'slow_max': slow_max,
            'slow_min': slow_min,
            'er_high_threshold': er_high_threshold,
            'er_low_threshold': er_low_threshold,
            'src_type': src_type,
            'use_kalman_filter': use_kalman_filter,
            'kalman_filter_type': kalman_filter_type,
            'kalman_process_noise': kalman_process_noise,
            'kalman_observation_noise': kalman_observation_noise,
            'use_zero_lag': use_zero_lag,
            'position_mode': position_mode,
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
        self.position_mode = position_mode
        
        # HyperMAMAエントリーシグナルの初期化
        self.hyper_mama_entry_signal = HyperMAMACrossoverEntrySignal(
            trigger_type=trigger_type,
            hyper_er_period=hyper_er_period,
            hyper_er_midline_period=hyper_er_midline_period,
            # HyperERの詳細パラメータ
            hyper_er_er_period=hyper_er_er_period,
            hyper_er_src_type=hyper_er_src_type,
            # HyperER ルーフィングフィルターパラメータ
            hyper_er_use_roofing_filter=hyper_er_use_roofing_filter,
            hyper_er_roofing_hp_cutoff=hyper_er_roofing_hp_cutoff,
            hyper_er_roofing_ss_band_edge=hyper_er_roofing_ss_band_edge,
            # HyperER ラゲールフィルターパラメータ
            hyper_er_use_laguerre_filter=hyper_er_use_laguerre_filter,
            hyper_er_laguerre_gamma=hyper_er_laguerre_gamma,
            # HyperER 平滑化オプション
            hyper_er_use_smoothing=hyper_er_use_smoothing,
            hyper_er_smoother_type=hyper_er_smoother_type,
            hyper_er_smoother_period=hyper_er_smoother_period,
            hyper_er_smoother_src_type=hyper_er_smoother_src_type,
            # HyperER エラーズ統合サイクル検出器パラメータ
            hyper_er_use_dynamic_period=hyper_er_use_dynamic_period,
            hyper_er_detector_type=hyper_er_detector_type,
            hyper_er_lp_period=hyper_er_lp_period,
            hyper_er_hp_period=hyper_er_hp_period,
            hyper_er_cycle_part=hyper_er_cycle_part,
            hyper_er_max_cycle=hyper_er_max_cycle,
            hyper_er_min_cycle=hyper_er_min_cycle,
            hyper_er_max_output=hyper_er_max_output,
            hyper_er_min_output=hyper_er_min_output,
            # HyperER パーセンタイルベーストレンド分析パラメータ
            hyper_er_enable_percentile_analysis=hyper_er_enable_percentile_analysis,
            hyper_er_percentile_lookback_period=hyper_er_percentile_lookback_period,
            hyper_er_percentile_low_threshold=hyper_er_percentile_low_threshold,
            hyper_er_percentile_high_threshold=hyper_er_percentile_high_threshold,
            # HyperTrendIndexパラメータ
            hyper_trend_period=hyper_trend_period,
            hyper_trend_midline_period=hyper_trend_midline_period,
            hyper_trend_src_type=hyper_trend_src_type,
            hyper_trend_use_kalman_filter=hyper_trend_use_kalman_filter,
            hyper_trend_kalman_filter_type=hyper_trend_kalman_filter_type,
            hyper_trend_use_dynamic_period=hyper_trend_use_dynamic_period,
            hyper_trend_detector_type=hyper_trend_detector_type,
            hyper_trend_use_roofing_filter=hyper_trend_use_roofing_filter,
            hyper_trend_roofing_hp_cutoff=hyper_trend_roofing_hp_cutoff,
            hyper_trend_roofing_ss_band_edge=hyper_trend_roofing_ss_band_edge,
            # 動的適応パラメータ
            fast_max=fast_max,
            fast_min=fast_min,
            slow_max=slow_max,
            slow_min=slow_min,
            er_high_threshold=er_high_threshold,
            er_low_threshold=er_low_threshold,
            src_type=src_type,
            use_kalman_filter=use_kalman_filter,
            kalman_filter_type=kalman_filter_type,
            kalman_process_noise=kalman_process_noise,
            kalman_observation_noise=kalman_observation_noise,
            use_zero_lag=use_zero_lag,
            position_mode=position_mode
        )
        
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
        self._hyper_mama_signals = None
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
                    # HyperMAMAシグナルの計算
                    hyper_mama_signals = self.hyper_mama_entry_signal.generate(df)
                    self._hyper_mama_signals = hyper_mama_signals
                    
                    # フィルターシグナルの計算
                    self._filter_signals = self._get_filter_signals(df)
                    
                    # シグナルの統合
                    filter_type_int = list(FilterType).index(self.filter_type)
                    self._long_signals, self._short_signals = combine_signals_numba(
                        hyper_mama_signals, self._filter_signals, filter_type_int
                    )
                    
                except Exception as e:
                    self.logger.error(f"シグナル計算中にエラー: {str(e)}")
                    # エラー時はゼロシグナルを設定
                    self._long_signals = np.zeros(current_len, dtype=np.int8)
                    self._short_signals = np.zeros(current_len, dtype=np.int8)
                    self._hyper_mama_signals = np.zeros(current_len, dtype=np.int8)
                    self._filter_signals = np.zeros(current_len, dtype=np.int8)
                
                self._data_len = current_len
        except Exception as e:
            self.logger.error(f"calculate_signals全体でエラー: {str(e)}")
            # エラー時はゼロシグナルを設定
            if data is not None:
                self._long_signals = np.zeros(len(data), dtype=np.int8)
                self._short_signals = np.zeros(len(data), dtype=np.int8)
                self._hyper_mama_signals = np.zeros(len(data), dtype=np.int8)
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
        if self._hyper_mama_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        
        if index == -1:
            index = len(data) - 1
        
        # Numba最適化された関数を使用
        return check_exit_conditions_numba(self._hyper_mama_signals, position, index)
    
    def get_hyper_mama_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """HyperMAMAシグナル取得"""
        if self._hyper_mama_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._hyper_mama_signals
    
    def get_filter_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """フィルターシグナル取得"""
        if self._filter_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._filter_signals
    
    def get_mama_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """HyperMAMA値を取得"""
        try:
            if data is not None:
                self.calculate_signals(data)
            return self.hyper_mama_entry_signal.get_mama_values()
        except Exception as e:
            self.logger.error(f"HyperMAMA値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_fama_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """HyperFAMA値を取得"""
        try:
            if data is not None:
                self.calculate_signals(data)
            return self.hyper_mama_entry_signal.get_fama_values()
        except Exception as e:
            self.logger.error(f"HyperFAMA値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_hyper_er_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """HyperER値を取得"""
        try:
            if data is not None:
                self.calculate_signals(data)
            return self.hyper_mama_entry_signal.get_hyper_er_values()
        except Exception as e:
            self.logger.error(f"HyperER値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_adaptive_limits(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Optional[tuple]:
        """動的適応されたfastlimitとslowlimitを取得"""
        try:
            if data is not None:
                self.calculate_signals(data)
            return self.hyper_mama_entry_signal.get_adaptive_limits()
        except Exception as e:
            self.logger.error(f"動的適応制限取得中にエラー: {str(e)}")
            return None
    
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
            HyperMAMAとフィルターの全メトリクス
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            metrics = {
                # HyperMAMAメトリクス
                'mama_values': self.get_mama_values(),
                'fama_values': self.get_fama_values(),
                'hyper_er_values': self.get_hyper_er_values(),
                'hyper_mama_signals': self.get_hyper_mama_signals(data),
                # シグナルメトリクス
                'long_signals': self.get_long_signals(data),
                'short_signals': self.get_short_signals(data),
                'filter_signals': self.get_filter_signals(data),
                # フィルター詳細
                'filter_type': self.filter_type.value
            }
            
            # 動的適応制限の追加
            adaptive_limits = self.get_adaptive_limits()
            if adaptive_limits is not None:
                metrics.update({
                    'adaptive_fast_limits': adaptive_limits[0],
                    'adaptive_slow_limits': adaptive_limits[1]
                })
            
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
        self._hyper_mama_signals = None
        self._filter_signals = None
        
        if hasattr(self.hyper_mama_entry_signal, 'reset'):
            self.hyper_mama_entry_signal.reset()
        if self.hyper_er_filter is not None and hasattr(self.hyper_er_filter, 'reset'):
            self.hyper_er_filter.reset()
        if self.hyper_trend_index_filter is not None and hasattr(self.hyper_trend_index_filter, 'reset'):
            self.hyper_trend_index_filter.reset()
        if self.hyper_adx_filter is not None and hasattr(self.hyper_adx_filter, 'reset'):
            self.hyper_adx_filter.reset()