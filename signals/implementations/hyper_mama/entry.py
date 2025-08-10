#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, Optional
import numpy as np
import pandas as pd
from numba import njit, prange
from enum import Enum

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.hyper_mama import HyperMAMA
from indicators.trend_filter.hyper_er import HyperER
from indicators.hyper_trend_index import HyperTrendIndex
from indicators.trend_filter.hyper_adx import HyperADX


class FilterType(Enum):
    """HyperMAMAエントリーシグナル用のフィルタータイプ"""
    NONE = "none"
    HYPER_ER = "hyper_er"
    HYPER_TREND_INDEX = "hyper_trend_index"
    HYPER_ADX = "hyper_adx"
    CONSENSUS = "consensus"  # 3つのうち2つが1の場合に1を出力


@njit(fastmath=True)
def consensus_filter_numba(
    hyper_er_signals: np.ndarray,
    hyper_trend_signals: np.ndarray, 
    hyper_adx_signals: np.ndarray
) -> np.ndarray:
    """
    3つのフィルターの合意判定（高速化版）
    3つのうち2つ以上が1を出力した場合に1を出力
    
    Args:
        hyper_er_signals: HyperERのシグナル配列
        hyper_trend_signals: HyperTrendIndexのシグナル配列
        hyper_adx_signals: HyperADXのシグナル配列
    
    Returns:
        np.ndarray: 合意シグナル（1: 合意あり、-1: 逆合意、0: 合意なし）
    """
    length = len(hyper_er_signals)
    consensus_signals = np.zeros(length, dtype=np.int8)
    
    for i in range(length):
        # 各フィルターのシグナルを取得
        er_sig = hyper_er_signals[i]
        trend_sig = hyper_trend_signals[i]
        adx_sig = hyper_adx_signals[i]
        
        # 1のシグナル数をカウント
        positive_count = 0
        negative_count = 0
        
        if er_sig == 1:
            positive_count += 1
        elif er_sig == -1:
            negative_count += 1
            
        if trend_sig == 1:
            positive_count += 1
        elif trend_sig == -1:
            negative_count += 1
            
        if adx_sig == 1:
            positive_count += 1
        elif adx_sig == -1:
            negative_count += 1
        
        # 2つ以上が同じ方向なら合意とみなす
        if positive_count >= 2:
            consensus_signals[i] = 1
        elif negative_count >= 2:
            consensus_signals[i] = -1
        else:
            consensus_signals[i] = 0
    
    return consensus_signals


@njit(fastmath=True, parallel=True)
def calculate_position_signals(
    mama_values: np.ndarray, 
    fama_values: np.ndarray
) -> np.ndarray:
    """
    HyperMAMAとHyperFAMAの位置関係シグナルを計算する（高速化版）
    
    Args:
        mama_values: HyperMAMA値の配列
        fama_values: HyperFAMA値の配列
    
    Returns:
        シグナルの配列（1: ロング, -1: ショート, 0: シグナルなし）
    """
    length = len(mama_values)
    signals = np.zeros(length, dtype=np.int8)
    
    # 位置関係の判定（並列処理化）
    for i in prange(length):
        # HyperMAMA値とHyperFAMA値が有効かチェック
        if np.isnan(mama_values[i]) or np.isnan(fama_values[i]):
            signals[i] = 0
            continue
            
        # HyperMAMA > HyperFAMA: ロングシグナル
        if mama_values[i] > fama_values[i]:
            signals[i] = 1
        # HyperMAMA < HyperFAMA: ショートシグナル
        elif mama_values[i] < fama_values[i]:
            signals[i] = -1
    
    return signals


@njit(fastmath=True)
def calculate_crossover_signals(
    mama_values: np.ndarray, 
    fama_values: np.ndarray
) -> np.ndarray:
    """
    HyperMAMAとHyperFAMAのクロスオーバーシグナルを計算する（高速化版）
    
    Args:
        mama_values: HyperMAMA値の配列
        fama_values: HyperFAMA値の配列
    
    Returns:
        シグナルの配列（1: ゴールデンクロス, -1: デッドクロス, 0: シグナルなし）
    """
    length = len(mama_values)
    signals = np.zeros(length, dtype=np.int8)
    
    # 前の値との比較でクロスオーバーを検出
    for i in range(1, length):
        # 現在と前の値が有効かチェック
        if (np.isnan(mama_values[i]) or np.isnan(fama_values[i]) or 
            np.isnan(mama_values[i-1]) or np.isnan(fama_values[i-1])):
            signals[i] = 0
            continue
            
        # 前の期間
        prev_mama = mama_values[i-1]
        prev_fama = fama_values[i-1]
        
        # 現在の期間
        curr_mama = mama_values[i]
        curr_fama = fama_values[i]
        
        # ゴールデンクロス: 前期間でMAMA <= FAMA、現期間でMAMA > FAMA
        if prev_mama <= prev_fama and curr_mama > curr_fama:
            signals[i] = 1
        # デッドクロス: 前期間でMAMA >= FAMA、現期間でMAMA < FAMA
        elif prev_mama >= prev_fama and curr_mama < curr_fama:
            signals[i] = -1
    
    return signals


class HyperMAMACrossoverEntrySignal(BaseSignal, IEntrySignal):
    """
    HyperMAMA/HyperFAMAクロスオーバーによるエントリーシグナル
    
    特徴:
    - HyperMAMA (Hyper Mother of Adaptive Moving Average) / HyperFAMA (Hyper Following Adaptive Moving Average)
    - HyperERインジケーターによる効率性ベースの動的適応
    - fastlimit: 0.1〜0.5、slowlimit: 0.01〜0.05 の動的調整
    - Ehlers's MESA (Maximum Entropy Spectrum Analysis) アルゴリズムベース
    - カルマンフィルターとゼロラグ処理統合版
    - トレンド効率性に応じて応答速度を自動調整
    
    シグナル条件:
    - position_mode=True: HyperMAMA > HyperFAMA: ロングシグナル (1), HyperMAMA < HyperFAMA: ショートシグナル (-1)
    - position_mode=False: ゴールデンクロス: ロングシグナル (1), デッドクロス: ショートシグナル (-1)
    """
    
    def __init__(
        self,
        # 動的適応のトリガータイプ
        trigger_type: str = 'hyper_er',          # 'hyper_er' または 'hyper_trend_index'
        
        # HyperMAMA関連パラメータ
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
        # カルマンフィルターパラメータ
        use_kalman_filter: bool = False,         # カルマンフィルターを使用するか
        kalman_filter_type: str = 'unscented',   # カルマンフィルタータイプ
        kalman_process_noise: float = 0.01,      # プロセスノイズ
        kalman_observation_noise: float = 0.001, # 観測ノイズ
        # ゼロラグ処理パラメータ
        use_zero_lag: bool = True,               # ゼロラグ処理を使用するか
        # シグナル設定
        position_mode: bool = False,             # 位置関係シグナル(True)またはクロスオーバーシグナル(False)
        # フィルター設定
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
        """
        初期化
        
        Args:
            hyper_er_period: HyperER計算期間（デフォルト: 14）
            hyper_er_midline_period: HyperERミッドライン計算期間（デフォルト: 100）
            hyper_er_er_period: HyperER ER計算期間（デフォルト: 13）
            hyper_er_src_type: HyperER ソースタイプ（デフォルト: 'oc2'）
            hyper_er_use_roofing_filter: ルーフィングフィルターを使用するか（デフォルト: True）
            hyper_er_roofing_hp_cutoff: ルーフィングフィルターのHighPassカットオフ（デフォルト: 48.0）
            hyper_er_roofing_ss_band_edge: ルーフィングフィルターのSuperSmootherバンドエッジ（デフォルト: 10.0）
            hyper_er_use_smoothing: 平滑化を使用するか（デフォルト: True）
            hyper_er_smoother_type: 統合スムーサータイプ（デフォルト: 'frama'）
            hyper_er_smoother_period: スムーサー期間（デフォルト: 12）
            hyper_er_smoother_src_type: スムーサーソースタイプ（デフォルト: 'close'）
            hyper_er_use_dynamic_period: 動的期間適応を使用するか（デフォルト: True）
            hyper_er_detector_type: サイクル検出器タイプ（デフォルト: 'hody_e'）
            hyper_er_lp_period: ローパスフィルター期間（デフォルト: 13）
            hyper_er_hp_period: ハイパスフィルター期間（デフォルト: 124）
            hyper_er_cycle_part: サイクル部分（デフォルト: 0.5）
            hyper_er_max_cycle: 最大サイクル期間（デフォルト: 124）
            hyper_er_min_cycle: 最小サイクル期間（デフォルト: 13）
            hyper_er_max_output: 最大出力値（デフォルト: 89）
            hyper_er_min_output: 最小出力値（デフォルト: 5）
            hyper_er_enable_percentile_analysis: パーセンタイル分析を有効にするか（デフォルト: True）
            hyper_er_percentile_lookback_period: パーセンタイル分析のルックバック期間（デフォルト: 50）
            hyper_er_percentile_low_threshold: パーセンタイル分析の低閾値（デフォルト: 0.25）
            hyper_er_percentile_high_threshold: パーセンタイル分析の高閾値（デフォルト: 0.75）
            fast_max: fastlimitの最大値（デフォルト: 0.5）
            fast_min: fastlimitの最小値（デフォルト: 0.1）
            slow_max: slowlimitの最大値（デフォルト: 0.05）
            slow_min: slowlimitの最小値（デフォルト: 0.01）
            er_high_threshold: HyperERの高閾値（デフォルト: 0.8）
            er_low_threshold: HyperERの低閾値（デフォルト: 0.2）
            src_type: ソースタイプ（デフォルト: 'hlc3'）
                基本ソース: 'close', 'hlc3', 'hl2', 'ohlc4', 'high', 'low', 'open'
            use_kalman_filter: カルマンフィルターを使用するか（デフォルト: False）
            kalman_filter_type: カルマンフィルタータイプ（デフォルト: 'unscented'）
            kalman_process_noise: プロセスノイズ（デフォルト: 0.01）
            kalman_observation_noise: 観測ノイズ（デフォルト: 0.001）
            use_zero_lag: ゼロラグ処理を使用するか（デフォルト: True）
            position_mode: 位置関係シグナル(True)またはクロスオーバーシグナル(False)
        """
        signal_type = "Position" if position_mode else "Crossover"
        kalman_str = f"_kalman({kalman_filter_type})" if use_kalman_filter else ""
        zero_lag_str = "_zero_lag" if use_zero_lag else ""
        
        super().__init__(
            f"HyperMAMA{signal_type}EntrySignal(fast={fast_min}-{fast_max}, slow={slow_min}-{slow_max}, {src_type}{kalman_str}{zero_lag_str})"
        )
        
        # パラメータの保存
        self._params = {
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
            'filter_hyper_er_period': filter_hyper_er_period,
            'filter_hyper_er_midline_period': filter_hyper_er_midline_period,
            'filter_hyper_trend_index_period': filter_hyper_trend_index_period,
            'filter_hyper_trend_index_midline_period': filter_hyper_trend_index_midline_period,
            'filter_hyper_adx_period': filter_hyper_adx_period,
            'filter_hyper_adx_midline_period': filter_hyper_adx_midline_period
        }
        
        self.position_mode = position_mode
        self.filter_type = filter_type
        
        # HyperMAMAインジケーターの初期化
        self.hyper_mama = HyperMAMA(
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
            use_zero_lag=use_zero_lag
        )
        
        # フィルター用インジケーターの初期化
        self.filter_hyper_er = None
        self.filter_hyper_trend_index = None
        self.filter_hyper_adx = None
        
        if filter_type != FilterType.NONE:
            # HyperERフィルター
            if filter_type in [FilterType.HYPER_ER, FilterType.CONSENSUS]:
                self.filter_hyper_er = HyperER(
                    period=filter_hyper_er_period,
                    midline_period=filter_hyper_er_midline_period
                )
            
            # HyperTrendIndexフィルター
            if filter_type in [FilterType.HYPER_TREND_INDEX, FilterType.CONSENSUS]:
                self.filter_hyper_trend_index = HyperTrendIndex(
                    period=filter_hyper_trend_index_period,
                    midline_period=filter_hyper_trend_index_midline_period
                )
            
            # HyperADXフィルター
            if filter_type in [FilterType.HYPER_ADX, FilterType.CONSENSUS]:
                self.filter_hyper_adx = HyperADX(
                    period=filter_hyper_adx_period,
                    midline_period=filter_hyper_adx_midline_period
                )
        
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
            if all(col in ohlcv_data.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                ohlcv_array = ohlcv_data[['open', 'high', 'low', 'close', 'volume']].values
            else:
                ohlcv_array = ohlcv_data.values
        else:
            ohlcv_array = ohlcv_data
            
        # Numpy配列でない場合はエラー
        if not isinstance(ohlcv_array, np.ndarray):
            raise TypeError("ohlcv_data must be a numpy array or pandas DataFrame")
        
        # 配列のハッシュと設定パラメータのハッシュを組み合わせる
        return hash((ohlcv_array.tobytes(), *sorted(self._params.items())))
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
        """
        try:
            # キャッシュチェック - 同じデータの場合は計算をスキップ
            data_hash = self._get_data_hash(data)
            if data_hash in self._signals_cache:
                return self._signals_cache[data_hash]
                
            # HyperMAMAの計算
            hyper_mama_result = self.hyper_mama.calculate(data)
            
            # 計算が失敗した場合はゼロシグナルを返す
            if hyper_mama_result is None or len(hyper_mama_result.mama_values) == 0:
                self._signals_cache[data_hash] = np.zeros(len(data), dtype=np.int8)
                return self._signals_cache[data_hash]
            
            # HyperMAMA値とHyperFAMA値の取得
            mama_values = hyper_mama_result.mama_values
            fama_values = hyper_mama_result.fama_values
            
            # シグナルの計算（位置関係またはクロスオーバー）
            if self.position_mode:
                # 位置関係シグナル
                hyper_mama_signals = calculate_position_signals(
                    mama_values,
                    fama_values
                )
            else:
                # クロスオーバーシグナル
                hyper_mama_signals = calculate_crossover_signals(
                    mama_values,
                    fama_values
                )
            
            # フィルターの適用
            if self.filter_type == FilterType.NONE:
                # フィルターなし - HyperMAMAシグナルをそのまま使用
                final_signals = hyper_mama_signals
            else:
                # フィルターシグナルを取得
                filter_signals = self._get_filter_signals(data)
                
                # HyperMAMAシグナルとフィルターシグナルを組み合わせ
                final_signals = np.zeros_like(hyper_mama_signals, dtype=np.int8)
                for i in range(len(hyper_mama_signals)):
                    if hyper_mama_signals[i] != 0 and filter_signals[i] == hyper_mama_signals[i]:
                        final_signals[i] = hyper_mama_signals[i]
            
            # 結果をキャッシュ
            self._signals_cache[data_hash] = final_signals
            return final_signals
            
        except Exception as e:
            # エラーが発生した場合は警告を出力し、ゼロシグナルを返す
            print(f"HyperMAMACrossoverEntrySignal計算中にエラー: {str(e)}")
            # エラー時に新しいハッシュキーを生成せず、一時的なゼロシグナルを返す
            # キャッシュすると別のエラーの可能性があるため、ここではキャッシュしない
            return np.zeros(len(data), dtype=np.int8)
    
    def get_mama_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        HyperMAMA値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: HyperMAMA値
        """
        if data is not None:
            self.generate(data)
            
        return self.hyper_mama.get_mama_values()
    
    def get_fama_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        HyperFAMA値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: HyperFAMA値
        """
        if data is not None:
            self.generate(data)
            
        return self.hyper_mama.get_fama_values()
    
    def get_period_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Period値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: Period値
        """
        if data is not None:
            self.generate(data)
            
        return self.hyper_mama.get_period_values()
    
    def get_alpha_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Alpha値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: Alpha値
        """
        if data is not None:
            self.generate(data)
            
        return self.hyper_mama.get_alpha_values()
    
    def get_phase_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Phase値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: Phase値
        """
        if data is not None:
            self.generate(data)
            
        return self.hyper_mama.get_phase_values()
    
    def get_i1_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        I1値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: I1値
        """
        if data is not None:
            self.generate(data)
            
        return self.hyper_mama.get_i1_values()
    
    def get_q1_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Q1値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: Q1値
        """
        if data is not None:
            self.generate(data)
            
        return self.hyper_mama.get_q1_values()
    
    def get_hyper_er_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        HyperER値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: HyperER値
        """
        if data is not None:
            self.generate(data)
            
        return self.hyper_mama.get_hyper_er_values()
    
    def get_adaptive_limits(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Optional[tuple]:
        """
        動的適応されたfastlimitとslowlimitを取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (adaptive_fast_limits, adaptive_slow_limits)
        """
        if data is not None:
            self.generate(data)
            
        return self.hyper_mama.get_adaptive_limits()
    
    def _get_filter_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        フィルターシグナルを取得する
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: フィルターシグナル (1: ロング, -1: ショート, 0: シグナルなし)
        """
        if self.filter_type == FilterType.NONE:
            # フィルターなしの場合は全て1を返す（シグナルを通す）
            return np.ones(len(data), dtype=np.int8)
        
        try:
            if self.filter_type == FilterType.HYPER_ER:
                # HyperERフィルター単体
                return self.filter_hyper_er.trend_signal(data)
                
            elif self.filter_type == FilterType.HYPER_TREND_INDEX:
                # HyperTrendIndexフィルター単体
                return self.filter_hyper_trend_index.trend_signal(data)
                
            elif self.filter_type == FilterType.HYPER_ADX:
                # HyperADXフィルター単体
                return self.filter_hyper_adx.trend_signal(data)
                
            elif self.filter_type == FilterType.CONSENSUS:
                # 合意フィルター（3つのうち2つが1の場合に1を出力）
                hyper_er_signals = self.filter_hyper_er.trend_signal(data)
                hyper_trend_signals = self.filter_hyper_trend_index.trend_signal(data)
                hyper_adx_signals = self.filter_hyper_adx.trend_signal(data)
                
                return consensus_filter_numba(
                    hyper_er_signals,
                    hyper_trend_signals,
                    hyper_adx_signals
                )
                
        except Exception as e:
            print(f"フィルターシグナル計算中にエラー ({self.filter_type.value}): {str(e)}")
            # エラー時は全て0を返す（シグナルを通さない）
            return np.zeros(len(data), dtype=np.int8)
        
        # 未知のフィルタータイプの場合
        return np.zeros(len(data), dtype=np.int8)
        
    def reset(self) -> None:
        """
        シグナルの状態をリセットする
        """
        super().reset()
        self.hyper_mama.reset() if hasattr(self.hyper_mama, 'reset') else None
        
        # フィルターインジケーターのリセット
        if self.filter_hyper_er and hasattr(self.filter_hyper_er, 'reset'):
            self.filter_hyper_er.reset()
        if self.filter_hyper_trend_index and hasattr(self.filter_hyper_trend_index, 'reset'):
            self.filter_hyper_trend_index.reset()
        if self.filter_hyper_adx and hasattr(self.filter_hyper_adx, 'reset'):
            self.filter_hyper_adx.reset()
            
        self._signals_cache = {}