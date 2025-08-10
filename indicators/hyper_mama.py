#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional, List, NamedTuple, Any
import numpy as np
import pandas as pd
from numba import jit, prange, vectorize, njit, float64, types
import traceback
import math

from .indicator import Indicator
from .price_source import PriceSource
from .trend_filter.hyper_er import HyperER
from .hyper_trend_index import HyperTrendIndex

# 条件付きインポート（オプション機能）
try:
    from .kalman.unified_kalman import UnifiedKalman
    UNIFIED_KALMAN_AVAILABLE = True
except ImportError:
    try:
        # 絶対インポートを試行（パス調整付き）
        import sys
        import os
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        from indicators.kalman.unified_kalman import UnifiedKalman
        UNIFIED_KALMAN_AVAILABLE = True
    except ImportError:
        UnifiedKalman = None
        UNIFIED_KALMAN_AVAILABLE = False


@dataclass
class HyperMAMAResult:
    """HyperMAMA/HyperFAMAの計算結果"""
    mama_values: np.ndarray      # HyperMAMAライン値
    fama_values: np.ndarray      # HyperFAMAライン値
    period_values: np.ndarray    # 計算されたPeriod値
    alpha_values: np.ndarray     # 計算されたAlpha値
    phase_values: np.ndarray     # Phase値
    i1_values: np.ndarray        # InPhase component
    q1_values: np.ndarray        # Quadrature component
    filtered_price: np.ndarray   # カルマンフィルター後の価格（使用した場合）
    hyper_er_values: np.ndarray  # HyperER値
    adaptive_fast_limits: np.ndarray  # 動的適応されたfastlimit値
    adaptive_slow_limits: np.ndarray  # 動的適応されたslowlimit値


@njit(fastmath=True, cache=True)
def calculate_zero_lag_processing(mama_values: np.ndarray, alpha_values: np.ndarray) -> np.ndarray:
    """
    HyperMAMA値に対してゼロラグ処理を適用する（Numba最適化版）
    HyperMAMA内部のアルファ値を使用してゼロラグ処理を実行
    
    Args:
        mama_values: HyperMAMA値の配列
        alpha_values: HyperMAMA内部のアルファ値の配列
    
    Returns:
        ゼロラグ処理後の値配列
    """
    length = len(mama_values)
    result = np.full(length, np.nan, dtype=np.float64)
    
    if length == 0:
        return result
    
    # EMA値の配列（HyperMAMA値のEMA）
    ema_values = np.full(length, np.nan, dtype=np.float64)
    
    # ラグ除去データの配列
    lag_reduced_data = np.full(length, np.nan, dtype=np.float64)
    
    # 最初の有効な値を見つける
    first_valid_idx = -1
    for i in range(length):
        if not np.isnan(mama_values[i]):
            ema_values[i] = mama_values[i]
            first_valid_idx = i
            break
    
    if first_valid_idx == -1:
        return result
    
    # EMAの計算（HyperMAMA値のEMA、HyperMAMAのアルファ値を使用）
    for i in range(first_valid_idx + 1, length):
        if not np.isnan(mama_values[i]) and not np.isnan(alpha_values[i]):
            if not np.isnan(ema_values[i-1]):
                ema_values[i] = alpha_values[i] * mama_values[i] + (1.0 - alpha_values[i]) * ema_values[i-1]
            else:
                ema_values[i] = mama_values[i]
    
    # ラグ除去データの計算
    for i in range(length):
        if not np.isnan(mama_values[i]) and not np.isnan(ema_values[i]):
            lag_reduced_data[i] = 2.0 * mama_values[i] - ema_values[i]
    
    # ZLEMAの計算
    # 最初の値はラグ除去データと同じ
    start_idx = first_valid_idx
    if start_idx < length and not np.isnan(lag_reduced_data[start_idx]):
        result[start_idx] = lag_reduced_data[start_idx]
    
    # 以降はラグ除去データのEMAを計算（HyperMAMAのアルファ値を使用）
    for i in range(start_idx + 1, length):
        if not np.isnan(lag_reduced_data[i]) and not np.isnan(alpha_values[i]):
            if not np.isnan(result[i-1]):
                result[i] = alpha_values[i] * lag_reduced_data[i] + (1.0 - alpha_values[i]) * result[i-1]
            else:
                result[i] = lag_reduced_data[i]
    
    return result


@njit(fastmath=True, cache=True)
def calculate_adaptive_limits_numba(
    hyper_er_values: np.ndarray,
    fast_max: float = 0.5,
    fast_min: float = 0.1,
    slow_max: float = 0.05,
    slow_min: float = 0.01,
    er_high_threshold: float = 0.8,
    er_low_threshold: float = 0.2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    HyperER値に基づいてfastlimitとslowlimitを動的適応させる（Numba最適化版）
    
    Args:
        hyper_er_values: HyperER値の配列
        fast_max: fastlimitの最大値（デフォルト: 0.5）
        fast_min: fastlimitの最小値（デフォルト: 0.1）
        slow_max: slowlimitの最大値（デフォルト: 0.05）
        slow_min: slowlimitの最小値（デフォルト: 0.01）
        er_high_threshold: HyperERの高閾値（デフォルト: 0.8）
        er_low_threshold: HyperERの低閾値（デフォルト: 0.2）
    
    Returns:
        適応されたfastlimitとslowlimitの配列のタプル
    """
    length = len(hyper_er_values)
    adaptive_fast_limits = np.full(length, np.nan, dtype=np.float64)
    adaptive_slow_limits = np.full(length, np.nan, dtype=np.float64)
    
    for i in range(length):
        if np.isnan(hyper_er_values[i]):
            # NaN値の場合はデフォルト値を使用
            adaptive_fast_limits[i] = (fast_max + fast_min) / 2.0
            adaptive_slow_limits[i] = (slow_max + slow_min) / 2.0
            continue
        
        er_value = hyper_er_values[i]
        
        # HyperERを0-1の範囲にクランプ
        er_value = max(0.0, min(1.0, er_value))
        
        # Kaufman Efficiency Scalingアプローチを参考にした動的適応
        if er_value >= er_high_threshold:
            # 高効率（0.8以上）：最大値を使用
            adaptive_fast_limits[i] = fast_max
            adaptive_slow_limits[i] = slow_max
        elif er_value <= er_low_threshold:
            # 低効率（0.2以下）：最小値を使用
            adaptive_fast_limits[i] = fast_min
            adaptive_slow_limits[i] = slow_min
        else:
            # 中間値：線形補間
            # ER値を0.2-0.8の範囲を0-1にマッピング
            normalized_er = (er_value - er_low_threshold) / (er_high_threshold - er_low_threshold)
            
            # 線形補間で適応値を計算
            adaptive_fast_limits[i] = fast_min + normalized_er * (fast_max - fast_min)
            adaptive_slow_limits[i] = slow_min + normalized_er * (slow_max - slow_min)
    
    return adaptive_fast_limits, adaptive_slow_limits


@njit(fastmath=True, cache=True)
def calculate_hyper_mama_fama(
    price: np.ndarray,
    adaptive_fast_limits: np.ndarray,
    adaptive_slow_limits: np.ndarray,
    use_zero_lag: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    HyperMAMA/HyperFAMAを計算する（Numba最適化版）
    
    Args:
        price: 価格配列（通常は(H+L)/2）
        adaptive_fast_limits: 動的適応されたfastlimit配列
        adaptive_slow_limits: 動的適応されたslowlimit配列
        use_zero_lag: ゼロラグ処理を使用するか（デフォルト: True）
    
    Returns:
        Tuple[np.ndarray, ...]: HyperMAMA値, HyperFAMA値, Period値, Alpha値, Phase値, I1値, Q1値
    """
    length = len(price)
    
    # 変数の初期化
    smooth = np.zeros(length, dtype=np.float64)
    detrender = np.zeros(length, dtype=np.float64)
    i1 = np.zeros(length, dtype=np.float64)
    q1 = np.zeros(length, dtype=np.float64)
    j_i = np.zeros(length, dtype=np.float64)
    j_q = np.zeros(length, dtype=np.float64)
    i2 = np.zeros(length, dtype=np.float64)
    q2 = np.zeros(length, dtype=np.float64)
    re = np.zeros(length, dtype=np.float64)
    im = np.zeros(length, dtype=np.float64)
    period = np.zeros(length, dtype=np.float64)
    smooth_period = np.zeros(length, dtype=np.float64)
    phase = np.zeros(length, dtype=np.float64)
    delta_phase = np.zeros(length, dtype=np.float64)
    alpha = np.zeros(length, dtype=np.float64)
    mama = np.zeros(length, dtype=np.float64)
    fama = np.zeros(length, dtype=np.float64)
    
    # 初期値設定 - すべて有効な値で初期化
    for i in range(min(7, length)):
        smooth[i] = price[i] if i < length else 100.0
        detrender[i] = 0.0
        i1[i] = 0.0
        q1[i] = 0.0
        j_i[i] = 0.0
        j_q[i] = 0.0
        i2[i] = 0.0
        q2[i] = 0.0
        re[i] = 0.0
        im[i] = 0.0
        period[i] = 20.0  # 初期値として有効な値を設定
        smooth_period[i] = 20.0
        phase[i] = 0.0
        delta_phase[i] = 1.0
        # 初期アルファ値は動的slow_limitを使用
        initial_slow = adaptive_slow_limits[i] if not np.isnan(adaptive_slow_limits[i]) else 0.05
        alpha[i] = initial_slow
        mama[i] = price[i] if i < length else 100.0
        fama[i] = price[i] if i < length else 100.0
    
    # CurrentBar > 5の条件 (インデックス5から開始、0ベースなので)
    for i in range(5, length):
        # 価格のスムージング: Smooth = (4*Price + 3*Price[1] + 2*Price[2] + Price[3]) / 10
        if i >= 3:  # 最低4つの価格が必要
            smooth[i] = (4.0 * price[i] + 3.0 * price[i-1] + 2.0 * price[i-2] + price[i-3]) / 10.0
        else:
            smooth[i] = price[i]  # フォールバック
            continue
        
        # 前回のPeriod値を取得（初回は20に設定）
        prev_period = period[i-1] if i > 6 and not np.isnan(period[i-1]) else 20.0
        
        # Detrender計算
        if i >= 6:
            detrender[i] = (0.0962 * smooth[i] + 0.5769 * smooth[i-2] - 
                           0.5769 * smooth[i-4] - 0.0962 * smooth[i-6]) * (0.075 * prev_period + 0.54)
        else:
            detrender[i] = 0.0  # 初期値として0を設定
            continue
        
        # InPhaseとQuadratureコンポーネントの計算
        if i >= 6:
            q1[i] = (0.0962 * detrender[i] + 0.5769 * detrender[i-2] - 
                    0.5769 * detrender[i-4] - 0.0962 * detrender[i-6]) * (0.075 * prev_period + 0.54)
            i1[i] = detrender[i-3] if i >= 9 else 0.0
        else:
            q1[i] = 0.0
            i1[i] = 0.0
            continue
        
        # 90度位相を進める
        if i >= 6:
            j_i[i] = (0.0962 * i1[i] + 0.5769 * i1[i-2] - 
                     0.5769 * i1[i-4] - 0.0962 * i1[i-6]) * (0.075 * prev_period + 0.54)
            j_q[i] = (0.0962 * q1[i] + 0.5769 * q1[i-2] - 
                     0.5769 * q1[i-4] - 0.0962 * q1[i-6]) * (0.075 * prev_period + 0.54)
        else:
            j_i[i] = 0.0
            j_q[i] = 0.0
            continue
        
        # Phasor加算（3バー平均）
        i2[i] = i1[i] - j_q[i]
        q2[i] = q1[i] + j_i[i]
        
        # IとQコンポーネントのスムージング
        if i > 5:
            i2[i] = 0.2 * i2[i] + 0.8 * i2[i-1]
            q2[i] = 0.2 * q2[i] + 0.8 * q2[i-1]
        
        # Homodyne Discriminator
        if i > 5:
            re[i] = i2[i] * i2[i-1] + q2[i] * q2[i-1]
            im[i] = i2[i] * q2[i-1] - q2[i] * i2[i-1]
            
            # ReとImのスムージング
            re[i] = 0.2 * re[i] + 0.8 * re[i-1]
            im[i] = 0.2 * im[i] + 0.8 * im[i-1]
        else:
            re[i] = 0.0
            im[i] = 0.0
            continue
        
        # Period計算
        if not np.isnan(im[i]) and not np.isnan(re[i]) and im[i] != 0.0 and re[i] != 0.0:
            # ArcTangent計算 - atan2を使用してより安全に計算
            atan_result = math.atan2(im[i], re[i]) * 180.0 / math.pi
            if abs(atan_result) > 0.001:  # 0に近すぎる値を避ける
                period[i] = 360.0 / abs(atan_result)
            else:
                period[i] = period[i-1] if i > 6 and not np.isnan(period[i-1]) else 20.0
            
            # Period制限
            if i > 5 and not np.isnan(period[i-1]):
                if period[i] > 1.5 * period[i-1]:
                    period[i] = 1.5 * period[i-1]
                elif period[i] < 0.67 * period[i-1]:
                    period[i] = 0.67 * period[i-1]
            
            if period[i] < 6.0:
                period[i] = 6.0
            elif period[i] > 50.0:
                period[i] = 50.0
            
            # Periodのスムージング
            if i > 5 and not np.isnan(period[i-1]):
                period[i] = 0.2 * period[i] + 0.8 * period[i-1]
        else:
            period[i] = period[i-1] if i > 5 and not np.isnan(period[i-1]) else 20.0
        
        # SmoothPeriod計算
        if i > 5 and not np.isnan(smooth_period[i-1]):
            smooth_period[i] = 0.33 * period[i] + 0.67 * smooth_period[i-1]
        else:
            smooth_period[i] = period[i]
        
        # Phase計算
        if not np.isnan(i1[i]) and not np.isnan(q1[i]):
            if abs(i1[i]) > 1e-10:  # i1が0に近すぎない場合のみ計算
                phase[i] = math.atan2(q1[i], i1[i]) * 180.0 / math.pi
            else:
                phase[i] = phase[i-1] if i > 5 else 0.0
        else:
            phase[i] = phase[i-1] if i > 5 else 0.0
        
        # DeltaPhase計算
        if i > 5:
            delta_phase[i] = abs(phase[i-1] - phase[i])
            if delta_phase[i] < 1.0:
                delta_phase[i] = 1.0
        else:
            delta_phase[i] = 1.0
        
        # Alpha計算 - 動的適応されたlimitを使用
        current_fast_limit = adaptive_fast_limits[i] if not np.isnan(adaptive_fast_limits[i]) else 0.5
        current_slow_limit = adaptive_slow_limits[i] if not np.isnan(adaptive_slow_limits[i]) else 0.05
        
        if delta_phase[i] > 0:
            alpha[i] = current_fast_limit / delta_phase[i]
            if alpha[i] < current_slow_limit:
                alpha[i] = current_slow_limit
            elif alpha[i] > current_fast_limit:
                alpha[i] = current_fast_limit
        else:
            alpha[i] = current_slow_limit
        
        # HyperMAMA計算
        if i > 5 and not np.isnan(mama[i-1]) and not np.isnan(alpha[i]):
            mama[i] = alpha[i] * price[i] + (1.0 - alpha[i]) * mama[i-1]
        else:
            mama[i] = price[i]  # 初期値として価格を使用
        
        # HyperFAMA計算
        if i > 5 and not np.isnan(fama[i-1]) and not np.isnan(mama[i]) and not np.isnan(alpha[i]):
            fama[i] = 0.5 * alpha[i] * mama[i] + (1.0 - 0.5 * alpha[i]) * fama[i-1]
        else:
            fama[i] = mama[i]  # 初期値としてMAMA値を使用
    
    # ゼロラグ処理の適用（オプション）
    if use_zero_lag:
        # HyperMAMAにゼロラグ処理を適用（HyperMAMAのアルファ値を使用）
        mama_zero_lag = calculate_zero_lag_processing(mama, alpha)
        
        # HyperFAMAにゼロラグ処理を適用（HyperMAMAのアルファ値を使用）
        fama_zero_lag = calculate_zero_lag_processing(fama, alpha)
        
        # 有効な値のみを使用（NaN値は元の値を保持）
        for i in range(length):
            if not np.isnan(mama_zero_lag[i]):
                mama[i] = mama_zero_lag[i]
            
            if not np.isnan(fama_zero_lag[i]):
                fama[i] = fama_zero_lag[i]
    
    return mama, fama, period, alpha, phase, i1, q1


class HyperMAMA(Indicator):
    """
    HyperMAMA (Hyper Mother of Adaptive Moving Average) インジケーター
    
    X_MAMAをベースとし、HyperERの値に基づいてfastlimitとslowlimitを動的適応させる拡張版：
    - HyperERインジケーターによる効率性測定
    - HyperERが0.8以上で最大値、0.2以下で最小値となる動的適応
    - fastlimitの範囲: 0.1〜0.5
    - slowlimitの範囲: 0.01〜0.05
    - カルマンフィルターによる価格ソースの前処理（オプション）
    - ゼロラグ処理による応答性の向上（オプション）
    
    特徴:
    - HyperERベースの高度な適応性
    - 市場効率性に応じた応答速度の自動調整
    - トレンド強度に応じたノイズフィルタリング
    - カルマンフィルターとゼロラグ処理の統合
    """
    
    def __init__(
        self,
        # 動的適応のトリガータイプ
        trigger_type: str = 'hyper_er',          # 'hyper_er' または 'hyper_trend_index'
        
        # HyperER関連パラメータ
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
        hyper_er_use_laguerre_filter: bool = True,  # ラゲールフィルターを使用するか
        hyper_er_laguerre_gamma: float = 0.5,  # ラゲールフィルターのガンマパラメータ
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
        use_zero_lag: bool = True                # ゼロラグ処理を使用するか
    ):
        """
        コンストラクタ
        
        Args:
            hyper_er_period: HyperER計算期間（デフォルト: 14）
            hyper_er_midline_period: HyperERミッドライン計算期間（デフォルト: 100）
            fast_max: fastlimitの最大値（デフォルト: 0.5）
            fast_min: fastlimitの最小値（デフォルト: 0.1）
            slow_max: slowlimitの最大値（デフォルト: 0.05）
            slow_min: slowlimitの最小値（デフォルト: 0.01）
            er_high_threshold: HyperERの高閾値（デフォルト: 0.8）
            er_low_threshold: HyperERの低閾値（デフォルト: 0.2）
            src_type: ソースタイプ
                基本ソース: 'close', 'hlc3', 'hl2', 'ohlc4', 'high', 'low', 'open'
            use_kalman_filter: カルマンフィルターを使用するか（デフォルト: False）
            kalman_filter_type: カルマンフィルタータイプ（デフォルト: 'unscented'）
            kalman_process_noise: プロセスノイズ（デフォルト: 0.01）
            kalman_observation_noise: 観測ノイズ（デフォルト: 0.001）
            use_zero_lag: ゼロラグ処理を使用するか（デフォルト: True）
        """
        # パラメータを保存（インジケーター名作成前に必要）
        self.trigger_type = trigger_type.lower()
        
        # インジケーター名の作成
        trigger_period = hyper_er_period if self.trigger_type == 'hyper_er' else hyper_trend_period
        indicator_name = f"HyperMAMA(trigger={self.trigger_type}, period={trigger_period}, fast={fast_min}-{fast_max}, slow={slow_min}-{slow_max}, {src_type}"
        if use_kalman_filter:
            indicator_name += f", kalman={kalman_filter_type}"
        if use_zero_lag:
            indicator_name += ", zero_lag=True"
        indicator_name += ")"
        
        super().__init__(indicator_name)
        self.hyper_er_period = hyper_er_period
        self.hyper_er_midline_period = hyper_er_midline_period
        self.fast_max = fast_max
        self.fast_min = fast_min
        self.slow_max = slow_max
        self.slow_min = slow_min
        self.er_high_threshold = er_high_threshold
        self.er_low_threshold = er_low_threshold
        self.src_type = src_type.lower()
        self.use_kalman_filter = use_kalman_filter
        self.kalman_filter_type = kalman_filter_type
        self.kalman_process_noise = kalman_process_noise
        self.kalman_observation_noise = kalman_observation_noise
        self.use_zero_lag = use_zero_lag
        
        # HyperTrendIndexパラメータを保存
        self.hyper_trend_period = hyper_trend_period
        self.hyper_trend_midline_period = hyper_trend_midline_period
        self.hyper_trend_src_type = hyper_trend_src_type
        self.hyper_trend_use_kalman_filter = hyper_trend_use_kalman_filter
        self.hyper_trend_kalman_filter_type = hyper_trend_kalman_filter_type
        self.hyper_trend_use_dynamic_period = hyper_trend_use_dynamic_period
        self.hyper_trend_detector_type = hyper_trend_detector_type
        self.hyper_trend_use_roofing_filter = hyper_trend_use_roofing_filter
        self.hyper_trend_roofing_hp_cutoff = hyper_trend_roofing_hp_cutoff
        self.hyper_trend_roofing_ss_band_edge = hyper_trend_roofing_ss_band_edge
        
        # パラメータ検証
        if self.trigger_type not in ['hyper_er', 'hyper_trend_index']:
            raise ValueError(f"無効なトリガータイプです: {trigger_type}。有効なオプション: 'hyper_er', 'hyper_trend_index'")
        if fast_max <= 0 or fast_max > 1:
            raise ValueError("fast_maxは0より大きく1以下である必要があります")
        if fast_min <= 0 or fast_min > 1:
            raise ValueError("fast_minは0より大きく1以下である必要があります")
        if slow_max <= 0 or slow_max > 1:
            raise ValueError("slow_maxは0より大きく1以下である必要があります")
        if slow_min <= 0 or slow_min > 1:
            raise ValueError("slow_minは0より大きく1以下である必要があります")
        if fast_min >= fast_max:
            raise ValueError("fast_minはfast_maxより小さい必要があります")
        if slow_min >= slow_max:
            raise ValueError("slow_minはslow_maxより小さい必要があります")
        if er_low_threshold >= er_high_threshold:
            raise ValueError("er_low_thresholdはer_high_thresholdより小さい必要があります")
        if use_kalman_filter and kalman_process_noise <= 0:
            raise ValueError("kalman_process_noiseは0より大きい必要があります")
        
        # ソースタイプの検証（PriceSourceから利用可能なタイプを取得）
        try:
            available_sources = PriceSource.get_available_sources()
            if self.src_type not in available_sources:
                raise ValueError(f"無効なソースタイプです: {src_type}。有効なオプション: {', '.join(available_sources.keys())}")
        except AttributeError:
            # get_available_sources()がない場合は基本的なソースタイプのみチェック
            basic_sources = ['close', 'high', 'low', 'open', 'hl2', 'hlc3', 'ohlc4']
            if self.src_type not in basic_sources:
                raise ValueError(f"無効なソースタイプです: {src_type}。有効なオプション: {', '.join(basic_sources)}")
        
        # トリガーインジケーターの初期化
        self._init_trigger_indicator(
            hyper_er_er_period,
            hyper_er_src_type,
            hyper_er_use_roofing_filter,
            hyper_er_roofing_hp_cutoff,
            hyper_er_roofing_ss_band_edge,
            hyper_er_use_laguerre_filter,
            hyper_er_laguerre_gamma,
            hyper_er_use_smoothing,
            hyper_er_smoother_type,
            hyper_er_smoother_period,
            hyper_er_smoother_src_type,
            hyper_er_use_dynamic_period,
            hyper_er_detector_type,
            hyper_er_lp_period,
            hyper_er_hp_period,
            hyper_er_cycle_part,
            hyper_er_max_cycle,
            hyper_er_min_cycle,
            hyper_er_max_output,
            hyper_er_min_output,
            hyper_er_enable_percentile_analysis,
            hyper_er_percentile_lookback_period,
            hyper_er_percentile_low_threshold,
            hyper_er_percentile_high_threshold
        )
        
        # カルマンフィルターの初期化（オプション）
        self.kalman_filter = None
        if self.use_kalman_filter:
            if not UNIFIED_KALMAN_AVAILABLE:
                self.logger.error("統合カルマンフィルターが利用できません。indicators.kalman.unified_kalmanをインポートできません。")
                self.use_kalman_filter = False
                self.logger.warning("カルマンフィルター機能を無効にしました")
            else:
                try:
                    self.kalman_filter = UnifiedKalman(
                        filter_type=self.kalman_filter_type,
                        src_type=self.src_type,
                        process_noise_scale=self.kalman_process_noise,
                        observation_noise_scale=self.kalman_observation_noise
                    )
                    self.logger.info(f"統合カルマンフィルターを初期化しました: {self.kalman_filter_type}")
                except Exception as e:
                    self.logger.error(f"統合カルマンフィルターの初期化に失敗: {e}")
                    self.use_kalman_filter = False
                    self.logger.warning("カルマンフィルター機能を無効にしました")
        
        # 結果キャッシュ（サイズ制限付き）
        self._result_cache = {}
        self._max_cache_size = 20
        self._cache_keys = []
    
    def _init_trigger_indicator(
        self, 
        hyper_er_er_period,
        hyper_er_src_type,
        hyper_er_use_roofing_filter,
        hyper_er_roofing_hp_cutoff,
        hyper_er_roofing_ss_band_edge,
        hyper_er_use_laguerre_filter,
        hyper_er_laguerre_gamma,
        hyper_er_use_smoothing,
        hyper_er_smoother_type,
        hyper_er_smoother_period,
        hyper_er_smoother_src_type,
        hyper_er_use_dynamic_period,
        hyper_er_detector_type,
        hyper_er_lp_period,
        hyper_er_hp_period,
        hyper_er_cycle_part,
        hyper_er_max_cycle,
        hyper_er_min_cycle,
        hyper_er_max_output,
        hyper_er_min_output,
        hyper_er_enable_percentile_analysis,
        hyper_er_percentile_lookback_period,
        hyper_er_percentile_low_threshold,
        hyper_er_percentile_high_threshold
    ):
        """トリガータイプに応じてインジケーターを初期化"""
        if self.trigger_type == 'hyper_er':
            # HyperER インジケーターを初期化
            self.hyper_er = HyperER(
                period=self.hyper_er_period,
                midline_period=self.hyper_er_midline_period,
                # HyperERの詳細パラメータ
                er_period=hyper_er_er_period,
                er_src_type=hyper_er_src_type,
                # HyperER ルーフィングフィルターパラメータ
                use_roofing_filter=hyper_er_use_roofing_filter,
                roofing_hp_cutoff=hyper_er_roofing_hp_cutoff,
                roofing_ss_band_edge=hyper_er_roofing_ss_band_edge,
                # HyperER ラゲールフィルターパラメータ
                use_laguerre_filter=hyper_er_use_laguerre_filter,
                laguerre_gamma=hyper_er_laguerre_gamma,
                # HyperER 平滑化オプション
                use_smoothing=hyper_er_use_smoothing,
                smoother_type=hyper_er_smoother_type,
                smoother_period=hyper_er_smoother_period,
                smoother_src_type=hyper_er_smoother_src_type,
                # HyperER エラーズ統合サイクル検出器パラメータ
                use_dynamic_period=hyper_er_use_dynamic_period,
                detector_type=hyper_er_detector_type,
                lp_period=hyper_er_lp_period,
                hp_period=hyper_er_hp_period,
                cycle_part=hyper_er_cycle_part,
                max_cycle=hyper_er_max_cycle,
                min_cycle=hyper_er_min_cycle,
                max_output=hyper_er_max_output,
                min_output=hyper_er_min_output,
                # HyperER パーセンタイルベーストレンド分析パラメータ
                enable_percentile_analysis=hyper_er_enable_percentile_analysis,
                percentile_lookback_period=hyper_er_percentile_lookback_period,
                percentile_low_threshold=hyper_er_percentile_low_threshold,
                percentile_high_threshold=hyper_er_percentile_high_threshold
            )
            self.trigger_indicator = self.hyper_er
            
        elif self.trigger_type == 'hyper_trend_index':
            # HyperTrendIndex インジケーターを初期化
            self.hyper_trend_index = HyperTrendIndex(
                period=self.hyper_trend_period,
                midline_period=self.hyper_trend_midline_period,
                src_type=self.hyper_trend_src_type,
                use_kalman_filter=self.hyper_trend_use_kalman_filter,
                kalman_filter_type=self.hyper_trend_kalman_filter_type,
                use_dynamic_period=self.hyper_trend_use_dynamic_period,
                detector_type=self.hyper_trend_detector_type,
                use_roofing_filter=self.hyper_trend_use_roofing_filter,
                roofing_hp_cutoff=self.hyper_trend_roofing_hp_cutoff,
                roofing_ss_band_edge=self.hyper_trend_roofing_ss_band_edge
            )
            self.trigger_indicator = self.hyper_trend_index
            
        else:
            raise ValueError(f"サポートされていないトリガータイプ: {self.trigger_type}")
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """
        データのハッシュ値を計算してキャッシュに使用する（超高速版）
        
        Args:
            data: 価格データ
            
        Returns:
            データハッシュ文字列
        """
        # 超高速化のため最小限のサンプリング
        try:
            # データ情報の取得
            if isinstance(data, pd.DataFrame):
                length = len(data)
                first_val = float(data.iloc[0].get('close', data.iloc[0, -1])) if length > 0 else 0.0
                last_val = float(data.iloc[-1].get('close', data.iloc[-1, -1])) if length > 0 else 0.0
            else:
                length = len(data)
                if length > 0:
                    if data.ndim > 1:
                        first_val = float(data[0, -1])
                        last_val = float(data[-1, -1])
                    else:
                        first_val = float(data[0])
                        last_val = float(data[-1])
                else:
                    first_val = last_val = 0.0
            
            # 最小限のパラメータ情報
            kalman_sig = f"{self.kalman_filter_type}_{self.kalman_process_noise}" if self.use_kalman_filter else "None"
            zero_lag_sig = "True" if self.use_zero_lag else "False"
            params_sig = f"{self.fast_max}_{self.fast_min}_{self.slow_max}_{self.slow_min}_{self.src_type}_{kalman_sig}_{zero_lag_sig}"
            
            # 超高速ハッシュ
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(params_sig)}"
            
        except Exception:
            # フォールバック
            return f"{id(data)}_{self.fast_max}_{self.fast_min}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> HyperMAMAResult:
        """
        HyperMAMA/HyperFAMAを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、OHLC + 選択したソースタイプに必要なカラムが必要
        
        Returns:
            HyperMAMAResult: HyperMAMA/HyperFAMAの値と計算中間値を含む結果
        """
        try:
            # キャッシュチェック - 同じデータの場合は計算をスキップ（高速化）
            data_hash = self._get_data_hash(data)
            
            # キャッシュにある場合は取得して返す
            if data_hash in self._result_cache:
                # キャッシュキーの順序を更新（最も新しく使われたキーを最後に）
                if data_hash in self._cache_keys:
                    self._cache_keys.remove(data_hash)
                self._cache_keys.append(data_hash)
                cached_result = self._result_cache[data_hash]
                return HyperMAMAResult(
                    mama_values=cached_result.mama_values.copy(),
                    fama_values=cached_result.fama_values.copy(),
                    period_values=cached_result.period_values.copy(),
                    alpha_values=cached_result.alpha_values.copy(),
                    phase_values=cached_result.phase_values.copy(),
                    i1_values=cached_result.i1_values.copy(),
                    q1_values=cached_result.q1_values.copy(),
                    filtered_price=cached_result.filtered_price.copy(),
                    hyper_er_values=cached_result.hyper_er_values.copy(),
                    adaptive_fast_limits=cached_result.adaptive_fast_limits.copy(),
                    adaptive_slow_limits=cached_result.adaptive_slow_limits.copy()
                )
            
            # 1. トリガーインジケーターの計算
            trigger_result = self.trigger_indicator.calculate(data)
            trigger_values = trigger_result.values
            
            # 2. 動的適応されたfastlimitとslowlimitの計算
            adaptive_fast_limits, adaptive_slow_limits = calculate_adaptive_limits_numba(
                trigger_values,
                self.fast_max,
                self.fast_min,
                self.slow_max,
                self.slow_min,
                self.er_high_threshold,
                self.er_low_threshold
            )
            
            # 3. 価格ソースの計算
            price_source = PriceSource.calculate_source(data, self.src_type)
            
            # NumPy配列に変換（float64型で統一）
            price_source = np.asarray(price_source, dtype=np.float64)
            
            # 4. カルマンフィルターによる前処理（オプション）
            filtered_price = price_source.copy()
            if self.use_kalman_filter and self.kalman_filter is not None:
                try:
                    kalman_result = self.kalman_filter.calculate(data)
                    
                    # カルマンフィルター結果の詳細なデバッグ
                    self.logger.debug(f"カルマンフィルター結果タイプ: {type(kalman_result)}")
                    
                    # カルマンフィルター結果の形式を確認
                    kalman_values = None
                    
                    # UnifiedKalmanResult専用の値抽出
                    if hasattr(kalman_result, 'values'):
                        # UnifiedKalmanResult or 他の標準化結果の場合
                        kalman_values = kalman_result.values
                        self.logger.debug(f"values属性を使用: {type(kalman_values)}")
                    elif hasattr(kalman_result, 'filtered_values'):
                        # UKFResult や他のフィルター結果の場合
                        kalman_values = kalman_result.filtered_values
                        self.logger.debug(f"filtered_values属性を使用: {type(kalman_values)}")
                    elif isinstance(kalman_result, (np.ndarray, list)):
                        # 直接配列の場合
                        kalman_values = kalman_result
                        self.logger.debug(f"直接配列を使用: {type(kalman_values)}")
                    else:
                        # その他の場合
                        kalman_values = kalman_result
                        self.logger.debug(f"その他の形式を使用: {type(kalman_values)}")
                    
                    # Noneでない場合のみ処理続行
                    if kalman_values is not None:
                        # NumPy配列に変換（統一されたインターフェースで処理）
                        try:
                            # NumPy配列に変換
                            kalman_values = np.asarray(kalman_values, dtype=np.float64)
                            
                            # 配列の次元をチェック
                            if kalman_values.ndim == 0:
                                # スカラー値の場合はエラー
                                raise ValueError("カルマンフィルター結果がスカラー値です")
                            elif kalman_values.ndim > 1:
                                # 多次元配列の場合は1次元に変換
                                kalman_values = kalman_values.flatten()
                            
                            # フィルタリングされた価格の検証
                            if len(kalman_values) != len(price_source):
                                self.logger.warning(f"カルマンフィルター結果のサイズ不一致: {len(kalman_values)} != {len(price_source)}。元の価格を使用します。")
                                filtered_price = price_source.copy()
                            else:
                                # NaN値の処理
                                nan_mask = np.isnan(kalman_values)
                                if np.any(nan_mask):
                                    kalman_values[nan_mask] = price_source[nan_mask]
                                
                                filtered_price = kalman_values
                                self.logger.debug("カルマンフィルターによる価格前処理を適用しました")
                        except Exception as array_error:
                            self.logger.warning(f"カルマンフィルター結果の配列変換エラー: {array_error}。元の価格を使用します。")
                            filtered_price = price_source.copy()
                    else:
                        self.logger.warning("カルマンフィルター結果がNoneです。元の価格を使用します。")
                        filtered_price = price_source.copy()
                        
                except Exception as e:
                    self.logger.warning(f"カルマンフィルター適用中にエラー: {e}。元の価格を使用します。")
                    filtered_price = price_source.copy()
            
            # 5. データ長の検証
            data_length = len(filtered_price)
            if data_length == 0:
                raise ValueError("入力データが空です")
            
            if data_length < 10:
                self.logger.warning(f"データが短すぎます（{data_length}点）。最低10点以上を推奨します。")
            
            # 6. HyperMAMA/HyperFAMAの計算（Numba最適化関数を使用）
            mama_values, fama_values, period_values, alpha_values, phase_values, i1_values, q1_values = calculate_hyper_mama_fama(
                filtered_price, adaptive_fast_limits, adaptive_slow_limits, self.use_zero_lag
            )
            
            # 結果の保存（参照問題を避けるため必要な部分だけコピー）
            result = HyperMAMAResult(
                mama_values=mama_values.copy(),
                fama_values=fama_values.copy(),
                period_values=period_values.copy(),
                alpha_values=alpha_values.copy(),
                phase_values=phase_values.copy(),
                i1_values=i1_values.copy(),
                q1_values=q1_values.copy(),
                filtered_price=filtered_price.copy(),
                hyper_er_values=trigger_values.copy(),
                adaptive_fast_limits=adaptive_fast_limits.copy(),
                adaptive_slow_limits=adaptive_slow_limits.copy()
            )
            
            # キャッシュを更新
            # キャッシュサイズ管理
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                # 最も古いキャッシュを削除
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            self._values = mama_values  # 基底クラスの要件を満たすため（HyperMAMA値をメインとする）
            return result
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"HyperMAMA/HyperFAMA計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は空の結果を返す
            error_result = HyperMAMAResult(
                mama_values=np.array([]),
                fama_values=np.array([]),
                period_values=np.array([]),
                alpha_values=np.array([]),
                phase_values=np.array([]),
                i1_values=np.array([]),
                q1_values=np.array([]),
                filtered_price=np.array([]),
                hyper_er_values=np.array([]),
                adaptive_fast_limits=np.array([]),
                adaptive_slow_limits=np.array([])
            )
            return error_result
    
    def get_values(self) -> Optional[np.ndarray]:
        """HyperMAMA値のみを取得する（後方互換性のため）"""
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.mama_values.copy()
    
    def get_mama_values(self) -> Optional[np.ndarray]:
        """
        HyperMAMA値を取得する
        
        Returns:
            np.ndarray: HyperMAMA値
        """
        return self.get_values()
    
    def get_fama_values(self) -> Optional[np.ndarray]:
        """
        HyperFAMA値を取得する
        
        Returns:
            np.ndarray: HyperFAMA値
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.fama_values.copy()
    
    def get_period_values(self) -> Optional[np.ndarray]:
        """
        Period値を取得する
        
        Returns:
            np.ndarray: Period値
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.period_values.copy()
    
    def get_alpha_values(self) -> Optional[np.ndarray]:
        """
        Alpha値を取得する
        
        Returns:
            np.ndarray: Alpha値
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.alpha_values.copy()
    
    def get_phase_values(self) -> Optional[np.ndarray]:
        """
        Phase値を取得する
        
        Returns:
            np.ndarray: Phase値
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.phase_values.copy()
    
    def get_i1_values(self) -> Optional[np.ndarray]:
        """
        I1値を取得する
        
        Returns:
            np.ndarray: I1値
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.i1_values.copy()
    
    def get_q1_values(self) -> Optional[np.ndarray]:
        """
        Q1値を取得する
        
        Returns:
            np.ndarray: Q1値
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.q1_values.copy()
    
    def get_inphase_quadrature(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        InPhaseとQuadratureコンポーネントを取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (I1値, Q1値)
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.i1_values.copy(), result.q1_values.copy()
    
    def get_filtered_price(self) -> Optional[np.ndarray]:
        """
        カルマンフィルター後の価格を取得する
        
        Returns:
            np.ndarray: フィルタリングされた価格
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.filtered_price.copy()
    
    def get_hyper_er_values(self) -> Optional[np.ndarray]:
        """
        HyperER値を取得する
        
        Returns:
            np.ndarray: HyperER値
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.hyper_er_values.copy()
    
    def get_adaptive_limits(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        動的適応されたfastlimitとslowlimitを取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (adaptive_fast_limits, adaptive_slow_limits)
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.adaptive_fast_limits.copy(), result.adaptive_slow_limits.copy()
    
    def get_indicator_info(self) -> Dict[str, Any]:
        """インジケーター情報を取得"""
        return {
            'name': self.name,
            'trigger_type': self.trigger_type,
            'hyper_er_period': self.hyper_er_period,
            'hyper_er_midline_period': self.hyper_er_midline_period,
            'hyper_trend_period': self.hyper_trend_period,
            'hyper_trend_midline_period': self.hyper_trend_midline_period,
            'fast_max': self.fast_max,
            'fast_min': self.fast_min,
            'slow_max': self.slow_max,
            'slow_min': self.slow_min,
            'er_high_threshold': self.er_high_threshold,
            'er_low_threshold': self.er_low_threshold,
            'src_type': self.src_type,
            'use_kalman_filter': self.use_kalman_filter,
            'kalman_filter_type': self.kalman_filter_type if self.use_kalman_filter else None,
            'kalman_process_noise': self.kalman_process_noise if self.use_kalman_filter else None,
            'kalman_observation_noise': self.kalman_observation_noise if self.use_kalman_filter else None,
            'use_zero_lag': self.use_zero_lag,
            'description': 'HyperER適応型移動平均線（効率性ベース動的適応・カルマンフィルター・ゼロラグ処理対応）'
        }
    
    def reset(self) -> None:
        """
        インディケーターの状態をリセットする
        """
        super().reset()
        if hasattr(self, 'trigger_indicator') and self.trigger_indicator:
            self.trigger_indicator.reset()
        if self.kalman_filter:
            self.kalman_filter.reset()
        self._result_cache = {}
        self._cache_keys = []


# 便利関数
def calculate_hyper_mama(
    data: Union[pd.DataFrame, np.ndarray],
    trigger_type: str = 'hyper_er',
    hyper_er_period: int = 14,
    hyper_er_midline_period: int = 100,
    fast_max: float = 0.5,
    fast_min: float = 0.1,
    slow_max: float = 0.05,
    slow_min: float = 0.01,
    er_high_threshold: float = 0.8,
    er_low_threshold: float = 0.2,
    src_type: str = 'hlc3',
    use_kalman_filter: bool = False,
    kalman_filter_type: str = 'unscented',
    kalman_process_noise: float = 0.01,
    kalman_observation_noise: float = 0.001,
    use_zero_lag: bool = True,
    **kwargs
) -> np.ndarray:
    """
    HyperMAMAの計算（便利関数）
    
    Args:
        data: 価格データ
        trigger_type: トリガータイプ ('hyper_er' または 'hyper_trend_index')
        hyper_er_period: HyperER計算期間
        hyper_er_midline_period: HyperERミッドライン計算期間
        fast_max: fastlimitの最大値
        fast_min: fastlimitの最小値
        slow_max: slowlimitの最大値
        slow_min: slowlimitの最小値
        er_high_threshold: HyperERの高閾値
        er_low_threshold: HyperERの低閾値
        src_type: ソースタイプ
        use_kalman_filter: カルマンフィルターを使用するか
        kalman_filter_type: カルマンフィルタータイプ
        kalman_process_noise: プロセスノイズ
        kalman_observation_noise: 観測ノイズ
        use_zero_lag: ゼロラグ処理を使用するか
        **kwargs: その他のパラメータ
        
    Returns:
        HyperMAMA値
    """
    indicator = HyperMAMA(
        trigger_type=trigger_type,
        hyper_er_period=hyper_er_period,
        hyper_er_midline_period=hyper_er_midline_period,
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
        **kwargs
    )
    result = indicator.calculate(data)
    return result.mama_values


if __name__ == "__main__":
    """直接実行時のテスト"""
    import numpy as np
    import pandas as pd
    
    print("=== HyperMAMA インジケーターのテスト ===")
    
    # テストデータ生成
    np.random.seed(42)
    length = 200
    base_price = 100.0
    
    # トレンドとレンジが混在するデータを生成
    prices = [base_price]
    for i in range(1, length):
        if i < 50:  # トレンド相場
            change = 0.002 + np.random.normal(0, 0.01)
        elif i < 100:  # レンジ相場
            change = np.random.normal(0, 0.008)
        elif i < 150:  # 強いトレンド相場
            change = 0.004 + np.random.normal(0, 0.015)
        else:  # レンジ相場
            change = np.random.normal(0, 0.006)
        
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # OHLC データの生成
    data = []
    for i, close in enumerate(prices):
        daily_range = abs(np.random.normal(0, close * 0.01))
        
        high = close + daily_range * np.random.uniform(0.3, 1.0)
        low = close - daily_range * np.random.uniform(0.3, 1.0)
        
        if i == 0:
            open_price = close
        else:
            gap = np.random.normal(0, close * 0.005)
            open_price = prices[i-1] + gap
        
        # 論理的整合性の確保
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.uniform(1000, 10000)
        })
    
    df = pd.DataFrame(data)
    
    print(f"テストデータ: {len(df)}ポイント")
    print(f"価格範囲: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    # 基本版HyperMAMAをテスト
    print("\n基本版HyperMAMAをテスト中...")
    hyper_mama_basic = HyperMAMA(
        hyper_er_period=14,
        hyper_er_midline_period=100,
        fast_max=0.5,
        fast_min=0.1,
        slow_max=0.05,
        slow_min=0.01,
        src_type='hlc3',
        use_kalman_filter=False,
        use_zero_lag=False
    )
    try:
        result_basic = hyper_mama_basic.calculate(df)
        print(f"  HyperMAMA結果の型: {type(result_basic)}")
        print(f"  MAMA配列の形状: {result_basic.mama_values.shape}")
        print(f"  FAMA配列の形状: {result_basic.fama_values.shape}")
        print(f"  HyperER配列の形状: {result_basic.hyper_er_values.shape}")
    except Exception as e:
        print(f"  エラー: {e}")
        import traceback
        traceback.print_exc()
        result_basic = None
    
    if result_basic is not None:
        valid_count = np.sum(~np.isnan(result_basic.mama_values))
        mean_mama = np.nanmean(result_basic.mama_values)
        mean_fama = np.nanmean(result_basic.fama_values)
        mean_hyper_er = np.nanmean(result_basic.hyper_er_values)
        mean_fast_limit = np.nanmean(result_basic.adaptive_fast_limits)
        mean_slow_limit = np.nanmean(result_basic.adaptive_slow_limits)
        
        print(f"  有効値数: {valid_count}/{len(df)}")
        print(f"  平均HyperMAMA: {mean_mama:.4f}")
        print(f"  平均HyperFAMA: {mean_fama:.4f}")
        print(f"  平均HyperER: {mean_hyper_er:.4f}")
        print(f"  平均動的FastLimit: {mean_fast_limit:.4f}")
        print(f"  平均動的SlowLimit: {mean_slow_limit:.4f}")
    else:
        print("  基本版HyperMAMAの計算に失敗しました")
    
    # ゼロラグ処理版をテスト
    print("\nゼロラグ処理版HyperMAMAをテスト中...")
    hyper_mama_zero_lag = HyperMAMA(
        hyper_er_period=14,
        hyper_er_midline_period=100,
        fast_max=0.5,
        fast_min=0.1,
        slow_max=0.05,
        slow_min=0.01,
        src_type='hlc3',
        use_kalman_filter=False,
        use_zero_lag=True
    )
    try:
        result_zero_lag = hyper_mama_zero_lag.calculate(df)
        
        valid_count_zero_lag = np.sum(~np.isnan(result_zero_lag.mama_values))
        mean_mama_zero_lag = np.nanmean(result_zero_lag.mama_values)
        mean_fama_zero_lag = np.nanmean(result_zero_lag.fama_values)
        
        print(f"  有効値数: {valid_count_zero_lag}/{len(df)}")
        print(f"  平均HyperMAMA（ゼロラグ）: {mean_mama_zero_lag:.4f}")
        print(f"  平均HyperFAMA（ゼロラグ）: {mean_fama_zero_lag:.4f}")
        
        # 比較統計
        if result_basic is not None and valid_count > 0 and valid_count_zero_lag > 0:
            min_length = min(valid_count, valid_count_zero_lag)
            correlation = np.corrcoef(
                result_basic.mama_values[~np.isnan(result_basic.mama_values)][-min_length:],
                result_zero_lag.mama_values[~np.isnan(result_zero_lag.mama_values)][-min_length:]
            )[0, 1]
            print(f"  基本版とゼロラグ版の相関: {correlation:.4f}")
    except Exception as e:
        print(f"  ゼロラグ処理版でエラー: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== テスト完了 ===")