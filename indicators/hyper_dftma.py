#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional, List, NamedTuple, Any
import numpy as np
import pandas as pd
from numba import jit, prange, vectorize, njit, float64, types
import traceback
import math

# パス調整
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from .indicator import Indicator
    from .price_source import PriceSource
    from .trend_filter.hyper_er import HyperER
    from .kalman.unified_kalman import UnifiedKalman
    UNIFIED_KALMAN_AVAILABLE = True
except ImportError:
    try:
        from indicators.indicator import Indicator
        from indicators.price_source import PriceSource
        from indicators.trend_filter.hyper_er import HyperER
        from indicators.kalman.unified_kalman import UnifiedKalman
        UNIFIED_KALMAN_AVAILABLE = True
    except ImportError:
        try:
            from indicators.indicator import Indicator
            from indicators.price_source import PriceSource
            from indicators.trend_filter.hyper_er import HyperER
            UnifiedKalman = None
            UNIFIED_KALMAN_AVAILABLE = False
        except ImportError:
            raise ImportError("必要なモジュールをインポートできません")


@dataclass
class HyperDFTMAResult:
    """HyperDFTMA/HyperDFTFAMAの計算結果"""
    dftma_values: np.ndarray         # HyperDFTMAライン値
    dftfama_values: np.ndarray       # HyperDFTFAMAライン値
    period_values: np.ndarray        # DFTで計算されたPeriod値
    alpha_values: np.ndarray         # 計算されたAlpha値
    dominant_cycle: np.ndarray       # ドミナントサイクル値
    raw_period: np.ndarray          # 生の周期値
    filtered_price: np.ndarray       # カルマンフィルター後の価格（使用した場合）
    hyper_er_values: np.ndarray      # HyperER値
    adaptive_fast_limits: np.ndarray  # 動的適応されたfastlimit値
    adaptive_slow_limits: np.ndarray  # 動的適応されたslowlimit値


@njit(fastmath=True, cache=True)
def calculate_dft_dominant_cycle_numba(
    price: np.ndarray,
    window: int = 50,
    cycle_part: float = 0.5,
    max_output: int = 34,
    min_output: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    離散フーリエ変換ドミナントサイクル検出のNumba実装
    
    Args:
        price: 価格データの配列
        window: 分析ウィンドウ長
        cycle_part: サイクル部分の倍率
        max_output: 最大出力値
        min_output: 最小出力値
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (ドミナントサイクル, 生の周期)
    """
    n = len(price)
    pi = 2 * np.arcsin(1.0)
    
    # 初期化
    hp = np.zeros(n)
    cleaned_data = np.zeros(n)
    dominant_cycle_values = np.zeros(n)
    dom_cycle = np.zeros(n)
    
    for i in range(n):
        # ハイパスフィルターで40期間カットオフでデトレンド
        if i <= 5:
            hp[i] = price[i]
            cleaned_data[i] = price[i]
        else:
            per = 2 * pi / 40
            cos_per = np.cos(per)
            if cos_per != 0:
                alpha1 = (1 - np.sin(per)) / cos_per
            else:
                alpha1 = 0.0
            
            hp[i] = 0.5 * (1 + alpha1) * (price[i] - price[i-1]) + alpha1 * hp[i-1]
            
            # 6タップローパスFIRフィルター
            if i >= 5:
                cleaned_data[i] = (hp[i] + 2 * hp[i-1] + 3 * hp[i-2] + 3 * hp[i-3] + 2 * hp[i-4] + hp[i-5]) / 12
            else:
                cleaned_data[i] = hp[i]
        
        # DFT計算
        if i >= window:
            cosine_part = np.zeros(52)
            sine_part = np.zeros(52)
            pwr = np.zeros(52)
            db = np.zeros(52)
            
            # 各期間に対してDFT計算
            for period in range(8, 51):
                for k in range(window):
                    if i - k >= 0:
                        cyc_per = 2 * pi * k / period
                        cosine_part[period] += cleaned_data[i - k] * np.cos(cyc_per)
                        sine_part[period] += cleaned_data[i - k] * np.sin(cyc_per)
                
                pwr[period] = cosine_part[period] ** 2 + sine_part[period] ** 2
            
            # 正規化のための最大パワーレベルを見つける
            max_pwr = pwr[8]
            for period in range(8, 51):
                if pwr[period] > max_pwr:
                    max_pwr = pwr[period]
            
            # パワーレベルを正規化してデシベルに変換
            for period in range(8, 51):
                if max_pwr > 0 and pwr[period] > 0:
                    ratio = pwr[period] / max_pwr
                    if ratio > 0.01:  # 分母が0になるのを防ぐ
                        db[period] = -10 * np.log10(0.01 / (1 - 0.99 * ratio))
                    else:
                        db[period] = 20
                    
                    if db[period] > 20:
                        db[period] = 20
            
            # 重心アルゴリズムを使用してドミナントサイクルを見つける
            num = 0.0
            denom = 0.0
            for period in range(8, 51):
                if db[period] < 3:
                    three_minus = 3 - db[period]
                    num += period * three_minus
                    denom += three_minus
            
            if denom != 0:
                dominant_cycle_values[i] = num / denom
            else:
                dominant_cycle_values[i] = dominant_cycle_values[i-1] if i > 0 else 15.0
            
            # 最終出力計算
            dc_output = int(np.ceil(cycle_part * dominant_cycle_values[i]))
            if dc_output > max_output:
                dom_cycle[i] = max_output
            elif dc_output < min_output:
                dom_cycle[i] = min_output
            else:
                dom_cycle[i] = dc_output
        else:
            # 初期値
            dominant_cycle_values[i] = 15.0
            dom_cycle[i] = int(np.ceil(cycle_part * 15.0))
    
    return dom_cycle, dominant_cycle_values


@njit(fastmath=True, cache=True)
def calculate_zero_lag_processing(dftma_values: np.ndarray, alpha_values: np.ndarray) -> np.ndarray:
    """
    HyperDFTMA値に対してゼロラグ処理を適用する（Numba最適化版）
    HyperDFTMA内部のアルファ値を使用してゼロラグ処理を実行
    
    Args:
        dftma_values: HyperDFTMA値の配列
        alpha_values: HyperDFTMA内部のアルファ値の配列
    
    Returns:
        ゼロラグ処理後の値配列
    """
    length = len(dftma_values)
    result = np.full(length, np.nan, dtype=np.float64)
    
    if length == 0:
        return result
    
    # EMA値の配列（HyperDFTMA値のEMA）
    ema_values = np.full(length, np.nan, dtype=np.float64)
    
    # ラグ除去データの配列
    lag_reduced_data = np.full(length, np.nan, dtype=np.float64)
    
    # 最初の有効な値を見つける
    first_valid_idx = -1
    for i in range(length):
        if not np.isnan(dftma_values[i]):
            ema_values[i] = dftma_values[i]
            first_valid_idx = i
            break
    
    if first_valid_idx == -1:
        return result
    
    # EMAの計算（HyperDFTMA値のEMA、HyperDFTMAのアルファ値を使用）
    for i in range(first_valid_idx + 1, length):
        if not np.isnan(dftma_values[i]) and not np.isnan(alpha_values[i]):
            if not np.isnan(ema_values[i-1]):
                ema_values[i] = alpha_values[i] * dftma_values[i] + (1.0 - alpha_values[i]) * ema_values[i-1]
            else:
                ema_values[i] = dftma_values[i]
    
    # ラグ除去データの計算
    for i in range(length):
        if not np.isnan(dftma_values[i]) and not np.isnan(ema_values[i]):
            lag_reduced_data[i] = 2.0 * dftma_values[i] - ema_values[i]
    
    # ZLEMAの計算
    # 最初の値はラグ除去データと同じ
    start_idx = first_valid_idx
    if start_idx < length and not np.isnan(lag_reduced_data[start_idx]):
        result[start_idx] = lag_reduced_data[start_idx]
    
    # 以降はラグ除去データのEMAを計算（HyperDFTMAのアルファ値を使用）
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
def calculate_hyper_dftma_fama(
    price: np.ndarray,
    dominant_cycle: np.ndarray,
    adaptive_fast_limits: np.ndarray,
    adaptive_slow_limits: np.ndarray,
    use_zero_lag: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    HyperDFTMA/HyperDFTFAMAを計算する（Numba最適化版）
    DFTで計算されたドミナントサイクルを使用してアルファ値を動的に調整する
    """
    length = len(price)
    
    # 変数の初期化
    alpha = np.zeros(length, dtype=np.float64)
    dftma = np.zeros(length, dtype=np.float64)
    dftfama = np.zeros(length, dtype=np.float64)
    
    # 初期値設定
    for i in range(min(5, length)):
        # 初期アルファ値は動的slow_limitを使用
        initial_slow = adaptive_slow_limits[i] if not np.isnan(adaptive_slow_limits[i]) else 0.05
        alpha[i] = initial_slow
        dftma[i] = price[i] if i < length else 100.0
        dftfama[i] = price[i] if i < length else 100.0
    
    # メイン計算
    for i in range(5, length):
        # DFTベースの動的適応アルファ計算
        current_fast_limit = adaptive_fast_limits[i] if not np.isnan(adaptive_fast_limits[i]) else 0.5
        current_slow_limit = adaptive_slow_limits[i] if not np.isnan(adaptive_slow_limits[i]) else 0.05
        
        # ドミナントサイクルに基づくアルファ値計算
        if not np.isnan(dominant_cycle[i]) and dominant_cycle[i] > 0:
            # DFTサイクル期間を使用してアルファ値を動的に調整
            cycle_alpha = 2.0 / (dominant_cycle[i] + 1.0)
            
            # HyperERによる適応limitを適用
            if cycle_alpha > current_fast_limit:
                alpha[i] = current_fast_limit
            elif cycle_alpha < current_slow_limit:
                alpha[i] = current_slow_limit
            else:
                alpha[i] = cycle_alpha
        else:
            alpha[i] = current_slow_limit
        
        # HyperDFTMA計算
        if i > 4 and not np.isnan(dftma[i-1]) and not np.isnan(alpha[i]):
            dftma[i] = alpha[i] * price[i] + (1.0 - alpha[i]) * dftma[i-1]
        else:
            dftma[i] = price[i]  # 初期値として価格を使用
        
        # HyperDFTFAMA計算
        if i > 4 and not np.isnan(dftfama[i-1]) and not np.isnan(dftma[i]) and not np.isnan(alpha[i]):
            dftfama[i] = 0.5 * alpha[i] * dftma[i] + (1.0 - 0.5 * alpha[i]) * dftfama[i-1]
        else:
            dftfama[i] = dftma[i]  # 初期値としてDFTMA値を使用
    
    # ゼロラグ処理の適用（オプション）
    if use_zero_lag:
        # HyperDFTMAにゼロラグ処理を適用
        dftma_zero_lag = calculate_zero_lag_processing(dftma, alpha)
        
        # HyperDFTFAMAにゼロラグ処理を適用
        dftfama_zero_lag = calculate_zero_lag_processing(dftfama, alpha)
        
        # 有効な値のみを使用（NaN値は元の値を保持）
        for i in range(length):
            if not np.isnan(dftma_zero_lag[i]):
                dftma[i] = dftma_zero_lag[i]
            
            if not np.isnan(dftfama_zero_lag[i]):
                dftfama[i] = dftfama_zero_lag[i]
    
    return dftma, dftfama, alpha


class HyperDFTMA(Indicator):
    """
    HyperDFTMA (Hyper Discrete Fourier Transform Moving Average) インジケーター
    
    HyperMAMAをベースとし、離散フーリエ変換（DFT）アルゴリズムによる
    ドミナントサイクル検出とHyperERの値に基づく動的適応移動平均線：
    - DFTベースのドミナントサイクル検出による動的期間適応
    - HyperERインジケーターによる効率性測定
    - HyperERが0.8以上で最大値、0.2以下で最小値となる動的適応
    - fastlimitの範囲: 0.1〜0.5
    - slowlimitの範囲: 0.01〜0.05
    - カルマンフィルターによる価格ソースの前処理（オプション）
    - ゼロラグ処理による応答性の向上（オプション）
    
    特徴:
    - DFTによる高精度なサイクル検出
    - HyperERベースの高度な適応性
    - 市場効率性に応じた応答速度の自動調整
    - トレンド強度に応じたノイズフィルタリング
    - カルマンフィルターとゼロラグ処理の統合
    """
    
    def __init__(
        self,
        # DFT関連パラメータ
        dft_window: int = 50,                    # DFT分析ウィンドウ長
        dft_cycle_part: float = 0.5,             # DFTサイクル部分の倍率
        dft_max_output: int = 34,                # DFT最大出力値
        dft_min_output: int = 1,                 # DFT最小出力値
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
        """
        # インジケーター名の作成
        indicator_name = f"HyperDFTMA(window={dft_window}, period={hyper_er_period}, fast={fast_min}-{fast_max}, slow={slow_min}-{slow_max}, {src_type}"
        if use_kalman_filter:
            indicator_name += f", kalman={kalman_filter_type}"
        if use_zero_lag:
            indicator_name += ", zero_lag=True"
        indicator_name += ")"
        
        super().__init__(indicator_name)
        
        # DFTパラメータを保存
        self.dft_window = dft_window
        self.dft_cycle_part = dft_cycle_part
        self.dft_max_output = dft_max_output
        self.dft_min_output = dft_min_output
        
        # HyperERパラメータを保存
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
        
        # パラメータ検証
        if dft_window <= 0:
            raise ValueError("dft_windowは0より大きい必要があります")
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
        
        # HyperERインジケーターの初期化
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
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """
        データのハッシュ値を計算してキャッシュに使用する（超高速版）
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
            params_sig = f"{self.dft_window}_{self.fast_max}_{self.fast_min}_{self.slow_max}_{self.slow_min}_{self.src_type}_{kalman_sig}_{zero_lag_sig}"
            
            # 超高速ハッシュ
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(params_sig)}"
            
        except Exception:
            # フォールバック
            return f"{id(data)}_{self.dft_window}_{self.fast_max}_{self.fast_min}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> HyperDFTMAResult:
        """
        HyperDFTMA/HyperDFTFAMAを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、OHLC + 選択したソースタイプに必要なカラムが必要
        
        Returns:
            HyperDFTMAResult: HyperDFTMA/HyperDFTFAMAの値と計算中間値を含む結果
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
                return HyperDFTMAResult(
                    dftma_values=cached_result.dftma_values.copy(),
                    dftfama_values=cached_result.dftfama_values.copy(),
                    period_values=cached_result.period_values.copy(),
                    alpha_values=cached_result.alpha_values.copy(),
                    dominant_cycle=cached_result.dominant_cycle.copy(),
                    raw_period=cached_result.raw_period.copy(),
                    filtered_price=cached_result.filtered_price.copy(),
                    hyper_er_values=cached_result.hyper_er_values.copy(),
                    adaptive_fast_limits=cached_result.adaptive_fast_limits.copy(),
                    adaptive_slow_limits=cached_result.adaptive_slow_limits.copy()
                )
            
            # 1. HyperERの計算
            hyper_er_result = self.hyper_er.calculate(data)
            hyper_er_values = hyper_er_result.values
            
            # 2. 動的適応されたfastlimitとslowlimitの計算
            adaptive_fast_limits, adaptive_slow_limits = calculate_adaptive_limits_numba(
                hyper_er_values,
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
            
            if data_length < max(self.dft_window, 10):
                self.logger.warning(f"データが短すぎます（{data_length}点）。最低{max(self.dft_window, 10)}点以上を推奨します。")
            
            # 6. DFTベースのドミナントサイクル検出
            dominant_cycle, raw_period = calculate_dft_dominant_cycle_numba(
                filtered_price,
                self.dft_window,
                self.dft_cycle_part,
                self.dft_max_output,
                self.dft_min_output
            )
            
            # 7. HyperDFTMA/HyperDFTFAMAの計算（Numba最適化関数を使用）
            dftma_values, dftfama_values, alpha_values = calculate_hyper_dftma_fama(
                filtered_price, dominant_cycle, adaptive_fast_limits, adaptive_slow_limits, self.use_zero_lag
            )
            
            # 結果の保存（参照問題を避けるため必要な部分だけコピー）
            result = HyperDFTMAResult(
                dftma_values=dftma_values.copy(),
                dftfama_values=dftfama_values.copy(),
                period_values=dominant_cycle.copy(),  # period_valuesとしてdominant_cycleを使用
                alpha_values=alpha_values.copy(),
                dominant_cycle=dominant_cycle.copy(),
                raw_period=raw_period.copy(),
                filtered_price=filtered_price.copy(),
                hyper_er_values=hyper_er_values.copy(),
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
            
            self._values = dftma_values  # 基底クラスの要件を満たすため（HyperDFTMA値をメインとする）
            return result
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"HyperDFTMA/HyperDFTFAMA計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は空の結果を返す
            error_result = HyperDFTMAResult(
                dftma_values=np.array([]),
                dftfama_values=np.array([]),
                period_values=np.array([]),
                alpha_values=np.array([]),
                dominant_cycle=np.array([]),
                raw_period=np.array([]),
                filtered_price=np.array([]),
                hyper_er_values=np.array([]),
                adaptive_fast_limits=np.array([]),
                adaptive_slow_limits=np.array([])
            )
            return error_result
    
    def get_values(self) -> Optional[np.ndarray]:
        """HyperDFTMA値のみを取得する（後方互換性のため）"""
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.dftma_values.copy()
    
    def get_dftma_values(self) -> Optional[np.ndarray]:
        """HyperDFTMA値を取得する"""
        return self.get_values()
    
    def get_dftfama_values(self) -> Optional[np.ndarray]:
        """HyperDFTFAMA値を取得する"""
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.dftfama_values.copy()
    
    def get_period_values(self) -> Optional[np.ndarray]:
        """Period値を取得する"""
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
        """Alpha値を取得する"""
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.alpha_values.copy()
    
    def get_dominant_cycle(self) -> Optional[np.ndarray]:
        """ドミナントサイクル値を取得する"""
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.dominant_cycle.copy()
    
    def get_raw_period(self) -> Optional[np.ndarray]:
        """生の周期値を取得する"""
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.raw_period.copy()
    
    def get_filtered_price(self) -> Optional[np.ndarray]:
        """カルマンフィルター後の価格を取得する"""
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
        """HyperER値を取得する"""
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
        """動的適応されたfastlimitとslowlimitを取得する"""
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
            'dft_window': self.dft_window,
            'dft_cycle_part': self.dft_cycle_part,
            'dft_max_output': self.dft_max_output,
            'dft_min_output': self.dft_min_output,
            'hyper_er_period': self.hyper_er_period,
            'hyper_er_midline_period': self.hyper_er_midline_period,
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
            'description': 'DFTベース効率性適応型移動平均線（離散フーリエ変換・HyperER動的適応・カルマンフィルター・ゼロラグ処理対応）'
        }
    
    def reset(self) -> None:
        """インディケーターの状態をリセットする"""
        super().reset()
        if self.hyper_er:
            self.hyper_er.reset()
        if self.kalman_filter:
            self.kalman_filter.reset()
        self._result_cache = {}
        self._cache_keys = []


# 便利関数
def calculate_hyper_dftma(
    data: Union[pd.DataFrame, np.ndarray],
    dft_window: int = 50,
    dft_cycle_part: float = 0.5,
    dft_max_output: int = 34,
    dft_min_output: int = 1,
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
    HyperDFTMAの計算（便利関数）
    """
    indicator = HyperDFTMA(
        dft_window=dft_window,
        dft_cycle_part=dft_cycle_part,
        dft_max_output=dft_max_output,
        dft_min_output=dft_min_output,
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
    return result.dftma_values


if __name__ == "__main__":
    """直接実行時のテスト"""
    import sys
    import os
    
    # パッケージルートを追加
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # 相対インポートを絶対インポートに変更
    from indicators.indicator import Indicator
    from indicators.price_source import PriceSource
    from indicators.trend_filter.hyper_er import HyperER
    
    import numpy as np
    import pandas as pd
    
    print("=== HyperDFTMA インジケーターのテスト ===")
    
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
    
    # 基本版HyperDFTMAをテスト
    print("\n基本版HyperDFTMAをテスト中...")
    hyper_dftma_basic = HyperDFTMA(
        dft_window=50,
        dft_cycle_part=0.5,
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
        result_basic = hyper_dftma_basic.calculate(df)
        print(f"  HyperDFTMA結果の型: {type(result_basic)}")
        print(f"  DFTMA配列の形状: {result_basic.dftma_values.shape}")
        print(f"  DFTFAMA配列の形状: {result_basic.dftfama_values.shape}")
        print(f"  HyperER配列の形状: {result_basic.hyper_er_values.shape}")
        print(f"  ドミナントサイクル配列の形状: {result_basic.dominant_cycle.shape}")
    except Exception as e:
        print(f"  エラー: {e}")
        import traceback
        traceback.print_exc()
        result_basic = None
    
    if result_basic is not None:
        valid_count = np.sum(~np.isnan(result_basic.dftma_values))
        mean_dftma = np.nanmean(result_basic.dftma_values)
        mean_dftfama = np.nanmean(result_basic.dftfama_values)
        mean_hyper_er = np.nanmean(result_basic.hyper_er_values)
        mean_dominant_cycle = np.nanmean(result_basic.dominant_cycle)
        mean_fast_limit = np.nanmean(result_basic.adaptive_fast_limits)
        mean_slow_limit = np.nanmean(result_basic.adaptive_slow_limits)
        
        print(f"  有効値数: {valid_count}/{len(df)}")
        print(f"  平均HyperDFTMA: {mean_dftma:.4f}")
        print(f"  平均HyperDFTFAMA: {mean_dftfama:.4f}")
        print(f"  平均HyperER: {mean_hyper_er:.4f}")
        print(f"  平均ドミナントサイクル: {mean_dominant_cycle:.4f}")
        print(f"  平均動的FastLimit: {mean_fast_limit:.4f}")
        print(f"  平均動的SlowLimit: {mean_slow_limit:.4f}")
    else:
        print("  基本版HyperDFTMAの計算に失敗しました")
    
    # ゼロラグ処理版をテスト
    print("\nゼロラグ処理版HyperDFTMAをテスト中...")
    hyper_dftma_zero_lag = HyperDFTMA(
        dft_window=50,
        dft_cycle_part=0.5,
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
        result_zero_lag = hyper_dftma_zero_lag.calculate(df)
        
        valid_count_zero_lag = np.sum(~np.isnan(result_zero_lag.dftma_values))
        mean_dftma_zero_lag = np.nanmean(result_zero_lag.dftma_values)
        mean_dftfama_zero_lag = np.nanmean(result_zero_lag.dftfama_values)
        
        print(f"  有効値数: {valid_count_zero_lag}/{len(df)}")
        print(f"  平均HyperDFTMA（ゼロラグ）: {mean_dftma_zero_lag:.4f}")
        print(f"  平均HyperDFTFAMA（ゼロラグ）: {mean_dftfama_zero_lag:.4f}")
        
        # 比較統計
        if result_basic is not None and valid_count > 0 and valid_count_zero_lag > 0:
            min_length = min(valid_count, valid_count_zero_lag)
            correlation = np.corrcoef(
                result_basic.dftma_values[~np.isnan(result_basic.dftma_values)][-min_length:],
                result_zero_lag.dftma_values[~np.isnan(result_zero_lag.dftma_values)][-min_length:]
            )[0, 1]
            print(f"  基本版とゼロラグ版の相関: {correlation:.4f}")
    except Exception as e:
        print(f"  ゼロラグ処理版でエラー: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== テスト完了 ===")