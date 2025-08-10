#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultimate Trend Range Filter - 宇宙最強のトレンド/レンジ判別インジケーター

Ultimate Chop Trendをベースに、トレンド（方向問わず）とレンジの2つに特化した究極の判別システム：
- 最新の数学・統計学・機械学習・信号処理アルゴリズムを統合
- トレンドの存在そのものを検出（方向性は問わない）
- 0=完全なレンジ、1=強いトレンドの明確な判定
- 超低遅延・超高精度・超高信頼性

【統合アルゴリズム】
🔬 Advanced Trend Detection - 高度トレンド検出システム
📊 Range Consolidation Analysis - レンジ統合解析
🧠 Multi-Scale Trend Filtering - マルチスケールトレンドフィルタリング
🎯 Adaptive Noise Suppression - 適応ノイズ抑制
📡 Harmonic Pattern Recognition - ハーモニックパターン認識
🌊 Volatility Regime Classification - ボラティリティレジーム分類
🚀 Machine Learning Feature Extraction - 機械学習特徴抽出
"""

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Any, Optional, NamedTuple, List
import numpy as np
import pandas as pd
from numba import jit, njit, prange
import traceback
import math
import warnings
warnings.filterwarnings("ignore")

# Base classes
try:
    from .indicator import Indicator
    from .atr import ATR
    from .cycle.ehlers_unified_dc import EhlersUnifiedDC
except ImportError:
    print("Warning: Could not import from relative path. Assuming base classes are available.")
    class Indicator:
        def __init__(self, name): self.name = name; self.logger = self._get_logger()
        def reset(self): pass
        def _get_logger(self): import logging; return logging.getLogger(self.__class__.__name__)
    class ATR:
        def __init__(self, **kwargs): pass
        def calculate(self, data): return None
        def get_values(self): return np.array([])
        def reset(self): pass
    class EhlersUnifiedDC:
        def __init__(self, **kwargs): pass
        def calculate(self, data): return np.full(len(data), 10.0)
        def reset(self): pass


class UltimateTrendRangeResult(NamedTuple):
    """Ultimate Trend Range Filter結果"""
    # メイン指標
    trend_strength: np.ndarray          # 0=レンジ, 1=強いトレンド
    trend_classification: np.ndarray    # 0=レンジ, 1=トレンド
    range_probability: np.ndarray       # レンジ確率（0-1）
    trend_probability: np.ndarray       # トレンド確率（0-1）
    
    # 信頼度・品質指標
    confidence_score: np.ndarray        # 判定信頼度（0-1）
    signal_quality: np.ndarray          # シグナル品質（0-1）
    noise_level: np.ndarray             # ノイズレベル（0-1）
    
    # 詳細成分
    directional_movement: np.ndarray    # 方向性移動量
    consolidation_index: np.ndarray     # 横ばい指数
    volatility_regime: np.ndarray       # ボラティリティレジーム
    
    # 高度解析成分
    harmonic_strength: np.ndarray       # ハーモニック強度
    fractal_dimension: np.ndarray       # フラクタル次元
    persistence_factor: np.ndarray      # 持続性要因
    market_microstructure: np.ndarray   # 市場マイクロ構造
    
    # 予測成分
    trend_continuation: np.ndarray      # トレンド継続予測
    regime_change_probability: np.ndarray # レジーム変化確率
    
    # 現在状態
    current_state: str                  # "trend" or "range"
    current_strength: float             # 現在の強度
    current_confidence: float           # 現在の信頼度


@njit(fastmath=True, cache=True)
def advanced_trend_detection(
    prices: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    period: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    高度トレンド検出システム
    
    Returns:
        (trend_strength, directional_movement, trend_consistency)
    """
    n = len(prices)
    if n < period * 2:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    trend_strength = np.full(n, np.nan)
    directional_movement = np.full(n, np.nan)
    trend_consistency = np.full(n, np.nan)
    
    for i in range(period, n):
        price_window = prices[i-period:i+1]
        high_window = high[i-period:i+1]
        low_window = low[i-period:i+1]
        
        # 1. 線形トレンド強度
        x_vals = np.arange(len(price_window))
        n_points = len(price_window)
        sum_x = np.sum(x_vals)
        sum_y = np.sum(price_window)
        sum_xy = np.sum(x_vals * price_window)
        sum_x2 = np.sum(x_vals * x_vals)
        
        denom = n_points * sum_x2 - sum_x * sum_x
        if abs(denom) > 1e-15:
            slope = (n_points * sum_xy - sum_x * sum_y) / denom
            intercept = (sum_y - slope * sum_x) / n_points
            
            # 回帰直線からの平均絶対偏差
            predicted = slope * x_vals + intercept
            mae = np.mean(np.abs(price_window - predicted))
            
            # 価格レンジに対する相対的な傾き
            price_range = np.max(price_window) - np.min(price_window)
            if price_range > 0:
                relative_slope = abs(slope * period) / price_range
                trend_strength[i] = min(relative_slope, 1.0)
            else:
                trend_strength[i] = 0.0
        else:
            trend_strength[i] = 0.0
            slope = 0.0
        
        # 2. 方向性移動量（ADXライクな計算）
        dm_plus = 0.0
        dm_minus = 0.0
        true_range_sum = 0.0
        
        for j in range(1, len(high_window)):
            high_diff = high_window[j] - high_window[j-1]
            low_diff = low_window[j-1] - low_window[j]
            
            if high_diff > low_diff and high_diff > 0:
                dm_plus += high_diff
            elif low_diff > high_diff and low_diff > 0:
                dm_minus += low_diff
                
            # True Range
            tr = max(
                high_window[j] - low_window[j],
                abs(high_window[j] - price_window[j-1]),
                abs(low_window[j] - price_window[j-1])
            )
            true_range_sum += tr
        
        if true_range_sum > 0:
            di_plus = dm_plus / true_range_sum
            di_minus = dm_minus / true_range_sum
            directional_movement[i] = abs(di_plus - di_minus)
        else:
            directional_movement[i] = 0.0
        
        # 3. トレンド一貫性（価格の単調性）
        price_changes = np.diff(price_window)
        if len(price_changes) > 0:
            positive_changes = np.sum(price_changes > 0)
            negative_changes = np.sum(price_changes < 0)
            total_changes = len(price_changes)
            
            if total_changes > 0:
                consistency = abs(positive_changes - negative_changes) / total_changes
                trend_consistency[i] = consistency
            else:
                trend_consistency[i] = 0.0
        else:
            trend_consistency[i] = 0.0
    
    return trend_strength, directional_movement, trend_consistency


@njit(fastmath=True, cache=True)
def range_consolidation_analysis(
    prices: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    period: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    レンジ統合解析システム
    
    Returns:
        (consolidation_index, range_tightness, sideways_strength)
    """
    n = len(prices)
    if n < period * 2:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    consolidation_index = np.full(n, np.nan)
    range_tightness = np.full(n, np.nan)
    sideways_strength = np.full(n, np.nan)
    
    for i in range(period, n):
        price_window = prices[i-period:i+1]
        high_window = high[i-period:i+1]
        low_window = low[i-period:i+1]
        
        # 1. 統合指数（価格の集中度）
        price_mean = np.mean(price_window)
        price_std = np.std(price_window)
        
        if price_std > 0:
            # 正規化された分散の逆数
            consolidation_index[i] = 1.0 / (1.0 + price_std / price_mean)
        else:
            consolidation_index[i] = 1.0
        
        # 2. レンジの密集度
        price_range = np.max(price_window) - np.min(price_window)
        high_low_range = np.max(high_window) - np.min(low_window)
        
        if high_low_range > 0:
            # レンジ内での価格分布の均等性
            range_tightness[i] = 1.0 - (price_range / high_low_range)
        else:
            range_tightness[i] = 1.0
        
        # 3. 横ばい強度（価格の戻り傾向）
        # 中央値からの平均距離
        price_median = np.median(price_window)
        avg_deviation = np.mean(np.abs(price_window - price_median))
        
        if price_range > 0:
            # 正規化された平均偏差
            sideways_strength[i] = 1.0 - (2.0 * avg_deviation / price_range)
            sideways_strength[i] = max(sideways_strength[i], 0.0)
        else:
            sideways_strength[i] = 1.0
    
    return consolidation_index, range_tightness, sideways_strength


@njit(fastmath=True, cache=True)
def multi_scale_trend_filtering(
    prices: np.ndarray,
    scales: np.ndarray = np.array([5, 10, 20, 40])
) -> Tuple[np.ndarray, np.ndarray]:
    """
    マルチスケールトレンドフィルタリング
    
    Returns:
        (multi_scale_trend, scale_consistency)
    """
    n = len(prices)
    if n < np.max(scales) * 2:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    multi_scale_trend = np.full(n, np.nan)
    scale_consistency = np.full(n, np.nan)
    
    for i in range(np.max(scales), n):
        scale_trends = []
        
        for scale in scales:
            if i >= scale:
                # スケール固有のトレンド計算
                price_segment = prices[i-scale:i+1]
                
                # 線形回帰の傾き
                x_vals = np.arange(len(price_segment))
                n_points = len(price_segment)
                sum_x = np.sum(x_vals)
                sum_y = np.sum(price_segment)
                sum_xy = np.sum(x_vals * price_segment)
                sum_x2 = np.sum(x_vals * x_vals)
                
                denom = n_points * sum_x2 - sum_x * sum_x
                if abs(denom) > 1e-15:
                    slope = (n_points * sum_xy - sum_x * sum_y) / denom
                    
                    # 正規化された傾き
                    price_range = np.max(price_segment) - np.min(price_segment)
                    if price_range > 0:
                        normalized_slope = (slope * scale) / price_range
                        scale_trends.append(normalized_slope)
                    else:
                        scale_trends.append(0.0)
                else:
                    scale_trends.append(0.0)
        
        if len(scale_trends) > 0:
            # マルチスケールトレンドの統合
            trend_magnitudes = np.array([abs(t) for t in scale_trends])
            multi_scale_trend[i] = np.mean(trend_magnitudes)
            
            # スケール間の一貫性
            if len(trend_magnitudes) > 1:
                trend_std = np.std(trend_magnitudes)
                trend_mean = np.mean(trend_magnitudes)
                if trend_mean > 0:
                    scale_consistency[i] = 1.0 - (trend_std / trend_mean)
                    scale_consistency[i] = max(scale_consistency[i], 0.0)
                else:
                    scale_consistency[i] = 1.0
            else:
                scale_consistency[i] = 1.0
        else:
            multi_scale_trend[i] = 0.0
            scale_consistency[i] = 0.0
    
    return multi_scale_trend, scale_consistency


@njit(fastmath=True, cache=True)
def adaptive_noise_suppression(
    signal: np.ndarray,
    volatility: np.ndarray,
    adaptation_factor: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    適応ノイズ抑制システム
    
    Returns:
        (denoised_signal, noise_level)
    """
    n = len(signal)
    if n < 10:
        return signal.copy(), np.zeros(n)
    
    denoised_signal = signal.copy()
    noise_level = np.zeros(n)
    
    # 適応フィルタリング
    for i in range(5, n):
        # 局所的な信号統計
        window = signal[i-5:i+1]
        local_mean = np.mean(window)
        local_std = np.std(window)
        
        # ボラティリティベースの適応
        vol_factor = min(volatility[i] / np.mean(volatility[max(0, i-20):i+1]), 3.0)
        
        # 適応しきい値
        threshold = local_std * vol_factor * adaptation_factor
        
        # ノイズ抑制
        signal_deviation = abs(signal[i] - local_mean)
        if signal_deviation > threshold:
            # 強いシグナルは保持
            denoised_signal[i] = signal[i]
            noise_level[i] = 0.0
        else:
            # 弱いシグナルはスムージング
            denoised_signal[i] = 0.7 * signal[i] + 0.3 * local_mean
            noise_level[i] = signal_deviation / threshold if threshold > 0 else 0.0
    
    return denoised_signal, noise_level


@njit(fastmath=True, cache=True)
def harmonic_pattern_recognition(
    prices: np.ndarray,
    period: int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ハーモニックパターン認識システム
    
    Returns:
        (harmonic_strength, pattern_type)
    """
    n = len(prices)
    if n < period * 2:
        return np.full(n, np.nan), np.full(n, 0.0)
    
    harmonic_strength = np.full(n, np.nan)
    pattern_type = np.full(n, 0.0)  # 0=なし, 1=支持抵抗, 2=三角形, 3=フラッグ
    
    for i in range(period, n):
        price_window = prices[i-period:i+1]
        
        # 1. 支持・抵抗レベルの検出
        price_max = np.max(price_window)
        price_min = np.min(price_window)
        price_range = price_max - price_min
        
        if price_range > 0:
            # 価格レベルの分布解析
            resistance_touches = 0
            support_touches = 0
            
            for price in price_window:
                if abs(price - price_max) / price_range < 0.02:  # 上限の2%以内
                    resistance_touches += 1
                elif abs(price - price_min) / price_range < 0.02:  # 下限の2%以内
                    support_touches += 1
            
            # 支持抵抗の強度
            total_points = len(price_window)
            support_resistance_strength = (resistance_touches + support_touches) / total_points
            
            # 2. 三角形パターン（収束）
            first_half = price_window[:period//2]
            second_half = price_window[period//2:]
            
            first_range = np.max(first_half) - np.min(first_half)
            second_range = np.max(second_half) - np.min(second_half)
            
            if first_range > 0:
                convergence_ratio = 1.0 - (second_range / first_range)
                convergence_ratio = max(convergence_ratio, 0.0)
            else:
                convergence_ratio = 0.0
            
            # 3. フラッグパターン（平行チャネル）
            # 上部と下部の線形回帰
            x_vals = np.arange(len(price_window))
            
            # 上部エンベロープ
            upper_envelope = []
            lower_envelope = []
            
            for j in range(len(price_window)):
                local_window = price_window[max(0, j-5):j+6]
                upper_envelope.append(np.max(local_window))
                lower_envelope.append(np.min(local_window))
            
            upper_envelope = np.array(upper_envelope)
            lower_envelope = np.array(lower_envelope)
            
            # 上下エンベロープの平行性
            upper_slope = 0.0
            lower_slope = 0.0
            
            if len(x_vals) > 1:
                # 簡易回帰
                n_points = len(x_vals)
                sum_x = np.sum(x_vals)
                
                # 上部の傾き
                sum_y_upper = np.sum(upper_envelope)
                sum_xy_upper = np.sum(x_vals * upper_envelope)
                sum_x2 = np.sum(x_vals * x_vals)
                
                denom = n_points * sum_x2 - sum_x * sum_x
                if abs(denom) > 1e-15:
                    upper_slope = (n_points * sum_xy_upper - sum_x * sum_y_upper) / denom
                
                # 下部の傾き
                sum_y_lower = np.sum(lower_envelope)
                sum_xy_lower = np.sum(x_vals * lower_envelope)
                
                if abs(denom) > 1e-15:
                    lower_slope = (n_points * sum_xy_lower - sum_x * sum_y_lower) / denom
            
            # 平行性の測定
            slope_difference = abs(upper_slope - lower_slope)
            if price_range > 0:
                slope_similarity = 1.0 / (1.0 + slope_difference * period / price_range)
            else:
                slope_similarity = 0.0
            
            # 最終的なハーモニック強度
            pattern_scores = [
                support_resistance_strength,
                convergence_ratio,
                slope_similarity
            ]
            
            max_score = max(pattern_scores)
            harmonic_strength[i] = max_score
            
            # パターンタイプの決定
            if max_score == support_resistance_strength and max_score > 0.3:
                pattern_type[i] = 1.0  # 支持抵抗
            elif max_score == convergence_ratio and max_score > 0.4:
                pattern_type[i] = 2.0  # 三角形
            elif max_score == slope_similarity and max_score > 0.6:
                pattern_type[i] = 3.0  # フラッグ
            else:
                pattern_type[i] = 0.0  # パターンなし
        else:
            harmonic_strength[i] = 0.0
            pattern_type[i] = 0.0
    
    return harmonic_strength, pattern_type 


@njit(fastmath=True, cache=True)
def volatility_regime_classification(
    prices: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    period: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ボラティリティレジーム分類システム
    
    Returns:
        (volatility_regime, regime_stability, regime_transition_prob)
    """
    n = len(prices)
    if n < period * 3:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    volatility_regime = np.full(n, np.nan)  # 0=低ボラ, 1=中ボラ, 2=高ボラ
    regime_stability = np.full(n, np.nan)
    regime_transition_prob = np.full(n, np.nan)
    
    for i in range(period * 2, n):
        # 真の値幅（True Range）の計算
        tr_values = []
        for j in range(i-period, i):
            if j > 0:
                tr = max(
                    high[j] - low[j],
                    abs(high[j] - prices[j-1]),
                    abs(low[j] - prices[j-1])
                )
                tr_values.append(tr)
        
        if len(tr_values) == 0:
            continue
            
        tr_array = np.array(tr_values)
        current_volatility = np.mean(tr_array)
        volatility_std = np.std(tr_array)
        
        # 長期ボラティリティとの比較
        long_term_period = min(period * 2, i)
        max_tr_count = max(1, long_term_period)
        long_term_tr = np.zeros(max_tr_count)
        tr_count = 0
        
        for j in range(i-long_term_period, i):
            if j > 0 and tr_count < max_tr_count:
                tr = max(
                    high[j] - low[j],
                    abs(high[j] - prices[j-1]),
                    abs(low[j] - prices[j-1])
                )
                long_term_tr[tr_count] = tr
                tr_count += 1
        
        if tr_count > 0:
            long_term_tr_valid = long_term_tr[:tr_count]
            long_term_vol = np.mean(long_term_tr_valid)
            long_term_std = np.std(long_term_tr_valid)
            
            # ボラティリティレジームの分類
            if long_term_std > 0:
                vol_z_score = (current_volatility - long_term_vol) / long_term_std
                
                if vol_z_score < -0.5:
                    volatility_regime[i] = 0.0  # 低ボラティリティ
                elif vol_z_score > 0.5:
                    volatility_regime[i] = 2.0  # 高ボラティリティ
                else:
                    volatility_regime[i] = 1.0  # 中ボラティリティ
                
                # レジームの安定性
                if volatility_std > 0:
                    stability = 1.0 / (1.0 + volatility_std / current_volatility)
                else:
                    stability = 1.0
                regime_stability[i] = stability
                
                # レジーム変化確率
                vol_change_rate = abs(vol_z_score)
                regime_transition_prob[i] = min(vol_change_rate / 2.0, 1.0)
            else:
                volatility_regime[i] = 1.0
                regime_stability[i] = 1.0
                regime_transition_prob[i] = 0.0
        else:
            volatility_regime[i] = 1.0
            regime_stability[i] = 0.5
            regime_transition_prob[i] = 0.5
    
    return volatility_regime, regime_stability, regime_transition_prob


@njit(fastmath=True, cache=True)
def ml_feature_extraction(
    prices: np.ndarray,
    volume: np.ndarray,
    period: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    機械学習特徴抽出システム
    
    Returns:
        (momentum_features, volatility_features, volume_features, composite_features)
    """
    n = len(prices)
    if n < period * 2:
        return (np.full(n, np.nan), np.full(n, np.nan), 
                np.full(n, np.nan), np.full(n, np.nan))
    
    momentum_features = np.full(n, np.nan)
    volatility_features = np.full(n, np.nan)
    volume_features = np.full(n, np.nan)
    composite_features = np.full(n, np.nan)
    
    for i in range(period, n):
        price_window = prices[i-period:i+1]
        
        # ボリューム処理（利用可能な場合）
        if len(volume) > i:
            volume_window = volume[i-period:i+1]
        else:
            volume_window = np.ones(len(price_window))  # デフォルト値
        
        # 1. モメンタム特徴量
        returns = np.diff(price_window)
        if len(returns) > 0:
            # 複数期間のモメンタム
            momentum_1 = returns[-1] if len(returns) >= 1 else 0
            momentum_3 = np.mean(returns[-3:]) if len(returns) >= 3 else 0
            momentum_5 = np.mean(returns[-5:]) if len(returns) >= 5 else 0
            
            # 正規化
            price_std = np.std(price_window)
            if price_std > 0:
                momentum_features[i] = (momentum_1 + momentum_3 + momentum_5) / (3 * price_std)
            else:
                momentum_features[i] = 0.0
        else:
            momentum_features[i] = 0.0
        
        # 2. ボラティリティ特徴量
        if len(returns) > 1:
            # 実現ボラティリティ
            realized_vol = np.std(returns)
            
            # GARCH様の特徴量
            squared_returns = returns ** 2
            vol_persistence = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1] if len(squared_returns) > 1 else 0
            vol_persistence = vol_persistence if not np.isnan(vol_persistence) else 0
            
            # 正規化
            price_level = np.mean(price_window)
            if price_level > 0:
                volatility_features[i] = realized_vol / price_level + abs(vol_persistence) * 0.1
            else:
                volatility_features[i] = realized_vol
        else:
            volatility_features[i] = 0.0
        
        # 3. ボリューム特徴量
        if len(volume_window) > 1:
            # ボリューム・価格関係
            price_returns = returns
            volume_changes = np.diff(volume_window)
            
            if len(price_returns) == len(volume_changes) and len(price_returns) > 1:
                # 価格・ボリューム相関
                pv_correlation = np.corrcoef(price_returns, volume_changes)[0, 1]
                pv_correlation = pv_correlation if not np.isnan(pv_correlation) else 0
                
                # ボリューム移動平均との乖離
                volume_ma = np.mean(volume_window)
                current_volume = volume_window[-1]
                if volume_ma > 0:
                    volume_deviation = (current_volume - volume_ma) / volume_ma
                else:
                    volume_deviation = 0
                
                volume_features[i] = abs(pv_correlation) * 0.5 + abs(volume_deviation) * 0.5
            else:
                volume_features[i] = 0.0
        else:
            volume_features[i] = 0.0
        
        # 4. 複合特徴量
        # 各特徴量の統合
        feat1 = momentum_features[i] if not np.isnan(momentum_features[i]) else 0
        feat2 = volatility_features[i] if not np.isnan(volatility_features[i]) else 0
        feat3 = volume_features[i] if not np.isnan(volume_features[i]) else 0
        
        # 非線形結合
        composite_features[i] = 0.0
        weight1 = 1.0
        weight2 = 0.5
        weight3 = 0.33333
        
        # タンジェント関数で非線形変換
        transformed1 = math.tanh(feat1 * 2.0)
        transformed2 = math.tanh(feat2 * 2.0)
        transformed3 = math.tanh(feat3 * 2.0)
        
        composite_features[i] = weight1 * transformed1 + weight2 * transformed2 + weight3 * transformed3
        
        # 正規化
        total_weight = weight1 + weight2 + weight3
        composite_features[i] = composite_features[i] / total_weight
    
    return momentum_features, volatility_features, volume_features, composite_features


@njit(fastmath=True, cache=True)
def ultimate_trend_range_ensemble(
    trend_strength: np.ndarray,
    directional_movement: np.ndarray,
    consolidation_index: np.ndarray,
    multi_scale_trend: np.ndarray,
    harmonic_strength: np.ndarray,
    volatility_regime: np.ndarray,
    ml_features: np.ndarray,
    weights: np.ndarray = np.array([0.25, 0.20, 0.20, 0.15, 0.10, 0.05, 0.05])
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ultimate Trend Range アンサンブルシステム
    
    Returns:
        (final_trend_strength, trend_probability, confidence_score)
    """
    n = len(trend_strength)
    final_trend_strength = np.full(n, np.nan)
    trend_probability = np.full(n, np.nan)
    confidence_score = np.full(n, np.nan)
    
    for i in range(n):
        # 各指標の安全な取得
        ind1 = trend_strength[i] if not np.isnan(trend_strength[i]) else 0.5
        ind2 = directional_movement[i] if not np.isnan(directional_movement[i]) else 0.5
        ind3 = 1.0 - (consolidation_index[i] if not np.isnan(consolidation_index[i]) else 0.5)  # レンジ指標の逆転
        ind4 = multi_scale_trend[i] if not np.isnan(multi_scale_trend[i]) else 0.5
        ind5 = harmonic_strength[i] if not np.isnan(harmonic_strength[i]) else 0.5
        ind6 = (volatility_regime[i] / 2.0 if not np.isnan(volatility_regime[i]) and volatility_regime[i] != 0 else 0.5)  # 0-1に正規化
        ind7 = abs(ml_features[i]) if not np.isnan(ml_features[i]) else 0.5
        
        # 重み付き平均（安全な配列アクセス）
        if len(weights) >= 7:
            w1, w2, w3, w4, w5, w6, w7 = weights[0], weights[1], weights[2], weights[3], weights[4], weights[5], weights[6]
        else:
            w1 = w2 = w3 = w4 = w5 = w6 = w7 = 1.0 / 7.0
        
        # 値を0-1範囲に制限
        norm_ind1 = min(max(ind1, 0.0), 1.0)
        norm_ind2 = min(max(ind2, 0.0), 1.0)
        norm_ind3 = min(max(ind3, 0.0), 1.0)
        norm_ind4 = min(max(ind4, 0.0), 1.0)
        norm_ind5 = min(max(ind5, 0.0), 1.0)
        norm_ind6 = min(max(ind6, 0.0), 1.0)
        norm_ind7 = min(max(ind7, 0.0), 1.0)
        
        weighted_sum = (norm_ind1 * w1 + norm_ind2 * w2 + norm_ind3 * w3 + 
                       norm_ind4 * w4 + norm_ind5 * w5 + norm_ind6 * w6 + norm_ind7 * w7)
        weight_sum = w1 + w2 + w3 + w4 + w5 + w6 + w7
        
        if weight_sum > 1e-15:  # より厳密なゼロ除算対策
            final_trend_strength[i] = weighted_sum / weight_sum
        else:
            final_trend_strength[i] = 0.5
        
        # トレンド確率（シグモイド変換）
        # 0.5を中心とした変換
        trend_score = (final_trend_strength[i] - 0.5) * 4  # -2から2の範囲に拡張
        trend_probability[i] = 1.0 / (1.0 + math.exp(-trend_score))
        
        # 信頼度スコア（指標間の一致度）
        mean_indicator = (ind1 + ind2 + ind3 + ind4 + ind5 + ind6 + ind7) / 7.0
        var1 = (ind1 - mean_indicator) ** 2
        var2 = (ind2 - mean_indicator) ** 2
        var3 = (ind3 - mean_indicator) ** 2
        var4 = (ind4 - mean_indicator) ** 2
        var5 = (ind5 - mean_indicator) ** 2
        var6 = (ind6 - mean_indicator) ** 2
        var7 = (ind7 - mean_indicator) ** 2
        variance = (var1 + var2 + var3 + var4 + var5 + var6 + var7) / 7.0
        
        # ゼロ除算対策を強化した信頼度計算
        variance_safe = max(variance, 1e-15)  # 最小分散を保証
        confidence_score[i] = 1.0 / (1.0 + variance_safe * 4.0)  # 分散の逆数ベース
    
    return final_trend_strength, trend_probability, confidence_score


class UltimateTrendRangeFilter(Indicator):
    """
    Ultimate Trend Range Filter - 宇宙最強のトレンド/レンジ判別システム 🚀
    
    Ultimate Chop Trendをベースに、トレンドとレンジの2つに特化した究極の判別インジケーター：
    
    【統合アルゴリズム】
    🔬 Advanced Trend Detection - 多次元トレンド強度解析
    📊 Range Consolidation Analysis - 高度レンジ統合解析  
    🧠 Multi-Scale Trend Filtering - マルチスケールトレンドフィルタリング
    🎯 Adaptive Noise Suppression - 適応的ノイズ抑制システム
    📡 Harmonic Pattern Recognition - ハーモニックパターン自動認識
    🌊 Volatility Regime Classification - ボラティリティレジーム分類
    🚀 ML Feature Extraction - 機械学習特徴抽出エンジン
    🏆 Ultimate Ensemble System - 最強アンサンブル統合システム
    
    【出力】
    - 0.0 = 完全なレンジ相場
    - 1.0 = 強いトレンド相場
    - 方向性は問わず、トレンドの存在そのものを検出
    """
    
    def __init__(
        self,
        # コアパラメータ
        analysis_period: int = 20,
        ensemble_window: int = 50,
        
        # アルゴリズム有効化
        enable_advanced_trend: bool = True,
        enable_range_analysis: bool = True,
        enable_multi_scale: bool = True,
        enable_noise_suppression: bool = True,
        enable_harmonic_patterns: bool = True,
        enable_volatility_regime: bool = True,
        enable_ml_features: bool = True,
        
        # 判定しきい値（実践的な値に調整）
        trend_threshold: float = 0.4,        # トレンド判定しきい値（0.6→0.4に下げて実用的に）
        strong_trend_threshold: float = 0.7, # 強トレンド判定しきい値（0.8→0.7に下げて実用的に）
        
        # アンサンブル重み
        component_weights: Optional[np.ndarray] = None,
        
        # 高度設定
        multi_scale_periods: Optional[np.ndarray] = None,
        noise_adaptation_factor: float = 0.1,
        harmonic_detection_period: int = 30
    ):
        """
        Ultimate Trend Range Filter - 最強トレンド/レンジ判別システム
        
        Args:
            analysis_period: 基本分析期間
            ensemble_window: アンサンブル統合ウィンドウ
            enable_*: 各アルゴリズムの有効化フラグ
            trend_threshold: トレンド判定しきい値（0-1）
            strong_trend_threshold: 強トレンド判定しきい値（0-1）
            component_weights: アンサンブル成分重み
            multi_scale_periods: マルチスケール解析期間
            noise_adaptation_factor: ノイズ適応係数
            harmonic_detection_period: ハーモニック検出期間
        """
        super().__init__(f"UltimateTrendRangeFilter(P={analysis_period},T={trend_threshold})")
        
        self.analysis_period = analysis_period
        self.ensemble_window = ensemble_window
        
        # アルゴリズム有効化フラグ
        self.enable_advanced_trend = enable_advanced_trend
        self.enable_range_analysis = enable_range_analysis
        self.enable_multi_scale = enable_multi_scale
        self.enable_noise_suppression = enable_noise_suppression
        self.enable_harmonic_patterns = enable_harmonic_patterns
        self.enable_volatility_regime = enable_volatility_regime
        self.enable_ml_features = enable_ml_features
        
        # しきい値設定
        self.trend_threshold = trend_threshold
        self.strong_trend_threshold = strong_trend_threshold
        
        # アンサンブル重み（トレンド検出感度を向上）
        if component_weights is not None and np.sum(component_weights) > 1e-15:
            self.component_weights = component_weights
        else:
            # より実践的な重み配分（トレンド検出とマルチスケール分析を重視）
            self.component_weights = np.array([0.30, 0.25, 0.15, 0.20, 0.05, 0.03, 0.02])
        
        # 高度設定
        if multi_scale_periods is not None:
            self.multi_scale_periods = multi_scale_periods
        else:
            self.multi_scale_periods = np.array([5, 10, 20, 40])
        
        self.noise_adaptation_factor = noise_adaptation_factor
        self.harmonic_detection_period = harmonic_detection_period
        
        # 結果キャッシュ
        self._result: Optional[UltimateTrendRangeResult] = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> UltimateTrendRangeResult:
        """
        Ultimate Trend Range Filterを計算
        """
        try:
            # データ検証
            if len(data) == 0:
                return self._create_empty_result(0)
            
            # データ変換
            if isinstance(data, pd.DataFrame):
                prices = np.asarray(data['close'].values, dtype=np.float64)
                high = np.asarray(data['high'].values, dtype=np.float64)
                low = np.asarray(data['low'].values, dtype=np.float64)
                # ボリューム（オプション）
                if 'volume' in data.columns:
                    volume = np.asarray(data['volume'].values, dtype=np.float64)
                else:
                    volume = np.ones(len(prices))
            else:
                prices = np.asarray(data[:, 3], dtype=np.float64)  # close
                high = np.asarray(data[:, 1], dtype=np.float64)    # high
                low = np.asarray(data[:, 2], dtype=np.float64)     # low
                if data.shape[1] > 5:
                    volume = np.asarray(data[:, 5], dtype=np.float64)
                else:
                    volume = np.ones(len(prices))
            
            n = len(prices)
            
            # ボラティリティの計算（ATR近似）
            volatility = np.zeros(n)
            for i in range(1, n):
                tr = max(
                    high[i] - low[i],
                    abs(high[i] - prices[i-1]),
                    abs(low[i] - prices[i-1])
                )
                if i < 14:
                    volatility[i] = max(tr, 1e-10)
                else:
                    volatility[i] = max((volatility[i-1] * 13 + tr) / 14, 1e-10)
            volatility[0] = max(high[0] - low[0], 1e-10)
            
            # 各アルゴリズムの実行
            components = {}
            
            # 1. 高度トレンド検出
            if self.enable_advanced_trend:
                trend_str, dir_mov, trend_cons = advanced_trend_detection(
                    prices, high, low, self.analysis_period
                )
                components['trend_strength'] = trend_str
                components['directional_movement'] = dir_mov
                components['trend_consistency'] = trend_cons
            else:
                components['trend_strength'] = np.full(n, 0.5)
                components['directional_movement'] = np.full(n, 0.5)
                components['trend_consistency'] = np.full(n, 0.5)
            
            # 2. レンジ統合解析
            if self.enable_range_analysis:
                consol_idx, range_tight, sideways_str = range_consolidation_analysis(
                    prices, high, low, self.analysis_period
                )
                components['consolidation_index'] = consol_idx
                components['range_tightness'] = range_tight
                components['sideways_strength'] = sideways_str
            else:
                components['consolidation_index'] = np.full(n, 0.5)
                components['range_tightness'] = np.full(n, 0.5)
                components['sideways_strength'] = np.full(n, 0.5)
            
            # 3. マルチスケールトレンドフィルタリング
            if self.enable_multi_scale:
                multi_trend, scale_cons = multi_scale_trend_filtering(
                    prices, self.multi_scale_periods
                )
                components['multi_scale_trend'] = multi_trend
                components['scale_consistency'] = scale_cons
            else:
                components['multi_scale_trend'] = np.full(n, 0.5)
                components['scale_consistency'] = np.full(n, 0.5)
            
            # 4. 適応ノイズ抑制
            if self.enable_noise_suppression:
                # 初期シグナルとしてトレンド強度を使用
                initial_signal = components['trend_strength']
                denoised_signal, noise_level = adaptive_noise_suppression(
                    initial_signal, volatility, self.noise_adaptation_factor
                )
                components['denoised_signal'] = denoised_signal
                components['noise_level'] = noise_level
            else:
                components['denoised_signal'] = components['trend_strength']
                components['noise_level'] = np.zeros(n)
            
            # 5. ハーモニックパターン認識
            if self.enable_harmonic_patterns:
                harmonic_str, pattern_type = harmonic_pattern_recognition(
                    prices, self.harmonic_detection_period
                )
                components['harmonic_strength'] = harmonic_str
                components['pattern_type'] = pattern_type
            else:
                components['harmonic_strength'] = np.full(n, 0.5)
                components['pattern_type'] = np.zeros(n)
            
            # 6. ボラティリティレジーム分類
            if self.enable_volatility_regime:
                vol_regime, regime_stab, regime_trans = volatility_regime_classification(
                    prices, high, low, self.analysis_period
                )
                components['volatility_regime'] = vol_regime
                components['regime_stability'] = regime_stab
                components['regime_transition'] = regime_trans
            else:
                components['volatility_regime'] = np.ones(n)
                components['regime_stability'] = np.ones(n)
                components['regime_transition'] = np.zeros(n)
            
            # 7. 機械学習特徴抽出
            if self.enable_ml_features:
                momentum_feat, vol_feat, volume_feat, composite_feat = ml_feature_extraction(
                    prices, volume, self.analysis_period
                )
                components['momentum_features'] = momentum_feat
                components['volatility_features'] = vol_feat
                components['volume_features'] = volume_feat
                components['composite_features'] = composite_feat
            else:
                components['momentum_features'] = np.full(n, 0.5)
                components['volatility_features'] = np.full(n, 0.5)
                components['volume_features'] = np.full(n, 0.5)
                components['composite_features'] = np.full(n, 0.5)
            
            # 8. アンサンブル統合（安全性チェック付き）
            # 各成分のNaN値をデフォルト値で置換
            safe_components = {}
            for key, values in components.items():
                safe_values = values.copy()
                # NaN値を0.5で置換
                nan_mask = np.isnan(safe_values)
                safe_values[nan_mask] = 0.5
                # 無限大値を制限
                inf_mask = np.isinf(safe_values)
                safe_values[inf_mask] = 0.5
                # 範囲制限
                safe_values = np.clip(safe_values, 0.0, 1.0)
                safe_components[key] = safe_values
            
            final_trend_strength, trend_prob, confidence = ultimate_trend_range_ensemble(
                safe_components['trend_strength'],
                safe_components['directional_movement'],
                safe_components['consolidation_index'],
                safe_components['multi_scale_trend'],
                safe_components['harmonic_strength'],
                safe_components['volatility_regime'],
                safe_components['composite_features'],
                self.component_weights
            )
            
            # 最終判定（初期期間も含めて完全な分類を保証）
            trend_classification = np.zeros(n)
            range_probability = np.zeros(n)
            
            for i in range(n):
                strength = final_trend_strength[i] if not np.isnan(final_trend_strength[i]) else 0.5
                
                # 初期期間（analysis_period未満）の特別処理
                if i < self.analysis_period:
                    # 初期期間は価格変動の簡単な分析でトレンド/レンジを判定
                    if i >= 5:  # 最低5期間のデータがある場合
                        recent_prices = prices[max(0, i-4):i+1]
                        price_range = np.max(recent_prices) - np.min(recent_prices)
                        price_change = abs(recent_prices[-1] - recent_prices[0])
                        
                        # 価格変動が範囲の50%以上ならトレンド、そうでなければレンジ
                        if price_range > 0 and (price_change / price_range) > 0.5:
                            trend_classification[i] = 1.0  # トレンド
                            strength = 0.6  # デフォルトのトレンド強度
                        else:
                            trend_classification[i] = 0.0  # レンジ
                            strength = 0.3  # デフォルトのレンジ強度
                    else:
                        # データが不十分な場合はレンジとして扱う
                        trend_classification[i] = 0.0  # レンジ
                        strength = 0.3
                else:
                    # 通常の判定
                    if strength >= self.trend_threshold:
                        trend_classification[i] = 1.0  # トレンド
                    else:
                        trend_classification[i] = 0.0  # レンジ
                
                # レンジ確率
                range_probability[i] = 1.0 - strength
            
            # 追加メトリクス（初期期間も含めて処理）
            signal_quality = np.full(n, 0.5)  # デフォルト値で初期化
            persistence_factor = np.full(n, 0.5)  # デフォルト値で初期化
            market_microstructure = np.full(n, 0.5)  # デフォルト値で初期化
            trend_continuation = np.full(n, 0.5)  # デフォルト値で初期化
            regime_change_prob = np.full(n, 0.2)  # デフォルト値で初期化
            
            # 初期期間の処理
            for i in range(min(self.analysis_period, n)):
                if i >= 3:
                    # 簡単な品質指標を計算
                    recent_strengths = final_trend_strength[max(0, i-2):i+1]
                    recent_strengths_clean = recent_strengths[~np.isnan(recent_strengths)]
                    if len(recent_strengths_clean) > 1:
                        signal_quality[i] = max(0.3, 1.0 - np.std(recent_strengths_clean))
                    
                    # 簡単な持続性計算
                    if len(recent_strengths_clean) > 1:
                        changes = np.abs(np.diff(recent_strengths_clean))
                        persistence_factor[i] = max(0.3, 1.0 - np.mean(changes))
            
            # 通常期間の処理
            for i in range(self.analysis_period, n):
                # シグナル品質（一貫性）
                recent_strengths = final_trend_strength[max(0, i-10):i+1]
                signal_quality[i] = 1.0 - np.std(recent_strengths) if len(recent_strengths) > 1 else 1.0
                
                # 持続性要因
                strength_changes = np.diff(recent_strengths)
                if len(strength_changes) > 0:
                    persistence_factor[i] = 1.0 - np.mean(np.abs(strength_changes))
                    persistence_factor[i] = max(persistence_factor[i], 0.0)
                
                # 市場マイクロ構造（価格効率性）
                price_changes = np.diff(prices[max(0, i-10):i+1])
                if len(price_changes) > 1:
                    autocorr = np.corrcoef(price_changes[:-1], price_changes[1:])[0, 1]
                    market_microstructure[i] = 1.0 - abs(autocorr) if not np.isnan(autocorr) else 1.0
                
                # トレンド継続予測
                if i >= 5:
                    trend_momentum = final_trend_strength[i] - final_trend_strength[i-5]
                    trend_continuation[i] = max(0, trend_momentum + 0.5)
                
                # レジーム変化確率
                if i >= 3:
                    recent_volatility = np.std(prices[i-3:i+1])
                    historical_volatility = np.std(prices[max(0, i-20):i-3])
                    if historical_volatility > 1e-15:  # より厳密なゼロ除算対策
                        vol_change = abs(recent_volatility - historical_volatility) / historical_volatility
                        regime_change_prob[i] = min(vol_change, 1.0)
                    else:
                        regime_change_prob[i] = 0.5  # デフォルト値
            
            # 現在状態の判定
            latest_strength = final_trend_strength[-1] if len(final_trend_strength) > 0 else 0.5
            latest_confidence = confidence[-1] if len(confidence) > 0 else 0.5
            
            if latest_strength >= self.strong_trend_threshold:
                current_state = "strong_trend"
            elif latest_strength >= self.trend_threshold:
                current_state = "trend"
            else:
                current_state = "range"
            
            # 結果作成
            result = UltimateTrendRangeResult(
                trend_strength=final_trend_strength,
                trend_classification=trend_classification,
                range_probability=range_probability,
                trend_probability=trend_prob,
                confidence_score=confidence,
                signal_quality=signal_quality,
                noise_level=components['noise_level'],
                directional_movement=components['directional_movement'],
                consolidation_index=components['consolidation_index'],
                volatility_regime=components['volatility_regime'],
                harmonic_strength=components['harmonic_strength'],
                fractal_dimension=components['scale_consistency'],  # 代用
                persistence_factor=persistence_factor,
                market_microstructure=market_microstructure,
                trend_continuation=trend_continuation,
                regime_change_probability=regime_change_prob,
                current_state=current_state,
                current_strength=latest_strength,
                current_confidence=latest_confidence
            )
            
            self._result = result
            self._values = final_trend_strength
            
            return result
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.logger.error(f"UltimateTrendRangeFilter計算中にエラー: {e}\n詳細:\n{error_details}")
            return self._create_empty_result(len(data))
    
    def _create_empty_result(self, length: int) -> UltimateTrendRangeResult:
        """空の結果を作成"""
        return UltimateTrendRangeResult(
            trend_strength=np.full(length, np.nan),
            trend_classification=np.zeros(length),
            range_probability=np.full(length, 0.5),
            trend_probability=np.full(length, 0.5),
            confidence_score=np.zeros(length),
            signal_quality=np.zeros(length),
            noise_level=np.zeros(length),
            directional_movement=np.full(length, np.nan),
            consolidation_index=np.full(length, np.nan),
            volatility_regime=np.ones(length),
            harmonic_strength=np.full(length, np.nan),
            fractal_dimension=np.full(length, np.nan),
            persistence_factor=np.zeros(length),
            market_microstructure=np.zeros(length),
            trend_continuation=np.zeros(length),
            regime_change_probability=np.zeros(length),
            current_state="range",
            current_strength=0.0,
            current_confidence=0.0
        )
    
    def get_values(self) -> Optional[np.ndarray]:
        """メイン指標値（トレンド強度）を取得"""
        if self._result is not None:
            return self._result.trend_strength.copy()
        return None
    
    def get_classification(self) -> Optional[np.ndarray]:
        """トレンド分類（0=レンジ, 1=トレンド）を取得"""
        if self._result is not None:
            return self._result.trend_classification.copy()
        return None
    
    def get_result(self) -> Optional[UltimateTrendRangeResult]:
        """完全な結果を取得"""
        return self._result
    
    def reset(self) -> None:
        """リセット"""
        super().reset()
        self._result = None
    
    def is_trend(self, index: int = -1) -> bool:
        """指定されたインデックスでトレンド状態かどうかを判定"""
        if self._result is None:
            return False
        
        classification = self._result.trend_classification
        if len(classification) == 0:
            return False
        
        return bool(classification[index] == 1.0)
    
    def is_range(self, index: int = -1) -> bool:
        """指定されたインデックスでレンジ状態かどうかを判定"""
        return not self.is_trend(index)
    
    def get_trend_strength(self, index: int = -1) -> float:
        """指定されたインデックスでのトレンド強度を取得"""
        if self._result is None:
            return 0.5
        
        strength = self._result.trend_strength
        if len(strength) == 0:
            return 0.5
        
        return float(strength[index]) if not np.isnan(strength[index]) else 0.5
    
    def get_confidence(self, index: int = -1) -> float:
        """指定されたインデックスでの信頼度を取得"""
        if self._result is None:
            return 0.0
        
        confidence = self._result.confidence_score
        if len(confidence) == 0:
            return 0.0
        
        return float(confidence[index]) if not np.isnan(confidence[index]) else 0.0 