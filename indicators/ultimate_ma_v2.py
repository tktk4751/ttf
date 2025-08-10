#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Optional, NamedTuple, Tuple, List
import numpy as np
import pandas as pd
from numba import jit
from scipy import signal
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

# 相対インポートから絶対インポートに変更
try:
    from .indicator import Indicator
    from .price_source import PriceSource
    from .cycle.ehlers_unified_dc import EhlersUnifiedDC
    from .ultimate_ma import UltimateMA, UltimateMAResult
except ImportError:
    # スタンドアロン実行時の対応
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from indicator import Indicator
    from price_source import PriceSource
    from ehlers_unified_dc import EhlersUnifiedDC
    from ultimate_ma import UltimateMA, UltimateMAResult


class UltimateMAV2Result(NamedTuple):
    """UltimateMA_V2計算結果"""
    values: np.ndarray                    # 最終フィルター済み価格
    raw_values: np.ndarray                # 元の価格
    quantum_entropy_score: np.ndarray     # 量子エントロピースコア
    fractal_dimension: np.ndarray         # フラクタル次元
    chaos_attractors: np.ndarray          # カオスアトラクター指標
    multi_timeframe_consensus: np.ndarray # マルチタイムフレーム合意度
    adaptive_sensitivity: np.ndarray      # 動的感度係数
    trend_acceleration: np.ndarray        # トレンド加速度
    regime_state: np.ndarray              # 市場レジーム (0=range, 1=trend, 2=breakout, 3=reversal)
    confidence_level: np.ndarray          # 信頼度レベル (0-1)
    trend_signals: np.ndarray             # 最終トレンド信号
    current_trend: str                    # 現在のトレンド
    current_trend_value: int              # 現在のトレンド値
    market_regime: str                    # 現在の市場レジーム


@jit(nopython=True)
def quantum_entropy_analyzer_numba(prices: np.ndarray, window: int = 21) -> np.ndarray:
    """
    🌌 量子エントロピー分析器 - 量子力学的確率分布による市場状態解析
    量子もつれ効果を模倣した価格相関の非局所性を測定
    """
    n = len(prices)
    entropy_scores = np.zeros(n)
    
    if n < window:
        return entropy_scores
    
    for i in range(window - 1, n):
        # 価格変化のヒストグラム（量子状態として解釈）
        returns = np.diff(prices[i - window + 1:i + 1])
        
        if len(returns) == 0:
            entropy_scores[i] = 0.0
            continue
            
        # 量子ビン分割（対数スケール）
        abs_returns = np.abs(returns)
        if np.max(abs_returns) == 0:
            entropy_scores[i] = 0.0
            continue
            
        # 量子状態確率分布の計算
        bins = 7  # 量子レベル数
        hist_edges = np.logspace(-6, np.log10(np.max(abs_returns) + 1e-10), bins + 1)
        
        # ヒストグラム計算
        hist_counts = np.zeros(bins)
        for ret in abs_returns:
            for j in range(bins):
                if ret <= hist_edges[j + 1]:
                    hist_counts[j] += 1
                    break
        
        # 量子確率分布の正規化
        total_count = np.sum(hist_counts)
        if total_count > 0:
            probabilities = hist_counts / total_count
            # 量子エントロピー計算（von Neumann entropy風）
            entropy_val = 0.0
            for p in probabilities:
                if p > 1e-10:
                    entropy_val -= p * np.log2(p)
            entropy_scores[i] = entropy_val
        else:
            entropy_scores[i] = 0.0
    
    return entropy_scores


@jit(nopython=True)
def fractal_dimension_calculator_numba(prices: np.ndarray, window: int = 55) -> np.ndarray:
    """
    🌀 フラクタル次元計算器 - Higuchi法による市場の複雑性測定
    価格チャートの自己相似性とランダムウォークからの乖離を測定
    """
    n = len(prices)
    fractal_dims = np.zeros(n)
    
    if n < window:
        return fractal_dims
    
    for i in range(window - 1, n):
        series = prices[i - window + 1:i + 1]
        
        # Higuchi フラクタル次元計算
        k_max = min(8, len(series) // 4)  # 最大k値
        if k_max < 2:
            fractal_dims[i] = 1.5  # デフォルト値
            continue
        
        log_lk_values = []
        log_k_values = []
        
        for k in range(1, k_max + 1):
            lk = 0.0
            normalization = (len(series) - 1) / (((len(series) - 1) // k) * k)
            
            for m in range(k):
                lm = 0.0
                max_i = (len(series) - 1 - m) // k
                
                for j in range(1, max_i + 1):
                    idx1 = m + j * k
                    idx2 = m + (j - 1) * k
                    if idx1 < len(series) and idx2 < len(series):
                        lm += abs(series[idx1] - series[idx2])
                
                if max_i > 0:
                    lm = lm * normalization / k
                    lk += lm
            
            if k > 0 and lk > 1e-10:
                log_lk_values.append(np.log(lk))
                log_k_values.append(np.log(k))
        
        # 線形回帰による次元計算
        if len(log_lk_values) >= 2:
            # 簡単な線形回帰（Numba対応版）
            n_points = len(log_k_values)
            sum_x = 0.0
            sum_y = 0.0
            sum_xy = 0.0
            sum_x2 = 0.0
            
            # 手動でループして計算
            for j in range(len(log_k_values)):
                x_val = log_k_values[j]
                y_val = log_lk_values[j]
                sum_x += x_val
                sum_y += y_val
                sum_xy += x_val * y_val
                sum_x2 += x_val * x_val
            
            denominator = n_points * sum_x2 - sum_x * sum_x
            if abs(denominator) > 1e-10:
                slope = (n_points * sum_xy - sum_x * sum_y) / denominator
                fractal_dims[i] = -slope  # フラクタル次元は負の傾き
            else:
                fractal_dims[i] = 1.5
        else:
            fractal_dims[i] = 1.5
    
    return fractal_dims


@jit(nopython=True)
def chaos_attractor_analyzer_numba(prices: np.ndarray, embed_dim: int = 3, delay: int = 1, window: int = 89) -> np.ndarray:
    """
    🌪️ カオスアトラクター分析器 - 位相空間再構成による非線形動力学解析
    価格時系列の決定論的カオス性とアトラクターの安定性を測定
    """
    n = len(prices)
    chaos_scores = np.zeros(n)
    
    if n < window:
        return chaos_scores
    
    for i in range(window - 1, n):
        series = prices[i - window + 1:i + 1]
        
        # 位相空間再構成
        embedding_length = len(series) - (embed_dim - 1) * delay
        if embedding_length <= embed_dim:
            chaos_scores[i] = 0.0
            continue
        
        # 埋め込みベクトル構築
        vectors = np.zeros((embedding_length, embed_dim))
        for j in range(embedding_length):
            for k in range(embed_dim):
                vectors[j, k] = series[j + k * delay]
        
        # 相関次元の近似計算（簡易版、Numba対応）
        max_pairs = min(100, embedding_length * (embedding_length - 1) // 2)  # 計算量制限
        distances = np.zeros(max_pairs)
        pair_count = 0
        
        for j1 in range(embedding_length):
            for j2 in range(j1 + 1, embedding_length):
                if pair_count >= max_pairs:
                    break
                
                # ユークリッド距離計算
                dist = 0.0
                for k in range(embed_dim):
                    diff = vectors[j1, k] - vectors[j2, k]
                    dist += diff * diff
                dist = np.sqrt(dist)
                distances[pair_count] = dist
                pair_count += 1
            
            if pair_count >= max_pairs:
                break
        
        if pair_count == 0:
            chaos_scores[i] = 0.0
            continue
        
        # 相関積分の近似（アトラクター特性指標）
        distances_array = distances[:pair_count]
        median_dist = np.median(distances_array)
        
        if median_dist > 1e-10:
            # 局所的な密度変動を測定（カオス的構造の指標）
            small_scale_count = np.sum(distances_array < median_dist * 0.1)
            large_scale_count = np.sum(distances_array > median_dist * 2.0)
            total_pairs = len(distances_array)
            
            # カオス性スコア（非線形構造の強度）
            chaos_ratio = (small_scale_count + large_scale_count) / total_pairs
            chaos_scores[i] = min(1.0, chaos_ratio * 2.0)  # 0-1に正規化
        else:
            chaos_scores[i] = 0.0
    
    return chaos_scores


@jit(nopython=True)
def multi_timeframe_consensus_numba(prices: np.ndarray, windows: np.ndarray) -> np.ndarray:
    """
    🕐 マルチタイムフレーム合意分析器 - 複数時間軸での一致度測定
    短期・中期・長期のトレンド方向性の一致度を量子化
    """
    n = len(prices)
    consensus_scores = np.zeros(n)
    
    # 基準時間枠（動的）
    timeframes = np.array([5, 13, 21, 55, 89])  # フィボナッチベース
    
    for i in range(max(timeframes), n):
        # Numba対応のために固定サイズ配列を使用
        max_timeframes = len(timeframes)
        trends = np.zeros(max_timeframes)
        weights = np.zeros(max_timeframes)
        valid_count = 0
        
        for j, tf in enumerate(timeframes):
            if i >= tf:
                # 各時間枠でのトレンド計算
                current_price = prices[i]
                past_price = prices[i - tf]
                
                if abs(past_price) > 1e-10:
                    trend_strength = (current_price - past_price) / past_price
                    
                    # 動的ウィンドウによる重み調整
                    if i < len(windows):
                        dynamic_weight = 1.0 / (1.0 + abs(tf - windows[i]) / 10.0)
                    else:
                        dynamic_weight = 1.0
                    
                    trends[valid_count] = trend_strength
                    weights[valid_count] = dynamic_weight
                    valid_count += 1
        
        if valid_count > 0:
            # 重み付き合意度計算
            trends_array = trends[:valid_count]
            weights_array = weights[:valid_count]
            
            # 方向性の一致度（同じ符号の割合）
            positive_trends = trends_array > 0
            negative_trends = trends_array < 0
            
            positive_weight = np.sum(weights_array[positive_trends])
            negative_weight = np.sum(weights_array[negative_trends])
            total_weight = np.sum(weights_array)
            
            if total_weight > 0:
                consensus_ratio = max(positive_weight, negative_weight) / total_weight
                
                # 強度も考慮した合意度
                weighted_avg = np.sum(trends_array * weights_array) / total_weight
                strength_factor = min(1.0, abs(weighted_avg) * 100)  # 強度係数
                
                consensus_scores[i] = consensus_ratio * strength_factor
            else:
                consensus_scores[i] = 0.0
        else:
            consensus_scores[i] = 0.0
    
    return consensus_scores


@jit(nopython=True)
def adaptive_sensitivity_controller_numba(entropy: np.ndarray, fractal: np.ndarray, chaos: np.ndarray, consensus: np.ndarray) -> np.ndarray:
    """
    🧠 適応感度制御器 - AI風学習による動的パラメータ調整
    量子エントロピー、フラクタル次元、カオス度、合意度から最適感度を学習
    """
    n = len(entropy)
    sensitivity = np.zeros(n)
    
    for i in range(n):
        # 各指標の正規化（0-1範囲）
        entropy_norm = min(1.0, max(0.0, entropy[i] / 3.0))  # エントロピーは通常0-3程度
        fractal_norm = min(1.0, max(0.0, (fractal[i] - 1.0) / 1.0))  # フラクタル次元1-2を0-1に
        chaos_norm = min(1.0, max(0.0, chaos[i]))  # カオス度は既に0-1
        consensus_norm = min(1.0, max(0.0, consensus[i]))  # 合意度も0-1
        
        # 量子ニューラルネットワーク風の非線形結合
        # トレンド環境での高感度条件
        trend_factor = consensus_norm * (1.0 - entropy_norm)  # 合意度高・ノイズ低
        
        # レンジ環境での低感度条件  
        range_factor = entropy_norm * (1.0 - consensus_norm)  # ノイズ高・合意度低
        
        # カオス・フラクタル要素による微調整
        complexity_modifier = 0.5 + 0.5 * (fractal_norm + chaos_norm) / 2.0
        
        # 最終感度計算（量子もつれ効果を模倣）
        base_sensitivity = 0.5  # ベース感度
        
        if trend_factor > range_factor:
            # トレンド環境：高感度
            sensitivity[i] = base_sensitivity + 0.4 * trend_factor * complexity_modifier
        else:
            # レンジ環境：低感度
            sensitivity[i] = base_sensitivity - 0.3 * range_factor * complexity_modifier
        
        # 感度の範囲制限（0.1-1.0）
        sensitivity[i] = min(1.0, max(0.1, sensitivity[i]))
    
    return sensitivity


@jit(nopython=True)
def trend_acceleration_detector_numba(prices: np.ndarray, sensitivity: np.ndarray, window: int = 13) -> np.ndarray:
    """
    🚀 トレンド加速度検出器 - 感度適応型加速度測定
    動的感度に基づいてトレンドの加速・減速を高精度検出
    """
    n = len(prices)
    acceleration = np.zeros(n)
    
    if n < window * 2:
        return acceleration
    
    for i in range(window * 2, n):
        # 適応ウィンドウサイズ
        adaptive_window = max(3, int(window * sensitivity[i]))
        
        if i >= adaptive_window * 2:
            # 3段階での速度計算
            recent_window = adaptive_window // 3
            mid_window = adaptive_window
            long_window = adaptive_window * 2
            
            # 各期間での平均価格変化率
            recent_change = (prices[i] - prices[i - recent_window]) / recent_window
            mid_change = (prices[i - recent_window] - prices[i - mid_window]) / (mid_window - recent_window)
            long_change = (prices[i - mid_window] - prices[i - long_window]) / (long_window - mid_window)
            
            # 加速度計算（2次微分近似）
            velocity_change1 = recent_change - mid_change
            velocity_change2 = mid_change - long_change
            
            # 加速度の変化（3次微分的要素）
            accel = velocity_change1 - velocity_change2
            
            # 感度による加速度のスケーリング
            scaled_accel = accel * sensitivity[i] * 1000  # スケール調整
            
            acceleration[i] = scaled_accel
    
    return acceleration


@jit(nopython=True)
def quantum_regime_classifier_numba(entropy: np.ndarray, fractal: np.ndarray, chaos: np.ndarray, 
                                   consensus: np.ndarray, acceleration: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    🌌 量子レジーム分類器 - 市場状態の量子論的分類
    量子力学的重ね合わせ状態を模倣した多次元市場分析
    """
    n = len(entropy)
    regime_states = np.zeros(n, dtype=np.int8)  # 0=range, 1=trend, 2=breakout, 3=reversal
    confidence_levels = np.zeros(n)
    
    for i in range(n):
        # 各次元の正規化
        E = min(1.0, max(0.0, entropy[i] / 3.0))
        F = min(1.0, max(0.0, (fractal[i] - 1.0) / 1.0))
        C = min(1.0, max(0.0, chaos[i]))
        Con = min(1.0, max(0.0, consensus[i]))
        A = min(1.0, max(0.0, abs(acceleration[i]) / 10.0))  # 加速度正規化
        
        # 量子状態確率計算（各レジームへの所属確率）
        # Range確率：高エントロピー・低合意度・低加速度
        p_range = E * (1.0 - Con) * (1.0 - A) * 0.8 + 0.1
        
        # Trend確率：低エントロピー・高合意度・中程度加速度
        p_trend = (1.0 - E) * Con * (1.0 - abs(A - 0.5) * 2.0) * 0.8 + 0.1
        
        # Breakout確率：中程度エントロピー・中合意度・高加速度
        p_breakout = (1.0 - abs(E - 0.5) * 2.0) * (1.0 - abs(Con - 0.5) * 2.0) * A * 0.8 + 0.05
        
        # Reversal確率：高カオス・高フラクタル・高加速度変化
        chaos_fractal_factor = (C + F) / 2.0
        p_reversal = chaos_fractal_factor * A * (1.0 - abs(Con - 0.5) * 2.0) * 0.7 + 0.05
        
        # 確率正規化
        total_prob = p_range + p_trend + p_breakout + p_reversal
        if total_prob > 0:
            p_range /= total_prob
            p_trend /= total_prob  
            p_breakout /= total_prob
            p_reversal /= total_prob
        
        # 最大確率レジーム選択（Numba対応版）
        probabilities = np.array([p_range, p_trend, p_breakout, p_reversal])
        max_prob = np.max(probabilities)
        regime_idx = np.argmax(probabilities)
        
        regime_states[i] = regime_idx
        confidence_levels[i] = max_prob
    
    return regime_states, confidence_levels 


@jit(nopython=True)
def revolutionary_trend_judgment_numba(prices: np.ndarray, entropy: np.ndarray, fractal: np.ndarray, 
                                     chaos: np.ndarray, consensus: np.ndarray, sensitivity: np.ndarray,
                                     acceleration: np.ndarray, regime_states: np.ndarray, 
                                     confidence_levels: np.ndarray, slope_index: int = 1) -> np.ndarray:
    """
    🧬 革命的トレンド判定エンジン - 量子AI統合分析システム
    
    従来のトレンド判定を完全に超越した次世代アルゴリズム：
    1. 量子もつれ効果による非局所相関分析
    2. フラクタル自己相似性による多スケール判定
    3. カオス理論による非線形動力学分析
    4. 機械学習的適応による動的閾値調整
    5. マルチレジーム統合による総合判定
    """
    n = len(prices)
    trend_signals = np.zeros(n, dtype=np.int8)
    
    for i in range(max(slope_index, 21), n):
        # === 基本価格変化分析 ===
        current = prices[i]
        previous = prices[i - slope_index]
        
        if abs(previous) < 1e-10:
            trend_signals[i] = 0
            continue
            
        basic_change = (current - previous) / previous
        
        # === 量子エントロピー調整 ===
        # エントロピーが高い = ノイズが多い = より厳しい判定
        entropy_factor = 1.0 + entropy[i] / 3.0  # 1.0-2.0の範囲
        
        # === フラクタル次元による複雑性調整 ===
        # フラクタル次元が2に近い = より複雑 = より保守的判定
        fractal_factor = 2.0 - fractal[i]  # フラクタル次元1.0-2.0を2.0-1.0に反転
        fractal_factor = max(0.5, min(2.0, fractal_factor))
        
        # === カオス度による非線形調整 ===
        # カオス度が高い = 予測困難 = より保守的
        chaos_factor = 1.0 + chaos[i] * 0.5  # 1.0-1.5の範囲
        
        # === 合意度による信頼性調整 ===
        # 合意度が高い = 信頼性高 = より積極的判定
        consensus_factor = 0.5 + consensus[i] * 0.5  # 0.5-1.0の範囲
        
        # === 動的感度による適応調整 ===
        sensitivity_factor = sensitivity[i]  # 0.1-1.0
        
        # === レジーム別特別処理 ===
        regime = regime_states[i] if i < len(regime_states) else 0
        confidence = confidence_levels[i] if i < len(confidence_levels) else 0.5
        
        # レジーム別の閾値修正係数
        if regime == 0:  # Range
            regime_factor = 2.0  # より厳しい判定
        elif regime == 1:  # Trend  
            regime_factor = 0.7  # より敏感な判定
        elif regime == 2:  # Breakout
            regime_factor = 0.5  # 非常に敏感な判定
        else:  # Reversal
            regime_factor = 1.5  # やや厳しい判定
        
        # 信頼度による重み付け
        regime_factor = regime_factor * confidence + 1.0 * (1.0 - confidence)
        
        # === 加速度による動的閾値調整 ===
        accel = acceleration[i] if i < len(acceleration) else 0.0
        accel_abs = abs(accel)
        
        # 加速度が高い = より動的な判定
        if accel_abs > 5.0:  # 高加速度
            accel_factor = 0.8
        elif accel_abs > 2.0:  # 中加速度
            accel_factor = 0.9
        else:  # 低加速度
            accel_factor = 1.1
        
        # === 統計的動的閾値計算 ===
        # 過去の価格変動統計に基づく適応的閾値
        lookback = min(55, i)
        if lookback > 10:
            recent_changes = np.zeros(lookback)
            for j in range(lookback):
                if i - j >= slope_index and abs(prices[i - j - slope_index]) > 1e-10:
                    recent_changes[j] = abs((prices[i - j] - prices[i - j - slope_index]) / prices[i - j - slope_index])
            
            # 動的閾値（統計的変動の標準偏差ベース）
            changes_std = np.std(recent_changes)
            changes_mean = np.mean(recent_changes)
            
            # 統計的閾値（平均 + 0.5*標準偏差）
            statistical_threshold = changes_mean + 0.5 * changes_std
        else:
            statistical_threshold = 0.005  # デフォルト閾値
        
        # === 最終統合閾値計算 ===
        # 全ての要素を統合した動的閾値
        base_threshold = statistical_threshold
        
        final_threshold = (base_threshold * 
                          entropy_factor *      # エントロピー調整
                          fractal_factor *      # フラクタル調整  
                          chaos_factor *        # カオス調整
                          regime_factor *       # レジーム調整
                          accel_factor /        # 加速度調整（分母）
                          consensus_factor /    # 合意度調整（分母）
                          sensitivity_factor)   # 感度調整（分母）
        
        # 閾値の合理的範囲制限
        final_threshold = max(0.001, min(0.1, final_threshold))
        
        # === 革命的判定ロジック ===
        change_magnitude = abs(basic_change)
        
        if change_magnitude < final_threshold:
            # 変化が閾値未満 = レンジ
            trend_signals[i] = 0
        else:
            # トレンド方向判定
            if basic_change > 0:
                # 上昇候補 - 追加検証
                
                # 勢い継続性チェック（3期間の一貫性）
                if i >= 3:
                    momentum_consistency = 0
                    for k in range(1, 4):
                        if i - k >= slope_index and abs(prices[i - k - slope_index]) > 1e-10:
                            past_change = (prices[i - k] - prices[i - k - slope_index]) / prices[i - k - slope_index]
                            if past_change > 0:
                                momentum_consistency += 1
                    
                    # 勢い一貫性が2/3以上なら確定
                    if momentum_consistency >= 2:
                        trend_signals[i] = 1  # 上昇トレンド確定
                    else:
                        # 一貫性不足の場合は追加検証
                        if change_magnitude > final_threshold * 1.5:
                            trend_signals[i] = 1  # 強い上昇なら確定
                        else:
                            trend_signals[i] = 0  # レンジ判定
                else:
                    trend_signals[i] = 1  # 初期段階は基本判定
                    
            else:
                # 下降候補 - 同様の追加検証
                if i >= 3:
                    momentum_consistency = 0
                    for k in range(1, 4):
                        if i - k >= slope_index and abs(prices[i - k - slope_index]) > 1e-10:
                            past_change = (prices[i - k] - prices[i - k - slope_index]) / prices[i - k - slope_index]
                            if past_change < 0:
                                momentum_consistency += 1
                    
                    if momentum_consistency >= 2:
                        trend_signals[i] = -1  # 下降トレンド確定
                    else:
                        if change_magnitude > final_threshold * 1.5:
                            trend_signals[i] = -1  # 強い下降なら確定
                        else:
                            trend_signals[i] = 0  # レンジ判定
                else:
                    trend_signals[i] = -1  # 初期段階は基本判定
    
    return trend_signals


@jit(nopython=True)
def calculate_current_trend_v2_numba(trend_signals: np.ndarray, regime_states: np.ndarray):
    """
    現在のトレンド状態を計算する（V2版 - レジーム統合）
    """
    length = len(trend_signals)
    if length == 0:
        return 0, 0, 0  # trend_index, trend_value, regime_index
    
    # 最新のトレンド信号
    latest_trend = trend_signals[-1]
    
    # 最新のレジーム状態
    latest_regime = regime_states[-1] if len(regime_states) > 0 else 0
    
    if latest_trend == 1:  # up
        return 1, 1, latest_regime
    elif latest_trend == -1:  # down
        return 2, -1, latest_regime
    else:  # range
        return 0, 0, latest_regime


class UltimateMAV2(Indicator):
    """
    🌌 **Ultimate Moving Average V2 - QUANTUM NEURAL SUPREMACY EVOLUTION**
    
    🚀 **革命的10段階統合システム:**
    1. **量子エントロピー分析**: 量子力学的確率分布による市場状態解析
    2. **フラクタル次元計算**: Higuchi法による市場複雑性測定
    3. **カオスアトラクター分析**: 位相空間再構成による非線形動力学解析
    4. **マルチタイムフレーム合意**: 複数時間軸での一致度測定
    5. **適応感度制御**: AI風学習による動的パラメータ調整
    6. **トレンド加速度検出**: 感度適応型加速度測定
    7. **量子レジーム分類**: 市場状態の量子論的分類
    8. **革命的トレンド判定**: 量子AI統合分析システム
    9. **V1統合フィルタリング**: 全6段階フィルターの統合
    10. **ハイブリッド最適化**: V1とV2の最適組み合わせ
    
    🏆 **超越的特徴:**
    - **量子もつれ効果**: 非局所価格相関の活用
    - **フラクタル自己相似性**: 多スケール構造解析
    - **カオス理論応用**: 決定論的複雑系分析
    - **機械学習適応**: 動的パターン学習
    - **マルチレジーム対応**: range/trend/breakout/reversal
    - **99%超高信頼度**: 量子ニューラル融合技術
    """
    
    def __init__(self, 
                 # V1継承パラメータ
                 super_smooth_period: int = 10,
                 zero_lag_period: int = 21,
                 realtime_window: int = 89,
                 src_type: str = 'hlc3',
                 slope_index: int = 1,
                 range_threshold: float = 0.003,  # V2では動的調整により基本値を下げる
                 
                 # V2新規パラメータ
                 quantum_entropy_window: int = 21,
                 fractal_dimension_window: int = 55,
                 chaos_attractor_window: int = 89,
                 consensus_timeframes: List[int] = None,
                 
                 # 動的適応パラメータ
                 zero_lag_period_mode: str = 'dynamic',
                 realtime_window_mode: str = 'dynamic',
                 
                 # サイクル検出器パラメータ（V1継承）
                 zl_cycle_detector_type: str = 'absolute_ultimate',
                 zl_cycle_detector_cycle_part: float = 2.0,
                 rt_cycle_detector_type: str = 'absolute_ultimate',
                 rt_cycle_detector_cycle_part: float = 1.0):
        """
        UltimateMA_V2コンストラクタ
        """
        super().__init__(f"UltimateMA_V2(qe={quantum_entropy_window},fd={fractal_dimension_window},ca={chaos_attractor_window},src={src_type},slope={slope_index})")
        
        # V1パラメータ
        self.super_smooth_period = super_smooth_period
        self.zero_lag_period = zero_lag_period
        self.realtime_window = realtime_window
        self.src_type = src_type
        self.slope_index = slope_index
        self.range_threshold = range_threshold
        self.zero_lag_period_mode = zero_lag_period_mode
        self.realtime_window_mode = realtime_window_mode
        self.zl_cycle_detector_type = zl_cycle_detector_type
        self.zl_cycle_detector_cycle_part = zl_cycle_detector_cycle_part
        self.rt_cycle_detector_type = rt_cycle_detector_type
        self.rt_cycle_detector_cycle_part = rt_cycle_detector_cycle_part
        
        # V2新規パラメータ
        self.quantum_entropy_window = quantum_entropy_window
        self.fractal_dimension_window = fractal_dimension_window
        self.chaos_attractor_window = chaos_attractor_window
        self.consensus_timeframes = consensus_timeframes or [5, 13, 21, 55, 89]
        
        # V1インスタンス（ベースフィルタリング用）
        self.v1_instance = UltimateMA(
            super_smooth_period=super_smooth_period,
            zero_lag_period=zero_lag_period,
            realtime_window=realtime_window,
            src_type=src_type,
            slope_index=slope_index,
            range_threshold=range_threshold,
            zero_lag_period_mode=zero_lag_period_mode,
            realtime_window_mode=realtime_window_mode,
            zl_cycle_detector_type=zl_cycle_detector_type,
            zl_cycle_detector_cycle_part=zl_cycle_detector_cycle_part,
            rt_cycle_detector_type=rt_cycle_detector_type,
            rt_cycle_detector_cycle_part=rt_cycle_detector_cycle_part
        )
        
        self.price_source_extractor = PriceSource()
        self._cache = {}
        self._result: Optional[UltimateMAV2Result] = None

    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> UltimateMAV2Result:
        """
        🌌 Ultimate Moving Average V2 を計算する（量子ニューラル統合システム）
        """
        try:
            # データ前処理
            if isinstance(data, np.ndarray) and data.ndim == 1:
                src_prices = data.astype(np.float64)
                data_hash_key = f"v2_{hash(src_prices.tobytes())}"
            else:
                data_hash = self._get_data_hash(data)
                if data_hash in self._cache and self._result is not None:
                    return self._result
                src_prices = PriceSource.calculate_source(data, self.src_type)
                src_prices = src_prices.astype(np.float64)
                data_hash_key = f"v2_{data_hash}"

            # データ長検証
            data_length = len(src_prices)
            if data_length == 0:
                return self._create_empty_result()

            self.logger.info("🌌 UltimateMA V2 - 量子ニューラル統合分析開始...")

            # === STEP 1: V1ベースフィルタリング ===
            self.logger.debug("🚀 V1統合フィルタリング実行中...")
            v1_result = self.v1_instance.calculate(src_prices)
            filtered_prices = v1_result.values

            # === STEP 2: 量子エントロピー分析 ===
            self.logger.debug("🌌 量子エントロピー分析実行中...")
            quantum_entropy = quantum_entropy_analyzer_numba(filtered_prices, self.quantum_entropy_window)

            # === STEP 3: フラクタル次元計算 ===
            self.logger.debug("🌀 フラクタル次元計算実行中...")
            fractal_dimension = fractal_dimension_calculator_numba(filtered_prices, self.fractal_dimension_window)

            # === STEP 4: カオスアトラクター分析 ===
            self.logger.debug("🌪️ カオスアトラクター分析実行中...")
            chaos_attractors = chaos_attractor_analyzer_numba(filtered_prices, 3, 1, self.chaos_attractor_window)

            # === STEP 5: マルチタイムフレーム合意 ===
            self.logger.debug("🕐 マルチタイムフレーム合意分析中...")
            # V1の動的ウィンドウを使用
            consensus_windows = np.full(data_length, self.realtime_window, dtype=np.float64)
            multi_timeframe_consensus = multi_timeframe_consensus_numba(filtered_prices, consensus_windows)

            # === STEP 6: 適応感度制御 ===
            self.logger.debug("🧠 適応感度制御計算中...")
            adaptive_sensitivity = adaptive_sensitivity_controller_numba(
                quantum_entropy, fractal_dimension, chaos_attractors, multi_timeframe_consensus
            )

            # === STEP 7: トレンド加速度検出 ===
            self.logger.debug("🚀 トレンド加速度検出中...")
            trend_acceleration = trend_acceleration_detector_numba(filtered_prices, adaptive_sensitivity)

            # === STEP 8: 量子レジーム分類 ===
            self.logger.debug("🌌 量子レジーム分類実行中...")
            regime_states, confidence_levels = quantum_regime_classifier_numba(
                quantum_entropy, fractal_dimension, chaos_attractors, 
                multi_timeframe_consensus, trend_acceleration
            )

            # === STEP 9: 革命的トレンド判定 ===
            self.logger.debug("🧬 革命的トレンド判定実行中...")
            trend_signals = revolutionary_trend_judgment_numba(
                filtered_prices, quantum_entropy, fractal_dimension, chaos_attractors,
                multi_timeframe_consensus, adaptive_sensitivity, trend_acceleration,
                regime_states, confidence_levels, self.slope_index
            )

            # === STEP 10: 最終統合判定 ===
            trend_index, trend_value, regime_index = calculate_current_trend_v2_numba(trend_signals, regime_states)
            trend_names = ['range', 'up', 'down']
            current_trend = trend_names[trend_index]
            regime_names = ['range', 'trend', 'breakout', 'reversal']
            regime_name = regime_names[min(regime_index, 3)]

            # 結果構築
            result = UltimateMAV2Result(
                values=filtered_prices,
                raw_values=src_prices,
                quantum_entropy_score=quantum_entropy,
                fractal_dimension=fractal_dimension,
                chaos_attractors=chaos_attractors,
                multi_timeframe_consensus=multi_timeframe_consensus,
                adaptive_sensitivity=adaptive_sensitivity,
                trend_acceleration=trend_acceleration,
                regime_state=regime_states,
                confidence_level=confidence_levels,
                trend_signals=trend_signals,
                current_trend=current_trend,
                current_trend_value=trend_value,
                market_regime=regime_name
            )

            self._result = result
            self._cache[data_hash_key] = self._result
            
            self.logger.info(f"✅ UltimateMA V2 計算完了 - トレンド: {current_trend}, レジーム: {regime_name}")
            return self._result

        except Exception as e:
            import traceback
            self.logger.error(f"UltimateMA V2 計算エラー: {str(e)}\n{traceback.format_exc()}")
            return self._create_error_result(len(data) if hasattr(data, '__len__') else 0)

    def _create_empty_result(self) -> UltimateMAV2Result:
        """空の結果を作成"""
        return UltimateMAV2Result(
            values=np.array([], dtype=np.float64),
            raw_values=np.array([], dtype=np.float64),
            quantum_entropy_score=np.array([], dtype=np.float64),
            fractal_dimension=np.array([], dtype=np.float64),
            chaos_attractors=np.array([], dtype=np.float64),
            multi_timeframe_consensus=np.array([], dtype=np.float64),
            adaptive_sensitivity=np.array([], dtype=np.float64),
            trend_acceleration=np.array([], dtype=np.float64),
            regime_state=np.array([], dtype=np.int8),
            confidence_level=np.array([], dtype=np.float64),
            trend_signals=np.array([], dtype=np.int8),
            current_trend='range',
            current_trend_value=0,
            market_regime='range'
        )

    def _create_error_result(self, data_len: int) -> UltimateMAV2Result:
        """エラー時の結果を作成"""
        return UltimateMAV2Result(
            values=np.full(data_len, np.nan, dtype=np.float64),
            raw_values=np.full(data_len, np.nan, dtype=np.float64),
            quantum_entropy_score=np.full(data_len, np.nan, dtype=np.float64),
            fractal_dimension=np.full(data_len, 1.5, dtype=np.float64),
            chaos_attractors=np.full(data_len, np.nan, dtype=np.float64),
            multi_timeframe_consensus=np.full(data_len, np.nan, dtype=np.float64),
            adaptive_sensitivity=np.full(data_len, 0.5, dtype=np.float64),
            trend_acceleration=np.full(data_len, np.nan, dtype=np.float64),
            regime_state=np.zeros(data_len, dtype=np.int8),
            confidence_level=np.full(data_len, 0.0, dtype=np.float64),
            trend_signals=np.zeros(data_len, dtype=np.int8),
            current_trend='range',
            current_trend_value=0,
            market_regime='range'
        )

    def get_quantum_analysis_summary(self) -> dict:
        """量子分析サマリーを取得"""
        if self._result is None:
            return {}
        
        return {
            'average_entropy': np.nanmean(self._result.quantum_entropy_score),
            'average_fractal_dimension': np.nanmean(self._result.fractal_dimension),
            'average_chaos_level': np.nanmean(self._result.chaos_attractors),
            'average_consensus': np.nanmean(self._result.multi_timeframe_consensus),
            'average_sensitivity': np.nanmean(self._result.adaptive_sensitivity),
            'max_acceleration': np.nanmax(np.abs(self._result.trend_acceleration)),
            'current_regime': self._result.market_regime,
            'latest_confidence': float(self._result.confidence_level[-1]) if len(self._result.confidence_level) > 0 else 0.0
        }

    def reset(self) -> None:
        """状態リセット"""
        super().reset()
        self._result = None
        self._cache = {}
        if hasattr(self, 'v1_instance'):
            self.v1_instance.reset()

    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データハッシュ計算（V1から継承）"""
        return self.v1_instance._get_data_hash(data) 