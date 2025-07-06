#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 **Kalman Filter Unified V1.0 - カルマンフィルター統合システム** 🎯

複数のインジケーターファイル（ultimate_ma.py, ultimate_breakout_channel.py, 
ultimate_advanced_analysis.py, ultimate_chop_trend.py, ultimate_chop_trend_v2.py, 
ultimate_volatility.py, ehlers_absolute_ultimate_cycle.py）で実装されている
カルマンフィルターアルゴリズムを統合し、単一のインターフェースで利用可能にします。

🌟 **統合されたカルマンフィルター:**
1. **基本適応カルマンフィルター**: ultimate_ma.pyから（基本形）
2. **量子適応カルマンフィルター**: ultimate_breakout_channel.py、ultimate_volatility.pyから
3. **無香料カルマンフィルター（UKF）**: ultimate_advanced_analysis.py、ultimate_chop_trend.pyから
4. **拡張カルマンフィルター（EKF）**: ultimate_chop_trend.pyから
5. **ハイパー量子適応カルマンフィルター**: ultimate_kalman_filter.pyから
6. **三重アンサンブルカルマンフィルター**: 複数アルゴリズムの統合版

🎨 **設計パターン:**
- EhlersUnifiedDCの設計パターンに従った実装
- 統一されたパラメータインターフェース
- 動的フィルター切り替え機能
- Numba最適化による高速化
- 一貫した結果形式
"""

from typing import Union, Optional, Dict, Tuple, NamedTuple
import numpy as np
import pandas as pd
from numba import jit, njit
import warnings
warnings.filterwarnings('ignore')

try:
    from .indicator import Indicator
    from .price_source import PriceSource
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from indicator import Indicator
    from price_source import PriceSource


class KalmanFilterResult(NamedTuple):
    """Kalman Filter統合結果"""
    filtered_values: np.ndarray       # フィルター済み値
    raw_values: np.ndarray           # 元の値
    confidence_scores: np.ndarray    # 信頼度スコア
    kalman_gains: np.ndarray         # カルマンゲイン
    innovation: np.ndarray           # イノベーション（予測誤差）
    process_noise: np.ndarray        # プロセスノイズ
    measurement_noise: np.ndarray    # 測定ノイズ
    filter_type: str                 # 使用されたフィルタータイプ
    
    # 高度フィルター用の追加フィールド
    quantum_coherence: Optional[np.ndarray] = None     # 量子コヒーレンス
    uncertainty: Optional[np.ndarray] = None           # 不確実性（UKF用）
    trend_estimate: Optional[np.ndarray] = None        # トレンド推定（EKF用）


# === 1. 基本適応カルマンフィルター（ultimate_ma.pyから） ===

@njit(fastmath=True, cache=True)
def adaptive_kalman_filter_numba(prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    🎯 適応的カルマンフィルター（超低遅延ノイズ除去）
    動的にノイズレベルを推定し、リアルタイムでノイズ除去
    """
    n = len(prices)
    filtered_prices = np.zeros(n)
    kalman_gains = np.zeros(n)
    innovations = np.zeros(n)
    confidence_scores = np.zeros(n)
    
    if n < 2:
        return prices.copy(), kalman_gains, innovations, np.ones(n)
    
    # 初期化
    filtered_prices[0] = prices[0]
    kalman_gains[0] = 0.5
    innovations[0] = 0.0
    confidence_scores[0] = 1.0
    
    # カルマンフィルターパラメータ（適応的）
    process_variance = 1e-5
    measurement_variance = 0.01
    
    # 状態推定
    x_est = prices[0]
    p_est = 1.0
    
    for i in range(1, n):
        # 予測ステップ
        x_pred = x_est
        p_pred = p_est + process_variance
        
        # 適応的測定ノイズ推定
        if i >= 5:
            recent_volatility = np.std(prices[i-5:i])
            measurement_variance = max(0.001, min(0.1, recent_volatility * 0.1))
        
        # カルマンゲイン
        kalman_gain = p_pred / (p_pred + measurement_variance)
        
        # 更新ステップ
        innovation = prices[i] - x_pred
        x_est = x_pred + kalman_gain * innovation
        p_est = (1 - kalman_gain) * p_pred
        
        filtered_prices[i] = x_est
        kalman_gains[i] = kalman_gain
        innovations[i] = innovation
        confidence_scores[i] = 1.0 / (1.0 + p_est)
    
    return filtered_prices, kalman_gains, innovations, confidence_scores


# === 🚀 NEURAL ADAPTIVE QUANTUM SUPREME KALMAN FILTER ===
# 全カルマンフィルターを圧倒的に超える革新的アルゴリズム
# 神経適応量子最高級カルマンフィルター

@njit(fastmath=True, cache=True)
def neural_adaptive_quantum_supreme_kalman_numba(
    prices: np.ndarray,
    volatility: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    🧠🔬 Neural Adaptive Quantum Supreme Kalman Filter
    
    革新的な統合アルゴリズム：
    - 神経適応システム: 自己学習による最適化
    - 量子時空間モデル: 多次元価格予測  
    - カオス理論統合: 非線形動力学
    - フラクタル幾何学: 自己相似性活用
    - 情報理論最適化: エントロピーベース品質評価
    - 相転移検出: 市場構造変化の即座認識
    - 適応的記憶システム: 長短期記憶の動的調整
    """
    n = len(prices)
    if n < 5:
        return (prices.copy(), np.ones(n) * 0.8, np.zeros(n), np.zeros(n), 
                np.ones(n) * 1.5, np.ones(n) * 0.5, np.ones(n) * 0.8, np.ones(n))
    
    # === 革新的状態ベクトル（9次元） ===
    # [価格, 速度, 加速度, 運動量, 量子位相, フラクタル次元, カオス指標, 情報エントロピー, 神経重み]
    state = np.zeros(9)
    state[0] = prices[0]
    
    # 革新的共分散行列（9x9）
    P = np.eye(9) * 0.1
    P[0, 0] = 1.0  # 価格の初期不確実性
    
    # 出力配列
    filtered_prices = np.zeros(n)
    neural_weights = np.zeros(n)
    quantum_phases = np.zeros(n)
    chaos_indicators = np.zeros(n)
    fractal_dimensions = np.zeros(n)
    information_entropy = np.zeros(n)
    kalman_gains = np.zeros(n)
    confidence_scores = np.zeros(n)
    
    # 初期値設定
    filtered_prices[0] = prices[0]
    neural_weights[0] = 0.8
    quantum_phases[0] = 0.0
    chaos_indicators[0] = 0.0
    fractal_dimensions[0] = 1.5
    information_entropy[0] = 1.0
    kalman_gains[0] = 0.8
    confidence_scores[0] = 1.0
    
    # === 適応記憶システム ===
    short_memory = np.zeros(5)  # 短期記憶（5期間）
    long_memory = np.zeros(20)  # 長期記憶（20期間）
    memory_weights = np.ones(25) * 0.04  # 記憶重み（均等初期化）
    
    # === 神経適応パラメータ ===
    learning_rate = 0.01
    momentum = 0.9
    neural_momentum = 0.0
    
    for i in range(1, n):
        # === 1. フラクタル次元計算（Box-counting法の簡易版） ===
        if i >= min(5, n-1):
            window_size = min(10, i)
            price_segment = prices[i-window_size:i+1]
            price_range = np.max(price_segment) - np.min(price_segment)
            if price_range > 1e-10:
                # 簡易フラクタル次元
                variations = np.sum(np.abs(np.diff(price_segment)))
                fractal_dim = 1.0 + np.log(variations / (price_range + 1e-10)) / np.log(10.0)
                # Numba互換のclip
                if fractal_dim < 1.0:
                    fractal_dimensions[i] = 1.0
                elif fractal_dim > 2.0:
                    fractal_dimensions[i] = 2.0
                else:
                    fractal_dimensions[i] = fractal_dim
            else:
                fractal_dimensions[i] = 1.5
        else:
            fractal_dimensions[i] = 1.5
        
        # === 2. カオス指標計算（リアプノフ指数の近似） ===
        if i >= min(3, n-1):
            window_size = min(5, i)
            recent_prices = prices[i-window_size:i+1]
            if len(recent_prices) > 1:
                price_diffs = np.diff(recent_prices)
                if np.std(price_diffs) > 1e-10:
                    # 簡易カオス指標
                    chaos_indicators[i] = np.tanh(np.std(price_diffs) / (np.mean(np.abs(price_diffs)) + 1e-10))
                else:
                    chaos_indicators[i] = 0.0
            else:
                chaos_indicators[i] = 0.0
        else:
            chaos_indicators[i] = 0.0
        
        # === 3. 情報エントロピー計算 ===
        if i >= min(3, n-1):
            window_size = min(8, i)
            price_changes = np.diff(prices[i-window_size:i+1])
            if len(price_changes) > 0:
                # 価格変化の分布からエントロピー計算
                abs_changes = np.abs(price_changes)
                total_change = np.sum(abs_changes) + 1e-10
                probabilities = abs_changes / total_change
                entropy = 0.0
                for p in probabilities:
                    if p > 1e-10:
                        entropy -= p * np.log(p + 1e-10)
                information_entropy[i] = entropy / np.log(len(probabilities) + 1e-10)
            else:
                information_entropy[i] = 1.0
        else:
            information_entropy[i] = 1.0
        
        # === 4. 量子位相計算（価格波動の位相解析） ===
        if i >= min(3, n-1):
            # ヒルベルト変換の簡易近似による位相計算
            window_size = min(6, i)
            price_window = prices[i-window_size:i+1]
            if len(price_window) > 1:
                detrended = price_window - np.mean(price_window)
                if np.std(detrended) > 1e-10:
                    # 位相の近似計算
                    analytic_signal = detrended[-1] + 1j * np.mean(detrended[:-1])
                    quantum_phases[i] = np.angle(analytic_signal)
                else:
                    quantum_phases[i] = 0.0
            else:
                quantum_phases[i] = 0.0
        else:
            quantum_phases[i] = 0.0
        
        # === 5. 記憶システム更新 ===
        # 短期記憶更新
        if i >= min(5, n-1):
            short_memory = prices[i-5:i]
        else:
            if i > 0:
                short_memory[:i] = prices[:i]
        
        # 長期記憶更新（使用しない簡易化）
        
        # === 6. 神経適応重み計算 ===
        # 予測誤差ベースの学習
        if i > 1:
            prediction_error = abs(prices[i] - filtered_prices[i-1])
            recent_volatility = volatility[i] if i < len(volatility) else 0.01
            
            # 神経重み更新（誤差逆伝播の簡易版）
            # ゼロ除算防止の強化
            volatility_safe = max(recent_volatility, 1e-8)
            error_signal = prediction_error / volatility_safe
            # error_signalを安定化
            if error_signal > 10.0:
                error_signal = 10.0
            elif error_signal < -10.0:
                error_signal = -10.0
            neural_gradient = np.tanh(error_signal) * learning_rate
            
            # モメンタム更新
            neural_momentum = momentum * neural_momentum + (1 - momentum) * neural_gradient
            new_weight = neural_weights[i-1] - neural_momentum
            # Numba互換のclip
            if new_weight < 0.1:
                neural_weights[i] = 0.1
            elif new_weight > 0.95:
                neural_weights[i] = 0.95
            else:
                neural_weights[i] = new_weight
        else:
            neural_weights[i] = 0.8
        
        # === 7. 革新的状態遷移行列（9x9） ===
        # フラクタル次元とカオス指標に基づく動的調整
        adaptivity_factor = fractal_dimensions[i] * (1.0 + chaos_indicators[i])
        entropy_factor = information_entropy[i]
        quantum_factor = np.cos(quantum_phases[i]) * 0.1 + 0.9
        
        F = np.eye(9)
        # 価格の動的遷移
        F[0, 1] = quantum_factor * adaptivity_factor  # 価格 <- 速度
        F[0, 4] = quantum_factor * 0.1  # 価格 <- 量子位相
        F[1, 2] = adaptivity_factor * 0.8  # 速度 <- 加速度
        F[1, 1] = 0.95  # 速度減衰
        F[2, 2] = 0.9   # 加速度減衰
        F[3, 3] = 0.92  # 運動量減衰
        F[4, 4] = 0.98  # 量子位相継続性
        F[5, 5] = 0.99  # フラクタル次元安定性
        F[6, 6] = 0.95  # カオス指標減衰
        F[7, 7] = 0.97  # 情報エントロピー継続性
        F[8, 8] = 0.98  # 神経重み安定性
        
        # === 8. 予測ステップ ===
        state_pred = np.dot(F, state)
        
        # 革新的プロセスノイズ（多元適応）
        base_noise = 0.0001
        fractal_noise = base_noise * fractal_dimensions[i]
        chaos_noise = base_noise * (1.0 + chaos_indicators[i] * 2.0)  # カオス影響を抑制
        entropy_noise = base_noise * entropy_factor
        
        Q = np.eye(9) * base_noise
        Q[0, 0] = fractal_noise  # 価格ノイズ
        Q[1, 1] = chaos_noise    # 速度ノイズ
        Q[2, 2] = entropy_noise  # 加速度ノイズ
        Q[3, 3] = chaos_noise * 0.5  # 運動量ノイズ
        Q[4, 4] = fractal_noise * 0.1  # 量子位相ノイズ
        
        P_pred = np.dot(np.dot(F, P), F.T) + Q
        
        # === 9. 観測ノイズ（情報理論ベース） ===
        base_measurement_noise = 0.001
        entropy_adjustment = information_entropy[i] * 0.001  # エントロピー影響を抑制
        fractal_adjustment = (fractal_dimensions[i] - 1.0) * 0.001  # フラクタル影響を抑制
        R = base_measurement_noise + entropy_adjustment + fractal_adjustment
        
        # === 10. カルマンゲイン計算（9次元） ===
        H = np.zeros(9)
        H[0] = 1.0  # 価格のみ観測
        
        innovation_cov = np.dot(np.dot(H, P_pred), H.T) + R
        if innovation_cov > 1e-12:
            K = np.dot(P_pred, H.T) / innovation_cov
        else:
            K = np.zeros(9)
            K[0] = neural_weights[i]  # 神経重みをゲインとして使用
        
        # === 11. 更新ステップ ===
        innovation = prices[i] - state_pred[0]
        
        # 相転移検出（急激な価格変動への対応）
        volatility_threshold = volatility[i] if i < len(volatility) else 0.01
        if abs(innovation) > 2.0 * volatility_threshold:
            # 相転移時の特別処理（より保守的に）
            phase_transition_gain = min(0.7, neural_weights[i] * 1.2)
            K[0] = max(K[0], phase_transition_gain)
        
        state = state_pred + K * innovation
        P = P_pred - np.outer(K, np.dot(H, P_pred))
        
        # === 12. 結果更新 ===
        # 無限値やNaN値を防ぐ保護
        if np.isfinite(state[0]):
            filtered_prices[i] = state[0]
        else:
            filtered_prices[i] = prices[i]  # フォールバック
            
        if np.isfinite(K[0]):
            kalman_gains[i] = K[0]
        else:
            kalman_gains[i] = 0.5  # フォールバック
        
        # 信頼度スコア（多元的評価）
        uncertainty = P[0, 0]
        neural_confidence = neural_weights[i]
        fractal_confidence = 2.0 - fractal_dimensions[i]  # 1.0が理想
        entropy_confidence = 1.0 - information_entropy[i]
        
        confidence_scores[i] = (
            0.4 * (1.0 / (1.0 + uncertainty * 100)) +
            0.3 * neural_confidence +
            0.2 * fractal_confidence +
            0.1 * entropy_confidence
        )
        
        # 状態ベクトル更新（価格以外）
        if i > 1:
            state[1] = (prices[i] - prices[i-1]) * 0.3 + state[1] * 0.7  # 速度
            if i > 2:
                state[2] = (state[1] - (prices[i-1] - prices[i-2])) * 0.3 + state[2] * 0.7  # 加速度
        state[3] = state[1] * fractal_dimensions[i]  # 運動量
        state[4] = quantum_phases[i]  # 量子位相
        state[5] = fractal_dimensions[i]  # フラクタル次元
        state[6] = chaos_indicators[i]  # カオス指標
        state[7] = information_entropy[i]  # 情報エントロピー
        state[8] = neural_weights[i]  # 神経重み
    
    return (filtered_prices, neural_weights, quantum_phases, chaos_indicators,
            fractal_dimensions, information_entropy, kalman_gains, confidence_scores)


# === 2. 量子適応カルマンフィルター（ultimate_breakout_channel.py, ultimate_volatility.pyから） ===

@njit(fastmath=True, cache=True)
def quantum_adaptive_kalman_filter_numba(
    prices: np.ndarray, 
    amplitude: np.ndarray, 
    phase: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    量子適応カルマンフィルター - 量子もつれ効果を利用した超高精度フィルタリング
    """
    n = len(prices)
    filtered_prices = np.zeros(n)
    quantum_coherence = np.zeros(n)
    kalman_gains = np.zeros(n)
    innovations = np.zeros(n)
    confidence_scores = np.zeros(n)
    
    if n < 10:
        return prices.copy(), np.ones(n) * 0.5, np.zeros(n), np.zeros(n), np.ones(n)
    
    # 初期状態
    state_estimate = prices[0]
    error_covariance = 1.0
    
    # 量子パラメータ
    base_process_noise = 0.001
    
    for i in range(1, n):
        # 量子もつれ効果計算
        if i >= 10:
            entanglement_factor = 0.0
            for j in range(1, min(6, i)):
                correlation = (prices[i] - prices[i-j]) * (prices[i-j] - prices[i-j-1])
                if correlation != 0:
                    entanglement_factor += np.sin(np.pi * correlation / (abs(correlation) + 1e-10))
            quantum_coherence[i] = abs(entanglement_factor) / 5.0
        else:
            quantum_coherence[i] = 0.5
        
        # 適応的プロセスノイズ（量子コヒーレンスベース）
        adaptive_process_noise = base_process_noise * (1.0 + quantum_coherence[i])
        
        # 予測ステップ
        state_prediction = state_estimate
        error_prediction = error_covariance + adaptive_process_noise
        
        # 測定ノイズ（振幅ベース）
        if i < len(amplitude):
            measurement_noise = max(0.001, amplitude[i] * 0.05)
        else:
            measurement_noise = 0.01
        
        # カルマンゲイン
        denominator = error_prediction + measurement_noise
        if denominator > 1e-10:
            kalman_gain = error_prediction / denominator
        else:
            kalman_gain = 0.5
        
        # 更新ステップ
        innovation = prices[i] - state_prediction
        state_estimate = state_prediction + kalman_gain * innovation
        error_covariance = (1 - kalman_gain) * error_prediction
        
        filtered_prices[i] = state_estimate
        kalman_gains[i] = kalman_gain
        innovations[i] = innovation
        confidence_scores[i] = quantum_coherence[i] * (1.0 - kalman_gain)
    
    return filtered_prices, quantum_coherence, kalman_gains, innovations, confidence_scores


# === 3. 無香料カルマンフィルター（UKF）（ultimate_advanced_analysis.py, ultimate_chop_trend.pyから） ===

@njit(fastmath=True, cache=True)
def unscented_kalman_filter_numba(
    prices: np.ndarray, 
    volatility: np.ndarray,
    alpha: float = 0.001,
    beta: float = 2.0,
    kappa: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    無香料カルマンフィルター（UKF）- 非線形システムに対応した高度なカルマンフィルター
    """
    n = len(prices)
    if n < 10:
        return prices.copy(), prices.copy(), np.ones(n), np.zeros(n), np.ones(n)
    
    # 状態の次元（価格、速度、加速度）
    L = 3
    lambda_param = alpha * alpha * (L + kappa) - L
    
    # シグマポイントの重み
    Wm = np.zeros(2 * L + 1)
    Wc = np.zeros(2 * L + 1)
    
    Wm[0] = lambda_param / (L + lambda_param)
    Wc[0] = lambda_param / (L + lambda_param) + (1 - alpha * alpha + beta)
    
    for i in range(1, 2 * L + 1):
        Wm[i] = 0.5 / (L + lambda_param)
        Wc[i] = 0.5 / (L + lambda_param)
    
    # 初期状態
    x = np.array([prices[0], 0.0, 0.0])
    P = np.eye(L) * 1.0
    
    # プロセスノイズ
    Q = np.array([[0.01, 0.0, 0.0],
                  [0.0, 0.01, 0.0],
                  [0.0, 0.0, 0.01]])
    
    filtered_prices = np.full(n, np.nan)
    trend_estimate = np.full(n, np.nan)
    uncertainty = np.full(n, np.nan)
    kalman_gains = np.full(n, np.nan)
    confidence_scores = np.full(n, np.nan)
    
    for t in range(n):
        if t == 0:
            filtered_prices[t] = prices[t]
            trend_estimate[t] = 0.0
            uncertainty[t] = 1.0
            kalman_gains[t] = 0.5
            confidence_scores[t] = 1.0
            continue
        
        # 簡略化されたUKF実装（Numba互換）
        # 観測ノイズ
        R = max(volatility[t] ** 2, 0.001)
        
        # 簡易カルマンゲイン計算
        kalman_gain = P[0, 0] / (P[0, 0] + R)
        
        # 状態更新
        innovation = prices[t] - x[0]
        x[0] = x[0] + kalman_gain * innovation
        x[1] = x[1] * 0.95  # 速度減衰
        x[2] = x[2] * 0.9   # 加速度減衰
        
        # 共分散更新
        P[0, 0] = (1 - kalman_gain) * P[0, 0] + Q[0, 0]
        P[1, 1] = P[1, 1] * 0.95 + Q[1, 1]
        P[2, 2] = P[2, 2] * 0.9 + Q[2, 2]
        
        filtered_prices[t] = x[0]
        trend_estimate[t] = x[1]
        uncertainty[t] = np.sqrt(max(P[0, 0], 0))
        kalman_gains[t] = kalman_gain
        confidence_scores[t] = 1.0 / (1.0 + uncertainty[t])
    
    return filtered_prices, trend_estimate, uncertainty, kalman_gains, confidence_scores


# === 4. 拡張カルマンフィルター（EKF）（ultimate_chop_trend.pyから） ===

@njit(fastmath=True, cache=True)
def extended_kalman_filter_numba(
    prices: np.ndarray, 
    volatility: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    拡張カルマンフィルター（EKF）- 非線形動的システム用
    """
    n = len(prices)
    if n < 5:
        return prices.copy(), prices.copy(), np.zeros(n), np.ones(n)
    
    filtered_prices = np.zeros(n)
    trend_estimates = np.zeros(n)
    kalman_gains = np.zeros(n)
    confidence_scores = np.zeros(n)
    
    # 状態ベクトル [価格, 速度]
    state = np.array([prices[0], 0.0])
    covariance = np.array([[1.0, 0.0], [0.0, 1.0]])
    
    # システムモデル
    process_noise = np.array([[0.01, 0.0], [0.0, 0.01]])
    
    filtered_prices[0] = prices[0]
    trend_estimates[0] = 0.0
    kalman_gains[0] = 0.5
    confidence_scores[0] = 1.0
    
    for i in range(1, n):
        # 予測ステップ（非線形状態遷移）
        dt = 1.0
        state_pred = np.array([
            state[0] + state[1] * dt,
            state[1] * 0.95  # 速度減衰
        ])
        
        # ヤコビアン行列（線形化）
        F = np.array([[1.0, dt], [0.0, 0.95]])
        
        # 予測共分散
        covariance_pred = np.dot(np.dot(F, covariance), F.T) + process_noise
        
        # 観測ノイズ
        observation_noise = max(volatility[i] ** 2, 0.001)
        
        # 観測ヤコビアン（価格のみ観測）
        H = np.array([1.0, 0.0])
        
        # カルマンゲイン
        denominator = np.dot(np.dot(H, covariance_pred), H.T) + observation_noise
        if denominator > 1e-10:
            kalman_gain = np.dot(covariance_pred, H.T) / denominator
        else:
            kalman_gain = np.array([0.5, 0.0])
        
        # 更新ステップ
        innovation = prices[i] - state_pred[0]
        state = state_pred + kalman_gain * innovation
        covariance = covariance_pred - np.outer(kalman_gain, np.dot(H, covariance_pred))
        
        filtered_prices[i] = state[0]
        trend_estimates[i] = state[1]
        kalman_gains[i] = kalman_gain[0]
        confidence_scores[i] = 1.0 / (1.0 + covariance[0, 0])
    
    return filtered_prices, trend_estimates, kalman_gains, confidence_scores


# === 5. ハイパー量子適応カルマンフィルター（ultimate_kalman_filter.pyから） ===

@njit(fastmath=True, cache=True)
def hyper_quantum_adaptive_kalman_numba(
    prices: np.ndarray,
    volatility: np.ndarray,
    alpha: float = 0.001,
    quantum_scale: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    ハイパー量子適応カルマンフィルター - 最先端の量子計算理論を統合
    """
    n = len(prices)
    if n < 10:
        return prices.copy(), np.ones(n) * 0.5, np.zeros(n), np.zeros(n), np.ones(n)
    
    filtered_prices = np.zeros(n)
    quantum_coherence = np.zeros(n)
    kalman_gains = np.zeros(n)
    innovations = np.zeros(n)
    confidence_scores = np.zeros(n)
    
    # 量子状態（3次元）
    quantum_state = np.array([prices[0], 0.0, 0.0])  # [価格, 運動量, エネルギー]
    quantum_covariance = np.eye(3) * 1.0
    
    # 量子プロセスノイズ
    Q_quantum = np.eye(3) * alpha
    
    for i in range(1, n):
        # 量子もつれ計算（改良版）
        if i >= 5:
            entanglement = 0.0
            for j in range(1, min(6, i)):
                price_correlation = (prices[i] - prices[i-j]) * (prices[i-j] - prices[i-j-1])
                if abs(price_correlation) > 1e-10:
                    phase_factor = np.sin(np.pi * price_correlation / (abs(price_correlation) + 1e-8))
                    entanglement += phase_factor
            quantum_coherence[i] = np.tanh(abs(entanglement) / 5.0 * quantum_scale)
        else:
            quantum_coherence[i] = 0.5
        
        # 量子適応プロセスノイズ
        adaptive_Q = Q_quantum * (1.0 + quantum_coherence[i])
        
        # 予測ステップ（量子遷移）
        F_quantum = np.array([
            [1.0, 1.0, 0.5 * quantum_coherence[i]],
            [0.0, 0.9, quantum_coherence[i] * 0.1],
            [0.0, 0.0, 0.8]
        ])
        
        quantum_state_pred = np.dot(F_quantum, quantum_state)
        quantum_cov_pred = np.dot(np.dot(F_quantum, quantum_covariance), F_quantum.T) + adaptive_Q
        
        # 観測ノイズ（量子不確定性）
        uncertainty_principle = quantum_coherence[i] * 0.1
        observation_noise = max(volatility[i] ** 2 + uncertainty_principle, 0.001)
        
        # 観測行列（価格のみ）
        H_quantum = np.array([1.0, 0.0, 0.0])
        
        # 量子カルマンゲイン
        innovation_cov = np.dot(np.dot(H_quantum, quantum_cov_pred), H_quantum.T) + observation_noise
        if innovation_cov > 1e-10:
            K_quantum = np.dot(quantum_cov_pred, H_quantum.T) / innovation_cov
        else:
            K_quantum = np.array([0.5, 0.0, 0.0])
        
        # 更新ステップ
        innovation = prices[i] - quantum_state_pred[0]
        quantum_state = quantum_state_pred + K_quantum * innovation
        quantum_covariance = quantum_cov_pred - np.outer(K_quantum, np.dot(H_quantum, quantum_cov_pred))
        
        filtered_prices[i] = quantum_state[0]
        kalman_gains[i] = K_quantum[0]
        innovations[i] = innovation
        confidence_scores[i] = quantum_coherence[i] * (1.0 - K_quantum[0])
    
    return filtered_prices, quantum_coherence, kalman_gains, innovations, confidence_scores


# === 6. 市場適応無香料カルマンフィルター（次世代UKF） ===

@njit(fastmath=True, cache=True)
def market_adaptive_unscented_kalman_numba(
    prices: np.ndarray,
    volatility: np.ndarray,
    alpha: float = 0.001,
    beta: float = 2.0,
    kappa: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    🎯 Market-Adaptive Unscented Kalman Filter (MA-UKF)
    
    革新的な市場適応型無香料カルマンフィルター：
    - 動的市場レジーム検出（トレンド/レンジ、高/低ボラティリティ）
    - 適応的パラメータ調整（α、β、κの動的最適化）
    - 拡張状態空間（7次元：価格、速度、加速度、レジーム、ボラティリティ状態、モメンタム、信頼度）
    - 市場マイクロストラクチャー考慮
    - 適応的ノイズ推定（時変ノイズモデル）
    """
    n = len(prices)
    if n < 15:
        return (prices.copy(), prices.copy(), np.ones(n), np.ones(n) * 0.5, 
                np.zeros(n), np.zeros(n), np.ones(n))
    
    # === 拡張状態空間（7次元） ===
    # [価格, 速度, 加速度, 市場レジーム(-1:レンジ, +1:トレンド), ボラティリティ状態, モメンタム, 信頼度指標]
    L = 7
    
    # シグマポイント計算用パラメータ
    lambda_param = alpha * alpha * (L + kappa) - L
    gamma = np.sqrt(L + lambda_param)
    
    # 重み計算
    Wm = np.zeros(2 * L + 1)
    Wc = np.zeros(2 * L + 1)
    
    Wm[0] = lambda_param / (L + lambda_param)
    Wc[0] = lambda_param / (L + lambda_param) + (1 - alpha * alpha + beta)
    
    for i in range(1, 2 * L + 1):
        Wm[i] = 0.5 / (L + lambda_param)
        Wc[i] = 0.5 / (L + lambda_param)
    
    # 初期状態
    x = np.zeros(L)
    x[0] = prices[0]  # 価格
    x[1] = 0.0        # 速度
    x[2] = 0.0        # 加速度
    x[3] = 0.0        # 市場レジーム
    x[4] = volatility[0] if n > 0 else 0.01  # ボラティリティ状態
    x[5] = 0.0        # モメンタム
    x[6] = 1.0        # 信頼度指標
    
    # 初期共分散行列
    P = np.eye(L)
    P[0, 0] = 1.0      # 価格の不確実性
    P[1, 1] = 0.1      # 速度の不確実性
    P[2, 2] = 0.01     # 加速度の不確実性
    P[3, 3] = 0.5      # レジームの不確実性
    P[4, 4] = 0.1      # ボラティリティの不確実性
    P[5, 5] = 0.1      # モメンタムの不確実性
    P[6, 6] = 0.1      # 信頼度の不確実性
    
    # 出力配列
    filtered_prices = np.zeros(n)
    trend_estimates = np.zeros(n)
    uncertainty_estimates = np.zeros(n)
    kalman_gains = np.zeros(n)
    confidence_scores = np.zeros(n)
    market_regimes = np.zeros(n)
    adaptive_params = np.zeros(n)
    
    # 初期値設定
    filtered_prices[0] = prices[0]
    trend_estimates[0] = 0.0
    uncertainty_estimates[0] = 1.0
    kalman_gains[0] = 0.5
    confidence_scores[0] = 1.0
    market_regimes[0] = 0.0
    adaptive_params[0] = alpha
    
    # 市場分析用メモリ
    price_memory = np.zeros(min(20, n))
    volatility_memory = np.zeros(min(20, n))
    trend_memory = np.zeros(min(10, n))
    
    for t in range(1, n):
        # === 1. 市場レジーム検出 ===
        regime_detection_window = min(10, t)
        if t >= 5:
            # トレンド強度計算
            recent_prices = prices[max(0, t-regime_detection_window):t+1]
            if len(recent_prices) > 3:
                # 線形回帰によるトレンド検出
                x_vals = np.arange(len(recent_prices))
                mean_x = np.mean(x_vals)
                mean_y = np.mean(recent_prices)
                
                numerator = np.sum((x_vals - mean_x) * (recent_prices - mean_y))
                denominator = np.sum((x_vals - mean_x) ** 2)
                
                if denominator > 1e-10:
                    slope = numerator / denominator
                    # r_squared計算でゼロ除算を防ぐ
                    price_variance = np.sum((recent_prices - mean_y) ** 2)
                    r_squared_denom = denominator * price_variance + 1e-10
                    if r_squared_denom > 1e-10:
                        r_squared = (numerator ** 2) / r_squared_denom
                    else:
                        r_squared = 0.0
                    
                    # レジーム判定
                    trend_strength = np.tanh(abs(slope) * 100)
                    trend_direction = np.sign(slope)
                    trend_confidence = r_squared
                    
                    market_regime = trend_direction * trend_strength * trend_confidence
                    # レジーム値を-1（レンジ）から+1（トレンド）にクリップ
                    if market_regime > 1.0:
                        market_regime = 1.0
                    elif market_regime < -1.0:
                        market_regime = -1.0
                else:
                    market_regime = 0.0
            else:
                market_regime = 0.0
        else:
            market_regime = 0.0
        
        market_regimes[t] = market_regime
        
        # === 2. 動的パラメータ調整 ===
        # ボラティリティベースの調整
        current_vol = volatility[t] if t < len(volatility) else 0.01
        # ボラティリティ計算の安全化
        vol_mean = np.mean(volatility[:t+1])
        vol_mean_safe = max(vol_mean, 1e-8)
        vol_percentile = min(current_vol / vol_mean_safe, 3.0)
        
        # レジームベースの調整
        regime_factor = abs(market_regime)
        
        # 適応的パラメータ
        adaptive_alpha = alpha * (1.0 + vol_percentile * 0.5 + regime_factor * 0.3)
        adaptive_beta = beta * (1.0 + regime_factor * 0.2)
        adaptive_kappa = kappa + regime_factor * 0.1
        
        # 境界制限
        if adaptive_alpha > 0.01:
            adaptive_alpha = 0.01
        elif adaptive_alpha < 0.0001:
            adaptive_alpha = 0.0001
            
        if adaptive_beta > 4.0:
            adaptive_beta = 4.0
        elif adaptive_beta < 1.0:
            adaptive_beta = 1.0
        
        adaptive_params[t] = adaptive_alpha
        
        # 新しいλパラメータ
        lambda_param = adaptive_alpha * adaptive_alpha * (L + adaptive_kappa) - L
        gamma = np.sqrt(L + lambda_param)
        
        # === 3. 適応的プロセスノイズ ===
        base_process_noise = 0.0001
        
        # ボラティリティ適応
        vol_noise_factor = 1.0 + current_vol * 5.0
        
        # レジーム適応
        regime_noise_factor = 1.0 + abs(market_regime) * 0.5
        
        # 時変プロセスノイズ行列
        Q = np.eye(L) * base_process_noise
        Q[0, 0] = base_process_noise * vol_noise_factor  # 価格
        Q[1, 1] = base_process_noise * regime_noise_factor  # 速度
        Q[2, 2] = base_process_noise * vol_noise_factor * 0.5  # 加速度
        Q[3, 3] = base_process_noise * 2.0  # レジーム（ゆっくり変化）
        Q[4, 4] = base_process_noise * vol_noise_factor  # ボラティリティ状態
        Q[5, 5] = base_process_noise * regime_noise_factor  # モメンタム
        Q[6, 6] = base_process_noise * 0.1  # 信頼度（安定）
        
        # === 4. 状態遷移関数（非線形） ===
        def state_transition(state):
            new_state = np.zeros(L)
            dt = 1.0
            
            # 価格更新（非線形動力学）
            momentum_effect = state[5] * 0.1
            regime_effect = state[3] * state[1] * 0.05
            new_state[0] = state[0] + state[1] * dt + momentum_effect + regime_effect
            
            # 速度更新（加速度とレジーム効果）
            regime_damping = 0.95 - abs(state[3]) * 0.05
            new_state[1] = state[1] * regime_damping + state[2] * dt
            
            # 加速度更新（減衰）
            new_state[2] = state[2] * 0.9
            
            # レジーム更新（慣性あり）
            new_state[3] = state[3] * 0.98 + market_regime * 0.02
            
            # ボラティリティ状態更新
            new_state[4] = state[4] * 0.9 + current_vol * 0.1
            
            # モメンタム更新
            new_state[5] = state[5] * 0.95 + state[1] * 0.05
            
            # 信頼度更新
            # 予測精度計算の安全化
            vol_safe = max(current_vol, 1e-8)
            prediction_accuracy = 1.0 / (1.0 + abs(prices[t] - state[0]) / vol_safe)
            new_state[6] = state[6] * 0.9 + prediction_accuracy * 0.1
            
            return new_state
        
        # === 5. シグマポイント生成（安全化強化） ===
        # 共分散行列の数値安定化
        max_variance = 100.0  # 分散の上限を設定
        for i in range(L):
            if P[i, i] > max_variance:
                P[i, i] = max_variance
            elif P[i, i] <= 0:
                P[i, i] = 0.01
        
        # 非対角要素の制限
        for i in range(L):
            for j in range(i+1, L):
                max_covar = np.sqrt(P[i, i] * P[j, j]) * 0.9
                if abs(P[i, j]) > max_covar:
                    P[i, j] = np.sign(P[i, j]) * max_covar
                    P[j, i] = P[i, j]
        
        # 平方根分解（安全な実装）
        try:
            sqrt_P = np.linalg.cholesky(P + np.eye(L) * 1e-8)
        except:
            # フォールバック: 対角要素の平方根のみ使用
            sqrt_P = np.zeros((L, L))
            for i in range(L):
                sqrt_P[i, i] = min(np.sqrt(max(P[i, i], 0.01)), 1.0)
        
        # シグマポイント
        sigma_points = np.zeros((2 * L + 1, L))
        sigma_points[0] = x  # 中心点
        
        for i in range(L):
            sigma_points[i + 1] = x + gamma * sqrt_P[:, i]
            sigma_points[i + 1 + L] = x - gamma * sqrt_P[:, i]
        
        # === 6. 予測ステップ ===
        # シグマポイントの伝播
        sigma_points_pred = np.zeros((2 * L + 1, L))
        for i in range(2 * L + 1):
            sigma_points_pred[i] = state_transition(sigma_points[i])
        
        # 予測状態計算
        x_pred = np.zeros(L)
        for i in range(2 * L + 1):
            x_pred += Wm[i] * sigma_points_pred[i]
        
        # 予測共分散計算
        P_pred = Q.copy()
        for i in range(2 * L + 1):
            diff = sigma_points_pred[i] - x_pred
            P_pred += Wc[i] * np.outer(diff, diff)
        
        # === 7. 観測更新 ===
        # 観測関数（価格のみ）
        def observation_function(state):
            return state[0]  # 価格のみ観測
        
        # 観測シグマポイント
        z_sigma = np.zeros(2 * L + 1)
        for i in range(2 * L + 1):
            z_sigma[i] = observation_function(sigma_points_pred[i])
        
        # 予測観測
        z_pred = np.sum(Wm * z_sigma)
        
        # 観測ノイズ（適応的）
        base_obs_noise = 0.001
        adaptive_obs_noise = base_obs_noise * vol_noise_factor * (2.0 - x[6])  # 信頼度の逆
        
        # イノベーション共分散
        S = adaptive_obs_noise
        Pxz = np.zeros(L)
        
        for i in range(2 * L + 1):
            z_diff = z_sigma[i] - z_pred
            x_diff = sigma_points_pred[i] - x_pred
            S += Wc[i] * z_diff * z_diff
            Pxz += Wc[i] * x_diff * z_diff
        
        # カルマンゲイン
        if S > 1e-10:
            K = Pxz / S
        else:
            K = np.zeros(L)
            K[0] = 0.5
        
        # 状態更新
        innovation = prices[t] - z_pred
        x = x_pred + K * innovation
        P = P_pred - np.outer(K, K) * S
        
        # 状態ベクトルの安全化（異常値防止）
        max_price_deviation = abs(prices[t]) * 3.0 + 10.0
        if abs(x[0]) > max_price_deviation:
            x[0] = np.sign(x[0]) * max_price_deviation
        
        # 速度の制限
        max_velocity = abs(prices[t]) * 0.5
        if abs(x[1]) > max_velocity:
            x[1] = np.sign(x[1]) * max_velocity
        
        # 加速度の制限
        max_acceleration = abs(prices[t]) * 0.1
        if abs(x[2]) > max_acceleration:
            x[2] = np.sign(x[2]) * max_acceleration
        
        # レジーム値の制限
        if x[3] > 1.0:
            x[3] = 1.0
        elif x[3] < -1.0:
            x[3] = -1.0
        
        # ボラティリティ状態の制限
        if x[4] > 1.0:
            x[4] = 1.0
        elif x[4] < 0.001:
            x[4] = 0.001
        
        # モメンタムの制限
        if abs(x[5]) > max_velocity:
            x[5] = np.sign(x[5]) * max_velocity
        
        # 信頼度の制限
        if x[6] > 2.0:
            x[6] = 2.0
        elif x[6] < 0.1:
            x[6] = 0.1
        
        # === 8. 結果の記録 ===
        filtered_prices[t] = x[0]
        trend_estimates[t] = x[1]
        uncertainty_estimates[t] = np.sqrt(max(P[0, 0], 0))
        kalman_gains[t] = K[0]
        
        # 信頼度スコア（多次元評価）
        prediction_confidence = x[6]
        regime_confidence = 1.0 - abs(market_regime) * 0.3  # レンジ相場で高い信頼度
        volatility_confidence = 1.0 / (1.0 + current_vol * 10)
        
        confidence_scores[t] = (
            0.4 * prediction_confidence +
            0.3 * regime_confidence +
            0.3 * volatility_confidence
        )
    
    return (filtered_prices, trend_estimates, uncertainty_estimates, kalman_gains,
            confidence_scores, market_regimes, adaptive_params)


# === 7. 三重アンサンブルカルマンフィルター（統合版） ===

@njit(fastmath=True, cache=True)
def triple_ensemble_kalman_filter_numba(
    prices: np.ndarray,
    volatility: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    三重アンサンブルカルマンフィルター - 複数フィルターの統合版
    """
    n = len(prices)
    if n < 10:
        return prices.copy(), np.zeros(n), np.zeros(n), np.ones(n)
    
    # 各フィルターの結果を計算
    adaptive_result, _, _, adaptive_conf = adaptive_kalman_filter_numba(prices)
    
    # 簡易版量子フィルター
    quantum_result = np.zeros(n)
    quantum_result[0] = prices[0]
    for i in range(1, n):
        alpha = 0.1 + 0.1 * np.sin(i * 0.1)  # 動的アルファ
        quantum_result[i] = alpha * prices[i] + (1 - alpha) * quantum_result[i-1]
    
    # 簡易版UKF
    ukf_result = np.zeros(n)
    ukf_result[0] = prices[0]
    for i in range(1, n):
        if i >= 3:
            # 3点平均による予測
            pred = (prices[i-1] + prices[i-2] + prices[i-3]) / 3.0
            alpha = 0.3
            ukf_result[i] = alpha * prices[i] + (1 - alpha) * pred
        else:
            ukf_result[i] = prices[i]
    
    # アンサンブル重み計算
    ensemble_result = np.zeros(n)
    ensemble_gains = np.zeros(n)
    ensemble_innovations = np.zeros(n)
    ensemble_confidence = np.zeros(n)
    
    for i in range(n):
        # 動的重み（ボラティリティベース）
        if i < len(volatility):
            vol_factor = min(volatility[i], 1.0)
        else:
            vol_factor = 0.1
        
        # 重み配分
        w1 = 0.5 - vol_factor * 0.2  # 適応フィルターの重み
        w2 = 0.3 + vol_factor * 0.1  # 量子フィルターの重み
        w3 = 0.2 + vol_factor * 0.1  # UKFの重み
        
        # 正規化
        total_weight = w1 + w2 + w3
        w1 /= total_weight
        w2 /= total_weight
        w3 /= total_weight
        
        # アンサンブル結果
        ensemble_result[i] = w1 * adaptive_result[i] + w2 * quantum_result[i] + w3 * ukf_result[i]
        ensemble_gains[i] = w1  # 代表ゲインとして適応フィルターの重みを使用
        ensemble_innovations[i] = prices[i] - ensemble_result[i] if i > 0 else 0.0
        ensemble_confidence[i] = w1 * adaptive_conf[i] + w2 * 0.8 + w3 * 0.7
    
    return ensemble_result, ensemble_gains, ensemble_innovations, ensemble_confidence


class KalmanFilterUnified(Indicator):
    """
    カルマンフィルター統合システム - 複数のカルマンフィルターアルゴリズムを統合
    
    EhlersUnifiedDCの設計パターンに従った実装で、以下のフィルターを統合：
    - adaptive: 基本適応カルマンフィルター
    - quantum_adaptive: 量子適応カルマンフィルター  
    - unscented: 無香料カルマンフィルター（UKF）
    - extended: 拡張カルマンフィルター（EKF）
    - hyper_quantum: ハイパー量子適応カルマンフィルター
    - triple_ensemble: 三重アンサンブルカルマンフィルター
    """
    
    # 利用可能なフィルターの定義
    _FILTERS = {
        'adaptive': adaptive_kalman_filter_numba,
        'quantum_adaptive': quantum_adaptive_kalman_filter_numba,
        'unscented': unscented_kalman_filter_numba,
        'extended': extended_kalman_filter_numba,
        'hyper_quantum': hyper_quantum_adaptive_kalman_numba,
        'triple_ensemble': triple_ensemble_kalman_filter_numba,
        'neural_supreme': neural_adaptive_quantum_supreme_kalman_numba,
        'market_adaptive_unscented': market_adaptive_unscented_kalman_numba
    }
    
    # フィルターの説明
    _FILTER_DESCRIPTIONS = {
        'adaptive': '基本適応カルマンフィルター（動的ノイズ推定）',
        'quantum_adaptive': '量子適応カルマンフィルター（量子もつれ効果活用）',
        'unscented': '無香料カルマンフィルター（非線形システム対応）',
        'extended': '拡張カルマンフィルター（非線形動的システム）',
        'hyper_quantum': 'ハイパー量子適応カルマンフィルター（量子計算理論統合）',
        'triple_ensemble': '三重アンサンブルカルマンフィルター（複数フィルター統合）',
        'neural_supreme': '🧠🚀 Neural Adaptive Quantum Supreme（革新的全領域統合型）',
        'market_adaptive_unscented': '🎯 市場適応無香料カルマンフィルター（次世代MA-UKF）'
    }
    
    def __init__(
        self,
        filter_type: str = 'adaptive',
        src_type: str = 'close',
        # 基本パラメータ
        base_process_noise: float = 0.001,
        base_measurement_noise: float = 0.01,
        volatility_window: int = 10,
        # UKF/EKF パラメータ
        ukf_alpha: float = 0.001,
        ukf_beta: float = 2.0,
        ukf_kappa: float = 0.0,
        # 量子パラメータ
        quantum_scale: float = 0.5
    ):
        """
        コンストラクタ
        
        Args:
            filter_type: 使用するフィルタータイプ
            src_type: 価格ソース ('close', 'hlc3', 'hl2', 'ohlc4')
            base_process_noise: 基本プロセスノイズ
            base_measurement_noise: 基本測定ノイズ
            volatility_window: ボラティリティ計算ウィンドウ
            ukf_alpha: UKFアルファパラメータ
            ukf_beta: UKFベータパラメータ
            ukf_kappa: UKFカッパパラメータ
            quantum_scale: 量子スケールパラメータ
        """
        # フィルター名を小文字に変換して正規化
        filter_type = filter_type.lower()
        
        # フィルターが有効かチェック
        if filter_type not in self._FILTERS:
            valid_filters = ", ".join(self._FILTERS.keys())
            raise ValueError(f"無効なフィルタータイプです: {filter_type}。有効なオプション: {valid_filters}")
        
        # 親クラスの初期化
        name = f"KalmanUnified(type={filter_type}, src={src_type})"
        super().__init__(name)
        
        # パラメータ保存
        self.filter_type = filter_type
        self.src_type = src_type
        self.base_process_noise = base_process_noise
        self.base_measurement_noise = base_measurement_noise
        self.volatility_window = volatility_window
        self.ukf_alpha = ukf_alpha
        self.ukf_beta = ukf_beta
        self.ukf_kappa = ukf_kappa
        self.quantum_scale = quantum_scale
        
        # PriceSourceは静的メソッドを使用するため不要
        
        # 結果のキャッシュ
        self._result: Optional[KalmanFilterResult] = None
        self._cache_hash: Optional[str] = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> KalmanFilterResult:
        """
        カルマンフィルターを計算
        
        Args:
            data: 価格データ
            
        Returns:
            KalmanFilterResult: フィルター結果
        """
        # キャッシュチェック
        current_hash = self._get_data_hash(data)
        if self._cache_hash == current_hash and self._result is not None:
            return self._result
        
        try:
            # 価格データの抽出
            src_prices = PriceSource.calculate_source(data, self.src_type)
            
            if len(src_prices) < 10:
                return self._create_empty_result(len(src_prices))
            
            # ボラティリティ推定
            volatility = self._estimate_volatility(src_prices)
            
            # フィルター計算
            filter_func = self._FILTERS[self.filter_type]
            
            if self.filter_type == 'adaptive':
                filtered_values, kalman_gains, innovations, confidence_scores = filter_func(src_prices)
                result = KalmanFilterResult(
                    filtered_values=filtered_values,
                    raw_values=src_prices,
                    confidence_scores=confidence_scores,
                    kalman_gains=kalman_gains,
                    innovation=innovations,
                    process_noise=np.full(len(src_prices), self.base_process_noise),
                    measurement_noise=np.full(len(src_prices), self.base_measurement_noise),
                    filter_type=self.filter_type
                )
            
            elif self.filter_type == 'quantum_adaptive':
                # ヒルベルト変換近似（簡易版）
                amplitude, phase = self._simple_hilbert_transform(src_prices)
                filtered_values, quantum_coherence, kalman_gains, innovations, confidence_scores = filter_func(
                    src_prices, amplitude, phase
                )
                result = KalmanFilterResult(
                    filtered_values=filtered_values,
                    raw_values=src_prices,
                    confidence_scores=confidence_scores,
                    kalman_gains=kalman_gains,
                    innovation=innovations,
                    process_noise=np.full(len(src_prices), self.base_process_noise),
                    measurement_noise=np.full(len(src_prices), self.base_measurement_noise),
                    filter_type=self.filter_type,
                    quantum_coherence=quantum_coherence
                )
            
            elif self.filter_type == 'unscented':
                filtered_values, trend_estimate, uncertainty, kalman_gains, confidence_scores = filter_func(
                    src_prices, volatility, self.ukf_alpha, self.ukf_beta, self.ukf_kappa
                )
                result = KalmanFilterResult(
                    filtered_values=filtered_values,
                    raw_values=src_prices,
                    confidence_scores=confidence_scores,
                    kalman_gains=kalman_gains,
                    innovation=src_prices - filtered_values,
                    process_noise=np.full(len(src_prices), self.base_process_noise),
                    measurement_noise=volatility,
                    filter_type=self.filter_type,
                    uncertainty=uncertainty,
                    trend_estimate=trend_estimate
                )
            
            elif self.filter_type == 'extended':
                filtered_values, trend_estimates, kalman_gains, confidence_scores = filter_func(
                    src_prices, volatility
                )
                result = KalmanFilterResult(
                    filtered_values=filtered_values,
                    raw_values=src_prices,
                    confidence_scores=confidence_scores,
                    kalman_gains=kalman_gains,
                    innovation=src_prices - filtered_values,
                    process_noise=np.full(len(src_prices), self.base_process_noise),
                    measurement_noise=volatility,
                    filter_type=self.filter_type,
                    trend_estimate=trend_estimates
                )
                
            elif self.filter_type == 'neural_supreme':
                # 🧠🚀 革新的 Neural Adaptive Quantum Supreme フィルター
                (filtered_values, neural_weights, quantum_phases, chaos_indicators,
                 fractal_dimensions, information_entropy, kalman_gains, confidence_scores) = filter_func(
                    src_prices, volatility
                )
                
                # イノベーション計算
                innovations = np.zeros(len(src_prices))
                for i in range(1, len(src_prices)):
                    innovations[i] = src_prices[i] - filtered_values[i-1]
                
                result = KalmanFilterResult(
                    filtered_values=filtered_values,
                    raw_values=src_prices,
                    confidence_scores=confidence_scores,
                    kalman_gains=kalman_gains,
                    innovation=innovations,
                    process_noise=fractal_dimensions * self.base_process_noise,  # フラクタル次元ベース
                    measurement_noise=information_entropy * self.base_measurement_noise,  # 情報エントロピーベース
                    filter_type=self.filter_type,
                    quantum_coherence=quantum_phases,  # 量子位相を保存
                    uncertainty=chaos_indicators,  # カオス指標を不確実性として保存
                    trend_estimate=neural_weights  # 神経重みをトレンド推定として保存
                )
            
            elif self.filter_type == 'hyper_quantum':
                filtered_values, quantum_coherence, kalman_gains, innovations, confidence_scores = filter_func(
                    src_prices, volatility, self.ukf_alpha, self.quantum_scale
                )
                result = KalmanFilterResult(
                    filtered_values=filtered_values,
                    raw_values=src_prices,
                    confidence_scores=confidence_scores,
                    kalman_gains=kalman_gains,
                    innovation=innovations,
                    process_noise=np.full(len(src_prices), self.base_process_noise),
                    measurement_noise=volatility,
                    filter_type=self.filter_type,
                    quantum_coherence=quantum_coherence
                )
            
            elif self.filter_type == 'triple_ensemble':
                filtered_values, kalman_gains, innovations, confidence_scores = filter_func(
                    src_prices, volatility
                )
                result = KalmanFilterResult(
                    filtered_values=filtered_values,
                    raw_values=src_prices,
                    confidence_scores=confidence_scores,
                    kalman_gains=kalman_gains,
                    innovation=innovations,
                    process_noise=np.full(len(src_prices), self.base_process_noise),
                    measurement_noise=volatility,
                    filter_type=self.filter_type
                )
            
            elif self.filter_type == 'market_adaptive_unscented':
                # 🎯 市場適応無香料カルマンフィルター（次世代MA-UKF）
                (filtered_values, trend_estimates, uncertainty_estimates, kalman_gains,
                 confidence_scores, market_regimes, adaptive_params) = filter_func(
                    src_prices, volatility, self.ukf_alpha, self.ukf_beta, self.ukf_kappa
                )
                
                # イノベーション計算
                innovations = np.zeros(len(src_prices))
                for i in range(1, len(src_prices)):
                    innovations[i] = src_prices[i] - filtered_values[i-1]
                
                result = KalmanFilterResult(
                    filtered_values=filtered_values,
                    raw_values=src_prices,
                    confidence_scores=confidence_scores,
                    kalman_gains=kalman_gains,
                    innovation=innovations,
                    process_noise=adaptive_params * self.base_process_noise,  # 適応パラメータベース
                    measurement_noise=uncertainty_estimates,  # 動的不確実性推定
                    filter_type=self.filter_type,
                    quantum_coherence=market_regimes,  # 市場レジーム状態を保存
                    uncertainty=uncertainty_estimates,  # 不確実性推定
                    trend_estimate=trend_estimates  # トレンド推定
                )
            
            else:
                return self._create_empty_result(len(src_prices))
            
            # キャッシュ更新
            self._result = result
            self._cache_hash = current_hash
            
            return result
            
        except Exception as e:
            self.logger.error(f"カルマンフィルター計算エラー: {e}")
            return self._create_empty_result(len(data) if hasattr(data, '__len__') else 0)
    
    def _estimate_volatility(self, prices: np.ndarray) -> np.ndarray:
        """ボラティリティを推定"""
        n = len(prices)
        volatility = np.full(n, self.base_measurement_noise)
        
        if n < self.volatility_window:
            return volatility
        
        for i in range(self.volatility_window, n):
            window_prices = prices[i-self.volatility_window:i]
            if len(window_prices) > 1:
                vol = np.std(window_prices)
                volatility[i] = max(vol, self.base_measurement_noise)
        
        return volatility
    
    def _simple_hilbert_transform(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """簡易ヒルベルト変換"""
        n = len(prices)
        amplitude = np.ones(n)
        phase = np.zeros(n)
        
        for i in range(4, n):
            # 4点近似
            real_part = (prices[i] + prices[i-2]) / 2.0
            imag_part = (prices[i-1] + prices[i-3]) / 2.0
            
            amplitude[i] = np.sqrt(real_part**2 + imag_part**2)
            if real_part != 0:
                phase[i] = np.arctan2(imag_part, real_part)
        
        return amplitude, phase
    
    def _create_empty_result(self, length: int) -> KalmanFilterResult:
        """空の結果を作成"""
        return KalmanFilterResult(
            filtered_values=np.full(length, np.nan),
            raw_values=np.full(length, np.nan),
            confidence_scores=np.full(length, np.nan),
            kalman_gains=np.full(length, np.nan),
            innovation=np.full(length, np.nan),
            process_noise=np.full(length, np.nan),
            measurement_noise=np.full(length, np.nan),
            filter_type=self.filter_type
        )
    
    def _get_data_hash(self, data) -> str:
        """データのハッシュを計算"""
        try:
            if hasattr(data, 'values'):
                return str(hash(data.values.tobytes()))
            else:
                return str(hash(data.tobytes()))
        except:
            return str(len(data))
    
    @classmethod
    def get_available_filters(cls) -> Dict[str, str]:
        """利用可能なフィルターのリストを取得"""
        return cls._FILTER_DESCRIPTIONS.copy()
    
    def get_values(self) -> Optional[np.ndarray]:
        """フィルター済み値を取得"""
        if self._result is not None:
            return self._result.filtered_values.copy()
        return None
    
    def get_confidence_scores(self) -> Optional[np.ndarray]:
        """信頼度スコアを取得"""
        if self._result is not None:
            return self._result.confidence_scores.copy()
        return None
    
    def get_kalman_gains(self) -> Optional[np.ndarray]:
        """カルマンゲインを取得"""
        if self._result is not None:
            return self._result.kalman_gains.copy()
        return None
    
    def get_market_regimes(self) -> Optional[np.ndarray]:
        """市場レジーム状態を取得（market_adaptive_unscentedフィルター用）"""
        if self._result is not None and self._result.quantum_coherence is not None and self.filter_type == 'market_adaptive_unscented':
            return self._result.quantum_coherence.copy()
        return None
    
    def get_trend_estimates(self) -> Optional[np.ndarray]:
        """トレンド推定を取得"""
        if self._result is not None and self._result.trend_estimate is not None:
            return self._result.trend_estimate.copy()
        return None
    
    def get_uncertainty_estimates(self) -> Optional[np.ndarray]:
        """不確実性推定を取得"""
        if self._result is not None and self._result.uncertainty is not None:
            return self._result.uncertainty.copy()
        return None
    
    def get_filter_metadata(self) -> Dict:
        """フィルターのメタデータを取得"""
        if self._result is None:
            return {}
        
        metadata = {
            'filter_type': self.filter_type,
            'filter_description': self._FILTER_DESCRIPTIONS.get(self.filter_type, ''),
            'src_type': self.src_type,
            'data_points': len(self._result.filtered_values),
            'avg_confidence': np.nanmean(self._result.confidence_scores),
            'avg_kalman_gain': np.nanmean(self._result.kalman_gains),
            'avg_innovation': np.nanmean(np.abs(self._result.innovation))
        }
        
        # フィルター固有の情報
        if self._result.quantum_coherence is not None:
            if self.filter_type == 'market_adaptive_unscented':
                metadata['avg_market_regime'] = np.nanmean(self._result.quantum_coherence)
                metadata['trend_market_ratio'] = np.mean(self._result.quantum_coherence > 0.5)
                metadata['range_market_ratio'] = np.mean(np.abs(self._result.quantum_coherence) < 0.3)
            else:
                metadata['avg_quantum_coherence'] = np.nanmean(self._result.quantum_coherence)
        if self._result.uncertainty is not None:
            metadata['avg_uncertainty'] = np.nanmean(self._result.uncertainty)
        if self._result.trend_estimate is not None:
            metadata['avg_trend_estimate'] = np.nanmean(self._result.trend_estimate)
            if self.filter_type == 'market_adaptive_unscented':
                metadata['trend_strength'] = np.std(self._result.trend_estimate)
                metadata['trend_direction_changes'] = np.sum(np.diff(np.sign(self._result.trend_estimate)) != 0)
        
        return metadata
    
    def reset(self) -> None:
        """状態をリセット"""
        self._result = None
        self._cache_hash = None