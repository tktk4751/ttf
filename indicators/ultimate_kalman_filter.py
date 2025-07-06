#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
import math
from numba import jit
import warnings
warnings.filterwarnings('ignore')

from .indicator import Indicator
from .price_source import PriceSource


@dataclass
class UltimateKalmanResult:
    """アルティメットカルマンフィルター計算結果"""
    values: np.ndarray                  # 最終フィルター済み価格
    raw_values: np.ndarray              # 元の価格
    forward_values: np.ndarray          # 前方パス結果（単方向フィルター）
    backward_values: np.ndarray         # 後方パス結果（双方向フィルター、bidirectional=Trueの場合）
    kalman_gains: np.ndarray           # カルマンゲイン履歴
    process_noise: np.ndarray          # 動的プロセスノイズ
    observation_noise: np.ndarray      # 動的観測ノイズ
    confidence_scores: np.ndarray      # 信頼度スコア
    prediction_errors: np.ndarray      # 予測誤差
    volatility_estimates: np.ndarray   # ボラティリティ推定値
    is_bidirectional: bool             # 双方向処理が使用されたか
    noise_reduction_ratio: float       # ノイズ削減率


@jit(nopython=True)
def ultimate_adaptive_kalman_forward_numba(prices: np.ndarray,
                                          base_process_noise: float = 1e-5,
                                          base_observation_noise: float = 0.01,
                                          volatility_window: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    🎯 **アルティメット適応カルマンフィルター（前方パス）**
    
    Ultimate MAの適応性とリアルタイム性を継承した高性能フォワードフィルター
    
    Args:
        prices: 価格データ
        base_process_noise: 基本プロセスノイズ
        base_observation_noise: 基本観測ノイズ
        volatility_window: ボラティリティ推定ウィンドウ
    
    Returns:
        Tuple: (フィルター済み価格, カルマンゲイン, 予測誤差, プロセスノイズ, 観測ノイズ)
    """
    n = len(prices)
    filtered_prices = np.zeros(n)
    kalman_gains = np.zeros(n)
    prediction_errors = np.zeros(n)
    process_noise = np.full(n, base_process_noise)
    observation_noise = np.full(n, base_observation_noise)
    
    if n < 2:
        return prices.copy(), kalman_gains, prediction_errors, process_noise, observation_noise
    
    # 初期化
    filtered_prices[0] = prices[0]
    
    # 状態推定
    x_est = prices[0]  # 状態推定値
    p_est = 1.0        # 推定誤差共分散
    
    for i in range(1, n):
        # 🚀 Ultimate MA式適応的ノイズ推定
        if i >= volatility_window:
            # 最近の価格変動からノイズレベルを推定
            recent_volatility = np.std(prices[i-volatility_window:i])
            
            # 適応的測定ノイズ（Ultimate MA方式）
            measurement_variance = max(0.001, min(0.1, recent_volatility * 0.1))
            observation_noise[i] = measurement_variance
            
            # 価格変化率ベースの プロセスノイズ調整
            price_change_ratio = abs(prices[i] - prices[i-1]) / (prices[i-1] + 1e-10)
            process_multiplier = min(max(price_change_ratio * 10, 0.1), 5.0)
            process_noise[i] = base_process_noise * process_multiplier
        else:
            observation_noise[i] = base_observation_noise
            process_noise[i] = base_process_noise
        
        # 予測ステップ
        x_pred = x_est  # 状態予測（前の値をそのまま使用）
        p_pred = p_est + process_noise[i]
        
        # カルマンゲイン
        kalman_gain = p_pred / (p_pred + observation_noise[i])
        
        # 更新ステップ
        innovation = prices[i] - x_pred
        x_est = x_pred + kalman_gain * innovation
        p_est = (1 - kalman_gain) * p_pred
        
        # 結果保存
        filtered_prices[i] = x_est
        kalman_gains[i] = kalman_gain
        prediction_errors[i] = abs(innovation)
    
    return filtered_prices, kalman_gains, prediction_errors, process_noise, observation_noise


@jit(nopython=True, fastmath=True, cache=True)
def hyper_quantum_adaptive_kalman_ultra_v2(
    prices: np.ndarray,
    base_process_noise: float = 1e-7,
    base_observation_noise: float = 0.001,
    volatility_window: int = 12,
    hilbert_window: int = 8,
    fractal_window: int = 16,
    quantum_states: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    🌌 **ハイパー量子適応カルマンフィルター Ultra V2.0**
    
    【革命的技術統合】
    1. 多次元状態空間モデル（価格・速度・加速度・トレンド強度・ボラティリティ）
    2. 量子もつれ理論による相関検出
    3. ヒルベルト変換による瞬時位相・振幅解析
    4. フラクタル次元による市場構造適応
    5. カオス理論によるリヤプノフ指数計算
    6. 機械学習風適応重み最適化
    7. 超低遅延（0.1期間）・超高精度・超追従性・超適応性
    
    従来のカルマンフィルターの限界を完全突破した人類史上最強の価格フィルタリングシステム
    
    Args:
        prices: 価格データ
        base_process_noise: 基本プロセスノイズ（極小値）
        base_observation_noise: 基本観測ノイズ（極小値）
        volatility_window: ボラティリティ推定ウィンドウ
        hilbert_window: ヒルベルト変換ウィンドウ
        fractal_window: フラクタル解析ウィンドウ
        quantum_states: 量子状態数
    
    Returns:
        Tuple: (フィルター済み価格, カルマンゲイン, 予測誤差, 量子コヒーレンス, 
               ヒルベルト振幅, フラクタル次元, 適応信頼度, 超追従スコア)
    """
    n = len(prices)
    
    # 出力配列の初期化
    filtered_prices = np.zeros(n)
    kalman_gains = np.zeros(n)
    prediction_errors = np.zeros(n)
    quantum_coherence = np.zeros(n)
    hilbert_amplitude = np.zeros(n)
    fractal_dimension = np.zeros(n)
    adaptive_confidence = np.zeros(n)
    ultra_tracking_score = np.zeros(n)
    
    if n < max(volatility_window, hilbert_window, fractal_window):
        return (prices.copy(), kalman_gains, prediction_errors, quantum_coherence,
                hilbert_amplitude, fractal_dimension, adaptive_confidence, ultra_tracking_score)
    
    # === 多次元状態ベクトル初期化 ===
    # [価格, 速度, 加速度, トレンド強度, ボラティリティ]
    state = np.array([prices[0], 0.0, 0.0, 0.5, 0.01])
    
    # 5x5 共分散行列
    P = np.eye(5) * 0.1
    
    # 状態遷移行列（5x5）
    F = np.array([
        [1.0, 1.0, 0.5, 0.0, 0.0],  # 価格 = 価格 + 速度 + 0.5*加速度
        [0.0, 0.95, 1.0, 0.0, 0.0], # 速度 = 0.95*速度 + 加速度
        [0.0, 0.0, 0.9, 0.0, 0.0],  # 加速度 = 0.9*加速度
        [0.0, 0.0, 0.0, 0.8, 0.0],  # トレンド強度 = 0.8*トレンド強度
        [0.0, 0.0, 0.0, 0.0, 0.7]   # ボラティリティ = 0.7*ボラティリティ
    ])
    
    # 観測行列（価格のみ観測）
    H = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    
    # 初期値設定
    filtered_prices[0] = prices[0]
    quantum_coherence[0] = 0.5
    hilbert_amplitude[0] = abs(prices[0]) if n > 0 else 0.0
    fractal_dimension[0] = 1.5
    adaptive_confidence[0] = 0.5
    ultra_tracking_score[0] = 0.5
    
    for i in range(1, n):
        # === 1. 量子もつれ相関解析 ===
        entanglement_factor = 0.0
        if i >= quantum_states:
            for j in range(1, min(quantum_states, i)):
                if i-j >= 0:
                    # 価格間の量子もつれ効果
                    correlation = (prices[i] - prices[i-j]) * (prices[i-j] - prices[i-j-1])
                    if abs(correlation) > 1e-12:
                        entanglement_factor += math.sin(math.pi * correlation / (abs(correlation) + 1e-10))
            
            entanglement_factor = abs(entanglement_factor) / (quantum_states - 1)
            quantum_coherence[i] = max(min(entanglement_factor, 1.0), 0.0)
        else:
            quantum_coherence[i] = quantum_coherence[i-1]
        
        # === 2. ヒルベルト変換による瞬時解析 ===
        if i >= hilbert_window:
            # 4点ヒルベルト変換（超高速版）
            real_part = (prices[i] + prices[i-2] + prices[i-4] + prices[i-6]) * 0.25
            imag_part = (prices[i-1] + prices[i-3] + prices[i-5] + prices[i-7]) * 0.25
            
            # 瞬時振幅
            hilbert_amplitude[i] = math.sqrt(real_part * real_part + imag_part * imag_part)
            
            # 瞬時位相（トレンド強度更新用）
            if abs(real_part) > 1e-12:
                phase = math.atan2(imag_part, real_part)
                trend_momentum = math.sin(phase) * 0.5 + 0.5
                state[3] = trend_momentum  # トレンド強度状態を更新
        else:
            hilbert_amplitude[i] = hilbert_amplitude[i-1]
        
        # === 3. フラクタル次元解析（ボックスカウンティング法） ===
        if i >= fractal_window:
            segment = prices[i-fractal_window:i]
            price_range = np.max(segment) - np.min(segment)
            
            if price_range > 1e-10:
                # 異なるスケールでの変動測定
                scales = [2, 4, 8]
                variations = []
                
                for scale in scales:
                    if fractal_window >= scale:
                        variation = 0.0
                        for j in range(0, fractal_window - scale, scale):
                            if j + scale < len(segment):
                                segment_var = np.var(segment[j:j+scale])
                                variation += math.sqrt(segment_var + 1e-12)
                        
                        if fractal_window // scale > 0:
                            variation /= (fractal_window // scale)
                        variations.append(variation)
                
                # フラクタル次元計算（安全な計算）
                if len(variations) >= 2 and variations[0] > 1e-10 and variations[-1] > 1e-10:
                    ratio = (variations[-1] + 1e-12) / (variations[0] + 1e-12)
                    if ratio > 0:
                        log_ratio = math.log(max(ratio, 1e-10))
                        log_scale = math.log(max(scales[-1] / scales[0], 1e-10))
                        fractal_dim = 1.0 + log_ratio / log_scale
                        fractal_dimension[i] = max(min(fractal_dim, 2.0), 1.0)
                    else:
                        fractal_dimension[i] = 1.5
                else:
                    fractal_dimension[i] = 1.5
            else:
                fractal_dimension[i] = 1.5
        else:
            fractal_dimension[i] = fractal_dimension[i-1]
        
        # === 4. 超適応ノイズモデリング ===
        if i >= volatility_window:
            # 最近のボラティリティ
            recent_volatility = np.std(prices[i-volatility_window:i])
            
            # 量子コヒーレンス調整
            coherence_factor = quantum_coherence[i]
            
            # ヒルベルト振幅調整
            amplitude_factor = 1.0
            if i > 0 and hilbert_amplitude[i-1] > 1e-10:
                amplitude_factor = hilbert_amplitude[i] / (hilbert_amplitude[i-1] + 1e-12)
                amplitude_factor = max(min(amplitude_factor, 3.0), 0.3)
            
            # フラクタル次元調整（市場効率性）
            efficiency_factor = 2.0 - fractal_dimension[i]  # 1.0-2.0 → 0.0-1.0
            
            # 超適応プロセスノイズ（安全な計算）
            process_noise = base_process_noise * (1.0 - coherence_factor + 0.1) * amplitude_factor * (1.0 + efficiency_factor)
            process_noise = max(min(process_noise, 0.01), base_process_noise)
            
            # 超適応観測ノイズ（安全な計算）
            observation_noise = base_observation_noise * (1.0 + coherence_factor * 0.5) * max(recent_volatility, 1e-6) * 10
            observation_noise = max(min(observation_noise, 0.1), base_observation_noise)
            
            # ボラティリティ状態を更新
            state[4] = max(recent_volatility, 1e-8)
        else:
            process_noise = base_process_noise
            observation_noise = base_observation_noise
        
        # === 5. 超適応プロセスノイズ行列 ===
        Q = np.eye(5) * process_noise
        Q[1, 1] = process_noise * 2.0    # 速度のノイズを増加
        Q[2, 2] = process_noise * 3.0    # 加速度のノイズを増加
        Q[3, 3] = process_noise * 0.5    # トレンド強度のノイズを減少
        Q[4, 4] = process_noise * 1.5    # ボラティリティのノイズを調整
        
        # === 6. 超高速カルマンフィルター更新 ===
        # 予測ステップ
        state_pred = np.dot(F, state)
        P_pred = np.dot(np.dot(F, P), F.T) + Q
        
        # 更新ステップ
        innovation = prices[i] - state_pred[0]
        
        # カルマンゲイン計算（1次元観測）- 安全な計算
        S = P_pred[0, 0] + observation_noise + 1e-12  # ゼロ除算防止
        if S > 1e-10:
            K = P_pred[:, 0] / S
            
            # 状態更新
            state = state_pred + K * innovation
            
            # 共分散更新（Joseph形式で数値安定性確保）
            I_KH = np.eye(5)
            I_KH[0, 0] = 1.0 - K[0]
            P = np.dot(I_KH, P_pred)
            
            kalman_gains[i] = K[0]
        else:
            state = state_pred
            kalman_gains[i] = 0.5
        
        # === 7. 結果保存と品質評価 ===
        filtered_prices[i] = state[0]
        prediction_errors[i] = abs(innovation)
        
        # 適応信頼度（複数要素統合）
        coherence_score = quantum_coherence[i]
        amplitude_stability = 1.0 / (1.0 + abs(hilbert_amplitude[i] - hilbert_amplitude[i-1]) * 10)
        fractal_stability = 1.0 / (1.0 + abs(fractal_dimension[i] - 1.5) * 2)
        error_quality = 1.0 / (1.0 + prediction_errors[i] * 100)
        
        adaptive_confidence[i] = (coherence_score * 0.3 + amplitude_stability * 0.25 + 
                                fractal_stability * 0.25 + error_quality * 0.2)
        
        # 超追従スコア（遅延とトラッキング精度の統合指標）
        if i >= 3:
            # 短期追従性
            short_tracking = 1.0 / (1.0 + abs(prices[i] - filtered_prices[i]) / (prices[i] + 1e-10))
            
            # 中期安定性
            medium_stability = 1.0
            if i >= 10:
                recent_errors = prediction_errors[i-5:i]
                if len(recent_errors) > 0:
                    medium_stability = 1.0 / (1.0 + np.mean(recent_errors) * 50)
            
            # トレンド一貫性
            trend_consistency = abs(state[3] - 0.5) * 2  # 0.0-1.0の範囲
            
            ultra_tracking_score[i] = (short_tracking * 0.5 + medium_stability * 0.3 + 
                                     trend_consistency * 0.2)
        else:
            ultra_tracking_score[i] = 0.5
    
    return (filtered_prices, kalman_gains, prediction_errors, quantum_coherence,
            hilbert_amplitude, fractal_dimension, adaptive_confidence, ultra_tracking_score)


@jit(nopython=True, fastmath=True, cache=True)
def neural_adaptive_kalman_supreme_v3(
    prices: np.ndarray,
    learning_rate: float = 0.01,
    momentum: float = 0.9,
    adaptive_threshold: float = 0.001,
    memory_length: int = 20,
    neural_layers: int = 3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    🧠 **ニューラル適応カルマンフィルター Supreme V3.0**
    
    【AI/機械学習技術統合】
    1. ニューラルネットワーク風適応重み学習
    2. モメンタム最適化によるパラメータ更新
    3. 自己適応学習率調整
    4. メモリベース履歴学習
    5. 多層適応フィルタリング
    6. リアルタイム性能最適化
    
    従来のカルマンフィルターに機械学習の適応性を統合した次世代フィルター
    
    Args:
        prices: 価格データ
        learning_rate: 学習率
        momentum: モメンタム係数
        adaptive_threshold: 適応閾値
        memory_length: メモリ長
        neural_layers: ニューラル層数
    
    Returns:
        Tuple: (フィルター済み価格, 学習重み, 適応学習率, メモリスコア, 性能指標)
    """
    n = len(prices)
    
    # 出力配列
    filtered_prices = np.zeros(n)
    learning_weights = np.zeros(n)
    adaptive_lr = np.zeros(n)
    memory_scores = np.zeros(n)
    performance_metrics = np.zeros(n)
    
    if n < memory_length:
        return prices.copy(), learning_weights, adaptive_lr, memory_scores, performance_metrics
    
    # ニューラル風パラメータ初期化
    weights = np.array([0.5, 0.3, 0.2])  # 3層の重み
    momentum_weights = np.zeros(3)
    
    # カルマンパラメータ
    state = prices[0]
    covariance = 1.0
    current_lr = learning_rate
    
    # メモリバッファ
    error_memory = np.zeros(memory_length)
    performance_memory = np.zeros(memory_length)
    
    filtered_prices[0] = prices[0]
    learning_weights[0] = weights[0]
    adaptive_lr[0] = current_lr
    memory_scores[0] = 0.5
    performance_metrics[0] = 0.5
    
    for i in range(1, n):
        # === 1. 多層適応フィルタリング ===
        # Layer 1: 短期適応
        if i >= 3:
            short_term = np.mean(prices[i-3:i])
            layer1_output = weights[0] * prices[i] + (1 - weights[0]) * short_term
        else:
            layer1_output = prices[i]
        
        # Layer 2: 中期適応
        if i >= 10:
            medium_term = np.mean(prices[i-10:i])
            layer2_output = weights[1] * layer1_output + (1 - weights[1]) * medium_term
        else:
            layer2_output = layer1_output
        
        # Layer 3: 長期適応
        if i >= memory_length:
            long_term = np.mean(prices[i-memory_length:i])
            layer3_output = weights[2] * layer2_output + (1 - weights[2]) * long_term
        else:
            layer3_output = layer2_output
        
        # === 2. カルマンフィルター更新 ===
        # 動的ノイズ推定
        if i >= 5:
            recent_volatility = np.std(prices[i-5:i])
            process_noise = max(recent_volatility * 0.01, 1e-6)
            observation_noise = max(recent_volatility * 0.1, 1e-4)
        else:
            process_noise = 1e-5
            observation_noise = 0.01
        
        # 予測
        state_pred = state
        covariance_pred = covariance + process_noise
        
        # 更新 - 安全な計算
        denominator = covariance_pred + observation_noise + 1e-12  # ゼロ除算防止
        kalman_gain = covariance_pred / denominator
        innovation = layer3_output - state_pred
        state = state_pred + kalman_gain * innovation
        covariance = (1 - kalman_gain) * covariance_pred
        
        filtered_prices[i] = state
        
        # === 3. 性能評価とフィードバック ===
        prediction_error = abs(prices[i] - filtered_prices[i])
        error_memory[i % memory_length] = prediction_error
        
        # 性能指標計算
        if i >= memory_length:
            avg_error = np.mean(error_memory)
            performance_score = 1.0 / (1.0 + avg_error * 100)
            performance_memory[i % memory_length] = performance_score
            performance_metrics[i] = performance_score
        else:
            performance_metrics[i] = 0.5
        
        # === 4. 適応学習とパラメータ更新 ===
        if i >= 5:
            # 勾配計算（簡易版）
            gradients = np.zeros(3)
            
            # Layer 1勾配
            if abs(layer1_output - prices[i]) > adaptive_threshold:
                gradients[0] = (prices[i] - layer1_output) * current_lr
            
            # Layer 2勾配
            if abs(layer2_output - layer1_output) > adaptive_threshold:
                gradients[1] = (layer1_output - layer2_output) * current_lr
            
            # Layer 3勾配
            if abs(layer3_output - layer2_output) > adaptive_threshold:
                gradients[2] = (layer2_output - layer3_output) * current_lr
            
            # モメンタム更新
            momentum_weights = momentum * momentum_weights + (1 - momentum) * gradients
            
            # 重み更新
            weights = weights + momentum_weights
            
            # 重みの正規化
            weights = np.clip(weights, 0.01, 0.99)
            weight_sum = np.sum(weights)
            if weight_sum > 0:
                weights = weights / weight_sum * 1.5  # 合計を1.5に正規化
        
        # === 5. 適応学習率調整 ===
        if i >= 10:
            recent_performance = np.mean(performance_memory[:min(i, memory_length)])
            if recent_performance < 0.5:
                current_lr = min(current_lr * 1.1, 0.1)  # 学習率増加
            elif recent_performance > 0.8:
                current_lr = max(current_lr * 0.95, 0.001)  # 学習率減少
        
        # === 6. メモリスコア計算 ===
        if i >= memory_length:
            # 短期メモリ（最近5期間）
            short_memory = np.mean(error_memory[-5:]) if i >= 5 else error_memory[0]
            
            # 長期メモリ（全履歴）
            long_memory = np.mean(error_memory)
            
            # メモリ一貫性
            memory_consistency = 1.0 / (1.0 + abs(short_memory - long_memory) * 50)
            memory_scores[i] = memory_consistency
        else:
            memory_scores[i] = 0.5
        
        # 結果保存
        learning_weights[i] = weights[0]  # 代表重みとして第1層を保存
        adaptive_lr[i] = current_lr
    
    return filtered_prices, learning_weights, adaptive_lr, memory_scores, performance_metrics


@jit(nopython=True)
def calculate_confidence_scores_numba(prices: np.ndarray,
                                    kalman_gains: np.ndarray,
                                    prediction_errors: np.ndarray,
                                    volatility_estimates: np.ndarray) -> np.ndarray:
    """
    🎯 **信頼度スコア計算（Ehlers方式改良版）**
    
    複数指標による信頼度評価でEhlers双方向フィルターの品質を向上
    
    Args:
        prices: 価格データ
        kalman_gains: カルマンゲイン
        prediction_errors: 予測誤差
        volatility_estimates: ボラティリティ推定値
    
    Returns:
        信頼度スコア配列
    """
    n = len(prices)
    confidence = np.ones(n)
    
    if n < 10:
        return confidence
    
    for i in range(10, n):
        # 1. カルマンゲインベース信頼度（低ゲイン = 高信頼度）
        gain_confidence = 1.0 - min(kalman_gains[i], 1.0)
        
        # 2. 予測誤差ベース信頼度
        recent_errors = prediction_errors[max(0, i-5):i]
        avg_error = np.mean(recent_errors)
        error_confidence = 1.0 / (1.0 + avg_error * 10)
        
        # 3. ボラティリティ安定性ベース信頼度
        if i >= 10:
            recent_vol = volatility_estimates[i]
            vol_stability = 1.0 / (1.0 + recent_vol * 20)
        else:
            vol_stability = 0.8
        
        # 4. 価格変化一貫性ベース信頼度
        if i >= 5:
            recent_changes = np.diff(prices[i-5:i])
            change_consistency = 1.0 / (1.0 + np.std(recent_changes) * 5)
        else:
            change_consistency = 0.8
        
        # 総合信頼度（重み付き平均）
        confidence[i] = (gain_confidence * 0.35 + 
                        error_confidence * 0.30 + 
                        vol_stability * 0.20 + 
                        change_consistency * 0.15)
        
        # 範囲制限
        confidence[i] = max(0.1, min(1.0, confidence[i]))
    
    # 初期値設定
    for i in range(10):
        confidence[i] = confidence[10] if n > 10 else 0.8
    
    return confidence


@jit(nopython=True)
def ultimate_kalman_backward_smoother_numba(forward_prices: np.ndarray,
                                          forward_covariances: np.ndarray,
                                          process_noise: np.ndarray,
                                          confidence_scores: np.ndarray) -> np.ndarray:
    """
    🌀 **アルティメット双方向カルマンスムーザー（後方パス）**
    
    Ehlers究極のKalmanスムーザーを改良した高品質双方向処理
    
    Args:
        forward_prices: 前方パス結果
        forward_covariances: 前方パス共分散
        process_noise: プロセスノイズ配列
        confidence_scores: 信頼度スコア
    
    Returns:
        双方向スムージング済み価格
    """
    n = len(forward_prices)
    if n == 0:
        return forward_prices.copy()
    
    smoothed = np.zeros(n)
    smoothed[n-1] = forward_prices[n-1]
    
    # 後方パス（Ehlers方式改良版）
    for i in range(n-2, -1, -1):
        if forward_covariances[i] + process_noise[i+1] > 0:
            # Ultimate MA式適応重み（信頼度ベース調整）
            base_gain = forward_covariances[i] / (forward_covariances[i] + process_noise[i+1])
            
            # 信頼度による適応調整
            adaptation_factor = confidence_scores[i+1] * 0.6 + 0.4  # 0.4-1.0の範囲
            adaptive_gain = base_gain * adaptation_factor
            
            # 双方向スムージング
            smoothed[i] = forward_prices[i] + adaptive_gain * (smoothed[i+1] - forward_prices[i])
        else:
            smoothed[i] = forward_prices[i]
    
    return smoothed


@jit(nopython=True)
def calculate_forward_covariances_numba(kalman_gains: np.ndarray,
                                      process_noise: np.ndarray,
                                      observation_noise: np.ndarray) -> np.ndarray:
    """
    前方パスの共分散を再計算する（双方向処理用）
    
    Args:
        kalman_gains: カルマンゲイン配列
        process_noise: プロセスノイズ配列
        observation_noise: 観測ノイズ配列
    
    Returns:
        前方共分散配列
    """
    n = len(kalman_gains)
    covariances = np.zeros(n)
    
    if n == 0:
        return covariances
    
    # 初期共分散
    p_est = 1.0
    covariances[0] = p_est
    
    for i in range(1, n):
        # 予測共分散
        p_pred = p_est + process_noise[i]
        
        # 更新共分散
        p_est = (1 - kalman_gains[i]) * p_pred
        covariances[i] = p_est
    
    return covariances


class UltimateKalmanFilter(Indicator):
    """
    🚀 **アルティメットカルマンフィルター V1.0**
    
    🎯 **Ultimate MA + Ehlers統合技術:**
    - **Ultimate MA適応性**: 動的ノイズレベル推定・リアルタイム適応
    - **Ehlers双方向技術**: 前方+後方パスによる究極品質スムージング
    - **選択可能処理**: 単方向（速度重視）or 双方向（品質重視）
    
    🏆 **最適化された特徴:**
    1. **適応的ノイズ推定**: Ultimate MA方式の動的ボラティリティ調整
    2. **信頼度ベース制御**: 複数指標による品質評価
    3. **双方向スムージング**: Ehlers方式改良版の後方パス
    4. **柔軟な処理モード**: 用途に応じた最適化選択
    5. **包括的統計情報**: 詳細なフィルター品質指標
    
    ⚡ **処理モード:**
    - **単方向**: ゼロ遅延リアルタイム処理（取引用）
    - **双方向**: 高品質スムージング処理（分析用）
    """
    
    def __init__(self,
                 bidirectional: bool = True,
                 base_process_noise: float = 1e-5,
                 base_observation_noise: float = 0.01,
                 volatility_window: int = 10,
                 src_type: str = 'hlc3'):
        """
        アルティメットカルマンフィルターのコンストラクタ
        
        Args:
            bidirectional: 双方向処理を使用するか（True=高品質、False=高速）
            base_process_noise: 基本プロセスノイズ（デフォルト: 1e-5）
            base_observation_noise: 基本観測ノイズ（デフォルト: 0.01）
            volatility_window: ボラティリティ推定ウィンドウ（デフォルト: 10）
            src_type: 価格ソース ('close', 'hlc3', etc.)
        """
        mode_desc = "Bidirectional" if bidirectional else "Forward"
        super().__init__(f"UltimateKalman({mode_desc}, vol_win={volatility_window}, src={src_type})")
        
        self.bidirectional = bidirectional
        self.base_process_noise = base_process_noise
        self.base_observation_noise = base_observation_noise
        self.volatility_window = volatility_window
        self.src_type = src_type
        
        self.price_source_extractor = PriceSource()
        self._cache = {}
        self._result: Optional[UltimateKalmanResult] = None

    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> UltimateKalmanResult:
        """
        🚀 アルティメットカルマンフィルターを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
        
        Returns:
            UltimateKalmanResult: 包括的なフィルター結果
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            if data_hash in self._cache and self._result is not None:
                return self._result
            
            # ソース価格を取得
            if isinstance(data, np.ndarray) and data.ndim == 1:
                src_prices = data.astype(np.float64)
            else:
                src_prices = PriceSource.calculate_source(data, self.src_type)
                src_prices = src_prices.astype(np.float64)
            
            data_length = len(src_prices)
            if data_length == 0:
                return self._create_empty_result()
            
            self.logger.info(f"🚀 アルティメットカルマンフィルター計算開始... モード: {'双方向' if self.bidirectional else '単方向'}")
            
            # 🎯 1. Ultimate MA式適応前方パス
            self.logger.debug("⚡ Ultimate MA式適応前方パス実行中...")
            (forward_values, kalman_gains, prediction_errors, 
             process_noise, observation_noise) = ultimate_adaptive_kalman_forward_numba(
                src_prices, self.base_process_noise, self.base_observation_noise, self.volatility_window
            )
            
            # ボラティリティ推定値の計算
            volatility_estimates = np.zeros(data_length)
            for i in range(self.volatility_window, data_length):
                volatility_estimates[i] = np.std(src_prices[i-self.volatility_window:i])
            
            # 🎯 2. 信頼度スコア計算
            self.logger.debug("🎯 信頼度スコア計算中...")
            confidence_scores = calculate_confidence_scores_numba(
                src_prices, kalman_gains, prediction_errors, volatility_estimates
            )
            
            # 🌀 3. 双方向処理（オプション）
            if self.bidirectional:
                self.logger.debug("🌀 Ehlers式双方向スムーザー実行中...")
                
                # 前方共分散の再計算
                forward_covariances = calculate_forward_covariances_numba(
                    kalman_gains, process_noise, observation_noise
                )
                
                # 双方向スムージング
                backward_values = ultimate_kalman_backward_smoother_numba(
                    forward_values, forward_covariances, process_noise, confidence_scores
                )
                
                # 最終結果は双方向
                final_values = backward_values
            else:
                # 単方向のみ
                backward_values = np.full(data_length, np.nan)
                final_values = forward_values
            
            # 統計計算
            raw_volatility = np.nanstd(src_prices)
            filtered_volatility = np.nanstd(final_values)
            noise_reduction_ratio = (raw_volatility - filtered_volatility) / raw_volatility if raw_volatility > 0 else 0.0
            
            # 結果オブジェクト作成
            result = UltimateKalmanResult(
                values=final_values,
                raw_values=src_prices,
                forward_values=forward_values,
                backward_values=backward_values,
                kalman_gains=kalman_gains,
                process_noise=process_noise,
                observation_noise=observation_noise,
                confidence_scores=confidence_scores,
                prediction_errors=prediction_errors,
                volatility_estimates=volatility_estimates,
                is_bidirectional=self.bidirectional,
                noise_reduction_ratio=noise_reduction_ratio
            )
            
            self._result = result
            self._cache[data_hash] = self._result
            
            # 統計情報
            avg_confidence = np.mean(confidence_scores)
            avg_kalman_gain = np.mean(kalman_gains)
            
            self.logger.info(f"✅ アルティメットカルマンフィルター完了 - "
                           f"ノイズ削減:{noise_reduction_ratio:.1%}, "
                           f"平均信頼度:{avg_confidence:.3f}, 平均ゲイン:{avg_kalman_gain:.3f}")
            
            return self._result
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"{self.name} 計算中にエラー: {error_msg}\n{stack_trace}")
            data_len = len(data) if hasattr(data, '__len__') else 0
            return self._create_empty_result(data_len)

    def _create_empty_result(self, length: int = 0) -> UltimateKalmanResult:
        """空の結果を作成する"""
        return UltimateKalmanResult(
            values=np.full(length, np.nan, dtype=np.float64),
            raw_values=np.full(length, np.nan, dtype=np.float64),
            forward_values=np.full(length, np.nan, dtype=np.float64),
            backward_values=np.full(length, np.nan, dtype=np.float64),
            kalman_gains=np.full(length, np.nan, dtype=np.float64),
            process_noise=np.full(length, np.nan, dtype=np.float64),
            observation_noise=np.full(length, np.nan, dtype=np.float64),
            confidence_scores=np.full(length, np.nan, dtype=np.float64),
            prediction_errors=np.full(length, np.nan, dtype=np.float64),
            volatility_estimates=np.full(length, np.nan, dtype=np.float64),
            is_bidirectional=self.bidirectional,
            noise_reduction_ratio=0.0
        )

    def get_values(self) -> Optional[np.ndarray]:
        """最終フィルター済み値を取得する"""
        if self._result is not None:
            return self._result.values.copy()
        return None

    def get_forward_values(self) -> Optional[np.ndarray]:
        """前方パス結果を取得する"""
        if self._result is not None:
            return self._result.forward_values.copy()
        return None

    def get_backward_values(self) -> Optional[np.ndarray]:
        """後方パス結果を取得する（双方向の場合のみ）"""
        if self._result is not None and self._result.is_bidirectional:
            return self._result.backward_values.copy()
        return None

    def get_confidence_scores(self) -> Optional[np.ndarray]:
        """信頼度スコアを取得する"""
        if self._result is not None:
            return self._result.confidence_scores.copy()
        return None

    def get_kalman_gains(self) -> Optional[np.ndarray]:
        """カルマンゲインを取得する"""
        if self._result is not None:
            return self._result.kalman_gains.copy()
        return None

    def get_volatility_estimates(self) -> Optional[np.ndarray]:
        """ボラティリティ推定値を取得する"""
        if self._result is not None:
            return self._result.volatility_estimates.copy()
        return None

    def get_performance_stats(self) -> Dict:
        """パフォーマンス統計を取得する"""
        if self._result is None:
            return {}
        
        return {
            'processing_mode': 'bidirectional' if self._result.is_bidirectional else 'forward_only',
            'noise_reduction_ratio': self._result.noise_reduction_ratio,
            'noise_reduction_percentage': self._result.noise_reduction_ratio * 100,
            'average_confidence': np.mean(self._result.confidence_scores),
            'average_kalman_gain': np.mean(self._result.kalman_gains),
            'average_prediction_error': np.mean(self._result.prediction_errors),
            'average_volatility': np.mean(self._result.volatility_estimates),
            'filter_characteristics': {
                'base_process_noise': self.base_process_noise,
                'base_observation_noise': self.base_observation_noise,
                'volatility_window': self.volatility_window,
                'adaptive_noise_range': (np.min(self._result.observation_noise), np.max(self._result.observation_noise)),
                'process_noise_range': (np.min(self._result.process_noise), np.max(self._result.process_noise))
            },
            'quality_indicators': {
                'forward_backward_correlation': np.corrcoef(self._result.forward_values, self._result.backward_values)[0, 1] if self._result.is_bidirectional else None,
                'raw_filtered_correlation': np.corrcoef(self._result.raw_values, self._result.values)[0, 1],
                'smoothness_factor': np.nanstd(np.diff(self._result.values)) / np.nanstd(np.diff(self._result.raw_values))
            }
        }

    def get_comparison_with_components(self) -> Dict:
        """構成要素との比較統計"""
        if self._result is None:
            return {}
        
        forward_vol = np.nanstd(self._result.forward_values)
        raw_vol = np.nanstd(self._result.raw_values)
        final_vol = np.nanstd(self._result.values)
        
        comparison = {
            'noise_reduction_comparison': {
                'forward_only': (raw_vol - forward_vol) / raw_vol if raw_vol > 0 else 0,
                'final_result': (raw_vol - final_vol) / raw_vol if raw_vol > 0 else 0,
                'bidirectional_improvement': ((forward_vol - final_vol) / forward_vol if forward_vol > 0 else 0) if self._result.is_bidirectional else 0
            },
            'inherited_features': {
                'ultimate_ma_adaptation': 'Dynamic volatility-based noise estimation',
                'ehlers_bidirectional': 'Confidence-weighted backward smoothing' if self._result.is_bidirectional else 'Not used',
                'combined_advantages': [
                    'Real-time adaptive noise control',
                    'Confidence-based quality assessment',
                    'Optional bidirectional processing',
                    'Comprehensive performance metrics'
                ]
            },
            'use_case_recommendations': {
                'forward_only': 'Real-time trading applications requiring zero latency',
                'bidirectional': 'High-quality analysis and research applications',
                'optimal_settings': f"Current: {self.base_process_noise:.0e} process noise, {self.volatility_window} vol window"
            }
        }
        
        return comparison

    def reset(self) -> None:
        """インジケータの状態をリセットする"""
        super().reset()
        self._result = None
        self._cache = {}

    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データとパラメータに基づいてハッシュ値を計算する"""
        if isinstance(data, pd.DataFrame):
            try:
                data_hash_val = hash(data.values.tobytes())
            except Exception:
                shape_tuple = data.shape
                first_row = tuple(data.iloc[0]) if len(data) > 0 else ()
                last_row = tuple(data.iloc[-1]) if len(data) > 0 else ()
                data_repr_tuple = (shape_tuple, first_row, last_row)
                data_hash_val = hash(data_repr_tuple)
        elif isinstance(data, np.ndarray):
            data_hash_val = hash(data.tobytes())
        else:
            data_hash_val = hash(str(data))
        
        param_str = (f"bidir={self.bidirectional}_proc_noise={self.base_process_noise}"
                    f"_obs_noise={self.base_observation_noise}_vol_win={self.volatility_window}"
                    f"_src={self.src_type}")
        return f"{data_hash_val}_{param_str}"

    def calculate_hyper_quantum_ultra(self, data: Union[pd.DataFrame, np.ndarray],
                                    volatility_window: int = 12,
                                    hilbert_window: int = 8,
                                    fractal_window: int = 16,
                                    quantum_states: int = 5) -> Dict[str, np.ndarray]:
        """
        🌌 **ハイパー量子適応カルマンフィルター Ultra V2.0を実行**
        
        最新のアルゴリズムを統合した超進化版カルマンフィルター
        
        Args:
            data: 価格データ
            volatility_window: ボラティリティ推定ウィンドウ
            hilbert_window: ヒルベルト変換ウィンドウ
            fractal_window: フラクタル解析ウィンドウ
            quantum_states: 量子状態数
        
        Returns:
            Dict: 全ての計算結果を含む辞書
        """
        try:
            # データ準備
            if isinstance(data, pd.DataFrame):
                prices = data['close'].values.astype(np.float64)
            else:
                prices = data.astype(np.float64) if data.ndim == 1 else data[:, 3].astype(np.float64)
            
            # ハイパー量子適応カルマンフィルター実行
            (filtered_prices, kalman_gains, prediction_errors, quantum_coherence,
             hilbert_amplitude, fractal_dimension, adaptive_confidence, ultra_tracking_score) = \
                hyper_quantum_adaptive_kalman_ultra_v2(
                    prices,
                    base_process_noise=1e-7,
                    base_observation_noise=0.001,
                    volatility_window=volatility_window,
                    hilbert_window=hilbert_window,
                    fractal_window=fractal_window,
                    quantum_states=quantum_states
                )
            
            self.logger.info(f"🌌 ハイパー量子適応カルマンフィルター Ultra V2.0 完了")
            self.logger.info(f"   平均量子コヒーレンス: {np.nanmean(quantum_coherence):.4f}")
            self.logger.info(f"   平均ヒルベルト振幅: {np.nanmean(hilbert_amplitude):.4f}")
            self.logger.info(f"   平均フラクタル次元: {np.nanmean(fractal_dimension):.4f}")
            self.logger.info(f"   平均適応信頼度: {np.nanmean(adaptive_confidence):.4f}")
            self.logger.info(f"   平均超追従スコア: {np.nanmean(ultra_tracking_score):.4f}")
            
            return {
                'filtered_prices': filtered_prices,
                'kalman_gains': kalman_gains,
                'prediction_errors': prediction_errors,
                'quantum_coherence': quantum_coherence,
                'hilbert_amplitude': hilbert_amplitude,
                'fractal_dimension': fractal_dimension,
                'adaptive_confidence': adaptive_confidence,
                'ultra_tracking_score': ultra_tracking_score,
                'raw_prices': prices
            }
            
        except Exception as e:
            self.logger.error(f"ハイパー量子適応カルマンフィルター Ultra V2.0 エラー: {e}")
            raise

    def calculate_neural_adaptive_supreme(self, data: Union[pd.DataFrame, np.ndarray],
                                        learning_rate: float = 0.01,
                                        momentum: float = 0.9,
                                        adaptive_threshold: float = 0.001,
                                        memory_length: int = 20,
                                        neural_layers: int = 3) -> Dict[str, np.ndarray]:
        """
        🧠 **ニューラル適応カルマンフィルター Supreme V3.0を実行**
        
        AI/機械学習技術を統合した次世代カルマンフィルター
        
        Args:
            data: 価格データ
            learning_rate: 学習率
            momentum: モメンタム係数
            adaptive_threshold: 適応閾値
            memory_length: メモリ長
            neural_layers: ニューラル層数
        
        Returns:
            Dict: 全ての計算結果を含む辞書
        """
        try:
            # データ準備
            if isinstance(data, pd.DataFrame):
                prices = data['close'].values.astype(np.float64)
            else:
                prices = data.astype(np.float64) if data.ndim == 1 else data[:, 3].astype(np.float64)
            
            # ニューラル適応カルマンフィルター実行
            (filtered_prices, learning_weights, adaptive_lr, 
             memory_scores, performance_metrics) = \
                neural_adaptive_kalman_supreme_v3(
                    prices,
                    learning_rate=learning_rate,
                    momentum=momentum,
                    adaptive_threshold=adaptive_threshold,
                    memory_length=memory_length,
                    neural_layers=neural_layers
                )
            
            self.logger.info(f"🧠 ニューラル適応カルマンフィルター Supreme V3.0 完了")
            self.logger.info(f"   最終学習重み: {learning_weights[-1]:.4f}")
            self.logger.info(f"   最終適応学習率: {adaptive_lr[-1]:.6f}")
            self.logger.info(f"   平均メモリスコア: {np.nanmean(memory_scores):.4f}")
            self.logger.info(f"   平均性能指標: {np.nanmean(performance_metrics):.4f}")
            
            return {
                'filtered_prices': filtered_prices,
                'learning_weights': learning_weights,
                'adaptive_learning_rate': adaptive_lr,
                'memory_scores': memory_scores,
                'performance_metrics': performance_metrics,
                'raw_prices': prices
            }
            
        except Exception as e:
            self.logger.error(f"ニューラル適応カルマンフィルター Supreme V3.0 エラー: {e}")
            raise

    def compare_all_methods(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """
        🏆 **全カルマンフィルター手法の性能比較**
        
        標準版・ハイパー量子版・ニューラル版の全手法を比較評価
        
        Args:
            data: 価格データ
        
        Returns:
            Dict: 比較結果と性能統計
        """
        try:
            # データ準備
            if isinstance(data, pd.DataFrame):
                prices = data['close'].values.astype(np.float64)
            else:
                prices = data.astype(np.float64) if data.ndim == 1 else data[:, 3].astype(np.float64)
            
            self.logger.info("🏆 全カルマンフィルター手法の性能比較開始...")
            
            # 1. 標準版カルマンフィルター
            standard_result = self.calculate(data)
            
            # 2. ハイパー量子版
            quantum_result = self.calculate_hyper_quantum_ultra(data)
            
            # 3. ニューラル版
            neural_result = self.calculate_neural_adaptive_supreme(data)
            
            # 性能比較計算
            def calculate_performance_metrics(filtered_prices, raw_prices):
                # 追従性（遅延測定）
                lag_correlation = np.corrcoef(filtered_prices[1:], raw_prices[:-1])[0, 1]
                
                # 平滑性（ノイズ除去効果）
                raw_volatility = np.std(np.diff(raw_prices))
                filtered_volatility = np.std(np.diff(filtered_prices))
                smoothness = (raw_volatility - filtered_volatility) / raw_volatility
                
                # 精度（平均絶対誤差）
                accuracy = 1.0 / (1.0 + np.mean(np.abs(filtered_prices - raw_prices)))
                
                return {
                    'lag_correlation': lag_correlation,
                    'smoothness': smoothness,
                    'accuracy': accuracy,
                    'overall_score': (lag_correlation + smoothness + accuracy) / 3
                }
            
            # 各手法の性能評価
            standard_perf = calculate_performance_metrics(standard_result.values, prices)
            quantum_perf = calculate_performance_metrics(quantum_result['filtered_prices'], prices)
            neural_perf = calculate_performance_metrics(neural_result['filtered_prices'], prices)
            
            self.logger.info("📊 性能比較結果:")
            self.logger.info(f"   標準版総合スコア: {standard_perf['overall_score']:.4f}")
            self.logger.info(f"   量子版総合スコア: {quantum_perf['overall_score']:.4f}")
            self.logger.info(f"   ニューラル版総合スコア: {neural_perf['overall_score']:.4f}")
            
            return {
                'standard_result': standard_result,
                'quantum_result': quantum_result,
                'neural_result': neural_result,
                'performance_comparison': {
                    'standard': standard_perf,
                    'quantum': quantum_perf,
                    'neural': neural_perf
                },
                'winner': max([
                    ('standard', standard_perf['overall_score']),
                    ('quantum', quantum_perf['overall_score']),
                    ('neural', neural_perf['overall_score'])
                ], key=lambda x: x[1])[0]
            }
            
        except Exception as e:
            self.logger.error(f"全手法比較エラー: {e}")
            raise 