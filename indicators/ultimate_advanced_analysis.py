#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultimate Advanced Analysis Module
超高度な時系列分析・機械学習アルゴリズムモジュール

実装機能：
- State Space Models（状態空間モデル）
- GARCH Volatility Models（GARCHボラティリティモデル）
- Regime Switching Models（レジーム切り替えモデル）
- Online Machine Learning（オンライン機械学習）
- Spectral Analysis（スペクトル解析）
- Non-linear Dynamics（非線形力学系）
"""

import numpy as np
from numba import jit, njit, prange
import math
from typing import Tuple, Optional


@njit(fastmath=True, cache=True)
def unscented_kalman_filter(
    prices: np.ndarray, 
    volatility: np.ndarray,
    alpha: float = 0.001,
    beta: float = 2.0,
    kappa: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Unscented Kalman Filter（無香料カルマンフィルター）
    非線形システムに対応した高度なカルマンフィルター
    
    Returns:
        (filtered_prices, trend_estimate, uncertainty)
    """
    n = len(prices)
    if n < 10:
        return prices.copy(), prices.copy(), np.ones(n)
    
    # 状態の次元（価格、速度、加速度）
    L = 3
    
    # UKFパラメータ
    lambda_param = alpha * alpha * (L + kappa) - L
    
    # シグマポイントの重み
    Wm = np.zeros(2 * L + 1)  # 平均用の重み
    Wc = np.zeros(2 * L + 1)  # 共分散用の重み
    
    Wm[0] = lambda_param / (L + lambda_param)
    Wc[0] = lambda_param / (L + lambda_param) + (1 - alpha * alpha + beta)
    
    for i in range(1, 2 * L + 1):
        Wm[i] = 0.5 / (L + lambda_param)
        Wc[i] = 0.5 / (L + lambda_param)
    
    # 初期状態
    x = np.array([prices[0], 0.0, 0.0])  # [価格, 速度, 加速度]
    P = np.eye(L) * 1.0  # 初期共分散
    
    # プロセスノイズ
    Q = np.array([[0.01, 0.0, 0.0],
                  [0.0, 0.01, 0.0],
                  [0.0, 0.0, 0.01]])
    
    filtered_prices = np.full(n, np.nan)
    trend_estimate = np.full(n, np.nan)
    uncertainty = np.full(n, np.nan)
    
    for t in range(n):
        if t == 0:
            filtered_prices[t] = prices[t]
            trend_estimate[t] = 0.0
            uncertainty[t] = 1.0
            continue
        
        # シグマポイントの生成
        sqrt_LP = np.sqrt(L + lambda_param)
        sigma_points = np.zeros((2 * L + 1, L))
        
        # 平方根行列の計算（簡易版）
        sqrt_P = np.zeros((L, L))
        for i in range(L):
            for j in range(L):
                if i == j:
                    sqrt_P[i, j] = math.sqrt(max(P[i, j], 0.001))
        
        sigma_points[0] = x
        for i in range(L):
            sigma_points[i + 1] = x + sqrt_LP * sqrt_P[i]
            sigma_points[i + 1 + L] = x - sqrt_LP * sqrt_P[i]
        
        # 予測ステップ
        # 状態遷移関数 f(x) = [x[0] + x[1] + 0.5*x[2], x[1] + x[2], x[2]]
        predicted_sigma = np.zeros((2 * L + 1, L))
        for i in range(2 * L + 1):
            sp = sigma_points[i]
            predicted_sigma[i, 0] = sp[0] + sp[1] + 0.5 * sp[2]  # 価格
            predicted_sigma[i, 1] = sp[1] + sp[2]  # 速度
            predicted_sigma[i, 2] = sp[2]  # 加速度
        
        # 予測平均
        x_pred = np.zeros(L)
        for i in range(2 * L + 1):
            x_pred += Wm[i] * predicted_sigma[i]
        
        # 予測共分散
        P_pred = np.zeros((L, L))
        for i in range(2 * L + 1):
            diff = predicted_sigma[i] - x_pred
            P_pred += Wc[i] * np.outer(diff, diff)
        P_pred += Q
        
        # 観測更新
        # 観測関数 h(x) = x[0] (価格のみを観測)
        predicted_obs = np.zeros(2 * L + 1)
        for i in range(2 * L + 1):
            predicted_obs[i] = predicted_sigma[i, 0]
        
        # 予測観測値
        y_pred = np.sum(Wm * predicted_obs)
        
        # 観測ノイズ
        R = max(volatility[t] ** 2, 0.001)
        
        # 観測共分散
        Pyy = 0.0
        for i in range(2 * L + 1):
            diff_obs = predicted_obs[i] - y_pred
            Pyy += Wc[i] * diff_obs * diff_obs
        Pyy += R
        
        # 交差共分散
        Pxy = np.zeros(L)
        for i in range(2 * L + 1):
            diff_state = predicted_sigma[i] - x_pred
            diff_obs = predicted_obs[i] - y_pred
            Pxy += Wc[i] * diff_state * diff_obs
        
        # カルマンゲイン
        if Pyy > 0:
            K = Pxy / Pyy
        else:
            K = np.zeros(L)
        
        # 状態更新
        innovation = prices[t] - y_pred
        x = x_pred + K * innovation
        P = P_pred - np.outer(K, K) * Pyy
        
        filtered_prices[t] = x[0]
        trend_estimate[t] = x[1]  # 速度（トレンド）
        uncertainty[t] = math.sqrt(max(P[0, 0], 0))
    
    return filtered_prices, trend_estimate, uncertainty


@njit(fastmath=True, cache=True)
def garch_volatility_model(
    returns: np.ndarray,
    alpha: float = 0.1,
    beta: float = 0.85,
    omega: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
    """
    GARCH(1,1)ボラティリティモデル
    
    σ²(t) = ω + α * ε²(t-1) + β * σ²(t-1)
    
    Returns:
        (conditional_variance, volatility)
    """
    n = len(returns)
    if n < 10:
        return np.full(n, 1.0), np.full(n, 1.0)
    
    # 初期値
    unconditional_var = np.var(returns)
    
    conditional_variance = np.full(n, unconditional_var)
    volatility = np.full(n, math.sqrt(unconditional_var))
    
    for t in range(1, n):
        # GARCH(1,1)の更新式
        lagged_return_sq = returns[t-1] ** 2
        lagged_variance = conditional_variance[t-1]
        
        conditional_variance[t] = omega + alpha * lagged_return_sq + beta * lagged_variance
        conditional_variance[t] = max(conditional_variance[t], 1e-6)  # 下限設定
        
        volatility[t] = math.sqrt(conditional_variance[t])
    
    return conditional_variance, volatility


@njit(fastmath=True, cache=True)
def regime_switching_detection(
    prices: np.ndarray,
    window: int = 50,
    n_regimes: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    レジーム切り替え検出（マルコフ切り替えモデルの簡易版）
    
    Returns:
        (regime_probabilities, most_likely_regime)
    """
    n = len(prices)
    if n < window * 2:
        return np.zeros((n, n_regimes)), np.zeros(n)
    
    regime_probs = np.zeros((n, n_regimes))
    most_likely = np.zeros(n)
    
    # レジームの特徴を定義
    # 0: 低ボラティリティ（レンジ）
    # 1: 上昇トレンド
    # 2: 下降トレンド
    
    for i in range(window, n):
        price_window = prices[i-window:i]
        returns = np.diff(price_window)
        
        if len(returns) == 0:
            continue
        
        # 統計量の計算
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # 正規化
        if std_return > 0:
            normalized_return = mean_return / std_return
        else:
            normalized_return = 0
        
        # レジーム確率の計算（単純化したベイズ推定）
        # レンジレジーム（低ボラティリティ、低平均リターン）
        range_score = math.exp(-0.5 * (normalized_return ** 2 + (std_return - 0.01) ** 2))
        
        # 上昇トレンドレジーム（正のリターン、中程度のボラティリティ）
        uptrend_score = math.exp(-0.5 * ((normalized_return - 1.0) ** 2 + (std_return - 0.02) ** 2)) if normalized_return > 0 else 0
        
        # 下降トレンドレジーム（負のリターン、中程度のボラティリティ）
        downtrend_score = math.exp(-0.5 * ((normalized_return + 1.0) ** 2 + (std_return - 0.02) ** 2)) if normalized_return < 0 else 0
        
        # 正規化
        total_score = range_score + uptrend_score + downtrend_score
        if total_score > 0:
            regime_probs[i, 0] = range_score / total_score
            regime_probs[i, 1] = uptrend_score / total_score
            regime_probs[i, 2] = downtrend_score / total_score
        else:
            regime_probs[i, :] = 1.0 / n_regimes
        
        # 最も可能性の高いレジーム
        max_prob = regime_probs[i, 0]
        max_idx = 0
        for j in range(1, n_regimes):
            if regime_probs[i, j] > max_prob:
                max_prob = regime_probs[i, j]
                max_idx = j
        most_likely[i] = max_idx
    
    return regime_probs, most_likely


@njit(fastmath=True, cache=True)
def online_passive_aggressive_learning(
    features: np.ndarray,  # (n_samples, n_features)
    targets: np.ndarray,   # (n_samples,)
    C: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    オンライン Passive-Aggressive 学習アルゴリズム
    
    Returns:
        (weights, predictions)
    """
    n_samples, n_features = features.shape
    
    # 重みベクトルの初期化
    weights = np.zeros(n_features)
    predictions = np.zeros(n_samples)
    
    for t in range(n_samples):
        # 予測
        prediction = np.dot(weights, features[t])
        predictions[t] = prediction
        
        # 損失の計算
        loss = max(0, 1 - targets[t] * prediction)
        
        if loss > 0:
            # PA-I更新
            norm_sq = np.dot(features[t], features[t])
            if norm_sq > 0:
                tau = min(C, loss / norm_sq)
                # 重みの更新
                weights += tau * targets[t] * features[t]
    
    return weights, predictions


@njit(fastmath=True, cache=True)
def spectral_analysis(
    prices: np.ndarray,
    window: int = 64
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    スペクトル解析（簡易版FFT）
    
    Returns:
        (dominant_frequency, spectral_power, trend_component)
    """
    n = len(prices)
    if n < window * 2:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    dominant_freq = np.full(n, np.nan)
    spectral_power = np.full(n, np.nan)
    trend_component = np.full(n, np.nan)
    
    for i in range(window, n):
        price_segment = prices[i-window:i]
        
        # 線形トレンドの除去
        x_vals = np.arange(window)
        n_points = len(x_vals)
        sum_x = np.sum(x_vals)
        sum_y = np.sum(price_segment)
        sum_xy = np.sum(x_vals * price_segment)
        sum_x2 = np.sum(x_vals * x_vals)
        
        # 線形回帰の傾き
        denom = n_points * sum_x2 - sum_x * sum_x
        if denom != 0:
            slope = (n_points * sum_xy - sum_x * sum_y) / denom
            intercept = (sum_y - slope * sum_x) / n_points
            
            # トレンド除去
            detrended = price_segment - (slope * x_vals + intercept)
            trend_component[i] = slope
        else:
            detrended = price_segment - np.mean(price_segment)
            trend_component[i] = 0
        
        # 簡易DFT（離散フーリエ変換）
        # 主要周波数成分のみを計算
        max_power = 0.0
        dominant_f = 0.0
        
        for k in range(1, min(window // 2, 16)):  # 低周波数成分のみ
            # DFTの計算
            real_part = 0.0
            imag_part = 0.0
            
            for j in range(window):
                angle = -2.0 * math.pi * k * j / window
                real_part += detrended[j] * math.cos(angle)
                imag_part += detrended[j] * math.sin(angle)
            
            # パワーの計算
            power = real_part * real_part + imag_part * imag_part
            
            if power > max_power:
                max_power = power
                dominant_f = k / window  # 正規化周波数
        
        dominant_freq[i] = dominant_f
        spectral_power[i] = max_power / (window * window)  # 正規化
    
    return dominant_freq, spectral_power, trend_component


@njit(fastmath=True, cache=True)
def nonlinear_dynamics_analysis(
    prices: np.ndarray,
    embedding_dim: int = 3,
    delay: int = 1,
    window: int = 50
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    非線形力学系解析
    
    Returns:
        (correlation_dimension, largest_lyapunov, predictability)
    """
    n = len(prices)
    if n < window * 2:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    correlation_dim = np.full(n, np.nan)
    lyapunov_exp = np.full(n, np.nan)
    predictability = np.full(n, np.nan)
    
    for i in range(window, n):
        price_segment = prices[i-window:i]
        
        # 時間遅れ埋め込み
        max_idx = window - (embedding_dim - 1) * delay
        if max_idx <= 0:
            continue
        
        # 埋め込みベクトルの構築
        embedded_points = np.zeros((max_idx, embedding_dim))
        for j in range(max_idx):
            for k in range(embedding_dim):
                embedded_points[j, k] = price_segment[j + k * delay]
        
        # 相関次元の推定（簡易版）
        if max_idx > 5:
            # 近傍点のカウント
            radius = np.std(price_segment) * 0.1
            correlation_sum = 0.0
            total_pairs = 0
            
            for j in range(max_idx):
                for k in range(j + 1, max_idx):
                    # ユークリッド距離
                    dist = 0.0
                    for dim in range(embedding_dim):
                        diff = embedded_points[j, dim] - embedded_points[k, dim]
                        dist += diff * diff
                    dist = math.sqrt(dist)
                    
                    if dist < radius:
                        correlation_sum += 1.0
                    total_pairs += 1
            
            if total_pairs > 0:
                correlation_integral = correlation_sum / total_pairs
                if correlation_integral > 0:
                    correlation_dim[i] = math.log(correlation_integral) / math.log(radius)
                    correlation_dim[i] = min(max(correlation_dim[i], 0), embedding_dim)
        
        # 最大リアプノフ指数の推定
        if max_idx > 10:
            divergence_sum = 0.0
            divergence_count = 0
            
            for j in range(5, max_idx - 5):
                # 最近傍点を探す
                min_dist = float('inf')
                nearest_idx = -1
                
                for k in range(max_idx):
                    if abs(k - j) < 3:  # 時間的に近い点は除外
                        continue
                    
                    dist = 0.0
                    for dim in range(embedding_dim):
                        diff = embedded_points[j, dim] - embedded_points[k, dim]
                        dist += diff * diff
                    dist = math.sqrt(dist)
                    
                    if dist < min_dist:
                        min_dist = dist
                        nearest_idx = k
                
                # 発散の計算
                if nearest_idx >= 0 and j + 3 < max_idx and nearest_idx + 3 < max_idx:
                    future_dist = 0.0
                    for dim in range(embedding_dim):
                        diff = embedded_points[j + 3, dim] - embedded_points[nearest_idx + 3, dim]
                        future_dist += diff * diff
                    future_dist = math.sqrt(future_dist)
                    
                    if min_dist > 0 and future_dist > 0:
                        divergence = math.log(future_dist / min_dist) / 3.0
                        divergence_sum += divergence
                        divergence_count += 1
            
            if divergence_count > 0:
                lyapunov_exp[i] = divergence_sum / divergence_count
        
        # 予測可能性（カオス性の逆）
        if not np.isnan(correlation_dim[i]) and not np.isnan(lyapunov_exp[i]):
            # 低次元・負のリアプノフ指数 → 高い予測可能性
            dim_factor = max(0, 1 - correlation_dim[i] / embedding_dim)
            lyap_factor = max(0, 1 - max(lyapunov_exp[i], 0))
            predictability[i] = (dim_factor + lyap_factor) / 2
        else:
            predictability[i] = 0.5
    
    return correlation_dim, lyapunov_exp, predictability


@njit(fastmath=True, cache=True)
def adaptive_feature_selection(
    features: np.ndarray,  # (n_samples, n_features)
    targets: np.ndarray,   # (n_samples,)
    window: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    適応的特徴選択
    
    Returns:
        (feature_importance, selected_features)
    """
    n_samples, n_features = features.shape
    
    feature_importance = np.zeros((n_samples, n_features))
    selected_features = np.zeros((n_samples, n_features))
    
    for i in range(window, n_samples):
        # ウィンドウ内のデータ
        X_window = features[i-window:i]
        y_window = targets[i-window:i]
        
        # 各特徴量と目的変数の相関を計算
        for j in range(n_features):
            feature_col = X_window[:, j]
            
            # 相関係数の計算
            if np.std(feature_col) > 0 and np.std(y_window) > 0:
                # ピアソン相関係数
                mean_x = np.mean(feature_col)
                mean_y = np.mean(y_window)
                
                numerator = np.sum((feature_col - mean_x) * (y_window - mean_y))
                denominator = math.sqrt(np.sum((feature_col - mean_x) ** 2) * np.sum((y_window - mean_y) ** 2))
                
                if denominator > 0:
                    correlation = abs(numerator / denominator)
                    feature_importance[i, j] = correlation
                else:
                    feature_importance[i, j] = 0
            else:
                feature_importance[i, j] = 0
        
        # 上位特徴量を選択（上位50%）
        importance_threshold = np.median(feature_importance[i])
        for j in range(n_features):
            selected_features[i, j] = 1.0 if feature_importance[i, j] >= importance_threshold else 0.0
    
    return feature_importance, selected_features


@njit(fastmath=True, cache=True)
def multiscale_entropy(
    prices: np.ndarray,
    max_scale: int = 10,
    pattern_length: int = 2,
    tolerance_ratio: float = 0.15
) -> np.ndarray:
    """
    マルチスケールエントロピー
    複数の時間スケールでの複雑性を測定
    
    Returns:
        マルチスケールエントロピー値の配列
    """
    n = len(prices)
    if n < 50:
        return np.full(n, np.nan)
    
    mse_values = np.full(n, np.nan)
    
    for i in range(50, n):
        price_segment = prices[max(0, i-50):i]
        
        # 各スケールでのサンプルエントロピーを計算
        entropy_sum = 0.0
        valid_scales = 0
        
        for scale in range(1, min(max_scale + 1, len(price_segment) // 4)):
            # コース・グレイニング
            if scale == 1:
                coarse_grained = price_segment
            else:
                coarse_length = len(price_segment) // scale
                coarse_grained = np.zeros(coarse_length)
                for j in range(coarse_length):
                    start_idx = j * scale
                    end_idx = min(start_idx + scale, len(price_segment))
                    coarse_grained[j] = np.mean(price_segment[start_idx:end_idx])
            
            # サンプルエントロピーの計算
            if len(coarse_grained) > pattern_length + 1:
                tolerance = tolerance_ratio * np.std(coarse_grained)
                sample_entropy = calculate_sample_entropy(coarse_grained, pattern_length, tolerance)
                
                if not np.isnan(sample_entropy):
                    entropy_sum += sample_entropy
                    valid_scales += 1
        
        if valid_scales > 0:
            mse_values[i] = entropy_sum / valid_scales
        else:
            mse_values[i] = 0.5
    
    return mse_values


@njit(fastmath=True, cache=True)
def calculate_sample_entropy(
    data: np.ndarray,
    pattern_length: int,
    tolerance: float
) -> float:
    """
    サンプルエントロピーの計算
    """
    n = len(data)
    if n <= pattern_length:
        return np.nan
    
    # パターンマッチング
    matches_m = 0
    matches_m_plus_1 = 0
    
    for i in range(n - pattern_length):
        # 長さmのパターン
        pattern_m = data[i:i + pattern_length]
        
        # 長さm+1のパターン
        if i + pattern_length < n:
            pattern_m_plus_1 = data[i:i + pattern_length + 1]
        else:
            continue
        
        for j in range(i + 1, n - pattern_length):
            # 長さmの比較
            candidate_m = data[j:j + pattern_length]
            
            # 最大差を計算
            max_diff_m = 0.0
            for k in range(pattern_length):
                diff = abs(pattern_m[k] - candidate_m[k])
                if diff > max_diff_m:
                    max_diff_m = diff
            
            if max_diff_m <= tolerance:
                matches_m += 1
                
                # 長さm+1の比較
                if j + pattern_length < n:
                    candidate_m_plus_1 = data[j:j + pattern_length + 1]
                    
                    max_diff_m_plus_1 = 0.0
                    for k in range(pattern_length + 1):
                        diff = abs(pattern_m_plus_1[k] - candidate_m_plus_1[k])
                        if diff > max_diff_m_plus_1:
                            max_diff_m_plus_1 = diff
                    
                    if max_diff_m_plus_1 <= tolerance:
                        matches_m_plus_1 += 1
    
    # サンプルエントロピーの計算
    if matches_m > 0 and matches_m_plus_1 > 0:
        relative_prevalence = matches_m_plus_1 / matches_m
        if relative_prevalence > 0:
            return -math.log(relative_prevalence)
        else:
            return float('inf')
    else:
        return np.nan 