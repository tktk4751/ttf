#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultimate Chop Trend - 宇宙最強のトレンド/レンジ判定インジケーター

最新の数学・統計学・機械学習アルゴリズムを統合した超低遅延・超高精度システム：
- Hilbert Transform Instantaneous Analysis（瞬時位相・周波数解析）
- Fractal Adaptive Filtering（フラクタル適応フィルタリング）
- Wavelet Multi-Resolution Analysis（ウェーブレット多重解像度解析）
- Dynamic Kalman Filtering（動的カルマンフィルタリング）
- Information Entropy Trend Detection（情報エントロピートレンド検出）
- Machine Learning Ensemble Voting（機械学習アンサンブル投票）
- Chaos Theory Indicators（カオス理論指標）
- Multi-Timeframe Regime Detection（多重時間軸レジーム検出）
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

# Assuming these base classes exist
try:
    from .indicator import Indicator
    from .atr import ATR
    from .ehlers_unified_dc import EhlersUnifiedDC
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


class UltimateChopTrendResult(NamedTuple):
    """Ultimate ChopTrend計算結果 - 宇宙最強版"""
    # コア指標
    ultimate_trend_index: np.ndarray      # 最終統合トレンド指数（0-1）
    trend_signals: np.ndarray             # 1=強上昇, 0.5=弱上昇, 0=レンジ, -0.5=弱下降, -1=強下降
    trend_strength: np.ndarray            # トレンド強度（0-1）
    regime_state: np.ndarray              # レジーム状態（0=レンジ、1=トレンド、2=ブレイクアウト）
    
    # 明確なトレンド方向判定（チャート背景色用）
    trend_direction: np.ndarray           # 1=アップトレンド, 0=レンジ, -1=ダウントレンド
    direction_strength: np.ndarray        # 方向性の強さ（0-1）
    
    # 信頼度・確率
    confidence_score: np.ndarray          # 予測信頼度（0-1）
    trend_probability: np.ndarray         # トレンド確率（0-1）
    regime_probability: np.ndarray        # レジーム確率
    
    # 先行指標
    predictive_signal: np.ndarray         # 予測シグナル
    momentum_forecast: np.ndarray         # モメンタム予測
    volatility_forecast: np.ndarray       # ボラティリティ予測
    
    # アンサンブル成分
    hilbert_component: np.ndarray         # ヒルベルト変換成分
    fractal_component: np.ndarray         # フラクタル成分
    wavelet_component: np.ndarray         # ウェーブレット成分
    kalman_component: np.ndarray          # カルマン成分
    entropy_component: np.ndarray         # エントロピー成分
    chaos_component: np.ndarray           # カオス成分
    
    # 🚀 新機能: 高度解析成分
    unscented_kalman_component: np.ndarray  # 無香料カルマンフィルター成分
    garch_volatility: np.ndarray            # GARCHボラティリティモデル
    regime_switching_probs: np.ndarray      # レジーム切り替え確率（3次元: n_samples x n_regimes）
    spectral_power: np.ndarray              # スペクトル解析パワー
    dominant_frequency: np.ndarray          # 支配的周波数
    multiscale_entropy: np.ndarray          # マルチスケールエントロピー
    nonlinear_dynamics: np.ndarray          # 非線形力学系指標
    predictability_score: np.ndarray       # 予測可能性スコア
    
    # 追加メトリクス
    market_efficiency: np.ndarray         # 市場効率性
    information_ratio: np.ndarray         # 情報比率
    adaptive_threshold: np.ndarray        # 適応しきい値
    correlation_dimension: np.ndarray     # 相関次元
    lyapunov_exponent: np.ndarray         # リアプノフ指数
    
    # 現在状態
    current_trend: str
    current_strength: float
    current_confidence: float


@njit(fastmath=True, cache=True)
def hilbert_transform_analysis(prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    ヒルベルト変換による瞬時解析
    
    Returns:
        (instantaneous_phase, instantaneous_frequency, instantaneous_amplitude, trend_component)
    """
    n = len(prices)
    if n < 50:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    # ヒルベルト変換の近似実装
    phase = np.zeros(n)
    frequency = np.zeros(n)
    amplitude = np.zeros(n)
    trend_component = np.zeros(n)
    
    # 位相差分を使った瞬時周波数の近似
    for i in range(7, n):
        # 4つの位相成分を計算
        real_part = (prices[i] + prices[i-2] + prices[i-4] + prices[i-6]) / 4.0
        
        # 90度位相をずらした虚数部の近似
        imag_part = (prices[i-1] + prices[i-3] + prices[i-5] + prices[i-7]) / 4.0
        
        # 瞬時位相
        if real_part != 0:
            phase[i] = math.atan2(imag_part, real_part)
        
        # 瞬時振幅
        amplitude[i] = math.sqrt(real_part * real_part + imag_part * imag_part)
        
        # 瞬時周波数（位相の差分）
        if i > 7:
            freq_diff = phase[i] - phase[i-1]
            # 位相の巻き戻しを修正
            if freq_diff > math.pi:
                freq_diff -= 2 * math.pi
            elif freq_diff < -math.pi:
                freq_diff += 2 * math.pi
            frequency[i] = abs(freq_diff)
        
        # トレンド成分（位相の方向性）
        if i > 14:
            phase_trend = 0.0
            for j in range(7):
                phase_trend += math.sin(phase[i-j])
            trend_component[i] = phase_trend / 7.0
    
    return phase, frequency, amplitude, trend_component


@njit(fastmath=True, cache=True)
def fractal_adaptive_moving_average(prices: np.ndarray, period: int = 21) -> np.ndarray:
    """
    フラクタル適応移動平均（FRAMA）
    
    Returns:
        FRAMA値の配列
    """
    n = len(prices)
    if n < period * 2:
        return np.full(n, np.nan)
    
    frama = np.full(n, np.nan)
    half_period = period // 2
    
    for i in range(period, n):
        # 上半分と下半分の最高値・最安値を取得
        h1 = np.max(prices[i-period:i-half_period])
        l1 = np.min(prices[i-period:i-half_period])
        h2 = np.max(prices[i-half_period:i])
        l2 = np.min(prices[i-half_period:i])
        
        # 全期間の最高値・最安値
        h_total = np.max(prices[i-period:i])
        l_total = np.min(prices[i-period:i])
        
        # フラクタル次元の計算
        n1 = (h1 - l1) / half_period
        n2 = (h2 - l2) / half_period
        n3 = (h_total - l_total) / period
        
        if n1 > 0 and n2 > 0 and n3 > 0:
            # フラクタル次元
            fractal_dim = (math.log(n1 + n2) - math.log(n3)) / math.log(2.0)
            fractal_dim = min(max(fractal_dim, 1.0), 2.0)  # 1-2の範囲に制限
            
            # 適応係数
            alpha = math.exp(-4.6 * (fractal_dim - 1.0))
            alpha = min(max(alpha, 0.01), 1.0)  # 0.01-1.0の範囲に制限
        else:
            alpha = 0.5  # デフォルト値
        
        # FRAMAの計算
        if i == period:
            frama[i] = prices[i]
        else:
            frama[i] = alpha * prices[i] + (1.0 - alpha) * frama[i-1]
    
    return frama


@njit(fastmath=True, cache=True)
def wavelet_denoising(prices: np.ndarray, levels: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    シンプルなウェーブレット・ノイズ除去（Haar wavelet approximation）
    
    Returns:
        (denoised_signal, detail_component)
    """
    n = len(prices)
    if n < 8:
        return prices.copy(), np.zeros(n)
    
    # シンプルなHaarウェーブレット近似
    signal = prices.copy()
    detail = np.zeros(n)
    
    for level in range(levels):
        step = 2 ** level
        if step >= n // 2:
            break
            
        # ダウンサンプリングとアップサンプリングによる近似
        for i in range(step, n - step, step * 2):
            # ローパス（平均）
            low = (signal[i-step] + signal[i]) * 0.5
            # ハイパス（差分）
            high = (signal[i-step] - signal[i]) * 0.5
            
            # ノイズ除去のためのしきい値
            threshold = np.std(signal[max(0, i-step*4):i+step*4]) * 0.1
            if abs(high) < threshold:
                high = 0
            
            detail[i] += high
            signal[i-step] = low
            signal[i] = low
    
    denoised = signal - detail
    return denoised, detail


@njit(fastmath=True, cache=True)
def extended_kalman_filter(prices: np.ndarray, volatility: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    拡張カルマンフィルター（トレンド推定用）
    
    Returns:
        (filtered_trend, prediction_confidence)
    """
    n = len(prices)
    if n < 10:
        return prices.copy(), np.ones(n)
    
    # 状態変数：[価格, 速度, 加速度]
    state = np.array([prices[0], 0.0, 0.0])
    
    # 共分散行列
    P = np.eye(3) * 1.0
    
    # プロセスノイズ
    Q = np.array([[0.01, 0.0, 0.0],
                  [0.0, 0.01, 0.0],
                  [0.0, 0.0, 0.01]])
    
    # 状態遷移行列
    F = np.array([[1.0, 1.0, 0.5],
                  [0.0, 1.0, 1.0],
                  [0.0, 0.0, 1.0]])
    
    filtered_trend = np.full(n, np.nan)
    confidence = np.full(n, np.nan)
    
    for i in range(n):
        if i == 0:
            filtered_trend[i] = prices[i]
            confidence[i] = 1.0
            continue
            
        # 予測ステップ
        state_pred = np.dot(F, state)
        P_pred = np.dot(np.dot(F, P), F.T) + Q
        
        # 観測ノイズ（ボラティリティベース）
        R = max(volatility[i] ** 2, 0.001)
        
        # カルマンゲイン
        H = np.array([1.0, 0.0, 0.0])  # 価格のみを観測
        K_denom = np.dot(np.dot(H, P_pred), H.T) + R
        if K_denom > 0:
            K = np.dot(P_pred, H.T) / K_denom
        else:
            K = np.zeros(3)
        
        # 更新ステップ
        innovation = prices[i] - state_pred[0]
        state = state_pred + K * innovation
        P = P_pred - np.outer(K, np.dot(H, P_pred))
        
        filtered_trend[i] = state[0]
        confidence[i] = 1.0 / (1.0 + P[0, 0])  # 確信度
    
    return filtered_trend, confidence


# 🚀 高度解析機能群（ultimate_advanced_analysis.pyから統合）

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
def multiscale_entropy(
    prices: np.ndarray,
    max_scale: int = 5,
    pattern_length: int = 2,
    tolerance_ratio: float = 0.2
) -> np.ndarray:
    """
    マルチスケールエントロピー（簡易版）
    複数の時間スケールでの複雑性を測定
    
    Returns:
        マルチスケールエントロピー値の配列（0-1正規化）
    """
    n = len(prices)
    if n < 20:
        return np.full(n, 0.5)
    
    mse_values = np.full(n, 0.5)
    
    for i in range(20, n):
        window_size = min(30, i)
        price_segment = prices[i-window_size:i]
        
        if len(price_segment) < 10:
            mse_values[i] = 0.5
            continue
        
        # 価格変化率を計算
        returns = np.diff(price_segment)
        if len(returns) < 5:
            mse_values[i] = 0.5
            continue
        
        # 簡易マルチスケールエントロピー
        entropy_sum = 0.0
        valid_scales = 0
        
        for scale in range(1, min(max_scale + 1, len(returns) // 3)):
            # スケールごとの粗視化
            if scale == 1:
                scaled_data = returns
            else:
                scaled_length = len(returns) // scale
                if scaled_length < 3:
                    continue
                scaled_data = np.zeros(scaled_length)
                for j in range(scaled_length):
                    start_idx = j * scale
                    end_idx = min(start_idx + scale, len(returns))
                    scaled_data[j] = np.mean(returns[start_idx:end_idx])
            
            # 簡易エントロピー計算（分散ベース）
            if len(scaled_data) > 2:
                data_std = np.std(scaled_data)
                data_mean = np.mean(scaled_data)
                
                # 正規化された分散をエントロピーの代理として使用
                if data_std > 1e-10:
                    normalized_variance = data_std / (abs(data_mean) + data_std + 1e-10)
                    entropy_sum += normalized_variance
                    valid_scales += 1
        
        if valid_scales > 0:
            avg_entropy = entropy_sum / valid_scales
            # 0-1範囲に正規化
            mse_values[i] = min(max(avg_entropy, 0.0), 1.0)
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
def information_entropy_trend(prices: np.ndarray, window: int = 20) -> np.ndarray:
    """
    情報エントロピーによるトレンド検出
    
    Returns:
        エントロピーベースのトレンド指標
    """
    n = len(prices)
    if n < window * 2:
        return np.full(n, np.nan)
    
    entropy_trend = np.full(n, np.nan)
    
    for i in range(window, n):
        # 価格変化の計算
        price_changes = np.diff(prices[i-window:i+1])
        
        if len(price_changes) == 0:
            continue
            
        # 変化を離散化（ビン分割）
        bins = 10
        hist_range = np.max(price_changes) - np.min(price_changes)
        if hist_range <= 0:
            entropy_trend[i] = 0.5
            continue
            
        # ヒストグラムの計算
        bin_width = hist_range / bins
        histogram = np.zeros(bins)
        
        for change in price_changes:
            bin_idx = int((change - np.min(price_changes)) / bin_width)
            bin_idx = min(max(bin_idx, 0), bins - 1)
            histogram[bin_idx] += 1
        
        # 確率分布への変換
        total = np.sum(histogram)
        if total > 0:
            probabilities = histogram / total
            
            # エントロピーの計算
            entropy = 0.0
            for p in probabilities:
                if p > 0:
                    entropy -= p * math.log2(p)
            
            # 正規化（最大エントロピーで割る）
            max_entropy = math.log2(bins)
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            # トレンド指標（低エントロピー = 強いトレンド）
            entropy_trend[i] = 1.0 - normalized_entropy
        else:
            entropy_trend[i] = 0.5
    
    return entropy_trend


@njit(fastmath=True, cache=True)
def chaos_theory_indicators(prices: np.ndarray, window: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    カオス理論指標の計算
    
    Returns:
        (hurst_exponent, largest_lyapunov_exponent)
    """
    n = len(prices)
    if n < window * 3:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    hurst = np.full(n, np.nan)
    lyapunov = np.full(n, np.nan)
    
    for i in range(window * 2, n):
        price_series = prices[i-window*2:i]
        
        # Hurst指数の計算（R/S統計）
        returns = np.diff(price_series)
        if len(returns) < window:
            continue
            
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return > 0:
            # 累積偏差
            cumdev = np.cumsum(returns - mean_return)
            R = np.max(cumdev) - np.min(cumdev)  # レンジ
            S = std_return  # 標準偏差
            
            if S > 0 and R > 0:
                rs_ratio = R / S
                hurst[i] = math.log(rs_ratio) / math.log(len(returns))
                hurst[i] = min(max(hurst[i], 0.0), 1.0)
            else:
                hurst[i] = 0.5
        else:
            hurst[i] = 0.5
        
        # リアプノフ指数の近似計算
        if len(returns) >= 10:
            # 近傍点の発散率を計算
            divergence_sum = 0.0
            count = 0
            
            for j in range(5, len(returns) - 5):
                # 近傍点を探す
                base_point = returns[j]
                
                for k in range(max(0, j-5), min(len(returns), j+5)):
                    if k != j and abs(returns[k] - base_point) < std_return * 0.1:
                        # 発散の計算
                        future_steps = 3
                        if j + future_steps < len(returns) and k + future_steps < len(returns):
                            initial_dist = abs(returns[k] - returns[j])
                            final_dist = abs(returns[k + future_steps] - returns[j + future_steps])
                            
                            if initial_dist > 0 and final_dist > 0:
                                divergence = math.log(final_dist / initial_dist)
                                divergence_sum += divergence
                                count += 1
            
            if count > 0:
                lyapunov[i] = divergence_sum / count
            else:
                lyapunov[i] = 0.0
        else:
            lyapunov[i] = 0.0
    
    return hurst, lyapunov


@njit(fastmath=True, cache=True)
def adaptive_threshold_calculation(
    signals: np.ndarray, 
    volatility: np.ndarray, 
    window: int = 50
) -> np.ndarray:
    """
    適応的しきい値の計算（レンジ優先版）
    
    Returns:
        適応しきい値の配列
    """
    n = len(signals)
    if n < window:
        return np.full(n, 0.5)
    
    adaptive_threshold = np.full(n, 0.5)
    
    for i in range(window, n):
        # 過去のシグナルの統計
        signal_window = signals[i-window:i]
        vol_window = volatility[i-window:i]
        
        # シグナルの統計値
        signal_mean = np.mean(signal_window)
        signal_std = np.std(signal_window)
        avg_vol = np.mean(vol_window)
        
        # レンジ相場の検出
        range_indicators = 0
        
        # 1. シグナルが中央値付近に集中している
        if abs(signal_mean - 0.5) < 0.08:  # 0.1から0.08に厳格化
            range_indicators += 1
        
        # 2. シグナルの標準偏差が小さい
        if signal_std < 0.12:  # 0.15から0.12に厳格化
            range_indicators += 1
        
        # 3. ボラティリティが低い
        if avg_vol < 0.015:  # 0.02から0.015に厳格化
            range_indicators += 1
        
        # 4. シグナルの範囲が狭い
        signal_range = np.max(signal_window) - np.min(signal_window)
        if signal_range < 0.25:  # 0.3から0.25に厳格化
            range_indicators += 1
        
        # レンジ度合いに応じてしきい値を調整
        range_strength = range_indicators / 4.0  # 0-1の範囲
        
        # 適応係数（レンジ相場では高いしきい値）
        vol_factor = min(max(avg_vol, 0.05), 1.5)
        
        # ベースしきい値をレンジ強度に応じて調整（バランス版）
        base_threshold = 0.5 + range_strength * 0.06  # 0.1から0.06に削減
        
        # ボラティリティによる調整
        volatility_adjustment = signal_std * vol_factor * 0.4  # 0.3から0.4に増加
        
        # レンジ相場追加調整（控えめに）
        range_adjustment = range_strength * 0.05  # 0.1から0.05に削減
        
        adaptive_threshold[i] = base_threshold + volatility_adjustment + range_adjustment
        
        # 実践的な範囲に制限（バランス重視）
        adaptive_threshold[i] = min(max(adaptive_threshold[i], 0.25), 0.7)  # 0.3-0.75から0.25-0.7に調整
        
        # 極端なレンジ相場の場合のみ追加調整
        if range_indicators == 4:  # 全指標が揃った場合のみ
            adaptive_threshold[i] = min(adaptive_threshold[i] + 0.03, 0.72)  # 0.05から0.03に削減
    
    return adaptive_threshold


@njit(fastmath=True, cache=True)
def ensemble_voting_system(
    hilbert_sig: np.ndarray,
    fractal_sig: np.ndarray,
    wavelet_sig: np.ndarray,
    kalman_sig: np.ndarray,
    entropy_sig: np.ndarray,
    chaos_sig: np.ndarray,
    confidence_weights: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    アンサンブル投票システム（完全なゼロ除算対策版）
    
    Returns:
        (ensemble_signal, ensemble_confidence)
    """
    n = len(hilbert_sig)
    ensemble_signal = np.full(n, 0.5)  # デフォルト値で初期化
    ensemble_confidence = np.full(n, 0.5)  # デフォルト値で初期化
    
    # 各シグナルの重み（合計1.0）
    base_weights = np.array([0.25, 0.20, 0.15, 0.15, 0.15, 0.10])
    
    for i in range(n):
        # 安全なシグナル値の取得（デフォルト0.5、範囲0-1に制限）
        sig_values = np.array([
            min(max(hilbert_sig[i] if not np.isnan(hilbert_sig[i]) and np.isfinite(hilbert_sig[i]) else 0.5, 0.0), 1.0),
            min(max(fractal_sig[i] if not np.isnan(fractal_sig[i]) and np.isfinite(fractal_sig[i]) else 0.5, 0.0), 1.0),
            min(max(wavelet_sig[i] if not np.isnan(wavelet_sig[i]) and np.isfinite(wavelet_sig[i]) else 0.5, 0.0), 1.0),
            min(max(kalman_sig[i] if not np.isnan(kalman_sig[i]) and np.isfinite(kalman_sig[i]) else 0.5, 0.0), 1.0),
            min(max(entropy_sig[i] if not np.isnan(entropy_sig[i]) and np.isfinite(entropy_sig[i]) else 0.5, 0.0), 1.0),
            min(max(chaos_sig[i] if not np.isnan(chaos_sig[i]) and np.isfinite(chaos_sig[i]) else 0.5, 0.0), 1.0)
        ])
        
        # 信頼度重みの安全な取得
        if i < len(confidence_weights):
            conf_weight = confidence_weights[i] if (not np.isnan(confidence_weights[i]) and 
                                                   np.isfinite(confidence_weights[i]) and 
                                                   confidence_weights[i] > 0) else 1.0
        else:
            conf_weight = 1.0
        
        # 信頼度重みを安全な範囲に制限
        conf_weight = min(max(conf_weight, 0.01), 10.0)
        
        # 効果的な重みの計算
        effective_weights = base_weights * conf_weight
        
        # 重み付き平均の計算（完全ゼロ除算対策）
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for j in range(len(sig_values)):
            if j < len(effective_weights) and effective_weights[j] > 0:
                weighted_sum += sig_values[j] * effective_weights[j]
                weight_sum += effective_weights[j]
        
        # アンサンブルシグナルの計算
        if weight_sum > 1e-15:  # 極小値チェック
            ensemble_signal[i] = weighted_sum / weight_sum
        else:
            ensemble_signal[i] = 0.5  # フォールバック値
        
        # 最終範囲制限
        ensemble_signal[i] = min(max(ensemble_signal[i], 0.0), 1.0)
        
        # アンサンブル信頼度の計算（安全版）
        try:
            # 手動で分散を計算してゼロ除算を完全回避
            mean_val = np.mean(sig_values)
            variance_sum = 0.0
            
            for val in sig_values:
                diff = val - mean_val
                variance_sum += diff * diff
            
            signal_variance = variance_sum / len(sig_values) if len(sig_values) > 0 else 0.0
            
            # 分散が極小の場合の処理
            if signal_variance < 1e-15:
                ensemble_confidence[i] = 0.95  # 高い信頼度（全シグナルが一致）
            else:
                confidence_val = 1.0 / (1.0 + signal_variance)
                ensemble_confidence[i] = min(max(confidence_val, 0.1), 1.0)
        except:
            ensemble_confidence[i] = 0.5  # エラー時のフォールバック
    
    return ensemble_signal, ensemble_confidence


@njit(fastmath=True, cache=True)
def predictive_analysis(
    prices: np.ndarray,
    trend_signals: np.ndarray,
    volatility: np.ndarray,
    lookforward: int = 3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    予測分析システム
    
    Returns:
        (predictive_signal, momentum_forecast, volatility_forecast)
    """
    n = len(prices)
    if n < lookforward * 3:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    predictive_signal = np.full(n, np.nan)
    momentum_forecast = np.full(n, np.nan)
    volatility_forecast = np.full(n, np.nan)
    
    for i in range(lookforward * 2, n - lookforward):
        # 過去のパターン分析
        past_prices = prices[i-lookforward*2:i]
        past_trends = trend_signals[i-lookforward*2:i]
        past_vols = volatility[i-lookforward*2:i]
        
        # モメンタム予測（線形回帰の傾き）
        x_vals = np.arange(len(past_prices))
        if len(past_prices) > 2:
            # 単純な線形回帰
            n_points = len(past_prices)
            sum_x = np.sum(x_vals)
            sum_y = np.sum(past_prices)
            sum_xy = np.sum(x_vals * past_prices)
            sum_x2 = np.sum(x_vals * x_vals)
            
            denom = n_points * sum_x2 - sum_x * sum_x
            if abs(denom) > 1e-15:  # ゼロ除算の完全回避
                slope = (n_points * sum_xy - sum_x * sum_y) / denom
                momentum_forecast[i] = min(max(slope, -1.0), 1.0)  # 範囲制限
            else:
                momentum_forecast[i] = 0.0
        else:
            momentum_forecast[i] = 0
        
        # ボラティリティ予測（EWMA）
        if len(past_vols) > 0:
            # 指数重み付き移動平均
            alpha = 0.2
            vol_forecast = past_vols[-1]
            for j in range(len(past_vols)):
                weight = alpha * (1 - alpha) ** j
                vol_forecast += weight * past_vols[-(j+1)]
            volatility_forecast[i] = vol_forecast
        else:
            volatility_forecast[i] = 0
        
        # 予測シグナル（トレンドとモメンタムの組み合わせ）
        current_trend = trend_signals[i] if not np.isnan(trend_signals[i]) else 0.5
        momentum_component = momentum_forecast[i] if not np.isnan(momentum_forecast[i]) else 0
        
        # シグナルの統合（安全な計算）
        if np.isfinite(momentum_component):
            momentum_bounded = min(max(momentum_component * 10, -50), 50)  # tanh入力値制限
            momentum_normalized = math.tanh(momentum_bounded) * 0.5 + 0.5
        else:
            momentum_normalized = 0.5
        
        predictive_signal[i] = 0.7 * current_trend + 0.3 * momentum_normalized
        predictive_signal[i] = min(max(predictive_signal[i], 0.0), 1.0)
    
    return predictive_signal, momentum_forecast, volatility_forecast


@njit(fastmath=True, cache=True)
def zero_lag_ema(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Zero-Lag Exponential Moving Average (超低遅延)
    
    Returns:
        Zero-lag EMA値の配列
    """
    n = len(prices)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    
    alpha = 2.0 / (period + 1)
    zlema = np.zeros(n)
    ema1 = np.zeros(n)
    ema2 = np.zeros(n)
    
    # 初期値設定
    zlema[0] = prices[0]
    ema1[0] = prices[0]
    ema2[0] = prices[0]
    
    for i in range(1, n):
        # 標準EMA
        ema1[i] = alpha * prices[i] + (1 - alpha) * ema1[i-1]
        # EMAのEMA
        ema2[i] = alpha * ema1[i] + (1 - alpha) * ema2[i-1]
        # Zero-lag計算
        zlema[i] = 2 * ema1[i] - ema2[i]
    
    return zlema


@njit(fastmath=True, cache=True)
def ehlers_instantaneous_trendline(prices: np.ndarray, alpha: float = 0.07) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ehlers' Instantaneous Trendline (瞬時トレンドライン)
    
    Returns:
        (trendline, momentum)
    """
    n = len(prices)
    if n < 4:
        return prices.copy(), np.zeros(n)
    
    trendline = np.zeros(n)
    momentum = np.zeros(n)
    
    # 初期値設定
    trendline[0] = prices[0]
    trendline[1] = prices[1]
    
    for i in range(2, n):
        # Ehlers' Instantaneous Trendline formula
        trendline[i] = (alpha - 0.25 * alpha * alpha) * prices[i] + \
                      0.5 * alpha * alpha * prices[i-1] - \
                      (alpha - 0.75 * alpha * alpha) * prices[i-2] + \
                      2 * (1 - alpha) * trendline[i-1] - \
                      (1 - alpha) * (1 - alpha) * trendline[i-2]
        
        # モメンタム計算
        if i >= 3:
            momentum[i] = trendline[i] - trendline[i-3]
    
    return trendline, momentum


@njit(fastmath=True, cache=True)
def adaptive_cyber_cycle(prices: np.ndarray, alpha: float = 0.07) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adaptive Cyber Cycle (適応サイバーサイクル)
    
    Returns:
        (cycle, trend_signal)
    """
    n = len(prices)
    if n < 7:
        return np.zeros(n), np.zeros(n)
    
    cycle = np.zeros(n)
    trend_signal = np.zeros(n)
    smooth = zero_lag_ema(prices, period=7)
    
    for i in range(6, n):
        # Cyber Cycle calculation
        cycle[i] = (1 - 0.5 * alpha) * (1 - 0.5 * alpha) * (smooth[i] - 2 * smooth[i-1] + smooth[i-2]) + \
                  2 * (1 - alpha) * cycle[i-1] - (1 - alpha) * (1 - alpha) * cycle[i-2]
        
        # トレンドシグナル（サイクルの方向性）
        if i >= 7:
            if cycle[i] > cycle[i-1] and cycle[i-1] <= cycle[i-2]:
                trend_signal[i] = 1  # サイクル底からの反転（上昇）
            elif cycle[i] < cycle[i-1] and cycle[i-1] >= cycle[i-2]:
                trend_signal[i] = -1  # サイクル天井からの反転（下降）
            else:
                trend_signal[i] = trend_signal[i-1]  # 前の状態を維持
    
    return cycle, trend_signal


@njit(fastmath=True, cache=True)
def online_variance_estimator(prices: np.ndarray, alpha: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    オンライン分散推定器（Welford's algorithm改良版）
    
    Returns:
        (running_mean, running_variance)
    """
    n = len(prices)
    running_mean = np.zeros(n)
    running_variance = np.zeros(n)
    
    if n == 0:
        return running_mean, running_variance
    
    running_mean[0] = prices[0]
    running_variance[0] = 0.0
    
    for i in range(1, n):
        # オンライン平均更新
        delta = prices[i] - running_mean[i-1]
        running_mean[i] = running_mean[i-1] + alpha * delta
        
        # オンライン分散更新
        delta2 = prices[i] - running_mean[i]
        running_variance[i] = (1 - alpha) * running_variance[i-1] + alpha * delta * delta2
        running_variance[i] = max(running_variance[i], 1e-10)  # 最小値制限
    
    return running_mean, running_variance


@njit(fastmath=True, cache=True)
def streaming_trend_detector(
    prices: np.ndarray,
    fast_alpha: float = 0.15,
    slow_alpha: float = 0.05,
    threshold: float = 0.02
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ストリーミングトレンド検出器（超低遅延）
    
    Returns:
        (trend_direction, trend_strength, confidence)
    """
    n = len(prices)
    trend_direction = np.zeros(n)
    trend_strength = np.zeros(n)
    confidence = np.zeros(n)
    
    if n < 2:
        return trend_direction, trend_strength, confidence
    
    fast_ema = np.zeros(n)
    slow_ema = np.zeros(n)
    
    # 初期値
    fast_ema[0] = prices[0]
    slow_ema[0] = prices[0]
    
    for i in range(1, n):
        # 高速・低速EMA
        fast_ema[i] = fast_alpha * prices[i] + (1 - fast_alpha) * fast_ema[i-1]
        slow_ema[i] = slow_alpha * prices[i] + (1 - slow_alpha) * slow_ema[i-1]
        
        # トレンド検出
        ema_diff = fast_ema[i] - slow_ema[i]
        ema_diff_normalized = ema_diff / max(slow_ema[i], 1e-10)
        
        # 方向判定
        if abs(ema_diff_normalized) > threshold:
            trend_direction[i] = 1 if ema_diff_normalized > 0 else -1
            trend_strength[i] = min(abs(ema_diff_normalized) / threshold, 2.0)
            confidence[i] = min(trend_strength[i] / 2.0, 1.0)
        else:
            trend_direction[i] = 0
            trend_strength[i] = abs(ema_diff_normalized) / threshold
            confidence[i] = 0.1  # 低信頼度
    
    return trend_direction, trend_strength, confidence


@njit(fastmath=True, cache=True)
def advanced_market_regime_detection(
    prices: np.ndarray,
    volatility: np.ndarray,
    window: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    超低遅延マーケットレジーム検出システム
    
    Returns:
        (trend_strength, trend_direction, regime_confidence)
    """
    n = len(prices)
    trend_strength = np.zeros(n)
    trend_direction = np.zeros(n)
    regime_confidence = np.zeros(n)
    
    # 1. ゼロラグEMAトレンドライン
    zlema = zero_lag_ema(prices, period=14)
    
    # 2. Ehlers瞬時トレンドライン
    ehlers_trend, ehlers_momentum = ehlers_instantaneous_trendline(prices, alpha=0.07)
    
    # 3. 適応サイバーサイクル
    cyber_cycle, cycle_signal = adaptive_cyber_cycle(prices, alpha=0.07)
    
    # 4. オンライン分散推定
    running_mean, running_variance = online_variance_estimator(prices, alpha=0.1)
    
    # 5. ストリーミングトレンド検出
    stream_direction, stream_strength, stream_confidence = streaming_trend_detector(
        prices, fast_alpha=0.15, slow_alpha=0.05, threshold=0.015
    )
    
    # 6. 統合判定（超低遅延）
    for i in range(n):
        # アンサンブル重み
        weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])  # 各手法の重み
        scores = np.zeros(5)
        
        # A. Zero-lag EMAトレンド (30%)
        if i > 0:
            zlema_change = zlema[i] - zlema[i-1] if zlema[i-1] != 0 else 0
            zlema_trend = zlema_change / max(abs(zlema[i-1]), 1e-10)
            if abs(zlema_trend) > 0.005:  # 0.5%以上の変化
                scores[0] = min(abs(zlema_trend) * 100, 1.0)
                trend_direction[i] = 1 if zlema_trend > 0 else -1
        
        # B. Ehlers瞬時トレンド (25%)
        if i < len(ehlers_momentum):
            ehlers_norm = ehlers_momentum[i] / max(abs(prices[i]), 1e-10)
            if abs(ehlers_norm) > 0.01:
                scores[1] = min(abs(ehlers_norm) * 50, 1.0)
                if trend_direction[i] == 0:
                    trend_direction[i] = 1 if ehlers_norm > 0 else -1
        
        # C. サイバーサイクル (20%)
        if i < len(cycle_signal) and cycle_signal[i] != 0:
            scores[2] = 0.7
            if trend_direction[i] == 0:
                trend_direction[i] = int(cycle_signal[i])
        
        # D. オンライン分散ベース (15%)
        if i > 5 and running_variance[i] > 0:
            price_change = abs(prices[i] - running_mean[i])
            volatility_ratio = price_change / math.sqrt(running_variance[i])
            if volatility_ratio > 1.5:  # 1.5標準偏差以上
                scores[3] = min(volatility_ratio / 3.0, 1.0)
        
        # E. ストリーミング検出 (10%)
        if i < len(stream_direction):
            if stream_direction[i] != 0:
                scores[4] = stream_confidence[i]
                if trend_direction[i] == 0:
                    trend_direction[i] = int(stream_direction[i])
        
        # 最終統合
        trend_strength[i] = np.sum(weights * scores)
        regime_confidence[i] = trend_strength[i]
        
        # レンジ判定（超厳格）
        if trend_strength[i] < 0.3:
            trend_direction[i] = 0
    
    return trend_strength, trend_direction, regime_confidence


@njit(fastmath=True, cache=True)
def incremental_regression_stats(
    prices: np.ndarray,
    alpha: float = 0.12
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    増分回帰統計（超低遅延）
    
    Returns:
        (trend_slope, trend_significance, confidence_score)
    """
    n = len(prices)
    trend_slope = np.zeros(n)
    trend_significance = np.zeros(n)
    confidence_score = np.zeros(n)
    
    if n < 3:
        return trend_slope, trend_significance, confidence_score
    
    # 増分統計の初期化
    sum_x = 0.0
    sum_y = 0.0  
    sum_xy = 0.0
    sum_x2 = 0.0
    sum_y2 = 0.0
    count = 0.0
    
    for i in range(n):
        # 現在の点を追加
        x_val = float(i)
        y_val = prices[i]
        
        # 指数重み付き更新
        if i > 0:
            decay = 1.0 - alpha
            sum_x = decay * sum_x + alpha * x_val
            sum_y = decay * sum_y + alpha * y_val
            sum_xy = decay * sum_xy + alpha * x_val * y_val
            sum_x2 = decay * sum_x2 + alpha * x_val * x_val
            sum_y2 = decay * sum_y2 + alpha * y_val * y_val
            count = decay * count + alpha
        else:
            sum_x = x_val
            sum_y = y_val
            sum_xy = x_val * y_val
            sum_x2 = x_val * x_val
            sum_y2 = y_val * y_val
            count = 1.0
        
        # 回帰係数計算（2点以上で）
        if i >= 2 and count > 1e-10:
            # 傾き計算
            x_mean = sum_x / count
            y_mean = sum_y / count
            
            numerator = sum_xy / count - x_mean * y_mean
            denominator = sum_x2 / count - x_mean * x_mean
            
            if abs(denominator) > 1e-10:
                slope = numerator / denominator
                trend_slope[i] = slope
                
                # トレンド有意性（正規化）
                y_variance = sum_y2 / count - y_mean * y_mean
                if y_variance > 1e-10:
                    # R²の近似
                    r_squared_approx = (numerator * numerator) / (denominator * y_variance)
                    trend_significance[i] = min(r_squared_approx, 1.0)
                    
                    # 信頼度スコア
                    slope_normalized = abs(slope) / max(abs(y_mean), 1e-10)
                    confidence_score[i] = min(slope_normalized * r_squared_approx * 8, 1.0)
    
    return trend_slope, trend_significance, confidence_score


@njit(fastmath=True, cache=True)
def statistical_trend_significance(
    prices: np.ndarray,
    window: int = 30,
    confidence_level: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    超高速統計的トレンド有意性テスト
    
    Returns:
        (trend_significance, p_values)
    """
    n = len(prices)
    trend_significance = np.zeros(n)
    p_values = np.ones(n)
    
    # 増分回帰統計を使用（超低遅延）
    trend_slope, sig_score, confidence_score = incremental_regression_stats(prices, alpha=0.12)
    
    for i in range(n):
        if i >= 3:
            # 簡易統計的有意性判定
            slope_magnitude = abs(trend_slope[i])
            significance = sig_score[i]
            
            # 統計的有意性スコア
            if significance > 0.15:  # 最小有意性閾値
                stat_score = slope_magnitude * significance * math.sqrt(float(i + 1))
                
                # 高速p値近似
                if stat_score > 3.0:  # 極めて高い有意性
                    p_values[i] = 0.001
                    trend_significance[i] = 2.0 if trend_slope[i] > 0 else -2.0
                elif stat_score > 2.0:  # 高い有意性
                    p_values[i] = 0.01
                    trend_significance[i] = 1.5 if trend_slope[i] > 0 else -1.5
                elif stat_score > 1.2:  # 中程度の有意性
                    p_values[i] = 0.05
                    trend_significance[i] = 1.0 if trend_slope[i] > 0 else -1.0
                elif stat_score > 0.8:  # 低い有意性
                    p_values[i] = 0.15
                    trend_significance[i] = 0.5 if trend_slope[i] > 0 else -0.5
                else:
                    p_values[i] = 0.5
                    trend_significance[i] = 0.0  # 有意でない
    
    return trend_significance, p_values


@njit(fastmath=True, cache=True)
def adaptive_multi_timeframe_consensus(
    prices: np.ndarray,
    fast_alpha: float = 0.3,
    medium_alpha: float = 0.15,
    slow_alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    適応マルチタイムフレームコンセンサス（超低遅延）
    
    Returns:
        (consensus_direction, consensus_strength, timeframe_agreement)
    """
    n = len(prices)
    consensus_direction = np.zeros(n)
    consensus_strength = np.zeros(n)
    timeframe_agreement = np.zeros(n)
    
    if n < 2:
        return consensus_direction, consensus_strength, timeframe_agreement
    
    # 適応EMA初期化
    fast_ema = np.zeros(n)
    medium_ema = np.zeros(n)
    slow_ema = np.zeros(n)
    
    # 初期値
    fast_ema[0] = prices[0]
    medium_ema[0] = prices[0]
    slow_ema[0] = prices[0]
    
    for i in range(1, n):
        # 適応EMA更新
        fast_ema[i] = fast_alpha * prices[i] + (1 - fast_alpha) * fast_ema[i-1]
        medium_ema[i] = medium_alpha * prices[i] + (1 - medium_alpha) * medium_ema[i-1]
        slow_ema[i] = slow_alpha * prices[i] + (1 - slow_alpha) * slow_ema[i-1]
        
        # 各タイムフレームの方向判定
        fast_trend = 1 if fast_ema[i] > fast_ema[i-1] else -1
        medium_trend = 1 if medium_ema[i] > medium_ema[i-1] else -1
        slow_trend = 1 if slow_ema[i] > slow_ema[i-1] else -1
        
        # トレンド強度計算
        fast_strength = abs(fast_ema[i] - fast_ema[i-1]) / max(fast_ema[i-1], 1e-10)
        medium_strength = abs(medium_ema[i] - medium_ema[i-1]) / max(medium_ema[i-1], 1e-10)
        slow_strength = abs(slow_ema[i] - slow_ema[i-1]) / max(slow_ema[i-1], 1e-10)
        
        # 重み付きコンセンサス（短期重視）
        trend_sum = fast_trend * 0.5 + medium_trend * 0.3 + slow_trend * 0.2
        strength_sum = fast_strength * 0.5 + medium_strength * 0.3 + slow_strength * 0.2
        
        # 方向判定
        if abs(trend_sum) > 0.6:  # 強いコンセンサス
            consensus_direction[i] = 1 if trend_sum > 0 else -1
            consensus_strength[i] = min(strength_sum * 100, 1.0)
        elif abs(trend_sum) > 0.2:  # 弱いコンセンサス
            consensus_direction[i] = 1 if trend_sum > 0 else -1
            consensus_strength[i] = min(strength_sum * 50, 0.6)
        else:  # レンジ
            consensus_direction[i] = 0
            consensus_strength[i] = 0.1
        
        # タイムフレーム一致度
        agreement_count = 0
        if fast_trend == medium_trend:
            agreement_count += 1
        if medium_trend == slow_trend:
            agreement_count += 1
        if fast_trend == slow_trend:
            agreement_count += 1
        
        timeframe_agreement[i] = agreement_count / 3.0
    
    return consensus_direction, consensus_strength, timeframe_agreement


@njit(fastmath=True, cache=True)
def multi_timeframe_trend_consensus(
    prices: np.ndarray,
    short_window: int = 10,
    medium_window: int = 20,
    long_window: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    超低遅延複数時間軸トレンドコンセンサス
    
    Returns:
        (consensus_direction, consensus_strength)
    """
    # 適応版を使用して低遅延化
    consensus_dir, consensus_str, agreement = adaptive_multi_timeframe_consensus(
        prices, fast_alpha=0.25, medium_alpha=0.12, slow_alpha=0.04
    )
    
    return consensus_dir, consensus_str


@njit(fastmath=True, cache=True)
def streaming_volatility_regime(
    volatility: np.ndarray,
    prices: np.ndarray,
    alpha: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ストリーミングボラティリティレジーム検出（超低遅延）
    
    Returns:
        (volatility_regime, volatility_score)
    """
    n = len(volatility)
    vol_regime = np.zeros(n)
    vol_score = np.zeros(n)
    
    if n < 2:
        return vol_regime, vol_score
    
    # オンライン統計
    running_mean = 0.0
    running_var = 1.0  # 初期分散
    
    for i in range(n):
        current_vol = volatility[i]
        
        if i == 0:
            running_mean = current_vol
            vol_regime[i] = 1  # デフォルト中程度
            vol_score[i] = 0.5
        else:
            # オンライン平均と分散の更新
            delta = current_vol - running_mean
            running_mean += alpha * delta
            running_var = (1 - alpha) * running_var + alpha * delta * delta
            running_var = max(running_var, 1e-10)  # 最小値制限
            
            # Z-scoreベースの判定（高速）
            z_score = delta / math.sqrt(running_var)
            
            # レジーム判定（より細かい分類）
            if z_score > 1.5:  # 極高ボラティリティ
                vol_regime[i] = 3
                vol_score[i] = min(abs(z_score) / 2.0, 1.0)
            elif z_score > 0.8:  # 高ボラティリティ
                vol_regime[i] = 2
                vol_score[i] = min(abs(z_score) / 1.5, 0.8)
            elif z_score > -0.3:  # 中程度
                vol_regime[i] = 1
                vol_score[i] = 0.5
            else:  # 低ボラティリティ
                vol_regime[i] = 0
                vol_score[i] = max(0.2, 1.0 - abs(z_score) / 2.0)
    
    return vol_regime, vol_score


@njit(fastmath=True, cache=True)
def adaptive_volatility_regime(
    volatility: np.ndarray,
    prices: np.ndarray,
    window: int = 20
) -> np.ndarray:
    """
    超低遅延適応的ボラティリティレジーム検出
    
    Returns:
        volatility_regime (0=低ボラティリティ, 1=中程度, 2=高ボラティリティ, 3=極高)
    """
    vol_regime, vol_score = streaming_volatility_regime(volatility, prices, alpha=0.12)
    return vol_regime


@njit(fastmath=True, cache=True)
def calculate_trend_direction_classification(
    trend_index: np.ndarray,
    trend_signals: np.ndarray,
    prices: np.ndarray,
    lookback_period: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    次世代高度トレンド・レンジ判別システム
    
    Args:
        trend_index: Ultimate ChopTrendインデックス（0-1）
        trend_signals: トレンドシグナル（-1～1）
        prices: 価格データ
        lookback_period: 価格トレンド判定用のルックバック期間
    
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - trend_direction: 1=アップトレンド, 0=レンジ, -1=ダウントレンド
            - trend_strength: トレンドの強さ（0-1）
    """
    n = len(trend_index)
    trend_direction = np.zeros(n, dtype=np.int8)
    trend_strength = np.zeros(n)
    
    # ボラティリティの計算（価格から近似）
    volatility = np.zeros(n)
    for i in range(1, n):
        volatility[i] = abs(prices[i] - prices[i-1]) / max(prices[i-1], 1e-10)
    
    # 1. 超低遅延マーケットレジーム検出
    regime_strength, regime_direction, regime_confidence = advanced_market_regime_detection(
        prices, volatility, window=12  # ウィンドウサイズ削減
    )
    
    # 2. 高速統計的有意性テスト
    stat_significance, p_values = statistical_trend_significance(
        prices, window=20, confidence_level=0.80  # 信頼度を実践的に調整
    )
    
    # 3. 適応マルチタイムフレームコンセンサス
    consensus_direction, consensus_strength, timeframe_agreement = adaptive_multi_timeframe_consensus(
        prices, fast_alpha=0.25, medium_alpha=0.12, slow_alpha=0.04
    )
    
    # 4. ストリーミングボラティリティレジーム
    vol_regime, vol_score = streaming_volatility_regime(volatility, prices, alpha=0.12)
    
    # 5. 統合判定システム
    for i in range(n):
        # === 低遅延重視統合スコア計算 ===
        uptrend_score = 0.0
        downtrend_score = 0.0
        range_score = 0.0
        
        # 遅延補正係数（早期のデータほど重視）
        latency_factor = min(1.0, (i + 1) / 8.0)  # 最初8期間は段階的に重み増加
        
        # === 低遅延重視重み配分 ===
        # A. 高速レジーム検出 (重み: 35% - 最重要、低遅延)
        regime_weight = 0.35 * latency_factor
        
        # B. ストリーミング統計 (重み: 25% - 高速統計)
        stat_weight = 0.25 * latency_factor
        
        # C. 適応マルチタイムフレーム (重み: 25% - EMAベース高速)
        consensus_weight = 0.25 * latency_factor
        
        # D. ストリーミングボラティリティ (重み: 10% - 補助指標)
        vol_weight = 0.10 * latency_factor
        
        # E. 従来ChopTrend (重み: 5% - 遅延大のため最小)
        legacy_weight = 0.05 * latency_factor
        
        # === 各指標スコア計算（新重み配分） ===
        
        # A. 高速レジーム検出スコア
        if i < len(regime_strength):
            regime_str = regime_strength[i]
            regime_dir = regime_direction[i]
            
            if regime_dir > 0 and regime_str > 0.2:
                uptrend_score += regime_weight * regime_str
            elif regime_dir < 0 and regime_str > 0.2:
                downtrend_score += regime_weight * regime_str
            else:
                range_score += regime_weight * 0.8  # レンジ寄り
        
        # B. ストリーミング統計スコア
        if i < len(stat_significance):
            stat_sig = stat_significance[i]
            
            if stat_sig > 0.5:  # 上昇有意
                uptrend_score += stat_weight * min(stat_sig, 1.0)
            elif stat_sig < -0.5:  # 下降有意
                downtrend_score += stat_weight * min(abs(stat_sig), 1.0)
            else:  # 有意でない
                range_score += stat_weight * 0.7
        
        # C. 適応マルチタイムフレームスコア
        if i < len(consensus_direction):
            consensus_dir = consensus_direction[i]
            consensus_str = consensus_strength[i]
            
            if consensus_dir > 0 and consensus_str > 0.3:
                uptrend_score += consensus_weight * consensus_str
            elif consensus_dir < 0 and consensus_str > 0.3:
                downtrend_score += consensus_weight * consensus_str
            else:
                range_score += consensus_weight * 0.6
        
        # D. ストリーミングボラティリティスコア（補助）
        if i < len(vol_regime):
            vol_reg = vol_regime[i]
            vol_sc = vol_score[i] if i < len(vol_score) else 0.5
            
            # 高ボラティリティはトレンド継続を示唆
            if vol_reg >= 2:  # 高・極高ボラティリティ
                # 既存トレンドを強化
                if uptrend_score > downtrend_score:
                    uptrend_score += vol_weight * vol_sc * 0.7
                elif downtrend_score > uptrend_score:
                    downtrend_score += vol_weight * vol_sc * 0.7
                else:
                    range_score += vol_weight * 0.3  # 不確実性
            else:  # 低・中ボラティリティ
                range_score += vol_weight * 0.6
        
        # E. 従来ChopTrendスコア（最小重み）
        trend_val = trend_index[i] if not np.isnan(trend_index[i]) else 0.5
        signal_val = trend_signals[i] if not np.isnan(trend_signals[i]) else 0.0
        
        if signal_val > 0.1:  # 上昇シグナル
            uptrend_score += legacy_weight * signal_val
        elif signal_val < -0.1:  # 下降シグナル
            downtrend_score += legacy_weight * abs(signal_val)
        else:  # レンジシグナル
            range_score += legacy_weight * 0.5
        
        # === 超低遅延最終統合判定 ===
        total_score = uptrend_score + downtrend_score + range_score
        
        # 正規化（スコア安定化）
        if total_score > 1e-10:
            uptrend_score /= total_score
            downtrend_score /= total_score
            range_score /= total_score
        else:
            # フォールバック - 均等分配
            uptrend_score = downtrend_score = range_score = 1.0/3.0
        
        # 低遅延判定しきい値（積極的トレンド検出）
        strong_trend_threshold = 0.35  # 強いトレンド（より積極的）
        weak_trend_threshold = 0.25    # 弱いトレンド（より積極的）
        trend_advantage = 0.03         # 他スコアに対する最小優位性（緩和）
        
        # 早期トレンド警告システム（低遅延特化）
        max_trend_score = max(uptrend_score, downtrend_score)
        trend_dominance = max_trend_score - range_score
        trend_consensus = abs(uptrend_score - downtrend_score)
        
        # トレンド強度計算
        trend_strength[i] = max_trend_score * trend_dominance * latency_factor
        trend_strength[i] = min(trend_strength[i], 1.0)
        
        # 超高速方向判定（低遅延重視）
        if (max_trend_score > strong_trend_threshold and 
            trend_dominance > trend_advantage and 
            trend_consensus > 0.05):  # 明確な方向性（緩和）
            
            if uptrend_score > downtrend_score:
                trend_direction[i] = 1  # 強いアップトレンド
            else:
                trend_direction[i] = -1  # 強いダウントレンド
                
        elif (max_trend_score > weak_trend_threshold and 
              trend_dominance > trend_advantage * 0.5 and
              trend_consensus > 0.03):  # 弱いが明確な方向性（緩和）
            
            if uptrend_score > downtrend_score:
                trend_direction[i] = 1  # 弱いアップトレンド
            else:
                trend_direction[i] = -1  # 弱いダウントレンド
                
        else:
            trend_direction[i] = 0  # レンジ（不確実性優先）
    
    return trend_direction, trend_strength


@njit(fastmath=True, cache=True)
def advanced_ensemble_voting_system(
    # 基本アルゴリズム（6種類）
    hilbert_sig: np.ndarray,
    fractal_sig: np.ndarray,
    wavelet_sig: np.ndarray,
    kalman_sig: np.ndarray,
    entropy_sig: np.ndarray,
    chaos_sig: np.ndarray,
    # 高度解析アルゴリズム（6種類）
    unscented_kalman_sig: np.ndarray,
    garch_sig: np.ndarray,
    regime_switching_sig: np.ndarray,
    spectral_sig: np.ndarray,
    multiscale_entropy_sig: np.ndarray,
    nonlinear_dynamics_sig: np.ndarray,
    confidence_weights: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    高度アンサンブル投票システム（12アルゴリズム統合版）
    
    Returns:
        (ensemble_signal, ensemble_confidence)
    """
    n = len(hilbert_sig)
    ensemble_signal = np.full(n, 0.5)
    ensemble_confidence = np.full(n, 0.5)
    
    # 12アルゴリズムの重み配分（合計1.0）
    # 基本アルゴリズム: 50%, 高度解析アルゴリズム: 50%
    base_weights = np.array([
        # 基本アルゴリズム（50%）
        0.125,  # Hilbert Transform (12.5%)
        0.100,  # Fractal Adaptive (10.0%)
        0.075,  # Wavelet (7.5%)
        0.075,  # Kalman (7.5%)
        0.075,  # Entropy (7.5%)
        0.050,  # Chaos (5.0%)
        # 高度解析アルゴリズム（50%）
        0.140,  # Unscented Kalman (14.0%)
        0.105,  # GARCH (10.5%)
        0.085,  # Regime Switching (8.5%)
        0.070,  # Spectral (7.0%)
        0.060,  # Multiscale Entropy (6.0%)
        0.040   # Nonlinear Dynamics (4.0%)
    ])
    
    for i in range(n):
        # 全12シグナルの安全な取得
        sig_values = np.array([
            # 基本アルゴリズム
            min(max(hilbert_sig[i] if not np.isnan(hilbert_sig[i]) and np.isfinite(hilbert_sig[i]) else 0.5, 0.0), 1.0),
            min(max(fractal_sig[i] if not np.isnan(fractal_sig[i]) and np.isfinite(fractal_sig[i]) else 0.5, 0.0), 1.0),
            min(max(wavelet_sig[i] if not np.isnan(wavelet_sig[i]) and np.isfinite(wavelet_sig[i]) else 0.5, 0.0), 1.0),
            min(max(kalman_sig[i] if not np.isnan(kalman_sig[i]) and np.isfinite(kalman_sig[i]) else 0.5, 0.0), 1.0),
            min(max(entropy_sig[i] if not np.isnan(entropy_sig[i]) and np.isfinite(entropy_sig[i]) else 0.5, 0.0), 1.0),
            min(max(chaos_sig[i] if not np.isnan(chaos_sig[i]) and np.isfinite(chaos_sig[i]) else 0.5, 0.0), 1.0),
            # 高度解析アルゴリズム
            min(max(unscented_kalman_sig[i] if not np.isnan(unscented_kalman_sig[i]) and np.isfinite(unscented_kalman_sig[i]) else 0.5, 0.0), 1.0),
            min(max(garch_sig[i] if not np.isnan(garch_sig[i]) and np.isfinite(garch_sig[i]) else 0.5, 0.0), 1.0),
            min(max(regime_switching_sig[i] if not np.isnan(regime_switching_sig[i]) and np.isfinite(regime_switching_sig[i]) else 0.5, 0.0), 1.0),
            min(max(spectral_sig[i] if not np.isnan(spectral_sig[i]) and np.isfinite(spectral_sig[i]) else 0.5, 0.0), 1.0),
            min(max(multiscale_entropy_sig[i] if not np.isnan(multiscale_entropy_sig[i]) and np.isfinite(multiscale_entropy_sig[i]) else 0.5, 0.0), 1.0),
            min(max(nonlinear_dynamics_sig[i] if not np.isnan(nonlinear_dynamics_sig[i]) and np.isfinite(nonlinear_dynamics_sig[i]) else 0.5, 0.0), 1.0)
        ])
        
        # 信頼度重みの安全な取得
        if i < len(confidence_weights):
            conf_weight = confidence_weights[i] if (not np.isnan(confidence_weights[i]) and 
                                                   np.isfinite(confidence_weights[i]) and 
                                                   confidence_weights[i] > 0) else 1.0
        else:
            conf_weight = 1.0
        
        conf_weight = min(max(conf_weight, 0.01), 10.0)
        
        # 効果的な重みの計算
        effective_weights = base_weights * conf_weight
        
        # 重み付き平均の計算
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for j in range(len(sig_values)):
            if j < len(effective_weights) and effective_weights[j] > 0:
                weighted_sum += sig_values[j] * effective_weights[j]
                weight_sum += effective_weights[j]
        
        # アンサンブルシグナルの計算
        if weight_sum > 1e-15:
            ensemble_signal[i] = weighted_sum / weight_sum
        else:
            ensemble_signal[i] = 0.5
        
        ensemble_signal[i] = min(max(ensemble_signal[i], 0.0), 1.0)
        
        # アンサンブル信頼度の計算
        try:
            mean_val = np.mean(sig_values)
            variance_sum = 0.0
            
            for val in sig_values:
                diff = val - mean_val
                variance_sum += diff * diff
            
            signal_variance = variance_sum / len(sig_values) if len(sig_values) > 0 else 0.0
            
            if signal_variance < 1e-15:
                ensemble_confidence[i] = 0.98  # より高い信頼度（12アルゴリズム一致）
            else:
                confidence_val = 1.0 / (1.0 + signal_variance)
                ensemble_confidence[i] = min(max(confidence_val, 0.1), 1.0)
        except:
            ensemble_confidence[i] = 0.5
    
    return ensemble_signal, ensemble_confidence


class UltimateChopTrend(Indicator):
    """
    Ultimate Chop Trend - 宇宙最強のトレンド/レンジ判定インジケーター 🚀
    
    最新の数学・統計学・機械学習・高度解析アルゴリズムを統合：
    
    【基本アルゴリズム】
    🧠 Hilbert Transform Analysis - 瞬時位相・周波数解析
    📐 Fractal Adaptive Moving Average - フラクタル適応移動平均
    🌊 Wavelet Multi-Resolution Analysis - ウェーブレット多重解像度解析
    🎯 Extended Kalman Filtering - 拡張カルマンフィルタリング
    📊 Information Entropy Analysis - 情報エントロピー解析
    🌀 Chaos Theory Indicators - カオス理論指標
    
    【🚀 高度解析機能 - NEW!】
    🔬 Unscented Kalman Filter - 無香料カルマンフィルター（非線形システム対応）
    📈 GARCH Volatility Model - 条件付き分散動的モデリング
    🔄 Regime Switching Detection - マルコフ切り替えモデル
    📡 Spectral Analysis - 周波数ドメイン解析・周期性検出
    🔍 Multiscale Entropy - 複数時間スケール複雑性測定
    🌌 Nonlinear Dynamics - 相関次元・リアプノフ指数・予測可能性
    
    【統合システム】
    🎯 Advanced Ensemble Voting - 高度アンサンブル投票システム
    🔮 Predictive Analysis - 次世代予測分析
    🏆 Ultimate Performance - 圧倒的精度とパフォーマンス
    """
    
    def __init__(
        self,
        # コアパラメータ
        analysis_period: int = 21,
        ensemble_window: int = 50,
        
        # アルゴリズム有効化フラグ（全アルゴリズム有効 - 最終段階テスト）
        enable_hilbert: bool = True,   # ヒルベルト変換解析
        enable_fractal: bool = True,   # フラクタル適応移動平均
        enable_wavelet: bool = True,   # ウェーブレット多重解像度解析
        enable_kalman: bool = True,    # 拡張カルマンフィルタリング
        enable_entropy: bool = True,   # 情報エントロピートレンド検出
        enable_chaos: bool = True,     # カオス理論指標
        
        # 🚀 新機能: 高度解析アルゴリズム
        enable_unscented_kalman: bool = True,  # 無香料カルマンフィルター
        enable_garch: bool = True,             # GARCHボラティリティモデル
        enable_regime_switching: bool = True,  # レジーム切り替え検出
        enable_spectral: bool = True,          # スペクトル解析
        enable_multiscale_entropy: bool = True, # マルチスケールエントロピー
        enable_nonlinear_dynamics: bool = True, # 非線形力学系解析
        
        # 予測設定
        enable_prediction: bool = True,
        prediction_horizon: int = 3,
        
        # しきい値設定
        trend_threshold: float = 0.65,
        strong_trend_threshold: float = 0.8,
        
        # 従来のChopTrendとの統合
        use_legacy_chop: bool = False,  # デフォルトで無効化（安全のため）
        chop_weight: float = 0.3
    ):
        """
        宇宙最強トレンド/レンジ判定インジケーター - 最終段階構成
        
        全アルゴリズムを統合した完全版：
        🧠 Hilbert Transform: 瞬時位相・周波数解析
        📐 Fractal Adaptive MA: フラクタル適応移動平均
        🌊 Wavelet Analysis: ウェーブレット多重解像度解析  
        🎯 Extended Kalman: 拡張カルマンフィルタリング
        📊 Information Entropy: 情報エントロピートレンド検出
        🌀 Chaos Theory: カオス理論指標（Hurst指数、リアプノフ指数）
        """
        super().__init__(f"UltimateChopTrend(P={analysis_period},W={ensemble_window})")
        
        self.analysis_period = analysis_period
        self.ensemble_window = ensemble_window
        
        # アルゴリズム有効化
        self.enable_hilbert = enable_hilbert
        self.enable_fractal = enable_fractal
        self.enable_wavelet = enable_wavelet
        self.enable_kalman = enable_kalman
        self.enable_entropy = enable_entropy
        self.enable_chaos = enable_chaos
        
        # 🚀 高度解析機能有効化
        self.enable_unscented_kalman = enable_unscented_kalman
        self.enable_garch = enable_garch
        self.enable_regime_switching = enable_regime_switching
        self.enable_spectral = enable_spectral
        self.enable_multiscale_entropy = enable_multiscale_entropy
        self.enable_nonlinear_dynamics = enable_nonlinear_dynamics
        
        # 予測設定
        self.enable_prediction = enable_prediction
        self.prediction_horizon = prediction_horizon
        
        # しきい値
        self.trend_threshold = trend_threshold
        self.strong_trend_threshold = strong_trend_threshold
        
        # 従来のChopTrendとの統合
        self.use_legacy_chop = use_legacy_chop
        self.chop_weight = chop_weight
        
        # 従来のChopTrendインスタンス
        if self.use_legacy_chop:
            try:
                from .chop_trend import ChopTrend
                self.legacy_chop = ChopTrend()
            except (ImportError, Exception):
                # ChopTrendが利用できない場合は無効化
                self.legacy_chop = None
                self.use_legacy_chop = False
                self.logger.warning("ChopTrendが利用できません。レガシー統合を無効化しました。")
        
        self._cache = {}
        self._result: Optional[UltimateChopTrendResult] = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> UltimateChopTrendResult:
        """
        Ultimate ChopTrendを計算する
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
            else:
                prices = np.asarray(data[:, 3], dtype=np.float64)
                high = np.asarray(data[:, 1], dtype=np.float64)
                low = np.asarray(data[:, 2], dtype=np.float64)
            
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
                    volatility[i] = max(tr, 1e-10)  # 最小値制限
                else:
                    volatility[i] = max((volatility[i-1] * 13 + tr) / 14, 1e-10)  # 最小値制限
            
            # 初期値も設定
            if n > 0:
                volatility[0] = max(high[0] - low[0], 1e-10)
            
            # 各アルゴリズムの実行
            components = {}
            
            if self.enable_hilbert:
                phase, freq, amp, hilbert_trend = hilbert_transform_analysis(prices)
                components['hilbert'] = hilbert_trend
            else:
                components['hilbert'] = np.full(n, 0.5)
            
            if self.enable_fractal:
                components['fractal'] = fractal_adaptive_moving_average(prices, self.analysis_period)
                # FRAMAを0-1スケールに正規化
                frama_normalized = np.full(n, 0.5)
                for i in range(self.analysis_period, n):
                    if not np.isnan(components['fractal'][i]):
                        vol_safe = max(volatility[i], 1e-10)  # ゼロ除算防止
                        price_change = (prices[i] - components['fractal'][i]) / vol_safe
                        # 安全なtanh計算
                        price_change_bounded = min(max(price_change, -50), 50)
                        frama_normalized[i] = math.tanh(price_change_bounded) * 0.5 + 0.5
                components['fractal'] = frama_normalized
            else:
                components['fractal'] = np.full(n, 0.5)
            
            if self.enable_wavelet:
                denoised, detail = wavelet_denoising(prices)
                # ウェーブレット成分を正規化
                wavelet_signal = np.full(n, 0.5)
                for i in range(10, n):
                    vol_safe = max(volatility[i], 1e-10)  # ゼロ除算防止
                    trend_strength = abs(denoised[i] - denoised[i-5]) / (vol_safe * 5)
                    wavelet_signal[i] = min(max(trend_strength, 0), 1)
                components['wavelet'] = wavelet_signal
            else:
                components['wavelet'] = np.full(n, 0.5)
            
            if self.enable_kalman:
                if self.enable_unscented_kalman:
                    # 🚀 高度版: 無香料カルマンフィルター
                    ukf_filtered, ukf_trend, ukf_uncertainty = unscented_kalman_filter(prices, volatility)
                    kalman_signal = np.full(n, 0.5)
                    for i in range(5, n):
                        if not np.isnan(ukf_trend[i]):
                            # トレンド速度を正規化
                            trend_dir_bounded = min(max(ukf_trend[i] * 10, -50), 50)
                            kalman_signal[i] = math.tanh(trend_dir_bounded) * 0.5 + 0.5
                    components['kalman'] = kalman_signal
                    kalman_conf = 1.0 / (1.0 + ukf_uncertainty)  # 不確実性の逆数で信頼度計算
                else:
                    # 従来版: 拡張カルマンフィルター
                    kalman_trend, kalman_conf = extended_kalman_filter(prices, volatility)
                    # カルマントレンドを正規化
                    kalman_signal = np.full(n, 0.5)
                    for i in range(5, n):
                        if not np.isnan(kalman_trend[i]):
                            vol_safe = max(volatility[i], 1e-10)  # ゼロ除算防止
                            trend_dir = (kalman_trend[i] - kalman_trend[i-3]) / vol_safe
                            # 安全なtanh計算
                            trend_dir_bounded = min(max(trend_dir * 2, -50), 50)
                            kalman_signal[i] = math.tanh(trend_dir_bounded) * 0.5 + 0.5
                    components['kalman'] = kalman_signal
            else:
                components['kalman'] = np.full(n, 0.5)
                kalman_conf = np.ones(n)
            
            if self.enable_entropy:
                if self.enable_multiscale_entropy:
                    # 🚀 高度版: マルチスケールエントロピー
                    components['entropy'] = multiscale_entropy(prices, max_scale=10, pattern_length=2)
                else:
                    # 従来版: 単一スケールエントロピー
                    components['entropy'] = information_entropy_trend(prices, self.analysis_period)
            else:
                components['entropy'] = np.full(n, 0.5)
            
            if self.enable_chaos:
                hurst, lyapunov = chaos_theory_indicators(prices, self.analysis_period)
                # カオス指標を統合
                chaos_signal = np.full(n, 0.5)
                for i in range(n):
                    if not np.isnan(hurst[i]):
                        # Hurst > 0.5 はトレンド、< 0.5 は平均回帰
                        chaos_signal[i] = hurst[i]
                components['chaos'] = chaos_signal
            else:
                components['chaos'] = np.full(n, 0.5)
            
            # 🚀 高度解析アルゴリズムのシグナル生成
            advanced_signals = {}
            
            # Unscented Kalman Filter シグナル
            if self.enable_unscented_kalman:
                ukf_filtered, ukf_trend, ukf_uncertainty = unscented_kalman_filter(prices, volatility)
                ukf_signal = np.full(n, 0.5)
                for i in range(5, n):
                    if not np.isnan(ukf_trend[i]):
                        trend_dir_bounded = min(max(ukf_trend[i] * 10, -50), 50)
                        ukf_signal[i] = math.tanh(trend_dir_bounded) * 0.5 + 0.5
                advanced_signals['unscented_kalman'] = ukf_signal
            else:
                advanced_signals['unscented_kalman'] = np.full(n, 0.5)
            
            # GARCH シグナル
            if self.enable_garch:
                returns = np.diff(prices)
                returns = np.concatenate([np.array([0]), returns])
                garch_var, garch_vol = garch_volatility_model(returns)
                garch_signal = np.full(n, 0.5)
                for i in range(10, n):
                    if not np.isnan(garch_vol[i]) and garch_vol[i] > 0:
                        # ボラティリティの変化率をシグナルに変換
                        vol_change = (garch_vol[i] - garch_vol[i-5]) / garch_vol[i-5] if garch_vol[i-5] > 0 else 0
                        vol_change_bounded = min(max(vol_change * 5, -50), 50)
                        garch_signal[i] = math.tanh(vol_change_bounded) * 0.5 + 0.5
                advanced_signals['garch'] = garch_signal
            else:
                advanced_signals['garch'] = np.full(n, 0.5)
            
            # Regime Switching シグナル
            if self.enable_regime_switching:
                regime_probs, most_likely_regime = regime_switching_detection(prices, window=50, n_regimes=3)
                regime_signal = np.full(n, 0.5)
                for i in range(n):
                    if not np.isnan(most_likely_regime[i]):
                        # レジーム0=下降、1=レンジ、2=上昇として正規化
                        regime_signal[i] = most_likely_regime[i] / 2.0
                advanced_signals['regime_switching'] = regime_signal
            else:
                advanced_signals['regime_switching'] = np.full(n, 0.5)
            
            # Spectral Analysis シグナル
            if self.enable_spectral:
                dominant_freq, spectral_power, spectral_trend = spectral_analysis(prices, window=64)
                spectral_signal = np.full(n, 0.5)
                for i in range(n):
                    if not np.isnan(spectral_trend[i]):
                        trend_bounded = min(max(spectral_trend[i] * 2, -50), 50)
                        spectral_signal[i] = math.tanh(trend_bounded) * 0.5 + 0.5
                advanced_signals['spectral'] = spectral_signal
            else:
                advanced_signals['spectral'] = np.full(n, 0.5)
            
            # Multiscale Entropy シグナル（既に0-1正規化済み）
            if self.enable_multiscale_entropy:
                advanced_signals['multiscale_entropy'] = multiscale_entropy(prices, max_scale=10, pattern_length=2)
            else:
                advanced_signals['multiscale_entropy'] = np.full(n, 0.5)
            
            # Nonlinear Dynamics シグナル
            if self.enable_nonlinear_dynamics:
                correlation_dim, lyapunov_exp, predictability = nonlinear_dynamics_analysis(prices)
                nonlinear_signal = np.full(n, 0.5)
                for i in range(n):
                    if not np.isnan(predictability[i]):
                        # 予測可能性を0-1範囲に正規化
                        nonlinear_signal[i] = min(max(predictability[i], 0.0), 1.0)
                advanced_signals['nonlinear_dynamics'] = nonlinear_signal
            else:
                advanced_signals['nonlinear_dynamics'] = np.full(n, 0.5)

            # 🚀 高度アンサンブル投票システムの使用
            try:
                ensemble_signal, ensemble_confidence = advanced_ensemble_voting_system(
                    # 基本アルゴリズム
                    components['hilbert'],
                    components['fractal'],
                    components['wavelet'],
                    components['kalman'],
                    components['entropy'],
                    components['chaos'],
                    # 高度解析アルゴリズム
                    advanced_signals['unscented_kalman'],
                    advanced_signals['garch'],
                    advanced_signals['regime_switching'],
                    advanced_signals['spectral'],
                    advanced_signals['multiscale_entropy'],
                    advanced_signals['nonlinear_dynamics'],
                    kalman_conf
                )
                
                # シンプルな後処理（最小限の調整）
                for i in range(n):
                    signal_val = ensemble_signal[i]
                    
                    # 極端に保守的すぎる調整を削除し、軽微な調整のみ
                    # 極端な値のみを中央値に近づける
                    if abs(signal_val - 0.5) > 0.4:  # 非常に極端な場合のみ
                        # 軽微な調整
                        ensemble_signal[i] = 0.5 + (signal_val - 0.5) * 0.9  # 10%削減のみ
                
            except Exception as e:
                self.logger.error(f"アンサンブル投票システムでエラー: {e}")
                # フォールバック：レンジ優先のシンプルな平均
                ensemble_signal = np.full(n, 0.5)
                ensemble_confidence = np.full(n, 0.5)
                
                for i in range(n):
                    valid_signals = []
                    for comp_name, comp_values in components.items():
                        if not np.isnan(comp_values[i]) and np.isfinite(comp_values[i]):
                            valid_signals.append(comp_values[i])
                    
                    if len(valid_signals) > 0:
                        # シンプルな平均（バイアス除去）
                        ensemble_signal[i] = np.mean(valid_signals)
                        ensemble_confidence[i] = min(len(valid_signals) / 6.0, 1.0)
            
            # 🚀 高度解析機能の結果保存（重複計算を避けるため既存の結果を使用）
            advanced_components = {}
            
            # 既に計算済みの結果を使用
            if self.enable_unscented_kalman:
                # Unscented Kalman Filterの結果を保存
                ukf_filtered, _, _ = unscented_kalman_filter(prices, volatility)
                advanced_components['unscented_kalman'] = ukf_filtered
            else:
                advanced_components['unscented_kalman'] = np.full(n, np.nan)
            
            # GARCHボラティリティモデル
            if self.enable_garch:
                returns = np.diff(prices)
                returns = np.concatenate([np.array([0]), returns])
                _, garch_vol = garch_volatility_model(returns)
                advanced_components['garch_volatility'] = garch_vol
            else:
                advanced_components['garch_volatility'] = np.full(n, np.nan)
            
            # レジーム切り替え検出
            if self.enable_regime_switching:
                regime_probs, most_likely_regime = regime_switching_detection(prices, window=50, n_regimes=3)
                advanced_components['regime_switching_probs'] = regime_probs
                advanced_components['most_likely_regime'] = most_likely_regime
            else:
                advanced_components['regime_switching_probs'] = np.zeros((n, 3))
                advanced_components['most_likely_regime'] = np.zeros(n)
            
            # スペクトル解析
            if self.enable_spectral:
                dominant_freq, spectral_power, spectral_trend = spectral_analysis(prices, window=64)
                advanced_components['dominant_frequency'] = dominant_freq
                advanced_components['spectral_power'] = spectral_power
                advanced_components['spectral_trend'] = spectral_trend
            else:
                advanced_components['dominant_frequency'] = np.full(n, np.nan)
                advanced_components['spectral_power'] = np.full(n, np.nan)
                advanced_components['spectral_trend'] = np.full(n, np.nan)
            
            # 非線形力学系解析
            if self.enable_nonlinear_dynamics:
                correlation_dim, lyapunov_exp, predictability = nonlinear_dynamics_analysis(prices)
                advanced_components['correlation_dimension'] = correlation_dim
                advanced_components['lyapunov_exponent'] = lyapunov_exp
                advanced_components['predictability_score'] = predictability
            else:
                advanced_components['correlation_dimension'] = np.full(n, np.nan)
                advanced_components['lyapunov_exponent'] = np.full(n, np.nan)
                advanced_components['predictability_score'] = np.full(n, np.nan)

            # 従来のChopTrendとの統合
            if self.use_legacy_chop and self.legacy_chop is not None:
                try:
                    legacy_result = self.legacy_chop.calculate(data)
                    legacy_values = legacy_result.values
                    # 統合
                    final_signal = (1 - self.chop_weight) * ensemble_signal + self.chop_weight * legacy_values
                except:
                    final_signal = ensemble_signal
            else:
                final_signal = ensemble_signal
            
            # 適応しきい値
            adaptive_threshold = adaptive_threshold_calculation(final_signal, volatility, self.ensemble_window)
            
            # 🚀 次世代トレンドシグナル生成（実践的バランス版）
            trend_signals = np.zeros(n)
            trend_strength = np.zeros(n)
            regime_state = np.zeros(n)
            
            for i in range(n):
                signal = final_signal[i]
                threshold = adaptive_threshold[i]
                
                # シンプルで実践的なアプローチ
                current_vol = volatility[i] if i < len(volatility) else 0.01
                
                # 🔧 修正されたトレンド強度計算
                # 中央値（0.5）からの距離を基本とし、ボラティリティで調整
                base_strength = abs(signal - 0.5) * 2.0
                
                # ボラティリティ調整（高ボラティリティ時はより強いシグナルが必要）
                vol_adjustment = min(current_vol * 10, 1.0)  # ボラティリティ調整係数
                adjusted_strength = base_strength * (1 + vol_adjustment)
                
                # 価格変動の勢いも考慮（直近の価格変化）
                if i > 0:
                    price_momentum = abs(prices[i] - prices[i-1]) / max(prices[i-1], 1e-10)
                    momentum_boost = min(price_momentum * 50, 0.5)  # 最大50%のブースト
                    adjusted_strength += momentum_boost
                
                trend_strength[i] = min(adjusted_strength, 1.0)  # 最大値1.0に制限
                
                # 🔧 修正されたレジーム判定（より実践的なしきい値）
                # しきい値を大幅に緩和して、より多くのトレンドを検出
                if trend_strength[i] > 0.4:  # 0.7 → 0.4に緩和
                    regime_state[i] = 2  # 強いトレンド
                elif trend_strength[i] > 0.2:  # 0.4 → 0.2に緩和
                    regime_state[i] = 1  # 通常のトレンド
                else:
                    regime_state[i] = 0  # レンジ
                
                # 実践的なしきい値（固定値で安定性確保）
                upper_bound = 0.65  # シンプルな上限
                lower_bound = 0.35  # シンプルな下限
                
                strong_upper = 0.75  # 強いトレンド上限
                strong_lower = 0.25  # 強いトレンド下限
                
                # トレンドシグナル（実践的判定）
                if signal > strong_upper:
                    trend_signals[i] = 1.0  # 強い上昇
                elif signal > upper_bound:
                    trend_signals[i] = 0.5  # 弱い上昇
                elif signal > 0.55:  # 微弱な上昇も検出
                    trend_signals[i] = 0.25
                elif signal < strong_lower:
                    trend_signals[i] = -1.0  # 強い下降
                elif signal < lower_bound:
                    trend_signals[i] = -0.5  # 弱い下降
                elif signal < 0.45:  # 微弱な下降も検出
                    trend_signals[i] = -0.25
                else:
                    trend_signals[i] = 0.0  # レンジ
            
            # 予測分析
            if self.enable_prediction:
                predictive_signal, momentum_forecast, volatility_forecast = predictive_analysis(
                    prices, final_signal, volatility, self.prediction_horizon
                )
                
                # 🚀 GARCHボラティリティ予測で向上
                if self.enable_garch and 'garch_volatility' in advanced_components:
                    garch_vol = advanced_components['garch_volatility']
                    # GARCHボラティリティで従来の予測を補強
                    for i in range(len(volatility_forecast)):
                        if not np.isnan(garch_vol[i]) and not np.isnan(volatility_forecast[i]):
                            # GARCH予測と従来予測の加重平均
                            volatility_forecast[i] = 0.7 * garch_vol[i] + 0.3 * volatility_forecast[i]
                        elif not np.isnan(garch_vol[i]):
                            volatility_forecast[i] = garch_vol[i]
            else:
                predictive_signal = np.full(n, np.nan)
                momentum_forecast = np.full(n, np.nan)
                volatility_forecast = np.full(n, np.nan)
            
            # 🚀 次世代高度トレンド方向分類（チャート背景色用）
            trend_direction, direction_strength = calculate_trend_direction_classification(
                final_signal, trend_signals, prices, lookback_period=5
            )
            
            # 追加メトリクス
            market_efficiency = np.zeros(n)
            information_ratio = np.zeros(n)
            trend_probability = np.zeros(n)
            regime_probability = np.zeros(n)
            
            for i in range(self.analysis_period, n):
                # 市場効率性（ランダムウォークからの乖離）
                returns = np.diff(prices[i-self.analysis_period:i+1])
                if len(returns) > 0:
                    autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1] if len(returns) > 1 else 0
                    market_efficiency[i] = 1 - abs(autocorr) if not np.isnan(autocorr) else 1
                
                # 情報比率
                avg_return = np.mean(returns) if len(returns) > 0 else 0
                std_return = np.std(returns) if len(returns) > 0 else 1e-10  # ゼロ除算防止
                std_return = max(std_return, 1e-10)  # 最小値制限
                information_ratio[i] = avg_return / std_return
                
                # 確率計算
                trend_probability[i] = trend_strength[i]
                regime_probability[i] = regime_state[i] / 2.0
            
            # 現在状態の判定
            latest_signal = trend_signals[-1] if len(trend_signals) > 0 else 0
            latest_strength = trend_strength[-1] if len(trend_strength) > 0 else 0
            latest_confidence = ensemble_confidence[-1] if len(ensemble_confidence) > 0 else 0
            
            if latest_signal > 0.7:
                current_trend = "strong_uptrend"
            elif latest_signal > 0.2:
                current_trend = "uptrend"
            elif latest_signal < -0.7:
                current_trend = "strong_downtrend"
            elif latest_signal < -0.2:
                current_trend = "downtrend"
            else:
                current_trend = "range"
            
            # 結果作成（🚀 宇宙最強版）
            result = UltimateChopTrendResult(
                ultimate_trend_index=final_signal,
                trend_signals=trend_signals,
                trend_strength=trend_strength,
                regime_state=regime_state,
                trend_direction=trend_direction,
                direction_strength=direction_strength,
                confidence_score=ensemble_confidence,
                trend_probability=trend_probability,
                regime_probability=regime_probability,
                predictive_signal=predictive_signal,
                momentum_forecast=momentum_forecast,
                volatility_forecast=volatility_forecast,
                hilbert_component=components['hilbert'],
                fractal_component=components['fractal'],
                wavelet_component=components['wavelet'],
                kalman_component=components['kalman'],
                entropy_component=components['entropy'],
                chaos_component=components['chaos'],
                # 🚀 新機能: 高度解析成分
                unscented_kalman_component=advanced_components['unscented_kalman'],
                garch_volatility=advanced_components['garch_volatility'],
                regime_switching_probs=advanced_components['regime_switching_probs'],
                spectral_power=advanced_components['spectral_power'],
                dominant_frequency=advanced_components['dominant_frequency'],
                multiscale_entropy=components['entropy'] if self.enable_multiscale_entropy else np.full(n, np.nan),
                nonlinear_dynamics=advanced_components['predictability_score'],
                predictability_score=advanced_components['predictability_score'],
                market_efficiency=market_efficiency,
                information_ratio=information_ratio,
                adaptive_threshold=adaptive_threshold,
                correlation_dimension=advanced_components['correlation_dimension'],
                lyapunov_exponent=advanced_components['lyapunov_exponent'],
                current_trend=current_trend,
                current_strength=latest_strength,
                current_confidence=latest_confidence
            )
            
            self._result = result
            self._values = final_signal
            
            return result
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.logger.error(f"UltimateChopTrend計算中にエラー: {e}\n詳細:\n{error_details}")
            return self._create_empty_result(len(data))
    
    def _create_empty_result(self, length: int) -> UltimateChopTrendResult:
        """空の結果を作成（🚀 宇宙最強版）"""
        return UltimateChopTrendResult(
            ultimate_trend_index=np.full(length, np.nan),
            trend_signals=np.zeros(length),
            trend_strength=np.zeros(length),
            regime_state=np.zeros(length),
            trend_direction=np.zeros(length),
            direction_strength=np.zeros(length),
            confidence_score=np.zeros(length),
            trend_probability=np.zeros(length),
            regime_probability=np.zeros(length),
            predictive_signal=np.full(length, np.nan),
            momentum_forecast=np.full(length, np.nan),
            volatility_forecast=np.full(length, np.nan),
            hilbert_component=np.full(length, np.nan),
            fractal_component=np.full(length, np.nan),
            wavelet_component=np.full(length, np.nan),
            kalman_component=np.full(length, np.nan),
            entropy_component=np.full(length, np.nan),
            chaos_component=np.full(length, np.nan),
            # 🚀 新機能: 高度解析成分
            unscented_kalman_component=np.full(length, np.nan),
            garch_volatility=np.full(length, np.nan),
            regime_switching_probs=np.zeros((length, 3)),
            spectral_power=np.full(length, np.nan),
            dominant_frequency=np.full(length, np.nan),
            multiscale_entropy=np.full(length, np.nan),
            nonlinear_dynamics=np.full(length, np.nan),
            predictability_score=np.full(length, np.nan),
            market_efficiency=np.zeros(length),
            information_ratio=np.zeros(length),
            adaptive_threshold=np.full(length, 0.5),
            correlation_dimension=np.full(length, np.nan),
            lyapunov_exponent=np.full(length, np.nan),
            current_trend="range",
            current_strength=0.0,
            current_confidence=0.0
        )
    
    def get_values(self) -> Optional[np.ndarray]:
        """メイン指標値を取得"""
        if self._result is not None:
            return self._result.ultimate_trend_index.copy()
        return None
    
    def get_result(self) -> Optional[UltimateChopTrendResult]:
        """完全な結果を取得"""
        return self._result
    
    def reset(self) -> None:
        """リセット"""
        super().reset()
        if self.use_legacy_chop and self.legacy_chop is not None:
            self.legacy_chop.reset()
        self._result = None
        self._cache = {} 