#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 **ER-Adaptive Unscented Kalman Filter V1.0** 🎯

Efficiency Ratio（効率比）を使用して動的適応する無香料カルマンフィルター

革新的な2段階フィルタリングシステム：
1. Stage 1: HLC3価格を通常のUKFでフィルタリング
2. Stage 2: フィルタ済み価格からERを計算
3. Stage 3: ER値に基づいてUKFパラメータを動的調整して最終フィルタリング

🌟 **主要機能:**
- ER値によるα、β、κパラメータの動的調整
- トレンド/レンジ市場での最適化された応答性
- Numba最適化による高速計算
- 包括的なエラーハンドリング
- ALMAパターンに基づく統一インターフェース

🔬 **適応ロジック:**
- ER > 0.618: 強トレンド → 高感度、低ノイズ
- ER < 0.382: レンジ相場 → 低感度、安定化優先
- 0.382 ≤ ER ≤ 0.618: 中間状態 → バランス調整
"""

from typing import Union, Optional, NamedTuple, Tuple
import numpy as np
import pandas as pd
from numba import jit, njit
import warnings
warnings.filterwarnings('ignore')

try:
    from .indicator import Indicator
    from .price_source import PriceSource
    from .efficiency_ratio import EfficiencyRatio
    from .kalman_filter_unified import unscented_kalman_filter_numba
    from .ehlers_unified_dc import EhlersUnifiedDC
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from indicator import Indicator
    from price_source import PriceSource
    from efficiency_ratio import EfficiencyRatio
    from kalman_filter_unified import unscented_kalman_filter_numba
    from ehlers_unified_dc import EhlersUnifiedDC


class CycleERAdaptiveUKFResult(NamedTuple):
    """Cycle-ER-Adaptive UKF結果"""
    values: np.ndarray                    # 最終フィルタ済み値
    stage1_filtered: np.ndarray          # Stage1フィルタ済み値
    cycle_values: np.ndarray             # Absolute Ultimate Cycle値
    er_values: np.ndarray                # Efficiency Ratio値
    er_trend_signals: np.ndarray         # ERトレンド信号
    adaptive_alpha: np.ndarray           # 動的αパラメータ
    adaptive_beta: np.ndarray            # 動的βパラメータ
    adaptive_kappa: np.ndarray           # 動的κパラメータ
    uncertainty: np.ndarray              # 不確実性推定
    kalman_gains: np.ndarray             # カルマンゲイン
    confidence_scores: np.ndarray        # 信頼度スコア
    current_trend: str                   # 現在のトレンド状態
    current_trend_value: int             # 現在のトレンド値


@njit(fastmath=True, cache=True)
def calculate_cycle_adaptive_parameters(
    cycle_values: np.ndarray,
    er_values: np.ndarray,
    base_alpha: float = 0.001,
    base_beta: float = 2.0,
    base_kappa: float = 0.0,
    # 感度パラメータの範囲（αとκ）
    alpha_min: float = 0.0001,
    alpha_max: float = 0.01,
    kappa_min: float = -1.0,
    kappa_max: float = 3.0,
    # ノイズ調整パラメータの範囲（β）
    beta_min: float = 1.0,
    beta_max: float = 4.0,
    # サイクル基準の閾値（動的に設定）
    cycle_threshold_ratio_high: float = 0.8,  # 長期サイクル閾値（最大値の80%）
    cycle_threshold_ratio_low: float = 0.3    # 短期サイクル閾値（最大値の30%）
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    サイクル値とER値に基づいてUKFパラメータを動的調整
    
    Args:
        cycle_values: Absolute Ultimate Cycleの値配列
        er_values: Efficiency Ratio値の配列
        base_alpha: ベースαパラメータ
        base_beta: ベースβパラメータ
        base_kappa: ベースκパラメータ
        alpha_min/max: αパラメータの最小/最大値
        kappa_min/max: κパラメータの最小/最大値
        beta_min/max: βパラメータの最小/最大値
        cycle_threshold_ratio_high/low: サイクル閾値比率
        
    Returns:
        Tuple[adaptive_alpha, adaptive_beta, adaptive_kappa]
    """
    n = len(cycle_values)
    adaptive_alpha = np.full(n, base_alpha)
    adaptive_beta = np.full(n, base_beta)
    adaptive_kappa = np.full(n, base_kappa)
    
    # サイクル値の動的閾値計算
    valid_cycles = cycle_values[~np.isnan(cycle_values)]
    if len(valid_cycles) > 0:
        cycle_max = np.max(valid_cycles)
        cycle_min = np.min(valid_cycles)
        cycle_range = cycle_max - cycle_min
        
        # 動的閾値設定
        cycle_threshold_high = cycle_min + cycle_range * cycle_threshold_ratio_high
        cycle_threshold_low = cycle_min + cycle_range * cycle_threshold_ratio_low
    else:
        cycle_threshold_high = 30.0
        cycle_threshold_low = 10.0
    
    for i in range(n):
        if np.isnan(cycle_values[i]) or np.isnan(er_values[i]):
            continue
            
        cycle = cycle_values[i]
        er = er_values[i]
        
        # === 1. サイクルベースの基本適応（完全ゼロ除算防止） ===
        MIN_SAFE_VALUE = 1e-10
        
        if cycle >= cycle_threshold_high:
            # 長期サイクル: 低感度、高安定性
            denom = max(cycle_max - cycle_threshold_high, MIN_SAFE_VALUE)
            cycle_factor = (cycle - cycle_threshold_high) / denom
            cycle_factor = max(min(cycle_factor, 1.0), 0.0)  # 0-1範囲に制限
            base_alpha_mult = 0.5 - cycle_factor * 0.3  # α低下（感度低下）
            base_beta_mult = 1.5 + cycle_factor * 1.0   # β増加（安定性向上）
            base_kappa_mult = -0.5 - cycle_factor * 0.3 # κ低下
            
        elif cycle <= cycle_threshold_low:
            # 短期サイクル: 高感度、迅速応答
            denom = max(cycle_threshold_low - cycle_min, MIN_SAFE_VALUE)
            cycle_factor = (cycle_threshold_low - cycle) / denom
            cycle_factor = max(min(cycle_factor, 1.0), 0.0)  # 0-1範囲に制限
            base_alpha_mult = 1.5 + cycle_factor * 1.0  # α増加（感度向上）
            base_beta_mult = 0.8 - cycle_factor * 0.3   # β低下（応答性向上）
            base_kappa_mult = 0.5 + cycle_factor * 0.5  # κ増加
            
        else:
            # 中期サイクル: バランス調整
            mid_point = (cycle_threshold_high + cycle_threshold_low) / 2.0
            half_range = max(cycle_range / 2.0, MIN_SAFE_VALUE)
            mid_factor = abs(cycle - mid_point) / half_range
            mid_factor = max(min(mid_factor, 1.0), 0.0)  # 0-1範囲に制限
            base_alpha_mult = 1.0 + mid_factor * 0.2
            base_beta_mult = 1.0 + mid_factor * 0.1
            base_kappa_mult = 0.0 + mid_factor * 0.1
        
        # === 2. ER値による微調整 ===
        # ER値を0-1範囲に正規化（想定範囲: 0.0-1.0）
        er_normalized = max(0.0, min(1.0, er))
        
        if er_normalized > 0.618:
            # 高効率（強トレンド）: 感度微増、ノイズ低減
            er_factor = (er_normalized - 0.618) / (1.0 - 0.618)
            er_alpha_mult = 1.0 + er_factor * 0.3   # α微増
            er_beta_mult = 1.0 - er_factor * 0.2    # β微減（ノイズ低減）
            er_kappa_mult = 1.0 + er_factor * 0.2   # κ微増
            
        elif er_normalized < 0.382:
            # 低効率（レンジ相場）: 感度低下、安定化
            er_factor = (0.382 - er_normalized) / 0.382
            er_alpha_mult = 1.0 - er_factor * 0.4   # α低下
            er_beta_mult = 1.0 + er_factor * 0.5    # β増加（安定化）
            er_kappa_mult = 1.0 - er_factor * 0.3   # κ低下
            
        else:
            # 中効率: 中立
            er_alpha_mult = 1.0
            er_beta_mult = 1.0
            er_kappa_mult = 1.0
        
        # === 3. 最終パラメータ計算 ===
        # サイクル適応とER微調整の統合
        final_alpha_mult = base_alpha_mult * er_alpha_mult
        final_beta_mult = base_beta_mult * er_beta_mult
        final_kappa_mult = base_kappa_mult * er_kappa_mult
        
        # パラメータ値の計算と境界制限
        target_alpha = base_alpha * final_alpha_mult
        target_beta = base_beta * final_beta_mult
        target_kappa = base_kappa + final_kappa_mult
        
        # 動的範囲制限（サイクルに応じて範囲も調整、完全ゼロ除算防止）
        # 長期サイクルほど保守的な範囲、短期サイクルほど積極的な範囲
        safe_cycle_range = max(cycle_range, MIN_SAFE_VALUE)
        cycle_norm = (cycle - cycle_min) / safe_cycle_range
        cycle_norm = max(min(cycle_norm, 1.0), 0.0)  # 0-1範囲に制限
        
        # αの動的範囲
        dynamic_alpha_min = alpha_min + (alpha_max - alpha_min) * 0.1 * (1.0 - cycle_norm)
        dynamic_alpha_max = alpha_min + (alpha_max - alpha_min) * (0.3 + 0.7 * (1.0 - cycle_norm))
        
        # βの動的範囲
        dynamic_beta_min = beta_min + (beta_max - beta_min) * 0.1 * cycle_norm
        dynamic_beta_max = beta_min + (beta_max - beta_min) * (0.7 + 0.3 * cycle_norm)
        
        # κの動的範囲
        dynamic_kappa_min = kappa_min + (kappa_max - kappa_min) * 0.2 * (1.0 - cycle_norm)
        dynamic_kappa_max = kappa_min + (kappa_max - kappa_min) * (0.4 + 0.6 * (1.0 - cycle_norm))
        
        # 最終制限適用
        adaptive_alpha[i] = max(dynamic_alpha_min, min(dynamic_alpha_max, target_alpha))
        adaptive_beta[i] = max(dynamic_beta_min, min(dynamic_beta_max, target_beta))
        adaptive_kappa[i] = max(dynamic_kappa_min, min(dynamic_kappa_max, target_kappa))
    
    return adaptive_alpha, adaptive_beta, adaptive_kappa


@njit(fastmath=True, cache=True)
def cycle_er_adaptive_unscented_kalman_numba(
    prices: np.ndarray,
    cycle_values: np.ndarray,
    er_values: np.ndarray,
    adaptive_alpha: np.ndarray,
    adaptive_beta: np.ndarray,
    adaptive_kappa: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    サイクル値とER値に基づく動的適応UKF実装（完全ゼロ除算防止版）
    
    Args:
        prices: 価格データ
        cycle_values: Absolute Ultimate Cycleの値
        er_values: Efficiency Ratio値
        adaptive_alpha: 動的αパラメータ
        adaptive_beta: 動的βパラメータ
        adaptive_kappa: 動的κパラメータ
        
    Returns:
        Tuple[filtered_prices, trend_estimate, uncertainty, kalman_gains, confidence_scores]
    """
    n = len(prices)
    if n < 10:
        return (prices.copy(), np.zeros(n), np.ones(n), np.ones(n) * 0.5, np.ones(n))
    
    # 拡張状態空間（6次元）
    # [価格, 速度, 加速度, サイクル適応度, ER適応度, 信頼度指標]
    L = 6
    
    # 初期状態（安全化）
    x = np.zeros(L)
    safe_initial_price = prices[0] if np.isfinite(prices[0]) and prices[0] != 0 else 100.0
    x[0] = safe_initial_price                    # 価格
    x[1] = 0.0                                   # 速度
    x[2] = 0.0                                   # 加速度
    x[3] = cycle_values[0] if not np.isnan(cycle_values[0]) else 20.0  # サイクル適応度
    x[4] = 0.5                                   # ER適応度
    x[5] = 1.0                                   # 信頼度指標
    
    # 初期共分散行列（安全化）
    P = np.eye(L)
    P[0, 0] = 1.0        # 価格の不確実性
    P[1, 1] = 0.1        # 速度の不確実性
    P[2, 2] = 0.01       # 加速度の不確実性
    P[3, 3] = 0.5        # サイクル適応度の不確実性
    P[4, 4] = 0.1        # ER適応度の不確実性
    P[5, 5] = 0.1        # 信頼度の不確実性
    
    # 出力配列
    filtered_prices = np.zeros(n)
    trend_estimates = np.zeros(n)
    uncertainty_estimates = np.zeros(n)
    kalman_gains = np.zeros(n)
    confidence_scores = np.zeros(n)
    
    # 初期値設定
    filtered_prices[0] = safe_initial_price
    trend_estimates[0] = 0.0
    uncertainty_estimates[0] = 1.0
    kalman_gains[0] = 0.5
    confidence_scores[0] = 1.0
    
    # 安全化定数
    MIN_SAFE_VALUE = 1e-10
    MAX_SAFE_VALUE = 1e10
    
    for t in range(1, n):
        # 現在の価格の安全化
        current_price = prices[t] if np.isfinite(prices[t]) and prices[t] != 0 else safe_initial_price
        
        # 現在のサイクル値とER値を取得（安全化）
        current_cycle = cycle_values[t] if not np.isnan(cycle_values[t]) and np.isfinite(cycle_values[t]) else 20.0
        current_cycle = max(min(current_cycle, 120.0), 5.0)
        
        current_er = er_values[t] if not np.isnan(er_values[t]) and np.isfinite(er_values[t]) else 0.5
        current_er = max(min(current_er, 1.0), 0.0)
        
        current_alpha = adaptive_alpha[t] if np.isfinite(adaptive_alpha[t]) else 0.001
        current_beta = adaptive_beta[t] if np.isfinite(adaptive_beta[t]) else 2.0
        current_kappa = adaptive_kappa[t] if np.isfinite(adaptive_kappa[t]) else 0.0
        
        # UKFパラメータ計算（完全ゼロ除算防止）
        # αパラメータの厳格な安全化
        safe_alpha = max(min(current_alpha, 0.1), MIN_SAFE_VALUE)
        safe_beta = max(min(current_beta, 10.0), 0.1)
        safe_kappa = max(min(current_kappa, 5.0), -5.0)
        
        lambda_param = safe_alpha * safe_alpha * (L + safe_kappa) - L
        
        # gammaの計算で負の値を完全防止
        gamma_arg = L + lambda_param
        if gamma_arg <= MIN_SAFE_VALUE:
            gamma_arg = MIN_SAFE_VALUE
        gamma = np.sqrt(gamma_arg)
        gamma = min(gamma, 10.0)  # 上限制限
        
        # シグマポイント重み（完全ゼロ除算防止）
        denom = L + lambda_param
        if abs(denom) < MIN_SAFE_VALUE:
            denom = MIN_SAFE_VALUE if denom >= 0 else -MIN_SAFE_VALUE
        
        # 重みの安全計算
        Wm0 = lambda_param / denom
        Wc0 = lambda_param / denom + (1 - safe_alpha * safe_alpha + safe_beta)
        Wi = 0.5 / denom
        
        # 重みの範囲制限
        Wm0 = max(min(Wm0, 1.0), -1.0)
        Wc0 = max(min(Wc0, 2.0), -1.0)
        Wi = max(min(Wi, 1.0), MIN_SAFE_VALUE)
        
        # 適応的プロセスノイズ（サイクルとER依存）
        base_process_noise = 0.0001
        
        # サイクル依存ノイズファクター（長期サイクルほどノイズ増）
        cycle_norm = max(0.0, min(1.0, (current_cycle - 5.0) / (120.0 - 5.0)))
        cycle_noise_factor = 1.0 + cycle_norm * 1.5  # 長期サイクル→ノイズ増
        
        # ER依存ノイズファクター（低効率→ノイズ増）
        er_noise_factor = 1.0 + (1.0 - current_er) * 1.0
        
        # 統合ノイズファクター
        combined_noise_factor = (cycle_noise_factor + er_noise_factor) / 2.0
        
        Q = np.eye(L) * base_process_noise
        Q[0, 0] = base_process_noise * combined_noise_factor  # 価格ノイズ
        Q[1, 1] = base_process_noise * combined_noise_factor * 0.5  # 速度ノイズ
        Q[2, 2] = base_process_noise * 0.1                   # 加速度ノイズ
        Q[3, 3] = base_process_noise * 0.3                   # サイクル適応度ノイズ
        Q[4, 4] = base_process_noise * 0.5                   # ER適応度ノイズ
        Q[5, 5] = base_process_noise * 0.1                   # 信頼度ノイズ
        
        # 状態遷移関数（非線形、サイクル・ER適応、完全ゼロ除算防止）
        def state_transition(state):
            new_state = np.zeros(L)
            dt = 1.0
            
            # 状態値の安全化
            safe_state = np.zeros(L)
            for i in range(L):
                if np.isfinite(state[i]):
                    safe_state[i] = state[i]
                else:
                    safe_state[i] = x[i]  # フォールバック
            
            # サイクル適応度とER適応度による動的調整（完全安全化）
            safe_cycle = max(min(safe_state[3], 120.0), 5.0)
            safe_er = max(min(safe_state[4], 1.0), 0.0)
            
            # ゼロ除算防止のための分母安全化
            safe_cycle_denom = max(safe_cycle, MIN_SAFE_VALUE)
            cycle_factor = safe_cycle / 120.0  # サイクル正規化（0-1）
            er_factor = safe_er
            
            # 価格更新（サイクル・ER依存の動力学）
            # 長期サイクル→慣性大、短期サイクル→応答性重視
            momentum_factor = 1.0 + cycle_factor * 0.2  # 長期サイクルで慣性増
            er_response = 1.0 + er_factor * 0.1         # ER高で応答性向上
            
            # 速度の安全化（価格に対する相対的制限）
            max_velocity = max(abs(current_price) * 0.3, 1.0)  # 最小値1.0を保証
            safe_velocity = max(min(safe_state[1], max_velocity), -max_velocity)
            
            new_state[0] = safe_state[0] + safe_velocity * dt * momentum_factor * er_response
            
            # 速度更新（サイクル・ER依存の減衰）
            cycle_damping = 0.98 - cycle_factor * 0.08  # 長期サイクル→減衰小
            er_damping = 0.95 - er_factor * 0.1         # ER高→減衰小（トレンド持続）
            combined_damping = max(min((cycle_damping + er_damping) / 2.0, 0.99), 0.5)
            new_state[1] = safe_state[1] * combined_damping + safe_state[2] * dt
            
            # 加速度更新（安全化）
            new_state[2] = safe_state[2] * 0.9
            
            # サイクル適応度更新（現在のサイクル値に向かって緩やかに収束）
            new_state[3] = safe_state[3] * 0.98 + current_cycle * 0.02
            
            # ER適応度更新（現在のERに向かって緩やかに収束）
            new_state[4] = safe_state[4] * 0.95 + current_er * 0.05
            
            # 信頼度更新（サイクル・ER両方を考慮、完全ゼロ除算防止）
            price_denom = max(abs(safe_state[0]), MIN_SAFE_VALUE)
            price_diff = abs(current_price - safe_state[0])
            prediction_accuracy = 1.0 / (1.0 + price_diff / price_denom)
            
            cycle_denom = max(abs(safe_state[3]), MIN_SAFE_VALUE)
            cycle_diff = abs(current_cycle - safe_state[3])
            cycle_stability = 1.0 / (1.0 + cycle_diff / cycle_denom)
            
            new_state[5] = safe_state[5] * 0.9 + (prediction_accuracy * 0.7 + cycle_stability * 0.3) * 0.1
            
            # 結果の範囲制限
            new_state[0] = max(min(new_state[0], MAX_SAFE_VALUE), -MAX_SAFE_VALUE)
            new_state[1] = max(min(new_state[1], max_velocity), -max_velocity)
            new_state[2] = max(min(new_state[2], max_velocity * 0.1), -max_velocity * 0.1)
            new_state[3] = max(min(new_state[3], 120.0), 5.0)
            new_state[4] = max(min(new_state[4], 1.0), 0.0)
            new_state[5] = max(min(new_state[5], 2.0), 0.1)
            
            return new_state
        
        # 数値安定化
        for i in range(L):
            if P[i, i] > 100.0:
                P[i, i] = 100.0
            elif P[i, i] <= 0:
                P[i, i] = 0.01
        
        # シグマポイント生成（簡略化UKF）
        # 平方根分解
        try:
            sqrt_P = np.linalg.cholesky(P + np.eye(L) * 1e-8)
        except:
            sqrt_P = np.zeros((L, L))
            for i in range(L):
                sqrt_P[i, i] = min(np.sqrt(max(P[i, i], 0.01)), 1.0)
        
        # 簡略化シグマポイント（2L+1個）
        sigma_points = np.zeros((2 * L + 1, L))
        sigma_points[0] = x  # 中心点
        
        # シグマポイント生成（安全化）
        for i in range(L):
            # gammaとsqrt_Pの値を制限
            safe_gamma = min(gamma, 10.0)
            sqrt_col = sqrt_P[:, i]
            
            # 異常値を制限
            for j in range(L):
                if abs(sqrt_col[j]) > 10.0:
                    sqrt_col[j] = np.sign(sqrt_col[j]) * 10.0
            
            sigma_points[i + 1] = x + safe_gamma * sqrt_col
            sigma_points[i + 1 + L] = x - safe_gamma * sqrt_col
        
        # シグマポイント伝播
        sigma_points_pred = np.zeros((2 * L + 1, L))
        for i in range(2 * L + 1):
            sigma_points_pred[i] = state_transition(sigma_points[i])
        
        # 予測状態計算
        x_pred = Wm0 * sigma_points_pred[0]
        for i in range(1, 2 * L + 1):
            x_pred += Wi * sigma_points_pred[i]
        
        # 予測共分散計算
        P_pred = Q.copy()
        diff = sigma_points_pred[0] - x_pred
        P_pred += Wc0 * np.outer(diff, diff)
        for i in range(1, 2 * L + 1):
            diff = sigma_points_pred[i] - x_pred
            P_pred += Wi * np.outer(diff, diff)
        
        # 観測更新
        H = np.zeros(L)
        H[0] = 1.0  # 価格のみ観測
        
        # 観測ノイズ（サイクル・ER適応）
        base_obs_noise = 0.001
        cycle_obs_factor = 1.0 + cycle_norm * 0.5        # 長期サイクル→観測ノイズ微増
        er_obs_factor = 1.0 + (1.0 - current_er) * 1.0   # ER低→観測ノイズ増
        R = base_obs_noise * (cycle_obs_factor + er_obs_factor) / 2.0
        
        # カルマンゲイン（完全ゼロ除算防止）
        H_P_pred = np.dot(H, P_pred)
        innovation_cov_raw = np.dot(H_P_pred, H.T) + R
        
        # イノベーション共分散の完全安全化
        if not np.isfinite(innovation_cov_raw) or innovation_cov_raw <= MIN_SAFE_VALUE:
            innovation_cov = MIN_SAFE_VALUE
        else:
            innovation_cov = max(innovation_cov_raw, MIN_SAFE_VALUE)
        
        # カルマンゲイン計算（完全安全化）
        P_pred_HT = np.dot(P_pred, H.T)
        
        # 分母がゼロでないことを保証
        safe_innovation_cov = max(abs(innovation_cov), MIN_SAFE_VALUE)
        K = P_pred_HT / safe_innovation_cov
        
        # カルマンゲインの完全安全化
        for i in range(L):
            if not np.isfinite(K[i]):
                K[i] = 0.5 if i == 0 else 0.0
            else:
                # 範囲制限を厳格化
                K[i] = max(min(K[i], 1.0), -1.0)
        
        # 状態更新（完全安全化）
        innovation_raw = current_price - x_pred[0]
        
        # イノベーションの完全安全化
        if not np.isfinite(innovation_raw):
            innovation = 0.0
        else:
            max_innovation = max(abs(current_price) * 1.0, 1.0)  # 最小値1.0を保証
            innovation = max(min(innovation_raw, max_innovation), -max_innovation)
        
        # 状態更新
        x = x_pred + K * innovation
        
        # 共分散更新の完全安全化
        try:
            H_P_pred_dot = np.dot(H, P_pred)
            K_H_P_pred = np.outer(K, H_P_pred_dot)
            P = P_pred - K_H_P_pred
        except:
            # エラー時はP_predをそのまま使用
            P = P_pred.copy()
        
        # 共分散行列の対角要素安全化
        for i in range(L):
            if not np.isfinite(P[i, i]) or P[i, i] <= 0:
                P[i, i] = 0.01
            elif P[i, i] > 100.0:
                P[i, i] = 100.0
        
        # 状態値の完全安全化
        max_price_deviation = max(abs(current_price) * 1.5, 10.0)
        x[0] = max(min(x[0], max_price_deviation), -max_price_deviation)
        
        max_velocity = max(abs(current_price) * 0.2, 1.0)
        x[1] = max(min(x[1], max_velocity), -max_velocity)
        
        max_acceleration = max(abs(current_price) * 0.05, 0.1)
        x[2] = max(min(x[2], max_acceleration), -max_acceleration)
        
        # サイクル適応度の制限
        x[3] = max(min(x[3], 120.0), 5.0)
        
        # ER適応度の制限
        x[4] = max(min(x[4], 1.0), 0.0)
        
        # 信頼度の制限
        x[5] = max(min(x[5], 2.0), 0.1)
        
        # 結果記録（安全化）
        filtered_prices[t] = x[0] if np.isfinite(x[0]) else current_price
        trend_estimates[t] = x[1] if np.isfinite(x[1]) else 0.0
        uncertainty_estimates[t] = np.sqrt(max(P[0, 0], MIN_SAFE_VALUE))
        kalman_gains[t] = K[0] if np.isfinite(K[0]) else 0.5
        
        # 信頼度スコア（サイクル・ER考慮、完全ゼロ除算防止）
        cycle_denom = max(abs(x[3]), MIN_SAFE_VALUE)
        cycle_diff = abs(current_cycle - x[3])
        cycle_confidence = max(0.0, min(1.0, 1.0 - cycle_diff / cycle_denom))  # サイクル安定性
        
        er_confidence = max(0.0, min(1.0, current_er))                         # ER効率性
        prediction_confidence = max(0.0, min(1.0, x[5]))                       # 予測精度
        
        # 不確実性の完全安全化
        safe_uncertainty = max(MIN_SAFE_VALUE, min(10.0, uncertainty_estimates[t]))
        uncertainty_denom = 1.0 + safe_uncertainty * 10
        uncertainty_confidence = 1.0 / max(uncertainty_denom, MIN_SAFE_VALUE)  # 不確実性
        
        confidence_scores[t] = (
            0.25 * cycle_confidence +
            0.25 * er_confidence +
            0.25 * prediction_confidence +
            0.25 * uncertainty_confidence
        )
    
    return (filtered_prices, trend_estimates, uncertainty_estimates, 
            kalman_gains, confidence_scores)


class CycleERAdaptiveUKF(Indicator):
    """
    Cycle-ER-Adaptive Unscented Kalman Filter
    
    Absolute Ultimate CycleとEfficiency Ratio（効率比）を使用して動的適応する無香料カルマンフィルター
    
    🌟 **3段階フィルタリングシステム:**
    1. HLC3価格を通常のUKFでフィルタリング（Stage1）
    2. フィルタ済み価格からERを計算
    3. 同時にAbsolute Ultimate Cycleを計算
    4. サイクル値とER値に基づいてUKFパラメータを動的調整（Stage2）
    
    🎯 **サイクル基準適応ロジック:**
    - 長期サイクル（上位80%）: 低感度、高安定性、慣性重視
    - 短期サイクル（下位30%）: 高感度、迅速応答、応答性重視
    - 中期サイクル: バランス調整
    
    ⚡ **ER微調整ロジック:**
    - ER > 0.618: 高効率（強トレンド）→ 感度微増、ノイズ低減
    - ER < 0.382: 低効率（レンジ相場）→ 感度低下、安定化
    - 中効率: 中立調整
    
    🔧 **動的パラメータ範囲調整:**
    - 感度パラメータ（α, κ）: サイクルに応じて範囲動的調整
    - ノイズパラメータ（β）: 長期サイクル→保守的、短期サイクル→積極的
    """
    
    def __init__(
        self,
        # UKFパラメータ
        ukf_alpha: float = 0.001,
        ukf_beta: float = 2.0,
        ukf_kappa: float = 0.0,
        # ERパラメータ  
        er_period: int = 14,
        er_smoothing_method: str = 'hma',
        er_slope_index: int = 1,
        er_range_threshold: float = 0.005,
        # サイクルパラメータ
        cycle_part: float = 1.0,
        cycle_max_output: int = 120,
        cycle_min_output: int = 5,
        cycle_period_range: Tuple[int, int] = (5, 120),
        # 適応パラメータ範囲
        alpha_min: float = 0.0001,
        alpha_max: float = 0.01,
        beta_min: float = 1.0,
        beta_max: float = 4.0,
        kappa_min: float = -1.0,
        kappa_max: float = 3.0,
        # サイクル閾値
        cycle_threshold_ratio_high: float = 0.8,
        cycle_threshold_ratio_low: float = 0.3,
        # その他
        volatility_window: int = 10
    ):
        """
        コンストラクタ
        
        Args:
            ukf_alpha: UKFアルファパラメータ
            ukf_beta: UKFベータパラメータ
            ukf_kappa: UKFカッパパラメータ
            er_period: ER計算期間
            er_smoothing_method: ERスムージング方法
            er_slope_index: ERトレンド判定期間
            er_range_threshold: ERレンジ判定閾値
            cycle_part: サイクル部分の倍率
            cycle_max_output: サイクル最大出力値
            cycle_min_output: サイクル最小出力値
            cycle_period_range: サイクル期間の範囲
            alpha_min/max: αパラメータの最小/最大値
            beta_min/max: βパラメータの最小/最大値
            kappa_min/max: κパラメータの最小/最大値
            cycle_threshold_ratio_high/low: サイクル閾値比率
            volatility_window: ボラティリティ計算ウィンドウ
        """
        name = f"CycleERAdaptiveUKF(α={ukf_alpha},β={ukf_beta},κ={ukf_kappa},ER={er_period},Cycle={cycle_period_range})"
        super().__init__(name)
        
        # パラメータ保存
        self.ukf_alpha = ukf_alpha
        self.ukf_beta = ukf_beta
        self.ukf_kappa = ukf_kappa
        self.er_period = er_period
        self.er_smoothing_method = er_smoothing_method
        self.er_slope_index = er_slope_index
        self.er_range_threshold = er_range_threshold
        self.cycle_part = cycle_part
        self.cycle_max_output = cycle_max_output
        self.cycle_min_output = cycle_min_output
        self.cycle_period_range = cycle_period_range
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        self.cycle_threshold_ratio_high = cycle_threshold_ratio_high
        self.cycle_threshold_ratio_low = cycle_threshold_ratio_low
        self.volatility_window = volatility_window
        
        # Absolute Ultimate Cycleインジケーター（元データ用）- EhlersUnifiedDC経由
        self.cycle_indicator = EhlersUnifiedDC(
            detector_type='absolute_ultimate',
            cycle_part=cycle_part,
            max_output=cycle_max_output,
            min_output=cycle_min_output,
            src_type='hlc3',
            period_range=cycle_period_range
        )
        
        # Efficiency Ratioインジケーター（Stage1フィルタ済み価格用）
        self.er_indicator = EfficiencyRatio(
            period=er_period,
            src_type='hlc3',  # Stage1フィルタ済みHLC3価格を使用
            smoothing_method=er_smoothing_method,
            use_dynamic_period=False,  # 固定期間モード
            slope_index=er_slope_index,
            range_threshold=er_range_threshold
        )
        
        # 結果のキャッシュ
        self._result: Optional[CycleERAdaptiveUKFResult] = None
        self._cache_hash: Optional[str] = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> CycleERAdaptiveUKFResult:
        """
        Cycle-ER-Adaptive UKFを計算
        
        Args:
            data: 価格データ
            
        Returns:
            CycleERAdaptiveUKFResult: フィルター結果
        """
        # キャッシュチェック
        current_hash = self._get_data_hash(data)
        if self._cache_hash == current_hash and self._result is not None:
            return self._result
        
        # HLC3価格データの抽出（安全化）
        hlc3_prices = PriceSource.calculate_source(data, 'hlc3')
        n_data = len(hlc3_prices)
        
        # 価格データの安全化
        hlc3_prices = np.where(np.isfinite(hlc3_prices), hlc3_prices, 100.0)
        
        print(f"🔍 デバッグ情報:")
        print(f"  - データ長: {n_data}")
        print(f"  - HLC3価格範囲: {np.nanmin(hlc3_prices):.2f} - {np.nanmax(hlc3_prices):.2f}")
        
        # 最小データ長を10に緩和
        if n_data < 10:
            print(f"⚠️ データ不足: {n_data} < 10")
            return self._create_empty_result(n_data)
        
        # Stage 1: HLC3価格を通常のUKFでフィルタリング（安全化）
        print("  - Stage 1: UKFフィルタリング開始")
        volatility = self._estimate_volatility(hlc3_prices)
        stage1_filtered, _, stage1_uncertainty, stage1_gains, stage1_confidence = unscented_kalman_filter_numba(
            hlc3_prices, volatility, self.ukf_alpha, self.ukf_beta, self.ukf_kappa
        )
        
        # Stage1結果の安全化
        stage1_filtered = np.where(np.isfinite(stage1_filtered), stage1_filtered, hlc3_prices)
        
        # Stage1結果の確認
        stage1_valid = np.sum(~np.isnan(stage1_filtered))
        print(f"  - Stage1有効値: {stage1_valid}/{n_data}")
        
        # Stage 2a: 元データからAbsolute Ultimate Cycleを計算（安全化）
        print("  - Stage 2a: サイクル計算開始")
        cycle_values = self.cycle_indicator.calculate(data)
        
        # サイクル値の安全化
        cycle_values = np.where(np.isfinite(cycle_values), cycle_values, 20.0)
        cycle_values = np.clip(cycle_values, 5.0, 120.0)
        
        cycle_valid = np.sum(~np.isnan(cycle_values))
        print(f"  - サイクル有効値: {cycle_valid}/{n_data}")
        if cycle_valid > 0:
            print(f"  - サイクル値範囲: {np.nanmin(cycle_values):.2f} - {np.nanmax(cycle_values):.2f}")
        
        # Stage 2b: フィルタ済み価格からERを計算（安全化）
        print("  - Stage 2b: ER計算開始")
        
        # DataFrameを作成してER計算用のHLC3として使用
        if isinstance(data, pd.DataFrame):
            # フィルタ済み価格をHLC3として使用するためのダミーDataFrame作成
            filtered_df = data.copy()
            filtered_df['high'] = stage1_filtered
            filtered_df['low'] = stage1_filtered  
            filtered_df['close'] = stage1_filtered
        else:
            # NumPy配列の場合はDataFrameに変換
            filtered_df = pd.DataFrame({
                'high': stage1_filtered,
                'low': stage1_filtered,
                'close': stage1_filtered
            })
        
        # ERを計算
        er_result = self.er_indicator.calculate(filtered_df)
        er_values = er_result.values
        er_trend_signals = er_result.trend_signals
        current_trend = er_result.current_trend
        current_trend_value = er_result.current_trend_value
        
        # ER値の安全化
        er_values = np.where(np.isfinite(er_values), er_values, 0.5)
        er_values = np.clip(er_values, 0.0, 1.0)
        
        er_valid = np.sum(~np.isnan(er_values))
        print(f"  - ER有効値: {er_valid}/{n_data}")
        if er_valid > 0:
            print(f"  - ER値範囲: {np.nanmin(er_values):.3f} - {np.nanmax(er_values):.3f}")
        
        # Stage 3: サイクル値とER値に基づくパラメータ調整（安全化）
        print("  - Stage 3: パラメータ調整開始")
        
        adaptive_alpha, adaptive_beta, adaptive_kappa = calculate_cycle_adaptive_parameters(
            cycle_values,
            er_values,
            self.ukf_alpha,
            self.ukf_beta,
            self.ukf_kappa,
            self.alpha_min,
            self.alpha_max,
            self.kappa_min,
            self.kappa_max,
            self.beta_min,
            self.beta_max,
            self.cycle_threshold_ratio_high,
            self.cycle_threshold_ratio_low
        )
        
        # 適応パラメータの安全化
        adaptive_alpha = np.where(np.isfinite(adaptive_alpha), adaptive_alpha, self.ukf_alpha)
        adaptive_beta = np.where(np.isfinite(adaptive_beta), adaptive_beta, self.ukf_beta)
        adaptive_kappa = np.where(np.isfinite(adaptive_kappa), adaptive_kappa, self.ukf_kappa)
        
        adaptive_alpha = np.clip(adaptive_alpha, self.alpha_min, self.alpha_max)
        adaptive_beta = np.clip(adaptive_beta, self.beta_min, self.beta_max)
        adaptive_kappa = np.clip(adaptive_kappa, self.kappa_min, self.kappa_max)
        
        print(f"  - 適応α範囲: {np.nanmin(adaptive_alpha):.6f} - {np.nanmax(adaptive_alpha):.6f}")
        print(f"  - 適応β範囲: {np.nanmin(adaptive_beta):.3f} - {np.nanmax(adaptive_beta):.3f}")
        
        # Stage 4: 適応UKFによる最終フィルタリング（安全化）
        print("  - Stage 4: 最終フィルタリング開始")
        
        final_filtered, trend_estimates, uncertainty_estimates, kalman_gains, confidence_scores = cycle_er_adaptive_unscented_kalman_numba(
            hlc3_prices,
            cycle_values,
            er_values,
            adaptive_alpha,
            adaptive_beta,
            adaptive_kappa
        )
        
        # 最終結果の安全化
        final_filtered = np.where(np.isfinite(final_filtered), final_filtered, hlc3_prices)
        trend_estimates = np.where(np.isfinite(trend_estimates), trend_estimates, 0.0)
        uncertainty_estimates = np.where(np.isfinite(uncertainty_estimates), uncertainty_estimates, 1.0)
        kalman_gains = np.where(np.isfinite(kalman_gains), kalman_gains, 0.5)
        confidence_scores = np.where(np.isfinite(confidence_scores), confidence_scores, 0.5)
        
        # 範囲制限
        uncertainty_estimates = np.clip(uncertainty_estimates, 0.001, 10.0)
        kalman_gains = np.clip(kalman_gains, 0.0, 1.0)
        confidence_scores = np.clip(confidence_scores, 0.0, 1.0)
        
        final_valid = np.sum(~np.isnan(final_filtered))
        print(f"  - 最終フィルタ有効値: {final_valid}/{n_data}")
        if final_valid > 0:
            print(f"  - 最終フィルタ値範囲: {np.nanmin(final_filtered):.2f} - {np.nanmax(final_filtered):.2f}")
        
        # 結果作成
        result = CycleERAdaptiveUKFResult(
            values=final_filtered,
            stage1_filtered=stage1_filtered,
            cycle_values=cycle_values,
            er_values=er_values,
            er_trend_signals=er_trend_signals,
            adaptive_alpha=adaptive_alpha,
            adaptive_beta=adaptive_beta,
            adaptive_kappa=adaptive_kappa,
            uncertainty=uncertainty_estimates,
            kalman_gains=kalman_gains,
            confidence_scores=confidence_scores,
            current_trend=current_trend,
            current_trend_value=current_trend_value
        )
        
        print("✅ 計算完了")
        
        # キャッシュ更新
        self._result = result
        self._cache_hash = current_hash
        
        return result
    
    def _estimate_volatility(self, prices: np.ndarray) -> np.ndarray:
        """ボラティリティを推定"""
        n = len(prices)
        volatility = np.full(n, 0.01)
        
        if n < self.volatility_window:
            return volatility
        
        for i in range(self.volatility_window, n):
            window_prices = prices[i-self.volatility_window:i]
            if len(window_prices) > 1:
                vol = np.std(window_prices)
                volatility[i] = max(vol, 0.001)
        
        return volatility
    
    def _create_empty_result(self, length: int) -> CycleERAdaptiveUKFResult:
        """空の結果を作成"""
        return CycleERAdaptiveUKFResult(
            values=np.full(length, np.nan),
            stage1_filtered=np.full(length, np.nan),
            cycle_values=np.full(length, np.nan),
            er_values=np.full(length, np.nan),
            er_trend_signals=np.zeros(length, dtype=np.int8),
            adaptive_alpha=np.full(length, self.ukf_alpha),
            adaptive_beta=np.full(length, self.ukf_beta),
            adaptive_kappa=np.full(length, self.ukf_kappa),
            uncertainty=np.full(length, np.nan),
            kalman_gains=np.full(length, np.nan),
            confidence_scores=np.full(length, np.nan),
            current_trend='range',
            current_trend_value=0
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
    
    def get_values(self) -> Optional[np.ndarray]:
        """最終フィルタ済み値を取得"""
        if self._result is not None:
            return self._result.values.copy()
        return None
    
    def get_stage1_filtered(self) -> Optional[np.ndarray]:
        """Stage1フィルタ済み値を取得"""
        if self._result is not None:
            return self._result.stage1_filtered.copy()
        return None
    
    def get_cycle_values(self) -> Optional[np.ndarray]:
        """サイクル値を取得"""
        if self._result is not None:
            return self._result.cycle_values.copy()
        return None
    
    def get_er_values(self) -> Optional[np.ndarray]:
        """ER値を取得"""
        if self._result is not None:
            return self._result.er_values.copy()
        return None
    
    def get_adaptive_parameters(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """適応パラメータを取得"""
        if self._result is not None:
            return (
                self._result.adaptive_alpha.copy(),
                self._result.adaptive_beta.copy(),
                self._result.adaptive_kappa.copy()
            )
        return None
    
    def get_confidence_scores(self) -> Optional[np.ndarray]:
        """信頼度スコアを取得"""
        if self._result is not None:
            return self._result.confidence_scores.copy()
        return None
    
    def get_trend_info(self) -> Tuple[str, int, Optional[np.ndarray]]:
        """トレンド情報を取得"""
        if self._result is not None:
            return (
                self._result.current_trend,
                self._result.current_trend_value,
                self._result.er_trend_signals.copy()
            )
        return ('range', 0, None)
    
    def get_metadata(self) -> dict:
        """メタデータを取得"""
        if self._result is None:
            return {}
        
        valid_cycle = self._result.cycle_values[~np.isnan(self._result.cycle_values)]
        valid_er = self._result.er_values[~np.isnan(self._result.er_values)]
        valid_confidence = self._result.confidence_scores[~np.isnan(self._result.confidence_scores)]
        
        metadata = {
            'indicator_name': self.name,
            'data_points': len(self._result.values),
            'avg_cycle': np.mean(valid_cycle) if len(valid_cycle) > 0 else np.nan,
            'cycle_range': (np.min(valid_cycle), np.max(valid_cycle)) if len(valid_cycle) > 0 else (np.nan, np.nan),
            'avg_er': np.mean(valid_er) if len(valid_er) > 0 else np.nan,
            'er_range': (np.min(valid_er), np.max(valid_er)) if len(valid_er) > 0 else (np.nan, np.nan),
            'avg_confidence': np.mean(valid_confidence) if len(valid_confidence) > 0 else np.nan,
            'trend_state': self._result.current_trend,
            'trend_value': self._result.current_trend_value,
            # サイクル分析
            'long_cycle_ratio': np.mean(valid_cycle >= np.percentile(valid_cycle, 80)) if len(valid_cycle) > 0 else 0.0,
            'short_cycle_ratio': np.mean(valid_cycle <= np.percentile(valid_cycle, 30)) if len(valid_cycle) > 0 else 0.0,
            # ER分析
            'high_efficiency_ratio': np.mean(valid_er > 0.618) if len(valid_er) > 0 else 0.0,
            'low_efficiency_ratio': np.mean(valid_er < 0.382) if len(valid_er) > 0 else 0.0,
            # 適応パラメータ
            'avg_adaptive_alpha': np.mean(self._result.adaptive_alpha),
            'avg_adaptive_beta': np.mean(self._result.adaptive_beta),
            'avg_adaptive_kappa': np.mean(self._result.adaptive_kappa),
            'parameter_variation': {
                'alpha_std': np.std(self._result.adaptive_alpha),
                'beta_std': np.std(self._result.adaptive_beta), 
                'kappa_std': np.std(self._result.adaptive_kappa)
            },
            # 適応範囲情報
            'parameter_ranges': {
                'alpha_range': (self.alpha_min, self.alpha_max),
                'beta_range': (self.beta_min, self.beta_max),
                'kappa_range': (self.kappa_min, self.kappa_max)
            }
        }
        
        return metadata
    
    def reset(self) -> None:
        """状態をリセット"""
        self._result = None
        self._cache_hash = None
        if hasattr(self.cycle_indicator, 'reset'):
            self.cycle_indicator.reset()
        if hasattr(self.er_indicator, 'reset'):
            self.er_indicator.reset()


# デモ機能
def demo_cycle_er_adaptive_ukf():
    """Cycle-ER-Adaptive UKFのデモ"""
    print("🎯 Cycle-ER-Adaptive UKF Demo")
    print("=" * 60)
    
    # サンプルデータ生成
    np.random.seed(42)
    n_periods = 200
    
    # より複雑なトレンド + サイクル + ノイズのある価格データ
    trend = np.linspace(100, 120, n_periods)
    cycle = 5 * np.sin(np.linspace(0, 4 * np.pi, n_periods))  # サイクル成分
    noise = np.random.normal(0, 1, n_periods)
    prices = trend + cycle + noise
    
    # OHLC形式のデータ作成
    data = pd.DataFrame({
        'high': prices + np.abs(np.random.normal(0, 0.5, n_periods)),
        'low': prices - np.abs(np.random.normal(0, 0.5, n_periods)), 
        'close': prices
    })
    
    # Cycle-ER-Adaptive UKF実行
    cycle_er_ukf = CycleERAdaptiveUKF(
        ukf_alpha=0.001,
        ukf_beta=2.0,
        er_period=14,
        er_smoothing_method='hma',
        cycle_part=1.0,
        cycle_max_output=120,
        cycle_min_output=5
    )
    
    result = cycle_er_ukf.calculate(data)
    
    # 結果の表示
    print(f"データ期間: {n_periods}")
    print(f"最終フィルタ値範囲: {np.nanmin(result.values):.2f} - {np.nanmax(result.values):.2f}")
    print(f"平均サイクル値: {np.nanmean(result.cycle_values):.2f}")
    print(f"サイクル値範囲: {np.nanmin(result.cycle_values):.2f} - {np.nanmax(result.cycle_values):.2f}")
    print(f"平均ER値: {np.nanmean(result.er_values):.3f}")
    print(f"現在のトレンド: {result.current_trend}")
    print(f"平均信頼度: {np.nanmean(result.confidence_scores):.3f}")
    
    # 適応パラメータの統計
    print(f"\n🔧 適応パラメータ統計:")
    print(f"α範囲: {np.nanmin(result.adaptive_alpha):.6f} - {np.nanmax(result.adaptive_alpha):.6f}")
    print(f"β範囲: {np.nanmin(result.adaptive_beta):.3f} - {np.nanmax(result.adaptive_beta):.3f}")
    print(f"κ範囲: {np.nanmin(result.adaptive_kappa):.3f} - {np.nanmax(result.adaptive_kappa):.3f}")
    
    # メタデータ表示
    metadata = cycle_er_ukf.get_metadata()
    print("\n📊 メタデータ:")
    for key, value in metadata.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")
    
    print("\n✅ デモ完了")


if __name__ == "__main__":
    demo_cycle_er_adaptive_ukf() 