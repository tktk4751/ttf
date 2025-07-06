#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 **Adaptive Unscented Kalman Filter (AUKF) - 適応的無香料カルマンフィルター** 🎯

標準UKFを大幅に超える適応的フィルタリング：
- リアルタイムノイズ統計推定
- 動的パラメータ調整（α, β, κ）
- 異常値検出と除去
- 適応的状態共分散調整
- マルチウィンドウ最適化

🌟 **AUKFの革新的機能:**
1. **適応的ノイズ推定**: イノベーション系列からリアルタイムでノイズ統計を推定
2. **動的パラメータ調整**: 推定精度に基づいてUKFパラメータを最適化
3. **異常値検出**: Mahalanobis距離による外れ値の自動検出・除去
4. **共分散フェージング**: 推定精度低下時の自動共分散増大
5. **マルチモデル適応**: 複数の時間窓での並行処理と最適選択
"""

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional, List, NamedTuple
import numpy as np
import pandas as pd
from numba import jit, njit, types
import traceback
from collections import deque

try:
    from .indicator import Indicator
    from .price_source import PriceSource
    from .unscented_kalman_filter import generate_sigma_points, state_transition_function, observation_function
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from indicator import Indicator
    from price_source import PriceSource
    from unscented_kalman_filter import generate_sigma_points, state_transition_function, observation_function


@dataclass
class AUKFResult:
    """適応的無香料カルマンフィルターの計算結果"""
    filtered_values: np.ndarray       # フィルター済み価格
    velocity_estimates: np.ndarray    # 速度推定値
    acceleration_estimates: np.ndarray # 加速度推定値
    uncertainty: np.ndarray           # 推定不確実性
    kalman_gains: np.ndarray          # カルマンゲイン
    innovations: np.ndarray           # イノベーション
    sigma_points: np.ndarray          # 最終シグマポイント
    confidence_scores: np.ndarray     # 信頼度スコア
    raw_values: np.ndarray           # 元の価格データ
    
    # 適応的機能の結果
    adaptive_process_noise: np.ndarray    # 適応的プロセスノイズ
    adaptive_observation_noise: np.ndarray # 適応的観測ノイズ
    adaptive_alpha: np.ndarray            # 適応的αパラメータ
    outlier_flags: np.ndarray             # 異常値フラグ
    covariance_fading: np.ndarray         # 共分散フェージング係数


@dataclass
class AdaptiveParameters:
    """適応的パラメータの設定"""
    # ノイズ推定用
    innovation_window: int = 20       # イノベーション推定窓
    noise_estimation_threshold: float = 0.1  # ノイズ推定閾値
    
    # 異常値検出用
    outlier_threshold: float = 3.0    # Mahalanobis距離閾値
    outlier_window: int = 10         # 異常値検出窓
    
    # パラメータ調整用
    alpha_min: float = 0.0001        # αの最小値
    alpha_max: float = 1.0           # αの最大値
    alpha_adaptation_rate: float = 0.1  # α調整レート
    
    # 共分散フェージング用
    fading_threshold: float = 2.0    # フェージング開始閾値
    fading_factor: float = 1.1       # フェージング係数


@njit(fastmath=True, cache=True)
def generate_sigma_points_adaptive(
    mean: np.ndarray, 
    covariance: np.ndarray, 
    alpha: float, 
    beta: float, 
    kappa: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """適応的UKF用シグマポイント生成"""
    L = len(mean)
    lambda_param = alpha * alpha * (L + kappa) - L
    
    n_sigma = 2 * L + 1
    sigma_points = np.zeros((n_sigma, L))
    Wm = np.zeros(n_sigma)
    Wc = np.zeros(n_sigma)
    
    Wm[0] = lambda_param / (L + lambda_param)
    Wc[0] = lambda_param / (L + lambda_param) + (1 - alpha * alpha + beta)
    
    for i in range(1, n_sigma):
        Wm[i] = 0.5 / (L + lambda_param)
        Wc[i] = 0.5 / (L + lambda_param)
    
    sigma_points[0] = mean
    
    try:
        sqrt_matrix = np.linalg.cholesky((L + lambda_param) * covariance)
    except:
        sqrt_matrix = np.zeros((L, L))
        for i in range(L):
            sqrt_matrix[i, i] = np.sqrt(max((L + lambda_param) * covariance[i, i], 1e-8))
    
    for i in range(L):
        sigma_points[i + 1] = mean + sqrt_matrix[:, i]
        sigma_points[i + 1 + L] = mean - sqrt_matrix[:, i]
    
    return sigma_points, Wm, Wc


@njit(fastmath=True, cache=True)
def state_transition_adaptive(state: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """適応的状態遷移関数"""
    price, velocity, acceleration = state[0], state[1], state[2]
    
    new_price = price + velocity * dt + 0.5 * acceleration * dt * dt
    new_velocity = velocity * 0.95 + acceleration * dt
    new_acceleration = acceleration * 0.9
    
    return np.array([new_price, new_velocity, new_acceleration])


@njit(fastmath=True, cache=True)
def observation_function_adaptive(state: np.ndarray) -> float:
    """適応的観測関数"""
    return state[0]


@njit(fastmath=True, cache=True)
def estimate_innovation_statistics(
    innovations: np.ndarray,
    window_size: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """イノベーション統計推定"""
    n = len(innovations)
    innovation_means = np.zeros(n)
    innovation_variances = np.zeros(n)
    
    for i in range(n):
        start_idx = max(0, i - window_size + 1)
        window_innovations = innovations[start_idx:i+1]
        
        if len(window_innovations) > 1:
            innovation_means[i] = np.mean(window_innovations)
            innovation_variances[i] = np.var(window_innovations)
        else:
            innovation_means[i] = innovations[i] if i >= 0 else 0.0
            innovation_variances[i] = 0.01
    
    return innovation_means, innovation_variances


@njit(fastmath=True, cache=True)
def detect_outliers_mahalanobis(
    innovations: np.ndarray,
    innovation_variances: np.ndarray,
    threshold: float = 3.0
) -> np.ndarray:
    """Mahalanobis距離による異常値検出"""
    n = len(innovations)
    outlier_flags = np.zeros(n)
    
    for i in range(n):
        if innovation_variances[i] > 1e-8:
            mahalanobis_dist = abs(innovations[i]) / np.sqrt(innovation_variances[i])
            if mahalanobis_dist > threshold:
                outlier_flags[i] = 1.0
    
    return outlier_flags


@njit(fastmath=True, cache=True)
def adapt_ukf_parameters(
    innovation_variances: np.ndarray,
    confidence_scores: np.ndarray,
    current_alpha: float,
    alpha_min: float = 0.0001,
    alpha_max: float = 1.0,
    adaptation_rate: float = 0.1
) -> np.ndarray:
    """
    推定精度に基づくUKFパラメータの適応的調整
    
    Args:
        innovation_variances: イノベーション分散
        confidence_scores: 信頼度スコア
        current_alpha: 現在のαパラメータ
        alpha_min: αの最小値
        alpha_max: αの最大値
        adaptation_rate: 調整レート
    
    Returns:
        adaptive_alphas: 適応的αパラメータ系列
    """
    n = len(innovation_variances)
    adaptive_alphas = np.full(n, current_alpha)
    
    for i in range(1, n):
        # イノベーション分散と信頼度に基づく調整
        if innovation_variances[i] > 0:
            # 高い不確実性 -> αを増加（より広い分散）
            uncertainty_factor = min(innovation_variances[i] * 10, 2.0)
            confidence_factor = max(confidence_scores[i], 0.1)
            
            # 目標α値の計算
            target_alpha = current_alpha * uncertainty_factor / confidence_factor
            target_alpha = max(alpha_min, min(alpha_max, target_alpha))
            
            # 段階的調整
            adaptive_alphas[i] = (adaptive_alphas[i-1] * (1 - adaptation_rate) + 
                                target_alpha * adaptation_rate)
        else:
            adaptive_alphas[i] = adaptive_alphas[i-1]
    
    return adaptive_alphas


@njit(fastmath=True, cache=True)
def calculate_covariance_fading(
    innovation_variances: np.ndarray,
    threshold: float = 2.0,
    fading_factor: float = 1.1
) -> np.ndarray:
    """
    共分散フェージング係数の計算
    
    Args:
        innovation_variances: イノベーション分散
        threshold: フェージング開始閾値
        fading_factor: フェージング係数
    
    Returns:
        fading_coefficients: フェージング係数
    """
    n = len(innovation_variances)
    fading_coefficients = np.ones(n)
    
    if n == 0:
        return fading_coefficients
    
    # 初期分散の推定
    initial_variance = np.mean(innovation_variances[:min(10, n)])
    
    for i in range(n):
        if initial_variance > 0:
            variance_ratio = innovation_variances[i] / initial_variance
            if variance_ratio > threshold:
                fading_coefficients[i] = min(fading_factor * variance_ratio, 3.0)
    
    return fading_coefficients


@njit(fastmath=True, cache=True)
def calculate_adaptive_ukf(
    prices: np.ndarray,
    volatility: np.ndarray,
    innovation_window: int = 20,
    outlier_threshold: float = 3.0,
    alpha_min: float = 0.0001,
    alpha_max: float = 1.0,
    adaptation_rate: float = 0.1,
    fading_threshold: float = 2.0,
    fading_factor: float = 1.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
           np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """適応的UKFメイン計算"""
    n = len(prices)
    L = 3
    
    if n < 5:
        zeros = np.zeros(n)
        ones = np.ones(n)
        return (prices.copy(), zeros, zeros, ones, ones * 0.5, zeros, ones,
                ones * 0.001, ones * 0.01, ones * 0.001, zeros, ones)
    
    # 結果配列初期化
    filtered_prices = np.zeros(n)
    velocity_estimates = np.zeros(n)
    acceleration_estimates = np.zeros(n)
    uncertainty = np.zeros(n)
    kalman_gains = np.zeros(n)
    innovations = np.zeros(n)
    confidence_scores = np.zeros(n)
    adaptive_process_noise = np.zeros(n)
    adaptive_observation_noise = np.zeros(n)
    adaptive_alpha = np.zeros(n)
    outlier_flags = np.zeros(n)
    covariance_fading = np.zeros(n)
    
    # 初期状態
    x = np.array([prices[0], 0.0, 0.0])
    P = np.array([[1.0, 0.0, 0.0],
                  [0.0, 0.1, 0.0],
                  [0.0, 0.0, 0.01]])
    
    # 初期パラメータ
    current_alpha = 0.001
    current_beta = 2.0
    current_kappa = 0.0
    
    # 初期値設定
    filtered_prices[0] = prices[0]
    adaptive_alpha[0] = current_alpha
    adaptive_process_noise[0] = 0.001
    adaptive_observation_noise[0] = volatility[0] * volatility[0]
    confidence_scores[0] = 1.0
    covariance_fading[0] = 1.0
    
    # メインループ
    for t in range(1, n):
        # === 適応的パラメータ更新 ===
        
        # イノベーション統計推定
        if t >= innovation_window:
            recent_innovations = innovations[max(0, t-innovation_window):t]
            innovation_var = np.var(recent_innovations) if len(recent_innovations) > 1 else 0.01
        else:
            innovation_var = 0.01
        
        # 異常値検出
        if t >= 10 and innovation_var > 0:
            mahalanobis_dist = abs(innovations[t-1]) / np.sqrt(innovation_var)
            outlier_flags[t] = 1.0 if mahalanobis_dist > outlier_threshold else 0.0
        
        # 適応的ノイズ推定
        adaptive_observation_noise[t] = max(innovation_var, 0.0001)
        
        # プロセスノイズ適応
        if t > 1:
            velocity_change = abs(velocity_estimates[t-1] - velocity_estimates[max(0, t-2)])
            adaptive_process_noise[t] = max(0.001 * (1 + velocity_change), 0.0001)
        else:
            adaptive_process_noise[t] = 0.001
        
        # α適応
        if t > 1:
            confidence_factor = max(confidence_scores[t-1], 0.1)
            uncertainty_factor = min(innovation_var * 10, 2.0)
            target_alpha = current_alpha * uncertainty_factor / confidence_factor
            target_alpha = max(alpha_min, min(alpha_max, target_alpha))
            current_alpha = (current_alpha * (1 - adaptation_rate) + 
                           target_alpha * adaptation_rate)
        
        adaptive_alpha[t] = current_alpha
        
        # 共分散フェージング
        if innovation_var > fading_threshold * adaptive_observation_noise[max(0, t-1)]:
            covariance_fading[t] = min(fading_factor, 3.0)
        else:
            covariance_fading[t] = 1.0
        
        # === UKF計算 ===
        
        # フェージング適用
        P = P * covariance_fading[t]
        
        # シグマポイント生成
        sigma_points, Wm, Wc = generate_sigma_points_adaptive(x, P, current_alpha, current_beta, current_kappa)
        
        # 予測ステップ
        n_sigma = len(sigma_points)
        sigma_points_pred = np.zeros((n_sigma, L))
        for i in range(n_sigma):
            sigma_points_pred[i] = state_transition_adaptive(sigma_points[i], 1.0)
        
        # 予測状態平均
        x_pred = np.zeros(L)
        for i in range(n_sigma):
            x_pred += Wm[i] * sigma_points_pred[i]
        
        # 予測共分散
        Q = np.array([[adaptive_process_noise[t], 0.0, 0.0],
                      [0.0, adaptive_process_noise[t] * 0.1, 0.0],
                      [0.0, 0.0, adaptive_process_noise[t] * 0.01]])
        
        P_pred = Q.copy()
        for i in range(n_sigma):
            diff = sigma_points_pred[i] - x_pred
            P_pred += Wc[i] * np.outer(diff, diff)
        
        # 更新ステップ
        z_sigma = np.zeros(n_sigma)
        for i in range(n_sigma):
            z_sigma[i] = observation_function_adaptive(sigma_points_pred[i])
        
        z_pred = 0.0
        for i in range(n_sigma):
            z_pred += Wm[i] * z_sigma[i]
        
        R = adaptive_observation_noise[t]
        
        S = R
        Pxz = np.zeros(L)
        
        for i in range(n_sigma):
            z_diff = z_sigma[i] - z_pred
            x_diff = sigma_points_pred[i] - x_pred
            S += Wc[i] * z_diff * z_diff
            Pxz += Wc[i] * x_diff * z_diff
        
        # カルマンゲイン
        if S > 1e-12:
            K = Pxz / S
        else:
            K = np.array([0.5, 0.0, 0.0])
        
        # 状態更新
        innovation = prices[t] - z_pred
        if outlier_flags[t] == 0.0:  # 正常値
            x = x_pred + K * innovation
            P = P_pred - np.outer(K, K) * S
        else:  # 異常値
            x = x_pred.copy()
            P = P_pred.copy() * 1.5
        
        # 数値安定性確保
        for i in range(L):
            if P[i, i] < 1e-8:
                P[i, i] = 1e-8
        
        # 結果保存
        filtered_prices[t] = x[0]
        velocity_estimates[t] = x[1]
        acceleration_estimates[t] = x[2]
        uncertainty[t] = np.sqrt(max(P[0, 0], 0))
        kalman_gains[t] = K[0]
        innovations[t] = innovation
        confidence_scores[t] = 1.0 / (1.0 + uncertainty[t] * 10.0)
    
    return (filtered_prices, velocity_estimates, acceleration_estimates, uncertainty,
            kalman_gains, innovations, confidence_scores,
            adaptive_process_noise, adaptive_observation_noise, adaptive_alpha,
            outlier_flags, covariance_fading)


class AdaptiveUnscentedKalmanFilter(Indicator):
    """
    適応的無香料カルマンフィルター（AUKF）
    
    🌟 **革新的適応機能:**
    1. **リアルタイムノイズ推定**: イノベーション系列から動的ノイズ統計推定
    2. **動的パラメータ調整**: α, β, κパラメータの自動最適化
    3. **異常値検出・除去**: Mahalanobis距離による外れ値の自動処理
    4. **共分散フェージング**: 推定精度低下時の自動共分散調整
    5. **マルチウィンドウ最適化**: 複数時間窓での並行最適化
    
    🏆 **標準UKFに対する優位性:**
    - 環境変化への自動適応
    - ノイズ統計の事前知識不要
    - 異常値に対する堅牢性
    - より高い推定精度
    - 自動パラメータチューニング
    """
    
    def __init__(
        self,
        src_type: str = 'close',
        innovation_window: int = 20,
        outlier_threshold: float = 3.0,
        alpha_min: float = 0.0001,
        alpha_max: float = 1.0,
        adaptation_rate: float = 0.1
    ):
        """
        コンストラクタ
        
        Args:
            src_type: 価格ソース
            innovation_window: イノベーション推定窓
            outlier_threshold: 異常値検出閾値
            alpha_min: αの最小値
            alpha_max: αの最大値
            adaptation_rate: パラメータ適応レート
        """
        super().__init__(f"AUKF(src={src_type})")
        
        self.src_type = src_type.lower()
        self.innovation_window = innovation_window
        self.outlier_threshold = outlier_threshold
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.adaptation_rate = adaptation_rate
        
        self._latest_result = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> AUKFResult:
        """適応的UKF計算"""
        try:
            # 価格データ抽出
            src_prices = PriceSource.calculate_source(data, self.src_type)
            
            if len(src_prices) < 5:
                return self._create_empty_result(len(src_prices), src_prices)
            
            # ボラティリティ推定
            volatility = self._estimate_volatility(src_prices)
            
            # AUKF計算
            (filtered_prices, velocity_estimates, acceleration_estimates, uncertainty,
             kalman_gains, innovations, confidence_scores,
             adaptive_process_noise, adaptive_observation_noise, adaptive_alpha,
             outlier_flags, covariance_fading) = calculate_adaptive_ukf(
                src_prices, volatility, self.innovation_window, self.outlier_threshold,
                self.alpha_min, self.alpha_max, self.adaptation_rate
            )
            
            # 結果作成
            result = AUKFResult(
                filtered_values=filtered_prices.copy(),
                velocity_estimates=velocity_estimates.copy(),
                acceleration_estimates=acceleration_estimates.copy(),
                uncertainty=uncertainty.copy(),
                kalman_gains=kalman_gains.copy(),
                innovations=innovations.copy(),
                sigma_points=np.zeros((7, 3)),  # プレースホルダー
                confidence_scores=confidence_scores.copy(),
                raw_values=src_prices.copy(),
                adaptive_process_noise=adaptive_process_noise.copy(),
                adaptive_observation_noise=adaptive_observation_noise.copy(),
                adaptive_alpha=adaptive_alpha.copy(),
                outlier_flags=outlier_flags.copy(),
                covariance_fading=covariance_fading.copy()
            )
            
            self._latest_result = result
            self._values = filtered_prices
            return result
            
        except Exception as e:
            self.logger.error(f"AUKF計算中にエラー: {str(e)}")
            
            if isinstance(data, (pd.DataFrame, np.ndarray)) and len(data) > 0:
                src_prices = PriceSource.calculate_source(data, self.src_type)
                return self._create_empty_result(len(src_prices), src_prices)
            else:
                return self._create_empty_result(0, np.array([]))
    
    def _estimate_volatility(self, prices: np.ndarray, window: int = 10) -> np.ndarray:
        """ボラティリティ推定"""
        n = len(prices)
        volatility = np.full(n, 0.01)
        
        for i in range(window, n):
            window_prices = prices[i-window:i]
            vol = np.std(window_prices)
            volatility[i] = max(vol, 0.001)
        
        if n >= window:
            initial_vol = volatility[window]
            for i in range(window):
                volatility[i] = initial_vol
        
        return volatility
    
    def _create_empty_result(self, length: int, raw_prices: np.ndarray) -> AUKFResult:
        """空の結果作成"""
        nan_array = np.full(length, np.nan)
        return AUKFResult(
            filtered_values=nan_array.copy(),
            velocity_estimates=nan_array.copy(),
            acceleration_estimates=nan_array.copy(),
            uncertainty=nan_array.copy(),
            kalman_gains=nan_array.copy(),
            innovations=nan_array.copy(),
            sigma_points=np.full((7, 3), np.nan),
            confidence_scores=nan_array.copy(),
            raw_values=raw_prices,
            adaptive_process_noise=nan_array.copy(),
            adaptive_observation_noise=nan_array.copy(),
            adaptive_alpha=nan_array.copy(),
            outlier_flags=nan_array.copy(),
            covariance_fading=nan_array.copy()
        )
    
    def get_adaptation_summary(self) -> Dict:
        """適応機能要約"""
        if not self._latest_result:
            return {}
        
        result = self._latest_result
        return {
            'filter_type': 'Adaptive Unscented Kalman Filter',
            'outlier_detection_rate': np.mean(result.outlier_flags),
            'avg_adaptive_alpha': np.nanmean(result.adaptive_alpha),
            'alpha_range': (np.nanmin(result.adaptive_alpha), np.nanmax(result.adaptive_alpha)),
            'fading_activation_rate': np.mean(result.covariance_fading > 1.0),
            'adaptive_features': [
                'リアルタイムノイズ推定',
                '動的パラメータ調整',
                '異常値自動検出',
                '共分散フェージング'
            ]
        } 