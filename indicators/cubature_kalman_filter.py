#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 **Cubature Kalman Filter (CKF) - 立方体カルマンフィルター** 🎯

UKFを超える高精度な非線形フィルタリング：
- スフィリカル-ラジアル立方体ルールによる数値積分
- UKFより少ないシグマポイントで高精度
- より安定した数値特性
- 高次モーメントの正確な推定

🌟 **CKFの優位性:**
1. **効率性**: UKFの2L+1個に対し、CKFは2L個のシグマポイント
2. **精度**: 3次モーメントまでUKFと同等、計算コストは低い
3. **安定性**: より良い数値安定性
4. **理論的基盤**: 球面積分ルールに基づく厳密な理論
"""

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional, List, NamedTuple
import numpy as np
import pandas as pd
from numba import jit, njit, types
import traceback

try:
    from .indicator import Indicator
    from .price_source import PriceSource
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from indicator import Indicator
    from price_source import PriceSource


@dataclass
class CKFResult:
    """立方体カルマンフィルターの計算結果"""
    filtered_values: np.ndarray      # フィルター済み価格
    velocity_estimates: np.ndarray   # 速度推定値
    acceleration_estimates: np.ndarray  # 加速度推定値
    uncertainty: np.ndarray          # 推定不確実性（標準偏差）
    kalman_gains: np.ndarray         # カルマンゲイン
    innovations: np.ndarray          # イノベーション（予測誤差）
    cubature_points: np.ndarray      # 最終立方体ポイント
    confidence_scores: np.ndarray    # 信頼度スコア
    raw_values: np.ndarray          # 元の価格データ


@njit(fastmath=True, cache=True)
def generate_cubature_points(mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    スフィリカル-ラジアル立方体ルールによるシグマポイント生成
    
    Args:
        mean: 状態ベクトル
        covariance: 共分散行列
    
    Returns:
        cubature_points: 立方体ポイント行列 (2L x L)
        weights: 重み (全て等しく 1/(2L))
    """
    L = len(mean)  # 状態次元
    n_points = 2 * L  # CKFのポイント数
    
    # 立方体ポイント初期化
    cubature_points = np.zeros((n_points, L))
    weights = np.full(n_points, 1.0 / (2.0 * L))  # 全て等重み
    
    # 共分散行列の平方根計算（Cholesky分解）
    try:
        sqrt_matrix = np.linalg.cholesky(L * covariance)
    except:
        # 数値安定性のための対処
        sqrt_matrix = np.zeros((L, L))
        for i in range(L):
            sqrt_matrix[i, i] = np.sqrt(max(L * covariance[i, i], 1e-8))
    
    # スフィリカル-ラジアル立方体ポイント生成
    for i in range(L):
        # 正方向
        cubature_points[i] = mean + sqrt_matrix[:, i]
        # 負方向
        cubature_points[i + L] = mean - sqrt_matrix[:, i]
    
    return cubature_points, weights


@njit(fastmath=True, cache=True)
def state_transition_function_ckf(state: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """
    CKF用状態遷移関数（UKFと同じモデル）
    
    Args:
        state: 現在の状態ベクトル [価格, 速度, 加速度]
        dt: 時間差分
    
    Returns:
        次の状態ベクトル
    """
    price, velocity, acceleration = state[0], state[1], state[2]
    
    # 非線形動力学モデル
    new_price = price + velocity * dt + 0.5 * acceleration * dt * dt
    
    # 速度の減衰と加速度の影響
    damping_factor = 0.95
    new_velocity = velocity * damping_factor + acceleration * dt
    
    # 加速度の減衰（平均回帰）
    new_acceleration = acceleration * 0.9
    
    return np.array([new_price, new_velocity, new_acceleration])


@njit(fastmath=True, cache=True)
def observation_function_ckf(state: np.ndarray) -> float:
    """
    CKF用観測関数（価格のみ観測）
    
    Args:
        state: 状態ベクトル
    
    Returns:
        観測値（価格）
    """
    return state[0]


@njit(fastmath=True, cache=True)
def calculate_cubature_kalman_filter(
    prices: np.ndarray,
    volatility: np.ndarray,
    process_noise_scale: float = 0.001,
    adaptive_noise: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    立方体カルマンフィルター（CKF）の実装
    
    Args:
        prices: 価格データ
        volatility: ボラティリティ推定値
        process_noise_scale: プロセスノイズのスケール
        adaptive_noise: 適応的ノイズ推定を使用するか
    
    Returns:
        filtered_prices: フィルター済み価格
        velocity_estimates: 速度推定
        acceleration_estimates: 加速度推定
        uncertainty: 不確実性
        kalman_gains: カルマンゲイン
        innovations: イノベーション
        confidence_scores: 信頼度スコア
        final_cubature_points: 最終立方体ポイント
    """
    n = len(prices)
    L = 3  # 状態次元 [価格, 速度, 加速度]
    
    if n < 5:
        # データ不足の場合
        return (prices.copy(), np.zeros(n), np.zeros(n), np.ones(n), 
                np.ones(n) * 0.5, np.zeros(n), np.ones(n), np.zeros((6, L)))
    
    # 結果配列の初期化
    filtered_prices = np.zeros(n)
    velocity_estimates = np.zeros(n)
    acceleration_estimates = np.zeros(n)
    uncertainty = np.zeros(n)
    kalman_gains = np.zeros(n)
    innovations = np.zeros(n)
    confidence_scores = np.zeros(n)
    
    # 初期状態
    x = np.array([prices[0], 0.0, 0.0])  # [価格, 速度, 加速度]
    P = np.array([[1.0, 0.0, 0.0],       # 初期共分散行列
                  [0.0, 0.1, 0.0],
                  [0.0, 0.0, 0.01]])
    
    # プロセスノイズ行列
    Q = np.array([[process_noise_scale, 0.0, 0.0],
                  [0.0, process_noise_scale * 0.1, 0.0],
                  [0.0, 0.0, process_noise_scale * 0.01]])
    
    # 初期値設定
    filtered_prices[0] = prices[0]
    velocity_estimates[0] = 0.0
    acceleration_estimates[0] = 0.0
    uncertainty[0] = np.sqrt(P[0, 0])
    kalman_gains[0] = 0.5
    innovations[0] = 0.0
    confidence_scores[0] = 1.0
    
    # 最終立方体ポイント（デバッグ用）
    final_cubature_points = np.zeros((2 * L, L))
    
    # CKFメインループ
    for t in range(1, n):
        # === 予測ステップ ===
        
        # 1. 立方体ポイント生成
        cubature_points, weights = generate_cubature_points(x, P)
        
        # 2. 各立方体ポイントを状態遷移関数に通す
        n_points = len(cubature_points)
        cubature_points_pred = np.zeros((n_points, L))
        
        for i in range(n_points):
            cubature_points_pred[i] = state_transition_function_ckf(cubature_points[i], 1.0)
        
        # 3. 予測状態の平均計算
        x_pred = np.zeros(L)
        for i in range(n_points):
            x_pred += weights[i] * cubature_points_pred[i]
        
        # 4. 予測共分散計算
        P_pred = Q.copy()  # プロセスノイズを加算
        for i in range(n_points):
            diff = cubature_points_pred[i] - x_pred
            P_pred += weights[i] * np.outer(diff, diff)
        
        # === 更新ステップ ===
        
        # 5. 新しい立方体ポイント生成（予測値用）
        cubature_points_update, weights_update = generate_cubature_points(x_pred, P_pred)
        
        # 6. 観測予測値計算
        z_points = np.zeros(len(cubature_points_update))
        for i in range(len(cubature_points_update)):
            z_points[i] = observation_function_ckf(cubature_points_update[i])
        
        # 7. 予測観測値の平均
        z_pred = 0.0
        for i in range(len(z_points)):
            z_pred += weights_update[i] * z_points[i]
        
        # 8. 観測ノイズの適応的調整
        if adaptive_noise and t >= 5:
            recent_vol = np.mean(volatility[max(0, t-5):t])
            R = max(recent_vol * recent_vol, 0.0001)
        else:
            R = max(volatility[t] * volatility[t], 0.0001)
        
        # 9. イノベーション共分散と相互共分散計算
        S = R  # 観測ノイズ
        Pxz = np.zeros(L)  # 状態-観測間の相互共分散
        
        for i in range(len(z_points)):
            z_diff = z_points[i] - z_pred
            x_diff = cubature_points_update[i] - x_pred
            
            S += weights_update[i] * z_diff * z_diff
            Pxz += weights_update[i] * x_diff * z_diff
        
        # 10. カルマンゲイン計算
        if S > 1e-12:
            K = Pxz / S
        else:
            K = np.array([0.5, 0.0, 0.0])
        
        # 11. 状態更新
        innovation = prices[t] - z_pred
        x = x_pred + K * innovation
        P = P_pred - np.outer(K, K) * S
        
        # 12. 数値安定性の確保
        for i in range(L):
            if P[i, i] < 1e-8:
                P[i, i] = 1e-8
        
        # 状態値の境界制限
        max_velocity = abs(prices[t]) * 0.5
        if abs(x[1]) > max_velocity:
            x[1] = np.sign(x[1]) * max_velocity
            
        max_acceleration = abs(prices[t]) * 0.1
        if abs(x[2]) > max_acceleration:
            x[2] = np.sign(x[2]) * max_acceleration
        
        # 13. 結果の保存
        filtered_prices[t] = x[0]
        velocity_estimates[t] = x[1]
        acceleration_estimates[t] = x[2]
        uncertainty[t] = np.sqrt(max(P[0, 0], 0))
        kalman_gains[t] = K[0]
        innovations[t] = innovation
        
        # 信頼度計算
        confidence_scores[t] = 1.0 / (1.0 + uncertainty[t] * 10.0)
        
        # 最終立方体ポイントの保存
        if t == n - 1:
            final_cubature_points = cubature_points_update.copy()
    
    return (filtered_prices, velocity_estimates, acceleration_estimates, uncertainty,
            kalman_gains, innovations, confidence_scores, final_cubature_points)


@njit(fastmath=True, cache=True)
def estimate_volatility_ckf(prices: np.ndarray, window: int = 10) -> np.ndarray:
    """CKF用ボラティリティ推定"""
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


class CubatureKalmanFilter(Indicator):
    """
    立方体カルマンフィルター（CKF）インジケーター
    
    UKFを超える高精度な非線形フィルタリング：
    - スフィリカル-ラジアル立方体ルール
    - より少ないシグマポイントで高精度
    - 優れた数値安定性
    - 厳密な理論的基盤
    
    🏆 **UKFに対する優位性:**
    - 効率性: 2L個のポイント（UKFは2L+1個）
    - 精度: 同等の精度でより少ない計算
    - 安定性: より良い数値特性
    - 理論: 球面積分に基づく厳密性
    """
    
    def __init__(
        self,
        src_type: str = 'close',           # 価格ソース
        process_noise_scale: float = 0.001,  # プロセスノイズスケール
        volatility_window: int = 10,      # ボラティリティ計算窓
        adaptive_noise: bool = True       # 適応的ノイズ推定
    ):
        """
        コンストラクタ
        
        Args:
            src_type: 価格ソース ('close', 'hlc3', 'hl2', 'ohlc4')
            process_noise_scale: プロセスノイズのスケール
            volatility_window: ボラティリティ計算ウィンドウ
            adaptive_noise: 適応的ノイズ推定を使用するか
        """
        indicator_name = f"CKF(src={src_type}, noise={process_noise_scale})"
        super().__init__(indicator_name)
        
        self.src_type = src_type.lower()
        self.process_noise_scale = process_noise_scale
        self.volatility_window = volatility_window
        self.adaptive_noise = adaptive_noise
        
        # ソースタイプの検証
        valid_sources = ['close', 'hlc3', 'hl2', 'ohlc4', 'high', 'low', 'open',
                        'ukf', 'ukf_close', 'ukf_hlc3', 'ukf_hl2', 'ukf_ohlc4']
        if self.src_type not in valid_sources:
            raise ValueError(f"無効なソースタイプ: {src_type}")
        
        # 結果キャッシュ
        self._result_cache = {}
        self._max_cache_size = 10
        self._cache_keys = []
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> CKFResult:
        """
        立方体カルマンフィルターを計算
        
        Args:
            data: 価格データ（DataFrameまたは配列）
        
        Returns:
            CKFResult: フィルター結果
        """
        try:
            # 価格データの抽出
            src_prices = PriceSource.calculate_source(data, self.src_type)
            
            if len(src_prices) < 5:
                return self._create_empty_result(len(src_prices), src_prices)
            
            # ボラティリティ推定
            volatility = estimate_volatility_ckf(src_prices, self.volatility_window)
            
            # CKF計算
            (filtered_prices, velocity_estimates, acceleration_estimates, uncertainty,
             kalman_gains, innovations, confidence_scores, final_cubature_points) = calculate_cubature_kalman_filter(
                src_prices, volatility, self.process_noise_scale, self.adaptive_noise
            )
            
            # 結果作成
            result = CKFResult(
                filtered_values=filtered_prices.copy(),
                velocity_estimates=velocity_estimates.copy(),
                acceleration_estimates=acceleration_estimates.copy(),
                uncertainty=uncertainty.copy(),
                kalman_gains=kalman_gains.copy(),
                innovations=innovations.copy(),
                cubature_points=final_cubature_points.copy(),
                confidence_scores=confidence_scores.copy(),
                raw_values=src_prices.copy()
            )
            
            self._values = filtered_prices  # 基底クラス用
            return result
            
        except Exception as e:
            self.logger.error(f"CKF計算中にエラー: {str(e)}\n{traceback.format_exc()}")
            
            if isinstance(data, (pd.DataFrame, np.ndarray)) and len(data) > 0:
                src_prices = PriceSource.calculate_source(data, self.src_type)
                return self._create_empty_result(len(src_prices), src_prices)
            else:
                return self._create_empty_result(0, np.array([]))
    
    def _create_empty_result(self, length: int, raw_prices: np.ndarray) -> CKFResult:
        """空の結果を作成"""
        return CKFResult(
            filtered_values=np.full(length, np.nan),
            velocity_estimates=np.full(length, np.nan),
            acceleration_estimates=np.full(length, np.nan),
            uncertainty=np.full(length, np.nan),
            kalman_gains=np.full(length, np.nan),
            innovations=np.full(length, np.nan),
            cubature_points=np.full((6, 3), np.nan),  # 2*3=6立方体ポイント, 3次元状態
            confidence_scores=np.full(length, np.nan),
            raw_values=raw_prices
        )
    
    def get_filter_metadata(self) -> Dict:
        """フィルターのメタデータを取得"""
        return {
            'filter_type': 'Cubature Kalman Filter',
            'src_type': self.src_type,
            'process_noise_scale': self.process_noise_scale,
            'advantages_over_ukf': [
                'より少ないシグマポイント（2L vs 2L+1）',
                'より良い数値安定性',
                '厳密な球面積分ルール',
                '同等精度でより高効率'
            ]
        } 