#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 **Unscented Kalman Filter (UKF) - 無香料カルマンフィルター** 🎯

正確な無香料カルマンフィルターの実装：
- シグマポイント法による非線形システムの状態推定
- ヤコビアン計算不要で高精度な非線形フィルタリング
- 適応的ノイズ推定機能
- 複数の価格ソース対応
- 動的期間調整機能

🌟 **UKFの特徴:**
1. **シグマポイント生成**: 確率分布を代表する点群を生成
2. **非線形伝播**: 各シグマポイントを非線形関数で変換
3. **統計モーメント再構築**: 重み付き平均と共分散を計算
4. **高精度推定**: 3次のモーメントまで正確に推定
"""

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional, List, NamedTuple
import numpy as np
import pandas as pd
from numba import jit, njit, types
import traceback

try:
    from ..indicator import Indicator
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from indicator import Indicator

# PriceSourceは遅延インポートで循環インポートを回避
PriceSource = None


@dataclass
class UKFResult:
    """無香料カルマンフィルターの計算結果"""
    filtered_values: np.ndarray      # フィルター済み価格
    velocity_estimates: np.ndarray   # 速度推定値
    acceleration_estimates: np.ndarray  # 加速度推定値
    uncertainty: np.ndarray          # 推定不確実性（標準偏差）
    kalman_gains: np.ndarray         # カルマンゲイン
    innovations: np.ndarray          # イノベーション（予測誤差）
    sigma_points: np.ndarray         # 最終シグマポイント（デバッグ用）
    confidence_scores: np.ndarray    # 信頼度スコア
    raw_values: np.ndarray          # 元の価格データ

# Alias for compatibility
UnscentedKalmanResult = UKFResult


@njit(fastmath=True, cache=True)
def generate_sigma_points(
    mean: np.ndarray, 
    covariance: np.ndarray, 
    alpha: float, 
    beta: float, 
    kappa: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    UKF用シグマポイントを生成（修正版）
    
    Args:
        mean: 状態ベクトル
        covariance: 共分散行列
        alpha: シグマポイントの分散度合い
        beta: 高次モーメント情報（ガウス分布の場合は2.0）
        kappa: 二次パラメータ
    
    Returns:
        sigma_points: シグマポイント行列
        Wm: 平均計算用重み
        Wc: 共分散計算用重み
    """
    L = len(mean)  # 状態次元
    lambda_param = alpha * alpha * (L + kappa) - L
    
    # シグマポイント数: 2L + 1
    n_sigma = 2 * L + 1
    sigma_points = np.zeros((n_sigma, L))
    
    # 重み計算（修正版）
    Wm = np.zeros(n_sigma)
    Wc = np.zeros(n_sigma)
    
    # 中心点の重み
    Wm[0] = lambda_param / (L + lambda_param)
    Wc[0] = lambda_param / (L + lambda_param) + (1 - alpha * alpha + beta)
    
    # 周辺点の重み
    weight_peripheral = 0.5 / (L + lambda_param)
    for i in range(1, n_sigma):
        Wm[i] = weight_peripheral
        Wc[i] = weight_peripheral
    
    # 中心シグマポイント
    sigma_points[0] = mean
    
    # 共分散行列の数値安定化
    # 対角要素に小さな値を追加して正定値性を保証
    stabilized_cov = covariance.copy()
    for i in range(L):
        stabilized_cov[i, i] += 1e-9
    
    # スケーリング係数
    scale = L + lambda_param
    scaled_cov = scale * stabilized_cov
    
    # 共分散行列の平方根計算（改良版）
    try:
        sqrt_matrix = np.linalg.cholesky(scaled_cov)
    except:
        # フォールバック: 固有値分解
        try:
            eigenvals, eigenvecs = np.linalg.eigh(scaled_cov)
            # 負の固有値を修正
            eigenvals = np.maximum(eigenvals, 1e-8)
            sqrt_matrix = eigenvecs @ np.diag(np.sqrt(eigenvals))
        except:
            # 最終フォールバック: 対角行列
            sqrt_matrix = np.zeros((L, L))
            for i in range(L):
                sqrt_matrix[i, i] = np.sqrt(max(scaled_cov[i, i], 1e-8))
    
    # 正負のシグマポイント生成
    for i in range(L):
        sigma_points[i + 1] = mean + sqrt_matrix[:, i]
        sigma_points[i + 1 + L] = mean - sqrt_matrix[:, i]
    
    return sigma_points, Wm, Wc


@njit(fastmath=True, cache=True)
def state_transition_function(state: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """
    状態遷移関数（非線形）
    
    状態ベクトル: [価格, 速度, 加速度]
    
    Args:
        state: 現在の状態ベクトル
        dt: 時間差分
    
    Returns:
        次の状態ベクトル
    """
    price, velocity, acceleration = state[0], state[1], state[2]
    
    # 非線形動力学モデル
    # 価格 = 価格 + 速度*dt + 0.5*加速度*dt^2 + 非線形項
    new_price = price + velocity * dt + 0.5 * acceleration * dt * dt
    
    # 速度の減衰と加速度の影響
    damping_factor = 0.95  # 減衰係数
    new_velocity = velocity * damping_factor + acceleration * dt
    
    # 加速度の減衰（平均回帰）
    new_acceleration = acceleration * 0.9
    
    return np.array([new_price, new_velocity, new_acceleration])


@njit(fastmath=True, cache=True)
def observation_function(state: np.ndarray) -> float:
    """
    観測関数（価格のみ観測）
    
    Args:
        state: 状態ベクトル
    
    Returns:
        観測値（価格）
    """
    return state[0]  # 価格のみを観測


@njit(fastmath=True, cache=True)
def calculate_unscented_kalman_filter(
    prices: np.ndarray,
    volatility: np.ndarray,
    alpha: float = 0.1,
    beta: float = 2.0,
    kappa: float = 0.0,
    process_noise_scale: float = 0.01,
    adaptive_noise: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    正確な無香料カルマンフィルターの実装
    
    Args:
        prices: 価格データ
        volatility: ボラティリティ推定値
        alpha: UKFパラメータ（推奨値: 0.001）
        beta: UKFパラメータ（ガウス分布では2.0）
        kappa: UKFパラメータ（通常0.0）
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
        final_sigma_points: 最終シグマポイント
    """
    n = len(prices)
    L = 3  # 状態次元 [価格, 速度, 加速度]
    
    if n < 5:
        # データ不足の場合 - 生の価格をそのまま返す
        uncertainty_vals = np.full(n, 0.1)
        confidence_vals = np.full(n, 0.5)
        kalman_gains_vals = np.full(n, 0.1)
        return (prices.copy(), np.zeros(n), np.zeros(n), uncertainty_vals, 
                kalman_gains_vals, np.zeros(n), confidence_vals, np.zeros((2 * L + 1, L)))
    
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
    
    # 最終シグマポイント（デバッグ用）
    final_sigma_points = np.zeros((2 * L + 1, L))
    
    # UKFメインループ
    for t in range(1, n):
        # === 予測ステップ ===
        
        # 1. シグマポイント生成
        sigma_points, Wm, Wc = generate_sigma_points(x, P, alpha, beta, kappa)
        
        # 2. 各シグマポイントを状態遷移関数に通す
        n_sigma = len(sigma_points)
        sigma_points_pred = np.zeros((n_sigma, L))
        
        for i in range(n_sigma):
            sigma_points_pred[i] = state_transition_function(sigma_points[i], 1.0)
        
        # 3. 予測状態の平均計算
        x_pred = np.zeros(L)
        for i in range(n_sigma):
            x_pred += Wm[i] * sigma_points_pred[i]
        
        # 4. 予測共分散計算
        P_pred = Q.copy()  # プロセスノイズを加算
        for i in range(n_sigma):
            diff = sigma_points_pred[i] - x_pred
            P_pred += Wc[i] * np.outer(diff, diff)
        
        # === 更新ステップ ===
        
        # 5. 観測シグマポイント計算
        z_sigma = np.zeros(n_sigma)
        for i in range(n_sigma):
            z_sigma[i] = observation_function(sigma_points_pred[i])
        
        # 6. 予測観測値の平均
        z_pred = 0.0
        for i in range(n_sigma):
            z_pred += Wm[i] * z_sigma[i]
        
        # 7. 観測ノイズの適応的調整
        if adaptive_noise and t >= 5:
            # 最近のボラティリティを使用
            recent_vol = np.mean(volatility[max(0, t-5):t])
            R = max(recent_vol * recent_vol, 0.0001)
        else:
            R = max(volatility[t] * volatility[t], 0.0001)
        
        # 8. イノベーション共分散と相互共分散計算
        S = R  # 観測ノイズ
        Pxz = np.zeros(L)  # 状態-観測間の相互共分散
        
        for i in range(n_sigma):
            z_diff = z_sigma[i] - z_pred
            x_diff = sigma_points_pred[i] - x_pred
            
            S += Wc[i] * z_diff * z_diff
            Pxz += Wc[i] * x_diff * z_diff
        
        # 9. カルマンゲイン計算
        if S > 1e-12:
            K = Pxz / S
        else:
            K = np.array([0.5, 0.0, 0.0])
        
        # 10. 状態更新
        innovation = prices[t] - z_pred
        x = x_pred + K * innovation
        P = P_pred - np.outer(K, K) * S
        
        # 11. 数値安定性の確保
        # 共分散行列の対角要素が負にならないよう調整
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
        
        # 12. 結果の保存
        filtered_prices[t] = x[0]
        velocity_estimates[t] = x[1]
        acceleration_estimates[t] = x[2]
        uncertainty[t] = np.sqrt(max(P[0, 0], 0))
        kalman_gains[t] = K[0]
        innovations[t] = innovation
        
        # 信頼度計算（不確実性の逆数ベース）
        confidence_scores[t] = 1.0 / (1.0 + uncertainty[t] * 10.0)
        
        # 最終シグマポイントの保存（最新のもの）
        if t == n - 1:
            final_sigma_points = sigma_points_pred.copy()
    
    return (filtered_prices, velocity_estimates, acceleration_estimates, uncertainty,
            kalman_gains, innovations, confidence_scores, final_sigma_points)


@njit(fastmath=True, cache=True)
def estimate_volatility(prices: np.ndarray, window: int = 10) -> np.ndarray:
    """ボラティリティ推定"""
    n = len(prices)
    volatility = np.full(n, 0.01)  # デフォルト値
    
    for i in range(window, n):
        window_prices = prices[i-window:i]
        vol = np.std(window_prices)
        volatility[i] = max(vol, 0.001)  # 最小値設定
    
    # 初期値の補完
    if n >= window:
        initial_vol = volatility[window]
        for i in range(window):
            volatility[i] = initial_vol
    
    return volatility


class UnscentedKalmanFilter(Indicator):
    """
    無香料カルマンフィルター（UKF）インジケーター
    
    正確なUKF実装によるシグマポイント法ベースの非線形状態推定：
    - シグマポイント生成と伝播
    - 適応的ノイズ推定
    - 複数の価格ソース対応
    - 高精度な非線形フィルタリング
    
    特徴:
    - ヤコビアン計算不要
    - 3次モーメントまで正確
    - 動的パラメータ調整
    """
    
    def __init__(
        self,
        src_type: str = 'close',           # 価格ソース
        alpha: float = 0.1,               # UKFアルファパラメータ（修正: より大きな値）
        beta: float = 2.0,                # UKFベータパラメータ
        kappa: float = 0.0,               # UKFカッパパラメータ
        process_noise_scale: float = 0.01,  # プロセスノイズスケール（修正: より大きな値）
        volatility_window: int = 10,      # ボラティリティ計算窓
        adaptive_noise: bool = True       # 適応的ノイズ推定
    ):
        """
        コンストラクタ
        
        Args:
            src_type: 価格ソース ('close', 'hlc3', 'hl2', 'ohlc4')
            alpha: UKFアルファパラメータ（推奨: 0.001）
            beta: UKFベータパラメータ（ガウス分布: 2.0）
            kappa: UKFカッパパラメータ（通常: 0.0）
            process_noise_scale: プロセスノイズのスケール
            volatility_window: ボラティリティ計算ウィンドウ
            adaptive_noise: 適応的ノイズ推定を使用するか
        """
        # 指標名の作成
        indicator_name = f"UKF(src={src_type}, α={alpha}, β={beta}, κ={kappa})"
        super().__init__(indicator_name)
        
        # パラメータ保存
        self.src_type = src_type.lower()
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.process_noise_scale = process_noise_scale
        self.volatility_window = volatility_window
        self.adaptive_noise = adaptive_noise
        
        # ソースタイプの検証
        valid_sources = ['close', 'hlc3', 'hl2', 'ohlc4', 'high', 'low', 'open', 'oc2']
        if self.src_type not in valid_sources:
            raise ValueError(f"無効なソースタイプ: {src_type}。有効なオプション: {', '.join(valid_sources)}")
        
        # 結果キャッシュ
        self._result_cache = {}
        self._max_cache_size = 10
        self._cache_keys = []
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算"""
        try:
            if isinstance(data, pd.DataFrame):
                length = len(data)
                if length > 0:
                    first_val = float(data.iloc[0].get('close', data.iloc[0, -1]))
                    last_val = float(data.iloc[-1].get('close', data.iloc[-1, -1]))
                else:
                    first_val = last_val = 0.0
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
            
            params_sig = f"{self.alpha}_{self.beta}_{self.kappa}_{self.src_type}_{self.process_noise_scale}"
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(params_sig)}"
            
        except Exception:
            return f"{id(data)}_{self.alpha}_{self.beta}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> UKFResult:
        """
        無香料カルマンフィルターを計算
        
        Args:
            data: 価格データ（DataFrameまたは配列）
        
        Returns:
            UKFResult: フィルター結果
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            
            if data_hash in self._result_cache:
                if data_hash in self._cache_keys:
                    self._cache_keys.remove(data_hash)
                self._cache_keys.append(data_hash)
                cached_result = self._result_cache[data_hash]
                return UKFResult(
                    filtered_values=cached_result.filtered_values.copy(),
                    velocity_estimates=cached_result.velocity_estimates.copy(),
                    acceleration_estimates=cached_result.acceleration_estimates.copy(),
                    uncertainty=cached_result.uncertainty.copy(),
                    kalman_gains=cached_result.kalman_gains.copy(),
                    innovations=cached_result.innovations.copy(),
                    sigma_points=cached_result.sigma_points.copy(),
                    confidence_scores=cached_result.confidence_scores.copy(),
                    raw_values=cached_result.raw_values.copy()
                )
            
            # PriceSourceを遅延インポート
            global PriceSource
            if PriceSource is None:
                try:
                    from ..price_source import PriceSource
                except ImportError:
                    try:
                        from indicators.price_source import PriceSource
                    except ImportError:
                        # 基本的な価格抽出のフォールバック
                        if isinstance(data, pd.DataFrame):
                            if self.src_type == 'close':
                                src_prices = data['close'].values
                            elif self.src_type == 'hlc3':
                                src_prices = ((data['high'] + data['low'] + data['close']) / 3.0).values
                            else:
                                src_prices = data['close'].values
                        else:
                            src_prices = data[:, 3] if data.ndim > 1 else data
                        
                        if len(src_prices) < 5:
                            return self._create_empty_result(len(src_prices), src_prices)
                        
                        # ボラティリティ推定に直接進む
                        volatility = estimate_volatility(src_prices, self.volatility_window)
                        
                        # UKF計算に直接進む
                        (filtered_prices, velocity_estimates, acceleration_estimates, uncertainty,
                         kalman_gains, innovations, confidence_scores, final_sigma_points) = calculate_unscented_kalman_filter(
                            src_prices, volatility, self.alpha, self.beta, self.kappa,
                            self.process_noise_scale, self.adaptive_noise
                        )
                        
                        # 結果作成
                        result = UKFResult(
                            filtered_values=filtered_prices.copy(),
                            velocity_estimates=velocity_estimates.copy(),
                            acceleration_estimates=acceleration_estimates.copy(),
                            uncertainty=uncertainty.copy(),
                            kalman_gains=kalman_gains.copy(),
                            innovations=innovations.copy(),
                            sigma_points=final_sigma_points.copy(),
                            confidence_scores=confidence_scores.copy(),
                            raw_values=src_prices.copy()
                        )
                        
                        self._values = filtered_prices  # 基底クラス用
                        return result
            
            # 価格データの抽出
            src_prices = PriceSource.calculate_source(data, self.src_type)
            
            if len(src_prices) < 5:
                return self._create_empty_result(len(src_prices), src_prices)
            
            # ボラティリティ推定
            volatility = estimate_volatility(src_prices, self.volatility_window)
            
            # UKF計算
            (filtered_prices, velocity_estimates, acceleration_estimates, uncertainty,
             kalman_gains, innovations, confidence_scores, final_sigma_points) = calculate_unscented_kalman_filter(
                src_prices, volatility, self.alpha, self.beta, self.kappa,
                self.process_noise_scale, self.adaptive_noise
            )
            
            # 結果作成
            result = UKFResult(
                filtered_values=filtered_prices.copy(),
                velocity_estimates=velocity_estimates.copy(),
                acceleration_estimates=acceleration_estimates.copy(),
                uncertainty=uncertainty.copy(),
                kalman_gains=kalman_gains.copy(),
                innovations=innovations.copy(),
                sigma_points=final_sigma_points.copy(),
                confidence_scores=confidence_scores.copy(),
                raw_values=src_prices.copy()
            )
            
            # キャッシュ管理
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            self._values = filtered_prices  # 基底クラス用
            return result
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"UKF計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は空の結果を返す
            if isinstance(data, (pd.DataFrame, np.ndarray)) and len(data) > 0:
                src_prices = PriceSource.calculate_source(data, self.src_type)
                return self._create_empty_result(len(src_prices), src_prices)
            else:
                return self._create_empty_result(0, np.array([]))
    
    def _create_empty_result(self, length: int, raw_prices: np.ndarray) -> UKFResult:
        """空の結果を作成"""
        return UKFResult(
            filtered_values=np.full(length, np.nan),
            velocity_estimates=np.full(length, np.nan),
            acceleration_estimates=np.full(length, np.nan),
            uncertainty=np.full(length, np.nan),
            kalman_gains=np.full(length, np.nan),
            innovations=np.full(length, np.nan),
            sigma_points=np.full((7, 3), np.nan),  # 2*3+1=7シグマポイント, 3次元状態
            confidence_scores=np.full(length, np.nan),
            raw_values=raw_prices
        )
    
    def get_values(self) -> Optional[np.ndarray]:
        """フィルター済み価格を取得"""
        if not self._result_cache:
            return None
        
        result = self._get_latest_result()
        return result.filtered_values.copy() if result else None
    
    def get_velocity_estimates(self) -> Optional[np.ndarray]:
        """速度推定値を取得"""
        result = self._get_latest_result()
        return result.velocity_estimates.copy() if result else None
    
    def get_acceleration_estimates(self) -> Optional[np.ndarray]:
        """加速度推定値を取得"""
        result = self._get_latest_result()
        return result.acceleration_estimates.copy() if result else None
    
    def get_uncertainty(self) -> Optional[np.ndarray]:
        """不確実性を取得"""
        result = self._get_latest_result()
        return result.uncertainty.copy() if result else None
    
    def get_confidence_scores(self) -> Optional[np.ndarray]:
        """信頼度スコアを取得"""
        result = self._get_latest_result()
        return result.confidence_scores.copy() if result else None
    
    def get_innovations(self) -> Optional[np.ndarray]:
        """イノベーション（予測誤差）を取得"""
        result = self._get_latest_result()
        return result.innovations.copy() if result else None
    
    def get_sigma_points(self) -> Optional[np.ndarray]:
        """最終シグマポイントを取得"""
        result = self._get_latest_result()
        return result.sigma_points.copy() if result else None
    
    def _get_latest_result(self) -> Optional[UKFResult]:
        """最新の結果を取得"""
        if not self._result_cache:
            return None
        
        if self._cache_keys:
            return self._result_cache[self._cache_keys[-1]]
        else:
            return next(iter(self._result_cache.values()))
    
    def get_filter_metadata(self) -> Dict:
        """フィルターのメタデータを取得"""
        result = self._get_latest_result()
        if not result:
            return {}
        
        return {
            'filter_type': 'Unscented Kalman Filter',
            'src_type': self.src_type,
            'alpha': self.alpha,
            'beta': self.beta,
            'kappa': self.kappa,
            'process_noise_scale': self.process_noise_scale,
            'data_points': len(result.filtered_values),
            'avg_uncertainty': np.nanmean(result.uncertainty),
            'avg_confidence': np.nanmean(result.confidence_scores),
            'avg_velocity': np.nanmean(np.abs(result.velocity_estimates)),
            'avg_acceleration': np.nanmean(np.abs(result.acceleration_estimates)),
            'sigma_point_count': len(result.sigma_points)
        }
    
    def reset(self) -> None:
        """状態をリセット"""
        super().reset()
        self._result_cache = {}
        self._cache_keys = [] 