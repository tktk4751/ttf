#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 **Unscented Kalman Filter V2 (UKF V2) - 無香料カルマンフィルターV2** 🎯

提供されたアカデミックな無香料カルマンフィルターの実装：
- マルチレート処理対応（予測と更新の分離）
- 汎用的な状態遷移関数と観測関数の対応
- より厳密な数学的実装
- カスタマイズ可能なシステム関数

🌟 **UKF V2の特徴:**
1. **汎用システム関数**: f(x, u)とh(x)を自由に定義可能
2. **マルチレート処理**: 予測と更新を独立して実行
3. **厳密な実装**: アカデミックな理論に忠実
4. **柔軟性**: 様々な状態空間モデルに対応
"""

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional, List, Callable
import numpy as np
import pandas as pd
from numba import njit
import traceback
import math

try:
    from ..indicator import Indicator
    from ..price_source import PriceSource
except ImportError:
    # Fallback for potential execution context issues
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from indicator import Indicator
    from price_source import PriceSource


@dataclass
class UKFV2Result:
    """無香料カルマンフィルターV2の計算結果"""
    filtered_values: np.ndarray      # フィルター済み価格
    state_estimates: np.ndarray      # 状態推定値（全状態）
    error_covariance: np.ndarray     # エラー共分散（対角成分）
    innovations: np.ndarray          # イノベーション（予測誤差）
    confidence_scores: np.ndarray    # 信頼度スコア
    raw_values: np.ndarray          # 元の価格データ
    prediction_history: np.ndarray   # 予測履歴
    update_history: np.ndarray      # 更新履歴

# Aliases for compatibility
UKFResult = UKFV2Result
UnscentedKalmanResult = UKFV2Result


class UnscentedKalmanFilterV2(object):
    """
    無香料カルマンフィルターV2
    
    汎用的な無香料カルマンフィルターの実装：
    - 状態量: Xk+1 = f(Xk, Uk) + v
    - 観測量: Yk+1 = h(Xk+1) + w
    - マルチレート処理対応
    - カスタマイズ可能なシステム関数
    """
    
    def __init__(
        self, 
        f: Callable[[np.ndarray, np.ndarray], np.ndarray],
        h: Callable[[np.ndarray], np.ndarray], 
        Q: np.ndarray, 
        R: np.ndarray, 
        x0_estimate: np.ndarray, 
        P0: np.ndarray, 
        u_dim: int, 
        step: int, 
        kappa: float = 0.0
    ):
        """
        初期化
        
        Args:
            f: 状態遷移関数 f(xt, ut) -> x_dim次元のnp.ndarray
            h: 観測関数 h(xt) -> y_dim次元のnp.ndarray  
            Q: プロセスノイズ共分散行列 (x_dim x x_dim)
            R: 観測ノイズ共分散行列 (y_dim x y_dim)
            x0_estimate: 状態初期値 (x_dim,)
            P0: 初期誤差共分散行列 (x_dim x x_dim)
            u_dim: 制御入力次元
            step: 最大ステップ数
            kappa: UKFパラメータ
        """
        # 入力検証
        assert isinstance(Q, np.ndarray) and Q.shape[0] == Q.shape[1], \
               'Qは正方行列である必要があります。'
        self.Q = Q.copy()
        self.x_dim = Q.shape[0]  # 状態量次元
        
        assert isinstance(R, np.ndarray) and R.shape[0] == R.shape[1], \
               'Rは正方行列である必要があります。'
        self.R = R.copy()
        self.y_dim = R.shape[0]  # 観測量次元
        
        self.u_dim = u_dim  # 制御量次元
        
        # システム関数の設定
        self.f = f
        self.h = h
        
        # システム関数の妥当性チェック
        self._check_system_function()
        
        assert isinstance(x0_estimate, np.ndarray) and len(x0_estimate) == self.x_dim, \
               'x0は状態次元と一致する必要があります。'
        
        assert isinstance(P0, np.ndarray) and P0.shape == (self.x_dim, self.x_dim), \
               'P0は状態次元の正方行列である必要があります。'
        self.Pk = P0.copy()
        
        self.t_dim = step + 1  # 時刻次元
        self.kappa = kappa
        self._k = 0  # 現在の予測ステップ
        self._k_correct = 0  # 現在の更新ステップ
        
        # 状態履歴の初期化
        self.X = np.zeros((self.x_dim, self.t_dim))
        self.X[:, self._k] = x0_estimate.copy()
        
        # UKFパラメータ
        self.omega = 0.5 / (self.x_dim + self.kappa)
        self.omega0 = self.kappa / (self.x_dim + self.kappa)
        
        # 予測・更新履歴
        self.prediction_history = np.zeros((self.x_dim, self.t_dim))
        self.update_history = np.zeros((self.x_dim, self.t_dim))
        self.prediction_history[:, 0] = x0_estimate.copy()
        self.update_history[:, 0] = x0_estimate.copy()
    
    def _check_system_function(self):
        """システム関数の妥当性をチェック"""
        # ランダムな入力でテスト
        x_test = np.random.randn(self.x_dim)
        u_test = np.random.randn(self.u_dim)
        
        # 状態遷移関数のテスト
        x_next = self.f(x_test, u_test)
        assert isinstance(x_next, np.ndarray) and len(x_next) == self.x_dim, \
               'fの返り値は状態次元と一致する必要があります。'
        
        # 観測関数のテスト
        y = self.h(x_test)
        assert isinstance(y, np.ndarray) and len(y) == self.y_dim, \
               'hの返り値は観測次元と一致する必要があります。'
    
    def estimate(self, u: np.ndarray) -> bool:
        """
        予測ステップ
        
        Args:
            u: 制御入力 (u_dim,)
            
        Returns:
            bool: 予測が成功したかどうか
        """
        if self._k + 1 >= self.t_dim:
            return False
        
        assert isinstance(u, np.ndarray) and len(u) == self.u_dim, \
               'uは制御入力次元と一致する必要があります。'
        
        # 現在の状態推定値を取得
        X = self.get_current_estimate_value()
        
        # シグマポイント生成
        X_sigma = self._generate_sigma_points(X, self.Pk)
        
        # 各シグマポイントを状態遷移関数に通す
        for i in range(X_sigma.shape[1]):
            X_sigma[:, i] = self.f(X_sigma[:, i], u)
        
        # 予測状態の計算（重み付き平均）
        Xk_priori = self.omega0 * X_sigma[:, 0]
        for i in range(1, X_sigma.shape[1]):
            Xk_priori += self.omega * X_sigma[:, i]
        
        # 予測共分散の計算
        diff = X_sigma[:, 0] - Xk_priori
        P_priori = self.Q + self.omega0 * np.outer(diff, diff)
        for i in range(1, X_sigma.shape[1]):
            diff = X_sigma[:, i] - Xk_priori
            P_priori += self.omega * np.outer(diff, diff)
        
        # 状態を更新
        self._k += 1
        self.X[:, self._k] = Xk_priori
        self.Pk = P_priori
        self.prediction_history[:, self._k] = Xk_priori.copy()
        
        return True
    
    def correct(self, Y: np.ndarray) -> bool:
        """
        更新ステップ
        
        Args:
            Y: 観測値 (y_dim,)
            
        Returns:
            bool: 更新が成功したかどうか
        """
        if self._k_correct >= self._k:
            return False
        
        assert isinstance(Y, np.ndarray) and len(Y) == self.y_dim, \
               'Yは観測次元と一致する必要があります。'
        
        # 現在の状態推定値を取得
        X = self.get_current_estimate_value()
        
        # シグマポイント生成
        X_sigma = self._generate_sigma_points(X, self.Pk)
        
        # 観測シグマポイント計算
        Y_sigma = np.zeros((self.y_dim, X_sigma.shape[1]))
        for i in range(X_sigma.shape[1]):
            Y_sigma[:, i] = self.h(X_sigma[:, i])
        
        # 予測観測値の計算（重み付き平均）
        Y_estimate = self.omega0 * Y_sigma[:, 0]
        for i in range(1, Y_sigma.shape[1]):
            Y_estimate += self.omega * Y_sigma[:, i]
        
        # イノベーション共分散の計算
        diff = Y_sigma[:, 0] - Y_estimate
        P_yy = self.R + self.omega0 * np.outer(diff, diff)
        for i in range(1, Y_sigma.shape[1]):
            diff = Y_sigma[:, i] - Y_estimate
            P_yy += self.omega * np.outer(diff, diff)
        
        # 状態-観測間の相互共分散の計算
        xdiff = X_sigma[:, 0] - X
        ydiff = Y_sigma[:, 0] - Y_estimate
        P_xy = self.omega0 * np.outer(xdiff, ydiff)
        for i in range(1, X_sigma.shape[1]):
            xdiff = X_sigma[:, i] - X
            ydiff = Y_sigma[:, i] - Y_estimate
            P_xy += self.omega * np.outer(xdiff, ydiff)
        
        # カルマンゲイン計算
        try:
            K_gain = P_xy @ np.linalg.inv(P_yy)
        except np.linalg.LinAlgError:
            # 特異行列の場合のフォールバック
            K_gain = np.zeros((self.x_dim, self.y_dim))
        
        # 状態と共分散の更新
        innovation = Y - Y_estimate
        self.X[:, self._k] = X + K_gain @ innovation
        self.Pk = self.Pk - K_gain @ P_xy.T
        
        # 数値安定性の確保
        self.Pk = (self.Pk + self.Pk.T) / 2  # 対称性の保証
        eigenvals = np.linalg.eigvals(self.Pk)
        if np.any(eigenvals < 1e-8):
            self.Pk += np.eye(self.x_dim) * 1e-8
        
        self._k_correct = self._k
        self.update_history[:, self._k] = self.X[:, self._k].copy()
        
        return True
    
    def _generate_sigma_points(self, mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
        """
        シグマポイント生成
        
        Args:
            mean: 平均ベクトル
            covariance: 共分散行列
            
        Returns:
            シグマポイント行列 (x_dim, 2*x_dim+1)
        """
        X_sigma = np.zeros((self.x_dim, 1 + self.x_dim * 2))
        X_sigma[:, 0] = mean
        
        try:
            # コレスキー分解
            P_cholesky = np.linalg.cholesky(covariance)
            
            for i in range(self.x_dim):
                diff = math.sqrt(self.x_dim + self.kappa) * P_cholesky[:, i]
                X_sigma[:, i + 1] = mean + diff
                X_sigma[:, self.x_dim + i + 1] = mean - diff
                
        except np.linalg.LinAlgError:
            # フォールバック: 固有値分解
            try:
                eigenvals, eigenvecs = np.linalg.eigh(covariance)
                eigenvals = np.maximum(eigenvals, 1e-8)  # 負の固有値を修正
                sqrt_matrix = eigenvecs @ np.diag(np.sqrt(eigenvals))
                
                for i in range(self.x_dim):
                    diff = math.sqrt(self.x_dim + self.kappa) * sqrt_matrix[:, i]
                    X_sigma[:, i + 1] = mean + diff
                    X_sigma[:, self.x_dim + i + 1] = mean - diff
                    
            except:
                # 最終フォールバック: 対角行列
                for i in range(self.x_dim):
                    std_dev = math.sqrt(max(covariance[i, i], 1e-8))
                    diff = math.sqrt(self.x_dim + self.kappa) * std_dev
                    X_sigma[i, i + 1] = mean[i] + diff
                    X_sigma[i, self.x_dim + i + 1] = mean[i] - diff
        
        return X_sigma
    
    def get_estimate_value(self) -> np.ndarray:
        """全ステップの状態推定値を取得"""
        return self.X[:, :self._k + 1]
    
    def get_current_estimate_value(self) -> np.ndarray:
        """現在ステップの状態推定値を取得"""
        return self.X[:, self._k]


# 金融データ用のシステム関数定義
def financial_state_transition(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    金融データ用の状態遷移関数
    
    状態: [価格, 速度, 加速度]
    制御: [市場ドリフト]
    """
    if len(x) < 3:
        return x
    
    price, velocity, acceleration = x[0], x[1], x[2]
    market_drift = u[0] if len(u) > 0 else 0.0
    
    dt = 1.0
    damping = 0.95
    
    # 状態遷移
    new_price = price + velocity * dt + 0.5 * acceleration * dt * dt + market_drift
    new_velocity = velocity * damping + acceleration * dt
    new_acceleration = acceleration * 0.9
    
    return np.array([new_price, new_velocity, new_acceleration])


def financial_observation(x: np.ndarray) -> np.ndarray:
    """
    金融データ用の観測関数
    
    観測: [価格] (価格のみを観測)
    """
    return np.array([x[0]])


class UnscentedKalmanFilterV2Wrapper(Indicator):
    """
    UKF V2のIndicatorラッパー
    
    金融データ処理用の無香料カルマンフィルター：
    - 汎用的なUKF V2実装をベース
    - 金融データ特化の状態遷移・観測関数
    - マルチレート処理対応
    - 既存のIndicatorインターフェースに準拠
    """
    
    def __init__(
        self,
        src_type: str = 'close',
        kappa: float = 0.0,
        process_noise_scale: float = 0.01,
        observation_noise_scale: float = 0.001,
        max_steps: int = 1000
    ):
        """
        コンストラクタ
        
        Args:
            src_type: 価格ソース ('close', 'hlc3', 'hl2', 'ohlc4')
            kappa: UKFパラメータ
            process_noise_scale: プロセスノイズスケール
            observation_noise_scale: 観測ノイズスケール
            max_steps: 最大処理ステップ数
        """
        indicator_name = f"UKF_V2(src={src_type}, κ={kappa})"
        super().__init__(indicator_name)
        
        self.src_type = src_type.lower()
        self.kappa = kappa
        self.process_noise_scale = process_noise_scale
        self.observation_noise_scale = observation_noise_scale
        self.max_steps = max_steps
        
        # ソースタイプの検証
        valid_sources = ['close', 'hlc3', 'hl2', 'ohlc4', 'high', 'low', 'open', 'oc2']
        if self.src_type not in valid_sources:
            raise ValueError(f"無効なソースタイプ: {src_type}")
        
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
            
            params_sig = f"{self.kappa}_{self.src_type}_{self.process_noise_scale}_{self.observation_noise_scale}"
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(params_sig)}"
            
        except Exception:
            return f"{id(data)}_{self.kappa}_{self.src_type}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> UKFV2Result:
        """
        UKF V2を使用してフィルタリングを計算
        
        Args:
            data: 価格データ
            
        Returns:
            UKFV2Result: フィルタリング結果
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            
            if data_hash in self._result_cache:
                if data_hash in self._cache_keys:
                    self._cache_keys.remove(data_hash)
                self._cache_keys.append(data_hash)
                cached_result = self._result_cache[data_hash]
                return UKFV2Result(
                    filtered_values=cached_result.filtered_values.copy(),
                    state_estimates=cached_result.state_estimates.copy(),
                    error_covariance=cached_result.error_covariance.copy(),
                    innovations=cached_result.innovations.copy(),
                    confidence_scores=cached_result.confidence_scores.copy(),
                    raw_values=cached_result.raw_values.copy(),
                    prediction_history=cached_result.prediction_history.copy(),
                    update_history=cached_result.update_history.copy()
                )
            
            # 価格データの抽出
            src_prices = PriceSource.calculate_source(data, self.src_type)
            data_length = len(src_prices)
            
            if data_length < 5:
                return self._create_empty_result(data_length, src_prices)
            
            # UKF V2の設定
            x_dim = 3  # [価格, 速度, 加速度]
            y_dim = 1  # [価格]
            u_dim = 1  # [市場ドリフト]
            
            # ノイズ行列
            Q = np.array([
                [self.process_noise_scale, 0.0, 0.0],
                [0.0, self.process_noise_scale * 0.1, 0.0],
                [0.0, 0.0, self.process_noise_scale * 0.01]
            ])
            R = np.array([[self.observation_noise_scale]])
            
            # 初期状態
            x0 = np.array([src_prices[0], 0.0, 0.0])
            P0 = np.array([
                [1.0, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [0.0, 0.0, 0.01]
            ])
            
            # UKF V2インスタンス作成
            ukf = UnscentedKalmanFilterV2(
                f=financial_state_transition,
                h=financial_observation,
                Q=Q, R=R, x0_estimate=x0, P0=P0,
                u_dim=u_dim, step=min(data_length, self.max_steps),
                kappa=self.kappa
            )
            
            # フィルタリング実行
            filtered_values = np.zeros(data_length)
            state_estimates = np.zeros((data_length, x_dim))
            error_covariance = np.zeros((data_length, x_dim))
            innovations = np.zeros(data_length)
            confidence_scores = np.zeros(data_length)
            
            # 初期値設定
            filtered_values[0] = src_prices[0]
            state_estimates[0] = x0
            error_covariance[0] = np.diag(P0)
            innovations[0] = 0.0
            confidence_scores[0] = 1.0
            
            # メインループ
            for i in range(1, min(data_length, ukf.t_dim)):
                # 市場ドリフト推定（簡単な例）
                if i > 1:
                    market_drift = (src_prices[i-1] - src_prices[i-2]) * 0.1
                else:
                    market_drift = 0.0
                
                u = np.array([market_drift])
                
                # 予測ステップ
                if ukf.estimate(u):
                    # 更新ステップ
                    y_obs = np.array([src_prices[i]])
                    ukf.correct(y_obs)
                    
                    # 結果の保存
                    current_state = ukf.get_current_estimate_value()
                    filtered_values[i] = current_state[0]
                    state_estimates[i] = current_state
                    error_covariance[i] = np.diag(ukf.Pk)
                    
                    # イノベーション計算
                    predicted_obs = ukf.h(current_state)
                    innovations[i] = src_prices[i] - predicted_obs[0]
                    
                    # 信頼度計算
                    uncertainty = np.sqrt(ukf.Pk[0, 0])
                    confidence_scores[i] = 1.0 / (1.0 + uncertainty * 10.0)
                else:
                    # 予測が失敗した場合
                    filtered_values[i] = filtered_values[i-1] if i > 0 else src_prices[i]
                    state_estimates[i] = state_estimates[i-1] if i > 0 else x0
                    error_covariance[i] = error_covariance[i-1] if i > 0 else np.diag(P0)
                    innovations[i] = 0.0
                    confidence_scores[i] = 0.5
            
            # 履歴の取得
            ukf_history = ukf.get_estimate_value()
            prediction_history = ukf.prediction_history[:, :ukf._k+1]
            update_history = ukf.update_history[:, :ukf._k_correct+1]
            
            # 結果の作成
            result = UKFV2Result(
                filtered_values=filtered_values.copy(),
                state_estimates=state_estimates.copy(),
                error_covariance=error_covariance.copy(),
                innovations=innovations.copy(),
                confidence_scores=confidence_scores.copy(),
                raw_values=src_prices.copy(),
                prediction_history=prediction_history.copy(),
                update_history=update_history.copy()
            )
            
            # キャッシュ管理
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            self._values = filtered_values  # 基底クラス用
            return result
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"UKF V2計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は空の結果を返す
            if isinstance(data, (pd.DataFrame, np.ndarray)) and len(data) > 0:
                src_prices = PriceSource.calculate_source(data, self.src_type)
                return self._create_empty_result(len(src_prices), src_prices)
            else:
                return self._create_empty_result(0, np.array([]))
    
    def _create_empty_result(self, length: int, raw_prices: np.ndarray) -> UKFV2Result:
        """空の結果を作成"""
        return UKFV2Result(
            filtered_values=np.full(length, np.nan),
            state_estimates=np.full((length, 3), np.nan),
            error_covariance=np.full((length, 3), np.nan),
            innovations=np.full(length, np.nan),
            confidence_scores=np.full(length, np.nan),
            raw_values=raw_prices,
            prediction_history=np.full((3, length), np.nan),
            update_history=np.full((3, length), np.nan)
        )
    
    def get_values(self) -> Optional[np.ndarray]:
        """フィルター済み価格を取得"""
        if not self._result_cache:
            return None
        
        result = self._get_latest_result()
        return result.filtered_values.copy() if result else None
    
    def get_state_estimates(self) -> Optional[np.ndarray]:
        """状態推定値を取得"""
        result = self._get_latest_result()
        return result.state_estimates.copy() if result else None
    
    def get_velocity_estimates(self) -> Optional[np.ndarray]:
        """速度推定値を取得"""
        result = self._get_latest_result()
        if result and result.state_estimates.shape[1] >= 2:
            return result.state_estimates[:, 1].copy()
        return None
    
    def get_acceleration_estimates(self) -> Optional[np.ndarray]:
        """加速度推定値を取得"""
        result = self._get_latest_result()
        if result and result.state_estimates.shape[1] >= 3:
            return result.state_estimates[:, 2].copy()
        return None
    
    def get_confidence_scores(self) -> Optional[np.ndarray]:
        """信頼度スコアを取得"""
        result = self._get_latest_result()
        return result.confidence_scores.copy() if result else None
    
    def get_innovations(self) -> Optional[np.ndarray]:
        """イノベーションを取得"""
        result = self._get_latest_result()
        return result.innovations.copy() if result else None
    
    def get_prediction_history(self) -> Optional[np.ndarray]:
        """予測履歴を取得"""
        result = self._get_latest_result()
        return result.prediction_history.copy() if result else None
    
    def get_update_history(self) -> Optional[np.ndarray]:
        """更新履歴を取得"""
        result = self._get_latest_result()
        return result.update_history.copy() if result else None
    
    def _get_latest_result(self) -> Optional[UKFV2Result]:
        """最新の結果を取得"""
        if not self._result_cache:
            return None
        
        if self._cache_keys:
            return self._result_cache[self._cache_keys[-1]]
        else:
            return next(iter(self._result_cache.values()))
    
    def reset(self) -> None:
        """状態をリセット"""
        super().reset()
        self._result_cache = {}
        self._cache_keys = []