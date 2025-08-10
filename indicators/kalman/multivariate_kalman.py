#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Optional, Tuple
import numpy as np
import pandas as pd
from numba import njit
import traceback

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
class MultivariateKalmanResult:
    """多変量カルマンフィルタの計算結果"""
    filtered_prices: np.ndarray          # フィルタリングされた価格（真の価格推定値）
    filtered_high: np.ndarray            # フィルタリングされた高値
    filtered_low: np.ndarray             # フィルタリングされた安値
    filtered_close: np.ndarray           # フィルタリングされた終値
    velocity_estimates: np.ndarray       # 速度推定値
    volatility_estimates: np.ndarray     # ボラティリティ推定値
    price_range_estimates: np.ndarray    # 価格レンジ推定値（動的）
    state_estimates: np.ndarray          # 完全な状態推定値（4次元）
    error_covariance: np.ndarray         # エラー共分散（対角成分）
    kalman_gains: np.ndarray             # カルマンゲイン行列
    innovations: np.ndarray              # イノベーション（3次元：high, low, close）
    confidence_scores: np.ndarray        # 信頼度スコア
    raw_ohlc: np.ndarray                # 元のOHLCデータ


@njit(fastmath=True, cache=True)
def multivariate_kalman_filter(
    high_prices: np.ndarray,
    low_prices: np.ndarray, 
    close_prices: np.ndarray,
    process_noise: float = 1e-5,
    observation_noise: float = 1e-3,
    volatility_noise: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    多変量カルマンフィルタ（OHLC対応）
    
    状態モデル:
    x[k] = F * x[k-1] + w[k]  (プロセスモデル)
    z[k] = H * x[k] + v[k]    (観測モデル)
    
    状態ベクトル: [真の価格, 速度, ボラティリティ, トレンド強度]
    観測ベクトル: [高値, 安値, 終値]
    
    Args:
        high_prices: 高値配列
        low_prices: 安値配列
        close_prices: 終値配列
        process_noise: プロセスノイズ分散
        observation_noise: 観測ノイズ分散
        volatility_noise: ボラティリティノイズ分散
    
    Returns:
        Tuple[filtered_prices, velocities, volatilities, price_ranges, state_estimates, 
              error_covariance_diag, kalman_gains, innovations, confidence_scores]
    """
    length = len(close_prices)
    
    # 状態ベクトルは4次元：[真の価格, 速度, ボラティリティ, トレンド強度]
    state_dim = 4
    obs_dim = 3  # 観測次元：[高値, 安値, 終値]
    
    # 結果配列の初期化
    filtered_prices = np.zeros(length)
    velocity_estimates = np.zeros(length)
    volatility_estimates = np.zeros(length)
    price_range_estimates = np.zeros(length)
    state_estimates = np.zeros((length, state_dim))
    error_covariance_diag = np.zeros((length, state_dim))
    kalman_gains = np.zeros((length, state_dim * obs_dim))  # 平坦化されたゲイン行列
    innovations = np.zeros((length, obs_dim))
    confidence_scores = np.zeros(length)
    
    if length == 0:
        return (filtered_prices, velocity_estimates, volatility_estimates, price_range_estimates,
                state_estimates, error_covariance_diag, kalman_gains, innovations, confidence_scores)
    
    # 状態遷移行列 F
    # x[k] = [価格, 速度, ボラティリティ, トレンド強度]
    dt = 1.0  # 時間ステップ
    F = np.array([
        [1.0, dt, 0.0, 0.0],      # 価格 = 価格 + 速度*dt
        [0.0, 0.95, 0.0, 0.0],    # 速度（減衰）
        [0.0, 0.0, 0.98, 0.0],    # ボラティリティ（減衰）
        [0.0, 0.0, 0.0, 0.99]     # トレンド強度（減衰）
    ])
    
    # 観測行列 H
    # 観測: [高値, 安値, 終値] = f(真の価格, ボラティリティ)
    H = np.array([
        [1.0, 0.0, 0.5, 0.0],     # 高値 = 真の価格 + 0.5*ボラティリティ
        [1.0, 0.0, -0.5, 0.0],    # 安値 = 真の価格 - 0.5*ボラティリティ
        [1.0, 0.0, 0.0, 0.0]      # 終値 = 真の価格
    ])
    
    # プロセスノイズ共分散行列 Q
    Q = np.array([
        [process_noise * dt * dt * 0.25, process_noise * dt * 0.5, 0.0, 0.0],
        [process_noise * dt * 0.5, process_noise, 0.0, 0.0],
        [0.0, 0.0, volatility_noise, 0.0],
        [0.0, 0.0, 0.0, process_noise * 0.1]
    ])
    
    # 観測ノイズ共分散行列 R
    R = np.array([
        [observation_noise * 2.0, 0.0, 0.0],         # 高値のノイズ（大きめ）
        [0.0, observation_noise * 2.0, 0.0],         # 安値のノイズ（大きめ）
        [0.0, 0.0, observation_noise]                # 終値のノイズ
    ])
    
    # 初期状態と共分散
    # 初期値を合理的に設定
    initial_price = close_prices[0]
    initial_volatility = abs(high_prices[0] - low_prices[0]) * 0.5
    if initial_volatility < 1e-6:
        initial_volatility = initial_price * 0.01  # 価格の1%をデフォルト
    
    x = np.array([initial_price, 0.0, initial_volatility, 0.5])  # [価格, 速度, ボラティリティ, トレンド強度]
    P = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.1, 0.0, 0.0],
        [0.0, 0.0, initial_volatility * initial_volatility, 0.0],
        [0.0, 0.0, 0.0, 0.1]
    ])
    
    # 初期値設定
    filtered_prices[0] = x[0]
    velocity_estimates[0] = x[1]
    volatility_estimates[0] = x[2]
    price_range_estimates[0] = x[2] * 2.0  # ボラティリティの2倍を価格レンジとする
    state_estimates[0] = x.copy()
    error_covariance_diag[0] = np.array([P[i, i] for i in range(state_dim)])
    kalman_gains[0] = np.zeros(state_dim * obs_dim)
    innovations[0] = np.zeros(obs_dim)
    confidence_scores[0] = 1.0
    
    # カルマンフィルタのメインループ
    for k in range(1, length):
        # === 予測ステップ ===
        # 状態予測: x_pred = F * x
        x_pred = np.dot(F, x)
        
        # 共分散予測: P_pred = F * P * F^T + Q
        P_pred = np.dot(np.dot(F, P), F.T) + Q
        
        # === 更新ステップ ===
        # 観測ベクトル
        z = np.array([high_prices[k], low_prices[k], close_prices[k]])
        
        # 観測予測: z_pred = H * x_pred
        z_pred = np.dot(H, x_pred)
        
        # イノベーション: y = z - z_pred
        y = z - z_pred
        
        # イノベーション共分散: S = H * P_pred * H^T + R
        S = np.dot(np.dot(H, P_pred), H.T) + R
        
        # カルマンゲイン: K = P_pred * H^T * S^(-1)
        try:
            S_inv = np.linalg.inv(S)
            K = np.dot(np.dot(P_pred, H.T), S_inv)
        except:
            # 特異行列の場合のフォールバック
            K = np.zeros((state_dim, obs_dim))
            for i in range(min(state_dim, obs_dim)):
                K[i, i] = 0.5
        
        # 状態更新: x = x_pred + K * y
        x = x_pred + np.dot(K, y)
        
        # 共分散更新: P = (I - K * H) * P_pred
        I_KH = np.eye(state_dim) - np.dot(K, H)
        P = np.dot(I_KH, P_pred)
        
        # 数値安定性の確保
        for i in range(state_dim):
            if P[i, i] < 1e-8:
                P[i, i] = 1e-8
        
        # 状態値の制約
        # ボラティリティは正の値
        if x[2] < 1e-6:
            x[2] = 1e-6
        
        # トレンド強度は0-1の範囲
        if x[3] < 0.0:
            x[3] = 0.0
        elif x[3] > 1.0:
            x[3] = 1.0
        
        # 価格レンジの動的推定（ボラティリティベース）
        dynamic_range = x[2] * 2.0  # ボラティリティの2倍
        # 実際の高安レンジとの適応的ブレンド
        actual_range = abs(high_prices[k] - low_prices[k])
        if actual_range > 0:
            alpha_range = 0.3  # ブレンド係数
            dynamic_range = alpha_range * actual_range + (1 - alpha_range) * dynamic_range
        
        # 信頼度スコアの計算
        confidence = 1.0 / (1.0 + P[0, 0] * 100.0)  # 価格の不確実性ベース
        confidence = max(0.0, min(1.0, confidence))
        
        # 結果の保存
        filtered_prices[k] = x[0]
        velocity_estimates[k] = x[1]
        volatility_estimates[k] = x[2]
        price_range_estimates[k] = dynamic_range
        state_estimates[k] = x.copy()
        error_covariance_diag[k] = np.array([P[i, i] for i in range(state_dim)])
        
        # カルマンゲインを平坦化して保存
        kalman_gains[k] = K.flatten()
        
        innovations[k] = y.copy()
        confidence_scores[k] = confidence
    
    return (filtered_prices, velocity_estimates, volatility_estimates, price_range_estimates,
            state_estimates, error_covariance_diag, kalman_gains, innovations, confidence_scores)


@njit(fastmath=True, cache=True)
def estimate_filtered_ohlc(
    filtered_prices: np.ndarray,
    volatility_estimates: np.ndarray,
    price_range_estimates: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    フィルタリングされた価格とボラティリティから合成OHLC値を推定
    
    Args:
        filtered_prices: フィルタリングされた真の価格
        volatility_estimates: ボラティリティ推定値
        price_range_estimates: 価格レンジ推定値
    
    Returns:
        Tuple[filtered_high, filtered_low, filtered_close]
    """
    length = len(filtered_prices)
    filtered_high = np.zeros(length)
    filtered_low = np.zeros(length)
    filtered_close = np.zeros(length)
    
    for i in range(length):
        price = filtered_prices[i]
        volatility = volatility_estimates[i]
        
        # 合成高値・安値の計算
        filtered_high[i] = price + volatility * 0.5
        filtered_low[i] = price - volatility * 0.5
        filtered_close[i] = price  # 終値は真の価格そのもの
    
    return filtered_high, filtered_low, filtered_close


class MultivariateKalman(Indicator):
    """
    多変量カルマンフィルタインジケーター（OHLC対応）
    
    OHLC情報を同時に処理する高度なカルマンフィルタ：
    - 高値・安値・終値を観測ベクトルとして使用
    - 真の価格・速度・ボラティリティ・トレンド強度を状態として推定
    - 価格レンジの動的推定機能
    - より多くの市場情報を活用した高精度フィルタリング
    
    特徴:
    - 多変量観測による情報量増加
    - ボラティリティの同時推定
    - 価格レンジの動的推定
    - トレンド強度の定量化
    """
    
    def __init__(
        self,
        process_noise: float = 1e-5,          # プロセスノイズ分散
        observation_noise: float = 1e-3,     # 観測ノイズ分散
        volatility_noise: float = 1e-4       # ボラティリティノイズ分散
    ):
        """
        コンストラクタ
        
        Args:
            process_noise: プロセスノイズ分散（デフォルト: 1e-5）
            observation_noise: 観測ノイズ分散（デフォルト: 1e-3）
            volatility_noise: ボラティリティノイズ分散（デフォルト: 1e-4）
        """
        # 指標名の作成
        indicator_name = f"MultivariateKalman(process_noise={process_noise}, obs_noise={observation_noise}, vol_noise={volatility_noise})"
        super().__init__(indicator_name)
        
        # パラメータを保存
        self.process_noise = process_noise
        self.observation_noise = observation_noise
        self.volatility_noise = volatility_noise
        
        # パラメータの検証
        if process_noise <= 0:
            raise ValueError("プロセスノイズは正の値である必要があります")
        if observation_noise <= 0:
            raise ValueError("観測ノイズは正の値である必要があります")
        if volatility_noise <= 0:
            raise ValueError("ボラティリティノイズは正の値である必要があります")
        
        # 結果キャッシュ（サイズ制限付き）
        self._result_cache = {}
        self._max_cache_size = 20
        self._cache_keys = []
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """
        データのハッシュ値を計算してキャッシュに使用する
        
        Args:
            data: OHLC価格データ
            
        Returns:
            データハッシュ文字列
        """
        try:
            # データ情報の取得
            if isinstance(data, pd.DataFrame):
                length = len(data)
                if length > 0:
                    first_vals = [float(data.iloc[0].get(col, 0)) for col in ['open', 'high', 'low', 'close']]
                    last_vals = [float(data.iloc[-1].get(col, 0)) for col in ['open', 'high', 'low', 'close']]
                else:
                    first_vals = last_vals = [0.0, 0.0, 0.0, 0.0]
            else:
                length = len(data)
                if length > 0 and data.ndim > 1 and data.shape[1] >= 4:
                    first_vals = [float(data[0, i]) for i in range(4)]
                    last_vals = [float(data[-1, i]) for i in range(4)]
                else:
                    first_vals = last_vals = [0.0, 0.0, 0.0, 0.0]
            
            # パラメータ情報
            params_sig = f"{self.process_noise}_{self.observation_noise}_{self.volatility_noise}"
            
            # ハッシュ計算
            data_sig = (length, tuple(first_vals), tuple(last_vals))
            return f"{hash(data_sig)}_{hash(params_sig)}"
            
        except Exception:
            # フォールバック
            return f"{id(data)}_{self.process_noise}_{self.observation_noise}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> MultivariateKalmanResult:
        """
        多変量カルマンフィルタを計算する
        
        Args:
            data: OHLC価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'open', 'high', 'low', 'close'カラムが必要
                NumPy配列の場合、OHLC順の2次元配列が必要
        
        Returns:
            MultivariateKalmanResult: フィルタリング結果と関連情報を含む結果
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            
            if data_hash in self._result_cache:
                if data_hash in self._cache_keys:
                    self._cache_keys.remove(data_hash)
                self._cache_keys.append(data_hash)
                cached_result = self._result_cache[data_hash]
                return MultivariateKalmanResult(
                    filtered_prices=cached_result.filtered_prices.copy(),
                    filtered_high=cached_result.filtered_high.copy(),
                    filtered_low=cached_result.filtered_low.copy(),
                    filtered_close=cached_result.filtered_close.copy(),
                    velocity_estimates=cached_result.velocity_estimates.copy(),
                    volatility_estimates=cached_result.volatility_estimates.copy(),
                    price_range_estimates=cached_result.price_range_estimates.copy(),
                    state_estimates=cached_result.state_estimates.copy(),
                    error_covariance=cached_result.error_covariance.copy(),
                    kalman_gains=cached_result.kalman_gains.copy(),
                    innovations=cached_result.innovations.copy(),
                    confidence_scores=cached_result.confidence_scores.copy(),
                    raw_ohlc=cached_result.raw_ohlc.copy()
                )
            
            # OHLC データの抽出
            if isinstance(data, pd.DataFrame):
                # DataFrameの場合
                required_columns = ['high', 'low', 'close']
                for col in required_columns:
                    if col not in data.columns:
                        # カラム名の代替を試す
                        alt_names = {
                            'high': ['High'],
                            'low': ['Low'], 
                            'close': ['Close', 'close', 'adj close', 'Adj Close']
                        }
                        found = False
                        for alt_name in alt_names.get(col, []):
                            if alt_name in data.columns:
                                data = data.rename(columns={alt_name: col})
                                found = True
                                break
                        if not found:
                            raise ValueError(f"必要なカラムが見つかりません: {col}")
                
                high_prices = data['high'].values.astype(np.float64)
                low_prices = data['low'].values.astype(np.float64)
                close_prices = data['close'].values.astype(np.float64)
                
                # OHLCデータの作成（open列がない場合は前日終値を使用）
                if 'open' in data.columns:
                    open_prices = data['open'].values.astype(np.float64)
                else:
                    open_prices = np.roll(close_prices, 1)
                    open_prices[0] = close_prices[0]
                
                raw_ohlc = np.column_stack([open_prices, high_prices, low_prices, close_prices])
                
            elif isinstance(data, np.ndarray):
                # NumPy配列の場合
                if data.ndim != 2 or data.shape[1] < 4:
                    raise ValueError("NumPy配列はOHLC形式の2次元配列である必要があります")
                
                data = data.astype(np.float64)
                open_prices = data[:, 0]
                high_prices = data[:, 1]
                low_prices = data[:, 2]
                close_prices = data[:, 3]
                raw_ohlc = data.copy()
                
            else:
                raise ValueError("サポートされていないデータ型です")
            
            # データ長の検証
            data_length = len(close_prices)
            if data_length == 0:
                raise ValueError("入力データが空です")
            
            if data_length < 5:
                self.logger.warning(f"データが短すぎます（{data_length}点）。最低5点以上を推奨します。")
            
            # 多変量カルマンフィルタの計算
            (filtered_prices, velocity_estimates, volatility_estimates, price_range_estimates,
             state_estimates, error_covariance_diag, kalman_gains, innovations, confidence_scores) = multivariate_kalman_filter(
                high_prices, low_prices, close_prices, 
                self.process_noise, self.observation_noise, self.volatility_noise
            )
            
            # フィルタリングされたOHLC値の推定
            filtered_high, filtered_low, filtered_close = estimate_filtered_ohlc(
                filtered_prices, volatility_estimates, price_range_estimates
            )
            
            # 結果の保存
            result = MultivariateKalmanResult(
                filtered_prices=filtered_prices.copy(),
                filtered_high=filtered_high.copy(),
                filtered_low=filtered_low.copy(),
                filtered_close=filtered_close.copy(),
                velocity_estimates=velocity_estimates.copy(),
                volatility_estimates=volatility_estimates.copy(),
                price_range_estimates=price_range_estimates.copy(),
                state_estimates=state_estimates.copy(),
                error_covariance=error_covariance_diag.copy(),
                kalman_gains=kalman_gains.copy(),
                innovations=innovations.copy(),
                confidence_scores=confidence_scores.copy(),
                raw_ohlc=raw_ohlc.copy()
            )
            
            # キャッシュ管理
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            self._values = filtered_prices  # 基底クラスの要件を満たすため
            return result
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"MultivariateKalman計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は空の結果を返す
            error_result = MultivariateKalmanResult(
                filtered_prices=np.array([]),
                filtered_high=np.array([]),
                filtered_low=np.array([]),
                filtered_close=np.array([]),
                velocity_estimates=np.array([]),
                volatility_estimates=np.array([]),
                price_range_estimates=np.array([]),
                state_estimates=np.array([]).reshape(0, 4),
                error_covariance=np.array([]).reshape(0, 4),
                kalman_gains=np.array([]).reshape(0, 12),
                innovations=np.array([]).reshape(0, 3),
                confidence_scores=np.array([]),
                raw_ohlc=np.array([]).reshape(0, 4)
            )
            return error_result
    
    def get_values(self) -> Optional[np.ndarray]:
        """フィルタリングされた真の価格を取得する（後方互換性のため）"""
        if not self._result_cache:
            return None
            
        result = self._get_latest_result()
        return result.filtered_prices.copy() if result else None
    
    def get_filtered_ohlc(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        フィルタリングされたOHLC値を取得する
        
        Returns:
            Tuple[open, high, low, close]: フィルタリングされたOHLC配列
        """
        result = self._get_latest_result()
        if result:
            # openは前の終値を使用
            filtered_open = np.roll(result.filtered_close, 1)
            filtered_open[0] = result.filtered_close[0]
            return (filtered_open.copy(), result.filtered_high.copy(), 
                   result.filtered_low.copy(), result.filtered_close.copy())
        return None
    
    def get_velocity_estimates(self) -> Optional[np.ndarray]:
        """速度推定値を取得する"""
        result = self._get_latest_result()
        return result.velocity_estimates.copy() if result else None
    
    def get_volatility_estimates(self) -> Optional[np.ndarray]:
        """ボラティリティ推定値を取得する"""
        result = self._get_latest_result()
        return result.volatility_estimates.copy() if result else None
    
    def get_price_range_estimates(self) -> Optional[np.ndarray]:
        """価格レンジ推定値を取得する"""
        result = self._get_latest_result()
        return result.price_range_estimates.copy() if result else None
    
    def get_state_estimates(self) -> Optional[np.ndarray]:
        """完全な状態推定値を取得する"""
        result = self._get_latest_result()
        return result.state_estimates.copy() if result else None
    
    def get_confidence_scores(self) -> Optional[np.ndarray]:
        """信頼度スコアを取得する"""
        result = self._get_latest_result()
        return result.confidence_scores.copy() if result else None
    
    def get_innovations(self) -> Optional[np.ndarray]:
        """イノベーション（予測誤差）を取得する"""
        result = self._get_latest_result()
        return result.innovations.copy() if result else None
    
    def get_raw_ohlc(self) -> Optional[np.ndarray]:
        """元のOHLCデータを取得する"""
        result = self._get_latest_result()
        return result.raw_ohlc.copy() if result else None
    
    def _get_latest_result(self) -> Optional[MultivariateKalmanResult]:
        """最新の結果を取得"""
        if not self._result_cache:
            return None
        
        if self._cache_keys:
            return self._result_cache[self._cache_keys[-1]]
        else:
            return next(iter(self._result_cache.values()))
    
    def reset(self) -> None:
        """
        インディケーターの状態をリセットする
        """
        super().reset()
        self._result_cache = {}
        self._cache_keys = []