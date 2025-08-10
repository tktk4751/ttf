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
class KalmanResult:
    """線形カルマンフィルタの計算結果"""
    filtered_signal: np.ndarray      # フィルタリングされた信号
    state_estimates: np.ndarray      # 状態推定値（2次元配列：位置と速度）
    error_covariance: np.ndarray     # エラー共分散（対角成分）
    kalman_gains: np.ndarray         # カルマンゲイン（位置の値）
    innovations: np.ndarray          # イノベーション（観測値 - 予測値）
    confidence_score: np.ndarray     # 信頼度スコア


@njit(fastmath=True, cache=True)
def linear_kalman_filter(
    signal: np.ndarray, 
    process_noise: float = 1e-5,
    observation_noise: float = 1e-3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    線形カルマンフィルタ
    
    状態モデル:
    x[k] = F * x[k-1] + w[k]  (プロセスモデル)
    z[k] = H * x[k] + v[k]    (観測モデル)
    
    状態ベクトル: [位置, 速度]
    
    Args:
        signal: 入力信号
        process_noise: プロセスノイズ分散
        observation_noise: 観測ノイズ分散
    
    Returns:
        Tuple[filtered_signal, state_estimates, error_covariance_diag, kalman_gains, innovations]
    """
    length = len(signal)
    
    # 状態ベクトルは2次元：[位置, 速度]
    state_dim = 2
    
    # 結果配列の初期化
    filtered_signal = np.zeros(length)
    state_estimates = np.zeros((length, state_dim))
    error_covariance_diag = np.zeros((length, state_dim))  # 対角成分のみ保存
    kalman_gains = np.zeros(length)
    innovations = np.zeros(length)
    
    if length == 0:
        return filtered_signal, state_estimates, error_covariance_diag, kalman_gains, innovations
    
    # 状態遷移行列 F (位置-速度モデル)
    # x[k] = [1 dt] * x[k-1] + w[k]
    #        [0  1]
    dt = 1.0  # 時間ステップ
    F = np.array([[1.0, dt], [0.0, 1.0]])
    
    # 観測行列 H (位置のみを観測)
    # z[k] = [1 0] * x[k] + v[k]
    H = np.array([1.0, 0.0])
    
    # プロセスノイズ共分散行列 Q
    # 位置と速度に対するノイズ
    Q = np.array([[process_noise * dt * dt * 0.25, process_noise * dt * 0.5],
                  [process_noise * dt * 0.5, process_noise]])
    
    # 観測ノイズ分散 R
    R = observation_noise
    
    # 初期状態と共分散
    x = np.array([signal[0], 0.0])  # [位置, 速度]
    P = np.array([[1.0, 0.0], [0.0, 0.1]])  # 初期共分散行列
    
    # 初期値設定
    filtered_signal[0] = x[0]
    state_estimates[0] = x.copy()
    error_covariance_diag[0] = np.array([P[0, 0], P[1, 1]])
    kalman_gains[0] = 0.5
    innovations[0] = 0.0
    
    # カルマンフィルタのメインループ
    for k in range(1, length):
        # === 予測ステップ ===
        # 状態予測: x_pred = F * x
        x_pred = np.dot(F, x)
        
        # 共分散予測: P_pred = F * P * F^T + Q
        P_pred = np.dot(np.dot(F, P), F.T) + Q
        
        # === 更新ステップ ===
        # 観測予測: z_pred = H * x_pred
        z_pred = np.dot(H, x_pred)
        
        # イノベーション: y = z - z_pred
        y = signal[k] - z_pred
        
        # イノベーション共分散: S = H * P_pred * H^T + R
        S = np.dot(np.dot(H, P_pred), H.T) + R
        
        # カルマンゲイン: K = P_pred * H^T * S^(-1)
        if S > 1e-12:
            K = np.dot(P_pred, H) / S
        else:
            K = np.array([0.5, 0.0])
        
        # 状態更新: x = x_pred + K * y
        x = x_pred + K * y
        
        # 共分散更新: P = (I - K * H) * P_pred
        I_KH = np.eye(state_dim) - np.outer(K, H)
        P = np.dot(I_KH, P_pred)
        
        # 数値安定性の確保
        # 共分散行列の対角要素が負にならないよう調整
        for i in range(state_dim):
            if P[i, i] < 1e-8:
                P[i, i] = 1e-8
        
        # 結果の保存
        filtered_signal[k] = x[0]  # 位置（フィルタ済み信号）
        state_estimates[k] = x.copy()
        error_covariance_diag[k] = np.array([P[0, 0], P[1, 1]])
        kalman_gains[k] = K[0]  # 位置に対するゲイン
        innovations[k] = y
    
    return filtered_signal, state_estimates, error_covariance_diag, kalman_gains, innovations


@njit(fastmath=True, cache=True)
def calculate_confidence_score(error_covariance_diag: np.ndarray) -> np.ndarray:
    """
    信頼度スコアを計算する
    
    Args:
        error_covariance_diag: エラー共分散の対角成分
    
    Returns:
        信頼度スコア（0-1の範囲）
    """
    length = len(error_covariance_diag)
    confidence_score = np.zeros(length)
    
    for i in range(length):
        # 位置の不確実性（共分散の対角成分の最初の要素）を使用
        position_uncertainty = error_covariance_diag[i, 0]
        
        if position_uncertainty > 0:
            # 不確実性の逆数ベースの信頼度
            confidence_score[i] = 1.0 / (1.0 + position_uncertainty * 100.0)
        else:
            confidence_score[i] = 1.0
        
        # 0-1の範囲に制限
        confidence_score[i] = max(0.0, min(1.0, confidence_score[i]))
    
    return confidence_score


class Kalman(Indicator):
    """
    線形カルマンフィルタインジケーター
    
    古典的な線形カルマンフィルタによる価格データのノイズ除去：
    - 位置-速度モデルによる状態推定
    - プロセスノイズと観測ノイズのバランス調整
    - 複数のプライスソースに対応
    - 信頼度スコア計算
    
    特徴:
    - シンプルで理論的に確立された手法
    - 低計算コスト
    - 安定した性能
    - 線形システムに最適
    """
    
    def __init__(
        self,
        process_noise: float = 1e-5,          # プロセスノイズ分散
        observation_noise: float = 1e-3,     # 観測ノイズ分散
        src_type: str = 'close'               # ソースタイプ
    ):
        """
        コンストラクタ
        
        Args:
            process_noise: プロセスノイズ分散（デフォルト: 1e-5）
            observation_noise: 観測ノイズ分散（デフォルト: 1e-3）
            src_type: ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4')
                - 'close': 終値（デフォルト）
                - 'hlc3': (高値 + 安値 + 終値) / 3
                - 'hl2': (高値 + 安値) / 2
                - 'ohlc4': (始値 + 高値 + 安値 + 終値) / 4
        """
        # 指標名の作成
        indicator_name = f"Kalman(process_noise={process_noise}, obs_noise={observation_noise}, {src_type})"
        super().__init__(indicator_name)
        
        # パラメータを保存
        self.process_noise = process_noise
        self.observation_noise = observation_noise
        self.src_type = src_type.lower()
        
        # パラメータの検証
        if process_noise <= 0:
            raise ValueError("プロセスノイズは正の値である必要があります")
        if observation_noise <= 0:
            raise ValueError("観測ノイズは正の値である必要があります")
        
        # ソースタイプの検証
        valid_sources = ['close', 'hlc3', 'hl2', 'ohlc4', 'high', 'low', 'open']
        if self.src_type not in valid_sources:
            raise ValueError(f"無効なソースタイプです: {src_type}。有効なオプション: {', '.join(valid_sources)}")
        
        # 結果キャッシュ（サイズ制限付き）
        self._result_cache = {}
        self._max_cache_size = 20
        self._cache_keys = []
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """
        データのハッシュ値を計算してキャッシュに使用する（超高速版）
        
        Args:
            data: 価格データ
            
        Returns:
            データハッシュ文字列
        """
        # 超高速化のため最小限のサンプリング
        try:
            # データ情報の取得
            if isinstance(data, pd.DataFrame):
                length = len(data)
                first_val = float(data.iloc[0].get('close', data.iloc[0, -1])) if length > 0 else 0.0
                last_val = float(data.iloc[-1].get('close', data.iloc[-1, -1])) if length > 0 else 0.0
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
            
            # 最小限のパラメータ情報
            params_sig = f"{self.process_noise}_{self.observation_noise}_{self.src_type}"
            
            # 超高速ハッシュ
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(params_sig)}"
            
        except Exception:
            # フォールバック
            return f"{id(data)}_{self.process_noise}_{self.observation_noise}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> KalmanResult:
        """
        線形カルマンフィルタを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、選択したソースタイプに必要なカラムが必要
        
        Returns:
            KalmanResult: フィルタリング結果と関連情報を含む結果
        """
        try:
            # キャッシュチェック - 同じデータの場合は計算をスキップ（高速化）
            data_hash = self._get_data_hash(data)
            
            # キャッシュにある場合は取得して返す
            if data_hash in self._result_cache:
                # キャッシュキーの順序を更新（最も新しく使われたキーを最後に）
                if data_hash in self._cache_keys:
                    self._cache_keys.remove(data_hash)
                self._cache_keys.append(data_hash)
                cached_result = self._result_cache[data_hash]
                return KalmanResult(
                    filtered_signal=cached_result.filtered_signal.copy(),
                    state_estimates=cached_result.state_estimates.copy(),
                    error_covariance=cached_result.error_covariance.copy(),
                    kalman_gains=cached_result.kalman_gains.copy(),
                    innovations=cached_result.innovations.copy(),
                    confidence_score=cached_result.confidence_score.copy()
                )
            
            # データの準備 - PriceSourceを使用してソース価格を取得
            source_data = PriceSource.calculate_source(data, self.src_type)
            
            # NumPy配列に変換（float64型で統一）
            signal = np.asarray(source_data, dtype=np.float64)
            
            # データ長の検証
            data_length = len(signal)
            if data_length == 0:
                raise ValueError("入力データが空です")
            
            if data_length < 3:
                self.logger.warning(f"データが短すぎます（{data_length}点）。最低3点以上を推奨します。")
            
            # 線形カルマンフィルタの計算
            filtered_signal, state_estimates, error_covariance_diag, kalman_gains, innovations = linear_kalman_filter(
                signal, self.process_noise, self.observation_noise
            )
            
            # 信頼度スコアの計算
            confidence_score = calculate_confidence_score(error_covariance_diag)
            
            # 結果の保存（参照問題を避けるため必要な部分だけコピー）
            result = KalmanResult(
                filtered_signal=filtered_signal.copy(),
                state_estimates=state_estimates.copy(),
                error_covariance=error_covariance_diag.copy(),
                kalman_gains=kalman_gains.copy(),
                innovations=innovations.copy(),
                confidence_score=confidence_score.copy()
            )
            
            # キャッシュを更新
            # キャッシュサイズ管理
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                # 最も古いキャッシュを削除
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            self._values = filtered_signal  # 基底クラスの要件を満たすため
            return result
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"Kalman計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は空の結果を返す
            error_result = KalmanResult(
                filtered_signal=np.array([]),
                state_estimates=np.array([]).reshape(0, 2),
                error_covariance=np.array([]).reshape(0, 2),
                kalman_gains=np.array([]),
                innovations=np.array([]),
                confidence_score=np.array([])
            )
            return error_result
    
    def get_values(self) -> Optional[np.ndarray]:
        """フィルタリングされた信号値のみを取得する（後方互換性のため）"""
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.filtered_signal.copy()
    
    def get_state_estimates(self) -> Optional[np.ndarray]:
        """
        状態推定値を取得する（位置と速度）
        
        Returns:
            np.ndarray: 状態推定値（2次元配列：[位置, 速度]）
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.state_estimates.copy()
    
    def get_velocity_estimates(self) -> Optional[np.ndarray]:
        """
        速度推定値のみを取得する
        
        Returns:
            np.ndarray: 速度推定値
        """
        state_estimates = self.get_state_estimates()
        if state_estimates is not None and state_estimates.shape[1] >= 2:
            return state_estimates[:, 1].copy()
        return None
    
    def get_kalman_gains(self) -> Optional[np.ndarray]:
        """
        カルマンゲインを取得する
        
        Returns:
            np.ndarray: カルマンゲインの値
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.kalman_gains.copy()
    
    def get_innovations(self) -> Optional[np.ndarray]:
        """
        イノベーション（観測値 - 予測値）を取得する
        
        Returns:
            np.ndarray: イノベーションの値
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.innovations.copy()
    
    def get_confidence_score(self) -> Optional[np.ndarray]:
        """
        信頼度スコアを取得する
        
        Returns:
            np.ndarray: 信頼度スコアの値（0-1の範囲）
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.confidence_score.copy()
    
    def get_error_covariance(self) -> Optional[np.ndarray]:
        """
        エラー共分散を取得する
        
        Returns:
            np.ndarray: エラー共分散の対角成分（2次元配列：[位置分散, 速度分散]）
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.error_covariance.copy()
    
    def reset(self) -> None:
        """
        インディケーターの状態をリセットする
        """
        super().reset()
        self._result_cache = {}
        self._cache_keys = []