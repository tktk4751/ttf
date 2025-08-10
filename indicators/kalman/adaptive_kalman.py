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
class AdaptiveKalmanResult:
    """適応カルマンフィルタの計算結果"""
    filtered_signal: np.ndarray      # フィルタリングされた信号
    adaptive_gain: np.ndarray        # 適応的カルマンゲイン
    innovations: np.ndarray          # イノベーション（観測値 - 予測値）
    error_covariance: np.ndarray     # エラー共分散
    confidence_score: np.ndarray     # 信頼度スコア


@njit(fastmath=True, cache=True)
def adaptive_kalman_filter(signal: np.ndarray, process_noise: float = 1e-5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    適応カルマンフィルタ
    動的ノイズ推定付き
    
    Args:
        signal: 入力信号
        process_noise: プロセスノイズ（デフォルト: 1e-5）
    
    Returns:
        Tuple[filtered_signal, adaptive_gain, innovations, error_covariance]
    """
    length = len(signal)
    filtered_signal = np.zeros(length)
    adaptive_gain = np.zeros(length)
    innovations = np.zeros(length)
    error_covariance = np.zeros(length)
    
    # 初期化
    if length > 0:
        state = signal[0]
        error_cov = 1.0
        filtered_signal[0] = state
        adaptive_gain[0] = 0.5
        innovations[0] = 0.0
        error_covariance[0] = error_cov
    
    for i in range(1, length):
        # 予測ステップ
        predicted_state = state
        predicted_covariance = error_cov + process_noise
        
        # 適応的観測ノイズ推定
        if i > 5:
            recent_residuals = np.zeros(5)
            for j in range(5):
                if i - j >= 0:
                    recent_residuals[j] = abs(signal[i-j] - filtered_signal[i-j])
            observation_noise = np.var(recent_residuals) + 1e-6
        else:
            observation_noise = 1e-3
        
        # カルマンゲイン
        kalman_gain = predicted_covariance / (predicted_covariance + observation_noise)
        
        # 更新ステップ
        innovation = signal[i] - predicted_state
        state = predicted_state + kalman_gain * innovation
        error_cov = (1 - kalman_gain) * predicted_covariance
        
        filtered_signal[i] = state
        adaptive_gain[i] = kalman_gain
        innovations[i] = innovation
        error_covariance[i] = error_cov
    
    return filtered_signal, adaptive_gain, innovations, error_covariance


@njit(fastmath=True, cache=True)
def calculate_confidence_score(error_covariance: np.ndarray, adaptive_gain: np.ndarray) -> np.ndarray:
    """
    信頼度スコアを計算する
    
    Args:
        error_covariance: エラー共分散
        adaptive_gain: 適応的カルマンゲイン
    
    Returns:
        信頼度スコア（0-1の範囲）
    """
    length = len(error_covariance)
    confidence_score = np.zeros(length)
    
    for i in range(length):
        # エラー共分散が小さく、ゲインが安定している場合に高い信頼度
        if error_covariance[i] > 0:
            base_confidence = 1.0 / (1.0 + error_covariance[i])
        else:
            base_confidence = 1.0
        
        # ゲインの安定性を考慮
        if i > 0:
            gain_stability = 1.0 - min(1.0, abs(adaptive_gain[i] - adaptive_gain[i-1]))
            confidence_score[i] = min(1.0, base_confidence * gain_stability)
        else:
            confidence_score[i] = base_confidence
    
    return confidence_score


class AdaptiveKalman(Indicator):
    """
    適応カルマンフィルタインジケーター
    
    動的にノイズレベルを推定し、リアルタイムでノイズ除去を行うスムーザー：
    - 適応的観測ノイズ推定
    - 動的カルマンゲイン調整
    - 複数のプライスソースに対応
    - 信頼度スコア計算
    
    特徴:
    - ノイズの多い環境：高いノイズ除去効果
    - 安定した環境：低遅延でトレンドを追従
    - 適応性：市場状況に応じて自動調整
    """
    
    def __init__(
        self,
        process_noise: float = 1e-5,          # プロセスノイズ
        src_type: str = 'close',              # ソースタイプ
        min_observation_noise: float = 1e-6,  # 最小観測ノイズ
        adaptation_window: int = 5             # 適応ウィンドウサイズ
    ):
        """
        コンストラクタ
        
        Args:
            process_noise: プロセスノイズ（デフォルト: 1e-5）
            src_type: ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4')
                - 'close': 終値（デフォルト）
                - 'hlc3': (高値 + 安値 + 終値) / 3
                - 'hl2': (高値 + 安値) / 2
                - 'ohlc4': (始値 + 高値 + 安値 + 終値) / 4
            min_observation_noise: 最小観測ノイズ（デフォルト: 1e-6）
            adaptation_window: 適応ウィンドウサイズ（デフォルト: 5）
        """
        # 指標名の作成
        indicator_name = f"AdaptiveKalman(process_noise={process_noise}, {src_type})"
        super().__init__(indicator_name)
        
        # パラメータを保存
        self.process_noise = process_noise
        self.src_type = src_type.lower()
        self.min_observation_noise = min_observation_noise
        self.adaptation_window = adaptation_window
        
        # # ソースタイプの検証
        # if self.src_type not in self.SRC_TYPES:
        #     raise ValueError(f"無効なソースタイプです: {src_type}。有効なオプション: {', '.join(self.SRC_TYPES)}")
        
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
            params_sig = f"{self.process_noise}_{self.src_type}_{self.min_observation_noise}_{self.adaptation_window}"
            
            # 超高速ハッシュ
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(params_sig)}"
            
        except Exception:
            # フォールバック
            return f"{id(data)}_{self.process_noise}_{self.src_type}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> AdaptiveKalmanResult:
        """
        適応カルマンフィルタを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、選択したソースタイプに必要なカラムが必要
        
        Returns:
            AdaptiveKalmanResult: フィルタリング結果と関連情報を含む結果
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
                return AdaptiveKalmanResult(
                    filtered_signal=cached_result.filtered_signal.copy(),
                    adaptive_gain=cached_result.adaptive_gain.copy(),
                    innovations=cached_result.innovations.copy(),
                    error_covariance=cached_result.error_covariance.copy(),
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
            
            # 適応カルマンフィルタの計算
            filtered_signal, adaptive_gain, innovations, error_covariance = adaptive_kalman_filter(
                signal, self.process_noise
            )
            
            # 信頼度スコアの計算
            confidence_score = calculate_confidence_score(error_covariance, adaptive_gain)
            
            # 結果の保存（参照問題を避けるため必要な部分だけコピー）
            result = AdaptiveKalmanResult(
                filtered_signal=filtered_signal.copy(),
                adaptive_gain=adaptive_gain.copy(),
                innovations=innovations.copy(),
                error_covariance=error_covariance.copy(),
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
            self.logger.error(f"AdaptiveKalman計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は空の結果を返す
            error_result = AdaptiveKalmanResult(
                filtered_signal=np.array([]),
                adaptive_gain=np.array([]),
                innovations=np.array([]),
                error_covariance=np.array([]),
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
    
    def get_adaptive_gain(self) -> Optional[np.ndarray]:
        """
        適応的カルマンゲインを取得する
        
        Returns:
            np.ndarray: 適応的カルマンゲインの値
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.adaptive_gain.copy()
    
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
            np.ndarray: エラー共分散の値
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