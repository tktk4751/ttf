#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🌀 **Quantum Adaptive Kalman Filter - 量子適応カルマンフィルター** 🌀

量子もつれ効果を利用した超高精度フィルタリング：
- 量子コヒーレンス効果による適応的ノイズ制御
- 多次元相関による量子もつれ効果のシミュレーション
- 振幅と位相情報を活用した動的測定ノイズ調整
- 超高精度の価格フィルタリング

📊 **特徴:**
- 量子もつれ効果による超高精度フィルタリング
- 動的コヒーレンス計算
- プライスソース対応
- 包括的な信頼度スコア算出
"""

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
class QuantumAdaptiveKalmanResult:
    """量子適応カルマンフィルターの計算結果"""
    values: np.ndarray                # フィルタリングされた価格（メイン結果）
    quantum_coherence: np.ndarray     # 量子コヒーレンス値
    kalman_gains: np.ndarray          # カルマンゲイン
    innovations: np.ndarray           # イノベーション（観測値 - 予測値）
    confidence_scores: np.ndarray     # 信頼度スコア
    raw_values: np.ndarray            # 元の価格データ


@njit(fastmath=True, cache=True)
def quantum_adaptive_kalman_filter_numba(
    prices: np.ndarray, 
    amplitude: np.ndarray, 
    phase: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    量子適応カルマンフィルター - 量子もつれ効果を利用した超高精度フィルタリング
    
    Args:
        prices: 価格データ
        amplitude: 振幅データ（測定ノイズ計算用）
        phase: 位相データ（将来的な拡張用）
        
    Returns:
        Tuple[filtered_prices, quantum_coherence, kalman_gains, innovations, confidence_scores]
    """
    n = len(prices)
    filtered_prices = np.zeros(n)
    quantum_coherence = np.zeros(n)
    kalman_gains = np.zeros(n)
    innovations = np.zeros(n)
    confidence_scores = np.zeros(n)
    
    if n < 10:
        return prices.copy(), np.ones(n) * 0.5, np.zeros(n), np.zeros(n), np.ones(n)
    
    # 初期状態
    state_estimate = prices[0]
    error_covariance = 1.0
    
    # 量子パラメータ
    base_process_noise = 0.001
    
    for i in range(1, n):
        # 量子もつれ効果計算
        if i >= 10:
            entanglement_factor = 0.0
            for j in range(1, min(6, i)):
                correlation = (prices[i] - prices[i-j]) * (prices[i-j] - prices[i-j-1])
                if correlation != 0:
                    entanglement_factor += np.sin(np.pi * correlation / (abs(correlation) + 1e-10))
            quantum_coherence[i] = abs(entanglement_factor) / 5.0
        else:
            quantum_coherence[i] = 0.5
        
        # 適応的プロセスノイズ（量子コヒーレンスベース）
        adaptive_process_noise = base_process_noise * (1.0 + quantum_coherence[i])
        
        # 予測ステップ
        state_prediction = state_estimate
        error_prediction = error_covariance + adaptive_process_noise
        
        # 測定ノイズ（振幅ベース）
        if i < len(amplitude):
            measurement_noise = max(0.001, amplitude[i] * 0.05)
        else:
            measurement_noise = 0.01
        
        # カルマンゲイン
        denominator = error_prediction + measurement_noise
        if denominator > 1e-10:
            kalman_gain = error_prediction / denominator
        else:
            kalman_gain = 0.5
        
        # 更新ステップ
        innovation = prices[i] - state_prediction
        state_estimate = state_prediction + kalman_gain * innovation
        error_covariance = (1 - kalman_gain) * error_prediction
        
        filtered_prices[i] = state_estimate
        kalman_gains[i] = kalman_gain
        innovations[i] = innovation
        confidence_scores[i] = quantum_coherence[i] * (1.0 - kalman_gain)
    
    return filtered_prices, quantum_coherence, kalman_gains, innovations, confidence_scores


@njit(fastmath=True, cache=True)
def calculate_amplitude_phase(prices: np.ndarray, window: int = 14) -> Tuple[np.ndarray, np.ndarray]:
    """
    価格データから振幅と位相を計算
    
    Args:
        prices: 価格データ
        window: 計算ウィンドウサイズ
        
    Returns:
        Tuple[amplitude, phase]
    """
    n = len(prices)
    amplitude = np.zeros(n)
    phase = np.zeros(n)
    
    if n < window:
        amplitude.fill(0.01)
        return amplitude, phase
    
    for i in range(window, n):
        # 振幅計算（移動標準偏差ベース）
        price_window = prices[i-window:i]
        mean_price = np.mean(price_window)
        amplitude[i] = np.std(price_window)
        
        # 位相計算（価格トレンドベース）
        if i > window:
            price_change = prices[i] - prices[i-1]
            trend = prices[i] - mean_price
            if trend != 0:
                phase[i] = np.arctan2(price_change, trend)
            else:
                phase[i] = 0.0
    
    # 初期値の補完
    if n >= window:
        amplitude[:window] = amplitude[window]
        phase[:window] = phase[window]
    
    return amplitude, phase


class QuantumAdaptiveKalman(Indicator):
    """
    量子適応カルマンフィルター
    
    量子もつれ効果を利用した超高精度フィルタリング：
    - 量子コヒーレンス効果による適応的ノイズ制御
    - 多次元相関による量子もつれ効果のシミュレーション
    - 振幅と位相情報を活用した動的測定ノイズ調整
    """
    
    def __init__(
        self,
        src_type: str = 'close',
        base_process_noise: float = 0.001,
        amplitude_window: int = 14,
        coherence_lookback: int = 5
    ):
        """
        コンストラクタ
        
        Args:
            src_type: 価格ソース ('close', 'hlc3', 'hl2', 'ohlc4', etc.)
            base_process_noise: 基本プロセスノイズ
            amplitude_window: 振幅計算ウィンドウサイズ
            coherence_lookback: 量子コヒーレンス計算のルックバック期間
        """
        super().__init__("QuantumAdaptiveKalman")
        self.src_type = src_type.lower()
        self.base_process_noise = base_process_noise
        self.amplitude_window = amplitude_window
        self.coherence_lookback = coherence_lookback
        
        # 結果保存用
        self._last_result: Optional[QuantumAdaptiveKalmanResult] = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> QuantumAdaptiveKalmanResult:
        """
        量子適応カルマンフィルターを計算
        
        Args:
            data: OHLC価格データ
            
        Returns:
            QuantumAdaptiveKalmanResult: 計算結果
        """
        try:
            # 価格データの抽出
            if isinstance(data, pd.DataFrame):
                if data.empty:
                    return self._create_empty_result(0)
                prices = PriceSource.calculate_source(data, self.src_type)
            else:
                if len(data) == 0:
                    return self._create_empty_result(0)
                if data.ndim == 1:
                    prices = data.astype(float)
                else:
                    # 2次元配列の場合、最後の列を価格として使用
                    prices = data[:, -1].astype(float)
            
            if len(prices) < 10:
                return self._create_empty_result(len(prices), prices)
            
            # 振幅と位相の計算
            amplitude, phase = calculate_amplitude_phase(prices, self.amplitude_window)
            
            # 量子適応カルマンフィルターの実行
            filtered_prices, quantum_coherence, kalman_gains, innovations, confidence_scores = \
                quantum_adaptive_kalman_filter_numba(prices, amplitude, phase)
            
            # 結果の作成
            result = QuantumAdaptiveKalmanResult(
                values=filtered_prices,
                quantum_coherence=quantum_coherence,
                kalman_gains=kalman_gains,
                innovations=innovations,
                confidence_scores=confidence_scores,
                raw_values=prices
            )
            
            self._last_result = result
            self._values = filtered_prices
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"量子適応カルマンフィルター計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は空の結果を返す
            if isinstance(data, (pd.DataFrame, np.ndarray)) and len(data) > 0:
                if isinstance(data, pd.DataFrame):
                    prices = PriceSource.calculate_source(data, self.src_type)
                else:
                    prices = data[:, -1] if data.ndim > 1 else data
                return self._create_empty_result(len(prices), prices)
            else:
                return self._create_empty_result(0)
    
    def _create_empty_result(self, length: int, raw_prices: Optional[np.ndarray] = None) -> QuantumAdaptiveKalmanResult:
        """空の結果を作成"""
        if raw_prices is None:
            raw_prices = np.array([])
        
        return QuantumAdaptiveKalmanResult(
            values=np.full(length, np.nan),
            quantum_coherence=np.full(length, np.nan),
            kalman_gains=np.full(length, np.nan),
            innovations=np.full(length, np.nan),
            confidence_scores=np.full(length, np.nan),
            raw_values=raw_prices
        )
    
    def get_values(self) -> Optional[np.ndarray]:
        """フィルタリングされた価格を取得"""
        return self._values.copy() if self._values is not None else None
    
    def get_quantum_coherence(self) -> Optional[np.ndarray]:
        """量子コヒーレンス値を取得"""
        if self._last_result:
            return self._last_result.quantum_coherence.copy()
        return None
    
    def get_kalman_gains(self) -> Optional[np.ndarray]:
        """カルマンゲインを取得"""
        if self._last_result:
            return self._last_result.kalman_gains.copy()
        return None
    
    def get_confidence_scores(self) -> Optional[np.ndarray]:
        """信頼度スコアを取得"""
        if self._last_result:
            return self._last_result.confidence_scores.copy()
        return None
    
    def get_innovations(self) -> Optional[np.ndarray]:
        """イノベーションを取得"""
        if self._last_result:
            return self._last_result.innovations.copy()
        return None
    
    def reset(self) -> None:
        """状態をリセット"""
        super().reset()
        self._last_result = None


# テスト実行部分
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta
    
    print("=== 量子適応カルマンフィルターのテスト ===")
    
    # テストデータ生成
    np.random.seed(42)
    length = 500
    t = np.linspace(0, 4*np.pi, length)
    
    # 複雑なトレンド + ノイズ + 周期性
    base_signal = 100 + 10 * np.sin(t) + 5 * np.sin(3*t) + 2 * np.sin(7*t)
    trend = 0.02 * t
    noise = np.random.normal(0, 2, length)
    prices = base_signal + trend + noise
    
    # OHLC データの生成
    data = []
    for i, close in enumerate(prices):
        daily_range = abs(np.random.normal(0, 1.5))
        high = close + daily_range * np.random.uniform(0.3, 1.0)
        low = close - daily_range * np.random.uniform(0.3, 1.0)
        open_price = close + np.random.normal(0, 0.5)
        
        # 論理的整合性の確保
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.uniform(1000, 10000)
        })
    
    df = pd.DataFrame(data)
    start_date = datetime.now() - timedelta(days=length)
    df.index = [start_date + timedelta(hours=i) for i in range(length)]
    
    print(f"テストデータ: {len(df)}ポイント")
    print(f"価格範囲: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    # 量子適応カルマンフィルターの計算
    quantum_kalman = QuantumAdaptiveKalman(src_type='close')
    result = quantum_kalman.calculate(df)
    
    print(f"\n計算結果:")
    print(f"フィルタリング値の範囲: {np.nanmin(result.values):.2f} - {np.nanmax(result.values):.2f}")
    print(f"量子コヒーレンスの範囲: {np.nanmin(result.quantum_coherence):.4f} - {np.nanmax(result.quantum_coherence):.4f}")
    print(f"平均カルマンゲイン: {np.nanmean(result.kalman_gains):.4f}")
    print(f"平均信頼度スコア: {np.nanmean(result.confidence_scores):.4f}")
    
    # プロット
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 価格とフィルタリング結果
    axes[0].plot(df.index, df['close'], alpha=0.7, label='元の価格', color='blue')
    axes[0].plot(df.index, result.values, label='量子フィルタリング', color='red', linewidth=2)
    axes[0].set_title('量子適応カルマンフィルター - 価格フィルタリング')
    axes[0].set_ylabel('価格')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 量子コヒーレンス
    axes[1].plot(df.index, result.quantum_coherence, color='purple', linewidth=1.5)
    axes[1].set_title('量子コヒーレンス')
    axes[1].set_ylabel('コヒーレンス値')
    axes[1].grid(True, alpha=0.3)
    
    # カルマンゲインと信頼度スコア
    axes[2].plot(df.index, result.kalman_gains, alpha=0.7, label='カルマンゲイン', color='green')
    axes[2].plot(df.index, result.confidence_scores, alpha=0.7, label='信頼度スコア', color='orange')
    axes[2].set_title('カルマンゲインと信頼度スコア')
    axes[2].set_ylabel('値')
    axes[2].set_xlabel('時間')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== テスト完了 ===")