#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 **Simple Kalman Filter - シンプルカルマンフィルター** 🎯

パインスクリプトのカルマンフィルターをPythonで実装。
状態空間モデルを使用して価格の真の値を推定し、ノイズを除去。

📊 **アルゴリズム:**
1. **予測ステップ**: 前回の推定値から次の状態を予測
2. **更新ステップ**: 観測値（価格）で推定値を修正

🔧 **パラメータ:**
- R: 測定ノイズ分散 (観測の不確実性)
- Q: プロセスノイズ分散 (状態の変化の不確実性)
- 初期推定値: 最初の価格値
- 初期誤差共分散: 1.0

📈 **特徴:**
- リアルタイムフィルタリング
- 適応的な重み調整
- トレンド方向の色分け対応
- 永続的な状態管理

🌟 **使用例:**
```python
kalman = SimpleKalman(R=0.1, Q=0.01)
result = kalman.calculate(data)
```
"""

from dataclasses import dataclass
from typing import Union, Optional, Dict, Any
import numpy as np
import pandas as pd
from numba import njit

try:
    from ..indicator import Indicator
    from ..price_source import PriceSource
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)
    
    from indicators.indicator import Indicator
    from indicators.price_source import PriceSource


@dataclass
class SimpleKalmanResult:
    """シンプルカルマンフィルターの計算結果"""
    values: np.ndarray              # フィルタリングされた値
    kalman_gains: np.ndarray        # カルマンゲイン系列
    error_covariances: np.ndarray   # 誤差共分散系列
    predictions: np.ndarray         # 予測値系列
    raw_values: np.ndarray          # 元の価格データ
    trend_signals: np.ndarray       # トレンド方向 (1: 上昇, -1: 下降, 0: 横ばい)
    parameters: Dict[str, float]    # 使用されたパラメータ


@njit
def kalman_filter_core(
    measurements: np.ndarray,
    R: float,
    Q: float,
    initial_estimate: float,
    initial_covariance: float
) -> tuple:
    """
    カルマンフィルターのコア計算（Numba最適化）
    
    Args:
        measurements: 観測値（価格データ）
        R: 測定ノイズ分散
        Q: プロセスノイズ分散
        initial_estimate: 初期推定値
        initial_covariance: 初期誤差共分散
        
    Returns:
        tuple: (フィルタリング値, カルマンゲイン, 誤差共分散, 予測値)
    """
    n = len(measurements)
    
    # 結果配列の初期化
    estimates = np.zeros(n)
    kalman_gains = np.zeros(n)
    error_covariances = np.zeros(n)
    predictions = np.zeros(n)
    
    # 初期状態
    kalman_est = initial_estimate
    P = initial_covariance
    
    for i in range(n):
        measurement = measurements[i]
        
        # 1. 予測ステップ (Prediction Step)
        # 予測状態: 前回の推定値をそのまま使用（ランダムウォークモデル）
        pred = kalman_est
        predictions[i] = pred
        
        # 予測誤差共分散: プロセスノイズで不確実性を増加
        P_pred = P + Q
        
        # 2. 更新ステップ (Update Step)
        # カルマンゲインの計算
        K = P_pred / (P_pred + R)
        kalman_gains[i] = K
        
        # 推定値の更新: 予測値 + ゲイン * (観測値 - 予測値)
        kalman_est = pred + K * (measurement - pred)
        estimates[i] = kalman_est
        
        # 誤差共分散の更新
        P = (1 - K) * P_pred
        error_covariances[i] = P
    
    return estimates, kalman_gains, error_covariances, predictions


@njit
def calculate_trend_signals(values: np.ndarray) -> np.ndarray:
    """
    トレンド方向の計算（Numba最適化）
    
    Args:
        values: フィルタリングされた値
        
    Returns:
        トレンド信号配列 (1: 上昇, -1: 下降, 0: 横ばい)
    """
    n = len(values)
    trends = np.zeros(n)
    
    for i in range(1, n):
        if values[i] > values[i-1]:
            trends[i] = 1.0  # 上昇
        elif values[i] < values[i-1]:
            trends[i] = -1.0  # 下降
        else:
            trends[i] = 0.0  # 横ばい
    
    return trends


@njit
def validate_kalman_params(R: float, Q: float) -> bool:
    """カルマンフィルターパラメータの検証"""
    if R <= 0 or Q <= 0:
        return False
    if R > 100 or Q > 100:  # 異常に大きな値を防ぐ
        return False
    return True


class SimpleKalman(Indicator):
    """
    シンプルカルマンフィルター
    
    パインスクリプトのKalman Filterの完全Python実装。
    状態空間モデルを使用して価格の真の値を推定し、
    測定ノイズとプロセスノイズを考慮したフィルタリングを行う。
    """
    
    def __init__(
        self,
        R: float = 0.1,
        Q: float = 0.01,
        src_type: str = 'close',
        initial_covariance: float = 1.0,
        enable_trend_detection: bool = True
    ):
        """
        コンストラクタ
        
        Args:
            R: 測定ノイズ分散（観測の不確実性）
            Q: プロセスノイズ分散（状態変化の不確実性）
            src_type: 価格ソース ('close', 'hlc3', 'hl2', 'ohlc4', etc.)
            initial_covariance: 初期誤差共分散
            enable_trend_detection: トレンド検出を有効にするか
        """
        super().__init__(f"SimpleKalman(R={R}, Q={Q})")
        
        # パラメータ検証
        if not validate_kalman_params(R, Q):
            raise ValueError(f"無効なパラメータ: R={R}, Q={Q}")
        
        if initial_covariance <= 0:
            raise ValueError(f"初期誤差共分散は正の値である必要があります: {initial_covariance}")
        
        self.R = float(R)
        self.Q = float(Q)
        self.src_type = src_type.lower()
        self.initial_covariance = float(initial_covariance)
        self.enable_trend_detection = bool(enable_trend_detection)
        
        # ソースタイプ検証
        valid_sources = ['close', 'hlc3', 'hl2', 'ohlc4', 'high', 'low', 'open', 'oc2']
        if self.src_type not in valid_sources:
            raise ValueError(f"無効なソースタイプ: {src_type}")
        
        # 内部状態
        self._last_result = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> SimpleKalmanResult:
        """
        カルマンフィルターを計算
        
        Args:
            data: 価格データ
            
        Returns:
            SimpleKalmanResult: 計算結果
        """
        try:
            # 価格データの抽出
            src_prices = PriceSource.calculate_source(data, self.src_type)
            data_length = len(src_prices)
            
            if data_length < 2:
                return self._create_empty_result(data_length, src_prices)
            
            # 有効な価格データのチェック
            if np.all(np.isnan(src_prices)):
                return self._create_empty_result(data_length, src_prices)
            
            # NaN値の処理（最初の有効値で補間）
            valid_mask = ~np.isnan(src_prices)
            if not np.any(valid_mask):
                return self._create_empty_result(data_length, src_prices)
            
            # 最初の有効値を初期推定値とする
            first_valid_idx = np.where(valid_mask)[0][0]
            initial_estimate = src_prices[first_valid_idx]
            
            # NaN値を前方補間で埋める
            clean_prices = src_prices.copy()
            last_valid = initial_estimate
            for i in range(data_length):
                if np.isnan(clean_prices[i]):
                    clean_prices[i] = last_valid
                else:
                    last_valid = clean_prices[i]
            
            # カルマンフィルターの実行
            estimates, kalman_gains, error_covariances, predictions = kalman_filter_core(
                clean_prices, self.R, self.Q, initial_estimate, self.initial_covariance
            )
            
            # トレンド信号の計算
            if self.enable_trend_detection:
                trend_signals = calculate_trend_signals(estimates)
            else:
                trend_signals = np.zeros(data_length)
            
            # 結果の作成
            result = SimpleKalmanResult(
                values=estimates,
                kalman_gains=kalman_gains,
                error_covariances=error_covariances,
                predictions=predictions,
                raw_values=src_prices.copy(),
                trend_signals=trend_signals,
                parameters={
                    'R': self.R,
                    'Q': self.Q,
                    'initial_covariance': self.initial_covariance,
                    'src_type': self.src_type
                }
            )
            
            # 基底クラス用の値設定
            self._values = estimates
            self._last_result = result
            
            return result
            
        except Exception as e:
            error_msg = f"シンプルカルマンフィルター計算中にエラー: {str(e)}"
            self.logger.error(error_msg)
            
            # エラー時は空の結果を返す
            if isinstance(data, (pd.DataFrame, np.ndarray)) and len(data) > 0:
                src_prices = PriceSource.calculate_source(data, self.src_type)
                return self._create_empty_result(len(src_prices), src_prices)
            else:
                return self._create_empty_result(0, np.array([]))
    
    def _create_empty_result(self, length: int, raw_prices: np.ndarray) -> SimpleKalmanResult:
        """空の結果を作成"""
        return SimpleKalmanResult(
            values=np.full(length, np.nan),
            kalman_gains=np.full(length, np.nan),
            error_covariances=np.full(length, np.nan),
            predictions=np.full(length, np.nan),
            raw_values=raw_prices,
            trend_signals=np.zeros(length),
            parameters={
                'R': self.R,
                'Q': self.Q,
                'initial_covariance': self.initial_covariance,
                'src_type': self.src_type
            }
        )
    
    def get_values(self) -> Optional[np.ndarray]:
        """フィルタリングされた値を取得"""
        return self._values.copy() if self._values is not None else None
    
    def get_kalman_gains(self) -> Optional[np.ndarray]:
        """カルマンゲイン系列を取得"""
        if self._last_result is not None:
            return self._last_result.kalman_gains.copy()
        return None
    
    def get_error_covariances(self) -> Optional[np.ndarray]:
        """誤差共分散系列を取得"""
        if self._last_result is not None:
            return self._last_result.error_covariances.copy()
        return None
    
    def get_trend_signals(self) -> Optional[np.ndarray]:
        """トレンド信号系列を取得"""
        if self._last_result is not None:
            return self._last_result.trend_signals.copy()
        return None
    
    def get_trend_colors(self) -> Optional[np.ndarray]:
        """
        トレンド色を取得（パインスクリプト互換）
        
        Returns:
            色コード配列 (1: 緑/上昇, -1: 赤/下降, 0: 青/横ばい)
        """
        if self._last_result is not None:
            trends = self._last_result.trend_signals
            # パインスクリプトの色ロジック: 上昇なら緑、下降なら赤
            colors = np.zeros_like(trends)
            colors[trends > 0] = 1   # 緑（上昇）
            colors[trends < 0] = -1  # 赤（下降）
            # colors[trends == 0] = 0  # 青（横ばい）- デフォルト
            return colors
        return None
    
    def get_parameters(self) -> Dict[str, Any]:
        """パラメータ情報を取得"""
        return {
            'R': self.R,
            'Q': self.Q,
            'src_type': self.src_type,
            'initial_covariance': self.initial_covariance,
            'enable_trend_detection': self.enable_trend_detection
        }
    
    def reset(self) -> None:
        """状態をリセット"""
        super().reset()
        self._last_result = None
    
    @staticmethod
    def get_recommended_params(volatility_regime: str = 'medium') -> Dict[str, float]:
        """
        推奨パラメータを取得
        
        Args:
            volatility_regime: ボラティリティ環境 ('low', 'medium', 'high')
            
        Returns:
            推奨パラメータ辞書
        """
        params_map = {
            'low': {'R': 0.05, 'Q': 0.005},      # 低ボラティリティ: より敏感
            'medium': {'R': 0.1, 'Q': 0.01},     # 中ボラティリティ: バランス
            'high': {'R': 0.2, 'Q': 0.02}        # 高ボラティリティ: より保守的
        }
        return params_map.get(volatility_regime, params_map['medium'])


# 便利関数
def simple_kalman_filter(
    data: Union[pd.DataFrame, np.ndarray],
    R: float = 0.1,
    Q: float = 0.01,
    src_type: str = 'close'
) -> np.ndarray:
    """
    シンプルカルマンフィルターの計算（便利関数）
    
    Args:
        data: 価格データ
        R: 測定ノイズ分散
        Q: プロセスノイズ分散
        src_type: 価格ソース
        
    Returns:
        フィルタリングされた値
    """
    kalman = SimpleKalman(R=R, Q=Q, src_type=src_type)
    result = kalman.calculate(data)
    return result.values


def adaptive_kalman_params(
    data: Union[pd.DataFrame, np.ndarray],
    src_type: str = 'close',
    window: int = 20
) -> Dict[str, float]:
    """
    データに基づいて適応的にカルマンフィルターパラメータを推定
    
    Args:
        data: 価格データ
        src_type: 価格ソース
        window: ボラティリティ計算ウィンドウ
        
    Returns:
        推奨パラメータ
    """
    src_prices = PriceSource.calculate_source(data, src_type)
    
    if len(src_prices) < window:
        return SimpleKalman.get_recommended_params('medium')
    
    # 価格変化率のボラティリティを計算
    returns = np.diff(src_prices) / src_prices[:-1]
    volatility = np.nanstd(returns)
    
    # ボラティリティに基づくパラメータ調整
    if volatility < 0.01:
        regime = 'low'
    elif volatility > 0.03:
        regime = 'high'
    else:
        regime = 'medium'
    
    return SimpleKalman.get_recommended_params(regime)


if __name__ == "__main__":
    """直接実行時のテスト"""
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta
    
    print("=== シンプルカルマンフィルターのテスト ===")
    
    # テストデータ生成（ノイズ付きトレンド）
    np.random.seed(42)
    length = 100
    
    # 真の信号（サイン波 + トレンド）
    t = np.linspace(0, 4*np.pi, length)
    true_signal = 100 + 10 * np.sin(t) + 0.1 * t
    
    # 観測値（ノイズ付き）
    noise = np.random.normal(0, 2, length)
    observations = true_signal + noise
    
    # OHLC データの生成
    data = []
    for i, close in enumerate(observations):
        high = close + abs(np.random.normal(0, 1))
        low = close - abs(np.random.normal(0, 1))
        open_price = observations[i-1] if i > 0 else close
        
        data.append({
            'open': open_price,
            'high': max(high, open_price, close),
            'low': min(low, open_price, close),
            'close': close,
            'volume': np.random.uniform(1000, 5000)
        })
    
    df = pd.DataFrame(data)
    start_date = datetime.now() - timedelta(days=length)
    df.index = [start_date + timedelta(hours=i) for i in range(length)]
    
    print(f"テストデータ: {len(df)}ポイント")
    
    # 異なるパラメータでテスト
    test_params = [
        {'R': 0.1, 'Q': 0.01, 'label': 'デフォルト'},
        {'R': 0.05, 'Q': 0.005, 'label': '高感度'},
        {'R': 0.2, 'Q': 0.02, 'label': '低感度'}
    ]
    
    results = {}
    
    for params in test_params:
        print(f"\n{params['label']} (R={params['R']}, Q={params['Q']}) をテスト中...")
        
        kalman = SimpleKalman(R=params['R'], Q=params['Q'], src_type='close')
        result = kalman.calculate(df)
        
        results[params['label']] = result
        
        # 統計計算
        mae = np.mean(np.abs(result.values - true_signal))
        mse = np.mean((result.values - true_signal) ** 2)
        correlation = np.corrcoef(result.values, true_signal)[0, 1]
        
        print(f"  MAE: {mae:.4f}")
        print(f"  MSE: {mse:.4f}")
        print(f"  相関係数: {correlation:.4f}")
        print(f"  平均カルマンゲイン: {np.mean(result.kalman_gains):.4f}")
    
    # 適応的パラメータのテスト
    print(f"\n適応的パラメータ推定:")
    adaptive_params = adaptive_kalman_params(df)
    print(f"  推奨パラメータ: {adaptive_params}")
    
    print("\n=== テスト完了 ===")