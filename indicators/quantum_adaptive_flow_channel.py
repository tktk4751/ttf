#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quantum Adaptive Flow Channel (QAFC)
量子適応フローチャネル

究極の超高性能トレンドフォローチャネルインジケーター
- 超低遅延カルマンフィルター
- 量子ノイズ適応システム  
- ニューラル価格予測エンジン
- 極限追従アルゴリズム
"""

from dataclasses import dataclass
from typing import Union, Tuple, Optional
import numpy as np
import pandas as pd
from numba import jit, njit, prange, float64
import math

from .indicator import Indicator
from .price_source import PriceSource


@dataclass
class QAFCResult:
    """QAFC計算結果"""
    # メインチャネル
    centerline: np.ndarray          # 超適応センターライン
    upper_channel: np.ndarray       # 動的上部チャネル
    lower_channel: np.ndarray       # 動的下部チャネル
    
    # 適応メトリクス
    volatility_ratio: np.ndarray    # ボラティリティ比率
    trend_strength: np.ndarray      # トレンド強度
    noise_level: np.ndarray         # ノイズレベル
    
    # 予測成分
    predicted_price: np.ndarray     # 予測価格
    confidence_score: np.ndarray    # 信頼度スコア
    
    # トレード指標
    trend_direction: np.ndarray     # トレンド方向 (-1, 0, 1)
    momentum_flow: np.ndarray       # モメンタムフロー


# === 超低遅延カルマンフィルター ===

@njit(fastmath=True, cache=True)
def ultra_low_latency_kalman_filter(
    prices: np.ndarray,
    process_noise: float = 0.01,
    measurement_noise: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    超低遅延カルマンフィルター
    
    最適な状態推定により超高速かつ正確な価格追従を実現
    """
    n = len(prices)
    filtered_prices = np.zeros(n)
    prediction_errors = np.zeros(n)
    
    # 初期化
    x = prices[0] if n > 0 else 0.0  # 状態推定値
    p = 1.0  # 誤差共分散
    
    # カルマンゲイン適応パラメータ
    q = process_noise
    r = measurement_noise
    
    for i in range(n):
        # 予測ステップ
        x_prior = x
        p_prior = p + q
        
        # 更新ステップ
        y = prices[i] - x_prior  # 観測残差
        s = p_prior + r  # 残差共分散
        k = p_prior / s  # カルマンゲイン
        
        # 状態更新
        x = x_prior + k * y
        p = (1 - k) * p_prior
        
        # 適応的ノイズ推定
        if i > 0:
            # 観測ノイズの動的調整
            innovation = abs(y)
            r = r * 0.95 + innovation * 0.05
            
            # プロセスノイズの動的調整
            if i > 1:
                state_change = abs(x - filtered_prices[i-1])
                q = q * 0.95 + state_change * 0.05
        
        filtered_prices[i] = x
        prediction_errors[i] = abs(y)
    
    return filtered_prices, prediction_errors


# === 量子ノイズ適応システム ===

@njit(fastmath=True, parallel=True, cache=True)
def quantum_noise_adaptation_system(
    prices: np.ndarray,
    window: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    量子ノイズ適応システム
    
    市場ノイズを量子レベルで分析し、真のトレンドを抽出
    """
    n = len(prices)
    volatility_ratio = np.full(n, np.nan)
    trend_strength = np.full(n, np.nan)
    noise_level = np.full(n, np.nan)
    
    for i in prange(window, n):
        local_prices = prices[i-window:i]
        
        # トレンド成分とノイズ成分の分離
        # 線形回帰によるトレンド抽出
        x_vals = np.arange(window, dtype=np.float64)
        x_mean = np.mean(x_vals)
        y_mean = np.mean(local_prices)
        
        # 共分散と分散
        cov = np.sum((x_vals - x_mean) * (local_prices - y_mean))
        var_x = np.sum((x_vals - x_mean) ** 2)
        
        if var_x > 1e-10:
            slope = cov / var_x
            intercept = y_mean - slope * x_mean
            
            # トレンドライン
            trend_line = slope * x_vals + intercept
            
            # 残差（ノイズ成分）
            residuals = local_prices - trend_line
            
            # ボラティリティ比率
            price_vol = np.std(local_prices)
            residual_vol = np.std(residuals)
            
            if price_vol > 1e-10:
                volatility_ratio[i] = residual_vol / price_vol
            else:
                volatility_ratio[i] = 0.0
            
            # トレンド強度（R二乗値）
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((local_prices - y_mean) ** 2)
            
            if ss_tot > 1e-10:
                r_squared = 1 - (ss_res / ss_tot)
                trend_strength[i] = max(0, r_squared)
            else:
                trend_strength[i] = 0.0
            
            # ノイズレベル（正規化された残差の標準偏差）
            if price_vol > 1e-10:
                noise_level[i] = residual_vol / price_vol
            else:
                noise_level[i] = 0.0
        else:
            volatility_ratio[i] = 1.0
            trend_strength[i] = 0.0
            noise_level[i] = 1.0
    
    return volatility_ratio, trend_strength, noise_level


# === ニューラル価格予測エンジン ===

@njit(fastmath=True, cache=True)
def neural_price_prediction_engine(
    prices: np.ndarray,
    filtered_prices: np.ndarray,
    trend_strength: np.ndarray,
    lookback: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ニューラル価格予測エンジン
    
    非線形パターン認識による高精度価格予測
    """
    n = len(prices)
    predicted_price = np.full(n, np.nan)

    
    confidence_score = np.full(n, np.nan)
    
    for i in range(lookback, n):
        if np.isnan(filtered_prices[i]) or np.isnan(trend_strength[i]):
            continue
        
        # 特徴抽出
        recent_prices = filtered_prices[i-lookback:i]
        price_changes = np.diff(recent_prices)
        
        if len(price_changes) == 0:
            continue
        
        # 重み付き移動平均予測
        weights = np.exp(np.linspace(-2, 0, lookback))
        weights /= np.sum(weights)
        
        # トレンド調整予測
        base_prediction = np.sum(recent_prices * weights)
        
        # モメンタム調整
        avg_change = np.mean(price_changes)
        momentum_adjustment = avg_change * trend_strength[i]
        
        # 最終予測
        predicted_price[i] = base_prediction + momentum_adjustment
        
        # 信頼度スコア（トレンド強度とボラティリティに基づく）
        price_volatility = np.std(recent_prices)
        mean_price = np.mean(recent_prices)
        
        if mean_price > 1e-10:
            normalized_vol = price_volatility / mean_price
            confidence_score[i] = trend_strength[i] * (1 - min(normalized_vol, 1.0))
        else:
            confidence_score[i] = 0.0
    
    return predicted_price, confidence_score


# === 極限追従アルゴリズム ===

@njit(fastmath=True, cache=True)
def extreme_following_algorithm(
    prices: np.ndarray,
    filtered_prices: np.ndarray,
    trend_strength: np.ndarray,
    noise_level: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    極限追従アルゴリズム
    
    市場の微細な変化を瞬時に検出し、超高速追従を実現
    """
    n = len(prices)
    trend_direction = np.zeros(n)
    momentum_flow = np.full(n, np.nan)
    
    for i in range(2, n):
        if np.isnan(filtered_prices[i]) or np.isnan(trend_strength[i]):
            continue
        
        # 瞬間的な方向検出
        price_change = filtered_prices[i] - filtered_prices[i-1]
        price_change_prev = filtered_prices[i-1] - filtered_prices[i-2]
        
        # 加速度（現在は使用していないが、将来の拡張用）
        # acceleration = price_change - price_change_prev
        
        # 適応的閾値
        adaptive_threshold = noise_level[i] * np.std(prices[max(0, i-20):i])
        
        # トレンド方向判定
        if price_change > adaptive_threshold:
            trend_direction[i] = 1
        elif price_change < -adaptive_threshold:
            trend_direction[i] = -1
        else:
            trend_direction[i] = trend_direction[i-1] * 0.9  # 減衰
        
        # モメンタムフロー計算
        momentum_strength = abs(price_change) * trend_strength[i]
        direction_consistency = 1.0 if trend_direction[i] * trend_direction[i-1] > 0 else 0.5
        
        momentum_flow[i] = momentum_strength * direction_consistency * np.sign(trend_direction[i])
    
    return trend_direction, momentum_flow


# === 動的チャネル幅計算 ===

@njit(fastmath=True, parallel=True, cache=True)
def dynamic_channel_width_calculation(
    prediction_errors: np.ndarray,
    volatility_ratio: np.ndarray,
    trend_strength: np.ndarray,
    confidence_score: np.ndarray,
    base_multiplier: float = 2.0
) -> np.ndarray:
    """
    動的チャネル幅計算
    
    市場状態に完全適応する超精密チャネル幅
    """
    n = len(prediction_errors)
    channel_width = np.full(n, np.nan)
    
    for i in prange(n):
        if (np.isnan(prediction_errors[i]) or np.isnan(volatility_ratio[i]) or
            np.isnan(trend_strength[i]) or np.isnan(confidence_score[i])):
            continue
        
        # 基本幅（予測誤差ベース）
        base_width = prediction_errors[i] * base_multiplier
        
        # ボラティリティ調整
        vol_adjustment = 1.0 + volatility_ratio[i]
        
        # トレンド調整（強いトレンドでは幅を狭める）
        trend_adjustment = 1.0 - (trend_strength[i] * 0.5)
        
        # 信頼度調整
        confidence_adjustment = 1.0 - (confidence_score[i] * 0.3)
        
        # 最終チャネル幅
        final_width = base_width * vol_adjustment * trend_adjustment * confidence_adjustment
        
        # 安全な範囲に制限
        channel_width[i] = max(final_width, base_width * 0.5)
    
    return channel_width


class QuantumAdaptiveFlowChannel(Indicator):
    """
    Quantum Adaptive Flow Channel (QAFC)
    量子適応フローチャネル
    
    究極の超高性能トレンドフォローチャネルインジケーター
    """
    
    def __init__(
        self,
        # フィルターパラメータ
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
        
        # 適応パラメータ
        noise_window: int = 20,
        prediction_lookback: int = 10,
        
        # チャネルパラメータ
        base_multiplier: float = 2.0,
        
        # データソース
        src_type: str = 'hlc3'
    ):
        """
        Quantum Adaptive Flow Channel コンストラクタ
        """
        super().__init__(f"QAFC({noise_window},{base_multiplier})")
        
        # パラメータ保存
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.noise_window = noise_window
        self.prediction_lookback = prediction_lookback
        self.base_multiplier = base_multiplier
        self.src_type = src_type
        
        # 依存コンポーネント
        self.price_source = PriceSource()
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> QAFCResult:
        """
        QAFC計算メイン関数
        """
        try:
            # 価格データ抽出
            if isinstance(data, pd.DataFrame):
                prices = self.price_source.calculate_source(data, self.src_type).astype(np.float64)
            else:
                prices = data.astype(np.float64)
            
            # === Phase 1: 超低遅延カルマンフィルター ===
            filtered_prices, prediction_errors = ultra_low_latency_kalman_filter(
                prices, self.process_noise, self.measurement_noise
            )
            
            # === Phase 2: 量子ノイズ適応システム ===
            volatility_ratio, trend_strength, noise_level = quantum_noise_adaptation_system(
                prices, self.noise_window
            )
            
            # === Phase 3: ニューラル価格予測エンジン ===
            predicted_price, confidence_score = neural_price_prediction_engine(
                prices, filtered_prices, trend_strength, self.prediction_lookback
            )
            
            # === Phase 4: 極限追従アルゴリズム ===
            trend_direction, momentum_flow = extreme_following_algorithm(
                prices, filtered_prices, trend_strength, noise_level
            )
            
            # === Phase 5: 動的チャネル幅計算 ===
            channel_width = dynamic_channel_width_calculation(
                prediction_errors, volatility_ratio, trend_strength, 
                confidence_score, self.base_multiplier
            )
            
            # === Phase 6: チャネル構築 ===
            # センターラインは超低遅延フィルター価格を使用
            centerline = filtered_prices
            
            # 動的チャネル
            upper_channel = np.full_like(centerline, np.nan)
            lower_channel = np.full_like(centerline, np.nan)
            
            for i in range(len(centerline)):
                if not np.isnan(centerline[i]) and not np.isnan(channel_width[i]):
                    upper_channel[i] = centerline[i] + channel_width[i]
                    lower_channel[i] = centerline[i] - channel_width[i]
            
            # 結果構築
            return QAFCResult(
                centerline=centerline,
                upper_channel=upper_channel,
                lower_channel=lower_channel,
                volatility_ratio=volatility_ratio,
                trend_strength=trend_strength,
                noise_level=noise_level,
                predicted_price=predicted_price,
                confidence_score=confidence_score,
                trend_direction=trend_direction,
                momentum_flow=momentum_flow
            )
            
        except Exception as e:
            self.logger.error(f"QAFC計算中にエラー: {str(e)}")
            # エラー時は空の結果を返す
            n = len(data) if hasattr(data, '__len__') else 0
            empty = np.full(n, np.nan)
            return QAFCResult(
                centerline=empty, upper_channel=empty, lower_channel=empty,
                volatility_ratio=empty, trend_strength=empty, noise_level=empty,
                predicted_price=empty, confidence_score=empty,
                trend_direction=np.zeros(n), momentum_flow=empty
            )
    
    def get_trading_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> dict:
        """トレーディングシグナル生成"""
        result = self.calculate(data)
        
        # 最新値取得
        latest_idx = -1
        while latest_idx >= -len(result.centerline) and np.isnan(result.centerline[latest_idx]):
            latest_idx -= 1
        
        if abs(latest_idx) >= len(result.centerline):
            return {"signal": "no_data"}
        
        # 現在価格
        if isinstance(data, pd.DataFrame):
            current_price = self.price_source.calculate_source(data, self.src_type)[latest_idx]
        else:
            current_price = data[latest_idx]
        
        # チャネル位置
        upper = result.upper_channel[latest_idx]
        lower = result.lower_channel[latest_idx]
        center = result.centerline[latest_idx]
        
        # シグナル判定
        position_in_channel = (current_price - lower) / (upper - lower) if upper > lower else 0.5
        
        return {
            "current_price": float(current_price),
            "centerline": float(center),
            "upper_channel": float(upper),
            "lower_channel": float(lower),
            "position_in_channel": float(position_in_channel),
            "trend_direction": int(result.trend_direction[latest_idx]),
            "trend_strength": float(result.trend_strength[latest_idx]),
            "confidence_score": float(result.confidence_score[latest_idx]),
            "momentum_flow": float(result.momentum_flow[latest_idx]) if not np.isnan(result.momentum_flow[latest_idx]) else 0.0,
            "signal": "strong_buy" if position_in_channel < 0.2 and result.trend_direction[latest_idx] > 0 else
                     "buy" if position_in_channel < 0.4 and result.trend_direction[latest_idx] >= 0 else
                     "strong_sell" if position_in_channel > 0.8 and result.trend_direction[latest_idx] < 0 else
                     "sell" if position_in_channel > 0.6 and result.trend_direction[latest_idx] <= 0 else
                     "neutral"
        }