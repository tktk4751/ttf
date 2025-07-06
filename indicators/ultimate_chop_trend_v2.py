#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultimate Chop Trend V2 - シンプル化された最強トレンド/レンジ判定インジケーター

複雑な実装を見直し、本当に効果的な5つのコアアルゴリズムのみを厳選：
🧠 Hilbert Transform - 瞬時位相・周波数解析（最も重要）
📐 Adaptive Kalman - 動的ノイズフィルタリング（統合版）
🌊 Spectral Regime - 周波数ドメイン市場レジーム検出
🎯 Fractal Momentum - フラクタル適応モメンタム
⚡ Smart Volatility - インテリジェントボラティリティ分析

シンプルさと効果性の両立を目指した次世代システム
"""

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Any, Optional, NamedTuple
import numpy as np
import pandas as pd
from numba import jit, njit
import math
import warnings
warnings.filterwarnings("ignore")

# Base classes
try:
    from .indicator import Indicator
except ImportError:
    class Indicator:
        def __init__(self, name): 
            self.name = name
        def reset(self): pass
        def _get_logger(self): 
            import logging
            return logging.getLogger(self.__class__.__name__)


class UltimateChopTrendV2Result(NamedTuple):
    """Ultimate ChopTrend V2 計算結果 - シンプル強力版"""
    # コア指標
    trend_index: np.ndarray           # 統合トレンド指数（0-1）
    trend_direction: np.ndarray       # 1=上昇, 0=レンジ, -1=下降
    trend_strength: np.ndarray        # トレンド強度（0-1）
    confidence_score: np.ndarray      # 信頼度（0-1）
    
    # 市場状態
    regime_state: np.ndarray          # レジーム（0=レンジ, 1=トレンド, 2=ブレイクアウト）
    volatility_regime: np.ndarray     # ボラティリティレジーム（0=低, 1=中, 2=高）
    
    # 成分分析
    hilbert_component: np.ndarray     # ヒルベルト変換成分
    kalman_component: np.ndarray      # カルマンフィルター成分
    spectral_component: np.ndarray    # スペクトル成分
    fractal_component: np.ndarray     # フラクタル成分
    volatility_component: np.ndarray  # ボラティリティ成分
    
    # 現在状態
    current_trend: str
    current_strength: float
    current_confidence: float


@njit(fastmath=True, cache=True)
def hilbert_instantaneous_analysis(prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    ヒルベルト変換による瞬時解析（高速版）
    """
    n = len(prices)
    if n < 20:
        return np.full(n, 0.5), np.full(n, 0.5)
    
    trend_signal = np.zeros(n)
    confidence = np.zeros(n)
    
    # 改良版ヒルベルト変換近似
    for i in range(14, n):
        # 価格系列を平滑化
        smoothed = np.mean(prices[i-7:i+1])
        
        # 瞬時位相の近似計算
        real_part = 0.0
        imag_part = 0.0
        
        # 4ポイントヒルベルト変換
        for j in range(4):
            idx = i - j * 2
            if idx >= 0:
                weight = 1.0 / (j + 1)
                real_part += prices[idx] * weight
                if idx - 1 >= 0:
                    imag_part += prices[idx-1] * weight
        
        # 瞬時振幅と位相
        amplitude = math.sqrt(real_part * real_part + imag_part * imag_part)
        
        if amplitude > 1e-10:
            phase = math.atan2(imag_part, real_part)
            
            # トレンド方向の判定
            phase_normalized = (phase + math.pi) / (2 * math.pi)  # 0-1正規化
            trend_signal[i] = phase_normalized
            
            # 信頼度（振幅ベース）
            avg_price = np.mean(prices[max(0, i-20):i+1])
            if avg_price > 0:
                confidence[i] = min(amplitude / avg_price * 10, 1.0)
            else:
                confidence[i] = 0.5
        else:
            trend_signal[i] = 0.5
            confidence[i] = 0.1
    
    return trend_signal, confidence


@njit(fastmath=True, cache=True)
def adaptive_kalman_filter(prices: np.ndarray, volatility: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    適応カルマンフィルター（統合版）
    """
    n = len(prices)
    if n < 5:
        return prices.copy(), np.ones(n)
    
    filtered_prices = np.zeros(n)
    confidence = np.zeros(n)
    
    # 状態変数：[価格, 速度]
    state = np.array([prices[0], 0.0])
    P = np.array([[1.0, 0.0], [0.0, 1.0]])  # 共分散行列
    
    # システムモデル
    F = np.array([[1.0, 1.0], [0.0, 0.95]])  # 状態遷移
    Q = np.array([[0.01, 0.0], [0.0, 0.01]])  # プロセスノイズ
    H = np.array([1.0, 0.0])  # 観測行列
    
    filtered_prices[0] = prices[0]
    confidence[0] = 1.0
    
    for i in range(1, n):
        # 予測ステップ
        state_pred = np.dot(F, state)
        P_pred = np.dot(np.dot(F, P), F.T) + Q
        
        # 観測ノイズ（適応的）
        R = max(volatility[i] ** 2, 1e-6)
        
        # 更新ステップ
        innovation = prices[i] - np.dot(H, state_pred)
        S = np.dot(np.dot(H, P_pred), H.T) + R
        
        if S > 1e-10:
            K = np.dot(P_pred, H.T) / S
            state = state_pred + K * innovation
            P = P_pred - np.outer(K, np.dot(H, P_pred))
        else:
            state = state_pred
        
        filtered_prices[i] = state[0]
        confidence[i] = 1.0 / (1.0 + P[0, 0])
    
    return filtered_prices, confidence


@njit(fastmath=True, cache=True)
def spectral_regime_detector(prices: np.ndarray, window: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """
    スペクトル解析による市場レジーム検出（高速版）
    """
    n = len(prices)
    if n < window * 2:
        return np.full(n, 0.5), np.full(n, 0.5)
    
    regime_signal = np.zeros(n)
    spectral_power = np.zeros(n)
    
    for i in range(window, n):
        segment = prices[i-window:i]
        
        # 線形トレンド除去
        x_vals = np.arange(window)
        sum_x = np.sum(x_vals)
        sum_y = np.sum(segment)
        sum_xy = np.sum(x_vals * segment)
        sum_x2 = np.sum(x_vals * x_vals)
        
        denom = window * sum_x2 - sum_x * sum_x
        if abs(denom) > 1e-10:
            slope = (window * sum_xy - sum_x * sum_y) / denom
            intercept = (sum_y - slope * sum_x) / window
            detrended = segment - (slope * x_vals + intercept)
        else:
            detrended = segment - np.mean(segment)
        
        # 簡易DFT（主要周波数のみ）
        max_power = 0.0
        total_power = 0.0
        dominant_freq = 0.0
        
        for k in range(1, min(window // 4, 8)):  # 低周波数のみ
            real_sum = 0.0
            imag_sum = 0.0
            
            for j in range(window):
                angle = -2.0 * math.pi * k * j / window
                real_sum += detrended[j] * math.cos(angle)
                imag_sum += detrended[j] * math.sin(angle)
            
            power = real_sum * real_sum + imag_sum * imag_sum
            total_power += power
            
            if power > max_power:
                max_power = power
                dominant_freq = k / window
        
        # レジーム判定
        if total_power > 1e-10:
            spectral_ratio = max_power / total_power
            
            # 高い周波数集中度 = トレンド、低い = レンジ
            if spectral_ratio > 0.6:  # 支配的周波数あり
                regime_signal[i] = dominant_freq  # 周波数を信号に変換
            else:  # 周波数が分散（レンジ）
                regime_signal[i] = 0.5
            
            spectral_power[i] = math.sqrt(total_power) / window
        else:
            regime_signal[i] = 0.5
            spectral_power[i] = 0.0
    
    return regime_signal, spectral_power


@njit(fastmath=True, cache=True)
def fractal_momentum_analyzer(prices: np.ndarray, period: int = 14) -> Tuple[np.ndarray, np.ndarray]:
    """
    フラクタル適応モメンタム分析（改良版）
    """
    n = len(prices)
    if n < period * 2:
        return np.full(n, 0.5), np.full(n, 0.5)
    
    momentum_signal = np.zeros(n)
    fractal_dimension = np.zeros(n)
    
    for i in range(period, n):
        # フラクタル次元の計算
        price_range = np.max(prices[i-period:i]) - np.min(prices[i-period:i])
        
        if price_range > 1e-10:
            # 価格変動の複雑さ測定
            total_variation = 0.0
            for j in range(i-period+1, i):
                total_variation += abs(prices[j] - prices[j-1])
            
            # フラクタル次元の近似
            if total_variation > 1e-10:
                dimension = math.log(total_variation / price_range) / math.log(period)
                dimension = min(max(dimension, 1.0), 2.0)
            else:
                dimension = 1.5
        else:
            dimension = 1.5
        
        fractal_dimension[i] = dimension
        
        # 適応係数の計算
        alpha = 2.0 / (dimension + 1.0)  # 低次元ほど高いアルファ
        
        # 適応モメンタム
        if i >= period + 5:
            price_change = prices[i] - prices[i-5]
            volatility = np.std(prices[i-period:i])
            
            if volatility > 1e-10:
                normalized_momentum = price_change / volatility
                # 適応的正規化
                momentum_signal[i] = alpha * math.tanh(normalized_momentum) + (1-alpha) * 0.5
                momentum_signal[i] = momentum_signal[i] * 0.5 + 0.5  # 0-1範囲
            else:
                momentum_signal[i] = 0.5
        else:
            momentum_signal[i] = 0.5
    
    return momentum_signal, fractal_dimension


@njit(fastmath=True, cache=True)
def smart_volatility_analyzer(prices: np.ndarray, fast_period: int = 7, slow_period: int = 21) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    インテリジェントボラティリティ分析
    """
    n = len(prices)
    if n < slow_period:
        return np.full(n, 0.5), np.full(n, 1.0), np.full(n, 0.01)
    
    volatility_signal = np.zeros(n)
    volatility_regime = np.zeros(n)  # float64として統一
    volatility_values = np.zeros(n)
    
    # 高速・低速ボラティリティ
    fast_vol = np.zeros(n)
    slow_vol = np.zeros(n)
    
    for i in range(1, n):
        # 価格変化率
        return_val = (prices[i] - prices[i-1]) / max(prices[i-1], 1e-10)
        
        # 指数移動平均ボラティリティ
        if i == 1:
            fast_vol[i] = abs(return_val)
            slow_vol[i] = abs(return_val)
        else:
            alpha_fast = 2.0 / (fast_period + 1)
            alpha_slow = 2.0 / (slow_period + 1)
            
            fast_vol[i] = alpha_fast * abs(return_val) + (1 - alpha_fast) * fast_vol[i-1]
            slow_vol[i] = alpha_slow * abs(return_val) + (1 - alpha_slow) * slow_vol[i-1]
        
        volatility_values[i] = fast_vol[i]
        
        # ボラティリティレジーム
        if i >= slow_period:
            vol_ratio = fast_vol[i] / max(slow_vol[i], 1e-10)
            
            if vol_ratio > 1.5:
                volatility_regime[i] = 2.0  # 高ボラティリティ
                volatility_signal[i] = 0.8
            elif vol_ratio > 1.1:
                volatility_regime[i] = 1.0  # 中ボラティリティ
                volatility_signal[i] = 0.6
            else:
                volatility_regime[i] = 0.0  # 低ボラティリティ
                volatility_signal[i] = 0.3
        else:
            volatility_signal[i] = 0.5
            volatility_regime[i] = 1.0
    
    return volatility_signal, volatility_regime, volatility_values


@njit(fastmath=True, cache=True)
def smart_ensemble_system(
    hilbert_sig: np.ndarray,
    kalman_sig: np.ndarray,
    spectral_sig: np.ndarray,
    fractal_sig: np.ndarray,
    volatility_sig: np.ndarray,
    confidence_weights: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    スマートアンサンブルシステム（実践的バランス版）
    """
    n = len(hilbert_sig)
    ensemble_signal = np.zeros(n)
    ensemble_confidence = np.zeros(n)
    
    # 各成分の重み（トレンド検出重視）
    weights = np.array([0.35, 0.25, 0.20, 0.12, 0.08])  # ヒルベルトとカルマンを強化
    
    for i in range(n):
        # 各シグナルの安全な取得
        signals = np.array([
            min(max(hilbert_sig[i] if np.isfinite(hilbert_sig[i]) else 0.5, 0.0), 1.0),
            min(max(kalman_sig[i] if np.isfinite(kalman_sig[i]) else 0.5, 0.0), 1.0),
            min(max(spectral_sig[i] if np.isfinite(spectral_sig[i]) else 0.5, 0.0), 1.0),
            min(max(fractal_sig[i] if np.isfinite(fractal_sig[i]) else 0.5, 0.0), 1.0),
            min(max(volatility_sig[i] if np.isfinite(volatility_sig[i]) else 0.5, 0.0), 1.0)
        ])
        
        # 信頼度重み
        conf_weight = confidence_weights[i] if i < len(confidence_weights) and np.isfinite(confidence_weights[i]) else 1.0
        conf_weight = min(max(conf_weight, 0.1), 2.0)
        
        # 重み付き平均
        effective_weights = weights * conf_weight
        ensemble_signal[i] = np.sum(signals * effective_weights) / np.sum(effective_weights)
        
        # より敏感な信頼度計算（トレンド検出重視）
        signal_variance = np.var(signals)
        signal_deviation = abs(ensemble_signal[i] - 0.5) * 2.0  # 中央からの距離
        
        # 基本信頼度
        base_confidence = 1.0 / (1.0 + signal_variance * 3.0)  # 感度を上げる
        
        # トレンド信号の場合、信頼度をブースト
        if signal_deviation > 0.15:  # トレンド傾向がある場合
            base_confidence *= (1.0 + signal_deviation * 0.5)
        
        ensemble_confidence[i] = min(max(base_confidence, 0.15), 1.0)
    
    return ensemble_signal, ensemble_confidence


@njit(fastmath=True, cache=True)
def calculate_trend_classification(
    trend_index: np.ndarray,
    confidence: np.ndarray,
    volatility_regime: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    トレンド分類システム（実践的バランス版）
    """
    n = len(trend_index)
    trend_direction = np.zeros(n)
    trend_strength = np.zeros(n)
    regime_state = np.zeros(n)
    
    for i in range(n):
        signal = trend_index[i]
        conf = confidence[i]
        vol_regime = volatility_regime[i] if i < len(volatility_regime) else 1
        
        # トレンド強度（中央値からの距離）
        base_strength = abs(signal - 0.5) * 2.0
        
        # 信頼度とボラティリティで調整
        adjusted_strength = base_strength * conf
        if vol_regime == 2:  # 高ボラティリティ
            adjusted_strength *= 1.3  # より積極的
        elif vol_regime == 0:  # 低ボラティリティ
            adjusted_strength *= 0.9  # 少し控えめに調整
        
        trend_strength[i] = min(adjusted_strength, 1.0)
        
        # 実践的な方向判定（感度を上げる）
        if signal > 0.58 and trend_strength[i] > 0.25:  # しきい値を下げる
            trend_direction[i] = 1  # 上昇
            regime_state[i] = 2 if trend_strength[i] > 0.55 else 1
        elif signal < 0.42 and trend_strength[i] > 0.25:  # しきい値を下げる
            trend_direction[i] = -1  # 下降
            regime_state[i] = 2 if trend_strength[i] > 0.55 else 1
        else:
            # より厳格なレンジ判定
            if 0.45 <= signal <= 0.55 and trend_strength[i] < 0.35:
                trend_direction[i] = 0  # レンジ
                regime_state[i] = 0
            else:
                # 弱いトレンドとして分類
                trend_direction[i] = 1 if signal > 0.5 else -1
                regime_state[i] = 1
    
    return trend_direction, trend_strength, regime_state


class UltimateChopTrendV2(Indicator):
    """
    Ultimate Chop Trend V2 - シンプル化された最強システム
    
    【厳選された5つのコアアルゴリズム】
    🧠 Hilbert Transform - 瞬時位相・周波数解析（最重要）
    📐 Adaptive Kalman - 統合カルマンフィルター
    🌊 Spectral Regime - 周波数ドメイン市場レジーム検出
    🎯 Fractal Momentum - フラクタル適応モメンタム
    ⚡ Smart Volatility - インテリジェントボラティリティ分析
    
    複雑さを排除し、効果的なアルゴリズムのみを統合
    """
    
    def __init__(
        self,
        # コアパラメータ
        analysis_period: int = 21,
        fast_period: int = 7,
        slow_period: int = 21,
        
        # アルゴリズム有効化
        enable_hilbert: bool = True,
        enable_kalman: bool = True,
        enable_spectral: bool = True,
        enable_fractal: bool = True,
        enable_volatility: bool = True,
        
        # しきい値（実践的に調整）
        trend_threshold: float = 0.55,  # より敏感に
        confidence_threshold: float = 0.3  # より低いしきい値
    ):
        """
        Ultimate ChopTrend V2 - シンプル強力版
        
        Args:
            analysis_period: 分析期間
            fast_period: 高速期間
            slow_period: 低速期間
            enable_*: 各アルゴリズムの有効化フラグ
            trend_threshold: トレンド判定しきい値
            confidence_threshold: 信頼度しきい値
        """
        super().__init__(f"UltimateChopTrendV2(P={analysis_period})")
        
        self.analysis_period = analysis_period
        self.fast_period = fast_period
        self.slow_period = slow_period
        
        # アルゴリズム有効化
        self.enable_hilbert = enable_hilbert
        self.enable_kalman = enable_kalman
        self.enable_spectral = enable_spectral
        self.enable_fractal = enable_fractal
        self.enable_volatility = enable_volatility
        
        # しきい値
        self.trend_threshold = trend_threshold
        self.confidence_threshold = confidence_threshold
        
        self._result: Optional[UltimateChopTrendV2Result] = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> UltimateChopTrendV2Result:
        """
        Ultimate ChopTrend V2を計算
        """
        try:
            # データ準備
            if len(data) == 0:
                return self._create_empty_result(0)
            
            if isinstance(data, pd.DataFrame):
                prices = np.asarray(data['close'].values, dtype=np.float64)
                high = np.asarray(data['high'].values, dtype=np.float64)
                low = np.asarray(data['low'].values, dtype=np.float64)
            else:
                prices = np.asarray(data[:, 3], dtype=np.float64)
                high = np.asarray(data[:, 1], dtype=np.float64)
                low = np.asarray(data[:, 2], dtype=np.float64)
            
            n = len(prices)
            
            # ボラティリティ計算（ATR近似）
            volatility = np.zeros(n)
            for i in range(1, n):
                tr = max(
                    high[i] - low[i],
                    abs(high[i] - prices[i-1]) if i > 0 else high[i] - low[i],
                    abs(low[i] - prices[i-1]) if i > 0 else high[i] - low[i]
                )
                if i < 14:
                    volatility[i] = max(tr, 1e-10)
                else:
                    volatility[i] = max((volatility[i-1] * 13 + tr) / 14, 1e-10)
            volatility[0] = max(high[0] - low[0], 1e-10)
            
            # 各コアアルゴリズムの実行
            components = {}
            confidences = {}
            
            # 1. ヒルベルト変換解析
            if self.enable_hilbert:
                hilbert_sig, hilbert_conf = hilbert_instantaneous_analysis(prices)
                components['hilbert'] = hilbert_sig
                confidences['hilbert'] = hilbert_conf
            else:
                components['hilbert'] = np.full(n, 0.5)
                confidences['hilbert'] = np.full(n, 0.5)
            
            # 2. 適応カルマンフィルター
            if self.enable_kalman:
                kalman_filtered, kalman_conf = adaptive_kalman_filter(prices, volatility)
                # カルマンシグナルの生成
                kalman_sig = np.full(n, 0.5)
                for i in range(5, n):
                    price_change = kalman_filtered[i] - kalman_filtered[i-3]
                    vol_norm = volatility[i] if volatility[i] > 0 else 1e-10
                    normalized_change = price_change / vol_norm
                    kalman_sig[i] = math.tanh(normalized_change) * 0.5 + 0.5
                
                components['kalman'] = kalman_sig
                confidences['kalman'] = kalman_conf
            else:
                components['kalman'] = np.full(n, 0.5)
                confidences['kalman'] = np.full(n, 0.5)
            
            # 3. スペクトル解析レジーム検出
            if self.enable_spectral:
                spectral_sig, spectral_power = spectral_regime_detector(prices, window=32)
                components['spectral'] = spectral_sig
                confidences['spectral'] = np.minimum(spectral_power * 5, 1.0)
            else:
                components['spectral'] = np.full(n, 0.5)
                confidences['spectral'] = np.full(n, 0.5)
            
            # 4. フラクタルモメンタム
            if self.enable_fractal:
                fractal_sig, fractal_dim = fractal_momentum_analyzer(prices, self.analysis_period)
                components['fractal'] = fractal_sig
                # フラクタル次元を信頼度に変換
                fractal_conf = np.abs(fractal_dim - 1.5) * 2  # 1.5から離れるほど高信頼度
                confidences['fractal'] = np.minimum(fractal_conf, 1.0)
            else:
                components['fractal'] = np.full(n, 0.5)
                confidences['fractal'] = np.full(n, 0.5)
            
            # 5. スマートボラティリティ
            if self.enable_volatility:
                vol_sig, vol_regime, vol_values = smart_volatility_analyzer(
                    prices, self.fast_period, self.slow_period
                )
                components['volatility'] = vol_sig
                confidences['volatility'] = np.full(n, 0.7)  # 固定信頼度
            else:
                vol_sig = np.full(n, 0.5)
                vol_regime = np.ones(n)
                vol_values = np.full(n, 0.01)
                components['volatility'] = vol_sig
                confidences['volatility'] = np.full(n, 0.5)
            
            # 統合信頼度重み
            avg_confidence = np.mean([confidences[k] for k in confidences.keys()], axis=0)
            
            # スマートアンサンブル統合
            ensemble_signal, ensemble_confidence = smart_ensemble_system(
                components['hilbert'],
                components['kalman'],
                components['spectral'],
                components['fractal'],
                components['volatility'],
                avg_confidence
            )
            
            # トレンド分類
            trend_direction, trend_strength, regime_state = calculate_trend_classification(
                ensemble_signal, ensemble_confidence, vol_regime
            )
            
            # 現在状態判定
            if len(trend_direction) > 0:
                latest_dir = trend_direction[-1]
                latest_strength = trend_strength[-1]
                latest_confidence = ensemble_confidence[-1]
                
                if latest_dir > 0:
                    current_trend = "strong_uptrend" if latest_strength > 0.7 else "uptrend"
                elif latest_dir < 0:
                    current_trend = "strong_downtrend" if latest_strength > 0.7 else "downtrend"
                else:
                    current_trend = "range"
            else:
                current_trend = "range"
                latest_strength = 0.0
                latest_confidence = 0.0
            
            # 結果作成
            result = UltimateChopTrendV2Result(
                trend_index=ensemble_signal,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                confidence_score=ensemble_confidence,
                regime_state=regime_state,
                volatility_regime=vol_regime,
                hilbert_component=components['hilbert'],
                kalman_component=components['kalman'],
                spectral_component=components['spectral'],
                fractal_component=components['fractal'],
                volatility_component=components['volatility'],
                current_trend=current_trend,
                current_strength=latest_strength,
                current_confidence=latest_confidence
            )
            
            self._result = result
            return result
            
        except Exception as e:
            self.logger.error(f"UltimateChopTrendV2計算エラー: {e}")
            return self._create_empty_result(len(data))
    
    def _create_empty_result(self, length: int) -> UltimateChopTrendV2Result:
        """空の結果を作成"""
        return UltimateChopTrendV2Result(
            trend_index=np.full(length, 0.5),
            trend_direction=np.zeros(length),
            trend_strength=np.zeros(length),
            confidence_score=np.zeros(length),
            regime_state=np.zeros(length),
            volatility_regime=np.ones(length),
            hilbert_component=np.full(length, 0.5),
            kalman_component=np.full(length, 0.5),
            spectral_component=np.full(length, 0.5),
            fractal_component=np.full(length, 0.5),
            volatility_component=np.full(length, 0.5),
            current_trend="range",
            current_strength=0.0,
            current_confidence=0.0
        )
    
    def get_values(self) -> Optional[np.ndarray]:
        """メイン指標値を取得"""
        if self._result is not None:
            return self._result.trend_index.copy()
        return None
    
    def get_result(self) -> Optional[UltimateChopTrendV2Result]:
        """完全な結果を取得"""
        return self._result
    
    def reset(self) -> None:
        """リセット"""
        super().reset()
        self._result = None 