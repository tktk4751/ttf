#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultimate Chop Trend V3 - 低遅延・高精度・シンプル版

厳選された5つの最強アルゴリズムによる超高速トレンド/レンジ判定：
🧠 Hilbert Transform - 瞬時位相・周波数解析（最重要・低遅延）
📊 Incremental Regression - 超低遅延統計的トレンド検出
⚡ Multi-timeframe Consensus - EMAベース高速コンセンサス
🌊 Streaming Volatility - リアルタイムボラティリティレジーム
🎯 Zero-lag EMA - 超低遅延移動平均

V2の複雑さを排除し、実証済みの効果的アルゴリズムのみを統合
"""

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Any, Optional, NamedTuple, List
import numpy as np
import pandas as pd
from numba import jit, njit, prange
import traceback
import math
import warnings
warnings.filterwarnings("ignore")

# Base classes
try:
    from .indicator import Indicator
except ImportError:
    class Indicator:
        def __init__(self, name): self.name = name; self.logger = self._get_logger()
        def reset(self): pass
        def _get_logger(self): import logging; return logging.getLogger(self.__class__.__name__)


class UltimateChopTrendV3Result(NamedTuple):
    """Ultimate ChopTrend V3計算結果 - シンプル版"""
    # コア指標
    trend_index: np.ndarray              # 統合トレンド指数（0-1）
    trend_direction: np.ndarray          # 1=アップトレンド, 0=レンジ, -1=ダウントレンド
    trend_strength: np.ndarray           # トレンド強度（0-1）
    confidence_score: np.ndarray         # 予測信頼度（0-1）
    
    # 成分指標
    hilbert_component: np.ndarray        # ヒルベルト変換成分
    regression_component: np.ndarray     # 回帰統計成分
    consensus_component: np.ndarray      # マルチタイムフレーム成分
    volatility_component: np.ndarray     # ボラティリティ成分
    zerollag_component: np.ndarray       # ゼロラグEMA成分
    
    # 現在状態
    current_trend: str
    current_strength: float
    current_confidence: float


@njit(fastmath=True, cache=True)
def hilbert_instantaneous_analysis_v3(prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    ヒルベルト変換による瞬時解析（V3最適化版）
    
    Returns:
        (trend_signal, confidence)
    """
    n = len(prices)
    if n < 16:
        return np.full(n, 0.5), np.full(n, 0.5)
    
    trend_signal = np.full(n, 0.5)
    confidence = np.full(n, 0.5)
    
    for i in range(8, n):
        # 4点ヒルベルト変換（最適化）
        real_part = (prices[i] + prices[i-2] + prices[i-4] + prices[i-6]) * 0.25
        imag_part = (prices[i-1] + prices[i-3] + prices[i-5] + prices[i-7]) * 0.25
        
        # 瞬時振幅
        amplitude = math.sqrt(real_part * real_part + imag_part * imag_part)
        
        # 瞬時位相
        if real_part != 0:
            phase = math.atan2(imag_part, real_part)
        else:
            phase = 0
        
        # トレンド成分（位相の方向性）
        if i >= 15:
            phase_trend = 0.0
            for j in range(7):
                phase_trend += math.sin(phase - j * 0.1)
            phase_trend /= 7.0
            
            # 正規化してシグナルに変換
            trend_signal[i] = math.tanh(phase_trend) * 0.5 + 0.5
            
            # 信頼度（振幅ベース）
            if i > 20:
                avg_amplitude = 0.0
                for j in range(5):
                    if i-j >= 0:
                        past_real = (prices[i-j] + prices[i-j-2] + prices[i-j-4] + prices[i-j-6]) * 0.25
                        past_imag = (prices[i-j-1] + prices[i-j-3] + prices[i-j-5] + prices[i-j-7]) * 0.25
                        avg_amplitude += math.sqrt(past_real * past_real + past_imag * past_imag)
                avg_amplitude /= 5.0
                
                if avg_amplitude > 0:
                    confidence[i] = min(amplitude / avg_amplitude, 2.0) * 0.5
                else:
                    confidence[i] = 0.5
    
    return trend_signal, confidence


@njit(fastmath=True, cache=True)
def incremental_regression_v3(prices: np.ndarray, alpha: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
    """
    増分回帰統計（V3最適化版）
    
    Returns:
        (trend_signal, confidence)
    """
    n = len(prices)
    if n < 3:
        return np.full(n, 0.5), np.full(n, 0.5)
    
    trend_signal = np.full(n, 0.5)
    confidence = np.full(n, 0.5)
    
    # 増分統計
    sum_x = 0.0
    sum_y = 0.0
    sum_xy = 0.0
    sum_x2 = 0.0
    count = 0.0
    
    for i in range(n):
        x_val = float(i)
        y_val = prices[i]
        
        # 指数重み付き更新
        if i > 0:
            decay = 1.0 - alpha
            sum_x = decay * sum_x + alpha * x_val
            sum_y = decay * sum_y + alpha * y_val
            sum_xy = decay * sum_xy + alpha * x_val * y_val
            sum_x2 = decay * sum_x2 + alpha * x_val * x_val
            count = decay * count + alpha
        else:
            sum_x = x_val
            sum_y = y_val
            sum_xy = x_val * y_val
            sum_x2 = x_val * x_val
            count = 1.0
        
        # 回帰係数計算
        if i >= 2 and count > 1e-10:
            x_mean = sum_x / count
            y_mean = sum_y / count
            
            numerator = sum_xy / count - x_mean * y_mean
            denominator = sum_x2 / count - x_mean * x_mean
            
            if abs(denominator) > 1e-10:
                slope = numerator / denominator
                
                # トレンドシグナル
                slope_normalized = math.tanh(slope * 5) * 0.5 + 0.5
                trend_signal[i] = slope_normalized
                
                # 信頼度（R²近似）
                if i > 5:
                    recent_var = 0.0
                    for j in range(min(5, i)):
                        diff = prices[i-j] - y_mean
                        recent_var += diff * diff
                    recent_var /= min(5, i)
                    
                    if recent_var > 1e-10:
                        r_squared = (numerator * numerator) / (denominator * recent_var)
                        confidence[i] = min(r_squared, 1.0)
    
    return trend_signal, confidence


@njit(fastmath=True, cache=True)
def multi_timeframe_consensus_v3(
    prices: np.ndarray,
    fast_alpha: float = 0.3,
    medium_alpha: float = 0.15,
    slow_alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    マルチタイムフレームコンセンサス（V3最適化版）
    
    Returns:
        (consensus_signal, agreement_strength)
    """
    n = len(prices)
    if n < 2:
        return np.full(n, 0.5), np.full(n, 0.5)
    
    consensus_signal = np.full(n, 0.5)
    agreement_strength = np.full(n, 0.5)
    
    # 適応EMA
    fast_ema = np.zeros(n)
    medium_ema = np.zeros(n)
    slow_ema = np.zeros(n)
    
    fast_ema[0] = prices[0]
    medium_ema[0] = prices[0]
    slow_ema[0] = prices[0]
    
    for i in range(1, n):
        # EMA更新
        fast_ema[i] = fast_alpha * prices[i] + (1 - fast_alpha) * fast_ema[i-1]
        medium_ema[i] = medium_alpha * prices[i] + (1 - medium_alpha) * medium_ema[i-1]
        slow_ema[i] = slow_alpha * prices[i] + (1 - slow_alpha) * slow_ema[i-1]
        
        # 各タイムフレームの方向
        fast_dir = 1 if fast_ema[i] > fast_ema[i-1] else -1
        medium_dir = 1 if medium_ema[i] > medium_ema[i-1] else -1
        slow_dir = 1 if slow_ema[i] > slow_ema[i-1] else -1
        
        # 重み付きコンセンサス
        consensus = fast_dir * 0.5 + medium_dir * 0.3 + slow_dir * 0.2
        
        # シグナル正規化
        if abs(consensus) > 0.6:
            consensus_signal[i] = 0.5 + consensus * 0.4  # 0.1-0.9の範囲
        else:
            consensus_signal[i] = 0.5  # レンジ
        
        # 一致度計算
        agreements = 0
        if fast_dir == medium_dir:
            agreements += 1
        if medium_dir == slow_dir:
            agreements += 1
        if fast_dir == slow_dir:
            agreements += 1
        
        agreement_strength[i] = agreements / 3.0
    
    return consensus_signal, agreement_strength


@njit(fastmath=True, cache=True)
def streaming_volatility_v3(prices: np.ndarray, alpha: float = 0.12) -> Tuple[np.ndarray, np.ndarray]:
    """
    ストリーミングボラティリティ分析（V3最適化版）
    
    Returns:
        (volatility_signal, regime_strength)
    """
    n = len(prices)
    if n < 2:
        return np.full(n, 0.5), np.full(n, 0.5)
    
    volatility_signal = np.full(n, 0.5)
    regime_strength = np.full(n, 0.5)
    
    # オンライン統計
    running_mean = prices[0]
    running_var = 1.0
    
    for i in range(1, n):
        # 価格変化
        price_change = prices[i] - prices[i-1]
        
        # オンライン平均・分散更新
        delta = price_change - running_mean
        running_mean += alpha * delta
        running_var = (1 - alpha) * running_var + alpha * delta * delta
        running_var = max(running_var, 1e-10)
        
        # ボラティリティレジーム判定
        if running_var > 0:
            z_score = abs(delta) / math.sqrt(running_var)
            
            # レジーム分類
            if z_score > 2.0:  # 高ボラティリティ
                volatility_signal[i] = 0.8
                regime_strength[i] = min(z_score / 3.0, 1.0)
            elif z_score > 1.0:  # 中ボラティリティ
                volatility_signal[i] = 0.6
                regime_strength[i] = z_score / 2.0
            else:  # 低ボラティリティ
                volatility_signal[i] = 0.3
                regime_strength[i] = 0.5 - z_score / 2.0
    
    return volatility_signal, regime_strength


@njit(fastmath=True, cache=True)
def zero_lag_ema_v3(prices: np.ndarray, period: int = 14) -> Tuple[np.ndarray, np.ndarray]:
    """
    Zero-lag EMA（V3最適化版）
    
    Returns:
        (zerollag_signal, momentum_strength)
    """
    n = len(prices)
    if n < period:
        return np.full(n, 0.5), np.full(n, 0.5)
    
    alpha = 2.0 / (period + 1)
    
    zerollag_signal = np.full(n, 0.5)
    momentum_strength = np.full(n, 0.5)
    
    ema1 = np.zeros(n)
    ema2 = np.zeros(n)
    
    ema1[0] = prices[0]
    ema2[0] = prices[0]
    
    for i in range(1, n):
        # 標準EMA
        ema1[i] = alpha * prices[i] + (1 - alpha) * ema1[i-1]
        # EMAのEMA
        ema2[i] = alpha * ema1[i] + (1 - alpha) * ema2[i-1]
        
        # Zero-lag計算
        zlema = 2 * ema1[i] - ema2[i]
        
        # トレンドシグナル
        if i >= period:
            price_vs_zlema = (prices[i] - zlema) / max(abs(zlema), 1e-10)
            zerollag_signal[i] = math.tanh(price_vs_zlema * 3) * 0.5 + 0.5
            
            # モメンタム強度
            if i > period:
                momentum = (zlema - ema1[i-period//2]) / max(abs(ema1[i-period//2]), 1e-10)
                momentum_strength[i] = min(abs(momentum) * 5, 1.0)
    
    return zerollag_signal, momentum_strength


@njit(fastmath=True, cache=True)
def ensemble_integration_v3(
    hilbert_sig: np.ndarray, hilbert_conf: np.ndarray,
    regression_sig: np.ndarray, regression_conf: np.ndarray,
    consensus_sig: np.ndarray, consensus_conf: np.ndarray,
    volatility_sig: np.ndarray, volatility_conf: np.ndarray,
    zerollag_sig: np.ndarray, zerollag_conf: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    アンサンブル統合（V3最適化版）
    
    Returns:
        (integrated_signal, integrated_confidence)
    """
    n = len(hilbert_sig)
    integrated_signal = np.full(n, 0.5)
    integrated_confidence = np.full(n, 0.5)
    
    # 重み配分（効果実証済み）
    weights = np.array([0.35, 0.25, 0.20, 0.12, 0.08])  # Hilbert最重要
    
    for i in range(n):
        # 各シグナルの安全な取得
        signals = np.array([
            min(max(hilbert_sig[i] if not np.isnan(hilbert_sig[i]) else 0.5, 0.0), 1.0),
            min(max(regression_sig[i] if not np.isnan(regression_sig[i]) else 0.5, 0.0), 1.0),
            min(max(consensus_sig[i] if not np.isnan(consensus_sig[i]) else 0.5, 0.0), 1.0),
            min(max(volatility_sig[i] if not np.isnan(volatility_sig[i]) else 0.5, 0.0), 1.0),
            min(max(zerollag_sig[i] if not np.isnan(zerollag_sig[i]) else 0.5, 0.0), 1.0)
        ])
        
        confidences = np.array([
            min(max(hilbert_conf[i] if not np.isnan(hilbert_conf[i]) else 0.5, 0.1), 1.0),
            min(max(regression_conf[i] if not np.isnan(regression_conf[i]) else 0.5, 0.1), 1.0),
            min(max(consensus_conf[i] if not np.isnan(consensus_conf[i]) else 0.5, 0.1), 1.0),
            min(max(volatility_conf[i] if not np.isnan(volatility_conf[i]) else 0.5, 0.1), 1.0),
            min(max(zerollag_conf[i] if not np.isnan(zerollag_conf[i]) else 0.5, 0.1), 1.0)
        ])
        
        # 信頼度重み付き統合
        effective_weights = weights * confidences
        weight_sum = np.sum(effective_weights)
        
        if weight_sum > 1e-10:
            integrated_signal[i] = np.sum(signals * effective_weights) / weight_sum
            integrated_confidence[i] = np.mean(confidences)
        else:
            integrated_signal[i] = 0.5
            integrated_confidence[i] = 0.5
        
        # 範囲制限
        integrated_signal[i] = min(max(integrated_signal[i], 0.0), 1.0)
        integrated_confidence[i] = min(max(integrated_confidence[i], 0.0), 1.0)
    
    return integrated_signal, integrated_confidence


@njit(fastmath=True, cache=True)
def calculate_trend_classification_v3(
    trend_index: np.ndarray,
    confidence: np.ndarray,
    prices: np.ndarray,
    trend_threshold: float = 0.58,
    confidence_threshold: float = 0.3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    トレンド分類（V3実践版）
    
    Returns:
        (trend_direction, trend_strength)
    """
    n = len(trend_index)
    trend_direction = np.zeros(n, dtype=np.int8)
    trend_strength = np.zeros(n)
    
    # 実践的しきい値
    upper_threshold = trend_threshold
    lower_threshold = 1.0 - trend_threshold
    strong_upper = 0.75
    strong_lower = 0.25
    
    for i in range(n):
        signal = trend_index[i] if not np.isnan(trend_index[i]) else 0.5
        conf = confidence[i] if not np.isnan(confidence[i]) else 0.5
        
        # トレンド強度計算
        trend_strength[i] = abs(signal - 0.5) * 2.0 * conf
        
        # 方向判定（信頼度考慮）
        if conf > confidence_threshold:
            if signal > strong_upper:
                trend_direction[i] = 1  # 強い上昇
            elif signal > upper_threshold:
                trend_direction[i] = 1  # 上昇
            elif signal < strong_lower:
                trend_direction[i] = -1  # 強い下降
            elif signal < lower_threshold:
                trend_direction[i] = -1  # 下降
            else:
                trend_direction[i] = 0  # レンジ
        else:
            trend_direction[i] = 0  # 低信頼度はレンジ
    
    return trend_direction, trend_strength


class UltimateChopTrendV3(Indicator):
    """
    Ultimate Chop Trend V3 - 低遅延・高精度・シンプル版
    
    厳選された5つの最強アルゴリズム：
    🧠 Hilbert Transform - 瞬時位相・周波数解析（最重要・低遅延）
    📊 Incremental Regression - 超低遅延統計的トレンド検出
    ⚡ Multi-timeframe Consensus - EMAベース高速コンセンサス
    🌊 Streaming Volatility - リアルタイムボラティリティレジーム
    🎯 Zero-lag EMA - 超低遅延移動平均
    
    V2の複雑さを排除し、実証済みの効果的アルゴリズムのみを統合
    """
    
    def __init__(
        self,
        # コアパラメータ
        analysis_period: int = 14,  # V2より短縮
        fast_period: int = 7,
        
        # しきい値（実践的）
        trend_threshold: float = 0.58,  # V2より敏感
        confidence_threshold: float = 0.3,  # V2より低い
        
        # アルゴリズム有効化（全て軽量で効果的）
        enable_hilbert: bool = True,
        enable_regression: bool = True,
        enable_consensus: bool = True,
        enable_volatility: bool = True,
        enable_zerollag: bool = True
    ):
        """
        Ultimate ChopTrend V3 - 低遅延・高精度・シンプル版
        
        Args:
            analysis_period: 分析期間（短縮で低遅延化）
            fast_period: 高速期間
            trend_threshold: トレンド判定しきい値（実践的）
            confidence_threshold: 信頼度しきい値（実践的）
            enable_*: 各アルゴリズムの有効化フラグ
        """
        super().__init__(f"UltimateChopTrendV3(P={analysis_period})")
        
        self.analysis_period = analysis_period
        self.fast_period = fast_period
        self.trend_threshold = trend_threshold
        self.confidence_threshold = confidence_threshold
        
        # アルゴリズム有効化
        self.enable_hilbert = enable_hilbert
        self.enable_regression = enable_regression
        self.enable_consensus = enable_consensus
        self.enable_volatility = enable_volatility
        self.enable_zerollag = enable_zerollag
        
        self._result: Optional[UltimateChopTrendV3Result] = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> UltimateChopTrendV3Result:
        """
        Ultimate ChopTrend V3を計算
        """
        try:
            # データ準備
            if len(data) == 0:
                return self._create_empty_result(0)
            
            if isinstance(data, pd.DataFrame):
                prices = np.asarray(data['close'].values, dtype=np.float64)
            else:
                prices = np.asarray(data[:, 3], dtype=np.float64)
            
            n = len(prices)
            
            # 各アルゴリズムの実行
            components = {}
            confidences = {}
            
            # 1. ヒルベルト変換解析（最重要）
            if self.enable_hilbert:
                hilbert_sig, hilbert_conf = hilbert_instantaneous_analysis_v3(prices)
                components['hilbert'] = hilbert_sig
                confidences['hilbert'] = hilbert_conf
            else:
                components['hilbert'] = np.full(n, 0.5)
                confidences['hilbert'] = np.full(n, 0.5)
            
            # 2. 増分回帰統計
            if self.enable_regression:
                regression_sig, regression_conf = incremental_regression_v3(prices)
                components['regression'] = regression_sig
                confidences['regression'] = regression_conf
            else:
                components['regression'] = np.full(n, 0.5)
                confidences['regression'] = np.full(n, 0.5)
            
            # 3. マルチタイムフレームコンセンサス
            if self.enable_consensus:
                consensus_sig, consensus_conf = multi_timeframe_consensus_v3(prices)
                components['consensus'] = consensus_sig
                confidences['consensus'] = consensus_conf
            else:
                components['consensus'] = np.full(n, 0.5)
                confidences['consensus'] = np.full(n, 0.5)
            
            # 4. ストリーミングボラティリティ
            if self.enable_volatility:
                volatility_sig, volatility_conf = streaming_volatility_v3(prices)
                components['volatility'] = volatility_sig
                confidences['volatility'] = volatility_conf
            else:
                components['volatility'] = np.full(n, 0.5)
                confidences['volatility'] = np.full(n, 0.5)
            
            # 5. Zero-lag EMA
            if self.enable_zerollag:
                zerollag_sig, zerollag_conf = zero_lag_ema_v3(prices, self.analysis_period)
                components['zerollag'] = zerollag_sig
                confidences['zerollag'] = zerollag_conf
            else:
                components['zerollag'] = np.full(n, 0.5)
                confidences['zerollag'] = np.full(n, 0.5)
            
            # アンサンブル統合
            trend_index, confidence_score = ensemble_integration_v3(
                components['hilbert'], confidences['hilbert'],
                components['regression'], confidences['regression'],
                components['consensus'], confidences['consensus'],
                components['volatility'], confidences['volatility'],
                components['zerollag'], confidences['zerollag']
            )
            
            # トレンド分類
            trend_direction, trend_strength = calculate_trend_classification_v3(
                trend_index, confidence_score, prices,
                self.trend_threshold, self.confidence_threshold
            )
            
            # 現在状態の判定
            latest_direction = trend_direction[-1] if len(trend_direction) > 0 else 0
            latest_strength = trend_strength[-1] if len(trend_strength) > 0 else 0
            latest_confidence = confidence_score[-1] if len(confidence_score) > 0 else 0
            
            if latest_direction > 0:
                if latest_strength > 0.7:
                    current_trend = "strong_uptrend"
                else:
                    current_trend = "uptrend"
            elif latest_direction < 0:
                if latest_strength > 0.7:
                    current_trend = "strong_downtrend"
                else:
                    current_trend = "downtrend"
            else:
                current_trend = "range"
            
            # 結果作成
            result = UltimateChopTrendV3Result(
                trend_index=trend_index,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                confidence_score=confidence_score,
                hilbert_component=components['hilbert'],
                regression_component=components['regression'],
                consensus_component=components['consensus'],
                volatility_component=components['volatility'],
                zerollag_component=components['zerollag'],
                current_trend=current_trend,
                current_strength=latest_strength,
                current_confidence=latest_confidence
            )
            
            self._result = result
            self._values = trend_index
            
            return result
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.logger.error(f"UltimateChopTrendV3計算中にエラー: {e}\n詳細:\n{error_details}")
            return self._create_empty_result(len(data))
    
    def _create_empty_result(self, length: int) -> UltimateChopTrendV3Result:
        """空の結果を作成"""
        return UltimateChopTrendV3Result(
            trend_index=np.full(length, np.nan),
            trend_direction=np.zeros(length),
            trend_strength=np.zeros(length),
            confidence_score=np.zeros(length),
            hilbert_component=np.full(length, np.nan),
            regression_component=np.full(length, np.nan),
            consensus_component=np.full(length, np.nan),
            volatility_component=np.full(length, np.nan),
            zerollag_component=np.full(length, np.nan),
            current_trend="range",
            current_strength=0.0,
            current_confidence=0.0
        )
    
    def get_values(self) -> Optional[np.ndarray]:
        """メイン指標値を取得"""
        if self._result is not None:
            return self._result.trend_index.copy()
        return None
    
    def get_result(self) -> Optional[UltimateChopTrendV3Result]:
        """完全な結果を取得"""
        return self._result
    
    def reset(self) -> None:
        """リセット"""
        super().reset()
        self._result = None 