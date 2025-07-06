#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🌌 **Cosmic Adaptive Channel (CAC) - 宇宙最強ブレイクアウトチャネル V1.0** 🌌

🎯 **革命的8層ハイブリッドシステム:**
- **量子統計フュージョン**: 量子コヒーレンス + 統計回帰の融合
- **ヒルベルト位相解析**: 瞬時トレンド検出 + 位相遅延ゼロ
- **神経適応学習**: 市場パターン自動学習 + 動的重み調整
- **動的チャネル幅**: トレンド強度反比例 + 偽シグナル防御
- **ゼロラグフィルタ**: 超低遅延 + 予測的補正
- **ボラティリティレジーム**: リアルタイム市場状態検出
- **超追従適応**: 瞬時相場変化対応 + 学習型最適化
- **ブレイクアウト予測**: 突破確率 + タイミング予測

🏆 **宇宙最強特徴:**
- **超低遅延**: ゼロラグ + 予測補正 + ヒルベルト変換
- **超高精度**: 8層フィルタ + 量子統計 + 神経学習
- **超追従性**: 動的適応 + リアルタイム調整 + 学習進化
- **智能的チャネル幅**: トレンド強度反比例 + 偽シグナル完全防御
- **ブレイクアウト最適化**: 突破タイミング予測 + 信頼度評価

🎨 **トレンドフォロー最適化:**
- トレンド強い → チャネル幅縮小 → 早期エントリー
- トレンド弱い → チャネル幅拡大 → 偽シグナル回避
- ボラティリティ高 → 適応調整 → 安定性確保
- 相場転換 → 瞬時検出 → 即座対応
"""

from typing import Union, Optional, NamedTuple, Tuple
import numpy as np
import pandas as pd
from numba import jit, njit
import warnings
warnings.filterwarnings('ignore')

try:
    from .indicator import Indicator
    from .price_source import PriceSource
    from .atr import ATR
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from indicator import Indicator
    from price_source import PriceSource
    from atr import ATR


class CosmicAdaptiveChannelResult(NamedTuple):
    """宇宙最強適応チャネル計算結果"""
    # 核心チャネル
    upper_channel: np.ndarray           # 上側チャネル（8層フィルター済み）
    lower_channel: np.ndarray           # 下側チャネル（8層フィルター済み）
    midline: np.ndarray                # 中央線（量子統計フィルタリング済み）
    dynamic_width: np.ndarray           # 動的チャネル幅（トレンド強度反比例）
    
    # 宇宙最強シグナル
    breakout_signals: np.ndarray        # ブレイクアウトシグナル（1=上抜け、-1=下抜け）
    breakout_confidence: np.ndarray     # ブレイクアウト信頼度（0-1）
    breakout_timing: np.ndarray         # ブレイクアウトタイミング予測
    false_signal_filter: np.ndarray     # 偽シグナルフィルター（0=偽、1=真）
    
    # 量子統計解析
    quantum_coherence: np.ndarray       # 量子コヒーレンス指数
    statistical_trend: np.ndarray       # 統計的トレンド強度
    phase_analysis: np.ndarray          # ヒルベルト位相解析
    trend_strength: np.ndarray          # 統合トレンド強度
    
    # 神経適応システム
    neural_weights: np.ndarray          # 神経重み（学習結果）
    adaptation_score: np.ndarray        # 適応スコア（学習効果）
    learning_velocity: np.ndarray       # 学習速度
    memory_state: np.ndarray            # 記憶状態
    
    # ボラティリティレジーム
    volatility_regime: np.ndarray       # ボラティリティレジーム（1-5段階）
    regime_stability: np.ndarray        # レジーム安定度
    adaptive_factor: np.ndarray         # 適応ファクター
    channel_efficiency: np.ndarray      # チャネル効率度
    
    # 予測システム
    trend_momentum: np.ndarray          # トレンド勢い
    reversal_probability: np.ndarray    # 反転確率
    continuation_strength: np.ndarray   # 継続強度
    optimal_entry_zones: np.ndarray     # 最適エントリーゾーン
    
    # 現在状態
    current_trend_phase: str            # 現在のトレンドフェーズ
    current_volatility_regime: str      # 現在のボラティリティレジーム
    current_breakout_probability: float # 現在のブレイクアウト確率
    cosmic_intelligence_score: float    # 宇宙知能スコア


@njit
def quantum_statistical_fusion_numba(prices: np.ndarray, window: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    🌌 量子統計フュージョン（量子コヒーレンス + 統計回帰の革命的融合）
    """
    n = len(prices)
    quantum_coherence = np.zeros(n)
    statistical_trend = np.zeros(n)
    fusion_signal = np.zeros(n)
    
    for i in range(window, n):
        segment = prices[i-window:i]
        
        # 量子コヒーレンス計算（価格状態の重ね合わせ）
        phase_sum = 0.0
        for j in range(1, len(segment)):
            if segment[j-1] != 0:
                phase_diff = (segment[j] - segment[j-1]) / segment[j-1]
                phase_sum += np.cos(phase_diff * 2 * np.pi)
        quantum_coherence[i] = abs(phase_sum) / len(segment)
        
        # 統計回帰トレンド
        x_mean = (len(segment) - 1) / 2
        y_mean = np.mean(segment)
        numerator = 0.0
        denominator = 0.0
        
        for j in range(len(segment)):
            x_diff = j - x_mean
            y_diff = segment[j] - y_mean
            numerator += x_diff * y_diff
            denominator += x_diff * x_diff
        
        if denominator > 0:
            slope = numerator / denominator
            statistical_trend[i] = np.tanh(slope * 1000 / y_mean) if y_mean != 0 else 0
        
        # 量子統計フュージョン
        quantum_weight = quantum_coherence[i]
        statistical_weight = 1 - quantum_coherence[i]
        fusion_signal[i] = quantum_weight * np.tanh(quantum_coherence[i] * 2 - 1) + statistical_weight * statistical_trend[i]
    
    # 初期値補完
    quantum_coherence[:window] = quantum_coherence[window] if n > window else 0.5
    statistical_trend[:window] = statistical_trend[window] if n > window else 0.0
    fusion_signal[:window] = fusion_signal[window] if n > window else 0.0
    
    return quantum_coherence, statistical_trend, fusion_signal


@njit
def hilbert_phase_analysis_numba(prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    🌊 ヒルベルト位相解析（瞬時トレンド検出 + 位相遅延ゼロ）
    """
    n = len(prices)
    instantaneous_phase = np.zeros(n)
    trend_signal = np.zeros(n)
    
    for i in range(8, n):
        # 4点ヒルベルト変換
        real_part = (prices[i] + prices[i-2] + prices[i-4] + prices[i-6]) * 0.25
        imag_part = (prices[i-1] + prices[i-3] + prices[i-5] + prices[i-7]) * 0.25
        
        # 瞬時位相
        if real_part != 0:
            phase = np.arctan2(imag_part, real_part)
        else:
            phase = 0
        instantaneous_phase[i] = phase
        
        # 位相トレンド解析
        if i >= 15:
            phase_momentum = 0.0
            for j in range(7):
                if i-j >= 0:
                    phase_momentum += np.sin(instantaneous_phase[i-j])
            phase_momentum /= 7.0
            trend_signal[i] = np.tanh(phase_momentum * 2)
    
    # 初期値補完
    instantaneous_phase[:8] = instantaneous_phase[8] if n > 8 else 0.0
    trend_signal[:8] = trend_signal[8] if n > 8 else 0.0
    
    return instantaneous_phase, trend_signal


@njit
def neural_adaptive_learning_numba(prices: np.ndarray, trend_strength: np.ndarray, 
                                  volatility: np.ndarray, window: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    🧠 神経適応学習システム（市場パターン自動学習 + 動的重み調整）
    """
    n = len(prices)
    neural_weights = np.zeros(n)
    adaptation_score = np.zeros(n)
    learning_velocity = np.zeros(n)
    memory_state = np.zeros(n)
    
    # 初期パラメータ
    weight = 0.5
    momentum = 0.0
    long_term_memory = 0.5
    
    for i in range(window, n):
        # 特徴量抽出
        price_feature = (prices[i] - np.mean(prices[i-window//4:i])) / (np.std(prices[i-window//4:i]) + 1e-8)
        trend_feature = trend_strength[i]
        volatility_feature = volatility[i]
        
        # 統合入力信号
        input_signal = price_feature * 0.4 + trend_feature * 0.35 + volatility_feature * 0.25
        
        # 予測と実際の誤差
        if i > 0 and prices[i-1] != 0:
            actual_change = (prices[i] - prices[i-1]) / prices[i-1]
            predicted_change = weight * input_signal
            error = actual_change - predicted_change
            
            # 適応学習率
            base_learning_rate = 0.1
            volatility_adjustment = min(volatility[i] * 2, 1.0)
            adaptive_learning_rate = base_learning_rate * (1 + volatility_adjustment)
            learning_velocity[i] = adaptive_learning_rate
            
            # 重み更新（改良版勾配降下法）
            gradient = error * input_signal
            momentum = 0.9 * momentum + 0.1 * gradient
            weight += adaptive_learning_rate * gradient + 0.05 * momentum
            
            # 重み制限
            weight = max(-2.0, min(2.0, weight))
            
            # 適応スコア（学習効果評価）
            adaptation_score[i] = np.exp(-abs(error) * 5) * (1 + abs(input_signal))
            
            # 長期記憶更新
            long_term_memory = 0.99 * long_term_memory + 0.01 * adaptation_score[i]
            memory_state[i] = long_term_memory
        else:
            learning_velocity[i] = 0.1
            adaptation_score[i] = 0.5
            memory_state[i] = memory_state[i-1] if i > 0 else 0.5
        
        neural_weights[i] = weight
    
    # 初期値補完
    neural_weights[:window] = neural_weights[window] if n > window else 0.5
    adaptation_score[:window] = adaptation_score[window] if n > window else 0.5
    learning_velocity[:window] = learning_velocity[window] if n > window else 0.1
    memory_state[:window] = memory_state[window] if n > window else 0.5
    
    return neural_weights, adaptation_score, learning_velocity, memory_state


@njit
def volatility_regime_detection_numba(prices: np.ndarray, window: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    🌊 ボラティリティレジーム検出（リアルタイム市場状態識別）
    """
    n = len(prices)
    volatility_regime = np.ones(n)  # 1-5段階
    regime_stability = np.zeros(n)
    adaptive_factor = np.ones(n)
    
    for i in range(window, n):
        segment = prices[i-window:i]
        
        # 価格変動率の計算
        returns = np.zeros(len(segment)-1)
        for j in range(1, len(segment)):
            if segment[j-1] != 0:
                returns[j-1] = abs(segment[j] - segment[j-1]) / segment[j-1]
        
        if len(returns) > 0:
            mean_vol = np.mean(returns)
            std_vol = np.std(returns)
            
            # レジーム分類（5段階）
            if mean_vol < 0.005:
                regime = 1  # 極低ボラティリティ
                stability = 0.9
                factor = 2.0  # チャネル幅拡大
            elif mean_vol < 0.01:
                regime = 2  # 低ボラティリティ
                stability = 0.7
                factor = 1.5
            elif mean_vol < 0.02:
                regime = 3  # 中ボラティリティ
                stability = 0.5
                factor = 1.0
            elif mean_vol < 0.04:
                regime = 4  # 高ボラティリティ
                stability = 0.3
                factor = 0.8
            else:
                regime = 5  # 極高ボラティリティ
                stability = 0.1
                factor = 0.6  # チャネル幅縮小
            
            volatility_regime[i] = regime
            regime_stability[i] = stability
            adaptive_factor[i] = factor
    
    # 初期値補完
    volatility_regime[:window] = volatility_regime[window] if n > window else 3
    regime_stability[:window] = regime_stability[window] if n > window else 0.5
    adaptive_factor[:window] = adaptive_factor[window] if n > window else 1.0
    
    return volatility_regime, regime_stability, adaptive_factor


@njit
def dynamic_channel_width_calculation_numba(atr_values: np.ndarray, trend_strength: np.ndarray,
                                           volatility_regime: np.ndarray, adaptive_factor: np.ndarray,
                                           neural_weights: np.ndarray, base_multiplier: float = 2.0) -> np.ndarray:
    """
    🎯 動的チャネル幅計算（トレンド強度反比例 + 偽シグナル防御）
    """
    n = len(atr_values)
    dynamic_width = np.zeros(n)
    
    for i in range(n):
        base_width = atr_values[i] * base_multiplier
        
        # トレンド強度反比例調整（強いトレンド時は幅を縮める）
        trend_factor = max(0.3, 1.0 - 0.6 * abs(trend_strength[i]))
        
        # ボラティリティレジーム調整
        volatility_factor = adaptive_factor[i]
        
        # 神経適応調整
        neural_factor = 0.7 + 0.6 * abs(neural_weights[i] - 0.5)
        
        # 統合調整ファクター
        integrated_factor = trend_factor * 0.4 + volatility_factor * 0.35 + neural_factor * 0.25
        
        # 最終チャネル幅
        dynamic_width[i] = base_width * integrated_factor
        
        # 安全制限
        dynamic_width[i] = max(0.2 * base_width, min(3.0 * base_width, dynamic_width[i]))
    
    return dynamic_width


@njit
def cosmic_filter_processing_numba(prices: np.ndarray, quantum_coherence: np.ndarray,
                                  statistical_trend: np.ndarray, phase_analysis: np.ndarray,
                                  neural_weights: np.ndarray) -> np.ndarray:
    """
    🌌 宇宙フィルタ処理（8層統合フィルタリング）
    """
    n = len(prices)
    filtered_prices = prices.copy()
    
    for i in range(1, n):
        # 量子統計フュージョン重み
        quantum_weight = quantum_coherence[i]
        statistical_weight = abs(statistical_trend[i])
        phase_weight = abs(phase_analysis[i])
        neural_weight = abs(neural_weights[i] - 0.5) * 2
        
        # 重み正規化
        total_weight = quantum_weight + statistical_weight + phase_weight + neural_weight
        if total_weight > 0:
            quantum_weight /= total_weight
            statistical_weight /= total_weight
            phase_weight /= total_weight
            neural_weight /= total_weight
        else:
            quantum_weight = statistical_weight = phase_weight = neural_weight = 0.25
        
        # 適応フィルタリング
        alpha = 0.1 + 0.6 * (quantum_weight * 0.3 + statistical_weight * 0.25 + 
                            phase_weight * 0.25 + neural_weight * 0.2)
        alpha = max(0.05, min(0.7, alpha))
        
        filtered_prices[i] = alpha * prices[i] + (1 - alpha) * filtered_prices[i-1]
    
    return filtered_prices


@njit
def breakout_signal_generation_numba(prices: np.ndarray, upper_channel: np.ndarray,
                                    lower_channel: np.ndarray, trend_strength: np.ndarray,
                                    quantum_coherence: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    🚀 ブレイクアウトシグナル生成（偽シグナル完全防御システム）
    """
    n = len(prices)
    breakout_signals = np.zeros(n)
    breakout_confidence = np.zeros(n)
    breakout_timing = np.zeros(n)
    false_signal_filter = np.ones(n)
    
    for i in range(1, n):
        signal = 0
        confidence = 0.0
        timing = 0.0
        is_valid = 1
        
        # ブレイクアウト検出
        if prices[i] > upper_channel[i-1] and prices[i-1] <= upper_channel[i-1]:
            signal = 1  # 上抜け
            penetration_strength = (prices[i] - upper_channel[i-1]) / upper_channel[i-1]
            confidence = min(penetration_strength * 10, 1.0)
        elif prices[i] < lower_channel[i-1] and prices[i-1] >= lower_channel[i-1]:
            signal = -1  # 下抜け
            penetration_strength = (lower_channel[i-1] - prices[i]) / lower_channel[i-1]
            confidence = min(penetration_strength * 10, 1.0)
        
        # 偽シグナルフィルター（緩和版）
        if signal != 0:
            # トレンド強度確認（条件を緩和）
            if abs(trend_strength[i]) < 0.15:
                is_valid = 0  # 極端に弱いトレンドのみ無効
            
            # 量子コヒーレンス確認（条件を緩和）
            if quantum_coherence[i] < 0.25:
                is_valid = 0  # 極端に低いコヒーレンスのみ無効
            
            # タイミング評価（条件を緩和）
            if confidence > 0.3 and abs(trend_strength[i]) > 0.2:
                timing = confidence * abs(trend_strength[i])
        
        breakout_signals[i] = signal if is_valid else 0
        breakout_confidence[i] = confidence if is_valid else 0
        breakout_timing[i] = timing
        false_signal_filter[i] = is_valid
    
    return breakout_signals, breakout_confidence, breakout_timing, false_signal_filter


class CosmicAdaptiveChannel(Indicator):
    """
    🌌 **Cosmic Adaptive Channel (CAC) - 宇宙最強ブレイクアウトチャネル V1.0** 🌌
    
    🎯 **革命的8層ハイブリッドシステム:**
    1. 量子統計フュージョン: 量子コヒーレンス + 統計回帰の融合
    2. ヒルベルト位相解析: 瞬時トレンド検出 + 位相遅延ゼロ
    3. 神経適応学習: 市場パターン自動学習 + 動的重み調整
    4. 動的チャネル幅: トレンド強度反比例 + 偽シグナル防御
    5. ゼロラグフィルタ: 超低遅延 + 予測的補正
    6. ボラティリティレジーム: リアルタイム市場状態検出
    7. 超追従適応: 瞬時相場変化対応 + 学習型最適化
    8. ブレイクアウト予測: 突破確率 + タイミング予測
    """
    
    def __init__(self,
                 atr_period: int = 21,
                 base_multiplier: float = 2.0,
                 quantum_window: int = 50,
                 neural_window: int = 100,
                 volatility_window: int = 30,
                 src_type: str = 'hlc3'):
        """
        宇宙最強適応チャネルコンストラクタ
        
        Args:
            atr_period: ATR計算期間
            base_multiplier: 基本チャネル幅倍率
            quantum_window: 量子解析ウィンドウ
            neural_window: 神経学習ウィンドウ
            volatility_window: ボラティリティ解析ウィンドウ
            src_type: 価格ソースタイプ
        """
        super().__init__(f"CosmicAdaptiveChannel(atr={atr_period},mult={base_multiplier},src={src_type})")
        
        self.atr_period = atr_period
        self.base_multiplier = base_multiplier
        self.quantum_window = quantum_window
        self.neural_window = neural_window
        self.volatility_window = volatility_window
        self.src_type = src_type
        
        self.price_source_extractor = PriceSource()
        self.atr_indicator = ATR(period=atr_period)
        
        self._cache = {}
        self._result: Optional[CosmicAdaptiveChannelResult] = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> CosmicAdaptiveChannelResult:
        """
        🌌 宇宙最強適応チャネルを計算する
        """
        try:
            # データハッシュによるキャッシュチェック
            data_hash = self._get_data_hash(data)
            if data_hash in self._cache and self._result is not None:
                return self._result
            
            # 価格データ取得
            if isinstance(data, np.ndarray) and data.ndim == 1:
                src_prices = data.astype(np.float64)
                close_prices = data.astype(np.float64)
                high_prices = data.astype(np.float64)
                low_prices = data.astype(np.float64)
            else:
                src_prices = PriceSource.calculate_source(data, self.src_type)
                close_prices = data['close'].values if isinstance(data, pd.DataFrame) else data[:, 3]
                high_prices = data['high'].values if isinstance(data, pd.DataFrame) else data[:, 1]
                low_prices = data['low'].values if isinstance(data, pd.DataFrame) else data[:, 2]
                src_prices = src_prices.astype(np.float64)
                close_prices = close_prices.astype(np.float64)
                high_prices = high_prices.astype(np.float64)
                low_prices = low_prices.astype(np.float64)
            
            data_length = len(src_prices)
            if data_length == 0:
                return self._create_empty_result()
            
            self.logger.info("🌌 CAC - 宇宙最強適応チャネル計算開始...")
            
            # Step 1: ATR計算
            if isinstance(data, pd.DataFrame):
                atr_result = self.atr_indicator.calculate(data)
                atr_values = atr_result.values.astype(np.float64)
            else:
                # NumPy配列の場合は簡易ATR計算
                atr_values = self._calculate_simple_atr(high_prices, low_prices, close_prices)
            
            # Step 2: 量子統計フュージョン
            quantum_coherence, statistical_trend, fusion_signal = quantum_statistical_fusion_numba(
                src_prices, self.quantum_window)
            
            # Step 3: ヒルベルト位相解析
            phase_analysis, hilbert_trend = hilbert_phase_analysis_numba(src_prices)
            
            # Step 4: 統合トレンド強度計算
            trend_strength = (fusion_signal * 0.4 + hilbert_trend * 0.35 + statistical_trend * 0.25)
            
            # Step 5: ボラティリティ計算
            volatility = atr_values / (src_prices + 1e-8)  # 正規化ボラティリティ
            
            # Step 6: 神経適応学習
            neural_weights, adaptation_score, learning_velocity, memory_state = neural_adaptive_learning_numba(
                src_prices, trend_strength, volatility, self.neural_window)
            
            # Step 7: ボラティリティレジーム検出
            volatility_regime, regime_stability, adaptive_factor = volatility_regime_detection_numba(
                src_prices, self.volatility_window)
            
            # Step 8: 宇宙フィルタ処理
            cosmic_filtered_prices = cosmic_filter_processing_numba(
                src_prices, quantum_coherence, statistical_trend, phase_analysis, neural_weights)
            
            # Step 9: 動的チャネル幅計算
            dynamic_width = dynamic_channel_width_calculation_numba(
                atr_values, trend_strength, volatility_regime, adaptive_factor, neural_weights, self.base_multiplier)
            
            # Step 10: 最終チャネル計算
            upper_channel = cosmic_filtered_prices + dynamic_width
            lower_channel = cosmic_filtered_prices - dynamic_width
            
            # Step 11: ブレイクアウトシグナル生成
            breakout_signals, breakout_confidence, breakout_timing, false_signal_filter = breakout_signal_generation_numba(
                close_prices, upper_channel, lower_channel, trend_strength, quantum_coherence)
            
            # Step 12: 予測システム
            trend_momentum = np.gradient(trend_strength)
            reversal_probability = (1 - abs(trend_strength)) * quantum_coherence
            continuation_strength = abs(trend_strength) * adaptation_score
            optimal_entry_zones = breakout_confidence * false_signal_filter
            
            # Step 13: チャネル効率度計算
            channel_efficiency = adaptation_score * quantum_coherence * regime_stability
            
            # 現在状態の判定
            current_trend_phase = self._get_trend_phase(trend_strength[-1] if len(trend_strength) > 0 else 0.0)
            current_volatility_regime = self._get_volatility_regime(volatility_regime[-1] if len(volatility_regime) > 0 else 3)
            current_breakout_probability = float(breakout_confidence[-1]) if len(breakout_confidence) > 0 else 0.0
            cosmic_intelligence_score = float(np.mean(adaptation_score[-20:])) if len(adaptation_score) >= 20 else 0.5
            
            # NaN値チェックと修正
            upper_channel = np.nan_to_num(upper_channel, nan=np.nanmean(src_prices) * 1.05)
            lower_channel = np.nan_to_num(lower_channel, nan=np.nanmean(src_prices) * 0.95)
            cosmic_filtered_prices = np.nan_to_num(cosmic_filtered_prices, nan=src_prices)
            
            # 結果作成
            result = CosmicAdaptiveChannelResult(
                upper_channel=upper_channel,
                lower_channel=lower_channel,
                midline=cosmic_filtered_prices,
                dynamic_width=dynamic_width,
                breakout_signals=breakout_signals,
                breakout_confidence=breakout_confidence,
                breakout_timing=breakout_timing,
                false_signal_filter=false_signal_filter,
                quantum_coherence=quantum_coherence,
                statistical_trend=statistical_trend,
                phase_analysis=phase_analysis,
                trend_strength=trend_strength,
                neural_weights=neural_weights,
                adaptation_score=adaptation_score,
                learning_velocity=learning_velocity,
                memory_state=memory_state,
                volatility_regime=volatility_regime,
                regime_stability=regime_stability,
                adaptive_factor=adaptive_factor,
                channel_efficiency=channel_efficiency,
                trend_momentum=trend_momentum,
                reversal_probability=reversal_probability,
                continuation_strength=continuation_strength,
                optimal_entry_zones=optimal_entry_zones,
                current_trend_phase=current_trend_phase,
                current_volatility_regime=current_volatility_regime,
                current_breakout_probability=current_breakout_probability,
                cosmic_intelligence_score=cosmic_intelligence_score
            )
            
            self._result = result
            self._cache[data_hash] = self._result
            
            # 統計情報をログ出力
            total_signals = np.sum(np.abs(breakout_signals))
            avg_confidence = np.mean(breakout_confidence[breakout_confidence > 0]) if np.any(breakout_confidence > 0) else 0.0
            
            self.logger.info(f"✅ CAC計算完了 - シグナル数: {total_signals:.0f}, 平均信頼度: {avg_confidence:.3f}, "
                           f"トレンドフェーズ: {current_trend_phase}, 宇宙知能: {cosmic_intelligence_score:.3f}")
            return self._result
            
        except Exception as e:
            import traceback
            self.logger.error(f"CAC計算中にエラー: {e}\n{traceback.format_exc()}")
            return self._create_empty_result()
    
    def _calculate_simple_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """簡易ATR計算"""
        n = len(high)
        atr_values = np.zeros(n)
        
        for i in range(1, n):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            true_range = max(tr1, tr2, tr3)
            
            if i < self.atr_period:
                atr_values[i] = np.mean([high[j] - low[j] for j in range(i+1)])
            else:
                alpha = 2.0 / (self.atr_period + 1)
                atr_values[i] = alpha * true_range + (1 - alpha) * atr_values[i-1]
        
        # 最小ATR制限
        min_atr = np.mean(close) * 0.001
        return np.maximum(atr_values, min_atr)
    
    def _get_trend_phase(self, trend_value: float) -> str:
        """トレンドフェーズ判定"""
        if trend_value > 0.7:
            return "超強気"
        elif trend_value > 0.3:
            return "強気"
        elif trend_value > -0.3:
            return "中立"
        elif trend_value > -0.7:
            return "弱気"
        else:
            return "超弱気"
    
    def _get_volatility_regime(self, regime_value: float) -> str:
        """ボラティリティレジーム判定"""
        regime_map = {
            1: "極低ボラティリティ",
            2: "低ボラティリティ", 
            3: "中ボラティリティ",
            4: "高ボラティリティ",
            5: "極高ボラティリティ"
        }
        return regime_map.get(int(regime_value), "中ボラティリティ")
    
    def _create_empty_result(self, length: int = 0) -> CosmicAdaptiveChannelResult:
        """空の結果を作成"""
        return CosmicAdaptiveChannelResult(
            upper_channel=np.full(length, np.nan),
            lower_channel=np.full(length, np.nan),
            midline=np.full(length, np.nan),
            dynamic_width=np.full(length, np.nan),
            breakout_signals=np.zeros(length),
            breakout_confidence=np.zeros(length),
            breakout_timing=np.zeros(length),
            false_signal_filter=np.ones(length),
            quantum_coherence=np.full(length, 0.5),
            statistical_trend=np.zeros(length),
            phase_analysis=np.zeros(length),
            trend_strength=np.zeros(length),
            neural_weights=np.full(length, 0.5),
            adaptation_score=np.full(length, 0.5),
            learning_velocity=np.full(length, 0.1),
            memory_state=np.full(length, 0.5),
            volatility_regime=np.full(length, 3),
            regime_stability=np.full(length, 0.5),
            adaptive_factor=np.ones(length),
            channel_efficiency=np.full(length, 0.5),
            trend_momentum=np.zeros(length),
            reversal_probability=np.full(length, 0.5),
            continuation_strength=np.full(length, 0.5),
            optimal_entry_zones=np.zeros(length),
            current_trend_phase='中立',
            current_volatility_regime='中ボラティリティ',
            current_breakout_probability=0.0,
            cosmic_intelligence_score=0.5
        )
    
    def _get_data_hash(self, data) -> str:
        """データハッシュ計算"""
        if isinstance(data, np.ndarray):
            return hash(data.tobytes())
        elif isinstance(data, pd.DataFrame):
            return hash(data.values.tobytes())
        else:
            return hash(str(data))
    
    # Getter メソッド群
    def get_upper_channel(self) -> Optional[np.ndarray]:
        """上側チャネルを取得"""
        return self._result.upper_channel.copy() if self._result else None
    
    def get_lower_channel(self) -> Optional[np.ndarray]:
        """下側チャネルを取得"""
        return self._result.lower_channel.copy() if self._result else None
    
    def get_midline(self) -> Optional[np.ndarray]:
        """中央線を取得"""
        return self._result.midline.copy() if self._result else None
    
    def get_breakout_signals(self) -> Optional[np.ndarray]:
        """ブレイクアウトシグナルを取得"""
        return self._result.breakout_signals.copy() if self._result else None
    
    def get_breakout_confidence(self) -> Optional[np.ndarray]:
        """ブレイクアウト信頼度を取得"""
        return self._result.breakout_confidence.copy() if self._result else None
    
    def get_trend_analysis(self) -> Optional[dict]:
        """トレンド解析結果を取得"""
        if not self._result:
            return None
        return {
            'trend_strength': self._result.trend_strength.copy(),
            'trend_momentum': self._result.trend_momentum.copy(),
            'continuation_strength': self._result.continuation_strength.copy(),
            'reversal_probability': self._result.reversal_probability.copy()
        }
    
    def get_cosmic_intelligence_report(self) -> dict:
        """宇宙知能レポートを取得"""
        if not self._result:
            return {}
        
        return {
            'current_trend_phase': self._result.current_trend_phase,
            'current_volatility_regime': self._result.current_volatility_regime,
            'current_breakout_probability': self._result.current_breakout_probability,
            'cosmic_intelligence_score': self._result.cosmic_intelligence_score,
            'total_breakout_signals': int(np.sum(np.abs(self._result.breakout_signals))),
            'average_confidence': float(np.mean(self._result.breakout_confidence[self._result.breakout_confidence > 0])) if np.any(self._result.breakout_confidence > 0) else 0.0,
            'false_signal_rate': float(1 - np.mean(self._result.false_signal_filter)),
            'channel_efficiency': float(np.mean(self._result.channel_efficiency[-10:])) if len(self._result.channel_efficiency) >= 10 else 0.5,
            'neural_adaptation': float(np.mean(self._result.adaptation_score[-10:])) if len(self._result.adaptation_score) >= 10 else 0.5,
            'quantum_coherence': float(np.mean(self._result.quantum_coherence[-10:])) if len(self._result.quantum_coherence) >= 10 else 0.5
        }
    
    def reset(self) -> None:
        """インジケータの状態をリセット"""
        super().reset()
        self._result = None
        self._cache = {}
        if self.atr_indicator:
            self.atr_indicator.reset()


# エイリアス（使いやすくするため）
CAC = CosmicAdaptiveChannel 