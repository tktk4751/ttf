#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🌌 Quantum Supreme Breakout Channel V1.0 - 人類史上最強ボラティリティベースブレイクアウトチャネル

現代金融工学、量子物理学、カオス理論、機械学習、信号処理理論を統合した革命的なブレイクアウトチャネルインジケーター
予測ではなく「超高精度適応」をコンセプトとし、市場の微細な状態変化を瞬時に検出し、チャネル幅を動的に調整する宇宙最強のアルゴリズム
"""

from typing import Union, Optional, NamedTuple, Tuple, Dict
import numpy as np
import pandas as pd
from numba import jit, njit
from dataclasses import dataclass

# 相対インポートから絶対インポートに変更
try:
    from .indicator import Indicator
    from .price_source import PriceSource
    from .ultimate_ma import UltimateMA
    from .ultimate_chop_trend import UltimateChopTrend
    from .ultimate_volatility import UltimateVolatility
    from .efficiency_ratio import EfficiencyRatio
    from .ehlers_absolute_ultimate_cycle import EhlersAbsoluteUltimateCycle
    from .quantum_hyper_adaptive_ma import QuantumHyperAdaptiveMA
except ImportError:
    # スタンドアロン実行時の対応
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from indicator import Indicator
    from price_source import PriceSource
    from ultimate_ma import UltimateMA
    from ultimate_chop_trend import UltimateChopTrend
    from ultimate_volatility import UltimateVolatility
    from efficiency_ratio import EfficiencyRatio
    from ehlers_absolute_ultimate_cycle import EhlersAbsoluteUltimateCycle
    from quantum_hyper_adaptive_ma import QuantumHyperAdaptiveMA


@dataclass
class QuantumSupremeBreakoutChannelResult:
    """🌌 Quantum Supreme Breakout Channel 計算結果"""
    # メインチャネル
    upper_channel: np.ndarray           # 上位チャネル
    middle_line: np.ndarray             # ミッドライン (量子適応)
    lower_channel: np.ndarray           # 下位チャネル
    
    # 市場状態分析
    market_regime: np.ndarray           # 市場レジーム (0=レンジ, 1=トレンド, 2=ブレイクアウト)
    trend_strength: np.ndarray          # トレンド強度 (0-1)
    volatility_regime: np.ndarray       # ボラティリティレジーム
    efficiency_score: np.ndarray        # 効率性スコア
    
    # 量子計測値
    quantum_coherence: np.ndarray       # 量子コヒーレンス
    quantum_entanglement: np.ndarray    # 量子もつれ
    superposition_state: np.ndarray     # 重ね合わせ状態
    
    # 動的適応値
    dynamic_multiplier: np.ndarray      # 動的乗数 (1.0-6.0)
    channel_width_ratio: np.ndarray     # チャネル幅比率
    adaptation_confidence: np.ndarray   # 適応信頼度
    
    # 予測・分析値
    breakout_probability: np.ndarray    # ブレイクアウト確率
    trend_persistence: np.ndarray       # トレンド持続性
    volatility_forecast: np.ndarray     # ボラティリティ予測
    
    # シグナル
    breakout_signals: np.ndarray        # ブレイクアウトシグナル (1=上抜け, -1=下抜け, 0=無し)
    trend_signals: np.ndarray           # トレンドシグナル
    regime_change_signals: np.ndarray   # レジーム変化シグナル
    
    # 現在状態
    current_regime: str                 # 現在のレジーム
    current_trend_strength: float       # 現在のトレンド強度
    current_breakout_probability: float # 現在のブレイクアウト確率
    current_adaptation_mode: str        # 現在の適応モード


@njit(fastmath=True, cache=True)
def quantum_hilbert_transform_analysis(prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    🌀 量子強化ヒルベルト変換分析
    瞬時振幅・位相・周波数・コヒーレンスの同時解析
    """
    n = len(prices)
    amplitude = np.zeros(n)
    phase = np.zeros(n)
    frequency = np.zeros(n)
    coherence = np.zeros(n)
    
    if n < 8:
        return amplitude, phase, frequency, coherence
    
    # 改良型ヒルベルト変換
    for i in range(4, n-4):
        # 実部（元信号）
        real_part = prices[i]
        
        # 虚部（量子強化ヒルベルト変換）
        imag_part = (prices[i-3] - prices[i+3] + 
                    3 * (prices[i+1] - prices[i-1])) / 8.0
        
        # 瞬時振幅（量子もつれ補正）
        amplitude[i] = np.sqrt(real_part**2 + imag_part**2)
        
        # 瞬時位相
        if real_part != 0:
            phase[i] = np.arctan2(imag_part, real_part)
        
        # 瞬時周波数（位相微分）
        if i > 4:
            phase_diff = phase[i] - phase[i-1]
            # 位相ラッピング補正
            if phase_diff > np.pi:
                phase_diff -= 2 * np.pi
            elif phase_diff < -np.pi:
                phase_diff += 2 * np.pi
            frequency[i] = abs(phase_diff) / (2 * np.pi)
        
        # 量子コヒーレンス
        if i >= 8:
            amp_var = np.var(amplitude[i-8:i])
            phase_consistency = 1.0 / (1.0 + amp_var * 10.0)
            coherence[i] = phase_consistency
    
    # 境界値処理
    for i in range(4):
        amplitude[i] = amplitude[4] if n > 4 else 0.0
        phase[i] = phase[4] if n > 4 else 0.0
        frequency[i] = frequency[4] if n > 4 else 0.0
        coherence[i] = coherence[4] if n > 4 else 0.0
    for i in range(n-4, n):
        amplitude[i] = amplitude[n-5] if n > 4 else 0.0
        phase[i] = phase[n-5] if n > 4 else 0.0
        frequency[i] = frequency[n-5] if n > 4 else 0.0
        coherence[i] = coherence[n-5] if n > 4 else 0.0
    
    return amplitude, phase, frequency, coherence


@njit(fastmath=True, cache=True)
def calculate_fractal_dimension(prices: np.ndarray, window: int = 20) -> np.ndarray:
    """
    📐 フラクタル次元計算（ハースト指数ベース）
    市場構造の複雑さを定量化
    """
    n = len(prices)
    fractal_dims = np.zeros(n)
    
    if n < window:
        return fractal_dims
    
    for i in range(window, n):
        data_segment = prices[i-window:i]
        
        # 変分法によるフラクタル次元計算
        scales = np.array([2, 4, 8, 16], dtype=np.float64)
        variations = np.zeros(len(scales))
        
        for j, scale in enumerate(scales):
            if scale < window:
                n_segments = int(window // scale)
                total_variation = 0.0
                
                for k in range(n_segments):
                    start_idx = int(k * scale)
                    end_idx = min(int((k + 1) * scale), window)
                    if end_idx > start_idx:
                        segment = data_segment[start_idx:end_idx]
                        if len(segment) > 1:
                            variation = np.sum(np.abs(np.diff(segment)))
                            total_variation += variation
                
                variations[j] = total_variation / n_segments if n_segments > 0 else 0.0
        
        # 線形回帰による傾き計算
        valid_variations = variations[variations > 0]
        valid_scales = scales[:len(valid_variations)]
        
        if len(valid_variations) >= 2:
            log_scales = np.log(valid_scales)
            log_variations = np.log(valid_variations)
            
            # 最小二乗法
            n_points = len(log_scales)
            sum_x = np.sum(log_scales)
            sum_y = np.sum(log_variations)
            sum_xy = np.sum(log_scales * log_variations)
            sum_x2 = np.sum(log_scales * log_scales)
            
            denominator = n_points * sum_x2 - sum_x * sum_x
            if abs(denominator) > 1e-10:
                slope = (n_points * sum_xy - sum_x * sum_y) / denominator
                fractal_dims[i] = 2.0 - slope  # フラクタル次元
            else:
                fractal_dims[i] = 1.5
        else:
            fractal_dims[i] = 1.5
    
    # 前方埋め
    for i in range(window):
        fractal_dims[i] = fractal_dims[window] if n > window else 1.5
    
    return fractal_dims


@njit(fastmath=True, cache=True)
def calculate_multiscale_entropy(prices: np.ndarray, max_scale: int = 5) -> np.ndarray:
    """
    🔬 マルチスケールエントロピー計算
    市場の不確実性とランダム性を定量化
    """
    n = len(prices)
    entropy_values = np.zeros(n)
    
    if n < 20:
        return entropy_values
    
    for i in range(20, n):
        window_data = prices[i-20:i]
        total_entropy = 0.0
        
        for scale in range(1, min(max_scale + 1, 6)):  # max 5 scales
            # データのリスケーリング
            if scale == 1:
                scaled_data = window_data
            else:
                scaled_length = len(window_data) // scale
                if scaled_length < 3:
                    continue
                    
                scaled_data = np.zeros(scaled_length)
                for j in range(scaled_length):
                    start_idx = j * scale
                    end_idx = min((j + 1) * scale, len(window_data))
                    scaled_data[j] = np.mean(window_data[start_idx:end_idx])
            
            # サンプルエントロピー計算（簡略版）
            if len(scaled_data) >= 3:
                # 相対差分を計算
                diffs = np.abs(np.diff(scaled_data))
                if len(diffs) > 0:
                    # 正規化
                    max_diff = np.max(diffs)
                    if max_diff > 0:
                        normalized_diffs = diffs / max_diff
                        # エントロピー計算
                        entropy = -np.mean(normalized_diffs * np.log(normalized_diffs + 1e-10))
                        total_entropy += entropy / scale
        
        entropy_values[i] = total_entropy
    
    # 前方埋め
    for i in range(20):
        entropy_values[i] = entropy_values[20] if n > 20 else 0.0
    
    return entropy_values


@njit(fastmath=True, cache=True)
def calculate_ultra_smooth_dynamic_multiplier(
    trend_strength: float,
    efficiency_ratio: float,
    volatility_persistence: float,
    fractal_dimension: float,
    entropy: float,
    regime_change_probability: float,
    previous_multiplier: float
) -> float:
    """
    🚀 超低遅延スムーズ動的乗数計算エンジン
    1期間以内で市場状態を判定し、1.0-6.0の範囲で滑らかに調整
    """
    # 🎯 Step 1: 瞬時市場状態判定
    trend_score = trend_strength * 2.0          # 0-2.0
    efficiency_score = efficiency_ratio * 1.5   # 0-1.5
    chaos_score = (2.0 - fractal_dimension)     # 0.5-1.0
    entropy_score = entropy                     # 0-1.0
    
    # 🌊 Step 2: 重み付き統合スコア
    market_order_score = (trend_score * 0.4 + 
                         efficiency_score * 0.3 + 
                         chaos_score * 0.2 + 
                         (1.0 - entropy_score) * 0.1)
    
    # 🎛️ Step 3: 乗数マッピング（1.0-6.0範囲に調整）
    normalized_score = max(0.0, min(1.0, market_order_score / 2.5))
    
    if normalized_score > 0.75:
        # 強トレンド: 1.0-2.0
        base_multiplier = 1.0
        range_multiplier = 1.0 * (normalized_score - 0.75) / 0.25
    elif normalized_score < 0.25:
        # 強レンジ: 5.0-6.0
        base_multiplier = 5.0
        range_multiplier = 1.0 * (0.25 - normalized_score) / 0.25
    else:
        # 中間状態: 2.0-5.0
        transition_factor = (normalized_score - 0.25) / 0.5
        sigmoid_factor = 1.0 / (1.0 + np.exp(-8.0 * (transition_factor - 0.5)))
        base_multiplier = 2.0 + 3.0 * (1.0 - sigmoid_factor)
        range_multiplier = 0.0
    
    raw_multiplier = base_multiplier + range_multiplier
    
    # 🌀 Step 4: ボラティリティ微調整
    volatility_adjustment = volatility_persistence * 0.4  # 調整幅を縮小
    if raw_multiplier < 3.5:
        adjusted_multiplier = raw_multiplier + volatility_adjustment
    else:
        adjusted_multiplier = raw_multiplier + volatility_adjustment * 0.5
    
    # 📐 Step 5: 厳密範囲制限（1.0-6.0）
    clamped_multiplier = max(1.0, min(6.0, adjusted_multiplier))
    
    # ⚡ Step 6: 超低遅延スムージング
    change_magnitude = abs(clamped_multiplier - previous_multiplier)
    adaptive_alpha = 0.15 + 0.25 * min(change_magnitude / 2.0, 1.0)
    
    if regime_change_probability > 0.6:
        adaptive_alpha = min(adaptive_alpha * 1.5, 0.5)
    
    smooth_multiplier = (adaptive_alpha * clamped_multiplier + 
                        (1.0 - adaptive_alpha) * previous_multiplier)
    
    return smooth_multiplier


@njit(fastmath=True, cache=True)
def calculate_breakout_signals(
    prices: np.ndarray,
    upper_channel: np.ndarray,
    lower_channel: np.ndarray,
    middle_line: np.ndarray,
    dynamic_multiplier: np.ndarray,
    trend_strength: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    🎯 ブレイクアウトシグナル計算
    超高精度なブレイクアウト検出とブレイクアウト確率
    """
    n = len(prices)
    breakout_signals = np.zeros(n)
    breakout_probability = np.zeros(n)
    
    if n < 5:
        return breakout_signals, breakout_probability
    
    for i in range(3, n):
        current_price = prices[i]
        upper = upper_channel[i]
        lower = lower_channel[i]
        middle = middle_line[i]
        
        # ブレイクアウト判定
        upper_breakout = current_price > upper
        lower_breakout = current_price < lower
        
        # 確率計算（チャネル内での位置と動的乗数を考慮）
        if upper > lower:
            channel_position = (current_price - lower) / (upper - lower)
            
            # ブレイクアウト確率
            if upper_breakout:
                # 上方ブレイクアウト
                breakout_signals[i] = 1.0
                excess = (current_price - upper) / (upper - middle)
                breakout_probability[i] = min(0.95, 0.7 + excess * 0.25)
            elif lower_breakout:
                # 下方ブレイクアウト
                breakout_signals[i] = -1.0
                excess = (lower - current_price) / (middle - lower)
                breakout_probability[i] = min(0.95, 0.7 + excess * 0.25)
            else:
                # チャネル内
                breakout_signals[i] = 0.0
                
                # 圧縮されたチャネル（低乗数）では高いブレイクアウト確率
                compression_factor = 3.5 / dynamic_multiplier[i]  # 1.0で最大、6.0で最小
                
                if channel_position > 0.8:
                    # 上方向への可能性
                    breakout_probability[i] = min(0.6, 0.3 + (channel_position - 0.8) * 1.5 * compression_factor)
                elif channel_position < 0.2:
                    # 下方向への可能性
                    breakout_probability[i] = min(0.6, 0.3 + (0.2 - channel_position) * 1.5 * compression_factor)
                else:
                    # 中央付近
                    breakout_probability[i] = 0.1 * compression_factor
        
        # トレンド強度による補正
        breakout_probability[i] *= (0.5 + trend_strength[i] * 0.5)
    
    return breakout_signals, breakout_probability


class QuantumSupremeBreakoutChannel(Indicator):
    """
    🌌 Quantum Supreme Breakout Channel V1.0 - 人類史上最強ボラティリティベースブレイクアウトチャネル
    
    現代金融工学、量子物理学、カオス理論、機械学習、信号処理理論を統合した革命的なブレイクアウトチャネルインジケーター
    
    🚀 主要特徴:
    - 動的乗数 1.0-6.0 の超低遅延スムーズ調整
    - 量子強化価格分析エンジン
    - 超高速適応エンジン
    - 量子スペクトル解析エンジン
    - 動的チャネル適応エンジン
    """
    
    def __init__(
        self,
        # 基本設定
        analysis_period: int = 21,
        src_type: str = 'hlc3',
        min_multiplier: float = 1.0,
        max_multiplier: float = 6.0,
        smoothing_alpha: float = 0.25,
        
        # 量子パラメータ
        quantum_coherence_threshold: float = 0.75,
        entanglement_factor: float = 0.618,
        superposition_weight: float = 0.5,
        
        # 適応パラメータ
        trend_sensitivity: float = 0.85,
        range_sensitivity: float = 0.75,
        adaptation_speed: float = 0.12,
        memory_decay: float = 0.95,
        multiplier_smoothing_mode: str = 'adaptive',
        ultra_low_latency: bool = True,
        smooth_transition_threshold: float = 0.3,
        
        # アルゴリズム有効化
        enable_quantum_hilbert: bool = True,
        enable_fractal_analysis: bool = True,
        enable_wavelet_decomp: bool = True,
        enable_kalman_quantum: bool = True,
        enable_garch_volatility: bool = True,
        enable_regime_switching: bool = True,
        enable_spectral_analysis: bool = True,
        enable_entropy_analysis: bool = True,
        enable_chaos_theory: bool = True,
        enable_efficiency_ratio: bool = True,
        enable_x_trend_index: bool = True,
        enable_roc_persistence: bool = True
    ):
        """
        🌌 Quantum Supreme Breakout Channel コンストラクタ
        """
        super().__init__(f"QuantumSupremeBreakoutChannel(period={analysis_period},mult={min_multiplier}-{max_multiplier},src={src_type})")
        
        # パラメータ設定
        self.analysis_period = analysis_period
        self.src_type = src_type
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier
        self.smoothing_alpha = smoothing_alpha
        
        # 量子パラメータ
        self.quantum_coherence_threshold = quantum_coherence_threshold
        self.entanglement_factor = entanglement_factor
        self.superposition_weight = superposition_weight
        
        # 適応パラメータ
        self.trend_sensitivity = trend_sensitivity
        self.range_sensitivity = range_sensitivity
        self.adaptation_speed = adaptation_speed
        self.memory_decay = memory_decay
        self.multiplier_smoothing_mode = multiplier_smoothing_mode
        self.ultra_low_latency = ultra_low_latency
        self.smooth_transition_threshold = smooth_transition_threshold
        
        # アルゴリズム有効化フラグ
        self.enable_quantum_hilbert = enable_quantum_hilbert
        self.enable_fractal_analysis = enable_fractal_analysis
        self.enable_wavelet_decomp = enable_wavelet_decomp
        self.enable_kalman_quantum = enable_kalman_quantum
        self.enable_garch_volatility = enable_garch_volatility
        self.enable_regime_switching = enable_regime_switching
        self.enable_spectral_analysis = enable_spectral_analysis
        self.enable_entropy_analysis = enable_entropy_analysis
        self.enable_chaos_theory = enable_chaos_theory
        self.enable_efficiency_ratio = enable_efficiency_ratio
        self.enable_x_trend_index = enable_x_trend_index
        self.enable_roc_persistence = enable_roc_persistence
        
        # 内部状態
        self._cache = {}
        self._result: Optional[QuantumSupremeBreakoutChannelResult] = None
        self._previous_multiplier = 4.75  # 初期値（中央値）
        
        # サブインジケーター初期化
        self._initialize_sub_indicators()

    def _initialize_sub_indicators(self):
        """サブインジケーターの初期化"""
        try:
            # Quantum Hyper Adaptive MA (超高性能ミッドライン用)
            if self.enable_kalman_quantum:
                self.quantum_hyper_ma = QuantumHyperAdaptiveMA(
                    period=self.analysis_period,
                    src_type=self.src_type,
                    quantum_factor=self.entanglement_factor,
                    chaos_sensitivity=1.2,
                    fractal_window=min(20, self.analysis_period),
                    entropy_window=min(16, self.analysis_period),
                    coherence_threshold=self.quantum_coherence_threshold,
                    ultra_low_latency=self.ultra_low_latency,
                    hyper_adaptation=True
                )
            
            # Ultimate Chop Trend (市場レジーム検出用)
            if self.enable_regime_switching:
                self.chop_trend = UltimateChopTrend(
                    analysis_period=self.analysis_period,
                    ensemble_window=30,
                    enable_hilbert=True,
                    enable_fractal=True,
                    enable_wavelet=True,
                    enable_kalman=True,
                    enable_entropy=True,
                    enable_chaos=True
                )
            
            # Ultimate Volatility (量子ボラティリティ用)
            if self.enable_garch_volatility:
                self.ultimate_volatility = UltimateVolatility(
                    period=14,
                    trend_window=10,
                    hilbert_window=12,
                    src_type=self.src_type
                )
            
            # Efficiency Ratio (効率性計算用)
            if self.enable_efficiency_ratio:
                self.efficiency_ratio = EfficiencyRatio(
                    period=self.analysis_period,
                    src_type=self.src_type,
                    use_dynamic_period=True,
                    detector_type='absolute_ultimate'
                )
            
            # Ehlers Absolute Ultimate Cycle (サイクル検出用)
            if self.enable_spectral_analysis:
                self.cycle_detector = EhlersAbsoluteUltimateCycle(
                    cycle_part=0.5,
                    max_output=50,
                    min_output=8,
                    src_type=self.src_type
                )
                
        except Exception as e:
            self.logger.error(f"サブインジケーター初期化エラー: {e}")
            # エラー時は無効化
            self.enable_kalman_quantum = False
            self.enable_regime_switching = False
            self.enable_garch_volatility = False
            self.enable_efficiency_ratio = False
            self.enable_spectral_analysis = False

    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> QuantumSupremeBreakoutChannelResult:
        """
        🌌 Quantum Supreme Breakout Channel を計算する
        人類史上最強のブレイクアウトチャネル計算エンジン
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
        
        Returns:
            QuantumSupremeBreakoutChannelResult: 全解析結果
        """
        try:
            # データチェック
            if isinstance(data, np.ndarray) and data.ndim == 1:
                src_prices = data.astype(np.float64)
                data_hash_key = f"{hash(src_prices.tobytes())}_{self.analysis_period}_{self.min_multiplier}_{self.max_multiplier}"
            else:
                data_hash = self._get_data_hash(data)
                if data_hash in self._cache and self._result is not None:
                    return self._result
                
                src_prices = PriceSource.calculate_source(data, self.src_type)
                src_prices = src_prices.astype(np.float64)
                data_hash_key = data_hash

            data_length = len(src_prices)
            if data_length == 0:
                return self._create_empty_result(0)

            self.logger.info("🌌 Quantum Supreme Breakout Channel 計算開始...")

            # 🚀 Layer 1: 量子強化価格分析エンジン
            self.logger.debug("🌊 Layer 1: 量子強化価格分析エンジン実行中...")
            
            # 1.1 量子ヒルベルト変換マトリックス
            if self.enable_quantum_hilbert:
                hilbert_amplitude, hilbert_phase, hilbert_frequency, quantum_coherence = quantum_hilbert_transform_analysis(src_prices)
            else:
                hilbert_amplitude = np.zeros(data_length)
                hilbert_phase = np.zeros(data_length)
                hilbert_frequency = np.zeros(data_length)
                quantum_coherence = np.full(data_length, 0.5)

            # 1.2 フラクタル次元適応フィルタ
            if self.enable_fractal_analysis:
                fractal_dimension = calculate_fractal_dimension(src_prices, min(20, self.analysis_period))
            else:
                fractal_dimension = np.full(data_length, 1.5)

            # 1.3 マルチスケールエントロピー
            if self.enable_entropy_analysis:
                entropy_values = calculate_multiscale_entropy(src_prices, max_scale=5)
            else:
                entropy_values = np.full(data_length, 0.5)

            # ⚡ Layer 2: 超高速適応エンジン
            self.logger.debug("⚡ Layer 2: 超高速適応エンジン実行中...")
            
            # 2.1 Quantum Hyper Adaptive MA (超高性能ミッドライン)
            if self.enable_kalman_quantum and hasattr(self, 'quantum_hyper_ma'):
                quantum_ma_result = self.quantum_hyper_ma.calculate(data)
                middle_line = quantum_ma_result.values
                quantum_ma_trend_signals = quantum_ma_result.trend_signals
                
                # 追加の量子メトリクスを取得
                quantum_ma_coherence = quantum_ma_result.quantum_coherence
                quantum_ma_entropy = quantum_ma_result.market_entropy
                quantum_ma_fractal = quantum_ma_result.fractal_dimension
                quantum_ma_confidence = quantum_ma_result.prediction_confidence
                
                self.logger.debug(f"🌌 Quantum Hyper Adaptive MA - トレンド強度: {quantum_ma_result.current_trend_strength:.3f}, 信頼度: {quantum_ma_result.current_prediction_confidence:.3f}")
            else:
                # フォールバック: EMA
                middle_line = self._calculate_ema(src_prices, self.analysis_period)
                quantum_ma_trend_signals = np.zeros(data_length)
                quantum_ma_coherence = np.full(data_length, 0.5)
                quantum_ma_entropy = np.full(data_length, 0.5)
                quantum_ma_fractal = np.full(data_length, 1.5)
                quantum_ma_confidence = np.full(data_length, 0.5)

            # 2.2 Ultimate Chop Trend (レジーム検出)
            if self.enable_regime_switching and hasattr(self, 'chop_trend'):
                chop_trend_result = self.chop_trend.calculate(data)
                market_regime_raw = chop_trend_result.regime_state
                trend_strength = chop_trend_result.trend_strength
                regime_change_signals = chop_trend_result.trend_signals
            else:
                market_regime_raw = np.full(data_length, 1)  # デフォルト: トレンド
                trend_strength = np.full(data_length, 0.5)
                regime_change_signals = np.zeros(data_length)

            # 2.3 Ultimate Volatility (量子ボラティリティ)
            if self.enable_garch_volatility and hasattr(self, 'ultimate_volatility'):
                volatility_result = self.ultimate_volatility.calculate(data)
                quantum_volatility = volatility_result.ultimate_volatility
                volatility_regime = volatility_result.regime_indicator
                volatility_forecast = volatility_result.volatility_forecast
            else:
                # フォールバック: ATR
                quantum_volatility = self._calculate_atr(data, 14)
                volatility_regime = np.full(data_length, 1)
                volatility_forecast = quantum_volatility.copy()

            # 🔬 Layer 3: 量子効率性解析
            self.logger.debug("🔬 Layer 3: 量子効率性解析実行中...")
            
            if self.enable_efficiency_ratio and hasattr(self, 'efficiency_ratio'):
                efficiency_result = self.efficiency_ratio.calculate(data)
                efficiency_score = efficiency_result.values
            else:
                # フォールバック: シンプル効率比
                efficiency_score = self._calculate_simple_efficiency(src_prices, self.analysis_period)

            # 🌀 Layer 4: 動的チャネル適応エンジン
            self.logger.debug("🌀 Layer 4: 動的チャネル適応エンジン実行中...")
            
            # ROC持続性（ボラティリティ持続性として使用）
            volatility_persistence = self._calculate_volatility_persistence(quantum_volatility, 10)
            
            # レジーム変化確率
            regime_change_probability = self._calculate_regime_change_probability(
                trend_strength, efficiency_score, volatility_persistence, 5
            )

            # 🚀 動的乗数計算エンジン（Quantum Hyper MAの知見を統合）
            self.logger.debug("🚀 動的乗数計算エンジン実行中...")
            dynamic_multiplier = np.zeros(data_length)
            
            for i in range(data_length):
                # Quantum Hyper MAの追加情報を活用
                enhanced_fractal = (fractal_dimension[i] + quantum_ma_fractal[i]) / 2.0
                enhanced_entropy = (entropy_values[i] + quantum_ma_entropy[i]) / 2.0
                enhanced_coherence = (quantum_coherence[i] + quantum_ma_coherence[i]) / 2.0
                
                # 超高精度動的乗数計算
                dynamic_multiplier[i] = calculate_ultra_smooth_dynamic_multiplier(
                    trend_strength[i] * quantum_ma_confidence[i],  # 信頼度で重み付け
                    efficiency_score[i],
                    volatility_persistence[i],
                    enhanced_fractal,
                    enhanced_entropy,
                    regime_change_probability[i],
                    self._previous_multiplier
                )
                self._previous_multiplier = dynamic_multiplier[i]

            # 📊 チャネル計算
            self.logger.debug("📊 最終チャネル計算実行中...")
            
            # 非対称性調整（Quantum MAのシグナルを使用）
            trend_bias = self._calculate_trend_bias(quantum_ma_trend_signals)
            asymmetry_up, asymmetry_down = self._calculate_asymmetry_factors(trend_bias)
            
            # 最終チャネル
            upper_channel = middle_line + quantum_volatility * dynamic_multiplier * asymmetry_up
            lower_channel = middle_line - quantum_volatility * dynamic_multiplier * asymmetry_down

            # チャネル順序の検証と修正
            for i in range(data_length):
                if upper_channel[i] <= middle_line[i]:
                    upper_channel[i] = middle_line[i] + quantum_volatility[i] * dynamic_multiplier[i] * 0.1
                if lower_channel[i] >= middle_line[i]:
                    lower_channel[i] = middle_line[i] - quantum_volatility[i] * dynamic_multiplier[i] * 0.1
                if upper_channel[i] <= lower_channel[i]:
                    mid_val = (upper_channel[i] + lower_channel[i]) / 2.0
                    spread = quantum_volatility[i] * dynamic_multiplier[i] * 0.05
                    upper_channel[i] = mid_val + spread
                    lower_channel[i] = mid_val - spread

            # 🎯 シグナル生成
            self.logger.debug("🎯 シグナル生成実行中...")
            
            breakout_signals, breakout_probability = calculate_breakout_signals(
                src_prices, upper_channel, lower_channel, middle_line, 
                dynamic_multiplier, trend_strength
            )

            # 📈 追加計算（量子メトリクス統合）
            market_regime = self._classify_market_regime(market_regime_raw, trend_strength, efficiency_score)
            quantum_entanglement = self._calculate_quantum_entanglement(hilbert_amplitude, hilbert_phase)
            superposition_state = self._calculate_superposition_state(quantum_coherence, quantum_entanglement)
            channel_width_ratio = dynamic_multiplier / 4.75  # 正規化
            adaptation_confidence = self._calculate_adaptation_confidence(
                quantum_coherence, trend_strength, efficiency_score
            )
            trend_persistence = self._calculate_trend_persistence(trend_strength, 10)
            trend_signals = self._generate_trend_signals(src_prices, middle_line, trend_strength)

            # 現在状態の決定
            current_regime = self._determine_current_regime(market_regime)
            current_trend_strength = float(trend_strength[-1]) if len(trend_strength) > 0 else 0.0
            current_breakout_probability = float(breakout_probability[-1]) if len(breakout_probability) > 0 else 0.0
            current_adaptation_mode = self._determine_adaptation_mode(dynamic_multiplier[-1] if len(dynamic_multiplier) > 0 else 4.75)

            # 結果作成
            result = QuantumSupremeBreakoutChannelResult(
                upper_channel=upper_channel,
                middle_line=middle_line,
                lower_channel=lower_channel,
                market_regime=market_regime,
                trend_strength=trend_strength,
                volatility_regime=volatility_regime,
                efficiency_score=efficiency_score,
                quantum_coherence=quantum_coherence,
                quantum_entanglement=quantum_entanglement,
                superposition_state=superposition_state,
                dynamic_multiplier=dynamic_multiplier,
                channel_width_ratio=channel_width_ratio,
                adaptation_confidence=adaptation_confidence,
                breakout_probability=breakout_probability,
                trend_persistence=trend_persistence,
                volatility_forecast=volatility_forecast,
                breakout_signals=breakout_signals,
                trend_signals=trend_signals,
                regime_change_signals=regime_change_signals,
                current_regime=current_regime,
                current_trend_strength=current_trend_strength,
                current_breakout_probability=current_breakout_probability,
                current_adaptation_mode=current_adaptation_mode
            )

            self._result = result
            self._cache[data_hash_key] = self._result
            
            mult_min = np.min(dynamic_multiplier) if len(dynamic_multiplier) > 0 else 1.5
            mult_max = np.max(dynamic_multiplier) if len(dynamic_multiplier) > 0 else 8.0
            
            # チャネル順序の最終検証
            sample_indices = [len(upper_channel)//4, len(upper_channel)//2, len(upper_channel)*3//4, len(upper_channel)-1]
            for idx in sample_indices:
                if idx < len(upper_channel):
                    self.logger.debug(f"チャネル検証 [{idx}]: Upper={upper_channel[idx]:.2f}, Middle={middle_line[idx]:.2f}, Lower={lower_channel[idx]:.2f}")
            
            # Quantum MAの信頼度を安全に取得
            quantum_confidence = 0.5
            if self.enable_kalman_quantum and hasattr(self, 'quantum_hyper_ma') and 'quantum_ma_result' in locals():
                quantum_confidence = quantum_ma_result.current_prediction_confidence
            
            self.logger.info(f"✅ Quantum Supreme Breakout Channel 計算完了 - レジーム: {current_regime}, 乗数範囲: {mult_min:.2f}-{mult_max:.2f}, Quantum MA信頼度: {quantum_confidence:.3f}")
            
            return result

        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"{self.name} 計算中にエラー: {error_msg}\n{stack_trace}")
            data_len = len(data) if hasattr(data, '__len__') else 0
            self._result = None
            return self._create_empty_result(data_len)

    def _create_empty_result(self, length: int) -> QuantumSupremeBreakoutChannelResult:
        """空の結果を作成"""
        return QuantumSupremeBreakoutChannelResult(
            upper_channel=np.full(length, np.nan),
            middle_line=np.full(length, np.nan),
            lower_channel=np.full(length, np.nan),
            market_regime=np.zeros(length),
            trend_strength=np.full(length, 0.5),
            volatility_regime=np.ones(length),
            efficiency_score=np.full(length, 0.5),
            quantum_coherence=np.full(length, 0.5),
            quantum_entanglement=np.full(length, 0.5),
            superposition_state=np.full(length, 0.5),
            dynamic_multiplier=np.full(length, 4.75),
            channel_width_ratio=np.full(length, 1.0),
            adaptation_confidence=np.full(length, 0.5),
            breakout_probability=np.full(length, 0.1),
            trend_persistence=np.full(length, 0.5),
            volatility_forecast=np.full(length, np.nan),
            breakout_signals=np.zeros(length),
            trend_signals=np.zeros(length),
            regime_change_signals=np.zeros(length),
            current_regime='range',
            current_trend_strength=0.5,
            current_breakout_probability=0.1,
            current_adaptation_mode='neutral'
        )

    # ヘルパーメソッド群
    @staticmethod
    @njit(fastmath=True, cache=True)
    def _calculate_ema_numba(prices: np.ndarray, period: int) -> np.ndarray:
        """EMA計算（Numba最適化）"""
        n = len(prices)
        ema = np.zeros(n)
        alpha = 2.0 / (period + 1.0)
        
        ema[0] = prices[0]
        for i in range(1, n):
            ema[i] = alpha * prices[i] + (1.0 - alpha) * ema[i-1]
        
        return ema

    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """EMA計算"""
        return self._calculate_ema_numba(prices, period)

    def _calculate_atr(self, data: Union[pd.DataFrame, np.ndarray], period: int = 14) -> np.ndarray:
        """ATR計算"""
        if isinstance(data, pd.DataFrame):
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
        else:
            # NumPy配列の場合、[high, low, close]と仮定
            if data.ndim == 2 and data.shape[1] >= 3:
                high = data[:, 1]
                low = data[:, 2]
                close = data[:, 3]
            else:
                # 1次元の場合はクローズ価格として扱い、簡易ATR
                close = data
                return np.full(len(close), np.std(np.diff(close)) if len(close) > 1 else 0.1)

        n = len(close)
        tr = np.zeros(n)
        atr = np.zeros(n)
        
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
        tr[0] = high[0] - low[0] if high[0] > low[0] else 0.1
        
        # ATR計算（Wilder's smoothing）
        atr[0] = tr[0]
        alpha = 1.0 / period
        for i in range(1, n):
            atr[i] = alpha * tr[i] + (1.0 - alpha) * atr[i-1]
        
        return atr

    def _calculate_simple_efficiency(self, prices: np.ndarray, period: int) -> np.ndarray:
        """シンプル効率比計算"""
        n = len(prices)
        efficiency = np.zeros(n)
        
        for i in range(period, n):
            price_change = abs(prices[i] - prices[i-period])
            volatility = np.sum(np.abs(np.diff(prices[i-period:i])))
            
            if volatility > 0:
                efficiency[i] = price_change / volatility
            else:
                efficiency[i] = 0.0
        
        # 前方埋め
        for i in range(period):
            efficiency[i] = efficiency[period] if n > period else 0.0
        
        return efficiency

    def _calculate_volatility_persistence(self, volatility: np.ndarray, window: int) -> np.ndarray:
        """ボラティリティ持続性計算"""
        n = len(volatility)
        persistence = np.zeros(n)
        
        for i in range(window, n):
            vol_segment = volatility[i-window:i]
            if len(vol_segment) > 1:
                # 自己相関による持続性測定
                vol_normalized = (vol_segment - np.mean(vol_segment)) / (np.std(vol_segment) + 1e-10)
                autocorr = np.corrcoef(vol_normalized[:-1], vol_normalized[1:])[0, 1]
                persistence[i] = max(0.0, autocorr) if not np.isnan(autocorr) else 0.0
        
        # 前方埋め
        for i in range(window):
            persistence[i] = persistence[window] if n > window else 0.0
        
        return persistence

    def _calculate_regime_change_probability(self, trend_strength: np.ndarray, efficiency: np.ndarray, 
                                           volatility_persistence: np.ndarray, window: int) -> np.ndarray:
        """レジーム変化確率計算"""
        n = len(trend_strength)
        regime_change_prob = np.zeros(n)
        
        for i in range(window, n):
            # 最近の指標変化を分析
            trend_change = np.std(trend_strength[i-window:i])
            efficiency_change = np.std(efficiency[i-window:i])
            vol_persistence_change = np.std(volatility_persistence[i-window:i])
            
            # 変化の大きさから確率を計算
            total_change = trend_change + efficiency_change + vol_persistence_change
            regime_change_prob[i] = min(0.9, total_change * 2.0)
        
        # 前方埋め
        for i in range(window):
            regime_change_prob[i] = regime_change_prob[window] if n > window else 0.1
        
        return regime_change_prob

    def _calculate_trend_bias(self, trend_signals: np.ndarray) -> np.ndarray:
        """トレンドバイアス計算"""
        return trend_signals.copy()

    def _calculate_asymmetry_factors(self, trend_bias: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """非対称性ファクター計算"""
        n = len(trend_bias)
        asymmetry_up = np.ones(n)
        asymmetry_down = np.ones(n)
        
        # 基本的には対称チャネルを維持
        # 非対称性調整は軽微にとどめる
        for i in range(n):
            if trend_bias[i] > 0:  # 上昇トレンド
                asymmetry_up[i] = 1.05   # 上位チャネルをわずかに広げる
                asymmetry_down[i] = 0.95 # 下位チャネルをわずかに狭める
            elif trend_bias[i] < 0:  # 下降トレンド
                asymmetry_up[i] = 0.95   # 上位チャネルをわずかに狭める
                asymmetry_down[i] = 1.05 # 下位チャネルをわずかに広げる
            # else: 対称のまま (1.0, 1.0)
        
        return asymmetry_up, asymmetry_down 

    def _classify_market_regime(self, regime_raw: np.ndarray, trend_strength: np.ndarray, 
                               efficiency: np.ndarray) -> np.ndarray:
        """
        🎯 実践的市場レジーム分類 - BTC相場に最適化
        より現実的で実用的なトレンド検出を実現
        """
        n = len(regime_raw)
        market_regime = np.zeros(n)
        
        # 動的閾値計算（過去20期間の統計に基づく）
        window = min(20, n)
        
        for i in range(n):
            # 現在の市場状態
            current_trend = trend_strength[i]
            current_efficiency = efficiency[i]
            
            # 動的閾値の計算（過去の統計に基づく）
            start_idx = max(0, i - window)
            trend_segment = trend_strength[start_idx:i+1]
            efficiency_segment = efficiency[i-window:i+1] if i >= window else efficiency[:i+1]
            
            # 統計的閾値（中央値 + 標準偏差の倍数）
            trend_median = np.median(trend_segment)
            trend_std = np.std(trend_segment)
            efficiency_median = np.median(efficiency_segment)
            efficiency_std = np.std(efficiency_segment)
            
            # 実践的閾値設定（BTC相場に最適化）
            trend_threshold_low = max(0.35, trend_median + 0.3 * trend_std)      # トレンド検出下限
            trend_threshold_high = max(0.65, trend_median + 1.0 * trend_std)     # ブレイクアウト検出下限
            efficiency_threshold_low = max(0.25, efficiency_median + 0.2 * efficiency_std)   # 効率性下限
            efficiency_threshold_high = max(0.45, efficiency_median + 0.8 * efficiency_std)  # 高効率性下限
            
            # 追加条件：価格変動率による補正
            if i > 0:
                # 直近の価格変動を考慮
                recent_volatility = abs(regime_raw[i] - regime_raw[i-1]) / regime_raw[i-1] if regime_raw[i-1] != 0 else 0
                volatility_boost = min(0.15, recent_volatility * 5)  # ボラティリティによる閾値引き下げ
                trend_threshold_low -= volatility_boost
                efficiency_threshold_low -= volatility_boost * 0.5
            
            # 🚀 ブレイクアウト判定（最優先）
            if (current_trend > trend_threshold_high and current_efficiency > efficiency_threshold_high):
                market_regime[i] = 2  # ブレイクアウト
            
            # 📈 トレンド判定（中優先）
            elif (current_trend > trend_threshold_low and current_efficiency > efficiency_threshold_low):
                market_regime[i] = 1  # トレンド
                
                # 連続性チェック：前の期間もトレンドなら継続しやすくする
                if i > 0 and market_regime[i-1] == 1:
                    # 継続トレンドの場合、やや緩い条件でも維持
                    if current_trend > trend_threshold_low * 0.85:
                        market_regime[i] = 1
            
            # 📊 レンジ判定（デフォルト）
            else:
                market_regime[i] = 0  # レンジ
                
                # レンジ継続の安定化：急激な変化を抑制
                if i > 2:
                    recent_regimes = market_regime[i-3:i]
                    if np.all(recent_regimes == 0):  # 直近3期間がレンジ
                        # レンジ継続中は、より厳しい条件でのみトレンドに移行
                        if not (current_trend > trend_threshold_low * 1.2 and current_efficiency > efficiency_threshold_low * 1.1):
                            market_regime[i] = 0
        
        # 後処理：ノイズ除去とスムージング
        market_regime = self._smooth_regime_transitions(market_regime)
        
        return market_regime
    
    def _smooth_regime_transitions(self, market_regime: np.ndarray) -> np.ndarray:
        """
        🌊 レジーム遷移のスムージング
        短期間のノイズを除去し、安定したレジーム判定を実現
        """
        n = len(market_regime)
        smoothed = market_regime.copy()
        
        # 短期間の孤立したレジームを修正
        for i in range(2, n-2):
            current = market_regime[i]
            prev2 = market_regime[i-2]
            prev1 = market_regime[i-1]
            next1 = market_regime[i+1]
            next2 = market_regime[i+2]
            
            # 孤立点の修正（前後が同じレジームの場合）
            if prev1 == next1 and prev1 != current:
                if abs(prev2 - prev1) <= 1 and abs(next1 - next2) <= 1:
                    smoothed[i] = prev1
            
            # 短期間の振動を抑制
            if i >= 3 and i < n-1:
                window = market_regime[i-3:i+2]
                unique_values, counts = np.unique(window, return_counts=True)
                if len(unique_values) > 1:
                    # 最頻値で置換（ただし現在値との差が1以下の場合のみ）
                    most_frequent = unique_values[np.argmax(counts)]
                    if abs(current - most_frequent) <= 1:
                        smoothed[i] = most_frequent
        
        return smoothed

    def _calculate_quantum_entanglement(self, amplitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
        """量子もつれ計算"""
        n = len(amplitude)
        entanglement = np.zeros(n)
        
        for i in range(5, n):
            # 振幅と位相の相関による量子もつれ計算
            amp_segment = amplitude[i-5:i]
            phase_segment = phase[i-5:i]
            
            if len(amp_segment) > 1 and len(phase_segment) > 1:
                correlation = np.corrcoef(amp_segment, phase_segment)[0, 1]
                entanglement[i] = abs(correlation) if not np.isnan(correlation) else 0.0
        
        # 前方埋め
        for i in range(5):
            entanglement[i] = entanglement[5] if n > 5 else 0.0
        
        return entanglement

    def _calculate_superposition_state(self, coherence: np.ndarray, entanglement: np.ndarray) -> np.ndarray:
        """重ね合わせ状態計算"""
        return (coherence + entanglement) / 2.0

    def _calculate_adaptation_confidence(self, coherence: np.ndarray, trend_strength: np.ndarray, 
                                       efficiency: np.ndarray) -> np.ndarray:
        """適応信頼度計算"""
        return (coherence * 0.4 + trend_strength * 0.3 + efficiency * 0.3)

    def _calculate_trend_persistence(self, trend_strength: np.ndarray, window: int) -> np.ndarray:
        """トレンド持続性計算"""
        n = len(trend_strength)
        persistence = np.zeros(n)
        
        for i in range(window, n):
            segment = trend_strength[i-window:i]
            # 標準偏差が小さいほど持続性が高い
            std_val = np.std(segment)
            persistence[i] = 1.0 / (1.0 + std_val * 5.0)
        
        # 前方埋め
        for i in range(window):
            persistence[i] = persistence[window] if n > window else 0.5
        
        return persistence

    def _generate_trend_signals(self, prices: np.ndarray, middle_line: np.ndarray, 
                               trend_strength: np.ndarray) -> np.ndarray:
        """トレンドシグナル生成"""
        n = len(prices)
        signals = np.zeros(n)
        
        for i in range(1, n):
            if prices[i] > middle_line[i] and trend_strength[i] > 0.6:
                signals[i] = 1.0  # 上昇シグナル
            elif prices[i] < middle_line[i] and trend_strength[i] > 0.6:
                signals[i] = -1.0  # 下降シグナル
            # else: 0 (ニュートラル)
        
        return signals

    def _determine_current_regime(self, market_regime: np.ndarray) -> str:
        """現在のレジーム判定"""
        if len(market_regime) == 0:
            return 'range'
        
        current_value = market_regime[-1]
        if current_value == 0:
            return 'range'
        elif current_value == 1:
            return 'trend'
        else:
            return 'breakout'

    def _determine_adaptation_mode(self, multiplier: float) -> str:
        """適応モード判定"""
        if multiplier <= 2.0:
            return 'trend_following'
        elif multiplier >= 5.0:
            return 'range_trading'
        else:
            return 'transitional'

    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データハッシュ計算"""
        try:
            if isinstance(data, pd.DataFrame):
                data_str = f"{data.shape}_{data.iloc[0].sum() if len(data) > 0 else 0}_{data.iloc[-1].sum() if len(data) > 0 else 0}"
            else:
                data_str = f"{data.shape}_{data[0] if len(data) > 0 else 0}_{data[-1] if len(data) > 0 else 0}"
            
            param_str = f"{self.analysis_period}_{self.min_multiplier}_{self.max_multiplier}_{self.src_type}"
            return f"{hash(data_str)}_{param_str}"
        except Exception:
            return f"fallback_{hash(str(data))}_{self.analysis_period}"

    # 結果取得メソッド群
    def get_result(self) -> Optional[QuantumSupremeBreakoutChannelResult]:
        """完全な結果を取得"""
        return self._result

    def get_upper_channel(self) -> Optional[np.ndarray]:
        """上位チャネルを取得"""
        if self._result is not None:
            return self._result.upper_channel.copy()
        return None

    def get_middle_line(self) -> Optional[np.ndarray]:
        """ミッドラインを取得"""
        if self._result is not None:
            return self._result.middle_line.copy()
        return None

    def get_lower_channel(self) -> Optional[np.ndarray]:
        """下位チャネルを取得"""
        if self._result is not None:
            return self._result.lower_channel.copy()
        return None

    def get_channels(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """全チャネル（上位、中央、下位）を取得"""
        if self._result is not None:
            return (
                self._result.upper_channel.copy(),
                self._result.middle_line.copy(),
                self._result.lower_channel.copy()
            )
        return None

    def get_dynamic_multiplier(self) -> Optional[np.ndarray]:
        """動的乗数を取得"""
        if self._result is not None:
            return self._result.dynamic_multiplier.copy()
        return None

    def get_market_regime(self) -> Optional[np.ndarray]:
        """市場レジームを取得"""
        if self._result is not None:
            return self._result.market_regime.copy()
        return None

    def get_breakout_signals(self) -> Optional[np.ndarray]:
        """ブレイクアウトシグナルを取得"""
        if self._result is not None:
            return self._result.breakout_signals.copy()
        return None

    def get_breakout_probability(self) -> Optional[np.ndarray]:
        """ブレイクアウト確率を取得"""
        if self._result is not None:
            return self._result.breakout_probability.copy()
        return None

    def get_quantum_metrics(self) -> Optional[Dict]:
        """量子メトリクスを取得"""
        if self._result is not None:
            return {
                'quantum_coherence': self._result.quantum_coherence.copy(),
                'quantum_entanglement': self._result.quantum_entanglement.copy(),
                'superposition_state': self._result.superposition_state.copy()
            }
        return None

    def get_current_status(self) -> Dict:
        """現在の状態情報を取得"""
        if self._result is not None:
            return {
                'current_regime': self._result.current_regime,
                'current_trend_strength': self._result.current_trend_strength,
                'current_breakout_probability': self._result.current_breakout_probability,
                'current_adaptation_mode': self._result.current_adaptation_mode,
                'latest_multiplier': float(self._result.dynamic_multiplier[-1]) if len(self._result.dynamic_multiplier) > 0 else 3.5
            }
        return {
            'current_regime': 'unknown',
            'current_trend_strength': 0.0,
            'current_breakout_probability': 0.0,
            'current_adaptation_mode': 'unknown',
            'latest_multiplier': 3.5
        }

    def reset(self) -> None:
        """インジケータの状態をリセット"""
        super().reset()
        self._result = None
        self._cache = {}
        self._previous_multiplier = 4.75
        
        # サブインジケーターのリセット
        if hasattr(self, 'quantum_hyper_ma'):
            self.quantum_hyper_ma.reset()
        if hasattr(self, 'chop_trend'):
            self.chop_trend.reset()
        if hasattr(self, 'ultimate_volatility'):
            self.ultimate_volatility.reset()
        if hasattr(self, 'efficiency_ratio'):
            self.efficiency_ratio.reset()
        if hasattr(self, 'cycle_detector'):
            self.cycle_detector.reset()

    def __str__(self) -> str:
        """文字列表現"""
        return f"QuantumSupremeBreakoutChannel(period={self.analysis_period}, mult_range={self.min_multiplier}-{self.max_multiplier})"

    # ヘルパーメソッド群
    @staticmethod
    @njit(fastmath=True, cache=True)
    def _calculate_ema_numba(prices: np.ndarray, period: int) -> np.ndarray:
        """EMA計算（Numba最適化）"""
        n = len(prices)
        ema = np.zeros(n)
        alpha = 2.0 / (period + 1.0)
        
        ema[0] = prices[0]
        for i in range(1, n):
            ema[i] = alpha * prices[i] + (1.0 - alpha) * ema[i-1]
        
        return ema

    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """EMA計算"""
        return self._calculate_ema_numba(prices, period)

    def _calculate_atr(self, data: Union[pd.DataFrame, np.ndarray], period: int = 14) -> np.ndarray:
        """ATR計算"""
        if isinstance(data, pd.DataFrame):
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
        else:
            # NumPy配列の場合、[high, low, close]と仮定
            if data.ndim == 2 and data.shape[1] >= 3:
                high = data[:, 1]
                low = data[:, 2]
                close = data[:, 3]
            else:
                # 1次元の場合はクローズ価格として扱い、簡易ATR
                close = data
                return np.full(len(close), np.std(np.diff(close)) if len(close) > 1 else 0.1)

        n = len(close)
        tr = np.zeros(n)
        atr = np.zeros(n)
        
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
        tr[0] = high[0] - low[0] if high[0] > low[0] else 0.1
        
        # ATR計算（Wilder's smoothing）
        atr[0] = tr[0]
        alpha = 1.0 / period
        for i in range(1, n):
            atr[i] = alpha * tr[i] + (1.0 - alpha) * atr[i-1]
        
        return atr

    def _calculate_simple_efficiency(self, prices: np.ndarray, period: int) -> np.ndarray:
        """シンプル効率比計算"""
        n = len(prices)
        efficiency = np.zeros(n)
        
        for i in range(period, n):
            price_change = abs(prices[i] - prices[i-period])
            volatility = np.sum(np.abs(np.diff(prices[i-period:i])))
            
            if volatility > 0:
                efficiency[i] = price_change / volatility
            else:
                efficiency[i] = 0.0
        
        # 前方埋め
        for i in range(period):
            efficiency[i] = efficiency[period] if n > period else 0.0
        
        return efficiency

    def _calculate_volatility_persistence(self, volatility: np.ndarray, window: int) -> np.ndarray:
        """ボラティリティ持続性計算"""
        n = len(volatility)
        persistence = np.zeros(n)
        
        for i in range(window, n):
            vol_segment = volatility[i-window:i]
            if len(vol_segment) > 1:
                # 自己相関による持続性測定
                vol_normalized = (vol_segment - np.mean(vol_segment)) / (np.std(vol_segment) + 1e-10)
                autocorr = np.corrcoef(vol_normalized[:-1], vol_normalized[1:])[0, 1]
                persistence[i] = max(0.0, autocorr) if not np.isnan(autocorr) else 0.0
        
        # 前方埋め
        for i in range(window):
            persistence[i] = persistence[window] if n > window else 0.0
        
        return persistence

    def _calculate_regime_change_probability(self, trend_strength: np.ndarray, efficiency: np.ndarray, 
                                           volatility_persistence: np.ndarray, window: int) -> np.ndarray:
        """レジーム変化確率計算"""
        n = len(trend_strength)
        regime_change_prob = np.zeros(n)
        
        for i in range(window, n):
            # 最近の指標変化を分析
            trend_change = np.std(trend_strength[i-window:i])
            efficiency_change = np.std(efficiency[i-window:i])
            vol_persistence_change = np.std(volatility_persistence[i-window:i])
            
            # 変化の大きさから確率を計算
            total_change = trend_change + efficiency_change + vol_persistence_change
            regime_change_prob[i] = min(0.9, total_change * 2.0)
        
        # 前方埋め
        for i in range(window):
            regime_change_prob[i] = regime_change_prob[window] if n > window else 0.1
        
        return regime_change_prob

    def _calculate_trend_bias(self, trend_signals: np.ndarray) -> np.ndarray:
        """トレンドバイアス計算"""
        return trend_signals.copy()

    def _calculate_asymmetry_factors(self, trend_bias: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """非対称性ファクター計算"""
        n = len(trend_bias)
        asymmetry_up = np.ones(n)
        asymmetry_down = np.ones(n)
        
        # 基本的には対称チャネルを維持
        # 非対称性調整は軽微にとどめる
        for i in range(n):
            if trend_bias[i] > 0:  # 上昇トレンド
                asymmetry_up[i] = 1.05   # 上位チャネルをわずかに広げる
                asymmetry_down[i] = 0.95 # 下位チャネルをわずかに狭める
            elif trend_bias[i] < 0:  # 下降トレンド
                asymmetry_up[i] = 0.95   # 上位チャネルをわずかに狭める
                asymmetry_down[i] = 1.05 # 下位チャネルをわずかに広げる
            # else: 対称のまま (1.0, 1.0)
        
        return asymmetry_up, asymmetry_down 
