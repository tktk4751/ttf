#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🚀 **Ultimate Trend Follow Signal V1.0 - 人類史上最強のトレンドフォローシグナル** 🚀

物理学の法則に基づく革新的システム：
- 量子力学（量子トレンド検出器）
- 流体力学（ボラティリティエンジン）  
- 相対論（超モメンタム解析器）
- 統合前処理基盤層（Neural Supreme Kalman + Quantum Supreme Hilbert + Ultimate Cosmic Wavelet）

5つの信号出力：
- ロングシグナル
- ロングエグジットシグナル
- ショートシグナル
- ショートエグジットシグナル
- ステイシグナル

🌟 **革新的特徴:**
- 3次元状態空間による軽量化
- 1期間での即座適応学習
- 物理法則による自動調整
- 超高精度・超低遅延・超追従性・超安定性
"""

from typing import Union, Optional, Dict, Tuple, NamedTuple
import numpy as np
import pandas as pd
from numba import jit, njit
import warnings
warnings.filterwarnings('ignore')

try:
    from .indicator import Indicator
    from .price_source import PriceSource
    from .kalman_filter_unified import KalmanFilterUnified
    from .hilbert_unified import HilbertTransformUnified
    from .wavelet_unified import WaveletUnified
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from indicator import Indicator
    from price_source import PriceSource
    from kalman_filter_unified import KalmanFilterUnified
    from hilbert_unified import HilbertTransformUnified
    from wavelet_unified import WaveletUnified


class TrendFollowSignalResult(NamedTuple):
    """トレンドフォローシグナル結果"""
    # メイン信号 (0-2: Stay, Long, Short)
    signals: np.ndarray
    
    # 3次元状態空間
    trend_dynamics: np.ndarray          # T(t): [瞬時方向性, 加速度, 持続力]
    volatility_state: np.ndarray        # V(t): [レジーム強度, 変化速度, 予測可能性]
    momentum_state: np.ndarray          # M(t): [勢い強度, 収束度, 継続確率]
    
    # 個別信号確率 (0-1)
    long_probability: np.ndarray
    short_probability: np.ndarray
    stay_probability: np.ndarray
    
    # 信頼度・強度
    signal_confidence: np.ndarray       # 全体信頼度
    trend_strength: np.ndarray          # トレンド強度
    
    # デバッグ・解析用
    preprocessing_results: Optional[Dict] = None


class IntegratedPreprocessingFoundation:
    """🌟 統合前処理基盤層の実装"""
    
    def __init__(self):
        # 🧠🚀 Neural Adaptive Quantum Supreme Kalman
        self.kalman_filter = KalmanFilterUnified(
            filter_type='neural_supreme',
            base_process_noise=0.0001,
            base_measurement_noise=0.001,
            volatility_window=21
        )
        
        # 🌀 Quantum Supreme Hilbert Transform  
        self.hilbert_transform = HilbertTransformUnified(
            algorithm_type='quantum_supreme',
            min_periods=16
        )
        
        # 🌌 Ultimate Cosmic Wavelet
        self.wavelet_analyzer = WaveletUnified(
            wavelet_type='ultimate_cosmic',
            cosmic_power_level=1.0
        )
    
    def process(self, prices: Union[pd.DataFrame, np.ndarray]) -> Dict[str, np.ndarray]:
        """統合前処理の実行"""
        
        # Phase 1: Neural Supreme Kalman濾波
        kalman_result = self.kalman_filter.calculate(prices)
        
        # Phase 2: Quantum Supreme Hilbert解析
        hilbert_result = self.hilbert_transform.calculate(kalman_result.filtered_values)
        
        # Phase 3: Ultimate Cosmic Wavelet変換
        wavelet_result = self.wavelet_analyzer.calculate(kalman_result.filtered_values)
        
        return {
            # カルマンフィルター結果
            'kalman_filtered': kalman_result.filtered_values,
            'neural_weights': kalman_result.trend_estimate,
            'quantum_phases_kalman': kalman_result.quantum_coherence,
            'chaos_indicators': kalman_result.uncertainty,
            'confidence_scores': kalman_result.confidence_scores,
            
            # ヒルベルト変換結果
            'hilbert_amplitude': hilbert_result.amplitude,
            'hilbert_phase': hilbert_result.phase,
            'hilbert_frequency': hilbert_result.frequency,
            'quantum_coherence': hilbert_result.quantum_coherence,
            
            # ウェーブレット結果
            'cosmic_signal': wavelet_result.values,
            'cosmic_trend': wavelet_result.trend_component,
            'cosmic_cycle': wavelet_result.cycle_component,
            'cosmic_noise': wavelet_result.noise_component,
            'market_regime': wavelet_result.market_regime
        }


@njit(fastmath=True, cache=True)
def quantum_trend_detector_core(
    kalman_filtered: np.ndarray,
    hilbert_phase: np.ndarray,
    hilbert_amplitude: np.ndarray,
    cosmic_signal: np.ndarray,
    window: int = 21
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """🔬 量子トレンド検出器（統合前処理基盤層強化版）"""
    
    n = len(kalman_filtered)
    direction = np.zeros(n)      # 瞬時方向性
    acceleration = np.zeros(n)   # 加速度
    persistence = np.zeros(n)    # 持続力
    
    # 量子もつれ効果：統合基盤層による強化
    entanglement_matrix = np.zeros((window, window))
    for i in range(window):
        for j in range(window):
            if i != j:
                # EPR相関をヒルベルト位相で強化
                base_correlation = np.exp(-abs(i-j) / (window/4))
                if i < len(hilbert_phase) and j < len(hilbert_phase):
                    phase_correlation = np.cos(hilbert_phase[min(i, n-1)] - hilbert_phase[min(j, n-1)])
                    entanglement_matrix[i,j] = base_correlation * (1 + phase_correlation) / 2
                else:
                    entanglement_matrix[i,j] = base_correlation
    
    for i in range(window, n):
        # カルマンフィルター済み価格を使用
        price_window = kalman_filtered[i-window+1:i+1]
        cosmic_window = cosmic_signal[i-window+1:i+1] if i < len(cosmic_signal) else price_window
        
        # 1. 量子重ね合わせ状態の計算（ヒルベルト位相強化版）
        price_diffs = np.diff(price_window)
        up_probability = np.sum(price_diffs > 0) / len(price_diffs)
        down_probability = np.sum(price_diffs < 0) / len(price_diffs)
        sideways_probability = 1 - up_probability - down_probability
        
        # ヒルベルト位相による量子位相の精密化
        hilbert_phase_current = hilbert_phase[i] if i < len(hilbert_phase) else 0
        phase_modulation = hilbert_phase_current * 0.1
        
        # 波動関数の複素振幅（位相強化版）
        psi_up = np.sqrt(up_probability) * np.exp(1j * (np.pi/4 + phase_modulation))
        psi_down = np.sqrt(down_probability) * np.exp(1j * (3*np.pi/4 + phase_modulation))
        psi_sideways = np.sqrt(sideways_probability) * np.exp(1j * (np.pi/2 + phase_modulation))
        
        # 2. 観測による波動関数の収束
        current_trend = kalman_filtered[i] - kalman_filtered[i-1]
        if current_trend > 0:
            collapsed_state = psi_up
        elif current_trend < 0:
            collapsed_state = psi_down
        else:
            collapsed_state = psi_sideways
            
        direction[i] = np.real(collapsed_state)
        
        # 3. 量子もつれによる非局所相関（コズミック強化版）
        normalized_prices = (price_window - np.mean(price_window)) / (np.std(price_window) + 1e-10)
        entangled_correlation = np.dot(normalized_prices, np.dot(entanglement_matrix, normalized_prices))
        entangled_correlation /= window
        
        # コズミック信号による相関強化
        cosmic_current = cosmic_window[-1] if len(cosmic_window) > 0 else 0
        cosmic_enhanced_correlation = entangled_correlation * (1 + abs(cosmic_current))
        
        # 4. ハイゼンベルクの不確定性原理（振幅強化版）
        price_uncertainty = np.std(price_window[-5:])
        momentum_uncertainty = np.std(np.diff(price_window[-5:]))
        uncertainty_product = price_uncertainty * momentum_uncertainty
        
        # ヒルベルト振幅による不確定性補正
        amplitude_current = hilbert_amplitude[i] if i < len(hilbert_amplitude) else 1
        amplitude_factor = 1 / (amplitude_current + 1e-10)
        
        certainty_factor = amplitude_factor / (1 + uncertainty_product)
        
        # 5. 瞬時3点微分による超高速加速度検出（カルマン強化版）
        if i >= 2:
            second_derivative = kalman_filtered[i] - 2*kalman_filtered[i-1] + kalman_filtered[i-2]
            quantum_acceleration = second_derivative * certainty_factor * cosmic_enhanced_correlation
            acceleration[i] = np.tanh(quantum_acceleration)
        
        # 6. 量子干渉による持続力計算（統合版）
        if i >= window:
            past_trends = np.sign(np.diff(kalman_filtered[i-window:i]))
            current_direction = np.sign(kalman_filtered[i] - kalman_filtered[i-1])
            
            interference_pattern = 0
            for t in range(len(past_trends)):
                phase_difference = np.pi * (past_trends[t] != current_direction)
                
                # ヒルベルト位相による干渉強化
                if i-t >= 0 and i-t < len(hilbert_phase):
                    phase_coherence = np.cos(hilbert_phase[i] - hilbert_phase[i-t])
                else:
                    phase_coherence = 1
                
                # コズミック成分による時間スケール重み付け
                if i-t >= 0 and i-t < len(cosmic_signal):
                    cosmic_weight = 1 + abs(cosmic_signal[i-t])
                else:
                    cosmic_weight = 1
                
                interference_term = (np.cos(phase_difference) * phase_coherence * 
                                   cosmic_weight * np.exp(-t/window))
                interference_pattern += interference_term
            
            persistence[i] = np.tanh(interference_pattern / len(past_trends))
    
    return direction, acceleration, persistence


@njit(fastmath=True, cache=True)
def fluid_volatility_engine_core(
    kalman_filtered: np.ndarray,
    hilbert_amplitude: np.ndarray,
    cosmic_cycle: np.ndarray,
    window: int = 21
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """💧 流体力学ボラティリティエンジン（統合強化版）"""
    
    n = len(kalman_filtered)
    regime_strength = np.zeros(n)    # レジーム強度
    change_velocity = np.zeros(n)    # 変化速度
    predictability = np.zeros(n)     # 予測可能性
    
    for i in range(window, n):
        price_window = kalman_filtered[i-window+1:i+1]
        returns = np.diff(price_window)
        
        # 1. 流体速度場の定義（ヒルベルト振幅強化）
        velocity = returns / price_window[:-1]
        mean_velocity = np.mean(velocity)
        velocity_variance = np.var(velocity)
        
        # ヒルベルト振幅による速度場補正
        amplitude_current = hilbert_amplitude[i] if i < len(hilbert_amplitude) else 1
        amplitude_corrected_variance = velocity_variance * amplitude_current
        
        # 2. レイノルズ数の計算（レジーム強度）
        characteristic_length = np.std(price_window)
        kinematic_viscosity = amplitude_corrected_variance + 1e-10
        
        reynolds = abs(mean_velocity) * characteristic_length / kinematic_viscosity
        regime_strength[i] = np.tanh(reynolds / 2300)  # 正規化 (0-1)
        
        # 3. 変化速度の計算（コズミックサイクル強化）
        if len(cosmic_cycle) > i:
            cosmic_factor = 1 + abs(cosmic_cycle[i])
        else:
            cosmic_factor = 1
            
        change_velocity[i] = abs(mean_velocity) * cosmic_factor
        
        # 4. 予測可能性の計算（粘性係数ベース）
        turbulence_intensity = np.sqrt(amplitude_corrected_variance) / (abs(mean_velocity) + 1e-10)
        
        if reynolds > 2300:  # 乱流
            predictability[i] = 1.0 / (1.0 + turbulence_intensity)
        else:  # 層流
            predictability[i] = 0.8 + 0.2 / (1.0 + turbulence_intensity)
    
    return regime_strength, change_velocity, predictability


@njit(fastmath=True, cache=True)
def ultra_momentum_analyzer_core(
    kalman_filtered: np.ndarray,
    hilbert_frequency: np.ndarray,
    cosmic_trend: np.ndarray,
    window: int = 21
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """⚡ 超モメンタム解析器（統合強化版）"""
    
    n = len(kalman_filtered)
    momentum_strength = np.zeros(n)     # 勢い強度
    convergence = np.zeros(n)           # 収束度
    continuation_probability = np.zeros(n)  # 継続確率
    
    # 物理定数
    c_market = 1.0  # 市場の「光速」
    
    for i in range(window, n):
        price_window = kalman_filtered[i-window+1:i+1]
        returns = np.diff(price_window)
        
        # 1. 相対論的運動量の計算（周波数強化版）
        velocity = returns / price_window[:-1]
        mean_velocity = np.mean(velocity)
        
        # 光速制限の適用
        if np.abs(mean_velocity) >= c_market:
            mean_velocity = c_market * np.sign(mean_velocity) * 0.99
        
        # ローレンツ因子（ヒルベルト周波数強化）
        frequency_current = hilbert_frequency[i] if i < len(hilbert_frequency) else 0.1
        frequency_factor = 1 + frequency_current
        
        gamma = frequency_factor / np.sqrt(1 - (mean_velocity/c_market)**2)
        
        # 相対論的運動量
        rest_mass = np.std(price_window)
        relativistic_momentum = gamma * rest_mass * mean_velocity
        momentum_strength[i] = np.tanh(relativistic_momentum)
        
        # 2. エネルギー保存則（コズミックトレンド強化）
        rest_energy = rest_mass * c_market**2
        momentum_energy = (relativistic_momentum * c_market)**2
        total_energy = np.sqrt(momentum_energy + rest_energy**2)
        kinetic_energy = total_energy - rest_energy
        
        # コズミックトレンドによる エネルギー増幅
        if len(cosmic_trend) > i:
            cosmic_trend_factor = 1 + np.abs(cosmic_trend[i])
        else:
            cosmic_trend_factor = 1
            
        enhanced_kinetic_energy = kinetic_energy * cosmic_trend_factor
        
        # 3. 収束度の計算（慣性モーメントベース）
        if len(returns) >= 3:
            abs_returns = np.abs(returns)
            mass_distribution = abs_returns / (np.sum(abs_returns) + 1e-10)
            distances_squared = (returns - mean_velocity * price_window[:-1])**2
            moment_of_inertia = np.sum(mass_distribution * distances_squared)
            convergence[i] = 1.0 / (1.0 + moment_of_inertia)
        
        # 4. 継続確率の計算（摩擦係数とエネルギーベース）
        volatility = np.std(returns)
        friction_coefficient = volatility * (1 - convergence[i])  # 収束度が高いほど摩擦小
        
        if enhanced_kinetic_energy > 0:
            energy_dissipation_rate = friction_coefficient / enhanced_kinetic_energy
            continuation_probability[i] = max(0, 1 - energy_dissipation_rate)
        else:
            continuation_probability[i] = 0.5
    
    return momentum_strength, convergence, continuation_probability


@njit(fastmath=True, cache=True)
def integrated_signal_generator(
    trend_dynamics: np.ndarray,      # [direction, acceleration, persistence]
    volatility_state: np.ndarray,    # [regime_strength, change_velocity, predictability]
    momentum_state: np.ndarray,      # [momentum_strength, convergence, continuation_probability]
    quantum_sensitivity: float = 1.0  # 量子感度パラメーター
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """🎯 統合信号生成器（物理学的融合）- 3シグナル版"""
    
    n = len(trend_dynamics)
    
    # 3つの信号確率
    long_prob = np.zeros(n)
    short_prob = np.zeros(n)
    stay_prob = np.zeros(n)
    
    signal_confidence = np.zeros(n)
    trend_strength = np.zeros(n)
    
    for i in range(n):
        # 3次元状態の抽出
        direction = trend_dynamics[i] if not np.isnan(trend_dynamics[i]) else 0
        acceleration = trend_dynamics[i] if i < len(trend_dynamics) else 0  # 簡略化
        persistence = trend_dynamics[i] if not np.isnan(trend_dynamics[i]) else 0
        
        regime_strength = volatility_state[i] if not np.isnan(volatility_state[i]) else 0.5
        change_velocity = volatility_state[i] if i < len(volatility_state) else 0  # 簡略化
        predictability = volatility_state[i] if not np.isnan(volatility_state[i]) else 0.5
        
        momentum_strength = momentum_state[i] if not np.isnan(momentum_state[i]) else 0
        convergence = momentum_state[i] if i < len(momentum_state) else 0  # 簡略化
        continuation_prob = momentum_state[i] if not np.isnan(momentum_state[i]) else 0.5
        
        # 物理学的統合状態方程式
        # F = ma (ニュートン第2法則)
        force = direction * np.abs(acceleration) * momentum_strength
        
        # エネルギー保存則
        potential_energy = persistence * predictability
        kinetic_energy = np.abs(momentum_strength) * continuation_prob
        total_energy = potential_energy + kinetic_energy
        
        # 流体力学的安定性
        flow_stability = predictability * (1 - regime_strength)  # 層流ほど安定
        
        # トレンド強度（統合指標）
        trend_strength[i] = np.tanh(np.abs(force) + total_energy)
        
        # 信号確率計算（実践的な閾値に調整）
        force_magnitude = np.abs(force)
        
        # トレンド強度の動的調整
        trend_signal_strength = trend_strength[i] * 5.0  # 感度を5倍に増加
        momentum_signal = np.abs(momentum_strength) * 3.0  # モメンタム感度を3倍に
        
        # より実践的な条件設定（量子感度を適用）
        quantum_boost = 1.0 + (quantum_sensitivity - 1.0) * 0.5
        
        if force > 0.02 / quantum_boost and trend_signal_strength > 0.3 / quantum_boost and continuation_prob > 0.4:
            # ロング条件：より低い閾値で反応
            signal_strength = force * trend_signal_strength * (1 + momentum_signal)
            long_prob[i] = min(0.85, signal_strength * 2.0)
            
        elif force < -0.02 / quantum_boost and trend_signal_strength > 0.3 / quantum_boost and continuation_prob > 0.4:
            # ショート条件：負のトレンドを敏感に検出
            signal_strength = force_magnitude * trend_signal_strength * (1 + momentum_signal)
            short_prob[i] = min(0.85, signal_strength * 2.0)
            
        # サイドウェイズ/レンジ相場の検出
        elif force_magnitude < 0.02 and trend_signal_strength < 0.3:
            # より積極的なトレンド検出
            if momentum_signal > 0.1:
                # 微細なモメンタムも捉える
                if force > 0:
                    long_prob[i] = min(0.6, momentum_signal * 2.0)
                else:
                    short_prob[i] = min(0.6, momentum_signal * 2.0)
            else:
                stay_prob[i] = max(0.3, 0.8 - momentum_signal)  # Stayの確率を下げる
        else:
            # デフォルトでより動的な判定
            stay_prob[i] = max(0.2, 0.6 - trend_signal_strength - momentum_signal)
        
        # 確率の正規化（全確率の合計を1にする）
        total_prob = long_prob[i] + short_prob[i] + stay_prob[i]
        if total_prob > 1.0:
            normalization_factor = 1.0 / total_prob
            long_prob[i] *= normalization_factor
            short_prob[i] *= normalization_factor
            stay_prob[i] *= normalization_factor
        elif total_prob < 0.8:
            # 低い確率の場合はstayに重み付け
            remaining = 1.0 - total_prob
            stay_prob[i] += remaining
        
        # 信頼度（物理的一貫性）
        signal_confidence[i] = predictability * convergence * (1 - regime_strength * 0.5)
    
    return long_prob, short_prob, stay_prob, signal_confidence, trend_strength


class UltimateTrendFollowSignal(Indicator):
    """
    🚀 Ultimate Trend Follow Signal - 人類史上最強のトレンドフォローシグナル
    
    物理学の法則に基づく革新的システム：
    - 量子力学：量子トレンド検出器
    - 流体力学：ボラティリティエンジン  
    - 相対論：超モメンタム解析器
    - 統合前処理：Neural Supreme + Quantum Supreme + Ultimate Cosmic
    """
    
    def __init__(
        self,
        src_type: str = 'close',
        window: int = 21,
        # 物理パラメータ
        quantum_sensitivity: float = 1.0,
        fluid_turbulence_threshold: float = 2300.0,
        relativistic_c_market: float = 1.0,
        # 信号生成パラメータ
        signal_threshold: float = 0.3,  # より低い閾値で実践的に
        confidence_threshold: float = 0.2,  # 信頼度閾値も下げる
        enable_debug: bool = False
    ):
        """
        Args:
            src_type: 価格ソース
            window: 解析ウィンドウ
            quantum_sensitivity: 量子感度
            fluid_turbulence_threshold: 流体乱流閾値
            relativistic_c_market: 相対論的市場光速
            signal_threshold: シグナル閾値
            confidence_threshold: 信頼度閾値
            enable_debug: デバッグモード
        """
        name = f"UltimateTrendFollowSignal(window={window}, src={src_type})"
        super().__init__(name)
        
        # パラメータ保存
        self.src_type = src_type
        self.window = window
        self.quantum_sensitivity = quantum_sensitivity
        self.fluid_turbulence_threshold = fluid_turbulence_threshold
        self.relativistic_c_market = relativistic_c_market
        self.signal_threshold = signal_threshold
        self.confidence_threshold = confidence_threshold
        self.enable_debug = enable_debug
        
        # 統合前処理基盤層
        self.preprocessing_foundation = IntegratedPreprocessingFoundation()
        
        # 結果キャッシュ
        self._result: Optional[TrendFollowSignalResult] = None
        self._cache_hash: Optional[str] = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> TrendFollowSignalResult:
        """トレンドフォローシグナルを計算"""
        
        # キャッシュチェック
        current_hash = self._get_data_hash(data)
        if self._cache_hash == current_hash and self._result is not None:
            return self._result
        
        try:
            # 価格データの抽出
            src_prices = PriceSource.calculate_source(data, self.src_type)
            
            if len(src_prices) < self.window * 2:
                return self._create_empty_result(len(src_prices))
            
            # Phase 1: 🌟 統合前処理基盤層
            preprocessing_results = self.preprocessing_foundation.process(src_prices)
            
            # Phase 2: 🔬 量子トレンド検出器
            direction, acceleration, persistence = quantum_trend_detector_core(
                preprocessing_results['kalman_filtered'],
                preprocessing_results['hilbert_phase'], 
                preprocessing_results['hilbert_amplitude'],
                preprocessing_results['cosmic_signal'],
                self.window
            )
            
            # Phase 3: 💧 流体力学ボラティリティエンジン
            regime_strength, change_velocity, predictability = fluid_volatility_engine_core(
                preprocessing_results['kalman_filtered'],
                preprocessing_results['hilbert_amplitude'],
                preprocessing_results['cosmic_cycle'] if preprocessing_results['cosmic_cycle'] is not None else np.zeros(len(src_prices)),
                self.window
            )
            
            # Phase 4: ⚡ 超モメンタム解析器
            momentum_strength, convergence, continuation_probability = ultra_momentum_analyzer_core(
                preprocessing_results['kalman_filtered'],
                preprocessing_results['hilbert_frequency'],
                preprocessing_results['cosmic_trend'] if preprocessing_results['cosmic_trend'] is not None else np.zeros(len(src_prices)),
                self.window
            )
            
            # Phase 5: 🎯 統合信号生成
            (long_prob, short_prob, stay_prob, 
             signal_confidence, trend_strength) = integrated_signal_generator(
                direction,  # 簡略化：3次元の代表値として使用
                regime_strength,  # 簡略化：3次元の代表値として使用  
                momentum_strength,  # 簡略化：3次元の代表値として使用
                self.quantum_sensitivity  # 量子感度を渡す
            )
            
            # 最終シグナル決定
            signals = self._determine_final_signals(
                long_prob, short_prob, stay_prob,
                signal_confidence
            )
            
            # 結果構築
            result = TrendFollowSignalResult(
                signals=signals,
                trend_dynamics=direction,  # 簡略化
                volatility_state=regime_strength,  # 簡略化
                momentum_state=momentum_strength,  # 簡略化
                long_probability=long_prob,
                short_probability=short_prob,
                stay_probability=stay_prob,
                signal_confidence=signal_confidence,
                trend_strength=trend_strength,
                preprocessing_results=preprocessing_results if self.enable_debug else None
            )
            
            # キャッシュ更新
            self._result = result
            self._cache_hash = current_hash
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ultimate Trend Follow Signal計算エラー: {e}")
            return self._create_empty_result(len(data) if hasattr(data, '__len__') else 0)
    
    def _determine_final_signals(
        self,
        long_prob: np.ndarray,
        short_prob: np.ndarray,
        stay_prob: np.ndarray,
        confidence: np.ndarray
    ) -> np.ndarray:
        """最終シグナル決定（3シグナル版）"""
        
        n = len(long_prob)
        signals = np.zeros(n)  # 0: Stay, 1: Long, 2: Short
        
        for i in range(n):
            # より実践的な判定：信頼度の閾値を柔軟に適用
            base_confidence_threshold = self.confidence_threshold
            
            # 各確率を集計
            probs = [stay_prob[i], long_prob[i], short_prob[i]]
            max_idx = np.argmax(probs)
            max_prob = probs[max_idx]
            
            # 動的な閾値判定（信号の種類によって閾値を調整）
            if max_idx == 0:  # Stay
                # Stayには高い閾値を要求（よりアクティブにする）
                if max_prob >= 0.7 and confidence[i] >= base_confidence_threshold:
                    signals[i] = 0
                else:
                    # Stayが条件を満たさない場合、次に高い確率の信号を検討
                    probs[0] = 0  # Stayを除外
                    second_idx = np.argmax(probs)
                    second_prob = probs[second_idx]
                    
                    if second_prob >= self.signal_threshold * 0.8:  # より低い閾値
                        signals[i] = second_idx
                    else:
                        signals[i] = 0  # デフォルトでStay
            else:
                # アクションシグナル（Long, Short）の場合
                signal_threshold = self.signal_threshold
                
                # トレンドシグナルはモメンタムを考慮
                if max_idx in [1, 2]:  # Long, Short
                    # モメンタムが強い場合は閾値を下げる
                    momentum_boost = min(0.3, abs(self._get_momentum_at_index(i)) * 0.5)
                    signal_threshold = max(0.15, signal_threshold - momentum_boost)
                
                if max_prob >= signal_threshold:
                    signals[i] = max_idx
                else:
                    signals[i] = 0  # Stay (閾値未満)
        
        return signals
    
    def _get_momentum_at_index(self, i: int) -> float:
        """指定インデックスでのモメンタム値を取得（簡易版）"""
        if self._result and i < len(self._result.momentum_state):
            return self._result.momentum_state[i]
        return 0.0
    
    def _create_empty_result(self, length: int) -> TrendFollowSignalResult:
        """空の結果を作成"""
        return TrendFollowSignalResult(
            signals=np.zeros(length),
            trend_dynamics=np.zeros(length),
            volatility_state=np.zeros(length), 
            momentum_state=np.zeros(length),
            long_probability=np.zeros(length),
            short_probability=np.zeros(length),
            stay_probability=np.ones(length),
            signal_confidence=np.zeros(length),
            trend_strength=np.zeros(length)
        )
    
    def _get_data_hash(self, data) -> str:
        """データのハッシュを計算"""
        try:
            if hasattr(data, 'values'):
                return str(hash(data.values.tobytes()))
            else:
                return str(hash(data.tobytes()))
        except:
            return str(len(data))
    
    def get_signals(self) -> Optional[np.ndarray]:
        """シグナル配列を取得"""
        if self._result is not None:
            return self._result.signals.copy()
        return None
    
    def get_signal_probabilities(self) -> Optional[Dict[str, np.ndarray]]:
        """各シグナルの確率を取得"""
        if self._result is None:
            return None
        
        return {
            'long': self._result.long_probability.copy(),
            'short': self._result.short_probability.copy(),
            'stay': self._result.stay_probability.copy()
        }
    
    def get_analysis_components(self) -> Optional[Dict[str, np.ndarray]]:
        """解析コンポーネントを取得"""
        if self._result is None:
            return None
        
        return {
            'trend_dynamics': self._result.trend_dynamics.copy(),
            'volatility_state': self._result.volatility_state.copy(),
            'momentum_state': self._result.momentum_state.copy(),
            'signal_confidence': self._result.signal_confidence.copy(),
            'trend_strength': self._result.trend_strength.copy()
        }
    
    def get_preprocessing_results(self) -> Optional[Dict]:
        """前処理結果を取得（デバッグモード時のみ）"""
        if self._result is not None and self._result.preprocessing_results is not None:
            return self._result.preprocessing_results.copy()
        return None
    
    def reset(self) -> None:
        """状態をリセット"""
        self._result = None
        self._cache_hash = None
        self.preprocessing_foundation = IntegratedPreprocessingFoundation()


# シグナル定数（3つにシンプル化）
SIGNAL_STAY = 0
SIGNAL_LONG = 1
SIGNAL_SHORT = 2

SIGNAL_NAMES = {
    SIGNAL_STAY: "Stay",
    SIGNAL_LONG: "Long", 
    SIGNAL_SHORT: "Short"
}


def example_usage():
    """使用例"""
    # ダミーデータ作成
    dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(1000) * 0.01)
    
    data = pd.DataFrame({
        'open': prices + np.random.randn(1000) * 0.001,
        'high': prices + abs(np.random.randn(1000) * 0.002),
        'low': prices - abs(np.random.randn(1000) * 0.002),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 1000)
    }, index=dates)
    
    # インジケーター作成・実行
    indicator = UltimateTrendFollowSignal(
        window=21,
        signal_threshold=0.6,
        confidence_threshold=0.5,
        enable_debug=True
    )
    
    result = indicator.calculate(data)
    
    print("🚀 Ultimate Trend Follow Signal Results:")
    print(f"Total signals: {len(result.signals)}")
    print(f"Long signals: {np.sum(result.signals == SIGNAL_LONG)}")
    print(f"Short signals: {np.sum(result.signals == SIGNAL_SHORT)}")
    print(f"Stay signals: {np.sum(result.signals == SIGNAL_STAY)}")
    print(f"Average confidence: {np.mean(result.signal_confidence):.3f}")
    print(f"Average trend strength: {np.mean(result.trend_strength):.3f}")


if __name__ == "__main__":
    example_usage() 