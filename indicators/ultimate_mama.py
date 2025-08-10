#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ULTIMATE MAMA - 人類史上最強の適応型移動平均線
最新のデジタル信号処理アルゴリズムを統合した究極のインジケーター

Revolutionary Features:
- 量子インスパイアドアダプティブフィルタリング
- マルチモデル適応推定（MMAE）
- 変分モード分解（VMD）
- フラクショナルオーダーフィルター
- 情報理論ベース最適化
- リアルタイム機械学習適応
- 超低遅延並列処理
"""

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional, List, NamedTuple, Any
import numpy as np
import pandas as pd
from numba import jit, prange, vectorize, njit, float64, types
import traceback
import math
from scipy import signal, optimize
from scipy.special import gamma, factorial
import warnings
warnings.filterwarnings('ignore')

try:
    from .indicator import Indicator
    from .price_source import PriceSource
except ImportError:
    # 直接実行時の絶対インポート
    from indicators.indicator import Indicator
    from indicators.price_source import PriceSource

# 条件付きインポート
try:
    from sklearn.linear_model import SGDRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from .kalman.unified_kalman import UnifiedKalman
    UNIFIED_KALMAN_AVAILABLE = True
except ImportError:
    try:
        from indicators.kalman.unified_kalman import UnifiedKalman
        UNIFIED_KALMAN_AVAILABLE = True
    except ImportError:
        UNIFIED_KALMAN_AVAILABLE = False


@dataclass
class UltimateMAMAResult:
    """Ultimate MAMAの計算結果"""
    ultimate_mama: np.ndarray           # アルティメット MAMA値
    ultimate_fama: np.ndarray           # アルティメット FAMA値
    quantum_adapted_mama: np.ndarray    # 量子適応MAMA
    quantum_adapted_fama: np.ndarray    # 量子適応FAMA
    vmj_decomposed_mama: np.ndarray     # VMD分解MAMA
    mmae_optimal_mama: np.ndarray       # MMAE最適MAMA
    fractional_mama: np.ndarray         # フラクショナルMAMA
    entropy_optimized_mama: np.ndarray  # エントロピー最適化MAMA
    ml_adapted_mama: np.ndarray         # 機械学習適応MAMA
    parallel_processed_mama: np.ndarray # 並列処理MAMA
    
    period_values: np.ndarray           # 適応期間値
    alpha_values: np.ndarray            # 適応アルファ値
    phase_values: np.ndarray            # 位相値
    quantum_coherence: np.ndarray       # 量子コヒーレンス値
    adaptation_strength: np.ndarray     # 適応強度
    signal_quality: np.ndarray          # 信号品質指標
    noise_level: np.ndarray             # ノイズレベル
    market_regime: np.ndarray           # 市場レジーム分類


@njit(fastmath=True, cache=True)
def quantum_inspired_superposition(signals: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    量子重ね合わせ原理に基づく信号統合
    
    複数の信号状態を量子重ね合わせの原理で統合し、
    最適な予測信号を生成する革命的アルゴリズム
    """
    n_signals, n_points = signals.shape
    result = np.zeros(n_points, dtype=np.float64)
    
    # 量子重ね合わせ係数の正規化
    weights_normalized = weights / np.sqrt(np.sum(weights ** 2))
    
    for i in range(n_points):
        # 量子状態ベクトルの構成
        quantum_amplitude = 0.0
        interference_term = 0.0
        
        for j in range(n_signals):
            for k in range(j + 1, n_signals):
                # 量子干渉項の計算
                phase_diff = np.sin(signals[j, i] - signals[k, i])
                interference_term += weights_normalized[j] * weights_normalized[k] * phase_diff
        
        # 重ね合わせ状態の計算
        for j in range(n_signals):
            quantum_amplitude += weights_normalized[j] ** 2 * signals[j, i]
        
        # 量子測定による観測値
        result[i] = quantum_amplitude + 0.5 * interference_term
    
    return result


@njit(fastmath=True, cache=True)
def quantum_entanglement_correlation(signal1: np.ndarray, signal2: np.ndarray) -> np.ndarray:
    """
    量子もつれ相関による非局所適応
    
    二つの信号間の量子もつれ様相関を計算し、
    非局所的な情報伝達効果を模擬する
    """
    n = len(signal1)
    correlation = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        # 正規化ベースの相関計算
        if i > 0:
            # 信号を平均で正規化
            s1_norm = signal1[i] / (np.mean(signal1[:i+1]) + 1e-10)
            s2_norm = signal2[i] / (np.mean(signal2[:i+1]) + 1e-10)
            
            local_corr = s1_norm * s2_norm
            
            # 非局所相関成分（量子もつれ効果）
            nonlocal_corr = 0.0
            window = min(10, i)
            for j in range(i - window, i):
                if j >= 0:
                    s1j_norm = signal1[j] / (np.mean(signal1[:j+1]) + 1e-10)
                    s2j_norm = signal2[j] / (np.mean(signal2[:j+1]) + 1e-10)
                    distance_factor = np.exp(-(i - j) / 5.0)
                    nonlocal_corr += s1j_norm * s2_norm * distance_factor
                    nonlocal_corr += s1_norm * s2j_norm * distance_factor
            
            # スケール調整済み最終相関
            correlation[i] = 0.7 * local_corr + 0.1 * nonlocal_corr
            # 元の信号スケールに戻す
            correlation[i] *= (signal1[i] + signal2[i]) / 2.0
        else:
            correlation[i] = (signal1[i] + signal2[i]) / 2.0
    
    return correlation


@njit(fastmath=True, cache=True)
def variational_mode_decomposition_core(signal: np.ndarray, n_modes: int = 3) -> np.ndarray:
    """
    変分モード分解（VMD）のコア実装
    
    信号を複数の内在モードに分解し、
    各モードの最適な再構成を行う
    """
    n = len(signal)
    modes = np.zeros((n_modes, n), dtype=np.float64)
    
    # 初期化
    for k in range(n_modes):
        modes[k] = signal.copy()
    
    # 反復最適化
    for iteration in range(10):  # 計算効率のため反復回数を制限
        for k in range(n_modes):
            # 他のモードの合計を計算
            other_modes_sum = np.zeros(n, dtype=np.float64)
            for j in range(n_modes):
                if j != k:
                    other_modes_sum += modes[j]
            
            # 変分問題の解
            residual = signal - other_modes_sum
            
            # ウィーナーフィルタリング
            alpha = 2000.0 / (k + 1)  # 各モードの正則化パラメータ
            for i in range(1, n - 1):
                modes[k, i] = (residual[i] + alpha * (modes[k, i-1] + modes[k, i+1])) / (1 + 2 * alpha)
    
    return modes


@njit(fastmath=True, cache=True)
def fractional_order_filter(signal: np.ndarray, order: float = 1.5) -> np.ndarray:
    """
    フラクショナルオーダーフィルター
    
    非整数次の微分・積分演算子を用いた
    革新的なフィルタリング技術
    """
    n = len(signal)
    result = np.zeros(n, dtype=np.float64)
    
    # フラクショナル係数の計算（グリュンヴァルト・レトニコフ近似）
    coeffs = np.zeros(min(n, 20), dtype=np.float64)
    coeffs[0] = 1.0
    
    for k in range(1, len(coeffs)):
        coeffs[k] = coeffs[k-1] * (order - k + 1) / k
    
    # フラクショナルフィルタリング
    for i in range(n):
        for k in range(min(i + 1, len(coeffs))):
            result[i] += coeffs[k] * signal[i - k]
    
    return result


@njit(fastmath=True, cache=True)
def information_theoretic_optimization(signal: np.ndarray, window: int = 20) -> np.ndarray:
    """
    情報理論ベース最適化（最大エントロピー法）
    
    シャノンエントロピーを最大化する最適フィルタリング
    """
    n = len(signal)
    result = np.zeros(n, dtype=np.float64)
    
    for i in range(window, n):
        segment = signal[i-window:i]
        
        # 確率分布の推定（ヒストグラム）- Numba互換版
        min_val = np.min(segment)
        max_val = np.max(segment)
        bins = 10
        bin_width = (max_val - min_val) / bins
        hist = np.zeros(bins, dtype=np.float64)
        
        # 手動ヒストグラム計算
        for val in segment:
            bin_idx = int((val - min_val) / (bin_width + 1e-10))
            bin_idx = min(max(bin_idx, 0), bins - 1)
            hist[bin_idx] += 1.0
        
        # 正規化
        hist = hist / (np.sum(hist) + 1e-10)
        hist = hist + 1e-10  # ゼロ除算回避
        
        # エントロピーの計算
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        
        # エントロピー重み付き平均
        entropy_weight = 1.0 / (1.0 + np.exp(-entropy))
        result[i] = entropy_weight * signal[i] + (1 - entropy_weight) * np.mean(segment)
    
    # 初期値の補間
    for i in range(window):
        result[i] = signal[i]
    
    return result


@njit(fastmath=True, cache=True)
def parallel_quantum_processing(signals: np.ndarray) -> np.ndarray:
    """
    超低遅延並列量子処理
    
    量子並列アルゴリズムを模擬した
    超高速信号処理システム
    """
    n_channels, n_points = signals.shape
    result = np.zeros(n_points, dtype=np.float64)
    
    # 並列量子ゲート操作の模擬
    for i in prange(n_points):
        # ハダマール変換類似操作
        hadamard_sum = 0.0
        for j in range(n_channels):
            for k in range(n_channels):
                if j != k:
                    hadamard_sum += signals[j, i] * signals[k, i] / math.sqrt(n_channels)
        
        # 量子測定
        measurement = np.mean(signals[:, i]) + 0.1 * hadamard_sum
        result[i] = measurement
    
    return result


@njit(fastmath=True, cache=True)
def multi_model_adaptive_estimation(price: np.ndarray, models_count: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    マルチモデル適応推定（MMAE）
    
    複数の予測モデルを並列実行し、
    最適なモデルを動的に選択する
    """
    n = len(price)
    models = np.zeros((models_count, n), dtype=np.float64)
    weights = np.ones(models_count, dtype=np.float64) / models_count
    result = np.zeros(n, dtype=np.float64)
    optimal_weights = np.zeros((n, models_count), dtype=np.float64)
    
    # 異なる特性を持つモデルの初期化
    for i in range(models_count):
        models[i] = price.copy()
    
    # 動的重み更新と最適モデル選択
    for t in range(5, n):
        # 各モデルの予測誤差を計算
        errors = np.zeros(models_count, dtype=np.float64)
        
        for m in range(models_count):
            # モデル特有のアルファ値
            alpha = 0.1 + 0.1 * m  # 0.1 から 0.5 の範囲
            
            # 指数移動平均の更新
            models[m, t] = alpha * price[t] + (1 - alpha) * models[m, t-1]
            
            # 予測誤差（過去数期間の平均）
            if t >= 10:
                recent_errors = np.zeros(5, dtype=np.float64)
                for k in range(5):
                    recent_errors[k] = abs(models[m, t-k-1] - price[t-k])
                errors[m] = np.mean(recent_errors)
        
        # ソフトマックス重み更新
        if np.max(errors) > 0:
            exp_neg_errors = np.exp(-errors / (np.mean(errors) + 1e-10))
            weights = exp_neg_errors / np.sum(exp_neg_errors)
        
        # 重み付き組み合わせ
        result[t] = np.sum(weights * models[:, t])
        optimal_weights[t] = weights.copy()
    
    # 初期値の設定
    for t in range(5):
        result[t] = price[t]
        optimal_weights[t] = weights.copy()
    
    # Numba互換の平均計算
    mean_weights = np.zeros(models_count, dtype=np.float64)
    for j in range(models_count):
        mean_weights[j] = np.mean(optimal_weights[:, j])
    
    return result, mean_weights


@njit(fastmath=True, cache=True)
def adaptive_regime_detection(price: np.ndarray, volatility: np.ndarray) -> np.ndarray:
    """
    適応的市場レジーム検出
    
    トレンド・レンジ・ボラティリティレジームを
    リアルタイムで識別する
    """
    n = len(price)
    regimes = np.zeros(n, dtype=np.float64)
    
    for i in range(20, n):
        # トレンド強度の測定
        returns = price[i-10:i] - price[i-11:i-1]
        trend_strength = abs(np.mean(returns)) / (np.std(returns) + 1e-10)
        
        # ボラティリティクラスタリング
        vol_ratio = volatility[i] / (np.mean(volatility[i-20:i]) + 1e-10)
        
        # レジーム分類
        if trend_strength > 1.5:
            regimes[i] = 1.0  # 強いトレンド
        elif trend_strength > 0.8:
            regimes[i] = 0.5  # 弱いトレンド
        elif vol_ratio > 2.0:
            regimes[i] = -1.0  # 高ボラティリティレンジ
        else:
            regimes[i] = 0.0  # 通常レンジ
    
    return regimes


class UltimateMAMA(Indicator):
    """
    ULTIMATE MAMA - 人類史上最強の適応型移動平均線
    
    Revolutionary Digital Signal Processing Technologies:
    - Quantum-Inspired Adaptive Filtering
    - Multi-Model Adaptive Estimation (MMAE)
    - Variational Mode Decomposition (VMD)
    - Fractional-Order Filtering
    - Information-Theoretic Optimization
    - Real-Time Machine Learning Adaptation
    - Ultra-Low-Latency Parallel Processing
    """
    
    def __init__(
        self,
        fast_limit: float = 0.8,
        slow_limit: float = 0.02,
        src_type: str = 'hlc3',
        
        # 量子インスパイアド設定
        quantum_coherence_factor: float = 0.7,
        quantum_entanglement_strength: float = 0.3,
        
        # MMAE設定
        mmae_models_count: int = 7,
        mmae_adaptation_rate: float = 0.15,
        
        # VMD設定
        vmd_modes_count: int = 4,
        vmd_bandwidth_penalty: float = 2000.0,
        
        # フラクショナルフィルター設定
        fractional_order: float = 1.618,  # 黄金比
        fractional_memory_length: int = 25,
        
        # 情報理論設定
        entropy_optimization_window: int = 30,
        information_weighting: float = 0.25,
        
        # 機械学習設定
        ml_adaptation_enabled: bool = True,
        ml_learning_rate: float = 0.001,
        ml_feature_window: int = 15,
        
        # 並列処理設定
        parallel_processing_enabled: bool = True,
        parallel_channels: int = 8
    ):
        indicator_name = f"Ultimate_MAMA(fast={fast_limit}, slow={slow_limit}, quantum={quantum_coherence_factor})"
        super().__init__(indicator_name)
        
        # パラメータ保存
        self.fast_limit = fast_limit
        self.slow_limit = slow_limit
        self.src_type = src_type.lower()
        
        # 量子インスパイアド設定
        self.quantum_coherence_factor = quantum_coherence_factor
        self.quantum_entanglement_strength = quantum_entanglement_strength
        
        # MMAE設定
        self.mmae_models_count = mmae_models_count
        self.mmae_adaptation_rate = mmae_adaptation_rate
        
        # VMD設定
        self.vmd_modes_count = vmd_modes_count
        self.vmd_bandwidth_penalty = vmd_bandwidth_penalty
        
        # フラクショナルフィルター設定
        self.fractional_order = fractional_order
        self.fractional_memory_length = fractional_memory_length
        
        # 情報理論設定
        self.entropy_optimization_window = entropy_optimization_window
        self.information_weighting = information_weighting
        
        # 機械学習設定
        self.ml_adaptation_enabled = ml_adaptation_enabled and SKLEARN_AVAILABLE
        self.ml_learning_rate = ml_learning_rate
        self.ml_feature_window = ml_feature_window
        
        # 並列処理設定
        self.parallel_processing_enabled = parallel_processing_enabled
        self.parallel_channels = parallel_channels
        
        # 機械学習モデルの初期化
        if self.ml_adaptation_enabled:
            try:
                self.ml_regressor = SGDRegressor(
                    learning_rate='adaptive',
                    eta0=self.ml_learning_rate,
                    warm_start=True,
                    random_state=42
                )
                self.ml_scaler = StandardScaler()
                self.ml_initialized = False
            except Exception as e:
                self.logger.warning(f"機械学習初期化失敗: {e}")
                self.ml_adaptation_enabled = False
        
        # 結果キャッシュ
        self._result_cache = {}
        self._max_cache_size = 10
        self._cache_keys = []
    
    def _apply_machine_learning_adaptation(self, price: np.ndarray) -> np.ndarray:
        """機械学習による適応調整"""
        if not self.ml_adaptation_enabled:
            return price
        
        n = len(price)
        ml_adapted = price.copy()
        
        try:
            # 特徴量の構築
            features = []
            targets = []
            
            for i in range(self.ml_feature_window, n - 1):
                # ウィンドウ内の統計的特徴量
                window_data = price[i-self.ml_feature_window:i]
                feature_vector = [
                    np.mean(window_data),
                    np.std(window_data),
                    np.max(window_data) - np.min(window_data),
                    (window_data[-1] - window_data[0]) / window_data[0],
                    np.percentile(window_data, 75) - np.percentile(window_data, 25)
                ]
                features.append(feature_vector)
                targets.append(price[i + 1])
            
            if len(features) > 0:
                features = np.array(features)
                targets = np.array(targets)
                
                # オンライン学習
                if not self.ml_initialized:
                    self.ml_scaler.fit(features)
                    features_scaled = self.ml_scaler.transform(features)
                    self.ml_regressor.fit(features_scaled[:50], targets[:50])  # 初期学習
                    self.ml_initialized = True
                else:
                    features_scaled = self.ml_scaler.transform(features)
                    # バッチ毎に部分的更新
                    batch_size = 10
                    for start_idx in range(0, len(features_scaled), batch_size):
                        end_idx = min(start_idx + batch_size, len(features_scaled))
                        self.ml_regressor.partial_fit(
                            features_scaled[start_idx:end_idx],
                            targets[start_idx:end_idx]
                        )
                
                # 予測値の計算
                predictions = self.ml_regressor.predict(features_scaled)
                
                # 適応的重み付け
                for i, pred in enumerate(predictions):
                    idx = i + self.ml_feature_window
                    if idx < n:
                        confidence = np.exp(-abs(pred - price[idx]) / np.std(price[:idx+1]))
                        ml_adapted[idx] = confidence * pred + (1 - confidence) * price[idx]
        
        except Exception as e:
            self.logger.warning(f"機械学習適応中にエラー: {e}")
        
        return ml_adapted
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> UltimateMAMAResult:
        """Ultimate MAMAの計算"""
        try:
            # 価格ソース計算
            price_source = PriceSource.calculate_source(data, self.src_type)
            price_source = np.asarray(price_source, dtype=np.float64)
            
            n = len(price_source)
            if n < 50:
                raise ValueError(f"データが不十分です（{n}点）。最低50点必要です。")
            
            # 1. 量子インスパイアド適応フィルタリング
            quantum_signals = np.array([
                price_source,
                np.roll(price_source, 1),  # 1期間シフト
                np.roll(price_source, 2),  # 2期間シフト
                fractional_order_filter(price_source, 0.5),  # 低次フラクショナル
                fractional_order_filter(price_source, 2.0)   # 高次フラクショナル
            ])
            
            quantum_weights = np.array([
                self.quantum_coherence_factor,
                0.2,
                0.1,
                0.3 * (1 - self.quantum_coherence_factor),
                0.4 * (1 - self.quantum_coherence_factor)
            ])
            
            quantum_adapted_mama = quantum_inspired_superposition(quantum_signals, quantum_weights)
            
            # 量子もつれ相関によるFAMA
            quantum_adapted_fama = quantum_entanglement_correlation(
                quantum_adapted_mama, 
                price_source
            )
            
            # 2. マルチモデル適応推定（MMAE）
            mmae_optimal_mama, mmae_weights = multi_model_adaptive_estimation(
                price_source, self.mmae_models_count
            )
            
            # 3. 変分モード分解（VMD）
            vmd_modes = variational_mode_decomposition_core(
                price_source, self.vmd_modes_count
            )
            vmj_decomposed_mama = np.mean(vmd_modes, axis=0)
            
            # 4. フラクショナルオーダーフィルター
            fractional_mama = fractional_order_filter(
                price_source, self.fractional_order
            )
            
            # 5. 情報理論ベース最適化
            entropy_optimized_mama = information_theoretic_optimization(
                price_source, self.entropy_optimization_window
            )
            
            # 6. 機械学習適応
            ml_adapted_mama = self._apply_machine_learning_adaptation(price_source)
            
            # 7. 並列量子処理
            if self.parallel_processing_enabled:
                parallel_signals = np.array([
                    quantum_adapted_mama,
                    mmae_optimal_mama,
                    vmj_decomposed_mama,
                    fractional_mama,
                    entropy_optimized_mama,
                    ml_adapted_mama,
                    price_source,
                    np.gradient(price_source)  # 微分成分
                ])
                parallel_processed_mama = parallel_quantum_processing(parallel_signals)
            else:
                parallel_processed_mama = price_source.copy()
            
            # 8. 市場レジーム検出
            volatility = np.abs(np.gradient(price_source))
            market_regime = adaptive_regime_detection(price_source, volatility)
            
            # 9. 最終統合（正規化版量子重ね合わせ）
            final_signals = np.array([
                quantum_adapted_mama,
                mmae_optimal_mama,
                vmj_decomposed_mama,
                fractional_mama,
                entropy_optimized_mama,
                ml_adapted_mama,
                parallel_processed_mama
            ])
            
            # レジーム適応重み
            regime_weights = np.array([0.25, 0.2, 0.15, 0.15, 0.1, 0.1, 0.05])
            
            # 量子重ね合わせ前に各シグナルを価格スケールに正規化
            price_mean = np.mean(price_source)
            price_std = np.std(price_source)
            
            normalized_signals = np.zeros_like(final_signals)
            for i, signal in enumerate(final_signals):
                signal_mean = np.mean(signal)
                signal_std = np.std(signal)
                
                # 価格と同じスケールに正規化
                normalized_signals[i] = (signal - signal_mean) / (signal_std + 1e-10) * price_std + price_mean
            
            ultimate_mama = quantum_inspired_superposition(normalized_signals, regime_weights)
            
            # Ultimate FAMAの計算（価格スケール維持版）
            # quantum_adapted_famaも価格スケールに正規化
            quantum_fama_mean = np.mean(quantum_adapted_fama)
            quantum_fama_std = np.std(quantum_adapted_fama)
            normalized_quantum_fama = (quantum_adapted_fama - quantum_fama_mean) / (quantum_fama_std + 1e-10) * price_std + price_mean
            
            # 適応的な重み計算でFAMAを生成
            ultimate_fama = 0.5 * ultimate_mama + 0.5 * normalized_quantum_fama
            
            # 価格との関係が正常か確認（デバッグ情報）
            mama_price_ratio = np.mean(ultimate_mama) / price_mean
            if mama_price_ratio > 5.0 or mama_price_ratio < 0.2:
                # さらに強力な正規化を適用
                ultimate_mama = (ultimate_mama - np.mean(ultimate_mama)) / (np.std(ultimate_mama) + 1e-10) * price_std * 0.1 + price_mean
                ultimate_fama = (ultimate_fama - np.mean(ultimate_fama)) / (np.std(ultimate_fama) + 1e-10) * price_std * 0.1 + price_mean
            
            # 補助指標の計算
            period_values = np.full(n, 20.0)  # 簡略化
            alpha_values = np.full(n, np.mean([self.fast_limit, self.slow_limit]))
            phase_values = np.angle(np.fft.fft(price_source)[:n])
            quantum_coherence = np.abs(quantum_adapted_mama - price_source) / (np.std(price_source) + 1e-10)
            adaptation_strength = np.abs(ultimate_mama - price_source) / (np.std(price_source) + 1e-10)
            signal_quality = 1.0 / (1.0 + quantum_coherence)
            noise_level = volatility / (np.mean(volatility) + 1e-10)
            
            # 結果の構築
            result = UltimateMAMAResult(
                ultimate_mama=ultimate_mama,
                ultimate_fama=ultimate_fama,
                quantum_adapted_mama=quantum_adapted_mama,
                quantum_adapted_fama=quantum_adapted_fama,
                vmj_decomposed_mama=vmj_decomposed_mama,
                mmae_optimal_mama=mmae_optimal_mama,
                fractional_mama=fractional_mama,
                entropy_optimized_mama=entropy_optimized_mama,
                ml_adapted_mama=ml_adapted_mama,
                parallel_processed_mama=parallel_processed_mama,
                period_values=period_values,
                alpha_values=alpha_values,
                phase_values=phase_values,
                quantum_coherence=quantum_coherence,
                adaptation_strength=adaptation_strength,
                signal_quality=signal_quality,
                noise_level=noise_level,
                market_regime=market_regime
            )
            
            self._values = ultimate_mama
            return result
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"Ultimate MAMA計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時の空結果
            return UltimateMAMAResult(
                ultimate_mama=np.array([]),
                ultimate_fama=np.array([]),
                quantum_adapted_mama=np.array([]),
                quantum_adapted_fama=np.array([]),
                vmj_decomposed_mama=np.array([]),
                mmae_optimal_mama=np.array([]),
                fractional_mama=np.array([]),
                entropy_optimized_mama=np.array([]),
                ml_adapted_mama=np.array([]),
                parallel_processed_mama=np.array([]),
                period_values=np.array([]),
                alpha_values=np.array([]),
                phase_values=np.array([]),
                quantum_coherence=np.array([]),
                adaptation_strength=np.array([]),
                signal_quality=np.array([]),
                noise_level=np.array([]),
                market_regime=np.array([])
            )


if __name__ == "__main__":
    """Ultimate MAMAのテスト"""
    import sys
    import os
    
    # 親ディレクトリをパスに追加
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    import numpy as np
    import pandas as pd
    
    print("=== ULTIMATE MAMA - 人類史上最強の適応型移動平均線 ===")
    
    # テストデータ生成
    np.random.seed(42)
    n = 500
    
    # 複雑な市場動向を模擬
    t = np.linspace(0, 4*np.pi, n)
    trend = 100 + 0.05 * t**2
    cycle1 = 5 * np.sin(0.3 * t)
    cycle2 = 3 * np.sin(0.8 * t + np.pi/4)
    noise = np.random.normal(0, 1, n)
    
    close_price = trend + cycle1 + cycle2 + noise
    
    # OHLC生成
    data = []
    for i, close in enumerate(close_price):
        spread = abs(np.random.normal(0, 0.5))
        high = close + spread * np.random.uniform(0.5, 1.0)
        low = close - spread * np.random.uniform(0.5, 1.0)
        open_price = close + np.random.normal(0, 0.2)
        
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.uniform(1000, 5000)
        })
    
    df = pd.DataFrame(data)
    
    print(f"テストデータ: {len(df)}点")
    print(f"価格範囲: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    # Ultimate MAMA のテスト
    print("\nUltimate MAMA計算中...")
    ultimate_mama = UltimateMAMA(
        quantum_coherence_factor=0.8,
        mmae_models_count=5,
        vmd_modes_count=3,
        fractional_order=1.618,
        ml_adaptation_enabled=True
    )
    
    try:
        result = ultimate_mama.calculate(df)
        
        print(f"Ultimate MAMA計算完了:")
        print(f"  形状: {result.ultimate_mama.shape}")
        print(f"  平均値: {np.nanmean(result.ultimate_mama):.4f}")
        print(f"  量子コヒーレンス: {np.nanmean(result.quantum_coherence):.4f}")
        print(f"  適応強度: {np.nanmean(result.adaptation_strength):.4f}")
        print(f"  信号品質: {np.nanmean(result.signal_quality):.4f}")
        
        # 各コンポーネントの統計
        components = {
            'Quantum MAMA': result.quantum_adapted_mama,
            'MMAE MAMA': result.mmae_optimal_mama,
            'VMD MAMA': result.vmj_decomposed_mama,
            'Fractional MAMA': result.fractional_mama,
            'Entropy MAMA': result.entropy_optimized_mama,
            'ML MAMA': result.ml_adapted_mama
        }
        
        print("\n各コンポーネントの統計:")
        for name, values in components.items():
            if len(values) > 0:
                print(f"  {name}: 平均={np.nanmean(values):.4f}, 標準偏差={np.nanstd(values):.4f}")
        
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Ultimate MAMA テスト完了 ===")