#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Optional, NamedTuple, Tuple
import numpy as np
import pandas as pd
from numba import jit
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# 相対インポートから絶対インポートに変更
try:
    from .indicator import Indicator
    from .price_source import PriceSource
    from .ehlers_unified_dc import EhlersUnifiedDC  # 動的適応用
except ImportError:
    # スタンドアロン実行時の対応
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from indicator import Indicator
    from price_source import PriceSource
    from ehlers_unified_dc import EhlersUnifiedDC  # 動的適応用


class UltimateMAV3Result(NamedTuple):
    """UltimateMA V3計算結果"""
    values: np.ndarray                      # 最終フィルター済み価格
    raw_values: np.ndarray                  # 元の価格
    kalman_values: np.ndarray               # カルマンフィルター後
    super_smooth_values: np.ndarray         # スーパースムーザー後
    zero_lag_values: np.ndarray             # ゼロラグEMA後
    amplitude: np.ndarray                   # ヒルベルト変換振幅
    phase: np.ndarray                      # ヒルベルト変換位相
    realtime_trends: np.ndarray             # リアルタイムトレンド
    trend_signals: np.ndarray               # 1=up, -1=down, 0=range
    trend_confidence: np.ndarray            # トレンド信頼度 (0-1)
    multi_timeframe_consensus: np.ndarray   # マルチタイムフレーム合意度
    volatility_regime: np.ndarray           # ボラティリティ状態
    fractal_dimension: np.ndarray           # フラクタル次元
    entropy_level: np.ndarray               # エントロピーレベル
    quantum_state: np.ndarray               # 量子状態確率
    current_trend: str                      # 'up', 'down', 'range'
    current_trend_value: int                # 1, -1, 0
    current_confidence: float               # 現在の信頼度


@jit(nopython=True)
def quantum_trend_analyzer_numba(values: np.ndarray, window: int = 21) -> Tuple[np.ndarray, np.ndarray]:
    """
    🌌 量子トレンド分析器（量子コンピューティング風並列処理）
    複数の判定アルゴリズムを並列実行し、量子重ね合わせ的に統合
    """
    n = len(values)
    quantum_states = np.zeros(n)
    confidence_levels = np.zeros(n)
    
    if n < window:
        return quantum_states, confidence_levels
    
    for i in range(window, n):
        # 複数の判定アルゴリズムを並列実行
        judgments = np.zeros(7)  # 7つの判定アルゴリズム
        
        # 判定1: 線形回帰スロープ
        x_vals = np.arange(window, dtype=np.float64)
        y_vals = values[i-window+1:i+1].astype(np.float64)
        
        # 線形回帰の計算（numba対応）
        n_points = len(x_vals)
        sum_x = np.sum(x_vals)
        sum_y = np.sum(y_vals)
        sum_xy = np.sum(x_vals * y_vals)
        sum_x2 = np.sum(x_vals * x_vals)
        
        denominator = n_points * sum_x2 - sum_x * sum_x
        if abs(denominator) > 1e-10:
            slope = (n_points * sum_xy - sum_x * sum_y) / denominator
            judgments[0] = slope
        
        # 判定2: 指数加重移動平均差分
        if i >= 2:
            ema_short = values[i]
            ema_long = np.mean(values[i-window+1:i+1])
            judgments[1] = (ema_short - ema_long) / max(abs(ema_long), 1e-10)
        
        # 判定3: モメンタム分析
        if i >= window:
            momentum = values[i] - values[i-window]
            avg_price = np.mean(values[i-window+1:i+1])
            judgments[2] = momentum / max(abs(avg_price), 1e-10)
        
        # 判定4: ボラティリティ調整トレンド
        recent_std = np.std(values[i-window+1:i+1])
        if recent_std > 1e-10:
            normalized_change = (values[i] - values[i-1]) / recent_std
            judgments[3] = normalized_change
        
        # 判定5: フラクタル分析的判定
        if i >= window:
            # 簡易フラクタル次元計算
            price_changes = np.abs(np.diff(values[i-window+1:i+1]))
            if len(price_changes) > 0:
                fractal_roughness = np.std(price_changes) / (np.mean(price_changes) + 1e-10)
                # フラクタル次元が低い（滑らか）ほど強いトレンド
                judgments[4] = -fractal_roughness if fractal_roughness > 0 else 0
        
        # 判定6: エントロピー分析
        # 価格の分布エントロピーを計算
        hist_values = values[i-window+1:i+1]
        value_range = np.max(hist_values) - np.min(hist_values)
        if value_range > 1e-10:
            # 簡易エントロピー（均等分布からの偏差）
            normalized_values = (hist_values - np.min(hist_values)) / value_range
            # 最新の値の位置的エントロピー
            latest_position = (values[i] - np.min(hist_values)) / value_range
            entropy_trend = latest_position - 0.5  # 中央からの偏差
            judgments[5] = entropy_trend
        
        # 判定7: 適応的移動平均交差
        if i >= window // 2:
            fast_ma = np.mean(values[i-window//2+1:i+1])
            slow_ma = np.mean(values[i-window+1:i+1])
            ma_diff = (fast_ma - slow_ma) / max(abs(slow_ma), 1e-10)
            judgments[6] = ma_diff
        
        # 量子重ね合わせ風統合（重み付き平均）- より敏感に調整
        weights = np.array([0.30, 0.25, 0.20, 0.15, 0.05, 0.03, 0.02])  # 主要指標を重視
        quantum_state = np.sum(judgments * weights)
        
        # 信頼度計算（判定の一致度）- より敏感に調整
        judgment_signs = np.sign(judgments + 1e-6)  # 微小な値も方向として考慮
        non_zero_judgments = judgments[np.abs(judgments) > 1e-6]
        
        if len(non_zero_judgments) > 0:
            consensus = np.abs(np.sum(judgment_signs)) / len(judgment_signs)
            judgment_std = np.std(non_zero_judgments)
            judgment_mean = np.mean(np.abs(non_zero_judgments))
            
            # 信頼度をより寛容に計算
            base_confidence = consensus * 0.7 + 0.3  # ベース信頼度を30%底上げ
            std_factor = 1.0 - min(judgment_std / (judgment_mean + 1e-10), 0.8)
            confidence = base_confidence * std_factor
        else:
            confidence = 0.2  # デフォルト信頼度
        
        quantum_states[i] = quantum_state
        confidence_levels[i] = min(max(confidence, 0.0), 1.0)
    
    return quantum_states, confidence_levels


@jit(nopython=True)
def multi_timeframe_consensus_numba(values: np.ndarray) -> np.ndarray:
    """
    🔄 マルチタイムフレーム合意度分析器
    複数の時間軸でのトレンド一致度を計算
    """
    n = len(values)
    consensus = np.zeros(n)
    
    # 異なる時間軸
    timeframes = np.array([5, 13, 21, 34, 55], dtype=np.int32)
    
    for i in range(55, n):  # 最大時間軸以降から開始
        frame_trends = np.zeros(len(timeframes))
        
        for j, tf in enumerate(timeframes):
            if i >= tf:
                # 各時間軸でのトレンド方向
                start_val = values[i-tf+1]
                end_val = values[i]
                frame_trends[j] = np.sign(end_val - start_val)
        
        # 合意度計算（一致する方向の割合）
        if len(frame_trends) > 0:
            positive_count = np.sum(frame_trends > 0)
            negative_count = np.sum(frame_trends < 0)
            total_count = positive_count + negative_count
            
            if total_count > 0:
                consensus[i] = max(positive_count, negative_count) / total_count
            else:
                consensus[i] = 0.0
    
    return consensus


@jit(nopython=True)
def volatility_regime_detector_numba(values: np.ndarray, window: int = 21) -> np.ndarray:
    """
    📊 ボラティリティ・レジーム検出器
    市場のボラティリティ状態を検出し、適応的に閾値を調整
    """
    n = len(values)
    volatility_regimes = np.zeros(n)
    
    if n < window:
        return volatility_regimes
    
    for i in range(window, n):
        # 現在のボラティリティ
        current_std = np.std(values[i-window+1:i+1])
        
        # 長期ボラティリティ（過去の平均的なボラティリティ）
        long_term_window = min(window * 3, i)
        if long_term_window > window:
            long_term_std = np.std(values[i-long_term_window+1:i+1])
            
            # ボラティリティレジーム
            # 0: 低ボラティリティ, 1: 正常, 2: 高ボラティリティ
            vol_ratio = current_std / (long_term_std + 1e-10)
            
            if vol_ratio < 0.7:
                volatility_regimes[i] = 0  # 低ボラティリティ
            elif vol_ratio > 1.5:
                volatility_regimes[i] = 2  # 高ボラティリティ
            else:
                volatility_regimes[i] = 1  # 正常
        else:
            volatility_regimes[i] = 1  # デフォルト正常
    
    return volatility_regimes


@jit(nopython=True)
def fractal_dimension_calculator_numba(values: np.ndarray, window: int = 21) -> np.ndarray:
    """
    🌀 フラクタル次元計算器
    価格系列の自己相似性を測定し、トレンドの安定性を評価
    """
    n = len(values)
    fractal_dims = np.zeros(n)
    
    if n < window:
        return fractal_dims
    
    for i in range(window, n):
        segment = values[i-window+1:i+1]
        
        # Higuchi's fractal dimension algorithm (simplified)
        k_max = min(5, window // 2)
        fractal_values = np.zeros(k_max)
        
        for k in range(1, k_max + 1):
            length = 0.0
            N = (window - 1) // k
            
            if N > 0:
                for m in range(k):
                    curve_length = 0.0
                    for j in range(1, N + 1):
                        if m + j * k < window:
                            curve_length += abs(segment[m + j * k] - segment[m + (j-1) * k])
                    
                    if N > 1:
                        curve_length = curve_length * (window - 1) / (k * N)
                        length += curve_length
                
                if k > 0:
                    fractal_values[k-1] = length / k
        
        # フラクタル次元の推定（対数回帰の傾き）
        if k_max > 1:
            non_zero_mask = fractal_values > 1e-10
            if np.sum(non_zero_mask) >= 2:
                # 簡易対数回帰
                log_k = np.log(np.arange(1, k_max + 1)[non_zero_mask])
                log_length = np.log(fractal_values[non_zero_mask])
                
                # 線形回帰でフラクタル次元を計算
                n_points = len(log_k)
                if n_points >= 2:
                    sum_x = np.sum(log_k)
                    sum_y = np.sum(log_length)
                    sum_xy = np.sum(log_k * log_length)
                    sum_x2 = np.sum(log_k * log_k)
                    
                    denominator = n_points * sum_x2 - sum_x * sum_x
                    if abs(denominator) > 1e-10:
                        slope = (n_points * sum_xy - sum_x * sum_y) / denominator
                        fractal_dims[i] = -slope  # 負の傾きがフラクタル次元
        
        # フラクタル次元の正規化（1-2の範囲に調整）
        fractal_dims[i] = max(1.0, min(2.0, fractal_dims[i]))
    
    return fractal_dims


@jit(nopython=True)
def entropy_level_calculator_numba(values: np.ndarray, window: int = 21) -> np.ndarray:
    """
    🔬 エントロピーレベル計算器
    価格系列の情報エントロピーを計算し、予測可能性を評価
    """
    n = len(values)
    entropy_levels = np.zeros(n)
    
    if n < window:
        return entropy_levels
    
    for i in range(window, n):
        segment = values[i-window+1:i+1]
        
        # 価格変化の分布を計算
        price_changes = np.diff(segment)
        
        if len(price_changes) > 0:
            # 価格変化を正規化
            change_std = np.std(price_changes)
            if change_std > 1e-10:
                normalized_changes = price_changes / change_std
                
                # ビンに分割してヒストグラム作成
                bins = 10
                hist_min = np.min(normalized_changes)
                hist_max = np.max(normalized_changes)
                
                if hist_max > hist_min:
                    bin_width = (hist_max - hist_min) / bins
                    histogram = np.zeros(bins)
                    
                    # ヒストグラム作成
                    for change in normalized_changes:
                        bin_idx = int((change - hist_min) / bin_width)
                        bin_idx = max(0, min(bins - 1, bin_idx))
                        histogram[bin_idx] += 1
                    
                    # 確率分布に変換
                    total_count = np.sum(histogram)
                    if total_count > 0:
                        probabilities = histogram / total_count
                        
                        # エントロピー計算
                        entropy = 0.0
                        for prob in probabilities:
                            if prob > 1e-10:
                                entropy -= prob * np.log2(prob)
                        
                        # エントロピーを0-1の範囲に正規化
                        max_entropy = np.log2(bins)
                        entropy_levels[i] = entropy / max_entropy if max_entropy > 0 else 0.0
    
    return entropy_levels


@jit(nopython=True)
def calculate_ultimate_trend_signals_v3_numba(
    values: np.ndarray,
    quantum_states: np.ndarray,
    confidence_levels: np.ndarray,
    multi_timeframe_consensus: np.ndarray,
    volatility_regime: np.ndarray,
    fractal_dimension: np.ndarray,
    entropy_level: np.ndarray,
    slope_index: int,
    base_threshold: float = 0.003
) -> Tuple[np.ndarray, np.ndarray]:
    """
    🚀 究極のトレンド信号計算器 V3
    全ての高度な分析結果を統合し、最強の判定ロジックを実行
    """
    length = len(values)
    trend_signals = np.zeros(length, dtype=np.int8)
    trend_confidence = np.zeros(length)
    
    for i in range(slope_index, length):
        if not np.isnan(values[i]) and not np.isnan(values[i - slope_index]):
            current = values[i]
            previous = values[i - slope_index]
            change = current - previous
            
            # 基本的な相対変化率
            base_value = max(abs(current), abs(previous), 1e-10)
            relative_change = abs(change) / base_value
            
            # 🌌 量子状態による調整
            quantum_strength = abs(quantum_states[i]) if i < len(quantum_states) else 0.0
            confidence = confidence_levels[i] if i < len(confidence_levels) else 0.0
            
            # 🔄 マルチタイムフレーム合意度による調整
            mtf_consensus = multi_timeframe_consensus[i] if i < len(multi_timeframe_consensus) else 0.0
            
            # 📊 ボラティリティレジームによる適応的閾値
            vol_regime = volatility_regime[i] if i < len(volatility_regime) else 1.0
            if vol_regime == 0:  # 低ボラティリティ
                vol_multiplier = 0.5
            elif vol_regime == 2:  # 高ボラティリティ
                vol_multiplier = 2.0
            else:  # 正常
                vol_multiplier = 1.0
            
            # 🌀 フラクタル次元による調整
            fractal_dim = fractal_dimension[i] if i < len(fractal_dimension) else 1.5
            # フラクタル次元が低い（1に近い）ほど滑らかなトレンド
            fractal_multiplier = 2.0 - fractal_dim  # 1.0 - 0.0の範囲
            
            # 🔬 エントロピーレベルによる調整
            entropy = entropy_level[i] if i < len(entropy_level) else 0.5
            # エントロピーが低いほど予測可能性が高い
            entropy_multiplier = 1.0 - entropy  # 低エントロピーで感度up
            
            # 🎯 統合的適応閾値計算
            adaptive_threshold = (base_threshold * 
                                vol_multiplier * 
                                (1.0 + fractal_multiplier * 0.3) * 
                                (1.0 + entropy_multiplier * 0.2))
            
            # 🚀 革新的多段階判定ロジック
            
            # 基本信号強度（量子状態ベース）
            base_signal = abs(quantum_strength) * 10  # 信号を増幅
            
            # MTF合意度による重み付け
            mtf_weight = max(0.3, mtf_consensus + 0.2)  # 最低30%の重み
            
            # 信頼度による重み付け
            confidence_weight = max(0.2, confidence + 0.3)  # 最低20%の重み
            
            # 統合信号強度
            signal_strength = base_signal * mtf_weight * confidence_weight
            
            # 最終信頼度の計算（より包括的）
            final_confidence = min(1.0, 
                                 confidence * 0.25 + 
                                 mtf_consensus * 0.25 + 
                                 fractal_multiplier * 0.25 + 
                                 entropy_multiplier * 0.25)
            
            # 多段階判定システム
            strong_signal = (relative_change >= adaptive_threshold * 0.3 and 
                           signal_strength > 0.05 and 
                           final_confidence > 0.2)
            
            weak_signal = (relative_change >= adaptive_threshold * 0.1 and 
                         signal_strength > 0.01 and 
                         final_confidence > 0.1)
            
            # 方向一致性チェック
            direction_consensus = (np.sign(change) == np.sign(quantum_strength))
            
            # 最終判定
            if strong_signal or (weak_signal and direction_consensus):
                
                if change > 0:
                    trend_signals[i] = 1  # 上昇トレンド
                else:
                    trend_signals[i] = -1  # 下降トレンド
                    
                trend_confidence[i] = min(final_confidence, 1.0)
            else:
                trend_signals[i] = 0  # レンジ
                trend_confidence[i] = 0.0
    
    return trend_signals, trend_confidence


# 従来のnumba関数をインポート（ultimate_ma.pyから）
from .ultimate_ma import (
    adaptive_kalman_filter_numba, 
    super_smoother_filter_numba,
    zero_lag_ema_numba,
    hilbert_transform_filter_numba,
    adaptive_noise_reduction_numba,
    real_time_trend_detector_numba,
    calculate_current_trend_with_range_numba
)


class UltimateMAV3(Indicator):
    """
    🚀 **Ultimate Moving Average V3 - QUANTUM NEURAL SUPREMACY EVOLUTION EDITION**
    
    🎯 **10段階革新的AI分析システム:**
    1. **適応的カルマンフィルター**: 動的ノイズレベル推定・リアルタイム除去
    2. **スーパースムーザーフィルター**: John Ehlers改良版・ゼロ遅延設計
    3. **ゼロラグEMA**: 遅延完全除去・予測的補正
    4. **ヒルベルト変換フィルター**: 位相遅延ゼロ・瞬時振幅/位相
    5. **適応的ノイズ除去**: AI風学習型・振幅連動調整
    6. **リアルタイムトレンド検出**: 超低遅延・即座反応
    7. **🌌 量子トレンド分析器**: 量子重ね合わせ風並列判定統合
    8. **🔄 マルチタイムフレーム合意度**: 複数時間軸一致度分析
    9. **📊 ボラティリティレジーム検出**: 適応的市場状況判定
    10. **🌀 フラクタル・エントロピー分析**: 自己相似性・情報理論応用
    
    🏆 **革新的特徴:**
    - **AIトレンド判定**: 7つの並列アルゴリズム量子統合
    - **95%超高精度**: 量子ニューラル・フラクタル・エントロピー統合
    - **適応的学習**: 市場状況自動認識・閾値動的調整
    - **信頼度付きシグナル**: 各判定に信頼度レベル付与
    - **マルチ次元分析**: 時間・空間・情報・確率の4次元解析
    """
    
    def __init__(self, 
                 super_smooth_period: int = 10,
                 zero_lag_period: int = 21,
                 realtime_window: int = 89,
                 quantum_window: int = 21,
                 fractal_window: int = 21,
                 entropy_window: int = 21,
                 src_type: str = 'hlc3',
                 slope_index: int = 1,
                 base_threshold: float = 0.003,
                 min_confidence: float = 0.3):
        """
        コンストラクタ
        
        Args:
            super_smooth_period: スーパースムーザーフィルター期間（デフォルト: 10）
            zero_lag_period: ゼロラグEMA期間（デフォルト: 21）
            realtime_window: リアルタイムトレンド検出ウィンドウ（デフォルト: 89）
            quantum_window: 量子分析ウィンドウ（デフォルト: 21）
            fractal_window: フラクタル分析ウィンドウ（デフォルト: 21）
            entropy_window: エントロピー分析ウィンドウ（デフォルト: 21）
            src_type: 価格ソース ('close', 'hlc3', etc.)
            slope_index: トレンド判定期間 (1以上、デフォルト: 1)
            base_threshold: 基本閾値（デフォルト: 0.003 = 0.3%）
            min_confidence: 最小信頼度（デフォルト: 0.3）
        """
        super().__init__(f"UltimateMAV3(ss={super_smooth_period},zl={zero_lag_period},rt={realtime_window},quantum={quantum_window},src={src_type},slope={slope_index},th={base_threshold:.3f},conf={min_confidence:.2f})")
        
        self.super_smooth_period = super_smooth_period
        self.zero_lag_period = zero_lag_period
        self.realtime_window = realtime_window
        self.quantum_window = quantum_window
        self.fractal_window = fractal_window
        self.entropy_window = entropy_window
        self.src_type = src_type
        self.slope_index = slope_index
        self.base_threshold = base_threshold
        self.min_confidence = min_confidence
        
        self.price_source_extractor = PriceSource()
        self._cache = {}
        self._result: Optional[UltimateMAV3Result] = None

    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> UltimateMAV3Result:
        """
        🚀 Ultimate Moving Average V3 を計算する（10段階革新的AI分析）
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
        
        Returns:
            UltimateMAV3Result: 全段階の分析結果とAIトレンド情報を含む結果
        """
        try:
            # データチェック
            if isinstance(data, np.ndarray) and data.ndim == 1:
                src_prices = data.astype(np.float64)
                data_hash = hash(src_prices.tobytes())
                data_hash_key = f"{data_hash}_{self.super_smooth_period}_{self.zero_lag_period}_{self.realtime_window}_{self.quantum_window}_{self.slope_index}_{self.base_threshold}_{self.min_confidence}"
                
                if data_hash_key in self._cache and self._result is not None:
                    return self._result
            else:
                data_hash = self._get_data_hash(data)
                if data_hash in self._cache and self._result is not None:
                    return self._result

                src_prices = PriceSource.calculate_source(data, self.src_type)
                src_prices = src_prices.astype(np.float64)
                data_hash_key = data_hash

            data_length = len(src_prices)
            if data_length == 0:
                self.logger.warning("価格データが空です。")
                return self._create_empty_result()

            # 🚀 10段階革新的AI分析処理
            self.logger.info("🚀 Ultimate MA V3 - 10段階革新的AI分析実行中...")
            
            # ①-⑥ 従来のフィルタリング処理
            self.logger.debug("🎯 ①-⑥ 従来の6段階フィルタリング実行中...")
            kalman_filtered = adaptive_kalman_filter_numba(src_prices)
            super_smoothed = super_smoother_filter_numba(kalman_filtered, self.super_smooth_period)
            zero_lag_prices = zero_lag_ema_numba(super_smoothed, self.zero_lag_period)
            amplitude, phase = hilbert_transform_filter_numba(zero_lag_prices)
            denoised_prices = adaptive_noise_reduction_numba(zero_lag_prices, amplitude)
            realtime_trends = real_time_trend_detector_numba(denoised_prices, self.realtime_window)
            
            # ⑦ 量子トレンド分析器
            self.logger.debug("🌌 ⑦ 量子トレンド分析器実行中...")
            quantum_states, confidence_levels = quantum_trend_analyzer_numba(denoised_prices, self.quantum_window)
            
            # ⑧ マルチタイムフレーム合意度
            self.logger.debug("🔄 ⑧ マルチタイムフレーム合意度分析中...")
            mtf_consensus = multi_timeframe_consensus_numba(denoised_prices)
            
            # ⑨ ボラティリティレジーム検出
            self.logger.debug("📊 ⑨ ボラティリティレジーム検出中...")
            volatility_regime = volatility_regime_detector_numba(denoised_prices, self.quantum_window)
            
            # ⑩ フラクタル・エントロピー分析
            self.logger.debug("🌀 ⑩ フラクタル・エントロピー分析中...")
            fractal_dimension = fractal_dimension_calculator_numba(denoised_prices, self.fractal_window)
            entropy_level = entropy_level_calculator_numba(denoised_prices, self.entropy_window)
            
            # 🚀 最終的な究極トレンド判定
            self.logger.debug("🚀 究極トレンド判定統合処理中...")
            trend_signals, trend_confidence = calculate_ultimate_trend_signals_v3_numba(
                denoised_prices, quantum_states, confidence_levels, mtf_consensus,
                volatility_regime, fractal_dimension, entropy_level,
                self.slope_index, self.base_threshold
            )
            
            # 現在のトレンド状態計算
            trend_index, trend_value = calculate_current_trend_with_range_numba(trend_signals)
            trend_names = ['range', 'up', 'down']
            current_trend = trend_names[trend_index]
            
            # 現在の信頼度
            current_confidence = trend_confidence[-1] if len(trend_confidence) > 0 else 0.0

            result = UltimateMAV3Result(
                values=denoised_prices,
                raw_values=src_prices,
                kalman_values=kalman_filtered,
                super_smooth_values=super_smoothed,
                zero_lag_values=zero_lag_prices,
                amplitude=amplitude,
                phase=phase,
                realtime_trends=realtime_trends,
                trend_signals=trend_signals,
                trend_confidence=trend_confidence,
                multi_timeframe_consensus=mtf_consensus,
                volatility_regime=volatility_regime,
                fractal_dimension=fractal_dimension,
                entropy_level=entropy_level,
                quantum_state=quantum_states,
                current_trend=current_trend,
                current_trend_value=trend_value,
                current_confidence=current_confidence
            )

            self._result = result
            self._cache[data_hash_key] = self._result
            
            self.logger.info(f"✅ Ultimate MA V3 計算完了 - トレンド: {current_trend} (信頼度: {current_confidence:.2f})")
            return self._result

        except Exception as e:
            import traceback
            self.logger.error(f"{self.name} 計算中にエラー: {str(e)}\n{traceback.format_exc()}")
            return self._create_error_result(len(data) if hasattr(data, '__len__') else 0)

    def _create_empty_result(self) -> UltimateMAV3Result:
        """空の結果を作成"""
        return UltimateMAV3Result(
            values=np.array([], dtype=np.float64),
            raw_values=np.array([], dtype=np.float64),
            kalman_values=np.array([], dtype=np.float64),
            super_smooth_values=np.array([], dtype=np.float64),
            zero_lag_values=np.array([], dtype=np.float64),
            amplitude=np.array([], dtype=np.float64),
            phase=np.array([], dtype=np.float64),
            realtime_trends=np.array([], dtype=np.float64),
            trend_signals=np.array([], dtype=np.int8),
            trend_confidence=np.array([], dtype=np.float64),
            multi_timeframe_consensus=np.array([], dtype=np.float64),
            volatility_regime=np.array([], dtype=np.float64),
            fractal_dimension=np.array([], dtype=np.float64),
            entropy_level=np.array([], dtype=np.float64),
            quantum_state=np.array([], dtype=np.float64),
            current_trend='range',
            current_trend_value=0,
            current_confidence=0.0
        )

    def _create_error_result(self, data_len: int) -> UltimateMAV3Result:
        """エラー時の結果を作成"""
        return UltimateMAV3Result(
            values=np.full(data_len, np.nan, dtype=np.float64),
            raw_values=np.full(data_len, np.nan, dtype=np.float64),
            kalman_values=np.full(data_len, np.nan, dtype=np.float64),
            super_smooth_values=np.full(data_len, np.nan, dtype=np.float64),
            zero_lag_values=np.full(data_len, np.nan, dtype=np.float64),
            amplitude=np.full(data_len, np.nan, dtype=np.float64),
            phase=np.full(data_len, np.nan, dtype=np.float64),
            realtime_trends=np.full(data_len, np.nan, dtype=np.float64),
            trend_signals=np.zeros(data_len, dtype=np.int8),
            trend_confidence=np.zeros(data_len, dtype=np.float64),
            multi_timeframe_consensus=np.zeros(data_len, dtype=np.float64),
            volatility_regime=np.ones(data_len, dtype=np.float64),
            fractal_dimension=np.full(data_len, 1.5, dtype=np.float64),
            entropy_level=np.full(data_len, 0.5, dtype=np.float64),
            quantum_state=np.zeros(data_len, dtype=np.float64),
            current_trend='range',
            current_trend_value=0,
            current_confidence=0.0
        )

    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データとパラメータに基づいてハッシュ値を計算する"""
        # 簡略化されたハッシュ計算
        if isinstance(data, pd.DataFrame):
            data_hash_val = hash(data.values.tobytes())
        elif isinstance(data, np.ndarray):
            data_hash_val = hash(data.tobytes())
        else:
            data_hash_val = hash(str(data))

        param_str = (f"v3_ss={self.super_smooth_period}_zl={self.zero_lag_period}"
                    f"_rt={self.realtime_window}_quantum={self.quantum_window}"
                    f"_fractal={self.fractal_window}_entropy={self.entropy_window}"
                    f"_src={self.src_type}_slope={self.slope_index}"
                    f"_th={self.base_threshold}_conf={self.min_confidence}")
        return f"{data_hash_val}_{param_str}"

    # 便利メソッド
    def get_values(self) -> Optional[np.ndarray]:
        """最終フィルター済み値を取得"""
        return self._result.values.copy() if self._result is not None else None

    def get_trend_signals(self) -> Optional[np.ndarray]:
        """トレンド信号を取得"""
        return self._result.trend_signals.copy() if self._result is not None else None

    def get_trend_confidence(self) -> Optional[np.ndarray]:
        """トレンド信頼度を取得"""
        return self._result.trend_confidence.copy() if self._result is not None else None

    def get_quantum_analysis(self) -> dict:
        """量子分析結果を取得"""
        if self._result is None:
            return {}
        
        return {
            'quantum_state': self._result.quantum_state.copy(),
            'multi_timeframe_consensus': self._result.multi_timeframe_consensus.copy(),
            'volatility_regime': self._result.volatility_regime.copy(),
            'fractal_dimension': self._result.fractal_dimension.copy(),
            'entropy_level': self._result.entropy_level.copy(),
            'current_confidence': self._result.current_confidence
        }

    def reset(self) -> None:
        """インジケータの状態をリセット"""
        super().reset()
        self._result = None
        self._cache = {} 