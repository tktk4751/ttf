#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Optional, Dict, Tuple
import numpy as np
import pandas as pd
from numba import njit, prange
import traceback

from .indicator import Indicator
from .price_source import PriceSource
from .str import STR
from .ultimate_smoother import UltimateSmoother


@dataclass
class UltraRefinedVolatilityStateResult:
    """超洗練されたボラティリティ状態判別結果"""
    state: np.ndarray                      # ボラティリティ状態 (1: 高, 0: 低)
    probability: np.ndarray                # 状態の確信度 (0.0-1.0)
    confidence: np.ndarray                 # 判定信頼度 (0.0-1.0)
    raw_score: np.ndarray                 # 生のボラティリティスコア
    str_values: np.ndarray                # STR値
    hilbert_envelope: np.ndarray          # ヒルベルト包絡線
    hilbert_phase: np.ndarray             # ヒルベルト位相
    instantaneous_frequency: np.ndarray   # 瞬間周波数
    wavelet_energy: np.ndarray            # ウェーブレットエネルギー
    adaptive_threshold: np.ndarray        # 適応的閾値
    spectral_entropy: np.ndarray          # スペクトラルエントロピー
    fractal_dimension: np.ndarray         # フラクタル次元
    adaptive_gain: np.ndarray             # 適応ゲイン


@njit(fastmath=True, cache=True)
def hilbert_transform_approximate(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ヒルベルト変換の近似計算（高速版）
    解析信号、包絡線、瞬間位相を計算
    """
    length = len(signal)
    envelope = np.zeros(length)
    phase = np.zeros(length)
    instantaneous_freq = np.zeros(length)
    
    # 簡易ヒルベルト変換（4段階移動平均フィルタによる近似）
    for i in range(4, length):
        # 実部（元信号）
        real_part = signal[i]
        
        # 虚部（90度位相シフト近似）
        if i >= 4:
            # 4点差分による90度位相シフト近似
            imag_part = (signal[i] - signal[i-2] + signal[i-1] - signal[i-3]) * 0.25
        else:
            imag_part = 0.0
        
        # 包絡線（振幅）
        envelope[i] = np.sqrt(real_part * real_part + imag_part * imag_part)
        
        # 位相
        if real_part != 0:
            phase[i] = np.arctan2(imag_part, real_part)
        else:
            phase[i] = phase[i-1] if i > 0 else 0
        
        # 瞬間周波数（位相の微分）
        if i > 0:
            phase_diff = phase[i] - phase[i-1]
            # 位相ラッピング調整
            if phase_diff > np.pi:
                phase_diff -= 2 * np.pi
            elif phase_diff < -np.pi:
                phase_diff += 2 * np.pi
            instantaneous_freq[i] = abs(phase_diff)
        
    return envelope, phase, instantaneous_freq


@njit(fastmath=True, cache=True)
def discrete_wavelet_transform_haar(signal: np.ndarray) -> np.ndarray:
    """
    ハールウェーブレット変換による多解像度解析
    エネルギー計算
    """
    length = len(signal)
    energy = np.zeros(length)
    
    # ハールウェーブレット係数計算
    for i in range(2, length):
        # スケール1のハールウェーブレット
        if i >= 2:
            detail_coeff = (signal[i] - signal[i-1]) / np.sqrt(2)
            approx_coeff = (signal[i] + signal[i-1]) / np.sqrt(2)
            
            # エネルギー（係数の二乗和）
            energy[i] = detail_coeff * detail_coeff + approx_coeff * approx_coeff
            
            # スケール2のウェーブレット（より大きな時間窓）
            if i >= 4:
                scale2_detail = (signal[i] + signal[i-1] - signal[i-2] - signal[i-3]) / 2.0
                energy[i] += scale2_detail * scale2_detail * 0.5
    
    return energy


@njit(fastmath=True, cache=True)
def calculate_spectral_entropy(signal: np.ndarray, window_size: int = 16) -> np.ndarray:
    """
    スペクトラルエントロピー計算（簡易版）
    信号の複雑性・不規則性を測定
    """
    length = len(signal)
    entropy = np.zeros(length)
    
    for i in range(window_size, length):
        window = signal[i-window_size:i]
        
        # 簡易パワースペクトラム計算（離散フーリエ変換の近似）
        n_bins = window_size // 2
        power_spectrum = np.zeros(n_bins)
        
        for k in range(n_bins):
            real_sum = 0.0
            imag_sum = 0.0
            for n in range(window_size):
                angle = 2.0 * np.pi * k * n / window_size
                real_sum += window[n] * np.cos(angle)
                imag_sum += window[n] * np.sin(angle)
            
            power_spectrum[k] = real_sum * real_sum + imag_sum * imag_sum
        
        # 正規化
        total_power = np.sum(power_spectrum)
        if total_power > 1e-10:
            # エントロピー計算
            entropy_sum = 0.0
            for k in range(n_bins):
                if power_spectrum[k] > 1e-10:
                    prob = power_spectrum[k] / total_power
                    entropy_sum -= prob * np.log2(prob + 1e-10)
            
            entropy[i] = entropy_sum / np.log2(n_bins)  # 正規化エントロピー
        else:
            entropy[i] = 0.0
    
    return entropy


@njit(fastmath=True, cache=True)
def calculate_fractal_dimension_higuchi(signal: np.ndarray, max_k: int = 8) -> np.ndarray:
    """
    東グチ法によるフラクタル次元計算
    時系列の複雑性を測定
    """
    length = len(signal)
    fractal_dim = np.zeros(length)
    window_size = 32
    
    for i in range(window_size, length):
        window = signal[i-window_size:i]
        
        # 各kに対する長さ計算
        lengths = np.zeros(max_k)
        
        for k in range(1, max_k + 1):
            length_sum = 0.0
            n_segments = (window_size - 1) // k
            
            for m in range(k):
                curve_length = 0.0
                points = 0
                
                for j in range(1, n_segments + 1):
                    if m + j * k < window_size:
                        curve_length += abs(window[m + j * k] - window[m + (j-1) * k])
                        points += 1
                
                if points > 0:
                    # 正規化
                    curve_length = curve_length * (window_size - 1) / (points * k)
                    length_sum += curve_length
            
            lengths[k-1] = length_sum / k if k > 0 else 0
        
        # フラクタル次元計算（対数回帰の近似）
        if lengths[0] > 1e-10:
            # 簡易対数線形回帰
            log_sum_x = 0.0
            log_sum_y = 0.0
            log_sum_xy = 0.0
            log_sum_xx = 0.0
            count = 0
            
            for k in range(1, max_k + 1):
                if lengths[k-1] > 1e-10:
                    log_k = np.log(k)
                    log_length = np.log(lengths[k-1])
                    
                    log_sum_x += log_k
                    log_sum_y += log_length
                    log_sum_xy += log_k * log_length
                    log_sum_xx += log_k * log_k
                    count += 1
            
            if count > 1:
                # 線形回帰の傾き
                denominator = count * log_sum_xx - log_sum_x * log_sum_x
                if abs(denominator) > 1e-10:
                    slope = (count * log_sum_xy - log_sum_x * log_sum_y) / denominator
                    fractal_dim[i] = -slope  # フラクタル次元
                else:
                    fractal_dim[i] = 1.0
            else:
                fractal_dim[i] = 1.0
        else:
            fractal_dim[i] = 1.0
        
        # 範囲制限（1.0-2.0）
        if fractal_dim[i] < 1.0:
            fractal_dim[i] = 1.0
        elif fractal_dim[i] > 2.0:
            fractal_dim[i] = 2.0
    
    return fractal_dim


@njit(fastmath=True, cache=True)
def adaptive_kalman_filter(signal: np.ndarray, process_noise: float = 1e-5) -> Tuple[np.ndarray, np.ndarray]:
    """
    適応カルマンフィルタ
    動的ノイズ推定付き
    """
    length = len(signal)
    filtered_signal = np.zeros(length)
    adaptive_gain = np.zeros(length)
    
    # 初期化
    if length > 0:
        state = signal[0]
        error_covariance = 1.0
        filtered_signal[0] = state
        adaptive_gain[0] = 0.5
    
    for i in range(1, length):
        # 予測ステップ
        predicted_state = state
        predicted_covariance = error_covariance + process_noise
        
        # 適応的観測ノイズ推定
        if i > 5:
            recent_residuals = np.zeros(5)
            for j in range(5):
                recent_residuals[j] = abs(signal[i-j] - filtered_signal[i-j])
            observation_noise = np.var(recent_residuals) + 1e-6
        else:
            observation_noise = 1e-3
        
        # カルマンゲイン
        kalman_gain = predicted_covariance / (predicted_covariance + observation_noise)
        
        # 更新ステップ
        innovation = signal[i] - predicted_state
        state = predicted_state + kalman_gain * innovation
        error_covariance = (1 - kalman_gain) * predicted_covariance
        
        filtered_signal[i] = state
        adaptive_gain[i] = kalman_gain
    
    return filtered_signal, adaptive_gain


@njit(fastmath=True, cache=True)
def calculate_adaptive_threshold(
    features: np.ndarray, 
    lookback: int = 50, 
    sensitivity: float = 2.0
) -> np.ndarray:
    """
    適応的閾値計算
    統計的特性とボラティリティ状況に基づく動的調整
    """
    length = len(features)
    threshold = np.zeros(length)
    
    for i in range(lookback, length):
        # 過去期間の統計
        window = features[i-lookback:i]
        
        # 外れ値除去（IQR法）
        sorted_window = np.sort(window)
        q1_idx = len(sorted_window) // 4
        q3_idx = 3 * len(sorted_window) // 4
        
        if q3_idx < len(sorted_window) and q1_idx >= 0:
            q1 = sorted_window[q1_idx]
            q3 = sorted_window[q3_idx]
            median = sorted_window[len(sorted_window) // 2]
            iqr = q3 - q1
            
            # 適応的閾値
            # 基本閾値 + IQRベースの動的調整
            base_threshold = median + sensitivity * iqr * 0.5
            
            # 最近の変動性による調整
            recent_volatility = 0.0
            for j in range(min(10, i)):
                if i - j - 1 >= 0:
                    recent_volatility += abs(features[i-j] - features[i-j-1])
            recent_volatility /= min(10, i)
            
            # 変動性が高い場合は閾値を下げる（感度アップ）
            volatility_adjustment = 1.0 - min(recent_volatility * 0.1, 0.3)
            threshold[i] = base_threshold * volatility_adjustment
            
            # 最小・最大閾値制限
            threshold[i] = max(min(threshold[i], 0.9), 0.1)
        else:
            threshold[i] = 0.5
    
    return threshold


@njit(fastmath=True, parallel=True, cache=True)
def ultra_refined_volatility_fusion(
    str_values: np.ndarray,
    hilbert_envelope: np.ndarray,
    instantaneous_frequency: np.ndarray,
    wavelet_energy: np.ndarray,
    spectral_entropy: np.ndarray,
    fractal_dimension: np.ndarray,
    adaptive_threshold: np.ndarray,
    adaptive_gain: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    超洗練されたボラティリティ融合アルゴリズム
    """
    length = len(str_values)
    state = np.zeros(length, dtype=np.int8)
    probability = np.zeros(length)
    confidence = np.zeros(length)
    raw_score = np.zeros(length)
    
    for i in prange(length):
        # 各特徴量の正規化
        str_score = min(str_values[i] * 100, 1.0) if not np.isnan(str_values[i]) else 0.5
        
        # ヒルベルト包絡線スコア（正規化）
        envelope_score = min(hilbert_envelope[i], 1.0) if not np.isnan(hilbert_envelope[i]) else 0.5
        
        # 瞬間周波数スコア（高周波数 = 高ボラティリティ）
        freq_score = min(instantaneous_frequency[i] * 10, 1.0) if not np.isnan(instantaneous_frequency[i]) else 0.5
        
        # ウェーブレットエネルギースコア
        wavelet_score = min(wavelet_energy[i] * 0.1, 1.0) if not np.isnan(wavelet_energy[i]) else 0.5
        
        # スペクトラルエントロピースコア（高エントロピー = 高ボラティリティ）
        entropy_score = spectral_entropy[i] if not np.isnan(spectral_entropy[i]) else 0.5
        
        # フラクタル次元スコア（高次元 = 高ボラティリティ）
        fractal_score = (fractal_dimension[i] - 1.0) if not np.isnan(fractal_dimension[i]) else 0.5
        
        # 適応ゲインスコア
        gain_score = adaptive_gain[i] if not np.isnan(adaptive_gain[i]) else 0.5
        
        # 高度な重み付き融合（動的重み調整）
        # 基本重み
        w_str = 0.25
        w_envelope = 0.20
        w_frequency = 0.15
        w_wavelet = 0.15
        w_entropy = 0.10
        w_fractal = 0.10
        w_gain = 0.05
        
        # 状況に応じた重み調整
        if freq_score > 0.7:  # 高周波数環境
            w_frequency += 0.05
            w_envelope += 0.05
            w_str -= 0.05
            w_wavelet -= 0.05
        
        if entropy_score > 0.7:  # 高複雑性環境
            w_entropy += 0.05
            w_fractal += 0.05
            w_str -= 0.05
            w_envelope -= 0.05
        
        # 最終スコア計算
        score = (w_str * str_score +
                 w_envelope * envelope_score +
                 w_frequency * freq_score +
                 w_wavelet * wavelet_score +
                 w_entropy * entropy_score +
                 w_fractal * fractal_score +
                 w_gain * gain_score)
        
        raw_score[i] = score
        
        # 信頼度計算（特徴量の一致度）
        feature_agreement = 0.0
        high_features = 0
        total_features = 6
        
        if str_score > 0.5: high_features += 1
        if envelope_score > 0.5: high_features += 1
        if freq_score > 0.5: high_features += 1
        if wavelet_score > 0.5: high_features += 1
        if entropy_score > 0.5: high_features += 1
        if fractal_score > 0.5: high_features += 1
        
        feature_agreement = high_features / total_features
        
        # 信頼度の調整
        if feature_agreement > 0.75:
            confidence[i] = 0.9 + 0.1 * gain_score
        elif feature_agreement > 0.5:
            confidence[i] = 0.7 + 0.2 * gain_score
        else:
            confidence[i] = 0.5 + 0.2 * gain_score
        
        # 確率計算（シグモイド + 信頼度調整）
        k = 8.0
        base_prob = 1.0 / (1.0 + np.exp(-k * (score - 0.5)))
        probability[i] = base_prob * confidence[i] + (1 - confidence[i]) * 0.5
        
        # 適応的状態判定
        current_threshold = adaptive_threshold[i] if i < len(adaptive_threshold) else 0.6
        
        if i > 0:
            prev_state = state[i-1]
            prev_confidence = confidence[i-1]
            
            # 信頼度ベースのヒステリシス調整
            confidence_factor = 0.8 + 0.4 * confidence[i]  # 0.8-1.2の範囲
            
            if prev_state == 0:  # 前回が低ボラティリティ
                effective_threshold = current_threshold * confidence_factor
                state[i] = 1 if score > effective_threshold else 0
            else:  # 前回が高ボラティリティ
                effective_threshold = current_threshold * (2.0 - confidence_factor)
                state[i] = 0 if score < effective_threshold else 1
        else:
            # 初回判定
            state[i] = 1 if score > current_threshold else 0
    
    return state, probability, confidence, raw_score


class UltraRefinedVolatilityState(Indicator):
    """
    超洗練されたボラティリティ状態判別インジケーター
    
    先進的デジタル信号処理技術を駆使:
    1. STRベース（超低遅延）
    2. ヒルベルト変換（包絡線・位相解析）
    3. ウェーブレット解析（多解像度解析）
    4. スペクトラルエントロピー（複雑性測定）
    5. フラクタル次元（自己相似性）
    6. 適応カルマンフィルタ（ノイズ除去）
    7. 適応的閾値（動的調整）
    8. 信頼度ベース判定（高精度）
    
    特徴:
    - 超高精度: 多次元特徴量の融合
    - 超低遅延: STRベースの高速応答
    - 動的適応: 市場状況に応じた自動調整
    - 高信頼性: 信頼度による判定品質保証
    """
    
    def __init__(
        self,
        str_period: int = 14,                 # STR計算期間
        lookback_period: int = 100,           # 適応的閾値計算期間
        hilbert_smooth: int = 4,              # ヒルベルト変換平滑化
        wavelet_scales: int = 2,              # ウェーブレットスケール数
        entropy_window: int = 16,             # エントロピー計算窓
        fractal_k: int = 8,                   # フラクタル次元計算K
        sensitivity: float = 2.0,             # 適応閾値感度
        confidence_threshold: float = 0.7,    # 最小信頼度閾値
        src_type: str = 'hlc3',               # 価格ソース
        smoothing: bool = True                # 最終スムージング
    ):
        """
        コンストラクタ
        
        Args:
            str_period: STR計算期間
            lookback_period: 適応的閾値計算の振り返り期間
            hilbert_smooth: ヒルベルト変換平滑化期間
            wavelet_scales: ウェーブレットスケール数
            entropy_window: スペクトラルエントロピー計算窓サイズ
            fractal_k: フラクタル次元計算の最大K値
            sensitivity: 適応閾値の感度パラメータ
            confidence_threshold: 判定に必要な最小信頼度
            src_type: 価格ソースタイプ
            smoothing: 最終結果のスムージング
        """
        super().__init__(f"UltraRefinedVolatilityState(str={str_period}, sensitivity={sensitivity}, confidence={confidence_threshold})")
        
        self.str_period = str_period
        self.lookback_period = lookback_period
        self.hilbert_smooth = hilbert_smooth
        self.wavelet_scales = wavelet_scales
        self.entropy_window = entropy_window
        self.fractal_k = fractal_k
        self.sensitivity = sensitivity
        self.confidence_threshold = confidence_threshold
        self.src_type = src_type.lower()
        self.smoothing = smoothing
        
        # STRインジケーター
        self.str_indicator = STR(
            period=str_period,
            src_type=src_type,
            period_mode='dynamic'  # 動的モード
        )
        
        # スムージング用
        if self.smoothing:
            self.smoother = UltimateSmoother(period=2, src_type='close')
        
        # 結果キャッシュ
        self._result_cache = {}
        self._max_cache_size = 3
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> UltraRefinedVolatilityStateResult:
        """
        超洗練されたボラティリティ状態を判定
        
        Args:
            data: 価格データ (OHLC必須)
            
        Returns:
            UltraRefinedVolatilityStateResult: 判定結果
        """
        try:
            # データ準備
            length = len(data)
            min_required = max(self.str_period, self.lookback_period // 2, 50)
            
            if length < min_required:
                return self._create_empty_result(length)
            
            # 1. STR計算（基本特徴量）
            str_result = self.str_indicator.calculate(data)
            str_values = str_result.values
            
            # 2. 適応カルマンフィルタによるSTRの平滑化
            filtered_str, adaptive_gain = adaptive_kalman_filter(str_values)
            
            # 3. ヒルベルト変換（包絡線・位相解析）
            hilbert_envelope, hilbert_phase, instantaneous_frequency = hilbert_transform_approximate(filtered_str)
            
            # 4. ウェーブレット解析（エネルギー計算）
            wavelet_energy = discrete_wavelet_transform_haar(filtered_str)
            
            # 5. スペクトラルエントロピー（複雑性測定）
            spectral_entropy = calculate_spectral_entropy(filtered_str, self.entropy_window)
            
            # 6. フラクタル次元（自己相似性）
            fractal_dimension = calculate_fractal_dimension_higuchi(filtered_str, self.fractal_k)
            
            # 7. 適応的閾値計算
            composite_feature = (filtered_str + hilbert_envelope + wavelet_energy * 0.1) / 3
            adaptive_threshold = calculate_adaptive_threshold(composite_feature, self.lookback_period, self.sensitivity)
            
            # 8. 超洗練された融合
            state, probability, confidence, raw_score = ultra_refined_volatility_fusion(
                filtered_str, hilbert_envelope, instantaneous_frequency,
                wavelet_energy, spectral_entropy, fractal_dimension,
                adaptive_threshold, adaptive_gain
            )
            
            # 9. 信頼度フィルタリング
            # 低信頼度の判定を中立化
            for i in range(length):
                if confidence[i] < self.confidence_threshold:
                    probability[i] = 0.5  # 中立確率
                    if i > 0:
                        state[i] = state[i-1]  # 前回状態を維持
            
            # 10. オプショナルスムージング
            if self.smoothing:
                # 状態のスムージング
                state_df = pd.DataFrame({'close': state.astype(np.float64)})
                smoothed_state_result = self.smoother.calculate(state_df)
                smoothed_state = (smoothed_state_result.values > 0.5).astype(np.int8)
                
                # 確率のスムージング
                prob_df = pd.DataFrame({'close': probability})
                smoothed_prob_result = self.smoother.calculate(prob_df)
                smoothed_probability = smoothed_prob_result.values
            else:
                smoothed_state = state
                smoothed_probability = probability
            
            # 結果作成
            result = UltraRefinedVolatilityStateResult(
                state=smoothed_state,
                probability=smoothed_probability,
                confidence=confidence,
                raw_score=raw_score,
                str_values=str_values,
                hilbert_envelope=hilbert_envelope,
                hilbert_phase=hilbert_phase,
                instantaneous_frequency=instantaneous_frequency,
                wavelet_energy=wavelet_energy,
                adaptive_threshold=adaptive_threshold,
                spectral_entropy=spectral_entropy,
                fractal_dimension=fractal_dimension,
                adaptive_gain=adaptive_gain
            )
            
            # キャッシュ管理
            data_hash = self._get_data_hash(data)
            if len(self._result_cache) >= self._max_cache_size:
                oldest_key = next(iter(self._result_cache))
                del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._values = smoothed_state.astype(np.float64)
            
            return result
            
        except Exception as e:
            self.logger.error(f"超洗練されたボラティリティ状態計算エラー: {str(e)}\n{traceback.format_exc()}")
            return self._create_empty_result(len(data))
    
    def _create_empty_result(self, length: int) -> UltraRefinedVolatilityStateResult:
        """空の結果を作成"""
        empty_array = np.zeros(length)
        return UltraRefinedVolatilityStateResult(
            state=empty_array.astype(np.int8),
            probability=empty_array,
            confidence=empty_array,
            raw_score=empty_array,
            str_values=empty_array,
            hilbert_envelope=empty_array,
            hilbert_phase=empty_array,
            instantaneous_frequency=empty_array,
            wavelet_energy=empty_array,
            adaptive_threshold=empty_array,
            spectral_entropy=empty_array,
            fractal_dimension=empty_array,
            adaptive_gain=empty_array
        )
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算"""
        try:
            if isinstance(data, pd.DataFrame):
                length = len(data)
                first_val = float(data.iloc[0]['close']) if length > 0 else 0.0
                last_val = float(data.iloc[-1]['close']) if length > 0 else 0.0
            else:
                length = len(data)
                first_val = float(data[0, 3]) if length > 0 else 0.0
                last_val = float(data[-1, 3]) if length > 0 else 0.0
            
            params_sig = f"{self.str_period}_{self.sensitivity}_{self.confidence_threshold}"
            return f"{length}_{first_val}_{last_val}_{params_sig}"
        except:
            return f"{id(data)}_{self.str_period}_{self.sensitivity}"
    
    def get_state(self) -> Optional[np.ndarray]:
        """現在のボラティリティ状態を取得"""
        if self._values is not None:
            return self._values.astype(np.int8)
        return None
    
    def get_advanced_analysis(self) -> Optional[Dict[str, np.ndarray]]:
        """高度分析結果を取得"""
        if self._result_cache:
            latest_result = list(self._result_cache.values())[-1]
            return {
                'str_values': latest_result.str_values,
                'hilbert_envelope': latest_result.hilbert_envelope,
                'instantaneous_frequency': latest_result.instantaneous_frequency,
                'wavelet_energy': latest_result.wavelet_energy,
                'spectral_entropy': latest_result.spectral_entropy,
                'fractal_dimension': latest_result.fractal_dimension,
                'adaptive_threshold': latest_result.adaptive_threshold,
                'confidence': latest_result.confidence
            }
        return None
    
    def get_current_confidence(self) -> Optional[float]:
        """現在の信頼度を取得"""
        if self._result_cache:
            latest_result = list(self._result_cache.values())[-1]
            if len(latest_result.confidence) > 0:
                return float(latest_result.confidence[-1])
        return None
    
    def get_signal_quality_metrics(self) -> Optional[Dict[str, float]]:
        """シグナル品質指標を取得"""
        if self._result_cache:
            latest_result = list(self._result_cache.values())[-1]
            confidence = latest_result.confidence
            spectral_entropy = latest_result.spectral_entropy
            fractal_dim = latest_result.fractal_dimension
            
            valid_confidence = confidence[confidence > 0]
            valid_entropy = spectral_entropy[spectral_entropy > 0]
            valid_fractal = fractal_dim[fractal_dim > 0]
            
            if len(valid_confidence) > 0:
                return {
                    'avg_confidence': float(np.mean(valid_confidence)),
                    'confidence_stability': float(1.0 - np.std(valid_confidence)),
                    'avg_complexity': float(np.mean(valid_entropy)) if len(valid_entropy) > 0 else 0.0,
                    'avg_fractal_dimension': float(np.mean(valid_fractal)) if len(valid_fractal) > 0 else 0.0,
                    'high_confidence_ratio': float(np.sum(valid_confidence > 0.8) / len(valid_confidence))
                }
        return None
    
    def reset(self) -> None:
        """インジケーターをリセット"""
        super().reset()
        self._result_cache = {}
        self.str_indicator.reset()
        if self.smoothing:
            self.smoother.reset()