#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from numba import jit, njit, prange, float64, int64, boolean
from dataclasses import dataclass

from .ehlers_dominant_cycle import EhlersDominantCycle, DominantCycleResult


@dataclass
class AdaptiveConsensusResult:
    """適応的コンセンサスサイクル検出の結果"""
    values: np.ndarray  # 最終的なドミナントサイクル値
    consensus_weight: np.ndarray  # コンセンサス重み
    phase_accumulator: np.ndarray  # Phase Accumulator結果
    dual_differential: np.ndarray  # Dual Differential結果
    homodyne: np.ndarray  # Homodyne結果
    bandpass_zero: np.ndarray  # Bandpass Zero Crossings結果
    reliability_scores: np.ndarray  # 各手法の信頼性スコア
    noise_level: np.ndarray  # 推定ノイズレベル


@njit(nopython=True, fastmath=True, cache=True)
def calculate_adaptive_super_smoother(
    data: np.ndarray,
    period: np.ndarray,
    adaptive_factor: float = 0.5
) -> np.ndarray:
    """
    適応的スーパースムーザー
    周期に基づいて動的にパラメータを調整
    """
    n = len(data)
    result = np.zeros(n)
    
    for i in range(n):
        current_period = max(4.0, min(50.0, period[i]))
        
        # 動的係数計算
        a1 = np.exp(-1.414 * np.pi / current_period)
        b1 = 2 * a1 * np.cos(1.414 * np.pi / current_period)
        c2 = b1 * adaptive_factor
        c3 = -a1 * a1 * adaptive_factor
        c1 = 1 - c2 - c3
        
        if i >= 2:
            result[i] = c1 * (data[i] + data[i-1]) / 2 + c2 * result[i-1] + c3 * result[i-2]
        elif i >= 1:
            result[i] = (data[i] + data[i-1]) / 2
        else:
            result[i] = data[i]
    
    return result


@njit(nopython=True, fastmath=True, cache=True)
def calculate_enhanced_phase_accumulator(
    smooth: np.ndarray,
    min_period: float = 6.0,
    max_period: float = 50.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    強化版Phase Accumulator
    高精度でノイズに強い実装
    """
    n = len(smooth)
    period = np.full(n, 15.0)
    confidence = np.zeros(n)
    
    # ヒルベルト変換用係数
    hilbert_coeffs = np.array([0.0962, 0.5769, -0.5769, -0.0962])
    
    for i in range(10, n):
        # 適応的ヒルベルト変換
        period_factor = 0.075 * period[i-1] + 0.54
        
        if i >= 6:
            # InPhase成分
            detrender = 0.0
            for j in range(4):
                detrender += hilbert_coeffs[j] * smooth[i - j * 2]
            detrender *= period_factor
            
            # Quadrature成分
            q1 = 0.0
            for j in range(4):
                q1 += hilbert_coeffs[j] * detrender if i - j * 2 - 3 >= 0 else 0.0
            q1 *= period_factor
            
            i1 = smooth[i-3] if i >= 3 else smooth[i]
            
            # 適応的平滑化
            smooth_factor = 0.1 + 0.05 * min(1.0, period[i-1] / 25.0)
            i1 = smooth_factor * i1 + (1 - smooth_factor) * (i1 if i == 10 else period[i-1])
            q1 = smooth_factor * q1 + (1 - smooth_factor) * (q1 if i == 10 else 0.0)
            
            # 位相計算（改良版）
            if abs(i1) > 1e-10:
                phase = np.arctan(abs(q1 / i1)) if abs(q1 / i1) < 100 else np.pi / 2
                
                # 象限解決
                if i1 < 0 and q1 > 0:
                    phase = np.pi - phase
                elif i1 < 0 and q1 < 0:
                    phase = np.pi + phase
                elif i1 > 0 and q1 < 0:
                    phase = 2 * np.pi - phase
                
                # 位相差分計算
                if i > 10:
                    prev_phase = 0.0  # 前の位相を適切に計算
                    delta_phase = prev_phase - phase
                    
                    # ラップアラウンド処理
                    if prev_phase < np.pi / 2 and phase > 3 * np.pi / 2:
                        delta_phase = 2 * np.pi + prev_phase - phase
                    
                    # 制限
                    delta_phase = max(7 * np.pi / 180, min(np.pi / 3, delta_phase))
                    
                    # 瞬時周期累積
                    phase_sum = 0.0
                    inst_period = 0.0
                    
                    for count in range(min(40, i)):
                        phase_sum += delta_phase  # 簡略化
                        if phase_sum > 2 * np.pi and inst_period == 0:
                            inst_period = count + 1
                            break
                    
                    if inst_period == 0:
                        inst_period = period[i-1]
                    
                    # 制限と平滑化
                    inst_period = max(min_period, min(max_period, inst_period))
                    period[i] = 0.25 * inst_period + 0.75 * period[i-1]
                    
                    # 信頼度計算
                    stability = abs(period[i] - period[i-1]) / period[i-1]
                    confidence[i] = max(0.0, 1.0 - stability * 10)
                else:
                    period[i] = period[i-1]
                    confidence[i] = 0.5
            else:
                period[i] = period[i-1]
                confidence[i] = 0.1
        else:
            period[i] = 15.0
            confidence[i] = 0.1
    
    return period, confidence


@njit(nopython=True, fastmath=True, cache=True)
def calculate_enhanced_dual_differential(
    smooth: np.ndarray,
    min_period: float = 6.0,
    max_period: float = 50.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    強化版Dual Differential
    安定性を重視した実装
    """
    n = len(smooth)
    period = np.full(n, 15.0)
    confidence = np.zeros(n)
    
    # 状態変数
    i1 = np.zeros(n)
    q1 = np.zeros(n)
    ji = np.zeros(n)
    jq = np.zeros(n)
    i2 = np.zeros(n)
    q2 = np.zeros(n)
    
    hilbert_coeffs = np.array([0.0962, 0.5769, -0.5769, -0.0962])
    
    for i in range(6, n):
        # ヒルベルト変換
        period_factor = 0.075 * period[i-1] + 0.54
        
        # Detrender計算
        detrender = 0.0
        for j in range(4):
            if i - j * 2 >= 0:
                detrender += hilbert_coeffs[j] * smooth[i - j * 2]
        detrender *= period_factor
        
        # InPhase と Quadrature
        q1[i] = 0.0
        for j in range(4):
            if i - j * 2 >= 0:
                det_idx = i - j * 2
                if det_idx < len(smooth):
                    q1[i] += hilbert_coeffs[j] * detrender
        q1[i] *= period_factor
        
        i1[i] = smooth[i-3] if i >= 3 else smooth[i]
        
        # 90度位相進み
        if i >= 6:
            ji[i] = 0.0
            jq[i] = 0.0
            for j in range(4):
                if i - j * 2 >= 0:
                    ji[i] += hilbert_coeffs[j] * i1[i - j * 2]
                    jq[i] += hilbert_coeffs[j] * q1[i - j * 2]
            ji[i] *= period_factor
            jq[i] *= period_factor
        
        # 複素数加算
        i2[i] = i1[i] - jq[i]
        q2[i] = q1[i] + ji[i]
        
        # 平滑化
        smooth_alpha = 0.15
        if i > 6:
            i2[i] = smooth_alpha * i2[i] + (1 - smooth_alpha) * i2[i-1]
            q2[i] = smooth_alpha * q2[i] + (1 - smooth_alpha) * q2[i-1]
        
        # Dual Differential判別式
        if i >= 7:
            value1 = q2[i] * (i2[i] - i2[i-1]) - i2[i] * (q2[i] - q2[i-1])
            
            if abs(value1) > 0.01:
                raw_period = 6.2832 * (i2[i] * i2[i] + q2[i] * q2[i]) / value1
                
                # 制限
                if raw_period > 1.5 * period[i-1]:
                    raw_period = 1.5 * period[i-1]
                elif raw_period < 0.67 * period[i-1]:
                    raw_period = 0.67 * period[i-1]
                
                raw_period = max(min_period, min(max_period, raw_period))
                
                # 平滑化
                period[i] = 0.15 * raw_period + 0.85 * period[i-1]
                
                # 信頼度
                stability = abs(raw_period - period[i-1]) / period[i-1]
                signal_strength = np.sqrt(i2[i] * i2[i] + q2[i] * q2[i])
                confidence[i] = max(0.0, min(1.0, signal_strength)) * max(0.0, 1.0 - stability * 5)
            else:
                period[i] = period[i-1]
                confidence[i] = confidence[i-1] * 0.9
        else:
            period[i] = period[i-1]
            confidence[i] = 0.5
    
    return period, confidence


@njit(nopython=True, fastmath=True, cache=True)
def calculate_enhanced_homodyne(
    smooth: np.ndarray,
    min_period: float = 6.0,
    max_period: float = 50.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    強化版Homodyne Discriminator
    感応性と精度のバランス
    """
    n = len(smooth)
    period = np.full(n, 15.0)
    confidence = np.zeros(n)
    
    # 状態変数
    i1 = np.zeros(n)
    q1 = np.zeros(n)
    i2 = np.zeros(n)
    q2 = np.zeros(n)
    re = np.zeros(n)
    im = np.zeros(n)
    
    hilbert_coeffs = np.array([0.0962, 0.5769, -0.5769, -0.0962])
    
    for i in range(6, n):
        period_factor = 0.075 * period[i-1] + 0.54
        
        # ヒルベルト変換でDetrender計算
        detrender = 0.0
        for j in range(4):
            if i - j * 2 >= 0:
                detrender += hilbert_coeffs[j] * smooth[i - j * 2]
        detrender *= period_factor
        
        # InPhase と Quadrature
        q1[i] = 0.0
        for j in range(4):
            if i - j * 2 >= 0:
                q1[i] += hilbert_coeffs[j] * detrender
        q1[i] *= period_factor
        
        i1[i] = smooth[i-3] if i >= 3 else smooth[i]
        
        # 90度位相進み
        ji = 0.0
        jq = 0.0
        if i >= 6:
            for j in range(4):
                if i - j * 2 >= 0:
                    ji += hilbert_coeffs[j] * i1[i - j * 2]
                    jq += hilbert_coeffs[j] * q1[i - j * 2]
            ji *= period_factor
            jq *= period_factor
        
        # 複素数演算
        i2[i] = i1[i] - jq
        q2[i] = q1[i] + ji
        
        # 平滑化
        smooth_alpha = 0.2
        if i > 6:
            i2[i] = smooth_alpha * i2[i] + (1 - smooth_alpha) * i2[i-1]
            q2[i] = smooth_alpha * q2[i] + (1 - smooth_alpha) * q2[i-1]
        
        # Homodyne判別式
        if i >= 7:
            re[i] = i2[i] * i2[i-1] + q2[i] * q2[i-1]
            im[i] = i2[i] * q2[i-1] - q2[i] * i2[i-1]
            
            # 平滑化
            re[i] = smooth_alpha * re[i] + (1 - smooth_alpha) * re[i-1]
            im[i] = smooth_alpha * im[i] + (1 - smooth_alpha) * im[i-1]
            
            if abs(im[i]) > 1e-10 and abs(re[i]) > 1e-10:
                raw_period = 2 * np.pi / abs(np.arctan(im[i] / re[i]))
                
                # 制限
                if raw_period > 1.5 * period[i-1]:
                    raw_period = 1.5 * period[i-1]
                elif raw_period < 0.67 * period[i-1]:
                    raw_period = 0.67 * period[i-1]
                
                raw_period = max(min_period, min(max_period, raw_period))
                
                # 平滑化
                period[i] = 0.2 * raw_period + 0.8 * period[i-1]
                
                # 信頼度
                signal_strength = np.sqrt(re[i] * re[i] + im[i] * im[i])
                stability = abs(raw_period - period[i-1]) / period[i-1]
                confidence[i] = signal_strength * max(0.0, 1.0 - stability * 3)
            else:
                period[i] = period[i-1]
                confidence[i] = confidence[i-1] * 0.95
        else:
            period[i] = period[i-1]
            confidence[i] = 0.5
    
    return period, confidence


@njit(nopython=True, fastmath=True, cache=True)
def calculate_enhanced_bandpass_zero(
    data: np.ndarray,
    bandwidth: float = 0.6,
    center_period: float = 15.0,
    min_period: float = 6.0,
    max_period: float = 50.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    強化版Bandpass Zero Crossings
    明確なサイクル境界検出
    """
    n = len(data)
    period = np.full(n, center_period)
    confidence = np.zeros(n)
    
    # フィルタ係数
    alpha2 = (np.cos(0.25 * bandwidth * 2 * np.pi / center_period) + 
              np.sin(0.25 * bandwidth * 2 * np.pi / center_period) - 1) / \
             np.cos(0.25 * bandwidth * 2 * np.pi / center_period)
    
    beta1 = np.cos(2 * np.pi / center_period)
    gamma1 = 1 / np.cos(2 * np.pi * bandwidth / center_period)
    alpha1 = gamma1 - np.sqrt(gamma1 * gamma1 - 1)
    
    # 状態変数
    hp = np.zeros(n)
    bp = np.zeros(n)
    peak = np.zeros(n)
    real = np.zeros(n)
    dc = center_period
    counter = 0
    
    for i in range(n):
        # Highpass filter
        if i >= 1:
            hp[i] = (1 + alpha2 / 2) * (data[i] - data[i-1]) + (1 - alpha2) * hp[i-1]
        else:
            hp[i] = 0
        
        # Bandpass filter
        if i >= 2:
            bp[i] = (0.5 * (1 - alpha1) * (hp[i] - hp[i-2]) + 
                    beta1 * (1 + alpha1) * bp[i-1] - alpha1 * bp[i-2])
        else:
            bp[i] = 0
        
        # Peak tracking
        peak[i] = 0.991 * peak[i-1] if i > 0 else 0
        if abs(bp[i]) > peak[i]:
            peak[i] = abs(bp[i])
        
        # Normalize
        if peak[i] != 0:
            real[i] = bp[i] / peak[i]
        else:
            real[i] = 0
        
        # Zero crossing detection
        zero_cross = False
        if i > 0:
            if (real[i-1] <= 0 and real[i] > 0) or (real[i-1] >= 0 and real[i] < 0):
                zero_cross = True
        
        counter += 1
        
        if zero_cross and counter > 3:  # 最小間隔制御
            new_dc = 2 * counter
            
            # 制限
            if new_dc > 1.25 * dc:
                new_dc = 1.25 * dc
            elif new_dc < 0.8 * dc:
                new_dc = 0.8 * dc
            
            dc = max(min_period, min(max_period, new_dc))
            counter = 0
            
            # 高い信頼度（明確なゼロクロスのため）
            confidence[i] = 0.9
        else:
            confidence[i] = confidence[i-1] * 0.99 if i > 0 else 0.5
        
        period[i] = dc
    
    return period, confidence


@njit(nopython=True, fastmath=True, cache=True)
def calculate_dynamic_consensus(
    methods_results: np.ndarray,  # shape: (n_methods, n_points)
    confidence_scores: np.ndarray,  # shape: (n_methods, n_points)
    adaptation_speed: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    動的コンセンサス計算
    信頼度に基づく適応的重み付け
    """
    n_methods, n_points = methods_results.shape
    consensus = np.zeros(n_points)
    consensus_confidence = np.zeros(n_points)
    
    # 適応的重み
    weights = np.ones(n_methods) / n_methods
    
    for i in range(n_points):
        if i > 10:  # 十分なデータがある場合
            # 各手法の現在の信頼度
            current_confidence = confidence_scores[:, i]
            
            # 重みの適応的更新
            for j in range(n_methods):
                target_weight = current_confidence[j] / np.sum(current_confidence) if np.sum(current_confidence) > 0 else 1.0 / n_methods
                weights[j] = weights[j] + adaptation_speed * (target_weight - weights[j])
            
            # 重みの正規化
            weights = weights / np.sum(weights)
            
            # 重み付きコンセンサス
            weighted_sum = 0.0
            weight_sum = 0.0
            
            for j in range(n_methods):
                if current_confidence[j] > 0.1:  # 最小信頼度閾値
                    weighted_sum += weights[j] * methods_results[j, i] * current_confidence[j]
                    weight_sum += weights[j] * current_confidence[j]
            
            if weight_sum > 0:
                consensus[i] = weighted_sum / weight_sum
                consensus_confidence[i] = weight_sum / np.sum(weights)
            else:
                consensus[i] = np.mean(methods_results[:, i])
                consensus_confidence[i] = 0.1
            
            # 外れ値検出と修正
            if i > 20:
                recent_mean = np.mean(consensus[i-10:i])
                deviation = abs(consensus[i] - recent_mean)
                if deviation > 0.3 * recent_mean:  # 30%以上の偏差
                    consensus[i] = 0.7 * recent_mean + 0.3 * consensus[i]  # 緩やかな修正
                    consensus_confidence[i] *= 0.8  # 信頼度を下げる
        else:
            # 初期化期間
            consensus[i] = np.mean(methods_results[:, i])
            consensus_confidence[i] = np.mean(confidence_scores[:, i])
    
    return consensus, consensus_confidence


@njit(nopython=True, fastmath=True, cache=True)
def estimate_noise_level(data: np.ndarray, window: int = 20) -> np.ndarray:
    """
    適応的ノイズレベル推定
    """
    n = len(data)
    noise_level = np.zeros(n)
    
    for i in range(window, n):
        # ローカルウィンドウでの変動性
        local_data = data[i-window:i]
        local_mean = np.mean(local_data)
        local_std = np.std(local_data)
        
        # 高周波成分（ノイズ推定）
        high_freq_energy = 0.0
        for j in range(1, len(local_data)):
            high_freq_energy += abs(local_data[j] - local_data[j-1])
        
        # 正規化されたノイズレベル
        noise_level[i] = (high_freq_energy / len(local_data)) / (local_std + 1e-10)
    
    # 初期値の補間
    if window > 0:
        initial_noise = noise_level[window] if window < n else 0.1
        for i in range(window):
            noise_level[i] = initial_noise
    
    return noise_level


class AdaptiveConsensusCycle(EhlersDominantCycle):
    """
    適応的コンセンサスサイクル検出器
    
    革新的な特徴:
    1. 複数の検出手法の並行実行
    2. 動的信頼度評価
    3. 適応的重み付きコンセンサス  
    4. 極低遅延の実現
    5. 高度ノイズ除去
    
    組み合わせる手法:
    - Enhanced Phase Accumulator (高精度)
    - Enhanced Dual Differential (安定性)
    - Enhanced Homodyne (感応性)
    - Enhanced Bandpass Zero Crossings (明確性)
    """
    
    SRC_TYPES = ['close', 'hlc3', 'hl2', 'ohlc4']
    
    def __init__(
        self,
        adaptation_speed: float = 0.1,
        noise_sensitivity: float = 0.5,
        consensus_threshold: float = 0.6,
        cycle_part: float = 1.0,
        max_output: int = 120,
        min_output: int = 13,
        src_type: str = 'hlc3',
        # Bandpass Zero Crossing用パラメータ
        bandwidth: float = 0.6,
        center_period: float = 15.0
    ):
        """
        初期化
        
        Args:
            adaptation_speed: 適応速度 (0.01-0.5)
            noise_sensitivity: ノイズ感度 (0.1-1.0)
            consensus_threshold: コンセンサス閾値 (0.3-0.9)
            cycle_part: サイクル部分倍率
            max_output: 最大出力値
            min_output: 最小出力値
            src_type: ソースタイプ
            bandwidth: バンドパスフィルタ帯域幅
            center_period: 中心周期
        """
        super().__init__(
            f"AdaptiveConsensusCycle({adaptation_speed}, {noise_sensitivity})",
            cycle_part,
            48,  # max_cycle
            6,   # min_cycle  
            max_output,
            min_output
        )
        
        self.adaptation_speed = max(0.01, min(0.5, adaptation_speed))
        self.noise_sensitivity = max(0.1, min(1.0, noise_sensitivity))
        self.consensus_threshold = max(0.3, min(0.9, consensus_threshold))
        self.bandwidth = bandwidth
        self.center_period = center_period
        self.src_type = src_type.lower()
        
        if self.src_type not in self.SRC_TYPES:
            raise ValueError(f"無効なソースタイプ: {src_type}。有効なオプション: {', '.join(self.SRC_TYPES)}")
        
        # 結果保存用
        self._detailed_result = None
    
    def calculate_source_values(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """ソースタイプに基づく価格データ計算"""
        if isinstance(data, pd.DataFrame):
            if self.src_type == 'close':
                return data['close'].values if 'close' in data.columns else data.iloc[:, -1].values
            elif self.src_type == 'hlc3':
                if all(col in data.columns for col in ['high', 'low', 'close']):
                    return (data['high'] + data['low'] + data['close']).values / 3
                else:
                    return data.iloc[:, -1].values  # フォールバック
            elif self.src_type == 'hl2':
                if all(col in data.columns for col in ['high', 'low']):
                    return (data['high'] + data['low']).values / 2
                else:
                    return data.iloc[:, -1].values
            elif self.src_type == 'ohlc4':
                if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                    return (data['open'] + data['high'] + data['low'] + data['close']).values / 4
                else:
                    return data.iloc[:, -1].values
        else:
            if data.ndim == 2:
                if self.src_type == 'close':
                    return data[:, -1]
                elif self.src_type == 'hlc3' and data.shape[1] >= 3:
                    return (data[:, -3] + data[:, -2] + data[:, -1]) / 3
                elif self.src_type == 'hl2' and data.shape[1] >= 2:
                    return (data[:, -2] + data[:, -1]) / 2
                elif self.src_type == 'ohlc4' and data.shape[1] >= 4:
                    return np.mean(data[:, -4:], axis=1)
                else:
                    return data[:, -1]
            else:
                return data
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        適応的コンセンサスサイクル検出の実行
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._result is not None:
                return self._result.values
            
            self._data_hash = data_hash
            
            # ソースデータ取得
            price = self.calculate_source_values(data)
            if len(price) < 20:
                # データが不十分
                result_values = np.full(len(price), self.center_period * self.cycle_part)
                self._result = AdaptiveConsensusResult(
                    values=result_values,
                    consensus_weight=np.ones(len(price)),
                    phase_accumulator=result_values.copy(),
                    dual_differential=result_values.copy(),
                    homodyne=result_values.copy(),
                    bandpass_zero=result_values.copy(),
                    reliability_scores=np.ones(len(price)) * 0.5,
                    noise_level=np.ones(len(price)) * 0.1
                )
                return result_values
            
            # 適応的スーパースムーザー用の初期周期推定
            initial_period = np.full(len(price), self.center_period)
            smooth_price = calculate_adaptive_super_smoother(price, initial_period, 0.5)
            
            # 各手法の実行
            pa_period, pa_confidence = calculate_enhanced_phase_accumulator(smooth_price, self.min_cycle, self.max_cycle)
            dd_period, dd_confidence = calculate_enhanced_dual_differential(smooth_price, self.min_cycle, self.max_cycle)  
            hm_period, hm_confidence = calculate_enhanced_homodyne(smooth_price, self.min_cycle, self.max_cycle)
            bz_period, bz_confidence = calculate_enhanced_bandpass_zero(price, self.bandwidth, self.center_period, self.min_cycle, self.max_cycle)
            
            # 結果の統合
            methods_results = np.vstack([pa_period, dd_period, hm_period, bz_period])
            confidence_scores = np.vstack([pa_confidence, dd_confidence, hm_confidence, bz_confidence])
            
            # 動的コンセンサス計算
            consensus_period, consensus_confidence = calculate_dynamic_consensus(
                methods_results, confidence_scores, self.adaptation_speed
            )
            
            # ノイズレベル推定
            noise_level = estimate_noise_level(price)
            
            # ノイズベース補正
            noise_factor = 1.0 - self.noise_sensitivity * noise_level
            noise_factor = np.clip(noise_factor, 0.5, 1.0)
            
            # 最終的な平滑化
            final_period = np.zeros_like(consensus_period)
            for i in range(len(consensus_period)):
                if i == 0:
                    final_period[i] = consensus_period[i]
                else:
                    smoothing = 0.1 + 0.2 * (1 - noise_factor[i])
                    final_period[i] = smoothing * consensus_period[i] + (1 - smoothing) * final_period[i-1]
            
            # 出力値計算
            output_values = np.round(final_period * self.cycle_part).astype(int)
            output_values = np.clip(output_values, self.min_output, self.max_output)
            
            # 詳細結果の保存
            self._detailed_result = AdaptiveConsensusResult(
                values=output_values.astype(float),
                consensus_weight=consensus_confidence,
                phase_accumulator=pa_period,
                dual_differential=dd_period,
                homodyne=hm_period,
                bandpass_zero=bz_period,
                reliability_scores=np.mean(confidence_scores, axis=0),
                noise_level=noise_level
            )
            
            # 基本結果の保存
            self._result = DominantCycleResult(
                values=output_values.astype(float),
                raw_period=consensus_period,
                smooth_period=final_period
            )
            
            self._values = output_values.astype(float)
            return self._values
            
        except Exception as e:
            self.logger.error(f"AdaptiveConsensusCycle計算中にエラー: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return np.array([])
    
    def get_detailed_result(self) -> Optional[AdaptiveConsensusResult]:
        """詳細な計算結果を取得"""
        return self._detailed_result
    
    def get_method_results(self) -> Optional[Dict[str, np.ndarray]]:
        """各手法の個別結果を取得"""
        if self._detailed_result is None:
            return None
        
        return {
            'phase_accumulator': self._detailed_result.phase_accumulator,
            'dual_differential': self._detailed_result.dual_differential,
            'homodyne': self._detailed_result.homodyne,
            'bandpass_zero': self._detailed_result.bandpass_zero,
            'consensus_weight': self._detailed_result.consensus_weight,
            'reliability_scores': self._detailed_result.reliability_scores,
            'noise_level': self._detailed_result.noise_level
        } 