#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
統合ハースト指数計算器 (Unified Hurst Exponent Calculator)

複数の手法でハースト指数を計算し、統合結果を提供します。

実装手法:
1. 古典的なR/S法 (Rescaled Range Statistics)
2. DFA法 (Detrended Fluctuation Analysis) 
3. ウェーブレット法 (Daubechies Wavelet Method)

ハースト指数 (H):
- H < 0.5: 反持続性 (平均回帰傾向)
- H = 0.5: ランダムウォーク (記憶なし)
- H > 0.5: 持続性 (トレンド継続傾向)
"""

from typing import Union, Optional, Dict, Tuple, NamedTuple
import numpy as np
import pandas as pd
from numba import njit, jit
import math

# 相対インポート
try:
    from .indicator import Indicator
    from .price_source import PriceSource
except ImportError:
    # スタンドアロン実行時
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from indicator import Indicator
    from price_source import PriceSource


class HurstResult(NamedTuple):
    """ハースト指数計算結果"""
    hurst_rs: np.ndarray          # R/S法によるハースト指数
    hurst_dfa: np.ndarray         # DFA法によるハースト指数
    hurst_wavelet: np.ndarray     # ウェーブレット法によるハースト指数
    hurst_consensus: np.ndarray   # 統合ハースト指数
    confidence_score: np.ndarray  # 信頼度スコア (0-1)
    persistence_regime: np.ndarray # 持続性レジーム (-1 to 1)


# === Numba最適化関数群 ===

@njit(fastmath=True, cache=True)
def calculate_rs_hurst_numba(
    prices: np.ndarray, 
    window_size: int, 
    min_subseries: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    古典的なR/S法によるハースト指数計算 (Numba最適化版)
    
    Args:
        prices: 価格配列
        window_size: 計算ウィンドウサイズ
        min_subseries: 最小サブシリーズ長
    
    Returns:
        (hurst_values, confidence_scores)
    """
    n = len(prices)
    hurst_values = np.full(n, np.nan)
    confidence_scores = np.full(n, np.nan)
    
    for i in range(window_size, n):
        window_start = i - window_size + 1
        window_data = prices[window_start:i+1]
        
        # 対数リターン計算
        log_returns = np.zeros(len(window_data) - 1)
        for j in range(len(window_data) - 1):
            if window_data[j] > 0 and window_data[j+1] > 0:
                log_returns[j] = math.log(window_data[j+1] / window_data[j])
            else:
                log_returns[j] = 0.0
        
        if len(log_returns) < min_subseries:
            continue
        
        # 平均リターンと標準偏差
        mean_return = np.mean(log_returns)
        std_return = np.std(log_returns)
        
        if std_return <= 1e-10:
            continue
        
        # R/S統計計算（複数のサブシリーズ長で）
        subseries_lengths = np.array([8, 12, 16, 20, 24])
        subseries_lengths = subseries_lengths[subseries_lengths <= len(log_returns)]
        
        if len(subseries_lengths) < 3:
            continue
        
        log_rs_values = np.zeros(len(subseries_lengths))
        log_lengths = np.zeros(len(subseries_lengths))
        
        for k, subseries_len in enumerate(subseries_lengths):
            n_subseries = len(log_returns) // subseries_len
            if n_subseries == 0:
                continue
            
            rs_values = np.zeros(n_subseries)
            
            for sub_idx in range(n_subseries):
                start_idx = sub_idx * subseries_len
                end_idx = start_idx + subseries_len
                subseries = log_returns[start_idx:end_idx]
                
                # 累積偏差
                cumdev = np.zeros(subseries_len)
                cumsum_val = 0.0
                for m in range(subseries_len):
                    cumsum_val += (subseries[m] - mean_return)
                    cumdev[m] = cumsum_val
                
                # レンジ (R)
                R = np.max(cumdev) - np.min(cumdev)
                
                # 標準偏差 (S)
                S = np.std(subseries)
                
                if S > 1e-10:
                    rs_values[sub_idx] = R / S
                else:
                    rs_values[sub_idx] = 1.0
            
            # 平均R/S
            avg_rs = np.mean(rs_values)
            if avg_rs > 0:
                log_rs_values[k] = math.log(avg_rs)
                log_lengths[k] = math.log(subseries_len)
        
        # 有効なデータポイントのみ使用
        valid_mask = ~np.isnan(log_rs_values) & ~np.isinf(log_rs_values)
        valid_mask = valid_mask & (~np.isnan(log_lengths)) & (~np.isinf(log_lengths))
        
        if np.sum(valid_mask) >= 3:
            valid_log_rs = log_rs_values[valid_mask]
            valid_log_lengths = log_lengths[valid_mask]
            
            # 線形回帰（手動実装 - Numba互換）
            n_points = len(valid_log_rs)
            sum_x = np.sum(valid_log_lengths)
            sum_y = np.sum(valid_log_rs)
            sum_xy = np.sum(valid_log_lengths * valid_log_rs)
            sum_x2 = np.sum(valid_log_lengths ** 2)
            
            denominator = n_points * sum_x2 - sum_x ** 2
            if abs(denominator) > 1e-10:
                slope = (n_points * sum_xy - sum_x * sum_y) / denominator
                
                # R²計算
                mean_y = sum_y / n_points
                ss_tot = np.sum((valid_log_rs - mean_y) ** 2)
                intercept = (sum_y - slope * sum_x) / n_points
                predicted_y = slope * valid_log_lengths + intercept
                ss_res = np.sum((valid_log_rs - predicted_y) ** 2)
                
                if ss_tot > 1e-10:
                    r_squared = 1 - (ss_res / ss_tot)
                else:
                    r_squared = 0.0
                
                hurst_values[i] = slope
                confidence_scores[i] = max(0, min(1, r_squared))
    
    return hurst_values, confidence_scores


@njit(fastmath=True, cache=True)
def calculate_dfa_hurst_numba(
    prices: np.ndarray, 
    window_size: int,
    scales: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    DFA法によるハースト指数計算 (Numba最適化版)
    
    Args:
        prices: 価格配列
        window_size: 計算ウィンドウサイズ
        scales: DFAスケール配列
    
    Returns:
        (hurst_values, confidence_scores)
    """
    n = len(prices)
    hurst_values = np.full(n, np.nan)
    confidence_scores = np.full(n, np.nan)
    
    # デフォルトスケール
    if scales is None:
        scales = np.array([8., 12., 16., 20., 24., 32., 40.])
    
    for i in range(window_size, n):
        window_start = i - window_size + 1
        window_data = prices[window_start:i+1]
        
        # 対数リターン計算
        log_returns = np.zeros(len(window_data) - 1)
        for j in range(len(window_data) - 1):
            if window_data[j] > 0 and window_data[j+1] > 0:
                log_returns[j] = math.log(window_data[j+1] / window_data[j])
            else:
                log_returns[j] = 0.0
        
        if len(log_returns) < int(scales[-1]) + 5:
            continue
        
        # プロファイル計算（累積合計）
        mean_return = np.mean(log_returns)
        profile = np.zeros(len(log_returns) + 1)
        for j in range(len(log_returns)):
            profile[j+1] = profile[j] + (log_returns[j] - mean_return)
        
        # 各スケールでの揺動計算
        log_scales = np.zeros(len(scales))
        log_fluctuations = np.zeros(len(scales))
        
        for k, scale in enumerate(scales):
            scale_int = int(scale)
            if scale_int >= len(profile) - 1:
                continue
            
            n_segments = len(profile) // scale_int
            if n_segments == 0:
                continue
            
            segment_variances = np.zeros(n_segments)
            
            for seg_idx in range(n_segments):
                start_idx = seg_idx * scale_int
                end_idx = start_idx + scale_int
                
                if end_idx <= len(profile):
                    segment = profile[start_idx:end_idx]
                    
                    # 線形トレンド除去
                    # y = ax + b の最小二乗法
                    x_vals = np.arange(len(segment), dtype=np.float64)
                    n_seg = len(segment)
                    
                    sum_x = np.sum(x_vals)
                    sum_y = np.sum(segment)
                    sum_xy = np.sum(x_vals * segment)
                    sum_x2 = np.sum(x_vals ** 2)
                    
                    denominator = n_seg * sum_x2 - sum_x ** 2
                    if abs(denominator) > 1e-10:
                        a = (n_seg * sum_xy - sum_x * sum_y) / denominator
                        b = (sum_y - a * sum_x) / n_seg
                        
                        # トレンド除去
                        detrended = segment - (a * x_vals + b)
                        
                        # 分散計算
                        segment_variances[seg_idx] = np.mean(detrended ** 2)
                    else:
                        segment_variances[seg_idx] = np.var(segment)
            
            # 平均揺動
            avg_fluctuation = math.sqrt(np.mean(segment_variances))
            if avg_fluctuation > 0:
                log_scales[k] = math.log(scale)
                log_fluctuations[k] = math.log(avg_fluctuation)
        
        # 有効なデータポイントのみ使用
        valid_mask = ~np.isnan(log_fluctuations) & ~np.isinf(log_fluctuations)
        valid_mask = valid_mask & (~np.isnan(log_scales)) & (~np.isinf(log_scales))
        
        if np.sum(valid_mask) >= 3:
            valid_log_scales = log_scales[valid_mask]
            valid_log_fluctuations = log_fluctuations[valid_mask]
            
            # 線形回帰（手動実装）
            n_points = len(valid_log_scales)
            sum_x = np.sum(valid_log_scales)
            sum_y = np.sum(valid_log_fluctuations)
            sum_xy = np.sum(valid_log_scales * valid_log_fluctuations)
            sum_x2 = np.sum(valid_log_scales ** 2)
            
            denominator = n_points * sum_x2 - sum_x ** 2
            if abs(denominator) > 1e-10:
                slope = (n_points * sum_xy - sum_x * sum_y) / denominator
                
                # R²計算
                mean_y = sum_y / n_points
                ss_tot = np.sum((valid_log_fluctuations - mean_y) ** 2)
                intercept = (sum_y - slope * sum_x) / n_points
                predicted_y = slope * valid_log_scales + intercept
                ss_res = np.sum((valid_log_fluctuations - predicted_y) ** 2)
                
                if ss_tot > 1e-10:
                    r_squared = 1 - (ss_res / ss_tot)
                else:
                    r_squared = 0.0
                
                hurst_values[i] = slope
                confidence_scores[i] = max(0, min(1, r_squared))
    
    return hurst_values, confidence_scores


@njit(fastmath=True, cache=True)
def calculate_wavelet_hurst_numba(
    prices: np.ndarray, 
    window_size: int,
    scales: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Daubechiesウェーブレット法によるハースト指数計算 (Numba最適化版)
    
    Args:
        prices: 価格配列
        window_size: 計算ウィンドウサイズ
        scales: ウェーブレットスケール配列
    
    Returns:
        (hurst_values, confidence_scores)
    """
    n = len(prices)
    hurst_values = np.full(n, np.nan)
    confidence_scores = np.full(n, np.nan)
    
    # デフォルトスケール
    if scales is None:
        scales = np.array([4., 8., 16., 32., 64.])
    
    # Daubechies-4ウェーブレット係数
    db4_h = np.array([
        0.6830127, 1.1830127, 0.3169873, -0.1830127,
        -0.0544158, 0.0094624, 0.0102581, -0.0017468
    ])
    
    for i in range(window_size, n):
        window_start = i - window_size + 1
        window_data = prices[window_start:i+1]
        
        # 対数リターン計算
        log_returns = np.zeros(len(window_data) - 1)
        for j in range(len(window_data) - 1):
            if window_data[j] > 0 and window_data[j+1] > 0:
                log_returns[j] = math.log(window_data[j+1] / window_data[j])
            else:
                log_returns[j] = 0.0
        
        if len(log_returns) < int(scales[-1]) + 10:
            continue
        
        # 各スケールでのウェーブレット係数分散計算
        log_scales = np.zeros(len(scales))
        log_variances = np.zeros(len(scales))
        
        for k, scale in enumerate(scales):
            scale_int = int(scale)
            if scale_int >= len(log_returns) // 2:
                continue
            
            # Daubechies-4風ウェーブレット変換
            signal = log_returns.copy()
            detail_coeffs = np.zeros(len(signal) // scale_int)
            
            # ダウンサンプリングとフィルタリング
            for j in range(len(detail_coeffs)):
                start_idx = j * scale_int
                end_idx = min(start_idx + scale_int, len(signal))
                
                if end_idx - start_idx >= 8:  # Daubechies-4の最小長
                    segment = signal[start_idx:end_idx]
                    
                    # Daubechies-4風ハイパスフィルタ
                    detail_val = 0.0
                    norm_factor = 0.0
                    
                    for m in range(min(8, len(segment))):
                        if m < len(db4_h):
                            # ハイパス係数（ローパスの符号反転版）
                            h_val = db4_h[m] * ((-1) ** m)
                            detail_val += segment[m] * h_val
                            norm_factor += h_val ** 2
                    
                    if norm_factor > 0:
                        detail_coeffs[j] = detail_val / math.sqrt(norm_factor)
            
            # ディテール係数の分散
            if len(detail_coeffs) > 0:
                detail_variance = np.var(detail_coeffs)
                if detail_variance > 0:
                    log_scales[k] = math.log(scale)
                    log_variances[k] = math.log(detail_variance)
        
        # 有効なデータポイントのみ使用
        valid_mask = ~np.isnan(log_variances) & ~np.isinf(log_variances)
        valid_mask = valid_mask & (~np.isnan(log_scales)) & (~np.isinf(log_scales))
        
        if np.sum(valid_mask) >= 3:
            valid_log_scales = log_scales[valid_mask]
            valid_log_variances = log_variances[valid_mask]
            
            # 線形回帰（手動実装）
            n_points = len(valid_log_scales)
            sum_x = np.sum(valid_log_scales)
            sum_y = np.sum(valid_log_variances)
            sum_xy = np.sum(valid_log_scales * valid_log_variances)
            sum_x2 = np.sum(valid_log_scales ** 2)
            
            denominator = n_points * sum_x2 - sum_x ** 2
            if abs(denominator) > 1e-10:
                slope = (n_points * sum_xy - sum_x * sum_y) / denominator
                
                # ウェーブレット法では slope = 2H - 1
                # よって H = (slope + 1) / 2
                hurst_estimate = (slope + 1) / 2.0
                
                # R²計算
                mean_y = sum_y / n_points
                ss_tot = np.sum((valid_log_variances - mean_y) ** 2)
                intercept = (sum_y - slope * sum_x) / n_points
                predicted_y = slope * valid_log_scales + intercept
                ss_res = np.sum((valid_log_variances - predicted_y) ** 2)
                
                if ss_tot > 1e-10:
                    r_squared = 1 - (ss_res / ss_tot)
                else:
                    r_squared = 0.0
                
                hurst_values[i] = hurst_estimate
                confidence_scores[i] = max(0, min(1, r_squared))
    
    return hurst_values, confidence_scores


@njit(fastmath=True, cache=True)
def calculate_consensus_hurst(
    hurst_rs: np.ndarray,
    hurst_dfa: np.ndarray,
    hurst_wavelet: np.ndarray,
    conf_rs: np.ndarray,
    conf_dfa: np.ndarray,
    conf_wavelet: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    複数手法のハースト指数を統合し、コンセンサスハースト指数を計算
    
    Returns:
        (consensus_hurst, consensus_confidence, persistence_regime)
    """
    n = len(hurst_rs)
    consensus_hurst = np.full(n, np.nan)
    consensus_confidence = np.full(n, np.nan)
    persistence_regime = np.full(n, np.nan)
    
    for i in range(n):
        # 有効な推定値を収集
        estimates = []
        weights = []
        
        if not (np.isnan(hurst_rs[i]) or np.isinf(hurst_rs[i])):
            estimates.append(hurst_rs[i])
            weights.append(conf_rs[i] + 0.1)  # 基本重み
        
        if not (np.isnan(hurst_dfa[i]) or np.isinf(hurst_dfa[i])):
            estimates.append(hurst_dfa[i])
            weights.append(conf_dfa[i] + 0.15)  # DFAを若干重視
        
        if not (np.isnan(hurst_wavelet[i]) or np.isinf(hurst_wavelet[i])):
            estimates.append(hurst_wavelet[i])
            weights.append(conf_wavelet[i] + 0.05)
        
        if len(estimates) == 0:
            continue
        
        estimates = np.array(estimates)
        weights = np.array(weights)
        
        # 外れ値除去（四分位範囲法）
        if len(estimates) >= 3:
            q1 = np.percentile(estimates, 25)
            q3 = np.percentile(estimates, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            valid_mask = (estimates >= lower_bound) & (estimates <= upper_bound)
            estimates = estimates[valid_mask]
            weights = weights[valid_mask]
        
        if len(estimates) == 0:
            continue
        
        # 重み付け平均
        total_weight = np.sum(weights)
        if total_weight > 0:
            consensus_hurst[i] = np.sum(estimates * weights) / total_weight
            
            # 信頼度：手法間の一致度 + 個別信頼度の平均
            if len(estimates) > 1:
                agreement = 1.0 - np.std(estimates) / (np.mean(estimates) + 1e-8)
                agreement = max(0, min(1, agreement))
            else:
                agreement = 0.7  # 単一手法の場合
            
            avg_confidence = np.mean(weights) / (np.max(weights) + 0.2)  # 正規化
            consensus_confidence[i] = (agreement + avg_confidence) / 2.0
            
            # 持続性レジーム分類
            h_val = consensus_hurst[i]
            conf_val = consensus_confidence[i]
            
            if h_val < 0.45:
                persistence_regime[i] = -1.0  # 強い反持続性
            elif h_val < 0.48:
                persistence_regime[i] = -0.5  # 弱い反持続性
            elif h_val < 0.52:
                persistence_regime[i] = 0.0   # ランダムウォーク
            elif h_val < 0.55:
                persistence_regime[i] = 0.5   # 弱い持続性
            else:
                persistence_regime[i] = 1.0   # 強い持続性
            
            # 低信頼度の場合は中立に近づける
            if conf_val < 0.3:
                persistence_regime[i] *= 0.5
        else:
            consensus_hurst[i] = 0.5  # デフォルト値
            consensus_confidence[i] = 0.1
            persistence_regime[i] = 0.0
    
    return consensus_hurst, consensus_confidence, persistence_regime


# === 統合ハースト指数インジケータークラス ===

class UnifiedHurstExponent(Indicator):
    """
    統合ハースト指数計算器
    
    複数の手法でハースト指数を計算し、コンセンサス結果を提供します。
    
    実装手法:
    1. 古典的なR/S法 (Rescaled Range Statistics)
    2. DFA法 (Detrended Fluctuation Analysis)
    3. ウェーブレット法 (Daubechies Wavelet Method)
    
    ハースト指数の解釈:
    - H < 0.5: 反持続性（平均回帰傾向）
    - H = 0.5: ランダムウォーク（記憶なし）
    - H > 0.5: 持続性（トレンド継続傾向）
    """
    
    def __init__(
        self,
        window_size: int = 100,
        src_type: str = 'close',
        enable_rs: bool = True,
        enable_dfa: bool = True,
        enable_wavelet: bool = True,
        dfa_scales: Optional[np.ndarray] = None,
        wavelet_scales: Optional[np.ndarray] = None
    ):
        """
        Args:
            window_size: 計算ウィンドウサイズ
            src_type: 価格ソースタイプ
            enable_rs: R/S法を有効にするか
            enable_dfa: DFA法を有効にするか
            enable_wavelet: ウェーブレット法を有効にするか
            dfa_scales: DFA法のスケール配列
            wavelet_scales: ウェーブレット法のスケール配列
        """
        super().__init__(f"UnifiedHurstExponent(window={window_size})")
        
        self.window_size = max(50, window_size)  # 最小50期間
        self.src_type = src_type
        self.enable_rs = enable_rs
        self.enable_dfa = enable_dfa
        self.enable_wavelet = enable_wavelet
        
        # デフォルトスケール設定
        self.dfa_scales = dfa_scales if dfa_scales is not None else np.array([8, 12, 16, 20, 24, 32, 40])
        self.wavelet_scales = wavelet_scales if wavelet_scales is not None else np.array([4, 8, 16, 32, 64])
        
        # 結果キャッシュ
        self._last_result: Optional[HurstResult] = None
        self._data_hash: Optional[str] = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> HurstResult:
        """
        統合ハースト指数を計算
        
        Args:
            data: 価格データ
        
        Returns:
            HurstResult: ハースト指数計算結果
        """
        try:
            # データハッシュチェック
            current_hash = self._get_data_hash(data)
            if self._last_result is not None and current_hash == self._data_hash:
                return self._last_result
            
            # 価格データの抽出
            if isinstance(data, np.ndarray) and data.ndim == 1:
                prices = data.copy()
            else:
                prices = PriceSource.calculate_source(data, self.src_type)
            
            if len(prices) < self.window_size:
                self.logger.warning(f"データ長({len(prices)})がウィンドウサイズ({self.window_size})より短いです")
                empty_array = np.full(len(prices), np.nan)
                return HurstResult(
                    hurst_rs=empty_array.copy(),
                    hurst_dfa=empty_array.copy(),
                    hurst_wavelet=empty_array.copy(),
                    hurst_consensus=empty_array.copy(),
                    confidence_score=empty_array.copy(),
                    persistence_regime=empty_array.copy()
                )
            
            n = len(prices)
            
            # 各手法でハースト指数を計算
            hurst_rs = np.full(n, np.nan)
            conf_rs = np.full(n, np.nan)
            
            hurst_dfa = np.full(n, np.nan)
            conf_dfa = np.full(n, np.nan)
            
            hurst_wavelet = np.full(n, np.nan)
            conf_wavelet = np.full(n, np.nan)
            
            # 1. R/S法
            if self.enable_rs:
                hurst_rs, conf_rs = calculate_rs_hurst_numba(prices, self.window_size)
                self.logger.info("R/S法ハースト指数計算完了")
            
            # 2. DFA法
            if self.enable_dfa:
                hurst_dfa, conf_dfa = calculate_dfa_hurst_numba(prices, self.window_size, self.dfa_scales)
                self.logger.info("DFA法ハースト指数計算完了")
            
            # 3. ウェーブレット法
            if self.enable_wavelet:
                hurst_wavelet, conf_wavelet = calculate_wavelet_hurst_numba(prices, self.window_size, self.wavelet_scales)
                self.logger.info("ウェーブレット法ハースト指数計算完了")
            
            # 4. コンセンサスハースト指数計算
            consensus_hurst, consensus_confidence, persistence_regime = calculate_consensus_hurst(
                hurst_rs, hurst_dfa, hurst_wavelet,
                conf_rs, conf_dfa, conf_wavelet
            )
            
            # 結果を作成
            result = HurstResult(
                hurst_rs=hurst_rs,
                hurst_dfa=hurst_dfa,
                hurst_wavelet=hurst_wavelet,
                hurst_consensus=consensus_hurst,
                confidence_score=consensus_confidence,
                persistence_regime=persistence_regime
            )
            
            # キャッシュを更新
            self._last_result = result
            self._data_hash = current_hash
            
            self.logger.info("統合ハースト指数計算完了")
            return result
            
        except Exception as e:
            self.logger.error(f"ハースト指数計算エラー: {e}")
            data_len = len(data) if hasattr(data, '__len__') else 0
            empty_array = np.full(data_len, np.nan)
            return HurstResult(
                hurst_rs=empty_array.copy(),
                hurst_dfa=empty_array.copy(),
                hurst_wavelet=empty_array.copy(),
                hurst_consensus=empty_array.copy(),
                confidence_score=empty_array.copy(),
                persistence_regime=empty_array.copy()
            )
    
    def get_consensus_hurst(self) -> Optional[np.ndarray]:
        """コンセンサスハースト指数を取得"""
        if self._last_result is not None:
            return self._last_result.hurst_consensus.copy()
        return None
    
    def get_persistence_regime(self) -> Optional[np.ndarray]:
        """持続性レジームを取得"""
        if self._last_result is not None:
            return self._last_result.persistence_regime.copy()
        return None
    
    def get_confidence_score(self) -> Optional[np.ndarray]:
        """信頼度スコアを取得"""
        if self._last_result is not None:
            return self._last_result.confidence_score.copy()
        return None
    
    def get_method_comparison(self) -> Optional[Dict[str, np.ndarray]]:
        """各手法の比較結果を取得"""
        if self._last_result is not None:
            return {
                'rs_method': self._last_result.hurst_rs.copy(),
                'dfa_method': self._last_result.hurst_dfa.copy(),
                'wavelet_method': self._last_result.hurst_wavelet.copy(),
                'consensus': self._last_result.hurst_consensus.copy()
            }
        return None
    
    def get_latest_analysis(self) -> Dict:
        """最新の分析結果サマリーを取得"""
        if self._last_result is None:
            return {}
        
        result = self._last_result
        
        # 最新の有効値を取得
        def get_latest_valid(arr):
            valid_mask = ~np.isnan(arr) & ~np.isinf(arr)
            if np.any(valid_mask):
                valid_indices = np.where(valid_mask)[0]
                return float(arr[valid_indices[-1]])
            return None
        
        latest_consensus = get_latest_valid(result.hurst_consensus)
        latest_confidence = get_latest_valid(result.confidence_score)
        latest_regime = get_latest_valid(result.persistence_regime)
        
        # 解釈
        interpretation = "不明"
        if latest_consensus is not None:
            if latest_consensus < 0.45:
                interpretation = "強い反持続性（平均回帰傾向）"
            elif latest_consensus < 0.48:
                interpretation = "弱い反持続性"
            elif latest_consensus < 0.52:
                interpretation = "ランダムウォーク"
            elif latest_consensus < 0.55:
                interpretation = "弱い持続性"
            else:
                interpretation = "強い持続性（トレンド継続傾向）"
        
        return {
            'latest_hurst_consensus': latest_consensus,
            'latest_confidence': latest_confidence,
            'latest_persistence_regime': latest_regime,
            'interpretation': interpretation,
            'method_results': {
                'rs_hurst': get_latest_valid(result.hurst_rs),
                'dfa_hurst': get_latest_valid(result.hurst_dfa),
                'wavelet_hurst': get_latest_valid(result.hurst_wavelet)
            },
            'enabled_methods': {
                'rs_method': self.enable_rs,
                'dfa_method': self.enable_dfa,
                'wavelet_method': self.enable_wavelet
            },
            'parameters': {
                'window_size': self.window_size,
                'src_type': self.src_type,
                'dfa_scales': self.dfa_scales.tolist(),
                'wavelet_scales': self.wavelet_scales.tolist()
            }
        }
    
    def reset(self) -> None:
        """状態をリセット"""
        super().reset()
        self._last_result = None
        self._data_hash = None
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データハッシュを計算"""
        # 簡単なハッシュ関数
        if isinstance(data, pd.DataFrame):
            data_str = f"{data.shape}_{self.src_type}_{data.iloc[0].sum() if len(data) > 0 else 0}_{data.iloc[-1].sum() if len(data) > 0 else 0}"
        elif isinstance(data, np.ndarray):
            data_str = f"{data.shape}_{data[0] if len(data) > 0 else 0}_{data[-1] if len(data) > 0 else 0}"
        else:
            data_str = str(data)
        
        param_str = f"{self.window_size}_{self.enable_rs}_{self.enable_dfa}_{self.enable_wavelet}"
        return f"{data_str}_{param_str}"


# === 使用例関数 ===

def example_usage():
    """統合ハースト指数の使用例"""
    print("=== 統合ハースト指数計算器 使用例 ===")
    
    # サンプルデータ生成
    np.random.seed(42)
    n_samples = 200
    
    # 1. ランダムウォーク（H ≈ 0.5）
    random_walk = np.cumsum(np.random.randn(n_samples)) + 100
    
    # 2. 平均回帰プロセス（H < 0.5）
    mean_reverting = np.zeros(n_samples)
    mean_reverting[0] = 100
    for i in range(1, n_samples):
        mean_reverting[i] = mean_reverting[i-1] + 0.05 * (100 - mean_reverting[i-1]) + np.random.randn() * 0.5
    
    # 3. トレンドプロセス（H > 0.5）
    trending = np.zeros(n_samples)
    trending[0] = 100
    trend = 0
    for i in range(1, n_samples):
        trend += np.random.randn() * 0.01
        trending[i] = trending[i-1] + trend + np.random.randn() * 0.3
    
    # 統合ハースト指数計算器
    hurst_calculator = UnifiedHurstExponent(
        window_size=80,
        enable_rs=True,
        enable_dfa=True,
        enable_wavelet=True
    )
    
    test_cases = [
        ("ランダムウォーク", random_walk),
        ("平均回帰プロセス", mean_reverting),
        ("トレンドプロセス", trending)
    ]
    
    for name, data in test_cases:
        print(f"\n--- {name} ---")
        
        result = hurst_calculator.calculate(data)
        analysis = hurst_calculator.get_latest_analysis()
        
        print(f"コンセンサスハースト指数: {analysis['latest_hurst_consensus']:.4f}")
        print(f"信頼度: {analysis['latest_confidence']:.4f}")
        print(f"解釈: {analysis['interpretation']}")
        
        print("手法別結果:")
        for method, value in analysis['method_results'].items():
            if value is not None:
                print(f"  {method}: {value:.4f}")
        
        hurst_calculator.reset()
    
    print("\n=== 完了 ===")


if __name__ == "__main__":
    example_usage() 