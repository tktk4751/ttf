#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Optional, Dict, Any
import numpy as np
import pandas as pd
from numba import njit, float64

from ..indicator import Indicator
from ..price_source import PriceSource
from ..smoother.unified_smoother import UnifiedSmoother
from ..utils.percentile_analysis import (
    calculate_percentile,
    calculate_trend_classification,
    PercentileAnalysisMixin
)

# 条件付きインポート（オプション機能）
try:
    from ..cycle.ehlers_unified_dc import EhlersUnifiedDC
    EHLERS_UNIFIED_DC_AVAILABLE = True
except ImportError:
    try:
        # 絶対インポートを試行（パス調整付き）
        import sys
        import os
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        from indicators.cycle.ehlers_unified_dc import EhlersUnifiedDC
        EHLERS_UNIFIED_DC_AVAILABLE = True
    except ImportError:
        EhlersUnifiedDC = None
        EHLERS_UNIFIED_DC_AVAILABLE = False

try:
    from ..kalman.unified_kalman import UnifiedKalman
    UNIFIED_KALMAN_AVAILABLE = True
except ImportError:
    try:
        # 絶対インポートを試行（パス調整付き）
        import sys
        import os
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        from indicators.kalman.unified_kalman import UnifiedKalman
        UNIFIED_KALMAN_AVAILABLE = True
    except ImportError:
        UnifiedKalman = None
        UNIFIED_KALMAN_AVAILABLE = False


@dataclass
class EhlersMarketModeResult:
    """Ehlers Market Mode Indicatorの計算結果"""
    smooth: np.ndarray                # スムーズされた価格
    detrender: np.ndarray            # デトレンダー値
    period: np.ndarray               # 測定されたサイクル期間
    smooth_period: np.ndarray        # スムーズされたサイクル期間
    dc_phase: np.ndarray             # DCフェーズ
    trend_line: np.ndarray           # トレンドライン
    trend_mode: np.ndarray           # トレンドモード（1=トレンド、0=サイクル）
    signal: np.ndarray               # 売買信号（1=買い、-1=売り、0=中立）
    days_in_trend: np.ndarray        # トレンド継続日数
    i_trend: np.ndarray              # iTrend値
    filtered_price: np.ndarray       # カルマンフィルター適用後の価格（オプション）
    smoothed_values: np.ndarray      # 平滑化された値（オプション）
    # パーセンタイルベースのトレンド分析
    percentiles: Optional[np.ndarray]     # パーセンタイル値
    trend_state: Optional[np.ndarray]     # トレンド状態（-1=下降、0=レンジ、1=上昇）
    trend_intensity: Optional[np.ndarray] # トレンド強度（0-1）


@njit(fastmath=True, cache=True)
def hyper_advanced_dft_numba(
    data: np.ndarray,
    window_size: int = 70,
    overlap: float = 0.85,
    zero_padding_factor: int = 8
) -> tuple:
    """
    超高度DFT解析 - 高精度な離散フーリエ変換アルゴリズム
    
    Args:
        data: 入力データ配列
        window_size: ウィンドウサイズ
        overlap: オーバーラップ率
        zero_padding_factor: ゼロパディング係数
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (周波数, 信頼度, コヒーレンス)
    """
    n = len(data)
    if n < window_size:
        window_size = n // 2
    
    step_size = max(1, int(window_size * (1 - overlap)))
    
    frequencies = np.zeros(n)
    confidences = np.zeros(n)
    coherences = np.zeros(n)
    
    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        window_data = data[start:end]
        
        # 高品質ウィンドウ関数の組み合わせ
        blackman_harris = np.zeros(window_size)
        flattop = np.zeros(window_size)
        gaussian = np.zeros(window_size)
        
        for i in range(window_size):
            t = 2 * np.pi * i / (window_size - 1)
            
            # Blackman-Harris ウィンドウ
            blackman_harris[i] = (0.35875 - 0.48829 * np.cos(t) + 
                                 0.14128 * np.cos(2*t) - 0.01168 * np.cos(3*t))
            
            # Flat-top ウィンドウ
            flattop[i] = (0.21557895 - 0.41663158 * np.cos(t) + 
                         0.277263158 * np.cos(2*t) - 0.083578947 * np.cos(3*t) + 
                         0.006947368 * np.cos(4*t))
            
            # Gaussian ウィンドウ
            sigma = 0.4
            gaussian[i] = np.exp(-0.5 * ((i - window_size/2) / (sigma * window_size/2))**2)
        
        # 最適組み合わせウィンドウ
        optimal_window = (0.5 * blackman_harris + 0.3 * flattop + 0.2 * gaussian)
        windowed_data = window_data * optimal_window
        
        # 超高度ゼロパディング
        padded_size = window_size * zero_padding_factor
        padded_data = np.zeros(padded_size)
        padded_data[:window_size] = windowed_data
        
        # 高精度DFT計算（6-50期間）
        period_count = 45
        freqs = np.zeros(period_count)
        powers = np.zeros(period_count)
        phases = np.zeros(period_count)
        
        for period_idx in range(period_count):
            period = 6 + period_idx
            if period < padded_size // 2:
                real_part = 0.0
                imag_part = 0.0
                
                for i in range(padded_size):
                    angle = 2.0 * np.pi * i / period
                    real_part += padded_data[i] * np.cos(angle)
                    imag_part += padded_data[i] * np.sin(angle)
                
                power = real_part**2 + imag_part**2
                phase = np.arctan2(imag_part, real_part)
                
                freqs[period_idx] = period
                powers[period_idx] = power
                phases[period_idx] = phase
        
        # 高度な重心アルゴリズム
        max_power = np.max(powers)
        if max_power > 0:
            normalized_powers = powers / max_power
        else:
            normalized_powers = powers
        
        # デシベル変換
        db_values = np.zeros(period_count)
        for i in range(period_count):
            if normalized_powers[i] > 0.001:
                ratio = normalized_powers[i]
                db_values[i] = -10 * np.log10(0.001 / (1 - 0.999 * ratio))
            else:
                db_values[i] = 30
            
            if db_values[i] > 30:
                db_values[i] = 30
        
        # 超高度重心計算
        numerator = 0.0
        denominator = 0.0
        total_weight = 0.0
        
        for i in range(period_count):
            weight = (30 - db_values[i]) ** 2
            total_weight += weight
            if db_values[i] <= 2:
                numerator += freqs[i] * weight
                denominator += weight
        
        if denominator > 1e-10:
            dominant_freq = numerator / denominator
            confidence = denominator / max(total_weight, 1e-10)
        else:
            dominant_freq = 20.0
            confidence = 0.1
        
        # 位相コヒーレンス計算
        max_idx = np.argmax(powers)
        coherence = 0.0
        
        if max_idx > 1 and max_idx < len(phases) - 2:
            phase_diffs = np.zeros(5)
            count = 0
            
            for j in range(-2, 3):
                if 0 <= max_idx + j < len(phases):
                    phase_diffs[count] = phases[max_idx + j]
                    count += 1
            
            if count > 2:
                mean_phase = 0.0
                for k in range(count):
                    mean_phase += phase_diffs[k]
                mean_phase /= max(count, 1)
                
                phase_variance = 0.0
                for k in range(count):
                    diff = phase_diffs[k] - mean_phase
                    phase_variance += diff * diff
                phase_variance /= max(count, 1)
                
                coherence = 1.0 / (1.0 + phase_variance * 10)
        
        # 結果保存
        mid_point = start + window_size // 2
        if mid_point < n:
            frequencies[mid_point] = dominant_freq
            confidences[mid_point] = confidence
            coherences[mid_point] = coherence
    
    # 高度な補間
    for i in range(n):
        if frequencies[i] == 0.0:
            left_vals = np.zeros(10)
            right_vals = np.zeros(10)
            left_count = 0
            right_count = 0
            
            for j in range(max(0, i-10), i):
                if frequencies[j] > 0 and left_count < 10:
                    left_vals[left_count] = frequencies[j]
                    left_count += 1
            
            for j in range(i+1, min(n, i+11)):
                if frequencies[j] > 0 and right_count < 10:
                    right_vals[right_count] = frequencies[j]
                    right_count += 1
            
            if left_count >= 2 and right_count >= 2:
                p0 = left_vals[left_count-2]
                p1 = left_vals[left_count-1]
                p2 = right_vals[0]
                p3 = right_vals[1]
                
                t = 0.5
                frequencies[i] = (0.5 * (2*p1 + (-p0 + p2)*t + 
                                 (2*p0 - 5*p1 + 4*p2 - p3)*t*t + 
                                 (-p0 + 3*p1 - 3*p2 + p3)*t*t*t))
            elif left_count > 0 and right_count > 0:
                frequencies[i] = (left_vals[left_count-1] + right_vals[0]) / 2
            elif left_count > 0:
                frequencies[i] = left_vals[left_count-1]
            elif right_count > 0:
                frequencies[i] = right_vals[0]
            else:
                frequencies[i] = 20.0
    
    return frequencies, confidences, coherences


@njit(fastmath=True, cache=True)
def calculate_detrender_numba(
    smooth: np.ndarray,
    period: np.ndarray
) -> np.ndarray:
    """
    デトレンダーを計算する（Numba最適化版）
    detrender = ((0.0962 * smooth) + (0.5769 * nz(smooth[2])) - (0.5769 * nz(smooth[4])) - (0.0962 * nz(smooth[6]))) * ((0.075 * nz(period[1])) + 0.54)
    
    Args:
        smooth: スムーズされた価格配列
        period: 周期配列
        
    Returns:
        デトレンダー配列
    """
    length = len(smooth)
    detrender = np.full(length, np.nan, dtype=np.float64)
    
    for i in range(length):
        # 前の期間値を取得（デフォルト値は20）
        prev_period = 20.0
        if i > 0 and not np.isnan(period[i-1]):
            prev_period = period[i-1]
        
        # フィルター係数
        filter_coef = (0.075 * prev_period) + 0.54
        
        # デトレンダー計算
        if i >= 6:
            detrender[i] = ((0.0962 * smooth[i]) + 
                           (0.5769 * smooth[i-2]) - 
                           (0.5769 * smooth[i-4]) - 
                           (0.0962 * smooth[i-6])) * filter_coef
        else:
            # 十分なデータがない場合は簡略化
            detrender[i] = 0.0
    
    return detrender


@njit(fastmath=True, cache=True)
def calculate_cycle_period_numba(
    detrender: np.ndarray
) -> tuple:
    """
    サイクル期間を計算する（Numba最適化版）
    
    Args:
        detrender: デトレンダー配列
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (period, smooth_period)
    """
    length = len(detrender)
    period = np.full(length, 20.0, dtype=np.float64)  # デフォルト期間20
    smooth_period = np.full(length, 20.0, dtype=np.float64)
    
    i1 = np.full(length, np.nan, dtype=np.float64)
    q1 = np.full(length, np.nan, dtype=np.float64)
    i2 = np.full(length, np.nan, dtype=np.float64)
    q2 = np.full(length, np.nan, dtype=np.float64)
    re = np.full(length, np.nan, dtype=np.float64)
    im = np.full(length, np.nan, dtype=np.float64)
    
    for i in range(length):
        # 前の期間値を取得
        prev_period = 20.0
        if i > 0:
            prev_period = period[i-1]
        
        # フィルター係数
        filter_coef = (0.075 * prev_period) + 0.54
        
        # Q1の計算
        if i >= 6:
            q1[i] = ((0.0962 * detrender[i]) + 
                    (0.5769 * detrender[i-2]) - 
                    (0.5769 * detrender[i-4]) - 
                    (0.0962 * detrender[i-6])) * filter_coef
        
        # I1の計算（3期間遅れのデトレンダー）
        if i >= 3:
            i1[i] = detrender[i-3]
        
        # JIとJQの計算
        jI = 0.0
        jQ = 0.0
        if i >= 6:
            jI = ((0.0962 * i1[i]) + 
                 (0.5769 * i1[i-2]) - 
                 (0.5769 * i1[i-4]) - 
                 (0.0962 * i1[i-6])) * filter_coef
            
            jQ = ((0.0962 * q1[i]) + 
                 (0.5769 * q1[i-2]) - 
                 (0.5769 * q1[i-4]) - 
                 (0.0962 * q1[i-6])) * filter_coef
        
        # I2とQ2の計算
        if not np.isnan(i1[i]) and not np.isnan(q1[i]):
            i2_raw = i1[i] - jQ
            q2_raw = q1[i] + jI
            
            # 平滑化
            if i > 0 and not np.isnan(i2[i-1]):
                i2[i] = 0.2 * i2_raw + 0.8 * i2[i-1]
            else:
                i2[i] = i2_raw
                
            if i > 0 and not np.isnan(q2[i-1]):
                q2[i] = 0.2 * q2_raw + 0.8 * q2[i-1]
            else:
                q2[i] = q2_raw
        
        # REとIMの計算
        if i > 0 and not np.isnan(i2[i]) and not np.isnan(q2[i]) and not np.isnan(i2[i-1]) and not np.isnan(q2[i-1]):
            re_raw = i2[i] * i2[i-1] + q2[i] * q2[i-1]
            im_raw = i2[i] * q2[i-1] - q2[i] * i2[i-1]
            
            # 平滑化
            if i > 0 and not np.isnan(re[i-1]):
                re[i] = 0.2 * re_raw + 0.8 * re[i-1]
            else:
                re[i] = re_raw
                
            if i > 0 and not np.isnan(im[i-1]):
                im[i] = 0.2 * im_raw + 0.8 * im[i-1]
            else:
                im[i] = im_raw
        
        # 周期の計算（ゼロ除算防止）
        if not np.isnan(im[i]) and not np.isnan(re[i]) and abs(im[i]) > 1e-10 and abs(re[i]) > 1e-10:
            # ゼロ除算防止
            ratio = im[i] / max(abs(re[i]), 1e-10)
            if abs(ratio) < 1e10:  # 極端に大きな値を防ぐ
                raw_period = 2.0 * np.pi / abs(np.arctan(ratio))
                
                # 周期の制限
                if i > 0:
                    raw_period = max(raw_period, 0.67 * period[i-1])
                    raw_period = min(raw_period, 1.5 * period[i-1])
                
                raw_period = max(6.0, min(50.0, raw_period))
                
                # 平滑化
                if i > 0:
                    period[i] = 0.2 * raw_period + 0.8 * period[i-1]
                else:
                    period[i] = raw_period
            else:
                if i > 0:
                    period[i] = period[i-1]
                else:
                    period[i] = 20.0
        else:
            if i > 0:
                period[i] = period[i-1]
            else:
                period[i] = 20.0
        
        # スムーズ期間の計算
        if i > 0:
            smooth_period[i] = 0.33 * period[i] + 0.67 * smooth_period[i-1]
        else:
            smooth_period[i] = period[i]
    
    return period, smooth_period


@njit(fastmath=True, cache=True)
def calculate_dc_phase_simple_numba(
    smooth: np.ndarray,
    smooth_period: np.ndarray
) -> np.ndarray:
    """
    DCフェーズを計算する（安定版）
    
    Args:
        smooth: スムーズされた価格配列
        smooth_period: スムーズされた周期配列
        
    Returns:
        DCフェーズ配列
    """
    length = len(smooth)
    dc_phase = np.full(length, np.nan, dtype=np.float64)
    
    for i in range(length):
        if np.isnan(smooth_period[i]) or smooth_period[i] <= 1e-10:
            continue
            
        dc_period = int(np.ceil(smooth_period[i] + 0.5))
        if dc_period <= 0 or i < dc_period:
            continue
            
        real = 0.0
        imag = 0.0
        
        # DFT計算（ゼロ除算防止）
        for j in range(dc_period):
            if i - j >= 0:
                angle = 2.0 * np.pi * j / max(dc_period, 1)
                real += np.sin(angle) * smooth[i - j]
                imag += np.cos(angle) * smooth[i - j]
        
        # フェーズ計算（ゼロ除算防止）
        if abs(imag) > 1e-10:
            phase = np.arctan(real / imag) * 180.0 / np.pi
        else:
            phase = 90.0 * (1.0 if real >= 0 else -1.0)
        
        phase += 90.0
        if smooth_period[i] > 1e-10:
            phase += 360.0 / smooth_period[i]
        
        if imag < 0:
            phase += 180.0
            
        # フェーズの正規化
        while phase > 315.0:
            phase -= 360.0
        while phase < -45.0:
            phase += 360.0
            
        dc_phase[i] = phase
    
    return dc_phase


@njit(fastmath=True, cache=True)
def calculate_trend_line_numba(
    src: np.ndarray,
    smooth_period: np.ndarray
) -> np.ndarray:
    """
    トレンドラインを計算する（Numba最適化版）
    
    Args:
        src: ソース価格配列
        smooth_period: スムーズされた周期配列
        
    Returns:
        トレンドライン配列
    """
    length = len(src)
    i_trend = np.full(length, np.nan, dtype=np.float64)
    trend_line = np.full(length, np.nan, dtype=np.float64)
    
    for i in range(length):
        if np.isnan(smooth_period[i]) or smooth_period[i] <= 0:
            continue
            
        dc_period = int(np.ceil(smooth_period[i] + 0.5))
        if dc_period <= 0 or i < dc_period:
            continue
        
        # iTrendの計算（期間内の平均）
        sum_val = 0.0
        count = 0
        for j in range(dc_period):
            if i - j >= 0:
                sum_val += src[i - j]
                count += 1
        
        if count > 0:
            i_trend[i] = sum_val / count
        
        # トレンドラインの計算（スムージング）
        if not np.isnan(i_trend[i]):
            if i >= 3:
                trend_line[i] = (4.0 * i_trend[i] + 
                               3.0 * (i_trend[i-1] if not np.isnan(i_trend[i-1]) else i_trend[i]) + 
                               2.0 * (i_trend[i-2] if not np.isnan(i_trend[i-2]) else i_trend[i]) + 
                               (i_trend[i-3] if not np.isnan(i_trend[i-3]) else i_trend[i])) / 10.0
            else:
                trend_line[i] = i_trend[i]
    
    return i_trend, trend_line


@njit(fastmath=True, cache=True)
def calculate_market_mode_numba(
    smooth: np.ndarray,
    trend_line: np.ndarray,
    dc_phase: np.ndarray,
    smooth_period: np.ndarray
) -> tuple:
    """
    マーケットモードを計算する（Numba最適化版）
    
    Args:
        smooth: スムーズされた価格配列
        trend_line: トレンドライン配列
        dc_phase: DCフェーズ配列
        smooth_period: スムーズされた周期配列
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (trend_mode, signal, days_in_trend)
    """
    length = len(smooth)
    trend_mode = np.full(length, 1.0, dtype=np.float64)  # デフォルトはトレンド
    signal = np.full(length, 0.0, dtype=np.float64)
    days_in_trend = np.full(length, 0.0, dtype=np.float64)
    
    current_days = 0.0
    
    for i in range(length):
        if i > 0 and not np.isnan(dc_phase[i]) and not np.isnan(dc_phase[i-1]) and not np.isnan(smooth_period[i]):
            # フェーズ変化の検出
            sin_current = np.sin(dc_phase[i] * np.pi / 180.0)
            sin_next = np.sin((dc_phase[i] + 45.0) * np.pi / 180.0)
            sin_prev = np.sin(dc_phase[i-1] * np.pi / 180.0)
            sin_prev_next = np.sin((dc_phase[i-1] + 45.0) * np.pi / 180.0)
            
            # クロスオーバー・クロスアンダーの検出
            crossover = (sin_current > sin_next) and (sin_prev <= sin_prev_next)
            crossunder = (sin_current < sin_next) and (sin_prev >= sin_prev_next)
            
            if crossover or crossunder:
                current_days = 0.0
                trend_mode[i] = 0.0  # サイクルモード
            
            current_days += 1.0
            days_in_trend[i] = current_days
            
            # トレンドモードの判定
            if current_days < 0.5 * smooth_period[i]:
                trend_mode[i] = 0.0  # サイクルモード
            else:
                trend_mode[i] = 1.0  # トレンドモード
            
            # フェーズ変化率による判定（ゼロ除算防止）
            if smooth_period[i] > 1e-10:
                phase_change = dc_phase[i] - dc_phase[i-1]
                expected_change = 360.0 / smooth_period[i]
                
                if (phase_change > 0.67 * expected_change and 
                    phase_change < 1.5 * expected_change):
                    trend_mode[i] = 0.0  # サイクルモード
            
            # 価格とトレンドラインの偏差による判定（ゼロ除算防止）
            if (not np.isnan(trend_line[i]) and abs(trend_line[i]) > 1e-10 and
                abs((smooth[i] - trend_line[i]) / trend_line[i]) >= 0.015):
                trend_mode[i] = 1.0  # トレンドモード
        else:
            if i > 0:
                current_days += 1.0
                days_in_trend[i] = current_days
                trend_mode[i] = trend_mode[i-1]
        
        # 売買信号の生成
        if not np.isnan(trend_line[i]) and not np.isnan(smooth[i]):
            if smooth[i] > trend_line[i]:
                signal[i] = 1.0   # 買いシグナル
            elif smooth[i] < trend_line[i]:
                signal[i] = -1.0  # 売りシグナル
            else:
                signal[i] = 0.0   # 中立
    
    return trend_mode, signal, days_in_trend


class EhlersMarketMode(Indicator, PercentileAnalysisMixin):
    """
    Ehlers Market Mode Indicator
    
    John F. Ehlersによる市場モード検出インジケーター：
    サイクル分析とフェーズ分析を用いて市場がトレンドモードか
    サイクルモードかを判定
    
    特徴:
    - 自動サイクル期間検出
    - DCフェーズ分析
    - トレンドライン計算
    - モード判定（トレンド vs サイクル）
    - 売買信号生成
    - カルマンフィルター・スムーサー統合対応
    - パーセンタイル分析機能付き
    
    計算手順:
    1. ソース価格データを取得
    2. カルマンフィルターを適用（オプション）
    3. 価格をスムージング
    4. デトレンダーを計算
    5. サイクル期間を検出
    6. DCフェーズを計算
    7. トレンドラインを計算
    8. マーケットモードを判定
    """
    
    def __init__(
        self,
        src_type: str = 'hl2',
        trend_threshold: float = 0.015,
        # 平滑化オプション
        use_smoothing: bool = False,
        smoother_type: str = 'super_smoother',
        smoother_period: int = 8,
        smoother_src_type: str = 'close',
        # エラーズ統合サイクル検出器パラメータ
        use_dynamic_period: bool = False,
        detector_type: str = 'phac_e',
        lp_period: int = 5,
        hp_period: int = 144,
        cycle_part: float = 0.5,
        max_cycle: int = 144,
        min_cycle: int = 5,
        max_output: int = 55,
        min_output: int = 5,
        # 統合カルマンフィルターパラメータ
        use_kalman_filter: bool = False,
        kalman_filter_type: str = 'unscented',
        kalman_process_noise: float = 0.01,
        kalman_observation_noise: float = 0.001,
        # パーセンタイルベーストレンド分析パラメータ
        enable_percentile_analysis: bool = True,
        percentile_lookback_period: int = 50,
        percentile_low_threshold: float = 0.25,
        percentile_high_threshold: float = 0.75
    ):
        """
        コンストラクタ
        
        Args:
            src_type: ソースタイプ（デフォルト: 'hl2'）
            trend_threshold: トレンド判定閾値（デフォルト: 0.015）
            use_smoothing: 平滑化を使用するか（デフォルト: False）
            smoother_type: 統合スムーサータイプ（デフォルト: 'super_smoother'）
            smoother_period: スムーサー期間（デフォルト: 8）
            smoother_src_type: スムーサーソースタイプ（デフォルト: 'close'）
            use_dynamic_period: 動的期間適応を使用するか（デフォルト: False）
            detector_type: サイクル検出器タイプ（デフォルト: 'phac_e'）
            lp_period: ローパスフィルター期間（デフォルト: 5）
            hp_period: ハイパスフィルター期間（デフォルト: 144）
            cycle_part: サイクル部分（デフォルト: 0.5）
            max_cycle: 最大サイクル期間（デフォルト: 144）
            min_cycle: 最小サイクル期間（デフォルト: 5）
            max_output: 最大出力値（デフォルト: 55）
            min_output: 最小出力値（デフォルト: 5）
            use_kalman_filter: カルマンフィルターを使用するか（デフォルト: False）
            kalman_filter_type: カルマンフィルタータイプ（デフォルト: 'unscented'）
            kalman_process_noise: カルマンフィルタープロセスノイズ（デフォルト: 0.01）
            kalman_observation_noise: カルマンフィルター観測ノイズ（デフォルト: 0.001）
            enable_percentile_analysis: パーセンタイル分析を有効にするか（デフォルト: True）
            percentile_lookback_period: パーセンタイル分析のルックバック期間（デフォルト: 50）
            percentile_low_threshold: パーセンタイル分析の低閾値（デフォルト: 0.25）
            percentile_high_threshold: パーセンタイル分析の高閾値（デフォルト: 0.75）
        """
        indicator_name = f"EhlersMarketMode(src={src_type}, threshold={trend_threshold:.3f}"
        if use_dynamic_period:
            indicator_name += f", dynamic={detector_type}"
        if use_kalman_filter:
            indicator_name += f", kalman={kalman_filter_type}"
        if use_smoothing:
            indicator_name += f", smooth={smoother_type}({smoother_period})"
        indicator_name += ")"
        
        super().__init__(indicator_name)
        
        # パラメータ保存
        self.src_type = src_type
        self.trend_threshold = trend_threshold
        self.use_smoothing = use_smoothing
        self.smoother_type = smoother_type
        self.smoother_period = smoother_period
        self.smoother_src_type = smoother_src_type
        
        # エラーズ統合サイクル検出器パラメータ
        self.use_dynamic_period = use_dynamic_period
        self.detector_type = detector_type
        self.lp_period = lp_period
        self.hp_period = hp_period
        self.cycle_part = cycle_part
        self.max_cycle = max_cycle
        self.min_cycle = min_cycle
        self.max_output = max_output
        self.min_output = min_output
        
        # 統合カルマンフィルターパラメータ
        self.use_kalman_filter = use_kalman_filter
        self.kalman_filter_type = kalman_filter_type
        self.kalman_process_noise = kalman_process_noise
        self.kalman_observation_noise = kalman_observation_noise
        
        # パーセンタイルベーストレンド分析パラメータの初期化
        self._add_percentile_analysis_params(
            enable_percentile_analysis=enable_percentile_analysis,
            percentile_lookback_period=percentile_lookback_period,
            percentile_low_threshold=percentile_low_threshold,
            percentile_high_threshold=percentile_high_threshold
        )
        
        # パラメータ検証
        if self.trend_threshold <= 0:
            raise ValueError("trend_thresholdは0より大きい必要があります")
        if self.use_dynamic_period and self.max_cycle <= self.min_cycle:
            raise ValueError("max_cycleはmin_cycleより大きい必要があります")
        if self.use_kalman_filter and self.kalman_process_noise <= 0:
            raise ValueError("kalman_process_noiseは0より大きい必要があります")
        
        # エラーズ統合サイクル検出器の初期化（動的期間適応が有効な場合）
        self.cycle_detector = None
        
        if self.use_dynamic_period:
            if not EHLERS_UNIFIED_DC_AVAILABLE:
                self.logger.error("エラーズ統合サイクル検出器が利用できません。indicators.cycle.ehlers_unified_dcをインポートできません。")
                self.use_dynamic_period = False
                self.logger.warning("動的期間適応機能を無効にしました")
            else:
                try:
                    self.cycle_detector = EhlersUnifiedDC(
                        detector_type=self.detector_type,
                        cycle_part=self.cycle_part,
                        max_cycle=self.max_cycle,
                        min_cycle=self.min_cycle,
                        max_output=self.max_output,
                        min_output=self.min_output,
                        src_type='hlc3',
                        use_kalman_filter=False,
                        lp_period=self.lp_period,
                        hp_period=self.hp_period
                    )
                    self.logger.info(f"エラーズ統合サイクル検出器を初期化しました: {self.detector_type}")
                except Exception as e:
                    self.logger.error(f"エラーズ統合サイクル検出器の初期化に失敗: {e}")
                    self.use_dynamic_period = False
                    self.logger.warning("動的期間適応機能を無効にしました")
        
        # 統合カルマンフィルターの初期化（カルマンフィルターが有効な場合）
        self.kalman_filter = None
        if self.use_kalman_filter:
            if not UNIFIED_KALMAN_AVAILABLE:
                self.logger.error("統合カルマンフィルターが利用できません。indicators.kalman.unified_kalmanをインポートできません。")
                self.use_kalman_filter = False
                self.logger.warning("カルマンフィルター機能を無効にしました")
            else:
                try:
                    self.kalman_filter = UnifiedKalman(
                        filter_type=self.kalman_filter_type,
                        src_type=self.src_type,
                        process_noise_scale=self.kalman_process_noise,
                        observation_noise_scale=self.kalman_observation_noise
                    )
                    self.logger.info(f"統合カルマンフィルターを初期化しました: {self.kalman_filter_type}")
                except Exception as e:
                    self.logger.error(f"統合カルマンフィルターの初期化に失敗: {e}")
                    self.use_kalman_filter = False
                    self.logger.warning("カルマンフィルター機能を無効にしました")
        
        # 統合スムーサーの初期化（価格スムージング用）
        self.price_smoother = None
        try:
            # 価格スムージング用のスムーサー（Pine Scriptの重み付き移動平均をエミュレート）
            self.price_smoother = UnifiedSmoother(
                smoother_type='alma',  # ALMA（重み付き移動平均に近い）
                src_type=self.src_type,
                length=4,  # Pine Scriptの4期間に対応
                offset=0.25,  # より新しい価格に重点
                sigma=4.0
            )
        except Exception as e:
            self.logger.error(f"価格スムーサーの初期化に失敗: {e}")
            self.price_smoother = None
        
        # 統合スムーサーの初期化（オプション）
        self.smoother = None
        if self.use_smoothing:
            try:
                self.smoother = UnifiedSmoother(
                    smoother_type=self.smoother_type,
                    src_type=self.smoother_src_type,
                    period=self.smoother_period
                )
            except Exception as e:
                self.logger.error(f"統合スムーサーの初期化に失敗: {e}")
                self.use_smoothing = False
                self.logger.warning("平滑化機能を無効にしました")
        
        # 結果キャッシュ
        self._result_cache = {}
        self._max_cache_size = 10
        self._cache_keys = []
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算してキャッシュに使用する"""
        try:
            if isinstance(data, pd.DataFrame):
                length = len(data)
                if length > 0:
                    first_val = float(data.iloc[0].get('close', data.iloc[0, -1]))
                    last_val = float(data.iloc[-1].get('close', data.iloc[-1, -1]))
                else:
                    first_val = last_val = 0.0
            else:
                length = len(data)
                if length > 0:
                    if data.ndim > 1:
                        first_val = float(data[0, -1])
                        last_val = float(data[-1, -1])
                    else:
                        first_val = float(data[0])
                        last_val = float(data[-1])
                else:
                    first_val = last_val = 0.0
            
            # パラメータ情報
            param_str = (f"{self.src_type}_{self.trend_threshold}_{self.use_smoothing}_"
                        f"{self.smoother_type}_{self.smoother_period}_{self.smoother_src_type}")
            
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(param_str)}"
            
        except Exception:
            return f"{id(data)}_{self.src_type}_{self.trend_threshold}"

    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> EhlersMarketModeResult:
        """
        Ehlers Market Mode Indicatorを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                必要なカラム: high, low, close, open（カルマンフィルター用）
        
        Returns:
            EhlersMarketModeResult: Ehlers Market Modeの計算結果
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            
            if data_hash in self._result_cache:
                # キャッシュヒット
                if data_hash in self._cache_keys:
                    self._cache_keys.remove(data_hash)
                self._cache_keys.append(data_hash)
                cached_result = self._result_cache[data_hash]
                return EhlersMarketModeResult(
                    smooth=cached_result.smooth.copy(),
                    detrender=cached_result.detrender.copy(),
                    period=cached_result.period.copy(),
                    smooth_period=cached_result.smooth_period.copy(),
                    dc_phase=cached_result.dc_phase.copy(),
                    trend_line=cached_result.trend_line.copy(),
                    trend_mode=cached_result.trend_mode.copy(),
                    signal=cached_result.signal.copy(),
                    days_in_trend=cached_result.days_in_trend.copy(),
                    i_trend=cached_result.i_trend.copy(),
                    filtered_price=cached_result.filtered_price.copy(),
                    smoothed_values=cached_result.smoothed_values.copy(),
                    percentiles=cached_result.percentiles.copy() if cached_result.percentiles is not None else None,
                    trend_state=cached_result.trend_state.copy() if cached_result.trend_state is not None else None,
                    trend_intensity=cached_result.trend_intensity.copy() if cached_result.trend_intensity is not None else None
                )
            
            # データの準備と検証
            if isinstance(data, pd.DataFrame):
                required_cols = ['high', 'low', 'close']
                if self.use_kalman_filter:
                    required_cols.extend(['open'])  # カルマンフィルター用
                missing_cols = [col for col in required_cols if col not in data.columns]
                if missing_cols:
                    raise ValueError(f"必要なカラムが不足しています: {missing_cols}")
            else:
                if data.ndim != 2 or data.shape[1] < 4:
                    raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")
            
            # データ長の検証
            data_length = len(data)
            if data_length == 0:
                raise ValueError("入力データが空です")
            
            min_required_length = 100  # 複雑な計算のため多めに必要
            if data_length < min_required_length:
                self.logger.warning(f"データ長（{data_length}）が推奨される長さ（{min_required_length}）より短いです")
            
            # 1. ソース価格データを取得
            source_prices = PriceSource.calculate_source(data, self.src_type)
            
            # NumPy配列に変換
            if not isinstance(source_prices, np.ndarray):
                source_prices = np.array(source_prices)
            if source_prices.dtype != np.float64:
                source_prices = source_prices.astype(np.float64)
            
            # 2. カルマンフィルターによる価格データのフィルタリング（オプション）
            filtered_prices = source_prices
            if self.use_kalman_filter and self.kalman_filter is not None:
                try:
                    # ソース価格をDataFrame形式に変換してカルマンフィルターに入力
                    if isinstance(data, pd.DataFrame):
                        kalman_input = data.copy()
                        # カルマンフィルター用のソースタイプが既に存在する場合はそのまま使用
                        if self.src_type not in kalman_input.columns:
                            kalman_input[self.src_type] = source_prices
                    else:
                        # NumPy配列の場合はDataFrameに変換
                        if data.shape[1] >= 4:  # OHLC
                            kalman_input = pd.DataFrame({
                                'open': data[:, 0],
                                'high': data[:, 1],
                                'low': data[:, 2],
                                'close': data[:, 3]
                            })
                            # ソースタイプを追加
                            kalman_input[self.src_type] = source_prices
                        else:
                            # 最小限のDataFrame
                            kalman_input = pd.DataFrame({self.src_type: source_prices})
                    
                    kalman_result = self.kalman_filter.calculate(kalman_input)
                    # カルマンフィルターの結果が辞書形式の場合は適切に処理
                    if hasattr(kalman_result, 'values'):
                        filtered_prices = kalman_result.values
                    elif hasattr(kalman_result, 'filtered_values'):
                        filtered_prices = kalman_result.filtered_values
                    else:
                        # 結果がarray形式の場合
                        filtered_prices = np.array(kalman_result)
                    
                    # NumPy配列として確保
                    if not isinstance(filtered_prices, np.ndarray):
                        filtered_prices = np.array(filtered_prices)
                    if filtered_prices.dtype != np.float64:
                        filtered_prices = filtered_prices.astype(np.float64)
                    
                    self.logger.debug("カルマンフィルターによる価格データのフィルタリングを適用しました")
                        
                except Exception as e:
                    self.logger.warning(f"カルマンフィルター適用中にエラー: {e}。元の値を使用します。")
                    filtered_prices = source_prices
            
            # 3. 価格をスムージング（UnifiedSmootherを使用）
            if self.price_smoother is not None:
                try:
                    # DataFrameまたは適切な形式でスムーサーに渡す
                    if isinstance(data, pd.DataFrame):
                        price_data = data.copy()
                        if self.src_type not in price_data.columns:
                            price_data[self.src_type] = filtered_prices
                    else:
                        # NumPy配列の場合
                        price_data = pd.DataFrame({
                            'open': data[:, 0] if data.shape[1] >= 4 else filtered_prices,
                            'high': data[:, 1] if data.shape[1] >= 4 else filtered_prices,
                            'low': data[:, 2] if data.shape[1] >= 4 else filtered_prices,
                            'close': data[:, 3] if data.shape[1] >= 4 else filtered_prices
                        })
                        price_data[self.src_type] = filtered_prices
                    
                    smoother_result = self.price_smoother.calculate(price_data)
                    smooth = smoother_result.values
                    
                    # NumPy配列として確保
                    if not isinstance(smooth, np.ndarray):
                        smooth = np.array(smooth)
                    if smooth.dtype != np.float64:
                        smooth = smooth.astype(np.float64)
                        
                    self.logger.debug("UnifiedSmootherによる価格スムージングを適用しました")
                        
                except Exception as e:
                    self.logger.warning(f"価格スムージング中にエラー: {e}。元の値を使用します。")
                    smooth = filtered_prices.copy()
            else:
                # フォールバック: シンプルなスムージング
                smooth = filtered_prices.copy()
            
            # 4. 初期期間配列を作成（サイクル期間計算で必要）
            period = np.full(len(filtered_prices), 20.0, dtype=np.float64)
            
            # 5. デトレンダーを計算
            detrender = calculate_detrender_numba(smooth, period)
            
            # 6. サイクル期間を検出
            period, smooth_period = calculate_cycle_period_numba(detrender)
            
            # 7. DCフェーズを計算（従来の安定版を使用）
            dc_phase = calculate_dc_phase_simple_numba(smooth, smooth_period)
            
            # 8. トレンドラインを計算
            i_trend, trend_line = calculate_trend_line_numba(filtered_prices, smooth_period)
            
            # 9. マーケットモードを判定
            trend_mode, signal, days_in_trend = calculate_market_mode_numba(
                smooth, trend_line, dc_phase, smooth_period
            )
            
            # 10. 平滑化（オプション）
            smoothed_values = np.full_like(signal, np.nan)
            if self.use_smoothing and self.smoother is not None:
                try:
                    # シグナルをDataFrame形式に変換
                    # FRAMAなどのスムーサーがhigh/lowを必要とする場合に対応
                    signal_df = pd.DataFrame({
                        'close': signal,
                        'high': signal,  # シグナルをhighとしても使用
                        'low': signal,   # シグナルをlowとしても使用
                        'open': signal   # シグナルをopenとしても使用
                    })
                    
                    # 平滑化を適用
                    smoother_result = self.smoother.calculate(signal_df)
                    if hasattr(smoother_result, 'values'):
                        smoothed_values = smoother_result.values
                    else:
                        smoothed_values = np.array(smoother_result)
                except Exception as e:
                    self.logger.warning(f"平滑化処理中にエラー: {e}。生の値を使用します。")
                    smoothed_values = signal.copy()
            else:
                smoothed_values = signal.copy()
            
            # 11. パーセンタイルベーストレンド分析（オプション）
            percentiles, trend_state, trend_intensity = self._calculate_percentile_analysis(
                trend_mode, 'trend'
            )
            
            # 結果の作成
            result = EhlersMarketModeResult(
                smooth=smooth.copy(),
                detrender=detrender.copy(),
                period=period.copy(),
                smooth_period=smooth_period.copy(),
                dc_phase=dc_phase.copy(),
                trend_line=trend_line.copy(),
                trend_mode=trend_mode.copy(),
                signal=signal.copy(),
                days_in_trend=days_in_trend.copy(),
                i_trend=i_trend.copy(),
                filtered_price=filtered_prices.copy(),
                smoothed_values=smoothed_values.copy(),
                percentiles=percentiles.copy() if percentiles is not None else None,
                trend_state=trend_state.copy() if trend_state is not None else None,
                trend_intensity=trend_intensity.copy() if trend_intensity is not None else None
            )
            
            # キャッシュ管理
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            # 基底クラス用の値設定（トレンドモードをメイン値として使用）
            self._values = trend_mode
            
            return result
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"Ehlers Market Mode計算中にエラー: {error_msg}\\n{stack_trace}")
            
            # エラー時は空の結果を返す
            empty_array = np.array([])
            return EhlersMarketModeResult(
                smooth=empty_array,
                detrender=empty_array,
                period=empty_array,
                smooth_period=empty_array,
                dc_phase=empty_array,
                trend_line=empty_array,
                trend_mode=empty_array,
                signal=empty_array,
                days_in_trend=empty_array,
                i_trend=empty_array,
                filtered_price=empty_array,
                smoothed_values=empty_array,
                percentiles=None,
                trend_state=None,
                trend_intensity=None
            )
    
    def get_values(self) -> Optional[np.ndarray]:
        """トレンドモード値を取得（後方互換性のため）"""
        if not self._result_cache:
            return None
        
        result = self._get_latest_result()
        return result.trend_mode.copy() if result else None
    
    def get_smooth(self) -> Optional[np.ndarray]:
        """スムーズされた価格を取得"""
        result = self._get_latest_result()
        return result.smooth.copy() if result else None
    
    def get_detrender(self) -> Optional[np.ndarray]:
        """デトレンダー値を取得"""
        result = self._get_latest_result()
        return result.detrender.copy() if result else None
    
    def get_period(self) -> Optional[np.ndarray]:
        """測定されたサイクル期間を取得"""
        result = self._get_latest_result()
        return result.period.copy() if result else None
    
    def get_smooth_period(self) -> Optional[np.ndarray]:
        """スムーズされたサイクル期間を取得"""
        result = self._get_latest_result()
        return result.smooth_period.copy() if result else None
    
    def get_dc_phase(self) -> Optional[np.ndarray]:
        """DCフェーズを取得"""
        result = self._get_latest_result()
        return result.dc_phase.copy() if result else None
    
    def get_trend_line(self) -> Optional[np.ndarray]:
        """トレンドラインを取得"""
        result = self._get_latest_result()
        return result.trend_line.copy() if result else None
    
    def get_trend_mode(self) -> Optional[np.ndarray]:
        """トレンドモードを取得"""
        result = self._get_latest_result()
        return result.trend_mode.copy() if result else None
    
    def get_signal(self) -> Optional[np.ndarray]:
        """売買信号を取得"""
        result = self._get_latest_result()
        return result.signal.copy() if result else None
    
    def get_days_in_trend(self) -> Optional[np.ndarray]:
        """トレンド継続日数を取得"""
        result = self._get_latest_result()
        return result.days_in_trend.copy() if result else None
    
    def get_i_trend(self) -> Optional[np.ndarray]:
        """iTrend値を取得"""
        result = self._get_latest_result()
        return result.i_trend.copy() if result else None
    
    def get_percentiles(self) -> Optional[np.ndarray]:
        """パーセンタイル値を取得"""
        result = self._get_latest_result()
        return result.percentiles.copy() if result and result.percentiles is not None else None
    
    def get_trend_state(self) -> Optional[np.ndarray]:
        """トレンド状態を取得"""
        result = self._get_latest_result()
        return result.trend_state.copy() if result and result.trend_state is not None else None
    
    def get_trend_intensity(self) -> Optional[np.ndarray]:
        """トレンド強度を取得"""
        result = self._get_latest_result()
        return result.trend_intensity.copy() if result and result.trend_intensity is not None else None
    
    def get_indicator_info(self) -> Dict[str, Any]:
        """インジケーター情報を取得"""
        return {
            'name': self.name,
            'src_type': self.src_type,
            'trend_threshold': self.trend_threshold,
            'use_smoothing': self.use_smoothing,
            'smoother_type': self.smoother_type if self.use_smoothing else None,
            'smoother_period': self.smoother_period if self.use_smoothing else None,
            'use_dynamic_period': self.use_dynamic_period,
            'detector_type': self.detector_type if self.use_dynamic_period else None,
            'use_kalman_filter': self.use_kalman_filter,
            'kalman_filter_type': self.kalman_filter_type if self.use_kalman_filter else None,
            'enable_percentile_analysis': self.enable_percentile_analysis,
            'percentile_lookback_period': self.percentile_lookback_period if self.enable_percentile_analysis else None,
            'percentile_low_threshold': self.percentile_low_threshold if self.enable_percentile_analysis else None,
            'percentile_high_threshold': self.percentile_high_threshold if self.enable_percentile_analysis else None,
            'description': 'Ehlers Market Mode Indicator - サイクル分析とフェーズ分析による市場モード検出（トレンド vs サイクル）'
        }
    
    def _get_latest_result(self) -> Optional[EhlersMarketModeResult]:
        """最新の結果を取得"""
        if not self._result_cache:
            return None
        
        if self._cache_keys:
            return self._result_cache[self._cache_keys[-1]]
        else:
            return next(iter(self._result_cache.values()))
    
    def reset(self) -> None:
        """インディケーターの状態をリセット"""
        super().reset()
        if self.smoother:
            self.smoother.reset()
        if self.price_smoother:
            self.price_smoother.reset()
        if self.cycle_detector:
            self.cycle_detector.reset()
        if self.kalman_filter:
            self.kalman_filter.reset()
        self._result_cache = {}
        self._cache_keys = []


# 便利関数
def calculate_ehlers_market_mode(
    data: Union[pd.DataFrame, np.ndarray],
    src_type: str = 'hl2',
    trend_threshold: float = 0.015,
    use_smoothing: bool = False,
    smoother_type: str = 'super_smoother',
    use_dynamic_period: bool = False,
    use_kalman_filter: bool = False,
    enable_percentile_analysis: bool = True,
    percentile_lookback_period: int = 50,
    percentile_low_threshold: float = 0.25,
    percentile_high_threshold: float = 0.75,
    **kwargs
) -> np.ndarray:
    """
    Ehlers Market Modeの計算（便利関数）
    
    Args:
        data: 価格データ
        src_type: ソースタイプ
        trend_threshold: トレンド判定閾値
        use_smoothing: 平滑化を使用するか
        smoother_type: スムーサータイプ
        use_dynamic_period: 動的期間適応を使用するか
        use_kalman_filter: カルマンフィルターを使用するか
        enable_percentile_analysis: パーセンタイル分析を有効にするか
        percentile_lookback_period: パーセンタイル分析のルックバック期間
        percentile_low_threshold: パーセンタイル分析の低閾値
        percentile_high_threshold: パーセンタイル分析の高閾値
        **kwargs: その他のパラメータ
        
    Returns:
        トレンドモード値
    """
    indicator = EhlersMarketMode(
        src_type=src_type,
        trend_threshold=trend_threshold,
        use_smoothing=use_smoothing,
        smoother_type=smoother_type,
        use_dynamic_period=use_dynamic_period,
        use_kalman_filter=use_kalman_filter,
        enable_percentile_analysis=enable_percentile_analysis,
        percentile_lookback_period=percentile_lookback_period,
        percentile_low_threshold=percentile_low_threshold,
        percentile_high_threshold=percentile_high_threshold,
        **kwargs
    )
    result = indicator.calculate(data)
    return result.trend_mode


if __name__ == "__main__":
    """直接実行時のテスト"""
    import numpy as np
    import pandas as pd
    
    print("=== Ehlers Market Mode インジケーターのテスト ===")
    
    # テストデータ生成
    np.random.seed(42)
    length = 300
    base_price = 100.0
    
    # Pine Scriptのテストに近いデータを生成
    prices = [base_price]
    for i in range(1, length):
        if i < 75:  # トレンド相場
            trend_component = 0.003  # 上昇トレンド
            cycle_component = 0.015 * np.sin(2 * np.pi * i / 20)  # 20期間サイクル
            noise = np.random.normal(0, 0.005)
        elif i < 150:  # サイクル相場
            trend_component = 0.0  # トレンドなし
            cycle_component = 0.025 * np.sin(2 * np.pi * i / 25)  # 25期間サイクル
            noise = np.random.normal(0, 0.008)
        elif i < 225:  # 下降トレンド相場
            trend_component = -0.002  # 下降トレンド
            cycle_component = 0.01 * np.sin(2 * np.pi * i / 18)  # 18期間サイクル
            noise = np.random.normal(0, 0.006)
        else:  # 複合相場
            trend_component = 0.004  # 強い上昇トレンド
            cycle_component = 0.02 * np.sin(2 * np.pi * i / 22)  # 22期間サイクル
            noise = np.random.normal(0, 0.007)
        
        change = trend_component + cycle_component + noise
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # OHLC データの生成
    data = []
    for i, close in enumerate(prices):
        daily_range = abs(np.random.normal(0, close * 0.008))
        
        high = close + daily_range * np.random.uniform(0.4, 1.0)
        low = close - daily_range * np.random.uniform(0.4, 1.0)
        
        if i == 0:
            open_price = close
        else:
            gap = np.random.normal(0, close * 0.003)
            open_price = prices[i-1] + gap
        
        # 論理的整合性の確保
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.uniform(1000, 10000)
        })
    
    df = pd.DataFrame(data)
    
    print(f"テストデータ: {len(df)}ポイント")
    print(f"価格範囲: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    # Ehlers Market Modeを計算（基本版）
    print("\\n基本版Ehlers Market Modeをテスト中...")
    emm = EhlersMarketMode(use_kalman_filter=False)
    result = emm.calculate(df)
    
    valid_trend_count = np.sum(~np.isnan(result.trend_mode))
    valid_signal_count = np.sum(~np.isnan(result.signal))
    mean_period = np.nanmean(result.period)
    
    # モード分析
    trend_mode_ratio = np.sum(result.trend_mode[~np.isnan(result.trend_mode)] == 1.0) / valid_trend_count if valid_trend_count > 0 else 0
    cycle_mode_ratio = np.sum(result.trend_mode[~np.isnan(result.trend_mode)] == 0.0) / valid_trend_count if valid_trend_count > 0 else 0
    
    # 信号分析
    buy_signals = np.sum(result.signal[~np.isnan(result.signal)] == 1.0)
    sell_signals = np.sum(result.signal[~np.isnan(result.signal)] == -1.0)
    neutral_signals = np.sum(result.signal[~np.isnan(result.signal)] == 0.0)
    
    print(f"  有効トレンドモード値数: {valid_trend_count}/{len(df)}")
    print(f"  有効信号数: {valid_signal_count}/{len(df)}")
    print(f"  平均サイクル期間: {mean_period:.1f}")
    print(f"  トレンドモード比率: {trend_mode_ratio:.2%}")
    print(f"  サイクルモード比率: {cycle_mode_ratio:.2%}")
    print(f"  買いシグナル数: {buy_signals}")
    print(f"  売りシグナル数: {sell_signals}")
    print(f"  中立シグナル数: {neutral_signals}")
    
    # カルマンフィルター版をテスト
    print("\\nカルマンフィルター版Ehlers Market Modeをテスト中...")
    emm_kalman = EhlersMarketMode(
        use_kalman_filter=True,
        kalman_filter_type='unscented',
        use_smoothing=True,
        smoother_type='frama'
    )
    result_kalman = emm_kalman.calculate(df)
    
    valid_trend_kalman = np.sum(~np.isnan(result_kalman.trend_mode))
    mean_period_kalman = np.nanmean(result_kalman.period)
    
    print(f"  有効トレンドモード値数: {valid_trend_kalman}/{len(df)}")
    print(f"  平均サイクル期間（カルマン版）: {mean_period_kalman:.1f}")
    
    # 比較統計
    if valid_trend_count > 50 and valid_trend_kalman > 50:
        # 両方の結果が有効な範囲で相関を計算
        min_valid = min(valid_trend_count, valid_trend_kalman)
        basic_trend = result.trend_mode[~np.isnan(result.trend_mode)][-min_valid:]
        kalman_trend = result_kalman.trend_mode[~np.isnan(result_kalman.trend_mode)][-min_valid:]
        
        if len(basic_trend) > 10 and len(kalman_trend) > 10:
            correlation = np.corrcoef(basic_trend, kalman_trend)[0, 1]
            print(f"  基本版とカルマン版の相関: {correlation:.4f}")
    
    print("\\n=== テスト完了 ===")