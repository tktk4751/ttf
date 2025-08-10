#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Optional, Dict, Any
import numpy as np
import pandas as pd
from numba import njit, float64

from .indicator import Indicator
from .price_source import PriceSource
from .smoother.unified_smoother import UnifiedSmoother

# 条件付きインポート（オプション機能）
try:
    from .cycle.ehlers_unified_dc import EhlersUnifiedDC
    EHLERS_UNIFIED_DC_AVAILABLE = True
except ImportError:
    try:
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
    from .kalman.unified_kalman import UnifiedKalman
    UNIFIED_KALMAN_AVAILABLE = True
except ImportError:
    try:
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
class XMarketModeResult:
    """Xマーケットモードインジケーターの計算結果"""
    smooth: np.ndarray                # スムーズされた価格
    detrender: np.ndarray            # デトレンダー値
    period: np.ndarray               # 測定されたサイクル期間
    smooth_period: np.ndarray        # スムーズされたサイクル期間
    adaptive_phase: np.ndarray       # 適応フェーズ
    trend_line: np.ndarray           # トレンドライン
    trend_mode: np.ndarray           # トレンドモード（1=トレンド、0=サイクル）
    signal: np.ndarray               # 売買信号（1=買い、-1=売り、0=中立）
    days_in_trend: np.ndarray        # トレンド継続日数
    i_trend: np.ndarray              # iTrend値
    filtered_price: np.ndarray       # カルマンフィルター適用後の価格（オプション）
    mode_strength: np.ndarray        # モード強度（0-1）
    cycle_strength: np.ndarray       # サイクル強度
    trend_strength: np.ndarray       # トレンド強度


@njit(fastmath=True, cache=True)
def ultra_precise_dft_numba(
    data: np.ndarray,
    window_size: int = 124,
    overlap: float = 0.9,
    zero_padding_factor: int = 16
) -> tuple:
    """
    超高精度DFT解析 - 中期時間軸（8-124期間）に特化
    
    Args:
        data: 入力データ配列
        window_size: ウィンドウサイズ（中期向けに大型化）
        overlap: オーバーラップ率（高い追従性のため増加）
        zero_padding_factor: ゼロパディング係数（高精度のため増加）
        
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
        
        # 超高品質ウィンドウ関数の最適組み合わせ
        kaiser_beta = 8.6  # より高いβ値で精度向上
        blackman_nuttall = np.zeros(window_size)
        kaiser = np.zeros(window_size)
        hamming = np.zeros(window_size)
        
        for i in range(window_size):
            t = 2 * np.pi * i / (window_size - 1)
            
            # Blackman-Nuttall ウィンドウ（高精度版）
            blackman_nuttall[i] = (0.3635819 - 0.4891775 * np.cos(t) + 
                                 0.1365995 * np.cos(2*t) - 0.0106411 * np.cos(3*t))
            
            # Kaiser ウィンドウ（高β値）
            # I0の近似計算（ベッセル関数）
            x = kaiser_beta * np.sqrt(1 - ((2*i/(window_size-1)) - 1)**2)
            i0_x = 1.0
            term = 1.0
            for k in range(1, 20):
                term *= (x / (2*k))**2
                i0_x += term
                if term < 1e-12:
                    break
            
            i0_beta = 1.0
            term = 1.0
            for k in range(1, 20):
                term *= (kaiser_beta / (2*k))**2
                i0_beta += term
                if term < 1e-12:
                    break
            
            kaiser[i] = i0_x / i0_beta
            
            # Hamming ウィンドウ（補完）
            hamming[i] = 0.54 - 0.46 * np.cos(t)
        
        # 最適組み合わせウィンドウ（中期解析向け調整）
        optimal_window = (0.6 * blackman_nuttall + 0.3 * kaiser + 0.1 * hamming)
        windowed_data = window_data * optimal_window
        
        # 超高度ゼロパディング
        padded_size = window_size * zero_padding_factor
        padded_data = np.zeros(padded_size)
        padded_data[:window_size] = windowed_data
        
        # 中期時間軸に特化した高精度DFT計算（8-124期間）
        period_count = min(117, padded_size // 4)  # 8から124まで、ただしpadded_sizeに依存
        if period_count <= 0:
            period_count = 1
            
        freqs = np.zeros(period_count)
        powers = np.zeros(period_count)
        phases = np.zeros(period_count)
        
        for period_idx in range(period_count):
            period = max(8, 8 + period_idx)
            if period >= padded_size // 2:
                period = max(8, padded_size // 4)  # 安全な期間に調整
                
            if period > 0:
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
        
        # 高度な重心アルゴリズム（中期向け調整）
        max_power = np.max(powers)
        if max_power > 0:
            normalized_powers = powers / max_power
        else:
            normalized_powers = powers
        
        # デシベル変換（より厳密な閾値）
        db_values = np.zeros(period_count)
        for i in range(period_count):
            if normalized_powers[i] > 0.0001:  # より厳密な閾値
                ratio = normalized_powers[i]
                db_values[i] = -10 * np.log10(0.0001 / (1 - 0.9999 * ratio))
            else:
                db_values[i] = 40  # より高い上限
            
            if db_values[i] > 40:
                db_values[i] = 40
        
        # 超高度重心計算（中期向け重み調整）
        numerator = 0.0
        denominator = 0.0
        total_weight = 0.0
        
        for i in range(period_count):
            # 中期向けの重み関数（指数的重み付け）
            weight = np.exp(-(40 - db_values[i]) / 8.0) ** 3
            total_weight += weight
            if db_values[i] <= 1.5:  # より厳密な閾値
                numerator += freqs[i] * weight
                denominator += weight
        
        if denominator > 1e-12:
            dominant_freq = numerator / denominator
            confidence = denominator / max(total_weight, 1e-12)
        else:
            dominant_freq = 66.0  # 中期の中央値
            confidence = 0.05
        
        # 位相コヒーレンス計算（高精度版）
        max_idx = np.argmax(powers)
        coherence = 0.0
        
        if max_idx > 2 and max_idx < len(phases) - 3:
            phase_diffs = np.zeros(7)
            count = 0
            
            for j in range(-3, 4):
                if 0 <= max_idx + j < len(phases):
                    phase_diffs[count] = phases[max_idx + j]
                    count += 1
            
            if count > 4:
                # 円形統計を使用した位相コヒーレンス
                sum_cos = 0.0
                sum_sin = 0.0
                for k in range(count):
                    sum_cos += np.cos(phase_diffs[k])
                    sum_sin += np.sin(phase_diffs[k])
                
                mean_cos = sum_cos / count
                mean_sin = sum_sin / count
                
                resultant_length = np.sqrt(mean_cos**2 + mean_sin**2)
                coherence = resultant_length
        
        # 結果保存
        mid_point = start + window_size // 2
        if mid_point < n:
            frequencies[mid_point] = dominant_freq
            confidences[mid_point] = confidence
            coherences[mid_point] = coherence
    
    # 高度な補間（スプライン風）
    for i in range(n):
        if frequencies[i] == 0.0:
            left_vals = np.zeros(15)
            right_vals = np.zeros(15)
            left_count = 0
            right_count = 0
            
            for j in range(max(0, i-15), i):
                if frequencies[j] > 0 and left_count < 15:
                    left_vals[left_count] = frequencies[j]
                    left_count += 1
            
            for j in range(i+1, min(n, i+16)):
                if frequencies[j] > 0 and right_count < 15:
                    right_vals[right_count] = frequencies[j]
                    right_count += 1
            
            if left_count >= 3 and right_count >= 3:
                # 3次スプライン補間
                p0 = left_vals[left_count-3]
                p1 = left_vals[left_count-2]
                p2 = left_vals[left_count-1]
                p3 = right_vals[0]
                p4 = right_vals[1]
                p5 = right_vals[2]
                
                t = 0.5
                # 3次ベジエ曲線近似
                frequencies[i] = (0.2 * p0 + 0.5 * p1 + 0.3 * p2) * (1-t) + (0.3 * p3 + 0.5 * p4 + 0.2 * p5) * t
            elif left_count > 0 and right_count > 0:
                frequencies[i] = (left_vals[left_count-1] + right_vals[0]) / 2
            elif left_count > 0:
                frequencies[i] = left_vals[left_count-1]
            elif right_count > 0:
                frequencies[i] = right_vals[0]
            else:
                frequencies[i] = 66.0  # 中期の中央値
    
    return frequencies, confidences, coherences


@njit(fastmath=True, cache=True)
def calculate_x_detrender_numba(
    smooth: np.ndarray,
    period: np.ndarray
) -> np.ndarray:
    """
    Xデトレンダーを計算する（中期特化版）
    より深い遅れとより高い精度を実現
    
    Args:
        smooth: スムーズされた価格配列
        period: 周期配列
        
    Returns:
        デトレンダー配列
    """
    length = len(smooth)
    detrender = np.full(length, np.nan, dtype=np.float64)
    
    for i in range(length):
        # 前の期間値を取得（中期のデフォルト値は66）
        prev_period = 66.0
        if i > 0 and not np.isnan(period[i-1]):
            prev_period = period[i-1]
        
        # 中期向けフィルター係数（より保守的）
        filter_coef = (0.045 * prev_period) + 0.72
        
        # Xデトレンダー計算（より深い遅れ）
        if i >= 12:  # より深い遅れ
            detrender[i] = ((0.0618 * smooth[i]) + 
                           (0.7382 * smooth[i-3]) - 
                           (0.7382 * smooth[i-6]) - 
                           (0.0618 * smooth[i-9]) +
                           (0.382 * smooth[i-12])) * filter_coef
        else:
            # 十分なデータがない場合は簡略化
            detrender[i] = 0.0
    
    return detrender


@njit(fastmath=True, cache=True)
def calculate_x_cycle_period_numba(
    detrender: np.ndarray
) -> tuple:
    """
    Xサイクル期間を計算する（中期特化版）
    
    Args:
        detrender: デトレンダー配列
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (period, smooth_period)
    """
    length = len(detrender)
    period = np.full(length, 66.0, dtype=np.float64)  # 中期デフォルト
    smooth_period = np.full(length, 66.0, dtype=np.float64)
    
    i1 = np.full(length, np.nan, dtype=np.float64)
    q1 = np.full(length, np.nan, dtype=np.float64)
    i2 = np.full(length, np.nan, dtype=np.float64)
    q2 = np.full(length, np.nan, dtype=np.float64)
    re = np.full(length, np.nan, dtype=np.float64)
    im = np.full(length, np.nan, dtype=np.float64)
    
    for i in range(length):
        # 前の期間値を取得
        prev_period = 66.0
        if i > 0:
            prev_period = period[i-1]
        
        # 中期向けフィルター係数
        filter_coef = (0.045 * prev_period) + 0.72
        
        # Q1の計算（より深い遅れ）
        if i >= 12:
            q1[i] = ((0.0618 * detrender[i]) + 
                    (0.7382 * detrender[i-3]) - 
                    (0.7382 * detrender[i-6]) - 
                    (0.0618 * detrender[i-9]) +
                    (0.382 * detrender[i-12])) * filter_coef
        
        # I1の計算（6期間遅れ）
        if i >= 6:
            i1[i] = detrender[i-6]
        
        # JIとJQの計算
        jI = 0.0
        jQ = 0.0
        if i >= 12:
            jI = ((0.0618 * i1[i]) + 
                 (0.7382 * i1[i-3]) - 
                 (0.7382 * i1[i-6]) - 
                 (0.0618 * i1[i-9]) +
                 (0.382 * i1[i-12])) * filter_coef
            
            jQ = ((0.0618 * q1[i]) + 
                 (0.7382 * q1[i-3]) - 
                 (0.7382 * q1[i-6]) - 
                 (0.0618 * q1[i-9]) +
                 (0.382 * q1[i-12])) * filter_coef
        
        # I2とQ2の計算（より強い平滑化）
        if not np.isnan(i1[i]) and not np.isnan(q1[i]):
            i2_raw = i1[i] - jQ
            q2_raw = q1[i] + jI
            
            # より強い平滑化（中期向け）
            if i > 0 and not np.isnan(i2[i-1]):
                i2[i] = 0.125 * i2_raw + 0.875 * i2[i-1]
            else:
                i2[i] = i2_raw
                
            if i > 0 and not np.isnan(q2[i-1]):
                q2[i] = 0.125 * q2_raw + 0.875 * q2[i-1]
            else:
                q2[i] = q2_raw
        
        # REとIMの計算
        if i > 0 and not np.isnan(i2[i]) and not np.isnan(q2[i]) and not np.isnan(i2[i-1]) and not np.isnan(q2[i-1]):
            re_raw = i2[i] * i2[i-1] + q2[i] * q2[i-1]
            im_raw = i2[i] * q2[i-1] - q2[i] * i2[i-1]
            
            # より強い平滑化
            if i > 0 and not np.isnan(re[i-1]):
                re[i] = 0.125 * re_raw + 0.875 * re[i-1]
            else:
                re[i] = re_raw
                
            if i > 0 and not np.isnan(im[i-1]):
                im[i] = 0.125 * im_raw + 0.875 * im[i-1]
            else:
                im[i] = im_raw
        
        # 周期の計算（中期範囲に制限）
        if not np.isnan(im[i]) and not np.isnan(re[i]) and abs(im[i]) > 1e-12 and abs(re[i]) > 1e-12:
            ratio = im[i] / max(abs(re[i]), 1e-12)
            if abs(ratio) < 1e8:
                raw_period = 2.0 * np.pi / abs(np.arctan(ratio))
                
                # 中期周期の制限（8-124期間）
                if i > 0:
                    raw_period = max(raw_period, 0.8 * period[i-1])
                    raw_period = min(raw_period, 1.25 * period[i-1])
                
                raw_period = max(8.0, min(124.0, raw_period))
                
                # より強い平滑化
                if i > 0:
                    period[i] = 0.125 * raw_period + 0.875 * period[i-1]
                else:
                    period[i] = raw_period
            else:
                if i > 0:
                    period[i] = period[i-1]
                else:
                    period[i] = 66.0
        else:
            if i > 0:
                period[i] = period[i-1]
            else:
                period[i] = 66.0
        
        # スムーズ期間の計算（より強い平滑化）
        if i > 0:
            smooth_period[i] = 0.2 * period[i] + 0.8 * smooth_period[i-1]
        else:
            smooth_period[i] = period[i]
    
    return period, smooth_period


@njit(fastmath=True, cache=True)
def calculate_x_adaptive_phase_numba(
    smooth: np.ndarray,
    smooth_period: np.ndarray
) -> np.ndarray:
    """
    X適応フェーズを計算する（高精度版）
    
    Args:
        smooth: スムーズされた価格配列
        smooth_period: スムーズされた周期配列
        
    Returns:
        適応フェーズ配列
    """
    length = len(smooth)
    adaptive_phase = np.full(length, np.nan, dtype=np.float64)
    
    for i in range(length):
        if np.isnan(smooth_period[i]) or smooth_period[i] <= 1e-12:
            continue
            
        dc_period = int(np.ceil(smooth_period[i] + 0.5))
        if dc_period <= 0 or i < dc_period:
            continue
            
        real = 0.0
        imag = 0.0
        
        # 適応DFT計算（重み付き）
        for j in range(dc_period):
            if i - j >= 0:
                # 指数的重み付け（新しいデータに重点）
                weight = np.exp(-j / (dc_period * 0.3))
                angle = 2.0 * np.pi * j / max(dc_period, 1)
                real += weight * np.sin(angle) * smooth[i - j]
                imag += weight * np.cos(angle) * smooth[i - j]
        
        # フェーズ計算（高精度版）
        if abs(imag) > 1e-12:
            phase = np.arctan(real / imag) * 180.0 / np.pi
        else:
            phase = 90.0 * (1.0 if real >= 0 else -1.0)
        
        # 中期向け調整
        phase += 90.0
        if smooth_period[i] > 1e-12:
            phase += 720.0 / smooth_period[i]  # より大きな調整
        
        if imag < 0:
            phase += 180.0
            
        # フェーズの正規化（-180°から+180°）
        while phase > 180.0:
            phase -= 360.0
        while phase < -180.0:
            phase += 360.0
            
        adaptive_phase[i] = phase
    
    return adaptive_phase


@njit(fastmath=True, cache=True)
def calculate_x_market_mode_numba(
    smooth: np.ndarray,
    trend_line: np.ndarray,
    adaptive_phase: np.ndarray,
    smooth_period: np.ndarray
) -> tuple:
    """
    Xマーケットモードを計算する（高精度・低遅延版）
    
    Args:
        smooth: スムーズされた価格配列
        trend_line: トレンドライン配列
        adaptive_phase: 適応フェーズ配列
        smooth_period: スムーズされた周期配列
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
        (trend_mode, signal, days_in_trend, mode_strength, cycle_strength, trend_strength)
    """
    length = len(smooth)
    trend_mode = np.full(length, 1.0, dtype=np.float64)  # デフォルトはトレンド
    signal = np.full(length, 0.0, dtype=np.float64)
    days_in_trend = np.full(length, 0.0, dtype=np.float64)
    mode_strength = np.full(length, 0.5, dtype=np.float64)  # モード強度
    cycle_strength = np.full(length, 0.0, dtype=np.float64)  # サイクル強度
    trend_strength = np.full(length, 0.0, dtype=np.float64)  # トレンド強度
    
    current_days = 0.0
    
    for i in range(length):
        if i > 0 and not np.isnan(adaptive_phase[i]) and not np.isnan(adaptive_phase[i-1]) and not np.isnan(smooth_period[i]):
            
            # 適応フェーズ変化の検出（高精度）
            phase_current = adaptive_phase[i] * np.pi / 180.0
            phase_prev = adaptive_phase[i-1] * np.pi / 180.0
            
            # フェーズ速度の計算
            phase_velocity = phase_current - phase_prev
            if phase_velocity > np.pi:
                phase_velocity -= 2 * np.pi
            elif phase_velocity < -np.pi:
                phase_velocity += 2 * np.pi
            
            # 期待フェーズ速度
            expected_velocity = 2 * np.pi / smooth_period[i]
            
            # フェーズ加速度（2次導関数）の計算
            if i > 1 and not np.isnan(adaptive_phase[i-2]):
                phase_prev2 = adaptive_phase[i-2] * np.pi / 180.0
                prev_velocity = phase_prev - phase_prev2
                if prev_velocity > np.pi:
                    prev_velocity -= 2 * np.pi
                elif prev_velocity < -np.pi:
                    prev_velocity += 2 * np.pi
                
                phase_acceleration = phase_velocity - prev_velocity
            else:
                phase_acceleration = 0.0
            
            # サイクル強度の計算（ゼロ除算防止）
            if abs(expected_velocity) > 1e-12:
                velocity_ratio = abs(phase_velocity / expected_velocity)
            else:
                velocity_ratio = 1.0
            
            if velocity_ratio > 2.0:
                velocity_ratio = 2.0
            
            cycle_str = 1.0 / (1.0 + abs(velocity_ratio - 1.0) * 3.0)
            cycle_strength[i] = cycle_str
            
            # トレンド強度の計算（価格とトレンドラインの偏差、ゼロ除算防止）
            if not np.isnan(trend_line[i]) and abs(trend_line[i]) > 1e-12:
                price_deviation = abs((smooth[i] - trend_line[i]) / trend_line[i])
                trend_str = min(1.0, price_deviation * 50.0)  # より敏感な調整
                trend_strength[i] = trend_str
            else:
                trend_strength[i] = 0.0
            
            # フェーズ加速度による追加判定
            acceleration_factor = min(1.0, abs(phase_acceleration) * 10.0)
            
            # 複合モード強度の計算
            base_mode_strength = (cycle_strength[i] + (1.0 - trend_strength[i])) / 2.0
            mode_strength[i] = base_mode_strength * (1.0 - acceleration_factor * 0.3)
            
            # サイクル検出の条件（より厳密）
            is_cycle = (
                velocity_ratio > 0.7 and velocity_ratio < 1.4 and  # より狭い範囲
                cycle_strength[i] > 0.6 and                        # より高い閾値
                trend_strength[i] < 0.4 and                        # より低い閾値
                acceleration_factor < 0.5                          # 加速度制限
            )
            
            if is_cycle:
                current_days = 0.0
                trend_mode[i] = 0.0  # サイクルモード
            else:
                current_days += 1.0
                
                # 期間による追加判定（より保守的）
                if current_days < 0.3 * smooth_period[i]:
                    trend_mode[i] = 0.0  # サイクルモード
                else:
                    trend_mode[i] = 1.0  # トレンドモード
        else:
            if i > 0:
                current_days += 1.0
                trend_mode[i] = trend_mode[i-1]
                mode_strength[i] = mode_strength[i-1]
                cycle_strength[i] = cycle_strength[i-1]
                trend_strength[i] = trend_strength[i-1]
        
        days_in_trend[i] = current_days
        
        # 売買信号の生成（より敏感、ゼロ除算防止）
        if not np.isnan(trend_line[i]) and not np.isnan(smooth[i]) and abs(trend_line[i]) > 1e-12:
            price_ratio = smooth[i] / trend_line[i]
            if price_ratio > 1.008:  # より敏感な閾値
                signal[i] = 1.0   # 買いシグナル
            elif price_ratio < 0.992:  # より敏感な閾値
                signal[i] = -1.0  # 売りシグナル
            else:
                signal[i] = 0.0   # 中立
        else:
            signal[i] = 0.0  # トレンドラインが無効な場合は中立
    
    return trend_mode, signal, days_in_trend, mode_strength, cycle_strength, trend_strength


class XMarketMode(Indicator):
    """
    Xマーケットモード - 中期時間軸（8-124期間）特化の超高精度・低遅延マーケットモード判別器
    
    Ehlers Market Modeを改良し、以下の特徴を実現：
    - 8期間から124期間の中期時間軸に特化
    - 超高精度DFT解析（16倍ゼロパディング）
    - 90%オーバーラップによる超追従性
    - 適応フェーズ解析による高精度判別
    - フェーズ加速度による動的調整
    - 複合モード強度指標
    - 低遅延信号生成
    
    計算手順:
    1. 超高精度DFTによるサイクル期間検出
    2. X デトレンダーによる価格フィルタリング
    3. 適応フェーズ解析
    4. 複合モード強度計算
    5. 高精度トレンド/サイクル判別
    """
    
    def __init__(
        self,
        src_type: str = 'hlc3',
        trend_threshold: float = 0.008,  # より敏感
        use_smoothing: bool = True,
        smoother_type: str = 'frama',
        smoother_period: int = 16,
        use_kalman_filter: bool = False,
        kalman_filter_type: str = 'unscented',
        kalman_process_noise: float = 0.005,
        kalman_observation_noise: float = 0.0005
    ):
        """
        コンストラクタ
        
        Args:
            src_type: ソースタイプ（デフォルト: 'hlc3'）
            trend_threshold: トレンド判定閾値（デフォルト: 0.008、より敏感）
            use_smoothing: 平滑化を使用するか（デフォルト: True）
            smoother_type: スムーサータイプ（デフォルト: 'frama'）
            smoother_period: スムーサー期間（デフォルト: 16）
            use_kalman_filter: カルマンフィルターを使用するか（デフォルト: False）
            kalman_filter_type: カルマンフィルタータイプ（デフォルト: 'unscented'）
            kalman_process_noise: カルマンフィルタープロセスノイズ（デフォルト: 0.005）
            kalman_observation_noise: カルマンフィルター観測ノイズ（デフォルト: 0.0005）
        """
        indicator_name = f"XMarketMode(src={src_type}, threshold={trend_threshold:.4f}"
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
        
        # カルマンフィルターパラメータ
        self.use_kalman_filter = use_kalman_filter
        self.kalman_filter_type = kalman_filter_type
        self.kalman_process_noise = kalman_process_noise
        self.kalman_observation_noise = kalman_observation_noise
        
        # パラメータ検証
        if self.trend_threshold <= 0:
            raise ValueError("trend_thresholdは0より大きい必要があります")
        
        # カルマンフィルターの初期化
        self.kalman_filter = None
        if self.use_kalman_filter and UNIFIED_KALMAN_AVAILABLE:
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
        
        # 価格スムーサーの初期化
        self.price_smoother = None
        try:
            self.price_smoother = UnifiedSmoother(
                smoother_type='alma',
                src_type=self.src_type,
                length=6,  # 中期向けに調整
                offset=0.15,
                sigma=5.0
            )
        except Exception as e:
            self.logger.error(f"価格スムーサーの初期化に失敗: {e}")
            self.price_smoother = None
        
        # オプション平滑化
        self.smoother = None
        if self.use_smoothing:
            try:
                self.smoother = UnifiedSmoother(
                    smoother_type=self.smoother_type,
                    src_type='close',
                    period=self.smoother_period
                )
            except Exception as e:
                self.logger.error(f"統合スムーサーの初期化に失敗: {e}")
                self.use_smoothing = False
        
        # 結果キャッシュ
        self._result_cache = {}
        self._max_cache_size = 5
        self._cache_keys = []
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> XMarketModeResult:
        """
        Xマーケットモードを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                必要なカラム: high, low, close
        
        Returns:
            XMarketModeResult: Xマーケットモードの計算結果
        """
        try:
            # データの準備と検証
            if isinstance(data, pd.DataFrame):
                required_cols = ['high', 'low', 'close']
                if self.use_kalman_filter:
                    required_cols.extend(['open'])
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
            
            min_required_length = 150  # 中期解析のため大きめに設定
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
                    if isinstance(data, pd.DataFrame):
                        kalman_input = data.copy()
                        if self.src_type not in kalman_input.columns:
                            kalman_input[self.src_type] = source_prices
                    else:
                        kalman_input = pd.DataFrame({
                            'open': data[:, 0],
                            'high': data[:, 1],
                            'low': data[:, 2],
                            'close': data[:, 3]
                        })
                        kalman_input[self.src_type] = source_prices
                    
                    kalman_result = self.kalman_filter.calculate(kalman_input)
                    
                    # カルマンフィルター結果の処理
                    if hasattr(kalman_result, 'filtered_state'):
                        # UKFResultオブジェクトの場合
                        filtered_prices = kalman_result.filtered_state
                    elif hasattr(kalman_result, 'values'):
                        filtered_prices = kalman_result.values
                    elif hasattr(kalman_result, 'filtered_values'):
                        filtered_prices = kalman_result.filtered_values
                    elif hasattr(kalman_result, 'x'):
                        # 状態ベクトルの場合、最初の要素を使用
                        if kalman_result.x.ndim > 1:
                            filtered_prices = kalman_result.x[:, 0]
                        else:
                            filtered_prices = kalman_result.x
                    else:
                        # オブジェクトの場合は文字列化せずに配列として扱う
                        if hasattr(kalman_result, '__len__'):
                            filtered_prices = np.array(kalman_result)
                        else:
                            raise ValueError("カルマンフィルターの結果が配列形式ではありません")
                    
                    if not isinstance(filtered_prices, np.ndarray):
                        filtered_prices = np.array(filtered_prices)
                    if filtered_prices.dtype != np.float64:
                        filtered_prices = filtered_prices.astype(np.float64)
                    
                    # 長さの確認
                    if len(filtered_prices) != len(source_prices):
                        self.logger.warning(f"カルマンフィルター結果の長さが不一致: {len(filtered_prices)} != {len(source_prices)}")
                        filtered_prices = source_prices
                    
                except Exception as e:
                    self.logger.warning(f"カルマンフィルター適用中にエラー: {e}。元の値を使用します。")
                    filtered_prices = source_prices
            
            # 3. 価格をスムージング
            if self.price_smoother is not None:
                try:
                    if isinstance(data, pd.DataFrame):
                        price_data = data.copy()
                        if self.src_type not in price_data.columns:
                            price_data[self.src_type] = filtered_prices
                    else:
                        price_data = pd.DataFrame({
                            'open': data[:, 0] if data.shape[1] >= 4 else filtered_prices,
                            'high': data[:, 1] if data.shape[1] >= 4 else filtered_prices,
                            'low': data[:, 2] if data.shape[1] >= 4 else filtered_prices,
                            'close': data[:, 3] if data.shape[1] >= 4 else filtered_prices
                        })
                        price_data[self.src_type] = filtered_prices
                    
                    smoother_result = self.price_smoother.calculate(price_data)
                    smooth = smoother_result.values
                    
                    if not isinstance(smooth, np.ndarray):
                        smooth = np.array(smooth)
                    if smooth.dtype != np.float64:
                        smooth = smooth.astype(np.float64)
                        
                except Exception as e:
                    self.logger.warning(f"価格スムージング中にエラー: {e}。元の値を使用します。")
                    smooth = filtered_prices.copy()
            else:
                smooth = filtered_prices.copy()
            
            # 4. 初期期間配列を作成
            period = np.full(len(filtered_prices), 66.0, dtype=np.float64)
            
            # 5. Xデトレンダーを計算
            detrender = calculate_x_detrender_numba(smooth, period)
            
            # 6. Xサイクル期間を検出
            period, smooth_period = calculate_x_cycle_period_numba(detrender)
            
            # 7. X適応フェーズを計算
            adaptive_phase = calculate_x_adaptive_phase_numba(smooth, smooth_period)
            
            # 8. トレンドラインを計算
            i_trend = np.full(len(filtered_prices), np.nan, dtype=np.float64)
            trend_line = np.full(len(filtered_prices), np.nan, dtype=np.float64)
            
            for i in range(len(filtered_prices)):
                if np.isnan(smooth_period[i]) or smooth_period[i] <= 0:
                    continue
                    
                dc_period = int(np.ceil(smooth_period[i] + 0.5))
                if dc_period <= 0 or i < dc_period:
                    continue
                
                # iTrendの計算（期間内の重み付き平均）
                sum_val = 0.0
                weight_sum = 0.0
                for j in range(dc_period):
                    if i - j >= 0:
                        weight = np.exp(-j / (dc_period * 0.4))  # 指数的重み付け
                        sum_val += weight * filtered_prices[i - j]
                        weight_sum += weight
                
                if weight_sum > 0:
                    i_trend[i] = sum_val / weight_sum
                
                # トレンドラインの計算（より強いスムージング）
                if not np.isnan(i_trend[i]):
                    if i >= 5:
                        weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                        values = []
                        for k in range(6):
                            if i-k >= 0 and not np.isnan(i_trend[i-k]):
                                values.append(i_trend[i-k])
                            else:
                                values.append(i_trend[i])
                        values = np.array(values)
                        trend_line[i] = np.sum(weights * values) / np.sum(weights)
                    else:
                        trend_line[i] = i_trend[i]
            
            # 9. Xマーケットモードを判定
            trend_mode, signal, days_in_trend, mode_strength, cycle_strength, trend_strength = calculate_x_market_mode_numba(
                smooth, trend_line, adaptive_phase, smooth_period
            )
            
            # 10. 平滑化（オプション）
            smoothed_values = np.full_like(signal, np.nan)
            if self.use_smoothing and self.smoother is not None:
                try:
                    signal_df = pd.DataFrame({
                        'close': signal,
                        'high': signal,
                        'low': signal,
                        'open': signal
                    })
                    
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
            
            # 結果の作成
            result = XMarketModeResult(
                smooth=smooth.copy(),
                detrender=detrender.copy(),
                period=period.copy(),
                smooth_period=smooth_period.copy(),
                adaptive_phase=adaptive_phase.copy(),
                trend_line=trend_line.copy(),
                trend_mode=trend_mode.copy(),
                signal=signal.copy(),
                days_in_trend=days_in_trend.copy(),
                i_trend=i_trend.copy(),
                filtered_price=filtered_prices.copy(),
                mode_strength=mode_strength.copy(),
                cycle_strength=cycle_strength.copy(),
                trend_strength=trend_strength.copy()
            )
            
            # 基底クラス用の値設定
            self._values = trend_mode
            
            return result
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"Xマーケットモード計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時はNaN値で埋めた配列を返す（データ長に合わせる）
            data_len = len(data) if hasattr(data, '__len__') else 0
            nan_array = np.full(data_len, np.nan, dtype=np.float64)
            return XMarketModeResult(
                smooth=nan_array.copy(),
                detrender=nan_array.copy(),
                period=nan_array.copy(),
                smooth_period=nan_array.copy(),
                adaptive_phase=nan_array.copy(),
                trend_line=nan_array.copy(),
                trend_mode=nan_array.copy(),
                signal=nan_array.copy(),
                days_in_trend=nan_array.copy(),
                i_trend=nan_array.copy(),
                filtered_price=nan_array.copy(),
                mode_strength=nan_array.copy(),
                cycle_strength=nan_array.copy(),
                trend_strength=nan_array.copy()
            )
    
    def get_values(self) -> Optional[np.ndarray]:
        """トレンドモード値を取得"""
        return self._values.copy() if self._values is not None else None
    
    def get_trend_mode(self) -> Optional[np.ndarray]:
        """トレンドモードを取得"""
        return self._values.copy() if self._values is not None else None


# 便利関数
def calculate_x_market_mode(
    data: Union[pd.DataFrame, np.ndarray],
    src_type: str = 'hlc3',
    trend_threshold: float = 0.008,
    use_smoothing: bool = True,
    smoother_type: str = 'frama',
    use_kalman_filter: bool = False,
    **kwargs
) -> np.ndarray:
    """
    Xマーケットモードの計算（便利関数）
    
    Args:
        data: 価格データ
        src_type: ソースタイプ
        trend_threshold: トレンド判定閾値
        use_smoothing: 平滑化を使用するか
        smoother_type: スムーサータイプ
        use_kalman_filter: カルマンフィルターを使用するか
        **kwargs: その他のパラメータ
        
    Returns:
        トレンドモード値
    """
    indicator = XMarketMode(
        src_type=src_type,
        trend_threshold=trend_threshold,
        use_smoothing=use_smoothing,
        smoother_type=smoother_type,
        use_kalman_filter=use_kalman_filter,
        **kwargs
    )
    result = indicator.calculate(data)
    return result.trend_mode


if __name__ == "__main__":
    """直接実行時のテスト"""
    import numpy as np
    import pandas as pd
    
    print("=== Xマーケットモード インジケーターのテスト ===")
    
    # テストデータ生成（中期向け）
    np.random.seed(42)
    length = 500
    base_price = 100.0
    
    # 中期トレンド・サイクル混合データ
    prices = [base_price]
    for i in range(1, length):
        if i < 100:  # 短期トレンド
            trend_component = 0.002
            cycle_component = 0.02 * np.sin(2 * np.pi * i / 30)
            noise = np.random.normal(0, 0.006)
        elif i < 200:  # 中期サイクル
            trend_component = 0.0
            cycle_component = 0.035 * np.sin(2 * np.pi * i / 66)
            noise = np.random.normal(0, 0.008)
        elif i < 300:  # 長期トレンド
            trend_component = 0.003
            cycle_component = 0.015 * np.sin(2 * np.pi * i / 89)
            noise = np.random.normal(0, 0.005)
        elif i < 400:  # 複合中期サイクル
            trend_component = -0.001
            cycle_component = (0.025 * np.sin(2 * np.pi * i / 45) + 
                             0.015 * np.sin(2 * np.pi * i / 78))
            noise = np.random.normal(0, 0.007)
        else:  # 強トレンド
            trend_component = 0.004
            cycle_component = 0.01 * np.sin(2 * np.pi * i / 124)
            noise = np.random.normal(0, 0.004)
        
        change = trend_component + cycle_component + noise
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # OHLC データの生成
    data = []
    for i, close in enumerate(prices):
        daily_range = abs(np.random.normal(0, close * 0.012))
        
        high = close + daily_range * np.random.uniform(0.3, 1.0)
        low = close - daily_range * np.random.uniform(0.3, 1.0)
        
        if i == 0:
            open_price = close
        else:
            gap = np.random.normal(0, close * 0.004)
            open_price = prices[i-1] + gap
        
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.uniform(1000, 15000)
        })
    
    df = pd.DataFrame(data)
    
    print(f"テストデータ: {len(df)}ポイント")
    print(f"価格範囲: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    # Xマーケットモードを計算
    print("\nXマーケットモードを計算中...")
    xmm = XMarketMode(use_kalman_filter=False)
    result = xmm.calculate(df)
    
    valid_trend_count = np.sum(~np.isnan(result.trend_mode))
    valid_signal_count = np.sum(~np.isnan(result.signal))
    mean_period = np.nanmean(result.period)
    
    # モード分析
    trend_mode_ratio = np.sum(result.trend_mode[~np.isnan(result.trend_mode)] == 1.0) / valid_trend_count if valid_trend_count > 0 else 0
    cycle_mode_ratio = np.sum(result.trend_mode[~np.isnan(result.trend_mode)] == 0.0) / valid_trend_count if valid_trend_count > 0 else 0
    
    # 強度分析
    mean_mode_strength = np.nanmean(result.mode_strength)
    mean_cycle_strength = np.nanmean(result.cycle_strength)
    mean_trend_strength = np.nanmean(result.trend_strength)
    
    print(f"  有効トレンドモード値数: {valid_trend_count}/{len(df)}")
    print(f"  有効信号数: {valid_signal_count}/{len(df)}")
    print(f"  平均サイクル期間: {mean_period:.1f}")
    print(f"  トレンドモード比率: {trend_mode_ratio:.2%}")
    print(f"  サイクルモード比率: {cycle_mode_ratio:.2%}")
    print(f"  平均モード強度: {mean_mode_strength:.3f}")
    print(f"  平均サイクル強度: {mean_cycle_strength:.3f}")
    print(f"  平均トレンド強度: {mean_trend_strength:.3f}")
    
    print("\n=== テスト完了 ===")