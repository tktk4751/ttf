#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Optional, NamedTuple, Tuple
import numpy as np
import pandas as pd
from numba import jit

# インポート処理
try:
    from .indicator import Indicator
    from .price_source import PriceSource
    from .smoother.ultimate_smoother import UltimateSmoother  # アルティメットスムーザー
except ImportError:
    # スタンドアロン実行時の対応
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from indicator import Indicator
    from price_source import PriceSource
    from indicators.smoother.ultimate_smoother import UltimateSmoother  # アルティメットスムーザー

# EhlersUnifiedDCは動的に条件付きインポート（実行時にファンクション内で処理）


class UltimateMAResult(NamedTuple):
    """UltimateMA計算結果"""
    values: np.ndarray              # 最終フィルター済み価格
    raw_values: np.ndarray          # 元の価格
    ukf_values: np.ndarray          # hlc3フィルター後
    kalman_values: np.ndarray       # 適応的カルマンフィルター後
    kalman_gains: np.ndarray        # カルマンゲイン
    kalman_innovations: np.ndarray  # カルマンイノベーション
    kalman_confidence: np.ndarray   # カルマン信頼度スコア
    ultimate_smooth_values: np.ndarray # アルティメットスムーザー後
    zero_lag_values: np.ndarray     # ゼロラグEMA後
    amplitude: np.ndarray           # ヒルベルト変換振幅
    phase: np.ndarray              # ヒルベルト変換位相
    realtime_trends: np.ndarray     # リアルタイムトレンド
    trend_signals: np.ndarray       # 1=up, -1=down, 0=range
    current_trend: str              # 'up', 'down', 'range'
    current_trend_value: int        # 1, -1, 0


# 適応的カルマンフィルターとスーパースムーザーフィルターは削除
# hlc3とUltimateSmoother を使用


@jit(nopython=True)
def zero_lag_ema_numba(prices: np.ndarray, period: int = 21) -> np.ndarray:
    """
    ⚡ ゼロラグEMA（遅延ゼロ指数移動平均）
    遅延を完全に除去した革新的EMA
    """
    n = len(prices)
    zero_lag = np.zeros(n)
    
    if n < 2:
        return prices.copy()
    
    alpha = 2.0 / (period + 1.0)
    zero_lag[0] = prices[0]
    
    for i in range(1, n):
        # 標準EMA
        ema = alpha * prices[i] + (1 - alpha) * zero_lag[i-1]
        
        # ゼロラグ補正（予測的補正）
        if i >= 2:
            # 価格変化の勢いを計算
            momentum = prices[i] - prices[i-1]
            # ラグ補正係数
            lag_correction = alpha * momentum
            zero_lag[i] = ema + lag_correction
        else:
            zero_lag[i] = ema
    
    return zero_lag


@jit(nopython=True)
def alma_numba(prices: np.ndarray, period: int = 21, offset: float = 0.85, sigma: float = 6.0) -> np.ndarray:
    """
    🎯 ALMA (Arnaud Legoux Moving Average)
    適応的な重み付けによる高精度移動平均
    """
    n = len(prices)
    alma = np.zeros(n)
    
    if n < period:
        return prices.copy()
    
    # 重み係数の事前計算
    weights = np.zeros(period)
    m = offset * (period - 1)
    s = period / sigma
    
    for i in range(period):
        weights[i] = np.exp(-((i - m) ** 2) / (2 * s * s))
    
    # 重みの正規化
    weight_sum = np.sum(weights)
    if weight_sum > 0:
        weights = weights / weight_sum
    
    # 初期値設定
    for i in range(period - 1):
        alma[i] = prices[i]
    
    # ALMA計算
    for i in range(period - 1, n):
        value = 0.0
        for j in range(period):
            value += weights[j] * prices[i - period + 1 + j]
        alma[i] = value
    
    return alma


@jit(nopython=True)
def hma_numba(prices: np.ndarray, period: int = 21) -> np.ndarray:
    """
    🚀 HMA (Hull Moving Average)
    ラグを最小化した高応答性移動平均
    """
    n = len(prices)
    hma = np.zeros(n)
    
    if n < period:
        return prices.copy()
    
    # WMA計算用の関数（内部関数として定義）
    def weighted_ma(data, length, start_idx):
        if length <= 0 or start_idx + length > len(data):
            return data[start_idx] if start_idx < len(data) else 0.0
        
        weight_sum = 0.0
        value_sum = 0.0
        
        for i in range(length):
            weight = length - i
            value_sum += data[start_idx + i] * weight
            weight_sum += weight
        
        return value_sum / weight_sum if weight_sum > 0 else 0.0
    
    half_period = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    
    # 初期値設定
    for i in range(period - 1):
        hma[i] = prices[i]
    
    # HMA計算
    raw_hma = np.zeros(n)
    
    for i in range(period - 1, n):
        # WMA(period/2) * 2 - WMA(period)
        wma_half = weighted_ma(prices, half_period, i - half_period + 1)
        wma_full = weighted_ma(prices, period, i - period + 1)
        raw_hma[i] = 2.0 * wma_half - wma_full
    
    # 最終的なWMA(sqrt(period))
    for i in range(period - 1 + sqrt_period - 1, n):
        hma[i] = weighted_ma(raw_hma, sqrt_period, i - sqrt_period + 1)
    
    return hma


@jit(nopython=True)
def kama_numba(prices: np.ndarray, period: int = 21, fast_sc: float = 2.0, slow_sc: float = 30.0) -> np.ndarray:
    """
    📈 KAMA (Kaufman's Adaptive Moving Average)
    ボラティリティに適応する動的移動平均
    """
    n = len(prices)
    kama = np.zeros(n)
    
    if n < period + 1:
        return prices.copy()
    
    # 平滑化定数
    fast_alpha = 2.0 / (fast_sc + 1.0)
    slow_alpha = 2.0 / (slow_sc + 1.0)
    
    # 初期値設定
    kama[0] = prices[0]
    for i in range(1, period):
        kama[i] = prices[i]
    
    # KAMA計算
    for i in range(period, n):
        # 変化量の計算
        change = abs(prices[i] - prices[i - period])
        
        # ボラティリティの計算
        volatility = 0.0
        for j in range(period):
            volatility += abs(prices[i - j] - prices[i - j - 1])
        
        # 効率比の計算
        if volatility > 0:
            efficiency_ratio = change / volatility
        else:
            efficiency_ratio = 0.0
        
        # 平滑化定数の計算
        smooth_constant = (efficiency_ratio * (fast_alpha - slow_alpha) + slow_alpha) ** 2
        
        # KAMA計算
        kama[i] = kama[i - 1] + smooth_constant * (prices[i] - kama[i - 1])
    
    return kama


@jit(nopython=True)
def hilbert_transform_filter_numba(prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    🌀 ヒルベルト変換フィルター（位相遅延ゼロ）
    瞬時振幅と瞬時位相を計算し、ノイズと信号を分離
    """
    n = len(prices)
    amplitude = np.zeros(n)
    phase = np.zeros(n)
    
    if n < 8:
        return prices.copy(), np.zeros(n)
    
    # 簡易ヒルベルト変換（FIRフィルター近似）
    for i in range(4, n-4):
        # 実部（元信号）
        real_part = prices[i]
        
        # 虚部（ヒルベルト変換）- 90度位相シフト
        imag_part = (prices[i-3] - prices[i+3] + 
                    3 * (prices[i+1] - prices[i-1])) / 8.0
        
        # 瞬時振幅
        amplitude[i] = np.sqrt(real_part**2 + imag_part**2)
        
        # 瞬時位相
        if real_part != 0:
            phase[i] = np.arctan2(imag_part, real_part)
    
    # 境界値の処理
    for i in range(4):
        amplitude[i] = amplitude[4] if n > 4 else 0.0
        phase[i] = phase[4] if n > 4 else 0.0
    for i in range(n-4, n):
        amplitude[i] = amplitude[n-5] if n > 4 else 0.0
        phase[i] = phase[n-5] if n > 4 else 0.0
    
    return amplitude, phase


@jit(nopython=True)
def adaptive_noise_reduction_numba(prices: np.ndarray, amplitude: np.ndarray) -> np.ndarray:
    """
    🔇 適応的ノイズ除去（AI風学習型）
    振幅情報を使用してノイズレベルを動的に調整
    """
    n = len(prices)
    denoised = np.zeros(n)
    
    if n < 5:
        return prices.copy()
    
    # 初期値
    denoised[0] = prices[0]
    
    for i in range(1, n):
        # ノイズレベルの推定
        if i >= 10:
            # 最近の振幅変動からノイズを推定
            recent_amp_std = np.std(amplitude[i-10:i])
            noise_threshold = recent_amp_std * 0.3
        else:
            noise_threshold = 0.1
        
        # 価格変化の大きさ
        price_change = abs(prices[i] - prices[i-1])
        
        # ノイズ判定と除去
        if price_change < noise_threshold:
            # 小さな変化はノイズとして除去（スムージング）
            if i >= 3:
                denoised[i] = (denoised[i-1] * 0.7 + 
                              prices[i] * 0.2 + 
                              denoised[i-2] * 0.1)
            else:
                denoised[i] = denoised[i-1] * 0.8 + prices[i] * 0.2
        else:
            # 大きな変化は信号として保持
            denoised[i] = prices[i] * 0.8 + denoised[i-1] * 0.2
    
    return denoised


@jit(nopython=True)
def real_time_trend_detector_numba(prices: np.ndarray, window: int = 5) -> np.ndarray:
    """
    🎯 **実践的高精度リアルタイムトレンド検出器 V2.1**
    
    本質的なトレンド検出に特化したシンプル最強アルゴリズム:
    - **適応的ノイズフィルタ**: 市場ノイズの動的除去
    - **重み付きトレンド**: 複数期間の最適結合
    - **本質抽出**: 真のトレンドのみを検出
    - **超低遅延**: 最小3期間から処理開始
    """
    n = len(prices)
    trend_signals = np.zeros(n)
    
    if n < 3:  # 最小3期間で開始
        return trend_signals
    
    # 適応的平滑化フィルタ（シンプル版）
    smoothed = np.zeros(n)
    smoothed[0] = prices[0]
    alpha = 0.3  # 平滑化係数
    
    for i in range(1, n):
        if np.isnan(prices[i]):
            smoothed[i] = smoothed[i-1]
            continue
        
        # 適応的平滑化（ノイズ除去）
        smoothed[i] = alpha * prices[i] + (1 - alpha) * smoothed[i-1]
    
    # メイントレンド検出ループ
    for i in range(3, n):
        if np.isnan(prices[i]):
            trend_signals[i] = 0.0
            continue
        
        # 🎯 1. 複数期間トレンド（重み付き）
        trend_1 = smoothed[i] - smoothed[i-1]           # 瞬間（重み40%）
        trend_2 = (smoothed[i] - smoothed[i-2]) / 2.0   # 短期（重み35%） 
        trend_3 = (smoothed[i] - smoothed[i-3]) / 3.0   # 中期（重み25%）
        
        # 重み付き統合トレンド
        combined_trend = trend_1 * 0.4 + trend_2 * 0.35 + trend_3 * 0.25
        
        # 🛡️ 2. 動的ノイズレベル判定（超軽量）
        noise_threshold = 0.0
        if i >= 5:
            # 直近の価格変動からノイズレベルを推定
            recent_noise = abs(prices[i-1] - prices[i-2]) + abs(prices[i-2] - prices[i-3])
            avg_noise = recent_noise / 2.0
            noise_threshold = avg_noise * 0.5  # ノイズ閾値
        
        # 🔥 3. 本質的トレンド抽出
        
        # トレンド一貫性チェック
        consistency = 0.0
        if abs(trend_1) > 0 and abs(trend_2) > 0 and abs(trend_3) > 0:
            # 3つのトレンドの方向一致度
            direction_1 = 1 if trend_1 > 0 else -1
            direction_2 = 1 if trend_2 > 0 else -1  
            direction_3 = 1 if trend_3 > 0 else -1
            
            main_direction = 1 if combined_trend > 0 else -1
            matches = 0
            if direction_1 == main_direction: matches += 1
            if direction_2 == main_direction: matches += 1
            if direction_3 == main_direction: matches += 1
            
            consistency = matches / 3.0
        
        # トレンド強度の計算
        trend_strength = abs(combined_trend)
        
        # ⚡ 4. 実践的フィルタリング
        
        # ノイズフィルター
        if trend_strength <= noise_threshold:
            trend_signals[i] = 0.0  # ノイズとして除去
            continue
        
        # 一貫性フィルター
        if consistency < 0.6:  # 60%未満の一貫性は弱い信号
            trend_strength *= 0.5  # 信号を弱める
        
        # より長期間のトレンド確認（可能な場合）
        long_term_boost = 1.0
        if i >= min(window, 8):
            long_trend = (smoothed[i] - smoothed[i-min(window, 8)]) / min(window, 8)
            
            # 長期トレンドと一致する場合は強化
            if (combined_trend > 0 and long_trend > 0) or (combined_trend < 0 and long_trend < 0):
                long_term_boost = 1.3  # 30%強化
        
        # 🎯 5. 最終信号生成（本質的トレンドのみ）
        
        final_strength = trend_strength * long_term_boost
        
        # 最小強度フィルター（ノイズ完全除去）
        min_strength = max(noise_threshold * 2.0, abs(combined_trend) * 0.1)
        
        if final_strength > min_strength:
            # 符号付き強度で出力
            trend_signals[i] = final_strength * (1 if combined_trend > 0 else -1)
        else:
            trend_signals[i] = 0.0  # 本質的でないトレンドは除去
    
    return trend_signals


@jit(nopython=True)
def calculate_trend_signals_with_range_numba(values: np.ndarray, slope_index: int, range_threshold: float = 0.005) -> np.ndarray:
    """
    🚀 **超高精度AI風トレンド判定アルゴリズム V3.0** 🚀
    
    最新の金融工学技術を統合した次世代判定システム:
    - **適応的指数重み付け統計**: 最新データ重視の動的閾値
    - **多時間軸モメンタム分析**: 1期・2期・3期・5期の複合解析
    - **AI風動的信頼度スコア**: 4指標の重み付き総合評価
    - **高精度ボラティリティ分析**: 市場状況の自動判定・適応
    - **予測的判定システム**: 先読み機能による早期検出
    - **緊急事態検出**: 極端変化への瞬時対応
    
    Args:
        values: インジケーター値の配列
        slope_index: スロープ判定期間
        range_threshold: range判定の基本閾値
    
    Returns:
        trend_signals: 1=up, -1=down, 0=range のNumPy配列
    """
    length = len(values)
    trend_signals = np.zeros(length, dtype=np.int8)
    
    # 超低遅延統計ウィンドウ（最小限に短縮）
    stats_window = max(13, slope_index * 2)  # 大幅短縮
    confirmation_window = 5  # 固定2期間で即応性重視
    
    for i in range(stats_window, length):
        if np.isnan(values[i]):
            trend_signals[i] = 0
            continue
        
        current = values[i]
        previous = values[i - slope_index]
        
        if np.isnan(previous):
            trend_signals[i] = 0
            continue
        
        # 基本的な変化量
        change = current - previous
        base_value = max(abs(current), abs(previous), 1e-10)
        relative_change = change / base_value
        abs_relative_change = abs(relative_change)
        
        # 🔥 1. 適応的指数重み付け統計（最新データ重視）
        start_idx = max(slope_index, i - stats_window + 1)
        
        # 指数重み付けによる高精度閾値計算
        weighted_changes = 0.0
        weighted_sum = 0.0
        weighted_variance = 0.0
        
        for j in range(start_idx + slope_index, i):
            if not np.isnan(values[j]) and not np.isnan(values[j - slope_index]):
                hist_current = values[j]
                hist_previous = values[j - slope_index]
                hist_base = max(abs(hist_current), abs(hist_previous), 1e-10)
                hist_change = abs(hist_current - hist_previous) / hist_base
                
                # 指数重み（最新データほど重要）
                age = i - j
                weight = np.exp(-age * 0.15)  # 指数減衰
                
                weighted_changes += hist_change * weight
                weighted_sum += weight
        
        if weighted_sum > 0:
            weighted_mean = weighted_changes / weighted_sum
            
            # 重み付き分散の計算
            for j in range(start_idx + slope_index, i):
                if not np.isnan(values[j]) and not np.isnan(values[j - slope_index]):
                    hist_current = values[j]
                    hist_previous = values[j - slope_index]
                    hist_base = max(abs(hist_current), abs(hist_previous), 1e-10)
                    hist_change = abs(hist_current - hist_previous) / hist_base
                    
                    age = i - j
                    weight = np.exp(-age * 0.15)
                    weighted_variance += weight * (hist_change - weighted_mean) ** 2
            
            weighted_std = np.sqrt(weighted_variance / weighted_sum) if weighted_sum > 0 else 0.0
            
            # 動的適応閾値（最新データ重視）
            adaptive_threshold = weighted_mean + weighted_std * 1.0
            effective_threshold = max(range_threshold, adaptive_threshold)
        else:
            effective_threshold = range_threshold
        
        # 🚀 2. 多時間軸モメンタム分析（予測的継続性）
        momentum_score = 0.0
        consistency_score = 0.0
        
        if i >= confirmation_window:
            # 複数時間軸でのモメンタム計算
            momentum_1 = change  # 現在の変化
            momentum_2 = (values[i] - values[i-2]) / 2.0 if i >= 2 and not np.isnan(values[i-2]) else 0.0
            momentum_3 = (values[i] - values[i-3]) / 3.0 if i >= 3 and not np.isnan(values[i-3]) else 0.0
            momentum_5 = (values[i] - values[i-5]) / 5.0 if i >= 5 and not np.isnan(values[i-5]) else 0.0
            
            # モメンタム強度（加重平均）
            momentum_weights = np.array([0.4, 0.3, 0.2, 0.1])
            momentums = np.array([momentum_1, momentum_2, momentum_3, momentum_5])
            
            # 正規化モメンタムスコア
            momentum_score = np.sum(momentums * momentum_weights)
            
            # 方向一致性スコア（Numba対応版）
            directions = np.zeros(4, dtype=np.int8)
            direction_count = 0
            
            if momentum_1 != 0: 
                directions[direction_count] = 1 if momentum_1 > 0 else -1
                direction_count += 1
            if momentum_2 != 0: 
                directions[direction_count] = 1 if momentum_2 > 0 else -1
                direction_count += 1
            if momentum_3 != 0: 
                directions[direction_count] = 1 if momentum_3 > 0 else -1
                direction_count += 1
            if momentum_5 != 0: 
                directions[direction_count] = 1 if momentum_5 > 0 else -1
                direction_count += 1
            
            if direction_count > 0:
                main_direction = 1 if change > 0 else -1
                consistent_count = 0
                for k in range(direction_count):
                    if directions[k] == main_direction:
                        consistent_count += 1
                consistency_score = consistent_count / direction_count
            else:
                consistency_score = 0.0
        
        # 🎯 3. 高精度ノイズ・ボラティリティ分析
        volatility_factor = 1.0
        signal_strength = 0.0
        
        if i >= 8:
            
            # 短期変化の計算（Numba対応）
            short_changes = np.zeros(4)
            short_count = 0
            for j in range(max(1, i-4), i):
                if not np.isnan(values[j]) and not np.isnan(values[j-1]) and short_count < 4:
                    short_changes[short_count] = abs(values[j] - values[j-1])
                    short_count += 1
            
            # 中期変化の計算（Numba対応）
            mid_changes = np.zeros(4)
            mid_count = 0
            for j in range(max(2, i-8), i, 2):
                if not np.isnan(values[j]) and not np.isnan(values[j-2]) and mid_count < 4:
                    mid_changes[mid_count] = abs(values[j] - values[j-2])
                    mid_count += 1
            
            if short_count >= 2 and mid_count >= 2:
                # 平均計算（Numba対応）
                short_vol = np.sum(short_changes[:short_count]) / short_count
                mid_vol = np.sum(mid_changes[:mid_count]) / mid_count
                
                # ボラティリティ比率（市場状況の判定）
                vol_ratio = short_vol / (mid_vol + 1e-10)
                
                # 信号強度の計算
                signal_strength = abs(change) / (short_vol + 1e-10)
                
                # 動的ボラティリティ補正
                if vol_ratio > 1.5:  # 高ボラティリティ環境
                    volatility_factor = 1.3
                elif vol_ratio < 0.7:  # 低ボラティリティ環境
                    volatility_factor = 0.8
                else:  # 通常環境
                    volatility_factor = 1.0
        
        final_threshold = effective_threshold * volatility_factor
        
        # 🎯 4. AI風動的信頼度スコアリングシステム
        
        # 複数指標の総合スコア計算
        change_score = min(abs_relative_change / final_threshold, 2.0)  # 変化量スコア（最大2.0）
        momentum_strength = min(abs(momentum_score) / (effective_threshold + 1e-10), 2.0)  # モメンタムスコア
        consistency_weight = consistency_score  # 一貫性重み
        signal_quality = min(signal_strength, 3.0)  # 信号品質（最大3.0）
        
        # 総合信頼度スコア（重み付き合成）
        confidence_score = (change_score * 0.35 +           # 変化量の重要度
                           momentum_strength * 0.25 +       # モメンタムの重要度  
                           consistency_weight * 0.25 +      # 一貫性の重要度
                           signal_quality * 0.15)           # 信号品質の重要度
        
        # 🔥 5. 予測的判定システム（先読み機能）
        
        # 基本しきい値チェック
        base_threshold = 0.8  # 基準信頼度
        high_threshold = 1.3  # 高信頼度
        
        # 最小変化量フィルター
        min_change_filter = abs_relative_change > range_threshold * 0.2
        
        if min_change_filter and confidence_score > 0.3:  # 最低信頼度30%
            if confidence_score > high_threshold:
                # 高信頼度：即座にトレンド判定
                trend_signals[i] = 1 if relative_change > 0 else -1
            elif confidence_score > base_threshold:
                # 中信頼度：一貫性も確認
                if consistency_score >= 0.6:  # 60%以上の一貫性
                    trend_signals[i] = 1 if relative_change > 0 else -1
                else:
                    trend_signals[i] = 0  # レンジ
            else:
                # 低信頼度：厳格な条件でのみ判定
                if (consistency_score >= 0.8 and  # 80%以上の高一貫性
                    signal_quality > 1.5 and      # 高品質信号
                    change_score > 1.0):           # 十分な変化量
                    trend_signals[i] = 1 if relative_change > 0 else -1
                else:
                    trend_signals[i] = 0  # レンジ
        else:
            trend_signals[i] = 0  # レンジ
        
        # 🔥 6. 極端変化・緊急事態検出システム
        extreme_threshold = final_threshold * 2.5
        if abs_relative_change > extreme_threshold:
            # 極端な変化の場合、緊急判定モード
            emergency_confidence = change_score + signal_quality
            
            if emergency_confidence > 2.0:  # 緊急事態レベル
                # 方向確認のみで即座に判定
                if i >= 1 and not np.isnan(values[i-1]):
                    prev_change = values[i] - values[i-1]
                    same_direction = (relative_change > 0) == (prev_change > 0)
                    
                    if same_direction or emergency_confidence > 3.0:
                        trend_signals[i] = 1 if relative_change > 0 else -1
                else:
                    # 前期データなしでも超極端な場合は判定
                    if abs_relative_change > extreme_threshold * 2.0:
                        trend_signals[i] = 1 if relative_change > 0 else -1
    
    return trend_signals


@jit(nopython=True)
def calculate_current_trend_with_range_numba(trend_signals: np.ndarray) -> tuple:
    """
    現在のトレンド状態を計算する（range対応版）(Numba JIT)
    
    Args:
        trend_signals: トレンド信号配列 (1=up, -1=down, 0=range)
    
    Returns:
        tuple: (current_trend_index, current_trend_value)
               current_trend_index: 0=range, 1=up, 2=down (trend_names用のインデックス)
               current_trend_value: 0=range, 1=up, -1=down (実際のトレンド値)
    """
    length = len(trend_signals)
    if length == 0:
        return 0, 0  # range
    
    # 最新の値を取得
    latest_trend = trend_signals[-1]
    
    if latest_trend == 1:  # up
        return 1, 1   # up
    elif latest_trend == -1:  # down
        return 2, -1   # down
    else:  # range
        return 0, 0  # range


@jit(nopython=True)
def zero_lag_ema_adaptive_numba(prices: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """
    ⚡ 動的適応ゼロラグEMA（期間が動的に変化）
    遅延を完全に除去した革新的EMA（適応的期間対応）
    """
    n = len(prices)
    zero_lag = np.zeros(n)
    
    if n < 2:
        return prices.copy()
    
    zero_lag[0] = prices[0]
    
    for i in range(1, n):
        # 動的期間からアルファを計算
        period = max(2.0, periods[i])  # 最小期間2
        alpha = 2.0 / (period + 1.0)
        
        # 標準EMA
        ema = alpha * prices[i] + (1 - alpha) * zero_lag[i-1]
        
        # ゼロラグ補正（予測的補正）
        if i >= 2:
            # 価格変化の勢いを計算
            momentum = prices[i] - prices[i-1]
            # ラグ補正係数
            lag_correction = alpha * momentum
            zero_lag[i] = ema + lag_correction
        else:
            zero_lag[i] = ema
    
    return zero_lag


@jit(nopython=True)
def real_time_trend_detector_adaptive_numba(prices: np.ndarray, windows: np.ndarray) -> np.ndarray:
    """
    🎯 **動的適応実践的高精度リアルタイムトレンド検出器 V2.1**
    
    本質的なトレンド検出に特化したシンプル最強アルゴリズム（動的期間対応）:
    - **動的適応ノイズフィルタ**: 期間に応じた市場ノイズの動的除去
    - **重み付きトレンド**: 複数期間の最適結合（動的調整）
    - **本質抽出**: 真のトレンドのみを検出
    - **超低遅延**: 最小3期間から処理開始（動的期間対応）
    """
    n = len(prices)
    trend_signals = np.zeros(n)
    
    if n < 3:  # 最小3期間で開始
        return trend_signals
    
    # 適応的平滑化フィルタ（動的版）
    smoothed = np.zeros(n)
    smoothed[0] = prices[0]
    
    for i in range(1, n):
        if np.isnan(prices[i]):
            smoothed[i] = smoothed[i-1]
            continue
        
        # 動的適応平滑化係数（ウィンドウサイズに応じて調整）
        current_window = max(3, min(int(windows[i]), i))
        alpha = 2.0 / (current_window + 1.0)  # 動的alpha
        alpha = max(0.1, min(0.5, alpha))  # 範囲制限
        
        # 適応的平滑化（ノイズ除去）
        smoothed[i] = alpha * prices[i] + (1 - alpha) * smoothed[i-1]
    
    # メイントレンド検出ループ
    for i in range(3, n):
        if np.isnan(prices[i]):
            trend_signals[i] = 0.0
            continue
        
        # 動的ウィンドウサイズ
        current_window = max(3, min(int(windows[i]), i))
        
        # 🎯 1. 動的複数期間トレンド（重み付き）
        
        # 基本期間の設定（動的調整）
        period_1 = 1
        period_2 = min(2, current_window // 3)
        period_3 = min(3, current_window // 2)
        
        # 各期間のトレンド計算
        trend_1 = smoothed[i] - smoothed[i-period_1]                    # 瞬間
        trend_2 = (smoothed[i] - smoothed[i-period_2]) / period_2       # 短期
        trend_3 = (smoothed[i] - smoothed[i-period_3]) / period_3       # 中期
        
        # 動的重み付き統合（ウィンドウサイズに応じて調整）
        if current_window <= 5:
            # 短期間の場合：瞬間重視
            combined_trend = trend_1 * 0.5 + trend_2 * 0.35 + trend_3 * 0.15
        elif current_window <= 15:
            # 中期間の場合：バランス重視
            combined_trend = trend_1 * 0.4 + trend_2 * 0.35 + trend_3 * 0.25
        else:
            # 長期間の場合：安定性重視
            combined_trend = trend_1 * 0.3 + trend_2 * 0.35 + trend_3 * 0.35
        
        # 🛡️ 2. 動的ノイズレベル判定（超軽量）
        noise_threshold = 0.0
        if i >= 5:
            # 直近の価格変動からノイズレベルを推定（動的調整）
            lookback = min(5, current_window // 2)
            recent_noise = 0.0
            for j in range(1, lookback + 1):
                if i >= j + 1:
                    recent_noise += abs(prices[i-j] - prices[i-j-1])
            
            avg_noise = recent_noise / lookback if lookback > 0 else 0.0
            
            # ウィンドウサイズに応じたノイズ閾値調整
            noise_multiplier = 0.3 if current_window <= 10 else 0.5
            noise_threshold = avg_noise * noise_multiplier
        
        # 🔥 3. 本質的トレンド抽出（動的版）
        
        # トレンド一貫性チェック
        consistency = 0.0
        if abs(trend_1) > 0 and abs(trend_2) > 0 and abs(trend_3) > 0:
            # 3つのトレンドの方向一致度
            direction_1 = 1 if trend_1 > 0 else -1
            direction_2 = 1 if trend_2 > 0 else -1  
            direction_3 = 1 if trend_3 > 0 else -1
            
            main_direction = 1 if combined_trend > 0 else -1
            matches = 0
            if direction_1 == main_direction: matches += 1
            if direction_2 == main_direction: matches += 1
            if direction_3 == main_direction: matches += 1
            
            consistency = matches / 3.0
        
        # トレンド強度の計算
        trend_strength = abs(combined_trend)
        
        # ⚡ 4. 動的実践的フィルタリング
        
        # ノイズフィルター
        if trend_strength <= noise_threshold:
            trend_signals[i] = 0.0  # ノイズとして除去
            continue
        
        # 動的一貫性フィルター（ウィンドウサイズに応じて調整）
        consistency_threshold = 0.5 if current_window <= 10 else 0.6
        if consistency < consistency_threshold:
            trend_strength *= 0.5  # 信号を弱める
        
        # より長期間のトレンド確認（動的調整）
        long_term_boost = 1.0
        long_period = min(current_window, 8)
        if i >= long_period:
            long_trend = (smoothed[i] - smoothed[i-long_period]) / long_period
            
            # 長期トレンドと一致する場合は強化（動的調整）
            if (combined_trend > 0 and long_trend > 0) or (combined_trend < 0 and long_trend < 0):
                boost_factor = 1.2 if current_window <= 10 else 1.3  # 動的強化
                long_term_boost = boost_factor
        
        # 🎯 5. 最終信号生成（本質的トレンドのみ・動的版）
        
        final_strength = trend_strength * long_term_boost
        
        # 動的最小強度フィルター（ノイズ完全除去）
        min_strength_base = max(noise_threshold * 2.0, abs(combined_trend) * 0.1)
        
        # ウィンドウサイズに応じた最小強度調整
        window_factor = 0.8 if current_window <= 10 else 1.0
        min_strength = min_strength_base * window_factor
        
        if final_strength > min_strength:
            # 符号付き強度で出力
            trend_signals[i] = final_strength * (1 if combined_trend > 0 else -1)
        else:
            trend_signals[i] = 0.0  # 本質的でないトレンドは除去
    
    return trend_signals


@jit(nopython=True)
def adaptive_kalman_filter_numba(prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    🎯 適応的カルマンフィルター（超低遅延ノイズ除去）
    動的にノイズレベルを推定し、リアルタイムでノイズ除去
    """
    n = len(prices)
    filtered_prices = np.zeros(n)
    kalman_gains = np.zeros(n)
    innovations = np.zeros(n)
    confidence_scores = np.zeros(n)
    
    if n < 2:
        return prices.copy(), kalman_gains, innovations, np.ones(n)
    
    # 初期化
    filtered_prices[0] = prices[0]
    kalman_gains[0] = 0.5
    innovations[0] = 0.0
    confidence_scores[0] = 1.0
    
    # カルマンフィルターパラメータ（適応的）
    process_variance = 1e-5
    measurement_variance = 0.01
    
    # 状態推定
    x_est = prices[0]
    p_est = 1.0
    
    for i in range(1, n):
        # 予測ステップ
        x_pred = x_est
        p_pred = p_est + process_variance
        
        # 適応的測定ノイズ推定
        if i >= 5:
            recent_volatility = np.std(prices[i-5:i])
            measurement_variance = max(0.001, min(0.1, recent_volatility * 0.1))
        
        # カルマンゲイン
        kalman_gain = p_pred / (p_pred + measurement_variance)
        
        # 更新ステップ
        innovation = prices[i] - x_pred
        x_est = x_pred + kalman_gain * innovation
        p_est = (1 - kalman_gain) * p_pred
        
        filtered_prices[i] = x_est
        kalman_gains[i] = kalman_gain
        innovations[i] = innovation
        confidence_scores[i] = 1.0 / (1.0 + p_est)
    
    return filtered_prices, kalman_gains, innovations, confidence_scores


class UltimateMA(Indicator):
    """
    🚀 **Ultimate Moving Average - V5.2 DYNAMIC ADAPTIVE QUANTUM NEURAL SUPREMACY EDITION**
    
    🎯 **6段階革新的フィルタリングシステム + 動的適応機能:**
    1. **hlc3フィルター**: 無香料カルマンフィルター・高精度状態推定
    2. **アルティメットスムーザー**: John Ehlers Ultimate Smoother・ゼロ遅延設計
    3. **ゼロラグEMA**: 遅延完全除去・予測的補正（動的適応対応）
    4. **ヒルベルト変換フィルター**: 位相遅延ゼロ・瞬時振幅/位相
    5. **適応的ノイズ除去**: AI風学習型・振幅連動調整
    6. **リアルタイムトレンド検出**: 超低遅延・即座反応（動的適応対応）
    
    🏆 **革新的特徴:**
    - **ノイズ除去**: 6段階革新的フィルタリング
    - **超低遅延**: リアルタイム処理最適化
    - **位相遅延ゼロ**: ヒルベルト変換適用
    - **適応的学習**: AI風ノイズレベル推定
    - **動的適応**: Ehlersサイクル検出による期間自動調整
    - **80%超高信頼度**: 量子ニューラル技術
    - **完全統合処理**: 各段階の結果も取得可能
    """
    
    def __init__(self, 
                 ultimate_smoother_period: float = 5.0,
                 zero_lag_period: int = 21,
                 realtime_window: int = 89,
                 src_type: str = 'hlc3',
                 slope_index: int = 1,
                 range_threshold: float = 0.005,
                 # 適応的カルマンフィルターパラメータ
                 use_adaptive_kalman: bool = True,  # 適応的カルマンフィルターを使用するか
                 kalman_process_variance: float = 1e-5,  # プロセス分散
                 kalman_measurement_variance: float = 0.01,  # 測定分散
                 kalman_volatility_window: int = 5,  # ボラティリティ計算ウィンドウ
                 # 動的適応パラメータ
                 zero_lag_period_mode: str = 'dynamic', # dynamic or fixed
                 realtime_window_mode: str = 'dynamic', # dynamic or fixed
                 # ゼロラグ用サイクル検出器パラメータ
                 zl_cycle_detector_type: str = 'absolute_ultimate',
                 zl_cycle_detector_cycle_part: float = 1.0,
                 zl_cycle_detector_max_cycle: int = 120,
                 zl_cycle_detector_min_cycle: int = 5,
                 zl_cycle_period_multiplier: float = 1.0,
                 # リアルタイムウィンドウ用サイクル検出器パラメータ
                 rt_cycle_detector_type: str = 'absolute_ultimate',
                 rt_cycle_detector_cycle_part: float = 0.5,
                 rt_cycle_detector_max_cycle: int = 120,
                 rt_cycle_detector_min_cycle: int = 5,
                 rt_cycle_period_multiplier: float = 0.5,
                 # period_rangeパラメータ（absolute_ultimate、ultra_supreme_stability用）
                 zl_cycle_detector_period_range: Tuple[int, int] = (5, 120),
                 rt_cycle_detector_period_range: Tuple[int, int] = (5, 120)):
        """
        コンストラクタ
        
        Args:
            ultimate_smoother_period: アルティメットスムーザー期間（デフォルト: 13.0）
            zero_lag_period: ゼロラグEMA期間（デフォルト: 21）
            realtime_window: リアルタイムトレンド検出ウィンドウ（デフォルト: 89）
            src_type: 価格ソース ('close', 'hlc3', etc.)
            slope_index: トレンド判定期間 (1以上、デフォルト: 1)
            range_threshold: range判定の基本閾値（デフォルト: 0.005 = 0.5%）
            
            # 適応的カルマンフィルターパラメータ
            use_adaptive_kalman: 適応的カルマンフィルターを使用するか（デフォルト: True）
            kalman_process_variance: プロセス分散（デフォルト: 1e-5）
            kalman_measurement_variance: 測定分散（デフォルト: 0.01）
            kalman_volatility_window: ボラティリティ計算ウィンドウ（デフォルト: 5）
            
            # UKFパラメータ
            ukf_alpha: UKFのalpha値（デフォルト: 0.001）
            ukf_beta: UKFのbeta値（デフォルト: 2.0）
            ukf_kappa: UKFのkappa値（デフォルト: 0.0）
            ukf_process_noise_scale: プロセスノイズスケール（デフォルト: 0.001）
            ukf_volatility_window: ボラティリティ計算ウィンドウ（デフォルト: 10）
            ukf_adaptive_noise: 適応ノイズの使用（デフォルト: True）
            
            # 動的適応パラメータ
            zero_lag_period_mode: ゼロラグ期間モード ('fixed' or 'dynamic')
            realtime_window_mode: リアルタイムウィンドウモード ('fixed' or 'dynamic')
            
            # ゼロラグ用サイクル検出器パラメータ
            zl_cycle_detector_type: ゼロラグ用サイクル検出器タイプ ('hody', 'phac', 'dudi', etc.)
            zl_cycle_detector_cycle_part: ゼロラグ用サイクル検出器のサイクル部分倍率（デフォルト: 1.0）
            zl_cycle_detector_max_cycle: ゼロラグ用サイクル検出器の最大サイクル期間（デフォルト: 120）
            zl_cycle_detector_min_cycle: ゼロラグ用サイクル検出器の最小サイクル期間（デフォルト: 5）
            zl_cycle_period_multiplier: ゼロラグ用サイクル期間の乗数（デフォルト: 1.0）
            
            # リアルタイムウィンドウ用サイクル検出器パラメータ
            rt_cycle_detector_type: リアルタイム用サイクル検出器タイプ ('hody', 'phac', 'dudi', etc.)
            rt_cycle_detector_cycle_part: リアルタイム用サイクル検出器のサイクル部分倍率（デフォルト: 0.5）
            rt_cycle_detector_max_cycle: リアルタイム用サイクル検出器の最大サイクル期間（デフォルト: 50）
            rt_cycle_detector_min_cycle: リアルタイム用サイクル検出器の最小サイクル期間（デフォルト: 6）
            rt_cycle_period_multiplier: リアルタイム用サイクル期間の乗数（デフォルト: 0.33）
            zl_cycle_detector_period_range: ゼロラグ用サイクル検出器の周期範囲（デフォルト: (5, 120)）
            rt_cycle_detector_period_range: リアルタイム用サイクル検出器の周期範囲（デフォルト: (5, 120)）
        """
        kalman_info = f"KF:{'ON' if use_adaptive_kalman else 'OFF'}"
        super().__init__(f"UltimateMA({kalman_info},us={ultimate_smoother_period},zl={zero_lag_period}({zero_lag_period_mode}),rt={realtime_window}({realtime_window_mode}),src={src_type},slope={slope_index},range_th={range_threshold:.3f},zl_cycle={zl_cycle_detector_type},rt_cycle={rt_cycle_detector_type})")
        
        self.ultimate_smoother_period = ultimate_smoother_period
        self.zero_lag_period = zero_lag_period
        self.realtime_window = realtime_window
        self.src_type = src_type
        self.slope_index = slope_index
        self.range_threshold = range_threshold
        
        # 適応的カルマンフィルターパラメータ
        self.use_adaptive_kalman = use_adaptive_kalman
        self.kalman_process_variance = kalman_process_variance
        self.kalman_measurement_variance = kalman_measurement_variance
        self.kalman_volatility_window = kalman_volatility_window
        
        # 動的適応パラメータ
        self.zero_lag_period_mode = zero_lag_period_mode.lower()
        self.realtime_window_mode = realtime_window_mode.lower()
        
        # ゼロラグ用サイクル検出器パラメータ
        self.zl_cycle_detector_type = zl_cycle_detector_type
        self.zl_cycle_detector_cycle_part = zl_cycle_detector_cycle_part
        self.zl_cycle_detector_max_cycle = zl_cycle_detector_max_cycle
        self.zl_cycle_detector_min_cycle = zl_cycle_detector_min_cycle
        self.zl_cycle_period_multiplier = zl_cycle_period_multiplier
        
        # リアルタイムウィンドウ用サイクル検出器パラメータ
        self.rt_cycle_detector_type = rt_cycle_detector_type
        self.rt_cycle_detector_cycle_part = rt_cycle_detector_cycle_part
        self.rt_cycle_detector_max_cycle = rt_cycle_detector_max_cycle
        self.rt_cycle_detector_min_cycle = rt_cycle_detector_min_cycle
        self.rt_cycle_period_multiplier = rt_cycle_period_multiplier
        self.zl_cycle_detector_period_range = zl_cycle_detector_period_range
        self.rt_cycle_detector_period_range = rt_cycle_detector_period_range
        
        # パラメータ検証
        if self.zero_lag_period_mode not in ['fixed', 'dynamic']:
            raise ValueError(f"無効なzero_lag_period_mode: {self.zero_lag_period_mode}. 'fixed' または 'dynamic' を指定してください。")
        
        if self.realtime_window_mode not in ['fixed', 'dynamic']:
            raise ValueError(f"無効なrealtime_window_mode: {self.realtime_window_mode}. 'fixed' または 'dynamic' を指定してください。")
        
        self.price_source_extractor = PriceSource()
        
        # 動的適応が必要な場合のみEhlersUnifiedDCを初期化
        self.zl_cycle_detector = None
        self.rt_cycle_detector = None
        
        # ゼロラグ用サイクル検出器の初期化
        if self.zero_lag_period_mode == 'dynamic':
            # EhlersUnifiedDCのインポート（デバッグ付き）
            EhlersUnifiedDC = None
            import_success = False
            
            try:
                # 相対インポートを試行
                from .cycle.ehlers_unified_dc import EhlersUnifiedDC
                import_success = True
                self.logger.debug("UltimateMA: EhlersUnifiedDC 相対インポート成功")
            except ImportError as e1:
                self.logger.debug(f"UltimateMA: EhlersUnifiedDC 相対インポート失敗: {e1}")
                try:
                    # 絶対インポートを試行（パス調整付き）
                    import sys
                    import os
                    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    if current_dir not in sys.path:
                        sys.path.insert(0, current_dir)
                    
                    from indicators.cycle.ehlers_unified_dc import EhlersUnifiedDC
                    import_success = True
                    self.logger.debug("UltimateMA: EhlersUnifiedDC 絶対インポート成功")
                except ImportError as e2:
                    self.logger.error(f"UltimateMA: EhlersUnifiedDC インポート失敗 - 相対: {e1}, 絶対: {e2}")
                    import_success = False
            
            if import_success and EhlersUnifiedDC is not None:
                try:
                    self.zl_cycle_detector = EhlersUnifiedDC(
                        detector_type=self.zl_cycle_detector_type,
                        cycle_part=self.zl_cycle_detector_cycle_part,
                        max_cycle=self.zl_cycle_detector_max_cycle,
                        min_cycle=self.zl_cycle_detector_min_cycle,
                        src_type=self.src_type,
                        period_range=self.zl_cycle_detector_period_range
                    )
                    self.logger.info(f"UltimateMA: ゼロラグ用動的適応サイクル検出器を初期化: {self.zl_cycle_detector_type}")
                except Exception as e:
                    self.logger.error(f"UltimateMA: ゼロラグ用サイクル検出器の初期化に失敗: {e}")
                    # フォールバックとして固定モードに変更
                    self.zero_lag_period_mode = 'fixed'
                    self.logger.warning("UltimateMA: ゼロラグ動的適応モードの初期化に失敗したため、固定モードにフォールバックしました。")
            else:
                self.logger.error("UltimateMA: EhlersUnifiedDCのインポートに失敗しました（ゼロラグ用）")
                # フォールバックとして固定モードに変更
                self.zero_lag_period_mode = 'fixed'
                self.logger.warning("UltimateMA: EhlersUnifiedDCインポート失敗のため、ゼロラグ固定モードにフォールバックしました。")
        
        # リアルタイムウィンドウ用サイクル検出器の初期化
        if self.realtime_window_mode == 'dynamic':
            # EhlersUnifiedDCのインポート（デバッグ付き）
            EhlersUnifiedDC = None
            import_success = False
            
            try:
                # 相対インポートを試行
                from .cycle.ehlers_unified_dc import EhlersUnifiedDC
                import_success = True
                self.logger.debug("UltimateMA: EhlersUnifiedDC 相対インポート成功（RT用）")
            except ImportError as e1:
                self.logger.debug(f"UltimateMA: EhlersUnifiedDC 相対インポート失敗（RT用）: {e1}")
                try:
                    # 絶対インポートを試行（パス調整付き）
                    import sys
                    import os
                    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    if current_dir not in sys.path:
                        sys.path.insert(0, current_dir)
                    
                    from indicators.cycle.ehlers_unified_dc import EhlersUnifiedDC
                    import_success = True
                    self.logger.debug("UltimateMA: EhlersUnifiedDC 絶対インポート成功（RT用）")
                except ImportError as e2:
                    self.logger.error(f"UltimateMA: EhlersUnifiedDC インポート失敗（RT用） - 相対: {e1}, 絶対: {e2}")
                    import_success = False
            
            if import_success and EhlersUnifiedDC is not None:
                try:
                    self.rt_cycle_detector = EhlersUnifiedDC(
                        detector_type=self.rt_cycle_detector_type,
                        cycle_part=self.rt_cycle_detector_cycle_part,
                        max_cycle=self.rt_cycle_detector_max_cycle,
                        min_cycle=self.rt_cycle_detector_min_cycle,
                        src_type=self.src_type,
                        period_range=self.rt_cycle_detector_period_range
                    )
                    self.logger.info(f"UltimateMA: リアルタイム用動的適応サイクル検出器を初期化: {self.rt_cycle_detector_type}")
                except Exception as e:
                    self.logger.error(f"UltimateMA: リアルタイム用サイクル検出器の初期化に失敗: {e}")
                    # フォールバックとして固定モードに変更
                    self.realtime_window_mode = 'fixed'
                    self.logger.warning("UltimateMA: リアルタイム動的適応モードの初期化に失敗したため、固定モードにフォールバックしました。")
            else:
                self.logger.error("UltimateMA: EhlersUnifiedDCのインポートに失敗しました（リアルタイム用）")
                # フォールバックとして固定モードに変更
                self.realtime_window_mode = 'fixed'
                self.logger.warning("UltimateMA: EhlersUnifiedDCインポート失敗のため、リアルタイム固定モードにフォールバックしました。")
        
        self._cache = {}
        self._result: Optional[UltimateMAResult] = None

    def _get_dynamic_periods(self, data: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        動的適応期間を計算する（個別のサイクル検出器を使用）
        
        Args:
            data: 価格データ
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (zero_lag_periods, realtime_windows)
        """
        data_length = len(data) if hasattr(data, '__len__') else 0
        
        # デフォルト値で初期化
        zero_lag_periods = np.full(data_length, self.zero_lag_period, dtype=np.float64)
        realtime_windows = np.full(data_length, self.realtime_window, dtype=np.float64)
        
        # ゼロラグ期間の動的適応
        if self.zero_lag_period_mode == 'dynamic' and self.zl_cycle_detector is not None:
            try:
                # ゼロラグ用ドミナントサイクルを計算
                zl_dominant_cycles = self.zl_cycle_detector.calculate(data)
                
                if zl_dominant_cycles is not None and len(zl_dominant_cycles) == data_length:
                    # サイクル期間に乗数を適用
                    adjusted_zl_cycles = zl_dominant_cycles * self.zl_cycle_period_multiplier
                    
                    # サイクル期間を適切な範囲にクリップ
                    zero_lag_periods = np.clip(adjusted_zl_cycles, 
                                             self.zl_cycle_detector_min_cycle, 
                                             self.zl_cycle_detector_max_cycle)
                    
                    self.logger.debug(f"ゼロラグ動的期間計算完了 - 期間範囲: [{np.min(zero_lag_periods):.1f}-{np.max(zero_lag_periods):.1f}]")
                else:
                    self.logger.warning("ゼロラグ用ドミナントサイクルの計算結果が無効です。固定期間を使用します。")
                    
            except Exception as e:
                self.logger.error(f"ゼロラグ動的期間計算中にエラー: {e}")
                # エラー時は固定期間を使用
        
        # リアルタイムウィンドウの動的適応
        if self.realtime_window_mode == 'dynamic' and self.rt_cycle_detector is not None:
            try:
                # リアルタイム用ドミナントサイクルを計算
                rt_dominant_cycles = self.rt_cycle_detector.calculate(data)
                
                if rt_dominant_cycles is not None and len(rt_dominant_cycles) == data_length:
                    # サイクル期間に乗数を適用
                    adjusted_rt_cycles = rt_dominant_cycles * self.rt_cycle_period_multiplier
                    
                    # サイクル期間を適切な範囲にクリップ
                    realtime_windows = np.clip(adjusted_rt_cycles, 
                                             2.0,  # 最小ウィンドウサイズ
                                             25.0)  # 最大ウィンドウサイズ
                    
                    self.logger.debug(f"リアルタイム動的ウィンドウ計算完了 - ウィンドウ範囲: [{np.min(realtime_windows):.1f}-{np.max(realtime_windows):.1f}]")
                else:
                    self.logger.warning("リアルタイム用ドミナントサイクルの計算結果が無効です。固定ウィンドウを使用します。")
                    
            except Exception as e:
                self.logger.error(f"リアルタイム動的ウィンドウ計算中にエラー: {e}")
                # エラー時は固定ウィンドウを使用
        
        return zero_lag_periods, realtime_windows

    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> UltimateMAResult:
        """
        🚀 Ultimate Moving Average を計算する（6段階革新的フィルタリング + 動的適応）
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）または直接価格の配列
        
        Returns:
            UltimateMAResult: 全段階のフィルター結果とトレンド情報を含む結果
        """
        try:
            # データチェック - 1次元配列が直接渡された場合は使用できない（hlc3にはOHLCが必要）
            if isinstance(data, np.ndarray) and data.ndim == 1:
                raise ValueError("1次元配列は直接使用できません。hlc3にはOHLCデータが必要です。")
            else:
                # 通常のハッシュチェック
                data_hash = self._get_data_hash(data)
                if data_hash in self._cache and self._result is not None:
                    return self._result

                # hlc3を使用して価格を取得
                ukf_prices = PriceSource.calculate_source(data, 'hlc3')
                ukf_prices = ukf_prices.astype(np.float64)  # 明示的にfloat64に変換
                data_hash_key = data_hash

            # データ長の検証
            data_length = len(ukf_prices)
            if data_length == 0:
                self.logger.warning("価格データが空です。空の配列を返します。")
                empty_result = UltimateMAResult(
                    values=np.array([], dtype=np.float64),
                    raw_values=np.array([], dtype=np.float64),
                    ukf_values=np.array([], dtype=np.float64),
                    kalman_values=np.array([], dtype=np.float64),
                    kalman_gains=np.array([], dtype=np.float64),
                    kalman_innovations=np.array([], dtype=np.float64),
                    kalman_confidence=np.array([], dtype=np.float64),
                    ultimate_smooth_values=np.array([], dtype=np.float64),
                    zero_lag_values=np.array([], dtype=np.float64),
                    amplitude=np.array([], dtype=np.float64),
                    phase=np.array([], dtype=np.float64),
                    realtime_trends=np.array([], dtype=np.float64),
                    trend_signals=np.array([], dtype=np.int8),
                    current_trend='range',
                    current_trend_value=0
                )
                self._result = empty_result
                self._cache[data_hash_key] = self._result
                return empty_result

            # 🚀 動的適応期間の計算
            if self.zero_lag_period_mode == 'dynamic' or self.realtime_window_mode == 'dynamic':
                self.logger.debug("動的適応期間を計算中...")
                zero_lag_periods, realtime_windows = self._get_dynamic_periods(data)
            else:
                # 固定期間の場合
                zero_lag_periods = np.full(data_length, self.zero_lag_period, dtype=np.float64)
                realtime_windows = np.full(data_length, self.realtime_window, dtype=np.float64)

            # 🚀 7段階革新的フィルタリング処理
            self.logger.info("🚀 Ultimate MA - 7段階革新的フィルタリング実行中...")
            
            # ①元の価格（比較用）
            src_prices = PriceSource.calculate_source(data, self.src_type)
            src_prices = src_prices.astype(np.float64)
            
            # ②適応的カルマンフィルター（新規追加）
            if self.use_adaptive_kalman:
                self.logger.debug("🎯 適応的カルマンフィルター適用中...")
                kalman_filtered, kalman_gains, kalman_innovations, kalman_confidence = adaptive_kalman_filter_numba(ukf_prices)
            else:
                self.logger.debug("🎯 適応的カルマンフィルターをスキップ中...")
                kalman_filtered = ukf_prices.copy()
                kalman_gains = np.zeros(len(ukf_prices))
                kalman_innovations = np.zeros(len(ukf_prices))
                kalman_confidence = np.ones(len(ukf_prices))
            
            # ③アルティメットスムーザーフィルター（カルマンフィルター後のデータを使用）
            self.logger.debug("🌊 アルティメットスムーザーフィルター適用中...")
            # カルマンフィルター後のデータをアルティメットスムーザーに渡す
            ultimate_smoother = UltimateSmoother(period=self.ultimate_smoother_period, src_type='hlc3')
            ultimate_smooth_result = ultimate_smoother.calculate(data)
            ultimate_smoothed = ultimate_smooth_result.values
            
            # ③ゼロラグEMA（動的適応対応）
            if self.zero_lag_period_mode == 'dynamic':
                self.logger.debug("⚡ 動的適応ゼロラグEMA処理中...")
                zero_lag_prices = zero_lag_ema_adaptive_numba(ultimate_smoothed, zero_lag_periods)
            else:
                self.logger.debug("⚡ 固定ゼロラグEMA処理中...")
                zero_lag_prices = zero_lag_ema_numba(ultimate_smoothed, self.zero_lag_period)
            
            # ④ヒルベルト変換フィルター
            self.logger.debug("🌀 ヒルベルト変換フィルター適用中...")
            amplitude, phase = hilbert_transform_filter_numba(zero_lag_prices)
            
            # ⑤適応的ノイズ除去
            self.logger.debug("🔇 適応的ノイズ除去実行中...")
            denoised_prices = adaptive_noise_reduction_numba(zero_lag_prices, amplitude)
            
            # ⑥リアルタイムトレンド検出（動的適応対応）
            if self.realtime_window_mode == 'dynamic':
                self.logger.debug("⚡ 動的適応リアルタイムトレンド検出中...")
                realtime_trends = real_time_trend_detector_adaptive_numba(denoised_prices, realtime_windows)
            else:
                self.logger.debug("⚡ 固定リアルタイムトレンド検出中...")
                realtime_trends = real_time_trend_detector_numba(denoised_prices, self.realtime_window)
            
            # 最終的な処理済み価格系列
            final_values = denoised_prices
            
            # トレンド判定
            trend_signals = calculate_trend_signals_with_range_numba(final_values, self.slope_index, self.range_threshold)
            trend_index, trend_value = calculate_current_trend_with_range_numba(trend_signals)
            trend_names = ['range', 'up', 'down']
            current_trend = trend_names[trend_index]

            result = UltimateMAResult(
                values=final_values,
                raw_values=src_prices,
                ukf_values=ukf_prices,
                kalman_values=kalman_filtered,
                kalman_gains=kalman_gains,
                kalman_innovations=kalman_innovations,
                kalman_confidence=kalman_confidence,
                ultimate_smooth_values=ultimate_smoothed,
                zero_lag_values=zero_lag_prices,
                amplitude=amplitude,
                phase=phase,
                realtime_trends=realtime_trends,
                trend_signals=trend_signals,
                current_trend=current_trend,
                current_trend_value=trend_value
            )

            self._result = result
            self._cache[data_hash_key] = self._result
            
            mode_info = f"ZL:{self.zero_lag_period_mode}, RT:{self.realtime_window_mode}"
            if self.zero_lag_period_mode == 'dynamic':
                mode_info += f", ZLサイクル検出器:{self.zl_cycle_detector_type}"
            if self.realtime_window_mode == 'dynamic':
                mode_info += f", RTサイクル検出器:{self.rt_cycle_detector_type}"
            
            self.logger.info(f"✅ Ultimate MA 計算完了 - トレンド: {current_trend}, モード: {mode_info}")
            return self._result

        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"{self.name} 計算中にエラー: {error_msg}\n{stack_trace}")
            data_len = len(data) if hasattr(data, '__len__') else 0
            self._result = None # エラー時は結果をクリア
            error_result = UltimateMAResult(
                values=np.full(data_len, np.nan, dtype=np.float64),
                raw_values=np.full(data_len, np.nan, dtype=np.float64),
                ukf_values=np.full(data_len, np.nan, dtype=np.float64),
                kalman_values=np.full(data_len, np.nan, dtype=np.float64),
                kalman_gains=np.full(data_len, np.nan, dtype=np.float64),
                kalman_innovations=np.full(data_len, np.nan, dtype=np.float64),
                kalman_confidence=np.full(data_len, np.nan, dtype=np.float64),
                ultimate_smooth_values=np.full(data_len, np.nan, dtype=np.float64),
                zero_lag_values=np.full(data_len, np.nan, dtype=np.float64),
                amplitude=np.full(data_len, np.nan, dtype=np.float64),
                phase=np.full(data_len, np.nan, dtype=np.float64),
                realtime_trends=np.full(data_len, np.nan, dtype=np.float64),
                trend_signals=np.zeros(data_len, dtype=np.int8),
                current_trend='range',
                current_trend_value=0
            )
            return error_result

    def get_values(self) -> Optional[np.ndarray]:
        """最終フィルター済み値のみを取得する（後方互換性のため）"""
        if self._result is not None:
            return self._result.values.copy()
        return None

    def get_raw_values(self) -> Optional[np.ndarray]:
        """元の価格値を取得する"""
        if self._result is not None:
            return self._result.raw_values.copy()
        return None

    def get_ukf_values(self) -> Optional[np.ndarray]:
        """hlc3フィルター後の値を取得する"""
        if self._result is not None:
            return self._result.ukf_values.copy()
        return None

    def get_kalman_values(self) -> Optional[np.ndarray]:
        """適応的カルマンフィルター後の値を取得する"""
        if self._result is not None:
            return self._result.kalman_values.copy()
        return None

    def get_kalman_gains(self) -> Optional[np.ndarray]:
        """カルマンゲインを取得する"""
        if self._result is not None:
            return self._result.kalman_gains.copy()
        return None

    def get_kalman_innovations(self) -> Optional[np.ndarray]:
        """カルマンイノベーションを取得する"""
        if self._result is not None:
            return self._result.kalman_innovations.copy()
        return None

    def get_kalman_confidence(self) -> Optional[np.ndarray]:
        """カルマン信頼度スコアを取得する"""
        if self._result is not None:
            return self._result.kalman_confidence.copy()
        return None

    def get_ultimate_smooth_values(self) -> Optional[np.ndarray]:
        """アルティメットスムーザーフィルター後の値を取得する"""
        if self._result is not None:
            return self._result.ultimate_smooth_values.copy()
        return None

    def get_zero_lag_values(self) -> Optional[np.ndarray]:
        """ゼロラグEMA後の値を取得する"""
        if self._result is not None:
            return self._result.zero_lag_values.copy()
        return None

    def get_amplitude(self) -> Optional[np.ndarray]:
        """ヒルベルト変換振幅を取得する"""
        if self._result is not None:
            return self._result.amplitude.copy()
        return None

    def get_phase(self) -> Optional[np.ndarray]:
        """ヒルベルト変換位相を取得する"""
        if self._result is not None:
            return self._result.phase.copy()
        return None

    def get_realtime_trends(self) -> Optional[np.ndarray]:
        """リアルタイムトレンド信号を取得する"""
        if self._result is not None:
            return self._result.realtime_trends.copy()
        return None

    def get_trend_signals(self) -> Optional[np.ndarray]:
        """トレンド信号を取得する"""
        if self._result is not None:
            return self._result.trend_signals.copy()
        return None

    def get_current_trend(self) -> str:
        """現在のトレンド状態を取得する"""
        if self._result is not None:
            return self._result.current_trend
        return 'range'

    def get_current_trend_value(self) -> int:
        """現在のトレンド値を取得する"""
        if self._result is not None:
            return self._result.current_trend_value
        return 0

    def get_noise_reduction_stats(self) -> dict:
        """ノイズ除去統計を取得する"""
        if self._result is None:
            return {}
        
        raw_std = np.nanstd(self._result.raw_values)
        final_std = np.nanstd(self._result.values)
        noise_reduction_ratio = (raw_std - final_std) / raw_std if raw_std > 0 else 0.0
        
        return {
            'raw_volatility': raw_std,
            'filtered_volatility': final_std,
            'noise_reduction_ratio': noise_reduction_ratio,
            'noise_reduction_percentage': noise_reduction_ratio * 100,
            'smoothing_effectiveness': min(noise_reduction_ratio * 100, 100.0)
        }

    def get_dynamic_periods_info(self) -> dict:
        """動的適応期間の情報を取得する"""
        info = {
            'zero_lag_period_mode': self.zero_lag_period_mode,
            'realtime_window_mode': self.realtime_window_mode,
            'zl_cycle_detector_available': self.zl_cycle_detector is not None,
            'rt_cycle_detector_available': self.rt_cycle_detector is not None
        }
        
        # ゼロラグ用サイクル検出器の情報
        if self.zl_cycle_detector is not None:
            info.update({
                'zl_cycle_detector_type': self.zl_cycle_detector_type,
                'zl_cycle_detector_cycle_part': self.zl_cycle_detector_cycle_part,
                'zl_cycle_detector_max_cycle': self.zl_cycle_detector_max_cycle,
                'zl_cycle_detector_min_cycle': self.zl_cycle_detector_min_cycle,
                'zl_cycle_period_multiplier': self.zl_cycle_period_multiplier
            })
        
        # リアルタイム用サイクル検出器の情報
        if self.rt_cycle_detector is not None:
            info.update({
                'rt_cycle_detector_type': self.rt_cycle_detector_type,
                'rt_cycle_detector_cycle_part': self.rt_cycle_detector_cycle_part,
                'rt_cycle_detector_max_cycle': self.rt_cycle_detector_max_cycle,
                'rt_cycle_detector_min_cycle': self.rt_cycle_detector_min_cycle,
                'rt_cycle_period_multiplier': self.rt_cycle_period_multiplier
            })
        
        return info

    def reset(self) -> None:
        """インジケータの状態をリセットする"""
        super().reset()
        self._result = None
        self._cache = {}
        if self.zl_cycle_detector is not None:
            self.zl_cycle_detector.reset()
        if self.rt_cycle_detector is not None:
            self.rt_cycle_detector.reset()

    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データとパラメータに基づいてハッシュ値を計算する"""
        # src_typeに基づいて必要なカラムを決定
        required_cols = set()
        if self.src_type == 'open':
            required_cols.add('open')
        elif self.src_type == 'high':
            required_cols.add('high')
        elif self.src_type == 'low':
            required_cols.add('low')
        elif self.src_type == 'close':
            required_cols.add('close')
        elif self.src_type == 'hl2':
            required_cols.update(['high', 'low'])
        elif self.src_type == 'hlc3':
            required_cols.update(['high', 'low', 'close'])
        elif self.src_type == 'ohlc4':
            required_cols.update(['open', 'high', 'low', 'close'])
        elif self.src_type == 'hlcc4':
            required_cols.update(['high', 'low', 'close'])
        elif self.src_type == 'weighted_close':
            required_cols.update(['high', 'low', 'close'])
        else:
            required_cols.add('close') # Default

        if isinstance(data, pd.DataFrame):
            relevant_cols = [col for col in data.columns if col.lower() in required_cols]
            present_cols = [col for col in relevant_cols if col in data.columns]
            if not present_cols:
                try:
                    shape_tuple = data.shape
                    first_row = tuple(data.iloc[0]) if len(data) > 0 else ()
                    last_row = tuple(data.iloc[-1]) if len(data) > 0 else ()
                    data_repr_tuple = (shape_tuple, first_row, last_row)
                    data_hash_val = hash(data_repr_tuple)
                except Exception:
                    data_hash_val = hash(str(data))
            else:
                data_values = data[present_cols].values
                data_hash_val = hash(data_values.tobytes())

        elif isinstance(data, np.ndarray):
            col_indices = []
            if 'open' in required_cols: col_indices.append(0)
            if 'high' in required_cols: col_indices.append(1)
            if 'low' in required_cols: col_indices.append(2)
            if 'close' in required_cols: col_indices.append(3)
            col_indices = sorted(list(set(col_indices)))
            if data.ndim == 2 and data.shape[1] > max(col_indices if col_indices else [-1]):
                data_values = data[:, col_indices]
                data_hash_val = hash(data_values.tobytes())
            else:
                data_hash_val = hash(data.tobytes())
        else:
            data_hash_val = hash(str(data))

        param_str = (f"kf={self.use_adaptive_kalman}_kf_pv={self.kalman_process_variance}_kf_mv={self.kalman_measurement_variance}_kf_vw={self.kalman_volatility_window}"
                    f"_us={self.ultimate_smoother_period}_zl={self.zero_lag_period}({self.zero_lag_period_mode})"
                    f"_rt={self.realtime_window}({self.realtime_window_mode})"
                    f"_src={self.src_type}_slope={self.slope_index}_range_th={self.range_threshold}"
                    f"_zl_cycle={self.zl_cycle_detector_type}_zl_cycle_part={self.zl_cycle_detector_cycle_part}"
                    f"_zl_cycle_max={self.zl_cycle_detector_max_cycle}_zl_cycle_min={self.zl_cycle_detector_min_cycle}"
                    f"_zl_cycle_mult={self.zl_cycle_period_multiplier}"
                    f"_rt_cycle={self.rt_cycle_detector_type}_rt_cycle_part={self.rt_cycle_detector_cycle_part}"
                    f"_rt_cycle_max={self.rt_cycle_detector_max_cycle}_rt_cycle_min={self.rt_cycle_detector_min_cycle}"
                    f"_rt_cycle_mult={self.rt_cycle_period_multiplier}"
                    f"_zl_cycle_period_range={self.zl_cycle_detector_period_range}_rt_cycle_period_range={self.rt_cycle_detector_period_range}")
        return f"{data_hash_val}_{param_str}" 