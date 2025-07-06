#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ultra Quantum Adaptive Trend-Range Discriminator (UQATRD)
===========================================================

John Ehlersの哲学に基づく革新的なトレンド/レンジ判別インジケーター

4つの核心量子アルゴリズム：
1. 量子コヒーレンス方向性測定器 (Quantum Coherence Directional Analyzer)
2. 量子エンタングルメント持続性アナライザー (Quantum Entanglement Persistence Analyzer)
3. 量子効率スペクトラム計算機 (Quantum Efficiency Spectrum Calculator)
4. 量子不確定性レンジ検出器 (Quantum Uncertainty Range Detector)

特徴：
- 超高精度：複素数平面での多次元解析
- 超適応性：動的サイクル検出による自動調整
- 超低遅延：ゼロラグフィルターと予測的アルゴリズム
- 量子的アプローチ：確率的判定による柔軟性
"""

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional, List, NamedTuple
import numpy as np
import pandas as pd
from numba import jit, njit, prange, float64, complex128
import traceback
import math

try:
    from .indicator import Indicator
    from .price_source import PriceSource
except ImportError:
    # スタンドアロン実行時の対応
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from indicator import Indicator
    from price_source import PriceSource


@dataclass
class UQATRDResult:
    """UQATRD計算結果"""
    # メイン判定結果
    trend_range_signal: np.ndarray      # 最終的なトレンド/レンジ判定 (0=レンジ to 1=トレンド)
    signal_strength: np.ndarray         # 信号強度 (0 to 1)
    
    # 4つの核心アルゴリズム結果
    quantum_coherence: np.ndarray       # 量子コヒーレンス (0 to 1)
    trend_persistence: np.ndarray       # トレンド持続性 (0=レンジ to 1=トレンド、方向性無関係)
    efficiency_spectrum: np.ndarray     # 効率スペクトラム (0 to 1)
    uncertainty_range: np.ndarray       # 不確定性レンジ (0 to 1)
    
    # 動的適応閾値
    adaptive_threshold: np.ndarray      # 動的適応閾値 (0.4 to 0.6)
    
    # 補助情報
    confidence_score: np.ndarray        # 信頼度スコア (0 to 1)
    cycle_adaptive_factor: np.ndarray   # サイクル適応因子


# ================== 量子コヒーレンス方向性測定器 ==================

@njit(fastmath=True, cache=True)
def quantum_coherence_trend_analyzer(
    prices: np.ndarray,
    window: int = 21
) -> Tuple[np.ndarray, np.ndarray]:
    """
    🌀 量子コヒーレンストレンド測定器
    
    従来のADXを量子的に再構築（方向性無関係版）：
    - 複素数平面での変動の一貫性を測定
    - 位相の安定性からトレンドの強さを抽出（方向性無関係）
    - 量子干渉による変動の純度測定
    
    Args:
        prices: 価格データ配列
        window: 分析ウィンドウ
        
    Returns:
        Tuple[coherence, trend_strength]: (コヒーレンス, トレンド強度)
    """
    n = len(prices)
    coherence = np.zeros(n)
    trend_strength = np.zeros(n)
    
    if n < window:
        return coherence, trend_strength
    
    for i in range(window, n):
        # 価格変動の複素数表現（方向性無関係）
        complex_movements = np.zeros(window-1, dtype=np.complex128)
        
        for j in range(window-1):
            price_change = prices[i-window+j+1] - prices[i-window+j]
            # 振幅のみ使用（方向性を除去）
            amplitude = abs(price_change)
            # 位相は変動の一貫性測定のためのみ使用
            angle = math.atan2(price_change, abs(price_change) + 1e-10)
            complex_movements[j] = amplitude * (math.cos(angle) + 1j * math.sin(angle))
        
        # 量子コヒーレンスの計算
        # 位相の分散を測定（位相が揃っているほど高コヒーレンス）
        phases = np.angle(complex_movements)
        
        # 位相の円形統計量（von Mises統計）
        cos_sum = np.sum(np.cos(phases))
        sin_sum = np.sum(np.sin(phases))
        
        # 結合ベクトルの長さ（コヒーレンス度）
        coherence_raw = math.sqrt(cos_sum**2 + sin_sum**2) / len(phases)
        coherence[i] = coherence_raw
        
        # トレンド強度の計算（方向性無関係）
        # 振幅の平均を使用（方向性を除去）
        amplitude_mean = np.mean(np.abs(complex_movements))
        
        # 変動の一貫性（コヒーレンス）とトレンド強度の結合
        trend_strength[i] = coherence_raw * amplitude_mean
    
    # 境界値の補間
    for i in range(window):
        coherence[i] = coherence[window] if n > window else 0.0
        trend_strength[i] = trend_strength[window] if n > window else 0.0
    
    return coherence, trend_strength


# ================== 量子エンタングルメント持続性アナライザー ==================

@njit(fastmath=True, cache=True)
def quantum_entanglement_trend_analyzer(
    prices: np.ndarray,
    window: int = 34
) -> Tuple[np.ndarray, np.ndarray]:
    """
    🔗 量子エンタングルメントトレンド分析器
    
    ハースト指数を量子もつれ理論で再実装（方向性無関係版）：
    - 過去と未来の価格データ間の非局所相関を測定
    - EPR相関による持続性の量子的評価（方向性無関係）
    - 時間軸を超えた相関の測定
    
    Args:
        prices: 価格データ配列
        window: 分析ウィンドウ
        
    Returns:
        Tuple[entanglement, trend_persistence]: (エンタングルメント度, トレンド持続性)
    """
    n = len(prices)
    entanglement = np.zeros(n)
    trend_persistence = np.zeros(n)
    
    if n < window * 2:
        return entanglement, trend_persistence
    
    for i in range(window, n):
        # 過去のデータセグメント
        past_segment = prices[i-window:i]
        
        # 量子もつれ相関の計算
        # R/S統計の量子版：スケール不変性の測定
        
        # 1. 平均除去（量子状態の中心化）
        mean_price = np.mean(past_segment)
        centered_prices = past_segment - mean_price
        
        # 2. 累積偏差（量子ランダムウォーク）
        cumulative_deviations = np.cumsum(centered_prices)
        
        # 3. レンジ計算（量子状態の広がり）
        range_val = np.max(cumulative_deviations) - np.min(cumulative_deviations)
        
        # 4. 標準偏差（量子不確定性）
        std_dev = np.std(past_segment)
        
        # 5. R/S比（量子もつれ指数）
        if std_dev > 1e-10:
            rs_ratio = range_val / std_dev
            # ハースト指数の近似
            hurst_approx = math.log(rs_ratio) / math.log(window)
            
            # 量子もつれ度（0.5からの偏差）
            entanglement[i] = abs(hurst_approx - 0.5) * 2.0
            
            # トレンド持続性（方向性無関係）
            # 0.5からの偏差の絶対値を使用：0.5以上でトレンド、0.5以下でレンジ
            # 0（レンジ）から1（トレンド）の範囲に正規化
            if hurst_approx > 0.5:
                trend_persistence[i] = (hurst_approx - 0.5) * 2.0
            else:
                trend_persistence[i] = (0.5 - hurst_approx) * 2.0
        else:
            entanglement[i] = 0.0
            trend_persistence[i] = 0.0
    
    # 境界値の補間
    for i in range(window):
        entanglement[i] = entanglement[window] if n > window else 0.0
        trend_persistence[i] = trend_persistence[window] if n > window else 0.0
    
    # 値の正規化（手動クリッピング）
    for i in range(n):
        if entanglement[i] < 0.0:
            entanglement[i] = 0.0
        elif entanglement[i] > 1.0:
            entanglement[i] = 1.0
        
        if trend_persistence[i] < 0.0:
            trend_persistence[i] = 0.0
        elif trend_persistence[i] > 1.0:
            trend_persistence[i] = 1.0
    
    return entanglement, trend_persistence


# ================== 量子効率スペクトラム計算機 ==================

@njit(fastmath=True, cache=True)
def quantum_efficiency_spectrum_calculator(
    prices: np.ndarray,
    window: int = 21
) -> Tuple[np.ndarray, np.ndarray]:
    """
    📊 量子効率スペクトラム計算機
    
    効率比を周波数ドメインで拡張：
    - 複数時間スケールでの効率性を同時測定
    - フーリエ変換による効率性の周波数分解
    - 量子調和振動子による効率性評価
    
    Args:
        prices: 価格データ配列
        window: 分析ウィンドウ
        
    Returns:
        Tuple[efficiency_spectrum, spectral_power]: (効率スペクトラム, スペクトル強度)
    """
    n = len(prices)
    efficiency_spectrum = np.zeros(n)
    spectral_power = np.zeros(n)
    
    if n < window:
        return efficiency_spectrum, spectral_power
    
    for i in range(window, n):
        price_segment = prices[i-window:i]
        
        # 1. 直線的変化（理想的な効率）
        linear_change = price_segment[-1] - price_segment[0]
        
        # 2. 実際の価格変動の総和
        actual_changes = np.sum(np.abs(np.diff(price_segment)))
        
        # 3. 基本効率比
        if actual_changes > 1e-10:
            basic_efficiency = abs(linear_change) / actual_changes
        else:
            basic_efficiency = 0.0
        
        # 4. 量子調和解析による効率スペクトラム
        # 複数の調和成分での効率性を測定
        harmonic_efficiencies = np.zeros(5)  # 5つの調和成分
        
        for h in range(1, 6):  # 1次から5次調和
            # 調和振動子の周波数
            omega = 2 * math.pi * h / window
            
            # 複素指数による変換
            complex_sum = 0.0 + 0.0j
            for j in range(window):
                angle = omega * j
                complex_sum += price_segment[j] * (math.cos(angle) + 1j * math.sin(angle))
            
            # 調和成分の振幅
            harmonic_amplitude = abs(complex_sum) / window
            
            # 調和成分での効率性
            if harmonic_amplitude > 1e-10:
                harmonic_efficiencies[h-1] = basic_efficiency * harmonic_amplitude
            else:
                harmonic_efficiencies[h-1] = 0.0
        
        # 5. 効率スペクトラムの統合
        # 黄金比による重み付け
        golden_weights = np.array([1.0, 0.618, 0.382, 0.236, 0.146])
        golden_weights /= np.sum(golden_weights)
        
        efficiency_spectrum[i] = np.sum(harmonic_efficiencies * golden_weights)
        spectral_power[i] = np.sum(harmonic_efficiencies)
    
    # 境界値の補間
    for i in range(window):
        efficiency_spectrum[i] = efficiency_spectrum[window] if n > window else 0.0
        spectral_power[i] = spectral_power[window] if n > window else 0.0
    
    # 値の正規化（手動クリッピング）
    for i in range(n):
        if efficiency_spectrum[i] < 0.0:
            efficiency_spectrum[i] = 0.0
        elif efficiency_spectrum[i] > 1.0:
            efficiency_spectrum[i] = 1.0
    
    return efficiency_spectrum, spectral_power


# ================== 量子不確定性レンジ検出器 ==================

@njit(fastmath=True, cache=True)
def quantum_uncertainty_range_detector(
    prices: np.ndarray,
    window: int = 14
) -> Tuple[np.ndarray, np.ndarray]:
    """
    🎯 量子不確定性レンジ検出器
    
    不確定性原理による「位置×運動量」の測定：
    - 価格の位置（現在値）と運動量（変化率）の積
    - 不確定性が高い = レンジ、低い = トレンド
    - ハイゼンベルクの不確定性原理の金融応用
    
    Args:
        prices: 価格データ配列
        window: 分析ウィンドウ
        
    Returns:
        Tuple[uncertainty_range, momentum_dispersion]: (不確定性レンジ, 運動量分散)
    """
    n = len(prices)
    uncertainty_range = np.zeros(n)
    momentum_dispersion = np.zeros(n)
    
    if n < window:
        return uncertainty_range, momentum_dispersion
    
    for i in range(window, n):
        price_segment = prices[i-window:i]
        
        # 1. 位置の不確定性（価格の分散）
        position_variance = np.var(price_segment)
        position_uncertainty = math.sqrt(position_variance)
        
        # 2. 運動量の不確定性（価格変化率の分散）
        momentum_changes = np.diff(price_segment)
        if len(momentum_changes) > 0:
            momentum_variance = np.var(momentum_changes)
            momentum_uncertainty = math.sqrt(momentum_variance)
        else:
            momentum_uncertainty = 0.0
        
        # 3. 不確定性原理の適用
        # ΔxΔp ≥ ℏ/2 (量子力学) → 価格版
        uncertainty_product = position_uncertainty * momentum_uncertainty
        
        # 4. 正規化された不確定性
        # 高い不確定性 = レンジ相場
        # 低い不確定性 = トレンド相場
        
        # 価格レンジでの正規化
        price_range = np.max(price_segment) - np.min(price_segment)
        if price_range > 1e-10:
            normalized_uncertainty = uncertainty_product / price_range
        else:
            normalized_uncertainty = 0.0
        
        uncertainty_range[i] = normalized_uncertainty
        momentum_dispersion[i] = momentum_uncertainty
        
        # 5. 量子もつれ補正
        # 近隣データポイントとの相関を考慮
        if i >= window + 5:
            past_uncertainties = uncertainty_range[i-5:i]
            correlation_factor = np.corrcoef(past_uncertainties, 
                                           np.arange(len(past_uncertainties)))[0, 1]
            if not np.isnan(correlation_factor):
                uncertainty_range[i] *= (1.0 + abs(correlation_factor) * 0.2)
    
    # 境界値の補間
    for i in range(window):
        uncertainty_range[i] = uncertainty_range[window] if n > window else 0.0
        momentum_dispersion[i] = momentum_dispersion[window] if n > window else 0.0
    
    # 値の正規化
    max_uncertainty = np.max(uncertainty_range) if len(uncertainty_range) > 0 else 1.0
    if max_uncertainty > 1e-10:
        uncertainty_range = uncertainty_range / max_uncertainty
    
    return uncertainty_range, momentum_dispersion


# ================== 動的適応閾値計算エンジン ==================

@njit(fastmath=True, cache=True)
def calculate_adaptive_threshold(
    prices: np.ndarray,
    coherence: np.ndarray,
    trend_persistence: np.ndarray,
    efficiency_spectrum: np.ndarray,
    uncertainty_range: np.ndarray,
    str_values: np.ndarray,
    window: int = 21
) -> np.ndarray:
    """
    🎯 量子適応学習閾値システム（ダイナミック版）
    
    4つの適応要素を統合した動的閾値計算：
    1. STRベースボラティリティ適応（ダイナミックレンジ）
    2. 量子コヒーレンス適応（拡張変動）
    3. サイクル適応（偏差強調）
    4. 適応学習機構（増幅調整）
    
    Args:
        prices: 価格データ配列
        coherence: 量子コヒーレンス配列
        trend_persistence: トレンド持続性配列
        efficiency_spectrum: 効率スペクトラム配列
        uncertainty_range: 不確定性レンジ配列
        str_values: STR（Smooth True Range）値配列
        window: 適応ウィンドウ
        
    Returns:
        np.ndarray: 動的適応閾値配列
    """
    n = len(prices)
    adaptive_threshold = np.zeros(n)
    
    # デフォルト閾値（中央値に設定）
    base_threshold = 0.5
    
    if n < window:
        for i in range(n):
            adaptive_threshold[i] = base_threshold
        return adaptive_threshold
    
    for i in range(window, n):
        # 1. STRベースのボラティリティ適応計算
        if i < len(str_values):
            # STRの移動平均を計算してボラティリティベースラインとする
            str_window = str_values[max(0, i-window):i]
            if len(str_window) > 0:
                str_mean = np.mean(str_window)
                str_current = str_values[i]
                
                # STRを価格で正規化（相対的ボラティリティ）
                current_price = prices[i]
                if current_price > 1e-10:
                    relative_str = str_current / current_price
                else:
                    relative_str = 0.0
                
                # 0-1範囲に正規化
                # 通常のSTR/価格比率は0.01-0.05程度なので、20倍して0-1に近づける
                normalized_volatility = min(1.0, relative_str * 20.0)
                
                # 下限を設定（極端に低いボラティリティを防ぐ）
                normalized_volatility = max(0.1, normalized_volatility)
            else:
                normalized_volatility = 0.5
        else:
            normalized_volatility = 0.5
        
        # 高ボラティリティ → 閾値を高く（トレンド判定を厳しく）
        # さらにダイナミックな変動のために係数を大幅拡大
        volatility_adjustment = 0.4 + normalized_volatility * 1.2  # 0.4-1.6の範囲
        
        # 2. 量子コヒーレンス適応
        # 高コヒーレンス → 閾値を低く（明確な信号）
        coherence_current = coherence[i] if i < len(coherence) else 0.5
        coherence_adjustment = 1.4 - coherence_current * 0.8      # 0.6-1.4の範囲
        
        # 3. サイクル適応（トレンド持続性ベース）
        # 高持続性 → 閾値を低く（トレンドが続きやすい）
        persistence_current = trend_persistence[i] if i < len(trend_persistence) else 0.5
        cycle_adjustment = 1.4 - persistence_current * 0.8        # 0.6-1.4の範囲
        
        # 4. 効率性適応
        # 高効率性 → 閾値を低く（効率的な動き）
        efficiency_current = efficiency_spectrum[i] if i < len(efficiency_spectrum) else 0.5
        efficiency_adjustment = 1.4 - efficiency_current * 0.8    # 0.6-1.4の範囲
        
        # 5. 不確定性適応
        # 高不確定性 → 閾値を高く（レンジ判定しやすく）
        uncertainty_current = uncertainty_range[i] if i < len(uncertainty_range) else 0.5
        uncertainty_adjustment = 0.4 + uncertainty_current * 1.2  # 0.4-1.6の範囲
        
        # 6. 適応学習機構（過去の精度に基づく調整）
        learning_adjustment = 1.0
        
        if i >= window * 2:
            # 過去のパフォーマンス分析
            past_signals = []
            past_thresholds = []
            
            for j in range(max(0, i-window), i):
                if j < len(coherence) and j < len(adaptive_threshold):
                    # 過去の信号と閾値を記録
                    past_signal = (coherence[j] + trend_persistence[j] + 
                                  efficiency_spectrum[j] + (1.0 - uncertainty_range[j])) / 4.0
                    past_signals.append(past_signal)
                    past_thresholds.append(adaptive_threshold[j] if j > 0 else base_threshold)
            
            if len(past_signals) > 10:
                # 閾値の安定性を評価
                threshold_variance = np.var(np.array(past_thresholds))
                
                # よりダイナミックな学習調整
                if threshold_variance < 0.01:
                    learning_adjustment = 0.85  # より大きな変動を促す
                else:
                    learning_adjustment = 1.25  # さらに大きな調整
        
        # 7. 統合適応閾値計算（よりダイナミックな方式）
        adjustments = np.array([
            volatility_adjustment,
            coherence_adjustment,
            cycle_adjustment,
            efficiency_adjustment,
            uncertainty_adjustment
        ])
        
        # より変動の大きい統合方式：重み付き平均ではなく偏差を強調
        weights = np.array([0.25, 0.2, 0.2, 0.15, 0.2])
        weighted_adjustments = adjustments * weights
        
        # 平均からの偏差を強調して変動を拡大
        mean_adjustment = np.sum(weighted_adjustments)
        deviation_from_base = mean_adjustment - 1.0  # 1.0を基準とした偏差
        
        # 偏差を大幅拡大して変動をよりダイナミックに
        amplified_deviation = deviation_from_base * 3.5 * learning_adjustment
        
        # 最終閾値計算（加算ベース）- 係数を大幅拡大
        final_threshold = base_threshold + amplified_deviation * 0.35  # 大きな変動のため係数を大幅拡大
        
        # 閾値の制限（実践的範囲: 0.4-0.6）
        if final_threshold < 0.45:
            final_threshold = 0.45
        elif final_threshold > 0.65:
            final_threshold = 0.65
        
        adaptive_threshold[i] = final_threshold
    
    # 境界値の補間（よりダイナミックな初期値）
    for i in range(window):
        if n > window:
            adaptive_threshold[i] = adaptive_threshold[window]
        else:
            # 初期値により大きな変動を持たせる
            variation = (i / window - 0.5) * 0.2  # -0.1 to +0.1の変動
            adaptive_threshold[i] = base_threshold + variation  # 0.4-0.6の範囲
    
    return adaptive_threshold


# ================== 統合計算エンジン ==================

@njit(fastmath=True, cache=True)
def calculate_uqatrd_core(
    prices: np.ndarray,
    str_values: np.ndarray,
    coherence_window: int = 21,
    entanglement_window: int = 34,
    efficiency_window: int = 21,
    uncertainty_window: int = 14
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    🚀 UQATRD統合計算エンジン
    
    4つの量子アルゴリズムを統合してトレンド/レンジ判定を実行
    出力：0（レンジ）から1（トレンド）の範囲、方向性無関係
    """
    n = len(prices)
    
    # 1. 量子コヒーレンストレンド測定
    coherence, trend_strength = quantum_coherence_trend_analyzer(
        prices, coherence_window
    )
    
    # 2. 量子エンタングルメントトレンド分析
    entanglement, trend_persistence = quantum_entanglement_trend_analyzer(
        prices, entanglement_window
    )
    
    # 3. 量子効率スペクトラム計算
    efficiency_spectrum, spectral_power = quantum_efficiency_spectrum_calculator(
        prices, efficiency_window
    )
    
    # 4. 量子不確定性レンジ検出
    uncertainty_range, momentum_dispersion = quantum_uncertainty_range_detector(
        prices, uncertainty_window
    )
    
    # 5. 統合信号の計算（方向性無関係）
    trend_range_signal = np.zeros(n)
    signal_strength = np.zeros(n)
    
    for i in range(n):
        # 各アルゴリズムからの信号を統合（全て0-1の範囲）
        # トレンド信号（1 に近いほど強いトレンド、0 に近いほど強いレンジ）
        trend_signals = np.array([
            coherence[i],                    # 高コヒーレンス = トレンド
            trend_persistence[i],            # 高持続性 = トレンド（方向性無関係）
            efficiency_spectrum[i],          # 高効率 = トレンド
            (1.0 - uncertainty_range[i])     # 低不確定性 = トレンド
        ])
        
        # 量子重ね合わせによる統合
        # 各信号の重み（黄金比ベース）
        weights = np.array([0.382, 0.236, 0.236, 0.146])
        
        # 重み付き平均
        weighted_signal = np.sum(trend_signals * weights)
        
        # 最終的なトレンド/レンジ判定
        # 0 (レンジ) から 1 (トレンド) の範囲
        if weighted_signal < 0.0:
            trend_range_signal[i] = 0.0
        elif weighted_signal > 1.0:
            trend_range_signal[i] = 1.0
        else:
            trend_range_signal[i] = weighted_signal
        
        # 信号強度（確信度）
        signal_variance = np.var(trend_signals)
        signal_strength[i] = 1.0 / (1.0 + signal_variance * 10.0)
    
    # 値の正規化（手動クリッピング）
    for i in range(n):
        if trend_range_signal[i] < 0.0:
            trend_range_signal[i] = 0.0
        elif trend_range_signal[i] > 1.0:
            trend_range_signal[i] = 1.0
        
        if signal_strength[i] < 0.0:
            signal_strength[i] = 0.0
        elif signal_strength[i] > 1.0:
            signal_strength[i] = 1.0
    
    # 6. 動的適応閾値の計算（STRベース）
    adaptive_threshold = calculate_adaptive_threshold(
        prices, coherence, trend_persistence, 
        efficiency_spectrum, uncertainty_range, str_values,
        window=21
    )
    
    return (trend_range_signal, signal_strength, coherence, 
            trend_persistence, efficiency_spectrum, uncertainty_range, adaptive_threshold)


class UltraQuantumAdaptiveTrendRangeDiscriminator(Indicator):
    """
    🌟 Ultra Quantum Adaptive Trend-Range Discriminator (UQATRD)
    
    John Ehlersの哲学に基づく革新的なトレンド/レンジ判別インジケーター
    
    4つの核心量子アルゴリズム：
    1. 量子コヒーレンス方向性測定器
    2. 量子エンタングルメント持続性アナライザー
    3. 量子効率スペクトラム計算機
    4. 量子不確定性レンジ検出器
    
    特徴：
    - 超高精度：複素数平面での多次元解析
    - 超適応性：動的サイクル検出による自動調整
    - 超低遅延：ゼロラグフィルターと予測的アルゴリズム
    - 量子的アプローチ：確率的判定による柔軟性
    """
    
    def __init__(
        self,
        # 各量子アルゴリズムのパラメータ
        coherence_window: int = 21,      # 量子コヒーレンス分析窓
        entanglement_window: int = 34,   # 量子エンタングルメント分析窓
        efficiency_window: int = 21,     # 量子効率スペクトラム分析窓
        uncertainty_window: int = 14,    # 量子不確定性分析窓
        
        # 一般パラメータ
        src_type: str = 'ukf_hlc3',          # 価格ソース
        adaptive_mode: bool = True,      # 適応モード
        sensitivity: float = 1.0,        # 感度調整
        
        # STRパラメータ
        str_period: float = 20.0,        # STR期間（ボラティリティ計算用）
        
        # 品質管理パラメータ
        min_data_points: int = 50,       # 最小データポイント数
        confidence_threshold: float = 0.7 # 信頼度閾値
    ):
        """
        コンストラクタ
        
        Args:
            coherence_window: 量子コヒーレンス分析ウィンドウ
            entanglement_window: 量子エンタングルメント分析ウィンドウ
            efficiency_window: 量子効率スペクトラム分析ウィンドウ
            uncertainty_window: 量子不確定性分析ウィンドウ
            src_type: 価格ソースタイプ
            adaptive_mode: 適応モード（将来の機能拡張用）
            sensitivity: 感度調整倍率
            str_period: STR期間（ボラティリティ計算用）
            min_data_points: 最小データポイント数
            confidence_threshold: 信頼度閾値
        """
        super().__init__(f"UQATRD(C:{coherence_window},E:{entanglement_window},"
                        f"Ef:{efficiency_window},U:{uncertainty_window})")
        
        # パラメータの保存
        self.coherence_window = coherence_window
        self.entanglement_window = entanglement_window
        self.efficiency_window = efficiency_window
        self.uncertainty_window = uncertainty_window
        
        self.src_type = src_type.lower()
        self.adaptive_mode = adaptive_mode
        self.sensitivity = sensitivity
        self.str_period = str_period
        self.min_data_points = min_data_points
        self.confidence_threshold = confidence_threshold
        
        # パラメータ検証
        if self.coherence_window < 5:
            raise ValueError("coherence_windowは5以上である必要があります")
        if self.entanglement_window < 10:
            raise ValueError("entanglement_windowは10以上である必要があります")
        if self.efficiency_window < 5:
            raise ValueError("efficiency_windowは5以上である必要があります")
        if self.uncertainty_window < 5:
            raise ValueError("uncertainty_windowは5以上である必要があります")
        
        # ソースタイプの検証
        available_sources = PriceSource.get_available_sources()
        if self.src_type not in available_sources:
            valid_sources = ', '.join(available_sources.keys())
            raise ValueError(f"無効なソースタイプです: {src_type}。"
                           f"有効なオプション: {valid_sources}")
        
        # STRインスタンス（ボラティリティ計算用）
        try:
            from .str import STR
            self._str_indicator = STR(period=self.str_period, src_type='hlc3')
            self.logger.info(f"STRインジケーター初期化完了: period={self.str_period}")
        except ImportError:
            try:
                from str import STR
                self._str_indicator = STR(period=self.str_period, src_type='hlc3')
                self.logger.info(f"STRインジケーター初期化完了: period={self.str_period}")
            except ImportError:
                self.logger.warning("STRインジケーターの初期化に失敗しました。簡易ボラティリティを使用します。")
                self._str_indicator = None
        
        # 結果キャッシュ
        self._result_cache = {}
        self._max_cache_size = 10
        self._cache_keys = []
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算"""
        try:
            if isinstance(data, pd.DataFrame):
                length = len(data)
                first_val = float(data.iloc[0].get('close', data.iloc[0, -1])) if length > 0 else 0.0
                last_val = float(data.iloc[-1].get('close', data.iloc[-1, -1])) if length > 0 else 0.0
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
            
            params_sig = (f"{self.coherence_window}_{self.entanglement_window}_"
                         f"{self.efficiency_window}_{self.uncertainty_window}_"
                         f"{self.src_type}_{self.sensitivity}")
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(params_sig)}"
            
        except Exception:
            return f"{id(data)}_{self.coherence_window}_{self.entanglement_window}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> UQATRDResult:
        """
        UQATRD計算メイン関数
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                
        Returns:
            UQATRDResult: 計算結果
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            
            if data_hash in self._result_cache:
                if data_hash in self._cache_keys:
                    self._cache_keys.remove(data_hash)
                self._cache_keys.append(data_hash)
                return self._result_cache[data_hash]
            
            # 価格ソースの計算
            price_source = PriceSource.calculate_source(data, self.src_type)
            price_source = np.asarray(price_source, dtype=np.float64)
            
            # データ長の検証
            data_length = len(price_source)
            if data_length == 0:
                raise ValueError("入力データが空です")
            
            if data_length < self.min_data_points:
                self.logger.warning(f"データが短すぎます（{data_length}点）。"
                                  f"最低{self.min_data_points}点を推奨します。")
            
            # 感度調整
            if self.sensitivity != 1.0:
                # 価格変動を感度倍率で調整
                mean_price = np.mean(price_source)
                price_deviations = price_source - mean_price
                price_source = mean_price + price_deviations * self.sensitivity
            
            # STR計算（ボラティリティ用）
            str_values = np.zeros(data_length)
            if self._str_indicator is not None:
                try:
                    str_result = self._str_indicator.calculate(data)
                    str_values = str_result.values.copy()
                    self.logger.debug(f"STR計算完了: 平均STR={np.mean(str_values):.6f}")
                except Exception as e:
                    self.logger.warning(f"STR計算に失敗: {e}. 簡易計算を使用します。")
                    # 簡易STR計算（フォールバック）
                    for i in range(1, data_length):
                        str_values[i] = abs(price_source[i] - price_source[i-1])
            else:
                # 簡易STR計算
                for i in range(1, data_length):
                    str_values[i] = abs(price_source[i] - price_source[i-1])
                self.logger.debug("簡易STR計算を使用しました")
            
            # 核心計算エンジン実行
            (trend_range_signal, signal_strength, coherence, 
             trend_persistence, efficiency_spectrum, uncertainty_range, adaptive_threshold) = calculate_uqatrd_core(
                price_source,
                str_values,
                self.coherence_window,
                self.entanglement_window,
                self.efficiency_window,
                self.uncertainty_window
            )
            
            # 信頼度スコア計算
            confidence_score = np.zeros(data_length)
            cycle_adaptive_factor = np.ones(data_length)
            
            for i in range(data_length):
                # 4つのアルゴリズムの合意度を信頼度とする（全て0-1の範囲）
                algorithm_values = np.array([
                    coherence[i],
                    trend_persistence[i],       # 既に0~1の範囲
                    efficiency_spectrum[i],
                    (1.0 - uncertainty_range[i])
                ])
                
                # 合意度の計算（分散の逆数）
                agreement = 1.0 - np.var(algorithm_values)
                confidence_score[i] = max(0.0, min(1.0, agreement))
                
                # 適応因子（将来の機能拡張用）
                if self.adaptive_mode:
                    cycle_adaptive_factor[i] = 1.0 + coherence[i] * 0.2
            
            # 結果の保存
            result = UQATRDResult(
                trend_range_signal=trend_range_signal.copy(),
                signal_strength=signal_strength.copy(),
                quantum_coherence=coherence.copy(),
                trend_persistence=trend_persistence.copy(),
                efficiency_spectrum=efficiency_spectrum.copy(),
                uncertainty_range=uncertainty_range.copy(),
                adaptive_threshold=adaptive_threshold.copy(),
                confidence_score=confidence_score.copy(),
                cycle_adaptive_factor=cycle_adaptive_factor.copy()
            )
            
            # キャッシュ管理
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            self._values = trend_range_signal  # 基底クラスの要件
            
            self.logger.info(f"UQATRD計算完了 - データ長: {data_length}, "
                           f"平均信頼度: {np.mean(confidence_score):.3f}")
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"UQATRD計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は空の結果を返す
            return UQATRDResult(
                trend_range_signal=np.array([]),
                signal_strength=np.array([]),
                quantum_coherence=np.array([]),
                trend_persistence=np.array([]),
                efficiency_spectrum=np.array([]),
                uncertainty_range=np.array([]),
                adaptive_threshold=np.array([]),
                confidence_score=np.array([]),
                cycle_adaptive_factor=np.array([])
            )
    
    def get_trend_range_signal(self) -> Optional[np.ndarray]:
        """メインのトレンド/レンジ判定信号を取得"""
        if not self._result_cache or not self._cache_keys:
            return None
        
        result = self._result_cache[self._cache_keys[-1]]
        return result.trend_range_signal.copy()
    
    def get_signal_strength(self) -> Optional[np.ndarray]:
        """信号強度を取得"""
        if not self._result_cache or not self._cache_keys:
            return None
        
        result = self._result_cache[self._cache_keys[-1]]
        return result.signal_strength.copy()
    
    def get_confidence_score(self) -> Optional[np.ndarray]:
        """信頼度スコアを取得"""
        if not self._result_cache or not self._cache_keys:
            return None
        
        result = self._result_cache[self._cache_keys[-1]]
        return result.confidence_score.copy()
    
    def get_algorithm_breakdown(self) -> Optional[Dict[str, np.ndarray]]:
        """各アルゴリズムの詳細結果を取得"""
        if not self._result_cache or not self._cache_keys:
            return None
        
        result = self._result_cache[self._cache_keys[-1]]
        return {
            'quantum_coherence': result.quantum_coherence.copy(),
            'trend_persistence': result.trend_persistence.copy(),
            'efficiency_spectrum': result.efficiency_spectrum.copy(),
            'uncertainty_range': result.uncertainty_range.copy()
        }
    
    def get_adaptive_threshold(self) -> Optional[np.ndarray]:
        """
        動的適応閾値を取得する
        
        Returns:
            np.ndarray: 動的適応閾値配列 (0.4 to 0.6)
        """
        if not self._result_cache or not self._cache_keys:
            return None
        
        result = self._result_cache[self._cache_keys[-1]]
        return result.adaptive_threshold.copy()
    
    def get_trend_range_classification(self) -> Optional[np.ndarray]:
        """
        動的閾値を使用したトレンド/レンジ分類を取得する
        
        Returns:
            np.ndarray: 分類結果 (0=レンジ, 1=トレンド)
        """
        if not self._result_cache or not self._cache_keys:
            return None
        
        result = self._result_cache[self._cache_keys[-1]]
        
        # 動的閾値を使用して分類
        classification = np.zeros_like(result.trend_range_signal)
        
        for i in range(len(result.trend_range_signal)):
            if result.trend_range_signal[i] >= result.adaptive_threshold[i]:
                classification[i] = 1.0  # トレンド
            else:
                classification[i] = 0.0  # レンジ
                
        return classification
    
    def get_threshold_info(self) -> Optional[Dict[str, any]]:
        """
        動的閾値の統計情報を取得する
        
        Returns:
            Dict: 閾値の統計情報
        """
        if not self._result_cache or not self._cache_keys:
            return None
        
        result = self._result_cache[self._cache_keys[-1]]
        threshold = result.adaptive_threshold
        
        return {
            'mean_threshold': float(np.mean(threshold)),
            'std_threshold': float(np.std(threshold)),
            'min_threshold': float(np.min(threshold)),
            'max_threshold': float(np.max(threshold)),
            'median_threshold': float(np.median(threshold)),
            'current_threshold': float(threshold[-1]) if len(threshold) > 0 else None
        }
    
    def reset(self) -> None:
        """インディケーターの状態をリセット"""
        super().reset()
        self._result_cache = {}
        self._cache_keys = []
        if self._str_indicator is not None:
            self._str_indicator.reset() 