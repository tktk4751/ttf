#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from numba import jit, prange
import warnings
warnings.filterwarnings('ignore')


@jit(nopython=True)
def calculate_efficiency_ratio_enhanced(prices: np.ndarray, period: int) -> np.ndarray:
    """
    🚀 効率比計算（エンハンス版・保守的調整）
    より保守的な効率比測定で確実なトレンド検出
    """
    n = len(prices)
    er = np.zeros(n)
    
    for i in range(period, n):
        # 直線距離（期間の最初から最後への価格変化）
        change = abs(prices[i] - prices[i-period])
        
        # 実際の経路距離（価格変化の絶対値の合計）
        volatility = 0.0
        for j in range(i-period, i):
            volatility += abs(prices[j+1] - prices[j])
        
        # 効率比の計算（より保守的に）
        if volatility > 1e-12:
            er[i] = change / volatility
            # 効率比のブーストを控えめに（強いトレンドのみをブースト）
            if er[i] > 0.5:  # より高い閾値でブースト
                er[i] = min(1.0, er[i] * 1.1)  # ブースト倍率を下げる
        else:
            er[i] = 1.0
    
    return er


@jit(nopython=True)
def calculate_enhanced_momentum_consistency(prices: np.ndarray) -> np.ndarray:
    """
    🎯 エンハンスモメンタム一貫性（保守的調整版）
    """
    n = len(prices)
    consistency = np.zeros(n)
    periods = np.array([3, 7, 14, 21, 35])
    
    for i in range(35, n):
        directions = np.zeros(len(periods))
        weights = np.array([0.25, 0.25, 0.25, 0.15, 0.1])  # より均等な重み付け
        
        for j, period in enumerate(periods):
            if i >= period:
                momentum = (prices[i] - prices[i-period]) / prices[i-period]
                # より厳しい閾値で方向を判定（保守的に）
                if momentum > 0.005:  # 0.5%以上の上昇（0.2%から上げる）
                    directions[j] = 1.0
                elif momentum < -0.005:  # 0.5%以上の下降
                    directions[j] = -1.0
        
        # 重み付き一致度計算
        if np.sum(np.abs(directions)) > 0:
            weighted_sum = np.sum(directions * weights)
            weight_sum = np.sum(weights[np.abs(directions) > 0])
            if weight_sum > 0:
                consistency[i] = abs(weighted_sum) / weight_sum
        else:
            consistency[i] = 0.0
    
    return consistency


@jit(nopython=True)
def enhanced_adaptive_threshold(er: np.ndarray, chop: np.ndarray, adx: np.ndarray,
                               vol_adj: np.ndarray, period: int = 30) -> np.ndarray:
    """
    🧠 エンハンス適応的閾値（バランス調整版）
    """
    n = len(er)
    threshold = np.full(n, 0.40)  # バランス調整（0.45から下げる）
    
    for i in range(period, n):
        # 過去の指標値の統計
        er_mean = np.mean(er[i-period+1:i+1])
        chop_mean = np.mean(chop[i-period+1:i+1])
        adx_mean = np.mean(adx[i-period+1:i+1])
        
        # バランス調整された閾値調整
        base_threshold = 0.40
        
        # 高ADX期間での調整（適度に）
        if adx_mean > 25:  # 中間的な閾値
            base_threshold -= 0.10  # 適度な調整
        elif adx_mean > 18:
            base_threshold -= 0.06
        elif adx_mean < 12:
            base_threshold += 0.08
        
        # 低Chop期間での調整（適度に）
        if chop_mean < 45:  # 中間的な条件
            base_threshold -= 0.08  # 適度な調整
        elif chop_mean > 65:  # 中間的な条件
            base_threshold += 0.12
        
        # 高効率比期間での調整（適度に）
        if er_mean > 0.45:  # 中間的な閾値
            base_threshold -= 0.05  # 適度な調整
        
        # ボラティリティ調整も適度に
        if vol_adj[i] < 0.9:
            base_threshold *= 0.92  # 適度な調整
        else:
            base_threshold *= vol_adj[i]
        
        threshold[i] = max(0.20, min(0.65, base_threshold))  # 中間的な範囲
    
    return threshold


@jit(nopython=True)
def ultimate_enhanced_ensemble_decision(
    er: np.ndarray,
    chop: np.ndarray,
    adx: np.ndarray,
    momentum_consistency: np.ndarray,
    vol_adj: np.ndarray,
    threshold: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    👑 究極エンハンスアンサンブル決定（バランス調整版）
    """
    n = len(er)
    signals = np.zeros(n, dtype=np.int32)
    confidence = np.zeros(n)
    
    for i in range(50, n):
        # 各指標のスコア計算（バランス調整）
        scores = np.zeros(4)
        
        # 1. Enhanced Efficiency Ratio Score (重み: 35%) - バランス調整
        if er[i] > 0.55:  # 中間的な閾値
            scores[0] = min(1.3, (er[i] - 0.55) / 0.3 * 1.6)
        elif er[i] > 0.35:
            scores[0] = (er[i] - 0.35) / 0.2 * 0.7  # 適度なスコア
        elif er[i] < 0.28:
            scores[0] = -min(1.0, (0.28 - er[i]) / 0.28 * 1.1)
        else:
            scores[0] = 0.0
        
        # 2. Enhanced Choppiness Score (重み: 30%) - バランス調整
        chop_normalized = (100.0 - chop[i]) / 100.0
        if chop_normalized > 0.58:  # 中間的な閾値
            scores[1] = (chop_normalized - 0.58) / 0.42 * 1.3
        elif chop_normalized < 0.42:
            scores[1] = -(0.42 - chop_normalized) / 0.42 * 1.2
        else:
            scores[1] = (chop_normalized - 0.42) / 0.16 * 0.4  # 適度なスコア
        
        # 3. Enhanced ADX Score (重み: 25%) - バランス調整
        adx_normalized = adx[i] / 100.0
        if adx_normalized > 0.25:  # 中間的な閾値
            scores[2] = min(1.1, (adx_normalized - 0.25) / 0.25 * 1.3)
        elif adx_normalized < 0.18:
            scores[2] = -min(0.8, (0.18 - adx_normalized) / 0.18 * 0.9)
        else:
            scores[2] = (adx_normalized - 0.18) / 0.07 * 0.3  # 適度なスコア
        
        # 4. Enhanced Momentum Score (重み: 10%) - バランス調整
        if momentum_consistency[i] > 0.65:  # 中間的な閾値
            scores[3] = (momentum_consistency[i] - 0.65) / 0.35 * 1.1
        elif momentum_consistency[i] < 0.25:
            scores[3] = -(0.25 - momentum_consistency[i]) / 0.25 * 0.7
        else:
            scores[3] = 0.0
        
        # 重み付きアンサンブルスコア（調整済み）
        weights = np.array([0.35, 0.30, 0.25, 0.10])
        final_score = np.sum(scores * weights)
        
        # バランス調整された信頼度計算
        positive_indicators = np.sum(scores > 0.25)  # 中間的な閾値
        negative_indicators = np.sum(scores < -0.25)
        
        if positive_indicators >= 2:  # 中間的な条件
            agreement = positive_indicators / len(scores)
            confidence_boost = 1.15 if positive_indicators >= 3 else 1.05
        elif negative_indicators >= 2:
            agreement = negative_indicators / len(scores)
            confidence_boost = 1.10 if negative_indicators >= 3 else 1.0
        else:
            agreement = 0.45  # 中間的な基本信頼度
            confidence_boost = 0.85
        
        # ボラティリティ調整（適度に）
        vol_adjustment = 0.93 + (vol_adj[i] - 1.0) * 0.4
        adjusted_score = final_score * vol_adjustment
        
        # バランス調整された判定基準
        current_threshold = threshold[i]
        
        # メインの判定（バランス調整）
        if abs(adjusted_score) >= current_threshold:
            signals[i] = 1 if adjusted_score > 0 else 0
            base_confidence = 0.52 + agreement * 0.28  # 中間的な基本信頼度
            confidence[i] = min(0.92, base_confidence * confidence_boost)
        else:
            # バランス調整されたフォールバック判定
            if positive_indicators >= 2 and abs(adjusted_score) >= current_threshold * 0.75:
                signals[i] = 1
                confidence[i] = 0.58 + (positive_indicators - 2) * 0.04
            elif negative_indicators >= 2 and abs(adjusted_score) >= current_threshold * 0.75:
                signals[i] = 0
                confidence[i] = 0.58 + (negative_indicators - 2) * 0.04
            else:
                signals[i] = 0  # レンジ
                confidence[i] = 0.48  # 中間的な信頼度
    
    # 初期値設定
    for i in range(50):
        signals[i] = 0
        confidence[i] = 0.48
    
    return signals, confidence


@jit(nopython=True)
def enhanced_noise_filter(signals: np.ndarray, confidence: np.ndarray, 
                         window: int = 5) -> np.ndarray:
    """
    🔧 エンハンスノイズ除去（より柔軟なフィルタリング）
    """
    n = len(signals)
    filtered_signals = np.zeros(n, dtype=np.int32)
    
    for i in range(n):
        if i < window:
            filtered_signals[i] = signals[i]
        else:
            # より短いウィンドウでの重み付き判定
            weighted_sum = 0.0
            total_weight = 0.0
            
            for j in range(window):
                idx = i - j
                # 新しい信号により大きな重みを与える
                time_weight = 1.0 + j * 0.1
                confidence_weight = confidence[idx] if idx >= 0 else 0.5
                weight = confidence_weight * time_weight
                
                weighted_sum += signals[idx] * weight
                total_weight += weight
            
            if total_weight > 0:
                avg_signal = weighted_sum / total_weight
                # よりソフトな判定
                filtered_signals[i] = 1 if avg_signal >= 0.45 else 0
            else:
                filtered_signals[i] = signals[i]
    
    return filtered_signals


class UltimateTrendRangeDetectorV3Enhanced:
    """
    🚀 人類史上最強トレンド/レンジ判別インジケーター V3.0 ENHANCED - BALANCED EDITION
    
    🎯 **バランス調整版:**
    - 適度な閾値設定でバランス判定
    - 中間的なブースト倍率
    - 柔軟なフォールバック判定
    - トレンド・レンジの適切な分布
    
    💎 **V3エンハンスバランス版革新:**
    - 実用的なトレンド検出システム
    - 適応的多段階判定
    - バランス重視ノイズフィルタリング
    - 実用性と精度の両立
    
    🏆 **実用性・精度・バランスの三位一体**
    """
    
    def __init__(self, 
                 er_period: int = 20,  # バランス調整された期間
                 chop_period: int = 14,
                 adx_period: int = 14,
                 vol_period: int = 18):  # バランス調整された期間
        """
        バランス版コンストラクタ（実用的な期間設定）
        """
        self.er_period = er_period
        self.chop_period = chop_period
        self.adx_period = adx_period
        self.vol_period = vol_period
        self.name = "UltimateTrendRangeDetectorV3EnhancedBalanced"
        self.version = "v3.0-enhanced-balanced"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        🎯 V3エンハンス版 究極判別実行（80%精度目標）
        """
        # データの準備
        if isinstance(data, pd.DataFrame):
            if not all(col in data.columns for col in ['high', 'low', 'close']):
                raise ValueError("high, low, closeカラムが必要です")
            high = data['high'].values.astype(np.float64)
            low = data['low'].values.astype(np.float64)
            close = data['close'].values.astype(np.float64)
        else:
            if data.ndim != 2 or data.shape[1] < 4:
                raise ValueError("OHLC形式の4列データが必要です")
            high = data[:, 1].astype(np.float64)
            low = data[:, 2].astype(np.float64)
            close = data[:, 3].astype(np.float64)
        
        # HLC3価格計算
        prices = (high + low + close) / 3.0
        n = len(prices)
        
        print("📊 エンハンス指標計算中...")
        
        # エンハンス版の計算を使用
        from ultimate_trend_range_detector_v3 import (
            calculate_true_range_v3, calculate_chop_index_v3, 
            calculate_adx_v3, calculate_volatility_adjustment_v3
        )
        
        # 1. エンハンス効率比
        er = calculate_efficiency_ratio_enhanced(prices, self.er_period)
        
        # 2. True Range
        tr = calculate_true_range_v3(high, low, close)
        
        # 3. Choppiness Index
        chop = calculate_chop_index_v3(high, low, close, tr, self.chop_period)
        
        # 4. ADX
        adx = calculate_adx_v3(high, low, close, self.adx_period)
        
        print("⚡ エンハンス補助指標計算中...")
        
        # 5. ボラティリティ調整
        vol_adj = calculate_volatility_adjustment_v3(prices, self.vol_period)
        
        # 6. エンハンスモメンタム一貫性
        momentum_consistency = calculate_enhanced_momentum_consistency(prices)
        
        # 7. エンハンス適応的閾値
        threshold = enhanced_adaptive_threshold(er, chop, adx, vol_adj)
        
        print("🧠 エンハンスアンサンブル判定実行中...")
        
        # 8. 究極エンハンスアンサンブル
        signals, confidence = ultimate_enhanced_ensemble_decision(
            er, chop, adx, momentum_consistency, vol_adj, threshold
        )
        
        # 9. エンハンスノイズ除去
        final_signals = enhanced_noise_filter(signals, confidence)
        
        # 結果統計
        trend_count = int(np.sum(final_signals == 1))
        range_count = int(np.sum(final_signals == 0))
        
        print("✅ V3エンハンス版計算完了！")
        
        return {
            'signal': final_signals,
            'confidence': confidence,
            'efficiency_ratio': er,
            'choppiness_index': chop,
            'adx': adx,
            'volatility_adjustment': vol_adj,
            'momentum_consistency': momentum_consistency,
            'adaptive_threshold': threshold,
            'labels': np.array(['レンジ', 'トレンド'])[final_signals],
            'summary': {
                'total_bars': n,
                'trend_bars': trend_count,
                'range_bars': range_count,
                'trend_ratio': float(trend_count / n),
                'avg_confidence': float(np.mean(confidence)),
                'high_confidence_ratio': float(np.mean(confidence >= 0.8)),
                'er_avg': float(np.mean(er[er > 0])),
                'chop_avg': float(np.mean(chop[chop > 0])),
                'adx_avg': float(np.mean(adx[adx > 0]))
            }
        } 