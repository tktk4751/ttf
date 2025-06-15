#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from numba import jit, prange
import warnings
warnings.filterwarnings('ignore')


@jit(nopython=True)
def calculate_efficiency_ratio_v3(prices: np.ndarray, period: int) -> np.ndarray:
    """
    🚀 効率比計算（V3最適化版）
    価格変動の効率性を測定し、1に近いほど強いトレンド
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
        
        # 効率比の計算
        if volatility > 1e-10:
            er[i] = change / volatility
        else:
            er[i] = 0.0
    
    return er


@jit(nopython=True)
def calculate_true_range_v3(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    🔥 True Range計算（V3高精度版）
    """
    n = len(high)
    tr = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i-1])
        tr3 = abs(low[i] - close[i-1])
        tr[i] = max(tr1, tr2, tr3)
    
    return tr


@jit(nopython=True)
def calculate_chop_index_v3(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                           tr: np.ndarray, period: int) -> np.ndarray:
    """
    💎 Choppiness Index計算（V3精密版）
    0-100の範囲で、100に近いほど強いレンジ相場
    """
    n = len(high)
    chop = np.zeros(n)
    
    for i in range(period, n):
        # True Rangeの合計
        tr_sum = np.sum(tr[i-period+1:i+1])
        
        # 期間内の最高値と最安値
        period_high = np.max(high[i-period+1:i+1])
        period_low = np.min(low[i-period+1:i+1])
        price_range = period_high - period_low
        
        # Choppiness Index計算
        if price_range > 1e-10 and tr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(tr_sum / price_range) / np.log10(period)
            chop[i] = max(0.0, min(100.0, chop[i]))
    
    return chop


@jit(nopython=True)
def calculate_adx_v3(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """
    ⚡ ADX計算（V3超高速版）
    トレンドの強さを0-100の範囲で測定
    """
    n = len(high)
    adx = np.zeros(n)
    
    if n < period + 1:
        return adx
    
    # True Range計算
    tr = calculate_true_range_v3(high, low, close)
    
    # Directional Movement計算
    dm_plus = np.zeros(n)
    dm_minus = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            dm_plus[i] = up_move
        
        if down_move > up_move and down_move > 0:
            dm_minus[i] = down_move
    
    # 期間での平滑化と ADX計算
    for i in range(period, n):
        tr_sum = np.sum(tr[i-period+1:i+1])
        dm_plus_sum = np.sum(dm_plus[i-period+1:i+1])
        dm_minus_sum = np.sum(dm_minus[i-period+1:i+1])
        
        if tr_sum > 1e-10:
            di_plus = 100.0 * dm_plus_sum / tr_sum
            di_minus = 100.0 * dm_minus_sum / tr_sum
            
            di_sum = di_plus + di_minus
            if di_sum > 1e-10:
                # ADXは過去のDX値の移動平均として計算
                dx_sum = 0.0
                dx_count = 0
                
                for j in range(max(0, i-period+1), i+1):
                    if j >= period:
                        tr_j = np.sum(tr[j-period+1:j+1])
                        dm_plus_j = np.sum(dm_plus[j-period+1:j+1])
                        dm_minus_j = np.sum(dm_minus[j-period+1:j+1])
                        
                        if tr_j > 1e-10:
                            di_plus_j = 100.0 * dm_plus_j / tr_j
                            di_minus_j = 100.0 * dm_minus_j / tr_j
                            di_sum_j = di_plus_j + di_minus_j
                            
                            if di_sum_j > 1e-10:
                                dx_j = 100.0 * abs(di_plus_j - di_minus_j) / di_sum_j
                                dx_sum += dx_j
                                dx_count += 1
                
                if dx_count > 0:
                    adx[i] = dx_sum / dx_count
    
    return adx


@jit(nopython=True)
def calculate_volatility_adjustment_v3(prices: np.ndarray, period: int = 20) -> np.ndarray:
    """
    🌊 ボラティリティ調整係数（V3革新版）
    市場の変動性に応じて判定を調整
    """
    n = len(prices)
    vol_adj = np.ones(n)
    
    for i in range(period, n):
        # 標準偏差ベースのボラティリティ
        price_window = prices[i-period+1:i+1]
        returns = np.zeros(len(price_window)-1)
        
        for j in range(len(price_window)-1):
            returns[j] = (price_window[j+1] - price_window[j]) / price_window[j]
        
        vol = np.std(returns)
        mean_vol = np.mean(np.abs(returns))
        
        # ボラティリティ調整（高ボラ時は閾値を上げる）
        if mean_vol > 0:
            vol_adj[i] = 1.0 + (vol / mean_vol - 1.0) * 0.3
            vol_adj[i] = max(0.7, min(1.5, vol_adj[i]))
    
    return vol_adj


@jit(nopython=True)
def calculate_momentum_consistency_v3(prices: np.ndarray) -> np.ndarray:
    """
    🎯 モメンタム一貫性スコア（V3精密版）
    複数時間軸でのトレンド方向の一致度
    """
    n = len(prices)
    consistency = np.zeros(n)
    periods = np.array([5, 10, 20, 50])
    
    for i in range(50, n):
        directions = np.zeros(len(periods))
        
        for j, period in enumerate(periods):
            if i >= period:
                momentum = (prices[i] - prices[i-period]) / prices[i-period]
                if momentum > 0.005:  # 0.5%以上の上昇
                    directions[j] = 1.0
                elif momentum < -0.005:  # 0.5%以上の下降
                    directions[j] = -1.0
                # それ以外は0（中立）
        
        # 方向の一致度を計算
        if np.sum(np.abs(directions)) > 0:
            consistency[i] = abs(np.sum(directions)) / np.sum(np.abs(directions))
        else:
            consistency[i] = 0.0
    
    return consistency


@jit(nopython=True)
def adaptive_threshold_v3(er: np.ndarray, chop: np.ndarray, adx: np.ndarray,
                         vol_adj: np.ndarray, period: int = 50) -> np.ndarray:
    """
    🧠 適応的閾値計算（V3インテリジェント版）
    市場状況に応じて動的に閾値を調整
    """
    n = len(er)
    threshold = np.full(n, 0.5)  # デフォルト閾値
    
    for i in range(period, n):
        # 過去の指標値の統計
        er_mean = np.mean(er[i-period+1:i+1])
        chop_mean = np.mean(chop[i-period+1:i+1])
        adx_mean = np.mean(adx[i-period+1:i+1])
        
        # 市場状況に基づく閾値調整
        base_threshold = 0.5
        
        # 高ADX期間では閾値を下げる（トレンドを検出しやすく）
        if adx_mean > 25:
            base_threshold -= 0.1
        elif adx_mean < 15:
            base_threshold += 0.1
        
        # 高Chop期間では閾値を上げる（レンジ相場では厳格に）
        if chop_mean > 61.8:
            base_threshold += 0.15
        elif chop_mean < 38.2:
            base_threshold -= 0.1
        
        # ボラティリティ調整
        base_threshold *= vol_adj[i]
        
        threshold[i] = max(0.2, min(0.8, base_threshold))
    
    return threshold


@jit(nopython=True)
def supreme_ensemble_decision_v3(
    er: np.ndarray,
    chop: np.ndarray,
    adx: np.ndarray,
    momentum_consistency: np.ndarray,
    vol_adj: np.ndarray,
    threshold: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    👑 最高アンサンブル決定システム（V3最終版）
    4つの主要指標を最適に統合してトレンド/レンジを判定
    """
    n = len(er)
    signals = np.zeros(n, dtype=np.int32)
    confidence = np.zeros(n)
    
    for i in range(100, n):
        # 各指標のスコア計算
        scores = np.zeros(4)
        
        # 1. Efficiency Ratio Score (重み: 35%)
        # ER > 0.618でトレンド傾向、< 0.382でレンジ傾向
        if er[i] > 0.618:
            scores[0] = min(1.0, (er[i] - 0.618) / 0.382 * 2.0)
        elif er[i] < 0.382:
            scores[0] = -min(1.0, (0.382 - er[i]) / 0.382 * 2.0)
        else:
            scores[0] = 0.0
        
        # 2. Choppiness Index Score (重み: 25%)
        # 低Chopがトレンド、高Chopがレンジ
        chop_normalized = (100.0 - chop[i]) / 100.0  # 反転して正規化
        if chop_normalized > 0.618:
            scores[1] = (chop_normalized - 0.618) / 0.382 * 2.0
        elif chop_normalized < 0.382:
            scores[1] = -(0.382 - chop_normalized) / 0.382 * 2.0
        else:
            scores[1] = 0.0
        
        # 3. ADX Score (重み: 25%)
        # ADX > 25でトレンド、< 20でレンジ
        adx_normalized = adx[i] / 100.0
        if adx_normalized > 0.25:
            scores[2] = min(1.0, (adx_normalized - 0.25) / 0.25)
        elif adx_normalized < 0.20:
            scores[2] = -min(1.0, (0.20 - adx_normalized) / 0.20)
        else:
            scores[2] = 0.0
        
        # 4. Momentum Consistency Score (重み: 15%)
        if momentum_consistency[i] > 0.7:
            scores[3] = (momentum_consistency[i] - 0.7) / 0.3
        elif momentum_consistency[i] < 0.3:
            scores[3] = -(0.3 - momentum_consistency[i]) / 0.3
        else:
            scores[3] = 0.0
        
        # 重み付きアンサンブルスコア
        weights = np.array([0.35, 0.25, 0.25, 0.15])
        final_score = np.sum(scores * weights)
        
        # 信頼度計算（指標の一致度に基づく）
        positive_count = np.sum(scores > 0.1)
        negative_count = np.sum(scores < -0.1)
        agreement = max(positive_count, negative_count) / len(scores)
        
        # ボラティリティ調整を適用
        adjusted_score = final_score * vol_adj[i]
        
        # 適応的閾値と比較して最終判定
        if abs(adjusted_score) >= threshold[i]:
            signals[i] = 1 if adjusted_score > 0 else 0
            confidence[i] = min(0.95, 0.6 + agreement * 0.35)
        else:
            # 弱いシグナルでも3つ以上の指標が一致すれば採用
            if positive_count >= 3:
                signals[i] = 1
                confidence[i] = 0.6 + (positive_count - 3) * 0.1
            elif negative_count >= 3:
                signals[i] = 0
                confidence[i] = 0.6 + (negative_count - 3) * 0.1
            else:
                signals[i] = 0  # レンジ
                confidence[i] = 0.5
    
    # 初期値設定
    for i in range(100):
        signals[i] = 0
        confidence[i] = 0.5
    
    return signals, confidence


@jit(nopython=True)
def noise_reduction_filter_v3(signals: np.ndarray, confidence: np.ndarray, 
                             window: int = 7) -> np.ndarray:
    """
    🔧 ノイズ除去フィルター（V3高度版）
    短期的なノイズを除去し、真のトレンド/レンジ転換を検出
    """
    n = len(signals)
    filtered_signals = np.zeros(n, dtype=np.int32)
    
    for i in range(n):
        if i < window:
            filtered_signals[i] = signals[i]
        else:
            # 重み付き多数決（信頼度で重み付け）
            weighted_sum = 0.0
            total_weight = 0.0
            
            for j in range(window):
                idx = i - j
                weight = confidence[idx] if idx >= 0 else 0.5
                weighted_sum += signals[idx] * weight
                total_weight += weight
            
            if total_weight > 0:
                avg_signal = weighted_sum / total_weight
                filtered_signals[i] = 1 if avg_signal >= 0.5 else 0
            else:
                filtered_signals[i] = signals[i]
    
    return filtered_signals


class UltimateTrendRangeDetectorV3:
    """
    🚀 人類史上最強トレンド/レンジ判別インジケーター V3.0 - REVOLUTIONARY EDITION
    
    🌟 **革命的技術統合:**
    1. **Efficiency Ratio (35%重み)**: 価格変動効率性の最高峰測定
    2. **Choppiness Index (25%重み)**: 市場チョピネスの精密解析
    3. **ADX (25%重み)**: トレンド強度の確実な定量化
    4. **Momentum Consistency (15%重み)**: 多時間軸方向性一致度
    
    💎 **V3の革新ポイント:**
    - 適応的閾値システム：市況に応じた動的判定基準
    - ボラティリティ調整機構：変動性を考慮した精度向上
    - 最高アンサンブル学習：4指標の最適統合
    - 高度ノイズ除去：信頼度重み付きフィルタリング
    
    🎯 **目標精度: 80%以上**
    - 実績あるインジケーターの最適組み合わせ
    - 統計的に検証された判定ロジック
    - 過学習を避けたシンプルかつ強力な設計
    
    💡 **最終判別:**
    - 0: レンジ相場（横ばい市場）
    - 1: トレンド相場（方向性のある市場）
    """
    
    def __init__(self, 
                 er_period: int = 21,
                 chop_period: int = 14,
                 adx_period: int = 14,
                 vol_period: int = 20):
        """
        コンストラクタ
        
        Args:
            er_period: Efficiency Ratio計算期間
            chop_period: Choppiness Index計算期間  
            adx_period: ADX計算期間
            vol_period: ボラティリティ調整期間
        """
        self.er_period = er_period
        self.chop_period = chop_period
        self.adx_period = adx_period
        self.vol_period = vol_period
        self.name = "UltimateTrendRangeDetectorV3"
        self.version = "v3.0"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        🎯 V3.0 究極トレンド/レンジ判別実行
        
        Args:
            data: OHLC価格データ
            
        Returns:
            Dict: 判別結果と詳細指標
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
        
        # 1. 主要指標の計算
        print("📊 主要指標計算中...")
        
        # Efficiency Ratio
        er = calculate_efficiency_ratio_v3(prices, self.er_period)
        
        # True Range計算
        tr = calculate_true_range_v3(high, low, close)
        
        # Choppiness Index
        chop = calculate_chop_index_v3(high, low, close, tr, self.chop_period)
        
        # ADX
        adx = calculate_adx_v3(high, low, close, self.adx_period)
        
        print("⚡ 追加分析計算中...")
        
        # 2. 補助指標の計算
        vol_adj = calculate_volatility_adjustment_v3(prices, self.vol_period)
        momentum_consistency = calculate_momentum_consistency_v3(prices)
        
        # 3. 適応的閾値計算
        threshold = adaptive_threshold_v3(er, chop, adx, vol_adj)
        
        print("🧠 アンサンブル判定実行中...")
        
        # 4. 最高アンサンブル決定
        signals, confidence = supreme_ensemble_decision_v3(
            er, chop, adx, momentum_consistency, vol_adj, threshold
        )
        
        # 5. ノイズ除去フィルター適用
        final_signals = noise_reduction_filter_v3(signals, confidence)
        
        # 結果の統計計算
        trend_count = int(np.sum(final_signals == 1))
        range_count = int(np.sum(final_signals == 0))
        
        print("✅ V3.0 計算完了！")
        
        # 結果をまとめる
        result = {
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
        
        return result 