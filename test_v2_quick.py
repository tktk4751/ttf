#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from indicators.enhanced_trend_state import EnhancedTrendState
from indicators.enhanced_trend_state_v2 import EnhancedTrendStateV2
import time

def compare_versions():
    """V1とV2の比較テスト"""
    print("=== Enhanced Trend State V1 vs V2 比較 ===\n")
    
    # テストデータ生成（実際の相場に近いパターン）
    np.random.seed(42)
    n = 500
    
    # より現実的な価格データ
    prices = [100.0]
    trend_periods = [(50, 150, 0.002), (250, 350, -0.0015), (400, 450, 0.001)]  # (start, end, trend_rate)
    
    for i in range(1, n):
        # トレンド判定
        is_trend = False
        trend_rate = 0
        for start, end, rate in trend_periods:
            if start <= i < end:
                is_trend = True
                trend_rate = rate
                break
        
        if is_trend:
            # トレンド期間
            noise = np.random.normal(0, 0.002)
            change = trend_rate + noise
        else:
            # レンジ期間
            center = np.mean(prices[-20:]) if len(prices) >= 20 else prices[-1]
            mean_reversion = -0.002 * (prices[-1] - center) / center
            noise = np.random.normal(0, 0.005)
            change = mean_reversion + noise
        
        prices.append(prices[-1] * (1 + change))
    
    # OHLC作成
    data = []
    for i, price in enumerate(prices):
        o = prices[i-1] if i > 0 else price
        c = price
        h = max(o, c) * (1 + np.random.uniform(0, 0.01))
        l = min(o, c) * (1 - np.random.uniform(0, 0.01))
        data.append([o, h, l, c])
    
    df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
    
    # 理論的なトレンド期間の割合
    theoretical_trend_bars = sum(end - start for start, end, _ in trend_periods)
    theoretical_trend_ratio = theoretical_trend_bars / n * 100
    print(f"理論的トレンド期間: {theoretical_trend_bars}バー ({theoretical_trend_ratio:.1f}%)\n")
    
    # V1テスト
    print("--- V1 (Original) ---")
    v1 = EnhancedTrendState(base_period=20, threshold=0.45, use_dynamic_period=False)
    start_time = time.time()
    result_v1 = v1.calculate(df)
    v1_time = time.time() - start_time
    
    v1_trend_ratio = np.sum(result_v1.trend_state) / len(result_v1.trend_state) * 100
    v1_avg_confidence = np.mean(result_v1.confidence[~np.isnan(result_v1.confidence)])
    v1_avg_composite = np.mean(result_v1.composite_score[~np.isnan(result_v1.composite_score)])
    
    print(f"トレンド判定率: {v1_trend_ratio:.1f}%")
    print(f"平均信頼度: {v1_avg_confidence:.3f}")
    print(f"平均複合スコア: {v1_avg_composite:.3f}")
    print(f"計算時間: {v1_time*1000:.2f}ms")
    
    # V2テスト
    print("\n--- V2 (Improved) ---")
    v2 = EnhancedTrendStateV2(base_period=20, threshold=0.45, use_dynamic_period=False)
    start_time = time.time()
    result_v2 = v2.calculate(df)
    v2_time = time.time() - start_time
    
    v2_trend_ratio = np.sum(result_v2.trend_state) / len(result_v2.trend_state) * 100
    v2_avg_confidence = np.mean(result_v2.confidence[~np.isnan(result_v2.confidence)])
    v2_avg_composite = np.mean(result_v2.composite_score[~np.isnan(result_v2.composite_score)])
    
    print(f"トレンド判定率: {v2_trend_ratio:.1f}%")
    print(f"平均信頼度: {v2_avg_confidence:.3f}")
    print(f"平均複合スコア: {v2_avg_composite:.3f}")
    print(f"計算時間: {v2_time*1000:.2f}ms")
    
    # 詳細分析
    print("\n--- 詳細分析 ---")
    print(f"V2の改善率: {(v2_trend_ratio/theoretical_trend_ratio*100):.1f}% (理論値に対して)")
    print(f"V1との差: {v2_trend_ratio - v1_trend_ratio:.1f}%ポイント")
    
    # コンポーネント分析
    print("\n--- コンポーネント統計 (V2) ---")
    valid_er = result_v2.efficiency_ratio[~np.isnan(result_v2.efficiency_ratio)]
    valid_chop = result_v2.choppiness_index[~np.isnan(result_v2.choppiness_index)]
    
    print(f"効率比: 平均={np.mean(valid_er):.3f}, 範囲=[{np.min(valid_er):.3f}-{np.max(valid_er):.3f}]")
    print(f"チョピネス: 平均={np.mean(valid_chop):.1f}, 範囲=[{np.min(valid_chop):.1f}-{np.max(valid_chop):.1f}]")
    
    # 各トレンド期間での検出精度
    print("\n--- 期間別検出精度 (V2) ---")
    for i, (start, end, rate) in enumerate(trend_periods):
        detected = np.sum(result_v2.trend_state[start:end])
        total = end - start
        accuracy = detected / total * 100
        print(f"期間{i+1} ({start}-{end}): {detected}/{total} ({accuracy:.1f}%)")

if __name__ == "__main__":
    compare_versions()