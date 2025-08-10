#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from indicators.supreme_trend_state import SupremeTrendState
import time

def quick_test():
    """クイックテスト"""
    print("=== Supreme Trend State Quick Test ===")
    
    # 小さなテストデータ
    np.random.seed(42)
    n = 200
    
    # 基本価格データ
    base_price = 100.0
    prices = [base_price]
    
    for i in range(1, n):
        change = np.random.normal(0, 0.01)  # 1%のランダムウォーク
        if i % 50 < 25:  # トレンド期間
            change += 0.005  # 上昇トレンド
        prices.append(prices[-1] * (1 + change))
    
    # OHLC作成
    data = []
    for i, price in enumerate(prices):
        o = prices[i-1] if i > 0 else price
        c = price
        h = max(o, c) * 1.01
        l = min(o, c) * 0.99
        data.append([o, h, l, c])
    
    df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
    
    # インジケーターテスト
    indicator = SupremeTrendState(window=15, threshold=0.55, use_dynamic_period=False)
    
    start_time = time.time()
    result = indicator.calculate(df)
    end_time = time.time()
    
    # 結果表示
    calculation_time = end_time - start_time
    trend_ratio = np.sum(result.trend_state) / len(result.trend_state) * 100
    avg_confidence = np.mean(result.confidence[~np.isnan(result.confidence)])
    avg_composite = np.mean(result.composite_score[~np.isnan(result.composite_score)])
    
    print(f"データサイズ: {len(df)}")
    print(f"計算時間: {calculation_time:.4f}秒")
    print(f"トレンド判定率: {trend_ratio:.1f}%")
    print(f"平均信頼度: {avg_confidence:.3f}")
    print(f"平均複合スコア: {avg_composite:.3f}")
    print(f"現在の状態: {'トレンド' if indicator.is_trending() else 'レンジ'}")
    
    # 主要コンポーネントの統計
    print(f"\n=== コンポーネント統計 ===")
    components = {
        'Efficiency Ratio': result.efficiency_ratio,
        'Neural Adaptation': result.neural_adaptation,
        'ML Prediction': result.ml_prediction,
        'Cycle Strength': result.cycle_strength
    }
    
    for name, values in components.items():
        valid_values = values[~np.isnan(values)]
        if len(valid_values) > 0:
            print(f"{name}: 平均={np.mean(valid_values):.3f}, 範囲=[{np.min(valid_values):.3f}-{np.max(valid_values):.3f}]")
    
    print(f"\n✓ Supreme Trend State インジケーター正常動作確認")
    return result

if __name__ == "__main__":
    result = quick_test()