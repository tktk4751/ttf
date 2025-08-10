#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from indicators.supreme_trend_state import SupremeTrendState
import time

def generate_test_data(n=1000):
    """テスト用データを生成"""
    np.random.seed(42)
    
    # 基本価格
    base_price = 100.0
    prices = [base_price]
    
    # トレンド期間とレンジ期間を交互に生成
    i = 0
    while i < n - 1:
        # トレンド期間 (200バー)
        trend_length = min(200, n - 1 - i)
        trend_direction = np.random.choice([1, -1])
        for j in range(trend_length):
            if i + j < n - 1:
                noise = np.random.normal(0, 0.5)
                trend = trend_direction * 0.1
                prices.append(prices[-1] * (1 + trend/100 + noise/100))
        i += trend_length
        
        # レンジ期間 (100バー)
        if i < n - 1:
            range_length = min(100, n - 1 - i)
            range_center = prices[-1]
            for j in range(range_length):
                if i + j < n - 1:
                    noise = np.random.normal(0, 1.0)
                    mean_reversion = -0.1 * (prices[-1] - range_center) / range_center
                    prices.append(prices[-1] * (1 + mean_reversion/100 + noise/100))
            i += range_length
    
    # OHLC形式のデータを作成
    data = []
    for i in range(len(prices)):
        if i == 0:
            o = h = l = c = prices[i]
        else:
            c = prices[i]
            o = prices[i-1]
            h = max(o, c) * (1 + np.random.uniform(0, 0.01))
            l = min(o, c) * (1 - np.random.uniform(0, 0.01))
        
        data.append([o, h, l, c])
    
    df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
    return df

def test_supreme_trend_state():
    """Supreme Trend Stateインジケーターのテスト"""
    print("=== Supreme Trend State Indicator Test ===")
    
    # テストデータの生成
    print("テストデータを生成中...")
    data = generate_test_data(1000)
    print(f"データ生成完了: {len(data)}行")
    
    # インジケーターの初期化
    print("\nインジケーターを初期化中...")
    indicators = {
        'conservative': SupremeTrendState(window=20, threshold=0.7, use_dynamic_period=False),
        'balanced': SupremeTrendState(window=20, threshold=0.6, use_dynamic_period=True),
        'aggressive': SupremeTrendState(window=15, threshold=0.5, use_dynamic_period=True)
    }
    
    results = {}
    
    # 各設定でテスト
    for name, indicator in indicators.items():
        print(f"\n{name.capitalize()} 設定でテスト実行中...")
        
        start_time = time.time()
        result = indicator.calculate(data)
        end_time = time.time()
        
        calculation_time = end_time - start_time
        print(f"計算時間: {calculation_time:.4f}秒")
        
        # 統計情報
        trend_ratio = np.sum(result.trend_state) / len(result.trend_state) * 100
        avg_confidence = np.mean(result.confidence[~np.isnan(result.confidence)])
        avg_composite = np.mean(result.composite_score[~np.isnan(result.composite_score)])
        
        print(f"トレンド判定率: {trend_ratio:.1f}%")
        print(f"平均信頼度: {avg_confidence:.3f}")
        print(f"平均複合スコア: {avg_composite:.3f}")
        
        # 最新の状態
        current_state = "トレンド" if indicator.is_trending() else "レンジ"
        print(f"現在の状態: {current_state}")
        
        results[name] = {
            'result': result,
            'calculation_time': calculation_time,
            'trend_ratio': trend_ratio,
            'avg_confidence': avg_confidence,
            'current_state': current_state
        }
    
    # 詳細分析
    print(f"\n=== 詳細分析 ===")
    result = results['balanced']['result']
    
    # 各コンポーネントの統計
    components = {
        'Information Entropy': result.entropy_score,
        'Fractal Dimension': result.fractal_dimension, 
        'Spectral Power': result.spectral_power,
        'Neural Adaptation': result.neural_adaptation,
        'ML Prediction': result.ml_prediction,
        'Efficiency Ratio': result.efficiency_ratio,
        'Volatility Regime': result.volatility_regime,
        'Cycle Strength': result.cycle_strength
    }
    
    for name, values in components.items():
        valid_values = values[~np.isnan(values)]
        if len(valid_values) > 0:
            mean_val = np.mean(valid_values)
            std_val = np.std(valid_values)
            print(f"{name}: 平均={mean_val:.4f}, 標準偏差={std_val:.4f}")
    
    # 可視化
    print(f"\n結果をプロット中...")
    create_visualization(data, results)
    
    print(f"\n=== テスト完了 ===")
    return results

def create_visualization(data, results):
    """結果の可視化"""
    fig, axes = plt.subplots(4, 1, figsize=(15, 16))
    
    # 価格チャート
    ax1 = axes[0]
    ax1.plot(data['close'], label='Close Price', color='black', linewidth=0.8)
    ax1.set_title('Price Chart')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # トレンド状態比較
    ax2 = axes[1]
    colors = ['blue', 'red', 'green']
    for i, (name, result_data) in enumerate(results.items()):
        trend_state = result_data['result'].trend_state
        # トレンド状態を可視化（1=トレンド、0=レンジ）
        y_offset = i * 0.5
        ax2.fill_between(range(len(trend_state)), y_offset, y_offset + trend_state * 0.4, 
                        alpha=0.7, color=colors[i], label=f'{name.capitalize()} Trend State')
    
    ax2.set_title('Trend State Comparison')
    ax2.set_ylabel('Trend State')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 複合スコアと信頼度
    ax3 = axes[2]
    balanced_result = results['balanced']['result']
    ax3.plot(balanced_result.composite_score, label='Composite Score', color='purple', linewidth=1)
    ax3.plot(balanced_result.confidence, label='Confidence', color='orange', linewidth=1)
    ax3.axhline(y=0.6, color='red', linestyle='--', alpha=0.7, label='Default Threshold')
    ax3.set_title('Composite Score and Confidence (Balanced Setting)')
    ax3.set_ylabel('Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 主要コンポーネント
    ax4 = axes[3]
    ax4.plot(balanced_result.efficiency_ratio, label='Efficiency Ratio', alpha=0.8)
    ax4.plot(balanced_result.neural_adaptation, label='Neural Adaptation', alpha=0.8)
    ax4.plot(balanced_result.ml_prediction, label='ML Prediction', alpha=0.8)
    ax4.plot(balanced_result.cycle_strength, label='Cycle Strength', alpha=0.8)
    ax4.set_title('Key Components (Balanced Setting)')
    ax4.set_ylabel('Component Value')
    ax4.set_xlabel('Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('supreme_trend_state_test.png', dpi=300, bbox_inches='tight')
    print("チャートを 'supreme_trend_state_test.png' に保存しました")
    plt.show()

def performance_test():
    """パフォーマンステスト"""
    print(f"\n=== パフォーマンステスト ===")
    
    data_sizes = [100, 500, 1000, 2000]
    
    for size in data_sizes:
        print(f"\nデータサイズ: {size}")
        data = generate_test_data(size)
        
        indicator = SupremeTrendState(window=20, threshold=0.6)
        
        # 複数回実行して平均を取る
        times = []
        for i in range(5):
            start_time = time.time()
            result = indicator.calculate(data)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        throughput = size / avg_time
        
        print(f"平均計算時間: {avg_time:.4f}秒")
        print(f"スループット: {throughput:.0f} データポイント/秒")

if __name__ == "__main__":
    # メインテスト
    results = test_supreme_trend_state()
    
    # パフォーマンステスト
    performance_test()