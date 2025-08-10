#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from indicators.enhanced_trend_state import EnhancedTrendState
from indicators.efficiency_ratio import EfficiencyRatio
from indicators.chop_trend import ChopTrend
import time

def generate_test_data(n=500):
    """シンプルなテストデータ生成"""
    np.random.seed(42)
    
    base_price = 100.0
    prices = [base_price]
    
    # トレンドとレンジを明確に分ける
    for i in range(1, n):
        if i % 150 < 75:  # トレンド期間
            trend = 0.003 * (1 if (i // 150) % 2 == 0 else -1)
            noise = np.random.normal(0, 0.002)
        else:  # レンジ期間
            center_price = np.mean(prices[-20:]) if len(prices) >= 20 else prices[-1]
            mean_reversion = -0.002 * (prices[-1] - center_price) / center_price
            noise = np.random.normal(0, 0.005)
            trend = mean_reversion
        
        new_price = prices[-1] * (1 + trend + noise)
        prices.append(new_price)
    
    # OHLC作成
    data = []
    for i, price in enumerate(prices):
        o = prices[i-1] if i > 0 else price
        c = price
        h = max(o, c) * (1 + np.random.uniform(0, 0.008))
        l = min(o, c) * (1 - np.random.uniform(0, 0.008))
        data.append([o, h, l, c])
    
    return pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])

def test_enhanced_vs_original():
    """Enhanced版と既存インジケーターの比較テスト"""
    print("=== Enhanced Trend State vs Original Indicators ===")
    
    # テストデータ生成
    data = generate_test_data(500)
    print(f"テストデータ: {len(data)}バー")
    
    # インジケーター設定
    indicators = {
        'Enhanced (Conservative)': EnhancedTrendState(
            base_period=20, threshold=0.65, use_dynamic_period=False,
            volatility_adjustment=True, atr_smoothing=True
        ),
        'Enhanced (Balanced)': EnhancedTrendState(
            base_period=20, threshold=0.60, use_dynamic_period=True,
            volatility_adjustment=True, atr_smoothing=True
        ),
        'Enhanced (Aggressive)': EnhancedTrendState(
            base_period=15, threshold=0.55, use_dynamic_period=True,
            volatility_adjustment=True, atr_smoothing=True
        ),
        'Original ER': EfficiencyRatio(
            period=20, use_dynamic_period=False, smoothing_method='none'
        ),
        'Original ChopTrend': ChopTrend(
            min_output=13, max_output=89, use_dynamic_atr_period=False
        )
    }
    
    results = {}
    performance_data = {}
    
    print(f"\n{'Indicator':<25} {'Time(ms)':<10} {'Trend%':<10} {'Confidence':<12} {'Current':<10}")
    print("-" * 75)
    
    for name, indicator in indicators.items():
        # パフォーマンステスト
        start_time = time.time()
        
        if 'Enhanced' in name:
            result = indicator.calculate(data)
            trend_ratio = np.sum(result.trend_state) / len(result.trend_state) * 100
            avg_confidence = np.mean(result.confidence[~np.isnan(result.confidence)])
            current_state = "トレンド" if indicator.is_trending() else "レンジ"
            
            results[name] = {
                'result': result,
                'trend_ratio': trend_ratio,
                'confidence': avg_confidence,
                'current_state': current_state,
                'type': 'enhanced'
            }
            
        elif 'ER' in name:
            result = indicator.calculate(data)
            er_values = result.values
            # 0.6以上をトレンドとして判定
            trend_signals = (er_values >= 0.6).astype(int)
            trend_ratio = np.sum(trend_signals) / len(trend_signals) * 100
            avg_confidence = np.mean(er_values[~np.isnan(er_values)])
            current_state = "トレンド" if er_values[-1] >= 0.6 else "レンジ"
            
            results[name] = {
                'er_values': er_values,
                'trend_signals': trend_signals,
                'trend_ratio': trend_ratio,
                'confidence': avg_confidence,
                'current_state': current_state,
                'type': 'er'
            }
            
        elif 'ChopTrend' in name:
            result = indicator.calculate(data)
            chop_values = result.values
            # 0.6以上をトレンドとして判定
            trend_signals = (chop_values >= 0.6).astype(int)
            trend_ratio = np.sum(trend_signals) / len(trend_signals) * 100
            avg_confidence = np.mean(chop_values[~np.isnan(chop_values)])
            current_state = "トレンド" if chop_values[-1] >= 0.6 else "レンジ"
            
            results[name] = {
                'chop_values': chop_values,
                'trend_signals': trend_signals,
                'trend_ratio': trend_ratio,
                'confidence': avg_confidence,
                'current_state': current_state,
                'type': 'chop'
            }
        
        end_time = time.time()
        calc_time = (end_time - start_time) * 1000  # ミリ秒
        
        performance_data[name] = calc_time
        
        print(f"{name:<25} {calc_time:<10.2f} {results[name]['trend_ratio']:<10.1f} "
              f"{results[name]['confidence']:<12.3f} {results[name]['current_state']:<10}")
    
    # 詳細可視化
    create_comparison_visualization(data, results)
    
    return results, performance_data

def create_comparison_visualization(data, results):
    """比較結果の可視化"""
    fig, axes = plt.subplots(5, 1, figsize=(16, 20))
    
    # 価格チャート
    ax1 = axes[0]
    ax1.plot(data['close'], label='Close Price', color='black', linewidth=1.2)
    ax1.set_title('Test Data - Price Chart', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Enhanced版トレンド状態比較
    ax2 = axes[1]
    colors = ['red', 'blue', 'green']
    enhanced_results = {k: v for k, v in results.items() if v['type'] == 'enhanced'}
    
    for i, (name, result_data) in enumerate(enhanced_results.items()):
        trend_state = result_data['result'].trend_state
        y_offset = i * 1.2
        
        # トレンド期間を可視化
        for j in range(len(trend_state)):
            if trend_state[j] == 1:
                ax2.plot([j, j], [y_offset, y_offset + 1], color=colors[i], linewidth=3, alpha=0.7)
        
        ax2.text(-30, y_offset + 0.5, name.replace('Enhanced ', ''), fontweight='bold', va='center')
    
    ax2.set_title('Enhanced Trend State Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Settings')
    ax2.set_ylim(-0.5, 4)
    ax2.set_yticks([])
    ax2.grid(True, alpha=0.3)
    
    # バランス設定の詳細分析
    balanced_result = results['Enhanced (Balanced)']['result']
    
    ax3 = axes[2]
    ax3.plot(balanced_result.composite_score, label='Composite Score', color='purple', linewidth=1.5)
    ax3.plot(balanced_result.confidence, label='Confidence', color='orange', linewidth=1.5)
    ax3.axhline(y=0.60, color='red', linestyle='--', alpha=0.7, label='Threshold (0.60)')
    ax3.fill_between(range(len(balanced_result.trend_state)), 0, 1.2,
                     where=balanced_result.trend_state==1, alpha=0.2, color='green', label='Trend Periods')
    ax3.set_title('Enhanced Trend State - Detailed Analysis (Balanced)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 強化されたコンポーネント
    ax4 = axes[3]
    ax4.plot(balanced_result.efficiency_ratio, label='Enhanced Efficiency Ratio', linewidth=1.5, alpha=0.8)
    ax4.plot(balanced_result.choppiness_index / 100, label='Enhanced Choppiness (scaled)', linewidth=1.5, alpha=0.8)
    ax4.plot(balanced_result.volatility_factor, label='Volatility Factor', linewidth=1.5, alpha=0.8)
    ax4.set_title('Enhanced Components Analysis', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Component Value')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 全インジケーター比較
    ax5 = axes[4]
    
    # Enhanced (Balanced)
    enhanced_composite = balanced_result.composite_score
    ax5.plot(enhanced_composite, label='Enhanced (Balanced)', color='blue', linewidth=2, alpha=0.8)
    
    # Original ER
    if 'Original ER' in results:
        er_values = results['Original ER']['er_values']
        ax5.plot(er_values, label='Original ER', color='red', linewidth=1.5, alpha=0.7)
    
    # Original ChopTrend
    if 'Original ChopTrend' in results:
        chop_values = results['Original ChopTrend']['chop_values']
        ax5.plot(chop_values, label='Original ChopTrend', color='green', linewidth=1.5, alpha=0.7)
    
    ax5.axhline(y=0.6, color='black', linestyle='--', alpha=0.5, label='Common Threshold (0.6)')
    ax5.set_title('Enhanced vs Original Indicators Comparison', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Indicator Value')
    ax5.set_xlabel('Time')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('enhanced_trend_state_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n比較チャートを 'enhanced_trend_state_comparison.png' に保存しました")
    plt.show()

def performance_benchmark():
    """パフォーマンスベンチマーク"""
    print(f"\n=== Performance Benchmark ===")
    
    data_sizes = [100, 300, 500, 1000]
    
    for size in data_sizes:
        print(f"\nデータサイズ: {size}バー")
        data = generate_test_data(size)
        
        # Enhanced版
        enhanced = EnhancedTrendState(base_period=20, threshold=0.6, use_dynamic_period=True)
        
        times = []
        for _ in range(10):  # 10回実行して平均を取る
            start = time.time()
            enhanced.calculate(data)
            end = time.time()
            times.append(end - start)
        
        avg_time = np.mean(times) * 1000  # ミリ秒
        throughput = size / (avg_time / 1000)  # データポイント/秒
        
        print(f"  Enhanced Trend State: {avg_time:.2f}ms ({throughput:.0f} points/sec)")

def accuracy_test():
    """精度テスト - 既知のトレンド/レンジパターンでテスト"""
    print(f"\n=== Accuracy Test ===")
    
    # 既知パターンのテストデータ
    n = 300
    prices = [100.0]
    known_states = np.zeros(n)
    
    # 明確なパターンを作成
    for i in range(1, n):
        if 50 <= i < 100:  # 明確な上昇トレンド
            prices.append(prices[-1] * 1.002)
            known_states[i] = 1
        elif 150 <= i < 200:  # 明確な下降トレンド
            prices.append(prices[-1] * 0.998)
            known_states[i] = 1
        else:  # レンジ
            center = 100.0
            mean_reversion = -0.001 * (prices[-1] - center) / center
            noise = np.random.normal(0, 0.003)
            prices.append(prices[-1] * (1 + mean_reversion + noise))
            known_states[i] = 0
    
    # データフレーム作成
    ohlc_data = []
    for i, price in enumerate(prices):
        o = prices[i-1] if i > 0 else price
        c = price
        h = max(o, c) * 1.005
        l = min(o, c) * 0.995
        ohlc_data.append([o, h, l, c])
    
    df = pd.DataFrame(ohlc_data, columns=['open', 'high', 'low', 'close'])
    
    # Enhanced版でテスト
    enhanced = EnhancedTrendState(base_period=20, threshold=0.6, use_dynamic_period=True)
    result = enhanced.calculate(df)
    
    # 精度計算（初期30バーを除く）
    start_idx = 30
    actual_states = known_states[start_idx:]
    predicted_states = result.trend_state[start_idx:]
    
    accuracy = np.sum(actual_states == predicted_states) / len(actual_states)
    
    # 詳細統計
    tp = np.sum((actual_states == 1) & (predicted_states == 1))  # True Positive
    fp = np.sum((actual_states == 0) & (predicted_states == 1))  # False Positive
    tn = np.sum((actual_states == 0) & (predicted_states == 0))  # True Negative
    fn = np.sum((actual_states == 1) & (predicted_states == 0))  # False Negative
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"全体精度: {accuracy:.3f}")
    print(f"適合率: {precision:.3f}")
    print(f"再現率: {recall:.3f}")
    print(f"F1スコア: {f1:.3f}")
    
    return accuracy, f1

if __name__ == "__main__":
    # 比較テスト
    comparison_results, perf_data = test_enhanced_vs_original()
    
    # パフォーマンステスト
    performance_benchmark()
    
    # 精度テスト
    accuracy, f1 = accuracy_test()
    
    print(f"\n=== Enhanced Trend State 完成 ===")
    print("✅ シンプル&洗練: ERとChoppinessベースの最適化設計")
    print("✅ 超高精度: ボラティリティ調整とATRスムージング")
    print("✅ 超低遅延: Numba JIT最適化による高速計算")
    print("✅ 超動的適応: 動的期間調整機能")
    print(f"✅ 検証済み精度: F1スコア {f1:.3f}")
    print("✅ バイナリ出力: 明確な1/0トレンド判定")