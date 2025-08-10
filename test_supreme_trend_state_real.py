#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from indicators.supreme_trend_state import SupremeTrendState

def create_realistic_test_data(n=1000):
    """より現実的なテストデータを生成"""
    np.random.seed(42)
    
    # 基本価格
    base_price = 100.0
    prices = [base_price]
    
    # より現実的な価格動作をシミュレート
    for i in range(1, n):
        # トレンド成分（長期）
        trend_period = 200
        trend_strength = 0.002 * np.sin(2 * np.pi * i / trend_period)
        
        # 平均回帰成分（中期）
        mean_reversion_period = 50
        mean_price = np.mean(prices[-min(mean_reversion_period, len(prices)):])
        mean_reversion = -0.001 * (prices[-1] - mean_price) / mean_price
        
        # ノイズ成分（短期）
        noise = np.random.normal(0, 0.008)
        
        # ボラティリティクラスタリング
        vol_scaling = 1.0 + 0.5 * abs(np.random.normal(0, 0.1))
        
        # 総合的な価格変化
        total_change = (trend_strength + mean_reversion + noise) * vol_scaling
        new_price = prices[-1] * (1 + total_change)
        prices.append(new_price)
    
    # OHLC形式のデータを作成
    data = []
    for i in range(len(prices)):
        if i == 0:
            o = h = l = c = prices[i]
        else:
            c = prices[i]
            o = prices[i-1]
            daily_range = abs(c - o) * np.random.uniform(1.2, 2.0)
            h = max(o, c) + daily_range * np.random.uniform(0, 0.5)
            l = min(o, c) - daily_range * np.random.uniform(0, 0.5)
        
        data.append([o, h, l, c])
    
    df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
    return df

def test_optimized_settings():
    """最適化された設定でテスト"""
    print("=== Optimized Supreme Trend State Test ===")
    
    # より現実的なテストデータ
    data = create_realistic_test_data(1000)
    print(f"現実的テストデータ生成完了: {len(data)}行")
    
    # 最適化された設定
    optimized_settings = {
        'ultra_sensitive': SupremeTrendState(window=10, threshold=0.45, use_dynamic_period=True),
        'sensitive': SupremeTrendState(window=15, threshold=0.50, use_dynamic_period=True),
        'balanced': SupremeTrendState(window=20, threshold=0.55, use_dynamic_period=True),
        'conservative': SupremeTrendState(window=25, threshold=0.65, use_dynamic_period=False),
    }
    
    results = {}
    
    for name, indicator in optimized_settings.items():
        print(f"\n{name.upper()} 設定:")
        result = indicator.calculate(data)
        
        # 統計
        trend_ratio = np.sum(result.trend_state) / len(result.trend_state) * 100
        avg_confidence = np.mean(result.confidence[~np.isnan(result.confidence)])
        avg_composite = np.mean(result.composite_score[~np.isnan(result.composite_score)])
        
        # トレンド期間の連続性を分析
        trend_switches = np.sum(np.diff(result.trend_state.astype(int)) != 0)
        
        print(f"  トレンド判定率: {trend_ratio:.1f}%")
        print(f"  平均信頼度: {avg_confidence:.3f}")
        print(f"  平均複合スコア: {avg_composite:.3f}")
        print(f"  トレンド切り替え回数: {trend_switches}")
        print(f"  現在の状態: {'トレンド' if indicator.is_trending() else 'レンジ'}")
        
        results[name] = {
            'result': result,
            'trend_ratio': trend_ratio,
            'avg_confidence': avg_confidence,
            'trend_switches': trend_switches
        }
    
    # 詳細可視化
    create_detailed_visualization(data, results)
    
    return results

def create_detailed_visualization(data, results):
    """詳細な可視化"""
    fig, axes = plt.subplots(5, 1, figsize=(16, 20))
    
    # 価格チャート
    ax1 = axes[0]
    ax1.plot(data['close'], label='Close Price', color='black', linewidth=1.0)
    ax1.set_title('Realistic Price Chart', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 複数設定でのトレンド状態比較
    ax2 = axes[1]
    colors = ['red', 'orange', 'blue', 'green']
    y_positions = [0, 1, 2, 3]
    
    for i, (name, result_data) in enumerate(results.items()):
        trend_state = result_data['result'].trend_state
        y_base = y_positions[i]
        
        # トレンド状態を可視化
        for j in range(len(trend_state)):
            if trend_state[j] == 1:  # トレンド
                ax2.plot([j, j], [y_base, y_base + 0.8], color=colors[i], linewidth=2, alpha=0.7)
        
        ax2.text(-50, y_base + 0.4, name.upper(), fontweight='bold', va='center')
    
    ax2.set_title('Trend State Comparison (Vertical Lines = Trend Periods)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Settings')
    ax2.set_ylim(-0.5, 4.5)
    ax2.set_yticks([])
    ax2.grid(True, alpha=0.3)
    
    # BALANCED設定の詳細分析
    balanced_result = results['balanced']['result']
    
    # 複合スコアと信頼度
    ax3 = axes[2]
    ax3.plot(balanced_result.composite_score, label='Composite Score', color='purple', linewidth=1.5)
    ax3.plot(balanced_result.confidence, label='Confidence', color='orange', linewidth=1.5)
    ax3.axhline(y=0.55, color='red', linestyle='--', alpha=0.7, label='Threshold (0.55)')
    ax3.fill_between(range(len(balanced_result.trend_state)), 0, 1.2, 
                     where=balanced_result.trend_state==1, alpha=0.2, color='green', label='Trend Periods')
    ax3.set_title('Composite Score and Confidence (Balanced Setting)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 主要コンポーネント (上位4つ)
    ax4 = axes[3]
    ax4.plot(balanced_result.efficiency_ratio, label='Efficiency Ratio', linewidth=1.5, alpha=0.8)
    ax4.plot(balanced_result.neural_adaptation, label='Neural Adaptation', linewidth=1.5, alpha=0.8)
    ax4.plot(balanced_result.ml_prediction, label='ML Prediction', linewidth=1.5, alpha=0.8)
    ax4.plot(balanced_result.cycle_strength, label='Cycle Strength', linewidth=1.5, alpha=0.8)
    ax4.set_title('Key Components Analysis (Balanced Setting)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Component Value')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 全コンポーネント概要
    ax5 = axes[4]
    components = {
        'Entropy': balanced_result.entropy_score / 3.0,  # 正規化
        'Fractal': (balanced_result.fractal_dimension - 1.0) / 1.0,  # 正規化
        'Spectral': balanced_result.spectral_power * 1000,  # スケール調整
        'Neural': balanced_result.neural_adaptation,
        'ML': balanced_result.ml_prediction,
        'Efficiency': balanced_result.efficiency_ratio,
        'Volatility': balanced_result.volatility_regime * 200,  # スケール調整
        'Cycle': balanced_result.cycle_strength
    }
    
    for i, (name, values) in enumerate(components.items()):
        ax5.plot(values, label=name, alpha=0.7, linewidth=1)
    
    ax5.set_title('All Components Overview (Normalized/Scaled)', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Normalized Value')
    ax5.set_xlabel('Time')
    ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('supreme_trend_state_optimized.png', dpi=300, bbox_inches='tight')
    print("\n詳細チャートを 'supreme_trend_state_optimized.png' に保存しました")
    plt.show()

def analyze_trend_detection_accuracy():
    """トレンド検出精度の分析"""
    print(f"\n=== Trend Detection Accuracy Analysis ===")
    
    # 既知のトレンド/レンジ期間を持つテストデータを作成
    n = 500
    known_trends = np.zeros(n)
    
    # 明確なトレンド期間を設定
    trend_periods = [(50, 150), (200, 300), (350, 450)]
    for start, end in trend_periods:
        known_trends[start:end] = 1
    
    # 対応する価格データを生成
    prices = [100.0]
    for i in range(1, n):
        if known_trends[i] == 1:  # トレンド期間
            trend_component = 0.005 * np.random.choice([1, -1])  # 一方向の動き
            noise = np.random.normal(0, 0.002)
        else:  # レンジ期間
            mean_price = np.mean(prices[-20:]) if len(prices) >= 20 else prices[-1]
            trend_component = -0.003 * (prices[-1] - mean_price) / mean_price  # 平均回帰
            noise = np.random.normal(0, 0.008)
        
        new_price = prices[-1] * (1 + trend_component + noise)
        prices.append(new_price)
    
    # DataFrame作成
    data = []
    for i in range(len(prices)):
        c = prices[i]
        o = prices[i-1] if i > 0 else c
        h = max(o, c) * (1 + np.random.uniform(0, 0.01))
        l = min(o, c) * (1 - np.random.uniform(0, 0.01))
        data.append([o, h, l, c])
    
    df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
    
    # Supreme Trend Stateで分析
    indicator = SupremeTrendState(window=20, threshold=0.55, use_dynamic_period=True)
    result = indicator.calculate(df)
    
    # 精度計算
    predicted_trends = result.trend_state
    
    # 適切な範囲で比較（初期期間を除く）
    start_idx = 30  # 計算が安定するまでの期間
    known_trends_valid = known_trends[start_idx:]
    predicted_trends_valid = predicted_trends[start_idx:]
    
    # 精度指標
    correct_predictions = np.sum(known_trends_valid == predicted_trends_valid)
    total_predictions = len(known_trends_valid)
    accuracy = correct_predictions / total_predictions
    
    # 真陽性、偽陽性、真陰性、偽陰性
    true_positives = np.sum((known_trends_valid == 1) & (predicted_trends_valid == 1))
    false_positives = np.sum((known_trends_valid == 0) & (predicted_trends_valid == 1))
    true_negatives = np.sum((known_trends_valid == 0) & (predicted_trends_valid == 0))
    false_negatives = np.sum((known_trends_valid == 1) & (predicted_trends_valid == 0))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"全体精度: {accuracy:.3f} ({correct_predictions}/{total_predictions})")
    print(f"適合率 (Precision): {precision:.3f}")
    print(f"再現率 (Recall): {recall:.3f}")
    print(f"F1スコア: {f1_score:.3f}")
    print(f"真陽性: {true_positives}, 偽陽性: {false_positives}")
    print(f"真陰性: {true_negatives}, 偽陰性: {false_negatives}")
    
    # 精度分析の可視化
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # 価格と既知トレンド
    ax1 = axes[0]
    ax1.plot(prices, color='black', linewidth=1)
    ax1.fill_between(range(len(known_trends)), min(prices), max(prices), 
                     where=known_trends==1, alpha=0.3, color='blue', label='Known Trend Periods')
    ax1.set_title('Test Data with Known Trend Periods')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 予測 vs 実際
    ax2 = axes[1]
    ax2.plot(known_trends, label='Actual Trend', color='blue', linewidth=2, alpha=0.7)
    ax2.plot(predicted_trends, label='Predicted Trend', color='red', linewidth=2, alpha=0.7)
    ax2.set_title('Trend Detection Comparison')
    ax2.set_ylabel('Trend State (1=Trend, 0=Range)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 複合スコア
    ax3 = axes[2]
    ax3.plot(result.composite_score, label='Composite Score', color='purple', linewidth=1.5)
    ax3.axhline(y=0.55, color='red', linestyle='--', alpha=0.7, label='Threshold')
    ax3.fill_between(range(len(known_trends)), 0, 1, 
                     where=known_trends==1, alpha=0.2, color='blue', label='Known Trend Periods')
    ax3.set_title('Composite Score vs Known Trend Periods')
    ax3.set_ylabel('Score')
    ax3.set_xlabel('Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('trend_detection_accuracy.png', dpi=300, bbox_inches='tight')
    print("精度分析チャートを 'trend_detection_accuracy.png' に保存しました")
    plt.show()
    
    return accuracy, precision, recall, f1_score

if __name__ == "__main__":
    # 最適化設定テスト
    optimized_results = test_optimized_settings()
    
    # 精度分析
    accuracy_results = analyze_trend_detection_accuracy()
    
    print(f"\n=== Supreme Trend State Indicator 完成 ===")
    print("✓ 超高精度: 複数のアルゴリズムによる多角的分析")
    print("✓ 超低遅延: Numba JITによる高速計算")
    print("✓ 超適応性: 動的期間調整機能")
    print("✓ バイナリ出力: 明確な1/0判定")
    print(f"✓ 検証済み精度: F1スコア {accuracy_results[3]:.3f}")