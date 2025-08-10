#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time

from indicators.ultimate_er import UltimateER
from indicators.efficiency_ratio import EfficiencyRatio

def generate_test_data(n_points=1000):
    """テスト用の市場データを生成"""
    dates = pd.date_range(start='2023-01-01', periods=n_points, freq='4h')
    
    # トレンドとレンジを含む価格データを生成
    np.random.seed(42)
    base_price = 100.0
    prices = []
    
    for i in range(n_points):
        if i < n_points * 0.3:  # 最初の30%は強いトレンド上昇
            trend = 0.15 * i
            noise = np.random.normal(0, 0.3)
        elif i < n_points * 0.5:  # 次の20%はレンジ
            trend = 0.15 * n_points * 0.3
            noise = np.random.normal(0, 1.5)
        elif i < n_points * 0.8:  # 次の30%は強いトレンド下降
            trend = 0.15 * n_points * 0.3 - 0.1 * (i - n_points * 0.5)
            noise = np.random.normal(0, 0.3)
        else:  # 最後の20%は再びレンジ
            trend = 0.15 * n_points * 0.3 - 0.1 * n_points * 0.3
            noise = np.random.normal(0, 1.5)
        
        price = base_price + trend + noise
        prices.append(max(price, base_price * 0.5))
    
    prices = np.array(prices)
    
    # OHLCデータを生成
    df = pd.DataFrame(index=dates)
    df['close'] = prices
    df['high'] = df['close'] + np.abs(np.random.normal(0, 0.5, n_points))
    df['low'] = df['close'] - np.abs(np.random.normal(0, 0.5, n_points))
    df['open'] = df['close'].shift(1)
    df.loc[df.index[0], 'open'] = df.loc[df.index[0], 'close']
    df['volume'] = np.random.uniform(1000, 10000, n_points)
    
    return df

def test_performance_comparison(data):
    """パフォーマンス比較テスト"""
    print("=== パフォーマンス比較テスト ===")
    
    # Traditional ER
    traditional_er = EfficiencyRatio(
        period=20,
        src_type='hlc3',
        smoothing_method='none',
        use_dynamic_period=False
    )
    start_time = time.time()
    traditional_result = traditional_er.calculate(data)
    traditional_time = time.time() - start_time
    
    # Ultimate ER (新ロジック: 5期間ER + UKF + 20期間Smoother)
    ultimate_er = UltimateER(
        er_period=5.0,  # 5期間のER
        src_type='hlc3',
        ukf_alpha=0.001,
        smoother_period=20.0  # 20期間のUltimate Smoother
    )
    start_time = time.time()
    ultimate_result = ultimate_er.calculate(data)
    ultimate_time = time.time() - start_time
    
    print(f"\n計算時間:")
    print(f"Traditional ER: {traditional_time:.4f}秒")
    print(f"Ultimate ER: {ultimate_time:.4f}秒")
    
    # 統計分析
    print(f"\n平均ER値:")
    print(f"Traditional: {np.nanmean(traditional_result.values):.4f}")
    print(f"Ultimate: {np.nanmean(ultimate_result.values):.4f}")
    
    # トレンド検出分析
    traditional_trending = np.sum(traditional_result.trend_signals != 0) / len(traditional_result.trend_signals) * 100
    ultimate_trending = np.sum(ultimate_result.trend_signals != 0) / len(ultimate_result.trend_signals) * 100
    
    print(f"\nトレンド判定率:")
    print(f"Traditional: {traditional_trending:.2f}%")
    print(f"Ultimate: {ultimate_trending:.2f}%")
    
    return {
        'traditional': traditional_result,
        'ultimate': ultimate_result,
        'times': {
            'traditional': traditional_time,
            'ultimate': ultimate_time
        }
    }

def analyze_filtering_effect(results):
    """フィルタリング効果の分析"""
    print("\n=== フィルタリング効果分析 ===")
    
    # Ultimate ERの各段階を分析
    ultimate_result = results['ultimate']
    
    # ノイズ除去効果（標準偏差の比較）
    raw_std = np.nanstd(ultimate_result.raw_er)
    ukf_std = np.nanstd(ultimate_result.ukf_filtered)
    final_std = np.nanstd(ultimate_result.values)
    
    print(f"\nノイズ除去効果（標準偏差）:")
    print(f"Raw ER (5期間): {raw_std:.4f}")
    print(f"UKF Filtered: {ukf_std:.4f} (削減率: {(1-ukf_std/raw_std)*100:.1f}%)")
    print(f"Ultimate ER: {final_std:.4f} (削減率: {(1-final_std/raw_std)*100:.1f}%)")
    
    # 信頼度スコアの統計
    print(f"\n信頼度スコア統計:")
    print(f"平均: {np.nanmean(ultimate_result.confidence_scores):.4f}")
    print(f"最小: {np.nanmin(ultimate_result.confidence_scores):.4f}")
    print(f"最大: {np.nanmax(ultimate_result.confidence_scores):.4f}")

def create_comparison_chart(data, results, save_path='ultimate_er_comparison.png'):
    """比較チャートの作成"""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(5, 1, height_ratios=[2, 1.5, 1.5, 1, 1], hspace=0.3)
    
    # 1. 価格チャート
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(data.index, data['close'], 'k-', linewidth=1.5, label='Close Price')
    ax1.set_title('Price Chart', fontsize=14)
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Traditional ER
    ax2 = fig.add_subplot(gs[1])
    traditional_values = results['traditional'].values
    ax2.plot(data.index, traditional_values, 'b-', linewidth=1, label='Traditional ER')
    ax2.axhline(y=0.618, color='g', linestyle='--', alpha=0.5, label='Strong Trend (0.618)')
    ax2.axhline(y=0.382, color='r', linestyle='--', alpha=0.5, label='Weak Trend (0.382)')
    ax2.fill_between(data.index, 0.618, 1.0, alpha=0.1, color='green')
    ax2.fill_between(data.index, 0, 0.382, alpha=0.1, color='red')
    ax2.set_ylabel('ER Value')
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Traditional Efficiency Ratio')
    
    # 3. Ultimate ER - 各段階
    ax3 = fig.add_subplot(gs[2])
    ultimate_result = results['ultimate']
    ax3.plot(data.index, ultimate_result.raw_er, 'gray', linewidth=0.5, alpha=0.5, label='Raw ER (5期間)')
    ax3.plot(data.index, ultimate_result.ukf_filtered, 'purple', linewidth=1, label='UKF Filtered')
    ax3.plot(data.index, ultimate_result.values, 'darkgreen', linewidth=2, label='Ultimate ER (20期間Smooth)')
    ax3.axhline(y=0.618, color='g', linestyle='--', alpha=0.5)
    ax3.axhline(y=0.382, color='r', linestyle='--', alpha=0.5)
    ax3.set_ylabel('ER Value')
    ax3.set_ylim(0, 1)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Ultimate ER - Processing Stages (5-period ER → UKF → 20-period Smoother)')
    
    # 4. 信頼度スコア
    ax4 = fig.add_subplot(gs[3])
    ax4.plot(data.index, ultimate_result.confidence_scores, 'blue', linewidth=1, label='UKF Confidence')
    ax4.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Baseline (0.5)')
    ax4.set_ylabel('Confidence')
    ax4.set_ylim(0, 1)
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.set_title('UKF Confidence Scores')
    
    # 5. トレンド信号
    ax5 = fig.add_subplot(gs[4])
    ax5.plot(data.index, ultimate_result.trend_signals, 'g-', linewidth=1.5, label='Ultimate ER Trend')
    ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax5.axhline(y=1, color='green', linestyle='--', alpha=0.3, label='Uptrend')
    ax5.axhline(y=-1, color='red', linestyle='--', alpha=0.3, label='Downtrend')
    ax5.set_ylabel('Trend Signal')
    ax5.set_ylim(-1.5, 1.5)
    ax5.set_xlabel('Date')
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)
    ax5.set_title('Trend Signals (1=Up, 0=Range, -1=Down)')
    
    plt.suptitle('Ultimate Efficiency Ratio Comparison', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nチャートを保存しました: {save_path}")
    plt.close()

def main():
    """メイン実行関数"""
    print("Ultimate Efficiency Ratio テスト開始")
    print("=" * 50)
    
    # データ生成
    print("\nテストデータを生成中...")
    data = generate_test_data(n_points=1500)
    print(f"データ期間: {data.index[0]} から {data.index[-1]}")
    print(f"データ数: {len(data)}")
    
    # パフォーマンステスト
    results = test_performance_comparison(data)
    
    # フィルタリング効果分析
    analyze_filtering_effect(results)
    
    # チャート作成
    print("\n比較チャートを作成中...")
    create_comparison_chart(data, results)
    
    # Ultimate ERの特徴まとめ
    print("\n=== Ultimate ER（新ロジック）の特徴 ===")
    print("1. 5期間の短期ERによる高感度な変化検出")
    print("2. UKFによる高精度ノイズ除去")
    print("3. 20期間Ultimate Smootherによる安定した平滑化")
    print("4. 信頼度スコアに基づく適応的重み付け")
    print("5. 短期感度と長期安定性のバランス")
    
    print("\n=== テスト完了 ===")

if __name__ == "__main__":
    main()