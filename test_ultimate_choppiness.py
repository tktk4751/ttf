#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime
import time

from indicators.ultimate_choppiness_index import UltimateChoppinessIndex
from indicators.choppiness import ChoppinessIndex

# サンプルデータ生成用
def generate_sample_data(n_points=1000):
    """サンプル市場データを生成"""
    dates = pd.date_range(start='2023-01-01', periods=n_points, freq='4h')
    
    # トレンドとレンジを含む価格データを生成
    np.random.seed(42)
    base_price = 100.0
    prices = []
    
    for i in range(n_points):
        if i < n_points * 0.3:  # 最初の30%はトレンド上昇
            trend = 0.1 * i
            noise = np.random.normal(0, 0.5)
        elif i < n_points * 0.5:  # 次の20%はレンジ
            trend = 0.1 * n_points * 0.3
            noise = np.random.normal(0, 1.0) * 2
        elif i < n_points * 0.8:  # 次の30%はトレンド下降
            trend = 0.1 * n_points * 0.3 - 0.05 * (i - n_points * 0.5)
            noise = np.random.normal(0, 0.5)
        else:  # 最後の20%は再びレンジ
            trend = 0.1 * n_points * 0.3 - 0.05 * n_points * 0.3
            noise = np.random.normal(0, 1.0) * 2
        
        price = base_price + trend + noise
        prices.append(max(price, base_price * 0.5))  # 最低価格制限
    
    prices = np.array(prices)
    
    # OHLCデータを生成
    df = pd.DataFrame(index=dates)
    df['close'] = prices
    
    # ランダムな変動でOHLを生成
    df['high'] = df['close'] + np.abs(np.random.normal(0, 0.5, n_points))
    df['low'] = df['close'] - np.abs(np.random.normal(0, 0.5, n_points))
    df['open'] = df['close'].shift(1)
    df.loc[df.index[0], 'open'] = df.loc[df.index[0], 'close']
    df['volume'] = np.random.uniform(1000, 10000, n_points)
    
    return df

def load_market_data(symbol='SUIUSDT', timeframe='4h', start_date='2021-01-01', end_date='2025-01-01'):
    """市場データをロード（サンプルデータで代替）"""
    # 実際のデータローダーがない場合はサンプルデータを使用
    print("サンプルデータを生成中...")
    return generate_sample_data(n_points=1500)

def test_performance_comparison(data):
    """パフォーマンス比較テスト"""
    print("=== パフォーマンス比較テスト ===")
    
    # Traditional Choppiness Index
    traditional_chop = ChoppinessIndex(period=14)
    start_time = time.time()
    traditional_values = traditional_chop.calculate(data)
    traditional_time = time.time() - start_time
    
    # Ultimate Choppiness Index (Fixed period)
    ultimate_chop_fixed = UltimateChoppinessIndex(
        period=14.0,
        period_mode='fixed',
        smooth_period=1  # スムージングなし
    )
    start_time = time.time()
    ultimate_result_fixed = ultimate_chop_fixed.calculate(data)
    ultimate_fixed_time = time.time() - start_time
    
    # Ultimate Choppiness Index (Dynamic period)
    ultimate_chop_dynamic = UltimateChoppinessIndex(
        period=14.0,
        period_mode='dynamic',
        smooth_period=3,
        cycle_detector_type='absolute_ultimate'
    )
    start_time = time.time()
    ultimate_result_dynamic = ultimate_chop_dynamic.calculate(data)
    ultimate_dynamic_time = time.time() - start_time
    
    print(f"\n計算時間:")
    print(f"Traditional Choppiness: {traditional_time:.4f}秒")
    print(f"Ultimate Choppiness (Fixed): {ultimate_fixed_time:.4f}秒")
    print(f"Ultimate Choppiness (Dynamic): {ultimate_dynamic_time:.4f}秒")
    
    # 相関分析
    mask = ~np.isnan(traditional_values) & ~np.isnan(ultimate_result_fixed.values)
    correlation_fixed = np.corrcoef(traditional_values[mask], ultimate_result_fixed.values[mask])[0, 1]
    
    mask_dynamic = ~np.isnan(traditional_values) & ~np.isnan(ultimate_result_dynamic.values)
    correlation_dynamic = np.corrcoef(traditional_values[mask_dynamic], ultimate_result_dynamic.values[mask_dynamic])[0, 1]
    
    print(f"\n相関係数:")
    print(f"Traditional vs Ultimate (Fixed): {correlation_fixed:.4f}")
    print(f"Traditional vs Ultimate (Dynamic): {correlation_dynamic:.4f}")
    
    return {
        'traditional': traditional_values,
        'ultimate_fixed': ultimate_result_fixed,
        'ultimate_dynamic': ultimate_result_dynamic,
        'times': {
            'traditional': traditional_time,
            'ultimate_fixed': ultimate_fixed_time,
            'ultimate_dynamic': ultimate_dynamic_time
        }
    }

def analyze_trend_detection(results):
    """トレンド検出分析"""
    print("\n=== トレンド検出分析 ===")
    
    # Traditional Choppiness
    traditional = results['traditional']
    trend_traditional = traditional <= 38.2
    trend_ratio_traditional = np.sum(trend_traditional) / len(traditional) * 100
    
    # Ultimate Fixed
    ultimate_fixed = results['ultimate_fixed']
    trend_ratio_fixed = np.sum(ultimate_fixed.trend_state == 1) / len(ultimate_fixed.trend_state) * 100
    
    # Ultimate Dynamic
    ultimate_dynamic = results['ultimate_dynamic']
    trend_ratio_dynamic = np.sum(ultimate_dynamic.trend_state == 1) / len(ultimate_dynamic.trend_state) * 100
    
    print(f"\nトレンド判定率:")
    print(f"Traditional Choppiness: {trend_ratio_traditional:.2f}%")
    print(f"Ultimate Choppiness (Fixed): {trend_ratio_fixed:.2f}%")
    print(f"Ultimate Choppiness (Dynamic): {trend_ratio_dynamic:.2f}%")
    
    # 平均値分析
    print(f"\n平均チョピネス値:")
    print(f"Traditional: {np.nanmean(traditional):.2f}")
    print(f"Ultimate (Fixed): {np.nanmean(ultimate_fixed.values):.2f}")
    print(f"Ultimate (Dynamic): {np.nanmean(ultimate_dynamic.values):.2f}")
    
    # STR値の分析
    print(f"\n平均STR値:")
    print(f"Ultimate (Fixed): {np.nanmean(ultimate_fixed.str_values):.4f}")
    print(f"Ultimate (Dynamic): {np.nanmean(ultimate_dynamic.str_values):.4f}")
    
    # 動的期間の分析
    print(f"\n動的期間統計 (Dynamic):")
    periods = ultimate_dynamic.dynamic_periods
    print(f"最小: {np.min(periods):.1f}")
    print(f"最大: {np.max(periods):.1f}")
    print(f"平均: {np.mean(periods):.1f}")
    print(f"標準偏差: {np.std(periods):.1f}")

def create_comparison_chart(data, results, save_path='ultimate_choppiness_comparison.png'):
    """比較チャートの作成"""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(5, 1, height_ratios=[2, 1.5, 1.5, 1.5, 1], hspace=0.3)
    
    # 価格チャート
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(data.index, data['close'], 'k-', linewidth=1.5, label='Close Price')
    
    # トレンド期間のハイライト（Ultimate Dynamic）
    trend_mask = results['ultimate_dynamic'].trend_state == 1
    for i in range(1, len(trend_mask)):
        if trend_mask[i] and not trend_mask[i-1]:
            start_idx = i
        elif not trend_mask[i] and trend_mask[i-1]:
            ax1.axvspan(data.index[start_idx], data.index[i-1], alpha=0.2, color='green')
    
    ax1.set_title('Price Chart with Trend Periods (Ultimate Dynamic)', fontsize=14)
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Traditional Choppiness
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(data.index, results['traditional'], 'b-', linewidth=1, label='Traditional')
    ax2.axhline(y=38.2, color='g', linestyle='--', alpha=0.5, label='Trend (38.2)')
    ax2.axhline(y=61.8, color='r', linestyle='--', alpha=0.5, label='Range (61.8)')
    ax2.fill_between(data.index, 0, 38.2, alpha=0.1, color='green')
    ax2.fill_between(data.index, 61.8, 100, alpha=0.1, color='red')
    ax2.set_ylabel('Choppiness')
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Traditional Choppiness Index')
    
    # Ultimate Choppiness (Fixed)
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(data.index, results['ultimate_fixed'].values, 'purple', linewidth=1, label='Ultimate (Fixed)')
    ax3.axhline(y=38.2, color='g', linestyle='--', alpha=0.5)
    ax3.axhline(y=61.8, color='r', linestyle='--', alpha=0.5)
    ax3.fill_between(data.index, 0, 38.2, alpha=0.1, color='green')
    ax3.fill_between(data.index, 61.8, 100, alpha=0.1, color='red')
    ax3.set_ylabel('Choppiness')
    ax3.set_ylim(0, 100)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Ultimate Choppiness Index (Fixed Period)')
    
    # Ultimate Choppiness (Dynamic)
    ax4 = fig.add_subplot(gs[3])
    ax4.plot(data.index, results['ultimate_dynamic'].values, 'orange', linewidth=1, label='Ultimate (Dynamic)')
    ax4.axhline(y=38.2, color='g', linestyle='--', alpha=0.5)
    ax4.axhline(y=61.8, color='r', linestyle='--', alpha=0.5)
    ax4.fill_between(data.index, 0, 38.2, alpha=0.1, color='green')
    ax4.fill_between(data.index, 61.8, 100, alpha=0.1, color='red')
    ax4.set_ylabel('Choppiness')
    ax4.set_ylim(0, 100)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Ultimate Choppiness Index (Dynamic Period)')
    
    # 動的期間
    ax5 = fig.add_subplot(gs[4])
    ax5.plot(data.index, results['ultimate_dynamic'].dynamic_periods, 'gray', linewidth=1)
    ax5.axhline(y=14, color='blue', linestyle='--', alpha=0.5, label='Base Period (14)')
    ax5.set_ylabel('Period')
    ax5.set_xlabel('Date')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_title('Dynamic Period (Ultimate Dynamic)')
    
    plt.suptitle(f'Ultimate Choppiness Index Comparison - {data.index[0].strftime("%Y-%m-%d")} to {data.index[-1].strftime("%Y-%m-%d")}', 
                 fontsize=16, y=0.995)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nチャートを保存しました: {save_path}")
    plt.close()

def main():
    """メイン実行関数"""
    print("Ultimate Choppiness Index テスト開始")
    print("=" * 50)
    
    # データ読み込み
    print("\n市場データを読み込み中...")
    data = load_market_data(symbol='SUIUSDT', timeframe='4h')
    print(f"データ期間: {data.index[0]} から {data.index[-1]}")
    print(f"データ数: {len(data)}")
    
    # パフォーマンステスト
    results = test_performance_comparison(data)
    
    # トレンド検出分析
    analyze_trend_detection(results)
    
    # チャート作成
    print("\n比較チャートを作成中...")
    create_comparison_chart(data, results)
    
    # 特徴の要約
    print("\n=== Ultimate Choppiness Indexの特徴 ===")
    print("1. STRによる超低遅延計算")
    print("2. 動的期間調整によるマーケット適応")
    print("3. ノイズ除去のための平滑化機能")
    print("4. より現実的なトレンド/レンジ判定")
    print("5. トレンド状態の明確なバイナリ出力")
    
    print("\n=== テスト完了 ===")

if __name__ == "__main__":
    main()