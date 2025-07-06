#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
効率比比較テスト：従来ER vs ハイパーER
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# インジケーターのインポート
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicators.efficiency_ratio import EfficiencyRatio
from indicators.hyper_efficiency_ratio import HyperEfficiencyRatio

# 英語フォントの設定
plt.rcParams['font.family'] = ['DejaVu Sans']
import matplotlib
matplotlib.use('Agg')  # フォント警告を防ぐためにAggバックエンドを使用


def generate_test_data(n_samples: int = 500, scenario: str = "mixed") -> pd.DataFrame:
    """
    テスト用データの生成
    
    Args:
        n_samples: サンプル数
        scenario: シナリオタイプ ('trend', 'range', 'volatile', 'mixed')
        
    Returns:
        OHLCVデータフレーム
    """
    dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='H')
    
    # ベース価格
    base_price = 100.0
    prices = [base_price]
    
    # シナリオ別価格生成
    for i in range(1, n_samples):
        if scenario == "trend":
            # トレンド相場（一方向性の強い動き）
            trend = 0.002 if i < n_samples // 2 else -0.001
            noise = np.random.normal(0, 0.001)
            change = trend + noise
        elif scenario == "range":
            # レンジ相場（往復動き）
            cycle_pos = (i % 100) / 100.0
            trend = 0.01 * np.sin(2 * np.pi * cycle_pos)
            noise = np.random.normal(0, 0.003)
            change = trend + noise
        elif scenario == "volatile":
            # 高ボラティリティ相場
            change = np.random.normal(0, 0.01)
        else:  # mixed
            # 混合シナリオ
            if i < n_samples // 3:
                # 前半：トレンド
                trend = 0.003
                noise = np.random.normal(0, 0.002)
                change = trend + noise
            elif i < 2 * n_samples // 3:
                # 中盤：レンジ
                cycle_pos = ((i - n_samples // 3) % 80) / 80.0
                trend = 0.02 * np.sin(2 * np.pi * cycle_pos)
                noise = np.random.normal(0, 0.004)
                change = trend + noise
            else:
                # 後半：ボラティリティ
                change = np.random.normal(0, 0.008)
        
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    prices = np.array(prices)
    
    # OHLC生成
    highs = prices * (1 + np.abs(np.random.normal(0, 0.005, n_samples)))
    lows = prices * (1 - np.abs(np.random.normal(0, 0.005, n_samples)))
    opens = np.roll(prices, 1)
    opens[0] = prices[0]
    
    # ボリューム生成
    volumes = np.random.randint(1000, 10000, n_samples)
    
    data = pd.DataFrame({
        'datetime': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': volumes
    })
    
    data.set_index('datetime', inplace=True)
    return data


def run_comparison_test(data: pd.DataFrame, period: int = 14) -> dict:
    """
    効率比比較テストの実行
    
    Args:
        data: 価格データ
        period: 計算期間
        
    Returns:
        比較結果辞書
    """
    print(f"🔬 効率比比較テスト開始 (期間: {period})")
    
    # 従来のEfficiency Ratio
    print("📊 従来ER計算中...")
    classic_er = EfficiencyRatio(period=period, src_type='hlc3', use_dynamic_period=False)
    classic_result = classic_er.calculate(data)
    
    # ハイパー効率比
    print("🚀 ハイパーER計算中...")
    hyper_er = HyperEfficiencyRatio(window=period, src_type='hlc3')
    hyper_result = hyper_er.calculate(data)
    
    # 統計比較
    classic_values = classic_result.values
    hyper_values = hyper_result.values
    
    # 有効データの抽出
    valid_mask = ~(np.isnan(classic_values) | np.isnan(hyper_values))
    classic_valid = classic_values[valid_mask]
    hyper_valid = hyper_values[valid_mask]
    
    # 統計計算
    stats = {
        'classic_er': {
            'mean': np.mean(classic_valid),
            'std': np.std(classic_valid),
            'min': np.min(classic_valid),
            'max': np.max(classic_valid),
            'median': np.median(classic_valid),
            'quality_score': 'N/A'
        },
        'hyper_er': {
            'mean': np.mean(hyper_valid),
            'std': np.std(hyper_valid),
            'min': np.min(hyper_valid),
            'max': np.max(hyper_valid),
            'median': np.median(hyper_valid),
            'quality_score': hyper_result.quality_score
        },
        'correlation': np.corrcoef(classic_valid, hyper_valid)[0, 1],
        'data_points': len(classic_valid)
    }
    
    # トレンド信号の比較
    classic_trends = classic_result.trend_signals[valid_mask]
    hyper_trends = hyper_result.trend_signals[valid_mask]
    
    # 信号一致率の計算
    signal_agreement = np.mean(classic_trends == hyper_trends)
    
    return {
        'stats': stats,
        'signal_agreement': signal_agreement,
        'classic_result': classic_result,
        'hyper_result': hyper_result,
        'data': data,
        'period': period
    }


def plot_comparison_results(results: dict, save_path: str = None):
    """
    比較結果をプロットする
    
    Args:
        results: 比較結果辞書
        save_path: 保存パス（Noneの場合は表示のみ）
    """
    classic_result = results['classic_result']
    hyper_result = results['hyper_result']
    data = results['data']
    period = results['period']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Efficiency Ratio Comparison Analysis (Period: {period})', fontsize=16, fontweight='bold')
    
    # 1. Price and ER time series
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()
    
    # Price plot
    ax1.plot(data.index, data['close'], 'k-', linewidth=1, alpha=0.7, label='Price')
    ax1.set_ylabel('Price', fontsize=10)
    ax1.set_title('Price and Efficiency Ratio Time Series', fontsize=12, fontweight='bold')
    
    # ER plot
    ax1_twin.plot(data.index, classic_result.values, 'b-', linewidth=1.5, alpha=0.8, label='Classic ER')
    ax1_twin.plot(data.index, hyper_result.values, 'r-', linewidth=1.5, alpha=0.8, label='Hyper ER')
    ax1_twin.set_ylabel('Efficiency Ratio', fontsize=10)
    ax1_twin.set_ylim(0, 1)
    ax1_twin.axhline(y=0.618, color='orange', linestyle='--', alpha=0.5, label='Golden Ratio Threshold')
    ax1_twin.legend(loc='upper right')
    
    # Date formatting
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Scatter plot (correlation analysis)
    ax2 = axes[0, 1]
    valid_mask = ~(np.isnan(classic_result.values) | np.isnan(hyper_result.values))
    classic_valid = classic_result.values[valid_mask]
    hyper_valid = hyper_result.values[valid_mask]
    
    ax2.scatter(classic_valid, hyper_valid, alpha=0.6, s=20)
    ax2.plot([0, 1], [0, 1], 'r--', alpha=0.8, label='y=x')
    ax2.set_xlabel('Classic ER')
    ax2.set_ylabel('Hyper ER')
    ax2.set_title(f'ER Correlation Analysis (r={results["stats"]["correlation"]:.3f})')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Distribution comparison (histogram)
    ax3 = axes[1, 0]
    bins = np.linspace(0, 1, 30)
    ax3.hist(classic_valid, bins=bins, alpha=0.6, label='Classic ER', color='blue', density=True)
    ax3.hist(hyper_valid, bins=bins, alpha=0.6, label='Hyper ER', color='red', density=True)
    ax3.set_xlabel('Efficiency Ratio Value')
    ax3.set_ylabel('Density')
    ax3.set_title('ER Value Distribution Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Volatility component analysis (Hyper ER only)
    ax4 = axes[1, 1]
    valid_hyper = ~np.isnan(hyper_result.values)
    idx_valid = data.index[valid_hyper]
    
    ax4.plot(idx_valid, hyper_result.linear_volatility[valid_hyper], 
             label='Linear Vol', linewidth=1, alpha=0.8)
    ax4.plot(idx_valid, hyper_result.nonlinear_volatility[valid_hyper], 
             label='Nonlinear Vol', linewidth=1, alpha=0.8)
    ax4.plot(idx_valid, hyper_result.adaptive_volatility[valid_hyper], 
             label='Adaptive Vol', linewidth=1, alpha=0.8)
    
    ax4.set_ylabel('Volatility')
    ax4.set_title('Hyper ER: Multi-dimensional Volatility Analysis')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Chart saved: {save_path}")
    else:
        plt.show()


def print_comparison_summary(results: dict):
    """
    Print comparison results summary
    
    Args:
        results: Comparison results dictionary
    """
    stats = results['stats']
    
    print("\n" + "="*80)
    print("📊 Efficiency Ratio Comparison Summary")
    print("="*80)
    
    print(f"📈 Data Points: {stats['data_points']}")
    print(f"🔗 Correlation: {stats['correlation']:.4f}")
    print(f"📡 Signal Agreement: {results['signal_agreement']:.1%}")
    
    print("\n📊 Statistical Comparison:")
    print(f"{'Metric':<15} {'Classic ER':<12} {'Hyper ER':<12} {'Difference':<12}")
    print("-" * 60)
    
    metrics = ['mean', 'std', 'min', 'max', 'median']
    for metric in metrics:
        classic_val = stats['classic_er'][metric]
        hyper_val = stats['hyper_er'][metric]
        diff = hyper_val - classic_val
        print(f"{metric:<15} {classic_val:<12.4f} {hyper_val:<12.4f} {diff:<+12.4f}")
    
    print(f"\n🎯 Quality Score (Hyper ER): {stats['hyper_er']['quality_score']:.4f}")
    
    # Feature analysis
    print("\n🔍 Feature Analysis:")
    
    # Variance comparison
    var_ratio = stats['hyper_er']['std'] / stats['classic_er']['std']
    if var_ratio > 1.1:
        print(f"• Hyper ER shows {var_ratio:.1f}x higher volatility than Classic ER")
    elif var_ratio < 0.9:
        print(f"• Hyper ER shows {1/var_ratio:.1f}x lower volatility than Classic ER")
    else:
        print("• Both indicators show similar volatility")
    
    # Mean comparison
    mean_diff = stats['hyper_er']['mean'] - stats['classic_er']['mean']
    if abs(mean_diff) > 0.05:
        direction = "higher" if mean_diff > 0 else "lower"
        print(f"• Hyper ER shows {direction} efficiency on average (diff: {mean_diff:+.3f})")
    
    # Correlation analysis
    correlation = stats['correlation']
    if correlation > 0.8:
        print(f"• Strong positive correlation (r={correlation:.3f}) - similar market assessment")
    elif correlation > 0.5:
        print(f"• Moderate positive correlation (r={correlation:.3f}) - partially similar")
    else:
        print(f"• Weak correlation (r={correlation:.3f}) - different market assessment")


def main():
    """
    Main execution function
    """
    print("🚀 Starting Efficiency Ratio comparison test")
    print("=" * 50)
    
    # Multiple scenario test
    scenarios = ['trend', 'range', 'volatile', 'mixed']
    periods = [10, 14, 21]
    
    all_results = {}
    
    for scenario in scenarios:
        print(f"\n📊 Scenario: {scenario.upper()}")
        print("-" * 30)
        
        # Generate test data
        data = generate_test_data(n_samples=300, scenario=scenario)
        
        for period in periods:
            print(f"\nPeriod: {period}")
            
            # Run comparison test
            results = run_comparison_test(data, period)
            
            # Save results
            key = f"{scenario}_{period}"
            all_results[key] = results
            
            # Print summary
            print_comparison_summary(results)
            
            # Create chart
            save_path = f"output/er_comparison_{scenario}_{period}.png"
            os.makedirs("output", exist_ok=True)
            plot_comparison_results(results, save_path)
    
    # Overall analysis
    print("\n" + "="*80)
    print("📈 Overall Analysis Results")
    print("="*80)
    
    # Average correlation and performance
    correlations = [results['stats']['correlation'] for results in all_results.values()]
    signal_agreements = [results['signal_agreement'] for results in all_results.values()]
    
    print(f"📊 Average Correlation: {np.mean(correlations):.4f} ± {np.std(correlations):.4f}")
    print(f"📡 Average Signal Agreement: {np.mean(signal_agreements):.1%} ± {np.std(signal_agreements):.1%}")
    
    # Quality score statistics
    quality_scores = [results['hyper_result'].quality_score for results in all_results.values()]
    print(f"🎯 Hyper ER Quality Score: {np.mean(quality_scores):.4f} ± {np.std(quality_scores):.4f}")
    
    print("\n✅ Efficiency Ratio comparison test completed")
    print(f"📁 Result files: output/er_comparison_*.png")


if __name__ == "__main__":
    main() 