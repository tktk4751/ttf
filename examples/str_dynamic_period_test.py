#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from indicators.str import STR

def generate_test_data(n_points=1000):
    """テスト用データを生成"""
    np.random.seed(42)
    
    # 基本的なトレンド
    trend = np.linspace(100, 150, n_points)
    
    # サイクル成分（複数の周期を含む）
    cycle1 = 10 * np.sin(2 * np.pi * np.arange(n_points) / 30)  # 30期間サイクル
    cycle2 = 5 * np.sin(2 * np.pi * np.arange(n_points) / 60)   # 60期間サイクル
    
    # ノイズ
    noise = np.random.normal(0, 2, n_points)
    
    # 価格データの生成
    close = trend + cycle1 + cycle2 + noise
    
    # OHLC データの生成
    volatility = 3 + np.random.normal(0, 0.5, n_points)
    
    high = close + np.abs(np.random.normal(0, volatility/2, n_points))
    low = close - np.abs(np.random.normal(0, volatility/2, n_points))
    open_price = close + np.random.normal(0, volatility/4, n_points)
    
    return pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close
    })

def test_str_dynamic_periods():
    """STRの動的期間対応をテスト"""
    
    print("🔄 STR動的期間対応テスト開始")
    print("=" * 60)
    
    # テストデータの生成
    data = generate_test_data(1000)
    print(f"📊 テストデータ生成完了: {len(data)} points")
    
    # 固定期間版STRの計算
    print("\n🔧 固定期間版STR計算中...")
    str_fixed = STR(
        period=20.0,
        period_mode='fixed',
        src_type='hlc3'
    )
    
    result_fixed = str_fixed.calculate(data)
    print(f"✅ 固定期間版STR計算完了")
    print(f"   期間: {str_fixed.period}")
    print(f"   STR値範囲: {np.min(result_fixed.values):.3f} - {np.max(result_fixed.values):.3f}")
    
    # 動的期間版STRの計算
    print("\n🔧 動的期間版STR計算中...")
    str_dynamic = STR(
        period=20.0,
        period_mode='dynamic',
        src_type='hlc3',
        cycle_detector_type='absolute_ultimate',
        cycle_detector_cycle_part=1.0,
        cycle_detector_max_cycle=60,
        cycle_detector_min_cycle=10,
        cycle_period_multiplier=1.0,
        cycle_detector_period_range=(10, 60)
    )
    
    result_dynamic = str_dynamic.calculate(data)
    print(f"✅ 動的期間版STR計算完了")
    
    # 動的期間情報の取得
    dynamic_info = str_dynamic.get_dynamic_periods_info()
    print(f"   期間モード: {dynamic_info['period_mode']}")
    print(f"   サイクル検出器: {dynamic_info.get('cycle_detector_type', 'N/A')}")
    print(f"   STR値範囲: {np.min(result_dynamic.values):.3f} - {np.max(result_dynamic.values):.3f}")
    
    # 結果の統計比較
    print("\n📊 結果統計比較:")
    print(f"   固定期間版 - 平均: {np.mean(result_fixed.values):.3f}, 標準偏差: {np.std(result_fixed.values):.3f}")
    print(f"   動的期間版 - 平均: {np.mean(result_dynamic.values):.3f}, 標準偏差: {np.std(result_dynamic.values):.3f}")
    
    # 差分の計算
    diff = result_dynamic.values - result_fixed.values
    print(f"   差分（動的-固定）- 平均: {np.mean(diff):.3f}, 標準偏差: {np.std(diff):.3f}")
    
    # 可視化
    print("\n📈 結果可視化中...")
    
    # データの最後500点を使用して可視化
    plot_start = max(0, len(data) - 500)
    plot_data = data.iloc[plot_start:].copy()
    plot_data.reset_index(drop=True, inplace=True)
    
    str_fixed_plot = result_fixed.values[plot_start:]
    str_dynamic_plot = result_dynamic.values[plot_start:]
    tr_plot = result_fixed.true_range[plot_start:]
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 1. 価格データ
    axes[0].plot(plot_data['close'], label='Close Price', color='blue', linewidth=1)
    axes[0].plot(plot_data['high'], label='High', color='lightgreen', alpha=0.7, linewidth=0.8)
    axes[0].plot(plot_data['low'], label='Low', color='lightcoral', alpha=0.7, linewidth=0.8)
    axes[0].set_title('Price Data (Last 500 points)')
    axes[0].set_ylabel('Price')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. True Range
    axes[1].plot(tr_plot, label='True Range', color='orange', linewidth=1)
    axes[1].set_title('True Range')
    axes[1].set_ylabel('True Range')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. STR比較
    axes[2].plot(str_fixed_plot, label='STR Fixed (20.0)', color='blue', linewidth=1.5)
    axes[2].plot(str_dynamic_plot, label='STR Dynamic (Adaptive)', color='red', linewidth=1.5)
    axes[2].set_title('STR Comparison: Fixed vs Dynamic Periods')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('STR Value')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 結果の保存
    output_path = 'output/str_dynamic_period_test.png'
    os.makedirs('output', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📁 結果グラフを保存: {output_path}")
    
    plt.show()
    
    # 統計レポートの保存
    report_path = 'output/str_dynamic_period_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("STR 動的期間対応テスト レポート\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"テストデータ: {len(data)} データポイント\n")
        f.write(f"テスト日時: {pd.Timestamp.now()}\n\n")
        
        f.write("固定期間版STR設定:\n")
        f.write(f"  期間: {str_fixed.period}\n")
        f.write(f"  期間モード: {str_fixed.period_mode}\n\n")
        
        f.write("動的期間版STR設定:\n")
        f.write(f"  期間: {str_dynamic.period}\n")
        f.write(f"  期間モード: {str_dynamic.period_mode}\n")
        f.write(f"  サイクル検出器: {str_dynamic.cycle_detector_type}\n")
        f.write(f"  サイクル期間範囲: {str_dynamic.cycle_detector_period_range}\n\n")
        
        f.write("結果統計:\n")
        f.write(f"  固定期間版STR - 平均: {np.mean(result_fixed.values):.6f}, 標準偏差: {np.std(result_fixed.values):.6f}\n")
        f.write(f"  動的期間版STR - 平均: {np.mean(result_dynamic.values):.6f}, 標準偏差: {np.std(result_dynamic.values):.6f}\n")
        f.write(f"  差分（動的-固定）- 平均: {np.mean(diff):.6f}, 標準偏差: {np.std(diff):.6f}\n\n")
        
        f.write("動的期間情報:\n")
        for key, value in dynamic_info.items():
            f.write(f"  {key}: {value}\n")
    
    print(f"📁 統計レポートを保存: {report_path}")
    
    print("\n✅ STR動的期間対応テスト完了")
    print("=" * 60)

if __name__ == "__main__":
    test_str_dynamic_periods() 