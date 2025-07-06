#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EfficiencyRatio V2 インジケーターの使用例

このスクリプトは、新しいER_V2インジケーターの基本的な使用方法を示します。
- UKF_HLC3プライスソース（カルマンフィルター適応済み価格）
- UltimateSmoother による高品質な平滑化
- 動的期間（ドミナントサイクル）対応
- 高精度なトレンド判定
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# TTFライブラリからのインポート
from indicators.efficiency_ratio_v2 import ER_V2
from data.data_loader import DataLoader

def generate_sample_data(length=1000):
    """
    サンプルデータを生成する
    
    Args:
        length: データ点数
    
    Returns:
        pandas.DataFrame: OHLCV形式のサンプルデータ
    """
    np.random.seed(42)
    
    # 基本価格の生成（トレンドとノイズを含む）
    dates = pd.date_range(start='2023-01-01', periods=length, freq='D')
    
    # 基本価格の生成
    base_price = 100.0
    trend = np.cumsum(np.random.randn(length) * 0.02)  # トレンド成分
    noise = np.random.randn(length) * 0.5  # ノイズ成分
    
    # 周期的なパターンの追加
    cycle = np.sin(np.arange(length) * 2 * np.pi / 50) * 2.0  # 50日周期
    
    # 終値の生成
    close = base_price + trend + noise + cycle
    
    # OHLC価格の生成
    high = close + np.abs(np.random.randn(length) * 0.5)
    low = close - np.abs(np.random.randn(length) * 0.5)
    
    # 始値の生成（前日の終値をベースに）
    open_price = np.zeros(length)
    open_price[0] = close[0] + np.random.randn() * 0.1
    open_price[1:] = close[:-1] + np.random.randn(length-1) * 0.1
    
    # ボリュームの生成
    volume = np.abs(np.random.randn(length) * 1000 + 10000)
    
    data = pd.DataFrame({
        'date': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    return data

def main():
    """メイン関数"""
    print("=" * 60)
    print("EfficiencyRatio V2 インジケーターの使用例")
    print("=" * 60)
    
    # サンプルデータの生成
    print("\n1. サンプルデータの生成...")
    data = generate_sample_data(1000)
    print(f"   データ点数: {len(data)}")
    print(f"   期間: {data['date'].min()} - {data['date'].max()}")
    print(f"   価格範囲: {data['close'].min():.2f} - {data['close'].max():.2f}")
    
    # ER_V2インジケーターの作成と設定
    print("\n2. ER_V2インジケーターの初期化...")
    
    # 基本設定
    er_v2_basic = ER_V2(
        period=5,                      # 基本期間
        src_type='ukf_hlc3',          # UKF_HLC3プライスソース
        use_ultimate_smoother=True,    # UltimateSmoother使用
        smoother_period=10.0,          # 平滑化期間
        use_dynamic_period=False,      # 固定期間モード
        slope_index=3,                 # トレンド判定期間
        range_threshold=0.005          # レンジ判定閾値
    )
    
    # 動的期間設定
    er_v2_dynamic = ER_V2(
        period=5,                      # 基本期間
        src_type='ukf_hlc3',          # UKF_HLC3プライスソース
        use_ultimate_smoother=True,    # UltimateSmoother使用
        smoother_period=10.0,          # 平滑化期間
        use_dynamic_period=True,       # 動的期間モード
        detector_type='absolute_ultimate',  # ドミナントサイクル検出方法
        max_cycle=120,                 # 最大サイクル
        min_cycle=5,                   # 最小サイクル
        slope_index=3,                 # トレンド判定期間
        range_threshold=0.005          # レンジ判定閾値
    )
    
    print(f"   基本設定: {er_v2_basic.name}")
    print(f"   動的期間設定: {er_v2_dynamic.name}")
    
    # 計算の実行
    print("\n3. ER_V2計算の実行...")
    
    # 基本設定での計算
    result_basic = er_v2_basic.calculate(data)
    print(f"   基本設定 - 計算完了")
    print(f"   ER値範囲: {np.nanmin(result_basic.values):.4f} - {np.nanmax(result_basic.values):.4f}")
    print(f"   平滑化値範囲: {np.nanmin(result_basic.smoothed_values):.4f} - {np.nanmax(result_basic.smoothed_values):.4f}")
    print(f"   現在のトレンド: {result_basic.current_trend}")
    
    # 動的期間設定での計算
    result_dynamic = er_v2_dynamic.calculate(data)
    print(f"   動的期間設定 - 計算完了")
    print(f"   ER値範囲: {np.nanmin(result_dynamic.values):.4f} - {np.nanmax(result_dynamic.values):.4f}")
    print(f"   平滑化値範囲: {np.nanmin(result_dynamic.smoothed_values):.4f} - {np.nanmax(result_dynamic.smoothed_values):.4f}")
    print(f"   現在のトレンド: {result_dynamic.current_trend}")
    print(f"   動的期間統計: 平均={np.nanmean(result_dynamic.dynamic_periods):.1f}, 範囲={np.nanmin(result_dynamic.dynamic_periods):.0f}-{np.nanmax(result_dynamic.dynamic_periods):.0f}")
    
    # トレンド信号の分析
    print("\n4. トレンド信号の分析...")
    
    def analyze_trend_signals(trend_signals, name):
        """トレンド信号の分析"""
        valid_signals = trend_signals[trend_signals != 0]  # 0以外の信号
        up_signals = np.sum(trend_signals == 1)
        down_signals = np.sum(trend_signals == -1)
        range_signals = np.sum(trend_signals == 0)
        total_signals = len(trend_signals)
        
        print(f"   {name}:")
        print(f"     上昇トレンド: {up_signals} ({up_signals/total_signals*100:.1f}%)")
        print(f"     下降トレンド: {down_signals} ({down_signals/total_signals*100:.1f}%)")
        print(f"     レンジ相場: {range_signals} ({range_signals/total_signals*100:.1f}%)")
    
    analyze_trend_signals(result_basic.trend_signals, "基本設定")
    analyze_trend_signals(result_dynamic.trend_signals, "動的期間設定")
    
    # 効率比の統計
    print("\n5. 効率比の統計情報...")
    
    def analyze_efficiency_ratio(values, smoothed_values, name):
        """効率比の統計分析"""
        valid_values = values[~np.isnan(values)]
        valid_smoothed = smoothed_values[~np.isnan(smoothed_values)]
        
        if len(valid_values) > 0:
            print(f"   {name}:")
            print(f"     原値 - 平均: {np.mean(valid_values):.4f}, 標準偏差: {np.std(valid_values):.4f}")
            print(f"     平滑値 - 平均: {np.mean(valid_smoothed):.4f}, 標準偏差: {np.std(valid_smoothed):.4f}")
            print(f"     効率的(>0.618): {np.sum(valid_smoothed > 0.618)} ({np.sum(valid_smoothed > 0.618)/len(valid_smoothed)*100:.1f}%)")
            print(f"     非効率(<0.382): {np.sum(valid_smoothed < 0.382)} ({np.sum(valid_smoothed < 0.382)/len(valid_smoothed)*100:.1f}%)")
    
    analyze_efficiency_ratio(result_basic.values, result_basic.smoothed_values, "基本設定")
    analyze_efficiency_ratio(result_dynamic.values, result_dynamic.smoothed_values, "動的期間設定")
    
    # 視覚化の作成
    print("\n6. 視覚化の作成...")
    
    # プロットの設定
    fig, axes = plt.subplots(4, 1, figsize=(15, 16))
    fig.suptitle('EfficiencyRatio V2 インジケーター分析', fontsize=16, fontweight='bold')
    
    # 日付インデックス
    dates = data['date'].values
    
    # 1. 価格チャート
    ax1 = axes[0]
    ax1.plot(dates, data['close'].values, label='終値', linewidth=1.5, color='blue')
    ax1.set_title('価格チャート', fontsize=14, fontweight='bold')
    ax1.set_ylabel('価格', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. ER_V2値の比較
    ax2 = axes[1]
    ax2.plot(dates, result_basic.values, label='基本設定 (原値)', alpha=0.7, color='orange')
    ax2.plot(dates, result_basic.smoothed_values, label='基本設定 (平滑化)', linewidth=2, color='red')
    ax2.plot(dates, result_dynamic.smoothed_values, label='動的期間設定 (平滑化)', linewidth=2, color='green')
    ax2.axhline(y=0.618, color='purple', linestyle='--', alpha=0.8, label='効率的閾値 (0.618)')
    ax2.axhline(y=0.382, color='brown', linestyle='--', alpha=0.8, label='非効率閾値 (0.382)')
    ax2.set_title('EfficiencyRatio V2 値', fontsize=14, fontweight='bold')
    ax2.set_ylabel('ER値', fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. トレンド信号
    ax3 = axes[2]
    ax3.plot(dates, result_basic.trend_signals, label='基本設定', alpha=0.8, color='red')
    ax3.plot(dates, result_dynamic.trend_signals, label='動的期間設定', alpha=0.8, color='green')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.set_title('トレンド信号 (1=上昇, -1=下降, 0=レンジ)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('信号', fontsize=12)
    ax3.set_ylim(-1.5, 1.5)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. 動的期間（動的期間設定のみ）
    ax4 = axes[3]
    if len(result_dynamic.dynamic_periods) > 0:
        ax4.plot(dates, result_dynamic.dynamic_periods, label='動的期間', linewidth=2, color='purple')
        ax4.set_title('動的期間（ドミナントサイクル）', fontsize=14, fontweight='bold')
        ax4.set_ylabel('期間', fontsize=12)
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, '動的期間データなし', ha='center', va='center', transform=ax4.transAxes, fontsize=14)
        ax4.set_title('動的期間（ドミナントサイクル）', fontsize=14, fontweight='bold')
    
    ax4.set_xlabel('日付', fontsize=12)
    
    # 日付軸の書式設定
    for ax in axes:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # 保存
    output_file = 'examples/output/efficiency_ratio_v2_example.png'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   チャートを保存しました: {output_file}")
    
    # 使用方法の説明
    print("\n7. 使用方法の説明...")
    print("   ER_V2インジケーターの主な特徴:")
    print("   - UKF_HLC3: カルマンフィルターで適応処理された価格を使用")
    print("   - UltimateSmoother: John Ehlersの高品質な平滑化フィルター")
    print("   - 動的期間対応: ドミナントサイクルに基づく適応的な期間調整")
    print("   - 高精度トレンド判定: 統計的閾値を用いたrange状態検出")
    print("")
    print("   効率比の解釈:")
    print("   - 0.618以上: 効率的な価格変動（強いトレンド）")
    print("   - 0.382以下: 非効率な価格変動（レンジ・ノイズ）")
    print("   - 0.382-0.618: 中程度の効率性")
    print("")
    print("   トレンド信号:")
    print("   - 1: 上昇トレンド")
    print("   - -1: 下降トレンド")
    print("   - 0: レンジ相場")
    
    print("\n=" * 60)
    print("ER_V2インジケーターの使用例が完了しました！")
    print("=" * 60)

if __name__ == "__main__":
    main() 