#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from signals.implementations.ultimate_channel.breakout_entry import UltimateChannelBreakoutEntrySignal

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

def test_ultimate_channel_breakout_signal():
    """アルティメットチャネルブレイクアウトシグナルのテスト"""
    
    print("🔄 アルティメットチャネルブレイクアウトシグナルテスト開始")
    print("=" * 70)
    
    # テストデータの生成
    data = generate_test_data(1000)
    print(f"📊 テストデータ生成完了: {len(data)} points")
    
    # 固定乗数版シグナルの計算
    print("\n🔧 固定乗数版シグナル計算中...")
    signal_fixed = UltimateChannelBreakoutEntrySignal(
        channel_lookback=1,
        ultimate_channel_params={
            'length': 20.0,
            'num_strs': 2.0,
            'multiplier_mode': 'fixed',
            'src_type': 'hlc3'
        }
    )
    
    signals_fixed = signal_fixed.generate(data)
    print(f"✅ 固定乗数版シグナル計算完了")
    print(f"   シグナル名: {signal_fixed.name}")
    
    # 動的乗数版シグナルの計算
    print("\n🔧 動的乗数版シグナル計算中...")
    signal_dynamic = UltimateChannelBreakoutEntrySignal(
        channel_lookback=1,
        ultimate_channel_params={
            'length': 20.0,
            'num_strs': 2.0,
            'multiplier_mode': 'dynamic',
            'src_type': 'hlc3',
            'uqatrd_str_period': 20.0
        }
    )
    
    signals_dynamic = signal_dynamic.generate(data)
    print(f"✅ 動的乗数版シグナル計算完了")
    print(f"   シグナル名: {signal_dynamic.name}")
    
    # シグナル統計の分析
    print("\n📊 シグナル統計分析:")
    
    # 固定乗数版の統計
    long_signals_fixed = np.sum(signals_fixed == 1)
    short_signals_fixed = np.sum(signals_fixed == -1)
    total_signals_fixed = long_signals_fixed + short_signals_fixed
    
    print(f"   固定乗数版:")
    print(f"     - ロングシグナル: {long_signals_fixed} ({long_signals_fixed/len(signals_fixed)*100:.1f}%)")
    print(f"     - ショートシグナル: {short_signals_fixed} ({short_signals_fixed/len(signals_fixed)*100:.1f}%)")
    print(f"     - 総シグナル数: {total_signals_fixed} ({total_signals_fixed/len(signals_fixed)*100:.1f}%)")
    
    # 動的乗数版の統計
    long_signals_dynamic = np.sum(signals_dynamic == 1)
    short_signals_dynamic = np.sum(signals_dynamic == -1)
    total_signals_dynamic = long_signals_dynamic + short_signals_dynamic
    
    print(f"   動的乗数版:")
    print(f"     - ロングシグナル: {long_signals_dynamic} ({long_signals_dynamic/len(signals_dynamic)*100:.1f}%)")
    print(f"     - ショートシグナル: {short_signals_dynamic} ({short_signals_dynamic/len(signals_dynamic)*100:.1f}%)")
    print(f"     - 総シグナル数: {total_signals_dynamic} ({total_signals_dynamic/len(signals_dynamic)*100:.1f}%)")
    
    # チャネル値の取得
    centerline_fixed, upper_fixed, lower_fixed = signal_fixed.get_channel_values()
    centerline_dynamic, upper_dynamic, lower_dynamic = signal_dynamic.get_channel_values()
    
    print(f"\n📈 チャネル値統計:")
    print(f"   固定乗数版 - 中心線範囲: {np.min(centerline_fixed):.2f} - {np.max(centerline_fixed):.2f}")
    print(f"   動的乗数版 - 中心線範囲: {np.min(centerline_dynamic):.2f} - {np.max(centerline_dynamic):.2f}")
    
    # 動的乗数の情報取得
    dynamic_multipliers = signal_dynamic.get_dynamic_multipliers()
    uqatrd_values = signal_dynamic.get_uqatrd_values()
    multiplier_mode = signal_dynamic.get_multiplier_mode()
    
    if len(dynamic_multipliers) > 0:
        print(f"\n🎯 動的乗数情報:")
        print(f"   乗数モード: {multiplier_mode}")
        print(f"   平均乗数: {np.mean(dynamic_multipliers):.3f}")
        print(f"   乗数範囲: {np.min(dynamic_multipliers):.3f} - {np.max(dynamic_multipliers):.3f}")
        print(f"   平均UQATRD値: {np.mean(uqatrd_values):.3f}")
    
    # 可視化
    print("\n📈 結果可視化中...")
    
    # データの最後500点を使用して可視化
    plot_start = max(0, len(data) - 500)
    plot_data = data.iloc[plot_start:].copy()
    plot_data.reset_index(drop=True, inplace=True)
    
    signals_fixed_plot = signals_fixed[plot_start:]
    signals_dynamic_plot = signals_dynamic[plot_start:]
    centerline_fixed_plot = centerline_fixed[plot_start:]
    upper_fixed_plot = upper_fixed[plot_start:]
    lower_fixed_plot = lower_fixed[plot_start:]
    centerline_dynamic_plot = centerline_dynamic[plot_start:]
    upper_dynamic_plot = upper_dynamic[plot_start:]
    lower_dynamic_plot = lower_dynamic[plot_start:]
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # 1. 価格データと固定乗数チャネル
    axes[0].plot(plot_data['close'], label='Close Price', color='blue', linewidth=1)
    axes[0].plot(centerline_fixed_plot, label='Centerline (Fixed)', color='green', linewidth=1.5)
    axes[0].plot(upper_fixed_plot, label='Upper Channel (Fixed)', color='red', linewidth=1, alpha=0.7)
    axes[0].plot(lower_fixed_plot, label='Lower Channel (Fixed)', color='red', linewidth=1, alpha=0.7)
    axes[0].fill_between(range(len(upper_fixed_plot)), upper_fixed_plot, lower_fixed_plot, alpha=0.1, color='gray')
    axes[0].set_title('Price Data with Fixed Multiplier Ultimate Channel (Last 500 points)')
    axes[0].set_ylabel('Price')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. 価格データと動的乗数チャネル
    axes[1].plot(plot_data['close'], label='Close Price', color='blue', linewidth=1)
    axes[1].plot(centerline_dynamic_plot, label='Centerline (Dynamic)', color='purple', linewidth=1.5)
    axes[1].plot(upper_dynamic_plot, label='Upper Channel (Dynamic)', color='orange', linewidth=1, alpha=0.7)
    axes[1].plot(lower_dynamic_plot, label='Lower Channel (Dynamic)', color='orange', linewidth=1, alpha=0.7)
    axes[1].fill_between(range(len(upper_dynamic_plot)), upper_dynamic_plot, lower_dynamic_plot, alpha=0.1, color='orange')
    axes[1].set_title('Price Data with Dynamic Multiplier Ultimate Channel (Last 500 points)')
    axes[1].set_ylabel('Price')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. シグナル比較
    # シグナルポイントの可視化
    long_points_fixed = np.where(signals_fixed_plot == 1)[0]
    short_points_fixed = np.where(signals_fixed_plot == -1)[0]
    long_points_dynamic = np.where(signals_dynamic_plot == 1)[0]
    short_points_dynamic = np.where(signals_dynamic_plot == -1)[0]
    
    axes[2].plot(signals_fixed_plot, label='Fixed Multiplier Signals', color='blue', linewidth=1, alpha=0.7)
    axes[2].plot(signals_dynamic_plot, label='Dynamic Multiplier Signals', color='red', linewidth=1, alpha=0.7)
    
    # シグナルポイントを強調表示
    if len(long_points_fixed) > 0:
        axes[2].scatter(long_points_fixed, [1] * len(long_points_fixed), color='green', s=20, alpha=0.8, label='Long (Fixed)')
    if len(short_points_fixed) > 0:
        axes[2].scatter(short_points_fixed, [-1] * len(short_points_fixed), color='red', s=20, alpha=0.8, label='Short (Fixed)')
    if len(long_points_dynamic) > 0:
        axes[2].scatter(long_points_dynamic, [1.2] * len(long_points_dynamic), color='darkgreen', s=20, alpha=0.8, label='Long (Dynamic)')
    if len(short_points_dynamic) > 0:
        axes[2].scatter(short_points_dynamic, [-1.2] * len(short_points_dynamic), color='darkred', s=20, alpha=0.8, label='Short (Dynamic)')
    
    axes[2].set_title('Signal Comparison: Fixed vs Dynamic Multiplier')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Signal')
    axes[2].set_ylim(-1.5, 1.5)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 結果の保存
    output_path = 'output/ultimate_channel_breakout_signal_test.png'
    os.makedirs('output', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📁 結果グラフを保存: {output_path}")
    
    plt.show()
    
    # 統計レポートの保存
    report_path = 'output/ultimate_channel_breakout_signal_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("アルティメットチャネルブレイクアウトシグナルテスト レポート\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"テストデータ: {len(data)} データポイント\n")
        f.write(f"テスト日時: {pd.Timestamp.now()}\n\n")
        
        f.write("固定乗数版シグナル設定:\n")
        f.write(f"  シグナル名: {signal_fixed.name}\n")
        f.write(f"  チャネル参照期間: {signal_fixed.channel_lookback}\n\n")
        
        f.write("動的乗数版シグナル設定:\n")
        f.write(f"  シグナル名: {signal_dynamic.name}\n")
        f.write(f"  チャネル参照期間: {signal_dynamic.channel_lookback}\n")
        f.write(f"  乗数モード: {multiplier_mode}\n\n")
        
        f.write("シグナル統計:\n")
        f.write(f"  固定乗数版 - ロング: {long_signals_fixed}, ショート: {short_signals_fixed}, 総数: {total_signals_fixed}\n")
        f.write(f"  動的乗数版 - ロング: {long_signals_dynamic}, ショート: {short_signals_dynamic}, 総数: {total_signals_dynamic}\n\n")
        
        if len(dynamic_multipliers) > 0:
            f.write("動的乗数情報:\n")
            f.write(f"  平均乗数: {np.mean(dynamic_multipliers):.6f}\n")
            f.write(f"  乗数範囲: {np.min(dynamic_multipliers):.6f} - {np.max(dynamic_multipliers):.6f}\n")
            f.write(f"  平均UQATRD値: {np.mean(uqatrd_values):.6f}\n")
    
    print(f"📁 統計レポートを保存: {report_path}")
    
    print("\n✅ アルティメットチャネルブレイクアウトシグナルテスト完了")
    print("=" * 70)

if __name__ == "__main__":
    test_ultimate_channel_breakout_signal() 