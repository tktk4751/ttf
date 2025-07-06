#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ultimate Channel 動的乗数テストスクリプト
==========================================

UQATRDによる動的乗数適応機能のテスト
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta
import sys
import os

# パスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from indicators.ultimate_channel import UltimateChannel


def generate_test_data(n_points=1000):
    """テスト用の価格データを生成（トレンドとレンジが明確に分かれている）"""
    np.random.seed(42)
    
    # 基本価格レベル
    base_price = 100.0
    
    # 時系列データの生成
    dates = [datetime.now() - timedelta(days=i) for i in range(n_points)]
    dates.reverse()
    
    # 価格データの生成（明確なトレンドとレンジの期間）
    prices = []
    price = base_price
    
    for i in range(n_points):
        # 強いトレンド期間（0-200）: 0.7以上の値になることが期待される
        if 0 <= i < 200:
            trend = 0.08 * np.sin(i * 0.005) + 0.05  # 強い上昇トレンド
            noise = np.random.normal(0, 0.2)
            price += trend + noise
        # レンジ相場期間（200-500）: 0.4-0.5の値になることが期待される
        elif 200 <= i < 500:
            range_factor = 0.3 * np.sin(i * 0.2)  # 狭いレンジ
            noise = np.random.normal(0, 0.1)
            price += range_factor + noise
        # 中程度のトレンド期間（500-700）: 0.5-0.6の値になることが期待される
        elif 500 <= i < 700:
            trend = 0.03 * np.sin(i * 0.01) + 0.02  # 中程度のトレンド
            noise = np.random.normal(0, 0.25)
            price += trend + noise
        # 強いレンジ相場期間（700-850）: 0.4以下の値になることが期待される
        elif 700 <= i < 850:
            range_factor = 0.8 * np.sin(i * 0.3)  # 広いレンジ
            noise = np.random.normal(0, 0.15)
            price += range_factor + noise
        # 再び強いトレンド期間（850-1000）: 0.6-0.7の値になることが期待される
        else:
            trend = 0.06 * np.sin(i * 0.008) + 0.04  # 強い下降トレンド
            noise = np.random.normal(0, 0.2)
            price -= abs(trend) + noise
        
        prices.append(price)
    
    # OHLC データの生成
    data = []
    for i, close in enumerate(prices):
        if i == 0:
            open_price = close
            high = close + np.random.uniform(0, 1.0)
            low = close - np.random.uniform(0, 1.0)
        else:
            open_price = prices[i-1]
            high = max(open_price, close) + np.random.uniform(0, 1.0)
            low = min(open_price, close) - np.random.uniform(0, 1.0)
        
        data.append({
            'date': dates[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': close
        })
    
    return pd.DataFrame(data)


def test_dynamic_multiplier():
    """動的乗数機能のテスト"""
    print("🌟 Ultimate Channel 動的乗数テスト開始")
    print("=" * 60)
    
    # テストデータの生成
    print("🔄 テストデータを生成中...")
    data = generate_test_data(1000)
    print(f"✅ データ生成完了: {len(data)} rows")
    
    # 固定乗数モードのテスト
    print("\n🔬 固定乗数モードでのテスト")
    print("-" * 40)
    
    try:
        fixed_channel = UltimateChannel(
            length=20.0,
            str_length=20.0,
            num_strs=2.0,
            multiplier_mode='fixed',
            src_type='hlc3'
        )
        
        print("✅ 固定乗数チャネル初期化完了")
        
        # チャネル計算
        fixed_result = fixed_channel.calculate(data)
        print(f"✅ 固定乗数チャネル計算完了")
        
        # 乗数情報の取得
        fixed_multiplier_info = fixed_channel.get_multiplier_info()
        print(f"📊 固定乗数情報:")
        print(f"   - モード: {fixed_multiplier_info['multiplier_mode']}")
        print(f"   - 固定乗数: {fixed_multiplier_info.get('fixed_multiplier', 'N/A')}")
        print(f"   - 平均乗数: {fixed_multiplier_info['mean_multiplier']:.2f}")
        
    except Exception as e:
        print(f"❌ 固定乗数モードテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 動的乗数モードのテスト
    print("\n🔬 動的乗数モードでのテスト")
    print("-" * 40)
    
    try:
        dynamic_channel = UltimateChannel(
            length=20.0,
            str_length=20.0,
            num_strs=2.0,  # 基準値（動的モードでは使用されない）
            multiplier_mode='dynamic',
            src_type='hlc3',
            uqatrd_coherence_window=21,
            uqatrd_entanglement_window=34,
            uqatrd_efficiency_window=21,
            uqatrd_uncertainty_window=14
        )
        
        print("✅ 動的乗数チャネル初期化完了")
        
        # チャネル計算
        dynamic_result = dynamic_channel.calculate(data)
        print(f"✅ 動的乗数チャネル計算完了")
        
        # 乗数情報の取得
        dynamic_multiplier_info = dynamic_channel.get_multiplier_info()
        print(f"📊 動的乗数情報:")
        print(f"   - モード: {dynamic_multiplier_info['multiplier_mode']}")
        print(f"   - 平均乗数: {dynamic_multiplier_info['mean_multiplier']:.2f}")
        print(f"   - 乗数範囲: {dynamic_multiplier_info['min_multiplier']:.1f} - {dynamic_multiplier_info['max_multiplier']:.1f}")
        print(f"   - 現在の乗数: {dynamic_multiplier_info['current_multiplier']:.1f}")
        
        # 詳細データの取得
        dynamic_multipliers = dynamic_channel.get_dynamic_multipliers()
        uqatrd_values = dynamic_channel.get_uqatrd_values()
        
        # 乗数分布の分析（線形補間版）
        print(f"\n🎯 動的乗数分布（線形補間版）:")
        
        # 乗数の範囲別統計
        ranges = [
            (0.5, 1.5, "0.5-1.5 (強いトレンド)"),
            (1.5, 2.5, "1.5-2.5 (中程度トレンド)"),
            (2.5, 3.5, "2.5-3.5 (弱いトレンド)"),
            (3.5, 4.5, "3.5-4.5 (弱いレンジ)"),
            (4.5, 5.5, "4.5-5.5 (中程度レンジ)"),
            (5.5, 6.0, "5.5-6.0 (強いレンジ)")
        ]
        
        for min_val, max_val, desc in ranges:
            count = np.sum((dynamic_multipliers >= min_val) & (dynamic_multipliers < max_val))
            percentage = count / len(dynamic_multipliers) * 100
            print(f"   - {desc}: {count}点 ({percentage:.1f}%)")
        
        # 連続値の統計
        print(f"\n📊 動的乗数の統計:")
        print(f"   - 平均値: {np.mean(dynamic_multipliers):.3f}")
        print(f"   - 標準偏差: {np.std(dynamic_multipliers):.3f}")
        print(f"   - 最小値: {np.min(dynamic_multipliers):.3f}")
        print(f"   - 最大値: {np.max(dynamic_multipliers):.3f}")
        
        # 線形補間式の検証
        print(f"\n🔬 線形補間式の検証:")
        print(f"   - 理論式: 乗数 = 6.0 - UQATRD値 * (6.0 - 0.5)")
        print(f"   - UQATRD=0.0 → 理論乗数=6.0")
        print(f"   - UQATRD=1.0 → 理論乗数=0.5")
        print(f"   - UQATRD=0.5 → 理論乗数=3.25")
        
        # 実際の検証（サンプル）
        sample_idx = len(dynamic_multipliers) // 2
        sample_uqatrd = uqatrd_values[sample_idx]
        sample_multiplier = dynamic_multipliers[sample_idx]
        theoretical_multiplier = 6.0 - sample_uqatrd * (6.0 - 0.5)
        print(f"   - 実際の検証（中央値）: UQATRD={sample_uqatrd:.3f}, 実際={sample_multiplier:.3f}, 理論={theoretical_multiplier:.3f}")
        
        # UQATRD値の分析
        print(f"\n🔍 UQATRD値の分析:")
        print(f"   - 平均値: {np.mean(uqatrd_values):.3f}")
        print(f"   - 標準偏差: {np.std(uqatrd_values):.3f}")
        print(f"   - 最小値: {np.min(uqatrd_values):.3f}")
        print(f"   - 最大値: {np.max(uqatrd_values):.3f}")
        
        # 市場状態の分析
        trend_threshold = 0.6
        range_threshold = 0.4
        strong_trend_count = np.sum(uqatrd_values >= trend_threshold)
        strong_range_count = np.sum(uqatrd_values <= range_threshold)
        neutral_count = np.sum((uqatrd_values > range_threshold) & (uqatrd_values < trend_threshold))
        
        print(f"\n📈 市場状態の分析:")
        print(f"   - 強いトレンド (UQATRD≥{trend_threshold}): {strong_trend_count}点 ({strong_trend_count/len(uqatrd_values)*100:.1f}%)")
        print(f"   - 強いレンジ (UQATRD≤{range_threshold}): {strong_range_count}点 ({strong_range_count/len(uqatrd_values)*100:.1f}%)")
        print(f"   - 中間状態 ({range_threshold}<UQATRD<{trend_threshold}): {neutral_count}点 ({neutral_count/len(uqatrd_values)*100:.1f}%)")
        
    except Exception as e:
        print(f"❌ 動的乗数モードテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 可視化
    print("\n📈 結果の可視化")
    print("-" * 40)
    
    try:
        fig, axes = plt.subplots(6, 1, figsize=(15, 18))
        
        # 価格チャート
        axes[0].plot(data['close'], label='Close Price', color='blue', linewidth=1)
        axes[0].set_title('Price Chart with Market Phases')
        axes[0].set_ylabel('Price')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 市場フェーズの背景色を追加
        axes[0].axvspan(0, 200, alpha=0.2, color='green', label='Strong Trend')
        axes[0].axvspan(200, 500, alpha=0.2, color='red', label='Range')
        axes[0].axvspan(500, 700, alpha=0.2, color='yellow', label='Medium Trend')
        axes[0].axvspan(700, 850, alpha=0.2, color='red', label='Strong Range')
        axes[0].axvspan(850, 1000, alpha=0.2, color='orange', label='Strong Trend 2')
        
        # UQATRDの値
        if len(uqatrd_values) > 0:
            axes[1].plot(uqatrd_values, label='UQATRD Signal', color='purple', linewidth=1)
            axes[1].axhline(y=0.4, color='red', linestyle='--', alpha=0.7, label='Range Threshold (0.4)')
            axes[1].axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Neutral (0.5)')
            axes[1].axhline(y=0.6, color='yellow', linestyle='--', alpha=0.7, label='Trend Threshold (0.6)')
            axes[1].set_title('UQATRD Signal - Linear Interpolation (0=Range, 1=Trend)')
            axes[1].set_ylabel('UQATRD Value')
            axes[1].set_ylim(0, 1)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # 動的乗数
        if len(dynamic_multipliers) > 0:
            axes[2].plot(dynamic_multipliers, label='Dynamic Multiplier', color='red', linewidth=1)
            axes[2].axhline(y=2.0, color='gray', linestyle=':', alpha=0.5, label='Fixed Multiplier (2.0)')
            axes[2].axhline(y=0.5, color='blue', linestyle='--', alpha=0.5, label='Min (0.5)')
            axes[2].axhline(y=6.0, color='purple', linestyle='--', alpha=0.5, label='Max (6.0)')
            axes[2].axhline(y=3.25, color='orange', linestyle='--', alpha=0.5, label='Neutral (3.25)')
            axes[2].set_title('Dynamic Multiplier - Linear Interpolation (Based on UQATRD)')
            axes[2].set_ylabel('Multiplier')
            axes[2].set_ylim(0, 6.5)
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        # 固定チャネル
        axes[3].plot(data['close'], label='Close Price', color='blue', linewidth=1)
        axes[3].plot(fixed_result.upper_channel, label='Upper Channel (Fixed)', color='red', linewidth=0.8, alpha=0.7)
        axes[3].plot(fixed_result.lower_channel, label='Lower Channel (Fixed)', color='red', linewidth=0.8, alpha=0.7)
        axes[3].plot(fixed_result.center_line, label='Center Line', color='black', linewidth=0.8, linestyle='--')
        axes[3].fill_between(range(len(fixed_result.upper_channel)), 
                            fixed_result.lower_channel, fixed_result.upper_channel,
                            alpha=0.1, color='red')
        axes[3].set_title('Fixed Multiplier Channel (Multiplier = 2.0)')
        axes[3].set_ylabel('Price')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        # 動的チャネル
        axes[4].plot(data['close'], label='Close Price', color='blue', linewidth=1)
        axes[4].plot(dynamic_result.upper_channel, label='Upper Channel (Dynamic)', color='green', linewidth=0.8, alpha=0.7)
        axes[4].plot(dynamic_result.lower_channel, label='Lower Channel (Dynamic)', color='green', linewidth=0.8, alpha=0.7)
        axes[4].plot(dynamic_result.center_line, label='Center Line', color='black', linewidth=0.8, linestyle='--')
        axes[4].fill_between(range(len(dynamic_result.upper_channel)), 
                            dynamic_result.lower_channel, dynamic_result.upper_channel,
                            alpha=0.1, color='green')
        axes[4].set_title('Dynamic Multiplier Channel - Linear Interpolation (UQATRD-Based)')
        axes[4].set_ylabel('Price')
        axes[4].legend()
        axes[4].grid(True, alpha=0.3)
        
        # チャネル幅の比較
        fixed_width = fixed_result.upper_channel - fixed_result.lower_channel
        dynamic_width = dynamic_result.upper_channel - dynamic_result.lower_channel
        
        axes[5].plot(fixed_width, label='Fixed Channel Width', color='red', linewidth=1, alpha=0.7)
        axes[5].plot(dynamic_width, label='Dynamic Channel Width', color='green', linewidth=1, alpha=0.7)
        axes[5].set_title('Channel Width Comparison')
        axes[5].set_ylabel('Channel Width')
        axes[5].set_xlabel('Time')
        axes[5].legend()
        axes[5].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 出力ディレクトリの確認
        output_dir = os.path.join(os.path.dirname(__file__), 'output')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_path = os.path.join(output_dir, 'ultimate_channel_dynamic_multiplier_test.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 可視化結果保存: {output_path}")
        
        # 統計情報のテキストファイル出力
        stats_path = os.path.join(output_dir, 'ultimate_channel_dynamic_multiplier_stats.txt')
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("Ultimate Channel 動的乗数テスト結果\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("固定乗数モード:\n")
            for key, value in fixed_multiplier_info.items():
                f.write(f"  {key}: {value}\n")
            
            f.write("\n動的乗数モード:\n")
            for key, value in dynamic_multiplier_info.items():
                f.write(f"  {key}: {value}\n")
            
            f.write("\n動的乗数分布（線形補間版）:\n")
            ranges = [
                (0.5, 1.5, "0.5-1.5 (強いトレンド)"),
                (1.5, 2.5, "1.5-2.5 (中程度トレンド)"),
                (2.5, 3.5, "2.5-3.5 (弱いトレンド)"),
                (3.5, 4.5, "3.5-4.5 (弱いレンジ)"),
                (4.5, 5.5, "4.5-5.5 (中程度レンジ)"),
                (5.5, 6.0, "5.5-6.0 (強いレンジ)")
            ]
            
            for min_val, max_val, desc in ranges:
                count = np.sum((dynamic_multipliers >= min_val) & (dynamic_multipliers < max_val))
                percentage = count / len(dynamic_multipliers) * 100
                f.write(f"  {desc}: {count}点 ({percentage:.1f}%)\n")
            
            f.write(f"\n動的乗数の統計:\n")
            f.write(f"  平均値: {np.mean(dynamic_multipliers):.3f}\n")
            f.write(f"  標準偏差: {np.std(dynamic_multipliers):.3f}\n")
            f.write(f"  最小値: {np.min(dynamic_multipliers):.3f}\n")
            f.write(f"  最大値: {np.max(dynamic_multipliers):.3f}\n")
            
            f.write(f"\n線形補間式の検証:\n")
            f.write(f"  理論式: 乗数 = 6.0 - UQATRD値 * (6.0 - 0.5)\n")
            f.write(f"  UQATRD=0.0 → 理論乗数=6.0\n")
            f.write(f"  UQATRD=1.0 → 理論乗数=0.5\n")
            f.write(f"  UQATRD=0.5 → 理論乗数=3.25\n")
            
            f.write(f"\nUQATRD統計:\n")
            f.write(f"  平均値: {np.mean(uqatrd_values):.3f}\n")
            f.write(f"  標準偏差: {np.std(uqatrd_values):.3f}\n")
            f.write(f"  最小値: {np.min(uqatrd_values):.3f}\n")
            f.write(f"  最大値: {np.max(uqatrd_values):.3f}\n")
        
        print(f"📝 統計情報保存: {stats_path}")
        
    except Exception as e:
        print(f"❌ 可視化エラー: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎯 テスト完了!")


if __name__ == "__main__":
    test_dynamic_multiplier() 