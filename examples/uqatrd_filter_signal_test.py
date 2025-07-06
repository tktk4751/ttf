#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UQATRD Filter Signal テストスクリプト
=====================================

Ultra Quantum Adaptive Trend-Range Discriminator (UQATRD) を使用した
フィルターシグナルの動作テスト
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

from signals.implementations.uqatrd.filter import UQATRDFilterSignal


def generate_test_data(n_points=1000):
    """テスト用の価格データを生成"""
    np.random.seed(42)
    
    # 基本価格レベル
    base_price = 100.0
    
    # 時系列データの生成
    dates = [datetime.now() - timedelta(days=i) for i in range(n_points)]
    dates.reverse()
    
    # 価格データの生成（トレンドとレンジが混在）
    prices = []
    price = base_price
    
    for i in range(n_points):
        # トレンド期間（0-300, 600-800）
        if 0 <= i < 300 or 600 <= i < 800:
            # 上昇トレンド
            trend = 0.05 * np.sin(i * 0.01) + 0.02
            noise = np.random.normal(0, 0.3)
            price += trend + noise
        elif 300 <= i < 600 or 800 <= i < 1000:
            # レンジ相場
            range_factor = 0.5 * np.sin(i * 0.1)
            noise = np.random.normal(0, 0.2)
            price += range_factor + noise
        
        prices.append(price)
    
    # OHLC データの生成
    data = []
    for i, close in enumerate(prices):
        if i == 0:
            open_price = close
            high = close + np.random.uniform(0, 0.5)
            low = close - np.random.uniform(0, 0.5)
        else:
            open_price = prices[i-1]
            high = max(open_price, close) + np.random.uniform(0, 0.5)
            low = min(open_price, close) - np.random.uniform(0, 0.5)
        
        data.append({
            'date': dates[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': close
        })
    
    return pd.DataFrame(data)


def test_uqatrd_filter_signal():
    """UQATRDFilterSignalのテスト"""
    print("🌟 UQATRD Filter Signal テスト開始")
    print("=" * 60)
    
    # テストデータの生成
    print("🔄 テストデータを生成中...")
    data = generate_test_data(1000)
    print(f"✅ データ生成完了: {len(data)} rows")
    
    # 動的しきい値モードのテスト
    print("\n🔬 動的しきい値モードでのテスト")
    print("-" * 40)
    
    try:
        # 動的しきい値モードのフィルターシグナル
        dynamic_filter = UQATRDFilterSignal(
            coherence_window=21,
            entanglement_window=34,
            efficiency_window=21,
            uncertainty_window=14,
            threshold_mode='dynamic',
            src_type='hlc3'
        )
        
        print("✅ 動的しきい値フィルター初期化完了")
        
        # シグナル生成
        dynamic_signals = dynamic_filter.generate(data)
        print(f"✅ 動的しきい値シグナル生成完了: {len(dynamic_signals)} points")
        
        # 結果の統計
        trend_count = np.sum(dynamic_signals == 1)
        range_count = np.sum(dynamic_signals == -1)
        nan_count = np.sum(np.isnan(dynamic_signals))
        
        print(f"📊 動的しきい値結果:")
        print(f"   - トレンド相場: {trend_count}点 ({trend_count/len(dynamic_signals)*100:.1f}%)")
        print(f"   - レンジ相場: {range_count}点 ({range_count/len(dynamic_signals)*100:.1f}%)")
        print(f"   - 無効値: {nan_count}点 ({nan_count/len(dynamic_signals)*100:.1f}%)")
        
        # しきい値情報の取得
        threshold_info = dynamic_filter.get_threshold_info()
        if threshold_info:
            print(f"🎯 動的しきい値情報:")
            print(f"   - 平均しきい値: {threshold_info['mean_threshold']:.3f}")
            print(f"   - しきい値範囲: {threshold_info['min_threshold']:.3f} - {threshold_info['max_threshold']:.3f}")
            print(f"   - 現在のしきい値: {threshold_info['current_threshold']:.3f}")
        
    except Exception as e:
        print(f"❌ 動的しきい値モードテストエラー: {e}")
        import traceback
        traceback.print_exc()
    
    # 固定しきい値モードのテスト
    print("\n🔬 固定しきい値モードでのテスト")
    print("-" * 40)
    
    try:
        # 固定しきい値モードのフィルターシグナル
        fixed_filter = UQATRDFilterSignal(
            coherence_window=21,
            entanglement_window=34,
            efficiency_window=21,
            uncertainty_window=14,
            threshold_mode='fixed',
            fixed_threshold=0.5,
            src_type='hlc3'
        )
        
        print("✅ 固定しきい値フィルター初期化完了")
        
        # シグナル生成
        fixed_signals = fixed_filter.generate(data)
        print(f"✅ 固定しきい値シグナル生成完了: {len(fixed_signals)} points")
        
        # 結果の統計
        trend_count = np.sum(fixed_signals == 1)
        range_count = np.sum(fixed_signals == -1)
        nan_count = np.sum(np.isnan(fixed_signals))
        
        print(f"📊 固定しきい値結果:")
        print(f"   - トレンド相場: {trend_count}点 ({trend_count/len(fixed_signals)*100:.1f}%)")
        print(f"   - レンジ相場: {range_count}点 ({range_count/len(fixed_signals)*100:.1f}%)")
        print(f"   - 無効値: {nan_count}点 ({nan_count/len(fixed_signals)*100:.1f}%)")
        
        # しきい値情報の取得
        threshold_info = fixed_filter.get_threshold_info()
        if threshold_info:
            print(f"🎯 固定しきい値情報:")
            print(f"   - 固定しきい値: {threshold_info['fixed_threshold']}")
            print(f"   - しきい値モード: {threshold_info['threshold_mode']}")
        
    except Exception as e:
        print(f"❌ 固定しきい値モードテストエラー: {e}")
        import traceback
        traceback.print_exc()
    
    # 可視化
    print("\n📈 結果の可視化")
    print("-" * 40)
    
    try:
        # 動的モードの詳細データ取得
        trend_range_values = dynamic_filter.get_trend_range_values()
        threshold_values = dynamic_filter.get_threshold_values()
        confidence_scores = dynamic_filter.get_confidence_score()
        
        # 可視化
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        # 価格チャート
        axes[0].plot(data['close'], label='Close Price', color='blue', linewidth=1)
        axes[0].set_title('Price Chart')
        axes[0].set_ylabel('Price')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # UQATRDトレンド/レンジ信号と動的しきい値
        if len(trend_range_values) > 0:
            axes[1].plot(trend_range_values, label='UQATRD Signal', color='green', linewidth=1)
            axes[1].plot(threshold_values, label='Dynamic Threshold', color='red', linewidth=1, linestyle='--')
            axes[1].axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Fixed Threshold (0.5)')
            axes[1].set_title('UQATRD Signal vs Dynamic Threshold')
            axes[1].set_ylabel('Signal Value')
            axes[1].set_ylim(0, 1)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # シグナル比較
        axes[2].plot(dynamic_signals, label='Dynamic Mode', color='blue', linewidth=1)
        axes[2].plot(fixed_signals, label='Fixed Mode', color='orange', linewidth=1, alpha=0.7)
        axes[2].set_title('Signal Comparison (1=Trend, -1=Range)')
        axes[2].set_ylabel('Signal')
        axes[2].set_ylim(-1.5, 1.5)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # 信頼度スコア
        if len(confidence_scores) > 0:
            axes[3].plot(confidence_scores, label='Confidence Score', color='purple', linewidth=1)
            axes[3].set_title('Confidence Score')
            axes[3].set_ylabel('Confidence')
            axes[3].set_xlabel('Time')
            axes[3].set_ylim(0, 1)
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 出力ディレクトリの確認
        output_dir = os.path.join(os.path.dirname(__file__), 'output')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_path = os.path.join(output_dir, 'uqatrd_filter_signal_test.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 可視化結果保存: {output_path}")
        
    except Exception as e:
        print(f"❌ 可視化エラー: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎯 テスト完了!")


if __name__ == "__main__":
    test_uqatrd_filter_signal() 