#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import pandas as pd
import math
from datetime import datetime, timedelta

# パスの追加
sys.path.append('.')

# 統合チャートのインポート
from visualization.unified_trend_cycle_chart import UnifiedTrendCycleChart


def create_sample_data(length: int = 300) -> pd.DataFrame:
    """
    テスト用の複雑な市場データを生成
    
    Args:
        length: データの長さ
        
    Returns:
        OHLCV データフレーム
    """
    np.random.seed(42)
    base_price = 100.0
    base_date = datetime(2024, 1, 1)
    
    # 複雑な市場シナリオ
    prices = [base_price]
    for i in range(1, length):
        if i < 60:  # 明確なサイクル相場（20期間周期）
            cycle_component = 8.0 * math.sin(2.0 * math.pi * i / 20.0)
            noise = np.random.normal(0, 1.5)
            change = (cycle_component + noise) / base_price
        elif i < 120:  # 強い上昇トレンド
            change = 0.005 + np.random.normal(0, 0.02)
        elif i < 180:  # 複雑なサイクル（異なる周期の混合）
            cycle1 = 5.0 * math.sin(2.0 * math.pi * i / 15.0)
            cycle2 = 3.0 * math.sin(2.0 * math.pi * i / 25.0)
            noise = np.random.normal(0, 1.2)
            change = (cycle1 + cycle2 + noise) / base_price
        elif i < 240:  # 下降トレンド
            change = -0.004 + np.random.normal(0, 0.015)
        else:  # 混合相場（トレンド+サイクル）
            trend_component = 0.002
            cycle_component = 4.0 * math.sin(2.0 * math.pi * i / 18.0)
            noise = np.random.normal(0, 1.0)
            change = (trend_component + cycle_component + noise) / base_price
        
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # OHLC データの生成
    data = []
    for i, close in enumerate(prices):
        date = base_date + timedelta(hours=i)  # 1時間足データとして生成
        
        daily_range = abs(np.random.normal(0, close * 0.015))
        high = close + daily_range * np.random.uniform(0.4, 1.0)
        low = close - daily_range * np.random.uniform(0.4, 1.0)
        
        if i == 0:
            open_price = close
        else:
            gap = np.random.normal(0, close * 0.008)
            open_price = prices[i-1] + gap
        
        # 論理的整合性の確保
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'datetime': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.uniform(1000, 15000)
        })
    
    df = pd.DataFrame(data)
    df.set_index('datetime', inplace=True)
    
    return df


def test_unified_chart():
    """統合チャートの動作テスト"""
    print("=== 統合トレンド・サイクル検出器チャートのテスト ===")
    
    # サンプルデータの生成
    print("サンプルデータを生成中...")
    sample_data = create_sample_data(300)
    print(f"サンプルデータ生成完了: {len(sample_data)}行")
    print(f"期間: {sample_data.index.min()} → {sample_data.index.max()}")
    print(f"価格範囲: {sample_data['close'].min():.2f} - {sample_data['close'].max():.2f}")
    
    # チャートクラスのインスタンス化
    chart = UnifiedTrendCycleChart()
    chart.data = sample_data  # サンプルデータを直接設定
    
    # 統合検出器の計算
    print("\n統合トレンド・サイクル検出器を計算中...")
    try:
        chart.calculate_indicators(
            period=20,
            trend_length=20,
            trend_threshold=0.3,
            adaptability_factor=0.7,
            src_type='close',
            enable_consensus_filter=True,
            min_consensus_threshold=0.6
        )
        print("統合検出器の計算が正常に完了しました")
    except Exception as e:
        print(f"統合検出器の計算中にエラー: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # チャートの描画テスト
    print("\nチャートを描画中...")
    try:
        chart.plot(
            title="統合トレンド・サイクル検出器 - テストデータ",
            show_volume=True,
            figsize=(16, 14),
            savefig="unified_trend_cycle_test_chart.png"
        )
        print("チャートの描画が正常に完了しました")
        print("チャートは 'unified_trend_cycle_test_chart.png' に保存されました")
    except Exception as e:
        print(f"チャート描画中にエラー: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 統計情報の追加出力
    result = chart.unified_detector.calculate(chart.data)
    
    print(f"\n=== 詳細統計情報 ===")
    print(f"統合トレンド強度 - 平均: {np.nanmean(result.unified_trend_strength):.4f}")
    print(f"統合サイクル信頼度 - 平均: {np.nanmean(result.unified_cycle_confidence):.4f}")
    print(f"コンセンサス強度 - 平均: {np.nanmean(result.consensus_strength):.4f}")
    print(f"フェーザー強度 - 平均: {np.nanmean(result.magnitude):.4f}")
    
    # 状態分布
    unique_states, state_counts = np.unique(result.unified_state, return_counts=True)
    print(f"\n=== 状態分布 ===")
    for state, count in zip(unique_states, state_counts):
        if state == 1:
            print(f"買い状態: {count}回 ({count/len(result.unified_state)*100:.1f}%)")
        elif state == -1:
            print(f"売り状態: {count}回 ({count/len(result.unified_state)*100:.1f}%)")
        elif state == 0:
            print(f"中立状態: {count}回 ({count/len(result.unified_state)*100:.1f}%)")
    
    # シグナル分布
    unique_signals, signal_counts = np.unique(result.unified_signal, return_counts=True)
    print(f"\n=== シグナル分布 ===")
    for signal, count in zip(unique_signals, signal_counts):
        if signal == 1:
            print(f"買いシグナル: {count}回")
        elif signal == -1:
            print(f"売りシグナル: {count}回")
        elif signal == 0:
            print(f"中立シグナル: {count}回")
    
    print("\n=== テスト完了 ===")


if __name__ == "__main__":
    test_unified_chart()