#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
修正されたHyperドンチャンエントリーシグナルのスーパースムーザーフィルタリング機能をテスト
"""

import numpy as np
import pandas as pd
import sys
import os

# パスの設定
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from signals.implementations.hyper_donchian.entry import HyperDonchianBreakoutEntrySignal

def create_test_data(length=200):
    """テスト用データを生成"""
    np.random.seed(42)
    base_price = 100.0
    
    # トレンドとブレイクアウトが混在するデータを生成
    prices = [base_price]
    for i in range(1, length):
        if i < 50:  # 上昇トレンド
            change = 0.004 + np.random.normal(0, 0.006)
        elif i < 100:  # レンジ相場
            change = np.random.normal(0, 0.008)
        elif i < 120:  # 強いブレイクアウト
            change = 0.008 + np.random.normal(0, 0.004)
        elif i < 150:  # 安定期間
            change = np.random.normal(0, 0.005)
        else:  # 下降トレンド
            change = -0.003 + np.random.normal(0, 0.007)
        
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # OHLC データの生成
    data = []
    for i, close in enumerate(prices):
        daily_range = abs(np.random.normal(0, close * 0.008))
        
        high = close + daily_range * np.random.uniform(0.4, 1.0)
        low = close - daily_range * np.random.uniform(0.4, 1.0)
        
        if i == 0:
            open_price = close
        else:
            gap = np.random.normal(0, close * 0.003)
            open_price = prices[i-1] + gap
        
        # 論理的整合性の確保
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.uniform(1000, 10000)
        })
    
    return pd.DataFrame(data)

def test_super_smoother_filter():
    """スーパースムーザーフィルタリング機能のテスト"""
    print("=== スーパースムーザーフィルタリング機能テスト ===")
    
    # テストデータの作成
    df = create_test_data(200)
    print(f"テストデータ: {len(df)}ポイント")
    print(f"価格範囲: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    print("\n1. スーパースムーザーフィルタリングなし（基本版）")
    signal_no_filter = HyperDonchianBreakoutEntrySignal(
        period=20,
        src_type='close',
        enable_super_smoother_filter=False  # フィルタリング無効
    )
    
    signals_no_filter = signal_no_filter.generate(df)
    long_no_filter = np.sum(signals_no_filter == 1)
    short_no_filter = np.sum(signals_no_filter == -1)
    no_signal_no_filter = np.sum(signals_no_filter == 0)
    
    print(f"  ロングエントリー: {long_no_filter}")
    print(f"  ショートエントリー: {short_no_filter}")
    print(f"  シグナルなし: {no_signal_no_filter}")
    
    print("\n2. スーパースムーザーフィルタリングあり（期間=10）")
    signal_with_filter_10 = HyperDonchianBreakoutEntrySignal(
        period=20,
        src_type='close',
        enable_super_smoother_filter=True,  # フィルタリング有効
        super_smoother_period=10,
        super_smoother_src_type='close'
    )
    
    signals_with_filter_10 = signal_with_filter_10.generate(df)
    long_with_filter_10 = np.sum(signals_with_filter_10 == 1)
    short_with_filter_10 = np.sum(signals_with_filter_10 == -1)
    no_signal_with_filter_10 = np.sum(signals_with_filter_10 == 0)
    
    print(f"  ロングエントリー: {long_with_filter_10}")
    print(f"  ショートエントリー: {short_with_filter_10}")
    print(f"  シグナルなし: {no_signal_with_filter_10}")
    
    print("\n3. スーパースムーザーフィルタリングあり（期間=20）")
    signal_with_filter_20 = HyperDonchianBreakoutEntrySignal(
        period=20,
        src_type='close',
        enable_super_smoother_filter=True,  # フィルタリング有効
        super_smoother_period=20,
        super_smoother_src_type='close'
    )
    
    signals_with_filter_20 = signal_with_filter_20.generate(df)
    long_with_filter_20 = np.sum(signals_with_filter_20 == 1)
    short_with_filter_20 = np.sum(signals_with_filter_20 == -1)
    no_signal_with_filter_20 = np.sum(signals_with_filter_20 == 0)
    
    print(f"  ロングエントリー: {long_with_filter_20}")
    print(f"  ショートエントリー: {short_with_filter_20}")
    print(f"  シグナルなし: {no_signal_with_filter_20}")
    
    print("\n4. スーパースムーザーフィルタリングあり（期間=50）")
    signal_with_filter_50 = HyperDonchianBreakoutEntrySignal(
        period=20,
        src_type='close',
        enable_super_smoother_filter=True,  # フィルタリング有効
        super_smoother_period=50,
        super_smoother_src_type='close'
    )
    
    signals_with_filter_50 = signal_with_filter_50.generate(df)
    long_with_filter_50 = np.sum(signals_with_filter_50 == 1)
    short_with_filter_50 = np.sum(signals_with_filter_50 == -1)
    no_signal_with_filter_50 = np.sum(signals_with_filter_50 == 0)
    
    print(f"  ロングエントリー: {long_with_filter_50}")
    print(f"  ショートエントリー: {short_with_filter_50}")
    print(f"  シグナルなし: {no_signal_with_filter_50}")
    
    # 結果の比較
    print("\n=== 比較結果 ===")
    print(f"基本版（フィルターなし）: Long={long_no_filter}, Short={short_no_filter}, Total={long_no_filter+short_no_filter}")
    print(f"フィルター期間10: Long={long_with_filter_10}, Short={short_with_filter_10}, Total={long_with_filter_10+short_with_filter_10}")
    print(f"フィルター期間20: Long={long_with_filter_20}, Short={short_with_filter_20}, Total={long_with_filter_20+short_with_filter_20}")
    print(f"フィルター期間50: Long={long_with_filter_50}, Short={short_with_filter_50}, Total={long_with_filter_50+short_with_filter_50}")
    
    # 違いがあることを確認
    all_results = [
        long_no_filter + short_no_filter,
        long_with_filter_10 + short_with_filter_10,
        long_with_filter_20 + short_with_filter_20,
        long_with_filter_50 + short_with_filter_50
    ]
    
    if len(set(all_results)) > 1:
        print("\n✓ フィルタリング機能が正常に動作しています（結果に違いがあります）")
    else:
        print("\n✗ フィルタリング機能に問題があります（すべて同じ結果です）")
    
    return all_results

if __name__ == "__main__":
    test_super_smoother_filter()