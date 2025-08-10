#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
import os

# プロジェクトのルートディレクトリを追加
sys.path.append('/home/vapor/dev/ttf')

from strategies.implementations.hyper_frama.strategy import HyperFRAMAEnhancedStrategy
from strategies.implementations.hyper_frama.signal_generator import FilterType

def test_current_default_strategy():
    """現在のデフォルト設定でのテスト"""
    print("=== 現在のデフォルト設定テスト ===")
    
    # 実際のデータに近いテストデータを生成
    np.random.seed(42)
    data_points = 200
    base_price = 50000
    prices = []
    
    for i in range(data_points):
        if i < 60:
            trend = -0.001  # 下降
        elif i < 120:
            trend = 0.002   # 上昇（ロングチャンス）
        else:
            trend = -0.0015 # 下降（ショートチャンス）
        
        noise = np.random.normal(0, 0.008)
        base_price *= (1 + trend + noise)
        prices.append(base_price)
    
    prices = np.array(prices)
    data = pd.DataFrame({
        'open': np.roll(prices, 1),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, data_points))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, data_points))),
        'close': prices,
        'volume': np.random.uniform(1000, 10000, data_points)
    })
    data.iloc[0, 0] = data.iloc[0, 3]  # open[0] = close[0]
    
    # デフォルト設定のストラテジー
    strategy = HyperFRAMAEnhancedStrategy()
    
    print(f"✓ デフォルトストラテジー設定:")
    print(f"  - Name: {strategy.name}")
    print(f"  - Position Mode: {strategy._parameters['position_mode']}")
    print(f"  - Filter Type: {strategy._parameters['filter_type']}")
    print(f"  - Period: {strategy._parameters['period']}")
    print(f"  - Alpha Multiplier: {strategy._parameters['alpha_multiplier']}")
    
    # エントリーシグナル生成
    entry_signals = strategy.generate_entry(data)
    long_signals = strategy.get_long_signals(data)
    short_signals = strategy.get_short_signals(data)
    
    long_count = np.sum(entry_signals == 1)
    short_count = np.sum(entry_signals == -1)
    
    print(f"\n✓ デフォルト設定結果:")
    print(f"  - ロングエントリー: {long_count}")
    print(f"  - ショートエントリー: {short_count}")
    
    if long_count == 0:
        print(f"  ⚠️ ロングシグナルが生成されていません")
        
        # HyperFRAMAの生の信号をチェック
        hyper_frama_signals = strategy.get_hyper_frama_signals(data)
        print(f"  - HyperFRAMA生シグナル: ロング={np.sum(hyper_frama_signals == 1)}, ショート={np.sum(hyper_frama_signals == -1)}")
        
        if np.sum(hyper_frama_signals == 1) == 0:
            print(f"  ❌ HyperFRAMAインジケーター自体がロングシグナルを生成していません")
            
            # FRAMA値を詳細に確認
            frama_values = strategy.get_frama_values(data)
            adjusted_frama_values = strategy.get_adjusted_frama_values(data)
            
            print(f"  FRAMA統計: min={np.nanmin(frama_values):.2f}, max={np.nanmax(frama_values):.2f}")
            print(f"  AdjFRAMA統計: min={np.nanmin(adjusted_frama_values):.2f}, max={np.nanmax(adjusted_frama_values):.2f}")
            
            # 位置関係分析
            valid_mask = ~(np.isnan(frama_values) | np.isnan(adjusted_frama_values))
            if np.any(valid_mask):
                frama_above_adj = frama_values[valid_mask] > adjusted_frama_values[valid_mask]
                print(f"  位置関係: FRAMA > Adj = {np.sum(frama_above_adj)}/{np.sum(valid_mask)} ({np.sum(frama_above_adj)/np.sum(valid_mask)*100:.1f}%)")
        else:
            print(f"  ✓ HyperFRAMAシグナルは生成されています - フィルター問題の可能性")
    else:
        print(f"  ✓ ロングシグナルが正常に生成されています")
    
    return long_count > 0

def test_optimized_settings():
    """最適化された設定でのテスト"""
    print(f"\n=== 最適化設定テスト ===")
    
    # 同じデータを使用
    np.random.seed(42)
    data_points = 150
    base_price = 50000
    prices = []
    
    for i in range(data_points):
        if i < 50:
            trend = -0.001
        elif i < 100:
            trend = 0.0025  # より強い上昇トレンド
        else:
            trend = -0.002
        
        noise = np.random.normal(0, 0.006)
        base_price *= (1 + trend + noise)
        prices.append(base_price)
    
    prices = np.array(prices)
    data = pd.DataFrame({
        'open': np.roll(prices, 1),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.004, data_points))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.004, data_points))),
        'close': prices,
        'volume': np.random.uniform(1000, 10000, data_points)
    })
    data.iloc[0, 0] = data.iloc[0, 3]
    
    # 最適化された設定
    strategy = HyperFRAMAEnhancedStrategy(
        period=8,           # より短い期間
        src_type='hl2',     # 標準的なソース
        fc=1,               # より敏感な設定
        sc=100,             # 短いSC
        alpha_multiplier=0.3,  # 小さなアルファ調整で違いを作る
        position_mode=False,   # クロスオーバーモード
        filter_type=FilterType.NONE  # フィルターなし
    )
    
    print(f"✓ 最適化ストラテジー設定:")
    print(f"  - Name: {strategy.name}")
    print(f"  - Period: {strategy._parameters['period']}")
    print(f"  - FC: {strategy._parameters['fc']}, SC: {strategy._parameters['sc']}")
    print(f"  - Alpha Multiplier: {strategy._parameters['alpha_multiplier']}")
    
    # シグナル生成
    entry_signals = strategy.generate_entry(data)
    long_count = np.sum(entry_signals == 1)
    short_count = np.sum(entry_signals == -1)
    
    print(f"\n✓ 最適化設定結果:")
    print(f"  - ロングエントリー: {long_count}")
    print(f"  - ショートエントリー: {short_count}")
    
    # 詳細な分析
    hyper_frama_signals = strategy.get_hyper_frama_signals(data)
    frama_values = strategy.get_frama_values(data)
    adjusted_frama_values = strategy.get_adjusted_frama_values(data)
    
    print(f"  - HyperFRAMAシグナル: ロング={np.sum(hyper_frama_signals == 1)}, ショート={np.sum(hyper_frama_signals == -1)}")
    
    # 実際のクロスオーバー発生箇所
    long_crossovers = np.where(hyper_frama_signals == 1)[0]
    short_crossovers = np.where(hyper_frama_signals == -1)[0]
    
    if len(long_crossovers) > 0:
        print(f"  ロングクロスオーバー発生インデックス: {long_crossovers}")
        # 最初のロングクロスオーバーの詳細
        idx = long_crossovers[0]
        if idx > 0:
            print(f"    [{idx}] FRAMA={frama_values[idx]:.2f} > Adj={adjusted_frama_values[idx]:.2f}")
            print(f"    前: FRAMA={frama_values[idx-1]:.2f} vs Adj={adjusted_frama_values[idx-1]:.2f}")
    
    if len(short_crossovers) > 0:
        print(f"  ショートクロスオーバー発生インデックス: {short_crossovers}")
    
    return long_count > 0

def main():
    """メイン関数"""
    print("HyperFRAMA クロスオーバー問題診断・修正テスト")
    print("=" * 60)
    
    results = []
    results.append(test_current_default_strategy())
    results.append(test_optimized_settings())
    
    print(f"\n{'='*60}")
    print(f"診断結果:")
    print(f"  デフォルト設定: {'✓ 正常' if results[0] else '✗ ロング未発生'}")
    print(f"  最適化設定: {'✓ 正常' if results[1] else '✗ ロング未発生'}")
    
    if not results[0]:
        print(f"\n推奨修正:")
        print(f"  1. デフォルトのposition_modeをFalse（クロスオーバー）に変更")
        print(f"  2. より敏感なパラメータ（period=8, sc=100, alpha_multiplier=0.3）を使用")
        print(f"  3. フィルターを無効化（FilterType.NONE）して純粋なクロスオーバーを使用")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    main()