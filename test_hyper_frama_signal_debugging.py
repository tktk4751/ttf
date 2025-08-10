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

def generate_test_data(n_periods=100):
    """デバッグ用テストデータを生成"""
    np.random.seed(42)
    
    # より明確なトレンド変化
    prices = []
    base_price = 50000
    
    for i in range(n_periods):
        if i < 25:
            # 明確な下降トレンド
            trend = -0.002
        elif i < 50:
            # 明確な上昇トレンド（ロングシグナルを期待）
            trend = 0.003
        elif i < 75:
            # 横ばい
            trend = 0.0001
        else:
            # 再び下降トレンド（ショートシグナルを期待）
            trend = -0.0025
        
        noise = np.random.normal(0, 0.005)
        base_price = base_price * (1 + trend + noise)
        prices.append(base_price)
    
    prices = np.array(prices)
    
    # OHLCV データの生成
    high = prices * (1 + np.abs(np.random.normal(0, 0.003, n_periods)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.003, n_periods)))
    open_prices = np.roll(prices, 1)
    open_prices[0] = prices[0]
    volume = np.random.uniform(1000, 10000, n_periods)
    
    data = pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': prices,
        'volume': volume
    })
    
    return data

def debug_signal_flow():
    """シグナルフローのデバッグ"""
    print("=== HyperFRAMA シグナルフローデバッグ ===")
    
    # テストデータ生成
    data = generate_test_data(100)
    
    # クロスオーバーストラテジー（フィルターなし）
    strategy = HyperFRAMAEnhancedStrategy(
        period=6,  # 短期間でクロスオーバーを発生させやすく
        src_type='hl2',
        fc=1,
        sc=80,     # SCを短くして反応を良くする
        alpha_multiplier=0.2,  # 小さなアルファで差を作る
        position_mode=False,   # クロスオーバーモード
        filter_type=FilterType.NONE  # フィルターなし
    )
    
    print(f"✓ ストラテジー初期化: {strategy.name}")
    print(f"  - Position Mode: {strategy._parameters['position_mode']}")
    print(f"  - Filter Type: {strategy._parameters['filter_type']}")
    
    # エントリーシグナル生成
    entry_signals = strategy.generate_entry(data)
    
    # 個別シグナル取得
    long_signals = strategy.get_long_signals(data)
    short_signals = strategy.get_short_signals(data)
    hyper_frama_signals = strategy.get_hyper_frama_signals(data)
    filter_signals = strategy.get_filter_signals(data)
    
    # FRAMA値取得
    frama_values = strategy.get_frama_values(data)
    adjusted_frama_values = strategy.get_adjusted_frama_values(data)
    
    print(f"\n✓ シグナル統計:")
    print(f"  エントリーシグナル: ロング={np.sum(entry_signals == 1)}, ショート={np.sum(entry_signals == -1)}")
    print(f"  個別ロングシグナル: {np.sum(long_signals == 1)}")
    print(f"  個別ショートシグナル: {np.sum(short_signals == 1)}")
    print(f"  HyperFRAMAシグナル: ロング={np.sum(hyper_frama_signals == 1)}, ショート={np.sum(hyper_frama_signals == -1)}")
    print(f"  フィルターシグナル: {np.unique(filter_signals, return_counts=True)}")
    
    # クロスオーバー発生箇所の詳細分析
    crossover_indices = np.where(hyper_frama_signals != 0)[0]
    print(f"\n✓ HyperFRAMAクロスオーバー詳細:")
    print(f"  発生回数: {len(crossover_indices)}")
    
    for i, idx in enumerate(crossover_indices[:10]):  # 最初の10個まで
        if idx > 0:  # 前のデータが必要
            signal_type = "ロング" if hyper_frama_signals[idx] == 1 else "ショート"
            prev_relation = "FRAMA > Adj" if frama_values[idx-1] > adjusted_frama_values[idx-1] else "FRAMA <= Adj"
            curr_relation = "FRAMA > Adj" if frama_values[idx] > adjusted_frama_values[idx] else "FRAMA <= Adj"
            
            long_generated = long_signals[idx] == 1
            short_generated = short_signals[idx] == 1
            entry_generated = entry_signals[idx] != 0
            
            print(f"  [{idx}] {signal_type}クロス: {prev_relation} -> {curr_relation}")
            print(f"      FRAMA: {frama_values[idx]:.2f}, Adj: {adjusted_frama_values[idx]:.2f}")
            print(f"      ロング生成: {'✓' if long_generated else '✗'}, ショート生成: {'✓' if short_generated else '✗'}, エントリー生成: {'✓' if entry_generated else '✗'}")
            print(f"      フィルター値: {filter_signals[idx]}")
    
    return np.sum(entry_signals == 1) > 0  # ロングシグナルが生成されたかどうか

def debug_filter_signals():
    """フィルターシグナルのデバッグ"""
    print(f"\n=== フィルターシグナル詳細デバッグ ===")
    
    data = generate_test_data(80)
    
    # HyperERフィルター付きストラテジー
    strategy_with_filter = HyperFRAMAEnhancedStrategy(
        period=8,
        src_type='hl2',
        position_mode=False,   # クロスオーバーモード
        filter_type=FilterType.HYPER_ER  # HyperERフィルター
    )
    
    print(f"✓ フィルター付きストラテジー: {strategy_with_filter.name}")
    
    # シグナル生成
    entry_signals = strategy_with_filter.generate_entry(data)
    filter_signals = strategy_with_filter.get_filter_signals(data)
    hyper_frama_signals = strategy_with_filter.get_hyper_frama_signals(data)
    
    print(f"  HyperFRAMAシグナル: ロング={np.sum(hyper_frama_signals == 1)}, ショート={np.sum(hyper_frama_signals == -1)}")
    print(f"  フィルターシグナル値の分布: {np.unique(filter_signals, return_counts=True)}")
    print(f"  エントリーシグナル: ロング={np.sum(entry_signals == 1)}, ショート={np.sum(entry_signals == -1)}")
    
    # フィルター条件での組み合わせ分析
    frama_long_indices = np.where(hyper_frama_signals == 1)[0]
    frama_short_indices = np.where(hyper_frama_signals == -1)[0]
    
    if len(frama_long_indices) > 0:
        print(f"\n  ロングHyperFRAMAシグナル時のフィルター状態:")
        for idx in frama_long_indices[:5]:
            filter_val = filter_signals[idx]
            entry_val = entry_signals[idx]
            print(f"    [{idx}] HyperFRAMA=1, フィルター={filter_val}, エントリー={entry_val}")
    
    if len(frama_short_indices) > 0:
        print(f"\n  ショートHyperFRAMAシグナル時のフィルター状態:")
        for idx in frama_short_indices[:5]:
            filter_val = filter_signals[idx]
            entry_val = entry_signals[idx]
            print(f"    [{idx}] HyperFRAMA=-1, フィルター={filter_val}, エントリー={entry_val}")
    
    return True

def main():
    """メインデバッグ関数"""
    print("HyperFRAMA ロングトレード問題デバッグ開始")
    print("=" * 60)
    
    # デバッグテストの実行
    has_long_signals = debug_signal_flow()
    debug_filter_signals()
    
    # 結果のまとめ
    print(f"\n{'='*60}")
    print(f"デバッグ結果まとめ:")
    if has_long_signals:
        print(f"✓ ロングシグナルが正常に生成されています")
    else:
        print(f"✗ ロングシグナルが生成されていません - 設定を確認してください")
        print(f"\n推奨対策:")
        print(f"  1. period、fc、sc、alpha_multiplierパラメータの調整")
        print(f"  2. より長期間またはトレンドの強いテストデータの使用")
        print(f"  3. フィルター条件の緩和")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    main()