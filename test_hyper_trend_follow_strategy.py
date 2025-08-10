#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
import os

# プロジェクトのルートディレクトリを追加
sys.path.append('/home/vapor/dev/ttf')

from strategies.implementations.hyper_trend_follow.strategy import HyperTrendFollowStrategy
from strategies.implementations.hyper_trend_follow.signal_generator import HyperTrendFollowSignalGenerator

def generate_test_data(n_periods=300):
    """テスト用のダミーデータを生成"""
    np.random.seed(42)
    
    dates = pd.date_range(start='2023-01-01', periods=n_periods, freq='4H')
    
    # トレンドのあるダミー価格データを生成
    price = 50000.0
    prices = []
    
    for i in range(n_periods):
        # トレンドとランダムウォーク
        trend = 0.0002 * (i - n_periods/2)  # 中間でトレンド転換
        noise = np.random.normal(0, 0.02)
        price = price * (1 + trend + noise)
        prices.append(price)
    
    prices = np.array(prices)
    
    # OHLCV データの生成
    high = prices * (1 + np.abs(np.random.normal(0, 0.015, n_periods)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.015, n_periods)))
    open_prices = np.roll(prices, 1)
    open_prices[0] = prices[0]
    volume = np.random.uniform(1000, 10000, n_periods)
    
    data = pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': prices,
        'volume': volume
    }, index=dates)
    
    return data

def test_signal_generator():
    """シグナルジェネレーターのテスト"""
    print("\n=== HyperTrendFollowSignalGenerator テスト ===")
    
    try:
        # テスト用データの生成
        data = generate_test_data(300)
        
        # シグナルジェネレーターの初期化
        signal_gen = HyperTrendFollowSignalGenerator(
            hyper_frama_period=16,
            hyper_frama_src_type='hl2',
            channel_period=20,
            channel_fixed_multiplier=2.0
        )
        
        print(f"✓ シグナルジェネレーターを初期化しました")
        
        # エントリーシグナルのテスト
        entry_signals = signal_gen.get_entry_signals(data)
        long_entries = np.sum(entry_signals == 1)
        short_entries = np.sum(entry_signals == -1)
        neutral_entries = np.sum(entry_signals == 0)
        
        print(f"エントリーシグナル統計:")
        print(f"  ロング: {long_entries} ({long_entries/len(entry_signals)*100:.1f}%)")
        print(f"  ショート: {short_entries} ({short_entries/len(entry_signals)*100:.1f}%)")
        print(f"  ニュートラル: {neutral_entries} ({neutral_entries/len(entry_signals)*100:.1f}%)")
        
        # ポジション計算
        positions = np.zeros_like(entry_signals)
        current_pos = 0
        for i in range(len(entry_signals)):
            if entry_signals[i] != 0:
                current_pos = entry_signals[i]
            positions[i] = current_pos
        
        # エグジットシグナルのテスト（簡単なテスト）
        exit_signal_test = signal_gen.get_exit_signals(data, 1, -1)  # ロングポジションのエグジット
        total_exits = 1 if exit_signal_test else 0
        
        print(f"エグジットシグナル統計:")
        print(f"  総決済: {total_exits}")
        
        print(f"\nシグナル情報:")
        print(f"  名前: HyperTrendFollowSignalGenerator")
        print(f"  種類: trend_follow")
        
        print(f"✓ シグナルジェネレーターテスト完了")
        return True
        
    except Exception as e:
        print(f"✗ シグナルジェネレーターテストでエラー: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_strategy():
    """戦略のテスト"""
    print("\n=== HyperTrendFollowStrategy テスト ===")
    
    try:
        # テスト用データの生成
        data = generate_test_data(300)
        
        # 戦略の初期化
        strategy = HyperTrendFollowStrategy(
            hyper_frama_period=16,
            hyper_frama_src_type='hl2',
            channel_period=20,
            channel_fixed_multiplier=2.0
        )
        
        print(f"✓ 戦略を初期化しました")
        
        # シグナル生成のテスト
        entry_signals = strategy.generate_entry(data)
        
        long_entries = np.sum(entry_signals == 1)
        short_entries = np.sum(entry_signals == -1)
        
        print(f"戦略シグナル統計:")
        print(f"  ロングエントリー: {long_entries}")
        print(f"  ショートエントリー: {short_entries}")
        print(f"  総エントリー: {long_entries + short_entries}")
        
        # エグジットテスト
        exit_test = strategy.generate_exit(data, 1, -1)
        print(f"  エグジットテスト: {exit_test}")
        
        # 戦略情報の表示
        strategy_info = strategy.get_strategy_info()
        print(f"\n戦略情報:")
        print(f"  名前: {strategy_info['name']}")
        print(f"  説明: {strategy_info['description']}")
        print(f"  機能数: {len(strategy_info['features'])}")
        
        print(f"✓ 戦略テスト完了")
        return True
        
    except Exception as e:
        print(f"✗ 戦略テストでエラー: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_signal_consistency():
    """シグナル一貫性のテスト"""
    print("\n=== シグナル一貫性テスト ===")
    
    try:
        # テスト用データの生成
        data = generate_test_data(200)
        
        # 同じパラメータで複数回テスト
        signal_gen1 = HyperTrendFollowSignalGenerator(
            hyper_frama_period=16,
            channel_period=20
        )
        
        signal_gen2 = HyperTrendFollowSignalGenerator(
            hyper_frama_period=16,
            channel_period=20
        )
        
        # シグナル生成
        signals1 = signal_gen1.get_entry_signals(data)
        signals2 = signal_gen2.get_entry_signals(data)
        
        # 一貫性チェック
        consistency = np.array_equal(signals1, signals2)
        print(f"シグナル一貫性: {'✓ 一致' if consistency else '✗ 不一致'}")
        
        if consistency:
            print(f"同一パラメータで同一のシグナルが生成されました")
        else:
            diff_count = np.sum(signals1 != signals2)
            print(f"不一致箇所: {diff_count}個 ({diff_count/len(signals1)*100:.2f}%)")
        
        return consistency
        
    except Exception as e:
        print(f"✗ 一貫性テストでエラー: {str(e)}")
        return False

def main():
    """メインテスト関数"""
    print("HyperTrendFollow戦略システム 総合テスト開始")
    print("=" * 60)
    
    results = []
    
    # テストの実行
    results.append(test_signal_generator())
    results.append(test_strategy())
    results.append(test_signal_consistency())
    
    # 結果のまとめ
    print(f"\n{'='*60}")
    print(f"テスト結果まとめ:")
    print(f"  実行済みテスト: {len(results)}")
    print(f"  成功: {sum(results)}")
    print(f"  失敗: {len(results) - sum(results)}")
    
    if all(results):
        print(f"✓ 全てのテストが成功しました！")
        print(f"\nHyperTrendFollow戦略システムが正常に動作しています。")
    else:
        print(f"✗ 一部のテストが失敗しました。")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    main()