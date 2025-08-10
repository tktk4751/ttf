#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
import os
import yaml

# プロジェクトのルートディレクトリを追加
sys.path.append('/home/vapor/dev/ttf')

from strategies.implementations.hyper_frama.strategy import HyperFRAMAEnhancedStrategy
from data.binance_data_source import BinanceDataSource
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from backtesting.backtester import Backtester
from position_sizing.x_position_sizing import XATRPositionSizing

def run_identical_backtest():
    """t.pyと同じ設定でバックテストを実行"""
    print("=== t.pyと同じ設定でのバックテスト実行 ===")
    
    # 設定ファイル読み込み
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # データの準備（t.pyと同じ）
    binance_config = config.get('binance_data', {})
    data_dir = binance_config.get('data_dir', 'data/binance')
    binance_data_source = BinanceDataSource(data_dir)
    
    dummy_csv_source = CSVDataSource("dummy")
    data_loader = DataLoader(
        data_source=dummy_csv_source,
        binance_data_source=binance_data_source
    )
    data_processor = DataProcessor()
    
    # データの読み込みと処理
    print("データ読み込み中...")
    raw_data = data_loader.load_data_from_config(config)
    processed_data = {
        symbol: data_processor.process(df)
        for symbol, df in raw_data.items()
    }
    
    # ストラテジーの設定（t.pyと同じ）
    strategy = HyperFRAMAEnhancedStrategy()
    
    print(f"✓ ストラテジー設定:")
    print(f"  名前: {strategy.name}")
    print(f"  Position Mode: {strategy._parameters['position_mode']}")
    print(f"  Filter Type: {strategy._parameters['filter_type']}")
    print(f"  Period: {strategy._parameters['period']}")
    
    # ポジションサイジング（t.pyと同じ）
    position_sizing = XATRPositionSizing()
    
    # バックテスターの作成（t.pyと同じ）
    initial_balance = config.get('backtest', {}).get('initial_balance', 10000)
    commission_rate = config.get('backtest', {}).get('commission', 0.001)
    backtester = Backtester(
        strategy=strategy,
        position_manager=position_sizing,
        initial_balance=initial_balance,
        commission=commission_rate,
        verbose=True  # 詳細なログを表示
    )
    
    print(f"\nバックテスト実行中...")
    print(f"  初期資金: ${initial_balance}")
    print(f"  手数料: {commission_rate}")
    
    # データからシグナルを事前に確認
    first_symbol = next(iter(processed_data))
    data = processed_data[first_symbol]
    
    print(f"\n事前シグナル解析:")
    entry_signals = strategy.generate_entry(data)
    long_signals = strategy.get_long_signals(data)
    short_signals = strategy.get_short_signals(data)
    
    print(f"  データ長: {len(data)}")
    print(f"  ロングエントリーシグナル: {np.sum(long_signals)}")
    print(f"  ショートエントリーシグナル: {np.sum(short_signals)}")
    print(f"  統合エントリーシグナル: ロング={np.sum(entry_signals == 1)}, ショート={np.sum(entry_signals == -1)}")
    
    # ロングシグナルの詳細
    if np.sum(long_signals) > 0:
        long_indices = np.where(long_signals == 1)[0]
        print(f"  ロングシグナルインデックス（最初の10個）: {long_indices[:10].tolist()}")
        
        # 最初のロングシグナルの詳細
        first_long_idx = long_indices[0]
        print(f"  最初のロングシグナル:")
        print(f"    インデックス: {first_long_idx}")
        print(f"    日時: {data.index[first_long_idx]}")
        print(f"    価格: ${data['close'].iloc[first_long_idx]:.2f}")
    
    # バックテストの実行
    trades = backtester.run(processed_data)
    
    # 結果の分析
    print(f"\n✓ バックテスト結果:")
    print(f"  総トレード数: {len(trades)}")
    
    if len(trades) > 0:
        long_trades = [t for t in trades if t.position_type == 'LONG']
        short_trades = [t for t in trades if t.position_type == 'SHORT']
        
        print(f"  ロングトレード: {len(long_trades)}")
        print(f"  ショートトレード: {len(short_trades)}")
        
        # 最初の数個のトレードの詳細
        print(f"\n  最初の5トレード詳細:")
        for i, trade in enumerate(trades[:5]):
            print(f"    [{i+1}] {trade.position_type}: {trade.entry_date} -> {trade.exit_date}")
            print(f"        エントリー: ${trade.entry_price:.2f}, エグジット: ${trade.exit_price:.2f}")
            print(f"        損益: ${trade.profit_loss:.2f} ({trade.profit_loss_pct:.2f}%)")
    else:
        print(f"  ❌ トレードが実行されませんでした")
        
        print(f"\n  トレード未実行の原因調査:")
        # ポジションサイジングをテスト
        test_price = data['close'].iloc[100]  # 適当な価格
        test_balance = initial_balance
        sizing_result = position_sizing.calculate_position_size(test_balance, test_price, data.iloc[100:110])
        
        print(f"    テスト用ポジションサイジング:")
        print(f"      価格: ${test_price:.2f}")
        print(f"      残高: ${test_balance}")
        print(f"      ポジションサイズ: {sizing_result}")
    
    return len(trades), len([t for t in trades if t.position_type == 'LONG']), len([t for t in trades if t.position_type == 'SHORT'])

def main():
    """メイン関数"""
    print("t.pyと同じ設定でのバックテスト診断")
    print("=" * 60)
    
    try:
        total_trades, long_trades, short_trades = run_identical_backtest()
        
        print(f"\n{'='*60}")
        print(f"最終結果:")
        print(f"  総トレード: {total_trades}")
        print(f"  ロングトレード: {long_trades}")
        print(f"  ショートトレード: {short_trades}")
        
        if total_trades == 0:
            print(f"\n⚠️ 問題特定: バックテスターがトレードを実行していません")
            print(f"  考えられる原因:")
            print(f"  1. ポジションサイジングの問題")
            print(f"  2. バックテスターのエントリー条件の問題")
            print(f"  3. データ処理の問題")
        elif long_trades == 0:
            print(f"\n⚠️ 問題特定: ロングトレードが実行されていません")
            print(f"  考えられる原因:")
            print(f"  1. ポジションサイジングがロングを拒否している")
            print(f"  2. バックテスターのロングエントリー条件に問題")
        else:
            print(f"\n✅ 正常: ロングトレードが実行されました")
        
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"エラー: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()