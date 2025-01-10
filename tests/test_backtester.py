#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import yaml
from datetime import datetime

# プロジェクトのルートディレクトリをPythonパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd
from backtesting.backtester import Backtester
from position_sizing.fixed_ratio import FixedRatioSizing
from strategies.supertrend_rsi_chopstrategy import SupertrendRsiChopStrategy
from data.data_loader import DataLoader
from data.data_processor import DataProcessor

def test_backtester():
    """バックテスターのテスト"""
    
    # 設定ファイルの読み込み
    config_path = os.path.join(project_root, 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # データの設定を取得
    data_config = config.get('data', {})
    data_dir = data_config.get('data_dir', 'data')
    symbol = data_config.get('symbol', 'BTCUSDT')
    timeframe = data_config.get('timeframe', '1h')
    start_date = data_config.get('start')
    end_date = data_config.get('end')
    
    # 日付文字列をdatetimeオブジェクトに変換
    start_dt = datetime.strptime(start_date, '%Y-%m-%d') if start_date else None
    end_dt = datetime.strptime(end_date, '%Y-%m-%d') if end_date else None
    
    # データの読み込みと処理
    loader = DataLoader(data_dir)
    processor = DataProcessor()
    
    data = loader.load_data(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_dt,
        end_date=end_dt
    )
    data = processor.process(data)
    
    # 戦略の設定
    strategy = SupertrendRsiChopStrategy(
        supertrend_params={'period': 5, 'multiplier': 3.0},
        rsi_entry_params={'period': 2, 'solid': {'rsi_long_entry': 20, 'rsi_short_entry': 80}},
        rsi_exit_params={'period': 14, 'solid': {'rsi_long_exit_solid': 85, 'rsi_short_exit_solid': 20}},
        chop_params={'period': 14, 'solid': {'chop_solid': 50}}
    )
    
    # ポジションサイジングの設定
    position_sizing = FixedRatioSizing({
        'ratio': config['position_sizing']['params']['ratio'],
        'min_position': 0.01,
        'max_position': 1.0
    })
    
    # バックテスターの作成と実行
    backtester = Backtester(
        strategy=strategy,
        position_sizing=position_sizing,
        initial_capital=config['backtest']['initial_balance'],
        max_positions=config['backtest']['max_positions']
    )
    
    results = backtester.run(data)
    
    # 結果の表示
    print("\n=== バックテスト結果 ===")
    print(f"総トレード数: {results['total_trades']}")
    print(f"勝率: {results['win_rate']:.2f}%")
    print(f"総利益: {results['total_profit']:.2f} USD")
    print(f"総損失: {results['total_loss']:.2f} USD")
    print(f"純利益: {results['net_profit']:.2f} USD")
    print(f"最終資金: {results['final_capital']:.2f} USD")
    
    # 各トレードの詳細表示
    print("\n=== 個別トレード ===")
    for i, trade in enumerate(backtester.trades, 1):
        print(f"\nトレード {i}:")
        print(f"タイプ: {trade.position_type}")
        print(f"エントリー日時: {trade.entry_date}")
        print(f"エントリー価格: {trade.entry_price:.2f}")
        print(f"エグジット日時: {trade.exit_date}")
        print(f"エグジット価格: {trade.exit_price:.2f}")
        print(f"損益: {trade.profit_loss:.2f} USD ({trade.profit_loss_pct:.2f}%)")

if __name__ == '__main__':
    test_backtester()
