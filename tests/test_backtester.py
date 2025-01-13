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
from analytics.analytics import Analytics

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
    
    # データを辞書形式に変換
    data_dict = {symbol: data}
    
    # 戦略の設定
    strategy = SupertrendRsiChopStrategy(
        supertrend_params={'period': 59, 'multiplier': 6},
        rsi_entry_params={'period': 2, 'solid': {'rsi_long_entry': 20, 'rsi_short_entry': 80}},
        rsi_exit_params={'period': 22, 'solid': {'rsi_long_exit_solid': 86, 'rsi_short_exit_solid': 14}},
        chop_params={'period': 46, 'solid': {'chop_solid': 50}}
    )
    
    # ポジションサイジングの設定
    position_sizing = FixedRatioSizing({
        'ratio': 1,  # 資金の99%を使用
        'min_position': None,  # 最小ポジションサイズの制限なし
        'max_position': None,  # 最大ポジションサイズの制限なし
        'leverage': 1  # レバレッジなし
    })
    
    # バックテスターの作成と実行
    backtester = Backtester(
        strategy=strategy,
        position_sizing=position_sizing,
        initial_balance=config['backtest']['initial_balance'],
        commission=config['backtest']['commission'],
        max_positions=config['backtest']['max_positions']
    )
    
    trades = backtester.run(data_dict)

    # バックテスト終了後の残高
    print(f"\n=== 最終結果 ===")
    print(f"初期資金: {config['backtest']['initial_balance']:.2f} USD")
    print(f"最終残高: {backtester.current_capital:.2f} USD")

    # パフォーマンス分析
    analytics = Analytics(trades, config['backtest']['initial_balance'])
    analytics.print_backtest_results()


if __name__ == '__main__':
    test_backtester()
