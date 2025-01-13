#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import yaml

# プロジェクトのルートディレクトリをPythonパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from backtesting.backtester import Backtester
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from strategies.supertrend_rsi_chopstrategy import SupertrendRsiChopStrategy
from position_sizing.fixed_ratio import FixedRatioSizing
from analytics.analytics import Analytics

class TestBacktester(unittest.TestCase):
    def setUp(self):
        """テストの準備"""
        # 設定ファイルの読み込み
        config_path = Path(project_root) / 'config.yaml'
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # データの準備
        data_dir = self.config['data']['data_dir']
        self.data_loader = DataLoader(CSVDataSource(data_dir))
        self.data_processor = DataProcessor()
        
        # データの読み込みと処理
        self.market_data = self._load_and_process_data()
        
        # バックテストの準備
        self.strategy = self._create_strategy()
        self.position_sizing = self._create_position_sizing()
        self.initial_balance = self.config.get('position', {}).get('initial_balance', 10000)
        self.commission_rate = self.config.get('position', {}).get('commission_rate', 0.001)
        self.backtester = Backtester(
            strategy=self.strategy,
            position_manager=self.position_sizing,
            initial_balance=self.initial_balance,
            commission=self.commission_rate
        )
    
    def _load_and_process_data(self) -> dict:
        """データの読み込みと処理"""
        # 設定ファイルからデータを読み込む
        raw_data = self.data_loader.load_data_from_config(self.config)
        
        # データの前処理
        processed_data = {}
        for symbol, df in raw_data.items():
            processed_data[symbol] = self.data_processor.process(df)
        
        return processed_data
    
    def _create_strategy(self) -> SupertrendRsiChopStrategy:
        """戦略の作成"""
        strategy_config = self.config.get('strategy', {})
        params = strategy_config.get('parameters', {})
        
        return SupertrendRsiChopStrategy(
            supertrend_params=params.get('supertrend_params', {
                'period': 10,
                'multiplier': 3.0
            }),
            rsi_entry_params=params.get('rsi_entry_params', {
                'period': 2,
                'solid': {
                    'rsi_long_entry': 20,
                    'rsi_short_entry': 80
                }
            }),
            rsi_exit_params=params.get('rsi_exit_params', {
                'period': 14,
                'solid': {
                    'rsi_long_exit_solid': 70,
                    'rsi_short_exit_solid': 30
                }
            }),
            chop_params=params.get('chop_params', {
                'period': 14,
                'solid': {
                    'chop_solid': 50
                }
            })
        )
    
    def _create_position_sizing(self) -> FixedRatioSizing:
        """ポジションサイジングの作成"""
        position_config = self.config.get('position', {})
        
        return FixedRatioSizing({
            'ratio': position_config.get('ratio', 0.99),
            'min_position': position_config.get('min_position'),
            'max_position': position_config.get('max_position'),
            'leverage': position_config.get('leverage', 1)
        })
    
    def test_backtest_execution(self):
        """バックテストの実行テスト"""
        # バックテストの実行
        trades = self.backtester.run(self.market_data)
        
        # トレードが生成されていることを確認
        self.assertIsInstance(trades, list)
        self.assertTrue(len(trades) > 0)
        
        # 分析の実行
        analytics = Analytics(trades, self.initial_balance)
        
        # 詳細な分析結果の出力
        analytics.print_backtest_results()
        
        # 基本的な検証
        if trades:
            first_trade = trades[0]
            self.assertIn('symbol', first_trade.__dict__)
            self.assertIn('entry_date', first_trade.__dict__)
            self.assertIn('exit_date', first_trade.__dict__)
            self.assertIn('entry_price', first_trade.__dict__)
            self.assertIn('exit_price', first_trade.__dict__)
            self.assertIn('profit_loss', first_trade.__dict__)
            self.assertIn('position_type', first_trade.__dict__)
        
        # パフォーマンス指標の検証
        analysis = analytics.get_full_analysis()
        
        self.assertIn('total_return', analysis)
        self.assertIn('win_rate', analysis)
        self.assertIn('sharpe_ratio', analysis)
        self.assertIn('max_drawdown', analysis)
        self.assertIn('calmar_ratio', analysis)
        self.assertIn('alpha_score', analysis)
        
        # 値の範囲チェック
        self.assertGreaterEqual(analysis['win_rate'], 0.0)
        self.assertLessEqual(analysis['win_rate'], 100.0)
        self.assertGreaterEqual(analysis['max_drawdown'], 0.0)
        self.assertLessEqual(analysis['max_drawdown'], 100.0)
    
    def test_multiple_symbols(self):
        """複数銘柄のバックテストテスト"""
        # 複数銘柄の設定
        symbols = self.data_loader.get_available_symbols()
        if len(symbols) > 1:
            # 複数銘柄のデータを読み込む
            market_data = {}
            for symbol in symbols[:2]:  # 最初の2銘柄のみテスト
                data = self.data_loader.load_market_data(symbol, '1h')
                market_data[symbol] = self.data_processor.process(data)
            
            # バックテストの実行
            trades = self.backtester.run(market_data)
            
            # 分析の実行
            analytics = Analytics(trades, self.initial_balance)
            
            # 各銘柄のトレードが存在することを確認
            trade_symbols = {trade.symbol for trade in trades}
            self.assertTrue(len(trade_symbols) > 1)
            
            # 銘柄別の分析
            for symbol in trade_symbols:
                symbol_trades = [t for t in trades if t.symbol == symbol]
                symbol_analytics = Analytics(symbol_trades, self.initial_balance)
                print(f"\n=== {symbol} の分析結果 ===")
                symbol_analytics.print_backtest_results()
    
    def test_strategy_signals(self):
        """戦略シグナルのテスト"""
        # 単一銘柄のデータを取得
        symbol = next(iter(self.market_data))
        data = self.market_data[symbol]
        
        # エントリーシグナルの生成
        entry_signals = self.strategy.generate_entry(data)
        
        # シグナルの基本的な検証
        self.assertIsInstance(entry_signals, np.ndarray)
        self.assertEqual(len(entry_signals), len(data))
        self.assertTrue(np.all(np.isin(entry_signals, [-1, 0, 1])))  # シグナルは-1, 0, 1のみ
        
        # エグジットシグナルのテスト
        for position in [1, -1]:  # ロングとショートのポジションをテスト
            exit_signal = bool(self.strategy.generate_exit(data, position))
            self.assertIsInstance(exit_signal, bool)

if __name__ == '__main__':
    unittest.main()
