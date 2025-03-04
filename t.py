#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import yaml
from pathlib import Path
from backtesting.backtester import Backtester
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor

from position_sizing.fixed_ratio import FixedRatioSizing
from analytics.analytics import Analytics
from optimization.bayesian_optimizer import BayesianOptimizer
from walkforward.walkforward_optimizer import WalkForwardOptimizer
from walkforward.walkforward_analyzer import WalkForwardAnalyzer
from walkforward.walkforward_optimizer import TimeSeriesDataSplitter
from typing import List
from backtesting.trade import Trade
from montecarlo.montecarlo import MonteCarlo

from data.binance_data_source import BinanceDataSource
from strategies.implementations.trend_alpha.strategy import TrendAlphaStrategy
from strategies.implementations.hyper_trend.strategy import HyperTrendStrategy


def run_backtest(config: dict):
    """バックテストを実行"""
    # データの準備
    binance_config = config.get('binance_data', {})
    data_dir = binance_config.get('data_dir', 'data/binance')
    binance_data_source = BinanceDataSource(data_dir)
    
    # CSVデータソースはダミーとして渡す（Binanceデータソースのみを使用）
    dummy_csv_source = CSVDataSource("dummy")
    data_loader = DataLoader(
        data_source=dummy_csv_source,
        binance_data_source=binance_data_source
    )
    data_processor = DataProcessor()
    
    # データの読み込みと処理
    print("\nLoading and processing data...")
    raw_data = data_loader.load_data_from_config(config)
    processed_data = {
        symbol: data_processor.process(df)
        for symbol, df in raw_data.items()
    }
    
    # 戦略の作成
    strategy = HyperTrendStrategy()
    
    # ポジションサイジングの作成
    position_config = config.get('position_sizing', {})
    position_sizing = FixedRatioSizing(
        ratio=position_config.get('ratio', 0.2),
        leverage=position_config.get('leverage', 1.0)
    )
        
    # バックテスターの作成
    initial_balance = config.get('backtest', {}).get('initial_balance', 10000)
    commission_rate = config.get('backtest', {}).get('commission', 0.001)
    backtester = Backtester(
        strategy=strategy,
        position_manager=position_sizing,
        initial_balance=initial_balance,
        commission=commission_rate,
        verbose=True  # 詳細なログを表示
    )
    
    print("\nExecuting backtest...")
    # バックテストの実行
    trades = backtester.run(processed_data)
    
    print("\nAnalyzing results...")
    # 分析の実行と結果の出力
    analytics = Analytics(trades, initial_balance)
    
    # 最初の銘柄の終値データを取得
    first_symbol = next(iter(processed_data))
    close_prices = processed_data[first_symbol]['close'].values
    
    # バックテスト結果を出力（終値データを渡す）
    analytics.print_backtest_results(close_prices=close_prices)

    backtester.plot_balance_history()

def run_optimization(config: dict):
    """Bayesian最適化を実行"""
    print("\nStarting Bayesian optimization...")

    optimizer = BayesianOptimizer(
        strategy_class=HyperTrendStrategy,
        param_generator=HyperTrendStrategy.create_optimization_params,
        config=config,
        n_trials=300,
        n_jobs=-1
    )
    
    optimizer.optimize()
  

def run_walkforward_test(config: dict):
    """ウォークフォワードテストを実行する"""
    print("\nStarting walk-forward test...")
    
    walkforward_config = config.get('walkforward', {})
    training_days = walkforward_config.get('training_days', 180)
    testing_days = walkforward_config.get('testing_days', 90)
    min_trades = walkforward_config.get('min_trades', 15)
    initial_balance = config.get('position', {}).get('initial_balance', 10000)

    # データの読み込みと前処理
    data_dir = config['data']['data_dir']
    data_loader = DataLoader(CSVDataSource(data_dir))
    data_processor = DataProcessor()
    
    print("\nLoading and processing data...")
    raw_data = data_loader.load_data_from_config(config)
    processed_data = {
        symbol: data_processor.process(df)
        for symbol, df in raw_data.items()
    }

    # データ分割器の作成
    data_splitter = TimeSeriesDataSplitter(training_days, testing_days)

    # Bayesian最適化器の作成


    bayesian_optimizer = BayesianOptimizer(
        strategy_class=TrendAlphaStrategy,
        param_generator=TrendAlphaStrategy.create_optimization_params,
        config=config,
        n_trials=300,
        n_jobs=-1
    )

    # ウォークフォワード最適化器の作成と実行



    optimizer = WalkForwardOptimizer(
        optimizer=bayesian_optimizer,
        data_splitter=data_splitter,
        config=config,

    )
    result = optimizer.run(processed_data)

    # 結果の分析
    analyzer = WalkForwardAnalyzer(initial_balance)
    analyzer.analyze(result)

def run_montecarlo(config: dict, trades: List[Trade] = None):
    """モンテカルロシミュレーションを実行する"""
    print("\nモンテカルロシミュレーションを開始します...")
    
    # 最適化の実行
    print("\nパラメータの最適化を実行中...")
    optimizer = BayesianOptimizer(
        strategy_class=TrendAlphaStrategy,
        param_generator=TrendAlphaStrategy.create_optimization_params,
        config=config,
        n_trials=500,
        n_jobs=-1
    )
    
    best_params, best_score = optimizer.optimize()
    
    print("\n最適化が完了しました")
    print(f"最適スコア: {best_score:.2f}")
    print("最適パラメータ:")
    for param_name, param_value in best_params.items():
        print(f"  {param_name}: {param_value}")
    
    # 最適化されたパラメータを戦略クラスの形式に変換
    strategy_params = TrendAlphaStrategy.convert_params_to_strategy_format(best_params)
    
    # データの準備
    data_dir = config['data']['data_dir']
    data_loader = DataLoader(CSVDataSource(data_dir))
    data_processor = DataProcessor()
    
    # 最適化されたパラメータで戦略を作成
    strategy = TrendAlphaStrategy(**strategy_params)
    
    # ポジションサイジングの作成
    position_config = config.get('position_sizing', {})
    position_sizing = FixedRatioSizing(
        ratio=position_config.get('ratio', 0.2),
        leverage=position_config.get('leverage', 1.0)
    )
    
    # バックテスターの作成
    initial_balance = config.get('position_sizing', {}).get('initial_balance', 10000)
    commission_rate = config.get('position_sizing', {}).get('commission_rate', 0.001)
    backtester = Backtester(
        strategy=strategy,
        position_manager=position_sizing,
        initial_balance=initial_balance,
        commission=commission_rate,
        verbose=False
    )
    
    # データの読み込みと処理
    print("\nデータを読み込んでいます...")
    raw_data = data_loader.load_data_from_config(config)
    processed_data = {
        symbol: data_processor.process(df)
        for symbol, df in raw_data.items()
    }
    
    print("最適化されたパラメータでバックテストを実行中...")
    trades = backtester.run(processed_data)
    
    # モンテカルロシミュレーションの実行
    print("\n=== モンテカルロシミュレーションの実行 ===")
    monte_carlo = MonteCarlo(
        trades=trades,
        initial_capital=config['backtest']['initial_balance'],
        num_simulations=config['montecarlo']['num_simulations'],
        confidence_level=0.95
    )
    
    monte_carlo.run()
    monte_carlo.print_simulation_results()


def main():
    """メイン関数"""
    # 設定ファイルの読み込み
    config_path = Path('config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # run_walkforward_test(config)
    run_optimization(config)
    # run_montecarlo(config)


    # run_backtest(config)  


    
    
    # run_signal_combination_optimization(config)

if __name__ == '__main__':
    main()