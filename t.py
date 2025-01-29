#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import yaml
from pathlib import Path
from backtesting.backtester import Backtester
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from strategies.implementations.supertrend_rsi_chop.strategy import SupertrendRsiChopStrategy
from strategies.implementations.supertrend_chop_mfi.strategy import SupertrendChopMfiStrategy
from strategies.implementations.alma_trend_following.strategy import ALMATrendFollowingStrategy
from strategies.implementations.alma_trend_following_v2.strategy import ALMATrendFollowingV2Strategy
from strategies.implementations.donchian_chop_long.strategy import DonchianChopLongStrategy
from strategies.implementations.donchian_adx_long.strategy import DonchianADXLongStrategy
from strategies.implementations.donchian_adx_short.strategy import DonchianADXShortStrategy
from strategies.implementations.donchian_chop_short.strategy import DonchianChopShortStrategy
from strategies.implementations.supertrend_chop_long.strategy import SupertrendChopLongStrategy
from strategies.implementations.supertrend_chop_short.strategy import SupertrendChopShortStrategy
from strategies.implementations.supertrend_adx_long.strategy import SupertrendADXLongStrategy
from strategies.implementations.squeeze_chop_long.strategy import SqueezeChopLongStrategy
from strategies.implementations.supertrend_adx_short.strategy import SupertrendADXShortStrategy
from strategies.implementations.squeeze_chop_short.strategy import SqueezeChopShortStrategy

from strategies.implementations.rsi_div_roc_long.strategy import RSIDivROCLongStrategy
from strategies.implementations.mfi_div_roc_short.strategy import MFIDivROCShortStrategy
from strategies.implementations.mfi_div_roc_long.strategy import MFIDivROCLongStrategy
from strategies.implementations.alma_cycle.strategy import ALMACycleStrategy
from position_sizing.fixed_ratio import FixedRatioSizing
from analytics.analytics import Analytics
from optimization.bayesian_optimizer import BayesianOptimizer
import optuna
from walkforward.walkforward_optimizer import WalkForwardOptimizer
from walkforward.walkforward_analyzer import WalkForwardAnalyzer
from walkforward.walkforward_optimizer import TimeSeriesDataSplitter
from typing import List
from backtesting.trade import Trade
from montecarlo.montecarlo import MonteCarlo
from optimization.signal_combination_optimizer import SignalCombinationOptimizer


from strategies.implementations.supertrend_adx_rsi_long.strategy import SupertrendADXRSILongStrategy
from strategies.implementations.supertrend_chop_rsi_long.strategy import SupertrendChopRSILongStrategy
from strategies.implementations.supertrend_adx_rsi_short.strategy import SupertrendADXRSIShortStrategy
from strategies.implementations.supertrend_chop_rsi_short.strategy import SupertrendChopRSIShortStrategy

from strategies.implementations.chop_rsi_donchian_long.strategy import ChopRSIDonchianLongStrategy
from strategies.implementations.chop_rsi_donchian_short.strategy import ChopRSIDonchianShortStrategy



def run_backtest(config: dict):
    """バックテストを実行"""
    # データの準備
    data_dir = config['data']['data_dir']
    data_loader = DataLoader(CSVDataSource(data_dir))
    data_processor = DataProcessor()
    
    # 戦略の作成

    strategy = MFIDivROCLongStrategy()
    
    # ポジションサイジングの作成
    position_config = config.get('position_sizing', {})
    position_sizing = FixedRatioSizing(
        ratio=position_config.get('ratio', 0.5),
        leverage=position_config.get('leverage', 1.0)
    )

    # # インスタンス化
    # position_sizing = ATRPositionSizing(
    # risk_percent=0.04,  # 4%のリスク
    # atr_period=14,
    # supertrend_period=10,


    # supertrend_multiplier=3.0
    # )

    # # ポジションサイズの計算
    # position_size = position_sizing.calculate(
    # capital=10000,      # 総資金
    # price=100,          # 現在価格
    # data=processed_data,    # 価格データ（OHLCV）
    # index=current_index,  # 現在のインデックス
    # direction=1         # ロングポジション
    # )
    
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
    
    # データの読み込みと処理
    print("\nLoading and processing data...")
    raw_data = data_loader.load_data_from_config(config)
    processed_data = {
        symbol: data_processor.process(df)
        for symbol, df in raw_data.items()
    }
    
    print("\nExecuting backtest...")
    # バックテストの実行
    trades = backtester.run(processed_data)
    
    print("\nAnalyzing results...")
    # 分析の実行と結果の出力
    analytics = Analytics(trades, initial_balance)
    analytics.print_backtest_results()
    
    # 銘柄別の分析
    trade_symbols = {trade.symbol for trade in trades}
    for symbol in trade_symbols:
        symbol_trades = [t for t in trades if t.symbol == symbol]
        symbol_analytics = Analytics(symbol_trades, initial_balance)
        print(f"\n=== {symbol} の分析結果 ===")
        symbol_analytics.print_backtest_results()

def run_optimization(config: dict):
    """Bayesian最適化を実行"""
    print("\nStarting Bayesian optimization...")

    optimizer = BayesianOptimizer(
        strategy_class=ChopRSIDonchianLongStrategy,
        param_generator=ChopRSIDonchianLongStrategy.create_optimization_params,
        config=config,
        n_trials=300,
        n_jobs=-1
    )
    
    optimizer.optimize()
    
    # # データの準備
    # data_dir = config['data']['data_dir']
    # data_loader = DataLoader(CSVDataSource(data_dir))
    # data_processor = DataProcessor()
    
    # # データの読み込みと処理
    # print("\nLoading and processing data...")
    # raw_data = data_loader.load_data_from_config(config)
    # processed_data = {
    #     symbol: data_processor.process(df)
    #     for symbol, df in raw_data.items()
    # }
    
    # print("\n=== 最終バックテスト結果 ===")
    # print(f"使用データ期間: {next(iter(processed_data.values())).index[0]} → {next(iter(processed_data.values())).index[-1]}")
    
    # # 最適化されたパラメータで戦略を作成
    # strategy_params = SupertrendChopMfiStrategy.convert_params_to_strategy_format(best_params)
    # strategy = SupertrendChopMfiStrategy(**strategy_params)
    
    # print("\nOptimization completed!")
    # print(f"Best score: {best_score:.2f}")
    # print("Best parameters:")
    # for param_name, param_value in best_params.items():
    #     print(f"  {param_name}: {param_value}")
    
    # # ポジションサイジングの作成
    # position_config = config.get('position_sizing', {})
    # position_sizing = FixedRatioSizing(
    #     ratio=position_config.get('ratio', 0.5),  # デフォルト値を0.5に設定
    #     leverage=position_config.get('leverage', 1.0)
    # )
    
    # # バックテスターの作成
    # initial_balance = config.get('position_sizing', {}).get('initial_balance', 10000)
    # commission_rate = config.get('position_sizing', {}).get('commission_rate', 0.001)
    # backtester = Backtester(
    #     strategy=strategy,
    #     position_manager=position_sizing,
    #     initial_balance=initial_balance,
    #     commission=commission_rate,
    #     verbose=True
    # )
    
    # print("\nExecuting backtest with optimized parameters...")
    # # バックテストの実行
    # backtester.run(processed_data)
    
    # print("\nAnalyzing results...")
    # # 分析の実行と結果の出力
    # analytics = Analytics(trades, initial_balance)
    # analytics.print_backtest_results()
    
    # # 銘柄別の分析
    # trade_symbols = {trade.symbol for trade in trades}
    # for symbol in trade_symbols:
    #     symbol_trades = [t for t in trades if t.symbol == symbol]
    #     symbol_analytics = Analytics(symbol_trades, initial_balance)
    #     print(f"\n=== {symbol} の分析結果 ===")
    #     symbol_analytics.print_backtest_results()

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
        strategy_class=DonchianChopLongStrategy,
        param_generator=DonchianChopLongStrategy.create_optimization_params,
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
        strategy_class=DonchianChopLongStrategy,
        param_generator=DonchianChopLongStrategy.create_optimization_params,
        config=config,
        n_trials=300,
        n_jobs=-1
    )
    
    best_params, best_score = optimizer.optimize()
    
    print("\n最適化が完了しました")
    print(f"最適スコア: {best_score:.2f}")
    print("最適パラメータ:")
    for param_name, param_value in best_params.items():
        print(f"  {param_name}: {param_value}")
    
    # 最適化されたパラメータを戦略クラスの形式に変換
    strategy_params = DonchianChopLongStrategy.convert_params_to_strategy_format(best_params)
    
    # データの準備
    data_dir = config['data']['data_dir']
    data_loader = DataLoader(CSVDataSource(data_dir))
    data_processor = DataProcessor()
    
    # 最適化されたパラメータで戦略を作成
    strategy = DonchianChopLongStrategy(**strategy_params)
    
    # ポジションサイジングの作成
    position_config = config.get('position_sizing', {})
    position_sizing = FixedRatioSizing(
        ratio=position_config.get('ratio', 0.5),
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

def run_signal_combination_optimization(config: dict):
    """シグナルの組み合わせの最適化を実行"""
    print("\nシグナルの組み合わせの最適化を開始します...")
    
    # カスタム評価指標の例（シャープレシオとアルファスコアの組み合わせ）
    def custom_metric(analytics: Analytics) -> float:
        return analytics.calculate_alpha_score() * 0.7 + analytics.calculate_sharpe_ratio() * 0.3
    
    optimizer = SignalCombinationOptimizer(
        config=config,
        n_trials=300,
        max_signals=3,
        metric_function=custom_metric,  # カスタム評価指標を使用（コメントアウトするとデフォルトのアルファスコアを使用）
        n_jobs=-1
    )
    
    best_params, best_score = optimizer.optimize()
    
    print("\n最適化が完了しました！")
    print(f"最高スコア: {best_score:.2f}")
    print("\n選択されたシグナルの組み合わせ:")
    for i, signal_name in enumerate(best_params['selected_signals'], 1):
        print(f"{i}. {signal_name}")

def main():
    """メイン関数"""
    # 設定ファイルの読み込み
    config_path = Path('config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # run_walkforward_test(config)
    run_optimization(config)
    # run_montecarlo(config)
    # run_backtest(config)  # この行を削除または#でコメントアウト


    
    
    # run_signal_combination_optimization(config)

if __name__ == '__main__':
    main()