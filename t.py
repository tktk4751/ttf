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
from position_sizing.atr_volatility import ATRVolatilitySizing
from position_sizing.volatility_std import AlphaVolatilitySizing
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

from position_sizing.z_position_sizing import ZATRPositionSizing

from strategies.implementations.trend_alpha_v2.strategy import TrendAlphaV2Strategy
from strategies.implementations.hyper_trend_v2.strategy import HyperTrendV2Strategy
from strategies.implementations.trend_alpha_v3.strategy import TrendAlphaV3Strategy
from strategies.implementations.true_momentum.strategy import TrueMomentumStrategy

from strategies.implementations.alpha_keltner_filter.strategy import AlphaKeltnerFilterStrategy
from strategies.implementations.alpha_trend_filter.strategy import AlphaTrendFilterStrategy
from strategies.implementations.alpha_momentum_filter.strategy import AlphaMomentumFilterStrategy

from strategies.implementations.kernel_ma_filter.strategy import KernelMAFilterStrategy
from strategies.implementations.alpha_circulation_strategy.strategy import AlphaCirculationStrategy

from strategies.implementations.alpha_roc_divergence_strategy.strategy import AlphaROCDivergenceStrategy

from strategies.implementations.alpha_trend_macd_hidden_divergence_strategy.strategy import AlphaTrendMACDHiddenDivergenceStrategy

from strategies.implementations.alpha_ma_filter.strategy import AlphaMAFilterStrategy
from strategies.implementations.alpha_mav2_keltner_filter.strategy import AlphaMAV2KeltnerFilterStrategy
from strategies.implementations.kama_keltner_chop_long_short.strategy import KAMAKeltnerChopLongShortStrategy
from strategies.implementations.alpha_donchian_filter.strategy import AlphaDonchianFilterStrategy

from strategies.implementations.alpha_band_trend_filter.strategy import AlphaBandTrendFilterStrategy      
from position_sizing.atr_risk_sizing import AlphaATRRiskSizing

from strategies.implementations.zc_breakout.strategy import ZCBreakoutStrategy
from strategies.implementations.squeeze_chop_short.strategy import SqueezeChopShortStrategy
from strategies.implementations.z_trend.strategy import ZTrendStrategy
from strategies.implementations.z_strategy.strategy import ZStrategy
from strategies.implementations.z_divergence.strategy import ZDivergenceStrategy
from strategies.implementations.z_donchian_trend.strategy import ZDonchianTrendStrategy
from strategies.implementations.z_bollinger_trend.strategy import ZBBTrendStrategy

from strategies.implementations.zc_simple.strategy import ZCSimpleStrategy
from strategies.implementations.zt_simple.strategy import ZTSimpleStrategy    
from strategies.implementations.zbb_simple.strategy import ZBBSimpleStrategy

from strategies.implementations.zv_breakout.strategy import ZVBreakoutStrategy

from strategies.implementations.zc_trend.strategy import ZCTrendStrategy
from strategies.implementations.zma_cross.strategy import ZMACrossStrategy
from strategies.implementations.zc_rsx_exit.strategy import ZCRSXExitStrategy

from strategies.implementations.simple_z_donchian.strategy import SimpleZDonchianStrategy
from strategies.implementations.simple_z_trend.strategy import SimpleZTrendStrategy

from strategies.implementations.z_breakout.strategy import ZBreakoutStrategy

from strategies.implementations.z_macd_breakout.strategy import ZMACDBreakoutStrategy

from strategies.implementations.cc_breakout.strategy import CCBreakoutStrategy
from position_sizing.c_position_sizing import CATRPositionSizing

from strategies.implementations.dual_cc_breakout.strategy import DualCCBreakoutStrategy
from strategies.implementations.cz_simple.strategy import CZSimpleStrategy
from strategies.implementations.cc_simple.strategy import CCSimpleStrategy
from strategies.implementations.supertrend_chop_long.strategy import SupertrendChopLongStrategy

from strategies.implementations.x_trend_simple.strategy import XTrendSimpleStrategy
from strategies.implementations.xc_simple.strategy import XCSimpleStrategy
from strategies.implementations.za_simple.strategy import ZASimpleStrategy
from strategies.implementations.z_adaptive_ma_crossover.strategy import ZAdaptiveMACrossoverStrategy
from strategies.implementations.kama_cross_chop_long.strategy import KAMACrossChopLongStrategy
from strategies.implementations.za_fillter.strategy import ZACTIStrategy
from strategies.implementations.za_chop.strategy import ZACHOPStrategy


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


    strategy = ZASimpleStrategy()
    
    # ポジションサイジングの作成
    position_config = config.get('position_sizing', {})
    # position_sizing = AlphaATRRiskSizing(
    #         risk_ratio=0.01,   # 1%リスク
    #         unit=1.0,         # 1倍のユニット
    #         max_position_percent=0.5  # 資金の50%まで
    #     )

    position_sizing = CATRPositionSizing()
    # position_sizing = AlphaVolatilitySizing()
    # position_sizing = FixedRatioSizing(
    #     ratio=position_config.get('ratio', 0.2),
    #     leverage=position_config.get('leverage', 1.0)
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
        strategy_class=ZASimpleStrategy,
        param_generator=ZASimpleStrategy.create_optimization_params,
        config=config,
        n_trials=100,
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

    # データ分割器の作成
    data_splitter = TimeSeriesDataSplitter(training_days, testing_days)

    # Bayesian最適化器の作成
    bayesian_optimizer = BayesianOptimizer(



        strategy_class=CCBreakoutStrategy,
        param_generator=CCBreakoutStrategy.create_optimization_params,
        config=config,
        n_trials=100,
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
        strategy_class=CCBreakoutStrategy,
        param_generator=CCBreakoutStrategy.create_optimization_params,
        config=config,
        n_trials=200,
        n_jobs=-1
    )
    
    best_params, best_score = optimizer.optimize()
    
    print("\n最適化が完了しました")
    print(f"最適スコア: {best_score:.2f}")
    print("最適パラメータ:")
    for param_name, param_value in best_params.items():
        print(f"  {param_name}: {param_value}")
    
    # 最適化されたパラメータを戦略クラスの形式に変換
    strategy_params = CCBreakoutStrategy.convert_params_to_strategy_format(best_params)
    
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
    

    
    # 最適化されたパラメータで戦略を作成
    strategy = CCBreakoutStrategy(**strategy_params)
    
    position_sizing = CATRPositionSizing()
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
    print("\nLoading and processing data...")
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

    # run_optimization(config)
    # run_montecarlo(config)
    run_backtest(config)

    
if __name__ == '__main__':
    main()


