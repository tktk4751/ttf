#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import pandas as pd

# プロジェクトのルートディレクトリをPythonパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from typing import Dict, Callable
from datetime import datetime
from data.data_loader import DataLoader
from data.data_processor import DataProcessor
from strategies.supertrend_rsi_chopstrategy import SupertrendRsiChopStrategy
from position_sizing.fixed_ratio import FixedRatioSizing
from backtesting.backtester import Backtester
from optimization.Bayesian_optimizer import BayesianOptimizer
from montecarlo.montecarlo import MonteCarlo


def create_supertrend_rsi_params(trial):
    """SupertrendRsiChopStrategyのパラメーター生成関数"""
    return {
        'supertrend_params': {
            'period': trial.suggest_int('supertrend_period', 5, 100, step=1),
            'multiplier': trial.suggest_float('supertrend_multiplier', 1.5, 8.0, step=0.5)
        },
        'rsi_entry_params': {
            'period': 2,
            'solid': {
                'rsi_long_entry': 20,
                'rsi_short_entry': 80
            }
        },
        'rsi_exit_params': {
            'period': trial.suggest_int('rsi_exit_period', 5, 34, step=1),
            'solid': {
                'rsi_long_exit_solid': 85,
                'rsi_short_exit_solid': 15
            }
        },
        'chop_params': {
            'period': trial.suggest_int('chop_period', 5, 100, step=1),
            'solid': {
                'chop_solid': 50
            }
        }
    }


def load_config() -> Dict:
    """設定ファイルを読み込む"""
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)


def run_optimization_and_montecarlo():
    """最適化とモンテカルロシミュレーションを実行"""
    # 設定を読み込む
    config = load_config()

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
    df = {symbol: data}
    
    # 最適化の実行
    print("\n=== パラメーター最適化の開始 ===")
    optimizer = BayesianOptimizer(
        config_path=os.path.join(project_root, 'config.yaml'),
        strategy_class=SupertrendRsiChopStrategy,
        param_generator=create_supertrend_rsi_params,
        n_trials=100,
        n_jobs=-1,
        timeout=None
    )
    
    best_params, best_score = optimizer.optimize()
    print("\n最適化結果:")
    print(f"最適パラメーター: {best_params}")
    print(f"最適スコア: {best_score}")
    
    # 最適化されたパラメーターでバックテストを実行
    print("\n=== 最適化パラメーターでのバックテスト実行 ===")
    strategy = SupertrendRsiChopStrategy(
        supertrend_params={
            'period': best_params['supertrend_period'],
            'multiplier': best_params['supertrend_multiplier']
        },
        rsi_entry_params={
            'period': 2,
            'solid': {
                'rsi_long_entry': 20,
                'rsi_short_entry': 80
            }
        },
        rsi_exit_params={
            'period': best_params['rsi_exit_period'],
            'solid': {
                'rsi_long_exit_solid': 85,
                'rsi_short_exit_solid': 15
            }
        },
        chop_params={
            'period': best_params['chop_period'],
            'solid': {
                'chop_solid': 50
            }
        }
    )
    
    # ポジションサイジングの設定
    position_sizing = FixedRatioSizing({
        'ratio': 1,  # 資金の99%を使用
        'min_position': None,  # 最小ポジションサイズの制限なし
        'max_position': None,  # 最大ポジションサイズの制限なし
        'leverage': 1  # レバレッジなし
    })
    
    backtest = Backtester(
        strategy=strategy,
        position_sizing=position_sizing,
        initial_balance=config['backtest']['initial_balance'],
        commission=config['backtest']['commission'],
        max_positions=1
    )
    
    trades = backtest.run(df)
    
    # モンテカルロシミュレーションの実行
    print("\n=== モンテカルロシミュレーションの開始 ===")
    monte_carlo = MonteCarlo(
        trades=trades,
        initial_capital=config['backtest']['initial_balance'],
        num_simulations=config['montecarlo']['num_simulations'],
        confidence_level=0.95
    )
    
    monte_carlo.run()
    monte_carlo.print_simulation_results()


if __name__ == "__main__":
    # スクリプトとして実行された場合、最適化とモンテカルロシミュレーションを実行
    run_optimization_and_montecarlo()