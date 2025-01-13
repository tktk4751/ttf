#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import yaml
from datetime import datetime
from pathlib import Path

# プロジェクトのルートディレクトリをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from data.data_loader import DataLoader
from data.data_processor import DataProcessor
from strategies.supertrend_rsi_chopstrategy import SupertrendRsiChopStrategy
from position_sizing.fixed_ratio import FixedRatioSizing
from optimization.Bayesian_optimizer import BayesianOptimizer
from backtesting.backtester import Backtester

def load_config():
    """設定ファイルを読み込む"""
    project_root = Path(__file__).parent.parent
    config_path = os.path.join(project_root, 'config.yaml')
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def convert_params_to_strategy_format(params):
    """最適化パラメータを戦略クラスの形式に変換"""
    return {
        'supertrend_params': {
            'period': params['supertrend_period'],
            'multiplier': params['supertrend_multiplier']
        },
        'rsi_entry_params': {
            'period': 2,
            'solid': {
                'rsi_long_entry': 20,
                'rsi_short_entry': 80
            }
        },
        'rsi_exit_params': {
            'period': params['rsi_exit_period'],
            'solid': {
                'rsi_long_exit_solid': 86,
                'rsi_short_exit_solid': 14
            }
        },
        'chop_params': {
            'period': params['chop_period'],
            'solid': {
                'chop_solid': 50
            }
        }
    }

def run_optimization_and_backtest():
    """最適化とバックテストを実行"""
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
    data_dict = {symbol: data}
    
   # ポジションサイジングの設定
    position_sizing = FixedRatioSizing({
        'ratio': 1,  # 資金の99%を使用
        'min_position': None,  # 最小ポジションサイズの制限なし
        'max_position': None,  # 最大ポジションサイズの制限なし
        'leverage': 1  # レバレッジなし
    })
        
    # オプティマイザーの初期化と実行
    optimizer = BayesianOptimizer(
        strategy_class=SupertrendRsiChopStrategy,
        param_generator=SupertrendRsiChopStrategy.create_optimization_params,
        n_trials=config['optimization']['n_trials'],
        n_jobs=-1,
        timeout=None
    )
    optimizer.data = data  # データを設定

    print("\n=== 最適化を開始 ===")
    best_params, best_score = optimizer.optimize()
    print(f"\n最適化結果:")
    print(f"最適パラメーター: {best_params}")
    print(f"最高スコア: {best_score:.2f}")
    
    # パラメータを戦略クラスの形式に変換
    strategy_params = convert_params_to_strategy_format(best_params)
    
    # 最適化されたパラメーターでバックテストを実行
    print("\n=== 最適化されたパラメーターでバックテストを実行 ===")
    optimized_strategy = SupertrendRsiChopStrategy(**strategy_params)
    
    backtester = Backtester(
        strategy=optimized_strategy,
        position_sizing=position_sizing,
        initial_balance=config['backtest']['initial_balance'],
        commission=config['backtest']['commission'],
        max_positions=config['backtest']['max_positions']
    )
    
    results = backtester.run(data_dict)
    analytics = results.get_analytics()
    analytics.print_backtest_results()

if __name__ == "__main__":
    run_optimization_and_backtest()
