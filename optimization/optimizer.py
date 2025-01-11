#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import optuna
import joblib
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Type, Callable
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from joblib import Parallel, delayed

# プロジェクトのルートディレクトリをPythonパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from backtesting.backtester import Backtester
from position_sizing.fixed_ratio import FixedRatioSizing
from strategies.strategy import Strategy
from strategies.supertrend_rsi_chopstrategy import SupertrendRsiChopStrategy
from data.data_loader import DataLoader
from data.data_processor import DataProcessor
from analytics.analytics import Analytics


class StrategyOptimizer:
    """戦略パラメーターの最適化を行うクラス"""
    
    def __init__(
        self,
        config_path: str,
        strategy_class: Type[Strategy],
        param_generator: Callable[[optuna.Trial], Dict[str, Any]],
        n_trials: int = 100,
        n_jobs: int = -1,
        timeout: Optional[int] = None
    ):
        """
        Args:
            config_path: 設定ファイルのパス
            strategy_class: 最適化対象の戦略クラス
            param_generator: 戦略パラメーターを生成する関数
            n_trials: 最適化の試行回数
            n_jobs: 並列処理数（-1で全CPU使用）
            timeout: タイムアウト時間（秒）
        """
        self.config_path = config_path
        self.strategy_class = strategy_class
        self.param_generator = param_generator
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.timeout = timeout
        self.best_params = None
        self.best_score = None
        self.best_trades = None
        
        # 設定ファイルの読み込み
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # データの読み込みと処理
        self._load_and_process_data()
    
    def _load_and_process_data(self) -> None:
        """データの読み込みと前処理"""
        data_config = self.config.get('data', {})
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
        self.data = processor.process(data)
        self.data_dict = {symbol: self.data}
    
    def _create_strategy(self, trial: optuna.Trial) -> Strategy:
        """Optunaのtrialから戦略インスタンスを生成

        Args:
            trial: Optunaのtrial

        Returns:
            Strategy: 戦略インスタンス
        """
        params = self.param_generator(trial)
        return self.strategy_class(**params)
    
    def _objective(self, trial: optuna.Trial) -> float:
        """最適化の目的関数

        Args:
            trial: Optunaのtrial

        Returns:
            float: アルファスコア
        """
        # 戦略の生成
        strategy = self._create_strategy(trial)
        
        # ポジションサイジングの設定
        position_sizing = FixedRatioSizing({
            'ratio': self.config['position_sizing']['params']['ratio'],
            'min_position': None,
            'max_position': None,
            'leverage': 1
        })
        
        # バックテスターの作成と実行
        backtester = Backtester(
            strategy=strategy,
            position_sizing=position_sizing,
            initial_balance=self.config['backtest']['initial_balance'],
            commission=self.config['backtest']['commission'],
            max_positions=self.config['backtest']['max_positions']
        )
        
        trades = backtester.run(self.data_dict)
        
        # トレード数が少なすぎる場合はPruning
        if len(trades) < 30:  # 最小トレード数の閾値
            raise optuna.TrialPruned()
        
        # アナリティクスの計算
        analytics = Analytics(trades, self.config['backtest']['initial_balance'])
        alpha_score = analytics.calculate_alpha_score()
        
        # 現在のベストスコアを更新
        if self.best_score is None or alpha_score > self.best_score:
            self.best_score = alpha_score
            self.best_params = trial.params
            self.best_trades = trades
        
        return alpha_score
    
    def optimize(self) -> Tuple[Dict[str, Any], float, List[Any]]:
        """最適化を実行

        Returns:
            Tuple[Dict[str, Any], float, List[Any]]: 最適パラメーター、最高スコア、最良トレード
        """
        # Optunaの設定
        study = optuna.create_study(
            study_name='alpha_score_optimization',
            direction='maximize',
            sampler=TPESampler(n_startup_trials=10),
            pruner=MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1
            )
        )
        
        # 最適化の実行
        study.optimize(
            self._objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=True
        )
        
        # 最適化結果の表示
        print("\n=== 最適化結果 ===")
        print(f"最適パラメーター: {study.best_params}")
        print(f"最高スコア: {study.best_value:.2f}")
        
        # 最適パラメーターでのバックテスト結果を表示
        if self.best_trades:
            analytics = Analytics(self.best_trades, self.config['backtest']['initial_balance'])
            print("\n=== 基本統計 ===")
            print(f"初期資金: {self.config['backtest']['initial_balance']:.2f} USD")
            print(f"最終残高: {self.best_trades.current_capital:.2f} USD")
            print(f"総リターン: {analytics.calculate_total_return():.2f}%")
            print(f"CAGR: {analytics.calculate_cagr():.2f}%")
            print(f"1トレードあたりの幾何平均リターン: {analytics.calculate_geometric_mean_return():.2f}%")
            print(f"勝率: {analytics.calculate_win_rate():.2f}%")
            print(f"総トレード数: {len(analytics.trades)}")
            print(f"勝ちトレード数: {analytics.get_winning_trades()}")
            print(f"負けトレード数: {analytics.get_losing_trades()}")
            print(f"平均保有期間（日）: {analytics.get_avg_bars_all_trades():.2f}")
            print(f"勝ちトレード平均保有期間（日）: {analytics.get_avg_bars_winning_trades():.2f}")
            print(f"負けトレード平均保有期間（日）: {analytics.get_avg_bars_losing_trades():.2f}")
            print(f"平均保有バー数: {analytics.get_avg_bars_all_trades() * 6:.2f}")  # 4時間足なので1日6バー
            print(f"勝ちトレード平均保有バー数: {analytics.get_avg_bars_winning_trades() * 6:.2f}")
            print(f"負けトレード平均保有バー数: {analytics.get_avg_bars_losing_trades() * 6:.2f}")

            # 損益統計の出力
            print("\n=== 損益統計 ===")
            print(f"総利益: {analytics.calculate_total_profit():.2f}")
            print(f"総損失: {analytics.calculate_total_loss():.2f}")
            print(f"純損益: {analytics.calculate_net_profit_loss():.2f}")
            max_profit, max_loss = analytics.calculate_max_win_loss()
            print(f"最大利益: {max_profit:.2f}")
            print(f"最大損失: {max_loss:.2f}")
            avg_profit, avg_loss = analytics.calculate_average_profit_loss()
            print(f"平均利益: {avg_profit:.2f}")
            print(f"平均損失: {avg_loss:.2f}")

            # ポジションタイプ別の分析
            print("\n=== ポジションタイプ別の分析 ===")
            print("LONG:")
            print(f"トレード数: {analytics.get_long_trade_count()}")
            print(f"勝率: {analytics.get_long_win_rate():.2f}%")
            print(f"総利益: {analytics.get_long_total_profit():.2f}")
            print(f"総損失: {analytics.get_long_total_loss():.2f}")
            print(f"純損益: {analytics.get_long_net_profit():.2f}")
            print(f"最大利益: {analytics.get_long_max_win():.2f}")
            print(f"最大損失: {analytics.get_long_max_loss():.2f}")
            print(f"総利益率: {analytics.get_long_total_profit_percentage():.2f}%")
            print(f"総損失率: {analytics.get_long_total_loss_percentage():.2f}%")
            print(f"純損益率: {analytics.get_long_net_profit_percentage():.2f}%")

            print("\nSHORT:")
            print(f"トレード数: {analytics.get_short_trade_count()}")
            print(f"勝率: {analytics.get_short_win_rate():.2f}%")
            print(f"総利益: {analytics.get_short_total_profit():.2f}")
            print(f"総損失: {analytics.get_short_total_loss():.2f}")
            print(f"純損益: {analytics.get_short_net_profit():.2f}")
            print(f"最大利益: {analytics.get_short_max_win():.2f}")
            print(f"最大損失: {analytics.get_short_max_loss():.2f}")
            print(f"総利益率: {analytics.get_short_total_profit_percentage():.2f}%")
            print(f"総損失率: {analytics.get_short_total_loss_percentage():.2f}%")
            print(f"純損益率: {analytics.get_short_net_profit_percentage():.2f}%")
            
            # リスク指標
            print("\n=== リスク指標 ===")
            max_dd, max_dd_start, max_dd_end = analytics.calculate_max_drawdown()
            print(f"最大ドローダウン: {max_dd:.2f}%")
            if max_dd_start and max_dd_end:
                print(f"最大ドローダウン期間: {max_dd_start.strftime('%Y-%m-%d %H:%M')} → {max_dd_end.strftime('%Y-%m-%d %H:%M')}")
                print(f"最大ドローダウン期間（日数）: {(max_dd_end - max_dd_start).days}日")
            
            # 全ドローダウン期間の表示
            print("\n=== ドローダウン期間 ===")
            drawdown_periods = analytics.calculate_drawdown_periods()
            for i, (dd_percent, dd_days, start_date, end_date) in enumerate(drawdown_periods[:5], 1):
                print(f"\nドローダウン {i}:")
                print(f"ドローダウン率: {dd_percent:.2f}%")
                print(f"期間: {start_date.strftime('%Y-%m-%d %H:%M')} → {end_date.strftime('%Y-%m-%d %H:%M')} ({dd_days}日)")
            
            print(f"\nシャープレシオ: {analytics.calculate_sharpe_ratio():.2f}")
            print(f"ソルティノレシオ: {analytics.calculate_sortino_ratio():.2f}")
            print(f"カルマーレシオ: {analytics.calculate_calmar_ratio():.2f}")
            print(f"VaR (95%): {analytics.calculate_value_at_risk():.2f}%")
            print(f"期待ショートフォール (95%): {analytics.calculate_expected_shortfall():.2f}%")
            print(f"ドローダウン回復効率: {analytics.calculate_drawdown_recovery_efficiency():.2f}")
            
            # トレード効率指標
            print("\n=== トレード効率指標 ===")
            print(f"プロフィットファクター: {analytics.calculate_profit_factor():.2f}")
            print(f"ペイオフレシオ: {analytics.calculate_payoff_ratio():.2f}")
            print(f"期待値: {analytics.calculate_expected_value():.2f}")
            # print(f"コモンセンスレシオ: {analytics.calculate_common_sense_ratio():.2f}")
            print(f"悲観的リターンレシオ: {analytics.calculate_pessimistic_return_ratio():.2f}")
            print(f"アルファスコア: {analytics.calculate_alpha_score():.2f}")
            print(f"SQNスコア: {analytics.calculate_sqn():.2f}")
            

        
        return study.best_params, study.best_value, self.best_trades


if __name__ == '__main__':
    # 設定ファイルのパス
    config_path = os.path.join(project_root, 'config.yaml')
    
    # 最適化の実行
    optimizer = StrategyOptimizer(
        config_path=config_path,
        strategy_class=SupertrendRsiChopStrategy,
        param_generator=lambda trial: {
            'supertrend_params': {
                'period': trial.suggest_int('supertrend_period', 5, 100, step=1),
                'multiplier': trial.suggest_float('supertrend_multiplier', 1.5, 5.0, step=0.5)
            },
            'rsi_entry_params': {
                'period': trial.suggest_int('rsi_entry_period', 2),
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
                'period': trial.suggest_int('chop_period', 3, 100, step=1),
                'solid': {
                    'chop_solid': 50
                }
            }
        },
        n_trials=100,  # 試行回数
        n_jobs=-1,     # 全CPU使用
        timeout=None   # タイムアウトなし
    )
    
    best_params, best_score, best_trades = optimizer.optimize()
