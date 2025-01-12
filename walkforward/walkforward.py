#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, List, Any, Type, Callable
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import yaml

# プロジェクトのルートディレクトリをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from optimization.Bayesian_optimizer import BayesianOptimizer
from strategies.strategy import Strategy
from analytics.analytics import Analytics
from data.data_loader import DataLoader
from data.data_processor import DataProcessor
from backtesting.backtester import Backtester
from position_sizing.position_sizing import PositionSizing
from position_sizing.fixed_ratio import FixedRatioSizing
from logger import get_logger

logger = get_logger(__name__)


class WalkForward:
    def __init__(
        self,
        strategy_class: Type[Strategy],
        config_path: str,
        param_generator: Callable,
    ):
        """ウォークフォワードテストの初期化

        Args:
            strategy_class: 戦略クラス
            config_path: 設定ファイルのパス
            param_generator: パラメータ生成関数
        """
        self.strategy_class = strategy_class
        self.param_generator = param_generator
        self.config_path = config_path

        # 設定ファイルの読み込み
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # walkforward設定の読み込み
        walkforward_config = self.config.get('walkforward', {})
        self.training_days = walkforward_config.get('training_days', 360)
        self.testing_days = walkforward_config.get('testing_days', 180)
        self.min_trades = walkforward_config.get('min_trades', 15)

        # データの読み込みと前処理
        self._load_and_process_data()

    def _load_and_process_data(self) -> None:
        """データの読み込みと前処理"""
        data_config = self.config.get('data', {})
        data_dir = data_config.get('data_dir', 'data')
        symbol = data_config.get('symbol', 'BTCUSDT')
        timeframe = data_config.get('timeframe', '1h')
        
        # データの読み込みと処理
        loader = DataLoader(data_dir)
        processor = DataProcessor()
        
        # 全期間のデータを読み込む
        data = loader.load_data(
            symbol=symbol,
            timeframe=timeframe,
            load_all=True  # 全データを読み込む
        )
        self.data = processor.process(data)
        
        logger.info(f"データを読み込みました: {symbol}/{timeframe}")
        logger.info(f"期間: {self.data.index.min()} → {self.data.index.max()}")
        logger.info(f"データ数: {len(self.data)}")

    def _split_data(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        """指定された期間のデータを抽出

        Args:
            start_date: 開始日
            end_date: 終了日

        Returns:
            pd.DataFrame: 抽出されたデータ
        """
        mask = (self.data.index >= start_date) & (self.data.index <= end_date)
        return self.data[mask].copy()

    def run(self) -> Dict[str, Any]:
        """ウォークフォワードテストを実行"""
        results = []
        data_start = self.data.index.min()
        data_end = self.data.index.max()

        logger.info(f"全データ期間: {data_start} → {data_end}")
        
        if (data_end - data_start).days < (self.training_days + self.testing_days):
            logger.error(f"データ期間が短すぎます。少なくとも {self.training_days + self.testing_days} 日必要です。")
            return results

        # 最初のトレーニング期間とテスト期間を設定
        test_start = data_start + pd.Timedelta(days=self.training_days)
        test_end = test_start + pd.Timedelta(days=self.testing_days)

        while test_end <= data_end:
            # トレーニング期間は現在のテスト期間の終わりから360日前まで
            training_start = test_start - pd.Timedelta(days=self.training_days)
            training_end = test_start  # テスト期間の開始まで
            
            training_data = self.data[training_start:training_end].copy()
            testing_data = self.data[test_start:test_end].copy()
            
            if len(training_data) < 100 or len(testing_data) < 50:  # 最小データ数のチェック
                logger.warning(f"データ数が不足しています: トレーニング {len(training_data)}, テスト {len(testing_data)}")
                test_start = test_end
                test_end = test_start + pd.Timedelta(days=self.testing_days)
                continue
            
            logger.info(f"トレーニング期間: {training_start} → {training_end} (データ数: {len(training_data)})")
            logger.info(f"テスト期間: {test_start} → {test_end} (データ数: {len(testing_data)})")

            try:
                # パラメータの最適化
                optimizer = BayesianOptimizer(
                    config_path=self.config_path,
                    strategy_class=self.strategy_class,
                    param_generator=self.param_generator,
                    n_trials=100,
                    n_jobs=1
                )
                optimizer.data = training_data
                best_params, best_value = optimizer.optimize()
                
                # パラメータを戦略クラスが期待する形式に変換
                strategy_params = {
                    'supertrend_params': {
                        'period': best_params['supertrend_period'],
                        'multiplier': best_params['supertrend_multiplier']
                    },
                    'rsi_entry_params': {
                        'period': 2,
                        'solid': {
                            'rsi_long_entry': 20,
                            'rsi_short_entry': 80
                        }
                    },
                    'rsi_exit_params': {
                        'period': best_params['rsi_exit_period'],
                        'solid': {
                            'rsi_long_exit_solid': 85,
                            'rsi_short_exit_solid': 15
                        }
                    },
                    'chop_params': {
                        'period': best_params['chop_period'],
                        'solid': {
                            'chop_solid': 50
                        }
                    }
                }
                
                # 変換したパラメータを使用
                strategy = self.strategy_class(**strategy_params)
                
                logger.info(f"最適化完了 - スコア: {best_value:.2f}")
                
                # トレーニングデータでのパフォーマンス計算
                training_trades = optimizer.best_trades
                if len(training_trades) < self.min_trades:
                    logger.warning(f"トレーニングのトレード数が不足: {len(training_trades)} < {self.min_trades}")
                    test_start = test_end
                    test_end = test_start + pd.Timedelta(days=self.testing_days)
                    continue
                
                training_analytics = Analytics(training_trades, self.config['backtest']['initial_balance'])
                training_alpha = training_analytics.calculate_alpha_score()
                
                # テストデータでのバックテスト
                position_sizing = FixedRatioSizing({
                    'ratio': self.config['position_sizing']['params']['ratio'],
                    'min_position': None,
                    'max_position': None,
                    'leverage': 1
                })

                backtester = Backtester(
                    strategy=strategy,
                    position_sizing=position_sizing,
                    initial_balance=self.config['backtest']['initial_balance'],
                    commission=self.config['backtest']['commission'],
                    max_positions=self.config['backtest'].get('max_positions', 1)
                )
                
                # シンボルはデータ設定から取得
                symbol = self.config['data']['symbol']
                test_trades = backtester.run({symbol: testing_data})
                
                if len(test_trades) >= self.min_trades:
                    test_analytics = Analytics(test_trades, self.config['backtest']['initial_balance'])
                    test_alpha = test_analytics.calculate_alpha_score()
                    
                    # WFEの計算
                    wfe = test_alpha / training_alpha if training_alpha > 0 else 0.0
                    
                    results.append({
                        'training_start': training_start,
                        'training_end': training_end,
                        'testing_start': test_start,
                        'testing_end': test_end,
                        'best_params': best_params,
                        'training_alpha': training_alpha,
                        'test_alpha': test_alpha,
                        'wfe': wfe,
                        'trades': test_trades
                    })
                    
                    logger.info(f"トレーニングアルファ: {training_alpha:.2f}")
                    logger.info(f"テストアルファ: {test_alpha:.2f}")
                    logger.info(f"WFE: {wfe:.2f}")
                else:
                    logger.warning(f"テストのトレード数が不足: {len(test_trades)} < {self.min_trades}")
            
            except Exception as e:
                logger.error(f"期間の処理中にエラーが発生: {str(e)}")
            
            # 次の期間へ
            test_start = test_end
            test_end = test_start + pd.Timedelta(days=self.testing_days)

        if not results:
            logger.warning("有効な結果が得られませんでした")
        else:
            logger.info(f"合計 {len(results)} 期間の結果が得られました")

        return results

    def print_results(self, results: List[Dict[str, Any]]) -> None:
        """ウォークフォワードテストの結果を表示"""
        if not results:
            logger.warning("結果がありません")
            return

        print("\n=== ウォークフォワードテスト結果 ===")
        
        # 平均パフォーマンス指標の計算
        avg_training_alpha = np.mean([r['training_alpha'] for r in results])
        avg_test_alpha = np.mean([r['test_alpha'] for r in results])
        avg_wfe = np.mean([r['wfe'] for r in results])
        
        print(f"\n平均トレーニングアルファ: {avg_training_alpha:.2f}")
        print(f"平均テストアルファ: {avg_test_alpha:.2f}")
        print(f"平均WFE: {avg_wfe:.2f}")
        
        print("\n各期間の詳細:")
        for i, result in enumerate(results, 1):
            print(f"\n期間 {i}:")
            print(f"トレーニング期間: {result['training_start'].strftime('%Y-%m-%d')} → {result['training_end'].strftime('%Y-%m-%d')}")
            print(f"テスト期間: {result['testing_start'].strftime('%Y-%m-%d')} → {result['testing_end'].strftime('%Y-%m-%d')}")
            
            # テスト期間の取引結果から各指標を計算
            test_analytics = Analytics(result['trades'], self.config['backtest']['initial_balance'])
            net_profit = test_analytics.calculate_net_profit_loss()
            sortino_ratio = test_analytics.calculate_sortino_ratio()
            calmar_ratio = test_analytics.calculate_calmar_ratio()
            
            print(f"トレーニングアルファ: {result['training_alpha']:.2f}")
            print(f"テストアルファ: {result['test_alpha']:.2f}")
            print(f"WFE: {result['wfe']:.2f}")
            print(f"純損益: {net_profit:,.0f} USD")
            print(f"ソルティノレシオ: {sortino_ratio:.2f}")
            print(f"カルマーレシオ: {calmar_ratio:.2f}")
            print("最適パラメータ:", result['best_params'])
