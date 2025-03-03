#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Protocol, Dict, List, Any, Tuple
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from optimization.bayesian_optimizer import BayesianOptimizer
from analytics.analytics import Analytics
from backtesting.backtester import Backtester
from position_sizing.fixed_ratio import FixedRatioSizing

class IDataSplitter(Protocol):
    """データ分割のインターフェース"""
    def split(self, data: Dict[str, pd.DataFrame], start_date: datetime, end_date: datetime) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """データを学習期間とテスト期間に分割"""
        ...

class IWalkForwardResult(Protocol):
    """ウォークフォワードテスト結果のインターフェース"""
    @property
    def trades(self) -> List[Any]:
        """全期間のトレード結果"""
        ...
    
    @property
    def period_results(self) -> List[Dict[str, Any]]:
        """各期間の結果"""
        ...

class WalkForwardResult:
    """ウォークフォワードテスト結果を保持するクラス"""
    def __init__(self):
        self._trades = []
        self._period_results = []
    
    @property
    def trades(self) -> List[Any]:
        return self._trades
    
    @property
    def period_results(self) -> List[Dict[str, Any]]:
        return self._period_results
    
    def add_period_result(self, result: Dict[str, Any]):
        """期間ごとの結果を追加"""
        self._period_results.append(result)
    
    def add_trades(self, trades: List[Any]):
        """トレード結果を追加"""
        self._trades.extend(trades)

class TimeSeriesDataSplitter(IDataSplitter):
    """時系列データを分割するクラス"""
    def __init__(self, training_days: int, testing_days: int):
        """
        Args:
            training_days: 学習期間の日数
            testing_days: テスト期間の日数
        """
        self.training_days = training_days
        self.testing_days = testing_days
    
    def split(self, data: Dict[str, pd.DataFrame], start_date: datetime, end_date: datetime) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """データを学習期間とテスト期間に分割
        
        Args:
            data: 分割対象のデータ
            start_date: 開始日
            end_date: 終了日
            
        Returns:
            Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]: 
            (学習データ, テストデータ)のタプル
        """
        # 学習期間とテスト期間の日付範囲を計算
        training_end = start_date + timedelta(days=self.training_days)
        testing_end = min(training_end + timedelta(days=self.testing_days), end_date)
        
        training_data = {}
        testing_data = {}
        
        for symbol, df in data.items():
            # データの日付範囲をチェック
            if df.index[-1] < start_date or df.index[0] > testing_end:
                continue
                
            # 学習データの抽出
            train_mask = (df.index >= start_date) & (df.index < training_end)
            training_data[symbol] = df[train_mask].copy()
            
            # テストデータの抽出
            test_mask = (df.index >= training_end) & (df.index < testing_end)
            testing_data[symbol] = df[test_mask].copy()
        
        return training_data, testing_data

class WalkForwardOptimizer:
    """ウォークフォワードテストを実行するクラス"""
    def __init__(
        self,
        optimizer: BayesianOptimizer,
        data_splitter: IDataSplitter,
        config: Dict[str, Any],
    ):
        """
        Args:
            optimizer: 最適化器
            data_splitter: データ分割器
            config: 設定
        """
        self.optimizer = optimizer
        self.data_splitter = data_splitter
        self.config = config
        self.result = WalkForwardResult()

    def _create_strategy(self, params: Dict[str, Any]) -> Any:
        """戦略インスタンスを作成

        Args:
            params: 最適化されたパラメータ

        Returns:
            戦略インスタンス
        """
        strategy_params = self.optimizer.strategy_class.convert_params_to_strategy_format(params)
        # strategy_params = {'params': self.optimizer.strategy_class.convert_params_to_strategy_format(params)}

        return self.optimizer.strategy_class(**strategy_params)

    def run(self, data: Dict[str, pd.DataFrame]) -> IWalkForwardResult:
        """ウォークフォワードテストを実行
        
        Args:
            data: バックテストに使用するデータ
        
        Returns:
            ウォークフォワードテストの結果
        """
        # 設定の取得
        walkforward_config = self.config.get('walkforward', {})
        start_date = pd.Timestamp(self.config['data']['start'])
        end_date = pd.Timestamp(self.config['data']['end'])
        min_trades = walkforward_config.get('min_trades', 15)
        
        # データの日付範囲をチェック
        valid_data = {}
        earliest_start = None
        latest_end = None
        
        for symbol, df in data.items():
            if df.empty:
                print(f"Warning: Empty data for {symbol}")
                continue
                
            df_start = df.index[0]
            df_end = df.index[-1]
            
            if earliest_start is None or df_start < earliest_start:
                earliest_start = df_start
            if latest_end is None or df_end > latest_end:
                latest_end = df_end
            
            if df_end < start_date:
                print(f"Warning: Data for {symbol} ends before start date")
                continue
            if df_start > end_date:
                print(f"Warning: Data for {symbol} starts after end date")
                continue
                
            valid_data[symbol] = df

        if not valid_data:
            print("Error: No valid data available for walk-forward test")
            return self.result
        
        # 有効な日付範囲を調整
        start_date = max(start_date, earliest_start)
        end_date = min(end_date, latest_end)
        
        print(f"\nUsing data range: {start_date} to {end_date}")
        
        # 期間ごとの結果を保存するリスト
        period_summaries = []

        current_date = start_date
        period_count = 0
        while current_date + timedelta(days=self.data_splitter.training_days) < end_date:
            try:
                # データの分割
                training_data, testing_data = self.data_splitter.split(valid_data, current_date, end_date)
                
                # データが十分にあるかチェック
                if not training_data or not testing_data:
                    print(f"Warning: Insufficient data for period starting {current_date}")
                    current_date += timedelta(days=self.data_splitter.testing_days)
                    continue
                
                # 学習データで最適化を実行
                original_data_config = self.config['data'].copy()
                self.config['data'].update({
                    'start': current_date.strftime('%Y-%m-%d'),
                    'end': (current_date + timedelta(days=self.data_splitter.training_days)).strftime('%Y-%m-%d')
                })
                
                # 最適化の実行
                best_params, best_score = self.optimizer.optimize()
                
                # configを元に戻す
                self.config['data'].update(original_data_config)
                
                # テストデータでバックテスト
                strategy = self._create_strategy(params=best_params)
                
                # ポジションサイジングの設定
                position_config = self.config.get('position_sizing', {})
                position_sizing = FixedRatioSizing(
                    ratio=position_config.get('ratio', 0.2),
                    leverage=position_config.get('leverage', 1.0)
                )
                
                # バックテスターの作成と実行
                initial_balance = self.config.get('position_sizing', {}).get('initial_balance', 10000)
                commission_rate = self.config.get('position_sizing', {}).get('commission_rate', 0.001)
                backtester = Backtester(
                    strategy=strategy,
                    position_manager=position_sizing,
                    initial_balance=initial_balance,
                    commission=commission_rate,
                    verbose=False
                )
                
                trades = backtester.run(testing_data)
                
                # 結果の保存
                if len(trades) >= min_trades:
                    period_count += 1
                    period_analytics = Analytics(trades, initial_balance)
                    period_wfe = period_analytics.calculate_alpha_score() / best_score if best_score != 0 else 0
                    
                    self.result.add_trades(trades)
                    self.result.add_period_result({
                        'start_date': current_date,
                        'end_date': current_date + timedelta(days=self.data_splitter.testing_days),
                        'parameters': best_params,
                        'training_score': best_score,
                        'trades': trades
                    })
                    
                    # 期間ごとの結果を保存
                    period_summaries.append({
                        'period': period_count,
                        'start_date': current_date,
                        'end_date': current_date + timedelta(days=self.data_splitter.testing_days),
                        'training_score': best_score,
                        'test_alpha_score': period_analytics.calculate_alpha_score(),
                        'wfe': period_wfe,
                        'trades': len(trades),
                        'return': period_analytics.calculate_total_return()
                    })
                else:
                    print(f"Warning: Insufficient trades ({len(trades)}) for period starting {current_date}")

            except Exception as e:
                print(f"Error processing period starting {current_date}: {str(e)}")
            
            # 次の期間へ
            current_date += timedelta(days=self.data_splitter.testing_days)
        
        if not self.result.period_results:
            print("Warning: No valid results were generated during the walk-forward test")
        else:
            print(f"\n=== Walk-Forward Test Summary ===")
            print(f"Total number of walk-forward periods: {period_count}")
            print("\n各期間の結果:")
            print("-" * 100)
            print(f"{'Period':^8} | {'Start Date':^12} | {'End Date':^12} | {'Training Score':^14} | {'Test Alpha':^10} | {'WFE':^8} | {'Trades':^8} | {'Return %':^8}")
            print("-" * 100)
            
            for summary in period_summaries:
                print(f"{summary['period']:^8} | {summary['start_date'].strftime('%Y-%m-%d'):^12} | {summary['end_date'].strftime('%Y-%m-%d'):^12} | {summary['training_score']:^14.4f} | {summary['test_alpha_score']:^10.4f} | {summary['wfe']:^8.4f} | {summary['trades']:^8} | {summary['return']:^8.2f}")
            
            print("-" * 100)
            
            # 全期間の結果を分析
            initial_balance = self.config.get('position', {}).get('initial_balance', 10000)
            analytics = Analytics(self.result.trades, initial_balance)
            
            # 各期間のWFEを計算
            wfe_values = []
            for period in self.result.period_results:
                in_sample_score = period['training_score']
                period_trades = period['trades']
                if period_trades and in_sample_score != 0:
                    period_analytics = Analytics(period_trades, initial_balance)
                    out_sample_score = period_analytics.calculate_alpha_score()
                    wfe = out_sample_score / in_sample_score
                    wfe_values.append(wfe)

            # 平均WFEを計算
            avg_wfe = np.mean(wfe_values) if wfe_values else 0
            
            # 最終結果の表示
            print("\n=== Walk-Forward Test Final Results ===")
            print(f"Total number of periods: {period_count}")
            print(f"Total trades: {len(self.result.trades)}")
            print(f"Average Walk-Forward Efficiency (WFE): {avg_wfe:.4f}")
            print(f"Total Return: {analytics.calculate_total_return():.2f}%")
            print(f"Calmar Ratio (Adjusted): {analytics.calculate_calmar_ratio():.4f}")
            print(f"Sortino Ratio: {analytics.calculate_sortino_ratio():.4f}")
            print(f"CAGR: {analytics.calculate_cagr():.2f}%")
            print("=" * 40)
        
        return self.result 