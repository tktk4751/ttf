#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Optional, List, Tuple, Type, Callable, Set
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import pandas as pd
from datetime import datetime
import importlib
import os
from pathlib import Path
import inspect

from optimization.optimizer import BaseOptimizer
from backtesting.backtester import Backtester
from position_sizing.fixed_ratio import FixedRatioSizing
from strategies.base.strategy import BaseStrategy
from analytics.analytics import Analytics
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from signals.base_signal import BaseSignal
from strategies.implementations.signal_combination.strategy import SignalCombinationStrategy

class SignalCombinationOptimizer(BaseOptimizer):
    """シグナルの組み合わせを最適化するクラス"""
    
    def __init__(
        self,
        config: Dict[str, Any],
        n_trials: int = 100,
        max_signals: int = 3,
        metric_function: Optional[Callable[[Analytics], float]] = None,
        n_jobs: int = 1
    ):
        """
        Args:
            config: 設定
            n_trials: 最適化の試行回数
            max_signals: 組み合わせる最大シグナル数
            metric_function: 評価指標を計算する関数（デフォルトはアルファスコア）
            n_jobs: 並列処理数（-1で全CPU使用）
        """
        super().__init__(None, None, config)
        self.n_trials = n_trials
        self.max_signals = max_signals
        self.metric_function = metric_function or (lambda x: x.calculate_alpha_score())
        self.n_jobs = n_jobs
        
        # 利用可能なシグナルクラスを動的に読み込む
        self.available_signals = self._discover_signals()
        
        # データの読み込みと処理
        self._data = None
        self._data_dict = None
        self._load_and_process_data()
    
    def _load_and_process_data(self) -> None:
        """データの読み込みと前処理"""
        # データの読み込みと処理
        data_dir = self.config['data']['data_dir']
        data_loader = DataLoader(CSVDataSource(data_dir))
        data_processor = DataProcessor()
        
        raw_data = data_loader.load_data_from_config(self.config)
        processed_data = {
            symbol: data_processor.process(df)
            for symbol, df in raw_data.items()
        }
        
        self._data_dict = processed_data
        
        # 設定ファイルで指定された銘柄のデータを取得
        symbol = self.config['data'].get('symbol', list(processed_data.keys())[0])
        self._data = processed_data[symbol]
    
    def _discover_signals(self) -> Dict[str, Type[BaseSignal]]:
        """signals/implementationsディレクトリから利用可能なシグナルクラスを動的に読み込む"""
        signals_dir = Path("signals/implementations")
        signal_classes = {}
        
        # 各サブディレクトリを探索
        for signal_dir in signals_dir.glob("*"):
            if not signal_dir.is_dir() or signal_dir.name.startswith("_"):
                continue
                
            # モジュールをインポート
            for py_file in signal_dir.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue
                    
                module_path = f"signals.implementations.{signal_dir.name}.{py_file.stem}"
                try:
                    module = importlib.import_module(module_path)
                    
                    # BaseSignalを継承したクラスを探す
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, BaseSignal) and 
                            obj != BaseSignal):
                            signal_classes[f"{signal_dir.name}.{name}"] = obj
                            
                except ImportError as e:
                    print(f"Warning: Failed to import {module_path}: {e}")
        
        return signal_classes
    
    def _create_signal_combination_strategy(
        self,
        selected_signals: List[str]
    ) -> BaseStrategy:
        """選択されたシグナルの組み合わせから戦略を生成"""
        signal_instances = []
        for signal_name in selected_signals:
            signal_class = self.available_signals[signal_name]
            signal_instances.append(signal_class())
        
        return SignalCombinationStrategy(signals=signal_instances)
    
    def _objective(self, trial: optuna.Trial) -> float:
        """最適化の目的関数"""
        # シグナルの数を選択（1からmax_signals）
        n_signals = trial.suggest_int("n_signals", 1, self.max_signals)
        
        # 利用可能なシグナルから指定した数だけランダムに選択
        signal_names = list(self.available_signals.keys())
        selected_signals = []
        for i in range(n_signals):
            # 既に選択されていないシグナルから選択
            remaining_signals = [s for s in signal_names if s not in selected_signals]
            if not remaining_signals:
                break
            signal_idx = trial.suggest_int(f"signal_{i}", 0, len(remaining_signals)-1)
            selected_signals.append(remaining_signals[signal_idx])
        
        # 選択されたシグナルで戦略を生成
        strategy = self._create_signal_combination_strategy(selected_signals)
        
        # ポジションサイジングの作成
        position_config = self.config.get('position', {})
        position_sizing = FixedRatioSizing(
            ratio=position_config.get('ratio', 0.99),
            leverage=position_config.get('leverage', 1.0)
        )
        
        # バックテスターの作成
        initial_balance = self.config.get('position', {}).get('initial_balance', 10000)
        commission_rate = self.config.get('position', {}).get('commission_rate', 0.001)
        backtester = Backtester(
            strategy=strategy,
            position_manager=position_sizing,
            initial_balance=initial_balance,
            commission=commission_rate,
            verbose=False
        )
        
        # バックテストを実行
        trades = backtester.run(self._data_dict)
        
        # 最小トレード数のチェック
        if len(trades) < 30:  # 最小トレード数の閾値
            raise optuna.TrialPruned()
        
        # 評価指標を計算
        analytics = Analytics(trades, initial_balance)
        score = self.metric_function(analytics)
        
        # 選択されたシグナルを記録
        trial.set_user_attr("selected_signals", selected_signals)
        
        return score
    
    def optimize(self) -> Tuple[Dict[str, List[str]], float]:
        """最適化を実行し、最適なシグナルの組み合わせとスコアを返す"""
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # 並列処理の設定
        if self.n_jobs != 1:
            study.optimize(
                self._objective,
                n_trials=self.n_trials,
                n_jobs=self.n_jobs
            )
        else:
            study.optimize(self._objective, n_trials=self.n_trials)
        
        # 最適なパラメータとスコアを取得
        best_params = {
            "selected_signals": study.best_trial.user_attrs["selected_signals"]
        }
        best_score = study.best_value
        
        return best_params, best_score 