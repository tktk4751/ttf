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
import pandas as pd

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
from logger import get_logger

class BayesianOptimizer:
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
        self.data = None
        self.data_dict = None
        
        # 設定ファイルの読み込み
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # ロガーの初期化
        self.logger = get_logger(__name__)

    @property
    def data(self) -> pd.DataFrame:
        """データを取得"""
        return self._data

    @data.setter
    def data(self, value: pd.DataFrame) -> None:
        """データを設定し、data_dictも更新"""
        self._data = value
        if value is not None:
            symbol = self.config['data']['symbol']
            self.data_dict = {symbol: value}
        else:
            self.data_dict = None

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

    @staticmethod
    def _run_backtest_static(
        strategy: Strategy,
        data_dict: Dict[str, Any],
        config: Dict[str, Any]
    ) -> List[Any]:
        """バックテストを実行する関数（並列処理対応のため静的メソッドに変更）"""
        position_sizing = FixedRatioSizing({
            'ratio': config['position_sizing']['params']['ratio'],
            'min_position': None,
            'max_position': None,
            'leverage': 1
        })

        backtester = Backtester(
            strategy=strategy,
            position_sizing=position_sizing,
            initial_balance=config['backtest']['initial_balance'],
            commission=config['backtest']['commission'],
            max_positions=config['backtest']['max_positions']
        )

        trades = backtester.run(data_dict)
        return trades

    def _objective(self, trial: optuna.Trial) -> float:
        """最適化の目的関数"""
        strategy = self._create_strategy(trial)
        
        # バックテストの実行を _run_backtest として切り出し
        trades = BayesianOptimizer._run_backtest_static(strategy, self.data_dict, self.config)

        if len(trades) < 30:
            raise optuna.TrialPruned()

        analytics = Analytics(trades, self.config['backtest']['initial_balance'])
        alpha_score = analytics.calculate_alpha_score()

        if self.best_score is None or alpha_score > self.best_score:
            self.best_score = alpha_score
            self.best_params = trial.params
            self.best_trades = trades

        return alpha_score

    def optimize(self):
        """最適化を実行し、最適なパラメータを返す"""
        if self.data is None:
            self._load_and_process_data()
            
        study = optuna.create_study(
            study_name='alpha_score_optimization',
            direction='maximize'
        )
        
        study.optimize(
            self._objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            show_progress_bar=True
        )
        
        # 最適なパラメータと最高スコアを取得
        best_params = study.best_params
        best_value = study.best_value
        
        self.logger.info(f"最適化完了 - 最高スコア: {best_value:.2f}")
        self.logger.info(f"最適パラメータ: {best_params}")
        
        return best_params, best_value  # タプルとして返す