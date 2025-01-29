from typing import Dict, Any, Optional, List, Tuple, Type, Callable
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import pandas as pd
from datetime import datetime

from optimization.optimizer import BaseOptimizer
from backtesting.backtester import Backtester
from position_sizing.fixed_ratio import FixedRatioSizing
from strategies.base.strategy import BaseStrategy
from analytics.analytics import Analytics
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor

class BayesianOptimizer(BaseOptimizer):
    """Bayesian最適化を行うクラス"""
    
    def __init__(
        self,
        strategy_class: Type[BaseStrategy],
        param_generator: Callable[[optuna.Trial], Dict[str, Any]],
        config: Dict[str, Any],
        n_trials: int ,
        n_jobs: int = -1,
        timeout: Optional[int] = None
    ):
        """
        Args:
            strategy_class: 最適化対象の戦略クラス
            param_generator: 戦略パラメーターを生成する関数
            config: 設定
            n_trials: 最適化の試行回数
            n_jobs: 並列処理数（-1で全CPU使用）
            timeout: タイムアウト時間（秒）
        """
        super().__init__(strategy_class, param_generator, config)
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.timeout = timeout
    
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
        data_loader = DataLoader(CSVDataSource(data_dir))
        data_processor = DataProcessor()
        
        raw_data = data_loader.load_data_from_config(self.config)
        processed_data = {
            symbol: data_processor.process(df)
            for symbol, df in raw_data.items()
        }
        
        self.data = processed_data[symbol]
        self._data_dict = processed_data
    
    def _create_strategy(self, trial: optuna.Trial) -> BaseStrategy:
        """Optunaのtrialから戦略インスタンスを生成"""
        params = self.param_generator(trial)
        strategy_params = self.strategy_class.convert_params_to_strategy_format(params)
        return self.strategy_class(**strategy_params)
    
    def _run_backtest(self, strategy: BaseStrategy) -> List[Any]:
        """バックテストを実行"""
        # ポジションサイジングの作成
        position_config = self.config.get('position_sizing', {})
        position_sizing = FixedRatioSizing(
            ratio=position_config.get('ratio', 0.5),
            leverage=position_config.get('leverage', 1.0)
        )
        
        # バックテスターの作成
        initial_balance = self.config.get('position_sizing', {}).get('initial_balance', 10000)
        commission_rate = self.config.get('position_sizing', {}).get('commission_rate', 0.001)
        backtester = Backtester(
            strategy=strategy,
            position_manager=position_sizing,
            initial_balance=initial_balance,
            commission=commission_rate,
            verbose=False
        )
        
        return backtester.run(self._data_dict)
    
    def _objective(self, trial: optuna.Trial) -> float:
        """最適化の目的関数"""
        strategy = self._create_strategy(trial)
        trades = self._run_backtest(strategy)
        
        if len(trades) < 30:  # 最小トレード数の閾値
            raise optuna.TrialPruned()
        
        analytics = Analytics(trades, self.config.get('position_sizing', {}).get('initial_balance', 10000))
        score = analytics.calculate_win_calmar_score()
        
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best_params = trial.params
            self.best_trades = trades
        
        return score
    
    def optimize(self) -> Tuple[Dict[str, Any], float]:
        """最適化を実行し、最適なパラメータとスコアを返す"""
        if self.data is None:
            self._load_and_process_data()
        
        study = optuna.create_study(
            study_name='strategy_optimization',
            direction='maximize',
            sampler=TPESampler(n_startup_trials=10),
            pruner=MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1
            )
        )
        
        study.optimize(
            self._objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=True
        )

        # 最適なパラメータでバックテストを実行
        best_params = study.best_params
        strategy_params = self.strategy_class.convert_params_to_strategy_format(best_params)
        best_strategy = self.strategy_class(**strategy_params)
        best_trades = self._run_backtest(best_strategy)
        analytics = Analytics(best_trades, self.config.get('position', {}).get('initial_balance', 10000))
        
        print("\n=== 最適化結果 ===")
        print(f"最適なパラメータ: {best_params}")
        print(f"最適なスコア: {study.best_value}")
        print("\n=== バックテスト結果 ===")
        print(analytics.print_backtest_results())
        
        return best_params, study.best_value 