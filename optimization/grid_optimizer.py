import numpy as np
from numba import jit
from typing import Dict, Any, Callable, Tuple, Type, List
import pandas as pd
from datetime import datetime

from optimization.optimizer import BaseOptimizer
from backtesting.backtester import Backtester
from position_sizing.fixed_ratio import FixedRatioSizing
from strategies.base.strategy import BaseStrategy
from analytics.analytics import Analytics
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor

class GridSearchOptimizer(BaseOptimizer):
    """グリッドサーチ最適化を行うクラス"""

    def __init__(
        self,
        strategy_class: Type[BaseStrategy],
        param_generator: Dict[str, list],
        config: Dict[str, Any],
    ):
        """
        Args:
            strategy_class: 最適化対象の戦略クラス
            param_generator: 戦略パラメーターのグリッド
            config: 設定
        """
        super().__init__(strategy_class, param_generator, config)
        self.param_grid = param_generator
        self.data = None
        self._data_dict = None
        self.best_score = None
        self.best_params = None
        self.best_trades = None

    def _load_and_process_data(self) -> None:
        """データの読み込みと前処理"""
        data_config = self.config.get("data", {})
        data_dir = data_config.get("data_dir", "data")
        symbol = data_config.get("symbol", "BTCUSDT")
        timeframe = data_config.get("timeframe", "1h")
        start_date = data_config.get("start")
        end_date = data_config.get("end")

        # 日付文字列をdatetimeオブジェクトに変換
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None

        # データの読み込みと処理
        data_loader = DataLoader(CSVDataSource(data_dir))
        data_processor = DataProcessor()

        raw_data = data_loader.load_data_from_config(self.config)
        processed_data = {
            symbol: data_processor.process(df) for symbol, df in raw_data.items()
        }

        self.data = processed_data[symbol]
        self._data_dict = processed_data

    @staticmethod
    @jit(nopython=True)
    def _calculate_score(trades: np.ndarray, initial_balance: float) -> float:
        """Numbaで高速化したスコア計算関数"""
        if len(trades) < 30:  # 最小トレード数の閾値
            return -np.inf

        # 累積利益、最大ドローダウン、勝ちトレード数の計算
        balance = initial_balance
        max_balance = initial_balance
        max_drawdown = 0
        win_trades = 0
        for trade in trades:
            balance += trade
            max_balance = max(max_balance, balance)
            drawdown = 1 - balance / max_balance
            max_drawdown = max(max_drawdown, drawdown)
            if trade > 0:
                win_trades += 1
        
        if max_drawdown == 0:
            return -np.inf

        # Win%の計算
        if len(trades) > 0:
            win_rate = win_trades / len(trades)
        else:
            win_rate = 0.0

        # 最終損益の計算
        final_pnl = balance - initial_balance

        # Calmarレシオの計算
        calmar_ratio = final_pnl / max_drawdown

        # スコアの計算
        score = win_rate * calmar_ratio

        return score

    def _create_strategy(self, params: Dict[str, Any]) -> BaseStrategy:
        """パラメータから戦略インスタンスを生成"""
        strategy_params = self.strategy_class.convert_params_to_strategy_format(params)
        return self.strategy_class(**strategy_params)

    def _run_backtest(self, strategy: BaseStrategy) -> list[float]:
        """バックテストを実行"""
        # ポジションサイジングの作成
        position_config = self.config.get("position_sizing", {})
        position_sizing = FixedRatioSizing(
            ratio=position_config.get("ratio", 0.5),
            leverage=position_config.get("leverage", 1.0),
        )

        # バックテスターの作成
        initial_balance = self.config.get("position_sizing", {}).get(
            "initial_balance", 10000
        )
        commission_rate = self.config.get("position_sizing", {}).get(
            "commission_rate", 0.001
        )
        backtester = Backtester(
            strategy=strategy,
            position_manager=position_sizing,
            initial_balance=initial_balance,
            commission=commission_rate,
            verbose=False,
        )

        return backtester.run(self._data_dict)

    def _generate_param_combinations(self) -> Tuple[np.ndarray, List[str]]:
        """パラメータの組み合わせを生成"""
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        # パラメータの組み合わせを生成
        param_combinations = np.array(np.meshgrid(*values)).T.reshape(-1, len(keys))
        return param_combinations, keys

    def optimize(self) -> Tuple[Dict[str, Any], float]:
        """最適化を実行し、最適なパラメータとスコアを返す"""
        if self.data is None:
            self._load_and_process_data()

        param_combinations, keys = self._generate_param_combinations()

        best_score = -np.inf
        best_params = None
        best_trades = None

        initial_balance = self.config.get("position_sizing", {}).get(
            "initial_balance", 10000
        )
        
        # パラメータの組み合わせごとにバックテストを実行
        for params in param_combinations:
            param_dict = dict(zip(keys, params))
            strategy = self._create_strategy(param_dict)
            trades = self._run_backtest(strategy)

            # 損益をNumpy配列に変換
            trades_pnl = np.array([trade.profit_loss for trade in trades], dtype=np.float64)

            score = self._calculate_score(trades_pnl, initial_balance)

            if score > best_score:
                best_score = score
                best_params = param_dict
                best_trades = trades

        # 最適なパラメータでバックテストを実行し、結果を表示
        best_strategy = self._create_strategy(best_params)
        best_trades = self._run_backtest(best_strategy)
        analytics = Analytics(best_trades, initial_balance)

        print("\n=== 最適化結果 ===")
        print(f"最適なパラメータ: {best_params}")
        print(f"最適なスコア: {best_score}")
        print("\n=== バックテスト結果 ===")
        print(analytics.print_backtest_results())

        return best_params, best_score