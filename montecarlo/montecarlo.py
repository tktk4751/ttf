from typing import List, Dict, Optional
from backtesting.trade import Trade
from analytics.analytics import Analytics
from .interfaces import ITradeSimulator, IEquityCurveCalculator, IStatisticsCalculator, IResultVisualizer
from .implementations import (
    BootstrapTradeSimulator,
    SimpleEquityCurveCalculator,
    MonteCarloStatisticsCalculator,
    MatplotlibVisualizer
)

class MonteCarlo:
    """モンテカルロシミュレーションを実行するクラス"""
    
    def __init__(
        self,
        trades: List[Trade],
        initial_capital: float,
        num_simulations: int = 2000,
        confidence_level: float = 0.95,
        trade_simulator: Optional[ITradeSimulator] = None,
        equity_calculator: Optional[IEquityCurveCalculator] = None,
        statistics_calculator: Optional[IStatisticsCalculator] = None,
        result_visualizer: Optional[IResultVisualizer] = None
    ):
        """モンテカルロシミュレーションを初期化

        Args:
            trades: 元のトレードリスト
            initial_capital: 初期資金
            num_simulations: シミュレーション回数
            confidence_level: 信頼水準
            trade_simulator: トレードシミュレータ
            equity_calculator: エクイティカーブ計算機
            statistics_calculator: 統計計算機
            result_visualizer: 結果可視化器
        """
        self.trades = trades
        self.initial_capital = initial_capital
        self.num_simulations = num_simulations
        self.confidence_level = confidence_level
        
        # 依存性注入
        self.trade_simulator = trade_simulator or BootstrapTradeSimulator()
        self.equity_calculator = equity_calculator or SimpleEquityCurveCalculator()
        self.statistics_calculator = statistics_calculator or MonteCarloStatisticsCalculator()
        self.result_visualizer = result_visualizer or MatplotlibVisualizer()
        
        # 結果の保存用
        self.simulation_results: List[Dict] = []
        self.equity_curves: List[List[float]] = []

    def run(self) -> Dict:
        """モンテカルロシミュレーションを実行

        Returns:
            Dict: シミュレーション結果の統計
        """
        for _ in range(self.num_simulations):
            # トレードをシミュレート
            simulated_trades = self.trade_simulator.simulate_trades(
                self.trades,
                self.initial_capital
            )
            
            # 分析を実行
            analytics = Analytics(simulated_trades, self.initial_capital)
            result = {
                'total_return': analytics.calculate_total_return(),
                'max_drawdown': analytics.calculate_max_drawdown()[0],
                'sharpe_ratio': analytics.calculate_sharpe_ratio(),
                'sortino_ratio': analytics.calculate_sortino_ratio(),
                'win_rate': analytics.calculate_win_rate(),
                'profit_factor': analytics.calculate_profit_factor(),
                'alpha_score': analytics.calculate_alpha_score(),
                'final_capital': analytics.final_capital
            }
            self.simulation_results.append(result)
            
            # エクイティカーブを計算
            equity_curve = self.equity_calculator.calculate(
                simulated_trades,
                self.initial_capital
            )
            self.equity_curves.append(equity_curve)

        return self.statistics_calculator.calculate(
            self.simulation_results,
            self.initial_capital,
            self.confidence_level
        )

    def get_worst_case_scenario(self) -> Dict:
        """最悪のシナリオを取得"""
        return min(self.simulation_results, key=lambda x: x['total_return'])

    def get_best_case_scenario(self) -> Dict:
        """最良のシナリオを取得"""
        return max(self.simulation_results, key=lambda x: x['total_return'])

    def plot_equity_curves(self) -> None:
        """エクイティカーブをプロット"""
        self.result_visualizer.visualize(
            self.equity_curves,
            self.confidence_level
        )

    def print_simulation_results(self) -> None:
        """シミュレーション結果を出力"""
        if not self.simulation_results:
            print("シミュレーションが実行されていません。")
            return

        stats = self.statistics_calculator.calculate(
            self.simulation_results,
            self.initial_capital,
            self.confidence_level
        )
        
        print("\n=== モンテカルロシミュレーション結果 ===")
        print(f"シミュレーション回数: {self.num_simulations}")
        print(f"信頼水準: {self.confidence_level * 100}%")
        
        print("\n--- リターン統計 ---")
        print(f"平均リターン: {stats['mean_return']:.2f}%")
        print(f"中央値リターン: {stats['median_return']:.2f}%")
        print(f"標準偏差: {stats['std_return']:.2f}%")
        print(f"最小リターン: {stats['min_return']:.2f}%")
        print(f"最大リターン: {stats['max_return']:.2f}%")
        
        ci = stats[f'confidence_interval_{int(self.confidence_level*100)}']
        print(f"\n{int(self.confidence_level*100)}%信頼区間:")
        print(f"下限: {ci['lower']:.2f}%")
        print(f"上限: {ci['upper']:.2f}%")
        
        print("\n--- 各指標の統計 ---")
        metrics_stats = stats['metrics_stats']
        for metric, metric_stats in metrics_stats.items():
            print(f"\n{metric}:")
            print(f"平均: {metric_stats['mean']:.2f}")
            print(f"中央値: {metric_stats['median']:.2f}")
            print(f"標準偏差: {metric_stats['std']:.2f}")
            print(f"最小値: {metric_stats['min']:.2f}")
            print(f"最大値: {metric_stats['max']:.2f}")
            print(f"{int(self.confidence_level*100)}%信頼区間:")
            print(f"下限: {metric_stats[f'percentile_{int((1-self.confidence_level)*100)}']:.2f}")
            print(f"上限: {metric_stats[f'percentile_{int(self.confidence_level*100)}']:.2f}")
        
        print("\n--- シナリオ分析 ---")
        worst_case = self.get_worst_case_scenario()
        best_case = self.get_best_case_scenario()
        
        print("\n最悪のシナリオ:")
        print(f"総リターン: {worst_case['total_return']:.2f}%")
        print(f"最大ドローダウン: {worst_case['max_drawdown']:.2f}%")
        print(f"シャープレシオ: {worst_case['sharpe_ratio']:.2f}")
        print(f"勝率: {worst_case['win_rate']:.2f}%")
        
        print("\n最良のシナリオ:")
        print(f"総リターン: {best_case['total_return']:.2f}%")
        print(f"最大ドローダウン: {best_case['max_drawdown']:.2f}%")
        print(f"シャープレシオ: {best_case['sharpe_ratio']:.2f}")
        print(f"勝率: {best_case['win_rate']:.2f}%")
        
        # エクイティカーブのグラフを表示
        self.plot_equity_curves()
