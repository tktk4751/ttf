import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict
from backtesting.trade import Trade
import copy
from .interfaces import ITradeSimulator, IEquityCurveCalculator, IStatisticsCalculator, IResultVisualizer
from analytics.analytics import Analytics

class BootstrapTradeSimulator(ITradeSimulator):
    """ブートストラップ法を使用したトレードシミュレータ"""
    def simulate_trades(self, trades: List[Trade], initial_capital: float) -> List[Trade]:
        simulated_trades = []
        current_balance = initial_capital
        
        # 元のトレード数と同じ数のトレードをランダムに選択
        n_trades = len(trades)
        selected_indices = np.random.choice(n_trades, size=n_trades, replace=True)
        
        for idx in selected_indices:
            original_trade = trades[idx]
            new_trade = copy.deepcopy(original_trade)
            
            # 損益にランダム性を加える
            returns = [t.profit_loss / t.position_size for t in trades]
            volatility = np.std(returns)
            original_return = original_trade.profit_loss / original_trade.position_size
            random_return = np.random.normal(original_return, volatility * 0.4)
            
            # 新しい損益を計算
            new_trade.profit_loss = random_return * original_trade.position_size
            new_trade.balance = current_balance + new_trade.profit_loss
            current_balance = new_trade.balance
            
            # 日付をランダムにシフト
            time_shift = np.random.randint(-5, 6)
            new_trade.entry_date += pd.Timedelta(days=time_shift)
            new_trade.exit_date += pd.Timedelta(days=time_shift)
            
            simulated_trades.append(new_trade)
        
        # 日付順にソート
        simulated_trades.sort(key=lambda x: x.entry_date)
        return simulated_trades

class SimpleEquityCurveCalculator(IEquityCurveCalculator):
    """シンプルなエクイティカーブ計算機"""
    def calculate(self, trades: List[Trade], initial_capital: float) -> List[float]:
        equity = [initial_capital]
        current_balance = initial_capital
        
        for trade in trades:
            current_balance += trade.profit_loss
            equity.append(current_balance)
        
        return equity

class MonteCarloStatisticsCalculator(IStatisticsCalculator):
    """モンテカルロシミュレーションの統計計算機"""
    def calculate(self, results: List[Dict], initial_capital: float, confidence_level: float) -> Dict:
        """シミュレーション結果の統計を計算

        Args:
            results: シミュレーション結果のリスト
            initial_capital: 初期資金
            confidence_level: 信頼水準

        Returns:
            Dict: 統計結果
        """
        # 各シミュレーションの分析結果を取得
        analytics_list = [Analytics(r['trades'], initial_capital) for r in results]
        
        # 計算する指標のリスト
        metrics = {
            'total_return': lambda x: x.calculate_total_return(),
            'max_drawdown': lambda x: x.calculate_max_drawdown()[0],
            'sharpe_ratio': lambda x: x.calculate_sharpe_ratio(),
            'sortino_ratio': lambda x: x.calculate_sortino_ratio(),
            'win_rate': lambda x: x.calculate_win_rate(),
            'profit_factor': lambda x: x.calculate_profit_factor(),
            'alpha_score': lambda x: x.calculate_alpha_score(),
            'alpha_score_v2': lambda x: x.calculate_alpha_score_v2(),
            'calmar_ratio': lambda x: x.calculate_calmar_ratio(),
            'pessimistic_return_ratio': lambda x: x.calculate_pessimistic_return_ratio(),
            'gpr': lambda x: x.calculate_gpr(),
            'sqn': lambda x: x.calculate_sqn()
        }
        
        # 各指標の統計を計算
        metrics_stats = {}
        for metric_name, metric_func in metrics.items():
            values = [metric_func(analytics) for analytics in analytics_list]
            metrics_stats[metric_name] = {
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                f'percentile_{int(confidence_level*100)}': np.percentile(values, confidence_level * 100),
                f'percentile_{int((1-confidence_level)*100)}': np.percentile(values, (1-confidence_level) * 100)
            }
        
        # リターンの統計を計算
        returns = [analytics.calculate_total_return() for analytics in analytics_list]
        confidence_lower = np.percentile(returns, (1 - confidence_level) * 100)
        confidence_upper = np.percentile(returns, confidence_level * 100)
        
        return {
            'mean_return': np.mean(returns),
            'median_return': np.median(returns),
            'std_return': np.std(returns),
            'min_return': np.min(returns),
            'max_return': np.max(returns),
            f'confidence_interval_{int(confidence_level*100)}': {
                'lower': confidence_lower,
                'upper': confidence_upper
            },
            'metrics_stats': metrics_stats
        }

class MatplotlibVisualizer(IResultVisualizer):
    """Matplotlibを使用した結果の可視化"""
    def visualize(self, equity_curves: List[List[float]], confidence_level: float, save_path: str = None) -> None:
        plt.figure(figsize=(15, 8))
        plt.style.use('seaborn-v0_8-darkgrid')
    
        
        # 全シミュレーションのプロット
        for equity_curve in equity_curves:
            plt.plot(equity_curve, alpha=0.1, linewidth=0.5, color='blue')
        
        # 平均エクイティカーブ
        avg_equity = np.mean(equity_curves, axis=0)
        plt.plot(avg_equity, color='red', linewidth=2, label='Average')
        
        # 信頼区間
        lower_percentile = (1 - confidence_level) * 100
        upper_percentile = confidence_level * 100
        lower_bound = np.percentile(equity_curves, lower_percentile, axis=0)
        upper_bound = np.percentile(equity_curves, upper_percentile, axis=0)
        
        plt.fill_between(
            range(len(avg_equity)),
            lower_bound,
            upper_bound,
            alpha=0.2,
            color='red',
            label=f'{int(confidence_level*100)}%Confidence interval'
        )
        
        plt.title('Monte Carlo simulation: equity curve', fontsize=14)
        plt.xlabel('Number of trades', fontsize=12)
        plt.ylabel('Account Balance', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        plt.show()
        plt.close() 