import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from backtesting.trade import Trade
from analytics.analytics import Analytics
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class MonteCarlo:
    def __init__(
        self,
        trades: List[Trade],
        initial_capital: float,
        num_simulations: int = 1000,
        confidence_level: float = 0.95
    ):
        """モンテカルロシミュレーションを初期化

        Args:
            trades: 元のトレードリスト
            initial_capital: 初期資金
            num_simulations: シミュレーション回数
            confidence_level: 信頼水準
        """
        self.trades = trades
        self.initial_capital = initial_capital
        self.num_simulations = num_simulations
        self.confidence_level = confidence_level
        self.simulation_results: List[Dict] = []
        self.equity_curves: List[List[float]] = []

    def run(self) -> Dict:
        """モンテカルロシミュレーションを実行

        Returns:
            Dict: シミュレーション結果の統計
        """
        for _ in range(self.num_simulations):
            # トレードリストをコピーしてランダム性を加える
            simulated_trades = self._generate_random_trades()
            
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
            equity_curve = self._calculate_equity_curve(simulated_trades)
            self.equity_curves.append(equity_curve)

        return self._calculate_statistics()

    def _generate_random_trades(self) -> List[Trade]:
        """ランダム性を加えたトレードリストを生成

        ブートストラップ法を使用して、実際のトレードデータから
        ランダムにトレードを選択し、新しいトレードシーケンスを生成します。
        
        Returns:
            List[Trade]: ランダムに生成された新しいトレードのリスト
        """
        simulated_trades = []
        current_balance = self.initial_capital
        
        # 元のトレード数と同じ数のトレードをランダムに選択
        n_trades = len(self.trades)
        selected_indices = np.random.choice(n_trades, size=n_trades, replace=True)
        
        for idx in selected_indices:
            original_trade = self.trades[idx]
            
            # トレードをコピー
            new_trade = copy.deepcopy(original_trade)
            
            # 損益にさらなるランダム性を加える
            # 実際のトレードの損益分布から標準偏差を計算
            returns = [t.profit_loss / t.position_size for t in self.trades]
            volatility = np.std(returns)
            
            # 元のリターンに対して、より大きなランダム変動を加える
            original_return = original_trade.profit_loss / original_trade.position_size
            random_return = np.random.normal(original_return, volatility * 0.5)  # ボラティリティの50%をランダム性として使用
            
            # 新しい損益を計算
            new_trade.profit_loss = random_return * original_trade.position_size
            new_trade.balance = current_balance + new_trade.profit_loss
            current_balance = new_trade.balance
            
            # エントリー日とイグジット日もランダムにシフト
            time_shift = np.random.randint(-5, 6)  # -5日から+5日のランダムなシフト
            new_trade.entry_date += pd.Timedelta(days=time_shift)
            new_trade.exit_date += pd.Timedelta(days=time_shift)
            
            simulated_trades.append(new_trade)
        
        # 日付順にソート
        simulated_trades.sort(key=lambda x: x.entry_date)
        
        return simulated_trades

    def _calculate_equity_curve(self, trades: List[Trade]) -> List[float]:
        """トレードのエクイティカーブを計算

        Args:
            trades: トレードリスト

        Returns:
            List[float]: エクイティカーブ（各時点での資金残高）
        """
        equity = [self.initial_capital]
        current_balance = self.initial_capital
        
        for trade in trades:
            current_balance += trade.profit_loss
            equity.append(current_balance)
        
        return equity

    def _calculate_statistics(self) -> Dict:
        """シミュレーション結果の統計を計算

        Returns:
            Dict: 統計結果
        """
        results = np.array([(r['final_capital'] - self.initial_capital) / self.initial_capital * 100 
                           for r in self.simulation_results])
        
        # 信頼区間の計算
        confidence_lower = np.percentile(results, (1 - self.confidence_level) * 100)
        confidence_upper = np.percentile(results, self.confidence_level * 100)
        
        # 各指標の統計量を計算
        stats = {
            'mean_return': np.mean(results),
            'median_return': np.median(results),
            'std_return': np.std(results),
            'min_return': np.min(results),
            'max_return': np.max(results),
            f'confidence_interval_{int(self.confidence_level*100)}': {
                'lower': confidence_lower,
                'upper': confidence_upper
            },
            'metrics_stats': self._calculate_metrics_statistics()
        }
        
        return stats

    def _calculate_metrics_statistics(self) -> Dict:
        """各指標の統計量を計算

        Returns:
            Dict: 各指標の統計量
        """
        metrics = ['total_return', 'max_drawdown', 'sharpe_ratio', 'sortino_ratio', 
                  'win_rate', 'profit_factor', 'alpha_score']
        
        stats = {}
        for metric in metrics:
            values = [r[metric] for r in self.simulation_results]
            stats[metric] = {
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                f'percentile_{int(self.confidence_level*100)}': np.percentile(values, self.confidence_level * 100),
                f'percentile_{int((1-self.confidence_level)*100)}': np.percentile(values, (1-self.confidence_level) * 100)
            }
        
        return stats

    def get_worst_case_scenario(self) -> Dict:
        """最悪のシナリオを取得

        Returns:
            Dict: 最も低いリターンを記録したシミュレーションの結果
        """
        return min(self.simulation_results, key=lambda x: x['total_return'])

    def get_best_case_scenario(self) -> Dict:
        """最良のシナリオを取得

        Returns:
            Dict: 最も高いリターンを記録したシミュレーションの結果
        """
        return max(self.simulation_results, key=lambda x: x['total_return'])

    def plot_equity_curves(self, save_path: Optional[str] = None):
        """エクイティカーブをプロット

        Args:
            save_path: グラフの保存パス（Noneの場合は表示のみ）
        """
        plt.figure(figsize=(15, 8))
        
        # スタイル設定
        plt.style.use('seaborn-v0_8-darkgrid')  # seabornのスタイルを正しく指定
        
        # 全シミュレーションのエクイティカーブをプロット
        for i, equity_curve in enumerate(self.equity_curves):
            plt.plot(equity_curve, alpha=0.1, linewidth=0.5, color='blue')
        
        # 平均エクイティカーブを計算してプロット
        avg_equity = np.mean(self.equity_curves, axis=0)
        plt.plot(avg_equity, color='red', linewidth=2, label='平均')
        
        # 信頼区間を計算してプロット
        lower_percentile = (1 - self.confidence_level) * 100
        upper_percentile = self.confidence_level * 100
        lower_bound = np.percentile(self.equity_curves, lower_percentile, axis=0)
        upper_bound = np.percentile(self.equity_curves, upper_percentile, axis=0)
        
        plt.fill_between(
            range(len(avg_equity)),
            lower_bound,
            upper_bound,
            alpha=0.2,
            color='red',
            label=f'{int(self.confidence_level*100)}%信頼区間'
        )
        
        # グラフの設定
        plt.title('モンテカルロシミュレーション: エクイティカーブ', fontsize=14)
        plt.xlabel('トレード数', fontsize=12)
        plt.ylabel('口座残高', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Y軸を通貨形式で表示
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()

    def print_simulation_results(self) -> None:
        """シミュレーション結果を出力し、グラフを表示"""
        if not self.simulation_results:
            print("シミュレーションが実行されていません。")
            return

        stats = self._calculate_statistics()
        
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
