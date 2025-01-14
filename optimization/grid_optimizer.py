import os
import sys
import yaml
import logging
import numpy as np
from datetime import datetime
from itertools import product
from typing import Dict, Any, List, Tuple, Type
from tqdm import tqdm
import multiprocessing

# プロジェクトのルートディレクトリをPythonパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from backtesting.backtester import Backtester
from position_sizing.fixed_ratio import FixedRatioSizing
from strategies.base.strategy import BaseStrategy
from data.data_loader import DataLoader
from data.data_processor import DataProcessor
from analytics.analytics import Analytics

class GridOptimizer:
    """グリッドサーチによる戦略パラメーターの最適化を行うクラス"""
    
    def __init__(
        self,
        config_path: str,
        strategy_class: Type[BaseStrategy],
        param_ranges: Dict[str, np.ndarray],
        n_jobs: int = -1
    ):
        """
        Args:
            config_path: 設定ファイルのパス
            strategy_class: 最適化対象の戦略クラス
            param_ranges: パラメーターの範囲（キーはパラメーター名、値はnumpy配列）
            n_jobs: 並列処理数（-1で全CPU使用）
        """
        self.logger = logging.getLogger('ttf')
        self.strategy_class = strategy_class
        self.param_ranges = param_ranges
        self.n_jobs = n_jobs if n_jobs != -1 else multiprocessing.cpu_count()
        
        # 設定ファイルの読み込み
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # データの読み込みと処理（初期化時に1回だけ実行）
        self._load_and_process_data()
        
        # バックテスターを事前に初期化
        self.backtester = Backtester(
            strategy=None,  # 後で設定
            position_sizing=FixedRatioSizing({
                'ratio': self.config['position_sizing']['params']['ratio'],
                'min_position': None,
                'max_position': None,
                'leverage': 1
            }),
            initial_balance=self.config['backtest']['initial_balance'],
            commission=self.config['backtest']['commission'],
            max_positions=self.config['backtest']['max_positions']
        )
        
        # 最適化結果の保存用
        self.best_params = None
        self.best_score = float('-inf')
        self.best_trades = None
    
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
    
    def _evaluate_params(self, params_tuple: Tuple) -> Tuple[float, Dict[str, Any], Any]:
        """パラメーターセットの評価

        Args:
            params_tuple: 評価するパラメーターセットのタプル

        Returns:
            Tuple[float, Dict[str, Any], Any]: アルファスコア, パラメータ, トレード結果
        """
        param_names = list(self.param_ranges.keys())
        params = dict(zip(param_names, params_tuple))
        
        # 戦略の生成
        strategy = self.strategy_class(**params)
        self.backtester.strategy = strategy

        # バックテストの実行
        trades = self.backtester.run(self.data_dict)
        
        # トレード数が少なすぎる場合はスキップ
        if len(trades) < 30:
            return float('-inf'), params, None

        # アナリティクスの計算
        analytics = Analytics(trades, self.config['backtest']['initial_balance'])
        alpha_score = analytics.calculate_alpha_score()
        
        return alpha_score, params, trades

    def optimize(self) -> Tuple[Dict[str, Any], float, List[Any]]:
        """最適化を実行

        Returns:
            Tuple[Dict[str, Any], float, List[Any]]: 最適パラメーター、最高スコア、最良トレード
        """
        self.logger.info("グリッドサーチによる最適化を開始します")
        
        # パラメーターの組み合わせを生成
        param_names = list(self.param_ranges.keys())
        param_values = list(self.param_ranges.values())
        combinations = list(product(*param_values))
        total_combinations = len(combinations)
        
        self.logger.info(f"パラメーター組み合わせ総数: {total_combinations}")
        
        # 並列処理で最適化を実行
        with multiprocessing.Pool(processes=self.n_jobs) as pool:
            results = list(tqdm(pool.imap(self._evaluate_params, combinations), total=total_combinations, desc="最適化の進捗"))

        # 結果を集約
        for alpha_score, params, trades in results:
            if alpha_score > self.best_score:
                self.best_score = alpha_score
                self.best_params = params
                self.best_trades = trades
                self.logger.info(f"新しいベストスコアを発見: {alpha_score:.2f} (パラメーター: {params})")
        
        self.logger.info(f"最適化が完了しました")
        self.logger.info(f"最適パラメーター: {self.best_params}")
        self.logger.info(f"最高スコア: {self.best_score:.2f}")
        
        # 最適パラメーターでのバックテスト結果を表示
        if self.best_trades:
            analytics = Analytics(self.best_trades, self.config['backtest']['initial_balance'])
            # ... (以下、最適化結果表示部分は変更なし)
            # 損益統計の出力
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

        
        return self.best_params, self.best_score, self.best_trades 