#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import yaml
from datetime import datetime

# プロジェクトのルートディレクトリをPythonパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd
from backtesting.backtester import Backtester
from position_sizing.fixed_ratio import FixedRatioSizing
from strategies.supertrend_rsi_chopstrategy import SupertrendRsiChopStrategy
from data.data_loader import DataLoader
from data.data_processor import DataProcessor
from analytics.analytics import Analytics

def test_backtester():
    """バックテスターのテスト"""
    
    # 設定ファイルの読み込み
    config_path = os.path.join(project_root, 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # データの設定を取得
    data_config = config.get('data', {})
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
    data = processor.process(data)
    
    # データを辞書形式に変換
    data_dict = {symbol: data}
    
    # 戦略の設定
    strategy = SupertrendRsiChopStrategy(
        supertrend_params={'period': 45, 'multiplier': 5.5},
        rsi_entry_params={'period': 2, 'solid': {'rsi_long_entry': 20, 'rsi_short_entry': 80}},
        rsi_exit_params={'period': 24, 'solid': {'rsi_long_exit_solid': 85, 'rsi_short_exit_solid': 15}},
        chop_params={'period': 9, 'solid': {'chop_solid': 50}}
    )
    
    # ポジションサイジングの設定
    position_sizing = FixedRatioSizing({
        'ratio': 1,  # 資金の99%を使用
        'min_position': None,  # 最小ポジションサイズの制限なし
        'max_position': None,  # 最大ポジションサイズの制限なし
        'leverage': 1  # レバレッジなし
    })
    
    # バックテスターの作成と実行
    backtester = Backtester(
        strategy=strategy,
        position_sizing=position_sizing,
        initial_balance=config['backtest']['initial_balance'],
        commission=config['backtest']['commission'],
        max_positions=config['backtest']['max_positions']
    )
    
    trades = backtester.run(data_dict)
    
    # パフォーマンス分析
    analytics = Analytics(trades, config['backtest']['initial_balance'])
    
    # 基本統計の出力
    print("\n=== 基本統計 ===")
    print(f"初期資金: {config['backtest']['initial_balance']:.2f} USD")
    print(f"最終残高: {backtester.current_capital:.2f} USD")
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
    
    # バックテスト終了後の残高
    print(f"\n=== 最終結果 ===")
    print(f"初期資金: {config['backtest']['initial_balance']:.2f} USD")
    print(f"最終残高: {backtester.current_capital:.2f} USD")

if __name__ == '__main__':
    test_backtester()
