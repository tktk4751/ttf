#!/usr/bin/env python3
"""
ドンチャンFRAMAストラテジーのテストスクリプト
全フィルタータイプとパフォーマンス比較を実行
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import yaml
from datetime import datetime
import matplotlib.pyplot as plt
from data.binance_data_source import BinanceDataSource
from strategies.implementations.donchian_frama.strategy import DonchianFRAMAStrategy
from strategies.implementations.donchian_frama.signal_generator import FilterType


def load_test_data():
    """テストデータの読み込み"""
    print("データを読み込み中...")
    
    # config.yamlから設定を読み込み
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"config.yaml読み込みエラー: {e}")
        return None
    
    # Binanceデータソース
    data_source = BinanceDataSource(config['binance_data']['data_dir'])
    
    # SOL/USDTの4時間足データを取得
    symbol = 'SOL'
    timeframe = '4h'
    limit = 1000
    
    try:
        data = data_source.load_data(symbol, timeframe)
        if data is None or len(data) < 100:
            print(f"データが不十分: {len(data) if data is not None else 0}行")
            return None
            
        print(f"データ読み込み完了: {len(data)}行 ({data.index[0]} から {data.index[-1]})")
        return data
        
    except Exception as e:
        print(f"データ取得エラー: {e}")
        return None


def create_strategy_configs():
    """各フィルタータイプの設定を作成"""
    base_config = {
        'name': 'DonchianFRAMA',
        'stop_loss_pct': 0.03,
        'take_profit_pct': 0.06,
        'min_position_size': 0.01,
        'max_position_size': 0.95,
        'entry': {
            'donchian_period': 20,
            'frama_period': 16,
            'frama_fc': 1,
            'frama_sc': 200,
            'signal_mode': 'position'
        },
        'hyper_er': {
            'period': 14,
            'midline_period': 100
        },
        'hyper_trend_index': {
            'chop_period': 14,
            'atr_period': 14,
            'midline_period': 100,
            'volatility_period': 20
        },
        'hyper_adx': {
            'adx_period': 14,
            'midline_period': 100,
            'smoothing_period': 3
        }
    }
    
    # 各フィルタータイプの設定
    configs = {}
    
    for filter_type in FilterType:
        config = base_config.copy()
        config['filter_type'] = filter_type.value
        config['name'] = f'DonchianFRAMA_{filter_type.value}'
        configs[filter_type.value] = config
    
    return configs


def run_backtest(strategy, data):
    """簡易バックテスト実行"""
    initial_capital = 10000
    current_capital = initial_capital
    position = 0
    entry_price = 0
    trades = []
    equity_curve = []
    
    signals = strategy.generate_signals(data)
    entry_signals = signals['entry']
    
    for i in range(len(data)):
        current_price = data.iloc[i]['close']
        signal = entry_signals[i]
        
        # ポジション管理
        if position == 0 and signal != 0:
            # 新規エントリー
            position = signal
            entry_price = current_price
            
        elif position != 0:
            # エグジット判定
            should_exit = False
            exit_reason = ''
            
            # 反対シグナル
            if (position > 0 and signal < 0) or (position < 0 and signal > 0):
                should_exit = True
                exit_reason = 'reverse_signal'
            
            # ストップロス/利確（簡易版）
            if position > 0:
                pnl_pct = (current_price - entry_price) / entry_price
                if pnl_pct <= -0.03:  # 3%ロス
                    should_exit = True
                    exit_reason = 'stop_loss'
                elif pnl_pct >= 0.06:  # 6%利確
                    should_exit = True
                    exit_reason = 'take_profit'
            else:
                pnl_pct = (entry_price - current_price) / entry_price
                if pnl_pct <= -0.03:
                    should_exit = True
                    exit_reason = 'stop_loss'
                elif pnl_pct >= 0.06:
                    should_exit = True
                    exit_reason = 'take_profit'
            
            if should_exit:
                # トレード記録
                if position > 0:
                    pnl = (current_price - entry_price) / entry_price
                else:
                    pnl = (entry_price - current_price) / entry_price
                
                current_capital *= (1 + pnl * 0.95)  # 95%投資と仮定
                
                trades.append({
                    'entry_time': data.index[i-1],
                    'exit_time': data.index[i],
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'position': position,
                    'pnl_pct': pnl,
                    'reason': exit_reason
                })
                
                position = 0
        
        equity_curve.append(current_capital)
    
    return {
        'final_capital': current_capital,
        'total_return': (current_capital - initial_capital) / initial_capital,
        'trades': trades,
        'equity_curve': equity_curve,
        'num_trades': len(trades),
        'win_rate': len([t for t in trades if t['pnl_pct'] > 0]) / max(len(trades), 1)
    }


def test_all_filters(data):
    """全フィルタータイプをテスト"""
    print("\n=== ドンチャンFRAMAストラテジー全フィルターテスト ===")
    
    configs = create_strategy_configs()
    results = {}
    
    for filter_name, config in configs.items():
        print(f"\n--- {filter_name}フィルターテスト ---")
        
        try:
            strategy = DonchianFRAMAStrategy(config)
            result = run_backtest(strategy, data)
            results[filter_name] = result
            
            print(f"最終資本: ${result['final_capital']:.2f}")
            print(f"総リターン: {result['total_return']:.2%}")
            print(f"トレード数: {result['num_trades']}")
            print(f"勝率: {result['win_rate']:.2%}")
            
        except Exception as e:
            print(f"エラー ({filter_name}): {e}")
            results[filter_name] = None
    
    return results


def create_comparison_chart(results, data):
    """結果比較チャート作成"""
    print("\n比較チャートを作成中...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. エクイティカーブ比較
    for filter_name, result in results.items():
        if result and 'equity_curve' in result:
            ax1.plot(result['equity_curve'], label=filter_name, alpha=0.8)
    
    ax1.set_title('エクイティカーブ比較')
    ax1.set_xlabel('期間')
    ax1.set_ylabel('資本')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. リターン比較
    filter_names = []
    returns = []
    for filter_name, result in results.items():
        if result:
            filter_names.append(filter_name)
            returns.append(result['total_return'] * 100)
    
    bars = ax2.bar(filter_names, returns, alpha=0.7)
    ax2.set_title('総リターン比較')
    ax2.set_ylabel('リターン (%)')
    ax2.tick_params(axis='x', rotation=45)
    
    # バーに数値表示
    for bar, ret in zip(bars, returns):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{ret:.1f}%', ha='center', va='bottom')
    
    # 3. トレード数比較
    trade_counts = []
    for filter_name in filter_names:
        result = results[filter_name]
        trade_counts.append(result['num_trades'] if result else 0)
    
    ax3.bar(filter_names, trade_counts, alpha=0.7, color='orange')
    ax3.set_title('トレード数比較')
    ax3.set_ylabel('トレード数')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. 勝率比較
    win_rates = []
    for filter_name in filter_names:
        result = results[filter_name]
        win_rates.append(result['win_rate'] * 100 if result else 0)
    
    ax4.bar(filter_names, win_rates, alpha=0.7, color='green')
    ax4.set_title('勝率比較')
    ax4.set_ylabel('勝率 (%)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.set_ylim(0, 100)
    
    plt.tight_layout()
    
    # 保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'donchian_frama_strategy_comparison_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"チャート保存: {filename}")
    
    plt.show()


def test_signal_generator_detail(data):
    """シグナルジェネレーターの詳細テスト"""
    print("\n=== シグナルジェネレーター詳細テスト ===")
    
    # コンセンサスフィルターのテスト
    config = {
        'filter_type': 'consensus',
        'entry': {
            'donchian_period': 20,
            'frama_period': 16,
            'signal_mode': 'position'
        },
        'hyper_er': {'period': 14, 'midline_period': 100},
        'hyper_trend_index': {'chop_period': 14, 'atr_period': 14, 'midline_period': 100},
        'hyper_adx': {'adx_period': 14, 'midline_period': 100}
    }
    
    from strategies.implementations.donchian_frama.signal_generator import DonchianFRAMASignalGenerator
    
    try:
        generator = DonchianFRAMASignalGenerator(config)
        signal_info = generator.get_signal_info(data)
        
        print("シグナル情報:")
        for key, values in signal_info.items():
            if isinstance(values, np.ndarray):
                print(f"{key}: {len(values)}要素, 最後の10値: {values[-10:]}")
            else:
                print(f"{key}: {values}")
        
        # 統計情報
        if 'entry_filtered' in signal_info:
            entry_signals = signal_info['entry_filtered']
            long_signals = np.sum(entry_signals == 1)
            short_signals = np.sum(entry_signals == -1)
            neutral_signals = np.sum(entry_signals == 0)
            
            print(f"\nシグナル統計:")
            print(f"ロング: {long_signals} ({long_signals/len(entry_signals)*100:.1f}%)")
            print(f"ショート: {short_signals} ({short_signals/len(entry_signals)*100:.1f}%)")
            print(f"ニュートラル: {neutral_signals} ({neutral_signals/len(entry_signals)*100:.1f}%)")
        
    except Exception as e:
        print(f"シグナルジェネレーターテストエラー: {e}")


def main():
    """メイン実行関数"""
    print("ドンチャンFRAMAストラテジーテスト開始")
    
    # データ読み込み
    data = load_test_data()
    if data is None:
        print("テスト終了: データ読み込み失敗")
        return
    
    # 全フィルターテスト
    results = test_all_filters(data)
    
    # 結果表示
    print("\n=== 最終結果サマリー ===")
    best_return = -999
    best_filter = None
    
    for filter_name, result in results.items():
        if result:
            print(f"{filter_name:15} | リターン: {result['total_return']:7.2%} | "
                  f"トレード: {result['num_trades']:3} | 勝率: {result['win_rate']:6.2%}")
            
            if result['total_return'] > best_return:
                best_return = result['total_return']
                best_filter = filter_name
    
    if best_filter:
        print(f"\n最高パフォーマンス: {best_filter} ({best_return:.2%})")
    
    # 比較チャート作成
    valid_results = {k: v for k, v in results.items() if v is not None}
    if len(valid_results) > 1:
        create_comparison_chart(valid_results, data)
    
    # シグナルジェネレーター詳細テスト
    test_signal_generator_detail(data)
    
    print("\nテスト完了!")


if __name__ == "__main__":
    main()