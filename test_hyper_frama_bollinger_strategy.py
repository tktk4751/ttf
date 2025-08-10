#!/usr/bin/env python3
"""
HyperFRAMA Bollinger Strategy Test

HyperFRAMAボリンジャー戦略とシグナルジェネレーターの動作確認テスト
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# パスを追加
sys.path.append(str(Path(__file__).parent))

from strategies.implementations.hyper_frama_bollinger.strategy import HyperFRAMABollingerStrategy
from strategies.implementations.hyper_frama_bollinger.signal_generator import SignalType, FilterType
from data.binance_data_source import BinanceDataSource


def test_strategy_basic_functionality():
    """戦略の基本機能テスト"""
    print("=== HyperFRAMA Bollinger Strategy 基本機能テスト ===")
    
    # テストデータ生成
    np.random.seed(42)
    n_points = 200
    
    # ボラティリティクラスタリング付きの価格生成
    base_price = 50000
    returns = []
    volatilities = []
    
    # GARCH風のボラティリティ
    alpha = 0.1
    beta = 0.85
    omega = 0.001
    
    vol = 0.02
    for i in range(n_points):
        # ボラティリティ更新
        if i > 0:
            vol = np.sqrt(omega + alpha * (returns[-1]**2) + beta * (vol**2))
        volatilities.append(vol)
        
        # リターン生成
        ret = np.random.normal(0, vol)
        returns.append(ret)
    
    # 価格系列生成
    prices = [base_price]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices[:-1])
    
    # OHLCV生成
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, n_points)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.003, n_points))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.003, n_points))),
        'close': prices,
        'volume': np.random.uniform(100, 1000, n_points)
    })
    
    # ブレイクアウト戦略テスト
    print("\n1. ブレイクアウト戦略テスト")
    breakout_strategy = HyperFRAMABollingerStrategy(
        signal_type=SignalType.BREAKOUT,
        lookback=2,
        exit_mode=2,  # ミッドラインクロス
        filter_type=FilterType.NONE
    )
    
    entry_signals = breakout_strategy.generate_entry(data)
    
    print(f"エントリーシグナル数: {np.sum(entry_signals != 0)}")
    print(f"ロングエントリー: {np.sum(entry_signals == 1)}")
    print(f"ショートエントリー: {np.sum(entry_signals == -1)}")
    
    # エグジットテスト（いくつかのポジションで）
    exit_tests = 0
    for i in range(len(entry_signals) - 10, len(entry_signals)):
        if entry_signals[i] == 1:  # ロングポジション想定
            exit_result = breakout_strategy.generate_exit(data, 1, i + 5)
            if exit_result:
                exit_tests += 1
    
    print(f"エグジット発生回数: {exit_tests}")
    print(f"戦略名: {breakout_strategy.name}")
    
    return breakout_strategy, data


def test_strategy_signal_types():
    """異なるシグナルタイプのテスト"""
    print("\n=== シグナルタイプ比較テスト ===")
    
    # より振動的なデータ生成（リバーサル検出用）
    np.random.seed(123)
    n_points = 200
    
    # サイン波ベースの価格生成（レンジ相場）
    t = np.linspace(0, 4*np.pi, n_points)
    base_trend = 50000 + 2000 * np.sin(t) + 1000 * np.sin(3*t)
    noise = np.random.normal(0, 300, n_points)
    prices = base_trend + noise
    
    # OHLCV生成
    data = pd.DataFrame({
        'open': prices * 0.999,
        'high': prices * 1.002,
        'low': prices * 0.998,
        'close': prices,
        'volume': np.random.uniform(100, 1000, n_points)
    })
    
    signal_types = {
        SignalType.BREAKOUT: "ブレイクアウト",
        SignalType.REVERSAL: "リバーサル"
    }
    
    results = {}
    
    for signal_type, name in signal_types.items():
        print(f"\n--- {name}戦略 ---")
        
        strategy = HyperFRAMABollingerStrategy(
            signal_type=signal_type,
            lookback=2,
            exit_mode=2,
            filter_type=FilterType.NONE,
            bollinger_period=20
        )
        
        entry_signals = strategy.generate_entry(data)
        
        print(f"  エントリーシグナル: {np.sum(entry_signals != 0)}")
        print(f"  ロングエントリー: {np.sum(entry_signals == 1)}")
        print(f"  ショートエントリー: {np.sum(entry_signals == -1)}")
        
        results[signal_type] = {
            'strategy': strategy,
            'entry_signals': entry_signals
        }
    
    return results, data


def test_strategy_filter_types():
    """フィルタータイプテスト"""
    print("\n=== フィルタータイプ比較テスト ===")
    
    # トレンドデータ生成
    np.random.seed(456)
    n_points = 150
    
    # トレンド + ノイズ
    trend = np.linspace(100, 150, n_points)
    noise = np.random.normal(0, 3, n_points)
    prices = trend + noise
    
    data = pd.DataFrame({
        'open': prices * 0.999,
        'high': prices * 1.002,
        'low': prices * 0.998,
        'close': prices,
        'volume': np.random.uniform(100, 1000, n_points)
    })
    
    filter_types = {
        FilterType.NONE: "フィルターなし",
        FilterType.HYPER_ER: "HyperER",
        FilterType.HYPER_TREND_INDEX: "HyperTrendIndex",
        FilterType.HYPER_ADX: "HyperADX"
        # FilterType.CONSENSUS: "コンセンサス"  # 時間がかかるためスキップ
    }
    
    for filter_type, name in filter_types.items():
        print(f"\n--- {name}フィルター ---")
        
        try:
            strategy = HyperFRAMABollingerStrategy(
                signal_type=SignalType.BREAKOUT,
                filter_type=filter_type,
                bollinger_period=15
            )
            
            entry_signals = strategy.generate_entry(data)
            filter_signals = strategy.get_filter_signals(data)
            
            print(f"  エントリーシグナル: {np.sum(entry_signals != 0)}")
            print(f"  フィルター許可: {np.sum(filter_signals == 1)}")
            print(f"  フィルター拒否: {np.sum(filter_signals == -1)}")
            
        except Exception as e:
            print(f"  エラー: {e}")


def test_strategy_exit_modes():
    """エグジットモードテスト"""
    print("\n=== エグジットモード比較テスト ===")
    
    # テストデータ生成
    np.random.seed(789)
    n_points = 100
    
    prices = 50000 + np.cumsum(np.random.normal(0, 100, n_points))
    
    data = pd.DataFrame({
        'open': prices * 0.999,
        'high': prices * 1.002,
        'low': prices * 0.998,
        'close': prices,
        'volume': np.random.uniform(100, 1000, n_points)
    })
    
    exit_modes = {
        1: "逆ブレイクアウト",
        2: "ミッドラインクロス", 
        3: "パーセントB反転"
    }
    
    for exit_mode, mode_name in exit_modes.items():
        print(f"\n--- {mode_name}エグジット ---")
        
        strategy = HyperFRAMABollingerStrategy(
            signal_type=SignalType.BREAKOUT,
            exit_mode=exit_mode,
            filter_type=FilterType.NONE
        )
        
        entry_signals = strategy.generate_entry(data)
        exit_signals = strategy.get_bollinger_exit_signals(data)
        
        print(f"  エントリーシグナル: {np.sum(entry_signals != 0)}")
        print(f"  エグジットシグナル: {np.sum(exit_signals != 0)}")
        print(f"  戦略名: {strategy.name}")


def test_strategy_advanced_metrics():
    """高度なメトリクステスト"""
    print("\n=== 高度なメトリクステスト ===")
    
    # 合成データ
    np.random.seed(999)
    n_points = 150
    
    base_price = 45000
    returns = np.random.normal(0.0002, 0.02, n_points)
    prices = [base_price]
    
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices[:-1])
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, n_points)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_points))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_points))),
        'close': prices,
        'volume': np.random.uniform(500, 2000, n_points)
    })
    
    strategy = HyperFRAMABollingerStrategy(
        signal_type=SignalType.BREAKOUT,
        filter_type=FilterType.NONE,
        bollinger_sigma_mode="dynamic"
    )
    
    # 基本メトリクス
    entry_signals = strategy.generate_entry(data)
    midline, upper_band, lower_band, percent_b, sigma_values = strategy.get_bollinger_values(data)
    
    print(f"データサイズ: {len(data)}")
    print(f"価格範囲: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    print(f"エントリーシグナル: {np.sum(entry_signals != 0)}")
    print(f"シグマ値範囲: {np.nanmin(sigma_values):.3f} - {np.nanmax(sigma_values):.3f}")
    print(f"パーセントB範囲: {np.nanmin(percent_b):.3f} - {np.nanmax(percent_b):.3f}")
    
    # 高度なメトリクス
    advanced_metrics = strategy.get_advanced_metrics(data)
    print(f"高度なメトリクス数: {len(advanced_metrics)}")
    
    # 戦略情報
    strategy_info = strategy.get_strategy_info()
    print(f"\n戦略情報:")
    print(f"  名前: {strategy_info['name']}")
    print(f"  説明: {strategy_info['description']}")
    print(f"  機能数: {len(strategy_info['features'])}")
    
    return strategy, data, advanced_metrics


def create_strategy_visualization(strategy, data, title="HyperFRAMA Bollinger Strategy"):
    """戦略可視化チャート"""
    
    # シグナルとメトリクス取得
    entry_signals = strategy.generate_entry(data)
    midline, upper_band, lower_band, percent_b, sigma_values = strategy.get_bollinger_values(data)
    bollinger_entry = strategy.get_bollinger_entry_signals(data)
    bollinger_exit = strategy.get_bollinger_exit_signals(data)
    filter_signals = strategy.get_filter_signals(data)
    
    plt.figure(figsize=(15, 16))
    
    # サブプロット1: 価格とボリンジャーバンド + エントリーシグナル
    plt.subplot(5, 1, 1)
    plt.plot(data['close'], label='Close Price', color='black', linewidth=1)
    plt.plot(midline, label='HyperFRAMA Midline', color='blue', linewidth=1.5)
    plt.plot(upper_band, label='Upper Band', color='red', linewidth=1, alpha=0.8)
    plt.plot(lower_band, label='Lower Band', color='green', linewidth=1, alpha=0.8)
    plt.fill_between(range(len(upper_band)), upper_band, lower_band, 
                     alpha=0.1, color='gray', label='Band Area')
    
    # エントリーシグナルをプロット
    long_entries = np.where(entry_signals == 1)[0]
    short_entries = np.where(entry_signals == -1)[0]
    
    if len(long_entries) > 0:
        plt.scatter(long_entries, data['close'].iloc[long_entries], 
                   color='blue', marker='^', s=100, label='Long Entry', zorder=5)
    if len(short_entries) > 0:
        plt.scatter(short_entries, data['close'].iloc[short_entries], 
                   color='red', marker='v', s=100, label='Short Entry', zorder=5)
    
    plt.title(f'{title} - Price, Bands and Entry Signals')
    plt.ylabel('Price')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # サブプロット2: 動的シグマ値
    plt.subplot(5, 1, 2)
    plt.plot(sigma_values, label='Dynamic Sigma', color='orange', linewidth=1.5)
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Sigma Min (1.0)')
    plt.axhline(y=2.5, color='red', linestyle='--', alpha=0.5, label='Sigma Max (2.5)')
    plt.title('Dynamic Sigma Values (HyperER-based)')
    plt.ylabel('Sigma')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # サブプロット3: パーセント B
    plt.subplot(5, 1, 3)
    plt.plot(percent_b, label='Percent B', color='brown', linewidth=1)
    plt.axhline(y=0.0, color='green', linestyle='--', alpha=0.5, label='Lower Band (0%)')
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Upper Band (100%)')
    plt.axhline(y=0.5, color='blue', linestyle='--', alpha=0.5, label='Midline (50%)')
    plt.fill_between(range(len(percent_b)), 0, 1, alpha=0.1, color='gray')
    
    # パーセントBのしきい値
    plt.axhline(y=0.2, color='green', linestyle=':', alpha=0.7, label='Oversold (20%)')
    plt.axhline(y=0.8, color='red', linestyle=':', alpha=0.7, label='Overbought (80%)')
    
    plt.title('Percent B')
    plt.ylabel('Percent B')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # サブプロット4: フィルターシグナル
    plt.subplot(5, 1, 4)
    plt.plot(filter_signals, label='Filter Signals', color='purple', linewidth=1.5)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Filter Allow')
    plt.axhline(y=-1, color='red', linestyle='--', alpha=0.5, label='Filter Deny')
    plt.title('Filter Signals')
    plt.ylabel('Signal')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # サブプロット5: エントリー・エグジットシグナル比較
    plt.subplot(5, 1, 5)
    plt.plot(bollinger_entry, label='Bollinger Entry', color='blue', marker='o', markersize=2, alpha=0.7)
    plt.plot(bollinger_exit, label='Bollinger Exit', color='red', marker='s', markersize=2, alpha=0.7)
    plt.plot(entry_signals, label='Final Entry (Filtered)', color='darkblue', linewidth=2)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.axhline(y=1, color='blue', linestyle='--', alpha=0.5)
    plt.axhline(y=-1, color='red', linestyle='--', alpha=0.5)
    plt.title('Signal Comparison')
    plt.ylabel('Signal')
    plt.xlabel('Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hyper_frama_bollinger_strategy_test.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """メイン実行"""
    print("HyperFRAMA Bollinger Strategy テスト開始\n")
    
    # 基本機能テスト
    strategy, data = test_strategy_basic_functionality()
    
    # シグナルタイプテスト
    signal_results, signal_data = test_strategy_signal_types()
    
    # フィルタータイプテスト
    test_strategy_filter_types()
    
    # エグジットモードテスト
    test_strategy_exit_modes()
    
    # 高度なメトリクステスト
    advanced_strategy, advanced_data, metrics = test_strategy_advanced_metrics()
    
    # 可視化（高度なメトリクス戦略の結果を使用）
    create_strategy_visualization(
        advanced_strategy, advanced_data,
        "HyperFRAMA Bollinger Strategy (Dynamic Sigma)"
    )
    
    print("\nテスト完了！")
    print("チャート: hyper_frama_bollinger_strategy_test.png")
    print("\n=== テスト結果サマリー ===")
    print(f"基本戦略名: {strategy.name}")
    print(f"高度戦略名: {advanced_strategy.name}")
    
    # 最適化パラメータのテスト例
    print(f"\n=== 最適化機能デモ ===")
    
    import optuna
    
    # ダミートライアルで最適化パラメータテスト
    study = optuna.create_study()
    trial = study.ask()
    
    try:
        opt_params = HyperFRAMABollingerStrategy.create_optimization_params(trial)
        strategy_params = HyperFRAMABollingerStrategy.convert_params_to_strategy_format(opt_params)
        
        print(f"最適化パラメータ例: {len(opt_params)} 個")
        print(f"戦略パラメータ例: {len(strategy_params)} 個")
        print("最適化機能: 正常")
    except Exception as e:
        print(f"最適化機能エラー: {e}")


if __name__ == "__main__":
    main()