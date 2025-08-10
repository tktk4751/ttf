#!/usr/bin/env python3
"""
HyperFRAMA Bollinger Bands Signals Test

HyperFRAMAボリンジャーバンドシグナルの動作確認テスト
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# パスを追加
sys.path.append(str(Path(__file__).parent))

from signals.implementations.hyper_frama_bollinger.bollinger_breakout_signal import HyperFRAMABollingerBreakoutSignal
from data.binance_data_source import BinanceDataSource
from indicators.price_source import PriceSource


def test_bollinger_breakout_signals():
    """ブレイクアウトシグナルテスト"""
    print("=== HyperFRAMA Bollinger ブレイクアウトシグナルテスト ===")
    
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
    
    # ブレイクアウトシグナルテスト
    print("\n1. ブレイクアウトシグナルテスト")
    breakout_signal = HyperFRAMABollingerBreakoutSignal(
        signal_type="breakout",
        lookback=2,
        bollinger_period=20,
        bollinger_sigma_mode="dynamic"
    )
    
    entry_signals = breakout_signal.generate_entry(data)
    exit_signals = breakout_signal.generate_exit(data)
    
    print(f"エントリーシグナル数: {np.sum(entry_signals != 0)}")
    print(f"ロングエントリー: {np.sum(entry_signals == 1)}")
    print(f"ショートエントリー: {np.sum(entry_signals == -1)}")
    print(f"エグジットシグナル数: {np.sum(exit_signals != 0)}")
    print(f"ロングエグジット: {np.sum(exit_signals == 1)}")
    print(f"ショートエグジット: {np.sum(exit_signals == -1)}")
    
    return breakout_signal, data, entry_signals, exit_signals


def test_bollinger_reversal_signals():
    """リバーサルシグナルテスト"""
    print("\n=== HyperFRAMA Bollinger リバーサルシグナルテスト ===")
    
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
    
    # リバーサルシグナルテスト
    print("\n1. リバーサルシグナルテスト")
    reversal_signal = HyperFRAMABollingerBreakoutSignal(
        signal_type="reversal",
        lookback=3,
        bollinger_period=20,
        bollinger_sigma_mode="dynamic",
        exit_mode=3  # パーセントB反転
    )
    
    entry_signals = reversal_signal.generate_entry(data)
    exit_signals = reversal_signal.generate_exit(data)
    
    print(f"エントリーシグナル数: {np.sum(entry_signals != 0)}")
    print(f"ロングエントリー: {np.sum(entry_signals == 1)}")
    print(f"ショートエントリー: {np.sum(entry_signals == -1)}")
    print(f"エグジットシグナル数: {np.sum(exit_signals != 0)}")
    print(f"ロングエグジット: {np.sum(exit_signals == 1)}")
    print(f"ショートエグジット: {np.sum(exit_signals == -1)}")
    
    return reversal_signal, data, entry_signals, exit_signals


def test_bollinger_signal_modes():
    """各エグジットモードテスト"""
    print("\n=== エグジットモードテスト ===")
    
    # テストデータ生成
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
    
    exit_modes = {
        1: "逆ブレイクアウト",
        2: "ミッドラインクロス", 
        3: "パーセントB反転"
    }
    
    for exit_mode, mode_name in exit_modes.items():
        print(f"\n{exit_mode}. {mode_name}モードテスト")
        
        signal = HyperFRAMABollingerBreakoutSignal(
            signal_type="breakout",
            exit_mode=exit_mode,
            bollinger_period=15
        )
        
        entry_signals = signal.generate_entry(data)
        exit_signals = signal.generate_exit(data)
        
        print(f"  エントリーシグナル: {np.sum(entry_signals != 0)}")
        print(f"  エグジットシグナル: {np.sum(exit_signals != 0)}")
        
        # ボリンジャー値取得テスト
        midline, upper_band, lower_band, percent_b, sigma_values = signal.get_bollinger_values(data)
        print(f"  シグマ値範囲: {np.nanmin(sigma_values):.3f} - {np.nanmax(sigma_values):.3f}")
        print(f"  パーセントB範囲: {np.nanmin(percent_b):.3f} - {np.nanmax(percent_b):.3f}")


def test_real_market_signals():
    """実市場データでのシグナルテスト"""
    print("\n=== 実市場データシグナルテスト ===")
    
    try:
        # 合成データを使用（実データ取得の代替）
        np.random.seed(789)
        n_points = 300
        
        # より現実的な価格データ
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
        
        print(f"データサイズ: {len(data)}")
        print(f"価格範囲: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
        
        # 複数のシグナル設定テスト
        signal_configs = [
            {"signal_type": "breakout", "exit_mode": 1, "name": "ブレイクアウト+逆ブレイク"},
            {"signal_type": "breakout", "exit_mode": 2, "name": "ブレイクアウト+ミッドライン"},
            {"signal_type": "reversal", "exit_mode": 3, "name": "リバーサル+パーセントB"}
        ]
        
        for config in signal_configs:
            print(f"\n--- {config['name']} ---")
            
            signal = HyperFRAMABollingerBreakoutSignal(
                signal_type=config["signal_type"],
                exit_mode=config["exit_mode"],
                bollinger_period=20,
                bollinger_sigma_mode="dynamic"
            )
            
            entry_signals = signal.generate_entry(data)
            exit_signals = signal.generate_exit(data)
            
            # 統計計算
            long_entries = np.sum(entry_signals == 1)
            short_entries = np.sum(entry_signals == -1)
            long_exits = np.sum(exit_signals == 1)
            short_exits = np.sum(exit_signals == -1)
            
            print(f"ロングエントリー: {long_entries}")
            print(f"ショートエントリー: {short_entries}")
            print(f"ロングエグジット: {long_exits}")
            print(f"ショートエグジット: {short_exits}")
            print(f"シグナル名: {signal.name}")
        
        return data, signal_configs
        
    except Exception as e:
        print(f"実市場データテストエラー: {e}")
        return None, []


def create_signal_visualization(data, signal, entry_signals, exit_signals, title="HyperFRAMA Bollinger Signals"):
    """シグナル可視化チャート"""
    
    # ボリンジャーバンド値取得
    midline, upper_band, lower_band, percent_b, sigma_values = signal.get_bollinger_values(data)
    
    plt.figure(figsize=(15, 12))
    
    # サブプロット1: 価格とボリンジャーバンド + シグナル
    plt.subplot(4, 1, 1)
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
    
    # エグジットシグナルをプロット
    long_exits = np.where(exit_signals == 1)[0]
    short_exits = np.where(exit_signals == -1)[0]
    
    if len(long_exits) > 0:
        plt.scatter(long_exits, data['close'].iloc[long_exits], 
                   color='orange', marker='x', s=100, label='Long Exit', zorder=5)
    if len(short_exits) > 0:
        plt.scatter(short_exits, data['close'].iloc[short_exits], 
                   color='purple', marker='x', s=100, label='Short Exit', zorder=5)
    
    plt.title(f'{title} - Price, Bands and Signals')
    plt.ylabel('Price')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # サブプロット2: シグマ値
    plt.subplot(4, 1, 2)
    plt.plot(sigma_values, label='Dynamic Sigma', color='orange', linewidth=1.5)
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Sigma Min (1.0)')
    plt.axhline(y=2.5, color='red', linestyle='--', alpha=0.5, label='Sigma Max (2.5)')
    plt.title('Dynamic Sigma Values')
    plt.ylabel('Sigma')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # サブプロット3: パーセント B
    plt.subplot(4, 1, 3)
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
    
    # サブプロット4: シグナル
    plt.subplot(4, 1, 4)
    plt.plot(entry_signals, label='Entry Signals', color='blue', marker='o', markersize=3)
    plt.plot(exit_signals, label='Exit Signals', color='red', marker='s', markersize=3)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.axhline(y=1, color='blue', linestyle='--', alpha=0.5, label='Long Signal')
    plt.axhline(y=-1, color='red', linestyle='--', alpha=0.5, label='Short Signal')
    plt.title('Entry/Exit Signals')
    plt.ylabel('Signal')
    plt.xlabel('Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hyper_frama_bollinger_signals_test.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """メイン実行"""
    print("HyperFRAMA Bollinger Signals テスト開始\n")
    
    # ブレイクアウトシグナルテスト
    breakout_signal, breakout_data, breakout_entry, breakout_exit = test_bollinger_breakout_signals()
    
    # リバーサルシグナルテスト
    reversal_signal, reversal_data, reversal_entry, reversal_exit = test_bollinger_reversal_signals()
    
    # エグジットモードテスト
    test_bollinger_signal_modes()
    
    # 実市場データテスト
    real_data, configs = test_real_market_signals()
    
    # 可視化（ブレイクアウトシグナルの結果を使用）
    create_signal_visualization(
        breakout_data, breakout_signal, breakout_entry, breakout_exit,
        "HyperFRAMA Bollinger Breakout Signals"
    )
    
    print("\nテスト完了！")
    print("チャート: hyper_frama_bollinger_signals_test.png")
    print("\n=== テスト結果サマリー ===")
    print(f"ブレイクアウトシグナル名: {breakout_signal.name}")
    print(f"リバーサルシグナル名: {reversal_signal.name}")


if __name__ == "__main__":
    main()