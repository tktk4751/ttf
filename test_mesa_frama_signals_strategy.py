#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from signals.implementations.mesa_frama.entry import MESAFRAMACrossoverEntrySignal
from strategies.implementations.mesa_frama.strategy import MESAFRAMAStrategy
from strategies.implementations.mesa_frama.signal_generator import MESAFRAMASignalGenerator

def generate_test_data(length=300):
    """より複雑なテストデータを生成"""
    np.random.seed(42)
    base_price = 100.0
    
    # 複数の市場状況を含むデータを生成
    prices = [base_price]
    trend_direction = 1
    
    for i in range(1, length):
        # 市場状況の切り替え
        if i % 50 == 0:
            trend_direction *= -1  # トレンド方向を反転
        
        if i < length // 6:  # 上昇トレンド
            change = 0.003 + np.random.normal(0, 0.01)
        elif i < 2 * length // 6:  # レンジ相場
            change = np.random.normal(0, 0.008)
        elif i < 3 * length // 6:  # 下降トレンド
            change = -0.002 + np.random.normal(0, 0.012)
        elif i < 4 * length // 6:  # ボラティリティの高いレンジ相場
            change = np.random.normal(0, 0.025)
        elif i < 5 * length // 6:  # 強い上昇トレンド
            change = 0.005 + np.random.normal(0, 0.015)
        else:  # 安定した下降トレンド
            change = -0.003 + np.random.normal(0, 0.008)
        
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # OHLC データの生成
    data = []
    for i, close in enumerate(prices):
        daily_range = abs(np.random.normal(0, close * 0.015))
        
        high = close + daily_range * np.random.uniform(0.3, 1.0)
        low = close - daily_range * np.random.uniform(0.3, 1.0)
        
        if i == 0:
            open_price = close
        else:
            gap = np.random.normal(0, close * 0.007)
            open_price = prices[i-1] + gap
        
        # 論理的整合性の確保
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.uniform(1000, 10000)
        })
    
    return pd.DataFrame(data)

def test_mesa_frama_entry_signals():
    """MESA_FRAMAエントリーシグナルのテスト"""
    print("=== MESA_FRAMAエントリーシグナルテスト ===")
    
    # テストデータ生成
    df = generate_test_data(300)
    print(f"テストデータ: {len(df)}ポイント")
    print(f"価格範囲: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    # 1. クロスオーバーモードテスト
    print("\n1. クロスオーバーモードテスト")
    entry_signal_crossover = MESAFRAMACrossoverEntrySignal(
        base_period=16,
        src_type='hl2',
        signal_mode='crossover',
        price_source='close',
        use_zero_lag=True
    )
    
    signals_crossover = entry_signal_crossover.generate(df)
    mesa_frama_values = entry_signal_crossover.get_mesa_frama_values()
    dynamic_periods = entry_signal_crossover.get_dynamic_periods()
    
    print(f"  クロスオーバーシグナル数: ロング {np.sum(signals_crossover == 1)}, ショート {np.sum(signals_crossover == -1)}")
    print(f"  MESA_FRAMA平均: {np.nanmean(mesa_frama_values):.4f}")
    print(f"  動的期間平均: {np.nanmean(dynamic_periods):.2f}")
    
    # 2. 位置関係モードテスト
    print("\n2. 位置関係モードテスト")
    entry_signal_position = MESAFRAMACrossoverEntrySignal(
        base_period=16,
        src_type='hl2',
        signal_mode='position',
        price_source='close',
        use_zero_lag=True
    )
    
    signals_position = entry_signal_position.generate(df)
    
    print(f"  位置関係シグナル数: ロング {np.sum(signals_position == 1)}, ショート {np.sum(signals_position == -1)}")
    
    # 3. 異なるパラメータでのテスト
    print("\n3. 異なるパラメータでのテスト")
    
    # 高感度版
    entry_signal_fast = MESAFRAMACrossoverEntrySignal(
        base_period=8,
        mesa_fast_limit=0.8,
        mesa_slow_limit=0.1,
        signal_mode='crossover',
        use_zero_lag=True
    )
    signals_fast = entry_signal_fast.generate(df)
    
    # 低感度版
    entry_signal_slow = MESAFRAMACrossoverEntrySignal(
        base_period=32,
        mesa_fast_limit=0.3,
        mesa_slow_limit=0.02,
        signal_mode='crossover',
        use_zero_lag=False
    )
    signals_slow = entry_signal_slow.generate(df)
    
    print(f"  高感度版シグナル数: ロング {np.sum(signals_fast == 1)}, ショート {np.sum(signals_fast == -1)}")
    print(f"  低感度版シグナル数: ロング {np.sum(signals_slow == 1)}, ショート {np.sum(signals_slow == -1)}")
    
    return {
        'crossover_signals': signals_crossover,
        'position_signals': signals_position,
        'fast_signals': signals_fast,
        'slow_signals': signals_slow,
        'mesa_frama_values': mesa_frama_values,
        'test_data': df
    }

def test_mesa_frama_signal_generator():
    """MESA_FRAMAシグナルジェネレーターのテスト"""
    print("\n=== MESA_FRAMAシグナルジェネレーターテスト ===")
    
    # テストデータ生成
    df = generate_test_data(300)
    
    # 1. 基本的なシグナルジェネレーターテスト
    print("\n1. 基本的なシグナルジェネレーターテスト")
    signal_generator = MESAFRAMASignalGenerator(
        base_period=16,
        src_type='hl2',
        signal_mode='crossover',
        price_source='close',
        use_zero_lag=True
    )
    
    # エントリーシグナル取得
    entry_signals = signal_generator.get_entry_signals(df)
    print(f"  エントリーシグナル数: ロング {np.sum(entry_signals == 1)}, ショート {np.sum(entry_signals == -1)}")
    
    # エグジットシグナルテスト
    exit_tests = []
    for i in range(50, len(df), 20):  # 一定間隔でエグジットテスト
        # ロングポジションのエグジット
        exit_long = signal_generator.get_exit_signals(df, position=1, index=i)
        # ショートポジションのエグジット
        exit_short = signal_generator.get_exit_signals(df, position=-1, index=i)
        exit_tests.append((exit_long, exit_short))
    
    exit_long_count = sum(1 for exit_long, _ in exit_tests if exit_long)
    exit_short_count = sum(1 for _, exit_short in exit_tests if exit_short)
    print(f"  エグジットシグナル数: ロング {exit_long_count}, ショート {exit_short_count} (テスト数: {len(exit_tests)})")
    
    # 2. 指標値取得テスト
    print("\n2. 指標値取得テスト")
    mesa_frama_vals = signal_generator.get_mesa_frama_values()
    fractal_dim = signal_generator.get_fractal_dimension()
    dynamic_periods = signal_generator.get_dynamic_periods()
    mesa_phase = signal_generator.get_mesa_phase()
    alpha_vals = signal_generator.get_alpha_values()
    
    print(f"  MESA_FRAMA: 平均 {np.nanmean(mesa_frama_vals):.4f}, 標準偏差 {np.nanstd(mesa_frama_vals):.4f}")
    print(f"  フラクタル次元: 平均 {np.nanmean(fractal_dim):.4f}, 範囲 [{np.nanmin(fractal_dim):.4f}, {np.nanmax(fractal_dim):.4f}]")
    print(f"  動的期間: 平均 {np.nanmean(dynamic_periods):.2f}, 範囲 [{np.nanmin(dynamic_periods):.2f}, {np.nanmax(dynamic_periods):.2f}]")
    print(f"  MESA位相: 平均 {np.nanmean(mesa_phase):.2f}, 範囲 [{np.nanmin(mesa_phase):.2f}, {np.nanmax(mesa_phase):.2f}]")
    print(f"  フラクタルアルファ: 平均 {np.nanmean(alpha_vals):.4f}, 範囲 [{np.nanmin(alpha_vals):.4f}, {np.nanmax(alpha_vals):.4f}]")
    
    return {
        'signal_generator': signal_generator,
        'entry_signals': entry_signals,
        'mesa_frama_values': mesa_frama_vals,
        'fractal_dimension': fractal_dim,
        'dynamic_periods': dynamic_periods,
        'test_data': df
    }

def test_mesa_frama_strategy():
    """MESA_FRAMAストラテジーのテスト"""
    print("\n=== MESA_FRAMAストラテジーテスト ===")
    
    # テストデータ生成
    df = generate_test_data(300)
    
    # 1. 基本的なストラテジーテスト
    print("\n1. 基本的なストラテジーテスト")
    strategy = MESAFRAMAStrategy(
        base_period=16,
        src_type='hl2',
        signal_mode='crossover',
        price_source='close',
        use_zero_lag=True
    )
    
    # エントリーシグナル生成
    entry_signals = strategy.generate_entry(df)
    print(f"  エントリーシグナル数: ロング {np.sum(entry_signals == 1)}, ショート {np.sum(entry_signals == -1)}")
    
    # エグジットシグナルテスト
    exit_tests = []
    for i in range(50, len(df), 15):
        # ロングポジションのエグジット
        exit_long = strategy.generate_exit(df, position=1, index=i)
        # ショートポジションのエグジット
        exit_short = strategy.generate_exit(df, position=-1, index=i)
        exit_tests.append((exit_long, exit_short))
    
    exit_long_count = sum(1 for exit_long, _ in exit_tests if exit_long)
    exit_short_count = sum(1 for _, exit_short in exit_tests if exit_short)
    print(f"  エグジットシグナル数: ロング {exit_long_count}, ショート {exit_short_count} (テスト数: {len(exit_tests)})")
    
    # 2. 最適化パラメータテスト
    print("\n2. 最適化パラメータテスト")
    
    # Mock Optuna Trial
    class MockTrial:
        def __init__(self):
            self.params = {}
        
        def suggest_int(self, name, low, high, step=1):
            if name == 'base_period':
                return 16
            elif name == 'fc':
                return 1
            elif name == 'sc':
                return 198
            return low
        
        def suggest_float(self, name, low, high, step=None, log=False):
            if name == 'mesa_fast_limit':
                return 0.5
            elif name == 'mesa_slow_limit':
                return 0.05
            elif name == 'crossover_threshold':
                return 0.0
            return (low + high) / 2
        
        def suggest_categorical(self, name, choices):
            if name == 'src_type':
                return 'hl2'
            elif name == 'signal_mode':
                return 'crossover'
            elif name == 'price_source':
                return 'close'
            elif name == 'use_zero_lag':
                return True
            elif name == 'use_kalman_filter':
                return False
            return choices[0]
    
    mock_trial = MockTrial()
    optimization_params = MESAFRAMAStrategy.create_optimization_params(mock_trial)
    strategy_params = MESAFRAMAStrategy.convert_params_to_strategy_format(optimization_params)
    
    print(f"  最適化パラメータ数: {len(optimization_params)}")
    print(f"  ストラテジーパラメータ数: {len(strategy_params)}")
    print(f"  基本期間: {strategy_params['base_period']}")
    print(f"  シグナルモード: {strategy_params['signal_mode']}")
    
    # 3. 指標値取得テスト
    print("\n3. 指標値取得テスト")
    mesa_frama_vals = strategy.get_mesa_frama_values(df)
    fractal_dim = strategy.get_fractal_dimension(df)
    dynamic_periods = strategy.get_dynamic_periods(df)
    
    print(f"  MESA_FRAMA: 有効数 {np.sum(~np.isnan(mesa_frama_vals))}, 平均 {np.nanmean(mesa_frama_vals):.4f}")
    print(f"  フラクタル次元: 有効数 {np.sum(~np.isnan(fractal_dim))}, 平均 {np.nanmean(fractal_dim):.4f}")
    print(f"  動的期間: 有効数 {np.sum(~np.isnan(dynamic_periods))}, 平均 {np.nanmean(dynamic_periods):.2f}")
    
    # 4. パフォーマンステスト
    print("\n4. パフォーマンステスト")
    import time
    
    times = []
    for _ in range(10):
        start_time = time.time()
        strategy.generate_entry(df)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    print(f"  平均計算時間: {avg_time:.4f}秒 (10回平均)")
    print(f"  1ポイント当たり: {avg_time / len(df) * 1000:.4f}ms")
    
    return {
        'strategy': strategy,
        'entry_signals': entry_signals,
        'optimization_params': optimization_params,
        'strategy_params': strategy_params,
        'test_data': df
    }

def test_error_handling():
    """エラーハンドリングテスト"""
    print("\n=== エラーハンドリングテスト ===")
    
    # 1. 短すぎるデータ
    print("\n1. 短すぎるデータテスト")
    short_df = pd.DataFrame({
        'open': [100, 101, 102],
        'high': [101, 102, 103],
        'low': [99, 100, 101],
        'close': [100.5, 101.5, 102.5],
        'volume': [1000, 1100, 1200]
    })
    
    strategy = MESAFRAMAStrategy(base_period=16)
    signals = strategy.generate_entry(short_df)
    print(f"  短いデータ結果: シグナル数 {len(signals)}, 非ゼロシグナル {np.sum(signals != 0)}")
    
    # 2. 空のデータ
    print("\n2. 空のデータテスト")
    empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    
    try:
        signals_empty = strategy.generate_entry(empty_df)
        print(f"  空のデータ結果: シグナル数 {len(signals_empty)}")
    except Exception as e:
        print(f"  空のデータでエラー: {e}")
    
    # 3. 不正なパラメータ
    print("\n3. 不正なパラメータテスト")
    try:
        invalid_strategy = MESAFRAMAStrategy(
            base_period=15,  # 奇数（不正）
            signal_mode='invalid_mode'  # 不正なモード
        )
        print("  不正なパラメータが受け入れられました（予期しない）")
    except Exception as e:
        print(f"  不正なパラメータが正しく拒否されました: {type(e).__name__}")

def main():
    """メインテスト関数"""
    print("=== MESA_FRAMAシグナル・ストラテジー総合テスト ===")
    
    # 各テストの実行
    entry_results = test_mesa_frama_entry_signals()
    signal_gen_results = test_mesa_frama_signal_generator()
    strategy_results = test_mesa_frama_strategy()
    test_error_handling()
    
    # 全体統計
    print("\n=== 全体統計 ===")
    print(f"エントリーシグナルテスト: クロスオーバー {np.sum(entry_results['crossover_signals'] != 0)} シグナル")
    print(f"シグナルジェネレーターテスト: {np.sum(signal_gen_results['entry_signals'] != 0)} シグナル")
    print(f"ストラテジーテスト: {np.sum(strategy_results['entry_signals'] != 0)} シグナル")
    print(f"テストデータ長: {len(entry_results['test_data'])} ポイント")
    
    print("\n=== テスト完了 ===")
    
    return {
        'entry_results': entry_results,
        'signal_gen_results': signal_gen_results,
        'strategy_results': strategy_results
    }

if __name__ == "__main__":
    results = main()