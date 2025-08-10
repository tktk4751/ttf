#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X_MAMAストラテジーとシグナルジェネレーターのテストスクリプト
"""

import numpy as np
import pandas as pd
import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from strategies.implementations.x_mama.strategy import XMAMAStrategy
    from strategies.implementations.x_mama.signal_generator import XMAMASignalGenerator
    import yaml
    from data.data_loader import DataLoader, CSVDataSource
    from data.data_processor import DataProcessor
    from data.binance_data_source import BinanceDataSource
    print("必要なモジュールのインポート完了")
except ImportError as e:
    print(f"インポートエラー: {e}")
    sys.exit(1)

def load_test_data():
    """テストデータを読み込む"""
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print(f"設定ファイルが見つかりません: {config_path}")
        return None
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # データの準備
    binance_config = config.get('binance_data', {})
    data_dir = binance_config.get('data_dir', 'data/binance')
    binance_data_source = BinanceDataSource(data_dir)
    
    # CSVデータソースはダミーとして渡す
    dummy_csv_source = CSVDataSource("dummy")
    data_loader = DataLoader(
        data_source=dummy_csv_source,
        binance_data_source=binance_data_source
    )
    data_processor = DataProcessor()
    
    # データの読み込みと処理
    print("データを読み込み中...")
    try:
        raw_data = data_loader.load_data_from_config(config)
        processed_data = {
            symbol: data_processor.process(df)
            for symbol, df in raw_data.items()
        }
        
        # 最初のシンボルのデータを取得
        first_symbol = next(iter(processed_data))
        data = processed_data[first_symbol]
        
        # 2024年のデータに絞り込む
        start_date = '2024-01-01'
        end_date = '2024-12-31'
        data = data[(data.index >= start_date) & (data.index <= end_date)]
        
        print(f"データ読み込み完了: {first_symbol}")
        print(f"期間: {data.index.min()} → {data.index.max()}")
        print(f"データ数: {len(data)}")
        
        return data
        
    except Exception as e:
        print(f"データ読み込みエラー: {e}")
        return None

def test_x_mama_signal_generator():
    """X_MAMAシグナルジェネレーターをテストする"""
    print("\n=== X_MAMAシグナルジェネレーターテスト ===")
    
    data = load_test_data()
    if data is None:
        return
    
    # 1. クロスオーバーシグナルジェネレーター
    print("\n1. クロスオーバーシグナルジェネレーター")
    crossover_generator = XMAMASignalGenerator(
        fast_limit=0.5,
        slow_limit=0.05,
        src_type='hlc3',
        use_kalman_filter=False,
        use_zero_lag=True,
        position_mode=False
    )
    
    entry_signals = crossover_generator.get_entry_signals(data)
    
    # シグナル統計
    long_signals = np.sum(entry_signals == 1)
    short_signals = np.sum(entry_signals == -1)
    no_signals = np.sum(entry_signals == 0)
    
    print(f"エントリーシグナル - ロング: {long_signals}, ショート: {short_signals}, なし: {no_signals}")
    
    # エグジットシグナルのテスト
    print("エグジットシグナルテスト:")
    exit_count = 0
    for i in range(len(data)):
        if entry_signals[i] == 1:  # ロングエントリー
            # 後続の時点でエグジットシグナルをチェック
            for j in range(i + 1, min(i + 50, len(data))):  # 最大50期間後まで
                if crossover_generator.get_exit_signals(data, 1, j):
                    exit_count += 1
                    break
    
    print(f"ロングエントリー後のエグジット発生数: {exit_count}/{long_signals}")
    
    # 2. 位置関係シグナルジェネレーター
    print("\n2. 位置関係シグナルジェネレーター")
    position_generator = XMAMASignalGenerator(
        fast_limit=0.5,
        slow_limit=0.05,
        src_type='hlc3',
        use_kalman_filter=False,
        use_zero_lag=True,
        position_mode=True
    )
    
    position_signals = position_generator.get_entry_signals(data)
    
    # シグナル統計
    long_signals_pos = np.sum(position_signals == 1)
    short_signals_pos = np.sum(position_signals == -1)
    no_signals_pos = np.sum(position_signals == 0)
    
    print(f"位置関係シグナル - ロング: {long_signals_pos}, ショート: {short_signals_pos}, なし: {no_signals_pos}")
    
    # 3. 高度なメトリクスの取得
    print("\n3. 高度なメトリクスの取得")
    advanced_metrics = crossover_generator.get_advanced_metrics(data)
    
    for metric_name, metric_values in advanced_metrics.items():
        if len(metric_values) > 0:
            valid_count = len(metric_values) - np.isnan(metric_values).sum()
            print(f"{metric_name}: 長さ {len(metric_values)}, 有効値 {valid_count}")
        else:
            print(f"{metric_name}: 空の配列")
    
    # 4. カルマンフィルター + ゼロラグ処理
    print("\n4. カルマンフィルター + ゼロラグ処理")
    try:
        advanced_generator = XMAMASignalGenerator(
            fast_limit=0.5,
            slow_limit=0.05,
            src_type='hlc3',
            use_kalman_filter=True,
            kalman_filter_type='unscented',
            kalman_process_noise=0.01,
            kalman_observation_noise=0.001,
            use_zero_lag=True,
            position_mode=False
        )
        
        advanced_signals = advanced_generator.get_entry_signals(data)
        
        # シグナル統計
        long_signals_adv = np.sum(advanced_signals == 1)
        short_signals_adv = np.sum(advanced_signals == -1)
        no_signals_adv = np.sum(advanced_signals == 0)
        
        print(f"高度シグナル - ロング: {long_signals_adv}, ショート: {short_signals_adv}, なし: {no_signals_adv}")
        
    except Exception as e:
        print(f"高度シグナルジェネレーターエラー: {e}")

def test_x_mama_strategy():
    """X_MAMAストラテジーをテストする"""
    print("\n=== X_MAMAストラテジーテスト ===")
    
    data = load_test_data()
    if data is None:
        return
    
    # 1. 基本的なX_MAMAストラテジー
    print("\n1. 基本的なX_MAMAストラテジー")
    basic_strategy = XMAMAStrategy(
        fast_limit=0.5,
        slow_limit=0.05,
        src_type='hlc3',
        use_kalman_filter=False,
        use_zero_lag=True,
        position_mode=False
    )
    
    # エントリーシグナル生成
    entry_signals = basic_strategy.generate_entry(data)
    
    # シグナル統計
    long_signals = np.sum(entry_signals == 1)
    short_signals = np.sum(entry_signals == -1)
    no_signals = np.sum(entry_signals == 0)
    
    print(f"エントリーシグナル - ロング: {long_signals}, ショート: {short_signals}, なし: {no_signals}")
    
    # エグジットシグナルのテスト
    print("エグジットシグナルテスト:")
    exit_tests = []
    for i in range(len(data)):
        if entry_signals[i] == 1:  # ロングエントリー
            # 後続の時点でエグジットシグナルをチェック
            for j in range(i + 1, min(i + 50, len(data))):
                if basic_strategy.generate_exit(data, 1, j):
                    exit_tests.append((i, j, j - i))
                    break
    
    print(f"ロングエントリー後のエグジット発生数: {len(exit_tests)}/{long_signals}")
    if exit_tests:
        avg_hold_time = np.mean([hold_time for _, _, hold_time in exit_tests])
        print(f"平均保有期間: {avg_hold_time:.1f}期間")
    
    # 2. 高度なX_MAMAストラテジー
    print("\n2. 高度なX_MAMAストラテジー")
    try:
        advanced_strategy = XMAMAStrategy(
            fast_limit=0.5,
            slow_limit=0.05,
            src_type='hlc3',
            use_kalman_filter=True,
            kalman_filter_type='unscented',
            kalman_process_noise=0.01,
            kalman_observation_noise=0.001,
            use_zero_lag=True,
            position_mode=False
        )
        
        advanced_entry_signals = advanced_strategy.generate_entry(data)
        
        # シグナル統計
        long_signals_adv = np.sum(advanced_entry_signals == 1)
        short_signals_adv = np.sum(advanced_entry_signals == -1)
        no_signals_adv = np.sum(advanced_entry_signals == 0)
        
        print(f"高度エントリーシグナル - ロング: {long_signals_adv}, ショート: {short_signals_adv}, なし: {no_signals_adv}")
        
        # ストラテジー情報の取得
        strategy_info = advanced_strategy.get_strategy_info()
        print(f"ストラテジー名: {strategy_info['name']}")
        print(f"機能: {len(strategy_info['features'])}個")
        
    except Exception as e:
        print(f"高度ストラテジーエラー: {e}")
    
    # 3. X_MAMA値の取得
    print("\n3. X_MAMA値の取得")
    mama_values = basic_strategy.get_mama_values(data)
    fama_values = basic_strategy.get_fama_values(data)
    
    print(f"X_MAMA値: 長さ {len(mama_values)}, 有効値 {len(mama_values) - np.isnan(mama_values).sum()}")
    print(f"X_FAMA値: 長さ {len(fama_values)}, 有効値 {len(fama_values) - np.isnan(fama_values).sum()}")
    
    if len(mama_values) > 0 and len(fama_values) > 0:
        print(f"X_MAMA - 平均: {np.nanmean(mama_values):.2f}, 標準偏差: {np.nanstd(mama_values):.2f}")
        print(f"X_FAMA - 平均: {np.nanmean(fama_values):.2f}, 標準偏差: {np.nanstd(fama_values):.2f}")
    
    # 4. 最適化パラメータの生成テスト
    print("\n4. 最適化パラメータの生成テスト")
    try:
        import optuna
        
        # ダミーのトライアルを作成
        study = optuna.create_study()
        trial = study.ask()
        
        # 最適化パラメータの生成
        optimization_params = XMAMAStrategy.create_optimization_params(trial)
        strategy_params = XMAMAStrategy.convert_params_to_strategy_format(optimization_params)
        
        print(f"最適化パラメータ数: {len(optimization_params)}")
        print(f"戦略パラメータ数: {len(strategy_params)}")
        
        # パラメータでストラテジーを作成
        optimized_strategy = XMAMAStrategy(**strategy_params)
        print("最適化パラメータでのストラテジー作成成功")
        
    except ImportError:
        print("Optunaが利用できません。最適化パラメータテストをスキップします。")
    except Exception as e:
        print(f"最適化パラメータテストエラー: {e}")

def test_performance():
    """パフォーマンステスト"""
    print("\n=== パフォーマンステスト ===")
    
    data = load_test_data()
    if data is None:
        return
    
    # 基本的なX_MAMAストラテジー
    basic_strategy = XMAMAStrategy(
        fast_limit=0.5,
        slow_limit=0.05,
        src_type='hlc3',
        use_kalman_filter=False,
        use_zero_lag=True,
        position_mode=False
    )
    
    # エントリーシグナル生成の計算時間
    print("1. エントリーシグナル生成の計算時間")
    start_time = time.time()
    for _ in range(10):
        basic_strategy.generate_entry(data)
    end_time = time.time()
    
    print(f"基本ストラテジー: {(end_time - start_time) / 10:.4f}秒/回 (平均10回)")
    
    # シグナルジェネレーター単体のテスト
    print("2. シグナルジェネレーター単体のテスト")
    signal_generator = XMAMASignalGenerator(
        fast_limit=0.5,
        slow_limit=0.05,
        src_type='hlc3',
        use_kalman_filter=False,
        use_zero_lag=True,
        position_mode=False
    )
    
    start_time = time.time()
    for _ in range(10):
        signal_generator.get_entry_signals(data)
    end_time = time.time()
    
    print(f"シグナルジェネレーター: {(end_time - start_time) / 10:.4f}秒/回 (平均10回)")
    
    # キャッシュ効果のテスト
    print("3. キャッシュ効果のテスト")
    start_time = time.time()
    for _ in range(100):
        signal_generator.get_entry_signals(data)  # 同じデータで繰り返し
    end_time = time.time()
    
    print(f"キャッシュ利用時: {(end_time - start_time) / 100:.4f}秒/回 (平均100回)")

def main():
    """メイン関数"""
    print("=== X_MAMAストラテジーとシグナルジェネレーター総合テスト ===")
    
    # 各テストを実行
    test_x_mama_signal_generator()
    test_x_mama_strategy()
    test_performance()
    
    print("\n=== 全テスト完了 ===")

if __name__ == "__main__":
    main()