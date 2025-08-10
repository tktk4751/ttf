#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X_MAMAエントリーシグナルのテストスクリプト
"""

import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from signals.implementations.x_mama.entry import XMAMACrossoverEntrySignal
    from indicators.x_mama import X_MAMA
    import yaml
    from data.data_loader import DataLoader, CSVDataSource
    from data.data_processor import DataProcessor
    from data.binance_data_source import BinanceDataSource
    print("必要なモジュールのインポート完了")
except ImportError as e:
    print(f"インポートエラー: {e}")
    sys.exit(1)

def test_x_mama_entry_signals():
    """X_MAMAエントリーシグナルをテストする"""
    print("\n=== X_MAMAエントリーシグナルテスト ===")
    
    # 設定ファイルからデータを読み込む
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print(f"設定ファイルが見つかりません: {config_path}")
        return
    
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
        
    except Exception as e:
        print(f"データ読み込みエラー: {e}")
        return
    
    # 1. クロスオーバーシグナルをテスト
    print("\n1. クロスオーバーシグナル（基本設定）")
    crossover_signal = XMAMACrossoverEntrySignal(
        fast_limit=0.5,
        slow_limit=0.05,
        src_type='hlc3',
        use_kalman_filter=False,
        use_zero_lag=True,
        position_mode=False  # クロスオーバーシグナル
    )
    
    crossover_signals = crossover_signal.generate(data)
    
    # クロスオーバーシグナルの統計
    long_signals = np.sum(crossover_signals == 1)
    short_signals = np.sum(crossover_signals == -1)
    no_signals = np.sum(crossover_signals == 0)
    
    print(f"クロスオーバーシグナル - ロング: {long_signals}, ショート: {short_signals}, なし: {no_signals}")
    
    # 2. 位置関係シグナルをテスト
    print("\n2. 位置関係シグナル（基本設定）")
    position_signal = XMAMACrossoverEntrySignal(
        fast_limit=0.5,
        slow_limit=0.05,
        src_type='hlc3',
        use_kalman_filter=False,
        use_zero_lag=True,
        position_mode=True  # 位置関係シグナル
    )
    
    position_signals = position_signal.generate(data)
    
    # 位置関係シグナルの統計
    long_signals_pos = np.sum(position_signals == 1)
    short_signals_pos = np.sum(position_signals == -1)
    no_signals_pos = np.sum(position_signals == 0)
    
    print(f"位置関係シグナル - ロング: {long_signals_pos}, ショート: {short_signals_pos}, なし: {no_signals_pos}")
    
    # 3. カルマンフィルター + ゼロラグ処理 + クロスオーバー
    print("\n3. カルマンフィルター + ゼロラグ処理 + クロスオーバー")
    try:
        advanced_signal = XMAMACrossoverEntrySignal(
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
        
        advanced_signals = advanced_signal.generate(data)
        
        # 高度シグナルの統計
        long_signals_adv = np.sum(advanced_signals == 1)
        short_signals_adv = np.sum(advanced_signals == -1)
        no_signals_adv = np.sum(advanced_signals == 0)
        
        print(f"高度シグナル - ロング: {long_signals_adv}, ショート: {short_signals_adv}, なし: {no_signals_adv}")
        
    except Exception as e:
        print(f"高度シグナルエラー: {e}")
    
    # 4. X_MAMA値の取得テスト
    print("\n4. X_MAMA値の取得テスト")
    mama_values = crossover_signal.get_mama_values(data)
    fama_values = crossover_signal.get_fama_values(data)
    period_values = crossover_signal.get_period_values(data)
    alpha_values = crossover_signal.get_alpha_values(data)
    phase_values = crossover_signal.get_phase_values(data)
    i1_values = crossover_signal.get_i1_values(data)
    q1_values = crossover_signal.get_q1_values(data)
    
    print(f"X_MAMA値 - 長さ: {len(mama_values)}, 有効値: {len(mama_values) - np.isnan(mama_values).sum()}")
    print(f"X_FAMA値 - 長さ: {len(fama_values)}, 有効値: {len(fama_values) - np.isnan(fama_values).sum()}")
    print(f"Period値 - 長さ: {len(period_values)}, 有効値: {len(period_values) - np.isnan(period_values).sum()}")
    print(f"Alpha値 - 長さ: {len(alpha_values)}, 有効値: {len(alpha_values) - np.isnan(alpha_values).sum()}")
    print(f"Phase値 - 長さ: {len(phase_values)}, 有効値: {len(phase_values) - np.isnan(phase_values).sum()}")
    print(f"I1値 - 長さ: {len(i1_values)}, 有効値: {len(i1_values) - np.isnan(i1_values).sum()}")
    print(f"Q1値 - 長さ: {len(q1_values)}, 有効値: {len(q1_values) - np.isnan(q1_values).sum()}")
    
    # 5. シグナルの詳細分析
    print("\n5. シグナルの詳細分析")
    
    # クロスオーバーシグナルの分析
    crossover_indices = np.where(crossover_signals != 0)[0]
    if len(crossover_indices) > 0:
        print(f"クロスオーバーシグナル発生インデックス（最初の10個）:")
        for i in range(min(10, len(crossover_indices))):
            idx = crossover_indices[i]
            signal_type = "ロング" if crossover_signals[idx] == 1 else "ショート"
            date = data.index[idx]
            mama_val = mama_values[idx]
            fama_val = fama_values[idx]
            print(f"  [{idx}] {date}: {signal_type} - X_MAMA: {mama_val:.2f}, X_FAMA: {fama_val:.2f}")
    
    # 連続するシグナルの分析
    print("\n6. 連続するシグナルの分析")
    
    # 位置関係シグナルでの連続性
    position_changes = np.diff(position_signals)
    change_indices = np.where(position_changes != 0)[0]
    
    if len(change_indices) > 0:
        print(f"位置関係の変化点（最初の10個）:")
        for i in range(min(10, len(change_indices))):
            idx = change_indices[i] + 1  # diffは1つ前を見るため+1
            if idx < len(data):
                prev_signal = position_signals[idx-1]
                curr_signal = position_signals[idx]
                date = data.index[idx]
                
                prev_str = "ロング" if prev_signal == 1 else ("ショート" if prev_signal == -1 else "なし")
                curr_str = "ロング" if curr_signal == 1 else ("ショート" if curr_signal == -1 else "なし")
                
                print(f"  [{idx}] {date}: {prev_str} → {curr_str}")
    
    # 7. パフォーマンステスト
    print("\n7. パフォーマンステスト")
    
    import time
    
    # 基本クロスオーバーシグナルの計算時間
    start_time = time.time()
    for _ in range(10):
        crossover_signal.generate(data)
    end_time = time.time()
    
    print(f"基本クロスオーバーシグナル: {(end_time - start_time) / 10:.4f}秒/回 (平均10回)")
    
    # 位置関係シグナルの計算時間
    start_time = time.time()
    for _ in range(10):
        position_signal.generate(data)
    end_time = time.time()
    
    print(f"位置関係シグナル: {(end_time - start_time) / 10:.4f}秒/回 (平均10回)")
    
    print("\n=== テスト完了 ===")

if __name__ == "__main__":
    test_x_mama_entry_signals()