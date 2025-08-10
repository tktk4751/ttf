#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ultimate MAMA シグナル生成デバッグスクリプト
"""

import sys
import os
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource
from strategies.implementations.ultimate_mama.strategy import UltimateMAMAStrategy

def debug_ultimate_mama_signals():
    """Ultimate MAMAシグナルのデバッグ"""
    print("=== Ultimate MAMA シグナル生成デバッグ ===")
    
    # 設定ファイルの読み込み
    config_path = Path('config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # データの準備
    binance_config = config.get('binance_data', {})
    data_dir = binance_config.get('data_dir', 'data/binance')
    binance_data_source = BinanceDataSource(data_dir)
    
    dummy_csv_source = CSVDataSource("dummy")
    data_loader = DataLoader(
        data_source=dummy_csv_source,
        binance_data_source=binance_data_source
    )
    data_processor = DataProcessor()
    
    # データの読み込み
    print("データを読み込み中...")
    raw_data = data_loader.load_data_from_config(config)
    processed_data = {
        symbol: data_processor.process(df)
        for symbol, df in raw_data.items()
    }
    
    # 最初のシンボルのデータを取得
    first_symbol = list(processed_data.keys())[0]
    data = processed_data[first_symbol]
    
    print(f"データサイズ: {len(data)}行")
    print(f"価格範囲: {data['close'].min():.2f} - {data['close'].max():.2f}")
    
    # 異なる設定でUltimate MAMAをテスト
    test_configs = [
        {
            "name": "非常に感度の高い設定",
            "params": {
                "fast_limit": 0.7,
                "slow_limit": 0.05,
                "base_confidence_threshold": 0.4,
                "minimum_signal_quality": 0.2,
                "quantum_exit_threshold": 0.2
            }
        },
        {
            "name": "中程度の感度設定",
            "params": {
                "fast_limit": 0.6,
                "slow_limit": 0.03,
                "base_confidence_threshold": 0.6,
                "minimum_signal_quality": 0.3,
                "quantum_exit_threshold": 0.25
            }
        },
        {
            "name": "デフォルト設定",
            "params": {}
        }
    ]
    
    for config_test in test_configs:
        print(f"\n--- {config_test['name']} ---")
        
        # ストラテジー作成
        strategy = UltimateMAMAStrategy(**config_test['params'])
        
        # エントリーシグナル生成
        entry_signals = strategy.generate_entry(data)
        
        # シグナル統計
        long_signals = np.sum(entry_signals == 1)
        short_signals = np.sum(entry_signals == -1)
        total_signals = np.sum(entry_signals != 0)
        signal_rate = total_signals / len(entry_signals) * 100
        
        print(f"  ロングシグナル数: {long_signals}")
        print(f"  ショートシグナル数: {short_signals}")
        print(f"  総シグナル数: {total_signals}")
        print(f"  シグナル率: {signal_rate:.2f}%")
        
        # 高度なメトリクス取得
        try:
            advanced_metrics = strategy.get_advanced_metrics(data)
            if 'signal_quality' in advanced_metrics:
                avg_quality = np.nanmean(advanced_metrics['signal_quality'])
                print(f"  平均信号品質: {avg_quality:.4f}")
            
            if 'quantum_coherence' in advanced_metrics:
                avg_coherence = np.nanmean(advanced_metrics['quantum_coherence'])
                print(f"  平均量子コヒーレンス: {avg_coherence:.4f}")
                
        except Exception as e:
            print(f"  メトリクス取得エラー: {e}")
        
        # 最初の20個のシグナルを表示
        print(f"  最初の20個のシグナル: {entry_signals[:20]}")
    
    # データの期間別分析
    print(f"\n--- 期間別シグナル分析 ---")
    
    # 最近のデータ（最後の1000ポイント）でテスト
    recent_data = data.tail(1000)
    print(f"最近のデータ（{len(recent_data)}点）でのテスト:")
    
    strategy = UltimateMAMAStrategy(
        fast_limit=0.6,
        slow_limit=0.03,
        base_confidence_threshold=0.5,
        minimum_signal_quality=0.25
    )
    
    recent_signals = strategy.generate_entry(recent_data)
    recent_signal_rate = np.sum(recent_signals != 0) / len(recent_signals) * 100
    
    print(f"  最近のシグナル率: {recent_signal_rate:.2f}%")
    print(f"  最近のロング数: {np.sum(recent_signals == 1)}")
    print(f"  最近のショート数: {np.sum(recent_signals == -1)}")

if __name__ == "__main__":
    debug_ultimate_mama_signals()