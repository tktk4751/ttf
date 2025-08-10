#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ultimate MAMA シグナル詳細テスト
実際に生成されているエントリー・エグジットシグナルを詳しく分析
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

def test_ultimate_mama_signals():
    """Ultimate MAMAシグナルの詳細テスト"""
    print("=== Ultimate MAMA シグナル詳細テスト ===")
    
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
    
    print(f"データシンボル: {first_symbol}")
    print(f"総データポイント数: {len(data)}")
    print(f"データ期間: {data.index[0]} - {data.index[-1]}")
    print(f"価格範囲: {data['close'].min():.2f} - {data['close'].max():.2f}")
    
    # Ultimate MAMAストラテジー
    strategy = UltimateMAMAStrategy(
        fast_limit=0.8,
        slow_limit=0.02,
        src_type='hlc3',
        quantum_coherence_factor=0.7,
        quantum_entanglement_strength=0.5,
        mmae_models_count=5,
        vmd_modes_count=3,
        base_confidence_threshold=0.25,
        minimum_signal_quality=0.15,
        quantum_exit_threshold=0.15,
        enable_adaptive_thresholds=True,
        enable_quantum_exit=True,
        enable_real_time_optimization=False,
        ml_adaptation_enabled=False
    )
    
    print("\n--- エントリーシグナル分析 ---")
    entry_signals = strategy.generate_entry(data)
    
    print(f"エントリーシグナル配列長: {len(entry_signals)}")
    print(f"ロングエントリー数: {np.sum(entry_signals == 1)}")
    print(f"ショートエントリー数: {np.sum(entry_signals == -1)}")
    print(f"シグナルなし: {np.sum(entry_signals == 0)}")
    print(f"総エントリー率: {(np.sum(entry_signals != 0) / len(entry_signals) * 100):.2f}%")
    
    # エントリーシグナルが発生したインデックスを取得
    long_entries = np.where(entry_signals == 1)[0]
    short_entries = np.where(entry_signals == -1)[0]
    
    print(f"\nロングエントリーの詳細:")
    if len(long_entries) > 0:
        for i, idx in enumerate(long_entries[:10]):  # 最初の10個まで表示
            date = data.index[idx]
            price = data.iloc[idx]['close']
            print(f"  {i+1}. Index {idx}: {date} @ {price:.2f}")
    else:
        print("  ロングエントリーなし")
    
    print(f"\nショートエントリーの詳細:")
    if len(short_entries) > 0:
        for i, idx in enumerate(short_entries[:10]):  # 最初の10個まで表示
            date = data.index[idx]
            price = data.iloc[idx]['close']
            print(f"  {i+1}. Index {idx}: {date} @ {price:.2f}")
    else:
        print("  ショートエントリーなし")
    
    # エグジットシグナルテスト
    print(f"\n--- エグジットシグナル分析 ---")
    
    if len(long_entries) > 0:
        # 最初のロングエントリーのエグジット条件をテスト
        entry_idx = long_entries[0]
        print(f"最初のロングエントリー（Index {entry_idx}）からのエグジット分析:")
        
        exit_signals = []
        exit_indices = []
        
        # エントリー後のエグジット条件をチェック
        for test_idx in range(entry_idx + 1, min(entry_idx + 200, len(data))):
            exit_condition = strategy.generate_exit(data, position=1, index=test_idx)
            if exit_condition:
                exit_signals.append(True)
                exit_indices.append(test_idx)
                print(f"  エグジット条件満足: Index {test_idx} @ {data.index[test_idx]} @ {data.iloc[test_idx]['close']:.2f}")
            else:
                exit_signals.append(False)
        
        print(f"  調査範囲: {min(200, len(data) - entry_idx - 1)}ポイント")
        print(f"  エグジット条件満足回数: {sum(exit_signals)}")
        
        if len(exit_indices) > 0:
            first_exit = exit_indices[0]
            holding_period = first_exit - entry_idx
            print(f"  最初のエグジット: Index {first_exit} (保有期間: {holding_period}バー)")
    
    # シグナル品質とメトリクス分析
    print(f"\n--- シグナル品質分析 ---")
    try:
        quantum_metrics = strategy.get_quantum_metrics(data)
        if quantum_metrics:
            print(f"量子メトリクス取得成功:")
            for name, values in quantum_metrics.items():
                if len(values) > 0:
                    print(f"  {name}: 平均={np.nanmean(values):.4f}, 標準偏差={np.nanstd(values):.4f}")
        
        advanced_metrics = strategy.get_advanced_metrics(data)
        if advanced_metrics:
            print(f"高度なメトリクス取得成功: {len(advanced_metrics)}項目")
            
            # 融合シグナル分析
            if 'fused_signals' in advanced_metrics:
                fused_signals = advanced_metrics['fused_signals']
                print(f"融合シグナル:")
                print(f"  平均: {np.nanmean(fused_signals):.4f}")
                print(f"  標準偏差: {np.nanstd(fused_signals):.4f}")
                print(f"  最小値: {np.nanmin(fused_signals):.4f}")
                print(f"  最大値: {np.nanmax(fused_signals):.4f}")
                
                # 強いシグナルの頻度
                strong_positive = np.sum(fused_signals > 0.5)
                strong_negative = np.sum(fused_signals < -0.5)
                print(f"  強いポジティブシグナル(>0.5): {strong_positive}回")
                print(f"  強いネガティブシグナル(<-0.5): {strong_negative}回")
            
            # 適応的閾値分析
            if 'adaptive_thresholds' in advanced_metrics:
                thresholds = advanced_metrics['adaptive_thresholds']
                print(f"適応的閾値:")
                print(f"  平均: {np.nanmean(thresholds):.4f}")
                print(f"  最小値: {np.nanmin(thresholds):.4f}")
                print(f"  最大値: {np.nanmax(thresholds):.4f}")
    
    except Exception as e:
        print(f"メトリクス取得エラー: {e}")
    
    # MAMA/FAMA値の確認
    print(f"\n--- MAMA/FAMA値分析 ---")
    try:
        mama_values = strategy.get_ultimate_mama_values(data)
        fama_values = strategy.get_ultimate_fama_values(data)
        
        if len(mama_values) > 0 and len(fama_values) > 0:
            print(f"MAMA値:")
            print(f"  平均: {np.nanmean(mama_values):.4f}")
            print(f"  最小値: {np.nanmin(mama_values):.4f}")
            print(f"  最大値: {np.nanmax(mama_values):.4f}")
            
            print(f"FAMA値:")
            print(f"  平均: {np.nanmean(fama_values):.4f}")
            print(f"  最小値: {np.nanmin(fama_values):.4f}")
            print(f"  最大値: {np.nanmax(fama_values):.4f}")
            
            # 価格との関係
            prices = data['close'].values
            mama_price_ratio = np.nanmean(mama_values) / np.mean(prices)
            print(f"MAMA/価格平均比: {mama_price_ratio:.4f}")
            
            # クロスオーバー分析
            crossovers = 0
            for i in range(1, len(mama_values)):
                mama_above_now = mama_values[i] > fama_values[i]
                mama_above_prev = mama_values[i-1] > fama_values[i-1]
                if mama_above_now != mama_above_prev:
                    crossovers += 1
            
            print(f"MAMA/FAMAクロスオーバー数: {crossovers}")
            
    except Exception as e:
        print(f"MAMA/FAMA値取得エラー: {e}")
    
    print(f"\n=== Ultimate MAMA シグナル詳細テスト完了 ===")

if __name__ == "__main__":
    test_ultimate_mama_signals()