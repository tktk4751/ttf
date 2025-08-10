#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ultimate MAMA 詳細デバッグスクリプト
生成されているMAMA/FAMA値を詳しく分析
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
from indicators.ultimate_mama import UltimateMAMA

def analyze_mama_values():
    """MAMA/FAMA値の詳細分析"""
    print("=== Ultimate MAMA 値の詳細分析 ===")
    
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
    
    # 最後の1000ポイントで分析
    recent_data = data.tail(1000).copy()
    print(f"分析データ: {len(recent_data)}行")
    print(f"価格範囲: {recent_data['close'].min():.2f} - {recent_data['close'].max():.2f}")
    
    # Ultimate MAMA計算
    ultimate_mama = UltimateMAMA(
        fast_limit=0.7,
        slow_limit=0.05,
        quantum_coherence_factor=0.6
    )
    
    result = ultimate_mama.calculate(recent_data)
    
    if result is None:
        print("❌ Ultimate MAMA計算失敗")
        return
    
    print("\n--- MAMA/FAMA値の分析 ---")
    
    # 基本統計
    mama_values = result.ultimate_mama
    fama_values = result.ultimate_fama
    
    print(f"MAMA統計:")
    print(f"  平均: {np.mean(mama_values):.4f}")
    print(f"  標準偏差: {np.std(mama_values):.4f}")
    print(f"  最小値: {np.min(mama_values):.4f}")
    print(f"  最大値: {np.max(mama_values):.4f}")
    
    print(f"\nFAMA統計:")
    print(f"  平均: {np.mean(fama_values):.4f}")
    print(f"  標準偏差: {np.std(fama_values):.4f}")
    print(f"  最小値: {np.min(fama_values):.4f}")
    print(f"  最大値: {np.max(fama_values):.4f}")
    
    # クロスオーバー分析
    crossovers_golden = 0  # MAMA > FAMA
    crossovers_dead = 0    # MAMA < FAMA
    
    for i in range(1, len(mama_values)):
        mama_above_now = mama_values[i] > fama_values[i]
        mama_above_prev = mama_values[i-1] > fama_values[i-1]
        
        if mama_above_now and not mama_above_prev:
            crossovers_golden += 1
        elif not mama_above_now and mama_above_prev:
            crossovers_dead += 1
    
    print(f"\nクロスオーバー分析:")
    print(f"  ゴールデンクロス数: {crossovers_golden}")
    print(f"  デッドクロス数: {crossovers_dead}")
    print(f"  総クロスオーバー数: {crossovers_golden + crossovers_dead}")
    
    # 現在の位置関係
    mama_above_fama = np.sum(mama_values > fama_values)
    mama_below_fama = np.sum(mama_values < fama_values)
    
    print(f"\n位置関係分析:")
    print(f"  MAMA > FAMA: {mama_above_fama}回 ({mama_above_fama/len(mama_values)*100:.1f}%)")
    print(f"  MAMA < FAMA: {mama_below_fama}回 ({mama_below_fama/len(mama_values)*100:.1f}%)")
    
    # 価格との相関
    prices = recent_data['close'].values
    price_changes = np.diff(prices)
    mama_changes = np.diff(mama_values)
    
    price_up = np.sum(price_changes > 0)
    price_down = np.sum(price_changes < 0)
    mama_up = np.sum(mama_changes > 0)
    mama_down = np.sum(mama_changes < 0)
    
    print(f"\n変化方向分析:")
    print(f"  価格上昇: {price_up}回 ({price_up/len(price_changes)*100:.1f}%)")
    print(f"  価格下降: {price_down}回 ({price_down/len(price_changes)*100:.1f}%)")
    print(f"  MAMA上昇: {mama_up}回 ({mama_up/len(mama_changes)*100:.1f}%)")
    print(f"  MAMA下降: {mama_down}回 ({mama_down/len(mama_changes)*100:.1f}%)")
    
    # 最後の20ポイントの詳細
    print(f"\n--- 最後の20ポイントの詳細 ---")
    for i in range(-20, 0):
        idx = len(mama_values) + i
        if idx >= 0:
            mama_val = mama_values[idx]
            fama_val = fama_values[idx]
            price_val = prices[idx]
            cross_status = "ABOVE" if mama_val > fama_val else "BELOW"
            print(f"  [{idx:4d}] Price: {price_val:7.2f}, MAMA: {mama_val:7.2f}, FAMA: {fama_val:7.2f}, {cross_status}")

if __name__ == "__main__":
    analyze_mama_values()