#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from indicators.ultimate_adaptive_ma import UltimateAdaptiveMA
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource
import yaml

def debug_ultimate_adaptive_ma():
    """Ultimate Adaptive MAの計算値をデバッグする"""
    
    # 設定ファイルからデータを読み込む
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("config.yamlファイルが見つかりません")
        return
    
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
    first_symbol = next(iter(processed_data))
    data = processed_data[first_symbol]
    
    print(f"データ読み込み完了: {first_symbol}")
    print(f"データ数: {len(data)}")
    print(f"価格範囲: {data['close'].min():.2f} - {data['close'].max():.2f}")
    
    # 最新の100データポイントを使用してテスト
    test_data = data.tail(100).copy()
    print(f"\nテストデータ範囲: {test_data['close'].min():.2f} - {test_data['close'].max():.2f}")
    
    # Ultimate Adaptive MAを計算
    print("\nUltimate Adaptive MAを計算中...")
    ultimate_ma = UltimateAdaptiveMA(
        base_period=21,
        adaptation_strength=1.0,
        src_type='hlc3'
    )
    
    try:
        result = ultimate_ma.calculate(test_data)
        
        # 各種値を取得
        ma_values = ultimate_ma.get_values()
        base_ma = ultimate_ma.get_base_ma()
        adaptive_factor = ultimate_ma.get_adaptive_factor()
        trend_strength = ultimate_ma.get_trend_strength()
        
        print(f"\n=== 計算結果の分析 ===")
        
        # 価格データの統計
        prices = test_data['close'].values
        print(f"価格データ統計:")
        print(f"  最小値: {prices.min():.6f}")
        print(f"  最大値: {prices.max():.6f}")
        print(f"  平均値: {prices.mean():.6f}")
        print(f"  標準偏差: {prices.std():.6f}")
        
        # Ultimate MA統計
        if ma_values is not None:
            valid_ma = ma_values[~np.isnan(ma_values)]
            print(f"\nUltimate MA統計:")
            print(f"  有効データ数: {len(valid_ma)} / {len(ma_values)}")
            if len(valid_ma) > 0:
                print(f"  最小値: {valid_ma.min():.6f}")
                print(f"  最大値: {valid_ma.max():.6f}")
                print(f"  平均値: {valid_ma.mean():.6f}")
                print(f"  標準偏差: {valid_ma.std():.6f}")
                
                # 価格との乖離度をチェック
                price_mean = prices.mean()
                ma_mean = valid_ma.mean()
                deviation_ratio = abs(ma_mean - price_mean) / price_mean
                print(f"  価格からの乖離率: {deviation_ratio:.2%}")
                
                if deviation_ratio > 0.5:  # 50%以上の乖離は異常
                    print(f"  ⚠️  WARNING: 価格から大きく乖離しています！")
        
        # ベース移動平均統計
        if base_ma is not None:
            valid_base = base_ma[~np.isnan(base_ma)]
            print(f"\nベース移動平均統計:")
            print(f"  有効データ数: {len(valid_base)} / {len(base_ma)}")
            if len(valid_base) > 0:
                print(f"  最小値: {valid_base.min():.6f}")
                print(f"  最大値: {valid_base.max():.6f}")
                print(f"  平均値: {valid_base.mean():.6f}")
                print(f"  標準偏差: {valid_base.std():.6f}")
        
        # 適応ファクター統計
        if adaptive_factor is not None:
            valid_factor = adaptive_factor[~np.isnan(adaptive_factor)]
            print(f"\n適応ファクター統計:")
            print(f"  有効データ数: {len(valid_factor)} / {len(adaptive_factor)}")
            if len(valid_factor) > 0:
                print(f"  最小値: {valid_factor.min():.6f}")
                print(f"  最大値: {valid_factor.max():.6f}")
                print(f"  平均値: {valid_factor.mean():.6f}")
        
        # 詳細な値の表示（最新10データポイント）
        print(f"\n=== 最新10データポイントの詳細 ===")
        print(f"{'Index':<6} {'Price':<12} {'Base MA':<12} {'Ultimate MA':<12} {'Factor':<8}")
        print("-" * 60)
        
        for i in range(max(0, len(test_data) - 10), len(test_data)):
            price = prices[i] if i < len(prices) else np.nan
            base = base_ma[i] if base_ma is not None and i < len(base_ma) else np.nan
            ultimate = ma_values[i] if ma_values is not None and i < len(ma_values) else np.nan
            factor = adaptive_factor[i] if adaptive_factor is not None and i < len(adaptive_factor) else np.nan
            
            print(f"{i:<6} {price:<12.6f} {base:<12.6f} {ultimate:<12.6f} {factor:<8.4f}")
        
        # 簡単なプロット作成
        print(f"\n簡易チャートを作成中...")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 価格とMA
        ax1.plot(prices, label='Price', color='black', linewidth=1)
        if base_ma is not None:
            ax1.plot(base_ma, label='Base MA', color='blue', alpha=0.7)
        if ma_values is not None:
            ax1.plot(ma_values, label='Ultimate Adaptive MA', color='red', linewidth=2)
        ax1.set_title('Price vs Moving Averages')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 適応ファクター
        if adaptive_factor is not None:
            ax2.plot(adaptive_factor, label='Adaptive Factor', color='green')
            ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            ax2.set_title('Adaptive Factor')
            ax2.set_ylabel('Factor')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('debug_ultimate_adaptive_ma.png', dpi=150)
        print("デバッグチャートを保存: debug_ultimate_adaptive_ma.png")
        
    except Exception as e:
        print(f"計算エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_ultimate_adaptive_ma()