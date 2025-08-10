#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
シンプルなスムーズ化価格ソーステスト
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 直接インポート
from indicators.price_source import PriceSource

def create_test_data(length=100):
    """テスト用のOHLCデータを生成"""
    np.random.seed(42)
    
    # 価格のトレンドとノイズを生成
    trend = np.linspace(100, 120, length)
    noise = np.random.normal(0, 2, length)
    
    # OHLCデータを生成
    close = trend + noise
    high = close + np.abs(np.random.normal(0, 1, length))
    low = close - np.abs(np.random.normal(0, 1, length))
    open_price = np.roll(close, 1)
    open_price[0] = close[0]
    
    # DataFrameに変換
    dates = [datetime.now() - timedelta(days=length-i) for i in range(length)]
    df = pd.DataFrame({
        'date': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': np.random.randint(1000, 10000, length)
    })
    
    return df

def main():
    """メイン処理"""
    print("スムーズ化価格ソーステスト開始...")
    
    # テストデータを作成
    data = create_test_data(200)
    print(f"テストデータ作成完了: {len(data)}行")
    
    # 利用可能なソースを表示
    sources = PriceSource.get_available_sources()
    print("\n利用可能な価格ソース:")
    for src_type, desc in sorted(sources.items()):
        print(f"  {src_type}: {desc}")
    
    # 基本ソースをテスト
    test_sources = ['close', 'hlc3', 'hl2']
    
    for src_type in test_sources:
        try:
            values = PriceSource.calculate_source(data, src_type)
            print(f"\n{src_type}: 成功 - {len(values)}データポイント")
        except Exception as e:
            print(f"\n{src_type}: エラー - {str(e)}")
    
    # UKFソースをテスト（利用可能な場合）
    if 'ukf_hlc3' in sources:
        try:
            ukf_values = PriceSource.calculate_source(data, 'ukf_hlc3')
            print(f"\nukf_hlc3: 成功 - {len(ukf_values)}データポイント")
        except Exception as e:
            print(f"\nukf_hlc3: エラー - {str(e)}")
    
    # スムーザーソースをテスト（利用可能な場合）
    smoother_sources = ['us_hlc3', 'ss_hlc3', 'us_close', 'ss_close']
    
    for src_type in smoother_sources:
        if src_type in sources:
            try:
                values = PriceSource.calculate_source(data, src_type)
                print(f"\n{src_type}: 成功 - {len(values)}データポイント")
                
                # ノイズ削減率を計算
                base_type = src_type[3:]  # 'us_' or 'ss_' を除去
                original = PriceSource.calculate_source(data, base_type)
                noise_reduction = 1 - np.std(np.diff(values)) / np.std(np.diff(original))
                print(f"  ノイズ削減率: {noise_reduction:.1%}")
                
            except Exception as e:
                print(f"\n{src_type}: エラー - {str(e)}")
    
    # 簡単な可視化
    if 'us_hlc3' in sources and 'ss_hlc3' in sources:
        try:
            plt.figure(figsize=(12, 6))
            
            # データを計算
            original = PriceSource.calculate_source(data, 'hlc3')
            us_smoothed = PriceSource.calculate_source(data, 'us_hlc3')
            ss_smoothed = PriceSource.calculate_source(data, 'ss_hlc3')
            
            # プロット
            x = range(len(data))
            plt.plot(x[-100:], original[-100:], 'b-', alpha=0.5, label='Original HLC3')
            plt.plot(x[-100:], us_smoothed[-100:], 'r-', linewidth=2, label='Ultimate Smoother')
            plt.plot(x[-100:], ss_smoothed[-100:], 'g-', linewidth=2, label='Super Smoother')
            
            plt.title('Price Source Smoothing Comparison (Last 100 points)')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('simple_smoother_test.png')
            print("\nチャートを simple_smoother_test.png に保存しました。")
            
        except Exception as e:
            print(f"\nチャート作成エラー: {str(e)}")

if __name__ == "__main__":
    main()