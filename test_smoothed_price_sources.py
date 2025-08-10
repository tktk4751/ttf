#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
スムーズ化された価格ソースのテストスクリプト
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# インジケーターのインポート
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

def test_smoothed_sources():
    """スムーズ化された価格ソースをテスト"""
    # テストデータを作成
    data = create_test_data(200)
    
    # 利用可能なソースタイプを取得
    available_sources = PriceSource.get_available_sources()
    print("利用可能な価格ソース:")
    for src_type, description in sorted(available_sources.items()):
        print(f"  {src_type}: {description}")
    
    # スムーザーソースタイプをフィルタリング
    smoother_sources = {k: v for k, v in available_sources.items() 
                       if k.startswith('us_') or k.startswith('ss_')}
    
    if not smoother_sources:
        print("\nスムーザーソースが利用できません。")
        return
    
    print(f"\n{len(smoother_sources)}個のスムーザーソースが利用可能です。")
    
    # 各スムーザーソースを計算してプロット
    fig, axes = plt.subplots(len(smoother_sources), 1, figsize=(12, 4 * len(smoother_sources)))
    if len(smoother_sources) == 1:
        axes = [axes]
    
    for idx, (src_type, description) in enumerate(sorted(smoother_sources.items())):
        ax = axes[idx]
        
        try:
            # スムーズ化されたソースを計算
            smoothed = PriceSource.calculate_source(data, src_type)
            
            # 元のソースタイプを特定
            base_type = src_type[3:]  # 'us_' or 'ss_' を除去
            original = PriceSource.calculate_source(data, base_type)
            
            # プロット
            x = range(len(data))
            ax.plot(x, original, 'b-', alpha=0.5, linewidth=1, label=f'Original {base_type}')
            ax.plot(x, smoothed, 'r-', linewidth=2, label=description)
            
            ax.set_title(f'{description} vs Original {base_type.upper()}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Price')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # ノイズ削減率を計算
            noise_reduction = 1 - np.std(np.diff(smoothed)) / np.std(np.diff(original))
            ax.text(0.02, 0.98, f'Noise Reduction: {noise_reduction:.1%}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            print(f"\n{src_type}: 計算成功")
            print(f"  - データ長: {len(smoothed)}")
            print(f"  - ノイズ削減率: {noise_reduction:.1%}")
            
        except Exception as e:
            print(f"\n{src_type}: エラー - {str(e)}")
            ax.text(0.5, 0.5, f'Error: {str(e)}', 
                   transform=ax.transAxes, ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig('smoothed_price_sources_test.png', dpi=150)
    print(f"\nチャートを smoothed_price_sources_test.png に保存しました。")
    
    # 比較テスト: Ultimate Smoother vs Super Smoother
    if 'us_hlc3' in smoother_sources and 'ss_hlc3' in smoother_sources:
        plt.figure(figsize=(12, 6))
        
        us_hlc3 = PriceSource.calculate_source(data, 'us_hlc3')
        ss_hlc3 = PriceSource.calculate_source(data, 'ss_hlc3')
        original_hlc3 = PriceSource.calculate_source(data, 'hlc3')
        
        x = range(len(data))
        plt.plot(x, original_hlc3, 'b-', alpha=0.3, linewidth=1, label='Original HLC3')
        plt.plot(x, us_hlc3, 'r-', linewidth=2, label='Ultimate Smoother HLC3')
        plt.plot(x, ss_hlc3, 'g-', linewidth=2, label='Super Smoother HLC3')
        
        plt.title('Ultimate Smoother vs Super Smoother (HLC3, 10期間)')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 統計情報
        us_noise_reduction = 1 - np.std(np.diff(us_hlc3)) / np.std(np.diff(original_hlc3))
        ss_noise_reduction = 1 - np.std(np.diff(ss_hlc3)) / np.std(np.diff(original_hlc3))
        
        stats_text = (f'Ultimate Smoother Noise Reduction: {us_noise_reduction:.1%}\n'
                     f'Super Smoother Noise Reduction: {ss_noise_reduction:.1%}')
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('us_vs_ss_comparison.png', dpi=150)
        print(f"\n比較チャートを us_vs_ss_comparison.png に保存しました。")

if __name__ == "__main__":
    test_smoothed_sources()