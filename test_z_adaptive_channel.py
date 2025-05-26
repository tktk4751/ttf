#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
import os

# indicators ディレクトリを Python パスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'indicators'))

from indicators.z_adaptive_channel import ZAdaptiveChannel

def create_test_data(length=100):
    """テスト用のOHLCデータを生成"""
    np.random.seed(42)
    price = 100 + np.cumsum(np.random.randn(length) * 0.5)
    
    # OHLCデータを生成
    high = price + np.abs(np.random.randn(length) * 0.2)
    low = price - np.abs(np.random.randn(length) * 0.2)
    open_price = price + np.random.randn(length) * 0.1
    close = price
    
    data = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close
    })
    
    return data

def test_adaptive_vs_simple():
    """アダプティブとシンプルな動的乗数計算方法を比較"""
    print("Zアダプティブチャネル - 乗数計算方法比較テスト")
    print("=" * 50)
    
    # テストデータ生成
    data = create_test_data(50)
    
    # アダプティブ方式
    adaptive_channel = ZAdaptiveChannel(
        multiplier_method='adaptive',
        multiplier_source='cer',
        max_max_multiplier=8.0,
        min_max_multiplier=3.0,
        max_min_multiplier=1.5,
        min_min_multiplier=0.5
    )
    
    # シンプル方式  
    simple_channel = ZAdaptiveChannel(
        multiplier_method='simple',
        multiplier_source='cer'
    )
    
    # 計算実行
    print("アダプティブ方式の計算...")
    adaptive_result = adaptive_channel.calculate(data)
    adaptive_bands = adaptive_channel.get_bands()
    adaptive_multiplier = adaptive_channel.get_dynamic_multiplier()
    adaptive_trigger = adaptive_channel.get_multiplier_trigger()
    
    print("シンプル方式の計算...")
    simple_result = simple_channel.calculate(data)
    simple_bands = simple_channel.get_bands()
    simple_multiplier = simple_channel.get_dynamic_multiplier()
    simple_trigger = simple_channel.get_multiplier_trigger()
    
    # 結果比較（最後の10個の値）
    print("\n結果比較（最後の10個の値）:")
    print("-" * 50)
    
    print("トリガー値:")
    print(f"アダプティブ: {adaptive_trigger[-10:].round(4)}")
    print(f"シンプル    : {simple_trigger[-10:].round(4)}")
    
    print("\n動的乗数:")
    print(f"アダプティブ: {adaptive_multiplier[-10:].round(4)}")
    print(f"シンプル    : {simple_multiplier[-10:].round(4)}")
    
    print("\n中心線:")
    print(f"アダプティブ: {adaptive_bands[0][-10:].round(4)}")
    print(f"シンプル    : {simple_bands[0][-10:].round(4)}")
    
    print("\n上限バンド:")
    print(f"アダプティブ: {adaptive_bands[1][-10:].round(4)}")
    print(f"シンプル    : {simple_bands[1][-10:].round(4)}")
    
    print("\n下限バンド:")
    print(f"アダプティブ: {adaptive_bands[2][-10:].round(4)}")
    print(f"シンプル    : {simple_bands[2][-10:].round(4)}")
    
    # シンプル方式の計算ロジック確認
    print("\nシンプル方式の計算ロジック確認:")
    print("動的乗数 = 16 - (CER*8) - (XTRENDINDEX*8)")
    
    # CERとXTRENDINDEXの値も取得
    simple_cer = simple_channel.get_efficiency_ratio()
    # XTRENDINDEXは直接取得できないので、simple_channelからx_trend_indexを使用
    x_trend_result = simple_channel.x_trend_index.calculate(data)
    simple_x_trend = x_trend_result.values
    
    for i in range(-5, 0):
        cer = simple_cer[i]
        x_trend = simple_x_trend[i]
        multiplier = simple_multiplier[i]
        expected = 16.0 - (abs(cer) * 8.0) - (x_trend * 8.0)
        print(f"CER: {cer:.4f}, XTREND: {x_trend:.4f}, 乗数: {multiplier:.4f}, 期待値: {expected:.4f}, 差: {abs(multiplier - expected):.6f}")

if __name__ == "__main__":
    test_adaptive_vs_simple() 