#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
直接スムーザーインポートテスト
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 直接インポートを試す
try:
    from indicators.smoother.ultimate_smoother import UltimateSmoother
    print("Ultimate Smoother import: 成功")
except ImportError as e:
    print(f"Ultimate Smoother import: 失敗 - {e}")

try:
    from indicators.smoother.super_smoother import SuperSmoother
    print("Super Smoother import: 成功")
except ImportError as e:
    print(f"Super Smoother import: 失敗 - {e}")

# PriceSourceをインポート
try:
    from indicators.price_source import PriceSource
    print("PriceSource import: 成功")
    
    # モジュール状態を確認
    print(f"\nPriceSource内のUltimateSmoother: {PriceSource.__dict__.get('UltimateSmoother', 'なし')}")
    print(f"PriceSource内のSuperSmoother: {PriceSource.__dict__.get('SuperSmoother', 'なし')}")
    
    # price_sourceモジュール内の変数を確認
    import indicators.price_source as ps_module
    print(f"\nprice_sourceモジュールのUltimateSmoother: {ps_module.UltimateSmoother}")
    print(f"price_sourceモジュールのSuperSmoother: {ps_module.SuperSmoother}")
    
except ImportError as e:
    print(f"PriceSource import: 失敗 - {e}")

# 簡単なテスト
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

# テストデータで確認
data = create_test_data(50)

# Ultimate Smootherを直接使用
if 'UltimateSmoother' in locals():
    try:
        us = UltimateSmoother(period=10.0, src_type='hlc3', period_mode='fixed')
        result = us.calculate(data)
        print(f"\nUltimate Smoother直接使用: 成功 - {len(result.values)}データポイント")
    except Exception as e:
        print(f"\nUltimate Smoother直接使用: エラー - {e}")

# Super Smootherを直接使用
if 'SuperSmoother' in locals():
    try:
        ss = SuperSmoother(length=10, num_poles=2, src_type='hlc3')
        result = ss.calculate(data)
        print(f"Super Smoother直接使用: 成功 - {len(result.values)}データポイント")
    except Exception as e:
        print(f"Super Smoother直接使用: エラー - {e}")