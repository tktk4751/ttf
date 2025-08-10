#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
X Position Sizingのテストスクリプト
"""

import numpy as np
import pandas as pd
import sys
import os

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from position_sizing.x_position_sizing import XATRPositionSizing
from position_sizing.position_sizing import PositionSizingParams


def generate_test_data(length: int = 200) -> pd.DataFrame:
    """テスト用のOHLCVデータを生成"""
    np.random.seed(42)
    base_price = 100.0
    trend = 0.001
    volatility = 0.02
    
    prices = [base_price]
    for i in range(1, length):
        change = trend + np.random.normal(0, volatility)
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # OHLC データの生成
    data = []
    for i, close in enumerate(prices):
        daily_range = abs(np.random.normal(0, volatility * close * 0.5))
        
        high = close + daily_range * np.random.uniform(0.3, 1.0)
        low = close - daily_range * np.random.uniform(0.3, 1.0)
        
        if i == 0:
            open_price = close
        else:
            gap = np.random.normal(0, volatility * close * 0.2)
            open_price = prices[i-1] + gap
        
        # 論理的整合性の確保
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.uniform(1000, 10000)
        })
    
    return pd.DataFrame(data)


def test_x_position_sizing():
    """X Position Sizingのテスト"""
    print("=== X Position Sizing テスト ===")
    
    # テストデータの生成
    test_data = generate_test_data(150)
    print(f"テストデータ生成完了: {len(test_data)}行")
    print(f"価格範囲: {test_data['close'].min():.2f} - {test_data['close'].max():.2f}")
    
    # テストケース1: Hyper ERトリガー
    print("\n1. Hyper ERトリガーのテスト")
    try:
        x_sizing_hyper_er = XATRPositionSizing(
            base_risk_ratio=0.02,
            unit=1.0,
            max_position_percent=0.5,
            leverage=1.0,
            trigger_type='hyper_er',
            apply_dynamic_adjustment=True,
            max_multiplier=5.0,
            min_multiplier=2.0,
            max_risk_ratio=0.03,
            min_risk_ratio=0.01
        )
        
        # ポジションサイジングパラメータ設定
        params = PositionSizingParams(
            entry_price=test_data['close'].iloc[-1],
            stop_loss_price=None,
            capital=10000.0,
            leverage=1.0,
            risk_per_trade=0.02,
            historical_data=test_data
        )
        
        result = x_sizing_hyper_er.calculate(params)
        
        print(f"  ポジションサイズ: ${result['position_size']:.2f}")
        print(f"  資産数量: {result['asset_quantity']:.4f}")
        print(f"  リスク金額: ${result['risk_amount']:.2f}")
        print(f"  X_ATR値: {result['x_atr_value']:.6f}")
        print(f"  ATR乗数: {result['atr_multiplier']:.2f}")
        print(f"  トリガー値: {result['trigger_value']:.4f}")
        print(f"  トリガー係数: {result['trigger_factor']:.4f}")
        print(f"  動的リスク比率: {result['risk_ratio']:.4f}")
        
    except Exception as e:
        print(f"  エラー: {e}")
    
    # テストケース2: Hyper Trend Indexトリガー
    print("\n2. Hyper Trend Indexトリガーのテスト")
    try:
        x_sizing_hyper_trend = XATRPositionSizing(
            base_risk_ratio=0.02,
            unit=1.0,
            max_position_percent=0.5,
            leverage=1.0,
            trigger_type='hyper_trend_index',
            apply_dynamic_adjustment=True,
            max_multiplier=5.0,
            min_multiplier=2.0,
            max_risk_ratio=0.03,
            min_risk_ratio=0.01
        )
        
        # ポジションサイジングパラメータ設定
        params = PositionSizingParams(
            entry_price=test_data['close'].iloc[-1],
            stop_loss_price=None,
            capital=10000.0,
            leverage=1.0,
            risk_per_trade=0.02,
            historical_data=test_data
        )
        
        result = x_sizing_hyper_trend.calculate(params)
        
        print(f"  ポジションサイズ: ${result['position_size']:.2f}")
        print(f"  資産数量: {result['asset_quantity']:.4f}")
        print(f"  リスク金額: ${result['risk_amount']:.2f}")
        print(f"  X_ATR値: {result['x_atr_value']:.6f}")
        print(f"  ATR乗数: {result['atr_multiplier']:.2f}")
        print(f"  トリガー値: {result['trigger_value']:.4f}")
        print(f"  トリガー係数: {result['trigger_factor']:.4f}")
        print(f"  動的リスク比率: {result['risk_ratio']:.4f}")
        
    except Exception as e:
        print(f"  エラー: {e}")
    
    # テストケース3: 固定リスクでのポジションサイジング
    print("\n3. 固定リスクでのポジションサイジングテスト")
    try:
        x_sizing = XATRPositionSizing(
            trigger_type='hyper_er',
            apply_dynamic_adjustment=True,
            fixed_risk_percent=0.01
        )
        
        result = x_sizing.calculate_position_size_with_fixed_risk(
            entry_price=test_data['close'].iloc[-1],
            capital=10000.0,
            historical_data=test_data,
            is_long=True,
            debug=True
        )
        
        print(f"  ポジションサイズ: ${result['position_size']:.2f}")
        print(f"  資産数量: {result['asset_quantity']:.4f}")
        print(f"  ストップロス価格: ${result['stop_loss_price']:.2f}")
        print(f"  リスク金額: ${result['risk_amount']:.2f}")
        print(f"  ストップロス幅: {result['stop_loss_distance']:.2f}")
        
    except Exception as e:
        print(f"  エラー: {e}")
    
    # テストケース4: IPositionManagerインターフェースのテスト
    print("\n4. IPositionManagerインターフェースのテスト")
    try:
        x_sizing = XATRPositionSizing(
            trigger_type='hyper_er',
            apply_dynamic_adjustment=False
        )
        
        position_size = x_sizing.calculate_position_size(
            price=test_data['close'].iloc[-1],
            capital=10000.0
        )
        
        print(f"  簡易ポジションサイズ: ${position_size:.2f}")
        print(f"  新規ポジション可能: {x_sizing.can_enter()}")
        
    except Exception as e:
        print(f"  エラー: {e}")
    
    print("\n=== テスト完了 ===")


if __name__ == "__main__":
    test_x_position_sizing()