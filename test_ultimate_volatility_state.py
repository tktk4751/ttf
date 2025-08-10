#!/usr/bin/env python3

import numpy as np
import pandas as pd
from indicators.ultimate_volatility_state import UltimateVolatilityState

def generate_test_data(length: int = 1000) -> pd.DataFrame:
    """テスト用の価格データを生成"""
    np.random.seed(42)
    
    # 基本価格
    base_price = 100.0
    prices = [base_price]
    
    # 高ボラティリティ期間と低ボラティリティ期間を交互に
    for i in range(1, length):
        if i % 200 < 100:  # 高ボラティリティ期間
            change = np.random.normal(0, 0.03)  # 3%の標準偏差
        else:  # 低ボラティリティ期間
            change = np.random.normal(0, 0.005)  # 0.5%の標準偏差
        
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 0.01))  # 最小価格を設定
    
    # OHLC データを生成
    data = []
    for i, close in enumerate(prices):
        if i == 0:
            open_price = close
            high = close
            low = close
        else:
            open_price = prices[i-1]
            volatility = abs(close - open_price) / open_price if open_price > 0 else 0
            high = max(open_price, close) * (1 + volatility * 0.5)
            low = min(open_price, close) * (1 - volatility * 0.5)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.randint(1000, 10000)
        })
    
    return pd.DataFrame(data)

def test_indicator():
    """インジケーターの基本テスト"""
    print("=== Ultimate Volatility State Indicator Test ===")
    
    # テストデータ生成
    data = generate_test_data(500)
    print(f"Generated test data with {len(data)} records")
    
    # インジケーター作成
    indicator = UltimateVolatilityState(
        period=21,
        threshold=0.5,
        zscore_period=50,
        adaptive_threshold=True
    )
    
    # 計算実行
    try:
        result = indicator.calculate(data)
        print(f"Calculation completed successfully")
        print(f"State array length: {len(result.state)}")
        print(f"High volatility count: {np.sum(result.state)}")
        print(f"Low volatility count: {np.sum(result.state == 0)}")
        
        # 統計情報
        non_zero_indices = np.where(result.state > 0)[0]
        if len(non_zero_indices) > 0:
            print(f"First high volatility at index: {non_zero_indices[0]}")
            print(f"Last high volatility at index: {non_zero_indices[-1]}")
        
        # 確率の統計
        valid_prob = result.probability[result.probability > 0]
        if len(valid_prob) > 0:
            print(f"Average probability: {np.mean(valid_prob):.3f}")
            print(f"Max probability: {np.max(valid_prob):.3f}")
            print(f"Min probability: {np.min(valid_prob):.3f}")
        
        # コンポーネントの確認
        components = result.components
        print(f"Components available: {list(components.keys())}")
        
        # 最新の状態
        if len(result.state) > 0:
            latest_state = result.state[-1]
            latest_prob = result.probability[-1]
            print(f"Latest state: {'High' if latest_state == 1 else 'Low'} volatility")
            print(f"Latest probability: {latest_prob:.3f}")
        
        # メソッドのテスト
        print(f"Is high volatility: {indicator.is_high_volatility()}")
        print(f"Is low volatility: {indicator.is_low_volatility()}")
        
        print("✓ All tests passed!")
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_indicator()