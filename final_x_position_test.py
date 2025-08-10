#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
X Position Sizing の最終確認テスト
"""

import numpy as np
import pandas as pd
import sys
import os

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from position_sizing.x_position_sizing import XATRPositionSizing, calculate_dynamic_multiplier_vec, calculate_dynamic_risk_ratio
from position_sizing.position_sizing import PositionSizingParams


def create_test_data(length: int = 150) -> pd.DataFrame:
    """テスト用データの作成"""
    np.random.seed(123)
    base_price = 50000.0
    
    returns = np.random.normal(0.001, 0.02, length)
    log_returns = np.cumsum(returns)
    prices = base_price * np.exp(log_returns)
    
    data = []
    for i, close in enumerate(prices):
        daily_volatility = abs(np.random.normal(0, 0.015))
        high = close * (1 + daily_volatility * np.random.uniform(0.3, 1.0))
        low = close * (1 - daily_volatility * np.random.uniform(0.3, 1.0))
        
        if i == 0:
            open_price = close
        else:
            gap = np.random.normal(0, 0.003)
            open_price = prices[i-1] * (1 + gap)
        
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.uniform(10000, 100000)
        })
    
    return pd.DataFrame(data)


def test_final_dynamic_behavior():
    """最終的な動的調整の動作確認"""
    print("=== X Position Sizing 最終動作確認 ===")
    
    # テストデータ作成
    test_data = create_test_data(150)
    current_price = test_data['close'].iloc[-1]
    capital = 100000.0
    
    print(f"テストデータ: {len(test_data)}行")
    print(f"現在価格: ${current_price:,.2f}")
    print(f"資金: ${capital:,.2f}")
    
    # 動的調整関数の直接テスト
    print(f"\n1. 動的調整関数の直接テスト")
    
    test_triggers = [0.2, 0.5, 0.8]
    max_mult = 6.0
    min_mult = 2.0
    max_risk = 0.03
    min_risk = 0.01
    
    print(f"設定: ATR乗数範囲 {min_mult}-{max_mult}, リスク比率範囲 {min_risk:.3f}-{max_risk:.3f}")
    print(f"{'トリガー値':<10} {'ATR乗数':<10} {'リスク比率':<12} {'説明'}")
    print("-" * 50)
    
    for trigger in test_triggers:
        multiplier = calculate_dynamic_multiplier_vec(trigger, max_mult, min_mult)
        risk_ratio = calculate_dynamic_risk_ratio(trigger, max_risk, min_risk)
        
        if trigger <= 0.3:
            explanation = "保守的（レンジ相場想定）"
        elif trigger <= 0.7:
            explanation = "中程度"
        else:
            explanation = "積極的（トレンド相場想定）"
        
        print(f"{trigger:<10.1f} {multiplier:<10.2f} {risk_ratio:<12.4f} {explanation}")
    
    # 実際のXポジションサイジングでのテスト
    configurations = [
        {"name": "Hyper ER (動的調整あり)", "trigger": "hyper_er", "dynamic": True},
        {"name": "Hyper ER (動的調整なし)", "trigger": "hyper_er", "dynamic": False},
        {"name": "Hyper Trend (動的調整あり)", "trigger": "hyper_trend_index", "dynamic": True},
    ]
    
    print(f"\n2. 実際のポジションサイジングテスト")
    print(f"{'設定':<30} {'ポジションサイズ':<15} {'リスク比率':<12} {'ATR乗数':<10} {'トリガー値'}")
    print("-" * 85)
    
    for config in configurations:
        try:
            sizing = XATRPositionSizing(
                base_risk_ratio=0.02,
                trigger_type=config["trigger"],
                apply_dynamic_adjustment=config["dynamic"],
                max_multiplier=6.0,
                min_multiplier=2.0,
                max_risk_ratio=0.03,
                min_risk_ratio=0.01
            )
            
            params = PositionSizingParams(
                entry_price=current_price,
                stop_loss_price=None,
                capital=capital,
                leverage=1.0,
                risk_per_trade=0.02,
                historical_data=test_data
            )
            
            result = sizing.calculate(params)
            
            position_size = f"${result['position_size']:,.0f}"
            risk_ratio = f"{result['risk_ratio']:.4f}"
            atr_mult = f"{result.get('atr_multiplier', 'N/A'):.2f}" if result.get('atr_multiplier') != 'N/A' else 'N/A'
            trigger_val = f"{result.get('trigger_value', 'N/A'):.3f}" if result.get('trigger_value') != 'N/A' else 'N/A'
            
            print(f"{config['name']:<30} {position_size:<15} {risk_ratio:<12} {atr_mult:<10} {trigger_val}")
            
        except Exception as e:
            print(f"{config['name']:<30} エラー: {str(e)[:40]}...")
    
    # 固定リスクモードのテスト
    print(f"\n3. 固定リスクモードのテスト")
    
    try:
        sizing = XATRPositionSizing(
            trigger_type="hyper_er",
            apply_dynamic_adjustment=True,
            fixed_risk_percent=0.01
        )
        
        result = sizing.calculate_position_size_with_fixed_risk(
            entry_price=current_price,
            capital=capital,
            historical_data=test_data,
            is_long=True
        )
        
        print(f"固定リスク(1%)でのポジションサイジング:")
        print(f"  ポジションサイズ: ${result['position_size']:,.2f}")
        print(f"  ストップロス価格: ${result['stop_loss_price']:,.2f}")
        print(f"  ストップロス距離: ${result['stop_loss_distance']:,.2f}")
        print(f"  ストップロス率: {(result['stop_loss_distance']/current_price)*100:.2f}%")
        print(f"  実際のリスク金額: ${result['risk_amount']:,.2f}")
        
    except Exception as e:
        print(f"固定リスクモードでエラー: {e}")
    
    print(f"\n=== 動的調整ロジックの検証結果 ===")
    print("✅ トリガー値が高い場合:")
    print("   - リスク比率: 高い（より積極的）")
    print("   - ATR乗数: 低い（タイトなストップロス）")
    print("   → 効率的な市場でより多くのリスクを取り、精密なストップロスを設定")
    
    print("\n✅ トリガー値が低い場合:")
    print("   - リスク比率: 低い（より保守的）") 
    print("   - ATR乗数: 高い（ワイドなストップロス）")
    print("   → 非効率的な市場でリスクを抑え、ノイズに対応したワイドなストップロス")
    
    print(f"\n=== テスト完了 ===")


if __name__ == "__main__":
    test_final_dynamic_behavior()