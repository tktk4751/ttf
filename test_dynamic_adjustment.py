#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
動的調整の動作確認テスト
"""

import numpy as np
import pandas as pd
import sys
import os

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from position_sizing.x_position_sizing import calculate_dynamic_multiplier_vec, calculate_dynamic_risk_ratio


def test_dynamic_functions():
    """動的計算関数のテスト"""
    print("=== 動的調整関数の動作確認 ===")
    
    # テストケース: トリガー値の範囲
    trigger_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    max_multiplier = 6.0
    min_multiplier = 2.0
    max_risk = 0.03
    min_risk = 0.01
    
    print("\n動的ATR乗数の計算:")
    print(f"設定: 最大乗数={max_multiplier}, 最小乗数={min_multiplier}")
    print("期待される動作: トリガー値が高いほど乗数は小さくなる")
    print("トリガー値 -> ATR乗数")
    for trigger in trigger_values:
        multiplier = calculate_dynamic_multiplier_vec(trigger, max_multiplier, min_multiplier)
        print(f"  {trigger:.1f}     ->  {multiplier:.2f}")
    
    print("\n動的リスク比率の計算:")
    print(f"設定: 最大リスク={max_risk:.3f}, 最小リスク={min_risk:.3f}")
    print("期待される動作: トリガー値が高いほどリスク比率は大きくなる")
    print("トリガー値 -> リスク比率")
    for trigger in trigger_values:
        risk_ratio = calculate_dynamic_risk_ratio(trigger, max_risk, min_risk)
        print(f"  {trigger:.1f}     ->  {risk_ratio:.4f}")
    
    print("\n=== 論理確認 ===")
    print("✓ トリガー値が高い（市場が効率的/トレンド的）:")
    print("  - リスク比率は高く（積極的）")
    print("  - ATR乗数は低く（タイトなストップロス）")
    print("✓ トリガー値が低い（市場が非効率的/レンジ的）:")
    print("  - リスク比率は低く（保守的）")
    print("  - ATR乗数は高く（ワイドなストップロス）")
    
    # エッジケースのテスト
    print("\n=== エッジケースのテスト ===")
    
    # NaN値のテスト
    nan_multiplier = calculate_dynamic_multiplier_vec(np.nan, max_multiplier, min_multiplier)
    nan_risk = calculate_dynamic_risk_ratio(np.nan, max_risk, min_risk)
    print(f"NaN入力: 乗数={nan_multiplier:.2f}, リスク={nan_risk:.4f}")
    
    # 範囲外の値のテスト
    extreme_values = [-0.5, 1.5, 2.0]
    print("範囲外の値のテスト:")
    for val in extreme_values:
        multiplier = calculate_dynamic_multiplier_vec(val, max_multiplier, min_multiplier)
        risk_ratio = calculate_dynamic_risk_ratio(val, max_risk, min_risk)
        print(f"  {val:.1f}: 乗数={multiplier:.2f}, リスク={risk_ratio:.4f}")


def test_real_scenario():
    """実際のシナリオでの動作確認"""
    print("\n=== 実際のシナリオでの動作確認 ===")
    
    scenarios = [
        {"name": "強いトレンド市場", "hyper_er": 0.8, "hyper_trend": 0.9},
        {"name": "中程度のトレンド市場", "hyper_er": 0.6, "hyper_trend": 0.7},
        {"name": "レンジ市場", "hyper_er": 0.3, "hyper_trend": 0.4},
        {"name": "強いレンジ市場", "hyper_er": 0.1, "hyper_trend": 0.2},
    ]
    
    max_multiplier = 6.0
    min_multiplier = 2.0
    max_risk = 0.03
    min_risk = 0.01
    
    print(f"{'シナリオ':<15} {'インディケーター':<10} {'トリガー値':<8} {'ATR乗数':<8} {'リスク比率':<10}")
    print("-" * 65)
    
    for scenario in scenarios:
        for indicator, value in [("Hyper ER", scenario["hyper_er"]), ("Hyper Trend", scenario["hyper_trend"])]:
            multiplier = calculate_dynamic_multiplier_vec(value, max_multiplier, min_multiplier)
            risk_ratio = calculate_dynamic_risk_ratio(value, max_risk, min_risk)
            
            print(f"{scenario['name']:<15} {indicator:<10} {value:<8.1f} {multiplier:<8.2f} {risk_ratio:<10.4f}")


if __name__ == "__main__":
    test_dynamic_functions()
    test_real_scenario()