#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from position_sizing.position_sizing import FixedRatioSizing

def test_fixed_ratio_sizing():
    """固定比率ポジションサイジングのテスト"""
    
    # テストケース1: デフォルトパラメータ
    sizer = FixedRatioSizing()
    capital = 100000  # 10万
    price = 50000    # 5万
    
    position_size = sizer.calculate(capital, price)
    print("\n=== テストケース1: デフォルトパラメータ ===")
    print(f"総資金: {capital:,.0f}")
    print(f"価格: {price:,.0f}")
    print(f"ポジションサイズ: {position_size:.4f}")
    print(f"投資金額: {position_size * price:,.0f}")
    print(f"投資比率: {(position_size * price / capital) * 100:.2f}%")
    
    # テストケース2: カスタムパラメータ
    params = {
        'ratio': 0.05,  # 5%
        'min_position': 0.01,
        'max_position': 1.0
    }
    sizer = FixedRatioSizing(params)
    capital = 200000  # 20万
    price = 40000    # 4万
    
    position_size = sizer.calculate(capital, price)
    print("\n=== テストケース2: カスタムパラメータ ===")
    print(f"総資金: {capital:,.0f}")
    print(f"価格: {price:,.0f}")
    print(f"ポジションサイズ: {position_size:.4f}")
    print(f"投資金額: {position_size * price:,.0f}")
    print(f"投資比率: {(position_size * price / capital) * 100:.2f}%")
    
    # テストケース3: 最小ポジションサイズの制限
    params = {
        'ratio': 0.01,  # 1%
        'min_position': 0.1,
        'max_position': None
    }
    sizer = FixedRatioSizing(params)
    capital = 50000   # 5万
    price = 60000    # 6万
    
    position_size = sizer.calculate(capital, price)
    print("\n=== テストケース3: 最小ポジションサイズの制限 ===")
    print(f"総資金: {capital:,.0f}")
    print(f"価格: {price:,.0f}")
    print(f"ポジションサイズ: {position_size:.4f}")
    print(f"投資金額: {position_size * price:,.0f}")
    print(f"投資比率: {(position_size * price / capital) * 100:.2f}%")
    
    # テストケース4: 最大ポジションサイズの制限
    params = {
        'ratio': 0.1,   # 10%
        'min_position': 0.001,
        'max_position': 0.5
    }
    sizer = FixedRatioSizing(params)
    capital = 300000  # 30万
    price = 30000    # 3万
    
    position_size = sizer.calculate(capital, price)
    print("\n=== テストケース4: 最大ポジションサイズの制限 ===")
    print(f"総資金: {capital:,.0f}")
    print(f"価格: {price:,.0f}")
    print(f"ポジションサイズ: {position_size:.4f}")
    print(f"投資金額: {position_size * price:,.0f}")
    print(f"投資比率: {(position_size * price / capital) * 100:.2f}%")

if __name__ == '__main__':
    test_fixed_ratio_sizing() 