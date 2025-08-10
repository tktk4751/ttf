#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Super SmootherのNaN問題を詳しく調べるデバッグスクリプト
"""

import numpy as np
import pandas as pd
import sys
import os

# インジケーターをインポートするためのパス設定
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from indicators.smoother.super_smoother import SuperSmoother, calculate_super_smoother_2pole

def test_super_smoother_with_different_inputs():
    """様々な入力でSuper Smootherをテストする"""
    print("=== Super Smoother NaN問題デバッグ ===")
    
    # テスト1: シンプルな価格データ
    print("\n--- テスト1: シンプルな価格データ ---")
    simple_prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 104.0, 103.0, 102.0, 101.0] * 10)
    simple_df = pd.DataFrame({'close': simple_prices})
    
    print(f"入力データ: 長さ={len(simple_prices)}, 範囲={np.min(simple_prices)}-{np.max(simple_prices)}")
    print(f"NaN数: {np.isnan(simple_prices).sum()}")
    
    try:
        smoother = SuperSmoother(src_type='close', length=8, num_poles=2)
        result = smoother.calculate(simple_df)
        
        if hasattr(result, 'values'):
            values = result.values
            print(f"結果: 長さ={len(values)}, 有効値数={np.sum(~np.isnan(values))}")
            if np.sum(~np.isnan(values)) > 0:
                valid = values[~np.isnan(values)]
                print(f"範囲: {np.min(valid):.4f} - {np.max(valid):.4f}")
                print(f"最初の10値: {values[:10]}")
            else:
                print("❌ すべてNaN")
        else:
            print("❌ valuesアトリビュートなし")
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
    
    # テスト2: ER値のような小さな値（0-1の範囲）
    print("\n--- テスト2: ER値ライクデータ（0-1範囲） ---")
    er_like_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.8, 0.7, 0.6, 0.5] * 10)
    er_df = pd.DataFrame({'close': er_like_values})
    
    print(f"入力データ: 長さ={len(er_like_values)}, 範囲={np.min(er_like_values)}-{np.max(er_like_values)}")
    print(f"NaN数: {np.isnan(er_like_values).sum()}")
    
    try:
        smoother = SuperSmoother(src_type='close', length=8, num_poles=2)
        result = smoother.calculate(er_df)
        
        if hasattr(result, 'values'):
            values = result.values
            print(f"結果: 長さ={len(values)}, 有効値数={np.sum(~np.isnan(values))}")
            if np.sum(~np.isnan(values)) > 0:
                valid = values[~np.isnan(values)]
                print(f"範囲: {np.min(valid):.4f} - {np.max(valid):.4f}")
                print(f"最初の10値: {values[:10]}")
            else:
                print("❌ すべてNaN")
                # デバッグのために直接Numba関数を呼び出してみる
                print("直接Numba関数をテスト:")
                direct_result = calculate_super_smoother_2pole(er_like_values, 8)
                print(f"直接結果: 有効値数={np.sum(~np.isnan(direct_result))}")
                if np.sum(~np.isnan(direct_result)) > 0:
                    valid_direct = direct_result[~np.isnan(direct_result)]
                    print(f"直接範囲: {np.min(valid_direct):.4f} - {np.max(valid_direct):.4f}")
                    print(f"直接最初の10値: {direct_result[:10]}")
        else:
            print("❌ valuesアトリビュートなし")
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
    
    # テスト3: NaNを含むデータ
    print("\n--- テスト3: NaNを含むデータ ---")
    nan_data = er_like_values.copy()
    nan_data[:10] = np.nan  # 最初の10値をNaNに
    nan_df = pd.DataFrame({'close': nan_data})
    
    print(f"入力データ: 長さ={len(nan_data)}, NaN数={np.isnan(nan_data).sum()}")
    valid_data = nan_data[~np.isnan(nan_data)]
    if len(valid_data) > 0:
        print(f"有効値: 数={len(valid_data)}, 範囲={np.min(valid_data)}-{np.max(valid_data)}")
    
    try:
        smoother = SuperSmoother(src_type='close', length=8, num_poles=2)
        result = smoother.calculate(nan_df)
        
        if hasattr(result, 'values'):
            values = result.values
            print(f"結果: 長さ={len(values)}, 有効値数={np.sum(~np.isnan(values))}")
            if np.sum(~np.isnan(values)) > 0:
                valid = values[~np.isnan(values)]
                print(f"範囲: {np.min(valid):.4f} - {np.max(valid):.4f}")
                print(f"最初の10値: {values[:10]}")
                print(f"10-20値: {values[10:20]}")
        else:
            print("❌ valuesアトリビュートなし")
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
    
    # テスト4: 異なるパラメータでテスト
    print("\n--- テスト4: 異なるパラメータでテスト ---")
    test_params = [
        {'length': 5, 'num_poles': 2},
        {'length': 10, 'num_poles': 2},
        {'length': 8, 'num_poles': 3},
    ]
    
    for params in test_params:
        print(f"\nパラメータ: {params}")
        try:
            smoother = SuperSmoother(src_type='close', **params)
            result = smoother.calculate(er_df)
            
            if hasattr(result, 'values'):
                values = result.values
                valid_count = np.sum(~np.isnan(values))
                print(f"  結果: 有効値数={valid_count}/{len(values)}")
                
                if valid_count > 0:
                    valid = values[~np.isnan(values)]
                    print(f"  範囲: {np.min(valid):.4f} - {np.max(valid):.4f}")
        except Exception as e:
            print(f"  ❌ エラー: {e}")

def test_direct_numba_function():
    """Numba関数を直接テストする"""
    print("\n=== 直接Numba関数テスト ===")
    
    # テストデータ
    test_data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.8, 0.7, 0.6, 0.5] * 10)
    
    print(f"入力: 長さ={len(test_data)}, 範囲={np.min(test_data)}-{np.max(test_data)}")
    
    # 異なる期間でテスト
    for length in [5, 8, 10, 13]:
        print(f"\n期間={length}:")
        result = calculate_super_smoother_2pole(test_data, length)
        valid_count = np.sum(~np.isnan(result))
        print(f"  有効値数: {valid_count}/{len(result)}")
        
        if valid_count > 0:
            valid = result[~np.isnan(result)]
            print(f"  範囲: {np.min(valid):.6f} - {np.max(valid):.6f}")
            print(f"  最初の5値: {result[:5]}")
            print(f"  最後の5値: {result[-5:]}")
            
            # 異常な値があるかチェック
            inf_count = np.isinf(result).sum()
            if inf_count > 0:
                print(f"  ⚠️ 無限大値: {inf_count}個")
        else:
            print(f"  ❌ すべてNaN")

def main():
    """メイン実行関数"""
    print("Super Smoother NaN問題詳細デバッグ")
    print("=" * 50)
    
    test_super_smoother_with_different_inputs()
    test_direct_numba_function()
    
    print("\n" + "=" * 50)
    print("デバッグ完了")

if __name__ == "__main__":
    main()