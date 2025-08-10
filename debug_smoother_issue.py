#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HyperERスムーサー問題のデバッグスクリプト
"""

import numpy as np
import pandas as pd
import sys
import os

# インジケーターをインポートするためのパス設定
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from indicators.trend_filter.hyper_er import HyperER
from indicators.smoother.unified_smoother import UnifiedSmoother

def test_unified_smoother_directly():
    """UnifiedSmootherを直接テストする"""
    print("=== UnifiedSmoother直接テスト ===")
    
    # テストデータを作成
    test_data = pd.DataFrame({
        'open': [100, 101, 102, 103, 104] * 20,
        'high': [101, 102, 103, 104, 105] * 20,
        'low': [99, 100, 101, 102, 103] * 20,
        'close': [100.5, 101.5, 102.5, 103.5, 104.5] * 20,
        'volume': [1000] * 100
    })
    
    print(f"テストデータ: {len(test_data)}行")
    
    # 異なるスムーサータイプをテスト
    smoother_types = ['super_smoother', 'frama', 'alma', 'hma']
    
    for smoother_type in smoother_types:
        print(f"\n--- {smoother_type}のテスト ---")
        try:
            smoother = UnifiedSmoother(
                smoother_type=smoother_type,
                src_type='close',
                period=8
            )
            print(f"  初期化: OK")
            
            result = smoother.calculate(test_data)
            print(f"  計算: OK")
            print(f"  結果型: {type(result)}")
            
            if hasattr(result, 'values'):
                values = result.values
                print(f"  値数: {len(values)}")
                print(f"  有効値数: {np.sum(~np.isnan(values))}")
                if np.sum(~np.isnan(values)) > 0:
                    valid_values = values[~np.isnan(values)]
                    print(f"  範囲: {np.min(valid_values):.4f} - {np.max(valid_values):.4f}")
                else:
                    print(f"  ⚠️ すべてNaN")
            else:
                print(f"  ⚠️ valuesアトリビュートがない")
                print(f"  利用可能アトリビュート: {dir(result)}")
            
        except Exception as e:
            print(f"  ❌ エラー: {e}")
            import traceback
            traceback.print_exc()

def test_hyper_er_with_different_smoothers():
    """HyperERで異なるスムーサーをテスト"""
    print("\n=== HyperER + スムーサーテスト ===")
    
    # テストデータを作成
    test_data = pd.DataFrame({
        'open': [100, 101, 102, 103, 104] * 20,
        'high': [101, 102, 103, 104, 105] * 20,
        'low': [99, 100, 101, 102, 103] * 20,
        'close': [100.5, 101.5, 102.5, 103.5, 104.5] * 20,
        'volume': [1000] * 100
    })
    
    smoother_types = ['super_smoother', 'frama', 'alma']
    
    for smoother_type in smoother_types:
        print(f"\n--- HyperER + {smoother_type} ---")
        try:
            hyper_er = HyperER(
                period=14,
                er_period=13,
                use_roofing_filter=False,  # シンプル化
                use_dynamic_period=False,  # シンプル化
                use_smoothing=True,
                smoother_type=smoother_type,
                smoother_period=8,
                smoother_src_type='close'
            )
            print(f"  初期化: OK")
            print(f"  スムーサー初期化済み: {hyper_er.smoother is not None}")
            
            result = hyper_er.calculate(test_data)
            print(f"  計算: OK")
            
            # 結果確認
            print(f"  生ER有効値: {np.sum(~np.isnan(result.raw_er))}")
            print(f"  フィルター済みER有効値: {np.sum(~np.isnan(result.filtered_er))}")
            print(f"  平滑化ER有効値: {np.sum(~np.isnan(result.smoothed_er))}")
            print(f"  最終値有効値: {np.sum(~np.isnan(result.values))}")
            
            if np.sum(~np.isnan(result.values)) > 0:
                valid = result.values[~np.isnan(result.values)]
                print(f"  最終値範囲: {np.min(valid):.4f} - {np.max(valid):.4f}")
            else:
                print(f"  ⚠️ 最終値がすべてNaN")
            
        except Exception as e:
            print(f"  ❌ エラー: {e}")
            import traceback
            traceback.print_exc()

def test_smoother_with_er_like_data():
    """ER値のようなデータでスムーサーをテスト"""
    print("\n=== ER値ライクデータでスムーサーテスト ===")
    
    # ER値のようなデータ（0-1の範囲）を作成
    er_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.8, 0.7, 0.6, 0.5] * 10)
    er_df = pd.DataFrame({'close': er_values})
    
    print(f"ER値ライクデータ: {len(er_df)}行")
    print(f"値範囲: {np.min(er_values)} - {np.max(er_values)}")
    
    smoother_types = ['super_smoother', 'alma', 'hma']
    
    for smoother_type in smoother_types:
        print(f"\n--- {smoother_type} (ER値ライクデータ) ---")
        try:
            smoother = UnifiedSmoother(
                smoother_type=smoother_type,
                src_type='close',
                period=8
            )
            
            result = smoother.calculate(er_df)
            
            if hasattr(result, 'values'):
                values = result.values
                valid_count = np.sum(~np.isnan(values))
                print(f"  有効値数: {valid_count}/{len(values)}")
                
                if valid_count > 0:
                    valid_values = values[~np.isnan(values)]
                    print(f"  範囲: {np.min(valid_values):.4f} - {np.max(valid_values):.4f}")
                    print(f"  最初の10値: {values[:10]}")
                else:
                    print(f"  ⚠️ すべてNaN")
            
        except Exception as e:
            print(f"  ❌ エラー: {e}")

def main():
    """メイン実行関数"""
    print("HyperER スムーサー問題デバッグ")
    print("=" * 50)
    
    # 1. UnifiedSmootherの直接テスト
    test_unified_smoother_directly()
    
    # 2. HyperERでのスムーサー使用テスト
    test_hyper_er_with_different_smoothers()
    
    # 3. ER値ライクデータでのテスト
    test_smoother_with_er_like_data()
    
    print("\n" + "=" * 50)
    print("デバッグ完了")

if __name__ == "__main__":
    main()