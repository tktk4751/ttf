#!/usr/bin/env python3
"""
Hyper Adaptive Channel 簡単テストスクリプト

依存関係を最小限にした基本動作テスト
"""

import numpy as np
import pandas as pd
import sys

sys.path.append('.')

def test_simple():
    """シンプルテスト"""
    
    print("=== Hyper Adaptive Channel シンプルテスト ===")
    
    try:
        from indicators.hyper_adaptive_channel import HyperAdaptiveChannel
        
        # シンプルなテストデータ
        np.random.seed(42)
        n = 100
        
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='1H'),
            'open': 100 + np.random.randn(n) * 0.1,
            'high': 101 + np.random.randn(n) * 0.1,
            'low': 99 + np.random.randn(n) * 0.1,
            'close': 100 + np.random.randn(n) * 0.1,
            'volume': np.random.randint(1000, 2000, n)
        })
        
        print(f"テストデータ作成: {len(data)}件")
        
        # 各ミッドラインスムーザーをテスト
        smoothers = [
            "hyper_frama",
            "ultimate_ma",
            "laguerre_filter", 
            "z_adaptive_ma",
            "super_smoother"
        ]
        
        for smoother in smoothers:
            print(f"\n--- {smoother} テスト ---")
            
            try:
                # 固定乗数モード
                indicator = HyperAdaptiveChannel(
                    period=10,
                    midline_smoother=smoother,
                    multiplier_mode="fixed",
                    fixed_multiplier=2.0
                )
                
                result = indicator.calculate(data)
                
                # 基本チェック
                midline_valid = np.sum(~np.isnan(result.midline))
                upper_valid = np.sum(~np.isnan(result.upper_band))
                lower_valid = np.sum(~np.isnan(result.lower_band))
                
                print(f"✓ {smoother} 計算成功")
                print(f"  - Midline有効: {midline_valid}/{len(data)}")
                print(f"  - Upper Band有効: {upper_valid}/{len(data)}")
                print(f"  - Lower Band有効: {lower_valid}/{len(data)}")
                
                if midline_valid > 0:
                    midline_range = f"{np.nanmin(result.midline):.2f} - {np.nanmax(result.midline):.2f}"
                    print(f"  - Midline範囲: {midline_range}")
                
                if len(result.multiplier_values) > 0:
                    mult_range = f"{np.nanmin(result.multiplier_values):.2f} - {np.nanmax(result.multiplier_values):.2f}"
                    print(f"  - 乗数範囲: {mult_range}")
                    
            except Exception as e:
                print(f"✗ {smoother} エラー: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n=== 動的乗数テスト ===")
        
        try:
            # 動的乗数モードテスト
            indicator_dynamic = HyperAdaptiveChannel(
                period=10,
                midline_smoother="super_smoother",
                multiplier_mode="dynamic",
                er_period=8
            )
            
            result_dynamic = indicator_dynamic.calculate(data)
            
            print("✓ 動的乗数モード成功")
            
            if result_dynamic.er_values is not None:
                er_valid = np.sum(~np.isnan(result_dynamic.er_values))
                print(f"  - ER有効値: {er_valid}/{len(data)}")
                
                if er_valid > 0:
                    er_range = f"{np.nanmin(result_dynamic.er_values):.3f} - {np.nanmax(result_dynamic.er_values):.3f}"
                    print(f"  - ER範囲: {er_range}")
            
            mult_range = f"{np.nanmin(result_dynamic.multiplier_values):.2f} - {np.nanmax(result_dynamic.multiplier_values):.2f}"
            print(f"  - 動的乗数範囲: {mult_range}")
            
        except Exception as e:
            print(f"✗ 動的乗数テストエラー: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n✓ 全体テスト完了")
        
    except Exception as e:
        print(f"✗ インポートまたは全体エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_simple()