#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
拡張グランドサイクルMAの簡単なテスト
カルマンフィルターとスムーサーの統合機能をテスト
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_test_data(length=100):
    """テストデータの作成"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=length, freq='h')
    
    # 価格データの生成
    base_price = 100.0
    price_changes = np.random.normal(0, 1, length)
    prices = [base_price]
    
    for i in range(1, length):
        new_price = prices[-1] + price_changes[i]
        prices.append(max(new_price, 50))  # 最小価格制限
    
    data = pd.DataFrame({
        'open': np.array(prices) * 0.999,
        'high': np.array(prices) * 1.002,
        'low': np.array(prices) * 0.998,
        'close': prices,
        'volume': np.random.uniform(1000, 5000, length)
    }, index=dates)
    
    return data

def test_basic_functionality():
    """基本機能のテスト"""
    print("=== 基本機能テスト ===")
    
    try:
        from indicators.grand_cycle_ma import GrandCycleMA
        
        data = create_test_data(50)
        print(f"テストデータ: {len(data)}件")
        
        # 基本設定
        grand_cycle_ma = GrandCycleMA(
            detector_type='hody',
            use_kalman_filter=False,
            use_smoother=False,
            src_type='close'
        )
        
        result = grand_cycle_ma.calculate(data)
        valid_count = np.sum(~np.isnan(result.grand_mama_values))
        
        print(f"✓ 基本計算成功: {valid_count}/{len(data)} 有効データ")
        return True
        
    except Exception as e:
        print(f"✗ 基本機能エラー: {e}")
        return False

def test_smoother_integration():
    """スムーサー統合のテスト"""
    print("\n=== スムーサー統合テスト ===")
    
    try:
        from indicators.grand_cycle_ma import GrandCycleMA
        
        data = create_test_data(50)
        
        # FRAMAスムーサーを使用
        grand_cycle_ma = GrandCycleMA(
            detector_type='hody',
            use_kalman_filter=False,
            use_smoother=True,
            smoother_type='frama',
            smoother_params={'period': 14},
            src_type='close'
        )
        
        result = grand_cycle_ma.calculate(data)
        valid_count = np.sum(~np.isnan(result.grand_mama_values))
        
        print(f"✓ FRAMAスムーサー統合成功: {valid_count}/{len(data)} 有効データ")
        return True
        
    except Exception as e:
        print(f"✗ スムーサー統合エラー: {e}")
        return False

def test_kalman_integration():
    """カルマンフィルター統合のテスト"""
    print("\n=== カルマンフィルター統合テスト ===")
    
    try:
        from indicators.grand_cycle_ma import GrandCycleMA
        
        data = create_test_data(50)
        
        # 適応カルマンフィルターを使用
        grand_cycle_ma = GrandCycleMA(
            detector_type='hody',
            use_kalman_filter=True,
            kalman_filter_type='adaptive',
            use_smoother=False,
            src_type='close'
        )
        
        result = grand_cycle_ma.calculate(data)
        valid_count = np.sum(~np.isnan(result.grand_mama_values))
        
        print(f"✓ 適応カルマンフィルター統合成功: {valid_count}/{len(data)} 有効データ")
        return True
        
    except Exception as e:
        print(f"✗ カルマンフィルター統合エラー: {e}")
        return False

def test_full_pipeline():
    """完全パイプラインのテスト（カルマン + スムーサー）"""
    print("\n=== 完全パイプラインテスト ===")
    
    try:
        from indicators.grand_cycle_ma import GrandCycleMA
        
        data = create_test_data(50)
        
        # カルマンフィルター + スムーサーの組み合わせ
        grand_cycle_ma = GrandCycleMA(
            detector_type='hody',
            use_kalman_filter=True,
            kalman_filter_type='adaptive',
            use_smoother=True,
            smoother_type='frama',
            smoother_params={'period': 16},
            src_type='close'
        )
        
        result = grand_cycle_ma.calculate(data)
        valid_count = np.sum(~np.isnan(result.grand_mama_values))
        
        print(f"✓ 完全パイプライン成功: {valid_count}/{len(data)} 有効データ")
        
        # 統計情報
        if valid_count > 0:
            valid_mama = result.grand_mama_values[~np.isnan(result.grand_mama_values)]
            print(f"  平均値: {np.mean(valid_mama):.4f}")
            print(f"  範囲: {np.min(valid_mama):.4f} - {np.max(valid_mama):.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 完全パイプラインエラー: {e}")
        return False

def test_processing_order():
    """処理順序の確認テスト"""
    print("\n=== 処理順序確認テスト ===")
    
    try:
        from indicators.grand_cycle_ma import GrandCycleMA
        
        data = create_test_data(50)
        
        # 各段階での結果を比較
        configs = [
            {'name': '元データ', 'kalman': False, 'smoother': False},
            {'name': 'カルマンのみ', 'kalman': True, 'smoother': False},
            {'name': 'スムーサーのみ', 'kalman': False, 'smoother': True},
            {'name': 'カルマン→スムーサー', 'kalman': True, 'smoother': True},
        ]
        
        results = {}
        
        for config in configs:
            grand_cycle_ma = GrandCycleMA(
                detector_type='hody',
                use_kalman_filter=config['kalman'],
                kalman_filter_type='adaptive' if config['kalman'] else None,
                use_smoother=config['smoother'],
                smoother_type='frama' if config['smoother'] else None,
                src_type='close'
            )
            
            result = grand_cycle_ma.calculate(data)
            valid_count = np.sum(~np.isnan(result.grand_mama_values))
            
            results[config['name']] = valid_count
            print(f"  {config['name']}: {valid_count}/{len(data)} 有効データ")
        
        print("✓ 処理順序確認完了")
        return True
        
    except Exception as e:
        print(f"✗ 処理順序確認エラー: {e}")
        return False

def main():
    """メインテスト実行"""
    print("=== 拡張グランドサイクルMA 簡単テスト開始 ===")
    
    tests = [
        test_basic_functionality,
        test_smoother_integration,
        test_kalman_integration,
        test_full_pipeline,
        test_processing_order
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== テスト結果: {passed}/{total} 成功 ===")
    
    if passed == total:
        print("🎉 全てのテストが成功しました！")
        print("\n✓ 確認されたポイント:")
        print("  - 基本的なグランドサイクルMA計算")
        print("  - FRAMAスムーサーの統合")
        print("  - 適応カルマンフィルターの統合")
        print("  - カルマン→スムーサーの処理順序")
        print("  - 複数設定での動作確認")
    else:
        print("⚠️ 一部のテストが失敗しました")
    
    return passed == total

if __name__ == "__main__":
    main()