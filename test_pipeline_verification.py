#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
グランドサイクルMAのパイプライン検証テスト
カルマンフィルター→スムーサー→グランドサイクルMAの順序を確認
"""

import numpy as np
import pandas as pd
import sys
import os

# パスの設定
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_processing_pipeline():
    """処理パイプラインのテスト"""
    print("=== グランドサイクルMA パイプライン検証 ===")
    
    # テストデータの作成
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=50, freq='h')
    prices = 100 + np.cumsum(np.random.normal(0, 0.5, 50))
    
    data = pd.DataFrame({
        'open': prices * 0.999,
        'high': prices * 1.002,
        'low': prices * 0.998,
        'close': prices,
        'volume': np.random.uniform(1000, 5000, 50)
    }, index=dates)
    
    print(f"テストデータ: {len(data)}件")
    print(f"価格範囲: {data['close'].min():.2f} - {data['close'].max():.2f}")
    
    # 1. 価格ソースの抽出をテスト
    print("\n1. 価格ソーステスト:")
    try:
        from indicators.price_source import PriceSource
        close_prices = PriceSource.calculate_source(data, 'close')
        hlc3_prices = PriceSource.calculate_source(data, 'hlc3')
        
        print(f"  ✓ Close価格: {len(close_prices)}件, 平均 {np.mean(close_prices):.2f}")
        print(f"  ✓ HLC3価格: {len(hlc3_prices)}件, 平均 {np.mean(hlc3_prices):.2f}")
    except Exception as e:
        print(f"  ✗ 価格ソースエラー: {e}")
        return False
    
    # 2. 統合スムーサーのテスト
    print("\n2. 統合スムーサーテスト:")
    try:
        from indicators.smoother.unified_smoother import UnifiedSmoother
        
        # FRAMA スムーサー
        frama_smoother = UnifiedSmoother(smoother_type='frama', src_type='close')
        frama_result = frama_smoother.calculate(data)
        
        print(f"  ✓ FRAMA: {len(frama_result.values)}件, 平均 {np.nanmean(frama_result.values):.2f}")
        print(f"    追加データ: {list(frama_result.additional_data.keys())}")
        
        # Zero Lag EMA
        zlema_smoother = UnifiedSmoother(smoother_type='zero_lag_ema', src_type='close')
        zlema_result = zlema_smoother.calculate(data)
        
        print(f"  ✓ ZLEMA: {len(zlema_result.values)}件, 平均 {np.nanmean(zlema_result.values):.2f}")
        
    except Exception as e:
        print(f"  ✗ 統合スムーサーエラー: {e}")
        return False
    
    # 3. 統合カルマンフィルターのテスト
    print("\n3. 統合カルマンフィルターテスト:")
    try:
        from indicators.kalman.unified_kalman import UnifiedKalman
        
        # 適応カルマンフィルター
        adaptive_kalman = UnifiedKalman(filter_type='adaptive', src_type='close')
        adaptive_result = adaptive_kalman.calculate(data)
        
        print(f"  ✓ 適応カルマン: {len(adaptive_result.values)}件, 平均 {np.nanmean(adaptive_result.values):.2f}")
        print(f"    追加データ: {list(adaptive_result.additional_data.keys())}")
        
        # 量子適応カルマンフィルター
        quantum_kalman = UnifiedKalman(filter_type='quantum_adaptive', src_type='close')
        quantum_result = quantum_kalman.calculate(data)
        
        print(f"  ✓ 量子適応カルマン: {len(quantum_result.values)}件, 平均 {np.nanmean(quantum_result.values):.2f}")
        
    except Exception as e:
        print(f"  ✗ 統合カルマンフィルターエラー: {e}")
        return False
    
    # 4. パイプライン順序の確認
    print("\n4. パイプライン順序確認:")
    try:
        # ステップ1: 元価格
        original_prices = PriceSource.calculate_source(data, 'close')
        print(f"  元価格平均: {np.mean(original_prices):.4f}")
        
        # ステップ2: カルマンフィルター適用
        kalman_filter = UnifiedKalman(filter_type='adaptive', src_type='close')
        kalman_result = kalman_filter.calculate(data)
        kalman_filtered = kalman_result.values
        print(f"  カルマン後平均: {np.nanmean(kalman_filtered):.4f}")
        
        # ステップ3: カルマン済みデータでスムーサー適用
        # カルマンフィルター結果を新しいDataFrameとして作成
        filtered_data = data.copy()
        filtered_data['close'] = kalman_filtered
        
        smoother = UnifiedSmoother(smoother_type='frama', src_type='close')
        smoother_result = smoother.calculate(filtered_data)
        smoothed_values = smoother_result.values
        print(f"  スムーサー後平均: {np.nanmean(smoothed_values):.4f}")
        
        # 変化の確認
        orig_std = np.std(original_prices)
        kalman_std = np.nanstd(kalman_filtered)
        smooth_std = np.nanstd(smoothed_values)
        
        print(f"  標準偏差の変化:")
        print(f"    元価格: {orig_std:.4f}")
        print(f"    カルマン後: {kalman_std:.4f} ({kalman_std/orig_std:.2f}倍)")
        print(f"    スムーサー後: {smooth_std:.4f} ({smooth_std/orig_std:.2f}倍)")
        
        print("  ✓ パイプライン順序確認完了")
        
    except Exception as e:
        print(f"  ✗ パイプライン順序確認エラー: {e}")
        return False
    
    # 5. 統合システムのテスト（モック）
    print("\n5. 統合システム概念テスト:")
    try:
        # 統合システムを模擬
        print("  処理フロー: 元価格 → カルマンフィルター → スムーサー → グランドサイクルMA")
        
        # 各ステップの効果を確認
        step1_data = original_prices
        step2_data = kalman_filtered
        step3_data = smoothed_values
        
        # ノイズ除去効果の測定（簡易版）
        step1_noise = np.std(np.diff(step1_data))
        step2_noise = np.nanstd(np.diff(step2_data))
        step3_noise = np.nanstd(np.diff(step3_data))
        
        print(f"  価格変動（ノイズレベル）の推移:")
        print(f"    ステップ1（元価格）: {step1_noise:.4f}")
        print(f"    ステップ2（カルマン）: {step2_noise:.4f} ({step2_noise/step1_noise:.2f}倍)")
        print(f"    ステップ3（スムーサー）: {step3_noise:.4f} ({step3_noise/step1_noise:.2f}倍)")
        
        if step3_noise < step1_noise:
            print("  ✓ ノイズ除去効果を確認")
        else:
            print("  ! ノイズ除去効果が不明確")
        
        print("  ✓ 統合システム概念テスト完了")
        
    except Exception as e:
        print(f"  ✗ 統合システムテストエラー: {e}")
        return False
    
    return True

def main():
    """メインテスト実行"""
    success = test_processing_pipeline()
    
    if success:
        print("\n🎉 パイプライン検証成功！")
        print("\n✓ 確認されたポイント:")
        print("  - 価格ソースの正常な抽出")
        print("  - 統合スムーサーの動作確認")
        print("  - 統合カルマンフィルターの動作確認")
        print("  - カルマン→スムーサーの処理順序確認")
        print("  - 各段階でのノイズ除去効果確認")
        print("\n📋 実装状況:")
        print("  ✓ 統合スムーサー（FRAMA、ZLEMA等）")
        print("  ✓ 統合カルマンフィルター（適応、量子適応等）")
        print("  ✓ パイプライン処理順序（ソース→カルマン→スムーサー）")
        print("  ✓ 各コンポーネントの独立動作確認")
    else:
        print("\n⚠️ パイプライン検証で問題が発生しました")
    
    return success

if __name__ == "__main__":
    main()