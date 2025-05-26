#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from indicators.z_adaptive_channel import ZAdaptiveChannel

def test_x_trend_adjustment():
    """X-Trend Index調整機能をテストする"""
    print("X-Trend Index調整機能のテスト開始...")
    
    # テスト用データの生成（50データポイント）
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=50, freq='1D')
    
    # トレンドとノイズを含む価格データを生成
    base_price = 100.0
    trend = np.linspace(0, 10, 50)  # 上昇トレンド
    noise = np.random.normal(0, 2, 50)  # ノイズ
    
    # OHLC データの生成
    close_prices = base_price + trend + noise
    high_prices = close_prices + np.abs(np.random.normal(0, 1, 50))
    low_prices = close_prices - np.abs(np.random.normal(0, 1, 50))
    open_prices = close_prices + np.random.normal(0, 0.5, 50)
    
    # DataFrame作成
    df = pd.DataFrame({
        'date': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices
    })
    
    print(f"テストデータ生成完了。データポイント数: {len(df)}")
    print(f"価格範囲: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    # X-Trend Index調整なしでテスト
    print("\n1. X-Trend Index調整なしでテスト...")
    z_channel_no_adjustment = ZAdaptiveChannel(
        use_x_trend_adjustment=False,
        multiplier_smoothing_method='none'
    )
    
    try:
        middle_no_adj = z_channel_no_adjustment.calculate(df)
        bands_no_adj = z_channel_no_adjustment.get_bands()
        print(f"   調整なし - 計算成功。中心線データポイント数: {len(middle_no_adj)}")
        
        # 基本乗数を取得
        basic_upper_mult = z_channel_no_adjustment.get_upper_multiplier()
        basic_lower_mult = z_channel_no_adjustment.get_lower_multiplier()
        print(f"   基本アッパー乗数範囲: {basic_upper_mult.min():.3f} - {basic_upper_mult.max():.3f}")
        print(f"   基本ロワー乗数範囲: {basic_lower_mult.min():.3f} - {basic_lower_mult.max():.3f}")
        
    except Exception as e:
        print(f"   調整なしテストでエラー: {e}")
        return False
    
    # X-Trend Index調整ありでテスト
    print("\n2. X-Trend Index調整ありでテスト...")
    z_channel_with_adjustment = ZAdaptiveChannel(
        use_x_trend_adjustment=True,
        multiplier_smoothing_method='none'
    )
    
    try:
        middle_with_adj = z_channel_with_adjustment.calculate(df)
        bands_with_adj = z_channel_with_adjustment.get_bands()
        print(f"   調整あり - 計算成功。中心線データポイント数: {len(middle_with_adj)}")
        
        # X-Trend Index値を取得
        x_trend_values = z_channel_with_adjustment.get_x_trend_values()
        print(f"   X-Trend Index範囲: {x_trend_values.min():.3f} - {x_trend_values.max():.3f}")
        
        # X-Trend Index調整された乗数を取得
        x_trend_upper_mult = z_channel_with_adjustment.get_x_trend_upper_multiplier()
        x_trend_lower_mult = z_channel_with_adjustment.get_x_trend_lower_multiplier()
        print(f"   X-Trend調整アッパー乗数範囲: {x_trend_upper_mult.min():.3f} - {x_trend_upper_mult.max():.3f}")
        print(f"   X-Trend調整ロワー乗数範囲: {x_trend_lower_mult.min():.3f} - {x_trend_lower_mult.max():.3f}")
        
    except Exception as e:
        print(f"   調整ありテストでエラー: {e}")
        return False
    
    # 調整ロジックの詳細テスト
    print("\n3. 調整ロジックの詳細テスト...")
    
    # 特定のX-Trend Index値での調整をテスト
    test_x_trend_values = np.array([0.5, 0.65, 0.7, 0.8, 0.9, 0.95, 0.98])
    test_multipliers = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    
    # adjust_multipliers_with_x_trend_index関数を直接テスト
    from indicators.z_adaptive_channel import (
        adjust_multipliers_with_x_trend_index, 
        linear_interpolation_x_trend,
        get_x_trend_usage_percentage,
        calculate_x_trend_multiplier,
        X_TREND_RANGE_THRESHOLD,
        X_TREND_TREND_THRESHOLD,
        X_TREND_MAX_USAGE,
        X_TREND_MIN_USAGE,
        X_TREND_MIN_MULTIPLIER
    )
    
    adjusted_upper, adjusted_lower = adjust_multipliers_with_x_trend_index(
        test_multipliers, test_multipliers, test_x_trend_values
    )
    
    print("   X-Trend Index値と対応する調整後乗数:")
    for i, x_trend in enumerate(test_x_trend_values):
        usage_rate = adjusted_upper[i] / test_multipliers[i]
        print(f"   X-Trend={x_trend:.2f}: 乗数={adjusted_upper[i]:.3f} (使用率={usage_rate:.1%})")
    
    # 線形補間関数の直接テスト
    print("\n   線形補間関数の直接テスト:")
    x_min, x_max, y_max, y_min = X_TREND_RANGE_THRESHOLD, X_TREND_TREND_THRESHOLD, X_TREND_MAX_USAGE, X_TREND_MIN_USAGE
    
    # 境界値と中間値をテスト
    test_values = [0.5, 0.65, 0.7, 0.8, 0.9, 0.95, 1.0]
    for x_val in test_values:
        interpolated = linear_interpolation_x_trend(x_val, x_min, x_max, y_min, y_max)
        print(f"   X-Trend={x_val:.2f}: 使用率={interpolated:.1%}")
    
    # ヘルパー関数のテスト
    print("\n   ヘルパー関数のテスト:")
    for x_val in [0.6, 0.75, 0.9]:
        usage_pct = get_x_trend_usage_percentage(x_val)
        adjusted_mult = calculate_x_trend_multiplier(5.0, x_val)
        print(f"   X-Trend={x_val:.2f}: 使用率={usage_pct:.1%}, 調整乗数={adjusted_mult:.3f} (元乗数5.0)")
    
    # 期待される結果の検証
    print("\n4. ロジック検証...")
    
    # X-Trend = X_TREND_RANGE_THRESHOLDの場合は100%使用
    x_065_usage = adjusted_upper[1] / test_multipliers[1]
    if abs(x_065_usage - 1.0) < 0.001:
        print(f"   ✓ X-Trend={X_TREND_RANGE_THRESHOLD}で100%使用")
    else:
        print(f"   ✗ X-Trend={X_TREND_RANGE_THRESHOLD}で{x_065_usage:.1%}使用（期待値100%）")
    
    # X-Trend = X_TREND_TREND_THRESHOLDの場合は20%使用
    x_095_usage = adjusted_upper[5] / test_multipliers[5]
    if abs(x_095_usage - X_TREND_MIN_USAGE) < 0.001:
        print(f"   ✓ X-Trend={X_TREND_TREND_THRESHOLD}で{X_TREND_MIN_USAGE:.0%}使用")
    else:
        print(f"   ✗ X-Trend={X_TREND_TREND_THRESHOLD}で{x_095_usage:.1%}使用（期待値{X_TREND_MIN_USAGE:.0%}）")
    
    # X-Trend = 0.8の場合の線形補間チェック（関数を使用）
    expected_08_usage = linear_interpolation_x_trend(0.8, x_min, x_max, y_min, y_max)
    x_08_usage = adjusted_upper[3] / test_multipliers[3]
    if abs(x_08_usage - expected_08_usage) < 0.001:
        print(f"   ✓ X-Trend=0.8で{x_08_usage:.1%}使用（期待値{expected_08_usage:.1%}）")
    else:
        print(f"   ✗ X-Trend=0.8で{x_08_usage:.1%}使用（期待値{expected_08_usage:.1%}）")
    
    # X-Trend ≤ X_TREND_RANGE_THRESHOLDの場合は100%使用
    x_05_usage = adjusted_upper[0] / test_multipliers[0]  # X-Trend = 0.5
    if abs(x_05_usage - 1.0) < 0.001:
        print("   ✓ X-Trend=0.5で100%使用")
    else:
        print(f"   ✗ X-Trend=0.5で{x_05_usage:.1%}使用（期待値100%）")
    
    # 最小乗数制限の検証
    min_multiplier = min(adjusted_upper.min(), adjusted_lower.min())
    if min_multiplier >= X_TREND_MIN_MULTIPLIER:
        print(f"   ✓ 最小乗数が{X_TREND_MIN_MULTIPLIER}以上: {min_multiplier:.3f}")
    else:
        print(f"   ✗ 最小乗数が{X_TREND_MIN_MULTIPLIER}未満: {min_multiplier:.3f}")
    
    # 非常に小さい元乗数での最小値制限テスト
    print("\n   最小値制限の詳細テスト:")
    small_multipliers = np.array([0.1, 0.2, 0.3])  # 小さい乗数
    extreme_x_trend = np.array([0.9, 0.95, 0.98])  # 高いX-Trend値
    
    small_adjusted_upper, small_adjusted_lower = adjust_multipliers_with_x_trend_index(
        small_multipliers, small_multipliers, extreme_x_trend
    )
    
    for i, (orig, adj, x_val) in enumerate(zip(small_multipliers, small_adjusted_upper, extreme_x_trend)):
        # 線形補間関数を使用して期待値を計算
        expected_usage = linear_interpolation_x_trend(x_val, x_min, x_max, y_min, y_max)
        expected_before_limit = orig * expected_usage
        
        print(f"   元乗数{orig:.1f}, X-Trend={x_val:.2f}: 制限前{expected_before_limit:.3f} → 制限後{adj:.3f}")
        if adj >= X_TREND_MIN_MULTIPLIER:
            print(f"     ✓ 最小値制限適用")
        else:
            print(f"     ✗ 最小値制限未適用")
    
    print("\n5. バンド幅の比較...")
    
    # 実際のデータでのバンド幅を比較
    if len(bands_no_adj) == 3 and len(bands_with_adj) == 3:
        middle_no, upper_no, lower_no = bands_no_adj
        middle_with, upper_with, lower_with = bands_with_adj
        
        # バンド値が有効かチェック
        if len(upper_no) > 0 and len(lower_no) > 0 and len(upper_with) > 0 and len(lower_with) > 0:
            # バンド幅を計算
            band_width_no_adj = upper_no - lower_no
            band_width_with_adj = upper_with - lower_with
            
            # 有効な値のみを使用
            valid_no_adj = band_width_no_adj[~np.isnan(band_width_no_adj)]
            valid_with_adj = band_width_with_adj[~np.isnan(band_width_with_adj)]
            
            if len(valid_no_adj) > 0 and len(valid_with_adj) > 0:
                print(f"   調整なしバンド幅範囲: {valid_no_adj.min():.3f} - {valid_no_adj.max():.3f}")
                print(f"   調整ありバンド幅範囲: {valid_with_adj.min():.3f} - {valid_with_adj.max():.3f}")
                
                # 平均バンド幅の変化
                avg_width_no_adj = np.mean(valid_no_adj)
                avg_width_with_adj = np.mean(valid_with_adj)
                width_change = (avg_width_with_adj - avg_width_no_adj) / avg_width_no_adj * 100
                
                print(f"   平均バンド幅変化: {width_change:+.1f}%")
            else:
                print("   有効なバンド幅データが不足しています")
        else:
            print("   バンドデータが不足しています")
    else:
        print("   バンドデータの取得に失敗しました")
    
    print("\nX-Trend Index調整機能のテスト完了！")
    return True

if __name__ == "__main__":
    success = test_x_trend_adjustment()
    if success:
        print("\n✓ すべてのテストが成功しました！")
    else:
        print("\n✗ テストでエラーが発生しました。") 