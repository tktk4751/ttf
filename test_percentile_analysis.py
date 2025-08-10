#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
パーセンタイル分析機能のテストスクリプト

X_ERとX_Hurstインジケーターのパーセンタイル分析機能をテストし、
正常に動作することを確認します。
"""

import sys
import os
import numpy as np
import pandas as pd

# プロジェクトのルートディレクトリをパスに追加
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from indicators.trend_filter.x_er import XER, calculate_x_er
from indicators.trend_filter.x_hurst import XHurst, calculate_x_hurst


def generate_test_data(length=200):
    """テストデータを生成"""
    np.random.seed(42)
    
    # ベース価格
    base_price = 100.0
    prices = [base_price]
    
    # 異なる市場状態を模擬
    for i in range(1, length):
        if i < 50:  # 効率的トレンド相場
            change = 0.003 + np.random.normal(0, 0.008)
        elif i < 100:  # 非効率的レンジ相場
            change = np.random.normal(0, 0.012)
        elif i < 150:  # 非常に効率的なトレンド相場
            change = 0.005 + np.random.normal(0, 0.006)
        else:  # レンジ相場
            change = np.random.normal(0, 0.010)
        
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # OHLC データの生成
    data = []
    for i, close in enumerate(prices):
        daily_range = abs(np.random.normal(0, close * 0.01))
        
        high = close + daily_range * np.random.uniform(0.3, 1.0)
        low = close - daily_range * np.random.uniform(0.3, 1.0)
        
        if i == 0:
            open_price = close
        else:
            gap = np.random.normal(0, close * 0.005)
            open_price = prices[i-1] + gap
        
        # 論理的整合性の確保
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.uniform(1000, 10000)
        })
    
    return pd.DataFrame(data)


def test_x_er_percentile_analysis():
    """X_ERのパーセンタイル分析機能をテスト"""
    print("\n=== X_ER パーセンタイル分析テスト ===")
    
    # テストデータ生成
    df = generate_test_data(150)
    print(f"テストデータ: {len(df)}ポイント")
    
    # X_ERインスタンス作成（パーセンタイル分析有効）
    x_er = XER(
        period=14,
        midline_period=50,
        er_period=10,
        use_smoothing=True,
        use_kalman_filter=False,  # テスト用に無効
        enable_percentile_analysis=True,
        percentile_lookback_period=30,
        percentile_low_threshold=0.3,
        percentile_high_threshold=0.7
    )
    
    # 計算実行
    result = x_er.calculate(df)
    
    # 結果検証
    print(f"X_ER有効値数: {np.sum(~np.isnan(result.values))}/{len(df)}")
    print(f"平均X_ER値: {np.nanmean(result.values):.4f}")
    
    # パーセンタイル分析結果の検証
    if result.percentiles is not None:
        valid_percentiles = result.percentiles[~np.isnan(result.percentiles)]
        print(f"パーセンタイル有効値数: {len(valid_percentiles)}")
        print(f"パーセンタイル範囲: {np.min(valid_percentiles):.3f} - {np.max(valid_percentiles):.3f}")
        
        if result.trend_state is not None:
            valid_states = result.trend_state[~np.isnan(result.trend_state)]
            if len(valid_states) > 0:
                low_count = np.sum(valid_states == -1.0)
                mid_count = np.sum(valid_states == 0.0)
                high_count = np.sum(valid_states == 1.0)
                total_count = len(valid_states)
                
                print(f"トレンド状態分布:")
                print(f"  低トレンド: {low_count}/{total_count} ({low_count/total_count:.1%})")
                print(f"  中トレンド: {mid_count}/{total_count} ({mid_count/total_count:.1%})")
                print(f"  高トレンド: {high_count}/{total_count} ({high_count/total_count:.1%})")
        
        if result.trend_intensity is not None:
            valid_intensity = result.trend_intensity[~np.isnan(result.trend_intensity)]
            if len(valid_intensity) > 0:
                print(f"トレンド強度: 平均={np.mean(valid_intensity):.3f}, 範囲={np.min(valid_intensity):.3f}-{np.max(valid_intensity):.3f}")
    
    # getterメソッドのテスト
    percentiles = x_er.get_percentiles()
    trend_state = x_er.get_trend_state()
    trend_intensity = x_er.get_trend_intensity()
    
    print(f"getter結果:")
    print(f"  パーセンタイル: {percentiles is not None}")
    print(f"  トレンド状態: {trend_state is not None}")
    print(f"  トレンド強度: {trend_intensity is not None}")
    
    # 便利関数のテスト
    convenience_result = calculate_x_er(
        df, period=14, enable_percentile_analysis=True
    )
    print(f"便利関数結果: {np.sum(~np.isnan(convenience_result))}/{len(convenience_result)} 有効値")
    
    return True


def test_x_hurst_percentile_analysis():
    """X_Hurstのパーセンタイル分析機能をテスト"""
    print("\n=== X_Hurst パーセンタイル分析テスト ===")
    
    # テストデータ生成
    df = generate_test_data(150)
    print(f"テストデータ: {len(df)}ポイント")
    
    # X_Hurstインスタンス作成（パーセンタイル分析有効）
    x_hurst = XHurst(
        period=30,  # 短めに設定してテスト時間短縮
        midline_period=50,
        min_scale=4,
        max_scale=15,
        scale_steps=6,
        use_smoothing=True,
        use_dynamic_period=False,  # テスト用に無効
        use_kalman_filter=False,   # テスト用に無効
        enable_percentile_analysis=True,
        percentile_lookback_period=25,
        percentile_low_threshold=0.35,
        percentile_high_threshold=0.65
    )
    
    # 計算実行
    try:
        result = x_hurst.calculate(df)
        
        # 結果検証
        print(f"X_Hurst有効値数: {np.sum(~np.isnan(result.values))}/{len(df)}")
        if np.sum(~np.isnan(result.values)) > 0:
            print(f"平均X_Hurst値: {np.nanmean(result.values):.4f}")
            
            # パーセンタイル分析結果の検証
            if result.percentiles is not None:
                valid_percentiles = result.percentiles[~np.isnan(result.percentiles)]
                print(f"パーセンタイル有効値数: {len(valid_percentiles)}")
                if len(valid_percentiles) > 0:
                    print(f"パーセンタイル範囲: {np.min(valid_percentiles):.3f} - {np.max(valid_percentiles):.3f}")
                
                if result.trend_state is not None:
                    valid_states = result.trend_state[~np.isnan(result.trend_state)]
                    if len(valid_states) > 0:
                        low_count = np.sum(valid_states == -1.0)
                        mid_count = np.sum(valid_states == 0.0)
                        high_count = np.sum(valid_states == 1.0)
                        total_count = len(valid_states)
                        
                        print(f"トレンド状態分布:")
                        print(f"  低トレンド: {low_count}/{total_count} ({low_count/total_count:.1%})")
                        print(f"  中トレンド: {mid_count}/{total_count} ({mid_count/total_count:.1%})")
                        print(f"  高トレンド: {high_count}/{total_count} ({high_count/total_count:.1%})")
                
                if result.trend_intensity is not None:
                    valid_intensity = result.trend_intensity[~np.isnan(result.trend_intensity)]
                    if len(valid_intensity) > 0:
                        print(f"トレンド強度: 平均={np.mean(valid_intensity):.3f}, 範囲={np.min(valid_intensity):.3f}-{np.max(valid_intensity):.3f}")
        
        # getterメソッドのテスト
        percentiles = x_hurst.get_percentiles()
        trend_state = x_hurst.get_trend_state()
        trend_intensity = x_hurst.get_trend_intensity()
        
        print(f"getter結果:")
        print(f"  パーセンタイル: {percentiles is not None}")
        print(f"  トレンド状態: {trend_state is not None}")
        print(f"  トレンド強度: {trend_intensity is not None}")
        
        # 便利関数のテスト
        convenience_result = calculate_x_hurst(
            df, period=30, enable_percentile_analysis=True,
            use_dynamic_period=False, use_kalman_filter=False
        )
        print(f"便利関数結果: {np.sum(~np.isnan(convenience_result))}/{len(convenience_result)} 有効値")
        
        return True
        
    except Exception as e:
        print(f"X_Hurst計算中にエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_indicator_info():
    """インジケーター情報のテスト"""
    print("\n=== インジケーター情報テスト ===")
    
    # X_ERのインジケーター情報
    x_er = XER(enable_percentile_analysis=True)
    er_info = x_er.get_indicator_info()
    print("X_ER情報:")
    for key, value in er_info.items():
        print(f"  {key}: {value}")
    
    # X_Hurstのインジケーター情報
    x_hurst = XHurst(enable_percentile_analysis=True)
    hurst_info = x_hurst.get_indicator_info()
    print("\nX_Hurst情報:")
    for key, value in hurst_info.items():
        print(f"  {key}: {value}")


def main():
    """メイン関数"""
    print("パーセンタイル分析機能テスト開始")
    
    success_count = 0
    total_tests = 0
    
    # X_ERテスト
    total_tests += 1
    if test_x_er_percentile_analysis():
        success_count += 1
        print("✓ X_ER パーセンタイル分析テスト成功")
    else:
        print("✗ X_ER パーセンタイル分析テスト失敗")
    
    # X_Hurstテスト
    total_tests += 1
    if test_x_hurst_percentile_analysis():
        success_count += 1
        print("✓ X_Hurst パーセンタイル分析テスト成功")
    else:
        print("✗ X_Hurst パーセンタイル分析テスト失敗")
    
    # インジケーター情報テスト
    total_tests += 1
    try:
        test_indicator_info()
        success_count += 1
        print("✓ インジケーター情報テスト成功")
    except Exception as e:
        print(f"✗ インジケーター情報テスト失敗: {e}")
    
    # 結果サマリー
    print(f"\n=== テスト結果サマリー ===")
    print(f"成功: {success_count}/{total_tests}")
    print(f"成功率: {success_count/total_tests:.1%}")
    
    if success_count == total_tests:
        print("\n🎉 全てのテストが成功しました！")
        return True
    else:
        print(f"\n⚠️  {total_tests - success_count}個のテストが失敗しました。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)