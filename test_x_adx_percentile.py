#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
X_ADXパーセンタイル分析機能のテストスクリプト

X_ADXインジケーターのパーセンタイル分析機能をテストし、
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

from indicators.trend_filter.x_adx import XADX, calculate_x_adx


def generate_test_data(length=200):
    """テストデータを生成"""
    np.random.seed(42)
    
    # ベース価格
    base_price = 100.0
    prices = [base_price]
    
    # 異なる市場状態を模擬
    for i in range(1, length):
        if i < 50:  # 強いトレンド相場
            change = 0.004 + np.random.normal(0, 0.008)
        elif i < 100:  # レンジ相場
            change = np.random.normal(0, 0.012)
        elif i < 150:  # 中程度のトレンド相場
            change = 0.002 + np.random.normal(0, 0.010)
        else:  # 弱いトレンド相場
            change = 0.001 + np.random.normal(0, 0.015)
        
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # OHLC データの生成
    data = []
    for i, close in enumerate(prices):
        daily_range = abs(np.random.normal(0, close * 0.015))
        
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


def test_x_adx_percentile_analysis():
    """X_ADXのパーセンタイル分析機能をテスト"""
    print("\n=== X_ADX パーセンタイル分析テスト ===")
    
    # テストデータ生成
    df = generate_test_data(150)
    print(f"テストデータ: {len(df)}ポイント")
    
    # X_ADXインスタンス作成（パーセンタイル分析有効）
    x_adx = XADX(
        period=14,
        midline_period=50,
        tr_method='atr',  # ATR方式を使用
        use_smoothing=True,
        smoother_type='super_smoother',
        smoother_period=8,
        use_dynamic_period=False,  # テスト用に無効
        use_kalman_filter=False,   # テスト用に無効
        enable_percentile_analysis=True,
        percentile_lookback_period=30,
        percentile_low_threshold=0.3,
        percentile_high_threshold=0.7
    )
    
    # 計算実行
    try:
        result = x_adx.calculate(df)
        
        # 結果検証
        valid_values = np.sum(~np.isnan(result.values))
        print(f"X_ADX有効値数: {valid_values}/{len(df)}")
        
        if valid_values > 0:
            print(f"平均X_ADX値: {np.nanmean(result.values):.4f}")
            print(f"X_ADX範囲: {np.nanmin(result.values):.4f} - {np.nanmax(result.values):.4f}")
            
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
        percentiles = x_adx.get_percentiles()
        trend_state = x_adx.get_trend_state()
        trend_intensity = x_adx.get_trend_intensity()
        
        print(f"getter結果:")
        print(f"  パーセンタイル: {percentiles is not None}")
        print(f"  トレンド状態: {trend_state is not None}")
        print(f"  トレンド強度: {trend_intensity is not None}")
        
        # インジケーター情報のテスト
        info = x_adx.get_indicator_info()
        print(f"\nインジケーター情報:")
        print(f"  名前: {info['name']}")
        print(f"  パーセンタイル分析: {info['enable_percentile_analysis']}")
        
        # 便利関数のテスト
        convenience_result = calculate_x_adx(
            df, period=14, enable_percentile_analysis=True,
            use_dynamic_period=False, use_kalman_filter=False
        )
        valid_convenience = np.sum(~np.isnan(convenience_result))
        print(f"便利関数結果: {valid_convenience}/{len(convenience_result)} 有効値")
        
        return True
        
    except Exception as e:
        print(f"X_ADX計算中にエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_x_adx_str_method():
    """X_ADXのSTR方式でのテスト"""
    print("\n=== X_ADX STR方式テスト ===")
    
    # テストデータ生成
    df = generate_test_data(100)
    print(f"テストデータ: {len(df)}ポイント")
    
    # X_ADXインスタンス作成（STR方式、パーセンタイル分析有効）
    x_adx_str = XADX(
        period=14,
        midline_period=50,
        tr_method='str',  # STR方式を使用
        str_period=20.0,
        use_smoothing=False,  # シンプルなテスト
        use_dynamic_period=False,
        use_kalman_filter=False,
        enable_percentile_analysis=True,
        percentile_lookback_period=25
    )
    
    try:
        result = x_adx_str.calculate(df)
        
        valid_values = np.sum(~np.isnan(result.values))
        print(f"X_ADX(STR)有効値数: {valid_values}/{len(df)}")
        
        if valid_values > 0:
            print(f"平均X_ADX(STR)値: {np.nanmean(result.values):.4f}")
            
            # パーセンタイル分析の確認
            if result.percentiles is not None:
                valid_percentiles = result.percentiles[~np.isnan(result.percentiles)]
                print(f"STR方式パーセンタイル有効値数: {len(valid_percentiles)}")
        
        return True
        
    except Exception as e:
        print(f"X_ADX(STR)計算中にエラー: {e}")
        return False


def main():
    """メイン関数"""
    print("X_ADX パーセンタイル分析機能テスト開始")
    
    success_count = 0
    total_tests = 0
    
    # X_ADX パーセンタイル分析テスト
    total_tests += 1
    if test_x_adx_percentile_analysis():
        success_count += 1
        print("✓ X_ADX パーセンタイル分析テスト成功")
    else:
        print("✗ X_ADX パーセンタイル分析テスト失敗")
    
    # X_ADX STR方式テスト
    total_tests += 1
    if test_x_adx_str_method():
        success_count += 1
        print("✓ X_ADX STR方式テスト成功")
    else:
        print("✗ X_ADX STR方式テスト失敗")
    
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