#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
X_チョピネスチャート（パーセンタイル分析付き）のテストスクリプト

X_チョピネスのパーセンタイル分析機能を含むチャート機能をテストします。
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# プロジェクトのルートディレクトリをパスに追加
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from visualization.x_choppiness_chart import XChoppinessChart


def generate_test_data(length=300):
    """テストデータを生成（時系列インデックス付き）"""
    np.random.seed(42)
    
    # 時系列インデックス作成
    dates = pd.date_range(start='2023-01-01', periods=length, freq='4H')
    
    # ベース価格
    base_price = 100.0
    prices = [base_price]
    
    # 異なる市場状態を模擬
    for i in range(1, length):
        if i < 75:  # 強いトレンド相場
            change = 0.005 + np.random.normal(0, 0.008)
        elif i < 150:  # レンジ相場
            change = np.random.normal(0, 0.015)
        elif i < 225:  # 中程度のトレンド相場
            change = 0.003 + np.random.normal(0, 0.012)
        else:  # 弱いトレンド相場
            change = 0.001 + np.random.normal(0, 0.018)
        
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # OHLC データの生成
    data = []
    for i, close in enumerate(prices):
        daily_range = abs(np.random.normal(0, close * 0.02))
        
        high = close + daily_range * np.random.uniform(0.3, 1.0)
        low = close - daily_range * np.random.uniform(0.3, 1.0)
        
        if i == 0:
            open_price = close
        else:
            gap = np.random.normal(0, close * 0.008)
            open_price = prices[i-1] + gap
        
        # 論理的整合性の確保
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.uniform(1000, 50000)
        })
    
    return pd.DataFrame(data, index=dates)


def test_chart_without_percentile():
    """パーセンタイル分析なしのチャートテスト"""
    print("\n=== パーセンタイル分析なしのチャートテスト ===")
    
    try:
        # テストデータ生成
        df = generate_test_data(200)
        print(f"テストデータ生成: {len(df)}ポイント")
        
        # チャートオブジェクト作成
        chart = XChoppinessChart()
        chart.data = df
        
        # パーセンタイル分析なしでインジケーター計算
        chart.calculate_indicators(
            period=14,
            midline_period=50,
            str_period=20.0,
            use_smoothing=False,
            use_dynamic_period=False,
            use_kalman_filter=False,
            enable_percentile_analysis=False  # パーセンタイル分析を無効
        )
        
        # チャート描画（保存テスト）
        output_path = "test_x_choppiness_without_percentile.png"
        chart.plot(
            title="X_チョピネス（パーセンタイル分析なし）",
            show_volume=True,
            figsize=(16, 10),
            savefig=output_path
        )
        
        # ファイル確認
        if os.path.exists(output_path):
            print(f"✓ チャート保存成功: {output_path}")
            return True
        else:
            print("✗ チャート保存失敗")
            return False
            
    except Exception as e:
        print(f"✗ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chart_with_percentile():
    """パーセンタイル分析ありのチャートテスト"""
    print("\n=== パーセンタイル分析ありのチャートテスト ===")
    
    try:
        # テストデータ生成
        df = generate_test_data(200)
        print(f"テストデータ生成: {len(df)}ポイント")
        
        # チャートオブジェクト作成
        chart = XChoppinessChart()
        chart.data = df
        
        # パーセンタイル分析ありでインジケーター計算
        chart.calculate_indicators(
            period=14,
            midline_period=50,
            str_period=20.0,
            use_smoothing=True,
            smoother_type='super_smoother',
            smoother_period=8,
            use_dynamic_period=False,
            use_kalman_filter=False,
            enable_percentile_analysis=True,  # パーセンタイル分析を有効
            percentile_lookback_period=30,
            percentile_low_threshold=0.3,
            percentile_high_threshold=0.7
        )
        
        # チャート描画（保存テスト）
        output_path = "test_x_choppiness_with_percentile.png"
        chart.plot(
            title="X_チョピネス（パーセンタイル分析あり）",
            show_volume=True,
            figsize=(16, 12),
            savefig=output_path
        )
        
        # ファイル確認
        if os.path.exists(output_path):
            print(f"✓ チャート保存成功: {output_path}")
            return True
        else:
            print("✗ チャート保存失敗")
            return False
            
    except Exception as e:
        print(f"✗ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chart_percentile_only():
    """パーセンタイル分析のみ表示のチャートテスト"""
    print("\n=== パーセンタイル分析フォーカステスト ===")
    
    try:
        # テストデータ生成
        df = generate_test_data(150)
        print(f"テストデータ生成: {len(df)}ポイント")
        
        # チャートオブジェクト作成
        chart = XChoppinessChart()
        chart.data = df
        
        # パーセンタイル分析特化設定
        chart.calculate_indicators(
            period=21,
            midline_period=100,
            str_period=15.0,
            use_smoothing=False,
            use_dynamic_period=False,
            use_kalman_filter=False,
            enable_percentile_analysis=True,
            percentile_lookback_period=40,
            percentile_low_threshold=0.25,
            percentile_high_threshold=0.75
        )
        
        # チャート描画（出来高なし）
        output_path = "test_x_choppiness_percentile_focus.png"
        chart.plot(
            title="X_チョピネス - パーセンタイル分析フォーカス",
            show_volume=False,  # 出来高なし
            figsize=(14, 10),
            savefig=output_path
        )
        
        # ファイル確認
        if os.path.exists(output_path):
            print(f"✓ チャート保存成功: {output_path}")
            return True
        else:
            print("✗ チャート保存失敗")
            return False
            
    except Exception as e:
        print(f"✗ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def cleanup_test_files():
    """テストファイルをクリーンアップ"""
    test_files = [
        "test_x_choppiness_without_percentile.png",
        "test_x_choppiness_with_percentile.png", 
        "test_x_choppiness_percentile_focus.png"
    ]
    
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"クリーンアップ: {file}")


def main():
    """メイン関数"""
    print("X_チョピネスチャート（パーセンタイル分析付き）テスト開始")
    
    success_count = 0
    total_tests = 0
    
    # テスト1: パーセンタイル分析なし
    total_tests += 1
    if test_chart_without_percentile():
        success_count += 1
        print("✓ パーセンタイル分析なしチャートテスト成功")
    else:
        print("✗ パーセンタイル分析なしチャートテスト失敗")
    
    # テスト2: パーセンタイル分析あり
    total_tests += 1
    if test_chart_with_percentile():
        success_count += 1
        print("✓ パーセンタイル分析ありチャートテスト成功")
    else:
        print("✗ パーセンタイル分析ありチャートテスト失敗")
    
    # テスト3: パーセンタイル分析フォーカス
    total_tests += 1
    if test_chart_percentile_only():
        success_count += 1
        print("✓ パーセンタイル分析フォーカステスト成功")
    else:
        print("✗ パーセンタイル分析フォーカステスト失敗")
    
    # 結果サマリー
    print(f"\n=== テスト結果サマリー ===")
    print(f"成功: {success_count}/{total_tests}")
    print(f"成功率: {success_count/total_tests:.1%}")
    
    if success_count == total_tests:
        print("\n🎉 全てのテストが成功しました！")
        print("\n生成されたチャートファイル:")
        for file in ["test_x_choppiness_without_percentile.png", 
                     "test_x_choppiness_with_percentile.png",
                     "test_x_choppiness_percentile_focus.png"]:
            if os.path.exists(file):
                print(f"  - {file}")
        
        # クリーンアップの確認
        user_input = input("\nテストファイルを削除しますか？ (y/N): ").strip().lower()
        if user_input == 'y':
            cleanup_test_files()
        
        return True
    else:
        print(f"\n⚠️  {total_tests - success_count}個のテストが失敗しました。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)