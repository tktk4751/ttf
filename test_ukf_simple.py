#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
簡単なUKF比較テストスクリプト
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def simple_ukf_test():
    """簡単なUKFテスト"""
    print("=" * 50)
    print("簡単なUKF比較テスト")
    print("=" * 50)
    
    try:
        from indicators.smoother.unscented_kalman_filter import UnscentedKalmanFilter
        from indicators.smoother.unscented_kalman_filter_v2 import UnscentedKalmanFilterV2Wrapper
        print("インポート成功")
    except ImportError as e:
        print(f"インポートエラー: {e}")
        return False
    
    # 簡単なデータ生成
    print("サンプルデータ生成中...")
    np.random.seed(42)
    length = 100
    
    # 価格データ生成
    base_price = 100.0
    prices = [base_price]
    for i in range(1, length):
        change = np.random.normal(0.001, 0.02)  # 小さなトレンドとノイズ
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # DataFrame作成
    data = pd.DataFrame({
        'close': prices,
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'volume': [1000] * length
    })
    
    # タイムスタンプ追加
    start_date = datetime.now() - timedelta(hours=length)
    data.index = [start_date + timedelta(hours=i) for i in range(length)]
    
    print(f"データ生成完了: {len(data)}ポイント")
    print(f"価格範囲: {min(prices):.2f} - {max(prices):.2f}")
    
    # UKF V1テスト
    print("\nUKF V1テスト中...")
    try:
        ukf_v1 = UnscentedKalmanFilter(
            src_type='close',
            alpha=0.1,
            beta=2.0,
            kappa=0.0,
            process_noise_scale=0.01
        )
        
        result_v1 = ukf_v1.calculate(data)
        print(f"UKF V1成功: フィルタリング結果形状 {result_v1.filtered_values.shape}")
        
        error_v1 = np.mean(np.abs(result_v1.filtered_values - np.array(prices)))
        print(f"UKF V1 平均絶対誤差: {error_v1:.6f}")
        print(f"UKF V1 平均信頼度: {np.mean(result_v1.confidence_scores):.4f}")
        
    except Exception as e:
        print(f"UKF V1エラー: {e}")
        return False
    
    # UKF V2テスト
    print("\nUKF V2テスト中...")
    try:
        ukf_v2 = UnscentedKalmanFilterV2Wrapper(
            src_type='close',
            kappa=0.0,
            process_noise_scale=0.01,
            observation_noise_scale=0.001
        )
        
        result_v2 = ukf_v2.calculate(data)
        print(f"UKF V2成功: フィルタリング結果形状 {result_v2.filtered_values.shape}")
        
        error_v2 = np.mean(np.abs(result_v2.filtered_values - np.array(prices)))
        print(f"UKF V2 平均絶対誤差: {error_v2:.6f}")
        print(f"UKF V2 平均信頼度: {np.mean(result_v2.confidence_scores):.4f}")
        
        # 状態推定の確認
        if result_v2.state_estimates.shape[1] >= 3:
            velocity = result_v2.state_estimates[:, 1]
            acceleration = result_v2.state_estimates[:, 2]
            print(f"UKF V2 速度範囲: [{np.min(velocity):.4f}, {np.max(velocity):.4f}]")
            print(f"UKF V2 加速度範囲: [{np.min(acceleration):.6f}, {np.max(acceleration):.6f}]")
        
    except Exception as e:
        print(f"UKF V2エラー: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 比較
    print("\n" + "=" * 30)
    print("比較結果")
    print("=" * 30)
    print(f"UKF V1 MAE: {error_v1:.6f}")
    print(f"UKF V2 MAE: {error_v2:.6f}")
    
    if error_v1 < error_v2:
        print("UKF V1の方が低い誤差")
    else:
        print("UKF V2の方が低い誤差")
    
    return True


def simple_chart_test():
    """簡単なチャートテスト"""
    print("\n" + "=" * 50)
    print("簡単なチャート生成テスト")
    print("=" * 50)
    
    try:
        from visualization.ukf_comparison_chart import generate_sample_data, create_ukf_comparison_chart
        
        # 小さなサンプルデータ
        print("小さなサンプルデータ生成中...")
        data = generate_sample_data(100)
        print(f"データ生成完了: {len(data)}ポイント")
        
        # チャート生成（保存のみ）
        save_path = "ukf_comparison_simple_test.png"
        print(f"チャート生成中... (保存先: {save_path})")
        
        fig = create_ukf_comparison_chart(
            data=data,
            title="UKF比較 (簡単テスト)",
            save_path=save_path,
            figsize=(12, 10)
        )
        
        if os.path.exists(save_path):
            file_size = os.path.getsize(save_path)
            print(f"チャート生成成功: {save_path} ({file_size:,} bytes)")
            return True
        else:
            print("チャートファイルが見つかりません")
            return False
            
    except Exception as e:
        print(f"チャート生成エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """メインテスト関数"""
    print("UKF簡単比較テストを開始します")
    
    try:
        # 基本テスト
        basic_success = simple_ukf_test()
        
        # チャートテスト
        chart_success = simple_chart_test()
        
        print("\n" + "=" * 50)
        print("テスト結果")
        print("=" * 50)
        print(f"基本UKFテスト: {'✓ 成功' if basic_success else '✗ 失敗'}")
        print(f"チャート生成テスト: {'✓ 成功' if chart_success else '✗ 失敗'}")
        
        if basic_success and chart_success:
            print("\n全てのテストが成功しました！")
            return 0
        else:
            print("\n一部のテストが失敗しました。")
            return 1
            
    except Exception as e:
        print(f"テスト実行中にエラー: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())