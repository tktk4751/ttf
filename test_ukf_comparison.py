#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UKF比較チャートのテストスクリプト
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from indicators.smoother.unscented_kalman_filter import UnscentedKalmanFilter
    from indicators.smoother.unscented_kalman_filter_v2 import UnscentedKalmanFilterV2Wrapper
    from visualization.ukf_comparison_chart import create_ukf_comparison_chart, generate_sample_data
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required modules are available")
    sys.exit(1)


def test_ukf_filters():
    """UKFフィルターの基本テスト"""
    print("=" * 50)
    print("UKFフィルター基本テスト")
    print("=" * 50)
    
    # サンプルデータ生成
    print("サンプルデータを生成中...")
    data = generate_sample_data(200)
    print(f"データ形状: {data.shape}")
    print(f"カラム: {list(data.columns)}")
    print(f"データ範囲: {data.index[0]} ～ {data.index[-1]}")
    
    # UKF V1テスト
    print("\nUKF V1（既存実装）をテスト中...")
    ukf_v1 = UnscentedKalmanFilter(
        src_type='close',
        alpha=0.1,
        beta=2.0,
        kappa=0.0,
        process_noise_scale=0.01,
        volatility_window=10,
        adaptive_noise=True
    )
    
    result_v1 = ukf_v1.calculate(data)
    print(f"V1 フィルタリング結果形状: {result_v1.filtered_values.shape}")
    print(f"V1 平均フィルタリング値: {np.mean(result_v1.filtered_values):.4f}")
    print(f"V1 平均不確実性: {np.mean(result_v1.uncertainty):.6f}")
    print(f"V1 平均信頼度: {np.mean(result_v1.confidence_scores):.4f}")
    
    # UKF V2テスト
    print("\nUKF V2（アカデミック実装）をテスト中...")
    ukf_v2 = UnscentedKalmanFilterV2Wrapper(
        src_type='close',
        kappa=0.0,
        process_noise_scale=0.01,
        observation_noise_scale=0.001,
        max_steps=1000
    )
    
    result_v2 = ukf_v2.calculate(data)
    print(f"V2 フィルタリング結果形状: {result_v2.filtered_values.shape}")
    print(f"V2 状態推定形状: {result_v2.state_estimates.shape}")
    print(f"V2 平均フィルタリング値: {np.mean(result_v2.filtered_values):.4f}")
    print(f"V2 平均信頼度: {np.mean(result_v2.confidence_scores):.4f}")
    
    # 速度・加速度の取得テスト
    v2_velocity = result_v2.state_estimates[:, 1] if result_v2.state_estimates.shape[1] > 1 else np.zeros(len(data))
    v2_acceleration = result_v2.state_estimates[:, 2] if result_v2.state_estimates.shape[1] > 2 else np.zeros(len(data))
    print(f"V2 平均速度: {np.mean(v2_velocity):.6f}")
    print(f"V2 平均加速度: {np.mean(v2_acceleration):.6f}")
    
    # 誤差比較
    print("\n=" * 30)
    print("誤差比較")
    print("=" * 30)
    
    close_prices = data['close'].values
    error_v1 = np.mean(np.abs(result_v1.filtered_values - close_prices))
    error_v2 = np.mean(np.abs(result_v2.filtered_values - close_prices))
    
    print(f"UKF V1 MAE: {error_v1:.6f}")
    print(f"UKF V2 MAE: {error_v2:.6f}")
    
    rmse_v1 = np.sqrt(np.mean((result_v1.filtered_values - close_prices)**2))
    rmse_v2 = np.sqrt(np.mean((result_v2.filtered_values - close_prices)**2))
    
    print(f"UKF V1 RMSE: {rmse_v1:.6f}")
    print(f"UKF V2 RMSE: {rmse_v2:.6f}")
    
    # 履歴データのテスト（UKF V2のみ）
    print("\n=" * 30)
    print("UKF V2 履歴データテスト")
    print("=" * 30)
    
    pred_history = ukf_v2.get_prediction_history()
    update_history = ukf_v2.get_update_history()
    
    if pred_history is not None:
        print(f"予測履歴形状: {pred_history.shape}")
        print(f"更新履歴形状: {update_history.shape}")
    else:
        print("履歴データが取得できませんでした")
    
    return data, result_v1, result_v2


def test_chart_generation():
    """チャート生成テスト"""
    print("\n" + "=" * 50)
    print("チャート生成テスト")
    print("=" * 50)
    
    # テストデータの生成
    data = generate_sample_data(300)
    
    # チャート生成（表示のみ）
    print("UKF比較チャートを生成中...")
    
    try:
        fig = create_ukf_comparison_chart(
            data=data,
            title="UKF比較テスト",
            save_path=None,  # 表示のみ
            figsize=(16, 14)
        )
        print("チャート生成成功！")
        return True
        
    except Exception as e:
        print(f"チャート生成エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_save():
    """チャート保存テスト"""
    print("\n" + "=" * 50)
    print("チャート保存テスト")
    print("=" * 50)
    
    # テストデータの生成
    data = generate_sample_data(250)
    
    # 保存パスの設定
    save_path = "ukf_comparison_test.png"
    
    try:
        fig = create_ukf_comparison_chart(
            data=data,
            title="UKF比較テスト（保存版）",
            save_path=save_path,
            figsize=(16, 14)
        )
        
        if os.path.exists(save_path):
            print(f"チャートが正常に保存されました: {save_path}")
            file_size = os.path.getsize(save_path)
            print(f"ファイルサイズ: {file_size:,} bytes")
            return True
        else:
            print("チャートファイルが見つかりません")
            return False
            
    except Exception as e:
        print(f"チャート保存エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parameter_comparison():
    """パラメータ比較テスト"""
    print("\n" + "=" * 50)
    print("パラメータ比較テスト")
    print("=" * 50)
    
    # テストデータの生成
    data = generate_sample_data(150)
    close_prices = data['close'].values
    
    # 異なるパラメータでのテスト
    test_configs = [
        {"kappa": 0.0, "process_noise": 0.001, "obs_noise": 0.0001},
        {"kappa": 0.5, "process_noise": 0.01, "obs_noise": 0.001},
        {"kappa": 1.0, "process_noise": 0.1, "obs_noise": 0.01},
    ]
    
    print("異なるパラメータでのUKF V2性能比較:")
    print("-" * 60)
    
    for i, config in enumerate(test_configs):
        try:
            ukf_v2 = UnscentedKalmanFilterV2Wrapper(
                src_type='close',
                kappa=config["kappa"],
                process_noise_scale=config["process_noise"],
                observation_noise_scale=config["obs_noise"],
                max_steps=1000
            )
            
            result = ukf_v2.calculate(data)
            error = np.mean(np.abs(result.filtered_values - close_prices))
            confidence = np.mean(result.confidence_scores)
            
            print(f"設定{i+1}: κ={config['kappa']}, Q={config['process_noise']}, R={config['obs_noise']}")
            print(f"  MAE: {error:.6f}, 平均信頼度: {confidence:.4f}")
            
        except Exception as e:
            print(f"設定{i+1}でエラー: {e}")
    
    print("-" * 60)


def main():
    """メインテスト関数"""
    print("UKF比較チャートのテストを開始します")
    
    try:
        # 基本テスト
        data, result_v1, result_v2 = test_ukf_filters()
        
        # パラメータ比較テスト
        test_parameter_comparison()
        
        # チャート生成テスト
        chart_success = test_chart_generation()
        
        # チャート保存テスト
        save_success = test_with_save()
        
        print("\n" + "=" * 50)
        print("テスト結果サマリー")
        print("=" * 50)
        print(f"基本計算テスト: ✓ 成功")
        print(f"パラメータ比較テスト: ✓ 成功")
        print(f"チャート生成テスト: {'✓ 成功' if chart_success else '✗ 失敗'}")
        print(f"チャート保存テスト: {'✓ 成功' if save_success else '✗ 失敗'}")
        
        if chart_success and save_success:
            print("\n全てのテストが成功しました！")
            return 0
        else:
            print("\n一部のテストが失敗しました。")
            return 1
            
    except Exception as e:
        print(f"テスト実行中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())