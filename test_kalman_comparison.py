#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
カルマンフィルター比較チャートのテストスクリプト
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from indicators.smoother.kalman import Kalman
    from indicators.smoother.multivariate_kalman import MultivariateKalman
    from visualization.kalman_comparison_chart import create_kalman_comparison_chart, generate_sample_data
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required modules are available")
    sys.exit(1)


def test_kalman_filters():
    """カルマンフィルターの基本テスト"""
    print("=" * 50)
    print("カルマンフィルター基本テスト")
    print("=" * 50)
    
    # サンプルデータ生成
    print("サンプルデータを生成中...")
    data = generate_sample_data(200)
    print(f"データ形状: {data.shape}")
    print(f"カラム: {list(data.columns)}")
    print(f"データ範囲: {data.index[0]} ～ {data.index[-1]}")
    
    # 従来のカルマンフィルターテスト
    print("\n従来のカルマンフィルター（終値）をテスト中...")
    kalman_close = Kalman(
        process_noise=1e-5,
        observation_noise=1e-3,
        src_type='close'
    )
    
    result_close = kalman_close.calculate(data)
    print(f"フィルタリング結果形状: {result_close.filtered_signal.shape}")
    print(f"状態推定形状: {result_close.state_estimates.shape}")
    print(f"平均フィルタリング値: {np.mean(result_close.filtered_signal):.4f}")
    
    # 従来のカルマンフィルター（HLC3）テスト
    print("\n従来のカルマンフィルター（HLC3）をテスト中...")
    kalman_hlc3 = Kalman(
        process_noise=1e-5,
        observation_noise=1e-3,
        src_type='hlc3'
    )
    
    result_hlc3 = kalman_hlc3.calculate(data)
    print(f"フィルタリング結果形状: {result_hlc3.filtered_signal.shape}")
    print(f"平均フィルタリング値: {np.mean(result_hlc3.filtered_signal):.4f}")
    
    # 多変量カルマンフィルターテスト
    print("\n多変量カルマンフィルターをテスト中...")
    multivariate_kalman = MultivariateKalman(
        process_noise=1e-5,
        observation_noise=1e-3,
        volatility_noise=1e-4
    )
    
    result_multivariate = multivariate_kalman.calculate(data)
    print(f"フィルタリング結果形状: {result_multivariate.filtered_prices.shape}")
    print(f"状態推定形状: {result_multivariate.state_estimates.shape}")
    print(f"平均フィルタリング値: {np.mean(result_multivariate.filtered_prices):.4f}")
    print(f"平均ボラティリティ: {np.mean(result_multivariate.volatility_estimates):.6f}")
    print(f"平均価格レンジ: {np.mean(result_multivariate.price_range_estimates):.6f}")
    print(f"平均信頼度: {np.mean(result_multivariate.confidence_scores):.4f}")
    
    # 誤差比較
    print("\n=" * 30)
    print("誤差比較")
    print("=" * 30)
    
    close_prices = data['close'].values
    error_close = np.mean(np.abs(result_close.filtered_signal - close_prices))
    error_hlc3 = np.mean(np.abs(result_hlc3.filtered_signal - close_prices))
    error_multivariate = np.mean(np.abs(result_multivariate.filtered_prices - close_prices))
    
    print(f"従来カルマン（終値）MAE: {error_close:.6f}")
    print(f"従来カルマン（HLC3）MAE: {error_hlc3:.6f}")
    print(f"多変量カルマン MAE: {error_multivariate:.6f}")
    
    rmse_close = np.sqrt(np.mean((result_close.filtered_signal - close_prices)**2))
    rmse_hlc3 = np.sqrt(np.mean((result_hlc3.filtered_signal - close_prices)**2))
    rmse_multivariate = np.sqrt(np.mean((result_multivariate.filtered_prices - close_prices)**2))
    
    print(f"従来カルマン（終値）RMSE: {rmse_close:.6f}")
    print(f"従来カルマン（HLC3）RMSE: {rmse_hlc3:.6f}")
    print(f"多変量カルマン RMSE: {rmse_multivariate:.6f}")
    
    return data, result_close, result_hlc3, result_multivariate


def test_chart_generation():
    """チャート生成テスト"""
    print("\n" + "=" * 50)
    print("チャート生成テスト")
    print("=" * 50)
    
    # テストデータの生成
    data = generate_sample_data(300)
    
    # チャート生成（表示のみ）
    print("比較チャートを生成中...")
    
    try:
        fig = create_kalman_comparison_chart(
            data=data,
            title="カルマンフィルター比較テスト",
            save_path=None,  # 表示のみ
            figsize=(16, 12)
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
    save_path = "kalman_comparison_test.png"
    
    try:
        fig = create_kalman_comparison_chart(
            data=data,
            title="カルマンフィルター比較テスト（保存版）",
            save_path=save_path,
            figsize=(16, 12)
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


def main():
    """メインテスト関数"""
    print("カルマンフィルター比較チャートのテストを開始します")
    
    try:
        # 基本テスト
        data, result_close, result_hlc3, result_multivariate = test_kalman_filters()
        
        # チャート生成テスト
        chart_success = test_chart_generation()
        
        # チャート保存テスト
        save_success = test_with_save()
        
        print("\n" + "=" * 50)
        print("テスト結果サマリー")
        print("=" * 50)
        print(f"基本計算テスト: ✓ 成功")
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