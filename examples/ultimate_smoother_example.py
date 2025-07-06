#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
アルティメットスムーザーの使用例
John Ehlersの"The Ultimate Smoother"に基づく実装のテスト
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicators.ultimate_smoother import UltimateSmoother


def create_test_data(length: int = 1000) -> pd.DataFrame:
    """
    テスト用の価格データを作成する
    
    Args:
        length: データの長さ
        
    Returns:
        pd.DataFrame: OHLC価格データ
    """
    # 基本的なトレンドとノイズを含む価格データを生成
    np.random.seed(42)
    
    # 基本価格とトレンド
    base_price = 100.0
    trend = np.linspace(0, 20, length)
    
    # サイクルとノイズを追加
    cycle1 = 10 * np.sin(np.linspace(0, 4 * np.pi, length))
    cycle2 = 5 * np.sin(np.linspace(0, 8 * np.pi, length))
    noise = np.random.normal(0, 2, length)
    
    # 組み合わせた価格
    close = base_price + trend + cycle1 + cycle2 + noise
    
    # OHLC価格を生成
    high = close + np.abs(np.random.normal(0, 1, length))
    low = close - np.abs(np.random.normal(0, 1, length))
    open_price = close + np.random.normal(0, 0.5, length)
    
    return pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close
    })


def test_ultimate_smoother():
    """アルティメットスムーザーのテスト"""
    print("=== アルティメットスムーザーのテスト ===")
    
    # テストデータの作成
    data = create_test_data(500)
    print(f"テストデータ作成: {len(data)}点")
    
    # 異なる期間でのテスト
    periods = [10, 20, 30]
    
    for period in periods:
        print(f"\n--- Period {period} のテスト ---")
        
        # UltimateSmoother作成
        smoother = UltimateSmoother(period=period, src_type='ukf_hlc3')
        
        # 計算実行
        result = smoother.calculate(data)
        
        print(f"計算結果: {len(result.values)}点")
        print(f"有効な値: {np.sum(~np.isnan(result.values))}点")
        print(f"最終値: {result.values[-1]:.4f}")
        print(f"最終係数: {result.coefficients[-1]:.6f}")
        
        # 基本統計
        valid_values = result.values[~np.isnan(result.values)]
        if len(valid_values) > 0:
            print(f"平均値: {np.mean(valid_values):.4f}")
            print(f"標準偏差: {np.std(valid_values):.4f}")
            print(f"最小値: {np.min(valid_values):.4f}")
            print(f"最大値: {np.max(valid_values):.4f}")


def test_comparison_with_ema():
    """EMA、UltimateSmootherの比較"""
    print("\n=== EMA、UltimateSmootherの比較 ===")
    
    # テストデータの作成
    data = create_test_data(200)
    period = 20
    
    # UltimateSmoother
    smoother = UltimateSmoother(period=period, src_type='close')
    result = smoother.calculate(data)
    
    # EMA（簡易版）
    alpha = 3 / period
    ema = np.zeros(len(data))
    ema[0] = data['close'].iloc[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data['close'].iloc[i] + (1 - alpha) * ema[i-1]
    
    # 比較統計
    print(f"期間: {period}")
    print(f"UltimateSmoother最終値: {result.values[-1]:.4f}")
    print(f"EMA最終値: {ema[-1]:.4f}")
    print(f"実際の価格: {data['close'].iloc[-1]:.4f}")
    
    # ラグ測定（簡易版）
    close_prices = data['close'].values
    
    # 有効な値のみで比較
    valid_idx = ~np.isnan(result.values)
    if np.sum(valid_idx) > 10:
        us_values = result.values[valid_idx]
        close_valid = close_prices[valid_idx]
        ema_valid = ema[valid_idx]
        
        # 相関係数（応答性の指標）
        us_corr = np.corrcoef(us_values, close_valid)[0, 1]
        ema_corr = np.corrcoef(ema_valid, close_valid)[0, 1]
        
        print(f"\n価格との相関係数:")
        print(f"UltimateSmoother: {us_corr:.4f}")
        print(f"EMA: {ema_corr:.4f}")


def test_ukf_integration():
    """UKFとの統合テスト"""
    print("\n=== UKF統合テスト ===")
    
    # テストデータの作成
    data = create_test_data(300)
    
    # UKFパラメータ
    ukf_params = {
        'alpha': 0.001,
        'process_noise_scale': 0.0005,
        'volatility_window': 15
    }
    
    try:
        # UKFソースでのテスト
        smoother_ukf = UltimateSmoother(
            period=20, 
            src_type='ukf_close', 
            ukf_params=ukf_params
        )
        result_ukf = smoother_ukf.calculate(data)
        
        # 基本ソースでのテスト
        smoother_basic = UltimateSmoother(period=20, src_type='close')
        result_basic = smoother_basic.calculate(data)
        
        print("UKF統合成功")
        print(f"UKF版最終値: {result_ukf.values[-1]:.4f}")
        print(f"基本版最終値: {result_basic.values[-1]:.4f}")
        
        # 違いを確認
        valid_ukf = ~np.isnan(result_ukf.values)
        valid_basic = ~np.isnan(result_basic.values)
        common_valid = valid_ukf & valid_basic
        
        if np.sum(common_valid) > 0:
            diff = np.mean(np.abs(result_ukf.values[common_valid] - result_basic.values[common_valid]))
            print(f"平均差分: {diff:.4f}")
        
    except Exception as e:
        print(f"UKF統合テストでエラー: {e}")
        print("UKFモジュールが利用できない可能性があります")


def plot_comparison():
    """比較チャートの作成"""
    print("\n=== 比較チャート作成 ===")
    
    try:
        # テストデータの作成
        data = create_test_data(300)
        period = 20
        
        # UltimateSmoother
        smoother = UltimateSmoother(period=period, src_type='ukf_hlc3')
        result = smoother.calculate(data)
        
        # EMA（簡易版）
        alpha = 3 / period
        ema = np.zeros(len(data))
        ema[0] = data['close'].iloc[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data['close'].iloc[i] + (1 - alpha) * ema[i-1]
        
        # プロット
        plt.figure(figsize=(12, 6))
        
        # 価格とスムーザー
        plt.plot(data['close'].values, label='Price', color='black', alpha=0.7)
        plt.plot(result.values, label='UltimateSmoother', color='red', linewidth=2)
        plt.plot(ema, label='EMA', color='green', linewidth=2)
        plt.title(f'Price Smoothing Comparison (Period={period})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # ファイル保存
        output_path = os.path.join(os.path.dirname(__file__), 'output', 'ultimate_smoother_comparison.png')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"チャートを保存しました: {output_path}")
        
        # 表示
        plt.show()
        
    except Exception as e:
        print(f"チャート作成でエラー: {e}")


def main():
    """メイン関数"""
    print("John Ehlers' Ultimate Smoother テスト開始")
    print("=" * 50)
    
    # 基本テスト
    test_ultimate_smoother()
    
    # 比較テスト
    test_comparison_with_ema()
    
    # UKF統合テスト
    test_ukf_integration()
    
    # チャート作成
    plot_comparison()
    
    print("\n" + "=" * 50)
    print("テスト完了")


if __name__ == "__main__":
    main() 