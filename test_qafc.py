#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quantum Adaptive Flow Channel テストスクリプト
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# プロジェクトのインジケーターをインポート
from indicators import QuantumAdaptiveFlowChannel, CosmicUniversalAdaptiveChannel


def generate_test_data(n_points=500):
    """テスト用の価格データを生成"""
    # 時系列インデックス
    dates = pd.date_range(end=datetime.now(), periods=n_points, freq='h')
    
    # トレンドとノイズを含む価格データを生成
    trend = np.linspace(100, 120, n_points) + 10 * np.sin(np.linspace(0, 4*np.pi, n_points))
    noise = np.random.normal(0, 1, n_points)
    prices = trend + noise
    
    # OHLCV形式のDataFrame作成
    df = pd.DataFrame({
        'datetime': dates,
        'open': prices + np.random.uniform(-0.5, 0.5, n_points),
        'high': prices + np.random.uniform(0, 1.5, n_points),
        'low': prices - np.random.uniform(0, 1.5, n_points),
        'close': prices,
        'volume': np.random.uniform(1000, 5000, n_points)
    })
    
    return df


def test_qafc():
    """QAFCインジケーターのテスト"""
    print("=== Quantum Adaptive Flow Channel Test ===\n")
    
    # テストデータ生成
    df = generate_test_data(500)
    
    # QAFCインジケーター作成
    qafc = QuantumAdaptiveFlowChannel(
        process_noise=0.01,
        measurement_noise=0.1,
        noise_window=20,
        prediction_lookback=10,
        base_multiplier=2.0,
        src_type='close'
    )
    
    # 計算実行
    print("Calculating QAFC...")
    result = qafc.calculate(df)
    
    # 結果の統計情報表示
    print("\n--- QAFC Results Summary ---")
    print(f"Centerline - Mean: {np.nanmean(result.centerline):.4f}")
    print(f"Channel Width - Mean: {np.nanmean(result.upper_channel - result.lower_channel):.4f}")
    print(f"Trend Strength - Mean: {np.nanmean(result.trend_strength):.4f}")
    print(f"Confidence Score - Mean: {np.nanmean(result.confidence_score):.4f}")
    
    # トレーディングシグナル取得
    signals = qafc.get_trading_signals(df)
    print("\n--- Latest Trading Signal ---")
    for key, value in signals.items():
        print(f"{key}: {value}")
    
    # プロット
    plt.figure(figsize=(14, 8))
    
    # 価格とチャネル
    plt.subplot(3, 1, 1)
    plt.plot(df.index, df['close'], 'k-', label='Close Price', alpha=0.7)
    plt.plot(df.index, result.centerline, 'b-', label='QAFC Centerline', linewidth=2)
    plt.fill_between(df.index, result.upper_channel, result.lower_channel, 
                     alpha=0.2, color='blue', label='QAFC Channel')
    plt.plot(df.index, result.predicted_price, 'g--', label='Predicted Price', alpha=0.7)
    plt.legend()
    plt.title('Quantum Adaptive Flow Channel')
    plt.ylabel('Price')
    
    # トレンド強度と信頼度
    plt.subplot(3, 1, 2)
    plt.plot(df.index, result.trend_strength, 'r-', label='Trend Strength')
    plt.plot(df.index, result.confidence_score, 'g-', label='Confidence Score')
    plt.legend()
    plt.ylabel('Score')
    plt.ylim(0, 1)
    
    # モメンタムフロー
    plt.subplot(3, 1, 3)
    plt.plot(df.index, result.momentum_flow, 'purple', label='Momentum Flow')
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.legend()
    plt.ylabel('Momentum')
    plt.xlabel('Time')
    
    plt.tight_layout()
    plt.savefig('qafc_test_result.png', dpi=150)
    print("\nPlot saved as 'qafc_test_result.png'")
    
    return result


def compare_channels():
    """QAFCとCUAVCの比較"""
    print("\n\n=== Comparing QAFC vs CUAVC ===\n")
    
    # テストデータ
    df = generate_test_data(300)
    
    # QAFC
    qafc = QuantumAdaptiveFlowChannel()
    qafc_result = qafc.calculate(df)
    
    # CUAVC
    cuavc = CosmicUniversalAdaptiveChannel()
    cuavc_result = cuavc.calculate(df)
    
    # 比較プロット
    plt.figure(figsize=(14, 10))
    
    # QAFC
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df['close'], 'k-', label='Close Price', alpha=0.7)
    plt.plot(df.index, qafc_result.centerline, 'b-', label='QAFC Centerline', linewidth=2)
    plt.fill_between(df.index, qafc_result.upper_channel, qafc_result.lower_channel, 
                     alpha=0.2, color='blue', label='QAFC Channel')
    plt.legend()
    plt.title('Quantum Adaptive Flow Channel (QAFC)')
    plt.ylabel('Price')
    
    # CUAVC
    plt.subplot(2, 1, 2)
    plt.plot(df.index, df['close'], 'k-', label='Close Price', alpha=0.7)
    plt.plot(df.index, cuavc_result.cosmic_centerline, 'r-', label='CUAVC Centerline', linewidth=2)
    plt.fill_between(df.index, cuavc_result.upper_channel, cuavc_result.lower_channel, 
                     alpha=0.2, color='red', label='CUAVC Channel')
    plt.legend()
    plt.title('Cosmic Universal Adaptive Channel (CUAVC)')
    plt.ylabel('Price')
    plt.xlabel('Time')
    
    plt.tight_layout()
    plt.savefig('qafc_vs_cuavc_comparison.png', dpi=150)
    print("Comparison plot saved as 'qafc_vs_cuavc_comparison.png'")
    
    # パフォーマンス比較
    print("\n--- Performance Comparison ---")
    
    # 遅延測定（センターラインの価格への追従性）
    qafc_lag = np.nanmean(np.abs(df['close'].values - qafc_result.centerline))
    cuavc_lag = np.nanmean(np.abs(df['close'].values - cuavc_result.cosmic_centerline))
    
    print(f"QAFC Average Lag: {qafc_lag:.4f}")
    print(f"CUAVC Average Lag: {cuavc_lag:.4f}")
    print(f"Improvement: {((cuavc_lag - qafc_lag) / cuavc_lag * 100):.1f}% lower lag")
    
    # チャネル幅の適応性
    qafc_width_std = np.nanstd(qafc_result.upper_channel - qafc_result.lower_channel)
    cuavc_width_std = np.nanstd(cuavc_result.upper_channel - cuavc_result.lower_channel)
    
    print(f"\nQAFC Channel Width Std: {qafc_width_std:.4f}")
    print(f"CUAVC Channel Width Std: {cuavc_width_std:.4f}")
    print(f"QAFC shows {((qafc_width_std / cuavc_width_std - 1) * 100):.1f}% more adaptive width variation")


if __name__ == "__main__":
    # QAFCテスト実行
    test_qafc()
    
    # チャネル比較
    compare_channels()
    
    print("\n\nTest completed successfully!")