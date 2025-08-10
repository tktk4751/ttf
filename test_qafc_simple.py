#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quantum Adaptive Flow Channel 簡易テストスクリプト
"""

import numpy as np
import pandas as pd
import time

# プロジェクトのインジケーターをインポート
from indicators import QuantumAdaptiveFlowChannel


def test_qafc_simple():
    """QAFCの簡易テスト"""
    print("=== Quantum Adaptive Flow Channel Simple Test ===\n")
    
    # 簡単なテストデータ（100ポイント）
    n_points = 100
    prices = 100 + np.cumsum(np.random.randn(n_points) * 0.5)
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices + np.abs(np.random.randn(n_points) * 0.2),
        'low': prices - np.abs(np.random.randn(n_points) * 0.2),
        'close': prices,
        'volume': np.ones(n_points) * 1000
    })
    
    # QAFCインジケーター
    qafc = QuantumAdaptiveFlowChannel(
        process_noise=0.01,
        measurement_noise=0.1,
        noise_window=10,  # 小さめの窓
        prediction_lookback=5,  # 小さめのlookback
        base_multiplier=2.0,
        src_type='close'
    )
    
    # 計算実行
    print("Calculating QAFC...")
    start_time = time.time()
    result = qafc.calculate(df)
    calc_time = time.time() - start_time
    print(f"Calculation completed in {calc_time:.3f} seconds")
    
    # 結果確認
    print("\n--- Results Check ---")
    print(f"Centerline shape: {result.centerline.shape}")
    print(f"Non-NaN values in centerline: {np.sum(~np.isnan(result.centerline))}")
    print(f"Non-NaN values in upper channel: {np.sum(~np.isnan(result.upper_channel))}")
    
    # 最後の10個の値を表示
    print("\n--- Last 10 Values ---")
    print("Close prices:", prices[-10:].round(2))
    print("Centerline:", result.centerline[-10:].round(2))
    print("Upper channel:", result.upper_channel[-10:].round(2))
    print("Lower channel:", result.lower_channel[-10:].round(2))
    
    # 統計情報
    valid_centerline = result.centerline[~np.isnan(result.centerline)]
    if len(valid_centerline) > 0:
        print("\n--- Statistics ---")
        print(f"Centerline Mean: {np.mean(valid_centerline):.4f}")
        print(f"Centerline Std: {np.std(valid_centerline):.4f}")
        
        valid_width = result.upper_channel - result.lower_channel
        valid_width = valid_width[~np.isnan(valid_width)]
        if len(valid_width) > 0:
            print(f"Channel Width Mean: {np.mean(valid_width):.4f}")
            print(f"Channel Width Std: {np.std(valid_width):.4f}")
    
    # トレーディングシグナル
    signals = qafc.get_trading_signals(df)
    print("\n--- Trading Signal ---")
    print(f"Signal: {signals.get('signal', 'N/A')}")
    print(f"Position in channel: {signals.get('position_in_channel', 'N/A')}")
    print(f"Trend direction: {signals.get('trend_direction', 'N/A')}")
    print(f"Confidence score: {signals.get('confidence_score', 'N/A')}")
    
    print("\n✓ Test completed successfully!")


if __name__ == "__main__":
    test_qafc_simple()