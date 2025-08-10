#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EMDの中間結果デバッグスクリプト
"""

import sys
import os
import numpy as np
import pandas as pd

# プロジェクトのルートディレクトリをパスに追加
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from indicators.trend_filter.empirical_mode_decomposition import (
    EMD, 
    calculate_averaged_peaks_valleys_improved_numba,
    detect_peaks_valleys_numba
)


def debug_emd_intermediate():
    """EMDの中間結果をデバッグ"""
    print("=== EMD 中間結果デバッグ ===")
    
    # 簡単なテストデータ
    np.random.seed(42)
    length = 50
    
    t = np.arange(length)
    cycle = 5 * np.sin(2 * np.pi * t / 10)
    trend = 0.1 * t
    noise = np.random.normal(0, 0.5, length)
    prices = 100 + trend + cycle + noise
    
    # OHLC データの生成
    data = []
    for i, close in enumerate(prices):
        high = close + abs(np.random.normal(0, 0.5))
        low = close - abs(np.random.normal(0, 0.5))
        
        data.append({
            'open': close,
            'high': high,
            'low': low,
            'close': close,
            'volume': 1000
        })
    
    df = pd.DataFrame(data)
    print(f"テストデータ: {len(df)}ポイント")
    
    # EMDのパラメータ
    period = 10
    delta = 0.3
    avg_period = 8
    fraction = 0.25
    
    print(f"パラメータ: period={period}, delta={delta}, avg_period={avg_period}, fraction={fraction}")
    
    # EMDを計算してすべての中間結果を確認
    emd = EMD(
        period=period,
        delta=delta,
        avg_period=avg_period,
        fraction=fraction,
        src_type='close',
        use_kalman_filter=False,
        use_smoothing=False
    )
    
    # ソース価格を直接取得
    source_prices = df['close'].values
    print(f"ソース価格サンプル: {source_prices[:5]}")
    
    # バンドパスフィルターを手動実行（EMDクラス内と同じ処理）
    from indicators.trend_filter.empirical_mode_decomposition import calculate_bandpass_filter_numba
    
    bandpass = calculate_bandpass_filter_numba(source_prices, period, delta)
    valid_bandpass = np.sum(~np.isnan(bandpass))
    print(f"バンドパス有効値数: {valid_bandpass}/{len(bandpass)}")
    print(f"バンドパス値サンプル: {bandpass[~np.isnan(bandpass)][:5]}")
    
    # ピーク・バレー検出を手動実行
    peaks, valleys = detect_peaks_valleys_numba(bandpass)
    valid_peaks = np.sum(~np.isnan(peaks))
    valid_valleys = np.sum(~np.isnan(valleys))
    print(f"検出ピーク数: {valid_peaks}, バレー数: {valid_valleys}")
    
    # ピーク・バレーの位置と値を詳細表示
    peak_indices = np.where(~np.isnan(peaks))[0]
    valley_indices = np.where(~np.isnan(valleys))[0]
    
    if len(peak_indices) > 0:
        peak_values = peaks[peak_indices]
        print(f"ピーク位置: {peak_indices}")
        print(f"ピーク値: {peak_values}")
    
    if len(valley_indices) > 0:
        valley_values = valleys[valley_indices]
        print(f"バレー位置: {valley_indices}")
        print(f"バレー値: {valley_values}")
    
    # 平均化を手動実行
    print(f"\n=== 平均化関数呼び出し ===")
    avg_peak, avg_valley, upper_threshold, lower_threshold = calculate_averaged_peaks_valleys_improved_numba(
        peaks, valleys, avg_period, fraction
    )
    
    # 結果確認
    valid_avg_peaks = np.sum(~np.isnan(avg_peak))
    valid_avg_valleys = np.sum(~np.isnan(avg_valley))
    valid_upper = np.sum(~np.isnan(upper_threshold))
    valid_lower = np.sum(~np.isnan(lower_threshold))
    
    print(f"平均化結果:")
    print(f"  平均ピーク数: {valid_avg_peaks}")
    print(f"  平均バレー数: {valid_avg_valleys}")
    print(f"  上部閾値数: {valid_upper}")
    print(f"  下部閾値数: {valid_lower}")
    
    if valid_avg_peaks > 0:
        non_nan_avg_peaks = avg_peak[~np.isnan(avg_peak)]
        print(f"  平均ピーク値サンプル: {non_nan_avg_peaks[:3]}")
    
    if valid_avg_valleys > 0:
        non_nan_avg_valleys = avg_valley[~np.isnan(avg_valley)]
        print(f"  平均バレー値サンプル: {non_nan_avg_valleys[:3]}")
    
    # EMDクラスの結果と比較
    print(f"\n=== EMDクラス結果と比較 ===")
    result = emd.calculate(df)
    
    emd_valid_avg_peaks = np.sum(~np.isnan(result.avg_peak))
    emd_valid_avg_valleys = np.sum(~np.isnan(result.avg_valley))
    
    print(f"EMD結果:")
    print(f"  平均ピーク数: {emd_valid_avg_peaks}")
    print(f"  平均バレー数: {emd_valid_avg_valleys}")
    
    # 直接比較
    peaks_match = np.array_equal(avg_peak, result.avg_peak, equal_nan=True)
    valleys_match = np.array_equal(avg_valley, result.avg_valley, equal_nan=True)
    
    print(f"  手動実行とEMDの一致: ピーク={peaks_match}, バレー={valleys_match}")
    
    if not peaks_match or not valleys_match:
        print("  ✗ 不一致が検出されました！")
    else:
        print("  ✓ 手動実行とEMDの結果が一致しています。")
    
    print("=== デバッグ完了 ===")


if __name__ == "__main__":
    debug_emd_intermediate()