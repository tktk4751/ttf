#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Numba版と非Numba版の平均化関数比較テスト
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
    calculate_averaged_peaks_valleys_improved_numba,
    detect_peaks_valleys_numba,
    calculate_bandpass_filter_numba
)


def calculate_averaged_peaks_valleys_non_numba(
    peaks: np.ndarray,
    valleys: np.ndarray,
    avg_period: int,
    fraction: float
) -> tuple:
    """
    平均化されたピーク・バレーと閾値を計算する（非Numba版、デバッグ用）
    """
    length = len(peaks)
    avg_peak = np.full(length, np.nan, dtype=np.float64)
    avg_valley = np.full(length, np.nan, dtype=np.float64)
    upper_threshold = np.full(length, np.nan, dtype=np.float64)
    lower_threshold = np.full(length, np.nan, dtype=np.float64)
    
    min_required_points = 1  # 最低必要なピーク・バレー数（実用性を重視）
    
    print(f"非Numba版: 総データ長={length}, avg_period={avg_period}, fraction={fraction}")
    print(f"非Numba版: ピーク有効数={np.sum(~np.isnan(peaks))}, バレー有効数={np.sum(~np.isnan(valleys))}")
    
    successful_peak_calculations = 0
    successful_valley_calculations = 0
    
    for i in range(avg_period - 1, length):
        # ピークの平均（論文に忠実な実装 + 実用的改善）
        peak_sum = 0.0
        peak_count = 0
        
        # 基本期間での検索
        basic_range_start = i - avg_period + 1
        basic_range_end = i + 1
        
        for j in range(basic_range_start, basic_range_end):
            if not np.isnan(peaks[j]):
                peak_sum += peaks[j]
                peak_count += 1
        
        # 十分なピークが見つからない場合、拡張検索（実用的改善）
        if peak_count < min_required_points:
            # 拡張検索：現在位置より前の全有効ピークを後方から探索
            search_sum = 0.0
            search_count = 0
            
            # 後方から検索して最新のピークを見つける
            for j in range(i, -1, -1):
                if not np.isnan(peaks[j]):
                    search_sum += peaks[j]
                    search_count += 1
                    # 必要数に達したら終了
                    if search_count >= min_required_points:
                        break
            
            if search_count >= min_required_points:
                peak_sum = search_sum
                peak_count = search_count
                
                if i < 10:  # 最初の10個のみデバッグ出力
                    print(f"非Numba版[{i}]: 拡張ピーク検索成功 - search_count={search_count}, peak_sum={peak_sum}")
        
        if peak_count >= min_required_points:
            avg_peak[i] = peak_sum / peak_count
            successful_peak_calculations += 1
            
            if i < 10:  # 最初の10個のみデバッグ出力
                print(f"非Numba版[{i}]: ピーク平均={avg_peak[i]:.4f}")
        
        # バレーの平均（同様の改善）
        valley_sum = 0.0
        valley_count = 0
        
        # 基本期間での検索
        for j in range(basic_range_start, basic_range_end):
            if not np.isnan(valleys[j]):
                valley_sum += valleys[j]
                valley_count += 1
        
        # 十分なバレーが見つからない場合、拡張検索
        if valley_count < min_required_points:
            # 拡張検索：現在位置より前の全有効バレーを後方から探索
            search_sum = 0.0
            search_count = 0
            
            # 後方から検索して最新のバレーを見つける
            for j in range(i, -1, -1):
                if not np.isnan(valleys[j]):
                    search_sum += valleys[j]
                    search_count += 1
                    # 必要数に達したら終了
                    if search_count >= min_required_points:
                        break
            
            if search_count >= min_required_points:
                valley_sum = search_sum
                valley_count = search_count
                
                if i < 10:  # 最初の10個のみデバッグ出力
                    print(f"非Numba版[{i}]: 拡張バレー検索成功 - search_count={search_count}, valley_sum={valley_sum}")
        
        if valley_count >= min_required_points:
            avg_valley[i] = valley_sum / valley_count
            successful_valley_calculations += 1
            
            if i < 10:  # 最初の10個のみデバッグ出力
                print(f"非Numba版[{i}]: バレー平均={avg_valley[i]:.4f}")
        
        # 閾値の計算（論文通り）
        if not np.isnan(avg_peak[i]):
            upper_threshold[i] = fraction * avg_peak[i]
        
        if not np.isnan(avg_valley[i]):
            lower_threshold[i] = fraction * avg_valley[i]
    
    print(f"非Numba版: ピーク計算成功数={successful_peak_calculations}, バレー計算成功数={successful_valley_calculations}")
    
    return avg_peak, avg_valley, upper_threshold, lower_threshold


def main():
    """メイン関数"""
    print("=== Numba vs 非Numba 平均化関数比較テスト ===")
    
    # 簡単なテストデータ
    np.random.seed(42)
    length = 50
    
    t = np.arange(length)
    cycle = 5 * np.sin(2 * np.pi * t / 10)
    trend = 0.1 * t
    noise = np.random.normal(0, 0.5, length)
    prices = 100 + trend + cycle + noise
    
    # バンドパスフィルター適用
    period = 10
    delta = 0.3
    avg_period = 8
    fraction = 0.25
    
    bandpass = calculate_bandpass_filter_numba(prices, period, delta)
    peaks, valleys = detect_peaks_valleys_numba(bandpass)
    
    print(f"入力データ:")
    print(f"  ピーク数: {np.sum(~np.isnan(peaks))}")
    print(f"  バレー数: {np.sum(~np.isnan(valleys))}")
    print(f"  ピーク位置: {np.where(~np.isnan(peaks))[0]}")
    print(f"  バレー位置: {np.where(~np.isnan(valleys))[0]}")
    
    # 非Numba版で実行
    print(f"\n=== 非Numba版実行 ===")
    non_numba_avg_peak, non_numba_avg_valley, non_numba_upper, non_numba_lower = calculate_averaged_peaks_valleys_non_numba(
        peaks, valleys, avg_period, fraction
    )
    
    non_numba_peak_count = np.sum(~np.isnan(non_numba_avg_peak))
    non_numba_valley_count = np.sum(~np.isnan(non_numba_avg_valley))
    
    print(f"非Numba版結果:")
    print(f"  平均ピーク数: {non_numba_peak_count}")
    print(f"  平均バレー数: {non_numba_valley_count}")
    
    # Numba版で実行
    print(f"\n=== Numba版実行 ===")
    numba_avg_peak, numba_avg_valley, numba_upper, numba_lower = calculate_averaged_peaks_valleys_improved_numba(
        peaks, valleys, avg_period, fraction
    )
    
    numba_peak_count = np.sum(~np.isnan(numba_avg_peak))
    numba_valley_count = np.sum(~np.isnan(numba_avg_valley))
    
    print(f"Numba版結果:")
    print(f"  平均ピーク数: {numba_peak_count}")
    print(f"  平均バレー数: {numba_valley_count}")
    
    # 結果比較
    print(f"\n=== 結果比較 ===")
    peaks_match = np.array_equal(non_numba_avg_peak, numba_avg_peak, equal_nan=True)
    valleys_match = np.array_equal(non_numba_avg_valley, numba_avg_valley, equal_nan=True)
    
    print(f"一致: ピーク={peaks_match}, バレー={valleys_match}")
    
    if non_numba_peak_count > 0 and numba_peak_count == 0:
        print("✗ Numba版でピーク平均化が失敗しています")
    elif non_numba_valley_count > 0 and numba_valley_count == 0:
        print("✗ Numba版でバレー平均化が失敗しています")
    elif non_numba_peak_count > 0 and non_numba_valley_count > 0:
        print("✓ 非Numba版は正常に動作しています")
        if numba_peak_count > 0 and numba_valley_count > 0:
            print("✓ Numba版も正常に動作しています")
        else:
            print("✗ Numba版に問題があります")
    else:
        print("✗ 両バージョンとも平均化に失敗しています")
    
    print("=== テスト完了 ===")


if __name__ == "__main__":
    main()