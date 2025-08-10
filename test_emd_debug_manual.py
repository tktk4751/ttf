#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EMDの平均化ロジック手動デバッグ
"""

import sys
import os
import numpy as np
import pandas as pd

# プロジェクトのルートディレクトリをパスに追加
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from indicators.trend_filter.empirical_mode_decomposition import EMD


def calculate_averaged_peaks_valleys_debug(
    peaks: np.ndarray,
    valleys: np.ndarray,
    avg_period: int,
    fraction: float
) -> tuple:
    """
    平均化されたピーク・バレーと閾値を計算する（デバッグ版）
    """
    length = len(peaks)
    avg_peak = np.full(length, np.nan, dtype=np.float64)
    avg_valley = np.full(length, np.nan, dtype=np.float64)
    upper_threshold = np.full(length, np.nan, dtype=np.float64)
    lower_threshold = np.full(length, np.nan, dtype=np.float64)
    
    min_required_points = 2  # 最低必要なピーク・バレー数
    
    print(f"デバッグ: 総データ長={length}, avg_period={avg_period}, fraction={fraction}")
    print(f"デバッグ: ピーク有効数={np.sum(~np.isnan(peaks))}, バレー有効数={np.sum(~np.isnan(valleys))}")
    
    # ピーク・バレーの位置を表示
    peak_indices = np.where(~np.isnan(peaks))[0]
    valley_indices = np.where(~np.isnan(valleys))[0]
    print(f"デバッグ: ピーク位置={peak_indices[:10]}")  # 最初の10個
    print(f"デバッグ: バレー位置={valley_indices[:10]}")  # 最初の10個
    
    successful_calculations = 0
    
    for i in range(avg_period - 1, length):
        # ピークの平均（基本期間での検索）
        peak_sum = 0.0
        peak_count = 0
        
        # 基本期間での検索
        basic_range_start = i - avg_period + 1
        basic_range_end = i + 1
        
        for j in range(basic_range_start, basic_range_end):
            if not np.isnan(peaks[j]):
                peak_sum += peaks[j]
                peak_count += 1
        
        # 十分なピークが見つからない場合、拡張検索
        if peak_count < min_required_points:
            # 利用可能な全てのピークから最新のものを使用
            extended_peaks = []
            for j in range(i + 1):
                if not np.isnan(peaks[j]):
                    extended_peaks.append(peaks[j])
            
            if len(extended_peaks) >= min_required_points:
                # 最新のmin_required_points個を使用
                recent_peaks = extended_peaks[-min_required_points:]
                peak_sum = sum(recent_peaks)
                peak_count = len(recent_peaks)
                
                if i < 10:  # 最初の10個のみデバッグ出力
                    print(f"デバッグ[{i}]: 拡張ピーク検索 - recent_peaks={recent_peaks}, count={peak_count}")
        
        # バレーの平均（同様の処理）
        valley_sum = 0.0
        valley_count = 0
        
        # 基本期間での検索
        for j in range(basic_range_start, basic_range_end):
            if not np.isnan(valleys[j]):
                valley_sum += valleys[j]
                valley_count += 1
        
        # 十分なバレーが見つからない場合、拡張検索
        if valley_count < min_required_points:
            extended_valleys = []
            for j in range(i + 1):
                if not np.isnan(valleys[j]):
                    extended_valleys.append(valleys[j])
            
            if len(extended_valleys) >= min_required_points:
                # 最新のmin_required_points個を使用
                recent_valleys = extended_valleys[-min_required_points:]
                valley_sum = sum(recent_valleys)
                valley_count = len(recent_valleys)
                
                if i < 10:  # 最初の10個のみデバッグ出力
                    print(f"デバッグ[{i}]: 拡張バレー検索 - recent_valleys={recent_valleys}, count={valley_count}")
        
        # 平均値と閾値を計算
        if peak_count >= min_required_points:
            avg_peak[i] = peak_sum / peak_count
            upper_threshold[i] = fraction * avg_peak[i]
            successful_calculations += 1
            
            if i < 10:  # 最初の10個のみデバッグ出力
                print(f"デバッグ[{i}]: ピーク平均={avg_peak[i]:.4f}, 上部閾値={upper_threshold[i]:.4f}")
        
        if valley_count >= min_required_points:
            avg_valley[i] = valley_sum / valley_count
            lower_threshold[i] = fraction * avg_valley[i]
            
            if i < 10:  # 最初の10個のみデバッグ出力
                print(f"デバッグ[{i}]: バレー平均={avg_valley[i]:.4f}, 下部閾値={lower_threshold[i]:.4f}")
    
    print(f"デバッグ: 成功した計算数={successful_calculations}")
    
    return avg_peak, avg_valley, upper_threshold, lower_threshold


def main():
    """メイン関数"""
    print("=== EMD 手動デバッグテスト ===")
    
    # シンプルなテストデータ生成（明確なサイクルを持つ）
    np.random.seed(42)
    length = 50  # 短くして見やすくする
    
    # より明確なサイクル
    t = np.arange(length)
    cycle = 5 * np.sin(2 * np.pi * t / 10)  # 10期間サイクル
    trend = 0.1 * t  # 緩やかなトレンド
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
    
    # EMDを計算して中間結果を取得
    emd = EMD(
        period=10,
        delta=0.3,
        avg_period=8,
        fraction=0.25,
        src_type='close',
        use_kalman_filter=False,
        use_smoothing=False
    )
    
    result = emd.calculate(df)
    
    print(f"\n=== EMD標準結果 ===")
    print(f"ピーク数: {np.sum(~np.isnan(result.peaks))}")
    print(f"バレー数: {np.sum(~np.isnan(result.valleys))}")
    print(f"平均ピーク数: {np.sum(~np.isnan(result.avg_peak))}")
    print(f"平均バレー数: {np.sum(~np.isnan(result.avg_valley))}")
    
    print(f"\n=== 手動デバッグ実行 ===")
    # 手動で平均化を実行
    debug_avg_peak, debug_avg_valley, debug_upper, debug_lower = calculate_averaged_peaks_valleys_debug(
        result.peaks, result.valleys, 8, 0.25
    )
    
    print(f"\n=== 手動デバッグ結果 ===")
    print(f"手動平均ピーク数: {np.sum(~np.isnan(debug_avg_peak))}")
    print(f"手動平均バレー数: {np.sum(~np.isnan(debug_avg_valley))}")
    print(f"手動上部閾値数: {np.sum(~np.isnan(debug_upper))}")
    print(f"手動下部閾値数: {np.sum(~np.isnan(debug_lower))}")
    
    # 最初の数値を比較
    if np.sum(~np.isnan(debug_avg_peak)) > 0:
        valid_debug_peaks = debug_avg_peak[~np.isnan(debug_avg_peak)]
        print(f"手動平均ピーク値サンプル: {valid_debug_peaks[:3]}")
    
    if np.sum(~np.isnan(debug_avg_valley)) > 0:
        valid_debug_valleys = debug_avg_valley[~np.isnan(debug_avg_valley)]
        print(f"手動平均バレー値サンプル: {valid_debug_valleys[:3]}")
    
    print("=== テスト完了 ===")


if __name__ == "__main__":
    main()