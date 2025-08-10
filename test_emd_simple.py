#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EMDの簡単なテストスクリプト
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


def main():
    """メイン関数"""
    print("=== EMD 改善テスト ===")
    
    # シンプルなテストデータ生成
    np.random.seed(42)
    length = 100
    
    # サイン波 + トレンド + ノイズ
    t = np.arange(length)
    trend = 0.02 * t  # 線形トレンド
    cycle = 10 * np.sin(2 * np.pi * t / 20)  # 20期間サイクル
    noise = np.random.normal(0, 2, length)
    prices = 100 + trend + cycle + noise
    
    # OHLC データの生成
    data = []
    for i, close in enumerate(prices):
        high = close + abs(np.random.normal(0, 1))
        low = close - abs(np.random.normal(0, 1))
        
        data.append({
            'open': close,
            'high': high,
            'low': low,
            'close': close,
            'volume': 1000
        })
    
    df = pd.DataFrame(data)
    print(f"テストデータ: {len(df)}ポイント")
    
    # EMDを計算（改善されたパラメータで）
    emd = EMD(
        period=20,
        delta=0.3,
        avg_period=10,
        fraction=0.2,
        src_type='close',
        use_kalman_filter=False,
        use_smoothing=False
    )
    
    result = emd.calculate(df)
    
    # 結果確認
    print(f"バンドパス有効値: {np.sum(~np.isnan(result.bandpass))}/{len(df)}")
    print(f"トレンド有効値: {np.sum(~np.isnan(result.trend))}/{len(df)}")
    
    # ピーク・バレー
    valid_peaks = result.peaks[~np.isnan(result.peaks)]
    valid_valleys = result.valleys[~np.isnan(result.valleys)]
    print(f"検出ピーク数: {len(valid_peaks)}")
    print(f"検出バレー数: {len(valid_valleys)}")
    
    # 平均化結果
    valid_avg_peaks = result.avg_peak[~np.isnan(result.avg_peak)]
    valid_avg_valleys = result.avg_valley[~np.isnan(result.avg_valley)]
    print(f"平均化ピーク数: {len(valid_avg_peaks)}")
    print(f"平均化バレー数: {len(valid_avg_valleys)}")
    
    # 閾値
    valid_upper = result.upper_threshold[~np.isnan(result.upper_threshold)]
    valid_lower = result.lower_threshold[~np.isnan(result.lower_threshold)]
    print(f"上部閾値有効数: {len(valid_upper)}")
    print(f"下部閾値有効数: {len(valid_lower)}")
    
    # モード信号
    valid_modes = result.mode_signal[~np.isnan(result.mode_signal)]
    if len(valid_modes) > 0:
        unique_modes, counts = np.unique(valid_modes, return_counts=True)
        print(f"モード信号分布: {dict(zip(unique_modes, counts))}")
        
        # サンプル値
        if len(valid_avg_peaks) > 0:
            print(f"平均ピーク値サンプル: {valid_avg_peaks[:3]}")
        if len(valid_avg_valleys) > 0:
            print(f"平均バレー値サンプル: {valid_avg_valleys[:3]}")
        if len(valid_upper) > 0:
            print(f"上部閾値サンプル: {valid_upper[:3]}")
        if len(valid_lower) > 0:
            print(f"下部閾値サンプル: {valid_lower[:3]}")
    
    success = (len(valid_avg_peaks) > 0 and len(valid_avg_valleys) > 0 and 
               len(valid_upper) > 0 and len(valid_lower) > 0)
    
    if success:
        print("\n✓ 改善されました！平均化とモード判定が正常に機能しています。")
    else:
        print("\n✗ まだ問題があります。平均化が機能していません。")
    
    print("=== テスト完了 ===")
    return success


if __name__ == "__main__":
    main()