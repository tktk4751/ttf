#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EMDのピーク・バレー検出と平均化のデバッグスクリプト
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


def generate_test_data(length=200):
    """テストデータを生成（論文に近いサイクル・トレンド混在データ）"""
    np.random.seed(42)
    
    # ベース価格
    base_price = 100.0
    prices = [base_price]
    
    # 異なる市場状態を模擬
    for i in range(1, length):
        if i < 50:  # 強いトレンド相場
            trend_component = 0.005  # 上昇トレンド
            cycle_component = 0.02 * np.sin(2 * np.pi * i / 20)  # 20期間サイクル
            noise = np.random.normal(0, 0.008)
        elif i < 100:  # サイクル相場
            trend_component = 0.0  # トレンドなし
            cycle_component = 0.03 * np.sin(2 * np.pi * i / 25)  # 25期間サイクル
            noise = np.random.normal(0, 0.010)
        elif i < 150:  # 弱いトレンド相場
            trend_component = -0.002  # 下降トレンド
            cycle_component = 0.015 * np.sin(2 * np.pi * i / 18)  # 18期間サイクル
            noise = np.random.normal(0, 0.006)
        else:  # 複合相場
            trend_component = 0.003  # 上昇トレンド
            cycle_component = 0.025 * np.sin(2 * np.pi * i / 22)  # 22期間サイクル
            noise = np.random.normal(0, 0.009)
        
        change = trend_component + cycle_component + noise
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # OHLC データの生成
    data = []
    for i, close in enumerate(prices):
        daily_range = abs(np.random.normal(0, close * 0.01))
        
        high = close + daily_range * np.random.uniform(0.3, 1.0)
        low = close - daily_range * np.random.uniform(0.3, 1.0)
        
        if i == 0:
            open_price = close
        else:
            gap = np.random.normal(0, close * 0.005)
            open_price = prices[i-1] + gap
        
        # 論理的整合性の確保
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.uniform(1000, 10000)
        })
    
    return pd.DataFrame(data)


def debug_peak_valley_detection():
    """ピーク・バレー検出のデバッグ"""
    print("=== EMD ピーク・バレー検出デバッグ ===")
    
    # テストデータ生成
    df = generate_test_data(200)
    print(f"テストデータ: {len(df)}ポイント")
    print(f"価格範囲: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    # EMDを計算（複数のパラメータで試す）
    test_configurations = [
        {"period": 20, "delta": 0.1, "avg_period": 10, "fraction": 0.25},
        {"period": 20, "delta": 0.5, "avg_period": 15, "fraction": 0.25},
        {"period": 20, "delta": 0.1, "avg_period": 20, "fraction": 0.1},
        {"period": 15, "delta": 0.3, "avg_period": 8, "fraction": 0.2},
    ]
    
    for i, config in enumerate(test_configurations):
        print(f"\n--- 設定 {i+1}: period={config['period']}, delta={config['delta']}, avg_period={config['avg_period']}, fraction={config['fraction']} ---")
        
        emd = EMD(
            period=config['period'],
            delta=config['delta'],
            avg_period=config['avg_period'],
            fraction=config['fraction'],
            src_type='hlc3',
            use_kalman_filter=False,
            use_smoothing=False
        )
        
        result = emd.calculate(df)
        
        # 結果分析
        valid_bandpass = np.sum(~np.isnan(result.bandpass))
        valid_trend = np.sum(~np.isnan(result.trend))
        
        # ピーク・バレー検出の分析
        valid_peaks = result.peaks[~np.isnan(result.peaks)]
        valid_valleys = result.valleys[~np.isnan(result.valleys)]
        
        print(f"  有効バンドパス値数: {valid_bandpass}/{len(df)}")
        print(f"  有効トレンド値数: {valid_trend}/{len(df)}")
        print(f"  検出されたピーク数: {len(valid_peaks)}")
        print(f"  検出されたバレー数: {len(valid_valleys)}")
        
        if len(valid_peaks) > 0:
            print(f"  ピーク値範囲: {valid_peaks.min():.4f} - {valid_peaks.max():.4f}")
            print(f"  ピーク値平均: {valid_peaks.mean():.4f}")
        
        if len(valid_valleys) > 0:
            print(f"  バレー値範囲: {valid_valleys.min():.4f} - {valid_valleys.max():.4f}")
            print(f"  バレー値平均: {valid_valleys.mean():.4f}")
        
        # 平均化の分析
        valid_avg_peaks = result.avg_peak[~np.isnan(result.avg_peak)]
        valid_avg_valleys = result.avg_valley[~np.isnan(result.avg_valley)]
        
        print(f"  平均化されたピーク数: {len(valid_avg_peaks)}")
        print(f"  平均化されたバレー数: {len(valid_avg_valleys)}")
        
        if len(valid_avg_peaks) > 0:
            print(f"  平均ピーク値範囲: {valid_avg_peaks.min():.4f} - {valid_avg_peaks.max():.4f}")
        
        if len(valid_avg_valleys) > 0:
            print(f"  平均バレー値範囲: {valid_avg_valleys.min():.4f} - {valid_avg_valleys.max():.4f}")
        
        # 上下閾値の分析
        valid_upper = result.upper_threshold[~np.isnan(result.upper_threshold)]
        valid_lower = result.lower_threshold[~np.isnan(result.lower_threshold)]
        
        print(f"  上部閾値有効数: {len(valid_upper)}")
        print(f"  下部閾値有効数: {len(valid_lower)}")
        
        # モード信号の分析
        valid_modes = result.mode_signal[~np.isnan(result.mode_signal)]
        if len(valid_modes) > 0:
            unique_modes, counts = np.unique(valid_modes, return_counts=True)
            print(f"  モード信号分布: {dict(zip(unique_modes, counts))}")
        
        # 詳細デバッグ: ピーク・バレーの位置分析
        peak_indices = np.where(~np.isnan(result.peaks))[0]
        valley_indices = np.where(~np.isnan(result.valleys))[0]
        
        if len(peak_indices) > 0:
            print(f"  ピーク位置サンプル（最初5個）: {peak_indices[:5]}")
        if len(valley_indices) > 0:
            print(f"  バレー位置サンプル（最初5個）: {valley_indices[:5]}")
        
        # avg_period期間内のピーク・バレー密度チェック
        if len(peak_indices) > 0:
            # 最後のavg_period期間でのピーク数
            recent_peaks = peak_indices[peak_indices >= len(df) - config['avg_period']]
            print(f"  最後{config['avg_period']}期間内のピーク数: {len(recent_peaks)}")
        
        if len(valley_indices) > 0:
            # 最後のavg_period期間でのバレー数
            recent_valleys = valley_indices[valley_indices >= len(df) - config['avg_period']]
            print(f"  最後{config['avg_period']}期間内のバレー数: {len(recent_valleys)}")


def main():
    """メイン関数"""
    debug_peak_valley_detection()
    print("\n=== デバッグ完了 ===")


if __name__ == "__main__":
    main()