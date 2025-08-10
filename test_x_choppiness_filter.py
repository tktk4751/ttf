#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from signals.implementations.x_choppiness.filter import XChoppinessFilterSignal
from indicators.trend_filter.x_choppiness import XChoppiness

def test_x_choppiness_filter():
    """X-Choppiness フィルターシグナルのテスト"""
    print("=== X-Choppiness フィルターシグナルのテスト ===")
    
    # テストデータ生成
    np.random.seed(42)
    length = 100
    base_price = 100.0
    
    # 明確なトレンドとレンジのデータを生成
    prices = [base_price]
    for i in range(1, length):
        if i < 50:  # 明確なトレンド相場（上昇）
            change = 0.005 + np.random.normal(0, 0.005)  # より明確なトレンド
        else:  # 明確なレンジ相場
            change = np.random.normal(0, 0.003)  # より狭いレンジ
        
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # OHLC データの生成
    data = []
    for i, close in enumerate(prices):
        daily_range = close * 0.005  # より小さいレンジ
        
        high = close + daily_range * np.random.uniform(0.3, 1.0)
        low = close - daily_range * np.random.uniform(0.3, 1.0)
        
        if i == 0:
            open_price = close
        else:
            gap = np.random.normal(0, close * 0.002)
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
    
    df = pd.DataFrame(data)
    
    print(f"テストデータ: {len(df)}ポイント")
    print(f"価格範囲: {df['close'].min():.2f} - {df['close'].max():.2f}")
    print(f"価格変化: {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:.2f}%")
    
    # まずX-Choppinessインジケーター単体でテスト
    print("\n--- X-Choppinessインジケーター単体テスト ---")
    x_chop = XChoppiness(
        period=10,  # 短い期間でテスト
        midline_period=20,  # 短いミッドライン期間
        use_smoothing=False,  # 平滑化を無効にしてシンプルに
        use_dynamic_period=False,
        use_kalman_filter=False,
        enable_percentile_analysis=False
    )
    
    try:
        result = x_chop.calculate(df)
        print(f"X-Choppiness値の範囲: {np.nanmin(result.values):.4f} - {np.nanmax(result.values):.4f}")
        print(f"ミッドラインの範囲: {np.nanmin(result.midline):.4f} - {np.nanmax(result.midline):.4f}")
        print(f"トレンドシグナル: {np.unique(result.trend_signal[~np.isnan(result.trend_signal)])}")
        print(f"有効なトレンドシグナル数: {np.sum(~np.isnan(result.trend_signal))}")
        
        # トレンドシグナルの統計
        trend_signals = result.trend_signal[~np.isnan(result.trend_signal)]
        if len(trend_signals) > 0:
            trend_count = np.sum(trend_signals == 1)
            range_count = np.sum(trend_signals == -1)
            print(f"トレンドシグナル = 1の数: {trend_count}")
            print(f"レンジシグナル = -1の数: {range_count}")
    except Exception as e:
        print(f"X-Choppinessインジケーター計算でエラー: {str(e)}")
        return
    
    # X-Choppinessフィルターシグナルのテスト
    print("\n--- X-Choppinessフィルターシグナルテスト ---")
    filter_signal = XChoppinessFilterSignal(
        period=10,  # 短い期間でテスト
        midline_period=20,  # 短いミッドライン期間
        use_smoothing=False,  # 平滑化を無効にしてシンプルに
        use_dynamic_period=False,
        use_kalman_filter=False,
        enable_percentile_analysis=False
    )
    
    try:
        signals = filter_signal.generate(df)
        print(f"生成されたシグナルの形状: {signals.shape}")
        print(f"ユニークなシグナル値: {np.unique(signals[~np.isnan(signals)])}")
        
        valid_signals = signals[~np.isnan(signals)]
        if len(valid_signals) > 0:
            trend_count = np.sum(valid_signals == 1)
            range_count = np.sum(valid_signals == -1)
            print(f"有効シグナル数: {len(valid_signals)}/{len(signals)}")
            print(f"トレンド判定 (1): {trend_count} ({trend_count/len(valid_signals)*100:.1f}%)")
            print(f"レンジ判定 (-1): {range_count} ({range_count/len(valid_signals)*100:.1f}%)")
        else:
            print("有効なシグナルが生成されませんでした")
            
        # X-Choppiness値の確認
        x_chop_values = filter_signal.get_x_choppiness_values()
        print(f"X-Choppiness値の数: {len(x_chop_values[~np.isnan(x_chop_values)])}")
        
        # トレンドシグナル値の確認
        trend_signal_values = filter_signal.get_trend_signal_values()
        print(f"トレンドシグナル値の数: {len(trend_signal_values[~np.isnan(trend_signal_values)])}")
        
    except Exception as e:
        print(f"フィルターシグナル生成でエラー: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n=== テスト完了 ===")

if __name__ == "__main__":
    test_x_choppiness_filter()