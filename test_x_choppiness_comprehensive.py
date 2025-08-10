#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from signals.implementations.x_choppiness.filter import XChoppinessFilterSignal
from signals.implementations.cycle_trend_index.fillter import CycleTrendIndexFilterSignal

def test_comprehensive():
    """包括的なX-Choppinessフィルターテスト"""
    print("=== X-Choppiness フィルター 包括テスト ===")
    
    # より長いテストデータを生成
    np.random.seed(123)
    length = 300
    base_price = 100.0
    
    # 複雑な市場状況を模擬
    prices = [base_price]
    for i in range(1, length):
        if i < 75:  # 強い上昇トレンド
            change = 0.006 + np.random.normal(0, 0.008)
        elif i < 150:  # レンジ相場
            change = np.random.normal(0, 0.004)
        elif i < 225:  # 強い下降トレンド
            change = -0.005 + np.random.normal(0, 0.008)
        else:  # レンジ相場
            change = np.random.normal(0, 0.003)
        
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, base_price * 0.5))  # 極端な下落を防ぐ
    
    # OHLC データの生成
    data = []
    for i, close in enumerate(prices):
        daily_range = close * 0.008
        
        high = close + daily_range * np.random.uniform(0.2, 1.0)
        low = close - daily_range * np.random.uniform(0.2, 1.0)
        
        if i == 0:
            open_price = close
        else:
            gap = np.random.normal(0, close * 0.003)
            open_price = prices[i-1] + gap
        
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
    print(f"総価格変化: {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:.2f}%")
    
    # 複数の設定でX-Choppinessフィルターをテスト
    test_configs = [
        {
            'name': '基本設定',
            'params': {
                'period': 14,
                'midline_period': 50,
                'use_smoothing': False,
                'use_dynamic_period': False,
                'use_kalman_filter': False
            }
        },
        {
            'name': '平滑化有効',
            'params': {
                'period': 14,
                'midline_period': 50,
                'use_smoothing': True,
                'smoother_type': 'super_smoother',
                'smoother_period': 8,
                'use_dynamic_period': False,
                'use_kalman_filter': False
            }
        },
        {
            'name': '短期設定',
            'params': {
                'period': 7,
                'midline_period': 25,
                'str_period': 10.0,
                'use_smoothing': False,
                'use_dynamic_period': False,
                'use_kalman_filter': False
            }
        },
        {
            'name': '長期設定',
            'params': {
                'period': 21,
                'midline_period': 100,
                'str_period': 30.0,
                'use_smoothing': True,
                'smoother_type': 'frama',
                'smoother_period': 12,
                'use_dynamic_period': False,
                'use_kalman_filter': False
            }
        }
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\n--- {config['name']}のテスト ---")
        
        try:
            filter_signal = XChoppinessFilterSignal(**config['params'])
            signals = filter_signal.generate(df)
            
            # 基本統計
            valid_signals = signals[~np.isnan(signals)]
            if len(valid_signals) > 0:
                trend_count = np.sum(valid_signals == 1)
                range_count = np.sum(valid_signals == -1)
                trend_ratio = trend_count / len(valid_signals)
                
                # X-Choppiness値の統計
                x_chop_values = filter_signal.get_x_choppiness_values()
                valid_x_chop = x_chop_values[~np.isnan(x_chop_values)]
                
                # ミッドライン値の統計
                midline_values = filter_signal.get_midline_values()
                valid_midline = midline_values[~np.isnan(midline_values)]
                
                result = {
                    'config': config['name'],
                    'valid_signals': len(valid_signals),
                    'trend_ratio': trend_ratio,
                    'range_ratio': 1 - trend_ratio,
                    'avg_x_choppiness': np.mean(valid_x_chop) if len(valid_x_chop) > 0 else np.nan,
                    'avg_midline': np.mean(valid_midline) if len(valid_midline) > 0 else np.nan,
                    'x_chop_std': np.std(valid_x_chop) if len(valid_x_chop) > 0 else np.nan
                }
                
                results.append(result)
                
                print(f"  有効シグナル: {len(valid_signals)}/{len(df)}")
                print(f"  トレンド判定: {trend_ratio:.1%}")
                print(f"  レンジ判定: {1-trend_ratio:.1%}")
                print(f"  平均X-Choppiness: {result['avg_x_choppiness']:.4f}")
                print(f"  平均ミッドライン: {result['avg_midline']:.4f}")
                print(f"  X-Choppiness標準偏差: {result['x_chop_std']:.4f}")
                
                # 時系列での変化を確認（4つの期間）
                period_length = len(valid_signals) // 4
                for p in range(4):
                    start_idx = p * period_length
                    end_idx = (p + 1) * period_length if p < 3 else len(valid_signals)
                    period_signals = valid_signals[start_idx:end_idx]
                    period_trend_ratio = np.sum(period_signals == 1) / len(period_signals) if len(period_signals) > 0 else 0
                    print(f"    期間{p+1} ({start_idx}-{end_idx}): トレンド比率 {period_trend_ratio:.1%}")
                
        except Exception as e:
            print(f"  エラー: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 結果比較
    print("\n=== 設定比較 ===")
    if results:
        print("設定名\t\t有効\tトレンド\tレンジ\t平均X-Chop\tX-Chop標準偏差")
        print("-" * 70)
        for result in results:
            print(f"{result['config'][:8]}\t{result['valid_signals']}\t{result['trend_ratio']:.1%}\t{result['range_ratio']:.1%}\t{result['avg_x_choppiness']:.3f}\t{result['x_chop_std']:.3f}")
    
    # CycleTrendIndexFilterSignalとの比較（参考用）
    print("\n--- 参考: CycleTrendIndexFilter vs X-ChoppinessFilter ---")
    try:
        cycle_filter = CycleTrendIndexFilterSignal()
        cycle_signals = cycle_filter.generate(df)
        cycle_valid = cycle_signals[~np.isnan(cycle_signals)]
        
        x_chop_filter = XChoppinessFilterSignal(period=14, midline_period=50, use_smoothing=False)
        x_chop_signals = x_chop_filter.generate(df)
        x_chop_valid = x_chop_signals[~np.isnan(x_chop_signals)]
        
        if len(cycle_valid) > 0 and len(x_chop_valid) > 0:
            print(f"CycleTrendIndex: トレンド比率 {np.sum(cycle_valid == 1)/len(cycle_valid):.1%}")
            print(f"X-Choppiness: トレンド比率 {np.sum(x_chop_valid == 1)/len(x_chop_valid):.1%}")
            
            # 一致度の計算（同じ長さの部分で比較）
            min_len = min(len(cycle_valid), len(x_chop_valid))
            if min_len > 0:
                agreement = np.sum(cycle_valid[-min_len:] == x_chop_valid[-min_len:]) / min_len
                print(f"シグナル一致度: {agreement:.1%}")
        
    except Exception as e:
        print(f"比較テストでエラー: {str(e)}")
    
    print("\n=== 包括テスト完了 ===")

if __name__ == "__main__":
    test_comprehensive()