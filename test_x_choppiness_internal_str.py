#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from indicators.trend_filter.x_choppiness import XChoppiness

def test_x_choppiness_internal_str():
    """内部STR実装を使用するX-Choppinessのテスト"""
    print("=== X-Choppiness（内部STR実装）のテスト ===")
    
    # テストデータ生成
    np.random.seed(42)
    length = 100
    base_price = 100.0
    
    # より明確なトレンドとレンジのデータを生成
    prices = [base_price]
    for i in range(1, length):
        if i < 40:  # トレンド相場
            change = 0.003 + np.random.normal(0, 0.005)
        elif i < 80:  # レンジ相場
            change = np.random.normal(0, 0.002)
        else:  # 再びトレンド相場
            change = -0.002 + np.random.normal(0, 0.004)
        
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # OHLC データの生成
    data = []
    for i, close in enumerate(prices):
        daily_range = close * 0.005
        
        high = close + daily_range * np.random.uniform(0.3, 1.0)
        low = close - daily_range * np.random.uniform(0.3, 1.0)
        
        if i == 0:
            open_price = close
        else:
            gap = np.random.normal(0, close * 0.001)
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
    
    # 複数の設定でX-Choppinessをテスト
    test_configs = [
        {
            'name': '基本設定（内部STR）',
            'params': {
                'period': 14,
                'midline_period': 50,
                'use_smoothing': False,
                'use_dynamic_period': False,
                'use_kalman_filter': False
            }
        },
        {
            'name': '動的期間設定',
            'params': {
                'period': 14,
                'midline_period': 50,
                'use_smoothing': False,
                'use_dynamic_period': True,
                'detector_type': 'hody_e',
                'lp_period': 10,
                'hp_period': 80,
                'max_cycle': 80,
                'min_cycle': 10,
                'use_kalman_filter': False
            }
        },
        {
            'name': '平滑化設定',
            'params': {
                'period': 14,
                'midline_period': 50,
                'use_smoothing': True,
                'smoother_type': 'frama',
                'smoother_period': 8,
                'use_dynamic_period': False,
                'use_kalman_filter': False
            }
        }
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\\n--- {config['name']}のテスト ---")
        
        try:
            x_chop = XChoppiness(**config['params'])
            result = x_chop.calculate(df)
            
            # 基本統計
            valid_values = result.values[~np.isnan(result.values)]
            valid_trend_signals = result.trend_signal[~np.isnan(result.trend_signal)]
            valid_str = result.str_values[~np.isnan(result.str_values)]
            
            if len(valid_values) > 0:
                trend_count = np.sum(valid_trend_signals == 1) if len(valid_trend_signals) > 0 else 0
                range_count = np.sum(valid_trend_signals == -1) if len(valid_trend_signals) > 0 else 0
                total_signals = trend_count + range_count
                
                config_result = {
                    'config': config['name'],
                    'valid_values': len(valid_values),
                    'avg_x_choppiness': np.mean(valid_values),
                    'std_x_choppiness': np.std(valid_values),
                    'avg_str': np.mean(valid_str) if len(valid_str) > 0 else np.nan,
                    'std_str': np.std(valid_str) if len(valid_str) > 0 else np.nan,
                    'trend_count': trend_count,
                    'range_count': range_count,
                    'trend_ratio': trend_count / total_signals if total_signals > 0 else 0
                }
                
                results.append(config_result)
                
                print(f"  有効X-Choppiness値: {len(valid_values)}/{len(df)}")
                print(f"  平均X-Choppiness: {config_result['avg_x_choppiness']:.4f}")
                print(f"  X-Choppiness標準偏差: {config_result['std_x_choppiness']:.4f}")
                print(f"  平均STR: {config_result['avg_str']:.4f}")
                print(f"  STR標準偏差: {config_result['std_str']:.4f}")
                print(f"  トレンド判定: {trend_count} ({config_result['trend_ratio']:.1%})")
                print(f"  レンジ判定: {range_count}")
                
                # 時系列での変化確認
                if len(valid_trend_signals) >= 60:  # 十分な長さがある場合
                    period_length = len(valid_trend_signals) // 3
                    periods = ['期間1 (トレンド)', '期間2 (レンジ)', '期間3 (トレンド)']
                    
                    for p in range(3):
                        start_idx = p * period_length
                        end_idx = (p + 1) * period_length if p < 2 else len(valid_trend_signals)
                        period_signals = valid_trend_signals[start_idx:end_idx]
                        period_trend_ratio = np.sum(period_signals == 1) / len(period_signals) if len(period_signals) > 0 else 0
                        print(f"    {periods[p]}: トレンド比率 {period_trend_ratio:.1%}")
                
            else:
                print("  有効なX-Choppiness値が生成されませんでした")
                
        except Exception as e:
            print(f"  エラー: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 結果比較
    print("\\n=== 設定比較 ===")
    if results:
        print("設定名\\t\\t有効\\t平均X-Chop\\tX-Chop標準偏差\\t平均STR\\tSTR標準偏差\\tトレンド比率")
        print("-" * 90)
        for result in results:
            print(f"{result['config'][:12]}\\t{result['valid_values']}\\t{result['avg_x_choppiness']:.4f}\\t\\t{result['std_x_choppiness']:.4f}\\t\\t{result['avg_str']:.4f}\\t{result['std_str']:.4f}\\t{result['trend_ratio']:.1%}")
    
    print("\\n=== 内部STR実装X-Choppinessテスト完了 ===")

if __name__ == "__main__":
    test_x_choppiness_internal_str()