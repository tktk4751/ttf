#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from strategies.implementations.x_mama_enhanced.strategy import XMAMAEnhancedStrategy, FilterType

def test_x_mama_enhanced_with_internal_str():
    """内部STR実装X-Choppinessフィルターを使用するX-MAMA Enhancedストラテジーのテスト"""
    print("=== X-MAMA Enhanced（内部STR X-Choppinessフィルター）のテスト ===")
    
    # テストデータ生成
    np.random.seed(42)
    length = 150
    base_price = 100.0
    
    # より明確なトレンドとレンジのデータを生成
    prices = [base_price]
    for i in range(1, length):
        if i < 50:  # 強いトレンド相場
            change = 0.003 + np.random.normal(0, 0.005)
        elif i < 100:  # レンジ相場
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
    
    # X-MAMA Enhanced戦略の設定をテスト
    test_configs = [
        {
            'name': 'X-Choppiness フィルターなし',
            'params': {
                'fast_limit': 0.5,
                'slow_limit': 0.05,
                'src_type': 'hlc3',
                'filter_type': FilterType.NONE,
                'position_mode': True
            }
        },
        {
            'name': 'X-Choppiness フィルター付き',
            'params': {
                'fast_limit': 0.5,
                'slow_limit': 0.05,
                'src_type': 'hlc3',
                'filter_type': FilterType.X_CHOPPINESS,
                'position_mode': True,
                # X-Choppinessパラメータ（サイクル検出器のみ）
                'x_choppiness_detector_type': 'hody_e',
                'x_choppiness_lp_period': 12,
                'x_choppiness_hp_period': 124,
                'x_choppiness_cycle_part': 0.5,
                'x_choppiness_max_cycle': 124,
                'x_choppiness_min_cycle': 12,
                'x_choppiness_max_output': 89,
                'x_choppiness_min_output': 5
            }
        }
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\n--- {config['name']}のテスト ---")
        
        try:
            # ストラテジーの初期化
            strategy = XMAMAEnhancedStrategy(**config['params'])
            
            # エントリーシグナルの生成
            entry_signals = strategy.generate_entry(df)
            
            # X_MAMAシグナルとフィルターシグナルの取得
            x_mama_signals = strategy.get_x_mama_signals(df)
            filter_signals = strategy.get_filter_signals(df)
            
            # ロング・ショートシグナルの分析
            long_signals = strategy.get_long_signals(df)
            short_signals = strategy.get_short_signals(df)
            
            # 統計計算
            valid_entries = np.sum(~np.isnan(entry_signals))
            total_long = np.sum(long_signals == 1) if len(long_signals) > 0 else 0
            total_short = np.sum(short_signals == 1) if len(short_signals) > 0 else 0
            total_entries = total_long + total_short
            
            x_mama_trend_count = np.sum(x_mama_signals == 1) if len(x_mama_signals) > 0 else 0
            x_mama_range_count = np.sum(x_mama_signals == -1) if len(x_mama_signals) > 0 else 0
            
            if config['params']['filter_type'] != FilterType.NONE:
                filter_trend_count = np.sum(filter_signals == 1) if len(filter_signals) > 0 else 0
                filter_range_count = np.sum(filter_signals == -1) if len(filter_signals) > 0 else 0
            else:
                filter_trend_count = filter_range_count = 0
            
            config_result = {
                'config': config['name'],
                'valid_entries': valid_entries,
                'total_long': total_long,
                'total_short': total_short,
                'total_entries': total_entries,
                'x_mama_trend_count': x_mama_trend_count,
                'x_mama_range_count': x_mama_range_count,
                'filter_trend_count': filter_trend_count,
                'filter_range_count': filter_range_count
            }
            
            results.append(config_result)
            
            print(f"  有効エントリー: {valid_entries}/{len(df)}")
            print(f"  ロングシグナル: {total_long}")
            print(f"  ショートシグナル: {total_short}")
            print(f"  総エントリー: {total_entries}")
            print(f"  X-MAMAトレンド判定: {x_mama_trend_count}")
            print(f"  X-MAMAレンジ判定: {x_mama_range_count}")
            
            if config['params']['filter_type'] != FilterType.NONE:
                print(f"  フィルタートレンド判定: {filter_trend_count}")
                print(f"  フィルターレンジ判定: {filter_range_count}")
            
            # フィルター詳細の取得
            if config['params']['filter_type'] == FilterType.X_CHOPPINESS:
                filter_details = strategy.get_filter_details(df)
                if 'x_choppiness_values' in filter_details:
                    x_chop_values = filter_details['x_choppiness_values']
                    str_values = filter_details.get('str_values', np.array([]))
                    
                    if len(x_chop_values) > 0:
                        print(f"  平均X-Choppiness: {np.nanmean(x_chop_values):.4f}")
                    if len(str_values) > 0:
                        print(f"  平均内部STR: {np.nanmean(str_values):.4f}")
            
        except Exception as e:
            print(f"  エラー: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 結果比較
    print("\n=== 設定比較 ===")
    if results:
        print("設定名\\t\\t\\t有効\\tロング\\tショート\\t総エントリー\\tX-MAMAトレンド\\tフィルタートレンド")
        print("-" * 100)
        for result in results:
            print(f"{result['config'][:20]}\\t{result['valid_entries']}\\t{result['total_long']}\\t{result['total_short']}\\t{result['total_entries']}\\t\\t{result['x_mama_trend_count']}\\t\\t{result['filter_trend_count']}")
    
    print("\n=== X-MAMA Enhanced（内部STR X-Choppinessフィルター）テスト完了 ===")

if __name__ == "__main__":
    test_x_mama_enhanced_with_internal_str()