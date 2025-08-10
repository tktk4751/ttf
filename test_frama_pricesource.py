#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FRAMA PriceSource Implementation Test
PriceSourceクラス対応版のFRAMAテスト
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from indicators.smoother.frama import FRAMA

def create_test_data(n_days=100):
    """テスト用のOHLCデータを作成"""
    np.random.seed(42)
    
    # 日付インデックス
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    # 価格データ
    base_price = 100
    trend = np.linspace(0, 20, n_days)
    cyclical = 5 * np.sin(np.linspace(0, 2*np.pi, n_days))
    noise = np.random.normal(0, 1, n_days)
    
    close_price = base_price + trend + cyclical + noise
    
    # OHLC作成
    open_price = close_price + np.random.normal(0, 0.5, n_days)
    high_price = np.maximum(open_price, close_price) + np.random.uniform(0, 2, n_days)
    low_price = np.minimum(open_price, close_price) - np.random.uniform(0, 2, n_days)
    volume = np.random.uniform(1000, 5000, n_days)
    
    # DataFrame作成
    data = pd.DataFrame({
        'open': open_price,
        'high': high_price,
        'low': low_price,
        'close': close_price,
        'volume': volume
    }, index=pd.DatetimeIndex(dates))
    
    return data

def test_frama_pricesource():
    """PriceSource対応FRAMAのテスト"""
    print("=== FRAMA PriceSource Implementation Test ===")
    
    # テストデータ作成
    data = create_test_data(100)
    print(f"テストデータ作成完了: {len(data)} 日分")
    
    # 異なるソースタイプでテスト
    source_types = ['close', 'hl2', 'hlc3', 'ohlc4']
    
    results = {}
    
    for src_type in source_types:
        print(f"\n--- {src_type} ソースタイプテスト ---")
        
        try:
            # FRAMAを計算（PineScript準拠設定）
            frama = FRAMA(
                period=16,
                src_type=src_type,
                fc=1,
                sc=198
            )
            
            result = frama.calculate(data)
            results[src_type] = result
            
            # 統計情報を表示
            valid_values = result.values[~np.isnan(result.values)]
            valid_dim = result.fractal_dimension[~np.isnan(result.fractal_dimension)]
            valid_alpha = result.alpha[~np.isnan(result.alpha)]
            
            print(f"  有効値数: {len(valid_values)}")
            print(f"  FRAMA値: {np.mean(valid_values):.4f} ± {np.std(valid_values):.4f}")
            print(f"  フラクタル次元: {np.mean(valid_dim):.4f} (範囲: {np.min(valid_dim):.4f} - {np.max(valid_dim):.4f})")
            print(f"  アルファ値: {np.mean(valid_alpha):.4f} (範囲: {np.min(valid_alpha):.4f} - {np.max(valid_alpha):.4f})")
            
            # PriceSourceが正常に動作していることを確認
            print(f"  ✓ PriceSource ({src_type}) 正常動作")
            
        except Exception as e:
            print(f"  ✗ エラー: {e}")
            import traceback
            traceback.print_exc()
    
    # NumPy配列での動作確認
    print(f"\n--- NumPy配列テスト ---")
    try:
        # DataFrameをNumPy配列に変換
        numpy_data = data[['open', 'high', 'low', 'close']].values
        
        frama = FRAMA(period=16, src_type='hl2', fc=1, sc=198)
        result = frama.calculate(numpy_data)
        
        valid_values = result.values[~np.isnan(result.values)]
        print(f"  NumPy配列: 有効値数 {len(valid_values)}, FRAMA平均 {np.mean(valid_values):.4f}")
        print(f"  ✓ NumPy配列対応正常動作")
        
    except Exception as e:
        print(f"  ✗ NumPy配列エラー: {e}")
    
    # キャッシュ機能のテスト
    print(f"\n--- キャッシュ機能テスト ---")
    try:
        frama = FRAMA(period=16, src_type='hlc3', fc=1, sc=198)
        
        # 1回目の計算
        result1 = frama.calculate(data)
        
        # 2回目の計算（キャッシュされるはず）
        result2 = frama.calculate(data)
        
        # 結果が同じであることを確認
        if np.array_equal(result1.values, result2.values, equal_nan=True):
            print(f"  ✓ キャッシュ機能正常動作")
        else:
            print(f"  ✗ キャッシュ機能エラー")
            
    except Exception as e:
        print(f"  ✗ キャッシュテストエラー: {e}")
    
    # エラーハンドリングのテスト
    print(f"\n--- エラーハンドリングテスト ---")
    
    # 不完全なDataFrame（高値・安値なし）
    try:
        incomplete_data = pd.DataFrame({'close': data['close']})  # high, lowなし
        frama = FRAMA(period=16, src_type='close', fc=1, sc=198)  # closeでもFRAMAは高値・安値が必要
        result = frama.calculate(incomplete_data)
        print(f"  ✗ high/lowなしでエラーが発生しませんでした")
    except ValueError as e:
        print(f"  ✓ FRAMAで適切にエラー（high/low必須）: {str(e)[:50]}...")
    except Exception as e:
        print(f"  ? 予期しないエラー: {e}")
    
    try:
        incomplete_data = pd.DataFrame({'close': data['close']})  # high, lowなし
        frama = FRAMA(period=16, src_type='hl2', fc=1, sc=198)  # hl2は高値・安値が必要
        result = frama.calculate(incomplete_data)
        print(f"  ✗ hl2ソースでエラーが発生しませんでした")
    except ValueError as e:
        print(f"  ✓ PriceSourceで適切にエラー（hl2）: {str(e)[:30]}...")
    except Exception as e:
        print(f"  ? 予期しないエラー: {e}")
    
    # 無効なソースタイプ
    try:
        frama = FRAMA(period=16, src_type='invalid', fc=1, sc=198)
        print(f"  ✗ 無効ソースタイプでエラーが発生しませんでした")
    except ValueError as e:
        print(f"  ✓ 無効ソースタイプで適切にエラー: {e}")
    except Exception as e:
        print(f"  ? 予期しないエラー: {e}")
    
    print("\n=== PriceSource Implementation Test 完了 ===")
    
    # 比較テスト結果
    if len(results) > 1:
        print(f"\n=== ソースタイプ比較 ===")
        for src_type, result in results.items():
            valid_values = result.values[~np.isnan(result.values)]
            if len(valid_values) > 0:
                print(f"{src_type:6}: 平均={np.mean(valid_values):7.4f}, 標準偏差={np.std(valid_values):6.4f}")

if __name__ == "__main__":
    test_frama_pricesource()