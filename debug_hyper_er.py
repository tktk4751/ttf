#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hyper_ERのデバッグスクリプト
"""

import numpy as np
import pandas as pd
import sys
import os

# インジケーターをインポートするためのパス設定
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from indicators.trend_filter.hyper_er import HyperER
from indicators.smoother.roofing_filter import RoofingFilter


def create_simple_test_data(length: int = 100) -> pd.DataFrame:
    """シンプルなテスト用の価格データを生成"""
    np.random.seed(42)
    base_price = 100.0
    
    # シンプルなトレンドデータ
    prices = []
    for i in range(length):
        price = base_price + i * 0.1 + np.random.normal(0, 0.5)
        prices.append(price)
    
    # OHLC データの生成
    data = []
    for i, close in enumerate(prices):
        high = close + np.random.uniform(0.1, 0.5)
        low = close - np.random.uniform(0.1, 0.5)
        open_price = close + np.random.uniform(-0.2, 0.2)
        
        # 論理的整合性の確保
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': 1000.0
        })
    
    return pd.DataFrame(data)


def debug_roofing_filter():
    """ルーフィングフィルターのデバッグ"""
    print("=== ルーフィングフィルターデバッグ ===")
    
    df = create_simple_test_data(100)
    print(f"テストデータ: {len(df)}ポイント")
    print(f"価格範囲: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    try:
        roofing = RoofingFilter(
            src_type='close',
            hp_cutoff=48.0,
            ss_band_edge=10.0
        )
        
        result = roofing.calculate(df)
        print(f"ルーフィングフィルター結果: {len(result.values)}ポイント")
        
        valid_values = result.values[~np.isnan(result.values)]
        print(f"有効値数: {len(valid_values)}")
        
        if len(valid_values) > 0:
            print(f"値域: {np.min(valid_values):.6f} - {np.max(valid_values):.6f}")
            print(f"平均: {np.mean(valid_values):.6f}")
            print(f"標準偏差: {np.std(valid_values):.6f}")
        else:
            print("有効値がありません")
        
        return True
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def debug_hyper_er_basic():
    """基本Hyper_ERのデバッグ"""
    print("\\n=== 基本Hyper_ERデバッグ ===")
    
    df = create_simple_test_data(100)
    
    try:
        # ルーフィングフィルターなし
        hyper_er_basic = HyperER(
            period=14,
            er_period=13,
            use_roofing_filter=False,
            use_dynamic_period=False,
            use_smoothing=False
        )
        
        result = hyper_er_basic.calculate(df)
        print(f"基本Hyper_ER結果: {len(result.values)}ポイント")
        
        valid_values = result.values[~np.isnan(result.values)]
        print(f"有効値数: {len(valid_values)}")
        
        if len(valid_values) > 0:
            print(f"値域: {np.min(valid_values):.4f} - {np.max(valid_values):.4f}")
            print(f"平均: {np.mean(valid_values):.4f}")
        
        return True
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def debug_hyper_er_roofing():
    """ルーフィングフィルター付きHyper_ERのデバッグ"""
    print("\\n=== ルーフィングフィルター付きHyper_ERデバッグ ===")
    
    df = create_simple_test_data(100)
    
    try:
        # ルーフィングフィルター付き
        hyper_er_roofing = HyperER(
            period=14,
            er_period=13,
            use_roofing_filter=True,
            roofing_hp_cutoff=48.0,
            roofing_ss_band_edge=10.0,
            use_dynamic_period=False,
            use_smoothing=False
        )
        
        result = hyper_er_roofing.calculate(df)
        print(f"ルーフィング付きHyper_ER結果: {len(result.values)}ポイント")
        
        valid_values = result.values[~np.isnan(result.values)]
        print(f"有効値数: {len(valid_values)}")
        
        if len(valid_values) > 0:
            print(f"値域: {np.min(valid_values):.4f} - {np.max(valid_values):.4f}")
            print(f"平均: {np.mean(valid_values):.4f}")
        else:
            print("有効値がありません - 詳細を調査")
            
            # 詳細調査
            print(f"\\n詳細調査:")
            print(f"raw_er有効値数: {np.sum(~np.isnan(result.raw_er))}")
            print(f"filtered_er有効値数: {np.sum(~np.isnan(result.filtered_er))}")
            print(f"roofing_values有効値数: {np.sum(~np.isnan(result.roofing_values))}")
            
            if np.sum(~np.isnan(result.roofing_values)) > 0:
                roofing_valid = result.roofing_values[~np.isnan(result.roofing_values)]
                print(f"roofing値域: {np.min(roofing_valid):.6f} - {np.max(roofing_valid):.6f}")
        
        return len(valid_values) > 0
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def debug_step_by_step():
    """ステップバイステップデバッグ"""
    print("\\n=== ステップバイステップデバッグ ===")
    
    df = create_simple_test_data(100)
    print(f"入力データ: {len(df)}ポイント")
    
    try:
        from indicators.price_source import PriceSource
        from indicators.trend_filter.hyper_er import calculate_efficiency_ratio_numba, calculate_hyper_er_numba
        
        # 1. ソース価格
        source_prices = PriceSource.calculate_source(df, 'hlc3')
        source_prices = np.array(source_prices, dtype=np.float64)
        print(f"1. ソース価格: {len(source_prices)}ポイント, 範囲: {np.min(source_prices):.2f} - {np.max(source_prices):.2f}")
        
        # 2. ルーフィングフィルター（オプション）
        roofing = RoofingFilter(src_type='hlc3', hp_cutoff=48.0, ss_band_edge=10.0)
        roofing_result = roofing.calculate(df)
        roofing_values = roofing_result.values
        print(f"2. ルーフィング値: {np.sum(~np.isnan(roofing_values))}有効値")
        
        if np.sum(~np.isnan(roofing_values)) > 0:
            roofing_valid = roofing_values[~np.isnan(roofing_values)]
            print(f"   ルーフィング範囲: {np.min(roofing_valid):.6f} - {np.max(roofing_valid):.6f}")
        
        # 3. フィルタリング済み価格
        filtered_prices = source_prices.copy()
        valid_roofing = np.sum(~np.isnan(roofing_values))
        if valid_roofing > len(roofing_values) * 0.5:
            roofing_range = np.nanmax(roofing_values) - np.nanmin(roofing_values)
            price_range = np.nanmax(source_prices) - np.nanmin(source_prices)
            if roofing_range > 0 and price_range > 0:
                scale_factor = price_range / roofing_range * 0.1
                filtered_prices = source_prices + roofing_values * scale_factor
                print(f"3. フィルタリング済み価格: スケール係数={scale_factor:.6f}")
            else:
                print(f"3. フィルタリング済み価格: スケール不可、元の価格を使用")
        else:
            print(f"3. フィルタリング済み価格: ルーフィング無効値が多い、元の価格を使用")
        
        print(f"   フィルタリング後範囲: {np.min(filtered_prices):.2f} - {np.max(filtered_prices):.2f}")
        
        # 4. ER計算
        raw_er = calculate_efficiency_ratio_numba(filtered_prices, 13)
        er_valid = np.sum(~np.isnan(raw_er))
        print(f"4. 生ER: {er_valid}有効値")
        
        if er_valid > 0:
            er_values = raw_er[~np.isnan(raw_er)]
            print(f"   ER範囲: {np.min(er_values):.4f} - {np.max(er_values):.4f}")
        
        # 5. Hyper_ER計算
        hyper_er_values = calculate_hyper_er_numba(raw_er, 14, None)
        hyper_valid = np.sum(~np.isnan(hyper_er_values))
        print(f"5. Hyper_ER: {hyper_valid}有効値")
        
        if hyper_valid > 0:
            hyper_values = hyper_er_values[~np.isnan(hyper_er_values)]
            print(f"   Hyper_ER範囲: {np.min(hyper_values):.4f} - {np.max(hyper_values):.4f}")
        
        return hyper_valid > 0
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """メイン実行関数"""
    print("Hyper_ER デバッグスクリプト")
    print("=" * 40)
    
    debug_results = []
    
    # 各デバッグを実行
    debug_results.append(("ルーフィングフィルター", debug_roofing_filter()))
    debug_results.append(("基本Hyper_ER", debug_hyper_er_basic()))
    debug_results.append(("ルーフィング付きHyper_ER", debug_hyper_er_roofing()))
    debug_results.append(("ステップバイステップ", debug_step_by_step()))
    
    # 結果サマリー
    print("\\n" + "=" * 40)
    print("デバッグ結果サマリー:")
    print("=" * 40)
    
    passed = 0
    total = len(debug_results)
    
    for test_name, result in debug_results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\\n合計: {passed}/{total} デバッグパス ({passed/total*100:.1f}%)")


if __name__ == "__main__":
    main()