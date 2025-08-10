#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hyper_ER計算ロジックのデバッグスクリプト
"""

import numpy as np
import pandas as pd
import sys
import os

# インジケーターをインポートするためのパス設定
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from indicators.trend_filter.hyper_er import HyperER, calculate_efficiency_ratio_numba, calculate_hyper_er_numba
from indicators.price_source import PriceSource
from indicators.smoother.roofing_filter import RoofingFilter
import yaml


def load_real_data():
    """実際のデータを読み込む"""
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        from data.data_loader import DataLoader, CSVDataSource
        from data.data_processor import DataProcessor
        from data.binance_data_source import BinanceDataSource
        
        # データの準備
        binance_config = config.get('binance_data', {})
        data_dir = binance_config.get('data_dir', 'data/binance')
        binance_data_source = BinanceDataSource(data_dir)
        
        dummy_csv_source = CSVDataSource("dummy")
        data_loader = DataLoader(
            data_source=dummy_csv_source,
            binance_data_source=binance_data_source
        )
        data_processor = DataProcessor()
        
        # データの読み込みと処理
        raw_data = data_loader.load_data_from_config(config)
        processed_data = {
            symbol: data_processor.process(df)
            for symbol, df in raw_data.items()
        }
        
        # 最初のシンボルのデータを取得
        first_symbol = next(iter(processed_data))
        data = processed_data[first_symbol]
        
        print(f"実データ読み込み完了: {first_symbol}")
        print(f"期間: {data.index.min()} → {data.index.max()}")
        print(f"データ数: {len(data)}")
        
        return data
        
    except Exception as e:
        print(f"実データ読み込みエラー: {e}")
        return None


def debug_step_by_step_calculation():
    """ステップバイステップでHyper_ER計算をデバッグ"""
    print("=== ステップバイステップ計算デバッグ ===")
    
    # 実データを使用
    data = load_real_data()
    if data is None:
        print("データ読み込みに失敗しました")
        return False
    
    # データを小さくして処理
    data = data.tail(200)  # 最新200ポイントのみ使用
    print(f"デバッグ用データサイズ: {len(data)}")
    
    try:
        # 1. ソース価格の取得
        print("\n1. ソース価格の取得")
        source_prices = PriceSource.calculate_source(data, 'hlc3')
        source_prices = np.array(source_prices, dtype=np.float64)
        print(f"   ソース価格: {len(source_prices)}ポイント")
        print(f"   範囲: {np.min(source_prices):.6f} - {np.max(source_prices):.6f}")
        print(f"   NaN数: {np.sum(np.isnan(source_prices))}")
        
        # 2. ルーフィングフィルター適用
        print("\n2. ルーフィングフィルター適用")
        roofing = RoofingFilter(src_type='hlc3', hp_cutoff=48.0, ss_band_edge=10.0)
        roofing_result = roofing.calculate(data)
        roofing_values = roofing_result.values
        print(f"   ルーフィング値: {len(roofing_values)}ポイント")
        print(f"   有効値数: {np.sum(~np.isnan(roofing_values))}")
        
        if np.sum(~np.isnan(roofing_values)) > 0:
            roofing_valid = roofing_values[~np.isnan(roofing_values)]
            print(f"   範囲: {np.min(roofing_valid):.6f} - {np.max(roofing_valid):.6f}")
        
        # 3. フィルタリング済み価格の計算
        print("\n3. フィルタリング済み価格の計算")
        filtered_prices = source_prices.copy()
        valid_roofing = np.sum(~np.isnan(roofing_values))
        
        if valid_roofing > len(roofing_values) * 0.5:
            roofing_range = np.nanmax(roofing_values) - np.nanmin(roofing_values)
            price_range = np.nanmax(source_prices) - np.nanmin(source_prices)
            print(f"   ルーフィング範囲: {roofing_range:.6f}")
            print(f"   価格範囲: {price_range:.6f}")
            
            if roofing_range > 0 and price_range > 0:
                scale_factor = price_range / roofing_range * 0.1
                filtered_prices = source_prices + roofing_values * scale_factor
                print(f"   スケール係数: {scale_factor:.6f}")
                print(f"   フィルタリング後範囲: {np.min(filtered_prices):.6f} - {np.max(filtered_prices):.6f}")
            else:
                print("   スケール計算不可、元の価格を使用")
        else:
            print("   ルーフィング値が少ない、元の価格を使用")
        
        print(f"   フィルタリング済み価格NaN数: {np.sum(np.isnan(filtered_prices))}")
        
        # 4. Efficiency Ratioの計算
        print("\n4. Efficiency Ratioの計算")
        er_period = 13
        raw_er = calculate_efficiency_ratio_numba(filtered_prices, er_period)
        print(f"   生ER: {len(raw_er)}ポイント")
        print(f"   有効値数: {np.sum(~np.isnan(raw_er))}")
        
        if np.sum(~np.isnan(raw_er)) > 0:
            er_valid = raw_er[~np.isnan(raw_er)]
            print(f"   範囲: {np.min(er_valid):.6f} - {np.max(er_valid):.6f}")
            print(f"   平均: {np.mean(er_valid):.6f}")
        
        # 5. Hyper_ERの計算
        print("\n5. Hyper_ERの計算")
        period = 14
        hyper_er_values = calculate_hyper_er_numba(raw_er, period, None)
        print(f"   Hyper_ER: {len(hyper_er_values)}ポイント")
        print(f"   有効値数: {np.sum(~np.isnan(hyper_er_values))}")
        
        if np.sum(~np.isnan(hyper_er_values)) > 0:
            hyper_valid = hyper_er_values[~np.isnan(hyper_er_values)]
            print(f"   範囲: {np.min(hyper_valid):.6f} - {np.max(hyper_valid):.6f}")
            print(f"   平均: {np.mean(hyper_valid):.6f}")
        else:
            print("   ⚠️ Hyper_ER値がすべてNaN!")
            
            # より詳細なデバッグ
            print("\n   詳細デバッグ:")
            print(f"   raw_er最初の20値: {raw_er[:20]}")
            print(f"   period: {period}")
            
            # 手動でHyper_ER計算をテスト
            manual_hyper_er = []
            for i in range(period - 1, len(raw_er)):
                period_data = raw_er[i - period + 1:i + 1]
                valid_data = period_data[~np.isnan(period_data)]
                if len(valid_data) >= period // 2:
                    avg_er = np.mean(valid_data)
                    manual_hyper_er.append(avg_er)
                    if len(manual_hyper_er) <= 5:  # 最初の5値のみ表示
                        print(f"   手動計算[{i}]: period_data={period_data}, valid_count={len(valid_data)}, avg={avg_er:.6f}")
                else:
                    manual_hyper_er.append(np.nan)
            
            manual_hyper_er = np.array(manual_hyper_er)
            print(f"   手動計算有効値数: {np.sum(~np.isnan(manual_hyper_er))}")
        
        # 6. 平滑化テスト（オプション）
        print("\n6. 平滑化テスト")
        if np.sum(~np.isnan(hyper_er_values)) > 0:
            try:
                from indicators.smoother.unified_smoother import UnifiedSmoother
                
                smoother = UnifiedSmoother(
                    smoother_type='super_smoother',
                    src_type='close',
                    period=8
                )
                
                # Hyper_ER値をDataFrame形式に変換
                er_df = pd.DataFrame({'close': hyper_er_values})
                smoother_result = smoother.calculate(er_df)
                smoothed_er = smoother_result.values
                
                print(f"   平滑化結果: {len(smoothed_er)}ポイント")
                print(f"   有効値数: {np.sum(~np.isnan(smoothed_er))}")
                
                if np.sum(~np.isnan(smoothed_er)) > 0:
                    smooth_valid = smoothed_er[~np.isnan(smoothed_er)]
                    print(f"   範囲: {np.min(smooth_valid):.6f} - {np.max(smooth_valid):.6f}")
                
            except Exception as e:
                print(f"   平滑化エラー: {e}")
        
        return np.sum(~np.isnan(hyper_er_values)) > 0
        
    except Exception as e:
        print(f"計算エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_calculate_hyper_er_numba():
    """calculate_hyper_er_numba関数の直接テスト"""
    print("\n=== calculate_hyper_er_numba関数テスト ===")
    
    # テストデータを作成
    test_er = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] * 10, dtype=np.float64)
    period = 5
    
    print(f"テストER値: {len(test_er)}ポイント")
    print(f"期間: {period}")
    print(f"最初の10値: {test_er[:10]}")
    
    try:
        result = calculate_hyper_er_numba(test_er, period, None)
        print(f"結果: {len(result)}ポイント")
        print(f"有効値数: {np.sum(~np.isnan(result))}")
        
        if np.sum(~np.isnan(result)) > 0:
            valid_result = result[~np.isnan(result)]
            print(f"範囲: {np.min(valid_result):.6f} - {np.max(valid_result):.6f}")
            print(f"最初の10値: {result[:10]}")
        else:
            print("⚠️ 結果がすべてNaN")
            
        return np.sum(~np.isnan(result)) > 0
        
    except Exception as e:
        print(f"関数テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """メイン実行関数"""
    print("Hyper_ER計算ロジックデバッグ")
    print("=" * 50)
    
    # 関数レベルのテスト
    function_test = test_calculate_hyper_er_numba()
    
    # 実データでのステップバイステップテスト
    step_test = debug_step_by_step_calculation()
    
    print("\n" + "=" * 50)
    print("デバッグ結果:")
    print(f"関数テスト: {'PASS' if function_test else 'FAIL'}")
    print(f"ステップテスト: {'PASS' if step_test else 'FAIL'}")


if __name__ == "__main__":
    main()