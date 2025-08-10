#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
詳細なスムーサーデバッグスクリプト - 実際のチャート実行時の状況を詳しく調べる
"""

import numpy as np
import pandas as pd
import sys
import os
import yaml

# インジケーターをインポートするためのパス設定
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from indicators.trend_filter.hyper_er import HyperER
from indicators.smoother.unified_smoother import UnifiedSmoother
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource

def test_real_data_smoother_integration():
    """実際のデータでスムーサー統合をテストする"""
    print("=== 実データスムーザー統合詳細テスト ===")
    
    # 設定ファイルからデータを読み込む
    config_path = 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    # データの準備
    binance_config = config.get('binance_data', {})
    data_dir = binance_config.get('data_dir', 'data/binance')
    binance_data_source = BinanceDataSource(data_dir)
    
    # CSVデータソースはダミーとして渡す（Binanceデータソースのみを使用）
    dummy_csv_source = CSVDataSource("dummy")
    data_loader = DataLoader(
        data_source=dummy_csv_source,
        binance_data_source=binance_data_source
    )
    data_processor = DataProcessor()
    
    # データの読み込みと処理（少量のデータのみ）
    print("データを読み込み・処理中...")
    raw_data = data_loader.load_data_from_config(config)
    processed_data = {
        symbol: data_processor.process(df)
        for symbol, df in raw_data.items()
    }
    
    # 最初のシンボルのデータを取得（テスト用に100行のみ）
    first_symbol = next(iter(processed_data))
    full_data = processed_data[first_symbol]
    test_data = full_data.head(200).copy()  # 200行に制限してテスト
    
    print(f"テストデータ: {len(test_data)}行")
    print(f"データ列: {test_data.columns.tolist()}")
    print(f"価格範囲 - 高値: {test_data['high'].min():.4f} - {test_data['high'].max():.4f}")
    print(f"価格範囲 - 安値: {test_data['low'].min():.4f} - {test_data['low'].max():.4f}")
    print(f"価格範囲 - 終値: {test_data['close'].min():.4f} - {test_data['close'].max():.4f}")
    
    # HyperERを計算（まず平滑化なしで）
    print("\n--- Step 1: 平滑化なしでHyperER計算 ---")
    hyper_er_no_smooth = HyperER(
        period=14,
        er_period=13,
        use_roofing_filter=False,
        use_dynamic_period=False,
        use_smoothing=False
    )
    
    result_no_smooth = hyper_er_no_smooth.calculate(test_data)
    print(f"平滑化なし - 最終値有効数: {np.sum(~np.isnan(result_no_smooth.values))}")
    
    if np.sum(~np.isnan(result_no_smooth.values)) > 0:
        valid_values = result_no_smooth.values[~np.isnan(result_no_smooth.values)]
        print(f"平滑化なし - 値範囲: {np.min(valid_values):.4f} - {np.max(valid_values):.4f}")
        print(f"平滑化なし - 最初の20値: {result_no_smooth.values[:20]}")
    
    # 今度は平滑化ありで
    smoother_types = ['super_smoother', 'frama', 'alma']
    
    for smoother_type in smoother_types:
        print(f"\n--- Step 2: {smoother_type}で平滑化 ---")
        
        # HyperER値を手動でスムーサーに渡してテスト
        er_values = result_no_smooth.values.copy()
        valid_er = er_values[~np.isnan(er_values)]
        
        print(f"スムーサー入力 - ER値有効数: {len(valid_er)}")
        
        if len(valid_er) > 0:
            print(f"スムーサー入力 - ER値範囲: {np.min(valid_er):.4f} - {np.max(valid_er):.4f}")
            
            # DataFrame作成の詳細テスト
            print("DataFrame作成テスト:")
            
            # 方法1: closeのみのDataFrame
            er_df_simple = pd.DataFrame({'close': er_values})
            print(f"  方法1 (closeのみ): 形状={er_df_simple.shape}, NaN数={er_df_simple['close'].isna().sum()}")
            
            # 方法2: 完全なOHLCデータ
            er_df_full = test_data.copy()
            er_df_full['close'] = er_values
            if len(er_df_full) != len(er_values):
                er_df_full = er_df_full.iloc[:len(er_values)].copy()
                er_df_full['close'] = er_values
            print(f"  方法2 (完全OHLC): 形状={er_df_full.shape}, close NaN数={er_df_full['close'].isna().sum()}")
            
            # 方法3: ER値でhigh, low, openも置き換え
            er_df_all_er = pd.DataFrame({
                'open': er_values,
                'high': er_values,
                'low': er_values,
                'close': er_values,
                'volume': np.ones_like(er_values) * 1000
            })
            print(f"  方法3 (全部ER値): 形状={er_df_all_er.shape}, close NaN数={er_df_all_er['close'].isna().sum()}")
            
            # それぞれの方法でスムーサーをテスト
            methods = [
                ("方法1 (closeのみ)", er_df_simple),
                ("方法2 (完全OHLC)", er_df_full),
                ("方法3 (全部ER値)", er_df_all_er)
            ]
            
            for method_name, df in methods:
                print(f"\n    {method_name}でスムーサー試行:")
                try:
                    smoother = UnifiedSmoother(
                        smoother_type=smoother_type,
                        src_type='close',
                        period=8
                    )
                    
                    smoother_result = smoother.calculate(df)
                    
                    if smoother_result is not None and hasattr(smoother_result, 'values'):
                        smooth_values = smoother_result.values
                        valid_smooth = np.sum(~np.isnan(smooth_values))
                        print(f"      結果: 有効値数={valid_smooth}/{len(smooth_values)}")
                        
                        if valid_smooth > 0:
                            valid_smooth_values = smooth_values[~np.isnan(smooth_values)]
                            print(f"      範囲: {np.min(valid_smooth_values):.4f} - {np.max(valid_smooth_values):.4f}")
                            print(f"      最初の10値: {smooth_values[:10]}")
                        else:
                            print(f"      ❌ すべてNaN")
                    else:
                        print(f"      ❌ 無効な結果")
                        
                except Exception as e:
                    print(f"      ❌ エラー: {e}")
                    import traceback
                    traceback.print_exc()
        
        # HyperERの統合テストも実行
        print(f"\n--- Step 3: HyperER統合で{smoother_type} ---")
        try:
            hyper_er_smooth = HyperER(
                period=14,
                er_period=13,
                use_roofing_filter=False,
                use_dynamic_period=False,
                use_smoothing=True,
                smoother_type=smoother_type,
                smoother_period=8,
                smoother_src_type='close'
            )
            
            result_smooth = hyper_er_smooth.calculate(test_data)
            
            print(f"HyperER統合 - 生ER有効値: {np.sum(~np.isnan(result_smooth.raw_er))}")
            print(f"HyperER統合 - 平滑化ER有効値: {np.sum(~np.isnan(result_smooth.smoothed_er))}")
            print(f"HyperER統合 - 最終値有効値: {np.sum(~np.isnan(result_smooth.values))}")
            
        except Exception as e:
            print(f"HyperER統合でエラー: {e}")
            import traceback
            traceback.print_exc()

def main():
    """メイン実行関数"""
    print("詳細スムーサー統合デバッグ")
    print("=" * 60)
    
    test_real_data_smoother_integration()
    
    print("\n" + "=" * 60)
    print("詳細デバッグ完了")

if __name__ == "__main__":
    main()