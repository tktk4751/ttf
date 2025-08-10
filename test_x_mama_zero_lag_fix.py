#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X_MAMAのゼロラグ処理修正版のテストスクリプト
"""

import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from indicators.x_mama import X_MAMA
    from indicators.price_source import PriceSource
    import yaml
    from data.data_loader import DataLoader, CSVDataSource
    from data.data_processor import DataProcessor
    from data.binance_data_source import BinanceDataSource
    print("必要なモジュールのインポート完了")
except ImportError as e:
    print(f"インポートエラー: {e}")
    sys.exit(1)

def test_zero_lag_processing():
    """X_MAMAのゼロラグ処理をテストする"""
    print("\n=== X_MAMAゼロラグ処理修正版テスト ===")
    
    # 設定ファイルからデータを読み込む
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print(f"設定ファイルが見つかりません: {config_path}")
        return
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # データの準備
    binance_config = config.get('binance_data', {})
    data_dir = binance_config.get('data_dir', 'data/binance')
    binance_data_source = BinanceDataSource(data_dir)
    
    # CSVデータソースはダミーとして渡す
    dummy_csv_source = CSVDataSource("dummy")
    data_loader = DataLoader(
        data_source=dummy_csv_source,
        binance_data_source=binance_data_source
    )
    data_processor = DataProcessor()
    
    # データの読み込みと処理
    print("データを読み込み中...")
    try:
        raw_data = data_loader.load_data_from_config(config)
        processed_data = {
            symbol: data_processor.process(df)
            for symbol, df in raw_data.items()
        }
        
        # 最初のシンボルのデータを取得
        first_symbol = next(iter(processed_data))
        data = processed_data[first_symbol]
        
        print(f"データ読み込み完了: {first_symbol}")
        print(f"期間: {data.index.min()} → {data.index.max()}")
        print(f"データ数: {len(data)}")
        
    except Exception as e:
        print(f"データ読み込みエラー: {e}")
        return
    
    # 通常のX_MAMA（ゼロラグ無し）をテスト
    print("\n1. 通常のX_MAMA（ゼロラグ無し）")
    x_mama_normal = X_MAMA(
        fast_limit=0.5,
        slow_limit=0.05,
        src_type='hlc3',
        use_kalman_filter=False,
        use_zero_lag=False
    )
    
    result_normal = x_mama_normal.calculate(data)
    mama_normal = result_normal.mama_values
    fama_normal = result_normal.fama_values
    
    print(f"通常のMAMA - 有効値: {len(mama_normal) - np.isnan(mama_normal).sum()}/{len(mama_normal)}")
    print(f"通常のFAMA - 有効値: {len(fama_normal) - np.isnan(fama_normal).sum()}/{len(fama_normal)}")
    
    # ゼロラグ処理X_MAMAをテスト
    print("\n2. ゼロラグ処理X_MAMA")
    x_mama_zero_lag = X_MAMA(
        fast_limit=0.5,
        slow_limit=0.05,
        src_type='hlc3',
        use_kalman_filter=False,
        use_zero_lag=True
    )
    
    result_zero_lag = x_mama_zero_lag.calculate(data)
    mama_zero_lag = result_zero_lag.mama_values
    fama_zero_lag = result_zero_lag.fama_values
    alpha_values = result_zero_lag.alpha_values
    
    print(f"ゼロラグMAMA - 有効値: {len(mama_zero_lag) - np.isnan(mama_zero_lag).sum()}/{len(mama_zero_lag)}")
    print(f"ゼロラグFAMA - 有効値: {len(fama_zero_lag) - np.isnan(fama_zero_lag).sum()}/{len(fama_zero_lag)}")
    print(f"アルファ値 - 有効値: {len(alpha_values) - np.isnan(alpha_values).sum()}/{len(alpha_values)}")
    
    # 統計的比較
    print("\n3. 統計的比較")
    valid_indices = ~(np.isnan(mama_normal) | np.isnan(mama_zero_lag))
    
    if valid_indices.sum() > 0:
        mama_normal_valid = mama_normal[valid_indices]
        mama_zero_lag_valid = mama_zero_lag[valid_indices]
        
        # 相関係数の計算
        correlation = np.corrcoef(mama_normal_valid, mama_zero_lag_valid)[0, 1]
        
        # 統計情報
        print(f"有効な比較点数: {valid_indices.sum()}")
        print(f"相関係数: {correlation:.6f}")
        print(f"通常MAMA - 平均: {mama_normal_valid.mean():.6f}, 標準偏差: {mama_normal_valid.std():.6f}")
        print(f"ゼロラグMAMA - 平均: {mama_zero_lag_valid.mean():.6f}, 標準偏差: {mama_zero_lag_valid.std():.6f}")
        
        # アルファ値の統計
        alpha_valid = alpha_values[~np.isnan(alpha_values)]
        if len(alpha_valid) > 0:
            print(f"アルファ値 - 平均: {alpha_valid.mean():.6f}, 範囲: {alpha_valid.min():.6f} - {alpha_valid.max():.6f}")
        
        # 差分分析
        diff = mama_zero_lag_valid - mama_normal_valid
        print(f"差分 - 平均: {diff.mean():.6f}, 標準偏差: {diff.std():.6f}")
        print(f"差分 - 最大: {diff.max():.6f}, 最小: {diff.min():.6f}")
        
        # 最後の10個の値を比較
        print(f"\n最後の10個の値の比較:")
        for i in range(max(0, len(mama_normal_valid) - 10), len(mama_normal_valid)):
            idx = i
            print(f"  [{idx}] 通常: {mama_normal_valid[i]:.6f}, ゼロラグ: {mama_zero_lag_valid[i]:.6f}, 差分: {mama_zero_lag_valid[i] - mama_normal_valid[i]:.6f}")
    else:
        print("有効な比較データがありません")
    
    # カルマンフィルターとゼロラグ処理の組み合わせテスト
    print("\n4. カルマンフィルター + ゼロラグ処理")
    try:
        x_mama_kalman_zero_lag = X_MAMA(
            fast_limit=0.5,
            slow_limit=0.05,
            src_type='hlc3',
            use_kalman_filter=True,
            kalman_filter_type='unscented',
            kalman_process_noise=0.01,
            kalman_observation_noise=0.001,
            use_zero_lag=True
        )
        
        result_kalman_zero_lag = x_mama_kalman_zero_lag.calculate(data)
        mama_kalman_zero_lag = result_kalman_zero_lag.mama_values
        fama_kalman_zero_lag = result_kalman_zero_lag.fama_values
        
        print(f"カルマン+ゼロラグMAMA - 有効値: {len(mama_kalman_zero_lag) - np.isnan(mama_kalman_zero_lag).sum()}/{len(mama_kalman_zero_lag)}")
        print(f"カルマン+ゼロラグFAMA - 有効値: {len(fama_kalman_zero_lag) - np.isnan(fama_kalman_zero_lag).sum()}/{len(fama_kalman_zero_lag)}")
        
        # 統計情報
        mama_kalman_valid = mama_kalman_zero_lag[~np.isnan(mama_kalman_zero_lag)]
        if len(mama_kalman_valid) > 0:
            print(f"カルマン+ゼロラグMAMA - 平均: {mama_kalman_valid.mean():.6f}, 標準偏差: {mama_kalman_valid.std():.6f}")
    except Exception as e:
        print(f"カルマンフィルター統合テストエラー: {e}")
    
    print("\n=== テスト完了 ===")

if __name__ == "__main__":
    test_zero_lag_processing()