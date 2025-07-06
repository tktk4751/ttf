#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 **Zero Lag EMA with MA-UKF テスト実行スクリプト** 🎯

Zero Lag EMA with Market-Adaptive UKFインジケーターの
包括的なテストとデモンストレーション

実行方法:
1. 基本テスト（ダミーデータ使用）:
   python examples/zero_lag_ema_ma_ukf_test.py

2. 設定ファイルからデータ読み込み:
   python examples/zero_lag_ema_ma_ukf_test.py --config config.yaml

3. パラメータ調整テスト:
   python examples/zero_lag_ema_ma_ukf_test.py --ema-period 21 --lag-adjustment 1.5

4. チャート保存:
   python examples/zero_lag_ema_ma_ukf_test.py --output output/zero_lag_ema_test.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List
from pathlib import Path

# パスの追加（インポートエラー対策）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from indicators.zero_lag_ema_ma_ukf import ZeroLagEMAWithMAUKF, calculate_zero_lag_ema_numba
    from indicators.price_source import PriceSource
    from visualization.zero_lag_ema_ma_ukf_chart import ZeroLagEMAMAUKFChart
except ImportError as e:
    print(f"インポートエラー: {e}")
    print("プロジェクトルートから実行してください")
    sys.exit(1)


def create_sample_config():
    """サンプル設定ファイルを作成"""
    sample_config = """
# Zero Lag EMA with MA-UKF テスト用設定ファイル

binance_data:
  data_dir: "data/binance"
  symbols:
    - "BTCUSDT"
    - "ETHUSDT"
  intervals:
    - "1h"
    - "4h"
  start_date: "2023-01-01"
  end_date: "2023-12-31"
  limit: 1000

# 使用しない場合はコメントアウト
csv_data:
  files:
    - "data/sample_btc.csv"
    - "data/sample_eth.csv"
"""
    
    with open("sample_config.yaml", "w", encoding="utf-8") as f:
        f.write(sample_config)
    
    print("サンプル設定ファイル 'sample_config.yaml' を作成しました")


def basic_test():
    """基本テスト - ダミーデータでの動作確認"""
    print("🔬 基本テスト開始（ダミーデータ使用）")
    print("=" * 50)
    
    # ダミーデータ生成
    print("📊 ダミーデータを生成中...")
    chart = ZeroLagEMAMAUKFChart()
    data = chart.generate_dummy_data(n_periods=300)
    chart.data = data
    print(f"ダミーデータ生成完了")
    print(f"期間: {data.index[0]} → {data.index[-1]}")
    print(f"価格範囲: {data['close'].min():.2f} - {data['close'].max():.2f}")
    print()
    
    # ゼロラグEMA計算
    print("📈 Zero Lag EMA with MA-UKFを計算中...")
    print()
    
    indicator = ZeroLagEMAWithMAUKF(
        ema_period=14,
        lag_adjustment=1.0,  # より安全な値
        slope_period=1,
        range_threshold=0.003
    )
    
    try:
        # まず、HLC3価格をチェック
        hlc3 = PriceSource.calculate_source(data, 'hlc3')
        print(f"HLC3価格統計:")
        print(f"  範囲: {hlc3.min():.2f} - {hlc3.max():.2f}")
        print(f"  平均: {hlc3.mean():.2f}")
        print(f"  標準偏差: {hlc3.std():.2f}")
        print()
        
        # MA-UKFフィルターの結果をチェック
        print("MA-UKFフィルタリング中...")
        ukf_result = indicator.ma_ukf.calculate(data)
        filtered_hlc3 = ukf_result.filtered_values
        
        print(f"フィルタリング済みHLC3統計:")
        print(f"  範囲: {filtered_hlc3.min():.2f} - {filtered_hlc3.max():.2f}")
        print(f"  平均: {filtered_hlc3.mean():.2f}")
        print(f"  標準偏差: {filtered_hlc3.std():.2f}")
        print(f"  有限値の数: {np.isfinite(filtered_hlc3).sum()}/{len(filtered_hlc3)}")
        print(f"  NaN値の数: {np.isnan(filtered_hlc3).sum()}")
        print(f"  無限値の数: {np.isinf(filtered_hlc3).sum()}")
        
        # 異常値をチェック
        abs_values = np.abs(filtered_hlc3[np.isfinite(filtered_hlc3)])
        if len(abs_values) > 0:
            max_abs = abs_values.max()
            print(f"  最大絶対値: {max_abs:.2e}")
            if max_abs > 1e6:
                print("  ⚠️ 警告: フィルタリング済み値に異常に大きな値が含まれています")
        
        print()
        
        # 通常のEMAとゼロラグEMAを直接計算
        print("ゼロラグEMA計算中...")
        zero_lag_values, ema_values = calculate_zero_lag_ema_numba(
            filtered_hlc3, 14, 1.0
        )
        
        print(f"ゼロラグEMA統計:")
        finite_zl = zero_lag_values[np.isfinite(zero_lag_values)]
        if len(finite_zl) > 0:
            print(f"  有限値の範囲: {finite_zl.min():.2f} - {finite_zl.max():.2f}")
            print(f"  有限値の平均: {finite_zl.mean():.2f}")
            print(f"  有限値の標準偏差: {finite_zl.std():.2f}")
        print(f"  有限値の数: {len(finite_zl)}/{len(zero_lag_values)}")
        print(f"  NaN値の数: {np.isnan(zero_lag_values).sum()}")
        print(f"  無限値の数: {np.isinf(zero_lag_values).sum()}")
        
        # EMA統計
        print(f"通常EMA統計:")
        finite_ema = ema_values[np.isfinite(ema_values)]
        if len(finite_ema) > 0:
            print(f"  有限値の範囲: {finite_ema.min():.2f} - {finite_ema.max():.2f}")
            print(f"  有限値の平均: {finite_ema.mean():.2f}")
            print(f"  有限値の標準偏差: {finite_ema.std():.2f}")
        print(f"  有限値の数: {len(finite_ema)}/{len(ema_values)}")
        print()
        
        # 最初の10個の値をチェック
        print("最初の10個の値:")
        for i in range(min(10, len(zero_lag_values))):
            print(f"  {i}: 価格={filtered_hlc3[i]:.2f}, EMA={ema_values[i]:.2f}, ZeroLag={zero_lag_values[i]:.2f}")
        print()
        
        # 完全な計算を実行
        result = indicator.calculate(data)
        
        print("✅ 計算完了!")
        
    except Exception as e:
        print(f"❌ 計算エラー: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 統計情報の表示
    print("計算結果の統計:")
    print(f"  Zero Lag EMA 有効値: {np.isfinite(result.values).sum()}/{len(result.values)}")
    if np.isfinite(result.values).any():
        finite_values = result.values[np.isfinite(result.values)]
        print(f"  範囲: {finite_values.min():.2f} - {finite_values.max():.2f}")
        print(f"  平均: {finite_values.mean():.2f}")
        print(f"  標準偏差: {finite_values.std():.2f}")
    print()
    
    # チャート描画
    print("📊 チャートを描画中...")
    try:
        chart.result = result
        chart.plot(
            title="Zero Lag EMA with MA-UKF - Debug Test",
            show_volume=True,
            figsize=(16, 14)
        )
        print("✅ チャート描画完了")
    except Exception as e:
        print(f"⚠️ チャート描画エラー: {e}")
    
    print("✅ 基本テスト完了!")
    print()


def run_parameter_comparison_test():
    """パラメータ比較テスト"""
    print("\n🔍 パラメータ比較テスト開始")
    print("=" * 50)
    
    try:
        # テストパラメータセット
        parameter_sets = [
                         {"ema_period": 10, "lag_adjustment": 1.2, "name": "Short Fast"},
             {"ema_period": 14, "lag_adjustment": 1.0, "name": "Standard"},
             {"ema_period": 21, "lag_adjustment": 0.8, "name": "Long Smooth"},
        ]
        
        # ダミーデータ生成
        chart = ZeroLagEMAMAUKFChart()
        dummy_data = chart.generate_dummy_data(n_periods=200)
        
        # 比較結果の収集
        results = []
        
        for params in parameter_sets:
            print(f"\n📊 {params['name']} パラメータをテスト中...")
            print(f"   EMA期間: {params['ema_period']}, 遅延調整: {params['lag_adjustment']}")
            
            # 新しいインジケーターインスタンス
            zero_lag_ema = ZeroLagEMAWithMAUKF(
                ema_period=params['ema_period'],
                lag_adjustment=params['lag_adjustment'],
                slope_period=1,
                range_threshold=0.003
            )
            
            # 計算実行
            result = zero_lag_ema.calculate(dummy_data)
            
            # 結果の分析
            valid_values = result.values[~np.isnan(result.values)]
            if len(valid_values) > 1:
                responsiveness = np.std(np.diff(valid_values))
                trend_changes = np.sum(np.diff(result.trend_signals) != 0)
                avg_confidence = np.nanmean(result.confidence_scores) if result.confidence_scores is not None else 0
                
                results.append({
                    'name': params['name'],
                    'responsiveness': responsiveness,
                    'trend_changes': trend_changes,
                    'avg_confidence': avg_confidence,
                    'valid_points': len(valid_values)
                })
        
        # 結果の表示
        print(f"\n📊 パラメータ比較結果:")
        print("-" * 70)
        print(f"{'パラメータ':<15} {'応答性':<10} {'トレンド変化':<12} {'平均信頼度':<12} {'有効点数':<10}")
        print("-" * 70)
        
        for result in results:
            print(f"{result['name']:<15} {result['responsiveness']:<10.4f} "
                  f"{result['trend_changes']:<12} {result['avg_confidence']:<12.3f} "
                  f"{result['valid_points']:<10}")
        
        print("\n✅ パラメータ比較テスト完了!")
        
    except Exception as e:
        print(f"\n❌ パラメータ比較テストでエラーが発生: {e}")
        import traceback
        traceback.print_exc()


def run_performance_test():
    """パフォーマンステスト"""
    print("\n⚡ パフォーマンステスト開始")
    print("=" * 50)
    
    import time
    
    try:
        # 異なるデータサイズでテスト
        data_sizes = [100, 500, 1000, 2000]
        
        print(f"{'データサイズ':<12} {'計算時間(秒)':<15} {'1点あたり(ms)':<15}")
        print("-" * 42)
        
        for size in data_sizes:
            # ダミーデータ生成
            chart = ZeroLagEMAMAUKFChart()
            dummy_data = chart.generate_dummy_data(n_periods=size)
            
            # 時間測定
            start_time = time.time()
            chart.data = dummy_data
            chart.calculate_indicators()
            end_time = time.time()
            
            calculation_time = end_time - start_time
            time_per_point = (calculation_time / size) * 1000  # ms
            
            print(f"{size:<12} {calculation_time:<15.4f} {time_per_point:<15.4f}")
        
        print("\n✅ パフォーマンステスト完了!")
        
    except Exception as e:
        print(f"\n❌ パフォーマンステストでエラーが発生: {e}")
        import traceback
        traceback.print_exc()


def run_robustness_test():
    """堅牢性テスト（異常データに対する耐性）"""
    print("\n🛡️ 堅牢性テスト開始")
    print("=" * 50)
    
    try:
        # 通常データ
        chart = ZeroLagEMAMAUKFChart()
        normal_data = chart.generate_dummy_data(n_periods=100)
        
        # 異常データパターンを追加
        test_cases = [
            ("通常データ", normal_data.copy()),
            ("NaN値混入", normal_data.copy()),
            ("極端な価格ジャンプ", normal_data.copy()),
            ("ゼロボラティリティ", normal_data.copy()),
            ("負の価格", normal_data.copy())
        ]
        
        # NaN値混入
        test_cases[1][1].loc[test_cases[1][1].index[20:25], 'close'] = np.nan
        
        # 極端な価格ジャンプ
        test_cases[2][1].loc[test_cases[2][1].index[50], 'close'] *= 10
        test_cases[2][1].loc[test_cases[2][1].index[51], 'close'] /= 10
        
        # ゼロボラティリティ
        test_cases[3][1].loc[test_cases[3][1].index[30:40], 'close'] = 100.0
        test_cases[3][1].loc[test_cases[3][1].index[30:40], 'high'] = 100.0
        test_cases[3][1].loc[test_cases[3][1].index[30:40], 'low'] = 100.0
        test_cases[3][1].loc[test_cases[3][1].index[30:40], 'open'] = 100.0
        
        # 負の価格（理論上起こらないが）
        test_cases[4][1].loc[test_cases[4][1].index[60:65], 'close'] = -1.0
        
        print(f"{'テストケース':<20} {'計算成功':<10} {'有効値数':<10} {'エラー内容':<30}")
        print("-" * 70)
        
        for case_name, test_data in test_cases:
            try:
                zero_lag_ema = ZeroLagEMAWithMAUKF()
                result = zero_lag_ema.calculate(test_data)
                
                valid_count = np.sum(~np.isnan(result.values))
                success = "✅"
                error_msg = "なし"
                
            except Exception as e:
                valid_count = 0
                success = "❌"
                error_msg = str(e)[:25] + "..." if len(str(e)) > 25 else str(e)
            
            print(f"{case_name:<20} {success:<10} {valid_count:<10} {error_msg:<30}")
        
        print("\n✅ 堅牢性テスト完了!")
        
    except Exception as e:
        print(f"\n❌ 堅牢性テストでエラーが発生: {e}")
        import traceback
        traceback.print_exc()


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Zero Lag EMA with MA-UKF 包括テスト')
    parser.add_argument('--config', '-c', type=str, help='設定ファイルのパス')
    parser.add_argument('--output', '-o', type=str, help='チャート出力パス')
    parser.add_argument('--ema-period', type=int, default=14, help='EMA期間')
    parser.add_argument('--lag-adjustment', type=float, default=1.0, help='遅延調整係数')
    parser.add_argument('--create-config', action='store_true', help='サンプル設定ファイルを作成')
    parser.add_argument('--test-basic', action='store_true', help='基本テストのみ実行')
    parser.add_argument('--test-params', action='store_true', help='パラメータ比較テストのみ実行')
    parser.add_argument('--test-performance', action='store_true', help='パフォーマンステストのみ実行')
    parser.add_argument('--test-robustness', action='store_true', help='堅牢性テストのみ実行')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    print("🎯 Zero Lag EMA with Market-Adaptive UKF 包括テスト")
    print("=" * 60)
    
    # サンプル設定ファイル作成
    if args.create_config:
        create_sample_config()
        return
    
    # 個別テスト実行
    if args.test_basic:
        basic_test()
        return
    elif args.test_params:
        run_parameter_comparison_test()
        return
    elif args.test_performance:
        run_performance_test()
        return
    elif args.test_robustness:
        run_robustness_test()
        return
    
    # 設定ファイルを使用したテスト
    if args.config:
        try:
            print(f"📁 設定ファイルを使用したテスト: {args.config}")
            
            chart = ZeroLagEMAMAUKFChart()
            chart.load_data_from_config(args.config)
            chart.calculate_indicators(
                ema_period=args.ema_period,
                lag_adjustment=args.lag_adjustment
            )
            chart.plot(
                start_date=args.start,
                end_date=args.end,
                savefig=args.output
            )
            
            print("✅ 設定ファイルテスト完了!")
            
        except Exception as e:
            print(f"❌ 設定ファイルテストでエラー: {e}")
            print("基本テストにフォールバックします...")
            basic_test()
    else:
        # 全テストを順次実行
        print("🔬 全テストを順次実行します...")
        
        basic_test()
        run_parameter_comparison_test() 
        run_performance_test()
        run_robustness_test()
        
        print(f"\n🎉 全テスト完了!")
        print("\n💡 次のステップ:")
        print("1. 実データでテスト: python examples/zero_lag_ema_ma_ukf_test.py --config config.yaml")
        print("2. パラメータ調整: --ema-period 21 --lag-adjustment 1.5")
        print("3. チャート保存: --output output/test_chart.png")
        print("4. サンプル設定作成: --create-config")


if __name__ == "__main__":
    main() 