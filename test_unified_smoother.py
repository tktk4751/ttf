#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
統合スムーサーのテストスクリプト
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def generate_test_data(length: int = 200, with_trend: bool = True, with_noise: bool = True) -> pd.DataFrame:
    """テスト用のOHLCデータを生成"""
    np.random.seed(42)
    
    base_price = 100.0
    trend = 0.001 if with_trend else 0.0
    volatility = 0.02 if with_noise else 0.005
    
    prices = [base_price]
    for i in range(1, length):
        # トレンド + ランダムウォーク
        change = trend + np.random.normal(0, volatility)
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # OHLC データの生成
    data = []
    for i, close in enumerate(prices):
        daily_range = abs(np.random.normal(0, volatility * close * 0.5))
        
        high = close + daily_range * np.random.uniform(0.3, 1.0)
        low = close - daily_range * np.random.uniform(0.3, 1.0)
        
        if i == 0:
            open_price = close
        else:
            gap = np.random.normal(0, volatility * close * 0.2)
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
    
    # タイムスタンプの追加
    start_date = datetime.now() - timedelta(days=length)
    df.index = [start_date + timedelta(hours=i) for i in range(length)]
    
    return df


def test_available_smoothers():
    """利用可能なスムーサーのテスト"""
    print("=" * 60)
    print("利用可能なスムーサーのテスト")
    print("=" * 60)
    
    try:
        from indicators.smoother.unified_smoother import UnifiedSmoother
        
        # 利用可能なスムーサーを取得
        smoothers = UnifiedSmoother.get_available_smoothers()
        print(f"利用可能なスムーサー数: {len(smoothers)}")
        print("\n利用可能なスムーサー:")
        print("-" * 60)
        
        for smoother_type, description in smoothers.items():
            print(f"{smoother_type:<20}: {description}")
        
        print("-" * 60)
        
        # デフォルトパラメータのテスト
        print("\nデフォルトパラメータの例:")
        test_types = ['frama', 'zero_lag_ema', 'kalman']
        for smoother_type in test_types:
            if smoother_type in smoothers:
                params = UnifiedSmoother.get_default_parameters(smoother_type)
                print(f"{smoother_type}: {params}")
        
        return True
        
    except ImportError as e:
        print(f"インポートエラー: {e}")
        return False


def test_basic_smoothing():
    """基本的なスムージングテスト"""
    print("\n" + "=" * 60)
    print("基本的なスムージングテスト")
    print("=" * 60)
    
    try:
        from indicators.smoother.unified_smoother import UnifiedSmoother
        
        # テストデータ生成
        data = generate_test_data(100)
        print(f"テストデータ: {len(data)}ポイント")
        print(f"価格範囲: {data['close'].min():.2f} - {data['close'].max():.2f}")
        
        # 各スムーサーをテスト
        test_smoothers = ['frama', 'super_smoother', 'zero_lag_ema', 'kalman']
        results = {}
        
        print("\nスムージング結果:")
        print("-" * 60)
        
        for smoother_type in test_smoothers:
            try:
                smoother = UnifiedSmoother(smoother_type=smoother_type, src_type='close')
                result = smoother.calculate(data)
                
                # 統計情報
                mean_smoothed = np.nanmean(result.values)
                mean_raw = np.nanmean(result.raw_values)
                valid_count = np.sum(~np.isnan(result.values))
                
                results[smoother_type] = result
                
                print(f"{smoother_type:<15}: 平均値={mean_smoothed:>8.4f}, "
                      f"元平均値={mean_raw:>8.4f}, 有効値={valid_count:>3}")
                
                # 追加データの確認
                if result.additional_data:
                    additional_info = ", ".join(result.additional_data.keys())
                    print(f"{'':>15}  追加データ: {additional_info}")
                
            except Exception as e:
                print(f"{smoother_type:<15}: エラー - {e}")
        
        print("-" * 60)
        return True
        
    except Exception as e:
        print(f"基本テストエラー: {e}")
        return False


def test_parameter_customization():
    """パラメータカスタマイゼーションテスト"""
    print("\n" + "=" * 60)
    print("パラメータカスタマイゼーションテスト")
    print("=" * 60)
    
    try:
        from indicators.smoother.unified_smoother import UnifiedSmoother
        
        data = generate_test_data(80)
        
        # FRAMAのパラメータテスト
        print("FRAMA パラメータテスト:")
        frama_configs = [
            {'period': 16, 'fc': 1, 'sc': 300},  # デフォルト
            {'period': 32, 'fc': 2, 'sc': 200},  # カスタム1
            {'period': 8, 'fc': 1, 'sc': 400},   # カスタム2
        ]
        
        for i, config in enumerate(frama_configs):
            smoother = UnifiedSmoother(smoother_type='frama', src_type='close', **config)
            result = smoother.calculate(data)
            
            mean_val = np.nanmean(result.values)
            valid_count = np.sum(~np.isnan(result.values))
            print(f"  設定{i+1} {config}: 平均値={mean_val:.4f}, 有効値={valid_count}")
        
        # Zero Lag EMAのパラメータテスト
        print("\nZero Lag EMA パラメータテスト:")
        zlema_configs = [
            {'period': 14, 'fast_mode': False},
            {'period': 21, 'fast_mode': True},
            {'period': 50, 'fast_mode': False},
        ]
        
        for i, config in enumerate(zlema_configs):
            smoother = UnifiedSmoother(smoother_type='zero_lag_ema', src_type='close', **config)
            result = smoother.calculate(data)
            
            mean_val = np.nanmean(result.values)
            valid_count = np.sum(~np.isnan(result.values))
            print(f"  設定{i+1} {config}: 平均値={mean_val:.4f}, 有効値={valid_count}")
        
        return True
        
    except Exception as e:
        print(f"パラメータテストエラー: {e}")
        return False


def test_different_sources():
    """異なるプライスソースのテスト"""
    print("\n" + "=" * 60)
    print("異なるプライスソースのテスト")
    print("=" * 60)
    
    try:
        from indicators.smoother.unified_smoother import UnifiedSmoother
        
        data = generate_test_data(60)
        sources = ['close', 'hlc3', 'hl2', 'ohlc4', 'high', 'low']
        
        print("FRAMA - 異なるプライスソース:")
        print("-" * 50)
        
        for src in sources:
            try:
                smoother = UnifiedSmoother(smoother_type='frama', src_type=src)
                result = smoother.calculate(data)
                
                mean_val = np.nanmean(result.values)
                valid_count = np.sum(~np.isnan(result.values))
                
                print(f"{src:>6}: 平均値={mean_val:>8.4f}, 有効値数={valid_count:>3}")
                
            except Exception as e:
                print(f"{src:>6}: エラー - {e}")
        
        print("-" * 50)
        return True
        
    except Exception as e:
        print(f"プライスソーステストエラー: {e}")
        return False


def test_convenience_functions():
    """便利関数のテスト"""
    print("\n" + "=" * 60)
    print("便利関数のテスト")
    print("=" * 60)
    
    try:
        from indicators.smoother.unified_smoother import smooth, compare_smoothers
        
        data = generate_test_data(50)
        
        # smooth関数のテスト
        print("smooth関数テスト:")
        smoothed_frama = smooth(data, smoother_type='frama', src_type='close')
        smoothed_zlema = smooth(data, smoother_type='zero_lag_ema', src_type='close', period=14)
        
        print(f"FRAMA結果: 形状={smoothed_frama.shape}, 平均値={np.nanmean(smoothed_frama):.4f}")
        print(f"ZLEMA結果: 形状={smoothed_zlema.shape}, 平均値={np.nanmean(smoothed_zlema):.4f}")
        
        # compare_smoothers関数のテスト
        print("\ncompare_smoothers関数テスト:")
        comparison = compare_smoothers(
            data, 
            smoother_types=['frama', 'zero_lag_ema', 'super_smoother'],
            src_type='close'
        )
        
        for smoother_type, values in comparison.items():
            mean_val = np.nanmean(values)
            valid_count = np.sum(~np.isnan(values))
            print(f"{smoother_type}: 平均値={mean_val:.4f}, 有効値={valid_count}")
        
        return True
        
    except Exception as e:
        print(f"便利関数テストエラー: {e}")
        return False


def test_cache_functionality():
    """キャッシュ機能のテスト"""
    print("\n" + "=" * 60)
    print("キャッシュ機能のテスト")
    print("=" * 60)
    
    try:
        from indicators.smoother.unified_smoother import UnifiedSmoother
        import time
        
        data = generate_test_data(200)
        smoother = UnifiedSmoother(smoother_type='frama', src_type='close')
        
        # 初回計算
        start_time = time.time()
        result1 = smoother.calculate(data)
        time1 = time.time() - start_time
        
        # 2回目計算（キャッシュヒット期待）
        start_time = time.time()
        result2 = smoother.calculate(data)
        time2 = time.time() - start_time
        
        # 結果比較
        print(f"初回計算時間: {time1:.4f}秒")
        print(f"2回目計算時間: {time2:.4f}秒")
        
        if time1 > 0:
            speedup = time1 / time2 if time2 > 0 else float('inf')
            print(f"速度向上: {speedup:.2f}倍")
        
        # 結果が同じかチェック
        values_equal = np.allclose(result1.values, result2.values, equal_nan=True)
        print(f"結果の一致: {'✓' if values_equal else '✗'}")
        
        return True
        
    except Exception as e:
        print(f"キャッシュテストエラー: {e}")
        return False


def test_edge_cases():
    """エッジケースのテスト"""
    print("\n" + "=" * 60)
    print("エッジケースのテスト")
    print("=" * 60)
    
    try:
        from indicators.smoother.unified_smoother import UnifiedSmoother
        
        # 小さなデータセット
        print("小さなデータセットテスト:")
        small_data = generate_test_data(5)
        smoother = UnifiedSmoother(smoother_type='frama', src_type='close')
        result = smoother.calculate(small_data)
        print(f"小データ結果: 有効値数={np.sum(~np.isnan(result.values))}")
        
        # 空のデータフレーム
        print("空データテスト:")
        empty_data = pd.DataFrame()
        try:
            result = smoother.calculate(empty_data)
            print(f"空データ結果: 形状={result.values.shape}")
        except Exception as e:
            print(f"空データエラー: {e}")
        
        # 無効なスムーサータイプ
        print("無効なスムーサータイプテスト:")
        try:
            invalid_smoother = UnifiedSmoother(smoother_type='invalid_type')
            print("無効なタイプが受け入れられました（問題）")
        except ValueError as e:
            print(f"期待通りのエラー: {e}")
        
        # 無効なプライスソース
        print("無効なプライスソーステスト:")
        try:
            invalid_source = UnifiedSmoother(smoother_type='frama', src_type='invalid_source')
            print("無効なソースが受け入れられました（問題）")
        except ValueError as e:
            print(f"期待通りのエラー: {e}")
        
        return True
        
    except Exception as e:
        print(f"エッジケーステストエラー: {e}")
        return False


def main():
    """メインテスト関数"""
    print("統合スムーサーのテストを開始します")
    
    try:
        # 利用可能なスムーサーテスト
        available_success = test_available_smoothers()
        
        # 基本スムージングテスト
        basic_success = test_basic_smoothing()
        
        # パラメータカスタマイゼーションテスト
        param_success = test_parameter_customization()
        
        # プライスソーステスト
        source_success = test_different_sources()
        
        # 便利関数テスト
        convenience_success = test_convenience_functions()
        
        # キャッシュ機能テスト
        cache_success = test_cache_functionality()
        
        # エッジケーステスト
        edge_success = test_edge_cases()
        
        print("\n" + "=" * 60)
        print("テスト結果サマリー")
        print("=" * 60)
        print(f"利用可能スムーサーテスト: {'✓ 成功' if available_success else '✗ 失敗'}")
        print(f"基本スムージングテスト: {'✓ 成功' if basic_success else '✗ 失敗'}")
        print(f"パラメータカスタマイゼーションテスト: {'✓ 成功' if param_success else '✗ 失敗'}")
        print(f"プライスソーステスト: {'✓ 成功' if source_success else '✗ 失敗'}")
        print(f"便利関数テスト: {'✓ 成功' if convenience_success else '✗ 失敗'}")
        print(f"キャッシュ機能テスト: {'✓ 成功' if cache_success else '✗ 失敗'}")
        print(f"エッジケーステスト: {'✓ 成功' if edge_success else '✗ 失敗'}")
        
        all_success = all([
            available_success, basic_success, param_success, source_success,
            convenience_success, cache_success, edge_success
        ])
        
        if all_success:
            print("\n全てのテストが成功しました！")
            return 0
        else:
            print("\n一部のテストが失敗しました。")
            return 1
            
    except Exception as e:
        print(f"テスト実行中にエラー: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())