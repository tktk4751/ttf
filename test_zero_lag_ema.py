#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ゼロラグEMAのテストスクリプト
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def generate_test_data(length: int = 200) -> pd.DataFrame:
    """テスト用のOHLCデータを生成"""
    np.random.seed(42)
    
    # トレンドのある価格データ生成
    base_price = 100.0
    trend = 0.002
    volatility = 0.02
    
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


def test_basic_zlema():
    """基本的なZLEMAテスト"""
    print("=" * 50)
    print("基本的なZLEMAテスト")
    print("=" * 50)
    
    try:
        from indicators.smoother.zero_lag_ema import ZeroLagEMA, zlema, fast_zlema
        print("インポート成功")
    except ImportError as e:
        print(f"インポートエラー: {e}")
        return False
    
    # テストデータ生成
    print("テストデータ生成中...")
    data = generate_test_data(100)
    print(f"データ形状: {data.shape}")
    print(f"価格範囲: {data['close'].min():.2f} - {data['close'].max():.2f}")
    
    # 基本ZLEMA テスト
    print("\n基本ZLEMAテスト中...")
    zlema_indicator = ZeroLagEMA(period=21, src_type='close')
    result = zlema_indicator.calculate(data)
    
    print(f"ZLEMA結果形状: {result.values.shape}")
    print(f"ZLEMA平均値: {np.nanmean(result.values):.4f}")
    print(f"有効値数: {np.sum(~np.isnan(result.values))}")
    
    # EMA値の確認
    if not np.all(np.isnan(result.ema_values)):
        print(f"EMA平均値: {np.nanmean(result.ema_values):.4f}")
    
    # ラグ除去データの確認
    if not np.all(np.isnan(result.lag_reduced_data)):
        print(f"ラグ除去データ平均値: {np.nanmean(result.lag_reduced_data):.4f}")
    
    return True


def test_different_sources():
    """異なるプライスソースのテスト"""
    print("\n" + "=" * 50)
    print("異なるプライスソースのテスト")
    print("=" * 50)
    
    try:
        from indicators.smoother.zero_lag_ema import ZeroLagEMA
    except ImportError as e:
        print(f"インポートエラー: {e}")
        return False
    
    data = generate_test_data(80)
    sources = ['close', 'hlc3', 'hl2', 'ohlc4', 'high', 'low']
    
    print("異なるプライスソースでのZLEMA比較:")
    print("-" * 50)
    
    results = {}
    for src in sources:
        try:
            indicator = ZeroLagEMA(period=14, src_type=src)
            result = indicator.calculate(data)
            
            mean_val = np.nanmean(result.values)
            valid_count = np.sum(~np.isnan(result.values))
            
            results[src] = mean_val
            print(f"{src:>6}: 平均値={mean_val:>8.4f}, 有効値数={valid_count:>3}")
            
        except Exception as e:
            print(f"{src:>6}: エラー - {e}")
    
    print("-" * 50)
    return True


def test_fast_mode():
    """高速モードのテスト"""
    print("\n" + "=" * 50)
    print("高速モードのテスト")
    print("=" * 50)
    
    try:
        from indicators.smoother.zero_lag_ema import ZeroLagEMA
        import time
    except ImportError as e:
        print(f"インポートエラー: {e}")
        return False
    
    data = generate_test_data(500)  # 大きなデータセット
    period = 21
    
    # 通常モード
    print("通常モードでの計算...")
    start_time = time.time()
    indicator_normal = ZeroLagEMA(period=period, fast_mode=False)
    result_normal = indicator_normal.calculate(data)
    normal_time = time.time() - start_time
    
    # 高速モード
    print("高速モードでの計算...")
    start_time = time.time()
    indicator_fast = ZeroLagEMA(period=period, fast_mode=True)
    result_fast = indicator_fast.calculate(data)
    fast_time = time.time() - start_time
    
    # 結果比較
    print("\n結果比較:")
    print(f"通常モード: 時間={normal_time:.4f}s, 平均値={np.nanmean(result_normal.values):.4f}")
    print(f"高速モード: 時間={fast_time:.4f}s, 平均値={np.nanmean(result_fast.values):.4f}")
    
    # 性能比較
    if normal_time > 0:
        speedup = normal_time / fast_time
        print(f"速度向上: {speedup:.2f}倍")
    
    # 精度比較（有効値のみ）
    valid_normal = ~np.isnan(result_normal.values)
    valid_fast = ~np.isnan(result_fast.values)
    valid_both = valid_normal & valid_fast
    
    if np.any(valid_both):
        mae = np.mean(np.abs(result_normal.values[valid_both] - result_fast.values[valid_both]))
        print(f"平均絶対誤差 (通常vs高速): {mae:.6f}")
    
    return True


def test_convenience_functions():
    """便利関数のテスト"""
    print("\n" + "=" * 50)
    print("便利関数のテスト")
    print("=" * 50)
    
    try:
        from indicators.smoother.zero_lag_ema import zlema, fast_zlema
    except ImportError as e:
        print(f"インポートエラー: {e}")
        return False
    
    data = generate_test_data(60)
    
    # zlema関数のテスト
    print("zlema関数テスト...")
    zlema_result = zlema(data, period=20, src_type='close')
    print(f"zlema結果: 形状={zlema_result.shape}, 平均値={np.nanmean(zlema_result):.4f}")
    
    # fast_zlema関数のテスト
    print("fast_zlema関数テスト...")
    fast_result = fast_zlema(data, period=20, src_type='close')
    print(f"fast_zlema結果: 形状={fast_result.shape}, 平均値={np.nanmean(fast_result):.4f}")
    
    # カスタムアルファのテスト
    print("カスタムアルファテスト...")
    custom_result = fast_zlema(data, period=20, src_type='close', alpha=0.2)
    print(f"カスタムアルファ結果: 形状={custom_result.shape}, 平均値={np.nanmean(custom_result):.4f}")
    
    return True


def test_edge_cases():
    """エッジケースのテスト"""
    print("\n" + "=" * 50)
    print("エッジケースのテスト")
    print("=" * 50)
    
    try:
        from indicators.smoother.zero_lag_ema import ZeroLagEMA
    except ImportError as e:
        print(f"インポートエラー: {e}")
        return False
    
    # 小さなデータセット
    print("小さなデータセットテスト...")
    small_data = generate_test_data(5)
    indicator = ZeroLagEMA(period=10)  # period > data length
    result = indicator.calculate(small_data)
    print(f"小データ結果: 有効値数={np.sum(~np.isnan(result.values))}")
    
    # 空のデータ
    print("空データテスト...")
    empty_data = pd.DataFrame()
    try:
        result = indicator.calculate(empty_data)
        print(f"空データ結果: 形状={result.values.shape}")
    except Exception as e:
        print(f"空データエラー: {e}")
    
    # 単一データポイント
    print("単一データポイントテスト...")
    single_data = generate_test_data(1)
    result = indicator.calculate(single_data)
    print(f"単一データ結果: 有効値数={np.sum(~np.isnan(result.values))}")
    
    return True


def main():
    """メインテスト関数"""
    print("ゼロラグEMAのテストを開始します")
    
    try:
        # 基本テスト
        basic_success = test_basic_zlema()
        
        # プライスソーステスト
        sources_success = test_different_sources()
        
        # 高速モードテスト
        fast_mode_success = test_fast_mode()
        
        # 便利関数テスト
        convenience_success = test_convenience_functions()
        
        # エッジケーステスト
        edge_cases_success = test_edge_cases()
        
        print("\n" + "=" * 50)
        print("テスト結果サマリー")
        print("=" * 50)
        print(f"基本テスト: {'✓ 成功' if basic_success else '✗ 失敗'}")
        print(f"プライスソーステスト: {'✓ 成功' if sources_success else '✗ 失敗'}")
        print(f"高速モードテスト: {'✓ 成功' if fast_mode_success else '✗ 失敗'}")
        print(f"便利関数テスト: {'✓ 成功' if convenience_success else '✗ 失敗'}")
        print(f"エッジケーステスト: {'✓ 成功' if edge_cases_success else '✗ 失敗'}")
        
        all_success = all([basic_success, sources_success, fast_mode_success, 
                          convenience_success, edge_cases_success])
        
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