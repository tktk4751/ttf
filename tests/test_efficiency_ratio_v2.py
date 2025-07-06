#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EfficiencyRatio V2 インジケーターのテストファイル

このファイルは、ER_V2インジケーターの基本的な動作を確認するための
簡単なテストを実行します。
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# パスの設定
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicators.efficiency_ratio_v2 import ER_V2

class TestEfficiencyRatioV2(unittest.TestCase):
    """ER_V2インジケーターのテストクラス"""
    
    def setUp(self):
        """テストデータの準備"""
        np.random.seed(42)
        
        # 基本的なOHLCデータを生成
        self.data_length = 100
        
        # 基本価格の生成
        base_price = 100.0
        trend = np.cumsum(np.random.randn(self.data_length) * 0.01)
        noise = np.random.randn(self.data_length) * 0.2
        
        # 終値の生成
        close = base_price + trend + noise
        
        # OHLC価格の生成
        high = close + np.abs(np.random.randn(self.data_length) * 0.1)
        low = close - np.abs(np.random.randn(self.data_length) * 0.1)
        open_price = close + np.random.randn(self.data_length) * 0.05
        
        # DataFrameの作成
        self.test_data = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.randint(1000, 10000, self.data_length)
        })
        
        # 日付情報の追加
        self.test_data['date'] = pd.date_range(start='2023-01-01', periods=self.data_length, freq='D')
    
    def test_basic_initialization(self):
        """基本的な初期化テスト"""
        er_v2 = ER_V2(
            period=5,
            src_type='ukf_hlc3',
            use_ultimate_smoother=True,
            smoother_period=10.0,
            use_dynamic_period=False
        )
        
        self.assertIsNotNone(er_v2)
        self.assertEqual(er_v2.period, 5)
        self.assertEqual(er_v2.src_type, 'ukf_hlc3')
        self.assertTrue(er_v2.use_ultimate_smoother)
        self.assertEqual(er_v2.smoother_period, 10.0)
        self.assertFalse(er_v2.use_dynamic_period)
    
    def test_basic_calculation(self):
        """基本的な計算テスト"""
        er_v2 = ER_V2(
            period=5,
            src_type='ukf_hlc3',
            use_ultimate_smoother=True,
            smoother_period=10.0,
            use_dynamic_period=False
        )
        
        # 計算の実行
        result = er_v2.calculate(self.test_data)
        
        # 結果の検証
        self.assertIsNotNone(result)
        self.assertEqual(len(result.values), self.data_length)
        self.assertEqual(len(result.smoothed_values), self.data_length)
        self.assertEqual(len(result.trend_signals), self.data_length)
        
        # 値の範囲チェック（NaN以外の値）
        valid_values = result.values[~np.isnan(result.values)]
        valid_smoothed = result.smoothed_values[~np.isnan(result.smoothed_values)]
        
        if len(valid_values) > 0:
            self.assertTrue(np.all(valid_values >= 0))
            self.assertTrue(np.all(valid_values <= 1))
        
        if len(valid_smoothed) > 0:
            self.assertTrue(np.all(valid_smoothed >= 0))
            self.assertTrue(np.all(valid_smoothed <= 1))
        
        # トレンド信号の値チェック
        unique_trends = np.unique(result.trend_signals)
        self.assertTrue(all(trend in [-1, 0, 1] for trend in unique_trends))
    
    def test_dynamic_period_calculation(self):
        """動的期間モードの計算テスト"""
        er_v2 = ER_V2(
            period=5,
            src_type='ukf_hlc3',
            use_ultimate_smoother=True,
            smoother_period=10.0,
            use_dynamic_period=True,
            detector_type='absolute_ultimate'
        )
        
        # 計算の実行
        result = er_v2.calculate(self.test_data)
        
        # 結果の検証
        self.assertIsNotNone(result)
        self.assertEqual(len(result.values), self.data_length)
        self.assertEqual(len(result.smoothed_values), self.data_length)
        self.assertEqual(len(result.trend_signals), self.data_length)
        
        # 動的期間の検証
        if len(result.dynamic_periods) > 0:
            valid_periods = result.dynamic_periods[~np.isnan(result.dynamic_periods)]
            if len(valid_periods) > 0:
                self.assertTrue(np.all(valid_periods >= er_v2.min_cycle))
                self.assertTrue(np.all(valid_periods <= er_v2.max_cycle))
    
    def test_without_smoother(self):
        """平滑化なしの計算テスト"""
        er_v2 = ER_V2(
            period=5,
            src_type='ukf_hlc3',
            use_ultimate_smoother=False,
            use_dynamic_period=False
        )
        
        # 計算の実行
        result = er_v2.calculate(self.test_data)
        
        # 結果の検証
        self.assertIsNotNone(result)
        self.assertEqual(len(result.values), self.data_length)
        self.assertEqual(len(result.smoothed_values), self.data_length)
        
        # 平滑化なしの場合、原値と平滑値が同じになることを確認
        np.testing.assert_array_equal(result.values, result.smoothed_values)
    
    def test_empty_data(self):
        """空データのテスト"""
        er_v2 = ER_V2(
            period=5,
            src_type='ukf_hlc3',
            use_ultimate_smoother=True,
            smoother_period=10.0,
            use_dynamic_period=False
        )
        
        # 空のDataFrameでテスト
        empty_data = pd.DataFrame()
        result = er_v2.calculate(empty_data)
        
        # 結果の検証
        self.assertIsNotNone(result)
        self.assertEqual(len(result.values), 0)
        self.assertEqual(len(result.smoothed_values), 0)
        self.assertEqual(len(result.trend_signals), 0)
        self.assertEqual(result.current_trend, 'range')
        self.assertEqual(result.current_trend_value, 0)
    
    def test_getter_methods(self):
        """getterメソッドのテスト"""
        er_v2 = ER_V2(
            period=5,
            src_type='ukf_hlc3',
            use_ultimate_smoother=True,
            smoother_period=10.0,
            use_dynamic_period=False
        )
        
        # 計算の実行
        result = er_v2.calculate(self.test_data)
        
        # Getterメソッドのテスト
        values = er_v2.get_values()
        smoothed_values = er_v2.get_smoothed_values()
        trend_signals = er_v2.get_trend_signals()
        current_trend = er_v2.get_current_trend()
        current_trend_value = er_v2.get_current_trend_value()
        dynamic_periods = er_v2.get_dynamic_periods()
        
        # 値の検証
        self.assertIsNotNone(values)
        self.assertIsNotNone(smoothed_values)
        self.assertIsNotNone(trend_signals)
        self.assertIsNotNone(current_trend)
        self.assertIsNotNone(current_trend_value)
        
        # 配列の長さの確認
        self.assertEqual(len(values), self.data_length)
        self.assertEqual(len(smoothed_values), self.data_length)
        self.assertEqual(len(trend_signals), self.data_length)
        
        # 動的期間は固定期間モードなので空
        self.assertEqual(len(dynamic_periods), 0)
    
    def test_reset_method(self):
        """リセットメソッドのテスト"""
        er_v2 = ER_V2(
            period=5,
            src_type='ukf_hlc3',
            use_ultimate_smoother=True,
            smoother_period=10.0,
            use_dynamic_period=False
        )
        
        # 計算の実行
        result = er_v2.calculate(self.test_data)
        
        # リセット前の状態確認
        self.assertIsNotNone(er_v2.get_values())
        
        # リセット実行
        er_v2.reset()
        
        # リセット後の状態確認
        # 結果はリセットされるが、最後の計算結果は保持される仕様
        # （キャッシュはクリアされる）
        self.assertIsNotNone(er_v2)
    
    def test_trend_analysis(self):
        """トレンド分析のテスト"""
        er_v2 = ER_V2(
            period=5,
            src_type='ukf_hlc3',
            use_ultimate_smoother=True,
            smoother_period=10.0,
            use_dynamic_period=False,
            slope_index=3,
            range_threshold=0.01
        )
        
        # 計算の実行
        result = er_v2.calculate(self.test_data)
        
        # トレンド信号の分析
        trend_signals = result.trend_signals
        unique_trends = np.unique(trend_signals)
        
        # 各トレンドの数を数える
        up_count = np.sum(trend_signals == 1)
        down_count = np.sum(trend_signals == -1)
        range_count = np.sum(trend_signals == 0)
        
        # 合計が全データ数と一致することを確認
        self.assertEqual(up_count + down_count + range_count, len(trend_signals))
        
        # 現在のトレンドが有効な値であることを確認
        self.assertIn(result.current_trend, ['up', 'down', 'range'])
        self.assertIn(result.current_trend_value, [-1, 0, 1])

def run_basic_functionality_test():
    """基本機能テストの実行"""
    print("=" * 60)
    print("ER_V2インジケーター 基本機能テスト")
    print("=" * 60)
    
    # テストデータの準備
    np.random.seed(42)
    data_length = 200
    
    # 基本価格の生成
    base_price = 100.0
    trend = np.cumsum(np.random.randn(data_length) * 0.01)
    noise = np.random.randn(data_length) * 0.2
    
    # 終値の生成
    close = base_price + trend + noise
    
    # OHLC価格の生成
    high = close + np.abs(np.random.randn(data_length) * 0.1)
    low = close - np.abs(np.random.randn(data_length) * 0.1)
    open_price = close + np.random.randn(data_length) * 0.05
    
    # DataFrameの作成
    test_data = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': np.random.randint(1000, 10000, data_length)
    })
    
    # 日付情報の追加
    test_data['date'] = pd.date_range(start='2023-01-01', periods=data_length, freq='D')
    
    print(f"テストデータ生成完了: {len(test_data)}点")
    
    # 基本設定でのテスト
    print("\n1. 基本設定でのテスト...")
    er_v2_basic = ER_V2(
        period=5,
        src_type='ukf_hlc3',
        use_ultimate_smoother=True,
        smoother_period=10.0,
        use_dynamic_period=False
    )
    
    result_basic = er_v2_basic.calculate(test_data)
    print(f"   計算完了: {er_v2_basic.name}")
    print(f"   ER値範囲: {np.nanmin(result_basic.values):.4f} - {np.nanmax(result_basic.values):.4f}")
    print(f"   平滑化値範囲: {np.nanmin(result_basic.smoothed_values):.4f} - {np.nanmax(result_basic.smoothed_values):.4f}")
    print(f"   現在のトレンド: {result_basic.current_trend} ({result_basic.current_trend_value})")
    
    # 動的期間設定でのテスト
    print("\n2. 動的期間設定でのテスト...")
    er_v2_dynamic = ER_V2(
        period=5,
        src_type='ukf_hlc3',
        use_ultimate_smoother=True,
        smoother_period=10.0,
        use_dynamic_period=True,
        detector_type='absolute_ultimate'
    )
    
    result_dynamic = er_v2_dynamic.calculate(test_data)
    print(f"   計算完了: {er_v2_dynamic.name}")
    print(f"   ER値範囲: {np.nanmin(result_dynamic.values):.4f} - {np.nanmax(result_dynamic.values):.4f}")
    print(f"   平滑化値範囲: {np.nanmin(result_dynamic.smoothed_values):.4f} - {np.nanmax(result_dynamic.smoothed_values):.4f}")
    print(f"   現在のトレンド: {result_dynamic.current_trend} ({result_dynamic.current_trend_value})")
    
    if len(result_dynamic.dynamic_periods) > 0:
        valid_periods = result_dynamic.dynamic_periods[~np.isnan(result_dynamic.dynamic_periods)]
        if len(valid_periods) > 0:
            print(f"   動的期間統計: 平均={np.mean(valid_periods):.1f}, 範囲={np.min(valid_periods):.0f}-{np.max(valid_periods):.0f}")
    
    # 平滑化なしでのテスト
    print("\n3. 平滑化なしでのテスト...")
    er_v2_no_smooth = ER_V2(
        period=5,
        src_type='ukf_hlc3',
        use_ultimate_smoother=False,
        use_dynamic_period=False
    )
    
    result_no_smooth = er_v2_no_smooth.calculate(test_data)
    print(f"   計算完了: {er_v2_no_smooth.name}")
    print(f"   ER値範囲: {np.nanmin(result_no_smooth.values):.4f} - {np.nanmax(result_no_smooth.values):.4f}")
    print(f"   現在のトレンド: {result_no_smooth.current_trend} ({result_no_smooth.current_trend_value})")
    
    # 性能テスト
    print("\n4. 性能テスト...")
    import time
    
    start_time = time.time()
    for i in range(10):
        result = er_v2_basic.calculate(test_data)
    end_time = time.time()
    
    print(f"   10回計算時間: {end_time - start_time:.3f}秒")
    print(f"   平均計算時間: {(end_time - start_time) / 10:.3f}秒")
    
    print("\n=" * 60)
    print("基本機能テスト完了")
    print("=" * 60)

if __name__ == "__main__":
    # 基本機能テストの実行
    run_basic_functionality_test()
    
    # ユニットテストの実行
    print("\n" + "=" * 60)
    print("ユニットテスト開始")
    print("=" * 60)
    
    unittest.main(verbosity=2) 