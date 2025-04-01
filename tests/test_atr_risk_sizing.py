#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import numpy as np
import pandas as pd
from position_sizing.atr_risk_sizing import AlphaATRRiskSizing
from position_sizing.position_sizing import PositionSizingParams
from indicators.cycle_efficiency_ratio import CycleEfficiencyRatio


class TestAlphaATRRiskSizing(unittest.TestCase):
    """AlphaATRRiskSizingクラスのテスト"""
    
    def setUp(self):
        """テスト用の共通セットアップ"""
        # テスト用の履歴データを生成（十分な長さを確保）
        np.random.seed(42)  # テストの一貫性のため
        n = 200  # AlphaATRとCERに十分なデータポイント
        close = 100 + np.cumsum(np.random.normal(0, 1, n))  # ランダムなウォーク
        high = close + np.random.uniform(0, 2, n)  # 高値
        low = close - np.random.uniform(0, 2, n)   # 安値
        open_price = close - np.random.uniform(-1, 1, n)  # 始値
        volume = np.random.uniform(100, 1000, n)   # 出来高
        
        # 一部にトレンドを入れる
        trend_start = 100
        trend_length = 40
        trend_slope = 0.5
        
        for i in range(trend_length):
            idx = trend_start + i
            close[idx] = close[trend_start] + i * trend_slope
            high[idx] = close[idx] + np.random.uniform(0, 1)
            low[idx] = close[idx] - np.random.uniform(0, 1)
        
        # データフレーム作成
        self.historical_data = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
        
        # 価格と残高の設定
        self.price = close[-1]
        self.capital = 10000.0
        
        # 基本的なポジションサイジングパラメータの設定
        self.basic_params = PositionSizingParams(
            entry_price=self.price,
            stop_loss_price=None,
            capital=self.capital,
            leverage=1.0,
            risk_per_trade=0.01,  # 1%リスク
            historical_data=self.historical_data
        )
    
    def test_basic_calculation(self):
        """基本的な計算のテスト"""
        # 標準パラメータでのインスタンス化
        sizer = AlphaATRRiskSizing(
            risk_ratio=0.01,  # 1%リスク
            unit=1.0
        )
        
        # 計算の実行
        result = sizer.calculate(self.basic_params)
        
        # 結果の検証
        self.assertIsNotNone(result)
        self.assertIn('position_size', result)
        self.assertIn('asset_quantity', result)
        self.assertIn('risk_amount', result)
        
        # ポジションサイズがゼロまたは負でないことを確認
        self.assertGreater(result['position_size'], 0)
        
        # リスク金額が資金の1%前後であることを検証
        expected_risk = self.capital * 0.01
        self.assertAlmostEqual(result['risk_amount'], expected_risk, delta=expected_risk * 0.5)
        
        # 資産数量が正しく計算されているか検証
        expected_quantity = result['position_size'] / self.price
        self.assertAlmostEqual(result['asset_quantity'], expected_quantity, delta=0.001)
    
    def test_risk_ratio_effect(self):
        """risk_ratioの変更が計算結果に反映されることを検証"""
        # ATR値を固定するためのモック関数
        def mock_get_absolute_atr():
            return np.array([1000.0])  # 固定ATR値 = 1000
            
        def mock_get_efficiency_ratio():
            return np.array([0.5])  # 固定効率比 = 0.5
            
        def mock_calculate(data, external_er=None):
            pass  # 何もしない
        
        # risk_ratio 1%での計算（効率比調整なし）
        sizer_1p = AlphaATRRiskSizing(
            risk_ratio=0.01,  # 1%リスク
            unit=1.0,
            apply_er_adjustment=False  # 効率比調整を無効化
        )
        
        # モンキーパッチを適用
        original_get_atr_1p = sizer_1p.alpha_atr.get_absolute_atr
        original_get_er_1p = sizer_1p.alpha_atr.get_efficiency_ratio
        original_calculate_1p = sizer_1p.alpha_atr.calculate
        
        sizer_1p.alpha_atr.get_absolute_atr = mock_get_absolute_atr
        sizer_1p.alpha_atr.get_efficiency_ratio = mock_get_efficiency_ratio
        sizer_1p.alpha_atr.calculate = mock_calculate
        
        # 計算を実行
        result_1p = sizer_1p.calculate(self.basic_params)
        
        # risk_ratio 2%での計算（効率比調整なし）
        sizer_2p = AlphaATRRiskSizing(
            risk_ratio=0.02,  # 2%リスク
            unit=1.0,
            apply_er_adjustment=False  # 効率比調整を無効化
        )
        
        # モンキーパッチを適用
        original_get_atr_2p = sizer_2p.alpha_atr.get_absolute_atr
        original_get_er_2p = sizer_2p.alpha_atr.get_efficiency_ratio
        original_calculate_2p = sizer_2p.alpha_atr.calculate
        
        sizer_2p.alpha_atr.get_absolute_atr = mock_get_absolute_atr
        sizer_2p.alpha_atr.get_efficiency_ratio = mock_get_efficiency_ratio
        sizer_2p.alpha_atr.calculate = mock_calculate
        
        # 計算を実行
        result_2p = sizer_2p.calculate(self.basic_params)
        
        # risk_ratio 5%での計算（効率比調整なし）
        sizer_5p = AlphaATRRiskSizing(
            risk_ratio=0.05,  # 5%リスク
            unit=1.0,
            apply_er_adjustment=False  # 効率比調整を無効化
        )
        
        # モンキーパッチを適用
        original_get_atr_5p = sizer_5p.alpha_atr.get_absolute_atr
        original_get_er_5p = sizer_5p.alpha_atr.get_efficiency_ratio
        original_calculate_5p = sizer_5p.alpha_atr.calculate
        
        sizer_5p.alpha_atr.get_absolute_atr = mock_get_absolute_atr
        sizer_5p.alpha_atr.get_efficiency_ratio = mock_get_efficiency_ratio
        sizer_5p.alpha_atr.calculate = mock_calculate
        
        # 計算を実行
        result_5p = sizer_5p.calculate(self.basic_params)
        
        try:
            # デバッグ出力
            print(f"\nテストデータ:")
            print(f"1% リスク - ポジションサイズ: {result_1p['position_size']}")
            print(f"2% リスク - ポジションサイズ: {result_2p['position_size']}")
            print(f"比率: {result_2p['position_size'] / result_1p['position_size']}")
            print(f"5% リスク - ポジションサイズ: {result_5p['position_size']}")
            print(f"比率: {result_5p['position_size'] / result_1p['position_size']}")
            
            # 計算式: 発注量USD = 残高USD × リスク比率 ÷ ATR × 現在価格USD × UNIT
            expected_1p = self.capital * 0.01 / 1000.0 * self.price * 1.0
            expected_2p = self.capital * 0.02 / 1000.0 * self.price * 1.0
            expected_5p = self.capital * 0.05 / 1000.0 * self.price * 1.0
            
            print(f"期待値 1%: {expected_1p}")
            print(f"期待値 2%: {expected_2p}")
            print(f"期待値 5%: {expected_5p}")
            print(f"期待比率 2%/1%: {expected_2p / expected_1p}")
            print(f"期待比率 5%/1%: {expected_5p / expected_1p}")
            
            # 実際の値と期待値を比較
            print(f"実際/期待 1%: {result_1p['position_size'] / expected_1p}")
            print(f"実際/期待 2%: {result_2p['position_size'] / expected_2p}")
            print(f"実際/期待 5%: {result_5p['position_size'] / expected_5p}")
            
            # 1%リスクの結果が期待通りか確認
            self.assertAlmostEqual(
                result_1p['position_size'],
                expected_1p,
                delta=expected_1p * 0.1
            )
            
            # 2%リスクの結果が期待通りか確認
            self.assertAlmostEqual(
                result_2p['position_size'],
                expected_2p,
                delta=expected_2p * 0.1
            )
            
            # 5%リスクの結果が期待通りか確認
            self.assertAlmostEqual(
                result_5p['position_size'],
                expected_5p,
                delta=expected_5p * 0.1
            )
        
        finally:
            # モンキーパッチを元に戻す
            sizer_1p.alpha_atr.get_absolute_atr = original_get_atr_1p
            sizer_1p.alpha_atr.get_efficiency_ratio = original_get_er_1p
            sizer_1p.alpha_atr.calculate = original_calculate_1p
            
            sizer_2p.alpha_atr.get_absolute_atr = original_get_atr_2p
            sizer_2p.alpha_atr.get_efficiency_ratio = original_get_er_2p
            sizer_2p.alpha_atr.calculate = original_calculate_2p
            
            sizer_5p.alpha_atr.get_absolute_atr = original_get_atr_5p
            sizer_5p.alpha_atr.get_efficiency_ratio = original_get_er_5p
            sizer_5p.alpha_atr.calculate = original_calculate_5p
    
    def test_different_parameters(self):
        """異なるパラメータでの計算のテスト"""
        # リスク比率とユニット係数を変更
        sizer = AlphaATRRiskSizing(
            risk_ratio=0.02,  # 2%リスク
            unit=1.5,         # 1.5倍のユニット
            atr_period=55     # ATR期間を55に設定
        )
        
        # 計算の実行
        result = sizer.calculate(self.basic_params)
        
        # 結果の検証
        # risk_ratioが2%であることを確認
        self.assertEqual(result['risk_ratio'], 0.02)
        
        # ユニット係数が1.5であることを確認
        self.assertEqual(result['unit'], 1.5)
    
    def test_max_position_limit(self):
        """最大ポジションサイズ制限のテスト"""
        # 高リスク比率と高ユニット係数
        sizer = AlphaATRRiskSizing(
            risk_ratio=0.1,   # 10%リスク
            unit=5.0,         # 5倍のユニット
            max_position_percent=0.3  # 資金の30%まで
        )
        
        # 計算の実行
        result = sizer.calculate(self.basic_params)
        
        # 結果の検証
        # risk_ratioが10%であることを確認
        self.assertEqual(result['risk_ratio'], 0.1)
        
        # ポジションサイズが資金の30%を超えていないことを確認
        max_allowed = self.capital * 0.3  # 最大許容ポジションサイズ
        self.assertLessEqual(result['position_size'], max_allowed)
    
    def test_calculation_with_fixed_atr(self):
        """固定ATR値での計算テスト（計算式の検証）"""
        # ATR値を固定するためのモック関数
        def mock_get_absolute_atr():
            return np.array([1000.0])  # 固定ATR値 = 1000
            
        def mock_get_efficiency_ratio():
            return np.array([0.5])  # 固定効率比 = 0.5
            
        def mock_calculate(data, external_er=None):
            pass  # 何もしない
        
        # 1%リスクでのテスト（効率比調整なし）
        sizer_1p = AlphaATRRiskSizing(risk_ratio=0.01, unit=1.0, apply_er_adjustment=False)
        
        # モンキーパッチを適用
        original_get_atr = sizer_1p.alpha_atr.get_absolute_atr
        original_get_er = sizer_1p.alpha_atr.get_efficiency_ratio
        original_calculate = sizer_1p.alpha_atr.calculate
        
        try:
            sizer_1p.alpha_atr.get_absolute_atr = mock_get_absolute_atr
            sizer_1p.alpha_atr.get_efficiency_ratio = mock_get_efficiency_ratio
            sizer_1p.alpha_atr.calculate = mock_calculate
            
            # 計算を実行
            result_1p = sizer_1p.calculate(self.basic_params)
            
            # 計算式: 発注量USD = 残高USD × リスク比率 ÷ ATR × 現在価格USD × UNIT
            # 10000 × 0.01 ÷ 1000 × price × 1.0 = price / 10
            expected_position_1p = self.capital * 0.01 / 1000.0 * self.price
            
            # 結果を検証
            self.assertAlmostEqual(result_1p['position_size'], expected_position_1p, delta=expected_position_1p * 0.1)
            
            # 2%リスクでのテスト（効率比調整なし）
            sizer_2p = AlphaATRRiskSizing(risk_ratio=0.02, unit=1.0, apply_er_adjustment=False)
            
            # モンキーパッチを適用
            original_get_atr_2p = sizer_2p.alpha_atr.get_absolute_atr
            original_get_er_2p = sizer_2p.alpha_atr.get_efficiency_ratio
            original_calculate_2p = sizer_2p.alpha_atr.calculate
            
            sizer_2p.alpha_atr.get_absolute_atr = mock_get_absolute_atr
            sizer_2p.alpha_atr.get_efficiency_ratio = mock_get_efficiency_ratio
            sizer_2p.alpha_atr.calculate = mock_calculate
            
            # 計算を実行
            result_2p = sizer_2p.calculate(self.basic_params)
            
            # 計算式: 発注量USD = 残高USD × リスク比率 ÷ ATR × 現在価格USD × UNIT
            # 10000 × 0.02 ÷ 1000 × price × 1.0 = price / 5
            expected_position_2p = self.capital * 0.02 / 1000.0 * self.price
            
            # 結果を検証
            self.assertAlmostEqual(result_2p['position_size'], expected_position_2p, delta=expected_position_2p * 0.1)
            
            # 2%のポジションサイズは1%の2倍になることを確認
            self.assertAlmostEqual(result_2p['position_size'] / result_1p['position_size'], 2.0, delta=0.1)
        
        finally:
            # モンキーパッチを元に戻す
            sizer_1p.alpha_atr.get_absolute_atr = original_get_atr
            sizer_1p.alpha_atr.get_efficiency_ratio = original_get_er
            sizer_1p.alpha_atr.calculate = original_calculate
            
            # 2%テストのためのモンキーパッチも元に戻す
            if 'sizer_2p' in locals():
                sizer_2p.alpha_atr.get_absolute_atr = original_get_atr_2p
                sizer_2p.alpha_atr.get_efficiency_ratio = original_get_er_2p
                sizer_2p.alpha_atr.calculate = original_calculate_2p


if __name__ == '__main__':
    unittest.main()