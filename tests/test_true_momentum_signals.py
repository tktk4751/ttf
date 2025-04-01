#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import numpy as np
import pandas as pd

from indicators.true_momentum import TrueMomentum
from signals.implementations.true_momentum import TrueMomentumEntrySignal, TrueMomentumDirectionSignal


class TestTrueMomentumSignals(unittest.TestCase):
    """トゥルーモメンタムシグナルのテスト"""
    
    def setUp(self):
        """テスト用のデータを準備"""
        np.random.seed(42)
        n = 200
        
        # 初期価格
        price = 100.0
        
        # 価格データの生成
        dates = pd.date_range(start='2023-01-01', periods=n)
        prices = []
        
        for i in range(n):
            # ランダムな価格変動を生成
            change = np.random.normal(0, 1)
            
            # トレンドを加える（50日周期）
            trend = 0.1 * np.sin(i / 25 * np.pi)
            
            # 価格を更新
            price += change + trend
            prices.append(price)
        
        # ボラティリティの高い期間を作成
        volatility_period = slice(80, 120)
        prices_array = np.array(prices)
        prices_array[volatility_period] += np.random.normal(0, 3, size=40)
        
        # トレンド期間を作成
        trend_period = slice(150, 180)
        trend_values = np.linspace(0, 10, 30)
        prices_array[trend_period] += trend_values
        
        # データフレームを作成
        self.data = pd.DataFrame({
            'date': dates,
            'open': prices_array,
            'high': prices_array + np.random.uniform(0.1, 1.0, size=n),
            'low': prices_array - np.random.uniform(0.1, 1.0, size=n),
            'close': prices_array,
            'volume': np.random.randint(1000, 10000, size=n)
        })
        
        self.data.set_index('date', inplace=True)
        
        # インジケーターとシグナルの初期化
        self.indicator = TrueMomentum(
            period=20,
            max_std_mult=2.0,
            min_std_mult=1.0,
            max_kama_slow=55,
            min_kama_slow=30,
            max_kama_fast=13,
            min_kama_fast=2,
            max_atr_period=120,
            min_atr_period=13,
            max_atr_mult=3.0,
            min_atr_mult=1.0,
            max_momentum_period=100,
            min_momentum_period=20
        )
        
        self.entry_signal = TrueMomentumEntrySignal(
            period=20,
            momentum_threshold=0.0
        )
        
        self.direction_signal = TrueMomentumDirectionSignal(
            period=20
        )
    
    def test_indicator_calculation(self):
        """インジケーターの計算結果のテスト"""
        # モメンタムの計算
        momentum = self.indicator.calculate(self.data)
        
        # ナンゼロ値であることを確認
        self.assertFalse(np.isnan(momentum[-1]))
        
        # 動的モメンタム期間のテスト
        dynamic_period = self.indicator.get_dynamic_momentum_period()
        # 期間が閾値内であることを確認
        self.assertTrue(all(20 <= p <= 100 for p in dynamic_period[~np.isnan(dynamic_period)]))
    
    def test_entry_signal_generation(self):
        """エントリーシグナルの生成テスト"""
        # シグナルの生成
        signals = self.entry_signal.generate(self.data)
        
        # シグナルの形状が正しいことを確認
        self.assertEqual(len(signals), len(self.data))
        
        # シグナル値が[-1, 0, 1]の範囲内であることを確認
        self.assertTrue(all(s in [-1, 0, 1] for s in signals))
        
        # 初期期間はシグナルがないことを確認
        self.assertTrue(all(s == 0 for s in signals[:20]))
    
    def test_direction_signal_generation(self):
        """方向シグナルの生成テスト"""
        # シグナルの生成
        signals = self.direction_signal.generate(self.data)
        
        # シグナルの形状が正しいことを確認
        self.assertEqual(len(signals), len(self.data))
        
        # シグナル値が[-1, 0, 1]の範囲内であることを確認
        self.assertTrue(all(s in [-1, 0, 1] for s in signals))
        
        # 初期期間はシグナルがないことを確認
        self.assertTrue(all(s == 0 for s in signals[:20]))
    
    def test_signal_alignment(self):
        """エントリーシグナルと方向シグナルの整合性テスト"""
        # モメンタムの計算
        momentum = self.indicator.calculate(self.data)
        
        # シグナルの生成
        entry_signals = self.entry_signal.generate(self.data)
        direction_signals = self.direction_signal.generate(self.data)
        
        # 正のモメンタムでは、方向シグナルがロング（1）であることを確認
        for i in range(len(momentum)):
            if not np.isnan(momentum[i]) and momentum[i] > 0:
                self.assertTrue(direction_signals[i] >= 0)
            elif not np.isnan(momentum[i]) and momentum[i] < 0:
                self.assertTrue(direction_signals[i] <= 0)


if __name__ == '__main__':
    unittest.main()