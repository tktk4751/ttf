#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import numpy as np
import pandas as pd
from strategies.implementations.z_donchian_trend.strategy import ZDonchianTrendStrategy
from strategies.implementations.z_donchian_trend.signal_generator import ZDonchianTrendSignalGenerator


class TestZDonchianTrend(unittest.TestCase):
    """ZドンチャンとZトレンドフィルター戦略のテスト"""

    @classmethod
    def setUpClass(cls):
        """テストデータの準備"""
        # データの作成
        np.random.seed(42)  # 再現性のある結果のため
        size = 500
        close = np.cumsum(np.random.normal(0, 1, size))
        high = close + np.random.uniform(0, 1, size)
        low = close - np.random.uniform(0, 1, size)
        open_price = close + np.random.normal(0, 0.5, size)
        volume = np.random.randint(100, 1000, size)
        
        # 上昇トレンドと下降トレンドを人工的に作成
        trend_up = np.arange(100) * 0.1
        trend_down = np.arange(100) * -0.1
        
        # トレンドをデータに挿入
        close[100:200] += trend_up
        close[300:400] += trend_down
        high[100:200] += trend_up
        high[300:400] += trend_down
        low[100:200] += trend_up
        low[300:400] += trend_down
        open_price[100:200] += trend_up
        open_price[300:400] += trend_down
        
        # DataFrameに変換
        cls.data = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })

    def test_initialization(self):
        """初期化のテスト"""
        strategy = ZDonchianTrendStrategy()
        self.assertIsInstance(strategy, ZDonchianTrendStrategy)
        self.assertIsInstance(strategy.signal_generator, ZDonchianTrendSignalGenerator)

    def test_entry_signal_generation(self):
        """エントリーシグナル生成のテスト"""
        strategy = ZDonchianTrendStrategy()
        entry_signals = strategy.generate_entry(self.data)
        self.assertEqual(len(entry_signals), len(self.data))
        
        # シグナルの値がLong(1)、Short(-1)、またはNoSignal(0)のみであることを確認
        unique_signals = np.unique(entry_signals)
        for signal in unique_signals:
            self.assertTrue(signal in [-1, 0, 1], f"不正なシグナル値: {signal}")

    def test_exit_signal_generation(self):
        """エグジットシグナル生成のテスト"""
        strategy = ZDonchianTrendStrategy()
        
        # ロングポジションからのエグジット
        exit_signal = strategy.generate_exit(self.data, 1)
        self.assertIsInstance(exit_signal, bool)
        
        # ショートポジションからのエグジット
        exit_signal = strategy.generate_exit(self.data, -1)
        self.assertIsInstance(exit_signal, bool)

    def test_optimization_params(self):
        """最適化パラメータの生成テスト"""
        import optuna
        trial = optuna.trial.Trial(optuna.study.Study(storage=None, sampler=optuna.samplers.RandomSampler()), 0)
        
        params = ZDonchianTrendStrategy.create_optimization_params(trial)
        self.assertIsInstance(params, dict)
        
        # 必要なパラメータがすべて含まれていることを確認
        self.assertIn('cycle_detector_type', params)
        self.assertIn('lp_period', params)
        self.assertIn('hp_period', params)
        self.assertIn('max_dc_cycle_part', params)
        self.assertIn('min_dc_cycle_part', params)
        self.assertIn('max_threshold', params)
        self.assertIn('min_threshold', params)

    def test_params_conversion(self):
        """パラメータ変換のテスト"""
        # テスト用パラメータ
        params = {
            'cycle_detector_type': 'hody_dc',
            'lp_period': 5,
            'hp_period': 144,
            'cycle_part': 0.5,
            'src_type': 'hlc3',
            'max_dc_cycle_part': 0.5,
            'max_dc_max_cycle': 144,
            'max_dc_min_cycle': 5,
            'max_dc_max_output': 89,
            'max_dc_min_output': 21,
            'min_dc_cycle_part': 0.25,
            'min_dc_max_cycle': 55,
            'min_dc_min_cycle': 5,
            'min_dc_max_output': 21,
            'min_dc_min_output': 8,
            'lookback': 1,
            'max_stddev_period': 13,
            'min_stddev_period': 5,
            'max_lookback_period': 13,
            'min_lookback_period': 5,
            'max_rms_window': 13,
            'min_rms_window': 5,
            'max_threshold': 0.75,
            'min_threshold': 0.55,
            'combination_weight': 0.6,
            'zadx_weight': 0.4,
            'combination_method': 'sigmoid',
            'max_chop_dc_cycle_part': 0.5,
            'max_chop_dc_max_cycle': 144,
            'max_chop_dc_min_cycle': 10,
            'max_chop_dc_max_output': 34,
            'max_chop_dc_min_output': 13,
            'min_chop_dc_cycle_part': 0.25,
            'min_chop_dc_max_cycle': 55,
            'min_chop_dc_min_cycle': 5,
            'min_chop_dc_max_output': 13,
            'min_chop_dc_min_output': 5
        }
        
        strategy_params = ZDonchianTrendStrategy.convert_params_to_strategy_format(params)
        self.assertIsInstance(strategy_params, dict)
        
        # パラメータが正しく変換されたことを確認
        for key, value in params.items():
            if isinstance(value, (int, float, str)):
                self.assertEqual(strategy_params[key], value, f"パラメータ {key} が正しく変換されていません")

    def test_signal_with_different_parameters(self):
        """異なるパラメータでのシグナル生成テスト"""
        # 基本パラメータのストラテジー
        default_strategy = ZDonchianTrendStrategy()
        default_signals = default_strategy.generate_entry(self.data)
        
        # パラメータを変更したストラテジー
        custom_strategy = ZDonchianTrendStrategy(
            cycle_detector_type='phac_dc',
            lp_period=8,
            hp_period=100,
            cycle_part=0.4,
            max_dc_cycle_part=0.4,
            min_dc_cycle_part=0.2,
            max_threshold=0.65,
            min_threshold=0.5
        )
        custom_signals = custom_strategy.generate_entry(self.data)
        
        # 異なるパラメータで異なるシグナルが生成されることを確認
        self.assertFalse(np.array_equal(default_signals, custom_signals),
                         "異なるパラメータ設定でも同じシグナルが生成されています")


if __name__ == '__main__':
    unittest.main() 