#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import unittest
from datetime import datetime
import pandas as pd
import numpy as np

# プロジェクトのルートディレクトリをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from walkforward.walkforward import WalkForward
from strategies.supertrend_rsi_chopstrategy import SupertrendRsiChopStrategy
from tests.test_optimizer import create_supertrend_rsi_params
from logger import get_logger

logger = get_logger(__name__)

def create_supertrend_rsi_params(trial):
    """SupertrendRsiChopStrategyのパラメータを生成"""
    return {
        'supertrend_params': {
            'period': trial.suggest_int('supertrend_period', 5, 100, step=1),
            'multiplier': trial.suggest_float('supertrend_multiplier', 1.5, 8.0, step=0.5)
        },
        'rsi_entry_params': {
            'period': 2,
            'solid': {
                'rsi_long_entry': 20,
                'rsi_short_entry': 80
            }
        },
        'rsi_exit_params': {
            'period': trial.suggest_int('rsi_exit_period', 5, 34, step=1),
            'solid': {
                'rsi_long_exit_solid': 85,
                'rsi_short_exit_solid': 15
            }
        },
        'chop_params': {
            'period': trial.suggest_int('chop_period', 5, 100, step=1),
            'solid': {
                'chop_solid': 50
            }
        }
    }

class TestWalkForward(unittest.TestCase):
    def setUp(self):
        """テストの準備"""
        self.config_path = os.path.join(project_root, 'config.yaml')
        self.strategy_class = SupertrendRsiChopStrategy
        self.param_generator = create_supertrend_rsi_params

    def test_walkforward(self):
        """ウォークフォワードテストの実行テスト"""
        # WalkForwardクラスのインスタンス化
        walkforward = WalkForward(
            strategy_class=self.strategy_class,
            config_path=self.config_path,
            param_generator=self.param_generator
        )

        # ウォークフォワードテストの実行
        results = walkforward.run()

        # 結果の検証
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

        # 各期間の結果を検証
        for result in results:
            # 必要なキーが存在することを確認
            required_keys = [
                'training_start', 'training_end',
                'test_start', 'test_end',
                'parameters', 'training_score',
                'test_trades', 'test_alpha_score'
            ]
            for key in required_keys:
                self.assertIn(key, result)

            # 日付の順序を確認
            self.assertLess(result['training_start'], result['training_end'])
            self.assertLess(result['test_start'], result['test_end'])
            self.assertEqual(result['training_end'], result['test_start'])

            # トレーニング期間とテスト期間の長さを確認
            training_days = (result['training_end'] - result['training_start']).days
            test_days = (result['test_end'] - result['test_start']).days
            self.assertAlmostEqual(training_days, 360, delta=1)
            self.assertAlmostEqual(test_days, 180, delta=1)

            # スコアと取引数の検証
            self.assertGreaterEqual(result['test_trades'], 30)
            self.assertGreaterEqual(result['test_alpha_score'], 0)

        # 結果の表示
        walkforward.print_results(results)

if __name__ == '__main__':
    unittest.main()
