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
from strategies.supertrend_rsi_chop_strategy import SupertrendRsiChopStrategy
from logger import get_logger

logger = get_logger(__name__)


class TestWalkForward(unittest.TestCase):
    def setUp(self):
        """テストの準備"""
        self.config_path = os.path.join(project_root, 'config.yaml')
        self.strategy_class = SupertrendRsiChopStrategy
        self.param_generator = SupertrendRsiChopStrategy.create_optimization_params

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
                'testing_start', 'testing_end',
                'best_params', 'training_alpha',
                'trades', 'test_alpha'
            ]
            for key in required_keys:
                self.assertIn(key, result)

            # 日付の順序を確認
            self.assertLess(result['training_start'], result['training_end'])
            self.assertLess(result['testing_start'], result['testing_end'])
            self.assertEqual(result['training_end'], result['testing_start'])

            # トレーニング期間とテスト期間の長さを確認
            training_days = (result['training_end'] - result['training_start']).days
            test_days = (result['testing_end'] - result['testing_start']).days
            self.assertAlmostEqual(training_days, 360, delta=1)
            self.assertAlmostEqual(test_days, 180, delta=1)

            # スコアと取引数の検証
            self.assertGreaterEqual(len(result['trades']), 15)  # min_tradesの値に合わせて変更
            self.assertGreaterEqual(result['test_alpha'], 0)

        # 結果の表示
        walkforward.print_results(results)

if __name__ == '__main__':
    unittest.main()
