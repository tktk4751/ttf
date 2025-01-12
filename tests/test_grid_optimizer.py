#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import unittest
import numpy as np
from datetime import datetime

# プロジェクトのルートディレクトリをPythonパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from optimization.grid_optimizer import GridOptimizer
from strategies.supertrend_rsi_chopstrategy import SupertrendRsiChopStrategy


class TestGridOptimizer(unittest.TestCase):
    """グリッドサーチ最適化のテストクラス"""
    
    def setUp(self):
        """テストの前準備"""
        self.config_path = os.path.join(project_root, 'config.yaml')
        
        # パラメーター範囲の定義
        supertrend_period = np.arange(5, 61, 5)  # 5から60まで5刻み
        supertrend_multiplier = np.arange(1.5, 5.5, 0.5)  # 1.5から5.0まで0.5刻み
        rsi_exit_period = np.arange(2, 22, 2)  # 2から20まで2刻み
        chop_period = np.arange(5, 61, 5)  # 5から60まで5刻み

        # パラメーターの組み合わせを生成
        self.param_ranges = {
            'supertrend_period': supertrend_period,
            'supertrend_multiplier': supertrend_multiplier,
            'rsi_exit_period': rsi_exit_period,
            'chop_period': chop_period
        }
    
    def test_grid_optimization(self):
        """グリッドサーチによる最適化のテスト"""
        # 最適化の実行
        optimizer = GridOptimizer(
            config_path=self.config_path,
            strategy_class=SupertrendRsiChopStrategy,
            param_ranges=self.param_ranges,
            n_jobs=1  # テスト時は並列処理を無効化
        )
        
        best_params, best_score, best_trades = optimizer.optimize()
        
        # アサーション
        self.assertIsNotNone(best_params)
        self.assertIsNotNone(best_score)
        self.assertIsNotNone(best_trades)
        self.assertGreater(best_score, 0)
        self.assertGreater(len(best_trades), 0)
        
        # パラメーターの範囲チェック
        self.assertGreaterEqual(best_params['supertrend_period'], 5)
        self.assertLessEqual(best_params['supertrend_period'], 60)
        self.assertGreaterEqual(best_params['supertrend_multiplier'], 1.5)
        self.assertLessEqual(best_params['supertrend_multiplier'], 5.0)
        self.assertGreaterEqual(best_params['rsi_exit_period'], 2)
        self.assertLessEqual(best_params['rsi_exit_period'], 20)
        self.assertGreaterEqual(best_params['chop_period'], 5)
        self.assertLessEqual(best_params['chop_period'], 60)


if __name__ == '__main__':
    unittest.main() 