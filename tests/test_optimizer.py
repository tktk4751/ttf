#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import unittest
from datetime import datetime

# プロジェクトのルートディレクトリをPythonパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from optimization.Bayesian_optimizer import StrategyOptimizer
from strategies.supertrend_rsi_chopstrategy import SupertrendRsiChopStrategy


def create_supertrend_rsi_params(trial):
    """SupertrendRsiChopStrategyのパラメーター生成関数"""
    return {
        'supertrend_params': {
            'period': trial.suggest_int('supertrend_period', 5, 100, step=1),
            'multiplier': trial.suggest_float('supertrend_multiplier', 1.5, 5.0, step=0.5)
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


class TestStrategyOptimizer(unittest.TestCase):
    """最適化機能のテストクラス"""
    
    def setUp(self):
        """テストの前準備"""
        self.config_path = os.path.join(project_root, 'config.yaml')
    
    def test_supertrend_rsi_optimization(self):
        """SupertrendRsiChopStrategyの最適化テスト"""
        # 最適化の実行
        optimizer = StrategyOptimizer(
            config_path=self.config_path,
            strategy_class=SupertrendRsiChopStrategy,
            param_generator=create_supertrend_rsi_params,
            n_trials=500,  # 試行回数
            n_jobs=-1,     # 全CPU使用
            timeout=None   # タイムアウトなし
        )
        
        best_params, best_score, best_trades = optimizer.optimize()
        
        # アサーション
        self.assertIsNotNone(best_params)
        self.assertIsNotNone(best_score)
        self.assertIsNotNone(best_trades)
        self.assertGreater(best_score, 0)
        self.assertGreater(len(best_trades), 0)
        
        # パラメーターの範囲チェック
        self.assertIn('supertrend_period', best_params)
        self.assertIn('supertrend_multiplier', best_params)
        self.assertIn('rsi_exit_period', best_params)
        self.assertIn('chop_period', best_params)
        
        # スーパートレンドのパラメーター範囲チェック
        self.assertGreaterEqual(best_params['supertrend_period'], 5)
        self.assertLessEqual(best_params['supertrend_period'], 100)
        self.assertGreaterEqual(best_params['supertrend_multiplier'], 1.5)
        self.assertLessEqual(best_params['supertrend_multiplier'], 5.0)
        
        # RSIのパラメーター範囲チェック

        self.assertGreaterEqual(best_params['rsi_exit_period'], 5)
        self.assertLessEqual(best_params['rsi_exit_period'], 34)
        
        # Choppinessのパラメーター範囲チェック
        self.assertGreaterEqual(best_params['chop_period'], 5)
        self.assertLessEqual(best_params['chop_period'], 100)


if __name__ == '__main__':
    unittest.main()
