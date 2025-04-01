#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict

from strategies.implementations.true_momentum.strategy import TrueMomentumStrategy
from signals.implementations.true_momentum import TrueMomentumEntrySignal, TrueMomentumDirectionSignal
from signals.implementations.guardian_angel.filter import GuardianAngelFilterSignal


class TestTrueMomentumStrategy(unittest.TestCase):
    """トゥルーモメンタムストラテジーのテスト"""
    
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
        
        # ストラテジーの初期化
        self.strategy = TrueMomentumStrategy(
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
            min_momentum_period=20,
            momentum_threshold=0.0,
            max_ga_period=100,
            min_ga_period=20,
            max_ga_threshold=61.8,
            min_ga_threshold=38.2
        )
    
    def test_entry_signal_generation(self):
        """エントリーシグナル生成のテスト"""
        # シグナルの生成
        entry_signals = self.strategy.generate_entry(self.data)
        
        # シグナルの形状が正しいことを確認
        self.assertEqual(len(entry_signals), len(self.data))
        
        # シグナル値が[-1, 0, 1]の範囲内であることを確認
        self.assertTrue(all(s in [-1, 0, 1] for s in entry_signals))
    
    def test_exit_signal_generation(self):
        """エグジットシグナル生成のテスト"""
        # ロングポジションでのエグジット
        long_exit = self.strategy.generate_exit(self.data, 1)
        
        # ショートポジションでのエグジット
        short_exit = self.strategy.generate_exit(self.data, -1)
        
        # シグナルの型がboolであることを確認
        self.assertIsInstance(long_exit, bool)
        self.assertIsInstance(short_exit, bool)
    
    def test_strategy_workflow(self):
        """戦略の一連の流れをテスト"""
        # エントリーシグナルを取得
        entry_signals = self.strategy.generate_entry(self.data)
        
        # エントリーポイントを特定
        long_entries = np.where(entry_signals == 1)[0]
        short_entries = np.where(entry_signals == -1)[0]
        
        if len(long_entries) > 0:
            # 最初のロングエントリーでのシミュレーション
            entry_index = long_entries[0]
            position = 1
            
            # エントリー後のデータでエグジットを検証
            for i in range(entry_index + 1, min(entry_index + 50, len(self.data))):
                should_exit = self.strategy.generate_exit(self.data.iloc[:i+1], position, i)
                if should_exit:
                    break
        
        if len(short_entries) > 0:
            # 最初のショートエントリーでのシミュレーション
            entry_index = short_entries[0]
            position = -1
            
            # エントリー後のデータでエグジットを検証
            for i in range(entry_index + 1, min(entry_index + 50, len(self.data))):
                should_exit = self.strategy.generate_exit(self.data.iloc[:i+1], position, i)
                if should_exit:
                    break
    
    def visualize_strategy(self):
        """戦略の視覚化（テストではなく、視覚的な確認用）"""
        # トゥルーモメンタムエントリーシグナル
        entry_signal = TrueMomentumEntrySignal(period=20)
        tm_entry = entry_signal.generate(self.data)
        
        # トゥルーモメンタム方向シグナル
        direction_signal = TrueMomentumDirectionSignal(period=20)
        tm_direction = direction_signal.generate(self.data)
        
        # ガーディアンエンジェルフィルター
        ga_filter = GuardianAngelFilterSignal(er_period=20)
        ga_signals = ga_filter.generate(self.data)
        
        # ストラテジーの組み合わせシグナル
        strategy_signals = self.strategy.generate_entry(self.data)
        
        # 可視化
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        
        # 価格チャート
        axes[0].plot(self.data.index, self.data['close'])
        axes[0].set_title('価格チャート')
        
        # トゥルーモメンタムエントリーシグナル
        axes[1].plot(self.data.index, tm_entry)
        axes[1].set_title('トゥルーモメンタムエントリーシグナル')
        axes[1].set_ylim(-1.5, 1.5)
        
        # トゥルーモメンタム方向シグナル
        axes[2].plot(self.data.index, tm_direction)
        axes[2].set_title('トゥルーモメンタム方向シグナル')
        axes[2].set_ylim(-1.5, 1.5)
        
        # ガーディアンエンジェルフィルター
        axes[3].plot(self.data.index, ga_signals)
        axes[3].set_title('ガーディアンエンジェルフィルター')
        axes[3].set_ylim(-1.5, 1.5)
        
        # 戦略エントリーシグナルをオーバーレイ
        for i, ax in enumerate(axes):
            if i > 0:  # 価格チャート以外
                buy_signals = np.where(strategy_signals == 1)[0]
                sell_signals = np.where(strategy_signals == -1)[0]
                
                if len(buy_signals) > 0:
                    ax.plot(self.data.index[buy_signals], [1] * len(buy_signals), '^', 
                            markersize=10, color='green', label='買いシグナル')
                
                if len(sell_signals) > 0:
                    ax.plot(self.data.index[sell_signals], [-1] * len(sell_signals), 'v', 
                            markersize=10, color='red', label='売りシグナル')
        
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # テスト実行
    unittest.main()
    
    # 可視化のみ実行する場合はコメントアウトを外す
    # test = TestTrueMomentumStrategy()
    # test.setUp()
    # test.visualize_strategy() 