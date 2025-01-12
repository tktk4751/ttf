#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import unittest
from datetime import datetime
import pandas as pd

# プロジェクトのルートディレクトリをPythonパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from data.data_loader import DataLoader


class TestDataLoader(unittest.TestCase):
    """DataLoaderのテストクラス"""
    
    def setUp(self):
        """テストの前準備"""
        self.data_dir = os.path.join(project_root, 'data/spot/monthly/klines')
        self.data_loader = DataLoader(data_dir=self.data_dir)
        self.symbol = 'BTCUSDT'
        self.timeframe = '4h'
    
    def test_load_data_with_date_range(self):
        """日付範囲を指定してデータを読み込むテスト"""
        # 2023年1月から3月までのデータを読み込む
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 3, 31)
        
        df = self.data_loader.load_data(
            symbol=self.symbol,
            timeframe=self.timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        # データフレームの検証
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(len(df) > 0)
        
        # 日付範囲の検証
        self.assertTrue(df.index.min() >= start_date)
        self.assertTrue(df.index.max() <= end_date)
        
        # カラムの存在確認
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            self.assertIn(col, df.columns)
    
    def test_load_all_data(self):
        """全データを読み込むテスト"""
        df = self.data_loader.load_data(
            symbol=self.symbol,
            timeframe=self.timeframe,
            load_all=True
        )
        
        # データフレームの検証
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(len(df) > 0)
        
        # 2020年8月から2024年11月までのデータが含まれているか確認
        self.assertTrue(df.index.min() <= datetime(2020, 8, 1))
        self.assertTrue(df.index.max() >= datetime(2024, 11, 1))
        
        # カラムの存在確認
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in expected_columns:
            self.assertIn(col, df.columns)
    
    def test_load_data_with_invalid_dates(self):
        """無効な日付範囲でのデータ読み込みテスト"""
        # データ期間外の日付を指定（2017年8月以前のデータは存在しない）
        start_date = datetime(2017, 1, 1)
        end_date = datetime(2017, 6, 30)
        
        df = self.data_loader.load_data(
            symbol=self.symbol,
            timeframe=self.timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        # 空のデータフレームが返されることを確認
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 0)
    
    def test_get_available_symbols(self):
        """利用可能な銘柄の取得テスト"""
        symbols = self.data_loader.get_available_symbols()
        
        self.assertIsInstance(symbols, list)
        self.assertIn(self.symbol, symbols)
    
    def test_get_available_timeframes(self):
        """利用可能な時間足の取得テスト"""
        timeframes = self.data_loader.get_available_timeframes(self.symbol)
        
        self.assertIsInstance(timeframes, list)
        self.assertIn(self.timeframe, timeframes)


if __name__ == '__main__':
    unittest.main()
