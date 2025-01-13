#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# プロジェクトのルートディレクトリをPythonパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from data.data_processor import DataProcessor

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        """テストの準備"""
        # テストデータの作成
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        self.test_data = pd.DataFrame({
            'open': np.random.uniform(100, 200, len(dates)),
            'high': np.random.uniform(150, 250, len(dates)),
            'low': np.random.uniform(50, 150, len(dates)),
            'close': np.random.uniform(100, 200, len(dates)),
            'volume': np.random.uniform(1000, 5000, len(dates))
        }, index=dates)
        
        # 一部のデータに欠損値を追加
        self.test_data_with_nan = self.test_data.copy()
        self.test_data_with_nan.loc[dates[5], 'close'] = np.nan
        self.test_data_with_nan.loc[dates[10], 'volume'] = np.nan
        
        # 一部のデータに異常値を追加
        self.test_data_with_invalid = self.test_data.copy()
        self.test_data_with_invalid.loc[dates[7], 'close'] = -100
        self.test_data_with_invalid.loc[dates[15], 'volume'] = 0
    
    def test_data_cleaning(self):
        """データクリーニングのテスト"""
        processor = DataProcessor()
        
        # 欠損値の処理
        cleaned_data = processor.process(self.test_data_with_nan)
        self.assertEqual(len(cleaned_data), len(self.test_data) - 2)
        self.assertTrue(cleaned_data['close'].notna().all())
        self.assertTrue(cleaned_data['volume'].notna().all())
        
        # 異常値の処理
        cleaned_data = processor.process(self.test_data_with_invalid)
        self.assertEqual(len(cleaned_data), len(self.test_data) - 2)
        self.assertTrue((cleaned_data['close'] > 0).all())
        self.assertTrue((cleaned_data['volume'] > 0).all())
    
    def test_empty_dataframe(self):
        """空のDataFrameの処理テスト"""
        processor = DataProcessor()
        empty_df = pd.DataFrame()
        processed_data = processor.process(empty_df)
        
        self.assertTrue(processed_data.empty)

if __name__ == '__main__':
    unittest.main()
