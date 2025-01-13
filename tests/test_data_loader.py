#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from data.data_loader import DataLoader, CSVDataSource, IDataSource

# プロジェクトのルートディレクトリをPythonパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

class MockDataSource(IDataSource):
    """テスト用のモックデータソース"""
    def __init__(self):
        self.data = {}
        
        # テストデータの作成
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        self.test_data = pd.DataFrame({
            'open': np.random.uniform(100, 200, len(dates)),
            'high': np.random.uniform(150, 250, len(dates)),
            'low': np.random.uniform(50, 150, len(dates)),
            'close': np.random.uniform(100, 200, len(dates)),
            'volume': np.random.uniform(1000, 5000, len(dates))
        }, index=dates)
    
    def load_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> pd.DataFrame:
        df = self.test_data.copy()
        
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
            
        return df
    
    def get_available_symbols(self) -> list:
        return ['BTCUSDT', 'ETHUSDT']
    
    def get_available_timeframes(self, symbol: str) -> list:
        return ['1h', '4h', '1d']

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        """テストの準備"""
        self.mock_source = MockDataSource()
        self.loader = DataLoader(self.mock_source)
        
        # 一時ディレクトリの作成（CSVデータソースのテスト用）
        self.temp_dir = tempfile.mkdtemp()
        self._create_test_csv_files()
    
    def tearDown(self):
        """テストの後処理"""
        # 一時ディレクトリの削除
        shutil.rmtree(self.temp_dir)
    
    def _create_test_csv_files(self):
        """テスト用のCSVファイルを作成"""
        # ディレクトリ構造の作成
        symbol = 'BTCUSDT'
        timeframe = '1h'
        data_dir = Path(self.temp_dir) / symbol / timeframe
        data_dir.mkdir(parents=True)
        
        # テストデータの作成
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='h')
        test_data = pd.DataFrame({
            'open_time': [int(d.timestamp() * 1000) for d in dates],
            'open': np.random.uniform(100, 200, len(dates)),
            'high': np.random.uniform(150, 250, len(dates)),
            'low': np.random.uniform(50, 150, len(dates)),
            'close': np.random.uniform(100, 200, len(dates)),
            'volume': np.random.uniform(1000, 5000, len(dates)),
            'close_time': [int((d + timedelta(hours=1)).timestamp() * 1000) for d in dates],
            'quote_volume': np.random.uniform(100000, 500000, len(dates)),
            'trades': np.random.randint(100, 1000, len(dates)),
            'taker_buy_volume': np.random.uniform(500, 2500, len(dates)),
            'taker_buy_quote_volume': np.random.uniform(50000, 250000, len(dates)),
            'ignore': np.zeros(len(dates))
        })
        
        # 複数のCSVファイルに分割して保存
        chunk_size = len(test_data) // 3
        for i in range(3):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < 2 else len(test_data)
            chunk = test_data.iloc[start_idx:end_idx]
            chunk.to_csv(data_dir / f'data_{i+1}.csv', index=False, header=False)
    
    def test_load_market_data(self):
        """市場データの読み込みテスト"""
        # 全期間のデータ読み込み
        data = self.loader.load_market_data('BTCUSDT', '1h')
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)
        
        # 期間を指定してデータ読み込み
        start_date = datetime(2024, 1, 10)
        end_date = datetime(2024, 1, 20)
        data = self.loader.load_market_data(
            'BTCUSDT',
            '1h',
            start_date=start_date,
            end_date=end_date
        )
        self.assertTrue(data.index.min() >= start_date)
        self.assertTrue(data.index.max() <= end_date)
    
    def test_cache_functionality(self):
        """キャッシュ機能のテスト"""
        # 最初のデータ読み込み
        data1 = self.loader.load_market_data('BTCUSDT', '1h', use_cache=True)
        
        # キャッシュからの読み込み
        data2 = self.loader.load_market_data('BTCUSDT', '1h', use_cache=True)
        
        # 同じデータが返されることを確認
        pd.testing.assert_frame_equal(data1, data2)
        
        # キャッシュをクリアして再読み込み
        self.loader.clear_cache()
        data3 = self.loader.load_market_data('BTCUSDT', '1h', use_cache=True)
        
        # データの内容は同じだが、異なるオブジェクトであることを確認
        pd.testing.assert_frame_equal(data1, data3)
        self.assertIsNot(data1, data3)
    
    def test_csv_data_source(self):
        """CSVデータソースのテスト"""
        csv_source = CSVDataSource(self.temp_dir)
        loader = DataLoader(csv_source)
        
        # データの読み込み
        data = loader.load_market_data('BTCUSDT', '1h')
        
        # 基本的なチェック
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)
        self.assertTrue(data.index.is_monotonic_increasing)  # インデックスが昇順であることを確認
        
        # 必要なカラムが存在することを確認
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            self.assertIn(col, data.columns)
    
    def test_error_handling(self):
        """エラー処理のテスト"""
        csv_source = CSVDataSource(self.temp_dir)
        loader = DataLoader(csv_source)
        
        # 存在しない銘柄のテスト
        with self.assertRaises(FileNotFoundError):
            loader.load_market_data('NONEXISTENT', '1h')
        
        # 存在しない時間足のテスト
        with self.assertRaises(FileNotFoundError):
            loader.load_market_data('BTCUSDT', 'invalid')
    
    def test_load_data_from_config(self):
        """設定ファイルからのデータ読み込みテスト"""
        # テスト用の設定
        config = {
            'data': {
                'symbol': 'BTCUSDT',
                'timeframe': '1h',
                'start': '2024-01-10',
                'end': '2024-01-20'
            }
        }
        
        # データの読み込み
        data_dict = self.loader.load_data_from_config(config)
        
        # 基本的なチェック
        self.assertIn('BTCUSDT', data_dict)
        data = data_dict['BTCUSDT']
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)
        
        # 日付範囲のチェック
        start_dt = datetime(2024, 1, 10)
        end_dt = datetime(2024, 1, 20)
        self.assertTrue(data.index.min() >= start_dt)
        self.assertTrue(data.index.max() <= end_dt)
    
    def test_load_data_from_config_without_dates(self):
        """日付指定なしでの設定ファイルからのデータ読み込みテスト"""
        # 日付指定のない設定
        config = {
            'data': {
                'symbol': 'BTCUSDT',
                'timeframe': '1h'
            }
        }
        
        # データの読み込み
        data_dict = self.loader.load_data_from_config(config)
        
        # 基本的なチェック
        self.assertIn('BTCUSDT', data_dict)
        data = data_dict['BTCUSDT']
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)
        
        # 全期間のデータが読み込まれていることを確認
        self.assertEqual(len(data), len(self.mock_source.test_data))

if __name__ == '__main__':
    unittest.main()
