#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import sys

# TTFシステムのインポート
sys.path.append(str(Path(__file__).parent.parent.parent))
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource


class TrendFollowDataPipeline:
    """
    トレンドフォローモデル用データパイプライン
    
    仕様書通りの実装:
    - visualization/z_adaptive_channel_chart.pyを参考にした実際の相場データ取得
    - CSVデータの%ベース分割
    - 時系列データの適切な処理
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        self.config_path = config_path
        self.config = None
        self.raw_data = None
        self.processed_data = None
        
    def load_config(self) -> Dict[str, Any]:
        """設定ファイルを読み込む"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        return self.config
    
    def load_market_data(self) -> pd.DataFrame:
        """
        設定ファイルから実際の相場データを取得
        visualization/z_adaptive_channel_chart.pyの実装を参考
        
        Returns:
            処理済みのOHLCVデータフレーム
        """
        if self.config is None:
            self.load_config()
        
        print("相場データを読み込み中...")
        
        # データの準備
        binance_config = self.config.get('binance_data', {})
        data_dir = binance_config.get('data_dir', 'data/binance')
        binance_data_source = BinanceDataSource(data_dir)
        
        # CSVデータソースはダミーとして渡す（Binanceデータソースのみを使用）
        dummy_csv_source = CSVDataSource("dummy")
        data_loader = DataLoader(
            data_source=dummy_csv_source,
            binance_data_source=binance_data_source
        )
        data_processor = DataProcessor()
        
        # データの読み込みと処理
        self.raw_data = data_loader.load_data_from_config(self.config)
        processed_data = {
            symbol: data_processor.process(df)
            for symbol, df in self.raw_data.items()
        }
        
        # 最初のシンボルのデータを取得
        first_symbol = next(iter(processed_data))
        self.processed_data = processed_data[first_symbol]
        
        print(f"データ読み込み完了: {first_symbol}")
        print(f"期間: {self.processed_data.index.min()} → {self.processed_data.index.max()}")
        print(f"データ数: {len(self.processed_data)}")
        
        return self.processed_data
    
    def split_data_by_percentage(self, 
                               data: pd.DataFrame, 
                               train_pct: float = 70, 
                               val_pct: float = 15, 
                               test_pct: float = 15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        CSVデータを%ベースで分割
        
        Args:
            data: 全CSVデータ
            train_pct: 訓練データの割合（デフォルト: 70%）
            val_pct: 検証データの割合（デフォルト: 15%）
            test_pct: テストデータの割合（デフォルト: 15%）
        
        Returns:
            train_data, val_data, test_data
        """
        total_len = len(data)
        
        train_end = int(total_len * train_pct / 100)
        val_end = int(total_len * (train_pct + val_pct) / 100)
        
        train_data = data.iloc[:train_end].copy()
        val_data = data.iloc[train_end:val_end].copy()
        test_data = data.iloc[val_end:].copy()
        
        print(f"データ分割完了:")
        print(f"  訓練データ: {len(train_data)}行 ({train_pct}%)")
        print(f"  検証データ: {len(val_data)}行 ({val_pct}%)")
        print(f"  テストデータ: {len(test_data)}行 ({test_pct}%)")
        
        return train_data, val_data, test_data
    
    def walk_forward_analysis(self, 
                            data: pd.DataFrame, 
                            train_pct: float = 70, 
                            step_pct: float = 10) -> list:
        """
        %ベースウォークフォワード分析
        
        Args:
            data: 全CSVデータ
            train_pct: 訓練期間の割合
            step_pct: スライドステップの割合
        
        Returns:
            データ分割結果のリスト
        """
        total_len = len(data)
        train_size = int(total_len * train_pct / 100)
        step_size = int(total_len * step_pct / 100)
        
        results = []
        
        for start_idx in range(0, total_len - train_size, step_size):
            train_end = start_idx + train_size
            test_end = min(train_end + step_size, total_len)
            
            if test_end <= train_end:
                break
            
            train_data = data.iloc[start_idx:train_end].copy()
            test_data = data.iloc[train_end:test_end].copy()
            
            results.append({
                'train_data': train_data,
                'test_data': test_data,
                'train_period': f"{train_data.index.min()} - {train_data.index.max()}",
                'test_period': f"{test_data.index.min()} - {test_data.index.max()}",
                'train_size': len(train_data),
                'test_size': len(test_data)
            })
        
        print(f"ウォークフォワード分析: {len(results)}回の分割を生成")
        return results
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        データの妥当性をチェック
        
        Args:
            data: チェック対象のデータフレーム
        
        Returns:
            データが妥当かどうか
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # 必要なカラムの存在チェック
        if not all(col in data.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in data.columns]
            print(f"エラー: 必要なカラムが不足しています: {missing_cols}")
            return False
        
        # NaN値のチェック
        nan_counts = data[required_columns].isnull().sum()
        if nan_counts.sum() > 0:
            print(f"警告: NaN値が検出されました:\n{nan_counts}")
        
        # データサイズのチェック
        if len(data) < 1000:
            print(f"警告: データサイズが小さすぎます: {len(data)}行")
            return False
        
        # 価格データの妥当性チェック
        if (data['high'] < data['low']).any():
            print("エラー: 高値が安値を下回る行があります")
            return False
        
        if (data['high'] < data['close']).any() or (data['low'] > data['close']).any():
            print("エラー: 終値が高値・安値の範囲外にあります")
            return False
        
        print("データ検証完了: 妥当なデータです")
        return True
    
    def get_data_info(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        データの基本情報を取得
        
        Args:
            data: 対象データフレーム
        
        Returns:
            データ情報の辞書
        """
        return {
            'total_rows': len(data),
            'start_date': data.index.min(),
            'end_date': data.index.max(),
            'columns': list(data.columns),
            'null_counts': data.isnull().sum().to_dict(),
            'price_range': {
                'min_close': data['close'].min(),
                'max_close': data['close'].max(),
                'mean_close': data['close'].mean()
            },
            'volume_stats': {
                'min_volume': data['volume'].min(),
                'max_volume': data['volume'].max(),
                'mean_volume': data['volume'].mean()
            }
        }


def main():
    """メイン実行関数"""
    # データパイプラインの初期化
    pipeline = TrendFollowDataPipeline()
    
    # データの読み込み
    data = pipeline.load_market_data()
    
    # データの検証
    if pipeline.validate_data(data):
        # データ分割の実行
        train_data, val_data, test_data = pipeline.split_data_by_percentage(data)
        
        # データ情報の表示
        print("\n=== データ情報 ===")
        for name, dataset in [("訓練", train_data), ("検証", val_data), ("テスト", test_data)]:
            info = pipeline.get_data_info(dataset)
            print(f"\n{name}データ:")
            print(f"  期間: {info['start_date']} - {info['end_date']}")
            print(f"  行数: {info['total_rows']}")
            print(f"  価格範囲: {info['price_range']['min_close']:.2f} - {info['price_range']['max_close']:.2f}")


if __name__ == "__main__":
    main()