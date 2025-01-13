#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Protocol
import pandas as pd
from logger import get_logger

class IDataProcessor(Protocol):
    """データ処理のインターフェース"""
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """データを処理する"""
        ...

class DataProcessor(IDataProcessor):
    """データ処理クラス"""
    def __init__(self):
        self.logger = get_logger(__name__)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """データを処理する
        
        Args:
            df: 処理対象のDataFrame
            
        Returns:
            処理済みのDataFrame
        """
        if df.empty:
            return df
            
        processed = df.copy()
        
        # 基本的なデータクリーニング
        processed = self._clean_data(processed)
        
        return processed

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """データのクリーニング処理
        
        - 欠損値の処理
        - 異常値の除去
        - データ型の変換
        など
        """
        # 欠損値の処理
        df = df.dropna()
        
        # 価格がゼロ以下の行を除外
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df.columns:
                df = df[df[col] > 0]
        
        # 出来高がゼロ以下の行を除外
        if 'volume' in df.columns:
            df = df[df['volume'] > 0]
        
        # インデックスの再設定
        df = df.sort_index()
        
        return df
