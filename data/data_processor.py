#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional
from datetime import datetime

import pandas as pd

from logger import get_logger


class DataProcessor:
    """
    データの前処理を行うクラス
    - タイムスタンプの変換
    - 不要なカラムの削除
    - データの整形
    """
    
    def __init__(self):
        """コンストラクタ"""
        self.logger = get_logger()
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        データを前処理する
        
        Args:
            df: 生データのDataFrame
        
        Returns:
            処理済みのDataFrame
        """
        # 深いコピーを作成
        processed = df.copy(deep=True)
        
        # タイムスタンプを変換
        processed = self._convert_timestamps(processed)
        
        # 不要なカラムを削除
        processed = self._drop_unnecessary_columns(processed)
        
        # インデックスを設定
        processed.set_index('datetime', inplace=True)
        
        self.logger.info(
            f"データを処理しました: {len(processed)}行 "
            f"({processed.index[0]} - {processed.index[-1]})"
        )
        
        return processed
    
    def _convert_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        タイムスタンプを人間が読める形式に変換
        
        Args:
            df: 変換前のDataFrame
        
        Returns:
            変換後のDataFrame
        """
        # UNIXタイムスタンプ（ミリ秒）をdatetimeに変換
        df['datetime'] = pd.to_datetime(df['open_time'] / 1000, unit='s')
        
        return df
    
    def _drop_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        不要なカラムを削除
        
        Args:
            df: 処理前のDataFrame
        
        Returns:
            処理後のDataFrame
        """
        # チャート表示に必要なカラムのみを残す
        columns_to_keep = [
            'datetime',
            'open',
            'high',
            'low',
            'close',
            'volume'
        ]
        
        return df[columns_to_keep]

