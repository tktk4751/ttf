#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger import get_logger

logger = get_logger(__name__)

class DataLoader:
    """
    Klineデータを読み込むクラス
    CSVファイルから時系列データを読み込み、pandas DataFrameとして提供
    """
    
    # CSVファイルのカラム名
    COLUMNS = [
        'open_time',          # 開始時刻（ミリ秒）
        'open',               # 始値
        'high',              # 高値
        'low',               # 安値
        'close',             # 終値
        'volume',            # 出来高
        'close_time',        # 終了時刻（ミリ秒）
        'quote_volume',      # 取引額
        'trades',            # 約定回数
        'taker_buy_volume',  # Takerの買い出来高
        'taker_buy_quote_volume',  # Takerの買い取引額
        'ignore'             # 無視するフィールド
    ]
    
    def __init__(self, data_dir: str):
        """データローダーの初期化
        
        Args:
            data_dir: データディレクトリのパス
        """
        self.data_dir = data_dir
        self.logger = logger

    def load_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        load_all: bool = False
    ) -> pd.DataFrame:
        """指定された銘柄と時間足のデータを読み込む
        
        Args:
            symbol: 銘柄名
            timeframe: 時間足
            start_date: 開始日（load_all=Falseの場合に使用）
            end_date: 終了日（load_all=Falseの場合に使用）
            load_all: Trueの場合、全データを読み込む。Falseの場合、日付範囲でフィルタリング
            
        Returns:
            pd.DataFrame: 読み込まれたデータ
        """
        # データファイルのパスを構築
        data_path = os.path.join(self.data_dir, symbol, timeframe)
        
        # ディレクトリ内の全CSVファイルを取得
        csv_files = sorted([f for f in os.listdir(data_path) if f.endswith('.csv')])
        
        if not csv_files:
            raise ValueError(f"データが見つかりません: {data_path}")
        
        # 全てのCSVファイルを読み込んで結合
        dfs = []
        for file in csv_files:
            file_path = os.path.join(data_path, file)
            df = self._load_csv(file_path)
            if df is not None:
                dfs.append(df)
        
        if not dfs:
            return pd.DataFrame(columns=self.COLUMNS)
        
        # データを結合
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # タイムスタンプをdatetimeに変換してインデックスに設定
        combined_df['timestamp'] = pd.to_datetime(combined_df['open_time'], unit='ms')
        combined_df.set_index('timestamp', inplace=True)
        
        # 不要なカラムを削除
        if 'open_time' in combined_df.columns:
            combined_df.drop('open_time', axis=1, inplace=True)
        if 'close_time' in combined_df.columns:
            combined_df.drop('close_time', axis=1, inplace=True)
        if 'ignore' in combined_df.columns:
            combined_df.drop('ignore', axis=1, inplace=True)
        
        # 重複を削除し、時系列でソート
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
        combined_df.sort_index(inplace=True)
        
        # データの日付範囲を取得
        data_start = combined_df.index.min()
        data_end = combined_df.index.max()
        
        # 日付範囲でフィルタリング（load_all=Falseの場合のみ）
        if not load_all and (start_date is not None or end_date is not None):
            # 指定された日付範囲がデータの範囲外の場合は空のDataFrameを返す
            if (start_date is not None and end_date is not None and 
                (start_date > data_end or end_date < data_start)):
                return pd.DataFrame(columns=[col for col in combined_df.columns])
            
            if start_date is not None:
                combined_df = combined_df[combined_df.index >= start_date]
            if end_date is not None:
                combined_df = combined_df[combined_df.index <= end_date]
        
        logger.info(f"データを読み込みました: {symbol}/{timeframe} (行数: {len(combined_df)}, 期間: {combined_df.index.min()} - {combined_df.index.max()})")
        
        return combined_df
    
    def _load_csv(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        CSVファイルを読み込む
        
        Args:
            file_path: CSVファイルのパス
        
        Returns:
            pandas DataFrame、エラー時はNone
        """
        try:
            df = pd.read_csv(
                file_path,
                names=self.COLUMNS,
                dtype={
                    'open_time': 'int64',
                    'open': 'float64',
                    'high': 'float64',
                    'low': 'float64',
                    'close': 'float64',
                    'volume': 'float64',
                    'close_time': 'int64',
                    'quote_volume': 'float64',
                    'trades': 'int64',
                    'taker_buy_volume': 'float64',
                    'taker_buy_quote_volume': 'float64',
                    'ignore': 'float64'
                }
            )
            return df
            
        except Exception as e:
            self.logger.error(f"CSVファイルの読み込みに失敗しました '{file_path}': {e}")
            return None
    
    def get_available_symbols(self) -> List[str]:
        """
        利用可能な銘柄のリストを取得
        
        Returns:
            銘柄のリスト
        """
        symbols = []
        for path in Path(self.data_dir).iterdir():
            if path.is_dir():
                symbols.append(path.name)
        return sorted(symbols)
    
    def get_available_timeframes(self, symbol: str) -> List[str]:
        """
        指定された銘柄で利用可能な時間足のリストを取得
        
        Args:
            symbol: 銘柄名
        
        Returns:
            時間足のリスト
        """
        symbol_dir = Path(self.data_dir) / symbol
        timeframes = []
        
        if symbol_dir.exists():
            for path in symbol_dir.iterdir():
                if path.is_dir():
                    timeframes.append(path.name)
        
        return sorted(timeframes)
