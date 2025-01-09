#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd

from logger import get_logger


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
    
    def __init__(self, data_dir: str = 'data/spot/monthly/klines'):
        """
        コンストラクタ
        
        Args:
            data_dir: データディレクトリのパス
        """
        self.logger = get_logger()
        self.data_dir = Path(data_dir)
        
        if not self.data_dir.exists():
            self.logger.warning(f"データディレクトリ '{self.data_dir}' が存在しません")
            self.data_dir.mkdir(parents=True)
    
    def load_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        指定された銘柄と時間足のデータを読み込む
        
        Args:
            symbol: 銘柄名 (例: "BTCUSDT")
            timeframe: 時間足 (例: "1h", "4h", "1d")
            start_date: 開始日時
            end_date: 終了日時
        
        Returns:
            pandas DataFrame
        """
        # 対象ディレクトリのパスを構築
        symbol_dir = self.data_dir / symbol / timeframe
        
        if not symbol_dir.exists():
            self.logger.error(f"指定された銘柄/時間足のディレクトリ '{symbol_dir}' が存在しません")
            return pd.DataFrame(columns=self.COLUMNS)
        
        # CSVファイルのリストを取得
        csv_files = sorted(glob.glob(str(symbol_dir / "*.csv")))
        
        if not csv_files:
            self.logger.warning(f"CSVファイルが見つかりません: {symbol_dir}")
            return pd.DataFrame(columns=self.COLUMNS)
        
        # 各CSVファイルを読み込んでリストに追加
        dfs: List[pd.DataFrame] = []
        for csv_file in csv_files:
            df = self._load_csv(csv_file)
            if df is not None:
                dfs.append(df)
        
        if not dfs:
            self.logger.warning("有効なデータが読み込めませんでした")
            return pd.DataFrame(columns=self.COLUMNS)
        
        # 全てのデータを結合
        data = pd.concat(dfs, ignore_index=True)
        
        # 日時でソート
        data = data.sort_values('open_time')
        
        # 重複を削除
        data = data.drop_duplicates(subset=['open_time'])
        
        # 日時でフィルタリング
        if start_date:
            start_ts = int(start_date.timestamp() * 1000)
            data = data[data['open_time'] >= start_ts]
        
        if end_date:
            end_ts = int(end_date.timestamp() * 1000)
            data = data[data['open_time'] <= end_ts]
        
        # インデックスをリセット
        data = data.reset_index(drop=True)
        
        self.logger.info(f"データを読み込みました: {symbol}/{timeframe} "
                      f"(行数: {len(data)}, 期間: {data['open_time'].iloc[0]} - {data['open_time'].iloc[-1]})")
        
        return data
    
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
        for path in self.data_dir.iterdir():
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
        symbol_dir = self.data_dir / symbol
        timeframes = []
        
        if symbol_dir.exists():
            for path in symbol_dir.iterdir():
                if path.is_dir():
                    timeframes.append(path.name)
        
        return sorted(timeframes)
