#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime
from pathlib import Path
from typing import Optional, List
import pandas as pd
from data.data_loader import IDataSource
from logger import get_logger

class BinanceDataSource(IDataSource):
    """Binanceの履歴データを読み込むクラス"""
    
    def __init__(self, base_dir: str = "data/binance"):
        """
        Args:
            base_dir: Binanceデータのベースディレクトリ
        """
        self.base_dir = Path(base_dir)
        self.logger = get_logger(__name__)

    def load_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        market_type: str = "spot"  # "spot" or "future"
    ) -> pd.DataFrame:
        """データを読み込む
        
        Args:
            symbol: 銘柄名（例: "BTC", "ETH"）
            timeframe: 時間足（例: "1h", "4h"）
            start_date: 開始日時
            end_date: 終了日時
            market_type: 市場タイプ（"spot" or "future"）
        
        Returns:
            pd.DataFrame: 読み込まれたデータ
        """
        # データファイルのパスを構築
        file_path = self.base_dir / symbol / market_type / timeframe / "historical_data.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"データファイルが見つかりません: {file_path}")
        
        try:
            # CSVファイルを読み込む
            df = pd.read_csv(file_path)
            
            # タイムスタンプをdatetimeに変換してインデックスに設定
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # データ型を変換
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_columns] = df[numeric_columns].astype(float)
            
            # 日付範囲でフィルタリング
            if start_date is not None:
                df = df[df.index >= start_date]
            if end_date is not None:
                df = df[df.index <= end_date]
            
            # 重複を削除し、時系列でソート
            df = df[~df.index.duplicated(keep='first')]
            df.sort_index(inplace=True)
            
            self.logger.info(
                f"Binanceデータを読み込みました: {symbol}/{market_type}/{timeframe} "
                f"(行数: {len(df)}, 期間: {df.index.min()} - {df.index.max()})"
            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"データファイルの読み込みに失敗しました '{file_path}': {e}")
            raise

    def get_available_symbols(self) -> List[str]:
        """利用可能な銘柄のリストを取得"""
        symbols = []
        for path in self.base_dir.iterdir():
            if path.is_dir():
                symbols.append(path.name)
        return sorted(symbols)

    def get_available_timeframes(self, symbol: str, market_type: str = "spot") -> List[str]:
        """指定された銘柄で利用可能な時間足のリストを取得"""
        timeframes = []
        timeframe_dir = self.base_dir / symbol / market_type
        
        if timeframe_dir.exists():
            for path in timeframe_dir.iterdir():
                if path.is_dir():
                    timeframes.append(path.name)
        
        return sorted(timeframes)

    def get_available_market_types(self, symbol: str) -> List[str]:
        """指定された銘柄で利用可能な市場タイプのリストを取得"""
        market_types = []
        symbol_dir = self.base_dir / symbol
        
        if symbol_dir.exists():
            for path in symbol_dir.iterdir():
                if path.is_dir():
                    market_types.append(path.name)
        
        return sorted(market_types) 