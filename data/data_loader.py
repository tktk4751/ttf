#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd

from logger import get_logger


class DataLoader:
    """
    Binanceのklineデータを読み込むクラス
    CSVファイルから時系列データを読み込み、pandas DataFrameとして提供
    """
    
    # CSVファイルのカラム定義
    COLUMNS = [
        'open_time',             # 開始時刻（ミリ秒）
        'open',                  # 始値
        'high',                 # 高値
        'low',                  # 安値
        'close',                # 終値
        'volume',               # 出来高
        'close_time',           # 終了時刻（ミリ秒）
        'quote_volume',         # 取引額
        'trades',               # 約定回数
        'taker_buy_volume',     # Takerの買い出来高
        'taker_buy_quote_volume', # Takerの買い取引額
        'ignore'                # 無視するフィールド
    ]
    
    # データ型の定義
    DTYPES = {
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
    
    def __init__(self, config: Dict[str, Any]):
        """
        コンストラクタ
        
        Args:
            config: 設定辞書（config.yamlの内容）
        """
        self.logger = get_logger()
        self.data_dir = Path(config.get('data', {}).get('data_dir', 'data/spot/monthly/klines'))
        self.symbol = config.get('data', {}).get('symbol', 'BTCUSDT')
        self.timeframe = config.get('data', {}).get('timeframe', '1h')
    
    def load_month(self, month: str) -> pd.DataFrame:
        """
        指定された月のデータを読み込む
        
        Args:
            month: 対象月（YYYY-MM形式）
        
        Returns:
            pandas DataFrame
        """
        # CSVファイルのパスを構築
        csv_path = self.data_dir / self.symbol / self.timeframe / f"{self.symbol}-{self.timeframe}-{month}.csv"
        
        if not csv_path.exists():
            raise FileNotFoundError(f"CSVファイルが見つかりません: {csv_path}")
        
        self.logger.info(f"データを読み込みます: {csv_path}")
        
        # CSVファイルを読み込む
        df = pd.read_csv(
            csv_path,
            names=self.COLUMNS,
            header=None,
            dtype=self.DTYPES
        )
        
        self.logger.info(f"データを読み込みました: {len(df)}行")
        
        return df
    
    def get_available_months(self) -> List[str]:
        """
        利用可能な月のリストを取得
        
        Returns:
            月のリスト（YYYY-MM形式）
        """
        data_dir = self.data_dir / self.symbol / self.timeframe
        if not data_dir.exists():
            return []
        
        months = []
        for file in data_dir.glob(f"{self.symbol}-{self.timeframe}-*.csv"):
            # ファイル名からYYYY-MM部分を抽出
            month = file.stem.split('-')[-1]
            months.append(month)
        
        return sorted(months)
