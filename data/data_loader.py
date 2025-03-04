#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Protocol, Any
import pandas as pd
from logger import get_logger

class IDataSource(Protocol):
    """データソースのインターフェース"""
    def load_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """データを読み込む"""
        ...

    def get_available_symbols(self) -> List[str]:
        """利用可能な銘柄のリストを取得"""
        ...

    def get_available_timeframes(self, symbol: str) -> List[str]:
        """指定された銘柄で利用可能な時間足のリストを取得"""
        ...

class CSVDataSource(IDataSource):
    """CSVファイルからデータを読み込むクラス"""
    
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
        """CSVデータソースの初期化
        
        Args:
            data_dir: データディレクトリのパス
        """
        self.data_dir = data_dir
        self.logger = get_logger(__name__)

    def load_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """CSVファイルからデータを読み込む"""
        # データファイルのパスを構築
        data_path = Path(self.data_dir) / symbol / timeframe
        
        if not data_path.exists():
            raise FileNotFoundError(f"データディレクトリが見つかりません: {data_path}")
        
        # ディレクトリ内の全CSVファイルを取得
        csv_files = sorted([f for f in data_path.glob('*.csv')])
        
        if not csv_files:
            raise FileNotFoundError(f"CSVファイルが見つかりません: {data_path}")
        
        # 全てのCSVファイルを読み込んで結合
        dfs = []
        for file in csv_files:
            df = self._load_csv(file)
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
        columns_to_drop = ['open_time', 'close_time', 'ignore']
        combined_df.drop(columns=[col for col in columns_to_drop if col in combined_df.columns], inplace=True)
        
        # 重複を削除し、時系列でソート
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
        combined_df.sort_index(inplace=True)
        
        # 日付範囲でフィルタリング
        if start_date is not None:
            combined_df = combined_df[combined_df.index >= start_date]
        if end_date is not None:
            combined_df = combined_df[combined_df.index <= end_date]
        
        self.logger.info(
            f"データを読み込みました: {symbol}/{timeframe} "
            f"(行数: {len(combined_df)}, 期間: {combined_df.index.min()} - {combined_df.index.max()})"
        )
        
        return combined_df
    
    def _load_csv(self, file_path: Path) -> Optional[pd.DataFrame]:
        """CSVファイルを読み込む"""
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
        """利用可能な銘柄のリストを取得"""
        symbols = []
        for path in Path(self.data_dir).iterdir():
            if path.is_dir():
                symbols.append(path.name)
        return sorted(symbols)
    
    def get_available_timeframes(self, symbol: str) -> List[str]:
        """指定された銘柄で利用可能な時間足のリストを取得"""
        symbol_dir = Path(self.data_dir) / symbol
        timeframes = []
        
        if symbol_dir.exists():
            for path in symbol_dir.iterdir():
                if path.is_dir():
                    timeframes.append(path.name)
        
        return sorted(timeframes)

class DataLoader:
    """データ読み込みのファサードクラス"""
    
    def __init__(self, data_source: IDataSource, binance_data_source: Optional[IDataSource] = None):
        """
        Args:
            data_source: 従来のデータソース
            binance_data_source: Binanceデータソース（オプション）
        """
        self.data_source = data_source
        self.binance_data_source = binance_data_source
        self.logger = get_logger(__name__)
        self._cache: Dict[str, pd.DataFrame] = {}

    def load_market_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        use_cache: bool = True,
        market_type: Optional[str] = None  # Binanceデータ用
    ) -> pd.DataFrame:
        """市場データを読み込む
        
        Args:
            symbol: 銘柄名
            timeframe: 時間足
            start_date: 開始日
            end_date: 終了日
            use_cache: キャッシュを使用するかどうか
            market_type: 市場タイプ（"spot" or "future"）- Binanceデータ用
        
        Returns:
            pd.DataFrame: 読み込まれたデータ
        """
        cache_key = f"{symbol}_{timeframe}"
        if market_type:
            cache_key = f"{cache_key}_{market_type}"
        
        # キャッシュチェック
        if use_cache and cache_key in self._cache:
            df = self._cache[cache_key]
            if start_date is not None:
                df = df[df.index >= start_date]
            if end_date is not None:
                df = df[df.index <= end_date]
            return df
        
        # データの読み込み
        if market_type and self.binance_data_source:
            df = self.binance_data_source.load_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                market_type=market_type
            )
        else:
            df = self.data_source.load_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
        
        # キャッシュの更新
        if use_cache:
            self._cache[cache_key] = df
        
        return df

    def load_data_from_config(self, config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """設定ファイルからデータを読み込む
        
        Args:
            config: 設定データ
                {
                    'data': {
                        'data_dir': 'data',
                        'symbol': 'BTCUSDT',
                        'timeframe': '1h',
                        'start': '2024-01-01',
                        'end': '2024-12-31'
                    },
                    'binance_data': {
                        'enabled': true,
                        'symbol': 'BTC',
                        'market_type': 'future',
                        'timeframe': '4h',
                        'start': '2019-01-01',
                        'end': '2024-12-31'
                    }
                }
        
        Returns:
            Dict[str, pd.DataFrame]: 銘柄をキーとするデータフレームの辞書
        """
        result = {}
        
        # Binanceデータ設定を処理
        binance_config = config.get('binance_data', {})
        if binance_config and binance_config.get('enabled', False) and self.binance_data_source:
            symbol = binance_config.get('symbol', 'BTC')
            market_type = binance_config.get('market_type', 'spot')
            timeframe = binance_config.get('timeframe', '4h')
            start_date = binance_config.get('start')
            end_date = binance_config.get('end')
            
            # 日付文字列をdatetimeオブジェクトに変換
            start_dt = datetime.strptime(start_date, '%Y-%m-%d') if start_date else None
            end_dt = datetime.strptime(end_date, '%Y-%m-%d') if end_date else None
            
            # データの読み込み
            data = self.load_market_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_dt,
                end_date=end_dt,
                market_type=market_type
            )
            result[f"{symbol}_{market_type}"] = data
            
            self.logger.info(
                f"Binance設定からデータを読み込みました: {symbol}/{market_type}/{timeframe} "
                f"(期間: {start_date or '全期間'} - {end_date or '全期間'})"
            )
            return result
        
        # 従来のデータ設定を処理（Binanceデータが無効な場合のみ）
        data_config = config.get('data', {})
        if data_config:
            symbol = data_config.get('symbol', 'BTCUSDT')
            timeframe = data_config.get('timeframe', '1h')
            start_date = data_config.get('start')
            end_date = data_config.get('end')
            
            # 日付文字列をdatetimeオブジェクトに変換
            start_dt = datetime.strptime(start_date, '%Y-%m-%d') if start_date else None
            end_dt = datetime.strptime(end_date, '%Y-%m-%d') if end_date else None
            
            # データの読み込み
            data = self.load_market_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_dt,
                end_date=end_dt
            )
            result[symbol] = data
            
            self.logger.info(
                f"従来の設定からデータを読み込みました: {symbol}/{timeframe} "
                f"(期間: {start_date or '全期間'} - {end_date or '全期間'})"
            )
        
        return result

    def get_available_symbols(self) -> List[str]:
        """利用可能な銘柄のリストを取得"""
        return self.data_source.get_available_symbols()

    def get_available_timeframes(self, symbol: str) -> List[str]:
        """指定された銘柄で利用可能な時間足のリストを取得"""
        return self.data_source.get_available_timeframes(symbol)

    def clear_cache(self) -> None:
        """キャッシュをクリア"""
        self._cache.clear()
