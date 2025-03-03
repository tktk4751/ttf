import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import List, Dict, Optional
import numpy as np

class BinanceDataFetcher:
    def __init__(self, market_type: str = 'spot'):
        """Binanceからデータを取得するクラスの初期化
        
        Args:
            market_type: 市場タイプ ('spot' または 'future')
        """
        self.market_type = market_type
        self.exchange = ccxt.binance({
            'enableRateLimit': True,  # レート制限を有効化
            'options': {
                'defaultType': market_type  # 市場タイプを指定
            }
        })

    def fetch_all_historical_data(
        self,
        symbol: str = 'BTC/USDT',
        timeframe: str = '4h',
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        指定された銘柄の全期間の履歴データを取得

        Args:
            symbol: 取得する銘柄（デフォルト: 'BTC/USDT'）
            timeframe: 時間足（デフォルト: '4h'）
            save_path: データを保存するパス（オプション）

        Returns:
            pd.DataFrame: 取得した価格データ
        """
        print(f"🔄 {symbol}の全期間データを取得中...")

        # 最も古いタイムスタンプを取得
        if self.market_type == 'spot':
            oldest_timestamp = self.exchange.parse8601('2017-08-17T00:00:00Z')  # Binanceスポット取引所の開始日
        else:
            oldest_timestamp = self.exchange.parse8601('2019-09-08T00:00:00Z')  # Binance先物取引所の開始日
        
        # 現在のタイムスタンプ
        now = self.exchange.milliseconds()
        
        all_candles = []
        current_timestamp = oldest_timestamp
        
        while current_timestamp < now:
            try:
                print(f"📊 {datetime.fromtimestamp(current_timestamp/1000)} のデータを取得中...")
                
                # ローソク足データを取得
                candles = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=current_timestamp,
                    limit=1000  # Binanceの制限に合わせる
                )
                
                if not candles:
                    break
                
                all_candles.extend(candles)
                
                # 次の取得開始時刻を設定
                current_timestamp = candles[-1][0] + 1
                
                # レート制限を考慮して待機
                time.sleep(self.exchange.rateLimit / 1000)
                
            except Exception as e:
                print(f"⚠️ エラーが発生しました: {e}")
                time.sleep(60)  # エラー時は1分待機
                continue
        
        if not all_candles:
            print(f"⚠️ データが取得できませんでした: {symbol}")
            return pd.DataFrame()
            
        # データフレームに変換
        df = pd.DataFrame(
            all_candles,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        # タイムスタンプを日時形式に変換
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # 重複行を削除
        df = df.drop_duplicates(subset=['timestamp'])
        
        # 時間でソート
        df = df.sort_values('timestamp')
        
        # インデックスを設定
        df.set_index('timestamp', inplace=True)
        
        print(f"✅ データ取得完了: {len(df)}件のデータを取得しました")
        print(f"📅 期間: {df.index[0]} から {df.index[-1]}")
        
        # データを保存
        if save_path:
            # インデックス（timestamp）を含めて保存
            df.to_csv(save_path, date_format='%Y-%m-%d %H:%M:%S')
            print(f"💾 データを保存しました: {save_path}")
            
            # 保存されたデータを確認
            saved_df = pd.read_csv(save_path, index_col=0, parse_dates=True)
            print(f"✅ 保存データの確認:")
            print(f"📊 行数: {len(saved_df)}")
            print(f"📅 保存データの期間: {saved_df.index[0]} から {saved_df.index[-1]}")
        
        return df

    def get_latest_data(
        self,
        symbol: str = 'BTC/USDT',
        timeframe: str = '4h',
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        最新のデータを取得

        Args:
            symbol: 取得する銘柄（デフォルト: 'BTC/USDT'）
            timeframe: 時間足（デフォルト: '4h'）
            limit: 取得するデータ数（デフォルト: 1000）

        Returns:
            pd.DataFrame: 取得した価格データ
        """
        try:
            candles = self.exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                limit=limit
            )
            
            if not candles:
                print(f"⚠️ データが取得できませんでした: {symbol}")
                return pd.DataFrame()
            
            df = pd.DataFrame(
                candles,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # タイムスタンプを日時形式に変換
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # 重複行を削除
            df = df.drop_duplicates(subset=['timestamp'])
            
            # 時間でソート
            df = df.sort_values('timestamp')
            
            # インデックスを設定
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            print(f"⚠️ エラーが発生しました: {e}")
            return pd.DataFrame() 