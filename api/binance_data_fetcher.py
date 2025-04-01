import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import List, Dict, Optional
import numpy as np
import os

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

    def update_historical_data(
        self,
        symbol: str = 'BTC/USDT',
        timeframe: str = '4h',
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        既存のCSVファイルがある場合は最新データのみを取得して追加し、
        ない場合は全期間のデータを取得する

        Args:
            symbol: 取得する銘柄（デフォルト: 'BTC/USDT'）
            timeframe: 時間足（デフォルト: '4h'）
            save_path: データを保存するパス（オプション）

        Returns:
            pd.DataFrame: 更新後の全データ
        """
        existing_df = pd.DataFrame()
        start_timestamp = None

        # 既存のファイルが存在するか確認
        if save_path and os.path.exists(save_path):
            try:
                print(f"📂 既存のデータファイルを読み込み中: {save_path}")
                existing_df = pd.read_csv(save_path, index_col=0, parse_dates=True)
                
                if not existing_df.empty:
                    # 最新のタイムスタンプを取得
                    latest_timestamp = existing_df.index[-1]
                    print(f"📅 既存データの最終日時: {latest_timestamp}")
                    
                    # ミリ秒単位のタイムスタンプに変換（+1ミリ秒して次のデータから取得）
                    start_timestamp = int(latest_timestamp.timestamp() * 1000) + 1
                    
                    # 現在の時刻と最新データの時刻を比較
                    now = datetime.now()
                    if (now - latest_timestamp).total_seconds() < 3600:  # 1時間以内の場合
                        print("✅ データは最新です。更新は必要ありません。")
                        return existing_df
            except Exception as e:
                print(f"⚠️ 既存ファイルの読み込み中にエラーが発生しました: {e}")
                print("🔄 全期間のデータを再取得します。")
                existing_df = pd.DataFrame()
                start_timestamp = None

        # 既存のデータがない場合は全期間を取得
        if existing_df.empty or not start_timestamp:
            return self.fetch_all_historical_data(symbol, timeframe, save_path)
            
        print(f"🔄 {symbol}の新規データを取得中...")
        
        # 現在のタイムスタンプ
        now = self.exchange.milliseconds()
        
        all_candles = []
        current_timestamp = start_timestamp
        
        while current_timestamp < now:
            try:
                print(f"📊 {datetime.fromtimestamp(current_timestamp/1000)} 以降のデータを取得中...")
                
                # ローソク足データを取得
                candles = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=current_timestamp,
                    limit=1000  # Binanceの制限に合わせる
                )
                
                if not candles:
                    print("📊 新しいデータはありません")
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
            print(f"📊 {symbol}の新規データはありません")
            return existing_df
            
        # 新規データをデータフレームに変換
        new_df = pd.DataFrame(
            all_candles,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        # タイムスタンプを日時形式に変換
        new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms')
        
        # 重複行を削除
        new_df = new_df.drop_duplicates(subset=['timestamp'])
        
        # 時間でソート
        new_df = new_df.sort_values('timestamp')
        
        # インデックスを設定
        new_df.set_index('timestamp', inplace=True)
        
        print(f"✅ 新規データ取得完了: {len(new_df)}件のデータを取得しました")
        if not new_df.empty:
            print(f"📅 新規データ期間: {new_df.index[0]} から {new_df.index[-1]}")
        
        # 既存データと新規データを結合
        combined_df = pd.concat([existing_df, new_df])
        
        # 重複を除去（念のため）
        combined_df = combined_df.loc[~combined_df.index.duplicated(keep='last')]
        
        # ソート
        combined_df = combined_df.sort_index()
        
        # データを保存
        if save_path:
            combined_df.to_csv(save_path, date_format='%Y-%m-%d %H:%M:%S')
            print(f"💾 更新データを保存しました: {save_path}")
            print(f"📊 総行数: {len(combined_df)}")
            print(f"📅 全期間: {combined_df.index[0]} から {combined_df.index[-1]}")
        
        return combined_df

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