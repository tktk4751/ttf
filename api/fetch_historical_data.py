from binance_data_fetcher import BinanceDataFetcher
import os
from pathlib import Path
from typing import List, Dict
import ccxt

def fetch_market_data(market_type: str, timeframes: List[str], symbols: List[str], base_dir: Path):
    """指定された市場タイプと時間足でデータを取得または更新

    Args:
        market_type: 市場タイプ ('spot' または 'future')
        timeframes: 時間足のリスト
        symbols: 銘柄のリスト
        base_dir: 基本ディレクトリ
    """
    # 市場タイプの検証
    if market_type not in ['spot', 'future']:
        print(f"⚠️ 無効な市場タイプです: {market_type}. 'spot'または'future'を指定してください。")
        return
    
    try:
        fetcher = BinanceDataFetcher(market_type=market_type)
    except Exception as e:
        print(f"⚠️ データフェッチャーの初期化中にエラーが発生しました: {e}")
        return
    
    # 有効な時間足リストを取得
    valid_timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
    
    for symbol in symbols:
        print(f"\n🔄 {symbol}のデータ取得/更新を開始します...")
        symbol_format = f"{symbol}/USDT" if market_type == 'spot' else f"{symbol}USDT"
        
        # 銘柄の存在確認
        try:
            # 銘柄が存在するか確認（簡易的にチェック）
            test_data = fetcher.exchange.fetch_ohlcv(
                symbol=symbol_format,
                timeframe='1d',
                limit=1
            )
            if not test_data:
                print(f"⚠️ 銘柄が見つかりません: {symbol_format}. スキップします。")
                continue
        except ccxt.BaseError as e:
            print(f"⚠️ 銘柄検証中にエラーが発生しました: {symbol_format} - {e}. スキップします。")
            continue
        except Exception as e:
            print(f"⚠️ 予期しないエラーが発生しました: {e}. スキップします。")
            continue
            
        for timeframe in timeframes:
            # 時間足の検証
            if timeframe not in valid_timeframes:
                print(f"⚠️ 無効な時間足です: {timeframe}. スキップします。")
                continue
                
            print(f"\n📈 {market_type.upper()}市場の{timeframe}データを取得/更新します...")
            
            # データ保存用のディレクトリを作成
            try:
                data_dir = base_dir / symbol / market_type / timeframe
                data_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"⚠️ ディレクトリ作成中にエラーが発生しました: {e}. スキップします。")
                continue
            
            # 保存ファイル名を設定
            save_path = data_dir / 'historical_data.csv'
            
            try:
                # データを取得または更新して保存
                df = fetcher.update_historical_data(
                    symbol=symbol_format,
                    timeframe=timeframe,
                    save_path=save_path
                )
                
                if df.empty:
                    print(f"⚠️ {symbol}の{timeframe}データが取得できませんでした。スキップします。")
                    continue
                    
                print(f"✅ {symbol}の{timeframe}データを正常に取得/更新しました。")
                
            except ccxt.ExchangeNotAvailable as e:
                print(f"⚠️ 取引所が利用できません: {e}. しばらく待ってから再試行します。")
                # 取引所の一時的な問題の場合、少し長めに待機
                import time
                time.sleep(60)  # 1分待機
                continue
            except ccxt.DDoSProtection as e:
                print(f"⚠️ レート制限に達しました: {e}. しばらく待ってから再試行します。")
                import time
                time.sleep(30)  # 30秒待機
                continue
            except ccxt.AuthenticationError as e:
                print(f"⚠️ 認証エラーが発生しました: {e}. 処理を中止します。")
                return
            except ccxt.BaseError as e:
                print(f"⚠️ CCXT基本エラーが発生しました: {e}. スキップします。")
                continue
            except Exception as e:
                print(f"⚠️ {symbol}のデータ取得/更新中にエラーが発生しました: {e}. スキップします。")
                continue

def main():
    """メイン関数"""
    try:
        # 基本ディレクトリを設定
        base_dir = Path('data/binance')
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # 取得する時間足を定義
        timeframes = ['5m','15m', '30m', '1h', '2h','4h','6h', '8h', '12h']
        
        # 取得する銘柄を定義
        symbols = [
            'BTC', 'SOL', 'ETH', 'SUI', 'DOGE', 'PEPE', 'AVAX', 'ZRO', 'TAO', 'INJ', 'NEAR',
            'APT', 'RENDER', 'TON', 'XRP', 'BONK', 'SEI', 'AAVE', 'SHIB', 'DOT', 'JUP',
            'TIA', 'PENDLE', 'RUNE','ICP','GRT','LUNA','ZK','ASTR','OSMO','ROSE','AKT','OM','PYTH',
            'LINK','ATOM','ADA','ORDI','ARB','LTC','WIF','FTM','SPX','DYDX','SAND','TRX','DYM','OP'
        ]
        
        # 現物市場のデータを取得または更新
        print("\n📊 現物市場のデータ取得/更新を開始します...")
        try:
            fetch_market_data('spot', timeframes, symbols, base_dir)
        except Exception as e:
            print(f"⚠️ 現物市場のデータ取得中に重大なエラーが発生しました: {e}")
        
        # 先物市場のデータを取得または更新
        print("\n📊 先物市場のデータ取得/更新を開始します...")
        try:
            fetch_market_data('future', timeframes, symbols, base_dir)
        except Exception as e:
            print(f"⚠️ 先物市場のデータ取得中に重大なエラーが発生しました: {e}")
            
        print("\n✅ すべての処理が完了しました。")
        
    except Exception as e:
        print(f"⚠️ メイン処理中に予期しないエラーが発生しました: {e}")

if __name__ == '__main__':
    main() 