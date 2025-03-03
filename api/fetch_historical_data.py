from binance_data_fetcher import BinanceDataFetcher
import os
from pathlib import Path
from typing import List, Dict

def fetch_market_data(market_type: str, timeframes: List[str], symbols: List[str], base_dir: Path):
    """指定された市場タイプと時間足でデータを取得

    Args:
        market_type: 市場タイプ ('spot' または 'future')
        timeframes: 時間足のリスト
        symbols: 銘柄のリスト
        base_dir: 基本ディレクトリ
    """
    fetcher = BinanceDataFetcher(market_type=market_type)
    
    for symbol in symbols:
        print(f"\n🔄 {symbol}のデータ取得を開始します...")
        symbol_format = f"{symbol}/USDT" if market_type == 'spot' else f"{symbol}USDT"
        
        for timeframe in timeframes:
            print(f"\n📈 {market_type.upper()}市場の{timeframe}データを取得します...")
            
            # データ保存用のディレクトリを作成
            data_dir = base_dir / symbol / market_type / timeframe
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存ファイル名を設定
            save_path = data_dir / 'historical_data.csv'
            
            try:
                # データを取得して保存
                df = fetcher.fetch_all_historical_data(
                    symbol=symbol_format,
                    timeframe=timeframe,
                    save_path=save_path
                )
            except Exception as e:
                print(f"⚠️ {symbol}のデータ取得中にエラーが発生しました: {e}")
                continue

def main():
    """メイン関数"""
    # 基本ディレクトリを設定
    base_dir = Path('data/binance')
    
    # 取得する時間足を定義
    timeframes = ['15m', '30m', '1h', '4h', '8h', '12h']
    
    # 取得する銘柄を定義
    symbols = [
        'BTC', 'SOL', 'ETH', 'SUI', 'DOGE', 'PEPE', 'AVAX', 'ZRO', 'TAO', 'INJ', 'NEAR',
        'APT', 'RENDER', 'TON', 'XRP', 'BONK', 'SEI', 'AAVE', 'SHIB', 'DOT', 'JUP',
        'TIA', 'PENDLE', 'RUNE'
    ]
    
    # 現物市場のデータを取得
    print("\n📊 現物市場のデータ取得を開始します...")
    fetch_market_data('spot', timeframes, symbols, base_dir)
    
    # 先物市場のデータを取得
    print("\n📊 先物市場のデータ取得を開始します...")
    fetch_market_data('future', timeframes, symbols, base_dir)

if __name__ == '__main__':
    main() 