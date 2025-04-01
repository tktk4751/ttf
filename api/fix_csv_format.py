#!/usr/bin/env python3
"""
既存のCSVファイルの数値表示形式を修正するスクリプト
科学的表記法(1.412e-05など)を通常の小数点表記に変換します
"""

import pandas as pd
import os
from pathlib import Path
from glob import glob
import argparse

def fix_csv_format(csv_path: str):
    """
    CSVファイルの数値表示形式を修正する

    Args:
        csv_path: 修正するCSVファイルのパス
    """
    try:
        # CSVファイルを読み込む
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        
        if df.empty:
            print(f"⚠️ {csv_path} は空のファイルです。スキップします。")
            return False
        
        # バックアップを作成
        backup_path = f"{csv_path}.bak"
        if not os.path.exists(backup_path):
            os.rename(csv_path, backup_path)
            print(f"📁 バックアップを作成しました: {backup_path}")
        
        # 小数点以下10桁まで表示するフォーマットで保存
        df.to_csv(csv_path, date_format='%Y-%m-%d %H:%M:%S', float_format='%.10f')
        print(f"✅ {csv_path} の数値表示形式を修正しました")
        return True
        
    except Exception as e:
        print(f"❌ {csv_path} の処理中にエラーが発生しました: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='CSVファイルの数値表示形式を修正する')
    parser.add_argument('--dir', type=str, help='処理するディレクトリ（デフォルト: data/binance）', default='data/binance')
    parser.add_argument('--symbol', type=str, help='特定の銘柄のみ処理する場合に指定')
    
    args = parser.parse_args()
    base_dir = Path(args.dir)
    
    if not base_dir.exists():
        print(f"❌ 指定されたディレクトリ {base_dir} は存在しません")
        return
    
    # 処理するCSVファイルを検索
    if args.symbol:
        # 特定の銘柄のみ処理
        symbols = [args.symbol]
    else:
        # すべての銘柄を処理
        symbols = [d.name for d in base_dir.iterdir() if d.is_dir()]
    
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    for symbol in symbols:
        print(f"\n🔄 {symbol} の処理を開始します...")
        
        # 銘柄のディレクトリ内のすべてのCSVファイルを検索
        csv_files = list(base_dir.glob(f"{symbol}/**/historical_data.csv"))
        
        if not csv_files:
            print(f"⚠️ {symbol} のCSVファイルが見つかりませんでした")
            continue
        
        for csv_file in csv_files:
            print(f"📄 {csv_file} を処理中...")
            result = fix_csv_format(str(csv_file))
            
            if result:
                processed_count += 1
            else:
                skipped_count += 1
    
    print(f"\n📊 処理結果:")
    print(f"✅ 修正完了: {processed_count}ファイル")
    print(f"⚠️ スキップ: {skipped_count}ファイル")
    print(f"❌ エラー: {error_count}ファイル")

if __name__ == "__main__":
    main() 