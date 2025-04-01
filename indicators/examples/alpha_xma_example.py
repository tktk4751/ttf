#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import yaml
from pathlib import Path

# インポートパスの設定
# プロジェクトのルートディレクトリを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)

# ルートディレクトリをパスに追加
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# データ取得用のクラスをインポート
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource

# インジケーターをインポート
from indicators.alpha_xma import AlphaXMA
from indicators.alpha_ma import AlphaMA


def main():
    """AlphaXMAとAlphaMAの使用例"""
    # 設定ファイルの読み込み
    config_path = Path(root_dir) / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # データの準備
    binance_config = config.get('binance_data', {})
    data_dir = binance_config.get('data_dir', 'data/binance')
    binance_data_source = BinanceDataSource(data_dir)
    
    # CSVデータソースはダミーとして渡す（Binanceデータソースのみを使用）
    dummy_csv_source = CSVDataSource("dummy")
    data_loader = DataLoader(
        data_source=dummy_csv_source,
        binance_data_source=binance_data_source
    )
    data_processor = DataProcessor()
    
    # データの読み込みと処理
    print("\nデータの読み込みと処理中...")
    raw_data = data_loader.load_data_from_config(config)
    processed_data = {
        symbol: data_processor.process(df)
        for symbol, df in raw_data.items()
    }
    
    # 最初の銘柄のデータを使用
    symbol = next(iter(processed_data))
    df = processed_data[symbol]
    print(f"使用するデータ: {symbol}")
    
    # データの状態を確認
    print(f"データフレームの形状: {df.shape}")
    print(f"データフレームのカラム: {df.columns.tolist()}")
    print(f"NaNの数: {df.isna().sum().sum()}")
    
    # AlphaMAとAlphaXMAを計算
    # AlphaMA - ハイパースムーサーを使用
    alpha_ma = AlphaMA(
        er_period=21,
        max_kama_period=144,
        min_kama_period=10,
        max_slow_period=89,
        min_slow_period=30,
        max_fast_period=13,
        min_fast_period=2,
        hyper_smooth_period=10  # ハイパースムーサーによる平滑化
    )
    
    # AlphaXMA - ALMAを使用
    alpha_xma = AlphaXMA(
        er_period=21,
        max_kama_period=144,
        min_kama_period=10,
        max_slow_period=89,
        min_slow_period=30,
        max_fast_period=13,
        min_fast_period=2,
        alma_period=9,         # ALMAの期間
        alma_offset=0.85,      # ALMAのオフセット
        alma_sigma=6           # ALMAのシグマ
    )
    
    # 計算実行
    print("\nAlphaMAとAlphaXMAを計算中...")
    try:
        # AlphaMAの計算（平滑化済みと生値）
        alpha_ma_values = alpha_ma.calculate(df)
        alpha_ma_raw_values = alpha_ma.get_raw_values()
        print(f"AlphaMA計算完了 - 配列の長さ: {len(alpha_ma_values)}, NaN数: {np.isnan(alpha_ma_values).sum()}")
        
        # AlphaXMAの計算（平滑化済みと生値）
        alpha_xma_values = alpha_xma.calculate(df)
        alpha_xma_raw_values = alpha_xma.get_raw_values()
        print(f"AlphaXMA計算完了 - 配列の長さ: {len(alpha_xma_values)}, NaN数: {np.isnan(alpha_xma_values).sum()}")
        
        # 結果をDataFrameに追加
        df['AlphaMA_HS'] = alpha_ma_values      # ハイパースムーサー平滑化済み
        df['AlphaMA_Raw'] = alpha_ma_raw_values # 生値
        df['AlphaXMA_ALMA'] = alpha_xma_values  # ALMA平滑化済み
        df['AlphaXMA_Raw'] = alpha_xma_raw_values # 生値
        
        # 効率比を取得
        er_values = alpha_xma.get_efficiency_ratio()
        df['ER'] = er_values
    except Exception as e:
        print(f"計算中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 先頭と末尾のデータを表示して確認
    print("\nデータ確認 (先頭5行):")
    print(df[['close', 'AlphaMA_HS', 'AlphaMA_Raw', 'AlphaXMA_ALMA', 'AlphaXMA_Raw']].head())
    print("\nデータ確認 (末尾5行):")
    print(df[['close', 'AlphaMA_HS', 'AlphaMA_Raw', 'AlphaXMA_ALMA', 'AlphaXMA_Raw']].tail())
    
    # NaNをフィルタリング
    not_na_mask = ~np.isnan(df['AlphaMA_HS']) & ~np.isnan(df['AlphaXMA_ALMA'])
    if not_na_mask.sum() == 0:
        print("警告: 有効なデータがありません。グラフは描画されません。")
        return
    
    df_plot = df[not_na_mask].copy()
    print(f"プロット用データの行数: {len(df_plot)}")
    
    # プロット
    print("\nグラフを作成中...")
    plt.figure(figsize=(16, 12))
    
    # プロット1: オリジナルMA
    plt.subplot(3, 1, 1)
    plt.plot(df_plot.index, df_plot['close'], label='Close', alpha=0.5, color='gray')
    plt.plot(df_plot.index, df_plot['AlphaMA_Raw'], label='AlphaMA Raw', color='blue', linewidth=1, alpha=0.6)
    plt.plot(df_plot.index, df_plot['AlphaXMA_Raw'], label='AlphaXMA Raw', color='red', linewidth=1, alpha=0.6)
    plt.title(f'{symbol} - 生のAlphaMA vs AlphaXMA（平滑化前）')
    plt.legend()
    plt.grid(True)
    
    # プロット2: 平滑化MA比較
    plt.subplot(3, 1, 2)
    plt.plot(df_plot.index, df_plot['close'], label='Close', alpha=0.5, color='gray')
    plt.plot(df_plot.index, df_plot['AlphaMA_HS'], label='AlphaMA (ハイパースムーサー)', color='green', linewidth=2)
    plt.plot(df_plot.index, df_plot['AlphaXMA_ALMA'], label='AlphaXMA (ALMA)', color='purple', linewidth=2)
    plt.title('平滑化した結果の比較')
    plt.legend()
    plt.grid(True)
    
    # プロット3: 効率比
    plt.subplot(3, 1, 3)
    plt.plot(df_plot.index, df_plot['ER'], label='Efficiency Ratio', color='purple')
    plt.axhline(y=0.618, color='r', linestyle='--', alpha=0.5, label='0.618')
    plt.axhline(y=0.382, color='g', linestyle='--', alpha=0.5, label='0.382')
    plt.fill_between(df_plot.index, df_plot['ER'], 0, where=(df_plot['ER'] >= 0.618), color='red', alpha=0.3, label='トレンド強')
    plt.fill_between(df_plot.index, df_plot['ER'], 0, where=(df_plot['ER'] <= 0.382), color='green', alpha=0.3, label='レンジ・ノイズ')
    plt.title('効率比 (Efficiency Ratio)')
    plt.legend()
    plt.grid(True)
    
    # 表示
    plt.tight_layout()
    plt.show()
    
    # ファイルに保存
    try:
        output_dir = Path(current_dir) / 'output'
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f'alpha_xma_vs_alpha_ma_{symbol}.png'
        plt.savefig(output_file)
        print(f"\nグラフを保存しました: {output_file}")
    except Exception as e:
        print(f"グラフの保存中にエラーが発生しました: {e}")
    
    print("\n完了しました。")


if __name__ == "__main__":
    main() 