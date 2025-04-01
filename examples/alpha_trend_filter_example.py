#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
アルファトレンドフィルター（AlphaTrendFilter）の例 - シグモイド強調版
configからデータを読み込むバージョン
"""

import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

# 親ディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# インジケーターをインポート
from indicators import AlphaTrendFilter
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource


def create_custom_cmap():
    """カスタムカラーマップを作成する"""
    # 範囲相場（赤）からニュートラル（黄色）、トレンド相場（緑）へのグラデーション
    colors = [(0.9, 0.2, 0.2), (0.9, 0.9, 0.2), (0.2, 0.8, 0.2)]
    return LinearSegmentedColormap.from_list('custom_cmap', colors, N=100)


def generate_sample_data(length=200):
    """サンプルデータを生成する"""
    np.random.seed(42)  # 再現性のために乱数シードを固定
    
    dates = [datetime.now() + timedelta(days=i) for i in range(length)]
    
    # トレンドと範囲の相場を模擬
    base = 100.0
    x = np.linspace(0, 4*np.pi, length)
    trend = np.sin(x) * 10.0 + x * 2.0
    noise = np.random.normal(0, 1.0, length)
    
    close_prices = base + trend + noise
    high_prices = close_prices + np.abs(np.random.normal(0, 1.5, length))
    low_prices = close_prices - np.abs(np.random.normal(0, 1.5, length))
    open_prices = close_prices - np.random.normal(0, 1.0, length)
    
    return dates, open_prices, high_prices, low_prices, close_prices


def load_config():
    """設定ファイルを読み込む"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_data_from_config(config):
    """設定ファイルからデータを読み込む"""
    # データソースの設定
    binance_config = config.get('binance_data', {})
    data_dir = binance_config.get('data_dir', 'data/binance')
    binance_data_source = BinanceDataSource(data_dir)
    
    # CSVデータソースはダミーとして渡す
    dummy_csv_source = CSVDataSource("dummy")
    data_loader = DataLoader(
        data_source=dummy_csv_source,
        binance_data_source=binance_data_source
    )
    data_processor = DataProcessor()
    
    # データの読み込みと処理
    print("\nデータを読み込み中...")
    raw_data = data_loader.load_data_from_config(config)
    processed_data = {
        symbol: data_processor.process(df)
        for symbol, df in raw_data.items()
    }
    
    return processed_data


def main():
    """メイン関数"""
    # 設定ファイルの読み込み
    config = load_config()
    use_config_data = True  # 設定ファイルからのデータ読み込みを使用するかどうか
    
    if use_config_data:
        print("アルファトレンドフィルター（シグモイド強調版）の実行: configからデータを読み込み中...")
        # configからデータを読み込む
        processed_data = load_data_from_config(config)
        
        if not processed_data:
            print("データが読み込めませんでした。サンプルデータを使用します。")
            use_config_data = False
        else:
            # 最初の銘柄のデータを使用
            first_symbol = next(iter(processed_data))
            print(f"銘柄 {first_symbol} のデータを使用します")
            df = processed_data[first_symbol]
            
            # データフレームからOHLCデータを取得
            dates = df.index
            open_prices = df['open'].values
            high_prices = df['high'].values
            low_prices = df['low'].values
            close_prices = df['close'].values
    
    if not use_config_data:
        # サンプルデータの生成
        print("アルファトレンドフィルター（シグモイド強調版）のサンプルデータを生成中...")
        dates, open_prices, high_prices, low_prices, close_prices = generate_sample_data(length=200)
    
    print("アルファトレンドフィルターを計算中...")
    # アルファトレンドフィルターのインスタンス化と計算
    filter_indicator = AlphaTrendFilter(
        er_period=21,
        max_chop_period=21,
        min_chop_period=8,
        max_atr_period=21,
        min_atr_period=10,
        max_stddev_period=21,
        min_stddev_period=14,
        max_lookback_period=14,
        min_lookback_period=7,
        max_rms_window=14,
        min_rms_window=5,
        combination_weight=0.6  # トレンドインデックスの重み（0.6でトレンドインデックスを重視）
    )
    
    result = filter_indicator.calculate(open_prices, high_prices, low_prices, close_prices)
    
    # フィルター値の取得
    filter_values = result.values
    
    print("結果をプロット中...")
    # プロットの設定
    plt.figure(figsize=(15, 10))
    plt.style.use('dark_background')
    
    # カスタムカラーマップ
    cmap = create_custom_cmap()
    
    # サブプロットの設定
    ax1 = plt.subplot(2, 1, 1)  # 価格チャート
    ax2 = plt.subplot(2, 1, 2)  # アルファトレンドフィルター
    
    # 価格チャートのプロット
    ax1.plot(dates[:len(close_prices)], close_prices, color='white', linewidth=1.5, label='Close')
    ax1.set_title('Price Chart', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 市場状態の背景色
    for i, date in enumerate(dates[:len(filter_values)]):
        if i < len(filter_values):
            # 色の設定: 0（赤）→0.5（黄）→1（緑）
            color = cmap(filter_values[i])
            # 1日分の幅で色付け
            if i < len(dates) - 1 and i+1 < len(dates):
                width = (mdates.date2num(dates[i+1]) - mdates.date2num(date))
            else:
                width = 1
            rect = patches.Rectangle(
                (mdates.date2num(date), 0), width, 1,
                transform=ax1.get_xaxis_transform(), alpha=0.3,
                facecolor=color, edgecolor='none'
            )
            ax1.add_patch(rect)
    
    # アルファトレンドフィルターのプロット
    ax2.plot(dates[:len(filter_values)], filter_values, color='cyan', linewidth=2, label='Alpha Trend Filter (Sigmoid Enhanced)')
    ax2.axhline(y=0.7, color='lime', linestyle='--', alpha=0.7, label='Trend (0.7)')
    ax2.axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='Range (0.3)')
    ax2.set_title('Alpha Trend Filter (Sigmoid Enhanced)', fontsize=14)
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # X軸の日付フォーマットを設定
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    output_filename = 'alpha_trend_filter_config_data.png' if use_config_data else 'alpha_trend_filter_sigmoid.png'
    plt.savefig(output_filename, dpi=150)
    plt.show()
    
    print(f"処理完了！結果は '{output_filename}' に保存されました。")


if __name__ == "__main__":
    main()
