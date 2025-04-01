#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
from pathlib import Path
from datetime import datetime, timedelta

# 親ディレクトリをパスに追加してインポートできるようにする
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 必要なクラスのインポート
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from indicators.z_hurst_exponent import ZHurstExponent
from data.binance_data_source import BinanceDataSource

# 日本語フォント設定
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['IPAGothic', 'IPAPGothic', 'VL Gothic', 'Noto Sans CJK JP', 'Takao']

# フォントが見つからない場合のフォールバック
import matplotlib.font_manager as fm
fonts = set([f.name for f in fm.fontManager.ttflist])
if not any(font in fonts for font in plt.rcParams['font.sans-serif']):
    # フォールバック: 日本語をASCIIで置き換える
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    
    # 警告を無効化
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    
    USE_ASCII_LABELS = True
else:
    USE_ASCII_LABELS = False


def load_data_from_config(config):
    """設定ファイルからデータを読み込む"""
    # データソースの設定
    binance_config = config.get('binance_data', {})
    data_dir = binance_config.get('data_dir', 'data/binance')
    symbol = binance_config.get('symbol', 'SOL')
    timeframe = binance_config.get('timeframe', '4h')
    market_type = binance_config.get('market_type', 'spot')
    
    print(f"\nデータを読み込み中... ({symbol}/{market_type}/{timeframe})")
    
    try:
        # BinanceDataSourceを使用してデータを直接読み込む
        binance_data_source = BinanceDataSource(data_dir)
        df = binance_data_source.load_data(
            symbol=symbol,
            timeframe=timeframe,
            market_type=market_type
        )
        
        # データの処理
        data_processor = DataProcessor()
        processed_df = data_processor.process(df)
        
        print(f"データを読み込みました: {len(processed_df)}行, 期間: {processed_df.index.min()} - {processed_df.index.max()}")
        
        return processed_df
        
    except Exception as e:
        print(f"データ読み込み中にエラーが発生しました: {str(e)}")
        print("ランダムデータを生成します...")
        
        # エラーが発生した場合はランダムデータを生成
        import numpy as np
        
        # 日付インデックスの作成（警告を避けるためにhを使用）
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        date_range = pd.date_range(start=start_date, end=end_date, freq='4h')
        
        # 価格データの生成
        np.random.seed(42)
        price_changes = np.random.normal(0, 1, len(date_range))
        prices = 100 + np.cumsum(price_changes)
        
        # データフレームの作成
        df = pd.DataFrame(index=date_range)
        df.loc[:, 'close'] = prices
        df.loc[:, 'high'] = df['close'] + np.random.uniform(0, 2, len(date_range))
        df.loc[:, 'low'] = df['close'] - np.random.uniform(0, 2, len(date_range))
        
        # 始値の計算（前日の終値から変動）
        df.loc[:, 'open'] = np.roll(df['close'], 1) + np.random.normal(0, 0.5, len(date_range))
        df.loc[df.index[0], 'open'] = df['close'].iloc[0] - 0.5
        
        # 出来高の生成
        df.loc[:, 'volume'] = np.random.uniform(1000, 5000, len(date_range))
        
        print(f"ランダムデータを生成しました: {len(df)}行")
        
        return df


def calculate_z_hurst_exponent(df):
    """Zハースト指数を計算する"""
    print("\nZハースト指数を計算中...")
    
    # Zハースト指数の設定
    z_hurst = ZHurstExponent(
        max_window_dc_cycle_part=0.75,
        max_window_dc_max_cycle=144,
        max_window_dc_min_cycle=8,
        max_window_dc_max_output=200,
        max_window_dc_min_output=50,
        
        min_window_dc_cycle_part=0.5,
        min_window_dc_max_cycle=55,
        min_window_dc_min_cycle=5,
        min_window_dc_max_output=50,
        min_window_dc_min_output=20,
        
        max_lag_ratio=0.5,
        min_lag_ratio=0.1,
        
        max_step=10,
        min_step=2,
        
        cycle_detector_type='dudi_dce',
        lp_period=10,
        hp_period=48,
        cycle_part=0.5,
        
        max_threshold=0.7,
        min_threshold=0.55
    )
    
    # Zハースト指数の計算
    z_hurst_values = z_hurst.calculate(df)
    
    # トレンド強度を取得（0.5からの距離、大きいほど強いトレンド）
    trend_strength = z_hurst.get_trend_strength()
    
    # 適応的なしきい値を取得
    adaptive_thresholds = z_hurst.get_adaptive_thresholds()
    
    # シグナルの取得（上昇トレンド、下降トレンド）
    uptrend_signals, downtrend_signals = z_hurst.get_signals()
    
    # 平均回帰シグナルの取得
    mean_reversion_signals = z_hurst.get_mean_reversion_signals()
    
    return {
        'values': z_hurst_values,
        'trend_strength': trend_strength,
        'adaptive_thresholds': adaptive_thresholds,
        'uptrend_signals': uptrend_signals,
        'downtrend_signals': downtrend_signals,
        'mean_reversion_signals': mean_reversion_signals
    }


def plot_chart_with_z_hurst(df, z_hurst_results, output_dir=None):
    """チャートとZハースト指数を描画する"""
    print("\nチャートとZハースト指数を描画中...")
    
    # 出力ディレクトリの設定
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    
    # 出力ディレクトリの作成
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # チャート用のデータを準備
    # 直近100バーのデータのみを表示
    last_n_bars = min(100, len(df))
    df_plot = df.tail(last_n_bars).copy()
    
    # mplfinance用にデータを準備
    df_mpf = df_plot.copy()
    df_mpf.index.name = 'Date'
    
    # Zハースト指数データを準備
    z_hurst_values = z_hurst_results['values'][-last_n_bars:]
    trend_strength = z_hurst_results['trend_strength'][-last_n_bars:]
    adaptive_thresholds = z_hurst_results['adaptive_thresholds'][-last_n_bars:]
    uptrend_signals = z_hurst_results['uptrend_signals'][-last_n_bars:]
    downtrend_signals = z_hurst_results['downtrend_signals'][-last_n_bars:]
    mean_reversion_signals = z_hurst_results['mean_reversion_signals'][-last_n_bars:]
    
    # チャートのスタイル設定
    mc = mpf.make_marketcolors(
        up='red',
        down='blue',
        edge='inherit',
        wick='inherit',
        volume='in'
    )
    style = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle='--',
        y_on_right=True
    )
    
    # サブプロットの設定 - この方法では2つのサブプロットをmpf.plotに渡す
    fig = plt.figure(figsize=(20, 16))
    
    # 3つのサブプロットを作成
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.1)
    ax1 = fig.add_subplot(gs[0])  # 価格チャート
    ax2 = fig.add_subplot(gs[1], sharex=ax1)  # Zハースト指数
    ax3 = fig.add_subplot(gs[2], sharex=ax1)  # トレンド強度
    
    # mpfでプロット
    mpf.plot(
        df_mpf,
        type='candle',
        style=style,
        ax=ax1,
        volume=False,
        warn_too_much_data=10000  # 警告を無効化
    )
    
    # サブチャートにZハースト指数をプロット
    # Zハースト指数プロット
    ax2.plot(df_plot.index, z_hurst_values, color='purple', linewidth=2, label='Zハースト指数')
    ax2.plot(df_plot.index, adaptive_thresholds, color='red', linestyle='--', linewidth=1, label='適応的しきい値')
    ax2.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='ランダムウォーク (0.5)')
    
    # トレンド強度プロット
    ax3.plot(df_plot.index, trend_strength, color='orange', linewidth=2, label='トレンド強度')
    ax3.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='強いトレンドしきい値 (0.2)')
    
    # シグナルマーカーを価格チャートに追加
    for i, signal in enumerate(uptrend_signals):
        if signal == 1:
            idx = df_plot.index[i]
            price = df_plot.loc[idx, 'low'] * 0.99
            ax1.plot(idx, price, '^', color='green', markersize=10, alpha=0.7)
    
    for i, signal in enumerate(downtrend_signals):
        if signal == 1:
            idx = df_plot.index[i]
            price = df_plot.loc[idx, 'high'] * 1.01
            ax1.plot(idx, price, 'v', color='red', markersize=10, alpha=0.7)
    
    for i, signal in enumerate(mean_reversion_signals):
        if signal == 1:
            idx = df_plot.index[i]
            price = df_plot.loc[idx, 'close']
            ax1.plot(idx, price, 'd', color='blue', markersize=10, alpha=0.7)
    
    # グラフタイトルと軸ラベル
    if not USE_ASCII_LABELS:
        ax1.set_title('価格チャートとZハースト指数シグナル', fontsize=14)
        ax2.set_title('Zハースト指数と適応的しきい値', fontsize=12)
        ax2.set_ylabel('指数値')
        ax3.set_title('トレンド強度 (0.5からの距離)', fontsize=12)
        ax3.set_ylabel('強度')
    else:
        ax1.set_title('Price Chart with Z-Hurst Exponent Signals', fontsize=14)
        ax2.set_title('Z-Hurst Exponent and Adaptive Threshold', fontsize=12)
        ax2.set_ylabel('Value')
        ax3.set_title('Trend Strength (Distance from 0.5)', fontsize=12)
        ax3.set_ylabel('Strength')
    
    # 凡例
    if not USE_ASCII_LABELS:
        ax1.legend(['上昇トレンド ▲', '下降トレンド ▼', '平均回帰 ◆'], loc='upper left')
    else:
        ax1.legend(['Uptrend ▲', 'Downtrend ▼', 'Mean Reversion ◆'], loc='upper left')
    ax2.legend(loc='upper left')
    ax3.legend(loc='upper left')
    
    # x軸の日付フォーマットを設定
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    # レイアウト調整
    plt.tight_layout()
    
    # グラフを保存
    output_file = os.path.join(output_dir, 'z_hurst_chart.png')
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"\nチャートを保存しました: {output_file}")
    
    # グラフを表示
    plt.show()


def main():
    """メイン関数"""
    # 設定ファイルの読み込み
    config_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # データの読み込み
    df = load_data_from_config(config)
    
    # Zハースト指数の計算
    z_hurst_results = calculate_z_hurst_exponent(df)
    
    # チャートの描画
    plot_chart_with_z_hurst(df, z_hurst_results)
    
    print("\n処理が完了しました。")


if __name__ == '__main__':
    main() 