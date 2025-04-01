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
from indicators.alpha_vol_band import AlphaVolBand
from indicators.alpha_keltner_channel import AlphaKeltnerChannel


def main():
    """AlphaVolBandとAlphaKeltnerChannelの比較例"""
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
    
    # 共通のパラメータ
    er_period = 21
    max_kama_period = 144
    min_kama_period = 8
    max_period = 100
    min_period = 10
    max_multiplier = 6.0
    min_multiplier = 3.0
    
    # AlphaVolBandとAlphaKeltnerChannelを計算
    print("\nAlphaVolBandとAlphaKeltnerChannelを計算中...")
    try:
        # AlphaVolBandの計算（金額ベースのボラティリティを使用）
        alpha_vol_band = AlphaVolBand(
            er_period=er_period,
            max_kama_period=max_kama_period,
            min_kama_period=min_kama_period,
            max_vol_period=max_period,
            min_vol_period=min_period,
            smoothing_period=14,
            max_multiplier=max_multiplier,
            min_multiplier=min_multiplier,
            use_percent_vol=False  # 金額ベースのボラティリティを使用
        )
        
        # AlphaKeltnerChannelの計算
        alpha_keltner = AlphaKeltnerChannel(
            er_period=er_period,
            max_kama_period=max_kama_period,
            min_kama_period=min_kama_period,
            max_atr_period=max_period,
            min_atr_period=min_period,
            max_multiplier=max_multiplier,
            min_multiplier=min_multiplier
        )
        
        # 計算実行
        alpha_vol_band.calculate(df)
        alpha_keltner.calculate(df)
        
        # バンド取得
        vol_middle, vol_upper, vol_lower = alpha_vol_band.get_bands()
        keltner_middle, keltner_upper, keltner_lower = alpha_keltner.get_bands()
        
        # ボラティリティと効率比の取得
        percent_vol = alpha_vol_band.get_percent_volatility()
        absolute_vol = alpha_vol_band.get_absolute_volatility()  # 金額ベースのボラティリティを取得
        alpha_atr = alpha_keltner.get_alpha_atr()
        # ATRを%ベースに変換（近似値）
        close = df['close'].values
        percent_atr = np.zeros_like(alpha_atr)
        for i in range(len(alpha_atr)):
            if not np.isnan(alpha_atr[i]) and not np.isnan(close[i]) and close[i] > 0:
                percent_atr[i] = (alpha_atr[i] / close[i]) * 100
        
        # 効率比
        er_vol = alpha_vol_band.get_efficiency_ratio()
        er_keltner = alpha_keltner.get_efficiency_ratio()
        
        # 動的乗数
        mult_vol = alpha_vol_band.get_dynamic_multiplier()
        mult_keltner = alpha_keltner.get_dynamic_multiplier()
        
        # 結果をDataFrameに追加
        df['VolBand_Middle'] = vol_middle
        df['VolBand_Upper'] = vol_upper
        df['VolBand_Lower'] = vol_lower
        df['Keltner_Middle'] = keltner_middle
        df['Keltner_Upper'] = keltner_upper
        df['Keltner_Lower'] = keltner_lower
        df['ER_Vol'] = er_vol
        df['ER_Keltner'] = er_keltner
        df['Percent_Vol'] = percent_vol
        df['Absolute_Vol'] = absolute_vol  # 金額ベースのボラティリティを保存
        df['Percent_ATR'] = percent_atr
        df['ATR'] = alpha_atr  # ATRの生値も保存
        df['Mult_Vol'] = mult_vol
        df['Mult_Keltner'] = mult_keltner
        
    except Exception as e:
        print(f"計算中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # NaNをフィルタリング
    not_na_mask = (
        ~np.isnan(df['VolBand_Middle']) & 
        ~np.isnan(df['Keltner_Middle']) &
        ~np.isnan(df['ER_Vol'])
    )
    
    if not_na_mask.sum() == 0:
        print("警告: 有効なデータがありません。グラフは描画されません。")
        return
    
    df_plot = df[not_na_mask].copy()
    print(f"プロット用データの行数: {len(df_plot)}")
    
    # プロット
    print("\nグラフを作成中...")
    plt.figure(figsize=(16, 16))
    
    # プロット1: 価格とバンド
    plt.subplot(3, 1, 1)
    plt.plot(df_plot.index, df_plot['close'], label='Close', alpha=0.7, color='black')
    plt.plot(df_plot.index, df_plot['VolBand_Middle'], label='VolBand Middle', color='blue', linewidth=1)
    plt.plot(df_plot.index, df_plot['VolBand_Upper'], label='VolBand Upper', color='blue', linewidth=1, linestyle='--')
    plt.plot(df_plot.index, df_plot['VolBand_Lower'], label='VolBand Lower', color='blue', linewidth=1, linestyle='--')
    plt.fill_between(df_plot.index, df_plot['VolBand_Upper'], df_plot['VolBand_Lower'], color='blue', alpha=0.1)
    
    plt.plot(df_plot.index, df_plot['Keltner_Middle'], label='Keltner Middle', color='red', linewidth=1)
    plt.plot(df_plot.index, df_plot['Keltner_Upper'], label='Keltner Upper', color='red', linewidth=1, linestyle='--')
    plt.plot(df_plot.index, df_plot['Keltner_Lower'], label='Keltner Lower', color='red', linewidth=1, linestyle='--')
    plt.fill_between(df_plot.index, df_plot['Keltner_Upper'], df_plot['Keltner_Lower'], color='red', alpha=0.1)
    
    plt.title(f'{symbol} - AlphaVolBand(金額ベース) vs AlphaKeltnerChannel')
    plt.legend(loc='upper left')
    plt.grid(True)
    
    # プロット2: ボラティリティ比較
    plt.subplot(3, 1, 2)
    plt.plot(df_plot.index, df_plot['Percent_Vol'], label='Volatility (%)', color='blue', linewidth=1.5)
    plt.plot(df_plot.index, df_plot['Percent_ATR'], label='ATR (%)', color='red', linewidth=1.5)
    
    # 第2軸: 金額ベースのボラティリティ
    ax3 = plt.twinx()
    ax3.plot(df_plot.index, df_plot['Absolute_Vol'], label='Volatility (金額)', color='darkblue', linewidth=1, linestyle=':')
    ax3.plot(df_plot.index, df_plot['ATR'], label='ATR (金額)', color='darkred', linewidth=1, linestyle=':')
    ax3.set_ylabel('金額ベース', color='gray')
    ax3.tick_params(axis='y', labelcolor='gray')
    ax3.legend(loc='upper right')
    
    # バンド幅の計算と表示（メインの軸）
    volband_width = (df_plot['VolBand_Upper'] - df_plot['VolBand_Lower']) / df_plot['close'] * 100
    keltner_width = (df_plot['Keltner_Upper'] - df_plot['Keltner_Lower']) / df_plot['close'] * 100
    
    plt.plot(df_plot.index, volband_width, label='VolBand Width (%)', color='blue', linewidth=1, linestyle=':')
    plt.plot(df_plot.index, keltner_width, label='Keltner Width (%)', color='red', linewidth=1, linestyle=':')
    
    plt.title('ボラティリティ vs ATR (% と 金額ベース)')
    plt.legend(loc='upper left')
    plt.grid(True)
    
    # プロット3: 効率比と乗数
    plt.subplot(3, 1, 3)
    plt.plot(df_plot.index, df_plot['ER_Vol'], label='ER (VolBand)', color='blue', linewidth=1.5)
    plt.plot(df_plot.index, df_plot['ER_Keltner'], label='ER (Keltner)', color='red', linewidth=1.5)
    
    # 補助線
    plt.axhline(y=0.618, color='green', linestyle='--', alpha=0.5, label='0.618')
    plt.axhline(y=0.382, color='orange', linestyle='--', alpha=0.5, label='0.382')
    
    # 第2軸: 乗数
    ax2 = plt.twinx()
    ax2.plot(df_plot.index, df_plot['Mult_Vol'], label='乗数 (VolBand)', color='blue', linewidth=1, linestyle=':')
    ax2.plot(df_plot.index, df_plot['Mult_Keltner'], label='乗数 (Keltner)', color='red', linewidth=1, linestyle=':')
    ax2.set_ylabel('乗数', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.legend(loc='upper right')
    
    plt.title('効率比 (Efficiency Ratio) と 動的乗数')
    plt.legend(loc='upper left')
    plt.grid(True)
    
    # 表示
    plt.tight_layout()
    plt.show()
    
    # ファイルに保存
    try:
        output_dir = Path(current_dir) / 'output'
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f'alpha_vol_band_absolute_vs_keltner_{symbol}.png'
        plt.savefig(output_file)
        print(f"\nグラフを保存しました: {output_file}")
    except Exception as e:
        print(f"グラフの保存中にエラーが発生しました: {e}")
    
    print("\n完了しました。")


if __name__ == "__main__":
    main() 