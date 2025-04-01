#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# 親ディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource
from indicators.kernel_ma import KernelMA


class CandlestickKernelMaChart:
    """
    ローソク足チャートとKernelMAを描画するクラス
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.kernel_ma = None
        self.fig = None
        self.ax1 = None  # ローソク足用のax
        self.ax2 = None  # 出来高用のax
        
    def load_data_from_config(self, config_path: str) -> pd.DataFrame:
        """
        設定ファイルからデータを読み込む
        
        Args:
            config_path: 設定ファイルのパス
            
        Returns:
            処理済みのデータフレーム
        """
        # 設定ファイルの読み込み
        with open(config_path, 'r', encoding='utf-8') as f:
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
        print("\nデータを読み込み・処理中...")
        raw_data = data_loader.load_data_from_config(config)
        processed_data = {
            symbol: data_processor.process(df)
            for symbol, df in raw_data.items()
        }
        
        # 最初のシンボルのデータを取得
        first_symbol = next(iter(processed_data))
        self.data = processed_data[first_symbol]
        
        print(f"データ読み込み完了: {first_symbol}")
        print(f"期間: {self.data.index.min()} → {self.data.index.max()}")
        print(f"データ数: {len(self.data)}")
        
        return self.data
    
    def calculate_kernel_ma(self, 
                          er_period: int = 21,
                          max_bandwidth: float = 10.0,
                          min_bandwidth: float = 2.0,
                          kernel_type: str = 'gaussian',
                          confidence_level: float = 0.95,
                          slope_period: int = 5) -> np.ndarray:
        """
        KernelMAを計算する
        
        Args:
            er_period: 効率比の計算期間
            max_bandwidth: バンド幅の最大値
            min_bandwidth: バンド幅の最小値
            kernel_type: カーネルの種類（'gaussian'または'epanechnikov'）
            confidence_level: 信頼区間のレベル
            slope_period: 傾きを計算する期間
            
        Returns:
            KernelMAの配列
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        # KernelMAの作成と計算
        self.kernel_ma = KernelMA(
            er_period=er_period,
            max_bandwidth=max_bandwidth,
            min_bandwidth=min_bandwidth,
            kernel_type=kernel_type,
            confidence_level=confidence_level,
            slope_period=slope_period
        )
        
        kernel_ma_values = self.kernel_ma.calculate(self.data)
        
        return kernel_ma_values
    
    def plot(self, 
            title: str = "ローソク足チャートとKernelMA", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            show_bands: bool = True,
            figsize: Tuple[int, int] = (12, 8),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとKernelMAを描画する
        
        Args:
            title: チャートのタイトル
            start_date: 表示開始日（フォーマット: YYYY-MM-DD）
            end_date: 表示終了日（フォーマット: YYYY-MM-DD）
            show_volume: 出来高を表示するか
            show_bands: 上下バンドを表示するか
            figsize: 図のサイズ
            style: mplfinanceのスタイル
            savefig: 保存先のパス（指定しない場合は表示のみ）
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        if self.kernel_ma is None:
            raise ValueError("KernelMAが計算されていません。calculate_kernel_ma()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # KernelMAをDataFrameに追加
        kernel_ma_values = self.kernel_ma._values
        
        # 期間絞り込み後のデータに対応するKernelMA値を取得
        # オリジナルデータのインデックスと絞り込み後のインデックスをマッピング
        if start_date or end_date:
            # 絞り込み後のインデックスがオリジナルデータのどの位置に対応するかを特定
            idx_map = [self.data.index.get_loc(idx) for idx in df.index]
            # 対応するKernelMA値を抽出
            df_kernel_ma = kernel_ma_values[idx_map]
            
            # バンドも同様に抽出
            if show_bands:
                upper_band, lower_band = self.kernel_ma.get_bands()
                df_upper_band = upper_band[idx_map]
                df_lower_band = lower_band[idx_map]
        else:
            df_kernel_ma = kernel_ma_values
            
            # バンドも取得
            if show_bands:
                upper_band, lower_band = self.kernel_ma.get_bands()
                df_upper_band = upper_band
                df_lower_band = lower_band
            
        df['kernel_ma'] = df_kernel_ma
        if show_bands:
            df['upper_band'] = df_upper_band
            df['lower_band'] = df_lower_band
        
        # mplfinanceでプロット
        ap = []  # 追加プロット用リスト
        
        # KernelMAのプロット設定
        ap.append(mpf.make_addplot(df['kernel_ma'], color='red', width=1.5, label='KernelMA'))
        
        # バンドのプロット設定
        if show_bands:
            ap.append(mpf.make_addplot(df['upper_band'], color='blue', width=0.8, linestyle='--', label='Upper Band'))
            ap.append(mpf.make_addplot(df['lower_band'], color='blue', width=0.8, linestyle='--', label='Lower Band'))
        
        # プロット
        kwargs = dict(
            type='candle',
            figsize=figsize,
            title=title,
            style=style,
            addplot=ap,
            volume=show_volume,
            panel_ratios=(4, 1) if show_volume else (1,),
            datetime_format='%Y-%m-%d',
            xrotation=45,
            returnfig=True
        )
        
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加
        if show_bands:
            axes[0].legend(['KernelMA', 'Upper Band', 'Lower Band'])
        else:
            axes[0].legend(['KernelMA'])
        
        self.fig = fig
        self.ax1 = axes[0]
        if show_volume:
            self.ax2 = axes[2]
        
        # 保存または表示
        if savefig:
            plt.savefig(savefig)
            print(f"チャートを保存しました: {savefig}")
        else:
            plt.tight_layout()
            plt.show()
    
    @staticmethod
    def create_and_plot_from_config(config_path: str, 
                                   start_date: Optional[str] = None,
                                   end_date: Optional[str] = None,
                                   show_bands: bool = True,
                                   savefig: Optional[str] = None) -> 'CandlestickKernelMaChart':
        """
        設定ファイルからチャートを作成して描画する
        
        Args:
            config_path: 設定ファイルのパス
            start_date: 表示開始日（フォーマット: YYYY-MM-DD）
            end_date: 表示終了日（フォーマット: YYYY-MM-DD）
            show_bands: 上下バンドを表示するか
            savefig: 保存先のパス（指定しない場合は表示のみ）
            
        Returns:
            CandlestickKernelMaChartインスタンス
        """
        chart = CandlestickKernelMaChart()
        chart.load_data_from_config(config_path)
        chart.calculate_kernel_ma()
        
        # 設定ファイルからデータを取得してタイトルを生成
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        binance_config = config.get('binance_data', {})
        symbol = binance_config.get('symbol', 'BTC')
        market_type = binance_config.get('market_type', 'spot')
        timeframe = binance_config.get('timeframe', '4h')
        
        title = f"{symbol}USDT ({market_type.capitalize()}) - {timeframe} - KernelMA Chart"
        
        chart.plot(title=title, start_date=start_date, end_date=end_date, show_bands=show_bands, savefig=savefig)
        
        return chart


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ローソク足チャートとKernelMAを描画')
    parser.add_argument('--config', default='config.yaml', help='設定ファイルパス')
    parser.add_argument('--start', help='表示開始日（YYYY-MM-DD）')
    parser.add_argument('--end', help='表示終了日（YYYY-MM-DD）')
    parser.add_argument('--no-bands', action='store_true', help='バンドを表示しない')
    parser.add_argument('--save', help='チャート保存先パス')
    
    args = parser.parse_args()
    
    CandlestickKernelMaChart.create_and_plot_from_config(
        config_path=args.config,
        start_date=args.start,
        end_date=args.end,
        show_bands=not args.no_bands,
        savefig=args.save
    )


if __name__ == "__main__":
    main() 