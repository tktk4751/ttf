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
from strategies.implementations.kernel_ma_filter.signal_generator import KernelMAFilterSignalGenerator


class CandlestickKernelMaWithFilterChart:
    """
    ローソク足チャートとKernelMA、アルファフィルターを描画するクラス
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.signal_generator = None
        self.fig = None
        self.ax1 = None  # ローソク足用のax
        self.ax2 = None  # フィルター用のax
        self.ax3 = None  # 出来高用のax
        
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
    
    def calculate_signals(self, 
                          er_period: int = 21,
                          # カーネルMA用パラメータ
                          max_bandwidth: float = 10.0,
                          min_bandwidth: float = 2.0,
                          kernel_type: str = 'gaussian',
                          confidence_level: float = 0.95,
                          slope_period: int = 5,
                          slope_threshold: float = 0.0001,
                          # アルファフィルター用パラメータ
                          max_chop_period: int = 55,
                          min_chop_period: int = 8,
                          max_adx_period: int = 21,
                          min_adx_period: int = 5,
                          alma_offset: float = 0.85,
                          alma_sigma: float = 6,
                          filter_threshold: float = 0.5) -> None:
        """
        KernelMAとアルファフィルターを計算する
        
        Args:
            er_period: 効率比の計算期間
            max_bandwidth: バンド幅の最大値
            min_bandwidth: バンド幅の最小値
            kernel_type: カーネルの種類（'gaussian'または'epanechnikov'）
            confidence_level: 信頼区間のレベル
            slope_period: 傾きを計算する期間
            slope_threshold: トレンド判定の傾き閾値
            max_chop_period: アルファチョピネスの最大期間
            min_chop_period: アルファチョピネスの最小期間
            max_adx_period: アルファADXの最大期間
            min_adx_period: アルファADXの最小期間
            alma_offset: ALMAのオフセット
            alma_sigma: ALMAのシグマ
            filter_threshold: アルファフィルターのしきい値
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        # シグナル生成器の初期化
        self.signal_generator = KernelMAFilterSignalGenerator(
            er_period=er_period,
            max_bandwidth=max_bandwidth,
            min_bandwidth=min_bandwidth,
            kernel_type=kernel_type,
            confidence_level=confidence_level,
            slope_period=slope_period,
            slope_threshold=slope_threshold,
            max_chop_period=max_chop_period,
            min_chop_period=min_chop_period,
            max_adx_period=max_adx_period,
            min_adx_period=min_adx_period,
            alma_offset=alma_offset,
            alma_sigma=alma_sigma,
            filter_threshold=filter_threshold
        )
        
        # シグナルの計算
        self.signal_generator.calculate_signals(self.data)
    
    def plot(self, 
            title: str = "ローソク足チャートとKernelMA+フィルター", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            show_bands: bool = True,
            figsize: Tuple[int, int] = (12, 10),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとKernelMA、アルファフィルターを描画する
        
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
            
        if self.signal_generator is None:
            raise ValueError("シグナルが計算されていません。calculate_signals()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # 絞り込み後のインデックスがオリジナルデータのどの位置に対応するかを特定
        if start_date or end_date:
            idx_map = [self.data.index.get_loc(idx) for idx in df.index]
        else:
            idx_map = list(range(len(self.data)))
            
        # KernelMA値を取得
        kernel_ma_values = self.signal_generator.get_ma_values()
        df_kernel_ma = kernel_ma_values[idx_map]
        df['kernel_ma'] = df_kernel_ma
        
        # バンドを取得
        if show_bands:
            upper_band, lower_band = self.signal_generator.get_bands()
            df['upper_band'] = upper_band[idx_map]
            df['lower_band'] = lower_band[idx_map]
        
        # 方向シグナルを取得
        direction = self.signal_generator.get_direction_signals()
        df['direction'] = direction[idx_map]
        
        # フィルター値を取得
        filter_values = self.signal_generator.get_filter_values()
        df['filter'] = filter_values[idx_map]
        
        # エントリーシグナルを取得
        entry_signals = self.signal_generator.get_entry_signals(self.data)
        df['entry'] = entry_signals[idx_map]
        
        # mplfinanceでプロット
        ap = []  # 追加プロット用リスト
        
        # KernelMAのプロット設定
        ap.append(mpf.make_addplot(df['kernel_ma'], color='red', width=1.5, label='KernelMA'))
        
        # バンドのプロット設定
        if show_bands:
            ap.append(mpf.make_addplot(df['upper_band'], color='blue', width=0.8, linestyle='--', label='Upper Band'))
            ap.append(mpf.make_addplot(df['lower_band'], color='blue', width=0.8, linestyle='--', label='Lower Band'))
        
        # フィルター値のプロット設定（サブプロット）
        ap.append(mpf.make_addplot(df['filter'], panel=1, color='purple', width=1.0, label='Alpha Filter'))
        
        # フィルター閾値ラインの追加
        filter_threshold_line = np.ones(len(df)) * 0.5
        ap.append(mpf.make_addplot(filter_threshold_line, panel=1, color='gray', width=0.5, linestyle='--'))
        
        # エントリーシグナルのマーカー
        long_entries = np.where(df['entry'] == 1, df['low'] * 0.99, np.nan)
        short_entries = np.where(df['entry'] == -1, df['high'] * 1.01, np.nan)
        
        ap.append(mpf.make_addplot(long_entries, type='scatter', markersize=100, marker='^', color='lime', label='Long Entry'))
        ap.append(mpf.make_addplot(short_entries, type='scatter', markersize=100, marker='v', color='red', label='Short Entry'))
        
        # プロット
        panel_ratios = (4, 1, 1) if show_volume else (4, 1)
        kwargs = dict(
            type='candle',
            figsize=figsize,
            title=title,
            style=style,
            addplot=ap,
            volume=show_volume,
            panel_ratios=panel_ratios,
            datetime_format='%Y-%m-%d',
            xrotation=45,
            returnfig=True
        )
        
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加
        if show_bands:
            axes[0].legend(['KernelMA', 'Upper Band', 'Lower Band', 'Long Entry', 'Short Entry'])
        else:
            axes[0].legend(['KernelMA', 'Long Entry', 'Short Entry'])
            
        # フィルターパネルの凡例
        axes[1].legend(['Alpha Filter', 'Threshold'])
        axes[1].set_ylabel('Filter Value')
        
        self.fig = fig
        self.ax1 = axes[0]
        self.ax2 = axes[1]
        if show_volume:
            self.ax3 = axes[2]
        
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
                                   savefig: Optional[str] = None) -> 'CandlestickKernelMaWithFilterChart':
        """
        設定ファイルからチャートを作成して描画する
        
        Args:
            config_path: 設定ファイルのパス
            start_date: 表示開始日（フォーマット: YYYY-MM-DD）
            end_date: 表示終了日（フォーマット: YYYY-MM-DD）
            show_bands: 上下バンドを表示するか
            savefig: 保存先のパス（指定しない場合は表示のみ）
            
        Returns:
            CandlestickKernelMaWithFilterChartインスタンス
        """
        chart = CandlestickKernelMaWithFilterChart()
        chart.load_data_from_config(config_path)
        chart.calculate_signals()
        
        # 設定ファイルからデータを取得してタイトルを生成
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        binance_config = config.get('binance_data', {})
        symbol = binance_config.get('symbol', 'BTC')
        market_type = binance_config.get('market_type', 'spot')
        timeframe = binance_config.get('timeframe', '4h')
        
        title = f"{symbol}USDT ({market_type.capitalize()}) - {timeframe} - KernelMA + Filter Chart"
        
        chart.plot(title=title, start_date=start_date, end_date=end_date, show_bands=show_bands, savefig=savefig)
        
        return chart


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ローソク足チャートとKernelMA+フィルターを描画')
    parser.add_argument('--config', default='config.yaml', help='設定ファイルパス')
    parser.add_argument('--start', help='表示開始日（YYYY-MM-DD）')
    parser.add_argument('--end', help='表示終了日（YYYY-MM-DD）')
    parser.add_argument('--no-bands', action='store_true', help='バンドを表示しない')
    parser.add_argument('--save', help='チャート保存先パス')
    
    args = parser.parse_args()
    
    CandlestickKernelMaWithFilterChart.create_and_plot_from_config(
        config_path=args.config,
        start_date=args.start,
        end_date=args.end,
        show_bands=not args.no_bands,
        savefig=args.save
    )


if __name__ == "__main__":
    main() 