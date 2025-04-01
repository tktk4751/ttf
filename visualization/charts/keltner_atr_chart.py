#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource
from indicators.alpha_keltner_channel import AlphaKeltnerChannel
from indicators.alpha_atr import AlphaATR


class KeltnerATRChart:
    """
    アルファケルトナーチャネルとアルファATRを表示するローソク足チャートクラス
    - ローソク足とアルファケルトナーチャネル
    - 出来高
    - アルファATR（オシレーター）
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.alpha_keltner = None
        self.alpha_atr = None
        self.fig = None
        self.axes = None
    
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
    
    def calculate_indicators(self, 
                            er_period: int = 21,
                            max_kama_period: int = 55,
                            min_kama_period: int = 8,
                            max_atr_period: int = 55,
                            min_atr_period: int = 8,
                            max_multiplier: float = 3.0,
                            min_multiplier: float = 1.5,
                            use_independent_atr: bool = True,  # 独立したアルファATRを使用するかどうか
                            alma_offset: float = 0.85,
                            alma_sigma: float = 6
                           ) -> None:
        """
        インジケーターを計算する
        
        Args:
            er_period: 効率比の計算期間
            max_kama_period: KAMAの最大期間
            min_kama_period: KAMAの最小期間
            max_atr_period: ATRの最大期間
            min_atr_period: ATRの最小期間
            max_multiplier: ATR乗数の最大値
            min_multiplier: ATR乗数の最小値
            use_independent_atr: 独立したアルファATRを使用するかどうか
            alma_offset: ALMAのオフセット
            alma_sigma: ALMAのシグマ
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        # アルファケルトナーチャネルの作成と計算
        print("アルファケルトナーチャネルを計算中...")
        self.alpha_keltner = AlphaKeltnerChannel(
            er_period=er_period,
            max_kama_period=max_kama_period,
            min_kama_period=min_kama_period,
            max_atr_period=max_atr_period,
            min_atr_period=min_atr_period,
            max_multiplier=max_multiplier,
            min_multiplier=min_multiplier,
            alma_offset=alma_offset,
            alma_sigma=alma_sigma
        )
        self.alpha_keltner.calculate(self.data)
        
        # 独立したアルファATRの作成と計算（オプション）
        if use_independent_atr:
            print("アルファATRを計算中...")
            self.alpha_atr = AlphaATR(
                er_period=er_period,
                max_atr_period=max_atr_period,
                min_atr_period=min_atr_period,
                alma_offset=alma_offset,
                alma_sigma=alma_sigma
            )
            self.alpha_atr.calculate(self.data)
        else:
            # ケルトナーチャネルが内部で使用しているアルファATRを取得
            self.alpha_atr = self.alpha_keltner.alpha_atr
    
    def plot(self, 
            title: str = "アルファケルトナーチャネルとアルファATR", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (14, 10),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとインジケーターを描画する
        
        Args:
            title: チャートのタイトル
            start_date: 表示開始日（フォーマット: YYYY-MM-DD）
            end_date: 表示終了日（フォーマット: YYYY-MM-DD）
            show_volume: 出来高を表示するか
            figsize: 図のサイズ
            style: mplfinanceのスタイル
            savefig: 保存先のパス（指定しない場合は表示のみ）
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        if any(i is None for i in [self.alpha_keltner, self.alpha_atr]):
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # 絞り込み後のインデックスがオリジナルデータのどの位置に対応するかを特定
        idx_map = [self.data.index.get_loc(idx) for idx in df.index]
        
        # アルファケルトナーチャネルの値を取得して追加
        middle, upper, lower = self.alpha_keltner.get_bands()
        df['keltner_middle'] = middle[idx_map]
        df['keltner_upper'] = upper[idx_map]
        df['keltner_lower'] = lower[idx_map]
        
        # アルファATRの値を取得
        df['alpha_atr'] = self.alpha_atr._values[idx_map]
        
        # 効率比を取得
        er = self.alpha_keltner.get_efficiency_ratio()
        df['er'] = er[idx_map]
        
        # 動的乗数を取得
        dynamic_mult = self.alpha_keltner.get_dynamic_multiplier()
        df['dynamic_mult'] = dynamic_mult[idx_map]
        
        # mplfinanceでプロット用の設定
        # 1. メインチャート上のプロット
        main_plots = []
        
        # ケルトナーチャネルのプロット設定
        main_plots.append(mpf.make_addplot(df['keltner_middle'], color='gray', width=1.5, label='KC Middle'))
        main_plots.append(mpf.make_addplot(df['keltner_upper'], color='green', width=1, label='KC Upper'))
        main_plots.append(mpf.make_addplot(df['keltner_lower'], color='blue', width=1, label='KC Lower'))
        
        # 2. オシレータープロット
        # ATRとパネル配置を設定
        atr_panel = mpf.make_addplot(df['alpha_atr'], panel=1, color='purple', width=1.2, 
                                     ylabel='Alpha ATR', secondary_y=False, label='AlphaATR')
        
        er_panel = mpf.make_addplot(df['er'], panel=2, color='blue', width=1.2, 
                                    ylabel='ER', secondary_y=False, label='ER')
        
        # mplfinanceの設定
        # パネル数に応じて設定を調整
        kwargs = dict(
            type='candle',
            figsize=figsize,
            title=title,
            style=style,
            datetime_format='%Y-%m-%d',
            xrotation=45,
            returnfig=True
        )
        
        # 出来高と追加パネルの設定
        if show_volume:
            kwargs['volume'] = True
            kwargs['panel_ratios'] = (4, 1, 1, 1)  # メイン:出来高:ATR:ER
            # 出来高を表示する場合は、オシレーターのパネル番号を+1する
            atr_panel = mpf.make_addplot(df['alpha_atr'], panel=2, color='purple', width=1.2, 
                                         ylabel='Alpha ATR', secondary_y=False, label='AlphaATR')
            er_panel = mpf.make_addplot(df['er'], panel=3, color='blue', width=1.2, 
                                        ylabel='ER', secondary_y=False, label='ER')
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (4, 1, 1)  # メイン:ATR:ER
        
        # すべてのプロットを結合
        all_plots = main_plots + [atr_panel, er_panel]
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加（プロットと同じ順序で設定）
        axes[0].legend(['KC Middle', 'KC Upper', 'KC Lower'], loc='upper left')
        
        self.fig = fig
        self.axes = axes
        
        # ATRとERパネルに参照線を追加
        if show_volume:
            # ATRパネル
            axes[2].axhline(y=df['alpha_atr'].mean(), color='black', linestyle='--', alpha=0.5)
            # ERパネル
            axes[3].axhline(y=0.7, color='black', linestyle='--', alpha=0.5)
            axes[3].axhline(y=0.3, color='black', linestyle='--', alpha=0.5)
        else:
            # ATRパネル
            axes[1].axhline(y=df['alpha_atr'].mean(), color='black', linestyle='--', alpha=0.5)
            # ERパネル
            axes[2].axhline(y=0.7, color='black', linestyle='--', alpha=0.5)
            axes[2].axhline(y=0.3, color='black', linestyle='--', alpha=0.5)
        
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
                                   savefig: Optional[str] = None) -> 'KeltnerATRChart':
        """
        設定ファイルからチャートを作成して描画する
        
        Args:
            config_path: 設定ファイルのパス
            start_date: 表示開始日（フォーマット: YYYY-MM-DD）
            end_date: 表示終了日（フォーマット: YYYY-MM-DD）
            savefig: 保存先のパス（指定しない場合は表示のみ）
            
        Returns:
            KeltnerATRChartインスタンス
        """
        chart = KeltnerATRChart()
        chart.load_data_from_config(config_path)
        chart.calculate_indicators()
        
        # 設定ファイルからデータを取得してタイトルを生成
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        binance_config = config.get('binance_data', {})
        symbol = binance_config.get('symbol', 'BTC')
        market_type = binance_config.get('market_type', 'spot')
        timeframe = binance_config.get('timeframe', '4h')
        
        title = f"{symbol}USDT ({market_type.capitalize()}) - {timeframe} - アルファケルトナーチャネルとアルファATR"
        
        chart.plot(title=title, start_date=start_date, end_date=end_date, savefig=savefig)
        
        return chart


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='アルファケルトナーチャネルとアルファATRのチャート表示')
    parser.add_argument('--config', default='config.yaml', help='設定ファイルパス')
    parser.add_argument('--start', help='表示開始日（YYYY-MM-DD）')
    parser.add_argument('--end', help='表示終了日（YYYY-MM-DD）')
    parser.add_argument('--save', help='チャート保存先パス')
    
    args = parser.parse_args()
    
    KeltnerATRChart.create_and_plot_from_config(
        config_path=args.config,
        start_date=args.start,
        end_date=args.end,
        savefig=args.save
    )


if __name__ == "__main__":
    main() 