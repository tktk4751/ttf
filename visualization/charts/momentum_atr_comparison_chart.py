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
from indicators.alpha_momentum import AlphaMomentum
from indicators.alpha_atr import AlphaATR
from indicators.atr import ATR


class MomentumATRComparisonChart:
    """
    アルファモメンタム、ATR、アルファATRを表示・比較するローソク足チャートクラス
    - ローソク足とアルファモメンタム
    - 出来高
    - ATRとアルファATRの比較
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.alpha_momentum = None
        self.alpha_atr = None
        self.standard_atr = None
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
                            # アルファモメンタムのパラメータ
                            bb_max_period: int = 55,
                            bb_min_period: int = 13,
                            kc_max_period: int = 55,
                            kc_min_period: int = 13,
                            bb_max_mult: float = 2.0,
                            bb_min_mult: float = 1.0,
                            kc_max_mult: float = 3.0,
                            kc_min_mult: float = 1.0,
                            max_length: int = 34,
                            min_length: int = 8,
                            # アルファATRのパラメータ
                            max_atr_period: int = 55,
                            min_atr_period: int = 8,
                            # 標準ATRのパラメータ
                            atr_period: int = 14,
                            # 共通パラメータ
                            alma_offset: float = 0.85,
                            alma_sigma: float = 6
                           ) -> None:
        """
        インジケーターを計算する
        
        Args:
            er_period: 効率比の計算期間
            bb_max_period: ボリンジャーバンドの最大期間
            bb_min_period: ボリンジャーバンドの最小期間
            kc_max_period: ケルトナーチャネルの最大期間
            kc_min_period: ケルトナーチャネルの最小期間
            bb_max_mult: ボリンジャーバンドの最大乗数
            bb_min_mult: ボリンジャーバンドの最小乗数
            kc_max_mult: ケルトナーチャネルの最大乗数
            kc_min_mult: ケルトナーチャネルの最小乗数
            max_length: 最大期間（モメンタム計算用）
            min_length: 最小期間（モメンタム計算用）
            max_atr_period: アルファATRの最大期間
            min_atr_period: アルファATRの最小期間
            atr_period: 標準ATRの期間
            alma_offset: ALMAのオフセット
            alma_sigma: ALMAのシグマ
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        # アルファモメンタムの作成と計算
        print("アルファモメンタムを計算中...")
        self.alpha_momentum = AlphaMomentum(
            er_period=er_period,
            bb_max_period=bb_max_period,
            bb_min_period=bb_min_period,
            kc_max_period=kc_max_period,
            kc_min_period=kc_min_period,
            bb_max_mult=bb_max_mult,
            bb_min_mult=bb_min_mult,
            kc_max_mult=kc_max_mult,
            kc_min_mult=kc_min_mult,
            max_length=max_length,
            min_length=min_length,
            alma_offset=alma_offset,
            alma_sigma=alma_sigma
        )
        self.alpha_momentum.calculate(self.data)
        
        # アルファATRの作成と計算
        print("アルファATRを計算中...")
        self.alpha_atr = AlphaATR(
            er_period=er_period,
            max_atr_period=max_atr_period,
            min_atr_period=min_atr_period,
            alma_offset=alma_offset,
            alma_sigma=alma_sigma
        )
        self.alpha_atr.calculate(self.data)
        
        # 標準ATRの作成と計算
        print("標準ATRを計算中...")
        self.standard_atr = ATR(period=atr_period)
        self.standard_atr.calculate(self.data)
    
    def plot(self, 
            title: str = "モメンタムとATR比較チャート", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (14, 12),
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
            
        if any(i is None for i in [self.alpha_momentum, self.alpha_atr, self.standard_atr]):
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # 絞り込み後のインデックスがオリジナルデータのどの位置に対応するかを特定
        idx_map = [self.data.index.get_loc(idx) for idx in df.index]
        
        # アルファモメンタムの値を取得して追加
        momentum = self.alpha_momentum.get_momentum()
        df['alpha_momentum'] = momentum[idx_map]
        
        # スクイーズ状態を取得
        sqz_on, sqz_off, no_sqz = self.alpha_momentum.get_squeeze_states()
        df['sqz_on'] = sqz_on[idx_map]
        df['sqz_off'] = sqz_off[idx_map]
        df['no_sqz'] = no_sqz[idx_map]
        
        # アルファATRと標準ATRの値を取得
        df['alpha_atr'] = self.alpha_atr._values[idx_map]
        df['standard_atr'] = self.standard_atr._values[idx_map]
        
        # 効率比を取得
        er = self.alpha_atr.get_efficiency_ratio()
        df['er'] = er[idx_map]
        
        # mplfinanceでプロット用の設定
        # 1. メインチャート上のプロット
        main_plots = []
        
        # モメンタムのプロット設定（ヒストグラム形式）
        # モメンタムが正の場合は青、負の場合は赤で表示
        colors = np.where(df['alpha_momentum'] >= 0, 'blue', 'red')
        main_plots.append(mpf.make_addplot(df['alpha_momentum'], type='bar', color=colors, 
                                          width=0.7, alpha=0.5, panel=0, secondary_y=True))
        
        # 2. ATRパネル（アルファATRと標準ATRを比較）
        atr_panel_alpha = mpf.make_addplot(df['alpha_atr'], panel=1, 
                                     color='purple', width=1.5, 
                                     ylabel='ATR比較', secondary_y=False)
        
        atr_panel_standard = mpf.make_addplot(df['standard_atr'], panel=1, 
                                     color='green', width=1.0, 
                                     ylabel='ATR比較', secondary_y=False)
        
        # 3. 効率比（ER）パネル
        er_panel = mpf.make_addplot(df['er'], panel=2, color='blue', width=1.2, 
                                   ylabel='効率比(ER)', secondary_y=False)
        
        # mplfinanceの設定
        # パネル数に応じて設定を調整
        kwargs = dict(
            type='candle',
            figsize=figsize,
            title=title,
            style=style,
            datetime_format='%Y-%m-%d',
            xrotation=45,
            ylabel='価格',
            ylabel_lower='',
            returnfig=True
        )
        
        # 出来高と追加パネルの設定
        if show_volume:
            kwargs['volume'] = True
            kwargs['panel_ratios'] = (4, 1, 2, 1)  # メイン:出来高:ATR比較:ER
            # 出来高を表示する場合は、オシレーターのパネル番号を+1する
            atr_panel_alpha = mpf.make_addplot(df['alpha_atr'], panel=2, 
                                        color='purple', width=1.5, 
                                        ylabel='ATR比較', secondary_y=False)
            
            atr_panel_standard = mpf.make_addplot(df['standard_atr'], panel=2, 
                                        color='green', width=1.0, 
                                        ylabel='ATR比較', secondary_y=False)
            
            er_panel = mpf.make_addplot(df['er'], panel=3, color='blue', width=1.2, 
                                       ylabel='効率比(ER)', secondary_y=False)
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (4, 2, 1)  # メイン:ATR比較:ER
        
        # すべてのプロットを結合
        all_plots = main_plots + [atr_panel_alpha, atr_panel_standard, er_panel]
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加
        # メインチャートにモメンタムの凡例
        if show_volume:
            atr_panel_idx = 2
            er_panel_idx = 3
        else:
            atr_panel_idx = 1
            er_panel_idx = 2
        
        # ATRパネルに凡例を追加
        axes[atr_panel_idx].legend(['Alpha ATR', '標準ATR'], loc='upper left')
        
        # スクイーズ状態の可視化（メインチャート上部に色付きマーカー）
        # スクイーズ状態に応じて色を設定
        # スクイーズオン：赤、スクイーズオフ：緑、スクイーズなし：灰色
        for i in range(len(df)):
            if df['sqz_on'].iloc[i]:
                color = 'red'
                label = 'スクイーズオン' if i == 0 else None
            elif df['sqz_off'].iloc[i]:
                color = 'green'
                label = 'スクイーズオフ' if i == 0 else None
            else:
                color = 'gray'
                label = 'スクイーズなし' if i == 0 else None
            
            if color:
                # チャートの上部に小さな円マーカーを配置
                axes[0].scatter(i, df['high'].iloc[i] * 1.01, color=color, s=15, alpha=0.7, label=label)
        
        # スクイーズ状態の凡例を追加
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(handles, labels, loc='upper right', title='スクイーズ状態')
        
        self.fig = fig
        self.axes = axes
        
        # ATRとERパネルに参照線を追加
        # ATRパネル
        axes[atr_panel_idx].axhline(y=df['standard_atr'].mean(), color='green', linestyle='--', alpha=0.5)
        axes[atr_panel_idx].axhline(y=df['alpha_atr'].mean(), color='purple', linestyle='--', alpha=0.5)
        
        # ERパネル
        axes[er_panel_idx].axhline(y=0.7, color='black', linestyle='--', alpha=0.5)
        axes[er_panel_idx].axhline(y=0.3, color='black', linestyle='--', alpha=0.5)
        
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
                                   savefig: Optional[str] = None) -> 'MomentumATRComparisonChart':
        """
        設定ファイルからチャートを作成して描画する
        
        Args:
            config_path: 設定ファイルのパス
            start_date: 表示開始日（フォーマット: YYYY-MM-DD）
            end_date: 表示終了日（フォーマット: YYYY-MM-DD）
            savefig: 保存先のパス（指定しない場合は表示のみ）
            
        Returns:
            MomentumATRComparisonChartインスタンス
        """
        chart = MomentumATRComparisonChart()
        chart.load_data_from_config(config_path)
        chart.calculate_indicators()
        
        # 設定ファイルからデータを取得してタイトルを生成
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        binance_config = config.get('binance_data', {})
        symbol = binance_config.get('symbol', 'BTC')
        market_type = binance_config.get('market_type', 'spot')
        timeframe = binance_config.get('timeframe', '4h')
        
        title = f"{symbol}USDT ({market_type.capitalize()}) - {timeframe} - モメンタムとATR比較"
        
        chart.plot(title=title, start_date=start_date, end_date=end_date, savefig=savefig)
        
        return chart


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='モメンタムとATR比較チャートの表示')
    parser.add_argument('--config', default='config.yaml', help='設定ファイルパス')
    parser.add_argument('--start', help='表示開始日（YYYY-MM-DD）')
    parser.add_argument('--end', help='表示終了日（YYYY-MM-DD）')
    parser.add_argument('--save', help='チャート保存先パス')
    
    args = parser.parse_args()
    
    MomentumATRComparisonChart.create_and_plot_from_config(
        config_path=args.config,
        start_date=args.start,
        end_date=args.end,
        savefig=args.save
    )


if __name__ == "__main__":
    main() 