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
from indicators.alpha_ma import AlphaMA
from indicators.alpha_trend import AlphaTrend
from indicators.alpha_choppiness import AlphaChoppiness
from indicators.alpha_filter import AlphaFilter


class MultiIndicatorChart:
    """
    複数のインジケーターを表示するローソク足チャートクラス
    - ローソク足と出来高
    - アルファMA
    - アルファトレンド
    - アルファチョピネス（オシレーター）
    - アルファフィルター（オシレーター）
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.alpha_ma = None
        self.alpha_trend = None
        self.alpha_choppiness = None
        self.alpha_filter = None
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
                            # AlphaMA パラメータ
                            max_kama_period: int = 144,
                            min_kama_period: int = 10,
                            max_slow_period: int = 89,
                            min_slow_period: int = 30,
                            max_fast_period: int = 13,
                            min_fast_period: int = 2,
                            # アルファトレンド パラメータ
                            max_percentile_length: int = 55,
                            min_percentile_length: int = 13,
                            max_atr_period: int = 89,
                            min_atr_period: int = 13,
                            max_multiplier: float = 3.0,
                            min_multiplier: float = 1.0,
                            # アルファチョピネス パラメータ
                            max_chop_period: int = 55,
                            min_chop_period: int = 8,
                            # 共通パラメータ
                            alma_offset: float = 0.85,
                            alma_sigma: float = 6
                           ) -> None:
        """
        各種インジケーターを計算する
        
        Args:
            er_period: 効率比の計算期間
            max_kama_period: KAMAピリオドの最大値
            min_kama_period: KAMAピリオドの最小値
            max_slow_period: 遅い移動平均の最大期間
            min_slow_period: 遅い移動平均の最小期間
            max_fast_period: 速い移動平均の最大期間
            min_fast_period: 速い移動平均の最小期間
            max_percentile_length: パーセンタイル計算の最大期間
            min_percentile_length: パーセンタイル計算の最小期間
            max_atr_period: AlphaATR期間の最大値
            min_atr_period: AlphaATR期間の最小値
            max_multiplier: ATR乗数の最大値
            min_multiplier: ATR乗数の最小値
            max_chop_period: チョピネス期間の最大値
            min_chop_period: チョピネス期間の最小値
            alma_offset: ALMAオフセット
            alma_sigma: ALMAシグマ
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        # AlphaMAの作成と計算
        print("アルファMAを計算中...")
        self.alpha_ma = AlphaMA(
            er_period=er_period,
            max_kama_period=max_kama_period,
            min_kama_period=min_kama_period,
            max_slow_period=max_slow_period,
            min_slow_period=min_slow_period,
            max_fast_period=max_fast_period,
            min_fast_period=min_fast_period
        )
        self.alpha_ma.calculate(self.data)
        
        # アルファトレンドの作成と計算
        print("アルファトレンドを計算中...")
        self.alpha_trend = AlphaTrend(
            er_period=er_period,
            max_percentile_length=max_percentile_length,
            min_percentile_length=min_percentile_length,
            max_atr_period=max_atr_period,
            min_atr_period=min_atr_period,
            max_multiplier=max_multiplier,
            min_multiplier=min_multiplier,
            alma_offset=alma_offset,
            alma_sigma=alma_sigma
        )
        self.alpha_trend.calculate(self.data)
        
        # アルファチョピネスの作成と計算
        print("アルファチョピネスを計算中...")
        self.alpha_choppiness = AlphaChoppiness(
            er_period=er_period,
            max_chop_period=max_chop_period,
            min_chop_period=min_chop_period,
            alma_period=10,
            alma_offset=alma_offset,
            alma_sigma=alma_sigma
        )
        self.alpha_choppiness.calculate(self.data)
        
        # アルファフィルターの作成と計算
        print("アルファフィルターを計算中...")
        self.alpha_filter = AlphaFilter(
            er_period=er_period,
            max_chop_period=max_chop_period,
            min_chop_period=min_chop_period,
            max_adx_period=21,
            min_adx_period=5,
            alma_period=10,
            alma_offset=alma_offset,
            alma_sigma=alma_sigma
        )
        self.alpha_filter.calculate(self.data)
    
    def plot(self, 
            title: str = "マルチインジケーターチャート", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (14, 12),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートと各種インジケーターを描画する
        
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
            
        if any(i is None for i in [self.alpha_ma, self.alpha_trend, self.alpha_choppiness, self.alpha_filter]):
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # 絞り込み後のインデックスがオリジナルデータのどの位置に対応するかを特定
        idx_map = [self.data.index.get_loc(idx) for idx in df.index]
        
        # AlphaMA値を取得して追加
        df['alpha_ma'] = self.alpha_ma._values[idx_map]
        
        # アルファトレンドの上下バンドを取得
        upper_band, lower_band = self.alpha_trend.get_bands()
        df['trend_upper'] = upper_band[idx_map]
        df['trend_lower'] = lower_band[idx_map]
        
        # アルファトレンドのトレンド方向を取得
        trend = self.alpha_trend.get_trend()
        df['trend'] = trend[idx_map]
        
        # アルファチョピネス値を取得
        df['choppiness'] = self.alpha_choppiness._values[idx_map]
        
        # アルファフィルター値を取得
        df['filter'] = self.alpha_filter._values[idx_map]
        
        # mplfinanceでプロット用の設定
        # 1. メインチャート上のプロット
        main_plots = []
        
        # AlphaMAのプロット設定
        main_plots.append(mpf.make_addplot(df['alpha_ma'], color='red', width=1.5, label='AlphaMA'))
        
        # アルファトレンドのバンドプロット設定
        main_plots.append(mpf.make_addplot(df['trend_upper'], color='green', width=1, label='AT Upper'))
        main_plots.append(mpf.make_addplot(df['trend_lower'], color='red', width=1, label='AT Lower'))
        
        # 2. オシレータープロット
        # アルファチョピネスとアルファフィルターのプロット設定
        chop_panel = mpf.make_addplot(df['choppiness'], panel=1, color='purple', width=1.2, 
                                      ylabel='Alpha Chop', secondary_y=False, label='ChopIndex')
        
        filter_panel = mpf.make_addplot(df['filter'], panel=2, color='blue', width=1.2, 
                                        ylabel='Alpha Filter', secondary_y=False, label='Filter')
        
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
        
        # 出来高表示の設定
        if show_volume:
            kwargs['volume'] = True
            kwargs['panel_ratios'] = (4, 1, 1, 1)  # メイン:出来高:チョピネス:フィルター
            # 出来高を表示する場合は、オシレーターのパネル番号を+1する
            chop_panel = mpf.make_addplot(df['choppiness'], panel=2, color='purple', width=1.2, 
                                          ylabel='Alpha Chop', secondary_y=False, label='ChopIndex')
            filter_panel = mpf.make_addplot(df['filter'], panel=3, color='blue', width=1.2, 
                                            ylabel='Alpha Filter', secondary_y=False, label='Filter')
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (4, 1, 1)  # メイン:チョピネス:フィルター
        
        # すべてのプロットを結合
        all_plots = main_plots + [chop_panel, filter_panel]
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加
        axes[0].legend(['AlphaMA', 'AT Upper', 'AT Lower'])
        
        # トレンド方向に基づいて背景色を変更
        if 'trend' in df.columns:
            # トレンド方向が変わる位置を特定
            trend_changes = np.where(df['trend'].diff() != 0)[0]
            
            # 最初のトレンド方向
            current_trend = df['trend'].iloc[0]
            
            # 背景色の設定
            if current_trend == 1:  # 上昇トレンド
                fillcolor = 'lightgreen'
                alpha = 0.15
            else:  # 下降トレンド
                fillcolor = 'lightcoral'
                alpha = 0.15
                
            # 最初の期間をプロット
            if len(trend_changes) > 0:
                start_idx = 0
                end_idx = trend_changes[0]
                axes[0].axvspan(df.index[start_idx], df.index[end_idx], 
                                facecolor=fillcolor, alpha=alpha)
            
            # 残りの期間をプロット
            for i in range(len(trend_changes)):
                start_idx = trend_changes[i]
                
                # 次のトレンド方向を設定
                current_trend = df['trend'].iloc[start_idx]
                
                if current_trend == 1:  # 上昇トレンド
                    fillcolor = 'lightgreen'
                else:  # 下降トレンド
                    fillcolor = 'lightcoral'
                
                # 終了位置を決定
                if i < len(trend_changes) - 1:
                    end_idx = trend_changes[i+1]
                else:
                    end_idx = len(df) - 1
                
                # 背景を塗る
                axes[0].axvspan(df.index[start_idx], df.index[end_idx], 
                                facecolor=fillcolor, alpha=alpha)
        
        self.fig = fig
        self.axes = axes
        
        # チョピネスとフィルターのパネルに水平線を追加
        if len(axes) > 2:  # チョピネスパネルが存在する場合
            # チョピネスパネルに0.5と0.7のラインを追加
            chop_ax_idx = 2 if show_volume else 1
            axes[chop_ax_idx].axhline(y=0.7, color='black', linestyle='--', alpha=0.5)
            axes[chop_ax_idx].axhline(y=0.3, color='black', linestyle='--', alpha=0.5)
            
            # フィルターパネルに0.5と0.7のラインを追加
            filter_ax_idx = 3 if show_volume else 2
            axes[filter_ax_idx].axhline(y=0.7, color='black', linestyle='--', alpha=0.5)
            axes[filter_ax_idx].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
            axes[filter_ax_idx].axhline(y=0.3, color='black', linestyle='--', alpha=0.5)
        
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
                                   savefig: Optional[str] = None) -> 'MultiIndicatorChart':
        """
        設定ファイルからチャートを作成して描画する
        
        Args:
            config_path: 設定ファイルのパス
            start_date: 表示開始日（フォーマット: YYYY-MM-DD）
            end_date: 表示終了日（フォーマット: YYYY-MM-DD）
            savefig: 保存先のパス（指定しない場合は表示のみ）
            
        Returns:
            MultiIndicatorChartインスタンス
        """
        chart = MultiIndicatorChart()
        chart.load_data_from_config(config_path)
        chart.calculate_indicators()
        
        # 設定ファイルからデータを取得してタイトルを生成
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        binance_config = config.get('binance_data', {})
        symbol = binance_config.get('symbol', 'BTC')
        market_type = binance_config.get('market_type', 'spot')
        timeframe = binance_config.get('timeframe', '4h')
        
        title = f"{symbol}USDT ({market_type.capitalize()}) - {timeframe} - マルチインジケーターチャート"
        
        chart.plot(title=title, start_date=start_date, end_date=end_date, savefig=savefig)
        
        return chart


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='複数インジケーター付きローソク足チャートの表示')
    parser.add_argument('--config', default='config.yaml', help='設定ファイルパス')
    parser.add_argument('--start', help='表示開始日（YYYY-MM-DD）')
    parser.add_argument('--end', help='表示終了日（YYYY-MM-DD）')
    parser.add_argument('--save', help='チャート保存先パス')
    
    args = parser.parse_args()
    
    MultiIndicatorChart.create_and_plot_from_config(
        config_path=args.config,
        start_date=args.start,
        end_date=args.end,
        savefig=args.save
    )


if __name__ == "__main__":
    main() 