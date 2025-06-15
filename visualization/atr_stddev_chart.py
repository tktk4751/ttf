#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# データ取得のための依存関係
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource

# インジケーター
from indicators.atr_stddev import ATRStdDev


class ATRStdDevChart:
    """
    ATR標準偏差 (ATR Standard Deviation) を表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - ATR値（サブパネル）
    - ATR標準偏差値（サブパネル）
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.atr_stddev = None
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
                            atr_period: int = 14,
                            stddev_period: int = 14
                           ) -> None:
        """
        ATR標準偏差を計算する
        
        Args:
            atr_period: ATR計算期間
            stddev_period: 標準偏差計算期間
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print(f"\nATR標準偏差を計算中...")
        print(f"パラメータ: ATR期間={atr_period}, 標準偏差期間={stddev_period}")
        
        # ATR標準偏差を計算
        self.atr_stddev = ATRStdDev(
            atr_period=atr_period,
            stddev_period=stddev_period
        )
        
        # ATR標準偏差の計算
        print("ATR標準偏差計算を実行します...")
        stddev_values = self.atr_stddev.calculate(self.data)
        atr_values = self.atr_stddev.get_atr_values()
        
        print(f"ATR標準偏差計算完了 - 配列長: {len(stddev_values)}")
        
        # NaN値のチェック
        atr_nan_count = np.isnan(atr_values).sum()
        atr_valid_count = len(atr_values) - atr_nan_count
        print(f"ATR値 - 有効値: {atr_valid_count}, NaN値: {atr_nan_count}")
        
        stddev_nan_count = np.isnan(stddev_values).sum()
        stddev_valid_count = len(stddev_values) - stddev_nan_count
        print(f"ATR標準偏差値 - 有効値: {stddev_valid_count}, NaN値: {stddev_nan_count}")
        
        if atr_valid_count > 0:
            print(f"ATR値範囲: {np.nanmin(atr_values):.6f} - {np.nanmax(atr_values):.6f}")
        
        if stddev_valid_count > 0:
            print(f"ATR標準偏差範囲: {np.nanmin(stddev_values):.6f} - {np.nanmax(stddev_values):.6f}")
        
        print("ATR標準偏差計算完了")
            
    def plot(self, 
            title: str = "ATR Standard Deviation", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (14, 12),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとATR標準偏差を描画する
        
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
            
        if self.atr_stddev is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # ATR標準偏差の値を取得
        print("ATR標準偏差データを取得中...")
        stddev_values = self.atr_stddev.calculate(self.data)
        atr_values = self.atr_stddev.get_atr_values()
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(index=self.data.index)
        full_df['atr'] = atr_values
        full_df['atr_stddev'] = stddev_values
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        print(f"ATRデータ確認 - NaN値: {df['atr'].isna().sum()}")
        print(f"ATR標準偏差データ確認 - NaN値: {df['atr_stddev'].isna().sum()}")
        
        # mplfinanceでプロット用の設定
        plots_list = []
        panel_count = 1
        
        # ATRパネル
        if not df['atr'].isna().all():
            atr_panel = mpf.make_addplot(df['atr'], panel=panel_count, color='orange', width=1.5, 
                                        ylabel='ATR', secondary_y=False)
            plots_list.append(atr_panel)
            panel_count += 1
        
        # ATR標準偏差パネル
        if not df['atr_stddev'].isna().all():
            stddev_panel = mpf.make_addplot(df['atr_stddev'], panel=panel_count, color='red', width=1.5, 
                                           ylabel='ATR StdDev', secondary_y=False)
            plots_list.append(stddev_panel)
            panel_count += 1
        
        # mplfinanceの設定
        kwargs = dict(
            type='candle',
            figsize=figsize,
            title=title,
            style=style,
            datetime_format='%Y-%m-%d',
            xrotation=45,
            returnfig=True,
            warn_too_much_data=len(df) * 2  # データ量の2倍に設定して警告を抑制
        )
        
        # 出来高と追加パネルの設定
        actual_panel_count = len(plots_list)  # 実際に作成されたパネル数
        
        if show_volume:
            kwargs['volume'] = True
            # 動的にパネル比率を設定
            panel_ratios = [4, 1]  # メイン、出来高
            panel_ratios.extend([1.5] * actual_panel_count)  # 追加パネル（少し大きめ）
            kwargs['panel_ratios'] = tuple(panel_ratios)
            
            # 出来高表示時はパネル番号を+1
            adjusted_plots = []
            for plot in plots_list:
                new_plot = mpf.make_addplot(
                    plot['data'], 
                    panel=plot['panel'] + 1, 
                    color=plot['color'], 
                    width=plot.get('width', 1),
                    ylabel=plot.get('ylabel', ''),
                    secondary_y=plot.get('secondary_y', False)
                )
                adjusted_plots.append(new_plot)
            plots_list = adjusted_plots
        else:
            kwargs['volume'] = False
            # 動的にパネル比率を設定
            panel_ratios = [4]  # メイン
            panel_ratios.extend([1.5] * actual_panel_count)  # 追加パネル（少し大きめ）
            if panel_ratios:
                kwargs['panel_ratios'] = tuple(panel_ratios)
        
        # すべてのプロットを結合
        if plots_list:
            kwargs['addplot'] = plots_list
        
        # プロット実行（エラーハンドリング）
        try:
            fig, axes = mpf.plot(df, **kwargs)
        except Exception as e:
            print(f"プロット中にエラーが発生しました: {e}")
            print("データの簡易表示を試行します...")
            # エラー時は最小限のプロットを試行
            simple_kwargs = dict(
                type='candle',
                figsize=figsize,
                title=title,
                style=style,
                datetime_format='%Y-%m-%d',
                xrotation=45,
                returnfig=True,
                volume=show_volume,
                warn_too_much_data=len(df) * 2
            )
            fig, axes = mpf.plot(df, **simple_kwargs)
            print("簡易表示でプロットしました。")
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        panel_offset = 1 if show_volume else 0
        panel_idx = 1 + panel_offset
        
        # ATRパネル
        if not df['atr'].isna().all() and panel_idx < len(axes):
            atr_mean = df['atr'].mean()
            if not np.isnan(atr_mean):
                axes[panel_idx].axhline(y=atr_mean, color='black', linestyle='-', alpha=0.3, 
                                      label=f'Mean ({atr_mean:.6f})')
                axes[panel_idx].legend(loc='upper right', fontsize=8)
            panel_idx += 1
        
        # ATR標準偏差パネル
        if not df['atr_stddev'].isna().all() and panel_idx < len(axes):
            stddev_mean = df['atr_stddev'].mean()
            if not np.isnan(stddev_mean):
                axes[panel_idx].axhline(y=stddev_mean, color='black', linestyle='-', alpha=0.3, 
                                      label=f'Mean ({stddev_mean:.6f})')
                axes[panel_idx].legend(loc='upper right', fontsize=8)
            
            # 標準偏差の閾値線を追加（高ボラティリティ変動の目安）
            stddev_std = df['atr_stddev'].std()
            if not np.isnan(stddev_std):
                high_volatility_line = stddev_mean + stddev_std
                axes[panel_idx].axhline(y=high_volatility_line, color='red', linestyle='--', alpha=0.5, 
                                      label=f'High Vol ({high_volatility_line:.6f})')
                axes[panel_idx].legend(loc='upper right', fontsize=8)
        
        # 統計情報の表示
        print(f"\n=== ATR標準偏差統計 ===")
        total_points = len(df[~df['atr_stddev'].isna()])
        print(f"総データ点数: {total_points}")
        
        valid_atr = df['atr'][~df['atr'].isna()]
        if len(valid_atr) > 0:
            print(f"ATR統計 - 平均: {valid_atr.mean():.6f}, 範囲: {valid_atr.min():.6f} - {valid_atr.max():.6f}")
        
        valid_stddev = df['atr_stddev'][~df['atr_stddev'].isna()]
        if len(valid_stddev) > 0:
            print(f"ATR標準偏差統計 - 平均: {valid_stddev.mean():.6f}, 範囲: {valid_stddev.min():.6f} - {valid_stddev.max():.6f}")
            
            # ボラティリティ分類
            stddev_75th = valid_stddev.quantile(0.75)
            stddev_25th = valid_stddev.quantile(0.25)
            print(f"ボラティリティ分類 - 高変動閾値(75%): {stddev_75th:.6f}, 低変動閾値(25%): {stddev_25th:.6f}")
        
        # 保存または表示
        if savefig:
            plt.savefig(savefig, dpi=150, bbox_inches='tight')
            print(f"チャートを保存しました: {savefig}")
        else:
            plt.tight_layout()
            plt.show()


def main():
    """メイン関数"""
    # コマンドライン引数を処理
    import argparse
    parser = argparse.ArgumentParser(description='ATR標準偏差の描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--atr-period', '-a', type=int, default=14, help='ATR計算期間')
    parser.add_argument('--stddev-period', '-d', type=int, default=14, help='標準偏差計算期間')
    parser.add_argument('--no-volume', action='store_true', help='出来高を非表示にする')
    parser.add_argument('--style', type=str, default='yahoo', help='mplfinanceスタイル')
    parser.add_argument('--width', type=int, default=14, help='チャート幅')
    parser.add_argument('--height', type=int, default=12, help='チャート高さ')
    args = parser.parse_args()
    
    # チャートを作成
    chart = ATRStdDevChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        atr_period=args.atr_period,
        stddev_period=args.stddev_period
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        show_volume=not args.no_volume,
        figsize=(args.width, args.height),
        style=args.style,
        savefig=args.output
    )


if __name__ == "__main__":
    main() 