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

# ハイパートレンドインデックス
from indicators.hyper_trend_index import HyperTrendIndex


class HyperTrendIndexChart:
    """
    ハイパートレンドインデックスを表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - ハイパートレンドインデックス値とミッドライン（メインパネル下の別パネル）
    - トレンド信号（1=緑、-1=赤）（別パネル）
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.hyper_trend = None
        self.fig = None
        self.axes = None
    
    def load_data_from_config(self, config_path: str, max_bars: int = 500) -> pd.DataFrame:
        """
        設定ファイルからデータを読み込む
        
        Args:
            config_path: 設定ファイルのパス
            max_bars: 最大データ数（デフォルト：500）
            
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
        full_data = processed_data[first_symbol]
        
        # 直近のmax_bars本に制限
        if len(full_data) > max_bars:
            self.data = full_data.tail(max_bars).copy()
        else:
            self.data = full_data.copy()
        
        print(f"データ読み込み完了: {first_symbol}")
        print(f"期間: {self.data.index.min()} → {self.data.index.max()}")
        print(f"データ数: {len(self.data)} (制限: {max_bars})")
        
        return self.data

    def calculate_indicators(self) -> None:
        """
        ハイパートレンドインデックスを計算する
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\nハイパートレンドインデックスを計算中...")
        
        # ハイパートレンドインデックスの計算
        self.hyper_trend = HyperTrendIndex(
            period=14,
            midline_period=100,
            src_type='hlc3',
            # 基本設定（依存関係の問題を避けるため、高度な機能を無効化）
            use_kalman_filter=True,
            use_dynamic_period=True,
            use_roofing_filter=True,
            use_smoothing=True
        )
        
        # インジケーターの計算
        print("計算を実行します...")
        self.hyper_trend_result = self.hyper_trend.calculate(self.data)
        
        print("ハイパートレンドインデックスの計算完了")
        print(f"有効値数: {np.sum(~np.isnan(self.hyper_trend_result.values))}/{len(self.hyper_trend_result.values)}")
            
    def plot(self, 
            title: str = "ハイパートレンドインデックスチャート", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (14, 14),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとハイパートレンドインデックスを描画する
        
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
            
        if self.hyper_trend is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # ハイパートレンドインデックスの値を取得
        print("インジケーターデータを取得中...")
        hyper_values = self.hyper_trend_result.values
        midline_values = self.hyper_trend_result.midline
        trend_signals = self.hyper_trend_result.trend_signal
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'hyper_trend': hyper_values,
                'midline': midline_values,
                'trend_signal': trend_signals
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        
        # トレンド信号に基づく色分け（1=緑、-1=赤）
        df['trend_green'] = np.where(df['trend_signal'] == 1, 1, np.nan)
        df['trend_red'] = np.where(df['trend_signal'] == -1, -1, np.nan)
        
        # mplfinanceでプロット用の設定
        # 追加パネルのプロット設定
        additional_plots = []
        
        if show_volume:
            # 出来高あり: パネル0=メイン、パネル1=出来高、パネル2=ハイパートレンド、パネル3=トレンド信号
            
            # ハイパートレンドインデックス値とミッドライン（パネル2）
            additional_plots.append(
                mpf.make_addplot(df['hyper_trend'], panel=2, color='blue', width=2, 
                               ylabel='Hyper Trend Index (0-1)', secondary_y=False, label='Hyper Trend')
            )
            additional_plots.append(
                mpf.make_addplot(df['midline'], panel=2, color='orange', width=1.5, 
                               linestyle='--', alpha=0.7, label='Midline')
            )
            
            # トレンド信号（パネル3）
            additional_plots.append(
                mpf.make_addplot(df['trend_green'], panel=3, color='green', width=3, 
                               ylabel='Trend Signal', secondary_y=False, label='Trend Up', type='line')
            )
            additional_plots.append(
                mpf.make_addplot(df['trend_red'], panel=3, color='red', width=3, 
                               label='Trend Down', type='line')
            )
            
            panel_ratios = (4, 1, 2, 1)  # メイン:出来高:ハイパートレンド:トレンド信号
            
        else:
            # 出来高なし: パネル0=メイン、パネル1=ハイパートレンド、パネル2=トレンド信号
            
            # ハイパートレンドインデックス値とミッドライン（パネル1）
            additional_plots.append(
                mpf.make_addplot(df['hyper_trend'], panel=1, color='blue', width=2, 
                               ylabel='Hyper Trend Index (0-1)', secondary_y=False, label='Hyper Trend')
            )
            additional_plots.append(
                mpf.make_addplot(df['midline'], panel=1, color='orange', width=1.5, 
                               linestyle='--', alpha=0.7, label='Midline')
            )
            
            # トレンド信号（パネル2）
            additional_plots.append(
                mpf.make_addplot(df['trend_green'], panel=2, color='green', width=3, 
                               ylabel='Trend Signal', secondary_y=False, label='Trend Up', type='line')
            )
            additional_plots.append(
                mpf.make_addplot(df['trend_red'], panel=2, color='red', width=3, 
                               label='Trend Down', type='line')
            )
            
            panel_ratios = (4, 2, 1)  # メイン:ハイパートレンド:トレンド信号
        
        # mplfinanceの設定
        kwargs = dict(
            type='candle',
            figsize=figsize,
            title=title,
            style=style,
            datetime_format='%Y-%m-%d',
            xrotation=45,
            returnfig=True,
            volume=show_volume,
            panel_ratios=panel_ratios,
            addplot=additional_plots
        )
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        if show_volume:
            # ハイパートレンドインデックスパネル（パネル2）に参照線
            axes[2].axhline(y=0.5, color='gray', linestyle='-', alpha=0.5, linewidth=1)
            axes[2].axhline(y=0.25, color='red', linestyle='--', alpha=0.5)
            axes[2].axhline(y=0.75, color='green', linestyle='--', alpha=0.5)
            axes[2].set_ylim(0, 1)
            
            # トレンド信号パネル（パネル3）に参照線
            axes[3].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[3].axhline(y=1, color='green', linestyle='--', alpha=0.3)
            axes[3].axhline(y=-1, color='red', linestyle='--', alpha=0.3)
            axes[3].set_ylim(-1.5, 1.5)
        else:
            # ハイパートレンドインデックスパネル（パネル1）に参照線
            axes[1].axhline(y=0.5, color='gray', linestyle='-', alpha=0.5, linewidth=1)
            axes[1].axhline(y=0.25, color='red', linestyle='--', alpha=0.5)
            axes[1].axhline(y=0.75, color='green', linestyle='--', alpha=0.5)
            axes[1].set_ylim(0, 1)
            
            # トレンド信号パネル（パネル2）に参照線
            axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[2].axhline(y=1, color='green', linestyle='--', alpha=0.3)
            axes[2].axhline(y=-1, color='red', linestyle='--', alpha=0.3)
            axes[2].set_ylim(-1.5, 1.5)
        
        # 統計情報の表示
        print(f"\n=== ハイパートレンドインデックス統計 ===")
        valid_hyper = df['hyper_trend'].dropna()
        valid_trend = df['trend_signal'].dropna()
        
        if len(valid_hyper) > 0:
            print(f"ハイパートレンド値範囲: {valid_hyper.min():.3f} - {valid_hyper.max():.3f}")
            print(f"ハイパートレンド平均: {valid_hyper.mean():.3f}")
            
        if len(valid_trend) > 0:
            total_signals = len(valid_trend)
            up_signals = len(valid_trend[valid_trend == 1])
            down_signals = len(valid_trend[valid_trend == -1])
            
            print(f"総信号数: {total_signals}")
            print(f"上昇トレンド信号: {up_signals} ({up_signals/total_signals*100:.1f}%)")
            print(f"下降トレンド信号: {down_signals} ({down_signals/total_signals*100:.1f}%)")
        
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
    parser = argparse.ArgumentParser(description='ハイパートレンドインデックスチャートの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--bars', '-b', type=int, default=1000, help='最大データ数 (デフォルト: 500)')
    args = parser.parse_args()
    
    # チャートを作成
    chart = HyperTrendIndexChart()
    chart.load_data_from_config(args.config, max_bars=args.bars)
    chart.calculate_indicators()
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output
    )


if __name__ == "__main__":
    main()