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
from indicators.hma import HMA


class HMAChart:
    """
    HMA（ハル移動平均線）を表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - HMAの中心線
    - トレンド方向のカラー表示
    - 動的期間の表示（動的期間モードの場合）
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.hma = None
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

    def calculate_indicators(self) -> None:
        """
        HMAを計算する
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\nHMAを計算中...")
        
        # HMAの計算
        self.hma = HMA(
            period=200,
            src_type='hlc3',
            use_dynamic_period=False,
            cycle_part=1.0,
            detector_type='absolute_ultimate',
            max_cycle=120,
            min_cycle=5,
            max_output=120,
            min_output=5,
            slope_index=1,
            range_threshold=0.005,
            lp_period=5,
            hp_period=120
        )
        
        # インジケーターの計算
        print("計算を実行します...")
        hma_result = self.hma.calculate(self.data)
        
        print("HMAの計算完了")
            
    def plot(self, 
            title: str = "HMAチャート", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (14, 12),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとHMAを描画する
        
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
            
        if self.hma is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # HMAの値を取得
        print("インジケーターデータを取得中...")
        hma_values = self.hma.get_values()
        trend_signals = self.hma.get_trend_signals()
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'hma': hma_values,
                'trend': trend_signals
            }
        )
        
        # 動的期間モードの場合、期間も取得
        if self.hma.use_dynamic_period:
            dynamic_periods = self.hma.get_dynamic_periods()
            full_df['period'] = dynamic_periods
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        
        # トレンド方向に基づくHMAの色分け
        df['hma_uptrend'] = np.where(df['trend'] == 1, df['hma'], np.nan)
        df['hma_downtrend'] = np.where(df['trend'] == -1, df['hma'], np.nan)
        df['hma_range'] = np.where(df['trend'] == 0, df['hma'], np.nan)
        
        # mplfinanceでプロット用の設定
        # 1. メインチャート上のプロット
        main_plots = []
        
        # HMAのプロット設定
        main_plots.append(mpf.make_addplot(df['hma_uptrend'], color='green', width=2, label='HMA (Up)'))
        main_plots.append(mpf.make_addplot(df['hma_downtrend'], color='red', width=2, label='HMA (Down)'))
        main_plots.append(mpf.make_addplot(df['hma_range'], color='gray', width=2, label='HMA (Range)'))
        
        # 2. オシレータープロット
        # トレンド方向パネル
        trend_panel = mpf.make_addplot(df['trend'], panel=1, color='orange', width=1.5, 
                                      ylabel='Trend Direction', secondary_y=False, label='Trend', type='line')
        
        # 動的期間モードの場合、期間パネルを追加
        if self.hma.use_dynamic_period:
            period_panel = mpf.make_addplot(df['period'], panel=2, color='blue', width=1.2,
                                          ylabel='Dynamic Period', secondary_y=False, label='Period')
        
        # mplfinanceの設定
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
            if self.hma.use_dynamic_period:
                kwargs['panel_ratios'] = (4, 1, 1, 1)  # メイン:出来高:トレンド:期間
                # 出来高を表示する場合は、オシレーターのパネル番号を+1する
                trend_panel = mpf.make_addplot(df['trend'], panel=2, color='orange', width=1.5, 
                                             ylabel='Trend Direction', secondary_y=False, label='Trend', type='line')
                period_panel = mpf.make_addplot(df['period'], panel=3, color='blue', width=1.2,
                                              ylabel='Dynamic Period', secondary_y=False, label='Period')
            else:
                kwargs['panel_ratios'] = (4, 1, 1)  # メイン:出来高:トレンド
                trend_panel = mpf.make_addplot(df['trend'], panel=2, color='orange', width=1.5, 
                                             ylabel='Trend Direction', secondary_y=False, label='Trend', type='line')
        else:
            kwargs['volume'] = False
            if self.hma.use_dynamic_period:
                kwargs['panel_ratios'] = (4, 1, 1)  # メイン:トレンド:期間
                period_panel = mpf.make_addplot(df['period'], panel=2, color='blue', width=1.2,
                                              ylabel='Dynamic Period', secondary_y=False, label='Period')
            else:
                kwargs['panel_ratios'] = (4, 1)  # メイン:トレンド
        
        # すべてのプロットを結合
        if self.hma.use_dynamic_period:
            all_plots = main_plots + [trend_panel, period_panel]
        else:
            all_plots = main_plots + [trend_panel]
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加
        axes[0].legend(['HMA (Up)', 'HMA (Down)', 'HMA (Range)'], 
                      loc='upper left')
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        if show_volume:
            # トレンド方向パネル
            if self.hma.use_dynamic_period:
                axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
                axes[2].axhline(y=1, color='green', linestyle='--', alpha=0.5)
                axes[2].axhline(y=-1, color='red', linestyle='--', alpha=0.5)
                
                # 期間パネル
                axes[3].axhline(y=self.hma.period, color='black', linestyle='--', alpha=0.5)
            else:
                axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
                axes[2].axhline(y=1, color='green', linestyle='--', alpha=0.5)
                axes[2].axhline(y=-1, color='red', linestyle='--', alpha=0.5)
        else:
            # トレンド方向パネル
            if self.hma.use_dynamic_period:
                axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
                axes[1].axhline(y=1, color='green', linestyle='--', alpha=0.5)
                axes[1].axhline(y=-1, color='red', linestyle='--', alpha=0.5)
                
                # 期間パネル
                axes[2].axhline(y=self.hma.period, color='black', linestyle='--', alpha=0.5)
            else:
                axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
                axes[1].axhline(y=1, color='green', linestyle='--', alpha=0.5)
                axes[1].axhline(y=-1, color='red', linestyle='--', alpha=0.5)
        
        # 統計情報の表示
        print(f"\n=== トレンド統計 ===")
        total_points = len(df[df['trend'] != 0])
        uptrend_points = len(df[df['trend'] == 1])
        downtrend_points = len(df[df['trend'] == -1])
        range_points = len(df[df['trend'] == 0])
        
        print(f"総データ点数: {total_points}")
        print(f"上昇トレンド: {uptrend_points} ({uptrend_points/total_points*100:.1f}%)")
        print(f"下降トレンド: {downtrend_points} ({downtrend_points/total_points*100:.1f}%)")
        print(f"レンジ相場: {range_points} ({range_points/total_points*100:.1f}%)")
        
        if self.hma.use_dynamic_period:
            valid_periods = df['period'].dropna()
            print(f"\n=== 動的期間統計 ===")
            print(f"平均期間: {valid_periods.mean():.1f}")
            print(f"最小期間: {valid_periods.min():.1f}")
            print(f"最大期間: {valid_periods.max():.1f}")
        
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
    parser = argparse.ArgumentParser(description='HMAチャートの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    args = parser.parse_args()
    
    # チャートを作成
    chart = HMAChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators()
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output
    )


if __name__ == "__main__":
    main() 