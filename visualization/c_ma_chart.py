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
from indicators.c_ma import CMA
from indicators.cycle_efficiency_ratio import CycleEfficiencyRatio


class CMAChart:
    """
    CMA（サイクル移動平均線）を表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - CMAの中心線
    - サイクル効率比（CER）の表示
    - ドミナントサイクルの表示
    - スムージング定数の表示
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.cma = None
        self.cer = None
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
        CMAとサイクル効率比を計算する
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\nCMAとサイクル効率比を計算中...")
        
        # サイクル効率比の計算
        self.cer = CycleEfficiencyRatio()
        
        # CMAの計算
        self.cma = CMA(
            detector_type='absolute_ultimate',
            cycle_part=1.0,
            lp_period=5,
            hp_period=120,
            max_cycle=120,
            min_cycle=5,
            max_output=120,
            min_output=5,
            fast_period=2,
            slow_period=120,
            src_type='hlc3'
        )
        
        # インジケーターの計算
        print("計算を実行します...")
        cer_result = self.cer.calculate(self.data)
        cma_result = self.cma.calculate(self.data, external_er=cer_result)
        
        print("CMAとサイクル効率比の計算完了")
            
    def plot(self, 
            title: str = "CMAチャート", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (14, 12),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとCMAを描画する
        
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
            
        if self.cma is None or self.cer is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # インジケーターの値を取得
        print("インジケーターデータを取得中...")
        cma_values = self.cma.get_values()
        cer_values = self.cma.get_efficiency_ratio()
        dc_values = self.cma.get_dc_values()
        sc_values = self.cma.get_smoothing_constants()
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'cma': cma_values,
                'cer': cer_values,
                'dc': dc_values,
                'sc': sc_values
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        
        # mplfinanceでプロット用の設定
        # 1. メインチャート上のプロット
        main_plots = []
        
        # CMAのプロット設定
        main_plots.append(mpf.make_addplot(df['cma'], color='blue', width=2, label='CMA'))
        
        # 2. オシレータープロット
        # サイクル効率比パネル
        cer_panel = mpf.make_addplot(df['cer'], panel=1, color='purple', width=1.2, 
                                    ylabel='Cycle Efficiency Ratio', secondary_y=False, label='CER')
        
        # ドミナントサイクルパネル
        dc_panel = mpf.make_addplot(df['dc'], panel=2, color='green', width=1.2,
                                   ylabel='Dominant Cycle', secondary_y=False, label='DC')
        
        # スムージング定数パネル
        sc_panel = mpf.make_addplot(df['sc'], panel=3, color='orange', width=1.2,
                                   ylabel='Smoothing Constant', secondary_y=False, label='SC')
        
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
            kwargs['panel_ratios'] = (4, 1, 1, 1, 1)  # メイン:出来高:CER:DC:SC
            # 出来高を表示する場合は、オシレーターのパネル番号を+1する
            cer_panel = mpf.make_addplot(df['cer'], panel=2, color='purple', width=1.2, 
                                        ylabel='Cycle Efficiency Ratio', secondary_y=False, label='CER')
            dc_panel = mpf.make_addplot(df['dc'], panel=3, color='green', width=1.2,
                                       ylabel='Dominant Cycle', secondary_y=False, label='DC')
            sc_panel = mpf.make_addplot(df['sc'], panel=4, color='orange', width=1.2,
                                       ylabel='Smoothing Constant', secondary_y=False, label='SC')
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (4, 1, 1, 1)  # メイン:CER:DC:SC
        
        # すべてのプロットを結合
        all_plots = main_plots + [cer_panel, dc_panel, sc_panel]
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加
        axes[0].legend(['CMA'], loc='upper left')
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        if show_volume:
            # サイクル効率比パネル
            axes[2].axhline(y=0.618, color='green', linestyle='--', alpha=0.5)
            axes[2].axhline(y=0.382, color='red', linestyle='--', alpha=0.5)
            
            # ドミナントサイクルパネル
            axes[3].axhline(y=self.cma.max_output, color='red', linestyle='--', alpha=0.5)
            axes[3].axhline(y=self.cma.min_output, color='green', linestyle='--', alpha=0.5)
            
            # スムージング定数パネル
            axes[4].axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
            axes[4].axhline(y=0.0, color='green', linestyle='--', alpha=0.5)
        else:
            # サイクル効率比パネル
            axes[1].axhline(y=0.618, color='green', linestyle='--', alpha=0.5)
            axes[1].axhline(y=0.382, color='red', linestyle='--', alpha=0.5)
            
            # ドミナントサイクルパネル
            axes[2].axhline(y=self.cma.max_output, color='red', linestyle='--', alpha=0.5)
            axes[2].axhline(y=self.cma.min_output, color='green', linestyle='--', alpha=0.5)
            
            # スムージング定数パネル
            axes[3].axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
            axes[3].axhline(y=0.0, color='green', linestyle='--', alpha=0.5)
        
        # 統計情報の表示
        print(f"\n=== サイクル効率比統計 ===")
        valid_cer = df['cer'].dropna()
        print(f"平均: {valid_cer.mean():.3f}")
        print(f"最小: {valid_cer.min():.3f}")
        print(f"最大: {valid_cer.max():.3f}")
        
        print(f"\n=== ドミナントサイクル統計 ===")
        valid_dc = df['dc'].dropna()
        print(f"平均期間: {valid_dc.mean():.1f}")
        print(f"最小期間: {valid_dc.min():.1f}")
        print(f"最大期間: {valid_dc.max():.1f}")
        
        print(f"\n=== スムージング定数統計 ===")
        valid_sc = df['sc'].dropna()
        print(f"平均: {valid_sc.mean():.3f}")
        print(f"最小: {valid_sc.min():.3f}")
        print(f"最大: {valid_sc.max():.3f}")
        
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
    parser = argparse.ArgumentParser(description='CMAチャートの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    args = parser.parse_args()
    
    # チャートを作成
    chart = CMAChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators()
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output
    )


if __name__ == "__main__":
    main() 