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
from indicators.c_atr import CATR


class CATRChart:
    """
    CATR（Cycle Average True Range）を表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - CATR（%ベースと金額ベース）
    - 動的ATR期間
    - True Range
    - 価格にATRバンドを重ねた表示
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.catr = None
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
                            # CATRパラメータ
                            detector_type: str = 'phac_e',
                            cycle_part: float = 0.5,
                            lp_period: int = 13,
                            hp_period: int = 50,
                            max_cycle: int = 55,
                            min_cycle: int = 5,
                            max_output: int = 34,
                            min_output: int = 5,
                            smoother_type: str = 'alma',
                            src_type: str = 'hlc3',
                            use_kalman_filter: bool = False,
                            kalman_measurement_noise: float = 1.0,
                            kalman_process_noise: float = 0.01,
                            kalman_n_states: int = 5
                           ) -> None:
        """
        CATR（Cycle Average True Range）を計算する
        
        Args:
            detector_type: ドミナントサイクル検出器タイプ
            cycle_part: サイクル部分の倍率
            lp_period: ローパスフィルター期間
            hp_period: ハイパスフィルター期間
            max_cycle: 最大サイクル期間
            min_cycle: 最小サイクル期間
            max_output: 最大出力値
            min_output: 最小出力値
            smoother_type: 平滑化アルゴリズムのタイプ（'alma'または'hyper'）
            src_type: 価格ソース
            use_kalman_filter: カルマンフィルター使用有無
            kalman_measurement_noise: カルマン測定ノイズ
            kalman_process_noise: カルマンプロセスノイズ
            kalman_n_states: カルマン状態数
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print(f"\nCATR（Cycle Average True Range）を計算中...")
        print(f"パラメータ: 検出器={detector_type}, スムーザー={smoother_type}, ソース={src_type}, カルマン={use_kalman_filter}")
        
        # CATRを計算
        self.catr = CATR(
            detector_type=detector_type,
            cycle_part=cycle_part,
            lp_period=lp_period,
            hp_period=hp_period,
            max_cycle=max_cycle,
            min_cycle=min_cycle,
            max_output=max_output,
            min_output=min_output,
            smoother_type=smoother_type,
            src_type=src_type,
            use_kalman_filter=use_kalman_filter,
            kalman_measurement_noise=kalman_measurement_noise,
            kalman_process_noise=kalman_process_noise,
            kalman_n_states=kalman_n_states
        )
        
        # CATRの計算
        print("CATR計算を実行します...")
        catr_values = self.catr.calculate(self.data)
        print(f"CATR計算完了 - 配列長: {len(catr_values)}")
        
        # NaN値のチェック
        nan_count = np.isnan(catr_values).sum()
        valid_count = len(catr_values) - nan_count
        print(f"CATR値 - 有効値: {valid_count}, NaN値: {nan_count}")
        
        if valid_count > 0:
            print(f"CATR値範囲: {np.nanmin(catr_values)*100:.4f}% - {np.nanmax(catr_values)*100:.4f}%")
        
        print("CATR計算完了")
            
    def plot(self, 
            title: str = "CATR（Cycle Average True Range）", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            show_atr_bands: bool = True,
            atr_multiplier: float = 2.0,
            figsize: Tuple[int, int] = (14, 12),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとCATRを描画する
        
        Args:
            title: チャートのタイトル
            start_date: 表示開始日（フォーマット: YYYY-MM-DD）
            end_date: 表示終了日（フォーマット: YYYY-MM-DD）
            show_volume: 出来高を表示するか
            show_atr_bands: ATRバンドを価格チャートに表示するか
            atr_multiplier: ATRバンドの乗数
            figsize: 図のサイズ
            style: mplfinanceのスタイル
            savefig: 保存先のパス（指定しない場合は表示のみ）
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        if self.catr is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # CATRの値を取得
        print("CATRデータを取得中...")
        catr_values = self.catr.calculate(self.data)  # %ベース
        percent_atr = self.catr.get_percent_atr()  # %ベース（100倍済み）
        absolute_atr = self.catr.get_absolute_atr()  # 金額ベース
        true_range = self.catr.get_true_range()
        atr_period = self.catr.get_atr_period()
        dc_values = self.catr.get_dc_values()
        
        # ATRバンドを計算（価格 ± ATR * multiplier）
        close_prices = df['close']
        atr_upper = close_prices + (absolute_atr * atr_multiplier)
        atr_lower = close_prices - (absolute_atr * atr_multiplier)
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'catr_percent': percent_atr,  # %ベース（100倍済み）
                'catr_absolute': absolute_atr,  # 金額ベース
                'true_range': true_range,
                'atr_period': atr_period,
                'dominant_cycle': dc_values,
                'atr_upper': atr_upper,
                'atr_lower': atr_lower
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        print(f"CATRデータ確認 - NaN値: {df['catr_percent'].isna().sum()}")
        
        # データ量が多い場合は絞り込み
        if len(df) > 2000:
            print(f"データ量が多いため、最新の2000データポイントに絞り込みます（元: {len(df)}行）")
            df = df.tail(2000)
            print(f"絞り込み後のデータ行数: {len(df)}")
        
        # NaN値を含む行を出力（最初の5行のみ）
        nan_rows = df[df['catr_percent'].isna()]
        if not nan_rows.empty:
            print(f"NaN値を含む行: {len(nan_rows)}行")
            if len(nan_rows) <= 5:
                print("NaN値を含む行（先頭5行）:")
                print(nan_rows.head())
        
        # mplfinanceでプロット用の設定
        # 1. メインチャート上のプロット
        main_plots = []
        
        # ATRバンドを表示
        if show_atr_bands:
            main_plots.append(mpf.make_addplot(df['atr_upper'], color='red', width=1, alpha=0.6, label=f'ATR Upper ({atr_multiplier}x)'))
            main_plots.append(mpf.make_addplot(df['atr_lower'], color='green', width=1, alpha=0.6, label=f'ATR Lower ({atr_multiplier}x)'))
        
        # 2. オシレータープロット
        # CATR %ベースパネル
        catr_percent_panel = mpf.make_addplot(df['catr_percent'], panel=1, color='purple', width=1.5, 
                                             ylabel='CATR (%)', secondary_y=False, label='CATR %')
        
        # True Rangeパネル
        tr_panel = mpf.make_addplot(df['true_range'], panel=2, color='orange', width=1.2, 
                                   ylabel='True Range', secondary_y=False, label='TR')
        
        # ATR期間パネル
        period_panel = mpf.make_addplot(df['atr_period'], panel=3, color='blue', width=1.2, 
                                       ylabel='ATR Period', secondary_y=False, label='ATR Period')
        
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
            kwargs['panel_ratios'] = (4, 1, 1, 1, 1)  # メイン:出来高:CATR%:TR:期間
            # パネル番号を+1する
            catr_percent_panel = mpf.make_addplot(df['catr_percent'], panel=2, color='purple', width=1.5, 
                                                 ylabel='CATR (%)', secondary_y=False, label='CATR %')
            tr_panel = mpf.make_addplot(df['true_range'], panel=3, color='orange', width=1.2, 
                                       ylabel='True Range', secondary_y=False, label='TR')
            period_panel = mpf.make_addplot(df['atr_period'], panel=4, color='blue', width=1.2, 
                                           ylabel='ATR Period', secondary_y=False, label='ATR Period')
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (4, 1, 1, 1)  # メイン:CATR%:TR:期間
        
        # すべてのプロットを結合
        all_plots = main_plots + [catr_percent_panel, tr_panel, period_panel]
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加（メインチャート）
        if show_atr_bands:
            axes[0].legend([f'ATR Upper ({atr_multiplier}x)', f'ATR Lower ({atr_multiplier}x)'], loc='upper left')
        
        self.fig = fig
        self.axes = axes
        
        # CATR%パネルに参照線を追加
        catr_panel_idx = 2 if show_volume else 1
        if catr_panel_idx < len(axes):
            catr_mean = df['catr_percent'].mean()
            if not np.isnan(catr_mean):
                axes[catr_panel_idx].axhline(y=catr_mean, color='black', linestyle='-', alpha=0.3, label=f'Mean ({catr_mean:.2f}%)')
                axes[catr_panel_idx].legend(loc='upper right', fontsize=8)
        
        # True Rangeパネルに参照線を追加
        tr_panel_idx = 3 if show_volume else 2
        if tr_panel_idx < len(axes):
            tr_mean = df['true_range'].mean()
            if not np.isnan(tr_mean):
                axes[tr_panel_idx].axhline(y=tr_mean, color='black', linestyle='-', alpha=0.3, label=f'Mean ({tr_mean:.4f})')
                axes[tr_panel_idx].legend(loc='upper right', fontsize=8)
        
        # ATR期間パネルに参照線を追加
        period_panel_idx = 4 if show_volume else 3
        if period_panel_idx < len(axes):
            period_mean = df['atr_period'].mean()
            if not np.isnan(period_mean):
                axes[period_panel_idx].axhline(y=period_mean, color='black', linestyle='-', alpha=0.3, label=f'Mean ({period_mean:.1f})')
                axes[period_panel_idx].legend(loc='upper right', fontsize=8)
        
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
    parser = argparse.ArgumentParser(description='CATR（Cycle Average True Range）の描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--detector', '-d', type=str, default='phac_e', help='検出器タイプ')
    parser.add_argument('--smoother', type=str, default='alma', choices=['alma', 'hyper'], help='平滑化アルゴリズム')
    parser.add_argument('--atr-multiplier', '-m', type=float, default=2.0, help='ATRバンドの乗数')
    parser.add_argument('--no-bands', action='store_true', help='ATRバンドを非表示にする')
    parser.add_argument('--kalman', action='store_true', help='カルマンフィルターを有効にする')
    args = parser.parse_args()
    
    # チャートを作成
    chart = CATRChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        detector_type=args.detector,
        smoother_type=args.smoother,
        use_kalman_filter=args.kalman
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        show_atr_bands=not args.no_bands,
        atr_multiplier=args.atr_multiplier,
        savefig=args.output
    )


if __name__ == "__main__":
    main() 