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
from indicators.cycle_efficiency_ratio import CycleEfficiencyRatio


class CycleEfficiencyRatioChart:
    """
    サイクル効率比（CER）を表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - サイクル効率比（CER）
    - 動的ドミナントサイクル値
    - セルフアダプティブ期間（有効時）
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.cycle_er = None
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
                            # CERパラメータ
                            detector_type: str = 'phac_e',
                            cycle_part: float = 0.5,
                            lp_period: int = 5,
                            hp_period: int = 55,
                            max_cycle: int = 55,
                            min_cycle: int = 5,
                            max_output: int = 34,
                            min_output: int = 5,
                            src_type: str = 'hlc3',
                            smooth_er: bool = True,
                            er_alma_period: int = 5,
                            er_alma_offset: float = 0.85,
                            er_alma_sigma: float = 6,
                            self_adaptive: bool = False,
                            # 新しい検出器用のパラメータ
                            alpha: float = 0.07,
                            bandwidth: float = 0.6,
                            center_period: float = 15.0,
                            avg_length: float = 3.0,
                            window: int = 50
                           ) -> None:
        """
        サイクル効率比（CER）を計算する
        
        Args:
            detector_type: ドミナントサイクル検出器タイプ
            cycle_part: サイクル部分
            lp_period: ローパスフィルター期間
            hp_period: ハイパスフィルター期間
            max_cycle: 最大サイクル期間
            min_cycle: 最小サイクル期間
            max_output: 最大出力値
            min_output: 最小出力値
            src_type: ソースタイプ
            smooth_er: ERスムージング有無
            er_alma_period: ALMAスムージング期間
            er_alma_offset: ALMAスムージングオフセット
            er_alma_sigma: ALMAスムージングシグマ
            self_adaptive: セルフアダプティブモード有無
            alpha: アルファパラメータ
            bandwidth: 帯域幅
            center_period: 中心周期
            avg_length: 平均長
            window: 分析ウィンドウ長
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print(f"\nサイクル効率比（CER）を計算中...")
        print(f"パラメータ: 検出器={detector_type}, ソース={src_type}, スムージング={smooth_er}, アダプティブ={self_adaptive}")
        
        # サイクル効率比を計算
        self.cycle_er = CycleEfficiencyRatio(
            detector_type=detector_type,
            cycle_part=cycle_part,
            lp_period=lp_period,
            hp_period=hp_period,
            max_cycle=max_cycle,
            min_cycle=min_cycle,
            max_output=max_output,
            min_output=min_output,
            src_type=src_type,
            smooth_er=smooth_er,
            er_alma_period=er_alma_period,
            er_alma_offset=er_alma_offset,
            er_alma_sigma=er_alma_sigma,
            self_adaptive=self_adaptive,
            alpha=alpha,
            bandwidth=bandwidth,
            center_period=center_period,
            avg_length=avg_length,
            window=window
        )
        
        # CERの計算
        print("CER計算を実行します...")
        cer_values = self.cycle_er.calculate(self.data)
        print(f"CER計算完了 - 配列長: {len(cer_values)}")
        
        # NaN値のチェック
        nan_count = np.isnan(cer_values).sum()
        valid_count = len(cer_values) - nan_count
        print(f"CER値 - 有効値: {valid_count}, NaN値: {nan_count}")
        
        if valid_count > 0:
            print(f"CER値範囲: {np.nanmin(cer_values):.4f} - {np.nanmax(cer_values):.4f}")
        
        print("サイクル効率比（CER）計算完了")
            
    def plot(self, 
            title: str = "サイクル効率比（CER）", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (14, 10),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとサイクル効率比（CER）を描画する
        
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
            
        if self.cycle_er is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # CERの値を取得
        print("CERデータを取得中...")
        cer_values = self.cycle_er.calculate(self.data)
        raw_cer = self.cycle_er.get_raw_values()
        smoothed_cer = self.cycle_er.get_smoothed_values()
        adaptive_periods = self.cycle_er.get_adaptive_periods()
        
        # ドミナントサイクル値を取得（dc_detectorからアクセス）
        dc_values = self.cycle_er.dc_detector.calculate(self.data) if hasattr(self.cycle_er, 'dc_detector') else np.full(len(self.data), np.nan)
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'cer': cer_values,
                'raw_cer': raw_cer if len(raw_cer) > 0 else cer_values,
                'smoothed_cer': smoothed_cer if len(smoothed_cer) > 0 else cer_values,
                'dominant_cycle': dc_values,
                'adaptive_periods': adaptive_periods if len(adaptive_periods) > 0 else np.full(len(self.data), np.nan)
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        print(f"CERデータ確認 - NaN値: {df['cer'].isna().sum()}")
        
        # データ量が多い場合は絞り込み
        if len(df) > 2000:
            print(f"データ量が多いため、最新の2000データポイントに絞り込みます（元: {len(df)}行）")
            df = df.tail(2000)
            print(f"絞り込み後のデータ行数: {len(df)}")
        
        # NaN値を含む行を出力（最初の5行のみ）
        nan_rows = df[df['cer'].isna()]
        if not nan_rows.empty:
            print(f"NaN値を含む行: {len(nan_rows)}行")
            if len(nan_rows) <= 5:
                print("NaN値を含む行（先頭5行）:")
                print(nan_rows.head())
        
        # mplfinanceでプロット用の設定
        # 1. メインチャート上のプロット
        main_plots = []
        
        # 2. オシレータープロット
        # CERパネル
        cer_panel = mpf.make_addplot(df['cer'], panel=1, color='purple', width=1.5, 
                                     ylabel='CER', secondary_y=False, label='CER')
        
        # 生とスムージングされたCERの両方を表示（異なる場合のみ）
        additional_plots = []
        if self.cycle_er.smooth_er and not df['raw_cer'].equals(df['smoothed_cer']):
            raw_cer_panel = mpf.make_addplot(df['raw_cer'], panel=1, color='lightpurple', width=1, 
                                           alpha=0.7, label='Raw CER')
            additional_plots.append(raw_cer_panel)
        
        # ドミナントサイクルパネル
        dc_panel = mpf.make_addplot(df['dominant_cycle'], panel=2, color='blue', width=1.2, 
                                   ylabel='Dominant Cycle', secondary_y=False, label='DC')
        
        # セルフアダプティブ期間パネル（有効時のみ）
        panels_count = 3  # 基本: メイン、CER、DC
        if self.cycle_er.self_adaptive and not df['adaptive_periods'].isna().all():
            adaptive_panel = mpf.make_addplot(df['adaptive_periods'], panel=3, color='orange', width=1.2,
                                            ylabel='Adaptive Period', secondary_y=False, label='Adaptive')
            additional_plots.append(adaptive_panel)
            panels_count = 4
        
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
            if panels_count == 3:
                kwargs['panel_ratios'] = (4, 1, 1, 1)  # メイン:出来高:CER:DC
                # パネル番号を+1する
                cer_panel = mpf.make_addplot(df['cer'], panel=2, color='purple', width=1.5, 
                                           ylabel='CER', secondary_y=False, label='CER')
                dc_panel = mpf.make_addplot(df['dominant_cycle'], panel=3, color='blue', width=1.2, 
                                          ylabel='Dominant Cycle', secondary_y=False, label='DC')
                if additional_plots:
                    for i, plot in enumerate(additional_plots):
                        if plot.panel == 1:  # CERパネルの追加プロット
                            additional_plots[i] = mpf.make_addplot(plot.data, panel=2, color=plot.color, 
                                                                 width=plot.width, alpha=plot.alpha if hasattr(plot, 'alpha') else 1.0)
            else:  # panels_count == 4
                kwargs['panel_ratios'] = (4, 1, 1, 1, 1)  # メイン:出来高:CER:DC:アダプティブ
                # パネル番号を+1する
                cer_panel = mpf.make_addplot(df['cer'], panel=2, color='purple', width=1.5, 
                                           ylabel='CER', secondary_y=False, label='CER')
                dc_panel = mpf.make_addplot(df['dominant_cycle'], panel=3, color='blue', width=1.2, 
                                          ylabel='Dominant Cycle', secondary_y=False, label='DC')
                # アダプティブ期間のパネル番号を修正
                for i, plot in enumerate(additional_plots):
                    if hasattr(plot, 'panel'):
                        if plot.panel == 1:  # CERパネル
                            additional_plots[i] = mpf.make_addplot(plot.data, panel=2, color=plot.color, 
                                                                 width=plot.width, alpha=plot.alpha if hasattr(plot, 'alpha') else 1.0)
                        elif plot.panel == 3:  # アダプティブパネル
                            additional_plots[i] = mpf.make_addplot(plot.data, panel=4, color=plot.color, 
                                                                 width=plot.width, alpha=plot.alpha if hasattr(plot, 'alpha') else 1.0)
        else:
            kwargs['volume'] = False
            if panels_count == 3:
                kwargs['panel_ratios'] = (4, 1, 1)  # メイン:CER:DC
            else:  # panels_count == 4
                kwargs['panel_ratios'] = (4, 1, 1, 1)  # メイン:CER:DC:アダプティブ
        
        # すべてのプロットを結合
        all_plots = main_plots + [cer_panel, dc_panel] + additional_plots
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        self.fig = fig
        self.axes = axes
        
        # CERパネルに参照線を追加
        cer_panel_idx = 2 if show_volume else 1
        if cer_panel_idx < len(axes):
            axes[cer_panel_idx].axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Strong Trend')
            axes[cer_panel_idx].axhline(y=0.0, color='black', linestyle='-', alpha=0.3, label='Neutral')
            axes[cer_panel_idx].axhline(y=-0.7, color='red', linestyle='--', alpha=0.5, label='Strong Trend (Down)')
            axes[cer_panel_idx].legend(loc='upper right', fontsize=8)
        
        # ドミナントサイクルパネルに参照線を追加
        dc_panel_idx = 3 if show_volume else 2
        if dc_panel_idx < len(axes):
            dc_mean = df['dominant_cycle'].mean()
            if not np.isnan(dc_mean):
                axes[dc_panel_idx].axhline(y=dc_mean, color='black', linestyle='-', alpha=0.3, label=f'Mean ({dc_mean:.1f})')
                axes[dc_panel_idx].legend(loc='upper right', fontsize=8)
        
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
    parser = argparse.ArgumentParser(description='サイクル効率比（CER）の描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--detector', '-d', type=str, default='phac_e', help='検出器タイプ')
    parser.add_argument('--smooth', action='store_true', help='ERスムージングを有効にする')
    parser.add_argument('--adaptive', action='store_true', help='セルフアダプティブモードを有効にする')
    args = parser.parse_args()
    
    # チャートを作成
    chart = CycleEfficiencyRatioChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        detector_type=args.detector,
        smooth_er=args.smooth,
        self_adaptive=args.adaptive
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output
    )


if __name__ == "__main__":
    main() 