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
from indicators.alma import ALMA


class ALMAChart:
    """
    ALMA (Arnaud Legoux Moving Average) を表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - ALMAライン（固定期間・動的期間対応）
    - 動的期間の値（動的期間使用時）
    - ドミナントサイクル値（動的期間使用時）
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.alma = None
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
                            # 基本ALMAパラメータ
                            period: int = 200,
                            offset: float = 0.85,
                            sigma: float = 6,
                            src_type: str = 'close',
                            # 動的期間計算オプション
                            use_dynamic_period: bool = False,
                            # EhlersUnifiedDCパラメータ
                            detector_type: str = 'cycle_period2',
                            cycle_part: float = 1.0,
                            max_cycle: int = 233,
                            min_cycle: int = 13,
                            max_output: int = 144,
                            min_output: int = 13
                           ) -> None:
        """
        ALMA (Arnaud Legoux Moving Average) を計算する
        
        Args:
            period: 期間（固定期間使用時）
            offset: オフセット (0-1)。1に近いほど最新のデータを重視
            sigma: シグマ。大きいほど重みの差が大きくなる
            src_type: 価格ソース ('close', 'hlc3', etc.)
            use_dynamic_period: 動的期間を使用するかどうか
            detector_type: Ehlers DC検出器タイプ
            cycle_part: サイクル部分の倍率
            max_cycle: 最大サイクル期間
            min_cycle: 最小サイクル期間
            max_output: 最大出力値（動的期間の最大値）
            min_output: 最小出力値（動的期間の最小値）
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print(f"\nALMA (Arnaud Legoux Moving Average) を計算中...")
        print(f"パラメータ: 期間={period}, 動的期間={use_dynamic_period}, 検出器={detector_type}, ソース={src_type}")
        
        # ALMAを計算
        self.alma = ALMA(
            period=period,
            offset=offset,
            sigma=sigma,
            src_type=src_type,
            use_dynamic_period=use_dynamic_period,
            detector_type=detector_type,
            cycle_part=cycle_part,
            max_cycle=max_cycle,
            min_cycle=min_cycle,
            max_output=max_output,
            min_output=min_output
        )
        
        # ALMAの計算
        print("ALMA計算を実行します...")
        alma_values = self.alma.calculate(self.data)
        print(f"ALMA計算完了 - 配列長: {len(alma_values)}")
        
        # NaN値のチェック
        nan_count = np.isnan(alma_values).sum()
        valid_count = len(alma_values) - nan_count
        print(f"ALMA値 - 有効値: {valid_count}, NaN値: {nan_count}")
        
        if valid_count > 0:
            print(f"ALMA値範囲: {np.nanmin(alma_values):.4f} - {np.nanmax(alma_values):.4f}")
        
        # 動的期間の場合の統計
        if use_dynamic_period:
            dynamic_periods = self.alma.get_dynamic_periods()
            if len(dynamic_periods) > 0:
                period_valid = ~np.isnan(dynamic_periods)
                if period_valid.any():
                    periods = dynamic_periods[period_valid]
                    print(f"期間統計 - 平均: {periods.mean():.1f}, 範囲: {periods.min():.0f} - {periods.max():.0f}")
        
        print("ALMA計算完了")
            
    def plot(self, 
            title: str = "ALMA (Arnaud Legoux Moving Average)", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (14, 10),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとALMAを描画する
        
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
            
        if self.alma is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # ALMAの値を取得
        print("ALMAデータを取得中...")
        alma_values = self.alma.calculate(self.data)
        
        # 動的期間データの取得
        dynamic_periods = None
        if self.alma.use_dynamic_period:
            dynamic_periods = self.alma.get_dynamic_periods()
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(index=self.data.index)
        full_df['alma'] = alma_values
        
        if dynamic_periods is not None and len(dynamic_periods) == len(self.data):
            full_df['dynamic_periods'] = dynamic_periods
            
            # ドミナントサイクル検出器からDC値を取得
            if hasattr(self.alma, 'dc_detector') and self.alma.dc_detector is not None:
                try:
                    dc_values = self.alma.dc_detector.calculate(self.data)
                    if len(dc_values) == len(self.data):
                        full_df['dc_values'] = dc_values
                except Exception as e:
                    print(f"DC値取得エラー: {e}")
                    full_df['dc_values'] = np.nan
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        print(f"ALMAデータ確認 - NaN値: {df['alma'].isna().sum()}")
        
        # データ量が多い場合は絞り込み（削除）
        # if len(df) > 2000:
        #     print(f"データ量が多いため、最新の2000データポイントに絞り込みます（元: {len(df)}行）")
        #     df = df.tail(2000)
        #     print(f"絞り込み後のデータ行数: {len(df)}")
        
        # mplfinanceでプロット用の設定
        # 1. メインチャート上のプロット
        main_plots = []
        
        # ALMAライン
        if not df['alma'].isna().all():
            main_plots.append(mpf.make_addplot(df['alma'], color='blue', width=2.5, label='ALMA'))
        
        # 2. サブプロット
        plots_list = []
        panel_count = 1
        
        # 動的期間使用時のみ追加パネルを作成
        if self.alma.use_dynamic_period:
            # 動的期間パネル
            if 'dynamic_periods' in df.columns and not df['dynamic_periods'].isna().all():
                period_panel = mpf.make_addplot(df['dynamic_periods'], panel=panel_count, color='orange', width=1.5, 
                                               ylabel='Dynamic Period', secondary_y=False)
                plots_list.append(period_panel)
                panel_count += 1
            
            # ドミナントサイクルパネル
            if 'dc_values' in df.columns and not df['dc_values'].isna().all():
                dc_panel = mpf.make_addplot(df['dc_values'], panel=panel_count, color='purple', width=1.5, 
                                           ylabel='Dominant Cycle', secondary_y=False)
                plots_list.append(dc_panel)
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
            panel_ratios.extend([1] * actual_panel_count)  # 追加パネル
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
            panel_ratios.extend([1] * actual_panel_count)  # 追加パネル
            if panel_ratios:
                kwargs['panel_ratios'] = tuple(panel_ratios)
        
        # すべてのプロットを結合
        if main_plots or plots_list:
            all_plots = main_plots + plots_list
            kwargs['addplot'] = all_plots
        
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
            # ALMAラインのみ追加
            if not df['alma'].isna().all():
                simple_kwargs['addplot'] = [
                    mpf.make_addplot(df['alma'], color='blue', width=2, label='ALMA')
                ]
            fig, axes = mpf.plot(df, **simple_kwargs)
            print("簡易表示でプロットしました。")
        
        # 凡例の追加（メインチャート）
        if len(axes) > 0 and not df['alma'].isna().all():
            axes[0].legend(['ALMA'], loc='upper left')
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        panel_offset = 1 if show_volume else 0
        
        # 動的期間使用時の追加パネル
        if self.alma.use_dynamic_period:
            panel_idx = 1 + panel_offset
            
            # 動的期間パネル
            if 'dynamic_periods' in df.columns and not df['dynamic_periods'].isna().all():
                if panel_idx < len(axes):
                    period_mean = df['dynamic_periods'].mean()
                    if not np.isnan(period_mean):
                        axes[panel_idx].axhline(y=period_mean, color='black', linestyle='-', alpha=0.3, 
                                              label=f'Mean ({period_mean:.1f})')
                        axes[panel_idx].legend(loc='upper right', fontsize=8)
                panel_idx += 1
            
            # DCパネル
            if 'dc_values' in df.columns and not df['dc_values'].isna().all():
                if panel_idx < len(axes):
                    dc_mean = df['dc_values'].mean()
                    if not np.isnan(dc_mean):
                        axes[panel_idx].axhline(y=dc_mean, color='black', linestyle='-', alpha=0.3, 
                                              label=f'Mean ({dc_mean:.1f})')
                        axes[panel_idx].legend(loc='upper right', fontsize=8)
        
        # 統計情報の表示
        print(f"\n=== ALMA統計 ===")
        total_points = len(df[~df['alma'].isna()])
        print(f"総データ点数: {total_points}")
        
        if self.alma.use_dynamic_period:
            if 'dynamic_periods' in df.columns:
                valid_periods = df['dynamic_periods'][~df['dynamic_periods'].isna()]
                if len(valid_periods) > 0:
                    print(f"期間統計 - 平均: {valid_periods.mean():.1f}, 範囲: {valid_periods.min():.0f} - {valid_periods.max():.0f}")
            
            if 'dc_values' in df.columns:
                valid_dc = df['dc_values'][~df['dc_values'].isna()]
                if len(valid_dc) > 0:
                    print(f"DC統計 - 平均: {valid_dc.mean():.1f}, 範囲: {valid_dc.min():.0f} - {valid_dc.max():.0f}")
        
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
    parser = argparse.ArgumentParser(description='ALMA (Arnaud Legoux Moving Average) の描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--period', '-p', type=int, default=200, help='ALMA期間（固定期間時）')
    parser.add_argument('--offset', type=float, default=0.85, help='ALMAオフセット')
    parser.add_argument('--sigma', type=float, default=6, help='ALMAシグマ')
    parser.add_argument('--src', type=str, default='hlc3', help='価格ソース')
    parser.add_argument('--dynamic', action='store_true', default=True, help='動的期間を使用する（デフォルト有効）')
    parser.add_argument('--fixed', action='store_true', help='固定期間を使用する（動的期間を無効化）')
    parser.add_argument('--detector', '-d', type=str, default='cycle_period2', help='検出器タイプ（動的期間時）')
    parser.add_argument('--max-output', type=int, default=200, help='最大出力値（動的期間時）')
    parser.add_argument('--min-output', type=int, default=13, help='最小出力値（動的期間時）')
    args = parser.parse_args()
    
    # 固定期間フラグが指定された場合は動的期間を無効化
    use_dynamic = args.dynamic and not args.fixed
    
    # チャートを作成
    chart = ALMAChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        period=args.period,
        offset=args.offset,
        sigma=args.sigma,
        src_type=args.src,
        use_dynamic_period=use_dynamic,
        detector_type=args.detector,
        max_output=args.max_output,
        min_output=args.min_output
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output
    )


if __name__ == "__main__":
    main() 