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
from indicators.trend_filter.ehlers_market_mode import EhlersMarketMode


class EhlersMarketModeChart:
    """
    Ehlers Market Modeを表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - スムーズされた価格
    - トレンドライン
    - マーケットモード（トレンド/サイクル）
    - DCフェーズ
    - サイクル期間
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.ehlers_market_mode = None
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
                            src_type: str = 'hlc3',
                            alpha: float = 0.1,
                            use_unified_smoother: bool = True,
                            smoother_type: str = 'alma',
                            smoother_length: int = 4,
                            enable_kalman_filter: bool = False,
                            kalman_type: str = 'ekf',
                            use_cycle_detector: bool = False,
                            enable_percentile_analysis: bool = False
                           ) -> None:
        """
        Ehlers Market Modeを計算する
        
        Args:
            src_type: ソースタイプ
            alpha: スムージングパラメータ
            use_unified_smoother: 統合スムーサーを使用するか
            smoother_type: スムーサータイプ
            smoother_length: スムーサー長さ
            enable_kalman_filter: カルマンフィルター使用有無
            kalman_type: カルマンフィルタータイプ
            use_cycle_detector: サイクル検出器使用有無
            enable_percentile_analysis: パーセンタイル分析使用有無
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\nEhlers Market Modeを計算中...")
        
        # Ehlers Market Modeを計算
        self.ehlers_market_mode = EhlersMarketMode(
            src_type=src_type,
            alpha=alpha,
            use_unified_smoother=use_unified_smoother,
            smoother_type=smoother_type,
            smoother_length=smoother_length,
            enable_kalman_filter=enable_kalman_filter,
            kalman_type=kalman_type,
            use_cycle_detector=use_cycle_detector,
            enable_percentile_analysis=enable_percentile_analysis
        )
        
        # EhlersMarketModeの計算
        print("計算を実行します...")
        result = self.ehlers_market_mode.calculate(self.data)
        
        print(f"Market Mode計算完了 - データ長: {len(result.smooth)}")
        
        # NaN値のチェック
        nan_count_smooth = np.isnan(result.smooth).sum()
        nan_count_trend_mode = np.isnan(result.trend_mode).sum()
        nan_count_signal = np.isnan(result.signal).sum()
        print(f"NaN値 - スムーズ: {nan_count_smooth}, トレンドモード: {nan_count_trend_mode}, シグナル: {nan_count_signal}")
        
        print("Ehlers Market Mode計算完了")
            
    def plot(self, 
            title: str = "Ehlers Market Mode", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (16, 12),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとEhlers Market Modeを描画する
        
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
            
        if self.ehlers_market_mode is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # Ehlers Market Modeの結果を取得
        print("Market Modeデータを取得中...")
        result = self.ehlers_market_mode.last_result
        
        if result is None:
            raise ValueError("Market Modeの計算結果がありません。")
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'smooth_price': result.smooth,
                'trend_line': result.trend_line,
                'trend_mode': result.trend_mode,
                'signal': result.signal,
                'dc_phase': result.dc_phase,
                'period': result.period,
                'smooth_period': result.smooth_period,
                'days_in_trend': result.days_in_trend,
                'i_trend': result.i_trend
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        print(f"Market Modeデータ確認 - スムーズNaN: {df['smooth_price'].isna().sum()}, モードNaN: {df['trend_mode'].isna().sum()}")
        
        # NaN値を含む行の確認（最初の5行のみ）
        nan_rows = df[df['smooth_price'].isna() | df['trend_mode'].isna()]
        if not nan_rows.empty:
            print(f"NaN値を含む行: {len(nan_rows)}行")
            if len(nan_rows) > 0:
                print(nan_rows.head())
        
        # トレンドモードのカラーマップを作成
        trend_colors = np.where(df['trend_mode'] == 1, 'green', 
                               np.where(df['trend_mode'] == 0, 'gray', 'red'))
        
        # mplfinanceでプロット用の設定
        # 1. メインチャート上のプロット
        main_plots = []
        
        # スムーズされた価格とトレンドライン
        main_plots.append(mpf.make_addplot(df['smooth_price'], color='blue', width=1.5, label='Smooth Price'))
        main_plots.append(mpf.make_addplot(df['trend_line'], color='orange', width=1.5, label='Trend Line'))
        
        # 2. オシレータープロット
        # トレンドモード表示
        trend_panel = mpf.make_addplot(df['trend_mode'], panel=1, color='purple', width=1.5, 
                                     ylabel='Trend Mode', secondary_y=False, label='Trend Mode')
        
        # シグナル表示
        signal_panel = mpf.make_addplot(df['signal'], panel=2, color='red', width=1.5, 
                                       ylabel='Signal', secondary_y=False, label='Signal')
        
        # DCフェーズ表示
        phase_panel = mpf.make_addplot(df['dc_phase'], panel=3, color='green', width=1.2, 
                                      ylabel='DC Phase', secondary_y=False, label='DC Phase')
        
        # サイクル期間表示
        period_panel = mpf.make_addplot(df['smooth_period'], panel=4, color='brown', width=1.2, 
                                       ylabel='Cycle Period', secondary_y=False, label='Period')
        
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
            kwargs['panel_ratios'] = (4, 1, 1, 1, 1, 1)  # メイン:出来高:モード:シグナル:フェーズ:期間
            # 出来高を表示する場合は、オシレーターのパネル番号を+1する
            trend_panel = mpf.make_addplot(df['trend_mode'], panel=2, color='purple', width=1.5, 
                                         ylabel='Trend Mode', secondary_y=False, label='Trend Mode')
            signal_panel = mpf.make_addplot(df['signal'], panel=3, color='red', width=1.5, 
                                           ylabel='Signal', secondary_y=False, label='Signal')
            phase_panel = mpf.make_addplot(df['dc_phase'], panel=4, color='green', width=1.2, 
                                          ylabel='DC Phase', secondary_y=False, label='DC Phase')
            period_panel = mpf.make_addplot(df['smooth_period'], panel=5, color='brown', width=1.2, 
                                           ylabel='Period', secondary_y=False, label='Period')
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (4, 1, 1, 1, 1)  # メイン:モード:シグナル:フェーズ:期間
        
        # すべてのプロットを結合
        all_plots = main_plots + [trend_panel, signal_panel, phase_panel, period_panel]
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加
        axes[0].legend(['Smooth Price', 'Trend Line'], loc='upper left')
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        panel_offset = 1 if show_volume else 0
        
        # トレンドモードパネル（0と1の線）
        trend_axis = axes[1 + panel_offset]
        trend_axis.axhline(y=0.5, color='black', linestyle='-', alpha=0.3)
        trend_axis.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        trend_axis.axhline(y=1, color='green', linestyle='--', alpha=0.5)
        
        # シグナルパネル（-1, 0, 1の線）
        signal_axis = axes[2 + panel_offset]
        signal_axis.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        signal_axis.axhline(y=1, color='green', linestyle='--', alpha=0.5)
        signal_axis.axhline(y=-1, color='red', linestyle='--', alpha=0.5)
        
        # DCフェーズパネル（0の線）
        phase_axis = axes[3 + panel_offset]
        phase_axis.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # サイクル期間パネル（平均値の線）
        period_axis = axes[4 + panel_offset]
        period_mean = df['smooth_period'].mean()
        period_axis.axhline(y=period_mean, color='black', linestyle='-', alpha=0.3)
        
        # 保存または表示
        if savefig:
            plt.savefig(savefig, dpi=300, bbox_inches='tight')
            print(f"チャートを保存しました: {savefig}")
        else:
            plt.tight_layout()
            plt.show()


def main():
    """メイン関数"""
    # コマンドライン引数を処理
    import argparse
    parser = argparse.ArgumentParser(description='Ehlers Market Modeの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--src', type=str, default='hlc3', help='価格ソース (close, hlc3, etc.)')
    parser.add_argument('--alpha', type=float, default=0.1, help='スムージングアルファ値')
    args = parser.parse_args()
    
    # チャートを作成
    chart = EhlersMarketModeChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        src_type=args.src,
        alpha=args.alpha
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output
    )


if __name__ == "__main__":
    main()