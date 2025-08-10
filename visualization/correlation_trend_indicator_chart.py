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
from indicators.trend_filter.correlation_trend_indicator import CorrelationTrendIndicator


class CorrelationTrendIndicatorChart:
    """
    Correlation Trend Indicator を表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - 相関値（-1から+1の範囲）
    - トレンドシグナル（+1: 上昇, -1: 下降, 0: 横這い）
    - トレンド強度（0-1の範囲）
    - 平滑化された相関値（オプション）
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.correlation_indicator = None
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
        print("\\nデータを読み込み・処理中...")
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
                            length: int = 20,
                            src_type: str = 'close',
                            trend_threshold: float = 0.3,
                            enable_smoothing: bool = False,
                            smooth_length: int = 5
                           ) -> None:
        """
        Correlation Trend Indicatorを計算する
        
        Args:
            length: 相関計算期間（デフォルト: 20）
            src_type: ソースタイプ（デフォルト: 'close'）
            trend_threshold: トレンド判定閾値（デフォルト: 0.3）
            enable_smoothing: 平滑化を有効にするか（デフォルト: False）
            smooth_length: 平滑化期間（デフォルト: 5）
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\\nCorrelation Trend Indicatorを計算中...")
        
        # Correlation Trend Indicatorを計算
        self.correlation_indicator = CorrelationTrendIndicator(
            length=length,
            src_type=src_type,
            trend_threshold=trend_threshold,
            enable_smoothing=enable_smoothing,
            smooth_length=smooth_length
        )
        
        # インジケーターの計算
        print("計算を実行します...")
        result = self.correlation_indicator.calculate(self.data)
        
        # 結果の取得
        correlation_values = result.values
        trend_signal = result.trend_signal
        trend_strength = result.trend_strength
        smoothed_values = result.smoothed_values
        
        print(f"Correlation Trend Indicator計算完了 - データ点数: {len(correlation_values)}")
        
        # 統計情報
        uptrend_count = (trend_signal == 1).sum()
        downtrend_count = (trend_signal == -1).sum()
        sideways_count = (trend_signal == 0).sum()
        
        print(f"状態統計:")
        print(f"  上昇トレンド: {uptrend_count} ({uptrend_count/len(trend_signal)*100:.1f}%)")
        print(f"  下降トレンド: {downtrend_count} ({downtrend_count/len(trend_signal)*100:.1f}%)")
        print(f"  横這い: {sideways_count} ({sideways_count/len(trend_signal)*100:.1f}%)")
        
        print(f"統計 - 相関値平均: {np.nanmean(correlation_values):.3f}")
        print(f"相関値範囲: {np.nanmin(correlation_values):.3f} - {np.nanmax(correlation_values):.3f}")
        print(f"トレンド強度平均: {np.nanmean(trend_strength):.3f}")
        
        if enable_smoothing:
            print(f"平滑化相関値平均: {np.nanmean(smoothed_values):.3f}")
        
        print("Correlation Trend Indicator計算完了")
            
    def plot(self, 
            title: str = "Correlation Trend Indicator", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (16, 14),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとCorrelation Trend Indicatorを描画する
        
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
            
        if self.correlation_indicator is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # Correlation Trend Indicatorの値を取得
        print("Correlation Trend Indicatorデータを取得中...")
        result = self.correlation_indicator.calculate(self.data)
        
        correlation_values = result.values
        trend_signal = result.trend_signal
        trend_strength = result.trend_strength
        smoothed_values = result.smoothed_values
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'correlation': correlation_values,
                'trend_signal': trend_signal,
                'trend_strength': trend_strength,
                'smoothed_correlation': smoothed_values
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        
        # トレンドシグナルに基づく色分け
        df['trend_up'] = np.where(df['trend_signal'] == 1, df['correlation'], np.nan)
        df['trend_down'] = np.where(df['trend_signal'] == -1, df['correlation'], np.nan)
        df['trend_sideways'] = np.where(df['trend_signal'] == 0, df['correlation'], np.nan)
        
        # デバッグ情報の出力
        up_count = (~np.isnan(df['trend_up'])).sum()
        down_count = (~np.isnan(df['trend_down'])).sum()
        sideways_count = (~np.isnan(df['trend_sideways'])).sum()
        print(f"Trend Up points: {up_count}, Down points: {down_count}, Sideways points: {sideways_count}")
        
        # 全てがNaNの場合は、適切なダミー値を追加
        if up_count == 0:
            df.loc[df.index[0], 'trend_up'] = 0.5
        if down_count == 0:
            df.loc[df.index[0], 'trend_down'] = -0.5
        if sideways_count == 0:
            df.loc[df.index[0], 'trend_sideways'] = 0.0
        
        # mplfinanceでプロット用の設定
        main_plots = []
        
        # 相関値パネル - 状態別に色分けしたプロット
        corr_up_panel = mpf.make_addplot(df['trend_up'], panel=1, color='green', width=2.0, 
                                        ylabel='Correlation\\n(-1 to +1)', secondary_y=False, 
                                        label='Up Trend', type='line')
        corr_down_panel = mpf.make_addplot(df['trend_down'], panel=1, color='red', width=2.0, 
                                          secondary_y=False, label='Down Trend', type='line')
        corr_sideways_panel = mpf.make_addplot(df['trend_sideways'], panel=1, color='gray', width=1.5, 
                                             secondary_y=False, label='Sideways', type='line')
        
        # 平滑化相関値パネル（有効な場合）
        smoothed_panel = None
        if self.correlation_indicator.enable_smoothing:
            smoothed_panel = mpf.make_addplot(df['smoothed_correlation'], panel=2, color='blue', width=2.0, 
                                            ylabel='Smoothed\\nCorrelation', secondary_y=False, label='Smoothed')
        
        # トレンドシグナルパネル
        signal_panel = mpf.make_addplot(df['trend_signal'], panel=3 if smoothed_panel else 2, color='purple', width=2.5, 
                                       ylabel='Trend Signal\\n(1=Up, -1=Down, 0=Side)', secondary_y=False, 
                                       label='Signal', type='line')
        
        # トレンド強度パネル
        strength_panel = mpf.make_addplot(df['trend_strength'], panel=4 if smoothed_panel else 3, color='orange', width=1.5, 
                                         ylabel='Trend Strength\\n(0-1)', secondary_y=False, label='Strength')
        
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
            if smoothed_panel:
                kwargs['panel_ratios'] = (5, 1, 1, 1, 1, 1)  # メイン:出来高:Corr:Smoothed:Signal:Strength
                # 出来高を表示する場合は、オシレーターのパネル番号を+1する
                corr_up_panel = mpf.make_addplot(df['trend_up'], panel=2, color='green', width=2.0, 
                                                ylabel='Correlation', secondary_y=False, label='Up Trend', type='line')
                corr_down_panel = mpf.make_addplot(df['trend_down'], panel=2, color='red', width=2.0, 
                                                  secondary_y=False, label='Down Trend', type='line')
                corr_sideways_panel = mpf.make_addplot(df['trend_sideways'], panel=2, color='gray', width=1.5, 
                                                     secondary_y=False, label='Sideways', type='line')
                smoothed_panel = mpf.make_addplot(df['smoothed_correlation'], panel=3, color='blue', width=2.0, 
                                                ylabel='Smoothed Correlation', secondary_y=False, label='Smoothed')
                signal_panel = mpf.make_addplot(df['trend_signal'], panel=4, color='purple', width=2.5, 
                                               ylabel='Trend Signal', secondary_y=False, label='Signal', type='line')
                strength_panel = mpf.make_addplot(df['trend_strength'], panel=5, color='orange', width=1.5, 
                                                 ylabel='Trend Strength', secondary_y=False, label='Strength')
            else:
                kwargs['panel_ratios'] = (5, 1, 1, 1, 1)  # メイン:出来高:Corr:Signal:Strength
                corr_up_panel = mpf.make_addplot(df['trend_up'], panel=2, color='green', width=2.0, 
                                                ylabel='Correlation', secondary_y=False, label='Up Trend', type='line')
                corr_down_panel = mpf.make_addplot(df['trend_down'], panel=2, color='red', width=2.0, 
                                                  secondary_y=False, label='Down Trend', type='line')
                corr_sideways_panel = mpf.make_addplot(df['trend_sideways'], panel=2, color='gray', width=1.5, 
                                                     secondary_y=False, label='Sideways', type='line')
                signal_panel = mpf.make_addplot(df['trend_signal'], panel=3, color='purple', width=2.5, 
                                               ylabel='Trend Signal', secondary_y=False, label='Signal', type='line')
                strength_panel = mpf.make_addplot(df['trend_strength'], panel=4, color='orange', width=1.5, 
                                                 ylabel='Trend Strength', secondary_y=False, label='Strength')
        else:
            kwargs['volume'] = False
            if smoothed_panel:
                kwargs['panel_ratios'] = (5, 1, 1, 1, 1)  # メイン:Corr:Smoothed:Signal:Strength
            else:
                kwargs['panel_ratios'] = (5, 1, 1, 1)  # メイン:Corr:Signal:Strength
        
        # すべてのプロットを結合
        all_plots = main_plots + [corr_up_panel, corr_down_panel, corr_sideways_panel]
        if smoothed_panel:
            all_plots.append(smoothed_panel)
        all_plots.extend([signal_panel, strength_panel])
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        panel_offset = 2 if show_volume else 1
        
        # 相関パネル
        corr_axis = axes[panel_offset]
        corr_axis.set_ylim(-1.1, 1.1)
        corr_axis.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=1)
        corr_axis.axhline(y=0.3, color='green', linestyle='--', alpha=0.6, linewidth=1)
        corr_axis.axhline(y=-0.3, color='red', linestyle='--', alpha=0.6, linewidth=1)
        corr_axis.axhline(y=1, color='black', linestyle='-', alpha=0.3, linewidth=1)
        corr_axis.axhline(y=-1, color='black', linestyle='-', alpha=0.3, linewidth=1)
        
        # 平滑化パネル（ある場合）
        if smoothed_panel:
            smooth_axis = axes[panel_offset + 1]
            smooth_axis.set_ylim(-1.1, 1.1)
            smooth_axis.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=1)
            smooth_axis.axhline(y=0.3, color='green', linestyle='--', alpha=0.6, linewidth=1)
            smooth_axis.axhline(y=-0.3, color='red', linestyle='--', alpha=0.6, linewidth=1)
            panel_offset += 1
        
        # シグナルパネル
        signal_axis = axes[panel_offset + 1]
        signal_axis.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        signal_axis.axhline(y=1, color='green', linestyle='--', alpha=0.5)
        signal_axis.axhline(y=-1, color='red', linestyle='--', alpha=0.5)
        signal_axis.set_ylim(-1.2, 1.2)
        
        # トレンド強度パネル
        strength_axis = axes[panel_offset + 2]
        strength_axis.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
        strength_axis.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        strength_axis.axhline(y=1, color='black', linestyle='-', alpha=0.3)
        strength_axis.set_ylim(0, 1)
        
        # 統計情報の表示
        print(f"\\n=== Correlation Trend Indicator統計 ===")
        total_points = len(df)
        uptrend_points = len(df[df['trend_signal'] == 1])
        downtrend_points = len(df[df['trend_signal'] == -1])
        sideways_points = len(df[df['trend_signal'] == 0])
        
        print(f"総データ点数: {total_points}")
        print(f"上昇トレンド: {uptrend_points} ({uptrend_points/total_points*100:.1f}%)")
        print(f"下降トレンド: {downtrend_points} ({downtrend_points/total_points*100:.1f}%)")
        print(f"横這い: {sideways_points} ({sideways_points/total_points*100:.1f}%)")
        print(f"相関値 - 平均: {df['correlation'].mean():.3f}, 範囲: {df['correlation'].min():.3f} - {df['correlation'].max():.3f}")
        print(f"トレンド強度 - 平均: {df['trend_strength'].mean():.3f}, 範囲: {df['trend_strength'].min():.3f} - {df['trend_strength'].max():.3f}")
        
        if self.correlation_indicator.enable_smoothing:
            print(f"平滑化相関値 - 平均: {df['smoothed_correlation'].mean():.3f}")
        
        # シグナル遷移の分析
        signal_changes = (df['trend_signal'] != df['trend_signal'].shift(1)).sum()
        print(f"シグナル変更: {signal_changes}回")
        
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
    parser = argparse.ArgumentParser(description='Correlation Trend Indicatorの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--length', type=int, default=20, help='相関計算期間')
    parser.add_argument('--threshold', type=float, default=0.3, help='トレンド判定閾値')
    parser.add_argument('--src-type', type=str, default='close', help='ソースタイプ')
    parser.add_argument('--smooth', action='store_true', help='平滑化を有効にする')
    parser.add_argument('--smooth-length', type=int, default=5, help='平滑化期間')
    args = parser.parse_args()
    
    # チャートを作成
    chart = CorrelationTrendIndicatorChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        length=args.length,
        src_type=args.src_type,
        trend_threshold=args.threshold,
        enable_smoothing=args.smooth,
        smooth_length=args.smooth_length
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output
    )


if __name__ == "__main__":
    main()