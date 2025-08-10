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
from indicators.trend_filter.empirical_mode_decomposition import EMD


class EMDChart:
    """
    EMD（Empirical Mode Decomposition）を表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - バンドパスフィルター（サイクル成分）
    - トレンド成分
    - ピーク・バレー閾値
    - モード信号
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.emd = None
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
                            period: int = 20,
                            delta: float = 0.1,
                            fraction: float = 0.25,
                            avg_period: int = 50,
                            use_smoothing: bool = False,
                            smoother_type: str = 'super_smoother',
                            smoother_period: int = 8,
                            use_dynamic_period: bool = False,
                            use_kalman_filter: bool = False,
                            enable_percentile_analysis: bool = True
                           ) -> None:
        """
        EMDを計算する
        
        Args:
            src_type: ソースタイプ
            period: フィルター周期
            delta: バンド幅パラメータ
            fraction: 閾値計算用係数
            avg_period: ピーク・バレー平均化期間
            use_smoothing: スムーサーを使用するか
            smoother_type: スムーサータイプ
            smoother_period: スムーサー期間
            use_dynamic_period: 動的期間を使用するか
            use_kalman_filter: カルマンフィルター使用有無
            enable_percentile_analysis: パーセンタイル分析使用有無
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\nEMDを計算中...")
        
        # EMDを計算
        self.emd = EMD(
            src_type=src_type,
            period=period,
            delta=delta,
            fraction=fraction,
            avg_period=avg_period,
            use_smoothing=use_smoothing,
            smoother_type=smoother_type,
            smoother_period=smoother_period,
            use_dynamic_period=use_dynamic_period,
            use_kalman_filter=use_kalman_filter,
            enable_percentile_analysis=enable_percentile_analysis
        )
        
        # EMDの計算
        print("計算を実行します...")
        self.emd_result = self.emd.calculate(self.data)
        
        print(f"EMD計算完了 - データ長: {len(self.emd_result.bandpass)}")
        
        # NaN値のチェック
        nan_count_bandpass = np.isnan(self.emd_result.bandpass).sum()
        nan_count_trend = np.isnan(self.emd_result.trend).sum()
        nan_count_mode = np.isnan(self.emd_result.mode_signal).sum()
        print(f"NaN値 - バンドパス: {nan_count_bandpass}, トレンド: {nan_count_trend}, モード: {nan_count_mode}")
        
        print("EMD計算完了")
            
    def plot(self, 
            title: str = "EMD (Empirical Mode Decomposition)", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (16, 14),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとEMDを描画する
        
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
            
        if self.emd is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        if not hasattr(self, 'emd_result') or self.emd_result is None:
            raise ValueError("EMDの計算結果がありません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # EMDの結果を取得
        print("EMDデータを取得中...")
        result = self.emd_result
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'bandpass': result.bandpass,
                'trend': result.trend,
                'peaks': result.peaks,
                'valleys': result.valleys,
                'avg_peak': result.avg_peak,
                'avg_valley': result.avg_valley,
                'upper_threshold': result.upper_threshold,
                'lower_threshold': result.lower_threshold,
                'mode_signal': result.mode_signal
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        print(f"EMDデータ確認 - バンドパスNaN: {df['bandpass'].isna().sum()}, トレンドNaN: {df['trend'].isna().sum()}")
        
        # NaN値を含む行の確認（最初の5行のみ）
        nan_rows = df[df['bandpass'].isna() | df['trend'].isna()]
        if not nan_rows.empty:
            print(f"NaN値を含む行: {len(nan_rows)}行")
            if len(nan_rows) > 0:
                print(nan_rows.head())
        
        # データの前処理: NaN値の処理
        # トレンド成分のNaN値を前方填補
        df['trend'] = df['trend'].ffill()
        
        # まだNaN値がある場合は後方填補
        df['trend'] = df['trend'].bfill()
        
        # まだNaN値がある場合は0で置換
        df['trend'] = df['trend'].fillna(0)
        
        # ピーク・バレー値: 散布図プロット用の前処理
        # NaN値のままにして、有効値のみをプロット
        df['peaks_plot'] = df['peaks'].copy()
        df['valleys_plot'] = df['valleys'].copy()
        
        # 平均ピーク・バレー値のNaN値を前方填補
        df['avg_peak'] = df['avg_peak'].ffill().bfill()
        df['avg_valley'] = df['avg_valley'].ffill().bfill()
        
        # まだNaN値がある場合は0で置換
        df['avg_peak'] = df['avg_peak'].fillna(0)
        df['avg_valley'] = df['avg_valley'].fillna(0)
        
        # 上下閾値のNaN値を前方填補
        df['upper_threshold'] = df['upper_threshold'].ffill().bfill()
        df['lower_threshold'] = df['lower_threshold'].ffill().bfill()
        
        # まだNaN値がある場合は0で置換
        df['upper_threshold'] = df['upper_threshold'].fillna(0)
        df['lower_threshold'] = df['lower_threshold'].fillna(0)
        
        print(f"データ前処理後 - トレンドNaN: {df['trend'].isna().sum()}, ピークNaN: {df['peaks_plot'].isna().sum()}")
        print(f"有効なピーク数: {df['peaks_plot'].notna().sum()}, 有効なバレー数: {df['valleys_plot'].notna().sum()}")
        
        # 各系列の値の範囲を確認
        print(f"バンドパス範囲: {df['bandpass'].min():.6f} - {df['bandpass'].max():.6f}")
        print(f"トレンド範囲: {df['trend'].min():.6f} - {df['trend'].max():.6f}")
        print(f"モード信号範囲: {df['mode_signal'].min():.6f} - {df['mode_signal'].max():.6f}")
        print(f"平均ピーク範囲: {df['avg_peak'].min():.6f} - {df['avg_peak'].max():.6f}")
        print(f"平均バレー範囲: {df['avg_valley'].min():.6f} - {df['avg_valley'].max():.6f}")
        
        # すべてのカラムに有効な値があることを確認
        for col in ['bandpass', 'trend', 'mode_signal', 'avg_peak', 'avg_valley', 'upper_threshold', 'lower_threshold']:
            valid_count = df[col].notna().sum()
            print(f"{col}: 有効値数 {valid_count}/{len(df)}")
            if valid_count == 0:
                print(f"警告: {col} にはすべてNaN値が含まれています")
                # すべてNaNの場合は0で置換
                df[col] = df[col].fillna(0)
        
        # mplfinanceでプロット用の設定
        # 1. メインチャート上のプロット（価格のみ）
        main_plots = []
        
        # 2. オシレータープロット（出来高によってパネル番号を調整）
        panel_plots = []
        
        # 出来高表示によるパネル番号の調整
        volume_offset = 1 if show_volume else 0
        
        try:
            # EMD本体パネル（トレンド成分 + 上限・下限チャネル）
            if df['trend'].notna().sum() > 0:
                emd_trend_panel = mpf.make_addplot(df['trend'], panel=1 + volume_offset, color='orange', width=2.0, 
                                                  ylabel='EMD Trend', secondary_y=False, label='EMD Trend')
                panel_plots.append(emd_trend_panel)
                print("EMDトレンド成分プロット追加")
            
            if df['upper_threshold'].notna().sum() > 0:
                emd_upper_panel = mpf.make_addplot(df['upper_threshold'], panel=1 + volume_offset, color='green', width=1.0, 
                                                  linestyle='--', alpha=0.7, secondary_y=False, label='Upper Channel')
                panel_plots.append(emd_upper_panel)
                print("EMD上限チャネルプロット追加")
            
            if df['lower_threshold'].notna().sum() > 0:
                emd_lower_panel = mpf.make_addplot(df['lower_threshold'], panel=1 + volume_offset, color='red', width=1.0, 
                                                  linestyle='--', alpha=0.7, secondary_y=False, label='Lower Channel')
                panel_plots.append(emd_lower_panel)
                print("EMD下限チャネルプロット追加")
            
            # バンドパスフィルター（サイクル成分）
            if df['bandpass'].notna().sum() > 0:
                bandpass_panel = mpf.make_addplot(df['bandpass'], panel=2 + volume_offset, color='blue', width=1.5, 
                                                 ylabel='Bandpass (Cycle)', secondary_y=False, label='Bandpass')
                panel_plots.append(bandpass_panel)
                print("バンドパスプロット追加")
            
            # モード信号
            if df['mode_signal'].notna().sum() > 0:
                mode_panel = mpf.make_addplot(df['mode_signal'], panel=3 + volume_offset, color='purple', width=2.0, 
                                             ylabel='Mode Signal', secondary_y=False, label='Mode Signal')
                panel_plots.append(mode_panel)
                print("モード信号プロット追加")
            
            # ピーク・バレー値
            if df['avg_peak'].notna().sum() > 0:
                peak_panel = mpf.make_addplot(df['avg_peak'], panel=4 + volume_offset, color='green', width=1.2, 
                                             ylabel='Peak/Valley', secondary_y=False, label='Avg Peak')
                panel_plots.append(peak_panel)
                print("ピーク平均プロット追加")
            
            if df['avg_valley'].notna().sum() > 0:
                valley_panel = mpf.make_addplot(df['avg_valley'], panel=4 + volume_offset, color='red', width=1.2, 
                                               secondary_y=False, label='Avg Valley')
                panel_plots.append(valley_panel)
                print("バレー平均プロット追加")
        except Exception as e:
            print(f"パネルプロット作成エラー: {e}")
            panel_plots = []
        
        # 個別のピーク・バレー検出点（有効なデータがある場合のみ）
        additional_plots = []
        
        # ピーク散布図: 有効値が10個以上ある場合のみ追加
        peaks_valid_count = df['peaks_plot'].notna().sum()
        if peaks_valid_count >= 10:
            try:
                peaks_scatter = mpf.make_addplot(df['peaks_plot'], panel=4 + volume_offset, type='scatter', markersize=20, 
                                                color='green', alpha=0.6, secondary_y=False, label='Peaks')
                additional_plots.append(peaks_scatter)
                print(f"ピーク散布図プロット追加 ({peaks_valid_count}個)")
            except Exception as e:
                print(f"ピーク散布図プロット作成エラー: {e}")
        else:
            print(f"ピーク散布図をスキップ (有効値数: {peaks_valid_count} < 10)")
        
        # バレー散布図: 有効値が10個以上ある場合のみ追加
        valleys_valid_count = df['valleys_plot'].notna().sum()
        if valleys_valid_count >= 10:
            try:
                valleys_scatter = mpf.make_addplot(df['valleys_plot'], panel=4 + volume_offset, type='scatter', markersize=20, 
                                                  color='red', alpha=0.6, secondary_y=False, label='Valleys')
                additional_plots.append(valleys_scatter)
                print(f"バレー散布図プロット追加 ({valleys_valid_count}個)")
            except Exception as e:
                print(f"バレー散布図プロット作成エラー: {e}")
        else:
            print(f"バレー散布図をスキップ (有効値数: {valleys_valid_count} < 10)")
        
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
        
        # パネル設定の修正（実際のパネル構成に基づく）
        if show_volume:
            kwargs['volume'] = True
            # 実際のパネル構成: メイン(0) + 出来高(1) + EMD本体(2) + バンドパス(3) + モード(4) + ピーク/バレー(5)
            # つまり、6つのパネル
            kwargs['panel_ratios'] = (4, 1, 1.8, 1.5, 1, 1.2)  # メイン:出来高:EMD本体:バンドパス:モード:ピーク/バレー
        else:
            kwargs['volume'] = False
            # 実際のパネル構成: メイン(0) + EMD本体(1) + バンドパス(2) + モード(3) + ピーク/バレー(4)
            # つまり、5つのパネル
            kwargs['panel_ratios'] = (4, 1.8, 1.5, 1, 1.2)  # メイン:EMD本体:バンドパス:モード:ピーク/バレー
        
        # すべてのプロットを結合
        all_plots = main_plots + panel_plots + additional_plots
        
        print(f"総プロット数: {len(all_plots)} (メイン: {len(main_plots)}, パネル: {len(panel_plots)}, 追加: {len(additional_plots)})")
        
        if len(all_plots) > 0:
            kwargs['addplot'] = all_plots
        else:
            print("警告: プロットするデータがありません。基本的なローソク足のみ表示します。")
            # addplotを設定しない（基本のローソク足と出来高のみ）
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加
        axes[0].legend(['Trend Component', 'Upper Threshold', 'Lower Threshold'], loc='upper left')
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        # volume_offsetを使用してパネル番号を調整
        
        # バンドパスパネル（0の線）
        bandpass_axis = axes[1 + volume_offset]
        bandpass_axis.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # モードシグナルパネル（-1, 0, 1の線）
        mode_axis = axes[2 + volume_offset]
        mode_axis.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        mode_axis.axhline(y=1, color='green', linestyle='--', alpha=0.5)
        mode_axis.axhline(y=-1, color='red', linestyle='--', alpha=0.5)
        
        # ピーク・バレーパネル（0の線）
        peak_valley_axis = axes[3 + volume_offset]
        peak_valley_axis.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
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
    parser = argparse.ArgumentParser(description='EMD (Empirical Mode Decomposition)の描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--src', type=str, default='hlc3', help='価格ソース (close, hlc3, etc.)')
    parser.add_argument('--period', type=int, default=20, help='フィルター周期')
    parser.add_argument('--delta', type=float, default=0.1, help='バンド幅パラメータ')
    parser.add_argument('--fraction', type=float, default=0.25, help='閾値計算用係数')
    parser.add_argument('--avg-period', type=int, default=50, help='ピーク・バレー平均化期間')
    args = parser.parse_args()
    
    # チャートを作成
    chart = EMDChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        src_type=args.src,
        period=args.period,
        delta=args.delta,
        fraction=args.fraction,
        avg_period=getattr(args, 'avg_period', 50)  # ハイフン付きの引数名を処理
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output
    )


if __name__ == "__main__":
    main()