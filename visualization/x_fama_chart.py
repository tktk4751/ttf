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
from indicators.x_fama import X_FAMA


class XFAMAChart:
    """
    X_FAMAを表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - X_FRAMA値（通常）
    - Fast X_FRAMA値（高速線）
    - フラクタル次元
    - アルファ値
    - トレンド判定用の交差シグナル
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.x_fama = None
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
                            period: int = 16,
                            src_type: str = 'hl2',
                            fc: int = 1,
                            sc: int = 198,
                            use_kalman_filter: bool = False,
                            kalman_filter_type: str = 'unscented',
                            kalman_process_noise: float = 0.01,
                            kalman_observation_noise: float = 0.001,
                            use_zero_lag: bool = True
                           ) -> None:
        """
        X_FAMAを計算する
        
        Args:
            period: 期間（偶数である必要がある、デフォルト: 16）
            src_type: ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4')
            fc: Fast Constant（デフォルト: 1）
            sc: Slow Constant（デフォルト: 198）
            use_kalman_filter: カルマンフィルターを使用するか（デフォルト: False）
            kalman_filter_type: カルマンフィルタータイプ（デフォルト: 'unscented'）
            kalman_process_noise: プロセスノイズ（デフォルト: 0.01）
            kalman_observation_noise: 観測ノイズ（デフォルト: 0.001）
            use_zero_lag: ゼロラグ処理を使用するか（デフォルト: True）
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\nX_FAMAを計算中...")
        
        # X_FAMAを計算
        self.x_fama = X_FAMA(
            period=period,
            src_type=src_type,
            fc=fc,
            sc=sc,
            use_kalman_filter=use_kalman_filter,
            kalman_filter_type=kalman_filter_type,
            kalman_process_noise=kalman_process_noise,
            kalman_observation_noise=kalman_observation_noise,
            use_zero_lag=use_zero_lag
        )
        
        # X_FAMAの計算
        print("計算を実行します...")
        self.x_fama.calculate(self.data)
        
        # 結果の取得テスト
        frama_values = self.x_fama.get_frama_values()
        fast_fama_values = self.x_fama.get_fast_fama_values()
        fractal_dim = self.x_fama.get_fractal_dimension()
        alpha_values = self.x_fama.get_alpha_values()
        
        print(f"計算完了 - FRAMA: {len(frama_values)}, Fast FRAMA: {len(fast_fama_values)}")
        print(f"Fractal Dimension: {len(fractal_dim)}, Alpha: {len(alpha_values)}")
        
        # NaN値のチェック
        nan_count_frama = np.isnan(frama_values).sum()
        nan_count_fast = np.isnan(fast_fama_values).sum()
        nan_count_dim = np.isnan(fractal_dim).sum()
        nan_count_alpha = np.isnan(alpha_values).sum()
        print(f"NaN値 - FRAMA: {nan_count_frama}, Fast FRAMA: {nan_count_fast}")
        print(f"Fractal Dim: {nan_count_dim}, Alpha: {nan_count_alpha}")
        
        # 統計情報
        print(f"FRAMA範囲: {np.nanmin(frama_values):.4f} - {np.nanmax(frama_values):.4f}")
        print(f"Fast FRAMA範囲: {np.nanmin(fast_fama_values):.4f} - {np.nanmax(fast_fama_values):.4f}")
        print(f"フラクタル次元範囲: {np.nanmin(fractal_dim):.4f} - {np.nanmax(fractal_dim):.4f}")
        print(f"アルファ値範囲: {np.nanmin(alpha_values):.4f} - {np.nanmax(alpha_values):.4f}")
        
        print("X_FAMA計算完了")
            
    def plot(self, 
            title: str = "X_FAMA（拡張フラクタル適応移動平均）", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (14, 12),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとX_FAMAを描画する
        
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
            
        if self.x_fama is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # X_FAMAの値を取得
        print("X_FAMAデータを取得中...")
        frama_values = self.x_fama.get_frama_values()
        fast_fama_values = self.x_fama.get_fast_fama_values()
        fractal_dim = self.x_fama.get_fractal_dimension()
        alpha_values = self.x_fama.get_alpha_values()
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'x_frama': frama_values,
                'fast_x_frama': fast_fama_values,
                'fractal_dimension': fractal_dim,
                'alpha_values': alpha_values
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        print(f"FRAMA データ確認 - NaN: {df['x_frama'].isna().sum()}, Fast FRAMA NaN: {df['fast_x_frama'].isna().sum()}")
        
        # トレンド判定用のクロスオーバーシグナル計算
        # Fast FRAMA > FRAMA の場合は上昇トレンド、逆は下降トレンド
        df['trend_signal'] = np.where(df['fast_x_frama'] > df['x_frama'], 1, -1)
        
        # クロスオーバーポイントを検出
        df['cross_signal'] = df['trend_signal'].diff()
        df['bullish_cross'] = np.where(df['cross_signal'] == 2, df['x_frama'], np.nan)
        df['bearish_cross'] = np.where(df['cross_signal'] == -2, df['x_frama'], np.nan)
        
        # トレンドに基づく色分け用データの準備
        df['frama_up'] = np.where(df['trend_signal'] == 1, df['x_frama'], np.nan)
        df['frama_down'] = np.where(df['trend_signal'] == -1, df['x_frama'], np.nan)
        df['fast_frama_up'] = np.where(df['trend_signal'] == 1, df['fast_x_frama'], np.nan)
        df['fast_frama_down'] = np.where(df['trend_signal'] == -1, df['fast_x_frama'], np.nan)
        
        # mplfinanceでプロット用の設定
        # 1. メインチャート上のプロット
        main_plots = []
        
        # X_FAMAのプロット設定（トレンドに応じた色分け）
        main_plots.append(mpf.make_addplot(df['frama_up'], color='green', width=2.5, label='X_FRAMA (Up)'))
        main_plots.append(mpf.make_addplot(df['frama_down'], color='red', width=2.5, label='X_FRAMA (Down)'))
        main_plots.append(mpf.make_addplot(df['fast_frama_up'], color='lightgreen', width=2, label='Fast X_FRAMA (Up)'))
        main_plots.append(mpf.make_addplot(df['fast_frama_down'], color='orange', width=2, label='Fast X_FRAMA (Down)'))
        
        # クロスオーバーシグナル
        main_plots.append(mpf.make_addplot(df['bullish_cross'], type='scatter', markersize=100, marker='^', color='green', alpha=0.8, label='Bullish Cross'))
        main_plots.append(mpf.make_addplot(df['bearish_cross'], type='scatter', markersize=100, marker='v', color='red', alpha=0.8, label='Bearish Cross'))
        
        # 2. オシレータープロット
        # フラクタル次元パネル
        fractal_panel = mpf.make_addplot(df['fractal_dimension'], panel=1, color='purple', width=1.5, 
                                        ylabel='Fractal Dimension', secondary_y=False, label='Fractal Dim')
        
        # アルファ値パネル
        alpha_panel = mpf.make_addplot(df['alpha_values'], panel=2, color='blue', width=1.5, 
                                      ylabel='Alpha Values', secondary_y=False, label='Alpha')
        
        # トレンドシグナルパネル
        trend_panel = mpf.make_addplot(df['trend_signal'], panel=3, color='orange', width=2, 
                                      ylabel='Trend Signal', secondary_y=False, label='Trend', type='line')
        
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
            kwargs['panel_ratios'] = (4, 1, 1, 1, 1)  # メイン:出来高:フラクタル:アルファ:トレンド
            # 出来高を表示する場合は、オシレーターのパネル番号を+1する
            fractal_panel = mpf.make_addplot(df['fractal_dimension'], panel=2, color='purple', width=1.5, 
                                            ylabel='Fractal Dimension', secondary_y=False, label='Fractal Dim')
            alpha_panel = mpf.make_addplot(df['alpha_values'], panel=3, color='blue', width=1.5, 
                                          ylabel='Alpha Values', secondary_y=False, label='Alpha')
            trend_panel = mpf.make_addplot(df['trend_signal'], panel=4, color='orange', width=2, 
                                          ylabel='Trend Signal', secondary_y=False, label='Trend', type='line')
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (4, 1, 1, 1)  # メイン:フラクタル:アルファ:トレンド
        
        # すべてのプロットを結合
        all_plots = main_plots + [fractal_panel, alpha_panel, trend_panel]
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加
        axes[0].legend(['X_FRAMA (Up)', 'X_FRAMA (Down)', 'Fast X_FRAMA (Up)', 'Fast X_FRAMA (Down)', 
                       'Bullish Cross', 'Bearish Cross'], loc='upper left')
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        if show_volume:
            # フラクタル次元パネル
            axes[2].axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='D=1 (Trend)')
            axes[2].axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='D=2 (Random)')
            axes[2].axhline(y=1.5, color='black', linestyle='-', alpha=0.3, label='D=1.5 (Mid)')
            
            # アルファ値パネル
            axes[3].axhline(y=0.01, color='red', linestyle='--', alpha=0.5, label='Min Alpha')
            axes[3].axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Max Alpha')
            axes[3].axhline(y=0.5, color='black', linestyle='-', alpha=0.3, label='Mid Alpha')
            
            # トレンドシグナルパネル
            axes[4].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[4].axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Bullish')
            axes[4].axhline(y=-1, color='red', linestyle='--', alpha=0.5, label='Bearish')
        else:
            # フラクタル次元パネル
            axes[1].axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='D=1 (Trend)')
            axes[1].axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='D=2 (Random)')
            axes[1].axhline(y=1.5, color='black', linestyle='-', alpha=0.3, label='D=1.5 (Mid)')
            
            # アルファ値パネル
            axes[2].axhline(y=0.01, color='red', linestyle='--', alpha=0.5, label='Min Alpha')
            axes[2].axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Max Alpha')
            axes[2].axhline(y=0.5, color='black', linestyle='-', alpha=0.3, label='Mid Alpha')
            
            # トレンドシグナルパネル
            axes[3].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[3].axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Bullish')
            axes[3].axhline(y=-1, color='red', linestyle='--', alpha=0.5, label='Bearish')
        
        # 統計情報の表示
        print(f"\n=== X_FRAMA統計 ===")
        total_points = len(df[df['trend_signal'] != 0])
        bullish_points = len(df[df['trend_signal'] == 1])
        bearish_points = len(df[df['trend_signal'] == -1])
        
        print(f"総データ点数: {total_points}")
        print(f"強気トレンド: {bullish_points} ({bullish_points/total_points*100:.1f}%)")
        print(f"弱気トレンド: {bearish_points} ({bearish_points/total_points*100:.1f}%)")
        
        # クロスオーバー統計
        bullish_crosses = df['bullish_cross'].notna().sum()
        bearish_crosses = df['bearish_cross'].notna().sum()
        print(f"強気クロス: {bullish_crosses}回")
        print(f"弱気クロス: {bearish_crosses}回")
        
        # フラクタル次元とアルファ値の統計
        print(f"フラクタル次元 - 平均: {df['fractal_dimension'].mean():.3f}, 範囲: {df['fractal_dimension'].min():.3f} - {df['fractal_dimension'].max():.3f}")
        print(f"アルファ値 - 平均: {df['alpha_values'].mean():.4f}, 範囲: {df['alpha_values'].min():.4f} - {df['alpha_values'].max():.4f}")
        
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
    parser = argparse.ArgumentParser(description='X_FAMAの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--period', type=int, default=16, help='期間')
    parser.add_argument('--src-type', type=str, default='hl2', help='ソースタイプ')
    parser.add_argument('--fc', type=int, default=1, help='Fast Constant')
    parser.add_argument('--sc', type=int, default=198, help='Slow Constant')
    parser.add_argument('--kalman', action='store_true', help='カルマンフィルターを使用')
    parser.add_argument('--no-zero-lag', action='store_true', help='ゼロラグ処理を無効化')
    args = parser.parse_args()
    
    # チャートを作成
    chart = XFAMAChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        period=args.period,
        src_type=args.src_type,
        fc=args.fc,
        sc=args.sc,
        use_kalman_filter=args.kalman,
        use_zero_lag=not args.no_zero_lag
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output
    )


if __name__ == "__main__":
    main()