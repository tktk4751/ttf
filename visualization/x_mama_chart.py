#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# データ取得のための依存関係
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource

# インジケーター
from indicators.x_mama import X_MAMA


class XMAMAChart:
    """
    X_MAMAを表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - X_MAMA（緑色）とX_FAMA（赤色）ライン
    - MAMAクロスオーバーシグナル
    - Period値とAlpha値の表示
    - InPhaseとQuadrature成分の表示
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.x_mama = None
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
                            # X_MAMAパラメータ
                            fast_limit: float = 0.5,
                            slow_limit: float = 0.05,
                            src_type: str = 'hlc3',
                            # カルマンフィルターパラメータ
                            use_kalman_filter: bool = False,
                            kalman_filter_type: str = 'unscented',
                            kalman_process_noise: float = 0.01,
                            kalman_observation_noise: float = 0.001,
                            # ゼロラグ処理パラメータ
                            use_zero_lag: bool = True
                           ) -> None:
        """
        X_MAMAインジケーターを計算する
        
        Args:
            fast_limit: 高速制限値
            slow_limit: 低速制限値
            src_type: ソースタイプ
            use_kalman_filter: カルマンフィルターを使用するか
            kalman_filter_type: カルマンフィルタータイプ
            kalman_process_noise: プロセスノイズ
            kalman_observation_noise: 観測ノイズ
            use_zero_lag: ゼロラグ処理を使用するか
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\\nX_MAMAインジケーターを計算中...")
        
        # X_MAMAを計算
        self.x_mama = X_MAMA(
            fast_limit=fast_limit,
            slow_limit=slow_limit,
            src_type=src_type,
            use_kalman_filter=use_kalman_filter,
            kalman_filter_type=kalman_filter_type,
            kalman_process_noise=kalman_process_noise,
            kalman_observation_noise=kalman_observation_noise,
            use_zero_lag=use_zero_lag
        )
        
        # X_MAMAの計算
        print("計算を実行します...")
        result = self.x_mama.calculate(self.data)
        
        # 結果の取得テスト
        mama_values = result.mama_values
        fama_values = result.fama_values
        period_values = result.period_values
        alpha_values = result.alpha_values
        
        print(f"X_MAMA計算完了 - MAMA: {len(mama_values)}, FAMA: {len(fama_values)}")
        print(f"Period: {len(period_values)}, Alpha: {len(alpha_values)}")
        
        # NaN値のチェック
        nan_count_mama = np.isnan(mama_values).sum()
        nan_count_fama = np.isnan(fama_values).sum()
        nan_count_period = np.isnan(period_values).sum()
        nan_count_alpha = np.isnan(alpha_values).sum()
        
        print(f"NaN値 - MAMA: {nan_count_mama}, FAMA: {nan_count_fama}")
        print(f"NaN値 - Period: {nan_count_period}, Alpha: {nan_count_alpha}")
        
        # 統計情報
        print(f"MAMA - 平均: {np.nanmean(mama_values):.4f}, 範囲: {np.nanmin(mama_values):.4f} - {np.nanmax(mama_values):.4f}")
        print(f"FAMA - 平均: {np.nanmean(fama_values):.4f}, 範囲: {np.nanmin(fama_values):.4f} - {np.nanmax(fama_values):.4f}")
        print(f"Period - 平均: {np.nanmean(period_values):.2f}, 範囲: {np.nanmin(period_values):.2f} - {np.nanmax(period_values):.2f}")
        print(f"Alpha - 平均: {np.nanmean(alpha_values):.4f}, 範囲: {np.nanmin(alpha_values):.4f} - {np.nanmax(alpha_values):.4f}")
        
        print("X_MAMAインジケーター計算完了")
            
    def plot(self, 
            title: str = "X_MAMA (eXtended Mother of Adaptive Moving Average)", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            show_crossovers: bool = True,
            figsize: Tuple[int, int] = (14, 12),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとX_MAMAを描画する
        
        Args:
            title: チャートのタイトル
            start_date: 表示開始日（フォーマット: YYYY-MM-DD）
            end_date: 表示終了日（フォーマット: YYYY-MM-DD）
            show_volume: 出来高を表示するか
            show_crossovers: クロスオーバーシグナルを表示するか
            figsize: 図のサイズ
            style: mplfinanceのスタイル
            savefig: 保存先のパス（指定しない場合は表示のみ）
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        if self.x_mama is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # X_MAMAの値を取得
        print("X_MAMAデータを取得中...")
        result = self.x_mama.calculate(self.data)
        
        mama_values = result.mama_values
        fama_values = result.fama_values
        period_values = result.period_values
        alpha_values = result.alpha_values
        phase_values = result.phase_values
        i1_values = result.i1_values
        q1_values = result.q1_values
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'x_mama': mama_values,
                'x_fama': fama_values,
                'period': period_values,
                'alpha': alpha_values,
                'phase': phase_values,
                'i1': i1_values,
                'q1': q1_values
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        print(f"X_MAMAデータ確認 - MAMA NaN: {df['x_mama'].isna().sum()}, FAMA NaN: {df['x_fama'].isna().sum()}")
        
        # クロスオーバーシグナルの計算
        if show_crossovers:
            df['mama_above_fama'] = df['x_mama'] > df['x_fama']
            df['crossover_up'] = (df['mama_above_fama'] == True) & (df['mama_above_fama'].shift(1) == False)
            df['crossover_down'] = (df['mama_above_fama'] == False) & (df['mama_above_fama'].shift(1) == True)
            
            # クロスオーバーポイントの価格を取得
            df['crossover_up_price'] = np.where(df['crossover_up'], df['close'], np.nan)
            df['crossover_down_price'] = np.where(df['crossover_down'], df['close'], np.nan)
        
        # NaN値を含む行を出力（最初の5行のみ）
        nan_rows = df[df['x_mama'].isna() | df['x_fama'].isna()]
        if not nan_rows.empty:
            print(f"NaN値を含む行: {len(nan_rows)}行")
            if len(nan_rows) > 0:
                print(nan_rows.head())
        
        # mplfinanceでプロット用の設定
        # 1. メインチャート上のプロット
        main_plots = []
        
        # X_MAMAとX_FAMAのプロット設定
        main_plots.append(mpf.make_addplot(df['x_mama'], color='green', width=2, label='X_MAMA'))
        main_plots.append(mpf.make_addplot(df['x_fama'], color='red', width=2, label='X_FAMA'))
        
        # クロスオーバーシグナルの追加
        if show_crossovers:
            main_plots.append(mpf.make_addplot(df['crossover_up_price'], type='scatter', 
                                             markersize=100, marker='^', color='green', 
                                             alpha=0.8, label='Bullish Cross'))
            main_plots.append(mpf.make_addplot(df['crossover_down_price'], type='scatter', 
                                             markersize=100, marker='v', color='red', 
                                             alpha=0.8, label='Bearish Cross'))
        
        # 2. オシレータープロット
        # Period値パネル
        period_panel = mpf.make_addplot(df['period'], panel=1, color='blue', width=1.2, 
                                       ylabel='Period', secondary_y=False, label='Period')
        
        # Alpha値パネル
        alpha_panel = mpf.make_addplot(df['alpha'], panel=2, color='purple', width=1.2, 
                                      ylabel='Alpha', secondary_y=False, label='Alpha')
        
        # Phase値パネル
        phase_panel = mpf.make_addplot(df['phase'], panel=3, color='orange', width=1.2, 
                                      ylabel='Phase', secondary_y=False, label='Phase')
        
        # InPhase & Quadrature パネル
        i1_panel = mpf.make_addplot(df['i1'], panel=4, color='cyan', width=1.0, 
                                   ylabel='I1 & Q1', secondary_y=False, label='I1')
        q1_panel = mpf.make_addplot(df['q1'], panel=4, color='magenta', width=1.0, 
                                   ylabel='I1 & Q1', secondary_y=False, label='Q1')
        
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
            kwargs['panel_ratios'] = (4, 1, 1, 1, 1, 1)  # メイン:出来高:Period:Alpha:Phase:I1&Q1
            # 出来高を表示する場合は、オシレーターのパネル番号を+1する
            period_panel = mpf.make_addplot(df['period'], panel=2, color='blue', width=1.2, 
                                           ylabel='Period', secondary_y=False, label='Period')
            alpha_panel = mpf.make_addplot(df['alpha'], panel=3, color='purple', width=1.2, 
                                          ylabel='Alpha', secondary_y=False, label='Alpha')
            phase_panel = mpf.make_addplot(df['phase'], panel=4, color='orange', width=1.2, 
                                          ylabel='Phase', secondary_y=False, label='Phase')
            i1_panel = mpf.make_addplot(df['i1'], panel=5, color='cyan', width=1.0, 
                                       ylabel='I1 & Q1', secondary_y=False, label='I1')
            q1_panel = mpf.make_addplot(df['q1'], panel=5, color='magenta', width=1.0, 
                                       ylabel='I1 & Q1', secondary_y=False, label='Q1')
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (4, 1, 1, 1, 1)  # メイン:Period:Alpha:Phase:I1&Q1
        
        # すべてのプロットを結合
        all_plots = main_plots + [period_panel, alpha_panel, phase_panel, i1_panel, q1_panel]
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加
        legend_labels = ['X_MAMA', 'X_FAMA']
        if show_crossovers:
            legend_labels.extend(['Bullish Cross', 'Bearish Cross'])
        axes[0].legend(legend_labels, loc='upper left')
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        panel_offset = 1 if show_volume else 0
        
        # Period値パネル
        period_mean = df['period'].mean()
        axes[1 + panel_offset].axhline(y=period_mean, color='black', linestyle='-', alpha=0.3)
        axes[1 + panel_offset].axhline(y=20, color='blue', linestyle='--', alpha=0.5)  # 標準的な期間
        
        # Alpha値パネル
        alpha_mean = df['alpha'].mean()
        axes[2 + panel_offset].axhline(y=alpha_mean, color='black', linestyle='-', alpha=0.3)
        axes[2 + panel_offset].axhline(y=0.5, color='purple', linestyle='--', alpha=0.5)  # 高速制限
        axes[2 + panel_offset].axhline(y=0.05, color='purple', linestyle='--', alpha=0.5)  # 低速制限
        
        # Phase値パネル
        axes[3 + panel_offset].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[3 + panel_offset].axhline(y=90, color='orange', linestyle='--', alpha=0.5)
        axes[3 + panel_offset].axhline(y=-90, color='orange', linestyle='--', alpha=0.5)
        
        # I1&Q1パネル
        axes[4 + panel_offset].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 統計情報の表示
        print(f"\\n=== X_MAMA統計 ===")
        valid_points = len(df.dropna(subset=['x_mama', 'x_fama']))
        
        if show_crossovers:
            bullish_crosses = df['crossover_up'].sum()
            bearish_crosses = df['crossover_down'].sum()
            print(f"クロスオーバー - 強気: {bullish_crosses}, 弱気: {bearish_crosses}")
        
        print(f"有効データ点数: {valid_points}")
        print(f"X_MAMA - 平均: {df['x_mama'].mean():.4f}, 標準偏差: {df['x_mama'].std():.4f}")
        print(f"X_FAMA - 平均: {df['x_fama'].mean():.4f}, 標準偏差: {df['x_fama'].std():.4f}")
        print(f"Period - 平均: {df['period'].mean():.2f}, 範囲: {df['period'].min():.2f} - {df['period'].max():.2f}")
        print(f"Alpha - 平均: {df['alpha'].mean():.4f}, 範囲: {df['alpha'].min():.4f} - {df['alpha'].max():.4f}")
        
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
    parser = argparse.ArgumentParser(description='X_MAMAの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--src-type', type=str, default='hlc3', help='ソースタイプ')
    parser.add_argument('--fast-limit', type=float, default=0.5, help='高速制限値')
    parser.add_argument('--slow-limit', type=float, default=0.05, help='低速制限値')
    parser.add_argument('--use-kalman', action='store_true',default=True, help='カルマンフィルターを使用')
    parser.add_argument('--use-zero-lag', action='store_true', default=True, help='ゼロラグ処理を使用')
    parser.add_argument('--no-crossovers', action='store_true', help='クロスオーバーシグナルを非表示')
    args = parser.parse_args()
    
    # チャートを作成
    chart = XMAMAChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        fast_limit=args.fast_limit,
        slow_limit=args.slow_limit,
        src_type=args.src_type,
        use_kalman_filter=args.use_kalman,
        use_zero_lag=args.use_zero_lag
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        show_crossovers=not args.no_crossovers,
        savefig=args.output
    )


if __name__ == "__main__":
    main()