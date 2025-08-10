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
from indicators.mesa_frama import MESA_FRAMA


class MESAFRAMAChart:
    """
    MESA_FRAMAを表示するローソク足チャートクラス
    
    特徴:
    - ローソク足と出来高
    - デュアルMESA_FRAMA（短期線・長期線）の表示
    - MESA適応期間による動的調整
    - フラクタル次元とMESA位相の表示
    - 動的期間とアルファ値の表示
    - クロスオーバーシグナルの表示
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.fast_mesa_frama = None
        self.slow_mesa_frama = None
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
                            # 短期MESA_FRAMAパラメータ
                            fast_base_period: int = 8,
                            fast_src_type: str = 'close',
                            fast_fc: int = 1,
                            fast_sc: int = 198,
                            fast_mesa_fast_limit: float = 0.8,
                            fast_mesa_slow_limit: float = 0.15,
                            # 長期MESA_FRAMAパラメータ
                            slow_base_period: int = 32,
                            slow_src_type: str = 'hl2',
                            slow_fc: int = 1,
                            slow_sc: int = 198,
                            slow_mesa_fast_limit: float = 0.3,
                            slow_mesa_slow_limit: float = 0.02,
                            # 共通パラメータ
                            use_kalman_filter: bool = False,
                            kalman_filter_type: str = 'unscented',
                            kalman_process_noise: float = 0.01,
                            kalman_observation_noise: float = 0.001,
                            use_zero_lag: bool = True
                           ) -> None:
        """
        デュアルMESA_FRAMAを計算する
        
        Args:
            fast_base_period: 短期基本期間
            fast_src_type: 短期ソースタイプ
            fast_fc: 短期Fast Constant
            fast_sc: 短期Slow Constant
            fast_mesa_fast_limit: 短期MESA高速制限値
            fast_mesa_slow_limit: 短期MESA低速制限値
            slow_base_period: 長期基本期間
            slow_src_type: 長期ソースタイプ
            slow_fc: 長期Fast Constant
            slow_sc: 長期Slow Constant
            slow_mesa_fast_limit: 長期MESA高速制限値
            slow_mesa_slow_limit: 長期MESA低速制限値
            use_kalman_filter: カルマンフィルターを使用するか
            kalman_filter_type: カルマンフィルタータイプ
            kalman_process_noise: プロセスノイズ
            kalman_observation_noise: 観測ノイズ
            use_zero_lag: ゼロラグ処理を使用するか
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\nデュアルMESA_FRAMAを計算中...")
        
        # 短期MESA_FRAMAを計算
        print("短期MESA_FRAMAを計算中...")
        self.fast_mesa_frama = MESA_FRAMA(
            base_period=fast_base_period,
            src_type=fast_src_type,
            fc=fast_fc,
            sc=fast_sc,
            mesa_fast_limit=fast_mesa_fast_limit,
            mesa_slow_limit=fast_mesa_slow_limit,
            use_kalman_filter=use_kalman_filter,
            kalman_filter_type=kalman_filter_type,
            kalman_process_noise=kalman_process_noise,
            kalman_observation_noise=kalman_observation_noise,
            use_zero_lag=use_zero_lag
        )
        
        # 長期MESA_FRAMAを計算
        print("長期MESA_FRAMAを計算中...")
        self.slow_mesa_frama = MESA_FRAMA(
            base_period=slow_base_period,
            src_type=slow_src_type,
            fc=slow_fc,
            sc=slow_sc,
            mesa_fast_limit=slow_mesa_fast_limit,
            mesa_slow_limit=slow_mesa_slow_limit,
            use_kalman_filter=use_kalman_filter,
            kalman_filter_type=kalman_filter_type,
            kalman_process_noise=kalman_process_noise,
            kalman_observation_noise=kalman_observation_noise,
            use_zero_lag=use_zero_lag
        )
        
        # MESA_FRAMAの計算実行
        print("計算を実行します...")
        fast_result = self.fast_mesa_frama.calculate(self.data)
        slow_result = self.slow_mesa_frama.calculate(self.data)
        
        if fast_result is None or slow_result is None:
            raise ValueError("MESA_FRAMAの計算に失敗しました")
        
        # 値の取得テスト
        fast_values = self.fast_mesa_frama.get_values()
        slow_values = self.slow_mesa_frama.get_values()
        fast_periods = self.fast_mesa_frama.get_dynamic_periods()
        slow_periods = self.slow_mesa_frama.get_dynamic_periods()
        fast_fractal = self.fast_mesa_frama.get_fractal_dimension()
        slow_fractal = self.slow_mesa_frama.get_fractal_dimension()
        
        print(f"計算完了 - 短期MESA_FRAMA: {len(fast_values)}, 長期MESA_FRAMA: {len(slow_values)}")
        
        # NaN値のチェック
        fast_nan_count = np.isnan(fast_values).sum()
        slow_nan_count = np.isnan(slow_values).sum()
        print(f"NaN値 - 短期: {fast_nan_count}, 長期: {slow_nan_count}")
        print(f"短期MESA_FRAMA - 平均: {np.nanmean(fast_values):.4f}, 範囲: {np.nanmin(fast_values):.4f} - {np.nanmax(fast_values):.4f}")
        print(f"長期MESA_FRAMA - 平均: {np.nanmean(slow_values):.4f}, 範囲: {np.nanmin(slow_values):.4f} - {np.nanmax(slow_values):.4f}")
        print(f"短期動的期間 - 平均: {np.nanmean(fast_periods):.2f}, 範囲: {np.nanmin(fast_periods):.2f} - {np.nanmax(fast_periods):.2f}")
        print(f"長期動的期間 - 平均: {np.nanmean(slow_periods):.2f}, 範囲: {np.nanmin(slow_periods):.2f} - {np.nanmax(slow_periods):.2f}")
        
        print("デュアルMESA_FRAMA計算完了")
            
    def plot(self, 
            title: str = "デュアルMESA_FRAMA", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            show_crossover_signals: bool = True,
            figsize: Tuple[int, int] = (14, 16),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとデュアルMESA_FRAMAを描画する
        
        Args:
            title: チャートのタイトル
            start_date: 表示開始日（フォーマット: YYYY-MM-DD）
            end_date: 表示終了日（フォーマット: YYYY-MM-DD）
            show_volume: 出来高を表示するか
            show_crossover_signals: クロスオーバーシグナルを表示するか
            figsize: 図のサイズ
            style: mplfinanceのスタイル
            savefig: 保存先のパス（指定しない場合は表示のみ）
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        if self.fast_mesa_frama is None or self.slow_mesa_frama is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # MESA_FRAMAの値を取得
        print("MESA_FRAMAデータを取得中...")
        fast_values = self.fast_mesa_frama.get_values()
        slow_values = self.slow_mesa_frama.get_values()
        fast_periods = self.fast_mesa_frama.get_dynamic_periods()
        slow_periods = self.slow_mesa_frama.get_dynamic_periods()
        fast_fractal = self.fast_mesa_frama.get_fractal_dimension()
        slow_fractal = self.slow_mesa_frama.get_fractal_dimension()
        fast_mesa_phase = self.fast_mesa_frama.get_mesa_phase()
        slow_mesa_phase = self.slow_mesa_frama.get_mesa_phase()
        fast_alpha = self.fast_mesa_frama.get_alpha()
        slow_alpha = self.slow_mesa_frama.get_alpha()
        
        # クロスオーバーシグナルの計算
        crossover_signals = np.zeros(len(fast_values), dtype=np.int8)
        for i in range(1, len(fast_values)):
            if (not np.isnan(fast_values[i]) and not np.isnan(slow_values[i]) and 
                not np.isnan(fast_values[i-1]) and not np.isnan(slow_values[i-1])):
                
                # ゴールデンクロス（短期が長期を上抜け）
                if fast_values[i-1] <= slow_values[i-1] and fast_values[i] > slow_values[i]:
                    crossover_signals[i] = 1
                # デッドクロス（短期が長期を下抜け）
                elif fast_values[i-1] >= slow_values[i-1] and fast_values[i] < slow_values[i]:
                    crossover_signals[i] = -1
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'fast_mesa_frama': fast_values,
                'slow_mesa_frama': slow_values,
                'fast_dynamic_period': fast_periods,
                'slow_dynamic_period': slow_periods,
                'fast_fractal_dimension': fast_fractal,
                'slow_fractal_dimension': slow_fractal,
                'fast_mesa_phase': fast_mesa_phase,
                'slow_mesa_phase': slow_mesa_phase,
                'fast_alpha': fast_alpha,
                'slow_alpha': slow_alpha,
                'crossover_signals': crossover_signals
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        print(f"MESA_FRAMAデータ確認 - 短期NaN: {df['fast_mesa_frama'].isna().sum()}, 長期NaN: {df['slow_mesa_frama'].isna().sum()}")
        
        # クロスオーバーシグナルのマーカー準備（データ絞り込み後に適用）
        golden_cross_mask = df['crossover_signals'] == 1
        dead_cross_mask = df['crossover_signals'] == -1
        
        # クロスオーバーポイントを価格データから抽出
        golden_cross_points = np.where(golden_cross_mask, df['close'], np.nan)
        dead_cross_points = np.where(dead_cross_mask, df['close'], np.nan)
        
        golden_cross_count = np.sum(golden_cross_mask)
        dead_cross_count = np.sum(dead_cross_mask)
        print(f"クロスオーバーシグナル - ゴールデンクロス: {golden_cross_count}, デッドクロス: {dead_cross_count}")
        
        # mplfinanceでプロット用の設定
        # 1. メインチャート上のプロット
        main_plots = []
        
        # MESA_FRAMAのプロット設定
        main_plots.append(mpf.make_addplot(df['fast_mesa_frama'], color='blue', width=2, label='Fast MESA_FRAMA'))
        main_plots.append(mpf.make_addplot(df['slow_mesa_frama'], color='red', width=2, label='Slow MESA_FRAMA'))
        
        # クロスオーバーシグナルのマーカー
        if show_crossover_signals and (golden_cross_count > 0 or dead_cross_count > 0):
            if golden_cross_count > 0:
                main_plots.append(mpf.make_addplot(golden_cross_points, type='scatter', 
                                                 markersize=60, marker='^', color='green', 
                                                 label='Golden Cross'))
            if dead_cross_count > 0:
                main_plots.append(mpf.make_addplot(dead_cross_points, type='scatter', 
                                                 markersize=60, marker='v', color='red', 
                                                 label='Dead Cross'))
        
        # 2. オシレータープロット
        # 動的期間パネル
        period_panel = mpf.make_addplot(df['fast_dynamic_period'], panel=1, color='blue', width=1.2, 
                                       ylabel='Dynamic Period', secondary_y=False, label='Fast Period')
        period_panel_slow = mpf.make_addplot(df['slow_dynamic_period'], panel=1, color='red', width=1.2, 
                                           secondary_y=False, label='Slow Period')
        
        # フラクタル次元パネル
        fractal_panel = mpf.make_addplot(df['fast_fractal_dimension'], panel=2, color='purple', width=1.2, 
                                        ylabel='Fractal Dimension', secondary_y=False, label='Fast Fractal')
        fractal_panel_slow = mpf.make_addplot(df['slow_fractal_dimension'], panel=2, color='orange', width=1.2, 
                                             secondary_y=False, label='Slow Fractal')
        
        # MESA位相パネル
        phase_panel = mpf.make_addplot(df['fast_mesa_phase'], panel=3, color='cyan', width=1.2, 
                                      ylabel='MESA Phase', secondary_y=False, label='Fast Phase')
        phase_panel_slow = mpf.make_addplot(df['slow_mesa_phase'], panel=3, color='magenta', width=1.2, 
                                           secondary_y=False, label='Slow Phase')
        
        # アルファ値パネル
        alpha_panel = mpf.make_addplot(df['fast_alpha'], panel=4, color='green', width=1.2, 
                                      ylabel='Alpha Values', secondary_y=False, label='Fast Alpha')
        alpha_panel_slow = mpf.make_addplot(df['slow_alpha'], panel=4, color='brown', width=1.2, 
                                           secondary_y=False, label='Slow Alpha')
        
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
            kwargs['panel_ratios'] = (4, 1, 1, 1, 1, 1)  # メイン:出来高:期間:フラクタル:位相:アルファ
            # 出来高を表示する場合は、オシレーターのパネル番号を+1する
            period_panel = mpf.make_addplot(df['fast_dynamic_period'], panel=2, color='blue', width=1.2, 
                                           ylabel='Dynamic Period', secondary_y=False, label='Fast Period')
            period_panel_slow = mpf.make_addplot(df['slow_dynamic_period'], panel=2, color='red', width=1.2, 
                                               secondary_y=False, label='Slow Period')
            fractal_panel = mpf.make_addplot(df['fast_fractal_dimension'], panel=3, color='purple', width=1.2, 
                                            ylabel='Fractal Dimension', secondary_y=False, label='Fast Fractal')
            fractal_panel_slow = mpf.make_addplot(df['slow_fractal_dimension'], panel=3, color='orange', width=1.2, 
                                                 secondary_y=False, label='Slow Fractal')
            phase_panel = mpf.make_addplot(df['fast_mesa_phase'], panel=4, color='cyan', width=1.2, 
                                          ylabel='MESA Phase', secondary_y=False, label='Fast Phase')
            phase_panel_slow = mpf.make_addplot(df['slow_mesa_phase'], panel=4, color='magenta', width=1.2, 
                                               secondary_y=False, label='Slow Phase')
            alpha_panel = mpf.make_addplot(df['fast_alpha'], panel=5, color='green', width=1.2, 
                                          ylabel='Alpha Values', secondary_y=False, label='Fast Alpha')
            alpha_panel_slow = mpf.make_addplot(df['slow_alpha'], panel=5, color='brown', width=1.2, 
                                               secondary_y=False, label='Slow Alpha')
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (4, 1, 1, 1, 1)  # メイン:期間:フラクタル:位相:アルファ
        
        # すべてのプロットを結合
        all_plots = main_plots + [period_panel, period_panel_slow, fractal_panel, fractal_panel_slow,
                                 phase_panel, phase_panel_slow, alpha_panel, alpha_panel_slow]
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加
        legend_labels = ['Fast MESA_FRAMA', 'Slow MESA_FRAMA']
        if show_crossover_signals:
            if golden_cross_count > 0:
                legend_labels.append('Golden Cross')
            if dead_cross_count > 0:
                legend_labels.append('Dead Cross')
        
        axes[0].legend(legend_labels, loc='upper left')
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        panel_offset = 1 if show_volume else 0
        
        # 動的期間パネル
        period_mean = (df['fast_dynamic_period'].mean() + df['slow_dynamic_period'].mean()) / 2
        axes[panel_offset + 1].axhline(y=period_mean, color='black', linestyle='-', alpha=0.3)
        
        # フラクタル次元パネル
        axes[panel_offset + 2].axhline(y=1.5, color='black', linestyle='--', alpha=0.5, label='Middle')
        axes[panel_offset + 2].axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
        axes[panel_offset + 2].axhline(y=2.0, color='gray', linestyle=':', alpha=0.5)
        
        # MESA位相パネル
        axes[panel_offset + 3].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[panel_offset + 3].axhline(y=np.pi, color='black', linestyle='--', alpha=0.5)
        axes[panel_offset + 3].axhline(y=-np.pi, color='black', linestyle='--', alpha=0.5)
        
        # アルファ値パネル
        axes[panel_offset + 4].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
        axes[panel_offset + 4].axhline(y=0.1, color='gray', linestyle=':', alpha=0.5)
        axes[panel_offset + 4].axhline(y=0.9, color='gray', linestyle=':', alpha=0.5)
        
        # 統計情報の表示
        print(f"\n=== MESA_FRAMA統計 ===")
        print(f"短期MESA_FRAMA - 平均: {df['fast_mesa_frama'].mean():.4f}, 標準偏差: {df['fast_mesa_frama'].std():.4f}")
        print(f"長期MESA_FRAMA - 平均: {df['slow_mesa_frama'].mean():.4f}, 標準偏差: {df['slow_mesa_frama'].std():.4f}")
        print(f"短期動的期間 - 平均: {df['fast_dynamic_period'].mean():.2f}, 標準偏差: {df['fast_dynamic_period'].std():.2f}")
        print(f"長期動的期間 - 平均: {df['slow_dynamic_period'].mean():.2f}, 標準偏差: {df['slow_dynamic_period'].std():.2f}")
        print(f"短期フラクタル次元 - 平均: {df['fast_fractal_dimension'].mean():.4f}, 範囲: {df['fast_fractal_dimension'].min():.4f} - {df['fast_fractal_dimension'].max():.4f}")
        print(f"長期フラクタル次元 - 平均: {df['slow_fractal_dimension'].mean():.4f}, 範囲: {df['slow_fractal_dimension'].min():.4f} - {df['slow_fractal_dimension'].max():.4f}")
        
        crossover_count = (df['crossover_signals'] != 0).sum()
        golden_count = (df['crossover_signals'] == 1).sum()
        dead_count = (df['crossover_signals'] == -1).sum()
        print(f"クロスオーバー - 総数: {crossover_count}, ゴールデン: {golden_count}, デッド: {dead_count}")
        
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
    parser = argparse.ArgumentParser(description='デュアルMESA_FRAMAの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--fast-period', type=int, default=8, help='短期基本期間')
    parser.add_argument('--slow-period', type=int, default=32, help='長期基本期間')
    parser.add_argument('--fast-src', type=str, default='close', help='短期ソースタイプ')
    parser.add_argument('--slow-src', type=str, default='hl2', help='長期ソースタイプ')
    parser.add_argument('--no-volume', action='store_true', help='出来高を非表示')
    parser.add_argument('--no-signals', action='store_true', help='クロスオーバーシグナルを非表示')
    args = parser.parse_args()
    
    # チャートを作成
    chart = MESAFRAMAChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        fast_base_period=args.fast_period,
        fast_src_type=args.fast_src,
        slow_base_period=args.slow_period,
        slow_src_type=args.slow_src
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        show_volume=not args.no_volume,
        show_crossover_signals=not args.no_signals,
        savefig=args.output
    )


if __name__ == "__main__":
    main()