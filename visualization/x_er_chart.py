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
from indicators.trend_filter.x_er import XER


class XERChart:
    """
    X_ERを表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - X_ER値（0-1の範囲、高い値=効率的トレンド、低い値=非効率的レンジ）
    - ミッドライン
    - トレンド信号（1=効率的トレンド、-1=非効率的レンジ）
    - 生のER値
    - 平滑化ER値（オプション）
    - 動的期間値（動的期間適応が有効な場合）
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.x_er = None
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
                            period: int = 14,
                            midline_period: int = 100,
                            # ERパラメータ
                            er_period: int = 13,
                            er_src_type: str = 'hlc3',
                            # 平滑化オプション
                            use_smoothing: bool = True,
                            smoother_type: str = 'super_smoother',
                            smoother_period: int = 8,
                            smoother_src_type: str = 'close',
                            # エラーズ統合サイクル検出器パラメータ
                            use_dynamic_period: bool = True,
                            detector_type: str = 'phac_e',
                            lp_period: int = 5,
                            hp_period: int = 144,
                            cycle_part: float = 0.5,
                            max_cycle: int = 144,
                            min_cycle: int = 5,
                            max_output: int = 55,
                            min_output: int = 5,
                            # 統合カルマンフィルターパラメータ
                            use_kalman_filter: bool = True,
                            kalman_filter_type: str = 'unscented',
                            kalman_process_noise: float = 0.01,
                            kalman_observation_noise: float = 0.001
                           ) -> None:
        """
        X_ERを計算する
        
        Args:
            period: X_ER計算期間
            midline_period: ミッドライン計算期間
            er_period: ER期間
            er_src_type: ERソースタイプ
            use_smoothing: 平滑化を使用するか
            smoother_type: 統合スムーサータイプ
            smoother_period: スムーサー期間
            smoother_src_type: スムーサーソースタイプ
            use_dynamic_period: 動的期間適応を使用するか
            detector_type: サイクル検出器タイプ
            lp_period: ローパスフィルター期間
            hp_period: ハイパスフィルター期間
            cycle_part: サイクル部分
            max_cycle: 最大サイクル期間
            min_cycle: 最小サイクル期間
            max_output: 最大出力値
            min_output: 最小出力値
            use_kalman_filter: カルマンフィルターを使用するか
            kalman_filter_type: カルマンフィルタータイプ
            kalman_process_noise: カルマンフィルタープロセスノイズ
            kalman_observation_noise: カルマンフィルター観測ノイズ
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\\nX_ERを計算中...")
        
        # X_ERを計算
        self.x_er = XER(
            period=period,
            midline_period=midline_period,
            er_period=er_period,
            er_src_type=er_src_type,
            use_smoothing=use_smoothing,
            smoother_type=smoother_type,
            smoother_period=smoother_period,
            smoother_src_type=smoother_src_type,
            use_dynamic_period=use_dynamic_period,
            detector_type=detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            max_cycle=max_cycle,
            min_cycle=min_cycle,
            max_output=max_output,
            min_output=min_output,
            use_kalman_filter=use_kalman_filter,
            kalman_filter_type=kalman_filter_type,
            kalman_process_noise=kalman_process_noise,
            kalman_observation_noise=kalman_observation_noise
        )
        
        # X_ERの計算
        print("計算を実行します...")
        result = self.x_er.calculate(self.data)
        
        print(f"X_ER計算完了 - 値: {len(result.values)}")
        
        # NaN値のチェック
        nan_count = np.isnan(result.values).sum()
        valid_count = (~np.isnan(result.values)).sum()
        trend_count = (result.trend_signal != 0).sum()
        print(f"NaN値: {nan_count}, 有効値: {valid_count}")
        print(f"トレンド信号 - 有効: {trend_count}, 効率的トレンド: {(result.trend_signal == 1).sum()}, 非効率的レンジ: {(result.trend_signal == -1).sum()}")
        
        # 統計情報
        if valid_count > 0:
            valid_values = result.values[~np.isnan(result.values)]
            print(f"X_ER統計 - 平均: {np.mean(valid_values):.4f}, 範囲: {np.min(valid_values):.4f} - {np.max(valid_values):.4f}")
        
        print("X_ER計算完了")
            
    def plot(self, 
            title: str = "X_ER（効率性指標）", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (14, 12),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとX_ERを描画する
        
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
            
        if self.x_er is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # X_ERの値を取得
        print("X_ERデータを取得中...")
        result = self.x_er.calculate(self.data)
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'x_er': result.values,
                'raw_er': result.raw_er,
                'filtered_er': result.filtered_er,
                'smoothed_er': result.smoothed_er,
                'midline': result.midline,
                'trend_signal': result.trend_signal
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        print(f"X_ERデータ確認 - NaN: {df['x_er'].isna().sum()}")
        
        # 効率的トレンド/非効率的レンジ状態に基づく色分け
        df['er_efficient'] = np.where(df['trend_signal'] == 1, df['x_er'], np.nan)
        df['er_inefficient'] = np.where(df['trend_signal'] == -1, df['x_er'], np.nan)
        
        # ミッドラインの色分け（トレンド状態に応じて）
        df['midline_efficient'] = np.where(df['trend_signal'] == 1, df['midline'], np.nan)
        df['midline_inefficient'] = np.where(df['trend_signal'] == -1, df['midline'], np.nan)
        
        # NaN値を含む行を出力（最初の5行のみ）
        nan_rows = df[df['x_er'].isna() | df['midline'].isna()]
        if not nan_rows.empty:
            print(f"NaN値を含む行: {len(nan_rows)}行")
            if len(nan_rows) > 0:
                print(nan_rows.head())
        
        # mplfinanceでプロット用の設定
        # 1. メインチャート上のプロット（価格とボリューム用）
        main_plots = []
        
        # 2. X_ERパネル
        er_efficient_plot = mpf.make_addplot(df['er_efficient'], panel=1, color='green', width=2, 
                                          ylabel='X-ER', secondary_y=False, label='Efficient')
        er_inefficient_plot = mpf.make_addplot(df['er_inefficient'], panel=1, color='red', width=2, 
                                          secondary_y=False, label='Inefficient')
        midline_efficient_plot = mpf.make_addplot(df['midline_efficient'], panel=1, color='darkgreen', width=1, 
                                             alpha=0.7, secondary_y=False, label='Midline (Efficient)')
        midline_inefficient_plot = mpf.make_addplot(df['midline_inefficient'], panel=1, color='darkred', width=1, 
                                             alpha=0.7, secondary_y=False, label='Midline (Inefficient)')
        
        # 3. 生のER値パネル
        raw_er_panel = mpf.make_addplot(df['raw_er'], panel=2, color='blue', width=1.2, 
                                    ylabel='Raw ER', secondary_y=False, label='Raw ER')
        
        # 4. トレンド信号パネル
        trend_panel = mpf.make_addplot(df['trend_signal'], panel=3, color='orange', width=1.5, 
                                      ylabel='Trend Signal', secondary_y=False, label='Signal', type='line')
        
        # 5. 平滑化ER（使用している場合）
        smoothed_plots = []
        if self.x_er.use_smoothing and not df['smoothed_er'].isna().all():
            df['smoothed_efficient'] = np.where(df['trend_signal'] == 1, df['smoothed_er'], np.nan)
            df['smoothed_inefficient'] = np.where(df['trend_signal'] == -1, df['smoothed_er'], np.nan)
            
            smoothed_efficient_plot = mpf.make_addplot(df['smoothed_efficient'], panel=1, color='lightgreen', 
                                                  width=1.5, alpha=0.8, secondary_y=False, label='Smoothed (Efficient)')
            smoothed_inefficient_plot = mpf.make_addplot(df['smoothed_inefficient'], panel=1, color='lightcoral', 
                                                  width=1.5, alpha=0.8, secondary_y=False, label='Smoothed (Inefficient)')
            smoothed_plots = [smoothed_efficient_plot, smoothed_inefficient_plot]
        
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
            kwargs['panel_ratios'] = (4, 1, 2, 1, 1)  # メイン:出来高:X_ER:生ER:トレンド信号
            # 出来高を表示する場合は、オシレーターのパネル番号を+1する
            er_efficient_plot = mpf.make_addplot(df['er_efficient'], panel=2, color='green', width=2, 
                                              ylabel='X-ER', secondary_y=False, label='Efficient')
            er_inefficient_plot = mpf.make_addplot(df['er_inefficient'], panel=2, color='red', width=2, 
                                              secondary_y=False, label='Inefficient')
            midline_efficient_plot = mpf.make_addplot(df['midline_efficient'], panel=2, color='darkgreen', width=1, 
                                                 alpha=0.7, secondary_y=False, label='Midline (Efficient)')
            midline_inefficient_plot = mpf.make_addplot(df['midline_inefficient'], panel=2, color='darkred', width=1, 
                                                 alpha=0.7, secondary_y=False, label='Midline (Inefficient)')
            raw_er_panel = mpf.make_addplot(df['raw_er'], panel=3, color='blue', width=1.2, 
                                        ylabel='Raw ER', secondary_y=False, label='Raw ER')
            trend_panel = mpf.make_addplot(df['trend_signal'], panel=4, color='orange', width=1.5, 
                                          ylabel='Trend Signal', secondary_y=False, label='Signal', type='line')
            
            # 平滑化版も更新
            if smoothed_plots:
                smoothed_efficient_plot = mpf.make_addplot(df['smoothed_efficient'], panel=2, color='lightgreen', 
                                                      width=1.5, alpha=0.8, secondary_y=False, label='Smoothed (Efficient)')
                smoothed_inefficient_plot = mpf.make_addplot(df['smoothed_inefficient'], panel=2, color='lightcoral', 
                                                      width=1.5, alpha=0.8, secondary_y=False, label='Smoothed (Inefficient)')
                smoothed_plots = [smoothed_efficient_plot, smoothed_inefficient_plot]
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (4, 2, 1, 1)  # メイン:X_ER:生ER:トレンド信号
        
        # すべてのプロットを結合
        all_plots = main_plots + [er_efficient_plot, er_inefficient_plot, midline_efficient_plot, midline_inefficient_plot] + smoothed_plots + [raw_er_panel, trend_panel]
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加（X_ERパネル）
        er_panel_idx = 2 if show_volume else 1
        legend_labels = ['X-ER (Efficient)', 'X-ER (Inefficient)', 'Midline (Efficient)', 'Midline (Inefficient)']
        if smoothed_plots:
            legend_labels.extend(['Smoothed (Efficient)', 'Smoothed (Inefficient)'])
        axes[er_panel_idx].legend(legend_labels, loc='upper left', fontsize=8)
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        if show_volume:
            # X_ERパネル
            axes[2].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
            axes[2].axhline(y=0.618, color='green', linestyle=':', alpha=0.7)  # 効率的トレンド閾値
            axes[2].axhline(y=0.382, color='red', linestyle=':', alpha=0.7)    # 非効率的閾値
            axes[2].axhline(y=0.0, color='black', linestyle='-', alpha=0.3)
            axes[2].axhline(y=1.0, color='black', linestyle='-', alpha=0.3)
            axes[2].set_ylim(-0.05, 1.05)
            
            # 生ER値パネル
            er_mean = df['raw_er'].mean()
            axes[3].axhline(y=er_mean, color='black', linestyle='-', alpha=0.3)
            axes[3].axhline(y=0.618, color='green', linestyle=':', alpha=0.7)
            axes[3].axhline(y=0.382, color='red', linestyle=':', alpha=0.7)
            axes[3].set_ylim(-0.05, 1.05)
            
            # トレンド信号パネル
            axes[4].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[4].axhline(y=1, color='green', linestyle='--', alpha=0.5)
            axes[4].axhline(y=-1, color='red', linestyle='--', alpha=0.5)
            axes[4].set_ylim(-1.5, 1.5)
        else:
            # X_ERパネル
            axes[1].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
            axes[1].axhline(y=0.618, color='green', linestyle=':', alpha=0.7)
            axes[1].axhline(y=0.382, color='red', linestyle=':', alpha=0.7)
            axes[1].axhline(y=0.0, color='black', linestyle='-', alpha=0.3)
            axes[1].axhline(y=1.0, color='black', linestyle='-', alpha=0.3)
            axes[1].set_ylim(-0.05, 1.05)
            
            # 生ER値パネル
            er_mean = df['raw_er'].mean()
            axes[2].axhline(y=er_mean, color='black', linestyle='-', alpha=0.3)
            axes[2].axhline(y=0.618, color='green', linestyle=':', alpha=0.7)
            axes[2].axhline(y=0.382, color='red', linestyle=':', alpha=0.7)
            axes[2].set_ylim(-0.05, 1.05)
            
            # トレンド信号パネル
            axes[3].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[3].axhline(y=1, color='green', linestyle='--', alpha=0.5)
            axes[3].axhline(y=-1, color='red', linestyle='--', alpha=0.5)
            axes[3].set_ylim(-1.5, 1.5)
        
        # 統計情報の表示
        print(f"\\n=== X_ER統計 ===")
        valid_mask = ~np.isnan(df['x_er'])
        total_points = valid_mask.sum()
        efficient_points = (df['trend_signal'] == 1).sum()
        inefficient_points = (df['trend_signal'] == -1).sum()
        
        print(f"総データ点数: {total_points}")
        print(f"効率的トレンド状態: {efficient_points} ({efficient_points/total_points*100:.1f}%)")
        print(f"非効率的レンジ状態: {inefficient_points} ({inefficient_points/total_points*100:.1f}%)")
        
        if total_points > 0:
            valid_er = df['x_er'][valid_mask]
            print(f"X_ER - 平均: {valid_er.mean():.4f}, 範囲: {valid_er.min():.4f} - {valid_er.max():.4f}")
            
        if not df['raw_er'].isna().all():
            valid_raw_er = df['raw_er'][~np.isnan(df['raw_er'])]
            print(f"生ER - 平均: {valid_raw_er.mean():.4f}, 範囲: {valid_raw_er.min():.4f} - {valid_raw_er.max():.4f}")
        
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
    parser = argparse.ArgumentParser(description='X_ERの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--period', type=int, default=14, help='X_ER計算期間')
    parser.add_argument('--midline-period', type=int, default=100, help='ミッドライン期間')
    parser.add_argument('--er-period', type=int, default=13, help='ER期間')
    parser.add_argument('--er-src-type', type=str, default='hlc3', help='ERソースタイプ')
    parser.add_argument('--smooth', action='store_true', help='平滑化を有効にする')
    parser.add_argument('--smoother-type', type=str, default='super_smoother', help='スムーサータイプ')
    parser.add_argument('--dynamic', action='store_true', help='動的期間適応を有効にする')
    parser.add_argument('--detector-type', type=str, default='phac_e', help='サイクル検出器タイプ')
    parser.add_argument('--kalman', action='store_true', help='カルマンフィルターを有効にする')
    parser.add_argument('--kalman-type', type=str, default='unscented', help='カルマンフィルタータイプ')
    args = parser.parse_args()
    
    # チャートを作成
    chart = XERChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        period=args.period,
        midline_period=args.midline_period,
        er_period=args.er_period,
        er_src_type=args.er_src_type,
        use_smoothing=args.smooth,
        smoother_type=args.smoother_type,
        use_dynamic_period=args.dynamic,
        detector_type=args.detector_type,
        use_kalman_filter=args.kalman,
        kalman_filter_type=args.kalman_type
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output
    )


if __name__ == "__main__":
    main()