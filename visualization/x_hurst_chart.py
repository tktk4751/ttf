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
from indicators.trend_filter.x_hurst import XHurst


class XHurstChart:
    """
    X_Hurstを表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - X_Hurst値（0-1の範囲、高い値=持続性トレンド、低い値=反持続性レンジ）
    - ミッドライン
    - トレンド信号（1=持続性トレンド、-1=反持続性レンジ）
    - 生のDFAハースト値
    - 平滑化ハースト値（オプション）
    - 動的期間値（動的期間適応が有効な場合）
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.x_hurst = None
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
                            period: int = 55,
                            midline_period: int = 100,
                            # DFAハーストパラメータ
                            hurst_src_type: str = 'close',
                            min_scale: int = 4,
                            max_scale: int = 20,
                            scale_steps: int = 8,
                            # 平滑化オプション
                            use_smoothing: bool = False,
                            smoother_type: str = 'super_smoother',
                            smoother_period: int = 8,
                            smoother_src_type: str = 'close',
                            # エラーズ統合サイクル検出器パラメータ
                            use_dynamic_period: bool = True,
                            detector_type: str = 'phac_e',
                            lp_period: int = 13,
                            hp_period: int = 124,
                            cycle_part: float = 0.5,
                            max_cycle: int = 124,
                            min_cycle: int = 13,
                            max_output: int = 124,
                            min_output: int = 13,
                            # 統合カルマンフィルターパラメータ
                            use_kalman_filter: bool = False,
                            kalman_filter_type: str = 'unscented',
                            kalman_process_noise: float = 0.01,
                            kalman_observation_noise: float = 0.001
                           ) -> None:
        """
        X_Hurstを計算する
        
        Args:
            period: X_Hurst計算期間
            midline_period: ミッドライン計算期間
            hurst_src_type: ハースト計算用ソースタイプ
            min_scale: DFA最小スケール
            max_scale: DFA最大スケール
            scale_steps: DFAスケールステップ数
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
            
        print("\\nX_Hurstを計算中...")
        
        # X_Hurstを計算
        self.x_hurst = XHurst(
            period=period,
            midline_period=midline_period,
            hurst_src_type=hurst_src_type,
            min_scale=min_scale,
            max_scale=max_scale,
            scale_steps=scale_steps,
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
        
        # X_Hurstの計算
        print("計算を実行します...")
        result = self.x_hurst.calculate(self.data)
        
        print(f"X_Hurst計算完了 - 値: {len(result.values)}")
        
        # NaN値のチェック
        nan_count = np.isnan(result.values).sum()
        valid_count = (~np.isnan(result.values)).sum()
        trend_count = (result.trend_signal != 0).sum()
        print(f"NaN値: {nan_count}, 有効値: {valid_count}")
        print(f"トレンド信号 - 有効: {trend_count}, 持続性トレンド: {(result.trend_signal == 1).sum()}, 反持続性レンジ: {(result.trend_signal == -1).sum()}")
        
        # 統計情報
        if valid_count > 0:
            valid_values = result.values[~np.isnan(result.values)]
            print(f"X_Hurst統計 - 平均: {np.mean(valid_values):.4f}, 範囲: {np.min(valid_values):.4f} - {np.max(valid_values):.4f}")
            
            # ハースト指数の解釈統計
            persistent_ratio = (valid_values > 0.5).sum() / len(valid_values) * 100
            anti_persistent_ratio = (valid_values < 0.5).sum() / len(valid_values) * 100
            random_ratio = (np.abs(valid_values - 0.5) < 0.05).sum() / len(valid_values) * 100
            
            print(f"持続性レジーム (H>0.5): {persistent_ratio:.1f}%")
            print(f"反持続性レジーム (H<0.5): {anti_persistent_ratio:.1f}%")
            print(f"ランダムウォーク領域 (H≈0.5): {random_ratio:.1f}%")
        
        print("X_Hurst計算完了")
            
    def plot(self, 
            title: str = "X_Hurst（ハースト指数）", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (14, 12),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとX_Hurstを描画する
        
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
            
        if self.x_hurst is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # X_Hurstの値を取得
        print("X_Hurstデータを取得中...")
        result = self.x_hurst.calculate(self.data)
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'x_hurst': result.values,
                'raw_hurst': result.raw_hurst,
                'filtered_hurst': result.filtered_hurst,
                'smoothed_hurst': result.smoothed_hurst,
                'midline': result.midline,
                'trend_signal': result.trend_signal
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        print(f"X_Hurstデータ確認 - NaN: {df['x_hurst'].isna().sum()}")
        
        # 持続性/反持続性状態に基づく色分け
        df['hurst_persistent'] = np.where(df['trend_signal'] == 1, df['x_hurst'], np.nan)
        df['hurst_anti_persistent'] = np.where(df['trend_signal'] == -1, df['x_hurst'], np.nan)
        
        # ミッドラインの色分け（トレンド状態に応じて）
        df['midline_persistent'] = np.where(df['trend_signal'] == 1, df['midline'], np.nan)
        df['midline_anti_persistent'] = np.where(df['trend_signal'] == -1, df['midline'], np.nan)
        
        # NaN値を含む行を出力（最初の5行のみ）
        nan_rows = df[df['x_hurst'].isna() | df['midline'].isna()]
        if not nan_rows.empty:
            print(f"NaN値を含む行: {len(nan_rows)}行")
            if len(nan_rows) > 0:
                print(nan_rows.head())
        
        # mplfinanceでプロット用の設定
        # 1. メインチャート上のプロット（価格とボリューム用）
        main_plots = []
        
        # 2. X_Hurstパネル
        hurst_persistent_plot = mpf.make_addplot(df['hurst_persistent'], panel=1, color='green', width=2, 
                                              ylabel='X-Hurst', secondary_y=False, label='Persistent')
        hurst_anti_persistent_plot = mpf.make_addplot(df['hurst_anti_persistent'], panel=1, color='red', width=2, 
                                              secondary_y=False, label='Anti-Persistent')
        midline_persistent_plot = mpf.make_addplot(df['midline_persistent'], panel=1, color='darkgreen', width=1, 
                                             alpha=0.7, secondary_y=False, label='Midline (Persistent)')
        midline_anti_persistent_plot = mpf.make_addplot(df['midline_anti_persistent'], panel=1, color='darkred', width=1, 
                                             alpha=0.7, secondary_y=False, label='Midline (Anti-Persistent)')
        
        # 3. 生のDFAハースト値パネル
        raw_hurst_panel = mpf.make_addplot(df['raw_hurst'], panel=2, color='blue', width=1.2, 
                                    ylabel='Raw DFA Hurst', secondary_y=False, label='Raw Hurst')
        
        # 4. トレンド信号パネル
        trend_panel = mpf.make_addplot(df['trend_signal'], panel=3, color='orange', width=1.5, 
                                      ylabel='Trend Signal', secondary_y=False, label='Signal', type='line')
        
        # 5. 平滑化ハースト（使用している場合）
        smoothed_plots = []
        if self.x_hurst.use_smoothing and not df['smoothed_hurst'].isna().all():
            df['smoothed_persistent'] = np.where(df['trend_signal'] == 1, df['smoothed_hurst'], np.nan)
            df['smoothed_anti_persistent'] = np.where(df['trend_signal'] == -1, df['smoothed_hurst'], np.nan)
            
            smoothed_persistent_plot = mpf.make_addplot(df['smoothed_persistent'], panel=1, color='lightgreen', 
                                                  width=1.5, alpha=0.8, secondary_y=False, label='Smoothed (Persistent)')
            smoothed_anti_persistent_plot = mpf.make_addplot(df['smoothed_anti_persistent'], panel=1, color='lightcoral', 
                                                  width=1.5, alpha=0.8, secondary_y=False, label='Smoothed (Anti-Persistent)')
            smoothed_plots = [smoothed_persistent_plot, smoothed_anti_persistent_plot]
        
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
            kwargs['panel_ratios'] = (4, 1, 2, 1, 1)  # メイン:出来高:X_Hurst:生ハースト:トレンド信号
            # 出来高を表示する場合は、オシレーターのパネル番号を+1する
            hurst_persistent_plot = mpf.make_addplot(df['hurst_persistent'], panel=2, color='green', width=2, 
                                              ylabel='X-Hurst', secondary_y=False, label='Persistent')
            hurst_anti_persistent_plot = mpf.make_addplot(df['hurst_anti_persistent'], panel=2, color='red', width=2, 
                                              secondary_y=False, label='Anti-Persistent')
            midline_persistent_plot = mpf.make_addplot(df['midline_persistent'], panel=2, color='darkgreen', width=1, 
                                                 alpha=0.7, secondary_y=False, label='Midline (Persistent)')
            midline_anti_persistent_plot = mpf.make_addplot(df['midline_anti_persistent'], panel=2, color='darkred', width=1, 
                                                 alpha=0.7, secondary_y=False, label='Midline (Anti-Persistent)')
            raw_hurst_panel = mpf.make_addplot(df['raw_hurst'], panel=3, color='blue', width=1.2, 
                                        ylabel='Raw DFA Hurst', secondary_y=False, label='Raw Hurst')
            trend_panel = mpf.make_addplot(df['trend_signal'], panel=4, color='orange', width=1.5, 
                                          ylabel='Trend Signal', secondary_y=False, label='Signal', type='line')
            
            # 平滑化版も更新
            if smoothed_plots:
                smoothed_persistent_plot = mpf.make_addplot(df['smoothed_persistent'], panel=2, color='lightgreen', 
                                                      width=1.5, alpha=0.8, secondary_y=False, label='Smoothed (Persistent)')
                smoothed_anti_persistent_plot = mpf.make_addplot(df['smoothed_anti_persistent'], panel=2, color='lightcoral', 
                                                      width=1.5, alpha=0.8, secondary_y=False, label='Smoothed (Anti-Persistent)')
                smoothed_plots = [smoothed_persistent_plot, smoothed_anti_persistent_plot]
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (4, 2, 1, 1)  # メイン:X_Hurst:生ハースト:トレンド信号
        
        # すべてのプロットを結合
        all_plots = main_plots + [hurst_persistent_plot, hurst_anti_persistent_plot, midline_persistent_plot, midline_anti_persistent_plot] + smoothed_plots + [raw_hurst_panel, trend_panel]
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加（X_Hurstパネル）
        hurst_panel_idx = 2 if show_volume else 1
        legend_labels = ['X-Hurst (Persistent)', 'X-Hurst (Anti-Persistent)', 'Midline (Persistent)', 'Midline (Anti-Persistent)']
        if smoothed_plots:
            legend_labels.extend(['Smoothed (Persistent)', 'Smoothed (Anti-Persistent)'])
        axes[hurst_panel_idx].legend(legend_labels, loc='upper left', fontsize=8)
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        if show_volume:
            # X_Hurstパネル
            axes[2].axhline(y=0.5, color='black', linestyle='--', alpha=0.7, linewidth=2)  # ランダムウォーク線
            axes[2].axhline(y=0.55, color='green', linestyle=':', alpha=0.7)  # 持続性閾値
            axes[2].axhline(y=0.45, color='red', linestyle=':', alpha=0.7)    # 反持続性閾値
            axes[2].axhline(y=0.0, color='black', linestyle='-', alpha=0.3)
            axes[2].axhline(y=1.0, color='black', linestyle='-', alpha=0.3)
            axes[2].set_ylim(-0.05, 1.05)
            axes[2].text(0.02, 0.52, 'Random Walk', transform=axes[2].transAxes, fontsize=8, alpha=0.7)
            axes[2].text(0.02, 0.77, 'Persistent', transform=axes[2].transAxes, fontsize=8, color='green', alpha=0.7)
            axes[2].text(0.02, 0.23, 'Anti-Persistent', transform=axes[2].transAxes, fontsize=8, color='red', alpha=0.7)
            
            # 生ハースト値パネル
            hurst_mean = df['raw_hurst'].mean()
            axes[3].axhline(y=hurst_mean, color='black', linestyle='-', alpha=0.3)
            axes[3].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
            axes[3].axhline(y=0.55, color='green', linestyle=':', alpha=0.7)
            axes[3].axhline(y=0.45, color='red', linestyle=':', alpha=0.7)
            axes[3].set_ylim(-0.05, 1.05)
            
            # トレンド信号パネル
            axes[4].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[4].axhline(y=1, color='green', linestyle='--', alpha=0.5)
            axes[4].axhline(y=-1, color='red', linestyle='--', alpha=0.5)
            axes[4].axhline(y=0.5, color='green', linestyle=':', alpha=0.3)
            axes[4].axhline(y=-0.5, color='red', linestyle=':', alpha=0.3)
            axes[4].set_ylim(-1.5, 1.5)
        else:
            # X_Hurstパネル
            axes[1].axhline(y=0.5, color='black', linestyle='--', alpha=0.7, linewidth=2)
            axes[1].axhline(y=0.55, color='green', linestyle=':', alpha=0.7)
            axes[1].axhline(y=0.45, color='red', linestyle=':', alpha=0.7)
            axes[1].axhline(y=0.0, color='black', linestyle='-', alpha=0.3)
            axes[1].axhline(y=1.0, color='black', linestyle='-', alpha=0.3)
            axes[1].set_ylim(-0.05, 1.05)
            axes[1].text(0.02, 0.52, 'Random Walk', transform=axes[1].transAxes, fontsize=8, alpha=0.7)
            axes[1].text(0.02, 0.77, 'Persistent', transform=axes[1].transAxes, fontsize=8, color='green', alpha=0.7)
            axes[1].text(0.02, 0.23, 'Anti-Persistent', transform=axes[1].transAxes, fontsize=8, color='red', alpha=0.7)
            
            # 生ハースト値パネル
            hurst_mean = df['raw_hurst'].mean()
            axes[2].axhline(y=hurst_mean, color='black', linestyle='-', alpha=0.3)
            axes[2].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
            axes[2].axhline(y=0.55, color='green', linestyle=':', alpha=0.7)
            axes[2].axhline(y=0.45, color='red', linestyle=':', alpha=0.7)
            axes[2].set_ylim(-0.05, 1.05)
            
            # トレンド信号パネル
            axes[3].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[3].axhline(y=1, color='green', linestyle='--', alpha=0.5)
            axes[3].axhline(y=-1, color='red', linestyle='--', alpha=0.5)
            axes[3].axhline(y=0.5, color='green', linestyle=':', alpha=0.3)
            axes[3].axhline(y=-0.5, color='red', linestyle=':', alpha=0.3)
            axes[3].set_ylim(-1.5, 1.5)
        
        # 統計情報の表示
        print(f"\\n=== X_Hurst統計 ===")
        valid_mask = ~np.isnan(df['x_hurst'])
        total_points = valid_mask.sum()
        persistent_points = (df['trend_signal'] == 1).sum()
        anti_persistent_points = (df['trend_signal'] == -1).sum()
        
        print(f"総データ点数: {total_points}")
        print(f"持続性トレンド状態: {persistent_points} ({persistent_points/total_points*100:.1f}%)")
        print(f"反持続性レンジ状態: {anti_persistent_points} ({anti_persistent_points/total_points*100:.1f}%)")
        
        if total_points > 0:
            valid_hurst = df['x_hurst'][valid_mask]
            print(f"X_Hurst - 平均: {valid_hurst.mean():.4f}, 範囲: {valid_hurst.min():.4f} - {valid_hurst.max():.4f}")
            
        if not df['raw_hurst'].isna().all():
            valid_raw_hurst = df['raw_hurst'][~np.isnan(df['raw_hurst'])]
            print(f"生DFAハースト - 平均: {valid_raw_hurst.mean():.4f}, 範囲: {valid_raw_hurst.min():.4f} - {valid_raw_hurst.max():.4f}")
        
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
    parser = argparse.ArgumentParser(description='X_Hurstの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--period', type=int, default=55, help='X_Hurst計算期間')
    parser.add_argument('--midline-period', type=int, default=100, help='ミッドライン期間')
    parser.add_argument('--hurst-src-type', type=str, default='hlc3', help='ハースト計算用ソースタイプ')
    parser.add_argument('--min-scale', type=int, default=4, help='DFA最小スケール')
    parser.add_argument('--max-scale', type=int, default=20, help='DFA最大スケール')
    parser.add_argument('--scale-steps', type=int, default=8, help='DFAスケールステップ数')
    parser.add_argument('--smooth', action='store_true', help='平滑化を有効にする')
    parser.add_argument('--smoother-type', type=str, default='super_smoother', help='スムーサータイプ')
    parser.add_argument('--dynamic', action='store_true', help='動的期間適応を有効にする')
    parser.add_argument('--detector-type', type=str, default='phac_e', help='サイクル検出器タイプ')
    parser.add_argument('--kalman', action='store_true', help='カルマンフィルターを有効にする')
    parser.add_argument('--kalman-type', type=str, default='unscented', help='カルマンフィルタータイプ')
    args = parser.parse_args()
    
    # チャートを作成
    chart = XHurstChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        period=args.period,
        midline_period=args.midline_period,
        hurst_src_type=args.hurst_src_type,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        scale_steps=args.scale_steps,
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