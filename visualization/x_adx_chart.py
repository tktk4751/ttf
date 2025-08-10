#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from typing import Optional, Tuple

# データ取得のための依存関係
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource

# インジケーター
from indicators.trend_filter.x_adx import XADX


class XADXChart:
    """
    X_ADXを表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - X_ADX値（正規化されたADX、高い値=強いトレンド）
    - ミッドライン
    - トレンド信号（1=トレンド、-1=レンジ）
    - True Range値
    - +DI/-DI値
    - 平滑化ADX値（オプション）
    - 動的期間値（動的期間適応が有効な場合）
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.x_adx = None
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
                            period: int = 13,
                            midline_period: int = 100,
                            tr_method: str = 'atr',
                            str_period: float = 20.0,
                            src_type: str = 'hlc3',
                            # 平滑化オプション
                            use_smoothing: bool = True,
                            smoother_type: str = 'super_smoother',
                            smoother_period: int = 10,
                            smoother_src_type: str = 'close',
                            # 動的期間オプション
                            use_dynamic_period: bool = False,
                            detector_type: str = 'hody_e',
                            lp_period: int = 13,
                            hp_period: int = 124,
                            cycle_part: float = 0.5,
                            max_cycle: int = 124,
                            min_cycle: int = 13,
                            max_output: int = 124,
                            min_output: int = 13,
                            # カルマンフィルターオプション
                            use_kalman_filter: bool = False,
                            kalman_filter_type: str = 'unscented',
                            kalman_process_noise: float = 0.01,
                            kalman_observation_noise: float = 0.001
                           ) -> None:
        """
        X_ADXを計算する
        
        Args:
            period: ADX計算期間
            midline_period: ミッドライン計算期間
            tr_method: True Range計算方法（'atr' または 'str'）
            str_period: STR期間
            src_type: プライスソースタイプ
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
            
        print("\nX_ADXを計算中...")
        
        # X_ADXを計算
        self.x_adx = XADX(
            period=period,
            midline_period=midline_period,
            tr_method=tr_method,
            str_period=str_period,
            src_type=src_type,
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
        
        # X_ADXの計算
        print("計算を実行します...")
        result = self.x_adx.calculate(self.data)
        
        print(f"X_ADX計算完了 - 値: {len(result.values)}")
        
        # NaN値のチェック
        nan_count = np.isnan(result.values).sum()
        valid_count = (~np.isnan(result.values)).sum()
        trend_count = (result.trend_signal != 0).sum()
        print(f"NaN値: {nan_count}, 有効値: {valid_count}")
        print(f"トレンド信号 - 有効: {trend_count}, トレンド: {(result.trend_signal == 1).sum()}, レンジ: {(result.trend_signal == -1).sum()}")
        
        # 統計情報
        if valid_count > 0:
            valid_values = result.values[~np.isnan(result.values)]
            print(f"X_ADX統計 - 平均: {np.mean(valid_values):.4f}, 範囲: {np.min(valid_values):.4f} - {np.max(valid_values):.4f}")
        
        print("X_ADX計算完了")
            
    def plot(self, 
            title: str = "X_ADX分析", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (14, 16),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとX_ADXを描画する
        
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
            
        if self.x_adx is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # X_ADXの値を取得
        print("X_ADXデータを取得中...")
        result = self.x_adx.calculate(self.data)
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'x_adx': result.values,
                'raw_adx': result.raw_adx,
                'smoothed_adx': result.smoothed_adx,
                'midline': result.midline,
                'trend_signal': result.trend_signal,
                'tr_values': result.tr_values,
                'plus_di': result.plus_di,
                'minus_di': result.minus_di
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        print(f"X_ADXデータ確認 - NaN: {df['x_adx'].isna().sum()}")
        
        # トレンド/レンジ状態に基づく色分け
        df['adx_trend'] = np.where(df['trend_signal'] == 1, df['x_adx'], np.nan)
        df['adx_range'] = np.where(df['trend_signal'] == -1, df['x_adx'], np.nan)
        
        # ミッドラインの色分け（トレンド状態に応じて）
        df['midline_trend'] = np.where(df['trend_signal'] == 1, df['midline'], np.nan)
        df['midline_range'] = np.where(df['trend_signal'] == -1, df['midline'], np.nan)
        
        # +DI/-DIの色分け
        df['plus_di_trend'] = np.where(df['trend_signal'] == 1, df['plus_di'], np.nan)
        df['plus_di_range'] = np.where(df['trend_signal'] == -1, df['plus_di'], np.nan)
        df['minus_di_trend'] = np.where(df['trend_signal'] == 1, df['minus_di'], np.nan)
        df['minus_di_range'] = np.where(df['trend_signal'] == -1, df['minus_di'], np.nan)
        
        # NaN値を含む行を出力（最初の5行のみ）
        nan_rows = df[df['x_adx'].isna() | df['midline'].isna()]
        if not nan_rows.empty:
            print(f"NaN値を含む行: {len(nan_rows)}行")
            if len(nan_rows) > 0:
                print(nan_rows.head())
        
        # mplfinanceでプロット用の設定
        # 1. メインチャート上のプロット（価格とボリューム用）
        main_plots = []
        
        # 2. X_ADXパネル
        adx_trend_plot = mpf.make_addplot(df['adx_trend'], panel=1, color='green', width=2, 
                                          ylabel='X-ADX', secondary_y=False, label='ADX (Trend)')
        adx_range_plot = mpf.make_addplot(df['adx_range'], panel=1, color='red', width=2, 
                                          secondary_y=False, label='ADX (Range)')
        midline_trend_plot = mpf.make_addplot(df['midline_trend'], panel=1, color='darkgreen', width=1, 
                                             alpha=0.7, secondary_y=False, label='Midline (Trend)')
        midline_range_plot = mpf.make_addplot(df['midline_range'], panel=1, color='darkred', width=1, 
                                             alpha=0.7, secondary_y=False, label='Midline (Range)')
        
        # 3. True Rangeパネル
        tr_panel = mpf.make_addplot(df['tr_values'], panel=2, color='blue', width=1.2, 
                                    ylabel='True Range', secondary_y=False, label='TR')
        
        # 4. +DI/-DIパネル
        plus_di_trend_plot = mpf.make_addplot(df['plus_di_trend'], panel=3, color='green', width=1.5, 
                                             ylabel='+DI/-DI', secondary_y=False, label='+DI (Trend)')
        plus_di_range_plot = mpf.make_addplot(df['plus_di_range'], panel=3, color='lightgreen', width=1.5, 
                                             alpha=0.7, secondary_y=False, label='+DI (Range)')
        minus_di_trend_plot = mpf.make_addplot(df['minus_di_trend'], panel=3, color='red', width=1.5, 
                                              secondary_y=False, label='-DI (Trend)')
        minus_di_range_plot = mpf.make_addplot(df['minus_di_range'], panel=3, color='lightcoral', width=1.5, 
                                              alpha=0.7, secondary_y=False, label='-DI (Range)')
        
        # 5. トレンド信号パネル
        trend_panel = mpf.make_addplot(df['trend_signal'], panel=4, color='orange', width=1.5, 
                                      ylabel='Trend Signal', secondary_y=False, label='Signal', type='line')
        
        # 6. 平滑化ADX（使用している場合）
        smoothed_plots = []
        if self.x_adx.use_smoothing and not df['smoothed_adx'].isna().all():
            df['smoothed_trend'] = np.where(df['trend_signal'] == 1, df['smoothed_adx'], np.nan)
            df['smoothed_range'] = np.where(df['trend_signal'] == -1, df['smoothed_adx'], np.nan)
            
            smoothed_trend_plot = mpf.make_addplot(df['smoothed_trend'], panel=1, color='lightgreen', 
                                                  width=1.5, alpha=0.8, secondary_y=False, label='Smoothed (Trend)')
            smoothed_range_plot = mpf.make_addplot(df['smoothed_range'], panel=1, color='lightcoral', 
                                                  width=1.5, alpha=0.8, secondary_y=False, label='Smoothed (Range)')
            smoothed_plots = [smoothed_trend_plot, smoothed_range_plot]
        
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
            kwargs['panel_ratios'] = (4, 1, 2, 1, 2, 1)  # メイン:出来高:X_ADX:TR:+DI/-DI:トレンド信号
            # 出来高を表示する場合は、オシレーターのパネル番号を+1する
            adx_trend_plot = mpf.make_addplot(df['adx_trend'], panel=2, color='green', width=2, 
                                              ylabel='X-ADX', secondary_y=False, label='ADX (Trend)')
            adx_range_plot = mpf.make_addplot(df['adx_range'], panel=2, color='red', width=2, 
                                              secondary_y=False, label='ADX (Range)')
            midline_trend_plot = mpf.make_addplot(df['midline_trend'], panel=2, color='darkgreen', width=1, 
                                                 alpha=0.7, secondary_y=False, label='Midline (Trend)')
            midline_range_plot = mpf.make_addplot(df['midline_range'], panel=2, color='darkred', width=1, 
                                                 alpha=0.7, secondary_y=False, label='Midline (Range)')
            tr_panel = mpf.make_addplot(df['tr_values'], panel=3, color='blue', width=1.2, 
                                        ylabel='True Range', secondary_y=False, label='TR')
            plus_di_trend_plot = mpf.make_addplot(df['plus_di_trend'], panel=4, color='green', width=1.5, 
                                                 ylabel='+DI/-DI', secondary_y=False, label='+DI (Trend)')
            plus_di_range_plot = mpf.make_addplot(df['plus_di_range'], panel=4, color='lightgreen', width=1.5, 
                                                 alpha=0.7, secondary_y=False, label='+DI (Range)')
            minus_di_trend_plot = mpf.make_addplot(df['minus_di_trend'], panel=4, color='red', width=1.5, 
                                                  secondary_y=False, label='-DI (Trend)')
            minus_di_range_plot = mpf.make_addplot(df['minus_di_range'], panel=4, color='lightcoral', width=1.5, 
                                                  alpha=0.7, secondary_y=False, label='-DI (Range)')
            trend_panel = mpf.make_addplot(df['trend_signal'], panel=5, color='orange', width=1.5, 
                                          ylabel='Trend Signal', secondary_y=False, label='Signal', type='line')
            
            # 平滑化版も更新
            if smoothed_plots:
                smoothed_trend_plot = mpf.make_addplot(df['smoothed_trend'], panel=2, color='lightgreen', 
                                                      width=1.5, alpha=0.8, secondary_y=False, label='Smoothed (Trend)')
                smoothed_range_plot = mpf.make_addplot(df['smoothed_range'], panel=2, color='lightcoral', 
                                                      width=1.5, alpha=0.8, secondary_y=False, label='Smoothed (Range)')
                smoothed_plots = [smoothed_trend_plot, smoothed_range_plot]
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (4, 2, 1, 2, 1)  # メイン:X_ADX:TR:+DI/-DI:トレンド信号
        
        # すべてのプロットを結合
        all_plots = (main_plots + [adx_trend_plot, adx_range_plot, midline_trend_plot, midline_range_plot] + 
                    smoothed_plots + [tr_panel, plus_di_trend_plot, plus_di_range_plot, minus_di_trend_plot, minus_di_range_plot, trend_panel])
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加（X_ADXパネル）
        adx_panel_idx = 2 if show_volume else 1
        legend_labels = ['X-ADX (Trend)', 'X-ADX (Range)', 'Midline (Trend)', 'Midline (Range)']
        if smoothed_plots:
            legend_labels.extend(['Smoothed (Trend)', 'Smoothed (Range)'])
        axes[adx_panel_idx].legend(legend_labels, loc='upper left', fontsize=8)
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        if show_volume:
            # X_ADXパネル
            axes[2].axhline(y=0.25, color='green', linestyle='--', alpha=0.5, label='Strong Trend (0.25)')
            axes[2].axhline(y=0.20, color='orange', linestyle='--', alpha=0.5, label='Weak Trend (0.20)')
            axes[2].axhline(y=0.0, color='black', linestyle='-', alpha=0.3)
            axes[2].axhline(y=1.0, color='black', linestyle='-', alpha=0.3)
            axes[2].set_ylim(-0.05, 1.05)
            
            # True Rangeパネル
            tr_mean = df['tr_values'].mean()
            axes[3].axhline(y=tr_mean, color='black', linestyle='-', alpha=0.3, label=f'Mean TR ({tr_mean:.4f})')
            
            # +DI/-DIパネル
            axes[4].axhline(y=0.0, color='black', linestyle='-', alpha=0.3)
            axes[4].axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, label='0.5 Level')
            
            # トレンド信号パネル
            axes[5].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[5].axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Trend')
            axes[5].axhline(y=-1, color='red', linestyle='--', alpha=0.5, label='Range')
            axes[5].set_ylim(-1.5, 1.5)
        else:
            # X_ADXパネル
            axes[1].axhline(y=0.25, color='green', linestyle='--', alpha=0.5, label='Strong Trend (0.25)')
            axes[1].axhline(y=0.20, color='orange', linestyle='--', alpha=0.5, label='Weak Trend (0.20)')
            axes[1].axhline(y=0.0, color='black', linestyle='-', alpha=0.3)
            axes[1].axhline(y=1.0, color='black', linestyle='-', alpha=0.3)
            axes[1].set_ylim(-0.05, 1.05)
            
            # True Rangeパネル
            tr_mean = df['tr_values'].mean()
            axes[2].axhline(y=tr_mean, color='black', linestyle='-', alpha=0.3, label=f'Mean TR ({tr_mean:.4f})')
            
            # +DI/-DIパネル
            axes[3].axhline(y=0.0, color='black', linestyle='-', alpha=0.3)
            axes[3].axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, label='0.5 Level')
            
            # トレンド信号パネル
            axes[4].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[4].axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Trend')
            axes[4].axhline(y=-1, color='red', linestyle='--', alpha=0.5, label='Range')
            axes[4].set_ylim(-1.5, 1.5)
        
        # 統計情報の表示
        print(f"\n=== X_ADX統計 ===")
        valid_mask = ~np.isnan(df['x_adx'])
        total_points = valid_mask.sum()
        trend_points = (df['trend_signal'] == 1).sum()
        range_points = (df['trend_signal'] == -1).sum()
        
        print(f"総データ点数: {total_points}")
        print(f"トレンド状態: {trend_points} ({trend_points/total_points*100:.1f}%)")
        print(f"レンジ状態: {range_points} ({range_points/total_points*100:.1f}%)")
        
        if total_points > 0:
            valid_adx = df['x_adx'][valid_mask]
            print(f"X_ADX - 平均: {valid_adx.mean():.4f}, 範囲: {valid_adx.min():.4f} - {valid_adx.max():.4f}")
            
        if not df['tr_values'].isna().all():
            valid_tr = df['tr_values'][~np.isnan(df['tr_values'])]
            print(f"True Range - 平均: {valid_tr.mean():.4f}, 範囲: {valid_tr.min():.4f} - {valid_tr.max():.4f}")
        
        if not df['plus_di'].isna().all():
            valid_plus_di = df['plus_di'][~np.isnan(df['plus_di'])]
            valid_minus_di = df['minus_di'][~np.isnan(df['minus_di'])]
            print(f"+DI - 平均: {valid_plus_di.mean():.4f}, 範囲: {valid_plus_di.min():.4f} - {valid_plus_di.max():.4f}")
            print(f"-DI - 平均: {valid_minus_di.mean():.4f}, 範囲: {valid_minus_di.min():.4f} - {valid_minus_di.max():.4f}")
        
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
    parser = argparse.ArgumentParser(description='X_ADXの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--period', type=int, default=13, help='ADX計算期間')
    parser.add_argument('--midline-period', type=int, default=100, help='ミッドライン期間')
    parser.add_argument('--tr-method', type=str, default='atr', choices=['atr', 'str'], help='True Range計算方法')
    parser.add_argument('--str-period', type=float, default=20.0, help='STR期間')
    parser.add_argument('--src-type', type=str, default='hlc3', help='プライスソースタイプ')
    parser.add_argument('--smooth', action='store_true', help='平滑化を有効にする')
    parser.add_argument('--smoother-type', type=str, default='super_smoother', help='スムーサータイプ')
    parser.add_argument('--dynamic', action='store_true', help='動的期間適応を有効にする')
    parser.add_argument('--detector-type', type=str, default='hody_e', help='サイクル検出器タイプ')
    parser.add_argument('--kalman', action='store_true', help='カルマンフィルターを有効にする')
    parser.add_argument('--kalman-type', type=str, default='unscented', help='カルマンフィルタータイプ')
    args = parser.parse_args()
    
    # チャートを作成
    chart = XADXChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        period=args.period,
        midline_period=args.midline_period,
        tr_method=args.tr_method,
        str_period=args.str_period,
        src_type=args.src_type,
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