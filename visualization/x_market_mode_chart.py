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
from indicators.x_market_mode import XMarketMode


class XMarketModeChart:
    """
    Xマーケットモードを表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - トレンドモード期間（緑）とサイクルモード期間（赤）
    - トレンドライン
    - モード強度インジケーター
    - サイクル期間
    - 適応フェーズ
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.x_market_mode = None
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
                            trend_threshold: float = 0.008,
                            use_smoothing: bool = True,
                            smoother_type: str = 'frama',
                            smoother_period: int = 16,
                            use_kalman_filter: bool = False,
                            kalman_filter_type: str = 'unscented',
                            kalman_process_noise: float = 0.005,
                            kalman_observation_noise: float = 0.0005) -> None:
        """
        Xマーケットモードを計算する
        
        Args:
            src_type: ソースタイプ
            trend_threshold: トレンド判定閾値
            use_smoothing: 平滑化を使用するか
            smoother_type: スムーサータイプ
            smoother_period: スムーサー期間
            use_kalman_filter: カルマンフィルターを使用するか
            kalman_filter_type: カルマンフィルタータイプ
            kalman_process_noise: カルマンフィルタープロセスノイズ
            kalman_observation_noise: カルマンフィルター観測ノイズ
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\nXマーケットモードを計算中...")
        
        # Xマーケットモードを計算
        self.x_market_mode = XMarketMode(
            src_type=src_type,
            trend_threshold=trend_threshold,
            use_smoothing=use_smoothing,
            smoother_type=smoother_type,
            smoother_period=smoother_period,
            use_kalman_filter=use_kalman_filter,
            kalman_filter_type=kalman_filter_type,
            kalman_process_noise=kalman_process_noise,
            kalman_observation_noise=kalman_observation_noise
        )
        
        # Xマーケットモードの計算
        print("計算を実行します...")
        result = self.x_market_mode.calculate(self.data)
        
        # 結果の確認
        print(f"計算完了 - トレンドモード: {len(result.trend_mode)}, トレンドライン: {len(result.trend_line)}")
        
        # NaN値のチェック
        nan_count_trend_mode = np.isnan(result.trend_mode).sum()
        nan_count_trend_line = np.isnan(result.trend_line).sum()
        trend_count = (~np.isnan(result.trend_mode)).sum()
        trend_mode_count = (result.trend_mode == 1).sum()
        cycle_mode_count = (result.trend_mode == 0).sum()
        
        print(f"NaN値 - トレンドモード: {nan_count_trend_mode}, トレンドライン: {nan_count_trend_line}")
        print(f"モード値 - 有効: {trend_count}, トレンド: {trend_mode_count}, サイクル: {cycle_mode_count}")
        
        # 統計情報
        if trend_count > 0:
            trend_ratio = trend_mode_count / trend_count
            cycle_ratio = cycle_mode_count / trend_count
            print(f"モード比率 - トレンド: {trend_ratio:.2%}, サイクル: {cycle_ratio:.2%}")
        
        mean_period = np.nanmean(result.period)
        mean_mode_strength = np.nanmean(result.mode_strength)
        print(f"平均サイクル期間: {mean_period:.1f}")
        print(f"平均モード強度: {mean_mode_strength:.3f}")
        
        print("Xマーケットモード計算完了")
            
    def plot(self, 
            title: str = "Xマーケットモード", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (16, 14),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとXマーケットモードを描画する
        
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
            
        if self.x_market_mode is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # Xマーケットモードの結果を取得
        print("マーケットモードデータを取得中...")
        result = self.x_market_mode.calculate(self.data)
        
        # 結果の長さを確認
        data_len = len(self.data)
        result_len = len(result.trend_mode) if len(result.trend_mode) > 0 else data_len
        
        # 結果が空の場合はNaN配列で埋める
        if len(result.trend_mode) == 0:
            nan_array = np.full(data_len, np.nan)
            result_data = {
                'trend_mode': nan_array.copy(),
                'trend_line': nan_array.copy(),
                'mode_strength': nan_array.copy(),
                'cycle_strength': nan_array.copy(),
                'trend_strength': nan_array.copy(),
                'period': nan_array.copy(),
                'adaptive_phase': nan_array.copy(),
                'signal': nan_array.copy()
            }
        else:
            # 長さが一致しない場合は適切にリサイズ
            def resize_array(arr, target_len):
                if len(arr) == target_len:
                    return arr
                elif len(arr) > target_len:
                    return arr[:target_len]
                else:
                    # 不足分をNaNで埋める
                    padded = np.full(target_len, np.nan)
                    padded[:len(arr)] = arr
                    return padded
            
            result_data = {
                'trend_mode': resize_array(result.trend_mode, data_len),
                'trend_line': resize_array(result.trend_line, data_len),
                'mode_strength': resize_array(result.mode_strength, data_len),
                'cycle_strength': resize_array(result.cycle_strength, data_len),
                'trend_strength': resize_array(result.trend_strength, data_len),
                'period': resize_array(result.period, data_len),
                'adaptive_phase': resize_array(result.adaptive_phase, data_len),
                'signal': resize_array(result.signal, data_len)
            }
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data=result_data
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        print(f"モードデータ確認 - トレンドモードNaN: {df['trend_mode'].isna().sum()}, トレンドラインNaN: {df['trend_line'].isna().sum()}")
        
        # トレンド・サイクルモード期間の色分け準備
        # トレンドモード（1）の期間を緑、サイクルモード（0）の期間を赤で表示
        df['trend_periods'] = np.where(df['trend_mode'] == 1, df['close'], np.nan)
        df['cycle_periods'] = np.where(df['trend_mode'] == 0, df['close'], np.nan)
        
        # トレンドラインの表示
        df['trend_line_display'] = df['trend_line']
        
        # モード切り替わりポイントの検出
        mode_changes = []
        prev_mode = np.nan
        for i, (idx, row) in enumerate(df.iterrows()):
            current_mode = row['trend_mode']
            if not np.isnan(current_mode) and current_mode != prev_mode:
                mode_changes.append((idx, current_mode))
                prev_mode = current_mode
        
        print(f"モード切り替わり回数: {len(mode_changes)}")
        
        # mplfinanceでプロット用の設定
        # 1. メインチャート上のプロット
        main_plots = []
        
        # トレンドライン
        main_plots.append(mpf.make_addplot(df['trend_line_display'], color='blue', width=2, label='Trend Line'))
        
        # トレンド期間の背景色（緑）
        if not df['trend_periods'].isna().all():
            main_plots.append(mpf.make_addplot(df['trend_periods'], type='scatter', 
                                             markersize=0.5, color='green', alpha=0.3, label='Trend Mode'))
        
        # サイクル期間の背景色（赤）
        if not df['cycle_periods'].isna().all():
            main_plots.append(mpf.make_addplot(df['cycle_periods'], type='scatter',
                                             markersize=0.5, color='red', alpha=0.3, label='Cycle Mode'))
        
        # 2. サブプロット設定
        # モード強度パネル
        mode_strength_panel = mpf.make_addplot(df['mode_strength'], panel=1, color='purple', width=1.5, 
                                              ylabel='Mode Strength', secondary_y=False, label='Mode')
        cycle_strength_panel = mpf.make_addplot(df['cycle_strength'], panel=1, color='red', width=1.0, 
                                               secondary_y=False, label='Cycle', alpha=0.7)
        trend_strength_panel = mpf.make_addplot(df['trend_strength'], panel=1, color='green', width=1.0, 
                                               secondary_y=False, label='Trend', alpha=0.7)
        
        # サイクル期間パネル
        period_panel = mpf.make_addplot(df['period'], panel=2, color='orange', width=1.5, 
                                       ylabel='Cycle Period', secondary_y=False, label='Period')
        
        # 適応フェーズパネル
        phase_panel = mpf.make_addplot(df['adaptive_phase'], panel=3, color='brown', width=1.2, 
                                      ylabel='Adaptive Phase', secondary_y=False, label='Phase')
        
        # トレンドモードパネル（バイナリ表示）
        trend_mode_panel = mpf.make_addplot(df['trend_mode'], panel=4, color='black', width=2, 
                                           ylabel='Trend Mode', secondary_y=False, label='Mode', type='line')
        
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
            kwargs['panel_ratios'] = (5, 1, 1.2, 1, 1, 0.8)  # メイン:出来高:強度:期間:フェーズ:モード
            # 出来高を表示する場合は、オシレーターのパネル番号を+1する
            mode_strength_panel = mpf.make_addplot(df['mode_strength'], panel=2, color='purple', width=1.5, 
                                                  ylabel='Mode Strength', secondary_y=False, label='Mode')
            cycle_strength_panel = mpf.make_addplot(df['cycle_strength'], panel=2, color='red', width=1.0, 
                                                   secondary_y=False, label='Cycle', alpha=0.7)
            trend_strength_panel = mpf.make_addplot(df['trend_strength'], panel=2, color='green', width=1.0, 
                                                   secondary_y=False, label='Trend', alpha=0.7)
            period_panel = mpf.make_addplot(df['period'], panel=3, color='orange', width=1.5, 
                                           ylabel='Cycle Period', secondary_y=False, label='Period')
            phase_panel = mpf.make_addplot(df['adaptive_phase'], panel=4, color='brown', width=1.2, 
                                          ylabel='Adaptive Phase', secondary_y=False, label='Phase')
            trend_mode_panel = mpf.make_addplot(df['trend_mode'], panel=5, color='black', width=2, 
                                               ylabel='Trend Mode', secondary_y=False, label='Mode', type='line')
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (5, 1.2, 1, 1, 0.8)  # メイン:強度:期間:フェーズ:モード
        
        # すべてのプロットを結合
        all_plots = main_plots + [mode_strength_panel, cycle_strength_panel, trend_strength_panel, 
                                 period_panel, phase_panel, trend_mode_panel]
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加
        axes[0].legend(['Trend Line', 'Trend Mode', 'Cycle Mode'], loc='upper left')
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        panel_offset = 1 if show_volume else 0
        
        # モード強度パネル
        strength_panel_idx = 1 + panel_offset
        axes[strength_panel_idx].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
        axes[strength_panel_idx].axhline(y=0.0, color='black', linestyle='-', alpha=0.3)
        axes[strength_panel_idx].axhline(y=1.0, color='black', linestyle='-', alpha=0.3)
        
        # サイクル期間パネル
        period_panel_idx = 2 + panel_offset
        period_mean = df['period'].mean()
        axes[period_panel_idx].axhline(y=period_mean, color='black', linestyle='-', alpha=0.3)
        axes[period_panel_idx].axhline(y=66, color='gray', linestyle='--', alpha=0.5)  # 中期中央値
        axes[period_panel_idx].axhline(y=8, color='green', linestyle=':', alpha=0.5)   # 最小期間
        axes[period_panel_idx].axhline(y=124, color='red', linestyle=':', alpha=0.5)  # 最大期間
        
        # 適応フェーズパネル
        phase_panel_idx = 3 + panel_offset
        axes[phase_panel_idx].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[phase_panel_idx].axhline(y=90, color='gray', linestyle='--', alpha=0.5)
        axes[phase_panel_idx].axhline(y=-90, color='gray', linestyle='--', alpha=0.5)
        axes[phase_panel_idx].axhline(y=180, color='red', linestyle=':', alpha=0.3)
        axes[phase_panel_idx].axhline(y=-180, color='red', linestyle=':', alpha=0.3)
        
        # トレンドモードパネル
        mode_panel_idx = 4 + panel_offset
        axes[mode_panel_idx].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
        axes[mode_panel_idx].axhline(y=1, color='green', linestyle='-', alpha=0.7, label='Trend')
        axes[mode_panel_idx].axhline(y=0, color='red', linestyle='-', alpha=0.7, label='Cycle')
        
        # モード切り替わりポイントにマーカーを追加
        for change_date, mode in mode_changes[:20]:  # 最初の20個まで表示
            if change_date in df.index:
                mode_color = 'green' if mode == 1 else 'red'
                mode_name = 'T' if mode == 1 else 'C'
                axes[0].annotate(mode_name, xy=(change_date, df.loc[change_date, 'high']), 
                               xytext=(5, 10), textcoords='offset points',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor=mode_color, alpha=0.7),
                               fontsize=8, color='white', weight='bold')
        
        # 統計情報の表示
        print(f"\n=== Xマーケットモード統計 ===")
        total_points = len(df[~df['trend_mode'].isna()])
        trend_points = len(df[df['trend_mode'] == 1])
        cycle_points = len(df[df['trend_mode'] == 0])
        
        print(f"総データ点数: {total_points}")
        print(f"トレンドモード: {trend_points} ({trend_points/total_points*100:.1f}%)")
        print(f"サイクルモード: {cycle_points} ({cycle_points/total_points*100:.1f}%)")
        print(f"平均サイクル期間: {df['period'].mean():.1f} (範囲: {df['period'].min():.1f} - {df['period'].max():.1f})")
        print(f"平均モード強度: {df['mode_strength'].mean():.3f}")
        print(f"平均サイクル強度: {df['cycle_strength'].mean():.3f}")
        print(f"平均トレンド強度: {df['trend_strength'].mean():.3f}")
        
        # モード持続期間の分析
        mode_durations = []
        current_mode = None
        current_duration = 0
        
        for mode in df['trend_mode'].dropna():
            if mode == current_mode:
                current_duration += 1
            else:
                if current_mode is not None:
                    mode_durations.append((current_mode, current_duration))
                current_mode = mode
                current_duration = 1
        
        if current_mode is not None:
            mode_durations.append((current_mode, current_duration))
        
        if mode_durations:
            trend_durations = [d for m, d in mode_durations if m == 1]
            cycle_durations = [d for m, d in mode_durations if m == 0]
            
            if trend_durations:
                print(f"トレンドモード平均持続期間: {np.mean(trend_durations):.1f}期間")
            if cycle_durations:
                print(f"サイクルモード平均持続期間: {np.mean(cycle_durations):.1f}期間")
        
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
    parser = argparse.ArgumentParser(description='Xマーケットモードの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--src-type', type=str, default='hlc3', help='ソースタイプ')
    parser.add_argument('--threshold', type=float, default=0.008, help='トレンド判定閾値')
    parser.add_argument('--smoother', type=str, default='frama', help='スムーサータイプ')
    parser.add_argument('--use-kalman', action='store_true', help='カルマンフィルターを使用')
    args = parser.parse_args()
    
    # チャートを作成
    chart = XMarketModeChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        src_type=args.src_type,
        trend_threshold=args.threshold,
        smoother_type=args.smoother,
        use_kalman_filter=args.use_kalman
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output
    )


if __name__ == "__main__":
    main()