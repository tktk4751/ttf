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
from indicators.cycle_donchian import CycleDonchian


class CycleDonchianChart:
    """
    サイクルドンチャンを表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - サイクルドンチャンの上限・下限・中心線バンド
    - サイクル検出器の値
    - サイクル期間値
    - 期間使用率
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.cycle_donchian = None
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
                            # 期間の範囲
                            min_period: int = 20,
                            max_period: int = 400,
                            # エラーズ統合DCパラメータ
                            detector_type: str = 'phac_e',
                            cycle_part: float = 0.7,
                            max_cycle: int = 377,
                            min_cycle: int = 13,
                            max_output: int = 233,
                            min_output: int = 20,
                            src_type: str = 'close',
                            use_kalman_filter: bool = False,
                            kalman_measurement_noise: float = 1.0,
                            kalman_process_noise: float = 0.01,
                            kalman_n_states: int = 5,
                            lp_period: int = 5,
                            hp_period: int = 55,
                            # 新しい検出器用のパラメータ
                            alpha: float = 0.07,
                            bandwidth: float = 0.6,
                            center_period: float = 15.0,
                            avg_length: float = 3.0,
                            window: int = 50
                           ) -> None:
        """
        サイクルドンチャンを計算する
        
        Args:
            min_period: 最小期間（制限用）
            max_period: 最大期間（制限用）
            detector_type: エラーズ統合DC検出器タイプ
            cycle_part: サイクル部分の倍率
            max_cycle: 最大サイクル期間
            min_cycle: 最小サイクル期間
            max_output: 最大出力値
            min_output: 最小出力値
            src_type: ソースタイプ
            use_kalman_filter: カルマンフィルター使用有無
            kalman_measurement_noise: カルマンフィルター測定ノイズ
            kalman_process_noise: カルマンフィルタープロセスノイズ
            kalman_n_states: カルマンフィルター状態数
            lp_period: ローパスフィルター期間（拡張検出器用）
            hp_period: ハイパスフィルター期間（拡張検出器用）
            alpha: アルファパラメータ（新しい検出器用）
            bandwidth: 帯域幅（新しい検出器用）
            center_period: 中心周期（新しい検出器用）
            avg_length: 平均長（新しい検出器用）
            window: 分析ウィンドウ長（新しい検出器用）
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\nサイクルドンチャンを計算中...")
        
        # サイクルドンチャンを計算
        self.cycle_donchian = CycleDonchian(
            min_period=min_period,
            max_period=max_period,
            detector_type=detector_type,
            cycle_part=cycle_part,
            max_cycle=max_cycle,
            min_cycle=min_cycle,
            max_output=max_output,
            min_output=min_output,
            src_type=src_type,
            use_kalman_filter=use_kalman_filter,
            kalman_measurement_noise=kalman_measurement_noise,
            kalman_process_noise=kalman_process_noise,
            kalman_n_states=kalman_n_states,
            lp_period=lp_period,
            hp_period=hp_period,
            alpha=alpha,
            bandwidth=bandwidth,
            center_period=center_period,
            avg_length=avg_length,
            window=window
        )
        
        # CycleDonchianの計算
        print("計算を実行します...")
        self.cycle_donchian.calculate(self.data)
        
        # バンドの取得テスト
        upper, lower, middle = self.cycle_donchian.get_bands()
        cycle_values = self.cycle_donchian.get_cycle_values()
        cycle_periods = self.cycle_donchian.get_cycle_periods()
        
        print(f"バンド計算完了 - 上限: {len(upper)}, 下限: {len(lower)}, 中心線: {len(middle)}")
        print(f"サイクル値: {len(cycle_values)}, サイクル期間: {len(cycle_periods)}")
        
        # NaN値のチェック
        nan_count_upper = np.isnan(upper).sum()
        nan_count_lower = np.isnan(lower).sum()
        nan_count_middle = np.isnan(middle).sum()
        valid_periods = (~np.isnan(cycle_periods)).sum()
        
        print(f"NaN値 - 上限: {nan_count_upper}, 下限: {nan_count_lower}, 中心線: {nan_count_middle}")
        print(f"有効期間数: {valid_periods}")
        print(f"期間範囲: {np.nanmin(cycle_periods):.1f} - {np.nanmax(cycle_periods):.1f}")
        print(f"サイクル値範囲: {np.nanmin(cycle_values):.3f} - {np.nanmax(cycle_values):.3f}")
        
        print("サイクルドンチャン計算完了")
            
    def plot(self, 
            title: str = "サイクルドンチャン", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (14, 12),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとサイクルドンチャンを描画する
        
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
            
        if self.cycle_donchian is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # サイクルドンチャンの値を取得
        print("バンドデータを取得中...")
        upper, lower, middle = self.cycle_donchian.get_bands()
        cycle_values = self.cycle_donchian.get_cycle_values()
        cycle_periods = self.cycle_donchian.get_cycle_periods()
        
        # 詳細結果から期間使用率を取得
        detailed_result = self.cycle_donchian.get_detailed_result()
        period_usage = detailed_result.period_usage
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'donchian_upper': upper,
                'donchian_lower': lower,
                'donchian_middle': middle,
                'cycle_values': cycle_values,
                'cycle_periods': cycle_periods,
                'period_usage': period_usage
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        print(f"バンドデータ確認 - 上限NaN: {df['donchian_upper'].isna().sum()}, 下限NaN: {df['donchian_lower'].isna().sum()}")
        
        # チャネルブレイクアウトの判定（参考情報として）
        df['price_vs_upper'] = np.where(df['close'] > df['donchian_upper'], 1, 0)  # 上抜け
        df['price_vs_lower'] = np.where(df['close'] < df['donchian_lower'], -1, 0)  # 下抜け
        df['breakout_signal'] = df['price_vs_upper'] + df['price_vs_lower']
        
        # NaN値を含む行を出力（最初の5行のみ）
        nan_rows = df[df['donchian_upper'].isna() | df['donchian_middle'].isna() | df['donchian_lower'].isna()]
        if not nan_rows.empty:
            print(f"NaN値を含む行: {len(nan_rows)}行")
            if len(nan_rows) > 0:
                print(nan_rows.head())
        
        # mplfinanceでプロット用の設定
        # 1. メインチャート上のプロット
        main_plots = []
        
        # サイクルドンチャンのプロット設定
        main_plots.append(mpf.make_addplot(df['donchian_upper'], color='red', width=1.5, alpha=0.8, label='Upper Band'))
        main_plots.append(mpf.make_addplot(df['donchian_lower'], color='green', width=1.5, alpha=0.8, label='Lower Band'))
        main_plots.append(mpf.make_addplot(df['donchian_middle'], color='blue', width=1.2, alpha=0.9, label='Middle Line'))
        
        # 2. オシレータープロット
        # サイクル値パネル
        cycle_panel = mpf.make_addplot(df['cycle_values'], panel=1, color='purple', width=1.2, 
                                      ylabel='Cycle Values', secondary_y=False, label='Cycle')
        
        # サイクル期間パネル
        period_panel = mpf.make_addplot(df['cycle_periods'], panel=2, color='orange', width=1.2, 
                                       ylabel='Cycle Period', secondary_y=False, label='Period')
        
        # 期間使用率パネル
        usage_panel = mpf.make_addplot(df['period_usage'], panel=3, color='brown', width=1.2, 
                                      ylabel='Period Usage', secondary_y=False, label='Usage Rate')
        
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
            kwargs['panel_ratios'] = (4, 1, 1, 1, 1)  # メイン:出来高:サイクル値:期間:使用率
            # 出来高を表示する場合は、オシレーターのパネル番号を+1する
            cycle_panel = mpf.make_addplot(df['cycle_values'], panel=2, color='purple', width=1.2, 
                                          ylabel='Cycle Values', secondary_y=False, label='Cycle')
            period_panel = mpf.make_addplot(df['cycle_periods'], panel=3, color='orange', width=1.2, 
                                           ylabel='Cycle Period', secondary_y=False, label='Period')
            usage_panel = mpf.make_addplot(df['period_usage'], panel=4, color='brown', width=1.2, 
                                          ylabel='Period Usage', secondary_y=False, label='Usage Rate')
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (4, 1, 1, 1)  # メイン:サイクル値:期間:使用率
        
        # すべてのプロットを結合
        all_plots = main_plots + [cycle_panel, period_panel, usage_panel]
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加
        axes[0].legend(['Upper Band', 'Lower Band', 'Middle Line'], loc='upper left')
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        if show_volume:
            # サイクル値パネル
            cycle_mean = df['cycle_values'].mean()
            axes[2].axhline(y=cycle_mean, color='black', linestyle='-', alpha=0.3)
            axes[2].axhline(y=self.cycle_donchian.min_output, color='green', linestyle='--', alpha=0.5)
            axes[2].axhline(y=self.cycle_donchian.max_output, color='red', linestyle='--', alpha=0.5)
            
            # サイクル期間パネル
            period_mean = df['cycle_periods'].mean()
            axes[3].axhline(y=period_mean, color='black', linestyle='-', alpha=0.3)
            axes[3].axhline(y=self.cycle_donchian.min_period, color='green', linestyle='--', alpha=0.5)
            axes[3].axhline(y=self.cycle_donchian.max_period, color='red', linestyle='--', alpha=0.5)
            
            # 期間使用率パネル
            axes[4].axhline(y=0.5, color='black', linestyle='-', alpha=0.5)
            axes[4].axhline(y=0.0, color='green', linestyle='--', alpha=0.5)
            axes[4].axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
        else:
            # サイクル値パネル
            cycle_mean = df['cycle_values'].mean()
            axes[1].axhline(y=cycle_mean, color='black', linestyle='-', alpha=0.3)
            axes[1].axhline(y=self.cycle_donchian.min_output, color='green', linestyle='--', alpha=0.5)
            axes[1].axhline(y=self.cycle_donchian.max_output, color='red', linestyle='--', alpha=0.5)
            
            # サイクル期間パネル
            period_mean = df['cycle_periods'].mean()
            axes[2].axhline(y=period_mean, color='black', linestyle='-', alpha=0.3)
            axes[2].axhline(y=self.cycle_donchian.min_period, color='green', linestyle='--', alpha=0.5)
            axes[2].axhline(y=self.cycle_donchian.max_period, color='red', linestyle='--', alpha=0.5)
            
            # 期間使用率パネル
            axes[3].axhline(y=0.5, color='black', linestyle='-', alpha=0.5)
            axes[3].axhline(y=0.0, color='green', linestyle='--', alpha=0.5)
            axes[3].axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
        
        # 統計情報の表示
        print(f"\n=== サイクルドンチャンチャネル統計 ===")
        total_points = len(df.dropna())
        breakouts = len(df[df['breakout_signal'] != 0])
        upper_breaks = len(df[df['breakout_signal'] == 1])
        lower_breaks = len(df[df['breakout_signal'] == -1])
        
        print(f"総データ点数: {total_points}")
        print(f"ブレイクアウト総数: {breakouts} ({breakouts/total_points*100:.1f}%)")
        print(f"上抜け: {upper_breaks} ({upper_breaks/total_points*100:.1f}%)")
        print(f"下抜け: {lower_breaks} ({lower_breaks/total_points*100:.1f}%)")
        print(f"サイクル期間 - 平均: {df['cycle_periods'].mean():.1f}, 範囲: {df['cycle_periods'].min():.1f} - {df['cycle_periods'].max():.1f}")
        print(f"サイクル値 - 平均: {df['cycle_values'].mean():.3f}, 範囲: {df['cycle_values'].min():.3f} - {df['cycle_values'].max():.3f}")
        print(f"期間使用率 - 平均: {df['period_usage'].mean():.3f}")
        print(f"検出器: {self.cycle_donchian.detector_type}, ソース: {self.cycle_donchian.src_type}")
        
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
    parser = argparse.ArgumentParser(description='サイクルドンチャンの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--min-period', type=int, default=6, help='最小期間')
    parser.add_argument('--max-period', type=int, default=50, help='最大期間')
    parser.add_argument('--detector', type=str, default='phac_e', help='検出器タイプ')
    parser.add_argument('--src-type', type=str, default='close', help='価格ソースタイプ')
    args = parser.parse_args()
    
    # チャートを作成
    chart = CycleDonchianChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        min_period=args.min_period,
        max_period=args.max_period,
        detector_type=args.detector,
        src_type=args.src_type
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output
    )


if __name__ == "__main__":
    main() 