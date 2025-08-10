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
from indicators.x_keltner_channel import XKeltnerChannel


class XKeltnerChannelChart:
    """
    Xケルトナーチャネルを表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - ケルトナーチャネル（ミッドライン・上部チャネル・下部チャネル）
    - 価格位置インジケーター
    - 動的期間とバンド幅
    - ATR値の表示
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.x_keltner_channel = None
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
                            # 基本パラメータ
                            period: int = 20,
                            multiplier: float = 2.0,
                            src_type: str = 'hlc3',
                            # カルマンフィルターパラメータ
                            use_kalman_filter: bool = True,
                            kalman_measurement_noise: float = 1.0,
                            kalman_process_noise: float = 0.01,
                            # 動的期間パラメータ
                            use_dynamic_period: bool = True,
                            dc_detector_type: str = 'hody_e',
                            dc_min_period: int = 6,
                            dc_max_period: int = 50,
                            # 統合スムーサーパラメータ
                            smoother_type: str = 'super_smoother',
                            smoother_period: int = 20,
                            use_dynamic_smoother_period: bool = True,
                            # X_ATRパラメータ
                            atr_period: int = 14,
                            atr_smoothing: str = 'zlema',
                            adaptive_multiplier: bool = False
                           ) -> None:
        """
        Xケルトナーチャネルを計算する
        
        Args:
            period: 基本期間（デフォルト: 20）
            multiplier: ATR乗数（デフォルト: 2.0）
            src_type: ソースタイプ（デフォルト: 'hlc3'）
            use_kalman_filter: カルマンフィルター使用（デフォルト: True）
            kalman_measurement_noise: カルマン測定ノイズ（デフォルト: 1.0）
            kalman_process_noise: カルマンプロセスノイズ（デフォルト: 0.01）
            use_dynamic_period: 動的期間使用（デフォルト: True）
            dc_detector_type: サイクル検出器タイプ（デフォルト: 'hody_e'）
            dc_min_period: 最小サイクル期間（デフォルト: 6）
            dc_max_period: 最大サイクル期間（デフォルト: 50）
            smoother_type: 統合スムーサータイプ（デフォルト: 'frama'）
            smoother_period: スムーサー期間（デフォルト: 20）
            use_dynamic_smoother_period: スムーサーで動的期間使用（デフォルト: True）
            atr_period: ATR期間（デフォルト: 14）
            atr_smoothing: ATRスムージングタイプ（デフォルト: 'ultimate_smoother'）
            adaptive_multiplier: 適応的乗数（デフォルト: False）
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\nXケルトナーチャネルを計算中...")
        
        # Xケルトナーチャネルを計算
        self.x_keltner_channel = XKeltnerChannel(
            period=period,
            multiplier=multiplier,
            src_type=src_type,
            use_kalman_filter=use_kalman_filter,
            kalman_measurement_noise=kalman_measurement_noise,
            kalman_process_noise=kalman_process_noise,
            use_dynamic_period=use_dynamic_period,
            dc_detector_type=dc_detector_type,
            dc_min_period=dc_min_period,
            dc_max_period=dc_max_period,
            smoother_type=smoother_type,
            smoother_period=smoother_period,
            use_dynamic_smoother_period=use_dynamic_smoother_period,
            atr_period=atr_period,
            atr_smoothing=atr_smoothing,
            adaptive_multiplier=adaptive_multiplier
        )
        
        # XKeltnerChannelの計算
        print("計算を実行します...")
        result = self.x_keltner_channel.calculate(self.data)
        
        # チャネルデータの取得テスト
        midline = result.midline_values
        upper = result.upper_channel
        lower = result.lower_channel
        print(f"チャネル計算完了 - ミッドライン: {len(midline)}, 上部: {len(upper)}, 下部: {len(lower)}")
        
        # NaN値のチェック
        nan_count_midline = np.isnan(midline).sum()
        nan_count_upper = np.isnan(upper).sum()
        nan_count_lower = np.isnan(lower).sum()
        print(f"NaN値 - ミッドライン: {nan_count_midline}, 上部: {nan_count_upper}, 下部: {nan_count_lower}")
        
        # 統計情報
        print(f"ATR値 - 平均: {np.nanmean(result.atr_values):.4f}, 範囲: {np.nanmin(result.atr_values):.4f} - {np.nanmax(result.atr_values):.4f}")
        print(f"バンド幅 - 平均: {np.nanmean(result.bandwidth):.2f}%, 範囲: {np.nanmin(result.bandwidth):.2f}% - {np.nanmax(result.bandwidth):.2f}%")
        
        print("Xケルトナーチャネル計算完了")
            
    def plot(self, 
            title: str = "Xケルトナーチャネル", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (14, 12),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとXケルトナーチャネルを描画する
        
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
            
        if self.x_keltner_channel is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # Xケルトナーチャネルの値を取得
        print("チャネルデータを取得中...")
        midline = self.x_keltner_channel.get_midline_values()
        upper = self.x_keltner_channel.get_upper_channel()
        lower = self.x_keltner_channel.get_lower_channel()
        atr_values = self.x_keltner_channel.get_atr_values()
        dynamic_period = self.x_keltner_channel.get_dynamic_period()
        bandwidth = self.x_keltner_channel.get_bandwidth()
        position = self.x_keltner_channel.get_position()
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'kc_midline': midline,
                'kc_upper': upper,
                'kc_lower': lower,
                'atr_values': atr_values,
                'dynamic_period': dynamic_period,
                'bandwidth': bandwidth,
                'position': position
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        print(f"チャネルデータ確認 - ミッドラインNaN: {df['kc_midline'].isna().sum()}, 上部NaN: {df['kc_upper'].isna().sum()}")
        
        # NaN値を含む行を出力（最初の5行のみ）
        nan_rows = df[df['kc_upper'].isna() | df['kc_midline'].isna() | df['kc_lower'].isna()]
        if not nan_rows.empty:
            print(f"NaN値を含む行: {len(nan_rows)}行")
            if len(nan_rows) > 0:
                print(nan_rows.head())
        
        # mplfinanceでプロット用の設定
        # 1. メインチャート上のプロット
        main_plots = []
        
        # ケルトナーチャネルのプロット設定
        main_plots.append(mpf.make_addplot(df['kc_midline'], color='blue', width=2, label='KC Midline'))
        main_plots.append(mpf.make_addplot(df['kc_upper'], color='red', width=1.5, alpha=0.7, label='KC Upper'))
        main_plots.append(mpf.make_addplot(df['kc_lower'], color='green', width=1.5, alpha=0.7, label='KC Lower'))
        
        # 価格位置に基づく色分け（オプション）
        # NaN値を含む場合の安全な処理
        valid_mask = ~(df['position'].isna() | df['kc_upper'].isna() | df['kc_lower'].isna())
        df['price_above_upper'] = np.where(
            valid_mask & (df['position'] > 0.8), 
            df['close'], 
            np.nan
        )
        df['price_below_lower'] = np.where(
            valid_mask & (df['position'] < -0.8), 
            df['close'], 
            np.nan
        )
        
        # 有効なデータがある場合のみマーカーを追加
        if not df['price_above_upper'].isna().all():
            main_plots.append(mpf.make_addplot(df['price_above_upper'], type='scatter', markersize=30, color='red', alpha=0.6, label='Above Upper'))
        if not df['price_below_lower'].isna().all():
            main_plots.append(mpf.make_addplot(df['price_below_lower'], type='scatter', markersize=30, color='green', alpha=0.6, label='Below Lower'))
        
        # 2. オシレータープロット
        # 価格位置パネル
        position_panel = mpf.make_addplot(df['position'], panel=1, color='purple', width=1.2, 
                                        ylabel='Price Position', secondary_y=False, label='Position')
        
        # ATR値パネル
        atr_panel = mpf.make_addplot(df['atr_values'], panel=2, color='orange', width=1.2, 
                                   ylabel='ATR Values', secondary_y=False, label='ATR')
        
        # バンド幅パネル
        bandwidth_panel = mpf.make_addplot(df['bandwidth'], panel=3, color='brown', width=1.2, 
                                         ylabel='Bandwidth (%)', secondary_y=False, label='Bandwidth')
        
        # 動的期間パネル
        period_panel = mpf.make_addplot(df['dynamic_period'], panel=4, color='navy', width=1.5, 
                                      ylabel='Dynamic Period', secondary_y=False, label='Period', type='line')
        
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
            kwargs['panel_ratios'] = (4, 1, 1, 1, 1, 1)  # メイン:出来高:ポジション:ATR:バンド幅:期間
            # 出来高を表示する場合は、オシレーターのパネル番号を+1する
            position_panel = mpf.make_addplot(df['position'], panel=2, color='purple', width=1.2, 
                                            ylabel='Price Position', secondary_y=False, label='Position')
            atr_panel = mpf.make_addplot(df['atr_values'], panel=3, color='orange', width=1.2, 
                                       ylabel='ATR Values', secondary_y=False, label='ATR')
            bandwidth_panel = mpf.make_addplot(df['bandwidth'], panel=4, color='brown', width=1.2, 
                                             ylabel='Bandwidth (%)', secondary_y=False, label='Bandwidth')
            period_panel = mpf.make_addplot(df['dynamic_period'], panel=5, color='navy', width=1.5, 
                                          ylabel='Dynamic Period', secondary_y=False, label='Period', type='line')
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (4, 1, 1, 1, 1)  # メイン:ポジション:ATR:バンド幅:期間
        
        # すべてのプロットを結合
        all_plots = main_plots + [position_panel, atr_panel, bandwidth_panel, period_panel]
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加
        axes[0].legend(['KC Midline', 'KC Upper', 'KC Lower', 'Above Upper', 'Below Lower'], 
                      loc='upper left')
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        if show_volume:
            # 価格位置パネル
            axes[2].axhline(y=0.0, color='black', linestyle='-', alpha=0.5)
            axes[2].axhline(y=0.8, color='red', linestyle='--', alpha=0.5)
            axes[2].axhline(y=-0.8, color='green', linestyle='--', alpha=0.5)
            axes[2].axhline(y=1.0, color='red', linestyle=':', alpha=0.3)
            axes[2].axhline(y=-1.0, color='green', linestyle=':', alpha=0.3)
            
            # ATR値パネル
            atr_mean = df['atr_values'].mean()
            axes[3].axhline(y=atr_mean, color='black', linestyle='-', alpha=0.3)
            
            # バンド幅パネル
            bandwidth_mean = df['bandwidth'].mean()
            axes[4].axhline(y=bandwidth_mean, color='black', linestyle='-', alpha=0.3)
            
            # 動的期間パネル
            period_mean = df['dynamic_period'].mean()
            axes[5].axhline(y=period_mean, color='black', linestyle='-', alpha=0.3)
        else:
            # 価格位置パネル
            axes[1].axhline(y=0.0, color='black', linestyle='-', alpha=0.5)
            axes[1].axhline(y=0.8, color='red', linestyle='--', alpha=0.5)
            axes[1].axhline(y=-0.8, color='green', linestyle='--', alpha=0.5)
            axes[1].axhline(y=1.0, color='red', linestyle=':', alpha=0.3)
            axes[1].axhline(y=-1.0, color='green', linestyle=':', alpha=0.3)
            
            # ATR値パネル
            atr_mean = df['atr_values'].mean()
            axes[2].axhline(y=atr_mean, color='black', linestyle='-', alpha=0.3)
            
            # バンド幅パネル
            bandwidth_mean = df['bandwidth'].mean()
            axes[3].axhline(y=bandwidth_mean, color='black', linestyle='-', alpha=0.3)
            
            # 動的期間パネル
            period_mean = df['dynamic_period'].mean()
            axes[4].axhline(y=period_mean, color='black', linestyle='-', alpha=0.3)
        
        # 統計情報の表示
        print(f"\n=== ケルトナーチャネル統計 ===")
        # 有効なデータのみで統計を計算
        valid_data = df[~(df['position'].isna() | df['kc_upper'].isna() | df['kc_lower'].isna())]
        
        if len(valid_data) > 0:
            above_upper = len(valid_data[valid_data['position'] > 0.8])
            below_lower = len(valid_data[valid_data['position'] < -0.8])
            in_channel = len(valid_data[(valid_data['position'] >= -0.8) & (valid_data['position'] <= 0.8)])
            total_valid = len(valid_data)
        else:
            above_upper = below_lower = in_channel = total_valid = 0
        
        print(f"総データ点数: {total_valid}")
        if total_valid > 0:
            print(f"上部チャネル超過: {above_upper} ({above_upper/total_valid*100:.1f}%)")
            print(f"下部チャネル下抜け: {below_lower} ({below_lower/total_valid*100:.1f}%)")
            print(f"チャネル内: {in_channel} ({in_channel/total_valid*100:.1f}%)")
        else:
            print("上部チャネル超過: 0 (0.0%)")
            print("下部チャネル下抜け: 0 (0.0%)")
            print("チャネル内: 0 (0.0%)")
        
        # 各指標の統計も安全に計算
        atr_valid = df['atr_values'].dropna()
        bandwidth_valid = df['bandwidth'].dropna()
        period_valid = df['dynamic_period'].dropna()
        
        if len(atr_valid) > 0:
            print(f"ATR値 - 平均: {atr_valid.mean():.4f}, 範囲: {atr_valid.min():.4f} - {atr_valid.max():.4f}")
        else:
            print("ATR値 - データなし")
            
        if len(bandwidth_valid) > 0:
            print(f"バンド幅 - 平均: {bandwidth_valid.mean():.2f}%, 範囲: {bandwidth_valid.min():.2f}% - {bandwidth_valid.max():.2f}%")
        else:
            print("バンド幅 - データなし")
            
        if len(period_valid) > 0:
            print(f"動的期間 - 平均: {period_valid.mean():.1f}, 範囲: {period_valid.min():.0f} - {period_valid.max():.0f}")
        else:
            print("動的期間 - データなし")
        
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
    parser = argparse.ArgumentParser(description='Xケルトナーチャネルの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--period', type=int, default=20, help='基本期間')
    parser.add_argument('--multiplier', type=float, default=2.0, help='ATR乗数')
    parser.add_argument('--src-type', type=str, default='hlc3', help='ソースタイプ')
    parser.add_argument('--smoother', type=str, default='super_smoother', help='スムーサータイプ')
    parser.add_argument('--atr-period', type=int, default=14, help='ATR期間')
    args = parser.parse_args()
    
    # チャートを作成
    chart = XKeltnerChannelChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        period=args.period,
        multiplier=args.multiplier,
        src_type=args.src_type,
        smoother_type=args.smoother,
        atr_period=args.atr_period
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output
    )


if __name__ == "__main__":
    main()