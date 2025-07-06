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
from indicators.ultimate_trend import UltimateTrend


class UltimateTrendChart:
    """
    アルティメットトレンドを表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - アルティメットトレンドライン（トレンド方向に応じた色分け）
    - 上側・下側バンド（スーパートレンドロジック）
    - Ultimate MAフィルタ済みミッドライン
    - カルマンフィルター後の価格
    - ATR値の表示
    - トレンド方向の表示
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.ultimate_trend = None
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
                            # アルティメットトレンドパラメータ
                            length: int = 13,
                            multiplier: float = 2.0,
                            super_smooth_period: int = 10,
                            zero_lag_period: int = 21,
                            filtering_mode: int = 1
                           ) -> None:
        """
        アルティメットトレンドを計算する
        
        Args:
            length: ATR期間
            multiplier: ATR乗数
            super_smooth_period: スーパースムーザー期間
            zero_lag_period: ゼロラグEMA期間
            filtering_mode: フィルタリングモード
                           0 = ゼロラグEMA後の値をミッドラインに使用（①②③まで）
                           1 = 完全5段階フィルタ済み値をミッドラインに使用（①②③④⑤まで）
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\nアルティメットトレンドを計算中...")
        
        # アルティメットトレンドを計算
        self.ultimate_trend = UltimateTrend(
            length=length,
            multiplier=multiplier,
            super_smooth_period=super_smooth_period,
            zero_lag_period=zero_lag_period,
            filtering_mode=filtering_mode
        )
        
        # アルティメットトレンドの計算
        print("計算を実行します...")
        result = self.ultimate_trend.calculate(self.data)
        
        # 結果の確認
        print(f"アルティメットトレンド計算完了")
        print(f"アルティメットトレンドライン: {len(result.values)}")
        print(f"上側バンド: {len(result.upper_band)}")
        print(f"下側バンド: {len(result.lower_band)}")
        print(f"トレンド方向: {len(result.trend)}")
        print(f"ATR値: {len(result.atr_values)}")
        print(f"フィルタ済みミッドライン: {len(result.filtered_midline)}")
        print(f"元HLC3ミッドライン: {len(result.raw_midline)}")
        
        # NaN値のチェック
        nan_count_ut = np.isnan(result.values).sum()
        nan_count_upper = np.isnan(result.upper_band).sum()
        nan_count_lower = np.isnan(result.lower_band).sum()
        nan_count_atr = np.isnan(result.atr_values).sum()
        nan_count_filtered = np.isnan(result.filtered_midline).sum()
        nan_count_raw = np.isnan(result.raw_midline).sum()
        trend_count = (result.trend != 0).sum()
        
        print(f"NaN値 - UT: {nan_count_ut}, 上側: {nan_count_upper}, 下側: {nan_count_lower}, ATR: {nan_count_atr}")
        print(f"NaN値 - フィルタ済み: {nan_count_filtered}, 元HLC3: {nan_count_raw}")
        print(f"トレンド値 - 有効: {trend_count}, 上昇: {(result.trend == 1).sum()}, 下降: {(result.trend == -1).sum()}")
        
        # フィルタリング効果の確認
        filtering_stats = self.ultimate_trend.get_filtering_stats()
        print(f"フィルタリング効果:")
        print(f"  元HLC3標準偏差: {filtering_stats.get('raw_hlc3_volatility', 0):.6f}")
        print(f"  フィルタ済み標準偏差: {filtering_stats.get('filtered_hlc3_volatility', 0):.6f}")
        print(f"  ノイズ除去率: {filtering_stats.get('noise_reduction_percentage', 0):.2f}%")
        
        # 値の範囲をチェック
        if nan_count_ut < len(result.values):
            valid_ut = result.values[~np.isnan(result.values)]
            print(f"UT値範囲: {valid_ut.min():.4f} - {valid_ut.max():.4f}")
        if nan_count_upper < len(result.upper_band):
            valid_upper = result.upper_band[~np.isnan(result.upper_band)]
            print(f"上側バンド範囲: {valid_upper.min():.4f} - {valid_upper.max():.4f}")
        if nan_count_lower < len(result.lower_band):
            valid_lower = result.lower_band[~np.isnan(result.lower_band)]
            print(f"下側バンド範囲: {valid_lower.min():.4f} - {valid_lower.max():.4f}")
        
        print("アルティメットトレンド計算完了")
            
    def plot(self, 
            title: str = "アルティメットトレンド", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            show_bands: bool = True,
            figsize: Tuple[int, int] = (14, 10),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとアルティメットトレンドを描画する
        
        Args:
            title: チャートのタイトル
            start_date: 表示開始日（フォーマット: YYYY-MM-DD）
            end_date: 表示終了日（フォーマット: YYYY-MM-DD）
            show_volume: 出来高を表示するか
            show_bands: バンドを表示するか
            figsize: 図のサイズ
            style: mplfinanceのスタイル
            savefig: 保存先のパス（指定しない場合は表示のみ）
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        if self.ultimate_trend is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # アルティメットトレンドの結果を取得
        print("アルティメットトレンドデータを取得中...")
        result = self.ultimate_trend.calculate(self.data)
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'ut_line': result.values,
                'ut_upper': result.upper_band,
                'ut_lower': result.lower_band,
                'ut_final_upper': result.final_upper_band,
                'ut_final_lower': result.final_lower_band,
                'ut_trend': result.trend,
                'atr_values': result.atr_values,
                'filtered_midline': result.filtered_midline,
                'raw_midline': result.raw_midline,
                'kalman_values': result.kalman_values,
                'super_smooth_values': result.super_smooth_values,
                'zero_lag_values': result.zero_lag_values,
                'amplitude': result.amplitude,
                'phase': result.phase
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        print(f"アルティメットトレンドデータ確認 - UTライン NaN: {df['ut_line'].isna().sum()}")
        
        # トレンド方向に基づくアルティメットトレンドラインの色分け
        df['ut_uptrend'] = np.where(df['ut_trend'] == 1, df['ut_line'], np.nan)
        df['ut_downtrend'] = np.where(df['ut_trend'] == -1, df['ut_line'], np.nan)
        
        # バンドの表示用（必要に応じて）
        if show_bands:
            df['upper_band_display'] = df['ut_upper']
            df['lower_band_display'] = df['ut_lower']
            df['final_upper_band_display'] = df['ut_final_upper']
            df['final_lower_band_display'] = df['ut_final_lower']
        
        # NaN値の問題をチェックして報告
        print(f"データ診断:")
        print(f"  UT上昇トレンド有効値: {(~df['ut_uptrend'].isna()).sum()}")
        print(f"  UT下降トレンド有効値: {(~df['ut_downtrend'].isna()).sum()}")
        if show_bands:
            print(f"  上側バンド有効値: {(~df['upper_band_display'].isna()).sum()}")
            print(f"  下側バンド有効値: {(~df['lower_band_display'].isna()).sum()}")
            print(f"  調整済み上側バンド有効値: {(~df['final_upper_band_display'].isna()).sum()}")
            print(f"  調整済み下側バンド有効値: {(~df['final_lower_band_display'].isna()).sum()}")
        
        # mplfinanceでプロット用の設定
        # 1. メインチャート上のプロット
        main_plots = []
        
        # アルティメットトレンドラインのプロット設定（有効なデータがある場合のみ）
        if (~df['ut_uptrend'].isna()).sum() > 0:
            main_plots.append(mpf.make_addplot(df['ut_uptrend'], color='green', width=3, label='Ultimate Trend (Up)'))
        if (~df['ut_downtrend'].isna()).sum() > 0:
            main_plots.append(mpf.make_addplot(df['ut_downtrend'], color='red', width=3, label='Ultimate Trend (Down)'))
        
        # バンドの表示（有効なデータがある場合のみ）
        if show_bands:
            if (~df['upper_band_display'].isna()).sum() > 0:
                main_plots.append(mpf.make_addplot(df['upper_band_display'], color='gray', width=1, alpha=0.5, label='Upper Band'))
            if (~df['lower_band_display'].isna()).sum() > 0:
                main_plots.append(mpf.make_addplot(df['lower_band_display'], color='gray', width=1, alpha=0.5, label='Lower Band'))
            # 調整済みバンドの表示（点線で区別）
            if (~df['final_upper_band_display'].isna()).sum() > 0:
                main_plots.append(mpf.make_addplot(df['final_upper_band_display'], color='orange', width=1.5, alpha=0.7, linestyle='--', label='Final Upper Band'))
            if (~df['final_lower_band_display'].isna()).sum() > 0:
                main_plots.append(mpf.make_addplot(df['final_lower_band_display'], color='orange', width=1.5, alpha=0.7, linestyle='--', label='Final Lower Band'))
        
        # Ultimate MAフィルタ済みミッドラインの表示
        if (~df['filtered_midline'].isna()).sum() > 0:
            main_plots.append(mpf.make_addplot(df['filtered_midline'], color='blue', width=1.5, alpha=0.7, label='Ultimate MA'))
        
        # カルマンフィルター後の価格表示
        if (~df['kalman_values'].isna()).sum() > 0:
            main_plots.append(mpf.make_addplot(df['kalman_values'], color='orange', width=1.2, alpha=0.8, label='Kalman Filtered'))
        
        # 2. 追加パネルのプロット
        panel_idx = 1 if not show_volume else 2
        
        # 追加プロットリスト
        additional_plots = []
        
        # ATR値パネル（有効なデータがある場合のみ）
        if (~df['atr_values'].isna()).sum() > 0:
            atr_panel = mpf.make_addplot(df['atr_values'], panel=panel_idx, color='blue', width=1.2, 
                                        ylabel='ATR', secondary_y=False, label='ATR')
            additional_plots.append(atr_panel)
        
        # トレンド方向パネル（有効なデータがある場合のみ）
        if (df['ut_trend'] != 0).sum() > 0:
            trend_panel = mpf.make_addplot(df['ut_trend'], panel=panel_idx+1, color='orange', width=1.5, 
                                          ylabel='Trend Direction', secondary_y=False, label='Trend', type='line')
            additional_plots.append(trend_panel)
        
        # mplfinanceの設定
        kwargs = dict(
            type='candle',
            figsize=figsize,
            title=f"{title} (フィルタリングモード: {result.filtering_mode})",
            style=style,
            datetime_format='%Y-%m-%d',
            xrotation=45,
            returnfig=True
        )
        
        # パネル比率の設定（シンプル化）
        if show_volume:
            kwargs['panel_ratios'] = (4, 1, 1, 1)  # メイン:出来高:ATR:トレンド
        else:
            kwargs['panel_ratios'] = (4, 1, 1)  # メイン:ATR:トレンド
        
        kwargs['volume'] = show_volume
        
        # すべてのプロットを結合
        all_plots = main_plots + additional_plots
        
        # プロットが空でないかチェック
        if len(all_plots) == 0:
            print("⚠️  警告: 表示可能なプロットデータがありません。チャートを表示できません。")
            return
        
        kwargs['addplot'] = all_plots
        
        # プロット実行
        try:
            fig, axes = mpf.plot(df, **kwargs)
        except Exception as e:
            print(f"⚠️  プロットエラー: {e}")
            print("データの詳細診断:")
            for i, plot in enumerate(all_plots):
                data = plot['data']
                valid_count = (~pd.isna(data)).sum() if hasattr(data, 'isna') else len([x for x in data if not np.isnan(x)])
                print(f"  プロット{i}: 有効データ数 = {valid_count}")
            return
        
        # 凡例の追加（プロットが成功した場合のみ）
        legend_labels = []
        if (~df['ut_uptrend'].isna()).sum() > 0:
            legend_labels.append('Ultimate Trend (Up)')
        if (~df['ut_downtrend'].isna()).sum() > 0:
            legend_labels.append('Ultimate Trend (Down)')
        if show_bands:
            if (~df['upper_band_display'].isna()).sum() > 0:
                legend_labels.append('Upper Band')
            if (~df['lower_band_display'].isna()).sum() > 0:
                legend_labels.append('Lower Band')
            if (~df['final_upper_band_display'].isna()).sum() > 0:
                legend_labels.append('Final Upper Band')
            if (~df['final_lower_band_display'].isna()).sum() > 0:
                legend_labels.append('Final Lower Band')
        if (~df['filtered_midline'].isna()).sum() > 0:
            legend_labels.append('Ultimate MA')
        if (~df['kalman_values'].isna()).sum() > 0:
            legend_labels.append('Kalman Filtered')
        
        if len(legend_labels) > 0:
            axes[0].legend(legend_labels, loc='upper left')
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        atr_panel_idx = 1 if not show_volume else 2
        trend_panel_idx = atr_panel_idx + 1
        
        # ATR値パネル
        if (~df['atr_values'].isna()).sum() > 0:
            atr_mean = df['atr_values'].mean()
            axes[atr_panel_idx].axhline(y=atr_mean, color='black', linestyle='-', alpha=0.3, label=f'ATR平均: {atr_mean:.4f}')
        
        # トレンド方向パネル
        axes[trend_panel_idx].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[trend_panel_idx].axhline(y=1, color='green', linestyle='--', alpha=0.5, label='上昇トレンド')
        axes[trend_panel_idx].axhline(y=-1, color='red', linestyle='--', alpha=0.5, label='下降トレンド')
        
        # 統計情報の表示
        print(f"\n=== アルティメットトレンド統計 ===")
        total_points = len(df[df['ut_trend'] != 0])
        uptrend_points = len(df[df['ut_trend'] == 1])
        downtrend_points = len(df[df['ut_trend'] == -1])
        
        print(f"総データ点数: {total_points}")
        print(f"上昇トレンド: {uptrend_points} ({uptrend_points/total_points*100:.1f}%)")
        print(f"下降トレンド: {downtrend_points} ({downtrend_points/total_points*100:.1f}%)")
        print(f"ATR値 - 平均: {df['atr_values'].mean():.4f}, 範囲: {df['atr_values'].min():.4f} - {df['atr_values'].max():.4f}")
        
        # フィルタリング効果の統計
        filtering_stats = self.ultimate_trend.get_filtering_stats()
        print(f"フィルタリング効果:")
        print(f"  元HLC3標準偏差: {filtering_stats.get('raw_hlc3_volatility', 0):.6f}")
        print(f"  フィルタ済み標準偏差: {filtering_stats.get('filtered_hlc3_volatility', 0):.6f}")
        print(f"  ノイズ除去率: {filtering_stats.get('noise_reduction_percentage', 0):.2f}%")
        print(f"  フィルタリングモード: {result.filtering_mode} ({'ゼロラグEMA後まで' if result.filtering_mode == 0 else 'Ultimate MA完全フィルタ'})")
        
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
    parser = argparse.ArgumentParser(description='アルティメットトレンドの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--length', type=int, default=13, help='ATR期間')
    parser.add_argument('--multiplier', type=float, default=2.0, help='ATR乗数')
    parser.add_argument('--super-smooth-period', type=int, default=10, help='スーパースムーザー期間')
    parser.add_argument('--zero-lag-period', type=int, default=21, help='ゼロラグEMA期間')
    parser.add_argument('--filtering-mode', type=int, default=1, choices=[0, 1], 
                       help='フィルタリングモード (0: ゼロラグEMA後まで, 1: Ultimate MA完全フィルタ)')
    parser.add_argument('--no-bands', action='store_true', help='バンドを非表示')
    parser.add_argument('--no-volume', action='store_true', help='出来高を非表示')
    args = parser.parse_args()
    
    # チャートを作成
    chart = UltimateTrendChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        length=args.length,
        multiplier=args.multiplier,
        super_smooth_period=args.super_smooth_period,
        zero_lag_period=args.zero_lag_period,
        filtering_mode=args.filtering_mode
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        show_volume=not args.no_volume,
        show_bands=not args.no_bands,
        savefig=args.output
    )


if __name__ == "__main__":
    main() 