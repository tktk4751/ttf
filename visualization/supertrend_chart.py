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
from indicators.supertrend import Supertrend


class SupertrendChart:
    """
    スーパートレンドを表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - スーパートレンドライン（トレンド方向に応じた色分け）
    - 上側・下側バンド
    - ATR値の表示
    - トレンド方向の表示
    - 動的期間使用時は期間値も表示
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.supertrend = None
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
                            # スーパートレンドパラメータ
                            period: int = 10,
                            multiplier: float = 3.0,
                            src_type: str = 'hlc3',
                            atr_smoothing: str = 'wilder',
                            # 動的期間パラメータ
                            use_dynamic_period: bool = False,
                            cycle_part: float = 1.0,
                            detector_type: str = 'cycle_period2',
                            max_cycle: int = 233,
                            min_cycle: int = 13,
                            max_output: int = 144,
                            min_output: int = 13,
                            lp_period: int = 10,
                            hp_period: int = 48
                           ) -> None:
        """
        スーパートレンドを計算する
        
        Args:
            period: ATR期間
            multiplier: ATR乗数
            src_type: ソースタイプ
            atr_smoothing: ATRスムージング方法
            use_dynamic_period: 動的期間を使用するか
            cycle_part: サイクル部分の倍率
            detector_type: 検出器タイプ
            max_cycle: 最大サイクル期間
            min_cycle: 最小サイクル期間
            max_output: 最大出力値
            min_output: 最小出力値
            lp_period: ローパスフィルター期間
            hp_period: ハイパスフィルター期間
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\nスーパートレンドを計算中...")
        
        # スーパートレンドを計算
        self.supertrend = Supertrend(
            period=period,
            multiplier=multiplier,
            src_type=src_type,
            atr_smoothing=atr_smoothing,
            use_dynamic_period=use_dynamic_period,
            cycle_part=cycle_part,
            detector_type=detector_type,
            max_cycle=max_cycle,
            min_cycle=min_cycle,
            max_output=max_output,
            min_output=min_output,
            lp_period=lp_period,
            hp_period=hp_period
        )
        
        # スーパートレンドの計算
        print("計算を実行します...")
        result = self.supertrend.calculate(self.data)
        
        # 結果の確認
        print(f"スーパートレンド計算完了")
        print(f"スーパートレンドライン: {len(result.values)}")
        print(f"上側バンド: {len(result.upper_band)}")
        print(f"下側バンド: {len(result.lower_band)}")
        print(f"トレンド方向: {len(result.trend)}")
        print(f"ATR値: {len(result.atr_values)}")
        
        # NaN値のチェック
        nan_count_st = np.isnan(result.values).sum()
        nan_count_upper = np.isnan(result.upper_band).sum()
        nan_count_lower = np.isnan(result.lower_band).sum()
        nan_count_atr = np.isnan(result.atr_values).sum()
        trend_count = (result.trend != 0).sum()
        
        print(f"NaN値 - ST: {nan_count_st}, 上側: {nan_count_upper}, 下側: {nan_count_lower}, ATR: {nan_count_atr}")
        print(f"トレンド値 - 有効: {trend_count}, 上昇: {(result.trend == 1).sum()}, 下降: {(result.trend == -1).sum()}")
        
        # 下側バンドの値を詳細チェック
        if nan_count_lower == len(result.lower_band):
            print(f"⚠️  警告: 下側バンドがすべてNaNです。ATRの計算に問題がある可能性があります")
            print(f"   ATR値の範囲: {np.nanmin(result.atr_values):.6f} - {np.nanmax(result.atr_values):.6f}")
            print(f"   ATR値の最初の有効インデックス: {np.where(~np.isnan(result.atr_values))[0][:5] if len(np.where(~np.isnan(result.atr_values))[0]) > 0 else 'なし'}")
        
        # 値の範囲をチェック
        if nan_count_st < len(result.values):
            valid_st = result.values[~np.isnan(result.values)]
            print(f"ST値範囲: {valid_st.min():.4f} - {valid_st.max():.4f}")
        if nan_count_upper < len(result.upper_band):
            valid_upper = result.upper_band[~np.isnan(result.upper_band)]
            print(f"上側バンド範囲: {valid_upper.min():.4f} - {valid_upper.max():.4f}")
        if nan_count_lower < len(result.lower_band):
            valid_lower = result.lower_band[~np.isnan(result.lower_band)]
            print(f"下側バンド範囲: {valid_lower.min():.4f} - {valid_lower.max():.4f}")
        
        # HLの値をチェック（バンド計算の基礎データ）
        hl_avg = (self.data['high'] + self.data['low']) / 2.0
        print(f"HLの平均値範囲: {hl_avg.min():.4f} - {hl_avg.max():.4f}")
        
        # ATRが有効な範囲をチェック
        valid_atr_indices = np.where(~np.isnan(result.atr_values))[0]
        if len(valid_atr_indices) > 0:
            print(f"ATR有効範囲: インデックス {valid_atr_indices[0]} - {valid_atr_indices[-1]}")
            # 基本的なバンド計算値をチェック
            multiplier = self.supertrend.multiplier
            basic_upper = hl_avg.iloc[valid_atr_indices[0]] + multiplier * result.atr_values[valid_atr_indices[0]]
            basic_lower = hl_avg.iloc[valid_atr_indices[0]] - multiplier * result.atr_values[valid_atr_indices[0]]
            print(f"基本バンド例（インデックス{valid_atr_indices[0]}）: 上={basic_upper:.4f}, 下={basic_lower:.4f}")
        
        # 動的期間使用時の情報
        if use_dynamic_period:
            dynamic_periods = self.supertrend.get_dynamic_periods()
            if len(dynamic_periods) > 0:
                valid_periods = dynamic_periods[~np.isnan(dynamic_periods)]
                if len(valid_periods) > 0:
                    print(f"動的期間 - 平均: {valid_periods.mean():.1f}, 範囲: {valid_periods.min():.0f} - {valid_periods.max():.0f}")
        
        print("スーパートレンド計算完了")
            
    def plot(self, 
            title: str = "スーパートレンド", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            show_bands: bool = True,
            figsize: Tuple[int, int] = (14, 12),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとスーパートレンドを描画する
        
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
            
        if self.supertrend is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # スーパートレンドの結果を取得
        print("スーパートレンドデータを取得中...")
        result = self.supertrend.calculate(self.data)
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'st_line': result.values,
                'st_upper': result.upper_band,
                'st_lower': result.lower_band,
                'st_trend': result.trend,
                'atr_values': result.atr_values
            }
        )
        
        # 動的期間使用時は期間値も追加
        if self.supertrend.use_dynamic_period:
            dynamic_periods = self.supertrend.get_dynamic_periods()
            if len(dynamic_periods) > 0:
                full_df['dynamic_periods'] = dynamic_periods
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        print(f"スーパートレンドデータ確認 - STライン NaN: {df['st_line'].isna().sum()}")
        
        # トレンド方向に基づくスーパートレンドラインの色分け
        df['st_uptrend'] = np.where(df['st_trend'] == 1, df['st_line'], np.nan)
        df['st_downtrend'] = np.where(df['st_trend'] == -1, df['st_line'], np.nan)
        
        # 上昇トレンド時のスーパートレンドラインがNaNの場合の代替表示
        if (~df['st_uptrend'].isna()).sum() == 0 and (df['st_trend'] == 1).sum() > 0:
            print(f"ℹ️  上昇トレンド用の代替表示を使用します（上昇トレンド期間: {(df['st_trend'] == 1).sum()}個）")
            # 上昇トレンド時は上側バンドを緑色で表示（下側バンドがNaNの場合の代替）
            if (~df['st_lower'].isna()).sum() > 0:
                df['st_uptrend'] = np.where(df['st_trend'] == 1, df['st_lower'], np.nan)
            else:
                # 下側バンドもNaNの場合は上側バンドを使用
                df['st_uptrend'] = np.where(df['st_trend'] == 1, df['st_upper'], np.nan)
                print(f"ℹ️  下側バンドもNaNのため、上昇トレンド時は上側バンドを使用します")
        
        # バンドの表示用（必要に応じて）
        if show_bands:
            df['upper_band_display'] = df['st_upper']
            df['lower_band_display'] = df['st_lower']
        
        # NaN値の問題をチェックして報告
        print(f"データ診断:")
        print(f"  ST上昇トレンド有効値: {(~df['st_uptrend'].isna()).sum()}")
        print(f"  ST下降トレンド有効値: {(~df['st_downtrend'].isna()).sum()}")
        if show_bands:
            print(f"  上側バンド有効値: {(~df['upper_band_display'].isna()).sum()}")
            print(f"  下側バンド有効値: {(~df['lower_band_display'].isna()).sum()}")
        
        # mplfinanceでプロット用の設定
        # 1. メインチャート上のプロット
        main_plots = []
        
        # スーパートレンドラインのプロット設定（有効なデータがある場合のみ）
        if (~df['st_uptrend'].isna()).sum() > 0:
            main_plots.append(mpf.make_addplot(df['st_uptrend'], color='green', width=2.5, label='Supertrend (Up)'))
        if (~df['st_downtrend'].isna()).sum() > 0:
            main_plots.append(mpf.make_addplot(df['st_downtrend'], color='red', width=2.5, label='Supertrend (Down)'))
        
        # バンドの表示（有効なデータがある場合のみ）
        if show_bands:
            if (~df['upper_band_display'].isna()).sum() > 0:
                main_plots.append(mpf.make_addplot(df['upper_band_display'], color='gray', width=1, alpha=0.5, label='Upper Band'))
            if (~df['lower_band_display'].isna()).sum() > 0:
                main_plots.append(mpf.make_addplot(df['lower_band_display'], color='gray', width=1, alpha=0.5, label='Lower Band'))
        
        # 2. オシレータープロット
        panel_idx = 1 if not show_volume else 2
        
        # 追加プロットリスト
        additional_plots = []
        
        # ATR値パネル（有効なデータがある場合のみ）
        if (~df['atr_values'].isna()).sum() > 0:
            atr_panel = mpf.make_addplot(df['atr_values'], panel=panel_idx, color='blue', width=1.2, 
                                        ylabel='ATR', secondary_y=False, label='ATR')
            additional_plots.append(atr_panel)
        
        # トレンド方向パネル（有効なデータがある場合のみ）
        if (df['st_trend'] != 0).sum() > 0:
            trend_panel = mpf.make_addplot(df['st_trend'], panel=panel_idx+1, color='orange', width=1.5, 
                                          ylabel='Trend Direction', secondary_y=False, label='Trend', type='line')
            additional_plots.append(trend_panel)
        
        # 動的期間を使用している場合は期間パネルを追加
        if self.supertrend.use_dynamic_period and 'dynamic_periods' in df.columns:
            if (~df['dynamic_periods'].isna()).sum() > 0:
                period_panel = mpf.make_addplot(df['dynamic_periods'], panel=panel_idx+2, color='purple', width=1.2, 
                                               ylabel='Dynamic Period', secondary_y=False, label='Period')
                additional_plots.append(period_panel)
        
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
        
        # パネル比率の設定
        if show_volume:
            if self.supertrend.use_dynamic_period and 'dynamic_periods' in df.columns:
                kwargs['panel_ratios'] = (4, 1, 1, 1, 1)  # メイン:出来高:ATR:トレンド:期間
            else:
                kwargs['panel_ratios'] = (4, 1, 1, 1)  # メイン:出来高:ATR:トレンド
            kwargs['volume'] = True
        else:
            if self.supertrend.use_dynamic_period and 'dynamic_periods' in df.columns:
                kwargs['panel_ratios'] = (4, 1, 1, 1)  # メイン:ATR:トレンド:期間
            else:
                kwargs['panel_ratios'] = (4, 1, 1)  # メイン:ATR:トレンド
            kwargs['volume'] = False
        
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
        if (~df['st_uptrend'].isna()).sum() > 0:
            legend_labels.append('Supertrend (Up)')
        if (~df['st_downtrend'].isna()).sum() > 0:
            legend_labels.append('Supertrend (Down)')
        if show_bands:
            if (~df['upper_band_display'].isna()).sum() > 0:
                legend_labels.append('Upper Band')
            if (~df['lower_band_display'].isna()).sum() > 0:
                legend_labels.append('Lower Band')
        
        if len(legend_labels) > 0:
            axes[0].legend(legend_labels, loc='upper left')
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        atr_panel_idx = 1 if not show_volume else 2
        trend_panel_idx = atr_panel_idx + 1
        
        # ATR値パネル
        atr_mean = df['atr_values'].mean()
        axes[atr_panel_idx].axhline(y=atr_mean, color='black', linestyle='-', alpha=0.3, label=f'ATR平均: {atr_mean:.4f}')
        
        # トレンド方向パネル
        axes[trend_panel_idx].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[trend_panel_idx].axhline(y=1, color='green', linestyle='--', alpha=0.5, label='上昇トレンド')
        axes[trend_panel_idx].axhline(y=-1, color='red', linestyle='--', alpha=0.5, label='下降トレンド')
        
        # 動的期間パネル（使用時）
        if self.supertrend.use_dynamic_period and 'dynamic_periods' in df.columns:
            period_panel_idx = trend_panel_idx + 1
            period_mean = df['dynamic_periods'].mean()
            axes[period_panel_idx].axhline(y=period_mean, color='black', linestyle='-', alpha=0.3, 
                                          label=f'期間平均: {period_mean:.1f}')
        
        # 統計情報の表示
        print(f"\n=== スーパートレンド統計 ===")
        total_points = len(df[df['st_trend'] != 0])
        uptrend_points = len(df[df['st_trend'] == 1])
        downtrend_points = len(df[df['st_trend'] == -1])
        
        print(f"総データ点数: {total_points}")
        print(f"上昇トレンド: {uptrend_points} ({uptrend_points/total_points*100:.1f}%)")
        print(f"下降トレンド: {downtrend_points} ({downtrend_points/total_points*100:.1f}%)")
        print(f"ATR値 - 平均: {df['atr_values'].mean():.4f}, 範囲: {df['atr_values'].min():.4f} - {df['atr_values'].max():.4f}")
        
        if self.supertrend.use_dynamic_period and 'dynamic_periods' in df.columns:
            valid_periods = df['dynamic_periods'].dropna()
            if len(valid_periods) > 0:
                print(f"動的期間 - 平均: {valid_periods.mean():.1f}, 範囲: {valid_periods.min():.0f} - {valid_periods.max():.0f}")
        
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
    parser = argparse.ArgumentParser(description='スーパートレンドの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--period', type=int, default=10, help='ATR期間')
    parser.add_argument('--multiplier', type=float, default=3.0, help='ATR乗数')
    parser.add_argument('--src-type', type=str, default='hlc3', help='ソースタイプ')
    parser.add_argument('--atr-smoothing', type=str, default='wilder', help='ATRスムージング方法')
    parser.add_argument('--dynamic', action='store_true', help='動的期間を使用')
    parser.add_argument('--no-bands', action='store_true', help='バンドを非表示')
    parser.add_argument('--no-volume', action='store_true', help='出来高を非表示')
    args = parser.parse_args()
    
    # チャートを作成
    chart = SupertrendChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        period=args.period,
        multiplier=args.multiplier,
        src_type=args.src_type,
        atr_smoothing=args.atr_smoothing,
        use_dynamic_period=args.dynamic
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