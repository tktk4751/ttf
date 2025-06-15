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
from indicators.hma import HMA


class HMAAdaptiveChart:
    """
    アダプティブHMAを表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - 固定期間HMAと動的期間アダプティブHMA
    - 動的期間値
    - ドミナントサイクル値
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.hma_fixed = None
        self.hma_adaptive = None
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
                            # HMAパラメータ
                            hma_period: int = 21,
                            hma_src_type: str = 'close',
                            # アダプティブHMAパラメータ
                            detector_type: str = 'phac_e',
                            cycle_part: float = 1.0,
                            max_cycle: int = 233,
                            min_cycle: int = 13,
                            max_output: int = 144,
                            min_output: int = 13,
                            # 表示オプション
                            show_fixed_hma: bool = False  # 固定期間HMAを表示するかどうか
                           ) -> None:
        """
        HMAを計算する
        
        Args:
            hma_period: 固定HMAの期間
            hma_src_type: HMAの価格ソース
            detector_type: Ehlers DC検出器タイプ
            cycle_part: DCサイクル部分
            max_cycle: DC最大サイクル期間
            min_cycle: DC最小サイクル期間
            max_output: DC最大出力値（動的期間の最大値）
            min_output: DC最小出力値（動的期間の最小値）
            show_fixed_hma: 固定期間HMAを表示するかどうか
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\nHMAアダプティブインジケーターを計算中...")
        
        # 固定期間HMAの計算（比較用）
        if show_fixed_hma:
            print("固定期間HMAを計算中...")
            self.hma_fixed = HMA(
                period=hma_period,
                src_type=hma_src_type,
                use_dynamic_period=False
            )
            hma_fixed_result = self.hma_fixed.calculate(self.data)
            print(f"固定HMA計算完了 - NaN値: {np.isnan(hma_fixed_result).sum()}")
        else:
            self.hma_fixed = None
        
        # アダプティブHMA（動的期間対応）を計算
        print("アダプティブHMA（動的期間）を計算中...")
        self.hma_adaptive = HMA(
            period=hma_period,  # フォールバック期間
            src_type=hma_src_type,
            use_dynamic_period=True,
            detector_type=detector_type,
            cycle_part=cycle_part,
            max_cycle=max_cycle,
            min_cycle=min_cycle,
            max_output=max_output,
            min_output=min_output
        )
        
        hma_adaptive_result = self.hma_adaptive.calculate(self.data)
        print(f"アダプティブHMA計算完了 - NaN値: {np.isnan(hma_adaptive_result).sum()}")
        
        # 動的期間の取得
        self.dynamic_periods = self.hma_adaptive.get_dynamic_periods()
        
        # ドミナントサイクル値を取得
        if hasattr(self.hma_adaptive, 'dc_detector') and self.hma_adaptive.dc_detector is not None:
            try:
                self.dominant_cycles = self.hma_adaptive.dc_detector.calculate(self.data)
                print(f"ドミナントサイクル計算完了 - 配列長: {len(self.dominant_cycles)}")
            except Exception as e:
                print(f"ドミナントサイクル取得エラー: {e}")
                self.dominant_cycles = np.full(len(self.data), np.nan)
        else:
            self.dominant_cycles = np.full(len(self.data), np.nan)
        
        # 動的期間の統計
        if len(self.dynamic_periods) > 0:
            valid_periods = self.dynamic_periods[~np.isnan(self.dynamic_periods)]
            if len(valid_periods) > 0:
                print(f"動的期間範囲: {np.min(valid_periods):.1f} - {np.max(valid_periods):.1f}")
        
        print("HMAアダプティブインジケーター計算完了")
            
    def plot(self, 
            title: str = "アダプティブHMA", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (14, 10),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとアダプティブHMAを描画する
        
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
            
        if self.hma_adaptive is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # インジケーター値を取得
        print("インジケーターデータを取得中...")
        hma_adaptive_result = self.hma_adaptive.calculate(self.data)
        
        # 全データの時系列データフレームを作成
        chart_data = {
            'hma_adaptive': hma_adaptive_result,
            'dynamic_period': self.dynamic_periods,
            'dominant_cycle': self.dominant_cycles
        }
        
        # 固定期間HMAがある場合のみ追加
        if self.hma_fixed is not None:
            hma_fixed_result = self.hma_fixed.calculate(self.data)
            chart_data['hma_fixed'] = hma_fixed_result
        
        full_df = pd.DataFrame(index=self.data.index, data=chart_data)
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        print(f"HMAデータ確認 - アダプティブHMA NaN: {df['hma_adaptive'].isna().sum()}")
        if 'hma_fixed' in df.columns:
            print(f"固定HMA NaN: {df['hma_fixed'].isna().sum()}")
        
        # mplfinanceでプロット用の設定
        # 1. メインチャート上のプロット
        main_plots = []
        
        # 動的期間HMAのプロット設定
        if not df['hma_adaptive'].isna().all():
            main_plots.append(mpf.make_addplot(df['hma_adaptive'], color='blue', width=2.5, label='Dynamic HMA'))
        
        # 固定期間HMAがある場合のみプロット（比較用）
        if 'hma_fixed' in df.columns and not df['hma_fixed'].isna().all():
            main_plots.append(mpf.make_addplot(df['hma_fixed'], color='gray', width=1.5, alpha=0.7, label='Fixed HMA'))
        
        # 2. サブプロット
        plots_list = []
        panel_count = 1
        
        # ドミナントサイクルパネル
        if not df['dominant_cycle'].isna().all():
            dc_panel = mpf.make_addplot(df['dominant_cycle'], panel=panel_count, color='purple', width=1.5, 
                                        ylabel='Dominant Cycle', secondary_y=False)
            plots_list.append(dc_panel)
            panel_count += 1
        
        # 動的期間パネル
        if not df['dynamic_period'].isna().all():
            period_panel = mpf.make_addplot(df['dynamic_period'], panel=panel_count, color='orange', width=1.5, 
                                            ylabel='Dynamic Period', secondary_y=False)
            plots_list.append(period_panel)
            panel_count += 1
        
        # mplfinanceの設定
        kwargs = dict(
            type='candle',
            figsize=figsize,
            title=title,
            style=style,
            datetime_format='%Y-%m-%d',
            xrotation=45,
            returnfig=True,
            warn_too_much_data=len(df) * 2  # データ量の2倍に設定して警告を抑制
        )
        
        # 出来高と追加パネルの設定
        actual_panel_count = len(plots_list)
        
        if show_volume:
            kwargs['volume'] = True
            # 動的にパネル比率を設定
            panel_ratios = [4, 1]  # メイン、出来高
            panel_ratios.extend([1] * actual_panel_count)  # 追加パネル
            kwargs['panel_ratios'] = tuple(panel_ratios)
            
            # 出来高を表示する場合は、オシレーターのパネル番号を+1する
            adjusted_plots = []
            for plot in plots_list:
                new_plot = mpf.make_addplot(
                    plot['data'], 
                    panel=plot['panel'] + 1, 
                    color=plot['color'], 
                    width=plot.get('width', 1),
                    ylabel=plot.get('ylabel', ''),
                    secondary_y=plot.get('secondary_y', False)
                )
                adjusted_plots.append(new_plot)
            plots_list = adjusted_plots
        else:
            kwargs['volume'] = False
            # 動的にパネル比率を設定
            panel_ratios = [4]  # メイン
            panel_ratios.extend([1] * actual_panel_count)  # 追加パネル
            if panel_ratios:
                kwargs['panel_ratios'] = tuple(panel_ratios)
        
        # すべてのプロットを結合
        if main_plots or plots_list:
            all_plots = main_plots + plots_list
            kwargs['addplot'] = all_plots
        
        # プロット実行
        try:
            fig, axes = mpf.plot(df, **kwargs)
        except Exception as e:
            print(f"プロット中にエラーが発生しました: {e}")
            print("データの簡易表示を試行します...")
            # エラー時は最小限のプロットを試行
            simple_kwargs = dict(
                type='candle',
                figsize=figsize,
                title=title,
                style=style,
                datetime_format='%Y-%m-%d',
                xrotation=45,
                returnfig=True,
                volume=show_volume,
                warn_too_much_data=len(df) * 2
            )
            # HMAラインのみ追加
            if not df['hma_adaptive'].isna().all():
                simple_kwargs['addplot'] = [
                    mpf.make_addplot(df['hma_adaptive'], color='blue', width=2, label='Dynamic HMA')
                ]
            fig, axes = mpf.plot(df, **simple_kwargs)
            print("簡易表示でプロットしました。")
        
        # 凡例の追加
        legend_labels = []
        if not df['hma_adaptive'].isna().all():
            legend_labels.append('Dynamic HMA')
            
        # 固定期間HMAがある場合
        if 'hma_fixed' in df.columns and not df['hma_fixed'].isna().all():
            legend_labels.append('Fixed HMA (Reference)')
        
        if legend_labels and len(axes) > 0:
            axes[0].legend(legend_labels, loc='upper left')
        
        self.fig = fig
        self.axes = axes
        
        # 参照線を追加
        panel_offset = 1 if show_volume else 0
        panel_idx = 1 + panel_offset
        
        # ドミナントサイクルパネル
        if not df['dominant_cycle'].isna().all() and panel_idx < len(axes):
            dc_mean = df['dominant_cycle'].mean()
            if not np.isnan(dc_mean):
                axes[panel_idx].axhline(y=dc_mean, color='black', linestyle='-', alpha=0.3)
                axes[panel_idx].axhline(y=df['dominant_cycle'].quantile(0.75), color='black', linestyle='--', alpha=0.5)
                axes[panel_idx].axhline(y=df['dominant_cycle'].quantile(0.25), color='black', linestyle='--', alpha=0.5)
            panel_idx += 1
        
        # 動的期間パネル
        if not df['dynamic_period'].isna().all() and panel_idx < len(axes):
            period_mean = df['dynamic_period'].mean()
            if not np.isnan(period_mean):
                axes[panel_idx].axhline(y=period_mean, color='black', linestyle='-', alpha=0.3)
                axes[panel_idx].axhline(y=df['dynamic_period'].max(), color='red', linestyle=':', alpha=0.5)
                axes[panel_idx].axhline(y=df['dynamic_period'].min(), color='green', linestyle=':', alpha=0.5)
        
        # 統計情報の表示
        print(f"\n=== HMA統計 ===")
        total_points = len(df[~df['hma_adaptive'].isna()])
        print(f"総データ点数: {total_points}")
        
        # 動的期間の統計
        if not df['dynamic_period'].isna().all():
            valid_periods = df['dynamic_period'][~df['dynamic_period'].isna()]
            if len(valid_periods) > 0:
                print(f"期間統計 - 平均: {valid_periods.mean():.1f}, 範囲: {valid_periods.min():.0f} - {valid_periods.max():.0f}")
        
        # ドミナントサイクルの統計
        if not df['dominant_cycle'].isna().all():
            valid_dc = df['dominant_cycle'][~df['dominant_cycle'].isna()]
            if len(valid_dc) > 0:
                print(f"DC統計 - 平均: {valid_dc.mean():.1f}, 範囲: {valid_dc.min():.0f} - {valid_dc.max():.0f}")
        
        # 保存または表示
        if savefig:
            plt.savefig(savefig, dpi=300, bbox_inches='tight')
            print(f"チャートを保存しました: {savefig}")
        else:
            plt.tight_layout()
            plt.show()

    def get_statistics(self) -> Dict[str, Any]:
        """
        インジケーターの統計情報を取得する
        
        Returns:
            統計情報の辞書
        """
        if self.hma_adaptive is None:
            return {}
            
        stats = {}
        
        # HMA統計
        hma_adaptive_values = self.hma_adaptive.calculate(self.data)
        valid_adaptive = hma_adaptive_values[~np.isnan(hma_adaptive_values)]
        
        stats['hma_adaptive'] = {
            'mean': np.mean(valid_adaptive) if len(valid_adaptive) > 0 else np.nan,
            'std': np.std(valid_adaptive) if len(valid_adaptive) > 0 else np.nan,
            'min': np.min(valid_adaptive) if len(valid_adaptive) > 0 else np.nan,
            'max': np.max(valid_adaptive) if len(valid_adaptive) > 0 else np.nan,
            'nan_count': np.isnan(hma_adaptive_values).sum(),
            'total_count': len(hma_adaptive_values)
        }
        
        # 固定期間HMAがある場合のみ統計を追加
        if self.hma_fixed is not None:
            hma_fixed_values = self.hma_fixed.calculate(self.data)
            valid_fixed = hma_fixed_values[~np.isnan(hma_fixed_values)]
            
            stats['hma_fixed'] = {
                'mean': np.mean(valid_fixed) if len(valid_fixed) > 0 else np.nan,
                'std': np.std(valid_fixed) if len(valid_fixed) > 0 else np.nan,
                'min': np.min(valid_fixed) if len(valid_fixed) > 0 else np.nan,
                'max': np.max(valid_fixed) if len(valid_fixed) > 0 else np.nan,
                'nan_count': np.isnan(hma_fixed_values).sum(),
                'total_count': len(hma_fixed_values)
            }
        
        # ドミナントサイクル統計
        if hasattr(self, 'dominant_cycles'):
            valid_dc = self.dominant_cycles[~np.isnan(self.dominant_cycles)]
            stats['dominant_cycle'] = {
                'mean': np.mean(valid_dc) if len(valid_dc) > 0 else np.nan,
                'std': np.std(valid_dc) if len(valid_dc) > 0 else np.nan,
                'min': np.min(valid_dc) if len(valid_dc) > 0 else np.nan,
                'max': np.max(valid_dc) if len(valid_dc) > 0 else np.nan,
                'nan_count': np.isnan(self.dominant_cycles).sum(),
                'total_count': len(self.dominant_cycles)
            }
        
        # 動的期間統計
        if hasattr(self, 'dynamic_periods'):
            valid_periods = self.dynamic_periods[~np.isnan(self.dynamic_periods)]
            stats['dynamic_period'] = {
                'mean': np.mean(valid_periods) if len(valid_periods) > 0 else np.nan,
                'std': np.std(valid_periods) if len(valid_periods) > 0 else np.nan,
                'min': np.min(valid_periods) if len(valid_periods) > 0 else np.nan,
                'max': np.max(valid_periods) if len(valid_periods) > 0 else np.nan,
                'nan_count': np.isnan(self.dynamic_periods).sum(),
                'total_count': len(self.dynamic_periods)
            }
        
        return stats


def main():
    """メイン関数"""
    # コマンドライン引数を処理
    import argparse
    parser = argparse.ArgumentParser(description='アダプティブHMAの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--hma-period', type=int, default=21, help='固定HMA期間 (default: 21)')
    parser.add_argument('--detector', type=str, default='cycle_period2', help='Ehlers DC検出器タイプ (default: phac_e)')
    parser.add_argument('--min-output', type=int, default=13, help='Ehlers DC最小出力値（動的期間最小値） (default: 8)')
    parser.add_argument('--max-output', type=int, default=144, help='Ehlers DC最大出力値（動的期間最大値） (default: 34)')
    parser.add_argument('--show-fixed', action='store_true', help='固定期間HMAも表示する（比較用）')
    parser.add_argument('--no-volume', action='store_true', help='出来高を表示しない')
    parser.add_argument('--stats', action='store_true', help='統計情報を表示')
    args = parser.parse_args()
    
    # パラメータの妥当性チェック
    if args.min_output >= args.max_output:
        print(f"エラー: min-output ({args.min_output}) は max-output ({args.max_output}) より小さい必要があります。")
        return
    
    if args.min_output < 2:
        print(f"エラー: min-output ({args.min_output}) は2以上である必要があります（HMAの最小期間）。")
        return
    
    # チャートを作成
    chart = HMAAdaptiveChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        hma_period=args.hma_period,
        detector_type=args.detector,
        min_output=args.min_output,
        max_output=args.max_output,
        show_fixed_hma=args.show_fixed
    )
    
    # 統計情報の表示
    if args.stats:
        stats = chart.get_statistics()
        print("\n=== インジケーター統計情報 ===")
        for indicator, stat_data in stats.items():
            print(f"\n{indicator.upper()}:")
            for key, value in stat_data.items():
                if isinstance(value, float):
                    if np.isnan(value):
                        print(f"  {key}: NaN")
                    else:
                        print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
    
    print(f"\nチャートを描画します...")
    print(f"パラメータ: HMA期間={args.hma_period}, 検出器={args.detector}")
    print(f"動的期間範囲: {args.min_output}-{args.max_output}")
    if args.show_fixed:
        print("固定期間HMAも表示します（比較用）")
    
    chart.plot(
        title="動的期間HMA",
        start_date=args.start,
        end_date=args.end,
        show_volume=not args.no_volume,
        savefig=args.output
    )


if __name__ == "__main__":
    main() 