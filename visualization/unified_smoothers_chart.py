#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
統合スムーサーチャート

統合スムーサーファイルに実装されている全てのスムーサーを
実際の相場データを設定ファイルから取得してチャートに描画
"""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime

# データ取得のための依存関係
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource

# 統合スムーサー
from indicators.smoother.unified_smoother import UnifiedSmoother


class UnifiedSmoothersChart:
    """
    統合スムーサーを表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - 全スムーサーの比較
    - 固定期間vs動的期間の比較（対応スムーサーのみ）
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.smoothers = {}
        self.fig = None
        self.axes = None
        self.available_smoothers = [
            'frama',
            'super_smoother', 
            'ultimate_smoother',
            'zero_lag_ema',
            'laguerre_filter',
            'alma',
            'hma'
        ]
        self.dynamic_supported = [
            'frama',
            'super_smoother',
            'ultimate_smoother', 
            'zero_lag_ema',
            'laguerre_filter',
            'alma',
            'hma'
        ]
    
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

    def calculate_smoothers(self,
                          src_type: str = 'close',
                          period_mode: str = 'fixed',
                          enable_kalman: bool = False,
                          kalman_type: str = 'simple',
                          test_dynamic: bool = False,
                          **kwargs) -> None:
        """
        全スムーサーを計算する
        
        Args:
            src_type: ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4', etc.)
            period_mode: 期間モード ('fixed' または 'dynamic')
            enable_kalman: カルマンフィルターを有効にするか
            kalman_type: カルマンフィルタータイプ
            test_dynamic: 動的期間対応テストも実行するか
            **kwargs: スムーサー固有のパラメータ
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print(f"\n統合スムーサーを計算中... (期間モード: {period_mode})")
        
        self.smoothers = {}
        
        # 各スムーサーを計算
        for smoother_name in self.available_smoothers:
            print(f"  {smoother_name} を計算中...")
            
            try:
                # 固定期間モード
                smoother = UnifiedSmoother(
                    smoother_type=smoother_name,
                    src_type=src_type,
                    period_mode='fixed',
                    enable_kalman=enable_kalman,
                    kalman_type=kalman_type,
                    **kwargs
                )
                result = smoother.calculate(self.data)
                
                self.smoothers[f"{smoother_name}_fixed"] = {
                    'values': result.values,
                    'smoother_type': smoother_name,
                    'mode': 'fixed',
                    'description': f"{smoother_name.upper()} (固定期間)"
                }
                
                # 動的期間モード（対応スムーサーのみ）
                if test_dynamic and smoother_name in self.dynamic_supported:
                    try:
                        smoother_dynamic = UnifiedSmoother(
                            smoother_type=smoother_name,
                            src_type=src_type,
                            period_mode='dynamic',
                            enable_kalman=enable_kalman,
                            kalman_type=kalman_type,
                            cycle_detector_type='hody_e',
                            cycle_part=0.5,
                            max_output=50,
                            min_output=8,
                            **kwargs
                        )
                        result_dynamic = smoother_dynamic.calculate(self.data)
                        
                        self.smoothers[f"{smoother_name}_dynamic"] = {
                            'values': result_dynamic.values,
                            'smoother_type': smoother_name,
                            'mode': 'dynamic',
                            'description': f"{smoother_name.upper()} (動的期間)"
                        }
                        print(f"    動的期間モードも計算完了")
                    except Exception as e:
                        print(f"    動的期間モード計算エラー: {e}")
                
                print(f"    計算完了")
                
            except Exception as e:
                print(f"    エラー: {e}")
                
        print(f"計算完了: {len(self.smoothers)}個のスムーサー")
            
    def plot_comparison(self, 
                       title: str = "統合スムーサー比較", 
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       show_volume: bool = True,
                       figsize: Tuple[int, int] = (16, 12),
                       style: str = 'yahoo',
                       savefig: Optional[str] = None,
                       max_smoothers_per_chart: int = 4) -> None:
        """
        全スムーサーの比較チャートを描画する
        
        Args:
            title: チャートのタイトル
            start_date: 表示開始日（フォーマット: YYYY-MM-DD）
            end_date: 表示終了日（フォーマット: YYYY-MM-DD）
            show_volume: 出来高を表示するか
            figsize: 図のサイズ
            style: mplfinanceのスタイル
            savefig: 保存先のパス（指定しない場合は表示のみ）
            max_smoothers_per_chart: 1つのチャートに表示する最大スムーサー数
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        if not self.smoothers:
            raise ValueError("スムーサーが計算されていません。calculate_smoothers()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
        
        # スムーサー数が多い場合は複数のチャートに分割
        smoother_names = list(self.smoothers.keys())
        num_charts = (len(smoother_names) + max_smoothers_per_chart - 1) // max_smoothers_per_chart
        
        # カラーパレット
        colors = [
            'blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray',
            'olive', 'cyan', 'magenta', 'yellow', 'darkblue', 'darkred', 'darkgreen'
        ]
        
        for chart_idx in range(num_charts):
            start_idx = chart_idx * max_smoothers_per_chart
            end_idx = min(start_idx + max_smoothers_per_chart, len(smoother_names))
            chart_smoothers = smoother_names[start_idx:end_idx]
            
            print(f"チャート {chart_idx + 1}/{num_charts} を描画中...")
            
            # スムーサーデータをデータフレームに追加
            smoother_data = {}
            for i, smoother_name in enumerate(chart_smoothers):
                smoother_info = self.smoothers[smoother_name]
                # データの長さを調整
                values = smoother_info['values']
                if len(values) == len(self.data):
                    # 期間絞り込み
                    smoother_series = pd.Series(values, index=self.data.index)
                    if start_date:
                        smoother_series = smoother_series[smoother_series.index >= pd.to_datetime(start_date)]
                    if end_date:
                        smoother_series = smoother_series[smoother_series.index <= pd.to_datetime(end_date)]
                    smoother_data[smoother_name] = smoother_series
            
            # メインチャート上のプロット
            main_plots = []
            for i, (smoother_name, smoother_series) in enumerate(smoother_data.items()):
                color = colors[i % len(colors)]
                width = 1.5 if 'dynamic' in smoother_name else 1.0
                alpha = 0.8 if 'dynamic' in smoother_name else 0.6
                
                main_plots.append(
                    mpf.make_addplot(
                        smoother_series, 
                        color=color, 
                        width=width, 
                        alpha=alpha,
                        label=self.smoothers[smoother_name]['description']
                    )
                )
            
            # mplfinanceの設定
            kwargs = dict(
                type='candle',
                figsize=figsize,
                title=f"{title} (Part {chart_idx + 1}/{num_charts})",
                style=style,
                datetime_format='%Y-%m-%d',
                xrotation=45,
                returnfig=True,
                addplot=main_plots
            )
            
            # 出来高の設定
            if show_volume:
                kwargs['volume'] = True
            else:
                kwargs['volume'] = False
            
            # プロット実行
            fig, axes = mpf.plot(df, **kwargs)
            
            # 凡例の追加
            legend_labels = [self.smoothers[name]['description'] for name in chart_smoothers]
            axes[0].legend(legend_labels, loc='upper left', fontsize=8)
            
            # 統計情報の表示
            self._add_statistics_table(fig, axes[0], chart_smoothers, df)
            
            # 保存または表示
            if savefig:
                if num_charts > 1:
                    if '.' in savefig:
                        base_name, extension = savefig.rsplit('.', 1)
                        save_path = f"{base_name}_part{chart_idx + 1}.{extension}"
                    else:
                        save_path = f"{savefig}_part{chart_idx + 1}.png"
                else:
                    save_path = savefig if '.' in savefig else f"{savefig}.png"
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"チャートを保存しました: {save_path}")
            else:
                plt.tight_layout()
                plt.show()
    
    def _add_statistics_table(self, fig, ax, smoother_names: List[str], df: pd.DataFrame):
        """統計情報テーブルを追加"""
        stats_data = []
        
        for smoother_name in smoother_names:
            smoother_info = self.smoothers[smoother_name]
            values = smoother_info['values']
            
            if len(values) == len(self.data):
                # 有効値の統計
                valid_mask = ~np.isnan(values)
                if np.any(valid_mask):
                    valid_values = values[valid_mask]
                    stats_data.append([
                        smoother_info['description'][:15],  # 名前を短縮
                        f"{np.mean(valid_values):.2f}",
                        f"{np.std(valid_values):.2f}",
                        f"{np.sum(valid_mask)}"
                    ])
        
        if stats_data:
            # テーブルを図の右側に追加
            table_ax = fig.add_axes([0.02, 0.02, 0.25, 0.15])
            table_ax.axis('off')
            
            headers = ['スムーサー', '平均', '標準偏差', '有効数']
            table = table_ax.table(
                cellText=stats_data,
                colLabels=headers,
                cellLoc='center',
                loc='center',
                bbox=[0, 0, 1, 1]
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.5)
    
    def plot_individual_comparison(self,
                                 smoother_name: str,
                                 title: Optional[str] = None,
                                 start_date: Optional[str] = None,
                                 end_date: Optional[str] = None,
                                 show_volume: bool = True,
                                 figsize: Tuple[int, int] = (14, 10),
                                 style: str = 'yahoo',
                                 savefig: Optional[str] = None) -> None:
        """
        個別スムーサーの固定vs動的期間比較チャートを描画
        
        Args:
            smoother_name: スムーサー名
            title: チャートのタイトル
            start_date: 表示開始日
            end_date: 表示終了日
            show_volume: 出来高を表示するか
            figsize: 図のサイズ
            style: mplfinanceのスタイル
            savefig: 保存先のパス
        """
        if title is None:
            title = f"{smoother_name.upper()} 固定期間 vs 動的期間比較"
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
        
        # 対象スムーサーのデータを取得
        main_plots = []
        fixed_key = f"{smoother_name}_fixed"
        dynamic_key = f"{smoother_name}_dynamic"
        
        if fixed_key in self.smoothers:
            fixed_values = self.smoothers[fixed_key]['values']
            if len(fixed_values) == len(self.data):
                fixed_series = pd.Series(fixed_values, index=self.data.index)
                if start_date:
                    fixed_series = fixed_series[fixed_series.index >= pd.to_datetime(start_date)]
                if end_date:
                    fixed_series = fixed_series[fixed_series.index <= pd.to_datetime(end_date)]
                
                main_plots.append(
                    mpf.make_addplot(
                        fixed_series, 
                        color='blue', 
                        width=1.0, 
                        alpha=0.7,
                        label=f"{smoother_name.upper()} (固定期間)"
                    )
                )
        
        if dynamic_key in self.smoothers:
            dynamic_values = self.smoothers[dynamic_key]['values']
            if len(dynamic_values) == len(self.data):
                dynamic_series = pd.Series(dynamic_values, index=self.data.index)
                if start_date:
                    dynamic_series = dynamic_series[dynamic_series.index >= pd.to_datetime(start_date)]
                if end_date:
                    dynamic_series = dynamic_series[dynamic_series.index <= pd.to_datetime(end_date)]
                
                main_plots.append(
                    mpf.make_addplot(
                        dynamic_series, 
                        color='red', 
                        width=1.5, 
                        alpha=0.8,
                        label=f"{smoother_name.upper()} (動的期間)"
                    )
                )
        
        if not main_plots:
            print(f"エラー: {smoother_name} のデータが見つかりません")
            return
        
        # mplfinanceの設定
        kwargs = dict(
            type='candle',
            figsize=figsize,
            title=title,
            style=style,
            datetime_format='%Y-%m-%d',
            xrotation=45,
            returnfig=True,
            addplot=main_plots
        )
        
        # 出来高の設定
        if show_volume:
            kwargs['volume'] = True
        else:
            kwargs['volume'] = False
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加
        legend_labels = []
        if fixed_key in self.smoothers:
            legend_labels.append(f"{smoother_name.upper()} (固定期間)")
        if dynamic_key in self.smoothers:
            legend_labels.append(f"{smoother_name.upper()} (動的期間)")
        
        axes[0].legend(legend_labels, loc='upper left')
        
        # 保存または表示
        if savefig:
            plt.savefig(savefig, dpi=150, bbox_inches='tight')
            print(f"チャートを保存しました: {savefig}")
        else:
            plt.tight_layout()
            plt.show()
    
    def get_performance_summary(self) -> pd.DataFrame:
        """各スムーサーの性能サマリーを取得"""
        if not self.smoothers:
            return pd.DataFrame()
        
        summary_data = []
        
        for smoother_name, smoother_info in self.smoothers.items():
            values = smoother_info['values']
            
            if len(values) > 0:
                valid_mask = ~np.isnan(values)
                if np.any(valid_mask):
                    valid_values = values[valid_mask]
                    
                    summary_data.append({
                        'スムーサー': smoother_info['description'],
                        'モード': smoother_info['mode'],
                        '平均値': np.mean(valid_values),
                        '標準偏差': np.std(valid_values),
                        '最小値': np.min(valid_values),
                        '最大値': np.max(valid_values),
                        '有効データ数': np.sum(valid_mask),
                        '有効率(%)': (np.sum(valid_mask) / len(values)) * 100
                    })
        
        return pd.DataFrame(summary_data)


def main():
    """メイン関数"""
    # コマンドライン引数を処理
    import argparse
    parser = argparse.ArgumentParser(description='統合スムーサーの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--src-type', type=str, default='close', help='価格ソースタイプ')
    parser.add_argument('--test-dynamic', action='store_true', help='動的期間テストも実行')
    parser.add_argument('--individual', type=str, help='個別スムーサー比較 (スムーサー名を指定)')
    parser.add_argument('--max-per-chart', type=int, default=4, help='1チャートあたりの最大スムーサー数')
    args = parser.parse_args()
    
    # チャートを作成
    chart = UnifiedSmoothersChart()
    chart.load_data_from_config(args.config)
    chart.calculate_smoothers(
        src_type=args.src_type,
        test_dynamic=args.test_dynamic
    )
    
    if args.individual:
        # 個別比較チャート
        output_path = args.output
        if output_path and not output_path.endswith('.png'):
            output_path = f"{output_path}_{args.individual}_comparison.png"
        
        chart.plot_individual_comparison(
            smoother_name=args.individual,
            start_date=args.start,
            end_date=args.end,
            savefig=output_path
        )
    else:
        # 全体比較チャート
        chart.plot_comparison(
            start_date=args.start,
            end_date=args.end,
            savefig=args.output,
            max_smoothers_per_chart=args.max_per_chart
        )
    
    # 性能サマリーの表示
    print("\n=== 性能サマリー ===")
    summary = chart.get_performance_summary()
    if not summary.empty:
        print(summary.to_string(index=False, float_format='%.4f'))
    
    # ファイル名に現在時刻を追加
    if args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = args.output.rsplit('.', 1)[0] if '.' in args.output else args.output
        extension = args.output.rsplit('.', 1)[1] if '.' in args.output else 'png'
        final_output = f"{base_name}_{timestamp}.{extension}"
        print(f"最終出力ファイル: {final_output}")


if __name__ == "__main__":
    main()