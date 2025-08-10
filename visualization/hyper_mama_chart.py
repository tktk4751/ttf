#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import yaml
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path

# データ取得のための依存関係
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource

# インジケーター
from indicators.hyper_mama import HyperMAMA


class HyperMAMAChart:
    """
    HyperMAMAとHyperFAMAを表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - HyperMAMA、HyperFAMA（メインパネル上）
    - HyperER値（別パネル）
    - 動的適応されたFast/Slow Limits（別パネル）
    - Period値とAlpha値（別パネル）
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.hyper_mama = None
        self.hyper_mama_result = None
        self.fig = None
        self.axes = None
    
    def load_data_from_config(self, config_path: str, max_bars: int = 500) -> pd.DataFrame:
        """
        設定ファイルからデータを読み込む
        
        Args:
            config_path: 設定ファイルのパス
            max_bars: 最大データ数（デフォルト：500）
            
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
        full_data = processed_data[first_symbol]
        
        # 直近のmax_bars本に制限
        if len(full_data) > max_bars:
            self.data = full_data.tail(max_bars).copy()
        else:
            self.data = full_data.copy()
        
        print(f"データ読み込み完了: {first_symbol}")
        print(f"期間: {self.data.index.min()} → {self.data.index.max()}")
        print(f"データ数: {len(self.data)} (制限: {max_bars})")
        
        return self.data

    def calculate_indicators(
        self,
        trigger_type: str = 'hyper_er',
        hyper_er_period: int = 14,
        hyper_er_midline_period: int = 100,
        hyper_trend_period: int = 14,
        hyper_trend_midline_period: int = 100,
        fast_max: float = 0.5,
        fast_min: float = 0.1,
        slow_max: float = 0.05,
        slow_min: float = 0.01,
        er_high_threshold: float = 0.8,
        er_low_threshold: float = 0.2,
        src_type: str = 'hlc3',
        use_kalman_filter: bool = False,
        use_zero_lag: bool = True
    ) -> None:
        """
        HyperMAMAインジケーターを計算する
        
        Args:
            trigger_type: トリガータイプ ('hyper_er' または 'hyper_trend_index')
            hyper_er_period: HyperER計算期間
            hyper_er_midline_period: HyperERミッドライン計算期間
            hyper_trend_period: HyperTrendIndex計算期間
            hyper_trend_midline_period: HyperTrendIndexミッドライン計算期間
            fast_max: fastlimitの最大値
            fast_min: fastlimitの最小値
            slow_max: slowlimitの最大値
            slow_min: slowlimitの最小値
            er_high_threshold: HyperERの高閾値
            er_low_threshold: HyperERの低閾値
            src_type: ソースタイプ
            use_kalman_filter: カルマンフィルターを使用するか
            use_zero_lag: ゼロラグ処理を使用するか
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\\nHyperMAMAインジケーターを計算中...")
        
        # HyperMAMAの計算
        print(f"HyperMAMAを計算中（トリガー: {trigger_type}）...")
        self.hyper_mama = HyperMAMA(
            trigger_type=trigger_type,
            hyper_er_period=hyper_er_period,
            hyper_er_midline_period=hyper_er_midline_period,
            hyper_trend_period=hyper_trend_period,
            hyper_trend_midline_period=hyper_trend_midline_period,
            fast_max=fast_max,
            fast_min=fast_min,
            slow_max=slow_max,
            slow_min=slow_min,
            er_high_threshold=er_high_threshold,
            er_low_threshold=er_low_threshold,
            src_type=src_type,
            use_kalman_filter=use_kalman_filter,
            use_zero_lag=use_zero_lag
        )
        
        self.hyper_mama_result = self.hyper_mama.calculate(self.data)
        print(f"HyperMAMA計算完了: 有効値数 {np.sum(~np.isnan(self.hyper_mama_result.mama_values))}/{len(self.hyper_mama_result.mama_values)}")
            
    def plot(self, 
            title: str = "HyperMAMA & HyperFAMA チャート", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (16, 20),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとHyperMAMAインジケーターを描画する
        
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
            
        if self.hyper_mama is None or self.hyper_mama_result is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # インジケーターの値を取得
        print("インジケーターデータを取得中...")
        
        # HyperMAMA関連データ
        mama_values = self.hyper_mama_result.mama_values
        fama_values = self.hyper_mama_result.fama_values
        hyper_er_values = self.hyper_mama_result.hyper_er_values
        period_values = self.hyper_mama_result.period_values
        alpha_values = self.hyper_mama_result.alpha_values
        adaptive_fast_limits = self.hyper_mama_result.adaptive_fast_limits
        adaptive_slow_limits = self.hyper_mama_result.adaptive_slow_limits
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'mama': mama_values,
                'fama': fama_values,
                'hyper_er': hyper_er_values,
                'period': period_values,
                'alpha': alpha_values,
                'fast_limit': adaptive_fast_limits,
                'slow_limit': adaptive_slow_limits
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        
        # HyperERベースの効率性判定（0.8以上で高効率、0.2以下で低効率、中間を中効率）
        df['hyper_er_high'] = np.where(df['hyper_er'] >= 0.8, df['hyper_er'], np.nan)
        df['hyper_er_medium'] = np.where((df['hyper_er'] >= 0.2) & (df['hyper_er'] < 0.8), df['hyper_er'], np.nan)
        df['hyper_er_low'] = np.where(df['hyper_er'] < 0.2, df['hyper_er'], np.nan)
        
        # mplfinanceでプロット用の設定
        # 追加パネルのプロット設定
        additional_plots = []
        
        if show_volume:
            # 出来高あり: パネル0=メイン、パネル1=出来高、パネル2=HyperER、パネル3=動的Limits、パネル4=Period&Alpha
            
            # メインパネルにHyperMAMAとHyperFAMA（パネル0）
            additional_plots.append(
                mpf.make_addplot(df['mama'], panel=0, color='blue', width=2.5, 
                               label='HyperMAMA', alpha=0.9)
            )
            additional_plots.append(
                mpf.make_addplot(df['fama'], panel=0, color='red', width=2, 
                               label='HyperFAMA', alpha=0.8)
            )
            
            # HyperER効率性（パネル2）
            additional_plots.append(
                mpf.make_addplot(df['hyper_er_high'], panel=2, color='green', width=2, 
                               ylabel='HyperER Efficiency', secondary_y=False, label='High Efficiency (≥0.8)', type='line')
            )
            additional_plots.append(
                mpf.make_addplot(df['hyper_er_medium'], panel=2, color='orange', width=2, 
                               label='Medium Efficiency (0.2-0.8)', type='line')
            )
            additional_plots.append(
                mpf.make_addplot(df['hyper_er_low'], panel=2, color='red', width=2, 
                               label='Low Efficiency (<0.2)', type='line')
            )
            
            # 動的適応Limits（パネル3）
            additional_plots.append(
                mpf.make_addplot(df['fast_limit'], panel=3, color='blue', width=2, 
                               ylabel='Dynamic Limits', secondary_y=False, label='Fast Limit', type='line')
            )
            additional_plots.append(
                mpf.make_addplot(df['slow_limit'], panel=3, color='purple', width=2, 
                               label='Slow Limit', type='line')
            )
            
            # Period値とAlpha値（パネル4）
            additional_plots.append(
                mpf.make_addplot(df['period'], panel=4, color='cyan', width=2, 
                               ylabel='Period & Alpha', secondary_y=False, label='Period', type='line')
            )
            additional_plots.append(
                mpf.make_addplot(df['alpha'], panel=4, color='magenta', width=2, 
                               secondary_y=True, label='Alpha', type='line')
            )
            
            panel_ratios = (5, 1, 1.5, 1.2, 1.2)  # メイン:出来高:HyperER:Limits:Period&Alpha
            
        else:
            # 出来高なし: パネル0=メイン、パネル1=HyperER、パネル2=動的Limits、パネル3=Period&Alpha
            
            # メインパネルにHyperMAMAとHyperFAMA（パネル0）
            additional_plots.append(
                mpf.make_addplot(df['mama'], panel=0, color='blue', width=2.5, 
                               label='HyperMAMA', alpha=0.9)
            )
            additional_plots.append(
                mpf.make_addplot(df['fama'], panel=0, color='red', width=2, 
                               label='HyperFAMA', alpha=0.8)
            )
            
            # HyperER効率性（パネル1）
            additional_plots.append(
                mpf.make_addplot(df['hyper_er_high'], panel=1, color='green', width=2, 
                               ylabel='HyperER Efficiency', secondary_y=False, label='High Efficiency (≥0.8)', type='line')
            )
            additional_plots.append(
                mpf.make_addplot(df['hyper_er_medium'], panel=1, color='orange', width=2, 
                               label='Medium Efficiency (0.2-0.8)', type='line')
            )
            additional_plots.append(
                mpf.make_addplot(df['hyper_er_low'], panel=1, color='red', width=2, 
                               label='Low Efficiency (<0.2)', type='line')
            )
            
            # 動的適応Limits（パネル2）
            additional_plots.append(
                mpf.make_addplot(df['fast_limit'], panel=2, color='blue', width=2, 
                               ylabel='Dynamic Limits', secondary_y=False, label='Fast Limit', type='line')
            )
            additional_plots.append(
                mpf.make_addplot(df['slow_limit'], panel=2, color='purple', width=2, 
                               label='Slow Limit', type='line')
            )
            
            # Period値とAlpha値（パネル3）
            additional_plots.append(
                mpf.make_addplot(df['period'], panel=3, color='cyan', width=2, 
                               ylabel='Period & Alpha', secondary_y=False, label='Period', type='line')
            )
            additional_plots.append(
                mpf.make_addplot(df['alpha'], panel=3, color='magenta', width=2, 
                               secondary_y=True, label='Alpha', type='line')
            )
            
            panel_ratios = (5, 1.5, 1.2, 1.2)  # メイン:HyperER:Limits:Period&Alpha
        
        # mplfinanceの設定
        kwargs = dict(
            type='candle',
            figsize=figsize,
            title=title,
            style=style,
            datetime_format='%Y-%m-%d',
            xrotation=45,
            returnfig=True,
            volume=show_volume,
            panel_ratios=panel_ratios,
            addplot=additional_plots
        )
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        if show_volume:
            # HyperER効率性パネル（パネル2）に参照線
            axes[2].axhline(y=0.0, color='black', linestyle='-', alpha=0.3, label='Zero')
            axes[2].axhline(y=0.2, color='red', linestyle='--', alpha=0.7, label='Low Threshold')
            axes[2].axhline(y=0.5, color='gray', linestyle='-', alpha=0.5, label='Neutral')
            axes[2].axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='High Threshold')
            axes[2].axhline(y=1.0, color='black', linestyle='-', alpha=0.3, label='Maximum')
            axes[2].set_ylim(-0.1, 1.1)
            
            # 動的Limitsパネル（パネル3）に参照線
            axes[3].axhline(y=0.0, color='black', linestyle='-', alpha=0.3)
            axes[3].set_ylim(0.0, 0.6)
            
            # Period&Alphaパネル（パネル4）の設定
            axes[4].set_ylim(5, 55)  # Period範囲
            # Alpha値は secondary_y 軸なので別途設定が必要
            ax4_right = axes[4].right_ax if hasattr(axes[4], 'right_ax') else None
            if ax4_right:
                ax4_right.set_ylim(0.0, 0.6)  # Alpha範囲
        else:
            # HyperER効率性パネル（パネル1）に参照線
            axes[1].axhline(y=0.0, color='black', linestyle='-', alpha=0.3, label='Zero')
            axes[1].axhline(y=0.2, color='red', linestyle='--', alpha=0.7, label='Low Threshold')
            axes[1].axhline(y=0.5, color='gray', linestyle='-', alpha=0.5, label='Neutral')
            axes[1].axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='High Threshold')
            axes[1].axhline(y=1.0, color='black', linestyle='-', alpha=0.3, label='Maximum')
            axes[1].set_ylim(-0.1, 1.1)
            
            # 動的Limitsパネル（パネル2）に参照線
            axes[2].axhline(y=0.0, color='black', linestyle='-', alpha=0.3)
            axes[2].set_ylim(0.0, 0.6)
            
            # Period&Alphaパネル（パネル3）の設定
            axes[3].set_ylim(5, 55)  # Period範囲
            # Alpha値は secondary_y 軸なので別途設定が必要
            ax3_right = axes[3].right_ax if hasattr(axes[3], 'right_ax') else None
            if ax3_right:
                ax3_right.set_ylim(0.0, 0.6)  # Alpha範囲
        
        # 統計情報の表示
        print(f"\\n=== HyperMAMAインジケーター統計 ===")
        
        # HyperMAMA統計
        valid_mama = df['mama'].dropna()
        valid_fama = df['fama'].dropna()
        valid_hyper_er = df['hyper_er'].dropna()
        
        if len(valid_mama) > 0:
            print(f"HyperMAMA範囲: {valid_mama.min():.2f} - {valid_mama.max():.2f}")
            print(f"HyperMAMA平均: {valid_mama.mean():.2f}")
            
        if len(valid_fama) > 0:
            print(f"HyperFAMA範囲: {valid_fama.min():.2f} - {valid_fama.max():.2f}")
            print(f"HyperFAMA平均: {valid_fama.mean():.2f}")
            
        if len(valid_hyper_er) > 0:
            high_er_count = len(valid_hyper_er[valid_hyper_er >= 0.8])
            medium_er_count = len(valid_hyper_er[(valid_hyper_er >= 0.2) & (valid_hyper_er < 0.8)])
            low_er_count = len(valid_hyper_er[valid_hyper_er < 0.2])
            
            print(f"HyperER範囲: {valid_hyper_er.min():.3f} - {valid_hyper_er.max():.3f}")
            print(f"HyperER平均: {valid_hyper_er.mean():.3f}")
            print(f"高効率期間 (≥0.8): {high_er_count} ({high_er_count/len(valid_hyper_er)*100:.1f}%)")
            print(f"中効率期間 (0.2-0.8): {medium_er_count} ({medium_er_count/len(valid_hyper_er)*100:.1f}%)")
            print(f"低効率期間 (<0.2): {low_er_count} ({low_er_count/len(valid_hyper_er)*100:.1f}%)")
        
        # 動的適応統計
        valid_fast_limit = df['fast_limit'].dropna()
        valid_slow_limit = df['slow_limit'].dropna()
        
        if len(valid_fast_limit) > 0:
            print(f"動的FastLimit範囲: {valid_fast_limit.min():.4f} - {valid_fast_limit.max():.4f}")
            print(f"動的FastLimit平均: {valid_fast_limit.mean():.4f}")
            
        if len(valid_slow_limit) > 0:
            print(f"動的SlowLimit範囲: {valid_slow_limit.min():.4f} - {valid_slow_limit.max():.4f}")
            print(f"動的SlowLimit平均: {valid_slow_limit.mean():.4f}")
        
        # Period & Alpha統計
        valid_period = df['period'].dropna()
        valid_alpha = df['alpha'].dropna()
        
        if len(valid_period) > 0:
            print(f"Period範囲: {valid_period.min():.1f} - {valid_period.max():.1f}")
            print(f"Period平均: {valid_period.mean():.1f}")
            
        if len(valid_alpha) > 0:
            print(f"Alpha範囲: {valid_alpha.min():.4f} - {valid_alpha.max():.4f}")
            print(f"Alpha平均: {valid_alpha.mean():.4f}")
        
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
    parser = argparse.ArgumentParser(description='HyperMAMA & HyperFAMAチャートの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--bars', '-b', type=int, default=500, help='最大データ数 (デフォルト: 500)')
    
    # HyperMAMAパラメータ
    parser.add_argument('--trigger_type', type=str, default='hyper_er', 
                       choices=['hyper_er', 'hyper_trend_index'], help='トリガータイプ')
    parser.add_argument('--hyper_er_period', type=int, default=14, help='HyperER計算期間')
    parser.add_argument('--hyper_er_midline_period', type=int, default=100, help='HyperERミッドライン計算期間')
    parser.add_argument('--hyper_trend_period', type=int, default=14, help='HyperTrendIndex計算期間')
    parser.add_argument('--hyper_trend_midline_period', type=int, default=100, help='HyperTrendIndexミッドライン計算期間')
    parser.add_argument('--fast_max', type=float, default=0.8, help='fastlimitの最大値')
    parser.add_argument('--fast_min', type=float, default=0.1, help='fastlimitの最小値')
    parser.add_argument('--slow_max', type=float, default=0.08, help='slowlimitの最大値')
    parser.add_argument('--slow_min', type=float, default=0.01, help='slowlimitの最小値')
    parser.add_argument('--er_high_threshold', type=float, default=0.8, help='HyperERの高閾値')
    parser.add_argument('--er_low_threshold', type=float, default=0.2, help='HyperERの低閾値')
    parser.add_argument('--src_type', type=str, default='hlc3', help='ソースタイプ')
    parser.add_argument('--use_kalman_filter', action='store_true', help='カルマンフィルターを使用する')
    parser.add_argument('--use_zero_lag', action='store_true', default=True, help='ゼロラグ処理を使用する')
    
    args = parser.parse_args()
    
    # チャートを作成
    chart = HyperMAMAChart()
    chart.load_data_from_config(args.config, max_bars=args.bars)
    chart.calculate_indicators(
        trigger_type=args.trigger_type,
        hyper_er_period=args.hyper_er_period,
        hyper_er_midline_period=args.hyper_er_midline_period,
        hyper_trend_period=args.hyper_trend_period,
        hyper_trend_midline_period=args.hyper_trend_midline_period,
        fast_max=args.fast_max,
        fast_min=args.fast_min,
        slow_max=args.slow_max,
        slow_min=args.slow_min,
        er_high_threshold=args.er_high_threshold,
        er_low_threshold=args.er_low_threshold,
        src_type=args.src_type,
        use_kalman_filter=args.use_kalman_filter,
        use_zero_lag=args.use_zero_lag
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output
    )


if __name__ == "__main__":
    main()