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
from indicators.adaptive_ma import AdaptiveMA


class AdaptiveMAChart:
    """
    AdaptiveMA（期間適応移動平均線）を表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - 期間適応MA線（効率比に基づいて期間が動的変化）
    - 動的期間の変化
    - 効率比（Efficiency Ratio）
    - トレンド判定信号（Bullish/Bearish）
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.adaptive_ma = None
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
                            # AdaptiveMA パラメータ
                            ma_type: str = 'sma',
                            min_period: int = 5,
                            max_period: int = 50,
                            er_period: int = 10,
                            src_type: str = 'close',
                            slope_index: int = 1
                           ) -> None:
        """
        期間適応MAを計算する
        
        Args:
            ma_type: MAのタイプ ('sma', 'ema', 'hma', 'alma', 'zlema', 'kama', 'hyperma')
            min_period: 最小期間（高効率時）
            max_period: 最大期間（低効率時）
            er_period: 効率比の計算期間
            src_type: 価格ソース ('close', 'hlc3', etc.)
            slope_index: トレンド判定期間
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\n期間適応MAを計算中...")
        
        # AdaptiveMAインスタンスを作成
        self.adaptive_ma = AdaptiveMA(
            ma_type=ma_type,
            min_period=min_period,
            max_period=max_period,
            er_period=er_period,
            src_type=src_type,
            slope_index=slope_index
        )
        
        # AdaptiveMAの計算
        print("計算を実行します...")
        result = self.adaptive_ma.calculate(self.data)
        
        # 結果の確認
        print(f"計算完了:")
        print(f"  - 適応MA: {len(result.values)} データポイント")
        print(f"  - 動的期間: {len(result.dynamic_periods)} データポイント")
        print(f"  - 効率比: {len(result.efficiency_ratio)} データポイント")
        print(f"  - 現在のトレンド: {result.current_trend}")
        
        # NaN値のチェック
        nan_count_adaptive = np.isnan(result.values).sum()
        nan_count_periods = np.isnan(result.dynamic_periods).sum()
        nan_count_er = np.isnan(result.efficiency_ratio).sum()
        
        print(f"NaN値:")
        print(f"  - 適応MA: {nan_count_adaptive}")
        print(f"  - 動的期間: {nan_count_periods}")
        print(f"  - 効率比: {nan_count_er}")
        
        # 期間の統計情報
        valid_periods = result.dynamic_periods[~np.isnan(result.dynamic_periods)]
        if len(valid_periods) > 0:
            print(f"動的期間統計:")
            print(f"  - 平均期間: {valid_periods.mean():.1f}")
            print(f"  - 最小期間: {valid_periods.min():.0f}")
            print(f"  - 最大期間: {valid_periods.max():.0f}")
        
        print("期間適応MA計算完了")
            
    def plot(self, 
            title: str = "AdaptiveMA - 期間適応移動平均線", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            show_trend_signals: bool = True,
            show_periods: bool = True,
            figsize: Tuple[int, int] = (14, 12),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとAdaptiveMAを描画する
        
        Args:
            title: チャートのタイトル
            start_date: 表示開始日（フォーマット: YYYY-MM-DD）
            end_date: 表示終了日（フォーマット: YYYY-MM-DD）
            show_volume: 出来高を表示するか
            show_trend_signals: トレンド信号を表示するか
            show_periods: 動的期間を表示するか
            figsize: 図のサイズ
            style: mplfinanceのスタイル
            savefig: 保存先のパス（指定しない場合は表示のみ）
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        if self.adaptive_ma is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # AdaptiveMAの値を取得
        print("AdaptiveMAデータを取得中...")
        result = self.adaptive_ma.calculate(self.data)
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'adaptive_ma': result.values,
                'dynamic_periods': result.dynamic_periods,
                'efficiency_ratio': result.efficiency_ratio,
                'is_bullish': result.is_bullish.astype(int),
                'is_bearish': result.is_bearish.astype(int)
            }
        )
        
        # トレンド信号の数値変換（視覚化用）
        full_df['trend_signal'] = np.where(
            result.is_bullish, 1,
            np.where(result.is_bearish, -1, 0)
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        print(f"AdaptiveMAデータ確認:")
        print(f"  - 適応MA NaN: {df['adaptive_ma'].isna().sum()}")
        print(f"  - 動的期間 NaN: {df['dynamic_periods'].isna().sum()}")
        print(f"  - 効率比 NaN: {df['efficiency_ratio'].isna().sum()}")
        
        # データ妥当性チェック
        valid_adaptive_ma = df['adaptive_ma'].notna().sum()
        if valid_adaptive_ma == 0:
            raise ValueError(f"AdaptiveMA計算結果が全てNaNです。MAタイプ'{self.adaptive_ma.ma_type}'の計算に問題がある可能性があります。")
        elif valid_adaptive_ma < len(df) * 0.1:  # 有効データが10%未満
            print(f"警告: AdaptiveMAの有効データが少ないです ({valid_adaptive_ma}/{len(df)} = {valid_adaptive_ma/len(df)*100:.1f}%)")
        
        # NaN値を含む行の情報出力（最初の5行のみ）
        nan_rows = df[df['adaptive_ma'].isna() | df['dynamic_periods'].isna()]
        if not nan_rows.empty:
            print(f"NaN値を含む行: {len(nan_rows)}行")
            if len(nan_rows) > 0:
                print(nan_rows[['adaptive_ma', 'dynamic_periods', 'efficiency_ratio']].head())
        
        # mplfinanceでプロット用の設定
        # 1. メインチャート上のプロット
        main_plots = []
        
        # AdaptiveMAのプロット設定
        main_plots.append(mpf.make_addplot(df['adaptive_ma'], color='purple', width=2.5, label='Adaptive MA'))
        
        # 2. 効率比パネル
        er_panel = mpf.make_addplot(df['efficiency_ratio'], panel=1, color='orange', width=1.5, 
                                   ylabel='Efficiency Ratio', secondary_y=False, label='ER')
        
        # 3. 動的期間パネル（オプション）
        periods_panel = None
        if show_periods:
            periods_panel = mpf.make_addplot(df['dynamic_periods'], panel=2, color='blue', width=1.5, 
                                           ylabel='Dynamic Period', secondary_y=False, label='Period')
        
        # 4. トレンド信号パネル（オプション）
        trend_panel = None
        if show_trend_signals:
            panel_index = 3 if show_periods else 2
            trend_panel = mpf.make_addplot(df['trend_signal'], panel=panel_index, color='green', width=1.5, 
                                         type='scatter', markersize=30, marker='o', alpha=0.6,
                                         ylabel='Trend Signal', secondary_y=False, label='Trend')
        
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
        
        # パネル数と比率の設定
        all_plots = main_plots + [er_panel]
        if show_periods:
            all_plots.append(periods_panel)
        if show_trend_signals:
            all_plots.append(trend_panel)
        
        # 出来高とパネル設定
        panel_count = 1 + (1 if show_periods else 0) + (1 if show_trend_signals else 0)  # ER + オプション
        
        if show_volume:
            kwargs['volume'] = True
            if show_periods and show_trend_signals:
                kwargs['panel_ratios'] = (4, 1, 1, 1, 1)  # メイン:出来高:ER:期間:トレンド
                # 出来高表示時はパネル番号を+1
                er_panel = mpf.make_addplot(df['efficiency_ratio'], panel=2, color='orange', width=1.5, 
                                           ylabel='Efficiency Ratio', secondary_y=False, label='ER')
                if periods_panel:
                    periods_panel = mpf.make_addplot(df['dynamic_periods'], panel=3, color='blue', width=1.5, 
                                                   ylabel='Dynamic Period', secondary_y=False, label='Period')
                if trend_panel:
                    trend_panel = mpf.make_addplot(df['trend_signal'], panel=4, color='green', width=1.5, 
                                                 type='scatter', markersize=30, marker='o', alpha=0.6,
                                                 ylabel='Trend Signal', secondary_y=False, label='Trend')
                all_plots = main_plots + [er_panel, periods_panel, trend_panel]
            elif show_periods:
                kwargs['panel_ratios'] = (4, 1, 1, 1)  # メイン:出来高:ER:期間
                er_panel = mpf.make_addplot(df['efficiency_ratio'], panel=2, color='orange', width=1.5, 
                                           ylabel='Efficiency Ratio', secondary_y=False, label='ER')
                periods_panel = mpf.make_addplot(df['dynamic_periods'], panel=3, color='blue', width=1.5, 
                                               ylabel='Dynamic Period', secondary_y=False, label='Period')
                all_plots = main_plots + [er_panel, periods_panel]
            elif show_trend_signals:
                kwargs['panel_ratios'] = (4, 1, 1, 1)  # メイン:出来高:ER:トレンド
                er_panel = mpf.make_addplot(df['efficiency_ratio'], panel=2, color='orange', width=1.5, 
                                           ylabel='Efficiency Ratio', secondary_y=False, label='ER')
                trend_panel = mpf.make_addplot(df['trend_signal'], panel=3, color='green', width=1.5, 
                                             type='scatter', markersize=30, marker='o', alpha=0.6,
                                             ylabel='Trend Signal', secondary_y=False, label='Trend')
                all_plots = main_plots + [er_panel, trend_panel]
            else:
                kwargs['panel_ratios'] = (4, 1, 1)  # メイン:出来高:ER
                er_panel = mpf.make_addplot(df['efficiency_ratio'], panel=2, color='orange', width=1.5, 
                                           ylabel='Efficiency Ratio', secondary_y=False, label='ER')
                all_plots = main_plots + [er_panel]
        else:
            kwargs['volume'] = False
            if show_periods and show_trend_signals:
                kwargs['panel_ratios'] = (4, 1, 1, 1)  # メイン:ER:期間:トレンド
            elif show_periods or show_trend_signals:
                kwargs['panel_ratios'] = (4, 1, 1)  # メイン:ER:期間orトレンド
            else:
                kwargs['panel_ratios'] = (4, 1)  # メイン:ER
        
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加
        axes[0].legend(['Adaptive MA'], loc='upper left')
        
        self.fig = fig
        self.axes = axes
        
        # 効率比パネルに参照線を追加
        er_panel_index = 2 if show_volume else 1
        axes[er_panel_index].axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Perfect Trend')
        axes[er_panel_index].axhline(y=0.5, color='gray', linestyle='-', alpha=0.3, label='Neutral')
        axes[er_panel_index].axhline(y=0.0, color='black', linestyle='--', alpha=0.5, label='Random')
        axes[er_panel_index].set_ylim(-0.1, 1.1)
        
        # 動的期間パネルに参照線を追加
        if show_periods:
            periods_panel_index = (3 if show_volume else 2)
            min_period = self.adaptive_ma.min_period
            max_period = self.adaptive_ma.max_period
            axes[periods_panel_index].axhline(y=min_period, color='green', linestyle='--', alpha=0.5, 
                                            label=f'Min Period ({min_period})')
            axes[periods_panel_index].axhline(y=max_period, color='red', linestyle='--', alpha=0.5, 
                                            label=f'Max Period ({max_period})')
            avg_period = (min_period + max_period) / 2
            axes[periods_panel_index].axhline(y=avg_period, color='gray', linestyle='-', alpha=0.3, 
                                            label=f'Avg Period ({avg_period:.0f})')
            axes[periods_panel_index].set_ylim(min_period - 2, max_period + 2)
        
        # トレンド信号パネルに参照線を追加
        if show_trend_signals:
            if show_periods:
                trend_panel_index = 4 if show_volume else 3
            else:
                trend_panel_index = 3 if show_volume else 2
                
            axes[trend_panel_index].axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Bullish')
            axes[trend_panel_index].axhline(y=0, color='gray', linestyle='-', alpha=0.3, label='Neutral')
            axes[trend_panel_index].axhline(y=-1, color='red', linestyle='--', alpha=0.5, label='Bearish')
            axes[trend_panel_index].set_ylim(-1.5, 1.5)
        
        # 現在のトレンド状態と期間をタイトルに追加
        current_trend = result.current_trend.upper()
        trend_color = 'green' if current_trend == 'BULLISH' else 'red' if current_trend == 'BEARISH' else 'gray'
        
        # 現在の期間を取得
        current_period = result.dynamic_periods[-1] if not np.isnan(result.dynamic_periods[-1]) else "N/A"
        
        fig.suptitle(f"{title} - Current Trend: {current_trend} | Period: {current_period}", 
                    fontsize=14, color=trend_color)
        
        # 保存または表示
        if savefig:
            plt.savefig(savefig, dpi=300, bbox_inches='tight')
            print(f"チャートを保存しました: {savefig}")
        else:
            plt.tight_layout()
            plt.show()

    def print_statistics(self) -> None:
        """統計情報を表示する"""
        if self.adaptive_ma is None:
            print("インジケーターが計算されていません。")
            return
            
        result = self.adaptive_ma.calculate(self.data)
        
        print("\n=== AdaptiveMA 統計情報 ===")
        print(f"MAタイプ: {self.adaptive_ma.ma_type.upper()}")
        print(f"最小期間: {self.adaptive_ma.min_period}")
        print(f"最大期間: {self.adaptive_ma.max_period}")
        print(f"効率比期間: {self.adaptive_ma.er_period}")
        print(f"価格ソース: {self.adaptive_ma.src_type.upper()}")
        print(f"トレンド判定期間: {self.adaptive_ma.slope_index}")
        
        print(f"\n現在のトレンド状態: {result.current_trend.upper()}")
        print(f"現在上昇トレンド: {result.is_currently_bullish}")
        print(f"現在下降トレンド: {result.is_currently_bearish}")
        
        # 効率比の統計
        er_valid = result.efficiency_ratio[~np.isnan(result.efficiency_ratio)]
        if len(er_valid) > 0:
            print(f"\n効率比統計:")
            print(f"  平均: {er_valid.mean():.4f}")
            print(f"  最大: {er_valid.max():.4f}")
            print(f"  最小: {er_valid.min():.4f}")
            print(f"  標準偏差: {er_valid.std():.4f}")
        
        # 動的期間の統計
        periods_valid = result.dynamic_periods[~np.isnan(result.dynamic_periods)]
        if len(periods_valid) > 0:
            print(f"\n動的期間統計:")
            print(f"  平均期間: {periods_valid.mean():.1f}")
            print(f"  最大期間: {periods_valid.max():.0f}")
            print(f"  最小期間: {periods_valid.min():.0f}")
            print(f"  標準偏差: {periods_valid.std():.1f}")
            
            # 期間分布
            short_count = (periods_valid <= self.adaptive_ma.min_period + 2).sum()
            long_count = (periods_valid >= self.adaptive_ma.max_period - 2).sum()
            mid_count = len(periods_valid) - short_count - long_count
            total_count = len(periods_valid)
            
            print(f"\n期間分布:")
            print(f"  短期間({self.adaptive_ma.min_period}-{self.adaptive_ma.min_period+2}): {short_count} ({short_count/total_count*100:.1f}%)")
            print(f"  中期間: {mid_count} ({mid_count/total_count*100:.1f}%)")
            print(f"  長期間({self.adaptive_ma.max_period-2}-{self.adaptive_ma.max_period}): {long_count} ({long_count/total_count*100:.1f}%)")
        
        # トレンド信号の統計
        bullish_count = result.is_bullish.sum()
        bearish_count = result.is_bearish.sum()
        neutral_count = len(result.is_bullish) - bullish_count - bearish_count
        total_count = len(result.is_bullish)
        
        print(f"\nトレンド信号統計:")
        print(f"  上昇トレンド: {bullish_count} ({bullish_count/total_count*100:.1f}%)")
        print(f"  下降トレンド: {bearish_count} ({bearish_count/total_count*100:.1f}%)")
        print(f"  ニュートラル: {neutral_count} ({neutral_count/total_count*100:.1f}%)")


def main():
    """メイン関数"""
    # コマンドライン引数を処理
    import argparse
    parser = argparse.ArgumentParser(description='期間適応MAの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--ma-type', type=str, default='hma', help='MAタイプ (sma, ema, hma, alma, zlema, hyperma)')
    parser.add_argument('--min-period', type=int, default=2, help='最小期間（高効率時）')
    parser.add_argument('--max-period', type=int, default=144, help='最大期間（低効率時）')
    parser.add_argument('--er-period', type=int, default=10, help='効率比計算期間')
    parser.add_argument('--src-type', type=str, default='close', help='価格ソースタイプ')
    parser.add_argument('--slope-index', type=int, default=1, help='トレンド判定期間')
    parser.add_argument('--no-volume', action='store_true', help='出来高を非表示')
    parser.add_argument('--no-trend', action='store_true', help='トレンド信号を非表示')
    parser.add_argument('--no-periods', action='store_true', help='動的期間を非表示')
    parser.add_argument('--stats', action='store_true', help='統計情報を表示')
    args = parser.parse_args()
    
    # チャートを作成
    chart = AdaptiveMAChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        ma_type=args.ma_type,
        min_period=args.min_period,
        max_period=args.max_period,
        er_period=args.er_period,
        src_type=args.src_type,
        slope_index=args.slope_index
    )
    
    # 統計情報表示
    if args.stats:
        chart.print_statistics()
    
    # チャート描画
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        show_volume=not args.no_volume,
        show_trend_signals=not args.no_trend,
        show_periods=not args.no_periods,
        savefig=args.output
    )


if __name__ == "__main__":
    main() 