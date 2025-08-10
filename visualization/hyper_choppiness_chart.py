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
from indicators.trend_filter.hyper_choppiness import HyperChoppiness


class HyperChoppinessChart:
    """
    Hyper Choppinessを表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - Hyper Choppiness値（0-100スケール）
    - 相場レジーム分類（チョピー/中立/トレンド）
    - 動的サイクル期間
    - ルーフィングフィルター済み高値・安値
    - 生のチョピネス値と平滑化後の値
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.hyper_choppiness = None
        self.fig = None
        self.axes = None
    
    def load_data_from_config(self, config_path: str, limit: int = 500) -> pd.DataFrame:
        """
        設定ファイルからデータを読み込む
        
        Args:
            config_path: 設定ファイルのパス
            limit: 取得するデータ数（最新からの件数）
            
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
        
        # CSVデータソースはダミーとして渡す
        dummy_csv_source = CSVDataSource("dummy")
        data_loader = DataLoader(
            data_source=dummy_csv_source,
            binance_data_source=binance_data_source
        )
        data_processor = DataProcessor()
        
        # データの読み込みと処理
        print(f"\nデータを読み込み・処理中... (最新{limit}件)")
        raw_data = data_loader.load_data_from_config(config)
        processed_data = {
            symbol: data_processor.process(df)
            for symbol, df in raw_data.items()
        }
        
        # 最初のシンボルのデータを取得し、最新のlimit件に制限
        first_symbol = next(iter(processed_data))
        full_data = processed_data[first_symbol]
        
        # 最新のlimit件を取得
        if len(full_data) > limit:
            self.data = full_data.tail(limit).copy()
        else:
            self.data = full_data.copy()
        
        print(f"データ読み込み完了: {first_symbol}")
        print(f"期間: {self.data.index.min()} → {self.data.index.max()}")
        print(f"データ数: {len(self.data)}")
        
        return self.data

    def calculate_indicators(self,
                            # ルーフィングフィルターパラメータ
                            use_roofing_filter: bool = True,
                            roofing_hp_cutoff: float = 48.0,
                            roofing_ss_band_edge: float = 10.0,
                            
                            # サイクル検出パラメータ
                            use_cycle_detection: bool = True,
                            min_cycle_period: int = 10,
                            max_cycle_period: int = 50,
                            default_period: int = 14,
                            
                            # 平滑化パラメータ
                            smoothing_period: int = 3,
                            
                            # 価格ソース
                            source_type: str = 'hlc3',
                            
                            # 相場レジーム判定
                            choppy_threshold: float = 61.8,
                            trending_threshold: float = 38.2
                           ) -> None:
        """
        Hyper Choppinessを計算する
        
        Args:
            use_roofing_filter: ルーフィングフィルターを使用するか
            roofing_hp_cutoff: ルーフィングフィルターのハイパスカットオフ
            roofing_ss_band_edge: ルーフィングフィルターのスーパースムーサーバンドエッジ
            use_cycle_detection: サイクル検出を使用するか
            min_cycle_period: 最小サイクル期間
            max_cycle_period: 最大サイクル期間
            default_period: デフォルトのチョピネス計算期間
            smoothing_period: 最終結果の平滑化期間
            source_type: 価格ソースタイプ
            choppy_threshold: チョピー相場の閾値
            trending_threshold: トレンド相場の閾値
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\nHyper Choppinessを計算中...")
        
        # Hyper Choppinessを計算
        self.hyper_choppiness = HyperChoppiness(
            use_roofing_filter=use_roofing_filter,
            roofing_hp_cutoff=roofing_hp_cutoff,
            roofing_ss_band_edge=roofing_ss_band_edge,
            use_cycle_detection=use_cycle_detection,
            min_cycle_period=min_cycle_period,
            max_cycle_period=max_cycle_period,
            default_period=default_period,
            smoothing_period=smoothing_period,
            source_type=source_type
        )
        
        # Hyper Choppinessの計算
        print("計算を実行します...")
        hyper_chop_result = self.hyper_choppiness.calculate(self.data)
        
        # 相場レジーム分類
        market_regime = self.hyper_choppiness.get_market_regime(
            choppy_threshold=choppy_threshold,
            trending_threshold=trending_threshold
        )
        
        print(f"Hyper Choppiness計算完了")
        print(f"チョピネス値範囲: {np.nanmin(hyper_chop_result):.2f} - {np.nanmax(hyper_chop_result):.2f}")
        print(f"平均サイクル期間: {np.nanmean(self.hyper_choppiness.get_cycle_periods()):.1f}")
        print(f"チョピー期間: {np.sum(market_regime == 1)}件")
        print(f"トレンド期間: {np.sum(market_regime == -1)}件")
        print(f"中立期間: {np.sum(market_regime == 0)}件")
            
    def plot(self, 
            title: str = "Hyper Choppiness Analysis", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (16, 14),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとHyper Choppinessを描画する
        
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
            
        if self.hyper_choppiness is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # Hyper Choppinessの値を取得
        print("Hyper Choppinessデータを取得中...")
        hyper_chop_values = self.hyper_choppiness.get_choppiness_values()
        raw_chop_values = self.hyper_choppiness.get_raw_choppiness()
        cycle_periods = self.hyper_choppiness.get_cycle_periods()
        roofing_high, roofing_low = self.hyper_choppiness.get_roofing_data()
        market_regime = self.hyper_choppiness.get_market_regime()
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'hyper_choppiness': hyper_chop_values,
                'raw_choppiness': raw_chop_values,
                'cycle_periods': cycle_periods,
                'roofing_high': roofing_high,
                'roofing_low': roofing_low,
                'market_regime': market_regime
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        print(f"Hyper Choppiness確認 - NaN: {df['hyper_choppiness'].isna().sum()}")
        
        # デバッグ情報の出力
        if df['hyper_choppiness'].isna().sum() == len(df):
            print("警告: すべてのHyper Choppiness値がNaNです")
            print("ルーフィングフィルターなしでの計算を試してください: --no-roofing")
        
        # NaN値を0で埋める（プロット用）
        df['hyper_choppiness'].fillna(0, inplace=True)
        df['raw_choppiness'].fillna(0, inplace=True)
        df['cycle_periods'].fillna(df['cycle_periods'].mean(), inplace=True)
        df['market_regime'].fillna(0, inplace=True)
        
        # 相場レジーム別の色分け用データ準備
        df['choppy_regime'] = np.where(df['market_regime'] == 1, df['hyper_choppiness'], np.nan)
        df['neutral_regime'] = np.where(df['market_regime'] == 0, df['hyper_choppiness'], np.nan)
        df['trending_regime'] = np.where(df['market_regime'] == -1, df['hyper_choppiness'], np.nan)
        
        # ルーフィングフィルターと原価格の比較用
        df['price_high'] = df['high']
        df['price_low'] = df['low']
        
        # mplfinanceでプロット用の設定
        main_plots = []
        
        # ルーフィングフィルター済み高値・安値をメインチャートに表示
        main_plots.append(mpf.make_addplot(df['roofing_high'], color='lightblue', width=1, alpha=0.6, label='Roofing High'))
        main_plots.append(mpf.make_addplot(df['roofing_low'], color='lightcoral', width=1, alpha=0.6, label='Roofing Low'))
        
        # Hyper Choppinessパネル（相場レジーム別色分け）
        hyper_chop_choppy = mpf.make_addplot(df['choppy_regime'], panel=1, color='red', width=2, 
                                           ylabel='Hyper Choppiness', secondary_y=False, label='Choppy')
        hyper_chop_neutral = mpf.make_addplot(df['neutral_regime'], panel=1, color='orange', width=2, 
                                            ylabel='Hyper Choppiness', secondary_y=False, label='Neutral')
        hyper_chop_trending = mpf.make_addplot(df['trending_regime'], panel=1, color='green', width=2, 
                                             ylabel='Hyper Choppiness', secondary_y=False, label='Trending')
        
        # 生のチョピネス値パネル
        raw_chop_panel = mpf.make_addplot(df['raw_choppiness'], panel=2, color='gray', width=1, alpha=0.7,
                                        ylabel='Raw Choppiness', secondary_y=False, label='Raw')
        
        # サイクル期間パネル
        cycle_panel = mpf.make_addplot(df['cycle_periods'], panel=3, color='blue', width=1.5,
                                      ylabel='Cycle Period', secondary_y=False, label='Cycle')
        
        # 相場レジームパネル
        regime_panel = mpf.make_addplot(df['market_regime'], panel=4, color='purple', width=2,
                                       ylabel='Market Regime', secondary_y=False, label='Regime', type='line')
        
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
            kwargs['panel_ratios'] = (4, 1, 1.5, 1, 1, 1)  # メイン:出来高:Hyper Chop:Raw Chop:Cycle:Regime
            # 出来高を表示する場合は、パネル番号を+1する
            hyper_chop_choppy = mpf.make_addplot(df['choppy_regime'], panel=2, color='red', width=2, 
                                               ylabel='Hyper Choppiness', secondary_y=False, label='Choppy')
            hyper_chop_neutral = mpf.make_addplot(df['neutral_regime'], panel=2, color='orange', width=2, 
                                                ylabel='Hyper Choppiness', secondary_y=False, label='Neutral')
            hyper_chop_trending = mpf.make_addplot(df['trending_regime'], panel=2, color='green', width=2, 
                                                 ylabel='Hyper Choppiness', secondary_y=False, label='Trending')
            raw_chop_panel = mpf.make_addplot(df['raw_choppiness'], panel=3, color='gray', width=1, alpha=0.7,
                                            ylabel='Raw Choppiness', secondary_y=False, label='Raw')
            cycle_panel = mpf.make_addplot(df['cycle_periods'], panel=4, color='blue', width=1.5,
                                          ylabel='Cycle Period', secondary_y=False, label='Cycle')
            regime_panel = mpf.make_addplot(df['market_regime'], panel=5, color='purple', width=2,
                                           ylabel='Market Regime', secondary_y=False, label='Regime', type='line')
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (4, 1.5, 1, 1, 1)  # メイン:Hyper Chop:Raw Chop:Cycle:Regime
        
        # すべてのプロットを結合
        all_plots = main_plots + [hyper_chop_choppy, hyper_chop_neutral, hyper_chop_trending, 
                                 raw_chop_panel, cycle_panel, regime_panel]
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加
        axes[0].legend(['Roofing High', 'Roofing Low'], loc='upper left')
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        panel_offset = 2 if show_volume else 1
        
        # Hyper Choppinessパネル - 閾値線
        axes[panel_offset].axhline(y=61.8, color='red', linestyle='--', alpha=0.7, label='Choppy Threshold')
        axes[panel_offset].axhline(y=38.2, color='green', linestyle='--', alpha=0.7, label='Trending Threshold')
        axes[panel_offset].axhline(y=50.0, color='black', linestyle='-', alpha=0.3)
        axes[panel_offset].set_ylim(0, 100)
        
        # Raw Choppinessパネル
        axes[panel_offset + 1].axhline(y=50.0, color='black', linestyle='-', alpha=0.3)
        
        # サイクル期間パネル
        cycle_mean = df['cycle_periods'].mean()
        axes[panel_offset + 2].axhline(y=cycle_mean, color='black', linestyle='-', alpha=0.3)
        
        # 相場レジームパネル
        axes[panel_offset + 3].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[panel_offset + 3].axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Choppy')
        axes[panel_offset + 3].axhline(y=-1, color='green', linestyle='--', alpha=0.5, label='Trending')
        axes[panel_offset + 3].set_ylim(-1.5, 1.5)
        
        # 統計情報の表示
        print(f"\n=== Hyper Choppiness統計 ===")
        total_points = len(df[~df['hyper_choppiness'].isna()])
        choppy_points = len(df[df['market_regime'] == 1])
        trending_points = len(df[df['market_regime'] == -1])
        neutral_points = len(df[df['market_regime'] == 0])
        
        print(f"総データ点数: {total_points}")
        print(f"チョピー相場: {choppy_points} ({choppy_points/total_points*100:.1f}%)")
        print(f"トレンド相場: {trending_points} ({trending_points/total_points*100:.1f}%)")
        print(f"中立相場: {neutral_points} ({neutral_points/total_points*100:.1f}%)")
        print(f"Hyper Choppiness - 平均: {df['hyper_choppiness'].mean():.2f}, 範囲: {df['hyper_choppiness'].min():.2f} - {df['hyper_choppiness'].max():.2f}")
        print(f"サイクル期間 - 平均: {df['cycle_periods'].mean():.1f}, 範囲: {df['cycle_periods'].min():.0f} - {df['cycle_periods'].max():.0f}")
        
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
    parser = argparse.ArgumentParser(description='Hyper Choppinessの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--limit', '-l', type=int, default=500, help='取得するデータ数')
    parser.add_argument('--source', type=str, default='hlc3', help='価格ソースタイプ')
    parser.add_argument('--roofing-hp', type=float, default=48.0, help='ルーフィングフィルターHPカットオフ')
    parser.add_argument('--roofing-ss', type=float, default=10.0, help='ルーフィングフィルターSSバンドエッジ')
    parser.add_argument('--min-cycle', type=int, default=10, help='最小サイクル期間')
    parser.add_argument('--max-cycle', type=int, default=50, help='最大サイクル期間')
    parser.add_argument('--smooth-period', type=int, default=3, help='平滑化期間')
    parser.add_argument('--no-roofing', action='store_true', help='ルーフィングフィルターを無効化')
    parser.add_argument('--no-cycle', action='store_true', help='サイクル検出を無効化')
    args = parser.parse_args()
    
    # チャートを作成
    chart = HyperChoppinessChart()
    chart.load_data_from_config(args.config, limit=args.limit)
    chart.calculate_indicators(
        use_roofing_filter=not args.no_roofing,
        roofing_hp_cutoff=args.roofing_hp,
        roofing_ss_band_edge=args.roofing_ss,
        use_cycle_detection=not args.no_cycle,
        min_cycle_period=args.min_cycle,
        max_cycle_period=args.max_cycle,
        smoothing_period=args.smooth_period,
        source_type=args.source
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output
    )


if __name__ == "__main__":
    main()