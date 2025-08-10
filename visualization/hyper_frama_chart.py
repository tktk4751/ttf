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
from indicators.hyper_frama import HyperFRAMA


class HyperFRAMAChart:
    """
    ハイパーFRAMAを表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - 通常のFRAMA（メインライン）
    - アルファ半分のFRAMA（シグナルライン）
    - フラクタル次元
    - アルファ値とアルファ半分値
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.hyper_frama = None
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
                            period: int = 16,
                            src_type: str = 'hl2',
                            fc: int = 1,
                            sc: int = 198,
                            alpha_multiplier: float = 0.5,
                            # 動的期間パラメータ
                            period_mode: str = 'fixed',
                            cycle_detector_type: str = 'hody_e',
                            lp_period: int = 13,
                            hp_period: int = 124,
                            cycle_part: float = 0.5,
                            max_cycle: int = 89,
                            min_cycle: int = 8,
                            max_output: int = 124,
                            min_output: int = 8
                           ) -> None:
        """
        ハイパーFRAMAを計算する
        
        Args:
            period: 期間（偶数である必要がある、デフォルト: 16）
            src_type: ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4')
            fc: Fast Constant（デフォルト: 1）
            sc: Slow Constant（デフォルト: 198）
            alpha_multiplier: アルファ調整係数（デフォルト: 0.5）
            period_mode: 期間モード ('fixed' または 'dynamic')
            cycle_detector_type: サイクル検出器タイプ
            lp_period: ローパスフィルター期間
            hp_period: ハイパスフィルター期間
            cycle_part: サイクル部分
            max_cycle: 最大サイクル期間
            min_cycle: 最小サイクル期間
            max_output: 最大出力値
            min_output: 最小出力値
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\nハイパーFRAMAを計算中...")
        
        # ハイパーFRAMAを計算
        self.hyper_frama = HyperFRAMA(
            period=period,
            src_type=src_type,
            fc=fc,
            sc=sc,
            alpha_multiplier=alpha_multiplier,
            period_mode=period_mode,
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            max_cycle=max_cycle,
            min_cycle=min_cycle,
            max_output=max_output,
            min_output=min_output
        )
        
        # ハイパーFRAMAの計算
        print("計算を実行します...")
        result = self.hyper_frama.calculate(self.data)
        
        # 結果の取得テスト
        frama_values = self.hyper_frama.get_frama_values()
        half_frama_values = self.hyper_frama.get_half_frama_values()
        fractal_dim = self.hyper_frama.get_fractal_dimension()
        alpha_values = self.hyper_frama.get_alpha()
        half_alpha_values = self.hyper_frama.get_half_alpha()
        
        print(f"計算完了 - FRAMA: {len(frama_values)}, Half-FRAMA: {len(half_frama_values)}")
        print(f"フラクタル次元: {len(fractal_dim)}, Alpha: {len(alpha_values)}, Half-Alpha: {len(half_alpha_values)}")
        
        # NaN値のチェック
        nan_count_frama = np.isnan(frama_values).sum()
        nan_count_half = np.isnan(half_frama_values).sum()
        nan_count_dim = np.isnan(fractal_dim).sum()
        print(f"NaN値 - FRAMA: {nan_count_frama}, Half-FRAMA: {nan_count_half}, フラクタル次元: {nan_count_dim}")
        
        print("ハイパーFRAMA計算完了")
            
    def plot(self, 
            title: str = "ハイパーFRAMA", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (14, 12),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとハイパーFRAMAを描画する
        
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
            
        if self.hyper_frama is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # ハイパーFRAMAの値を取得
        print("ハイパーFRAMAデータを取得中...")
        frama_values = self.hyper_frama.get_frama_values()
        half_frama_values = self.hyper_frama.get_half_frama_values()
        fractal_dim = self.hyper_frama.get_fractal_dimension()
        alpha_values = self.hyper_frama.get_alpha()
        half_alpha_values = self.hyper_frama.get_half_alpha()
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'frama': frama_values,
                'half_frama': half_frama_values,
                'fractal_dimension': fractal_dim,
                'alpha': alpha_values,
                'half_alpha': half_alpha_values
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        print(f"FRAMAデータ確認 - NaN: {df['frama'].isna().sum()}, Half-FRAMA NaN: {df['half_frama'].isna().sum()}")
        
        # クロスオーバーの検出（NaN値を適切に処理）
        df['frama_above_half'] = df['frama'] > df['half_frama']
        
        # トレンド方向の判定（FRAMAとHalf-FRAMAの関係）
        # NaN値がある場合はFalseとして扱う
        prev_frama = df['frama'].shift(1)
        prev_half_frama = df['half_frama'].shift(1)
        
        df['bullish_signal'] = (
            (df['frama'] > df['half_frama']) & 
            (prev_frama <= prev_half_frama) &
            df['frama'].notna() & df['half_frama'].notna() &
            prev_frama.notna() & prev_half_frama.notna()
        )
        
        df['bearish_signal'] = (
            (df['frama'] < df['half_frama']) & 
            (prev_frama >= prev_half_frama) &
            df['frama'].notna() & df['half_frama'].notna() &
            prev_frama.notna() & prev_half_frama.notna()
        )
        
        # mplfinanceでプロット用の設定
        # 1. メインチャート上のプロット
        main_plots = []
        
        # ハイパーFRAMAのプロット設定
        main_plots.append(mpf.make_addplot(df['frama'], color='blue', width=2.5, label='FRAMA'))
        main_plots.append(mpf.make_addplot(df['half_frama'], color='red', width=1.5, label='Adjusted FRAMA'))
        
        # クロスオーバーシグナルのプロット用データ準備
        # 全行でNaNを作成し、シグナルがある箇所のみFRAMA値を設定
        bullish_scatter = pd.Series(index=df.index, data=np.nan)
        bearish_scatter = pd.Series(index=df.index, data=np.nan)
        
        # シグナルがある箇所にFRAMA値を設定
        bullish_scatter.loc[df['bullish_signal']] = df.loc[df['bullish_signal'], 'frama']
        bearish_scatter.loc[df['bearish_signal']] = df.loc[df['bearish_signal'], 'frama']
        
        # シグナルが存在する場合のみプロットに追加
        if bullish_scatter.notna().any():
            main_plots.append(mpf.make_addplot(bullish_scatter, type='scatter', markersize=100, marker='^', color='green', alpha=0.8))
        if bearish_scatter.notna().any():
            main_plots.append(mpf.make_addplot(bearish_scatter, type='scatter', markersize=100, marker='v', color='red', alpha=0.8))
        
        # 2. オシレータープロット
        # フラクタル次元パネル
        fractal_panel = mpf.make_addplot(df['fractal_dimension'], panel=1, color='purple', width=1.2, 
                                        ylabel='Fractal Dimension', secondary_y=False, label='Fractal Dim')
        
        # アルファ値パネル
        alpha_panel = mpf.make_addplot(df['alpha'], panel=2, color='orange', width=1.2, 
                                      ylabel='Alpha Values', secondary_y=False, label='Alpha')
        alpha_half_panel = mpf.make_addplot(df['half_alpha'], panel=2, color='darkred', width=1.2, 
                                           secondary_y=False, label='Half Alpha')
        
        # トレンド強度パネル（FRAMAとHalf-FRAMAの差）
        df['trend_strength'] = (df['frama'] - df['half_frama']) / df['half_frama'] * 100
        strength_panel = mpf.make_addplot(df['trend_strength'], panel=3, color='navy', width=1.5, 
                                         ylabel='Trend Strength (%)', secondary_y=False, label='Trend Strength')
        
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
            kwargs['panel_ratios'] = (4, 1, 1, 1, 1)  # メイン:出来高:フラクタル次元:アルファ:トレンド強度
            # 出来高を表示する場合は、オシレーターのパネル番号を+1する
            fractal_panel = mpf.make_addplot(df['fractal_dimension'], panel=2, color='purple', width=1.2, 
                                            ylabel='Fractal Dimension', secondary_y=False, label='Fractal Dim')
            alpha_panel = mpf.make_addplot(df['alpha'], panel=3, color='orange', width=1.2, 
                                          ylabel='Alpha Values', secondary_y=False, label='Alpha')
            alpha_half_panel = mpf.make_addplot(df['half_alpha'], panel=3, color='darkred', width=1.2, 
                                               secondary_y=False, label='Half Alpha')
            strength_panel = mpf.make_addplot(df['trend_strength'], panel=4, color='navy', width=1.5, 
                                             ylabel='Trend Strength (%)', secondary_y=False, label='Trend Strength')
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (4, 1, 1, 1)  # メイン:フラクタル次元:アルファ:トレンド強度
        
        # すべてのプロットを結合
        all_plots = main_plots + [fractal_panel, alpha_panel, alpha_half_panel, strength_panel]
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加
        axes[0].legend(['FRAMA', 'Adjusted FRAMA', 'Bullish Signal', 'Bearish Signal'], 
                      loc='upper left')
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        if show_volume:
            # フラクタル次元パネル
            axes[2].axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Trend (1.0)')
            axes[2].axhline(y=1.5, color='black', linestyle='-', alpha=0.3)
            axes[2].axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='Range (2.0)')
            axes[2].legend(loc='upper right', fontsize=8)
            
            # アルファ値パネル
            axes[3].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
            axes[3].axhline(y=0.01, color='red', linestyle='--', alpha=0.5, label='Min Alpha')
            axes[3].axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Max Alpha')
            axes[3].legend(loc='upper right', fontsize=8)
            
            # トレンド強度パネル
            axes[4].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[4].axhline(y=5, color='green', linestyle='--', alpha=0.3)
            axes[4].axhline(y=-5, color='red', linestyle='--', alpha=0.3)
        else:
            # フラクタル次元パネル
            axes[1].axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Trend (1.0)')
            axes[1].axhline(y=1.5, color='black', linestyle='-', alpha=0.3)
            axes[1].axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='Range (2.0)')
            axes[1].legend(loc='upper right', fontsize=8)
            
            # アルファ値パネル
            axes[2].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
            axes[2].axhline(y=0.01, color='red', linestyle='--', alpha=0.5, label='Min Alpha')
            axes[2].axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Max Alpha')
            axes[2].legend(loc='upper right', fontsize=8)
            
            # トレンド強度パネル
            axes[3].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[3].axhline(y=5, color='green', linestyle='--', alpha=0.3)
            axes[3].axhline(y=-5, color='red', linestyle='--', alpha=0.3)
        
        # 統計情報の表示
        print(f"\n=== ハイパーFRAMA統計 ===")
        total_points = len(df.dropna())
        bullish_signals = int(df['bullish_signal'].sum())
        bearish_signals = int(df['bearish_signal'].sum())
        
        print(f"総データ点数: {total_points}")
        print(f"強気シグナル: {bullish_signals}")
        print(f"弱気シグナル: {bearish_signals}")
        
        # NaN値を除いた統計計算
        fractal_clean = df['fractal_dimension'].dropna()
        alpha_clean = df['alpha'].dropna()
        half_alpha_clean = df['half_alpha'].dropna()
        trend_strength_clean = df['trend_strength'].dropna()
        
        if len(fractal_clean) > 0:
            print(f"フラクタル次元 - 平均: {fractal_clean.mean():.3f}, 範囲: {fractal_clean.min():.3f} - {fractal_clean.max():.3f}")
        if len(alpha_clean) > 0:
            print(f"アルファ値 - 平均: {alpha_clean.mean():.3f}, 範囲: {alpha_clean.min():.3f} - {alpha_clean.max():.3f}")
        if len(half_alpha_clean) > 0:
            print(f"アルファ半分値 - 平均: {half_alpha_clean.mean():.3f}, 範囲: {half_alpha_clean.min():.3f} - {half_alpha_clean.max():.3f}")
        if len(trend_strength_clean) > 0:
            print(f"トレンド強度 - 平均: {trend_strength_clean.mean():.2f}%, 範囲: {trend_strength_clean.min():.2f}% - {trend_strength_clean.max():.2f}%")
        
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
    parser = argparse.ArgumentParser(description='ハイパーFRAMAの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--period', type=int, default=16, help='期間（偶数である必要がある）')
    parser.add_argument('--src-type', type=str, default='hl2', help='ソースタイプ')
    parser.add_argument('--fc', type=int, default=1, help='Fast Constant')
    parser.add_argument('--sc', type=int, default=198, help='Slow Constant')
    parser.add_argument('--alpha-mult', type=float, default=0.3, help='アルファ調整係数 (0.1-1.0)')
    parser.add_argument('--period-mode', type=str, default='fixed', help='期間モード (fixed または dynamic)')
    args = parser.parse_args()
    
    # チャートを作成
    chart = HyperFRAMAChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        period=args.period,
        src_type=args.src_type,
        fc=args.fc,
        sc=args.sc,
        alpha_multiplier=args.alpha_mult,
        period_mode=args.period_mode
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output
    )


if __name__ == "__main__":
    main()