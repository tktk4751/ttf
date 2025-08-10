#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import yaml
from typing import Optional, Tuple

# データ取得のための依存関係
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource

# インジケーター
from indicators.trend_filter.hyper_donchian import HyperDonchian
from indicators.smoother.frama import FRAMA


class HyperDonchianFRAMAChart:
    """
    HyperドンチャンとFRAMAを表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - Hyperドンチャンバンドとミッドライン（メインパネル上）
    - FRAMA（メインパネル上）
    - Hyperドンチャントレンド信号（別パネル）
    - FRAMAフラクタル次元（別パネル）
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.hyper_donchian = None
        self.frama = None
        self.hyper_donchian_result = None
        self.frama_result = None
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

    def calculate_indicators(self) -> None:
        """
        HyperドンチャンとFRAMAを計算する
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\\nインジケーターを計算中...")
        
        # Hyperドンチャンの計算
        print("Hyperドンチャンを計算中...")
        self.hyper_donchian = HyperDonchian(
            period=60,
            src_type='hlc3'
        )
        
        self.hyper_donchian_result = self.hyper_donchian.calculate(self.data)
        print(f"Hyperドンチャン計算完了: 有効値数 {np.sum(~np.isnan(self.hyper_donchian_result.values))}/{len(self.hyper_donchian_result.values)}")
        
        # FRAMAの計算
        print("FRAMAを計算中...")
        self.frama = FRAMA(
            period=16,
            src_type='hlc3',
            fc=2,
            sc=198,
            period_mode='dynamic'  # 動的期間適応
        )
        
        self.frama_result = self.frama.calculate(self.data)
        print(f"FRAMA計算完了: 有効値数 {np.sum(~np.isnan(self.frama_result.values))}/{len(self.frama_result.values)}")
            
    def plot(self, 
            title: str = "HyperドンチャンFRAMA チャート", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (16, 16),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとインジケーターを描画する
        
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
            
        if self.hyper_donchian is None or self.frama is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # インジケーターの値を取得
        print("インジケーターデータを取得中...")
        
        # HyperドンチャンFRAMA関連
        hyper_donchian_midline = self.hyper_donchian_result.values
        hyper_donchian_upper = self.hyper_donchian_result.upper_band
        hyper_donchian_lower = self.hyper_donchian_result.lower_band
        
        # FRAMA関連
        frama_values = self.frama_result.values
        frama_fractal_dim = self.frama_result.fractal_dimension
        frama_alpha = self.frama_result.alpha
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'hyper_donchian_midline': hyper_donchian_midline,
                'hyper_donchian_upper': hyper_donchian_upper,
                'hyper_donchian_lower': hyper_donchian_lower,
                'frama': frama_values,
                'frama_fractal_dim': frama_fractal_dim,
                'frama_alpha': frama_alpha
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        
        # 価格とHyperドンチャンミッドラインの関係でトレンド信号を作成
        # NaN値をマスク
        valid_mask = ~(pd.isna(df['close']) | pd.isna(df['hyper_donchian_midline']))
        
        df['trend_up'] = np.nan
        df['trend_down'] = np.nan  
        df['trend_neutral'] = np.nan
        
        # 有効なデータのみで信号を計算
        if valid_mask.any():
            df.loc[valid_mask & (df['close'] > df['hyper_donchian_midline']), 'trend_up'] = 1
            df.loc[valid_mask & (df['close'] < df['hyper_donchian_midline']), 'trend_down'] = -1
            df.loc[valid_mask & (df['close'] == df['hyper_donchian_midline']), 'trend_neutral'] = 0
        
        # フラクタル次元の色分け（1に近い=緑、2に近い=赤）
        df['fractal_smooth'] = np.where(df['frama_fractal_dim'] <= 1.5, df['frama_fractal_dim'], np.nan)
        df['fractal_choppy'] = np.where(df['frama_fractal_dim'] > 1.5, df['frama_fractal_dim'], np.nan)
        
        # mplfinanceでプロット用の設定
        # 追加パネルのプロット設定
        additional_plots = []
        
        if show_volume:
            # 出来高あり: パネル0=メイン、パネル1=出来高、パネル2=フラクタル次元
            
            # メインパネルにHyperドンチャンバンドとFRAMA（パネル0）
            additional_plots.append(
                mpf.make_addplot(df['hyper_donchian_upper'], panel=0, color='red', width=1.5, 
                               alpha=0.7, label='HyperDonchian Upper', linestyle='--')
            )
            additional_plots.append(
                mpf.make_addplot(df['hyper_donchian_midline'], panel=0, color='blue', width=2, 
                               label='HyperDonchian Midline')
            )
            additional_plots.append(
                mpf.make_addplot(df['hyper_donchian_lower'], panel=0, color='green', width=1.5, 
                               alpha=0.7, label='HyperDonchian Lower', linestyle='--')
            )
            additional_plots.append(
                mpf.make_addplot(df['frama'], panel=0, color='purple', width=2, 
                               label='FRAMA', alpha=0.8)
            )
            
            # フラクタル次元（パネル2）
            additional_plots.append(
                mpf.make_addplot(df['fractal_smooth'], panel=2, color='green', width=2, 
                               ylabel='FRAMA Fractal D', secondary_y=False, label='Smooth (D≤1.5)', type='line')
            )
            additional_plots.append(
                mpf.make_addplot(df['fractal_choppy'], panel=2, color='red', width=2, 
                               label='Choppy (D>1.5)', type='line')
            )
            
            panel_ratios = (5, 1, 1.5)  # メイン:出来高:フラクタル次元
            
        else:
            # 出来高なし: パネル0=メイン、パネル1=トレンド信号、パネル2=フラクタル次元
            
            # メインパネルにHyperドンチャンバンドとFRAMA（パネル0）
            additional_plots.append(
                mpf.make_addplot(df['hyper_donchian_upper'], panel=0, color='red', width=1.5, 
                               alpha=0.7, label='HyperDonchian Upper', linestyle='--')
            )
            additional_plots.append(
                mpf.make_addplot(df['hyper_donchian_midline'], panel=0, color='blue', width=2, 
                               label='HyperDonchian Midline')
            )
            additional_plots.append(
                mpf.make_addplot(df['hyper_donchian_lower'], panel=0, color='green', width=1.5, 
                               alpha=0.7, label='HyperDonchian Lower', linestyle='--')
            )
            additional_plots.append(
                mpf.make_addplot(df['frama'], panel=0, color='purple', width=2, 
                               label='FRAMA', alpha=0.8)
            )
            
            # Hyperドンチャントレンド信号（パネル1）
            additional_plots.append(
                mpf.make_addplot(df['trend_up'], panel=1, color='green', width=3, 
                               ylabel='HyperDonchian Trend', secondary_y=False, label='Uptrend', type='line')
            )
            additional_plots.append(
                mpf.make_addplot(df['trend_down'], panel=1, color='red', width=3, 
                               label='Downtrend', type='line')
            )
            additional_plots.append(
                mpf.make_addplot(df['trend_neutral'], panel=1, color='gray', width=2, 
                               label='Neutral', type='line', alpha=0.5)
            )
            
            # フラクタル次元（パネル2）
            additional_plots.append(
                mpf.make_addplot(df['fractal_smooth'], panel=2, color='green', width=2, 
                               ylabel='FRAMA Fractal D', secondary_y=False, label='Smooth (D≤1.5)', type='line')
            )
            additional_plots.append(
                mpf.make_addplot(df['fractal_choppy'], panel=2, color='red', width=2, 
                               label='Choppy (D>1.5)', type='line')
            )
            
            panel_ratios = (5, 1, 1.5)  # メイン:トレンド:フラクタル次元
        
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
            # フラクタル次元パネル（パネル2）に参照線
            axes[2].axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect Trend')
            axes[2].axhline(y=1.5, color='orange', linestyle='-', alpha=0.7, label='Threshold')
            axes[2].axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='Random Walk')
            axes[2].set_ylim(0.8, 2.2)
        else:
            # フラクタル次元パネル（パネル2）に参照線
            axes[2].axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect Trend')
            axes[2].axhline(y=1.5, color='orange', linestyle='-', alpha=0.7, label='Threshold')
            axes[2].axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='Random Walk')
            axes[2].set_ylim(0.8, 2.2)
        
        # 統計情報の表示
        print(f"\\n=== インジケーター統計 ===")
        
        # HyperドンチャンFRAMA統計
        valid_hyper_donchian = df['hyper_donchian_midline'].dropna()
        valid_trend_up = df['trend_up'].dropna()
        valid_trend_down = df['trend_down'].dropna()
        
        if len(valid_hyper_donchian) > 0:
            print(f"HyperドンチャンFRAMAミッドライン範囲: {valid_hyper_donchian.min():.2f} - {valid_hyper_donchian.max():.2f}")
            print(f"HyperドンチャンFRAMAミッドライン平均: {valid_hyper_donchian.mean():.2f}")
            
            total_trend_signals = len(valid_trend_up) + len(valid_trend_down)
            if total_trend_signals > 0:
                print(f"価格ポジション信号 - 総数: {total_trend_signals}")
                print(f"  ミッドライン上: {len(valid_trend_up)} ({len(valid_trend_up)/total_trend_signals*100:.1f}%)")
                print(f"  ミッドライン下: {len(valid_trend_down)} ({len(valid_trend_down)/total_trend_signals*100:.1f}%)")
        
        # FRAMA統計
        valid_frama = df['frama'].dropna()
        valid_fractal = df['frama_fractal_dim'].dropna()
        
        if len(valid_frama) > 0:
            print(f"FRAMA範囲: {valid_frama.min():.2f} - {valid_frama.max():.2f}")
            print(f"FRAMA平均: {valid_frama.mean():.2f}")
            
        if len(valid_fractal) > 0:
            smooth_fractal = len(valid_fractal[valid_fractal <= 1.5])
            choppy_fractal = len(valid_fractal[valid_fractal > 1.5])
            
            print(f"フラクタル次元範囲: {valid_fractal.min():.3f} - {valid_fractal.max():.3f}")
            print(f"フラクタル次元平均: {valid_fractal.mean():.3f}")
            print(f"スムーズ期間 (D≤1.5): {smooth_fractal} ({smooth_fractal/len(valid_fractal)*100:.1f}%)")
            print(f"チョッピー期間 (D>1.5): {choppy_fractal} ({choppy_fractal/len(valid_fractal)*100:.1f}%)")
        
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
    parser = argparse.ArgumentParser(description='HyperドンチャンFRAMAチャートの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--bars', '-b', type=int, default=500, help='最大データ数 (デフォルト: 500)')
    args = parser.parse_args()
    
    # チャートを作成
    chart = HyperDonchianFRAMAChart()
    chart.load_data_from_config(args.config, max_bars=args.bars)
    chart.calculate_indicators()
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output
    )


if __name__ == "__main__":
    main()