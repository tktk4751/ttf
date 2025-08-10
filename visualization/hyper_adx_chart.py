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

# Hyper ADX インジケーター
from indicators.trend_filter.hyper_adx import HyperADX


class HyperADXChart:
    """
    Hyper ADX（統合型Average Directional Index）を表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - Hyper ADX値とミッドライン（メインパネル下の別パネル）
    - 生DX値（Raw DX）（別パネル）
    - トレンド信号（1=緑、-1=赤）（別パネル）
    - +DI、-DI値（別パネル）
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.hyper_adx = None
        self.hyper_adx_result = None
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
        print("\nデータを読み込み・処理中...")
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
        Hyper ADXを計算する
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\nHyper ADXを計算中...")
        
        # Hyper ADXの計算（基本設定で依存関係の問題を回避）
        self.hyper_adx = HyperADX(
            period=5,
            midline_period=100,
            # 高度な機能を有効化（ハイパーERと同じフロー）
            use_kalman_filter=True,
            kalman_filter_type='unscented',
            use_roofing_filter=True,
            roofing_hp_cutoff=55.0,
            roofing_ss_band_edge=10.0,
            smoother_type='laguerre',
            smoother_period=21,
            use_dynamic_period=True,
            detector_type='dft_dominant',
            # 二次平滑化オプション（テスト用）
            use_secondary_smoothing=True,
            secondary_smoother_type='super_smoother',
            secondary_smoother_period=8
        )
        
        # インジケーターの計算
        print("計算を実行します...")
        self.hyper_adx_result = self.hyper_adx.calculate(self.data)
        
        print("Hyper ADXの計算完了")
        print(f"有効値数: {np.sum(~np.isnan(self.hyper_adx_result.values))}/{len(self.hyper_adx_result.values)}")
            
    def plot(self, 
            title: str = "Hyper ADX チャート", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (14, 16),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとHyper ADXを描画する
        
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
            
        if self.hyper_adx is None or self.hyper_adx_result is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # Hyper ADXの値を取得
        print("インジケーターデータを取得中...")
        hyper_adx_values = self.hyper_adx_result.values
        raw_dx_values = self.hyper_adx_result.raw_dx
        midline_values = self.hyper_adx_result.midline
        trend_signals = self.hyper_adx_result.trend_signal
        plus_di_values = self.hyper_adx_result.plus_di
        minus_di_values = self.hyper_adx_result.minus_di
        secondary_smoothed_values = self.hyper_adx_result.secondary_smoothed
        
        # 全データの時系列データフレームを作成
        chart_data = {
            'hyper_adx': hyper_adx_values,
            'raw_dx': raw_dx_values,
            'midline': midline_values,
            'trend_signal': trend_signals,
            'plus_di': plus_di_values,
            'minus_di': minus_di_values
        }
        
        # 二次平滑化された値がある場合は追加
        if secondary_smoothed_values is not None:
            chart_data['secondary_smoothed'] = secondary_smoothed_values
        
        full_df = pd.DataFrame(
            index=self.data.index,
            data=chart_data
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        
        # トレンド信号に基づく色分け（1=緑、-1=赤）
        df['trend_green'] = np.where(df['trend_signal'] == 1, 1, np.nan)
        df['trend_red'] = np.where(df['trend_signal'] == -1, -1, np.nan)
        
        # mplfinanceでプロット用の設定
        # 追加パネルのプロット設定
        additional_plots = []
        
        if show_volume:
            # 出来高あり: パネル0=メイン、パネル1=出来高、パネル2=Hyper ADX、パネル3=Raw DX、パネル4=DI値、パネル5=トレンド信号
            
            # Hyper ADX値とミッドライン（パネル2）
            additional_plots.append(
                mpf.make_addplot(df['hyper_adx'], panel=2, color='blue', width=2, 
                               ylabel='Hyper ADX (0-1)', secondary_y=False, label='Hyper ADX')
            )
            # 二次平滑化された値がある場合は追加
            if 'secondary_smoothed' in df.columns:
                additional_plots.append(
                    mpf.make_addplot(df['secondary_smoothed'], panel=2, color='cyan', width=2, 
                                   linestyle='-', alpha=0.8, label='Secondary Smoothed')
                )
            additional_plots.append(
                mpf.make_addplot(df['midline'], panel=2, color='orange', width=1.5, 
                               linestyle='--', alpha=0.7, label='Midline')
            )
            
            # Raw DX値（パネル3）
            additional_plots.append(
                mpf.make_addplot(df['raw_dx'], panel=3, color='purple', width=1.5, 
                               ylabel='Raw DX (0-1)', secondary_y=False, label='Raw DX')
            )
            
            # +DI、-DI値（パネル4）
            additional_plots.append(
                mpf.make_addplot(df['plus_di'], panel=4, color='green', width=1.5, 
                               ylabel='Directional Index', secondary_y=False, label='+DI')
            )
            additional_plots.append(
                mpf.make_addplot(df['minus_di'], panel=4, color='red', width=1.5, 
                               label='-DI')
            )
            
            # トレンド信号（パネル5）
            additional_plots.append(
                mpf.make_addplot(df['trend_green'], panel=5, color='green', width=3, 
                               ylabel='Trend Signal', secondary_y=False, label='Trend', type='line')
            )
            additional_plots.append(
                mpf.make_addplot(df['trend_red'], panel=5, color='red', width=3, 
                               label='Range', type='line')
            )
            
            panel_ratios = (4, 1, 2, 1.5, 1.5, 1)  # メイン:出来高:Hyper ADX:Raw DX:DI:トレンド信号
            
        else:
            # 出来高なし: パネル0=メイン、パネル1=Hyper ADX、パネル2=Raw DX、パネル3=DI値、パネル4=トレンド信号
            
            # Hyper ADX値とミッドライン（パネル1）
            additional_plots.append(
                mpf.make_addplot(df['hyper_adx'], panel=1, color='blue', width=2, 
                               ylabel='Hyper ADX (0-1)', secondary_y=False, label='Hyper ADX')
            )
            # 二次平滑化された値がある場合は追加
            if 'secondary_smoothed' in df.columns:
                additional_plots.append(
                    mpf.make_addplot(df['secondary_smoothed'], panel=1, color='cyan', width=2, 
                                   linestyle='-', alpha=0.8, label='Secondary Smoothed')
                )
            additional_plots.append(
                mpf.make_addplot(df['midline'], panel=1, color='orange', width=1.5, 
                               linestyle='--', alpha=0.7, label='Midline')
            )
            
            # Raw DX値（パネル2）
            additional_plots.append(
                mpf.make_addplot(df['raw_dx'], panel=2, color='purple', width=1.5, 
                               ylabel='Raw DX (0-1)', secondary_y=False, label='Raw DX')
            )
            
            # +DI、-DI値（パネル3）
            additional_plots.append(
                mpf.make_addplot(df['plus_di'], panel=3, color='green', width=1.5, 
                               ylabel='Directional Index', secondary_y=False, label='+DI')
            )
            additional_plots.append(
                mpf.make_addplot(df['minus_di'], panel=3, color='red', width=1.5, 
                               label='-DI')
            )
            
            # トレンド信号（パネル4）
            additional_plots.append(
                mpf.make_addplot(df['trend_green'], panel=4, color='green', width=3, 
                               ylabel='Trend Signal', secondary_y=False, label='Trend', type='line')
            )
            additional_plots.append(
                mpf.make_addplot(df['trend_red'], panel=4, color='red', width=3, 
                               label='Range', type='line')
            )
            
            panel_ratios = (4, 2, 1.5, 1.5, 1)  # メイン:Hyper ADX:Raw DX:DI:トレンド信号
        
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
            # Hyper ADXパネル（パネル2）に参照線
            axes[2].axhline(y=0.5, color='gray', linestyle='-', alpha=0.5, linewidth=1)
            axes[2].axhline(y=0.25, color='red', linestyle='--', alpha=0.5)
            axes[2].axhline(y=0.75, color='green', linestyle='--', alpha=0.5)
            axes[2].set_ylim(0, 1)
            
            # Raw DXパネル（パネル3）に参照線
            axes[3].axhline(y=0.5, color='gray', linestyle='-', alpha=0.5, linewidth=1)
            axes[3].set_ylim(0, 1)
            
            # DIパネル（パネル4）に参照線
            axes[4].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            # トレンド信号パネル（パネル5）に参照線
            axes[5].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[5].axhline(y=1, color='green', linestyle='--', alpha=0.3)
            axes[5].axhline(y=-1, color='red', linestyle='--', alpha=0.3)
            axes[5].set_ylim(-1.5, 1.5)
        else:
            # Hyper ADXパネル（パネル1）に参照線
            axes[1].axhline(y=0.5, color='gray', linestyle='-', alpha=0.5, linewidth=1)
            axes[1].axhline(y=0.25, color='red', linestyle='--', alpha=0.5)
            axes[1].axhline(y=0.75, color='green', linestyle='--', alpha=0.5)
            axes[1].set_ylim(0, 1)
            
            # Raw DXパネル（パネル2）に参照線
            axes[2].axhline(y=0.5, color='gray', linestyle='-', alpha=0.5, linewidth=1)
            axes[2].set_ylim(0, 1)
            
            # DIパネル（パネル3）に参照線
            axes[3].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            # トレンド信号パネル（パネル4）に参照線
            axes[4].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[4].axhline(y=1, color='green', linestyle='--', alpha=0.3)
            axes[4].axhline(y=-1, color='red', linestyle='--', alpha=0.3)
            axes[4].set_ylim(-1.5, 1.5)
        
        # 統計情報の表示
        print(f"\n=== Hyper ADX統計 ===")
        valid_hyper_adx = df['hyper_adx'].dropna()
        valid_raw_dx = df['raw_dx'].dropna()
        valid_trend = df['trend_signal'].dropna()
        valid_plus_di = df['plus_di'].dropna()
        valid_minus_di = df['minus_di'].dropna()
        
        if len(valid_hyper_adx) > 0:
            print(f"Hyper ADX値範囲: {valid_hyper_adx.min():.3f} - {valid_hyper_adx.max():.3f}")
            print(f"Hyper ADX平均: {valid_hyper_adx.mean():.3f}")
            
        # 二次平滑化値の統計がある場合は表示
        if 'secondary_smoothed' in df.columns:
            valid_secondary = df['secondary_smoothed'].dropna()
            if len(valid_secondary) > 0:
                print(f"二次平滑化ADX値範囲: {valid_secondary.min():.3f} - {valid_secondary.max():.3f}")
                print(f"二次平滑化ADX平均: {valid_secondary.mean():.3f}")
                
        if len(valid_raw_dx) > 0:
            print(f"Raw DX値範囲: {valid_raw_dx.min():.3f} - {valid_raw_dx.max():.3f}")
            print(f"Raw DX平均: {valid_raw_dx.mean():.3f}")
            
        if len(valid_plus_di) > 0:
            print(f"+DI値範囲: {valid_plus_di.min():.4f} - {valid_plus_di.max():.4f}")
            print(f"+DI平均: {valid_plus_di.mean():.4f}")
            
        if len(valid_minus_di) > 0:
            print(f"-DI値範囲: {valid_minus_di.min():.4f} - {valid_minus_di.max():.4f}")
            print(f"-DI平均: {valid_minus_di.mean():.4f}")
            
        if len(valid_trend) > 0:
            total_signals = len(valid_trend)
            trend_signals = len(valid_trend[valid_trend == 1])
            range_signals = len(valid_trend[valid_trend == -1])
            
            print(f"総信号数: {total_signals}")
            print(f"トレンド信号: {trend_signals} ({trend_signals/total_signals*100:.1f}%)")
            print(f"レンジ信号: {range_signals} ({range_signals/total_signals*100:.1f}%)")
        
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
    parser = argparse.ArgumentParser(description='Hyper ADXチャートの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--bars', '-b', type=int, default=500, help='最大データ数 (デフォルト: 500)')
    args = parser.parse_args()
    
    # チャートを作成
    chart = HyperADXChart()
    chart.load_data_from_config(args.config, max_bars=args.bars)
    chart.calculate_indicators()
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output
    )


if __name__ == "__main__":
    main()