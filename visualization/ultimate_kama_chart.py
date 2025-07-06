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
from indicators.ultimate_kama import UltimateKAMA


class UltimateKAMAChart:
    """
    UltimateKAMAを表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - UltimateKAMAの各段階の値（UKF、スムーサー、KAMA、ゼロラグEMA）
    - アルティメットER値
    - トレンド方向のカラー表示
    - 超高精度AI風トレンド判定結果
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.ultimate_kama = None
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
                            # KAMAパラメータ
                            kama_fast: int = 2,
                            kama_slow: int = 30,
                            # ゼロラグEMAパラメータ
                            zero_lag_period: int = 21,
                            # アルティメットスムーサーパラメータ
                            smoother_period: float = 13.0,
                            # ソースタイプ
                            src_type: str = 'ukf_hlc3',
                            # トレンド検出パラメータ
                            trend_window: int = 21,
                            slope_index: int = 1,
                            range_threshold: float = 0.005
                           ) -> None:
        """
        UltimateKAMAを計算する
        
        Args:
            kama_fast: KAMAのfast期間
            kama_slow: KAMAのslow期間
            zero_lag_period: ゼロラグEMA期間
            smoother_period: アルティメットスムーサー期間
            src_type: プライスソース
            trend_window: トレンド検出ウィンドウ
            slope_index: トレンド判定期間
            range_threshold: range判定の閾値
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\nUltimateKAMAを計算中...")
        
        # UltimateKAMAを計算
        self.ultimate_kama = UltimateKAMA(
            kama_fast=kama_fast,
            kama_slow=kama_slow,
            zero_lag_period=zero_lag_period,
            smoother_period=smoother_period,
            src_type=src_type,
            trend_window=trend_window,
            slope_index=slope_index,
            range_threshold=range_threshold
        )
        
        # UltimateKAMAの計算
        print("計算を実行します...")
        result = self.ultimate_kama.calculate(self.data)
        
        # 各段階の値を取得してテスト
        ukf_values = self.ultimate_kama.get_ukf_values()
        smooth_values = self.ultimate_kama.get_ultimate_smooth_values()
        kama_values = self.ultimate_kama.get_kama_values()
        zero_lag_values = self.ultimate_kama.get_zero_lag_values()
        er_values = self.ultimate_kama.get_er_values()
        trend_signals = self.ultimate_kama.get_trend_signals()
        
        print(f"計算完了 - UKF: {len(ukf_values)}, スムーサー: {len(smooth_values)}, KAMA: {len(kama_values)}")
        print(f"ゼロラグ: {len(zero_lag_values)}, ER: {len(er_values)}, トレンド: {len(trend_signals)}")
        
        # NaN値のチェック
        print(f"NaN値 - UKF: {np.isnan(ukf_values).sum()}, スムーサー: {np.isnan(smooth_values).sum()}")
        print(f"KAMA: {np.isnan(kama_values).sum()}, ゼロラグ: {np.isnan(zero_lag_values).sum()}")
        print(f"ER: {np.isnan(er_values).sum()}")
        
        # トレンド統計
        trend_count = (trend_signals != 0).sum()
        print(f"トレンド値 - 有効: {trend_count}, 上昇: {(trend_signals == 1).sum()}, 下降: {(trend_signals == -1).sum()}")
        
        # ノイズ除去統計
        noise_stats = self.ultimate_kama.get_noise_reduction_stats()
        print(f"ノイズ除去効果: {noise_stats.get('noise_reduction_percentage', 0):.1f}%")
        
        print("UltimateKAMA計算完了")
            
    def plot(self, 
            title: str = "UltimateKAMA", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (16, 10),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとUltimateKAMAを描画する
        
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
            
        if self.ultimate_kama is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # UltimateKAMAの値を取得
        print("UltimateKAMAデータを取得中...")
        ultimate_kama_values = self.ultimate_kama.get_values()
        trend_signals = self.ultimate_kama.get_trend_signals()
        current_trend = self.ultimate_kama.get_current_trend()
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'ultimate_kama': ultimate_kama_values,
                'trend_signals': trend_signals
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        print(f"現在のトレンド: {current_trend}")
        
        # トレンド方向に基づくUltimateKAMAの色分け
        df['ultimate_kama_uptrend'] = np.where(df['trend_signals'] == 1, df['ultimate_kama'], np.nan)
        df['ultimate_kama_downtrend'] = np.where(df['trend_signals'] == -1, df['ultimate_kama'], np.nan)
        df['ultimate_kama_range'] = np.where(df['trend_signals'] == 0, df['ultimate_kama'], np.nan)
        
        # mplfinanceでプロット用の設定
        main_plots = []
        
        # UltimateKAMAのプロット設定（トレンド方向で色分け）
        main_plots.append(mpf.make_addplot(df['ultimate_kama_uptrend'], color='green', width=2, label='UltimateKAMA (Up)'))
        main_plots.append(mpf.make_addplot(df['ultimate_kama_downtrend'], color='red', width=2, label='UltimateKAMA (Down)'))
        main_plots.append(mpf.make_addplot(df['ultimate_kama_range'], color='gray', width=1.5, alpha=0.6, label='UltimateKAMA (Range)'))
        
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
        
        # 出来高の設定
        if show_volume:
            kwargs['volume'] = True
            kwargs['panel_ratios'] = (4, 1)  # メイン:出来高
        else:
            kwargs['volume'] = False
        
        kwargs['addplot'] = main_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加
        axes[0].legend(['UltimateKAMA (Up)', 'UltimateKAMA (Down)', 'UltimateKAMA (Range)'], 
                      loc='upper left')
        
        self.fig = fig
        self.axes = axes
        
        # 統計情報の表示
        print(f"\n=== UltimateKAMA統計 ===")
        total_points = len(df[df['trend_signals'] != 0])
        uptrend_points = len(df[df['trend_signals'] == 1])
        downtrend_points = len(df[df['trend_signals'] == -1])
        range_points = len(df[df['trend_signals'] == 0])
        
        print(f"総データ点数: {len(df)}")
        print(f"上昇トレンド: {uptrend_points} ({uptrend_points/len(df)*100:.1f}%)")
        print(f"下降トレンド: {downtrend_points} ({downtrend_points/len(df)*100:.1f}%)")
        print(f"レンジ相場: {range_points} ({range_points/len(df)*100:.1f}%)")
        
        # ノイズ除去統計
        noise_stats = self.ultimate_kama.get_noise_reduction_stats()
        print(f"ノイズ除去効果: {noise_stats.get('noise_reduction_percentage', 0):.1f}%")
        print(f"平滑化効果: {noise_stats.get('smoothing_effectiveness', 0):.1f}%")
        
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
    parser = argparse.ArgumentParser(description='UltimateKAMAの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--kama-fast', type=int, default=2, help='KAMA fast期間')
    parser.add_argument('--kama-slow', type=int, default=30, help='KAMA slow期間')
    parser.add_argument('--zero-lag-period', type=int, default=21, help='ゼロラグEMA期間')
    parser.add_argument('--smoother-period', type=float, default=13.0, help='アルティメットスムーサー期間')
    parser.add_argument('--trend-window', type=int, default=21, help='トレンド検出ウィンドウ')
    parser.add_argument('--range-threshold', type=float, default=0.005, help='レンジ判定閾値')
    args = parser.parse_args()
    
    # チャートを作成
    chart = UltimateKAMAChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        kama_fast=args.kama_fast,
        kama_slow=args.kama_slow,
        zero_lag_period=args.zero_lag_period,
        smoother_period=args.smoother_period,
        trend_window=args.trend_window,
        range_threshold=args.range_threshold
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output
    )


if __name__ == "__main__":
    main() 