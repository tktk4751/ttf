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
from indicators.z_adaptive_channel import ZAdaptiveChannel


class ZAdaptiveChannelChart:
    """
    Zアダプティブチャネルを表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - Zアダプティブチャネルの中心線・上限バンド・下限バンド
    - サイクル効率比（CER）
    - 動的乗数値
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.z_adaptive_channel = None
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
                            # ZAdaptiveChannel パラメータ
                            max_max_multiplier: float = 8.0,
                            min_max_multiplier: float = 3.0,
                            max_min_multiplier: float = 1.5,
                            min_min_multiplier: float = 0.5,
                            src_type: str = 'hlc3',
                            # CERパラメータ
                            detector_type: str = 'phac_e',
                            cycle_part: float = 0.5,
                            lp_period: int = 5,
                            hp_period: int = 55,
                            max_cycle: int = 55,
                            min_cycle: int = 5,
                            max_output: int = 34,
                            min_output: int = 5,
                            use_kalman_filter: bool = False,
                            # ZAdaptiveMA用パラメータ
                            fast_period: int = 2,
                            slow_period: int = 144
                           ) -> None:
        """
        Zアダプティブチャネルを計算する
        
        Args:
            max_max_multiplier: 最大乗数の最大値（動的乗数使用時）
            min_max_multiplier: 最大乗数の最小値（動的乗数使用時）
            max_min_multiplier: 最小乗数の最大値（動的乗数使用時）
            min_min_multiplier: 最小乗数の最小値（動的乗数使用時）
            src_type: ソースタイプ
            detector_type: CER用ドミナントサイクル検出器タイプ
            cycle_part: CER用サイクル部分
            lp_period: CER用ローパスフィルター期間
            hp_period: CER用ハイパスフィルター期間
            max_cycle: CER用最大サイクル期間
            min_cycle: CER用最小サイクル期間
            max_output: CER用最大出力値
            min_output: CER用最小出力値
            use_kalman_filter: CER用カルマンフィルター使用有無
            fast_period: 速い移動平均の期間（固定値）
            slow_period: 遅い移動平均の期間（固定値）
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\nZアダプティブチャネルを計算中...")
        
        # Zアダプティブチャネルを計算
        self.z_adaptive_channel = ZAdaptiveChannel(
            max_max_multiplier=max_max_multiplier,
            min_max_multiplier=min_max_multiplier,
            max_min_multiplier=max_min_multiplier,
            min_min_multiplier=min_min_multiplier,
            src_type=src_type,
            detector_type=detector_type,
            cycle_part=cycle_part,
            lp_period=lp_period,
            hp_period=hp_period,
            max_cycle=max_cycle,
            min_cycle=min_cycle,
            max_output=max_output,
            min_output=min_output,
            use_kalman_filter=use_kalman_filter,
            fast_period=fast_period,
            slow_period=slow_period
        )
        
        # ZAdaptiveChannelの計算
        print("計算を実行します...")
        self.z_adaptive_channel.calculate(self.data)
        
        # バンドの取得テスト
        middle, upper, lower = self.z_adaptive_channel.get_bands()
        print(f"バンド計算完了 - 中心線: {len(middle)}, 上限: {len(upper)}, 下限: {len(lower)}")
        
        # NaN値のチェック
        nan_count_middle = np.isnan(middle).sum()
        nan_count_upper = np.isnan(upper).sum()
        nan_count_lower = np.isnan(lower).sum()
        print(f"NaN値 - 中心線: {nan_count_middle}, 上限: {nan_count_upper}, 下限: {nan_count_lower}")
        
        print("Zアダプティブチャネル計算完了")
            
    def plot(self, 
            title: str = "Zアダプティブチャネル", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (14, 10),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとZアダプティブチャネルを描画する
        
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
            
        if self.z_adaptive_channel is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # Zアダプティブチャネルの値を取得
        # 注意: 先にインジケーター全体のデータを取得してから絞り込み用データに追加
        print("バンドデータを取得中...")
        middle, upper, lower = self.z_adaptive_channel.get_bands()
        cer = self.z_adaptive_channel.get_cycle_er()
        dynamic_mult = self.z_adaptive_channel.get_dynamic_multiplier()
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'channel_middle': middle,
                'channel_upper': upper,
                'channel_lower': lower,
                'cer': cer,
                'dynamic_mult': dynamic_mult
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        print(f"バンドデータ確認 - 中心線NaN: {df['channel_middle'].isna().sum()}, 上限NaN: {df['channel_upper'].isna().sum()}")
        
        # NaN値を含む行を出力（最初の5行のみ）
        nan_rows = df[df['channel_upper'].isna() | df['channel_middle'].isna() | df['channel_lower'].isna()]
        if not nan_rows.empty:
            print(f"NaN値を含む行: {len(nan_rows)}行")
            if len(nan_rows) > 0:
                print(nan_rows.head())
        
        # mplfinanceでプロット用の設定
        # 1. メインチャート上のプロット
        main_plots = []
        
        # Zアダプティブチャネルのプロット設定
        main_plots.append(mpf.make_addplot(df['channel_middle'], color='gray', width=1.5, label='ZA Middle'))
        main_plots.append(mpf.make_addplot(df['channel_upper'], color='green', width=1, label='ZA Upper'))
        main_plots.append(mpf.make_addplot(df['channel_lower'], color='red', width=1, label='ZA Lower'))
        
        # 2. オシレータープロット
        # CERとパネル配置を設定
        cer_panel = mpf.make_addplot(df['cer'], panel=1, color='purple', width=1.2, 
                                     ylabel='CER', secondary_y=False, label='CER')
        
        mult_panel = mpf.make_addplot(df['dynamic_mult'], panel=2, color='blue', width=1.2, 
                                     ylabel='Dynamic Mult', secondary_y=False, label='Multiplier')
        
        # mplfinanceの設定
        # パネル数に応じて設定を調整
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
            kwargs['panel_ratios'] = (4, 1, 1, 1)  # メイン:出来高:CER:乗数
            # 出来高を表示する場合は、オシレーターのパネル番号を+1する
            cer_panel = mpf.make_addplot(df['cer'], panel=2, color='purple', width=1.2, 
                                         ylabel='CER', secondary_y=False, label='CER')
            mult_panel = mpf.make_addplot(df['dynamic_mult'], panel=3, color='blue', width=1.2, 
                                         ylabel='Dynamic Mult', secondary_y=False, label='Multiplier')
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (4, 1, 1)  # メイン:CER:乗数
        
        # すべてのプロットを結合
        all_plots = main_plots + [cer_panel, mult_panel]
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加
        axes[0].legend(['ZA Middle', 'ZA Upper', 'ZA Lower'], loc='upper left')
        
        self.fig = fig
        self.axes = axes
        
        # CERパネルに参照線を追加
        if show_volume:
            # CERパネル
            axes[2].axhline(y=0.7, color='black', linestyle='--', alpha=0.5)
            axes[2].axhline(y=0.0, color='black', linestyle='-', alpha=0.3)
            axes[2].axhline(y=-0.7, color='black', linestyle='--', alpha=0.5)
            
            # 動的乗数パネル
            mult_mean = df['dynamic_mult'].mean()
            axes[3].axhline(y=mult_mean, color='black', linestyle='-', alpha=0.3)
        else:
            # CERパネル
            axes[1].axhline(y=0.7, color='black', linestyle='--', alpha=0.5)
            axes[1].axhline(y=0.0, color='black', linestyle='-', alpha=0.3)
            axes[1].axhline(y=-0.7, color='black', linestyle='--', alpha=0.5)
            
            # 動的乗数パネル
            mult_mean = df['dynamic_mult'].mean()
            axes[2].axhline(y=mult_mean, color='black', linestyle='-', alpha=0.3)
        
        # 保存または表示
        if savefig:
            plt.savefig(savefig)
            print(f"チャートを保存しました: {savefig}")
        else:
            plt.tight_layout()
            plt.show()


def main():
    """メイン関数"""
    # コマンドライン引数を処理
    import argparse
    parser = argparse.ArgumentParser(description='Zアダプティブチャネルの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    args = parser.parse_args()
    
    # チャートを作成
    chart = ZAdaptiveChannelChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators()
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output
    )


if __name__ == "__main__":
    main() 