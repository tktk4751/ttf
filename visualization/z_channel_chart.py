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
from indicators.z_channel import ZChannel


class ZChannelChart:
    """
    Zチャネルを表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - Zチャネルの中心線・上限バンド・下限バンド
    - サイクル効率比（CER）
    - 動的乗数値
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.z_channel = None
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
                            # 基本パラメータ
                            detector_type: str = 'phac_e',
                            cer_detector_type: str = None,
                            lp_period: int = 5,
                            hp_period: int = 55,
                            cycle_part: float = 0.382,
                            max_multiplier: float = 7.0,
                            min_multiplier: float = 1.0,
                            # 動的乗数の範囲パラメータ
                            max_max_multiplier: float = 8.0,
                            min_max_multiplier: float = 3.0,
                            max_min_multiplier: float = 1.5,
                            min_min_multiplier: float = 0.5,
                            smoother_type: str = 'alma',
                            src_type: str = 'hlc3',
                            # CER用パラメータ
                            cer_max_cycle: int = 62,
                            cer_min_cycle: int = 5,
                            cer_max_output: int = 34,
                            cer_min_output: int = 5
                           ) -> None:
        """
        Zチャネルを計算する
        
        Args:
            detector_type: 検出器タイプ（ZMAとZATRに使用）
            cer_detector_type: CER用の検出器タイプ（指定しない場合はdetector_typeと同じ）
            lp_period: ローパスフィルターの期間
            hp_period: ハイパスフィルターの期間
            cycle_part: サイクル部分の倍率
            max_multiplier: ATR乗数の最大値（レガシーパラメータ）
            min_multiplier: ATR乗数の最小値（レガシーパラメータ）
            max_max_multiplier: 最大乗数の最大値
            min_max_multiplier: 最大乗数の最小値
            max_min_multiplier: 最小乗数の最大値
            min_min_multiplier: 最小乗数の最小値
            smoother_type: 平滑化アルゴリズムのタイプ
            src_type: ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4')
            cer_max_cycle: CER用の最大サイクル期間
            cer_min_cycle: CER用の最小サイクル期間
            cer_max_output: CER用の最大出力値
            cer_min_output: CER用の最小出力値
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\nZチャネルを計算中...")
        
        # Zチャネルを計算
        self.z_channel = ZChannel(
            detector_type=detector_type,
            cer_detector_type=cer_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            max_multiplier=max_multiplier,
            min_multiplier=min_multiplier,
            max_max_multiplier=max_max_multiplier,
            min_max_multiplier=min_max_multiplier,
            max_min_multiplier=max_min_multiplier,
            min_min_multiplier=min_min_multiplier,
            smoother_type=smoother_type,
            src_type=src_type,
            cer_max_cycle=cer_max_cycle,
            cer_min_cycle=cer_min_cycle,
            cer_max_output=cer_max_output,
            cer_min_output=cer_min_output
        )
        
        # Zチャネルの計算
        self.z_channel.calculate(self.data)
        
        print("Zチャネル計算完了")
            
    def plot(self, 
            title: str = "Zチャネル", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (14, 10),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとZチャネルを描画する
        
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
            
        if self.z_channel is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # 絞り込み後のインデックスがオリジナルデータのどの位置に対応するかを特定
        idx_map = [self.data.index.get_loc(idx) for idx in df.index]
        
        # Zチャネルの値を取得して追加
        middle, upper, lower = self.z_channel.get_bands()
        df['channel_middle'] = middle[idx_map]
        df['channel_upper'] = upper[idx_map]
        df['channel_lower'] = lower[idx_map]
        
        # サイクル効率比（CER）を取得
        cer = self.z_channel.get_cycle_er()
        df['cer'] = cer[idx_map]
        
        # 動的乗数を取得
        dynamic_mult = self.z_channel.get_dynamic_multiplier()
        df['dynamic_mult'] = dynamic_mult[idx_map]
        
        # Z-ATRを取得
        z_atr = self.z_channel.get_z_atr()
        df['z_atr'] = z_atr[idx_map]
        
        # mplfinanceでプロット用の設定
        # 1. メインチャート上のプロット
        main_plots = []
        
        # Zチャネルのプロット設定
        main_plots.append(mpf.make_addplot(df['channel_middle'], color='gray', width=1.5, label='Z Middle'))
        main_plots.append(mpf.make_addplot(df['channel_upper'], color='green', width=1, label='Z Upper'))
        main_plots.append(mpf.make_addplot(df['channel_lower'], color='red', width=1, label='Z Lower'))
        
        # 2. オシレータープロット
        # CERとパネル配置を設定
        cer_panel = mpf.make_addplot(df['cer'], panel=1, color='purple', width=1.2, 
                                     ylabel='CER', secondary_y=False, label='CER')
        
        mult_panel = mpf.make_addplot(df['dynamic_mult'], panel=2, color='blue', width=1.2, 
                                     ylabel='Dynamic Mult', secondary_y=False, label='Multiplier')
        
        atr_panel = mpf.make_addplot(df['z_atr'], panel=3, color='orange', width=1.2, 
                                     ylabel='Z-ATR', secondary_y=False, label='Z-ATR')
        
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
            kwargs['panel_ratios'] = (4, 1, 1, 1, 1)  # メイン:出来高:CER:乗数:ATR
            # 出来高を表示する場合は、オシレーターのパネル番号を+1する
            cer_panel = mpf.make_addplot(df['cer'], panel=2, color='purple', width=1.2, 
                                         ylabel='CER', secondary_y=False, label='CER')
            mult_panel = mpf.make_addplot(df['dynamic_mult'], panel=3, color='blue', width=1.2, 
                                         ylabel='Dynamic Mult', secondary_y=False, label='Multiplier')
            atr_panel = mpf.make_addplot(df['z_atr'], panel=4, color='orange', width=1.2, 
                                        ylabel='Z-ATR', secondary_y=False, label='Z-ATR')
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (4, 1, 1, 1)  # メイン:CER:乗数:ATR
        
        # すべてのプロットを結合
        all_plots = main_plots + [cer_panel, mult_panel, atr_panel]
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加
        axes[0].legend(['Z Middle', 'Z Upper', 'Z Lower'], loc='upper left')
        
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
            
            # ATRパネル
            atr_mean = df['z_atr'].mean()
            axes[4].axhline(y=atr_mean, color='black', linestyle='-', alpha=0.3)
        else:
            # CERパネル
            axes[1].axhline(y=0.7, color='black', linestyle='--', alpha=0.5)
            axes[1].axhline(y=0.0, color='black', linestyle='-', alpha=0.3)
            axes[1].axhline(y=-0.7, color='black', linestyle='--', alpha=0.5)
            
            # 動的乗数パネル
            mult_mean = df['dynamic_mult'].mean()
            axes[2].axhline(y=mult_mean, color='black', linestyle='-', alpha=0.3)
            
            # ATRパネル
            atr_mean = df['z_atr'].mean()
            axes[3].axhline(y=atr_mean, color='black', linestyle='-', alpha=0.3)
        
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
    parser = argparse.ArgumentParser(description='Zチャネルの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    args = parser.parse_args()
    
    # チャートを作成
    chart = ZChannelChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators()
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output
    )


if __name__ == "__main__":
    main() 