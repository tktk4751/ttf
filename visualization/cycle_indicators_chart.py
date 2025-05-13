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
from indicators.cycle_efficiency_ratio import CycleEfficiencyRatio
from indicators.cycle_chop import CycleChoppiness


class CycleIndicatorsChart:
    """
    サイクル効率比（CER）とサイクルチョピネスを表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - サイクル効率比（CER）
    - サイクルチョピネス
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.cycle_er = None
        self.cycle_chop = None
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
                            # サイクル効率比パラメータ
                            er_detector_type: str = 'phac_e',
                            er_cycle_part: float = 0.5,
                            er_lp_period: int = 5,
                            er_hp_period: int = 144,
                            er_max_cycle: int = 144,
                            er_min_cycle: int = 5,
                            er_max_output: int = 55,
                            er_min_output: int = 5,
                            er_src_type: str = 'hlc3',
                            er_use_kalman_filter: bool = False,
                            er_smooth: bool = True,
                            er_alma_period: int = 5,
                            er_alma_offset: float = 0.85,
                            er_alma_sigma: float = 6,
                            # サイクルチョピネスパラメータ
                            chop_detector_type: str = 'phac_e',
                            chop_cycle_part: float = 0.5,
                            chop_max_cycle: int = 144,
                            chop_min_cycle: int = 5,
                            chop_max_output: int = 75,
                            chop_min_output: int = 8,
                            chop_src_type: str = 'hlc3',
                            chop_lp_period: int = 5,
                            chop_hp_period: int = 144,
                            smooth_chop: bool = True,
                            chop_alma_period: int = 5,
                            chop_alma_offset: float = 0.85,
                            chop_alma_sigma: float = 6
                           ) -> None:
        """
        インジケーターを計算する
        
        Args:
            er_detector_type: CER用ドミナントサイクル検出器タイプ
            er_cycle_part: CER用サイクル部分
            er_lp_period: CER用ローパスフィルター期間
            er_hp_period: CER用ハイパスフィルター期間
            er_max_cycle: CER用最大サイクル期間
            er_min_cycle: CER用最小サイクル期間
            er_max_output: CER用最大出力値
            er_min_output: CER用最小出力値
            er_src_type: CER用ソースタイプ
            er_use_kalman_filter: CER用カルマンフィルター使用有無
            er_smooth: CER用スムージング有無
            er_alma_period: CER用ALMAスムージング期間
            er_alma_offset: CER用ALMAオフセット
            er_alma_sigma: CER用ALMAシグマ
            chop_detector_type: チョピネス用ドミナントサイクル検出器タイプ
            chop_cycle_part: チョピネス用サイクル部分
            chop_max_cycle: チョピネス用最大サイクル期間
            chop_min_cycle: チョピネス用最小サイクル期間
            chop_max_output: チョピネス用最大出力値
            chop_min_output: チョピネス用最小出力値
            chop_src_type: チョピネス用ソースタイプ
            chop_lp_period: チョピネス用ローパスフィルター期間
            chop_hp_period: チョピネス用ハイパスフィルター期間
            smooth_chop: チョピネス値にスムージングを適用するか
            chop_alma_period: チョピネス用ALMAスムージング期間
            chop_alma_offset: チョピネス用ALMAオフセット
            chop_alma_sigma: チョピネス用ALMAシグマ
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\nインジケーターを計算中...")
        
        # サイクル効率比（CER）を計算
        self.cycle_er = CycleEfficiencyRatio(
            detector_type=er_detector_type,
            cycle_part=er_cycle_part,
            lp_period=er_lp_period,
            hp_period=er_hp_period,
            max_cycle=er_max_cycle,
            min_cycle=er_min_cycle,
            max_output=er_max_output,
            min_output=er_min_output,
            src_type=er_src_type,
            use_kalman_filter=er_use_kalman_filter,
            smooth_er=er_smooth,
            er_alma_period=er_alma_period,
            er_alma_offset=er_alma_offset,
            er_alma_sigma=er_alma_sigma
        )
        
        # サイクルチョピネスを計算
        self.cycle_chop = CycleChoppiness(
            detector_type=chop_detector_type,
            lp_period=chop_lp_period,
            hp_period=chop_hp_period,
            cycle_part=chop_cycle_part,
            max_cycle=chop_max_cycle,
            min_cycle=chop_min_cycle,
            max_output=chop_max_output,
            min_output=chop_min_output,
            src_type=chop_src_type,
            smooth_chop=smooth_chop,
            chop_alma_period=chop_alma_period,
            chop_alma_offset=chop_alma_offset,
            chop_alma_sigma=chop_alma_sigma
        )
        
        # インジケーターの計算
        print("サイクル効率比（CER）を計算中...")
        self.cycle_er.calculate(self.data)
        
        print("サイクルチョピネスを計算中...")
        self.cycle_chop.calculate(self.data)
        
        print("インジケーター計算完了")
        
    def plot(self, 
            title: str = "サイクル効率比とサイクルチョピネス", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (14, 10),
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
            
        if self.cycle_er is None or self.cycle_chop is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # インジケーターの値を取得
        er_values = self.cycle_er.calculate(self.data)
        chop_values = self.cycle_chop.calculate(self.data)
        
        # 動的サイクル期間を取得
        er_cycle_periods = self.cycle_er.get_cycle_periods() if hasattr(self.cycle_er, 'get_cycle_periods') else None
        chop_cycle_periods = self.cycle_chop.get_cycle_periods() if hasattr(self.cycle_chop, 'get_cycle_periods') else None
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'er': er_values,
                'chop': chop_values
            }
        )
        
        # サイクル期間が取得できる場合は追加
        if er_cycle_periods is not None and len(er_cycle_periods) == len(self.data):
            full_df['er_cycle'] = er_cycle_periods
        
        if chop_cycle_periods is not None and len(chop_cycle_periods) == len(self.data):
            full_df['chop_cycle'] = chop_cycle_periods
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        
        # mplfinanceでプロット用の設定
        
        # サイクル効率比（CER）をパネル1に配置
        er_panel = mpf.make_addplot(df['er'], panel=1, color='purple', width=1.2, 
                                   ylabel='CER', secondary_y=False, label='CER')
        
        # サイクルチョピネスをパネル2に配置
        chop_panel = mpf.make_addplot(df['chop'], panel=2, color='blue', width=1.2, 
                                     ylabel='Choppiness', secondary_y=False, label='Choppiness')
        
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
            kwargs['panel_ratios'] = (4, 1, 1, 1)  # メイン:出来高:CER:Choppiness
            # 出来高を表示する場合は、オシレーターのパネル番号を+1する
            er_panel = mpf.make_addplot(df['er'], panel=2, color='purple', width=1.2, 
                                       ylabel='CER', secondary_y=False, label='CER')
            chop_panel = mpf.make_addplot(df['chop'], panel=3, color='blue', width=1.2, 
                                         ylabel='Choppiness', secondary_y=False, label='Choppiness')
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (4, 1, 1)  # メイン:CER:Choppiness
        
        # すべてのプロットを結合
        all_plots = [er_panel, chop_panel]
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        self.fig = fig
        self.axes = axes
        
        # CERパネルに参照線を追加
        if show_volume:
            # CERパネル
            axes[2].axhline(y=0.7, color='black', linestyle='--', alpha=0.5)
            axes[2].axhline(y=0.0, color='black', linestyle='-', alpha=0.3)
            axes[2].axhline(y=-0.7, color='black', linestyle='--', alpha=0.5)
            axes[2].legend(['CER'], loc='upper left')
            
            # チョピネスパネル
            axes[3].axhline(y=0.5, color='black', linestyle='-', alpha=0.3)
            axes[3].axhline(y=0.2, color='black', linestyle='--', alpha=0.5)
            axes[3].axhline(y=0.8, color='black', linestyle='--', alpha=0.5)
            axes[3].legend(['Choppiness'], loc='upper left')
        else:
            # CERパネル
            axes[1].axhline(y=0.7, color='black', linestyle='--', alpha=0.5)
            axes[1].axhline(y=0.0, color='black', linestyle='-', alpha=0.3)
            axes[1].axhline(y=-0.7, color='black', linestyle='--', alpha=0.5)
            axes[1].legend(['CER'], loc='upper left')
            
            # チョピネスパネル
            axes[2].axhline(y=0.5, color='black', linestyle='-', alpha=0.3)
            axes[2].axhline(y=0.2, color='black', linestyle='--', alpha=0.5)
            axes[2].axhline(y=0.8, color='black', linestyle='--', alpha=0.5)
            axes[2].legend(['Choppiness'], loc='upper left')
        
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
    parser = argparse.ArgumentParser(description='サイクル効率比とサイクルチョピネスの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    args = parser.parse_args()
    
    # チャートを作成
    chart = CycleIndicatorsChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators()
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output
    )


if __name__ == "__main__":
    main() 