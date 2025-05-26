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
from indicators.x_trend_index import XTrendIndex


class XTrendIndexChart:
    """
    Xトレンドインデックスとサイクル効率比（CER）を表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - Xトレンドインデックス
    - サイクル効率比（CER）
    - 固定しきい値
    - トレンド/レンジ状態
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.cycle_er = None
        self.x_trend_index = None
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
                            # CycleEfficiencyRatio パラメータ
                            detector_type: str = 'phac_e',
                            cycle_part: float = 0.5,
                            lp_period: int = 5,
                            hp_period: int = 144,
                            max_cycle: int = 144,
                            min_cycle: int = 5,
                            max_output: int = 55,
                            min_output: int = 5,
                            src_type: str = 'hlc3',
                            use_kalman_filter: bool = False,
                            smooth_er: bool = True,
                            er_alma_period: int = 5,
                            er_alma_offset: float = 0.85,
                            er_alma_sigma: float = 6,
                            # XTrendIndex パラメータ
                            x_detector_type: str = 'phac_e',
                            x_cycle_part: float = 0.5,
                            x_max_cycle: int = 55,
                            x_min_cycle: int = 5,
                            x_max_output: int = 34,
                            x_min_output: int = 5,
                            x_src_type: str = 'hlc3',
                            x_lp_period: int = 5,
                            x_hp_period: int = 55,
                            smoother_type: str = 'alma',
                            max_threshold: float = 0.75,  # 固定しきい値の計算用（平均値）
                            min_threshold: float = 0.55   # 固定しきい値の計算用（平均値）
                           ) -> None:
        """
        インジケーターを計算する
        
        Args:
            detector_type: CER用ドミナントサイクル検出器タイプ
            cycle_part: CER用サイクル部分
            lp_period: CER用ローパスフィルター期間
            hp_period: CER用ハイパスフィルター期間
            max_cycle: CER用最大サイクル期間
            min_cycle: CER用最小サイクル期間
            max_output: CER用最大出力値
            min_output: CER用最小出力値
            src_type: CER用ソースタイプ
            use_kalman_filter: CER用カルマンフィルター使用有無
            smooth_er: CER用スムージング有無
            er_alma_period: CER用ALMAスムージング期間
            er_alma_offset: CER用ALMAオフセット
            er_alma_sigma: CER用ALMAシグマ
            x_detector_type: XTI用ドミナントサイクル検出器タイプ
            x_cycle_part: XTI用サイクル部分
            x_max_cycle: XTI用最大サイクル期間
            x_min_cycle: XTI用最小サイクル期間
            x_max_output: XTI用最大出力値
            x_min_output: XTI用最小出力値
            x_src_type: XTI用ソースタイプ
            x_lp_period: XTI用ローパスフィルター期間
            x_hp_period: XTI用ハイパスフィルター期間
            smoother_type: XTI用CATRスムーザータイプ
            max_threshold: XTI用動的しきい値最大値
            min_threshold: XTI用動的しきい値最小値
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\nインジケーターを計算中...")
        
        # サイクル効率比（CER）を計算
        self.cycle_er = CycleEfficiencyRatio(
            detector_type=detector_type,
            cycle_part=cycle_part,
            lp_period=lp_period,
            hp_period=hp_period,
            max_cycle=max_cycle,
            min_cycle=min_cycle,
            max_output=max_output,
            min_output=min_output,
            src_type=src_type,
            use_kalman_filter=use_kalman_filter,
            smooth_er=smooth_er,
            er_alma_period=er_alma_period,
            er_alma_offset=er_alma_offset,
            er_alma_sigma=er_alma_sigma
        )
        
        # Xトレンドインデックスを計算
        self.x_trend_index = XTrendIndex(
            detector_type=x_detector_type,
            cycle_part=x_cycle_part,
            max_cycle=x_max_cycle,
            min_cycle=x_min_cycle,
            max_output=x_max_output,
            min_output=x_min_output,
            src_type=x_src_type,
            lp_period=x_lp_period,
            hp_period=x_hp_period,
            smoother_type=smoother_type,
            fixed_threshold=(max_threshold + min_threshold) / 2.0  # 固定しきい値として平均値を使用
        )
        
        # インジケーターの計算
        print("サイクル効率比（CER）を計算中...")
        self.cycle_er.calculate(self.data)
        
        print("Xトレンドインデックスを計算中...")
        self.x_trend_index.calculate(self.data)
        
        print("インジケーター計算完了")
        
    def plot(self, 
            title: str = "Xトレンドインデックスとサイクル効率比", 
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
            
        if self.cycle_er is None or self.x_trend_index is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # インジケーターの値を取得
        er_values = self.cycle_er.calculate(self.data)
        x_trend_result = self.x_trend_index.calculate(self.data)
        
        # X Trend Index の値を取得
        x_trend_values = x_trend_result.values
        fixed_threshold = x_trend_result.fixed_threshold  # 固定しきい値を取得
        trend_state = x_trend_result.trend_state
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'er': er_values,
                'x_trend': x_trend_values,
                'threshold': np.full_like(x_trend_values, fixed_threshold),  # 固定しきい値を全データに適用
                'trend_state': trend_state
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        
        # mplfinanceでプロット用の設定
        # 1. オシレータープロット
        # サイクル効率比（CER）をパネル1に配置
        er_panel = mpf.make_addplot(df['er'], panel=1, color='purple', width=1.2, 
                                   ylabel='CER', secondary_y=False, label='CER')
        
        # Xトレンドインデックスをパネル2に配置
        x_trend_panel = mpf.make_addplot(df['x_trend'], panel=2, color='blue', width=1.2, 
                                        ylabel='X Trend', secondary_y=False, label='X Trend')
        
        # 動的しきい値をパネル2に追加
        threshold_panel = mpf.make_addplot(df['threshold'], panel=2, color='red', width=1.0, 
                                          secondary_y=False, label='Fixed Threshold')
        
        # トレンド状態をパネル3に配置
        trend_state_panel = mpf.make_addplot(df['trend_state'], panel=3, color='green', type='bar',
                                           ylabel='Trend State', secondary_y=False, label='Trend State')
        
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
            kwargs['panel_ratios'] = (4, 1, 1, 1, 1)  # メイン:出来高:CER:X Trend:Trend State
            # 出来高を表示する場合は、オシレーターのパネル番号を+1する
            er_panel = mpf.make_addplot(df['er'], panel=2, color='purple', width=1.2, 
                                       ylabel='CER', secondary_y=False, label='CER')
            x_trend_panel = mpf.make_addplot(df['x_trend'], panel=3, color='blue', width=1.2, 
                                            ylabel='X Trend', secondary_y=False, label='X Trend')
            threshold_panel = mpf.make_addplot(df['threshold'], panel=3, color='red', width=1.0, 
                                              secondary_y=False, label='Fixed Threshold')
            trend_state_panel = mpf.make_addplot(df['trend_state'], panel=4, color='green', type='bar',
                                               ylabel='Trend State', secondary_y=False, label='Trend State')
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (4, 1, 1, 1)  # メイン:CER:X Trend:Trend State
        
        # すべてのプロットを結合
        all_plots = [er_panel, x_trend_panel, threshold_panel, trend_state_panel]
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
            
            # X Trendパネル
            axes[3].axhline(y=0.5, color='black', linestyle='-', alpha=0.3)
            axes[3].legend(['X Trend', 'Fixed Threshold'], loc='upper left')
            
            # Trend Stateパネル
            axes[4].axhline(y=0.5, color='black', linestyle='-', alpha=0.3)
            axes[4].legend(['Trend State'], loc='upper left')
        else:
            # CERパネル
            axes[1].axhline(y=0.7, color='black', linestyle='--', alpha=0.5)
            axes[1].axhline(y=0.0, color='black', linestyle='-', alpha=0.3)
            axes[1].axhline(y=-0.7, color='black', linestyle='--', alpha=0.5)
            axes[1].legend(['CER'], loc='upper left')
            
            # X Trendパネル
            axes[2].axhline(y=0.5, color='black', linestyle='-', alpha=0.3)
            axes[2].legend(['X Trend', 'Fixed Threshold'], loc='upper left')
            
            # Trend Stateパネル
            axes[3].axhline(y=0.5, color='black', linestyle='-', alpha=0.3)
            axes[3].legend(['Trend State'], loc='upper left')
        
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
    parser = argparse.ArgumentParser(description='XトレンドインデックスとサイクルERの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    args = parser.parse_args()
    
    # チャートを作成
    chart = XTrendIndexChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators()
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output
    )


if __name__ == "__main__":
    main() 