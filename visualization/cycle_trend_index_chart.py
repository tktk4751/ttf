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
from indicators.cycle_trend_index import CycleTrendIndex


class CycleTrendIndexChart:
    """
    サイクルトレンドインデックス、サイクル効率比（CER）、サイクルチョピネスを表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - サイクル効率比（CER）
    - サイクルチョピネス
    - サイクルトレンドインデックス
    - 動的しきい値
    - トレンド/レンジ状態
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.cycle_trend_index = None
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
                            # 共通パラメータ
                            detector_type: str = 'phac_e',
                            cycle_part: float = 0.5,
                            max_cycle: int = 144,
                            min_cycle: int = 5,
                            max_output: int = 55,
                            min_output: int = 5,
                            src_type: str = 'hlc3',
                            lp_period: int = 5,
                            hp_period: int = 144,
                            
                            # CER固有パラメータ
                            er_detector_type: Optional[str] = None,
                            er_cycle_part: Optional[float] = None,
                            er_max_cycle: Optional[int] = None,
                            er_min_cycle: Optional[int] = None,
                            er_max_output: Optional[int] = None,
                            er_min_output: Optional[int] = None,
                            er_src_type: Optional[str] = None,
                            er_lp_period: Optional[int] = None,
                            er_hp_period: Optional[int] = None,
                            use_kalman_filter: bool = False,
                            smooth_er: bool = True,
                            er_alma_period: int = 3,
                            er_alma_offset: float = 0.85,
                            er_alma_sigma: float = 6,
                            
                            # チョピネス固有パラメータ
                            chop_detector_type: Optional[str] = None,
                            chop_cycle_part: Optional[float] = None,
                            chop_max_cycle: Optional[int] = None,
                            chop_min_cycle: Optional[int] = None,
                            chop_max_output: Optional[int] = None,
                            chop_min_output: Optional[int] = None,
                            chop_src_type: Optional[str] = None,
                            chop_lp_period: Optional[int] = None,
                            chop_hp_period: Optional[int] = None,
                            smooth_chop: bool = True,
                            chop_alma_period: int = 3,
                            chop_alma_offset: float = 0.85,
                            chop_alma_sigma: float = 6,
                            
                            # 動的しきい値のパラメータ
                            max_threshold: float = 0.75,
                            min_threshold: float = 0.45
                           ) -> None:
        """
        インジケーターを計算する
        
        Args:
            detector_type: 共通のドミナントサイクル検出器タイプ
            cycle_part: 共通のサイクル部分
            max_cycle: 共通の最大サイクル期間
            min_cycle: 共通の最小サイクル期間
            max_output: 共通の最大出力値
            min_output: 共通の最小出力値
            src_type: 共通の価格ソース
            lp_period: 共通のローパスフィルター期間
            hp_period: 共通のハイパスフィルター期間
            
            er_detector_type: CER固有のドミナントサイクル検出器タイプ
            er_cycle_part: CER固有のサイクル部分
            er_max_cycle: CER固有の最大サイクル期間
            er_min_cycle: CER固有の最小サイクル期間
            er_max_output: CER固有の最大出力値
            er_min_output: CER固有の最小出力値
            er_src_type: CER固有の価格ソース
            er_lp_period: CER固有のローパスフィルター期間
            er_hp_period: CER固有のハイパスフィルター期間
            use_kalman_filter: CER用カルマンフィルター使用有無
            smooth_er: CER用スムージング有無
            er_alma_period: CER用ALMAスムージング期間
            er_alma_offset: CER用ALMAオフセット
            er_alma_sigma: CER用ALMAシグマ
            
            chop_detector_type: チョピネス固有のドミナントサイクル検出器タイプ
            chop_cycle_part: チョピネス固有のサイクル部分
            chop_max_cycle: チョピネス固有の最大サイクル期間
            chop_min_cycle: チョピネス固有の最小サイクル期間
            chop_max_output: チョピネス固有の最大出力値
            chop_min_output: チョピネス固有の最小出力値
            chop_src_type: チョピネス固有の価格ソース
            chop_lp_period: チョピネス固有のローパスフィルター期間
            chop_hp_period: チョピネス固有のハイパスフィルター期間
            smooth_chop: チョピネス値にスムージングを適用するか
            chop_alma_period: チョピネス用ALMAスムージング期間
            chop_alma_offset: チョピネス用ALMAオフセット
            chop_alma_sigma: チョピネス用ALMAシグマ
            
            max_threshold: 動的しきい値の最大値
            min_threshold: 動的しきい値の最小値
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\nインジケーターを計算中...")
        
        # サイクルトレンドインデックスを計算
        self.cycle_trend_index = CycleTrendIndex(
            # 共通パラメータ
            detector_type=detector_type,
            cycle_part=cycle_part,
            max_cycle=max_cycle,
            min_cycle=min_cycle,
            max_output=max_output,
            min_output=min_output,
            src_type=src_type,
            lp_period=lp_period,
            hp_period=hp_period,
            
            # CER固有パラメータ
            er_detector_type=er_detector_type,
            er_cycle_part=er_cycle_part,
            er_max_cycle=er_max_cycle,
            er_min_cycle=er_min_cycle,
            er_max_output=er_max_output,
            er_min_output=er_min_output,
            er_src_type=er_src_type,
            er_lp_period=er_lp_period,
            er_hp_period=er_hp_period,
            use_kalman_filter=use_kalman_filter,
            smooth_er=smooth_er,
            er_alma_period=er_alma_period,
            er_alma_offset=er_alma_offset,
            er_alma_sigma=er_alma_sigma,
            
            # チョピネス固有パラメータ
            chop_detector_type=chop_detector_type,
            chop_cycle_part=chop_cycle_part,
            chop_max_cycle=chop_max_cycle,
            chop_min_cycle=chop_min_cycle,
            chop_max_output=chop_max_output,
            chop_min_output=chop_min_output,
            chop_src_type=chop_src_type,
            chop_lp_period=chop_lp_period,
            chop_hp_period=chop_hp_period,
            smooth_chop=smooth_chop,
            chop_alma_period=chop_alma_period,
            chop_alma_offset=chop_alma_offset,
            chop_alma_sigma=chop_alma_sigma,
            
            # 動的しきい値のパラメータ
            max_threshold=max_threshold,
            min_threshold=min_threshold
        )
        
        # インジケーターの計算
        print("サイクルトレンドインデックスを計算中...")
        self.cycle_trend_index.calculate(self.data)
        
        print("インジケーター計算完了")
        
    def plot(self, 
            title: str = "サイクルトレンドインデックス", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (14, 12),
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
            
        if self.cycle_trend_index is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # インジケーターの計算結果を取得
        result = self.cycle_trend_index.calculate(self.data)
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'cti': result.values,             # サイクルトレンドインデックス
                'er': result.er_values,           # サイクル効率比
                'chop': result.chop_values,       # サイクルチョピネス
                'threshold': result.dynamic_threshold,  # 動的しきい値
                'trend_state': result.trend_state  # トレンド状態
            }
        )
        
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
        
        # サイクルトレンドインデックスをパネル3に配置
        cti_panel = mpf.make_addplot(df['cti'], panel=3, color='green', width=1.2, 
                                    ylabel='Cycle Trend', secondary_y=False, label='CTI')
        
        # 動的しきい値をパネル3に追加
        threshold_panel = mpf.make_addplot(df['threshold'], panel=3, color='red', width=1.0, 
                                          secondary_y=False, label='Threshold')
        
        # トレンド状態をパネル4に配置
        trend_state_panel = mpf.make_addplot(df['trend_state'], panel=4, color='green', type='bar',
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
            kwargs['panel_ratios'] = (4, 1, 1, 1, 1, 1)  # メイン:出来高:CER:Chop:CTI:State
            # 出来高を表示する場合は、オシレーターのパネル番号を+1する
            er_panel = mpf.make_addplot(df['er'], panel=2, color='purple', width=1.2, 
                                       ylabel='CER', secondary_y=False, label='CER')
            chop_panel = mpf.make_addplot(df['chop'], panel=3, color='blue', width=1.2, 
                                         ylabel='Choppiness', secondary_y=False, label='Choppiness')
            cti_panel = mpf.make_addplot(df['cti'], panel=4, color='green', width=1.2, 
                                        ylabel='Cycle Trend', secondary_y=False, label='CTI')
            threshold_panel = mpf.make_addplot(df['threshold'], panel=4, color='red', width=1.0, 
                                              secondary_y=False, label='Threshold')
            trend_state_panel = mpf.make_addplot(df['trend_state'], panel=5, color='green', type='bar',
                                               ylabel='Trend State', secondary_y=False, label='Trend State')
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (4, 1, 1, 1, 1)  # メイン:CER:Chop:CTI:State
        
        # すべてのプロットを結合
        all_plots = [er_panel, chop_panel, cti_panel, threshold_panel, trend_state_panel]
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
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
            
            # サイクルトレンドインデックスパネル
            axes[4].axhline(y=0.5, color='black', linestyle='-', alpha=0.3)
            axes[4].legend(['CTI', 'Threshold'], loc='upper left')
            
            # トレンド状態パネル
            axes[5].axhline(y=0.5, color='black', linestyle='-', alpha=0.3)
            axes[5].legend(['Trend State'], loc='upper left')
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
            
            # サイクルトレンドインデックスパネル
            axes[3].axhline(y=0.5, color='black', linestyle='-', alpha=0.3)
            axes[3].legend(['CTI', 'Threshold'], loc='upper left')
            
            # トレンド状態パネル
            axes[4].axhline(y=0.5, color='black', linestyle='-', alpha=0.3)
            axes[4].legend(['Trend State'], loc='upper left')
        
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
    parser = argparse.ArgumentParser(description='サイクルトレンドインデックスの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    args = parser.parse_args()
    
    # チャートを作成
    chart = CycleTrendIndexChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators()
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output
    )


if __name__ == "__main__":
    main() 