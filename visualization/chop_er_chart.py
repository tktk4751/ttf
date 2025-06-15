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
from indicators.chop_er import ChopER


class ChopERChart:
    """
    CHOP_ER（CHOP Trend Efficiency Ratio）を表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - CHOP_ER値（Efficiency Ratio）のプロット
    - CHOPトレンド値のプロット
    - トレンド信号のカラー表示
    - 動的期間値の表示（使用している場合）
    - 固定しきい値の参照線
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.chop_er = None
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
                            # CHOPトレンドパラメータ（固定期間13で計算）
                            chop_trend_period: int = 13,
                            
                            # Efficiency Ratioパラメータ
                            er_period: int = 13,
                            smoothing_method: str = 'hma',
                            use_dynamic_er_period: bool = False,
                            smoother_period: int = 5,
                            
                            # EhlersUnifiedDC パラメータ（ER用）
                            detector_type: str = 'cycle_period2',
                            cycle_part: float = 1.0,
                            max_cycle: int = 144,
                            min_cycle: int = 10,
                            max_output: int = 89,
                            min_output: int = 13,
                            src_type: str = 'hlc3',
                            lp_period: int = 10,
                            hp_period: int = 48,
                            
                            # トレンド判定パラメータ
                            slope_index: int = 3,
                            range_threshold: float = 0.005,

                            # 固定しきい値のパラメータ
                            fixed_threshold: float = 0.618
                           ) -> None:
        """
        CHOP_ERインジケーターを計算する
        
        Args:
            chop_trend_period: CHOPトレンドの期間（デフォルト: 13、固定）
            er_period: Efficiency Ratioの期間（デフォルト: 13）
            smoothing_method: スムージング方法 ('none', 'wilder', 'hma', 'alma', 'zlema')
            use_dynamic_er_period: ER計算で動的期間を使用するかどうか（デフォルト: False）
            smoother_period: スムージング期間（デフォルト: 5）
            detector_type: EhlersUnifiedDCで使用する検出器タイプ（デフォルト: 'cycle_period2'）
            cycle_part: DCのサイクル部分の倍率（デフォルト: 1.0）
            max_cycle: DCの最大サイクル期間（デフォルト: 144）
            min_cycle: DCの最小サイクル期間（デフォルト: 10）
            max_output: DCの最大出力値（デフォルト: 89）
            min_output: DCの最小出力値（デフォルト: 13）
            src_type: DC計算に使用する価格ソース（デフォルト: 'hlc3'）
            lp_period: 拡張DC用のローパスフィルター期間（デフォルト: 10）
            hp_period: 拡張DC用のハイパスフィルター期間（デフォルト: 48）
            slope_index: トレンド判定期間（デフォルト: 3）
            range_threshold: range判定の基本閾値（デフォルト: 0.005）
            fixed_threshold: 固定しきい値（デフォルト: 0.618）
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\nCHOP_ERインジケーターを計算中...")
        
        # CHOP_ERインジケーターを初期化
        self.chop_er = ChopER(
            chop_trend_period=chop_trend_period,
            er_period=er_period,
            smoothing_method=smoothing_method,
            use_dynamic_er_period=use_dynamic_er_period,
            smoother_period=smoother_period,
            detector_type=detector_type,
            cycle_part=cycle_part,
            max_cycle=max_cycle,
            min_cycle=min_cycle,
            max_output=max_output,
            min_output=min_output,
            src_type=src_type,
            lp_period=lp_period,
            hp_period=hp_period,
            slope_index=slope_index,
            range_threshold=range_threshold,
            fixed_threshold=fixed_threshold
        )
        
        # CHOP_ERの計算
        print("計算を実行します...")
        result = self.chop_er.calculate(self.data)
        
        # 計算結果の取得テスト
        chop_er_values = self.chop_er.get_values()
        chop_trend_values = self.chop_er.get_chop_trend_values()
        trend_signals = self.chop_er.get_trend_signals()
        dynamic_periods = self.chop_er.get_dynamic_periods()
        
        print(f"CHOP_ER計算完了 - CHOP_ER: {len(chop_er_values)}, CHOPトレンド: {len(chop_trend_values)}")
        print(f"トレンド信号: {len(trend_signals)}, 動的期間: {len(dynamic_periods)}")
        
        # NaN値のチェック
        nan_count_er = np.isnan(chop_er_values).sum()
        nan_count_chop = np.isnan(chop_trend_values).sum()
        trend_count = (trend_signals != 0).sum()
        print(f"NaN値 - CHOP_ER: {nan_count_er}, CHOPトレンド: {nan_count_chop}")
        print(f"トレンド信号 - 有効: {trend_count}, 上昇: {(trend_signals == 1).sum()}, 下降: {(trend_signals == -1).sum()}")
        
        # 統計情報
        valid_er = chop_er_values[~np.isnan(chop_er_values)]
        if len(valid_er) > 0:
            print(f"CHOP_ER統計 - 平均: {valid_er.mean():.3f}, 範囲: {valid_er.min():.3f} - {valid_er.max():.3f}")
        
        print("CHOP_ER計算完了")
            
    def plot(self, 
            title: str = "CHOP_ER (CHOP Trend Efficiency Ratio)", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (14, 12),
            style: str = 'yahoo',
            savefig: Optional[str] = None,
            show_continuous_line: bool = True,
            recent_bars: int = 100) -> None:
        """
        ローソク足チャートとCHOP_ERを描画する
        
        Args:
            title: チャートのタイトル
            start_date: 表示開始日（フォーマット: YYYY-MM-DD）
            end_date: 表示終了日（フォーマット: YYYY-MM-DD）
            show_volume: 出来高を表示するか
            figsize: 図のサイズ
            style: mplfinanceのスタイル
            savefig: 保存先のパス（指定しない場合は表示のみ）
            show_continuous_line: 連続ライン表示（Trueの場合、色分けと連続ラインの両方を表示）
            recent_bars: 出力する直近のバー数（デフォルト: 100）
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        if self.chop_er is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # CHOP_ERの値を取得
        print("CHOP_ERデータを取得中...")
        chop_er_values = self.chop_er.get_values()
        chop_trend_values = self.chop_er.get_chop_trend_values()
        trend_signals = self.chop_er.get_trend_signals()
        dynamic_periods = self.chop_er.get_dynamic_periods()
        fixed_threshold = self.chop_er.get_fixed_threshold()
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'chop_er': chop_er_values,
                'chop_trend': chop_trend_values,
                'trend_signals': trend_signals,
                'dynamic_periods': dynamic_periods if len(dynamic_periods) > 0 else np.full(len(chop_er_values), np.nan)
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        print(f"CHOP_ERデータ確認 - CHOP_ER NaN: {df['chop_er'].isna().sum()}, CHOPトレンド NaN: {df['chop_trend'].isna().sum()}")
        
        # トレンド信号に基づく色分け用データの準備
        df['chop_er_up'] = np.where(df['trend_signals'] == 1, df['chop_er'], np.nan)
        df['chop_er_down'] = np.where(df['trend_signals'] == -1, df['chop_er'], np.nan)
        df['chop_er_range'] = np.where(df['trend_signals'] == 0, df['chop_er'], np.nan)
        
        # CHOPトレンドの色分け
        df['chop_trend_up'] = np.where(df['trend_signals'] == 1, df['chop_trend'], np.nan)
        df['chop_trend_down'] = np.where(df['trend_signals'] == -1, df['chop_trend'], np.nan)
        df['chop_trend_range'] = np.where(df['trend_signals'] == 0, df['chop_trend'], np.nan)
        
        # NaN値を含む行を出力（最初の5行のみ）
        nan_rows = df[df['chop_er'].isna() | df['chop_trend'].isna()]
        if not nan_rows.empty:
            print(f"NaN値を含む行: {len(nan_rows)}行")
            if len(nan_rows) > 0:
                print(nan_rows.head())
        
        # mplfinanceでプロット用の設定
        # 1. メインチャート上のプロット（なし）
        main_plots = []
        
        # 2. オシレータープロット
        plots_list = []
        
        if show_continuous_line:
            # 連続ライン + 色分け表示
            # CHOP_ERパネル - 連続ライン（薄い色）
            chop_er_continuous = mpf.make_addplot(df['chop_er'], panel=1, color='lightblue', width=0.8, alpha=0.6,
                                                 ylabel='CHOP_ER', secondary_y=False, label='CHOP_ER (Continuous)')
            plots_list.append(chop_er_continuous)
            
            # CHOP_ERパネル - 色分け表示（太い線）
            plots_list.extend([
                mpf.make_addplot(df['chop_er_up'], panel=1, color='green', width=2.0, 
                               secondary_y=False, label='CHOP_ER (Up)'),
                mpf.make_addplot(df['chop_er_down'], panel=1, color='red', width=2.0, 
                               secondary_y=False, label='CHOP_ER (Down)'),
                mpf.make_addplot(df['chop_er_range'], panel=1, color='gray', width=1.5, 
                               secondary_y=False, label='CHOP_ER (Range)')
            ])
            
            # CHOPトレンドパネル - 連続ライン（薄い色）
            chop_trend_continuous = mpf.make_addplot(df['chop_trend'], panel=2, color='lightcyan', width=0.8, alpha=0.6,
                                                   ylabel='CHOP Trend', secondary_y=False, label='CHOP Trend (Continuous)')
            plots_list.append(chop_trend_continuous)
            
            # CHOPトレンドパネル - 色分け表示（太い線）
            plots_list.extend([
                mpf.make_addplot(df['chop_trend_up'], panel=2, color='darkgreen', width=2.0, 
                               secondary_y=False, label='CHOP Trend (Up)'),
                mpf.make_addplot(df['chop_trend_down'], panel=2, color='darkred', width=2.0, 
                               secondary_y=False, label='CHOP Trend (Down)'),
                mpf.make_addplot(df['chop_trend_range'], panel=2, color='darkgray', width=1.5, 
                               secondary_y=False, label='CHOP Trend (Range)')
            ])
        else:
            # 色分け表示のみ
            # CHOP_ERパネル（トレンド信号による色分け）
            plots_list.extend([
                mpf.make_addplot(df['chop_er_up'], panel=1, color='green', width=1.5, 
                               ylabel='CHOP_ER', secondary_y=False, label='CHOP_ER (Up)'),
                mpf.make_addplot(df['chop_er_down'], panel=1, color='red', width=1.5, 
                               secondary_y=False, label='CHOP_ER (Down)'),
                mpf.make_addplot(df['chop_er_range'], panel=1, color='gray', width=1.0, 
                               secondary_y=False, label='CHOP_ER (Range)')
            ])
            
            # CHOPトレンドパネル（トレンド信号による色分け）
            plots_list.extend([
                mpf.make_addplot(df['chop_trend_up'], panel=2, color='darkgreen', width=1.2, 
                               ylabel='CHOP Trend', secondary_y=False, label='CHOP Trend (Up)'),
                mpf.make_addplot(df['chop_trend_down'], panel=2, color='darkred', width=1.2, 
                               secondary_y=False, label='CHOP Trend (Down)'),
                mpf.make_addplot(df['chop_trend_range'], panel=2, color='darkgray', width=1.0, 
                               secondary_y=False, label='CHOP Trend (Range)')
            ])
        
        # トレンド信号パネル
        trend_signal_panel = mpf.make_addplot(df['trend_signals'], panel=3, color='orange', width=1.5, 
                                             ylabel='Trend Signal', secondary_y=False, label='Trend Signal', type='line')
        plots_list.append(trend_signal_panel)
        
        # 動的期間パネル（動的期間モードの場合のみ）
        if not df['dynamic_periods'].isna().all():
            dynamic_period_panel = mpf.make_addplot(df['dynamic_periods'], panel=4, color='purple', width=1.2, 
                                                   ylabel='Dynamic Period', secondary_y=False, label='Dynamic Period')
            plots_list.append(dynamic_period_panel)
        
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
            if not df['dynamic_periods'].isna().all():
                kwargs['volume'] = True
                kwargs['panel_ratios'] = (4, 1, 1.5, 1.5, 1, 1)  # メイン:出来高:CHOP_ER:CHOPトレンド:信号:期間
                # 出来高を表示する場合は、オシレーターのパネル番号を+1する
                plots_list = []
                plots_list.append(mpf.make_addplot(df['chop_er_up'], panel=2, color='green', width=1.5, 
                                                  ylabel='CHOP_ER', secondary_y=False, label='CHOP_ER (Up)'))
                plots_list.append(mpf.make_addplot(df['chop_er_down'], panel=2, color='red', width=1.5, 
                                                  secondary_y=False, label='CHOP_ER (Down)'))
                plots_list.append(mpf.make_addplot(df['chop_er_range'], panel=2, color='gray', width=1.0, 
                                                  secondary_y=False, label='CHOP_ER (Range)'))
                
                plots_list.append(mpf.make_addplot(df['chop_trend_up'], panel=3, color='darkgreen', width=1.2, 
                                                  ylabel='CHOP Trend', secondary_y=False, label='CHOP Trend (Up)'))
                plots_list.append(mpf.make_addplot(df['chop_trend_down'], panel=3, color='darkred', width=1.2, 
                                                  secondary_y=False, label='CHOP Trend (Down)'))
                plots_list.append(mpf.make_addplot(df['chop_trend_range'], panel=3, color='darkgray', width=1.0, 
                                                  secondary_y=False, label='CHOP Trend (Range)'))
                
                plots_list.append(mpf.make_addplot(df['trend_signals'], panel=4, color='orange', width=1.5, 
                                                  ylabel='Trend Signal', secondary_y=False, label='Trend Signal', type='line'))
                
                plots_list.append(mpf.make_addplot(df['dynamic_periods'], panel=5, color='purple', width=1.2, 
                                                  ylabel='Dynamic Period', secondary_y=False, label='Dynamic Period'))
            else:
                kwargs['volume'] = True
                kwargs['panel_ratios'] = (4, 1, 1.5, 1.5, 1)  # メイン:出来高:CHOP_ER:CHOPトレンド:信号
                plots_list = []
                plots_list.append(mpf.make_addplot(df['chop_er_up'], panel=2, color='green', width=1.5, 
                                                  ylabel='CHOP_ER', secondary_y=False, label='CHOP_ER (Up)'))
                plots_list.append(mpf.make_addplot(df['chop_er_down'], panel=2, color='red', width=1.5, 
                                                  secondary_y=False, label='CHOP_ER (Down)'))
                plots_list.append(mpf.make_addplot(df['chop_er_range'], panel=2, color='gray', width=1.0, 
                                                  secondary_y=False, label='CHOP_ER (Range)'))
                
                plots_list.append(mpf.make_addplot(df['chop_trend_up'], panel=3, color='darkgreen', width=1.2, 
                                                  ylabel='CHOP Trend', secondary_y=False, label='CHOP Trend (Up)'))
                plots_list.append(mpf.make_addplot(df['chop_trend_down'], panel=3, color='darkred', width=1.2, 
                                                  secondary_y=False, label='CHOP Trend (Down)'))
                plots_list.append(mpf.make_addplot(df['chop_trend_range'], panel=3, color='darkgray', width=1.0, 
                                                  secondary_y=False, label='CHOP Trend (Range)'))
                
                plots_list.append(mpf.make_addplot(df['trend_signals'], panel=4, color='orange', width=1.5, 
                                                  ylabel='Trend Signal', secondary_y=False, label='Trend Signal', type='line'))
        else:
            if not df['dynamic_periods'].isna().all():
                kwargs['volume'] = False
                kwargs['panel_ratios'] = (4, 1, 1.5, 1.5, 1)  # メイン:CHOP_ER:CHOPトレンド:信号
            else:
                kwargs['volume'] = False
                kwargs['panel_ratios'] = (4, 1, 1.5, 1)  # メイン:CHOP_ER:CHOPトレンド:信号
        
        # すべてのプロットを結合
        all_plots = main_plots + plots_list
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        if show_volume:
            panel_offset = 1  # 出来高パネルがある場合のオフセット
        else:
            panel_offset = 0
            
        # CHOP_ERパネルに固定しきい値の線を追加
        chop_er_panel_idx = 1 + panel_offset
        axes[chop_er_panel_idx].axhline(y=fixed_threshold, color='black', linestyle='--', alpha=0.7, label=f'Threshold ({fixed_threshold})')
        axes[chop_er_panel_idx].axhline(y=0.0, color='black', linestyle='-', alpha=0.3)
        axes[chop_er_panel_idx].axhline(y=1.0, color='black', linestyle='-', alpha=0.3)
        
        # CHOPトレンドパネルに参照線を追加
        chop_trend_panel_idx = 2 + panel_offset
        axes[chop_trend_panel_idx].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
        axes[chop_trend_panel_idx].axhline(y=0.0, color='black', linestyle='-', alpha=0.3)
        axes[chop_trend_panel_idx].axhline(y=1.0, color='black', linestyle='-', alpha=0.3)
        
        # トレンド信号パネルに参照線を追加
        trend_signal_panel_idx = 3 + panel_offset
        axes[trend_signal_panel_idx].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[trend_signal_panel_idx].axhline(y=1, color='green', linestyle='--', alpha=0.5)
        axes[trend_signal_panel_idx].axhline(y=-1, color='red', linestyle='--', alpha=0.5)
        
        # 動的期間パネルに参照線を追加（存在する場合）
        if not df['dynamic_periods'].isna().all():
            dynamic_panel_idx = 4 + panel_offset
            period_mean = df['dynamic_periods'].mean()
            if not np.isnan(period_mean):
                axes[dynamic_panel_idx].axhline(y=period_mean, color='black', linestyle='-', alpha=0.3)
        
        # 統計情報の表示
        print(f"\n=== CHOP_ER統計 ===")
        total_points = len(df[df['trend_signals'] != 0])
        uptrend_points = len(df[df['trend_signals'] == 1])
        downtrend_points = len(df[df['trend_signals'] == -1])
        range_points = len(df[df['trend_signals'] == 0])
        
        print(f"総データ点数: {len(df)}")
        print(f"有効トレンド点数: {total_points}")
        print(f"上昇トレンド: {uptrend_points} ({uptrend_points/len(df)*100:.1f}%)")
        print(f"下降トレンド: {downtrend_points} ({downtrend_points/len(df)*100:.1f}%)")
        print(f"レンジ状態: {range_points} ({range_points/len(df)*100:.1f}%)")
        
        # CHOP_ER値の統計
        valid_chop_er = df['chop_er'].dropna()
        if len(valid_chop_er) > 0:
            print(f"CHOP_ER統計 - 平均: {valid_chop_er.mean():.3f}, 中央値: {valid_chop_er.median():.3f}")
            print(f"CHOP_ER範囲: {valid_chop_er.min():.3f} - {valid_chop_er.max():.3f}")
            print(f"しきい値({fixed_threshold})以上: {(valid_chop_er >= fixed_threshold).sum()} ({(valid_chop_er >= fixed_threshold).mean()*100:.1f}%)")
        
        # CHOPトレンド値の統計
        valid_chop_trend = df['chop_trend'].dropna()
        if len(valid_chop_trend) > 0:
            print(f"CHOPトレンド統計 - 平均: {valid_chop_trend.mean():.3f}, 範囲: {valid_chop_trend.min():.3f} - {valid_chop_trend.max():.3f}")
        
        # 動的期間の統計（存在する場合）
        if not df['dynamic_periods'].isna().all():
            valid_periods = df['dynamic_periods'].dropna()
            if len(valid_periods) > 0:
                print(f"動的期間統計 - 平均: {valid_periods.mean():.1f}, 範囲: {valid_periods.min():.0f} - {valid_periods.max():.0f}")
        
        # 直近のデータを出力
        self._print_recent_values(df, recent_bars)
        
        # 保存または表示
        if savefig:
            plt.savefig(savefig, dpi=150, bbox_inches='tight')
            print(f"チャートを保存しました: {savefig}")
        else:
            plt.tight_layout()
            plt.show()

    def _print_recent_values(self, df: pd.DataFrame, num_bars: int = 100) -> None:
        """
        直近のデータ値を表形式で出力する
        
        Args:
            df: チャートデータのDataFrame
            num_bars: 出力する本数（デフォルト: 100）
        """
        print(f"\n=== CHOP_ER 直近{num_bars}本のデータ ===")
        
        # 直近のデータを取得
        recent_df = df.tail(num_bars).copy()
        
        if len(recent_df) == 0:
            print("表示可能なデータがありません。")
            return
        
        # 表示用の列を選択・整理
        display_columns = []
        column_headers = []
        
        # 日時
        display_columns.append(recent_df.index.strftime('%Y-%m-%d %H:%M'))
        column_headers.append('Date/Time')
        
        # CHOP_ER値
        if 'chop_er' in recent_df.columns:
            display_columns.append(recent_df['chop_er'].round(4))
            column_headers.append('CHOP_ER')
        
        # CHOPトレンド値
        if 'chop_trend' in recent_df.columns:
            display_columns.append(recent_df['chop_trend'].round(4))
            column_headers.append('CHOP_Trend')
        
        # トレンド信号
        if 'trend_signals' in recent_df.columns:
            trend_labels = recent_df['trend_signals'].map({1: 'Up', -1: 'Down', 0: 'Range'})
            display_columns.append(trend_labels)
            column_headers.append('Trend_Signal')
        
        # 動的期間（存在する場合）
        if 'dynamic_periods' in recent_df.columns and not recent_df['dynamic_periods'].isna().all():
            display_columns.append(recent_df['dynamic_periods'].round(1))
            column_headers.append('Dynamic_Period')
        
        # 固定しきい値との比較
        if 'chop_er' in recent_df.columns:
            fixed_threshold = self.chop_er.get_fixed_threshold()
            threshold_status = recent_df['chop_er'].apply(
                lambda x: 'Above' if x >= fixed_threshold else 'Below' if not pd.isna(x) else 'NaN'
            )
            display_columns.append(threshold_status)
            column_headers.append(f'vs_Threshold({fixed_threshold})')
        
        # DataFrameとして整理
        display_df = pd.DataFrame(dict(zip(column_headers, display_columns)))
        
        # 表示設定
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 200)
        pd.set_option('display.max_colwidth', 20)
        
        print(display_df.to_string(index=False))
        
        # 直近の値のサマリー
        if 'chop_er' in recent_df.columns:
            latest_chop_er = recent_df['chop_er'].iloc[-1]
            latest_trend = recent_df['trend_signals'].iloc[-1]
            trend_label = {1: '上昇', -1: '下降', 0: 'レンジ'}.get(latest_trend, '不明')
            
            print(f"\n【直近の状態】")
            print(f"CHOP_ER: {latest_chop_er:.4f}")
            print(f"トレンド: {trend_label} ({latest_trend})")
            
            if 'chop_trend' in recent_df.columns:
                latest_chop_trend = recent_df['chop_trend'].iloc[-1]
                print(f"CHOPトレンド: {latest_chop_trend:.4f}")
            
            if 'dynamic_periods' in recent_df.columns and not recent_df['dynamic_periods'].isna().all():
                latest_period = recent_df['dynamic_periods'].iloc[-1]
                if not pd.isna(latest_period):
                    print(f"動的期間: {latest_period:.1f}")
        
        # 設定をリセット
        pd.reset_option('display.max_rows')
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')
        pd.reset_option('display.max_colwidth')


def main():
    """メイン関数"""
    # コマンドライン引数を処理
    import argparse
    parser = argparse.ArgumentParser(description='CHOP_ERインジケーターの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--er-period', type=int, default=13, help='Efficiency Ratio期間')
    parser.add_argument('--smoothing', type=str, default='hma', help='スムージング方法')
    parser.add_argument('--dynamic-er', action='store_true', help='動的ER期間を使用')
    parser.add_argument('--threshold', type=float, default=0.618, help='固定しきい値')
    parser.add_argument('--slope-index', type=int, default=3, help='トレンド判定期間')
    parser.add_argument('--continuous-line', action='store_true', help='連続ライン表示を有効にする')
    parser.add_argument('--recent-bars', type=int, default=100, help='出力する直近のバー数（デフォルト: 100）')
    args = parser.parse_args()
    
    # チャートを作成
    chart = ChopERChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        er_period=args.er_period,
        smoothing_method=args.smoothing,
        use_dynamic_er_period=args.dynamic_er,
        fixed_threshold=args.threshold,
        slope_index=args.slope_index
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output,
        show_continuous_line=args.continuous_line,
        recent_bars=args.recent_bars
    )


if __name__ == "__main__":
    main() 