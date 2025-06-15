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
from indicators.adx_vol import ADXVol


class ADXVolChart:
    """
    ADX_VOL（ADX Volume Trend Indicator）を表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - ADX_VOL値（正規化ADX × 標準偏差係数）のプロット
    - 正規化ADX値のプロット
    - 標準偏差係数のプロット
    - ATR値のプロット
    - トレンド信号のカラー表示
    - 動的期間値の表示（使用している場合）
    - 固定しきい値の参照線
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.adx_vol = None
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
                            # ADXパラメータ
                            period: int = 13,
                            use_dynamic_period: bool = True,
                            
                            # EhlersUnifiedDC パラメータ
                            detector_type: str = 'cycle_period2',
                            cycle_part: float = 0.5,
                            max_cycle: int = 55,
                            min_cycle: int = 5,
                            max_output: int = 15,
                            min_output: int = 3,
                            src_type: str = 'hlc3',
                            lp_period: int = 10,
                            hp_period: int = 48,

                            # ATR パラメータ
                            atr_period: int = 13,
                            atr_smoothing_method: str = 'alma',
                            use_dynamic_atr_period: bool = True,
                            
                            # トレンド判定パラメータ
                            slope_index: int = 1,
                            range_threshold: float = 0.005,

                            # 固定しきい値のパラメータ
                            fixed_threshold: float = 0.25
                           ) -> None:
        """
        ADX_VOLインジケーターを計算する
        
        Args:
            period: ADXの期間（デフォルト: 13）
            use_dynamic_period: 動的ADX期間を使用するかどうか（デフォルト: True）
            detector_type: EhlersUnifiedDCで使用する検出器タイプ（デフォルト: 'cycle_period2'）
            cycle_part: DCのサイクル部分の倍率（デフォルト: 0.5）
            max_cycle: DCの最大サイクル期間（デフォルト: 55）
            min_cycle: DCの最小サイクル期間（デフォルト: 5）
            max_output: DCの最大出力値（デフォルト: 15）
            min_output: DCの最小出力値（デフォルト: 3）
            src_type: DC計算に使用する価格ソース（デフォルト: 'hlc3'）
            lp_period: 拡張DC用のローパスフィルター期間（デフォルト: 10）
            hp_period: 拡張DC用のハイパスフィルター期間（デフォルト: 48）
            atr_period: ATRの期間（デフォルト: 13）
            atr_smoothing_method: ATRで使用する平滑化アルゴリズム ('alma', 'hma', 'zlema', 'wilder', 'none')
            use_dynamic_atr_period: 動的ATR期間を使用するかどうか（デフォルト: True）
            slope_index: トレンド判定期間（デフォルト: 1）
            range_threshold: range判定の基本閾値（デフォルト: 0.005）
            fixed_threshold: 固定しきい値（デフォルト: 0.25）
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\nADX_VOLインジケーターを計算中...")
        
        # ADX_VOLインジケーターを初期化
        self.adx_vol = ADXVol(
            period=period,
            use_dynamic_period=use_dynamic_period,
            detector_type=detector_type,
            cycle_part=cycle_part,
            max_cycle=max_cycle,
            min_cycle=min_cycle,
            max_output=max_output,
            min_output=min_output,
            src_type=src_type,
            lp_period=lp_period,
            hp_period=hp_period,
            atr_period=atr_period,
            atr_smoothing_method=atr_smoothing_method,
            use_dynamic_atr_period=use_dynamic_atr_period,
            slope_index=slope_index,
            range_threshold=range_threshold,
            fixed_threshold=fixed_threshold
        )
        
        # ADX_VOLの計算
        print("計算を実行します...")
        result = self.adx_vol.calculate(self.data)
        
        # 計算結果の取得テスト
        adx_vol_values = self.adx_vol.get_values()
        normalized_adx = self.adx_vol.get_normalized_adx()
        stddev_factor = self.adx_vol.get_stddev_factor()
        atr_values = self.adx_vol.get_atr()
        trend_signals = self.adx_vol.get_trend_signals()
        dynamic_periods = self.adx_vol.get_dynamic_atr_period()
        
        print(f"ADX_VOL計算完了 - ADX_VOL: {len(adx_vol_values)}, 正規化ADX: {len(normalized_adx)}")
        print(f"標準偏差係数: {len(stddev_factor)}, ATR: {len(atr_values)}")
        print(f"トレンド信号: {len(trend_signals)}, 動的期間: {len(dynamic_periods)}")
        
        # NaN値のチェック
        nan_count_adx_vol = np.isnan(adx_vol_values).sum()
        nan_count_norm_adx = np.isnan(normalized_adx).sum()
        nan_count_stddev = np.isnan(stddev_factor).sum()
        trend_count = (trend_signals != 0).sum()
        print(f"NaN値 - ADX_VOL: {nan_count_adx_vol}, 正規化ADX: {nan_count_norm_adx}, 標準偏差係数: {nan_count_stddev}")
        print(f"トレンド信号 - 有効: {trend_count}, 上昇: {(trend_signals == 1).sum()}, 下降: {(trend_signals == -1).sum()}")
        
        # 統計情報
        valid_adx_vol = adx_vol_values[~np.isnan(adx_vol_values)]
        if len(valid_adx_vol) > 0:
            print(f"ADX_VOL統計 - 平均: {valid_adx_vol.mean():.3f}, 範囲: {valid_adx_vol.min():.3f} - {valid_adx_vol.max():.3f}")
        
        valid_norm_adx = normalized_adx[~np.isnan(normalized_adx)]
        if len(valid_norm_adx) > 0:
            print(f"正規化ADX統計 - 平均: {valid_norm_adx.mean():.3f}, 範囲: {valid_norm_adx.min():.3f} - {valid_norm_adx.max():.3f}")
        
        print("ADX_VOL計算完了")
            
    def plot(self, 
            title: str = "ADX_VOL (ADX Volume Trend Indicator)", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (14, 14),
            style: str = 'yahoo',
            savefig: Optional[str] = None,
            show_continuous_line: bool = True,
            recent_bars: int = 100) -> None:
        """
        ローソク足チャートとADX_VOLを描画する
        
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
            
        if self.adx_vol is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # ADX_VOLの値を取得
        print("ADX_VOLデータを取得中...")
        adx_vol_values = self.adx_vol.get_values()
        normalized_adx = self.adx_vol.get_normalized_adx()
        stddev_factor = self.adx_vol.get_stddev_factor()
        atr_values = self.adx_vol.get_atr()
        trend_signals = self.adx_vol.get_trend_signals()
        dynamic_periods = self.adx_vol.get_dynamic_atr_period()
        fixed_threshold = self.adx_vol.get_fixed_threshold()
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'adx_vol': adx_vol_values,
                'normalized_adx': normalized_adx,
                'stddev_factor': stddev_factor,
                'atr': atr_values,
                'trend_signals': trend_signals,
                'dynamic_periods': dynamic_periods if len(dynamic_periods) > 0 else np.full(len(adx_vol_values), np.nan)
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        print(f"ADX_VOLデータ確認 - ADX_VOL NaN: {df['adx_vol'].isna().sum()}, 正規化ADX NaN: {df['normalized_adx'].isna().sum()}")
        
        # トレンド信号に基づく色分け用データの準備
        df['adx_vol_up'] = np.where(df['trend_signals'] == 1, df['adx_vol'], np.nan)
        df['adx_vol_down'] = np.where(df['trend_signals'] == -1, df['adx_vol'], np.nan)
        df['adx_vol_range'] = np.where(df['trend_signals'] == 0, df['adx_vol'], np.nan)
        
        # 正規化ADXの色分け
        df['norm_adx_up'] = np.where(df['trend_signals'] == 1, df['normalized_adx'], np.nan)
        df['norm_adx_down'] = np.where(df['trend_signals'] == -1, df['normalized_adx'], np.nan)
        df['norm_adx_range'] = np.where(df['trend_signals'] == 0, df['normalized_adx'], np.nan)
        
        # 標準偏差係数の色分け
        df['stddev_up'] = np.where(df['trend_signals'] == 1, df['stddev_factor'], np.nan)
        df['stddev_down'] = np.where(df['trend_signals'] == -1, df['stddev_factor'], np.nan)
        df['stddev_range'] = np.where(df['trend_signals'] == 0, df['stddev_factor'], np.nan)
        
        # ATRの色分け
        df['atr_up'] = np.where(df['trend_signals'] == 1, df['atr'], np.nan)
        df['atr_down'] = np.where(df['trend_signals'] == -1, df['atr'], np.nan)
        df['atr_range'] = np.where(df['trend_signals'] == 0, df['atr'], np.nan)
        
        # NaN値を含む行を出力（最初の5行のみ）
        nan_rows = df[df['adx_vol'].isna() | df['normalized_adx'].isna()]
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
            # ADX_VOLパネル - 連続ライン（薄い色）
            adx_vol_continuous = mpf.make_addplot(df['adx_vol'], panel=1, color='lightblue', width=0.8, alpha=0.6,
                                                 ylabel='ADX_VOL', secondary_y=False, label='ADX_VOL (Continuous)')
            plots_list.append(adx_vol_continuous)
            
            # ADX_VOLパネル - 色分け表示（太い線）
            plots_list.extend([
                mpf.make_addplot(df['adx_vol_up'], panel=1, color='green', width=2.0, 
                               secondary_y=False, label='ADX_VOL (Up)'),
                mpf.make_addplot(df['adx_vol_down'], panel=1, color='red', width=2.0, 
                               secondary_y=False, label='ADX_VOL (Down)'),
                mpf.make_addplot(df['adx_vol_range'], panel=1, color='gray', width=1.5, 
                               secondary_y=False, label='ADX_VOL (Range)')
            ])
            
            # 正規化ADXパネル - 連続ライン（薄い色）
            norm_adx_continuous = mpf.make_addplot(df['normalized_adx'], panel=2, color='lightgreen', width=0.8, alpha=0.6,
                                                  ylabel='Normalized ADX', secondary_y=False, label='Norm ADX (Continuous)')
            plots_list.append(norm_adx_continuous)
            
            # 正規化ADXパネル - 色分け表示（太い線）
            plots_list.extend([
                mpf.make_addplot(df['norm_adx_up'], panel=2, color='darkgreen', width=2.0, 
                               secondary_y=False, label='Norm ADX (Up)'),
                mpf.make_addplot(df['norm_adx_down'], panel=2, color='darkred', width=2.0, 
                               secondary_y=False, label='Norm ADX (Down)'),
                mpf.make_addplot(df['norm_adx_range'], panel=2, color='darkgray', width=1.5, 
                               secondary_y=False, label='Norm ADX (Range)')
            ])
            
            # 標準偏差係数パネル - 連続ライン（薄い色）
            stddev_continuous = mpf.make_addplot(df['stddev_factor'], panel=3, color='lightcoral', width=0.8, alpha=0.6,
                                               ylabel='StdDev Factor', secondary_y=False, label='StdDev (Continuous)')
            plots_list.append(stddev_continuous)
            
            # 標準偏差係数パネル - 色分け表示（太い線）
            plots_list.extend([
                mpf.make_addplot(df['stddev_up'], panel=3, color='blue', width=2.0, 
                               secondary_y=False, label='StdDev (Up)'),
                mpf.make_addplot(df['stddev_down'], panel=3, color='purple', width=2.0, 
                               secondary_y=False, label='StdDev (Down)'),
                mpf.make_addplot(df['stddev_range'], panel=3, color='lightgray', width=1.5, 
                               secondary_y=False, label='StdDev (Range)')
            ])
            
            # ATRパネル - 連続ライン（薄い色）
            atr_continuous = mpf.make_addplot(df['atr'], panel=4, color='lightyellow', width=0.8, alpha=0.6,
                                            ylabel='ATR', secondary_y=False, label='ATR (Continuous)')
            plots_list.append(atr_continuous)
            
            # ATRパネル - 色分け表示（太い線）
            plots_list.extend([
                mpf.make_addplot(df['atr_up'], panel=4, color='cyan', width=2.0, 
                               secondary_y=False, label='ATR (Up)'),
                mpf.make_addplot(df['atr_down'], panel=4, color='magenta', width=2.0, 
                               secondary_y=False, label='ATR (Down)'),
                mpf.make_addplot(df['atr_range'], panel=4, color='lightsteelblue', width=1.5, 
                               secondary_y=False, label='ATR (Range)')
            ])
        else:
            # 色分け表示のみ
            # ADX_VOLパネル（トレンド信号による色分け）
            plots_list.extend([
                mpf.make_addplot(df['adx_vol_up'], panel=1, color='green', width=1.5, 
                               ylabel='ADX_VOL', secondary_y=False, label='ADX_VOL (Up)'),
                mpf.make_addplot(df['adx_vol_down'], panel=1, color='red', width=1.5, 
                               secondary_y=False, label='ADX_VOL (Down)'),
                mpf.make_addplot(df['adx_vol_range'], panel=1, color='gray', width=1.0, 
                               secondary_y=False, label='ADX_VOL (Range)')
            ])
            
            # 正規化ADXパネル（トレンド信号による色分け）
            plots_list.extend([
                mpf.make_addplot(df['norm_adx_up'], panel=2, color='darkgreen', width=1.2, 
                               ylabel='Normalized ADX', secondary_y=False, label='Norm ADX (Up)'),
                mpf.make_addplot(df['norm_adx_down'], panel=2, color='darkred', width=1.2, 
                               secondary_y=False, label='Norm ADX (Down)'),
                mpf.make_addplot(df['norm_adx_range'], panel=2, color='darkgray', width=1.0, 
                               secondary_y=False, label='Norm ADX (Range)')
            ])
            
            # 標準偏差係数パネル（トレンド信号による色分け）
            plots_list.extend([
                mpf.make_addplot(df['stddev_up'], panel=3, color='blue', width=1.2, 
                               secondary_y=False, label='StdDev (Up)'),
                mpf.make_addplot(df['stddev_down'], panel=3, color='purple', width=1.2, 
                               secondary_y=False, label='StdDev (Down)'),
                mpf.make_addplot(df['stddev_range'], panel=3, color='lightgray', width=1.0, 
                               secondary_y=False, label='StdDev (Range)')
            ])
            
            # ATRパネル（トレンド信号による色分け）
            plots_list.extend([
                mpf.make_addplot(df['atr_up'], panel=4, color='cyan', width=1.2, 
                               secondary_y=False, label='ATR (Up)'),
                mpf.make_addplot(df['atr_down'], panel=4, color='magenta', width=1.2, 
                               secondary_y=False, label='ATR (Down)'),
                mpf.make_addplot(df['atr_range'], panel=4, color='lightsteelblue', width=1.0, 
                               secondary_y=False, label='ATR (Range)')
            ])
        
        # トレンド信号パネル
        trend_signal_panel = mpf.make_addplot(df['trend_signals'], panel=5, color='orange', width=1.5, 
                                             ylabel='Trend Signal', secondary_y=False, label='Trend Signal', type='line')
        plots_list.append(trend_signal_panel)
        
        # 動的期間パネル（動的期間モードの場合のみ）
        if not df['dynamic_periods'].isna().all():
            dynamic_period_panel = mpf.make_addplot(df['dynamic_periods'], panel=6, color='brown', width=1.2, 
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
                kwargs['panel_ratios'] = (4, 1, 1.2, 1.2, 1, 1, 0.8, 0.8)  # メイン:出来高:ADX_VOL:正規化ADX:標準偏差:ATR:信号:期間
                # 出来高を表示する場合は、オシレーターのパネル番号を+1する
                plots_list = []
                # ADX_VOLパネル (panel=2)
                plots_list.extend([
                    mpf.make_addplot(df['adx_vol_up'], panel=2, color='green', width=1.5, 
                                    ylabel='ADX_VOL', secondary_y=False, label='ADX_VOL (Up)'),
                    mpf.make_addplot(df['adx_vol_down'], panel=2, color='red', width=1.5, 
                                    secondary_y=False, label='ADX_VOL (Down)'),
                    mpf.make_addplot(df['adx_vol_range'], panel=2, color='gray', width=1.0, 
                                    secondary_y=False, label='ADX_VOL (Range)')
                ])
                
                # 正規化ADXパネル (panel=3)
                plots_list.extend([
                    mpf.make_addplot(df['norm_adx_up'], panel=3, color='darkgreen', width=1.2, 
                                    ylabel='Normalized ADX', secondary_y=False, label='Norm ADX (Up)'),
                    mpf.make_addplot(df['norm_adx_down'], panel=3, color='darkred', width=1.2, 
                                    secondary_y=False, label='Norm ADX (Down)'),
                    mpf.make_addplot(df['norm_adx_range'], panel=3, color='darkgray', width=1.0, 
                                    secondary_y=False, label='Norm ADX (Range)')
                ])
                
                # 標準偏差係数パネル (panel=4)
                plots_list.extend([
                    mpf.make_addplot(df['stddev_up'], panel=4, color='blue', width=1.2, 
                                    ylabel='StdDev Factor', secondary_y=False, label='StdDev (Up)'),
                    mpf.make_addplot(df['stddev_down'], panel=4, color='purple', width=1.2, 
                                    secondary_y=False, label='StdDev (Down)'),
                    mpf.make_addplot(df['stddev_range'], panel=4, color='lightgray', width=1.0, 
                                    secondary_y=False, label='StdDev (Range)')
                ])
                
                # ATRパネル (panel=5)
                plots_list.extend([
                    mpf.make_addplot(df['atr_up'], panel=5, color='cyan', width=1.2, 
                                    ylabel='ATR', secondary_y=False, label='ATR (Up)'),
                    mpf.make_addplot(df['atr_down'], panel=5, color='magenta', width=1.2, 
                                    secondary_y=False, label='ATR (Down)'),
                    mpf.make_addplot(df['atr_range'], panel=5, color='lightsteelblue', width=1.0, 
                                    secondary_y=False, label='ATR (Range)')
                ])
                
                # トレンド信号パネル (panel=6)
                plots_list.append(mpf.make_addplot(df['trend_signals'], panel=6, color='orange', width=1.5, 
                                                  ylabel='Trend Signal', secondary_y=False, label='Trend Signal', type='line'))
                
                # 動的期間パネル (panel=7)
                plots_list.append(mpf.make_addplot(df['dynamic_periods'], panel=7, color='brown', width=1.2, 
                                                  ylabel='Dynamic Period', secondary_y=False, label='Dynamic Period'))
            else:
                kwargs['volume'] = True
                kwargs['panel_ratios'] = (4, 1, 1.2, 1.2, 1, 1, 0.8)  # メイン:出来高:ADX_VOL:正規化ADX:標準偏差:ATR:信号
                plots_list = []
                # ADX_VOLパネル (panel=2)
                plots_list.extend([
                    mpf.make_addplot(df['adx_vol_up'], panel=2, color='green', width=1.5, 
                                    ylabel='ADX_VOL', secondary_y=False, label='ADX_VOL (Up)'),
                    mpf.make_addplot(df['adx_vol_down'], panel=2, color='red', width=1.5, 
                                    secondary_y=False, label='ADX_VOL (Down)'),
                    mpf.make_addplot(df['adx_vol_range'], panel=2, color='gray', width=1.0, 
                                    secondary_y=False, label='ADX_VOL (Range)')
                ])
                
                # 正規化ADXパネル (panel=3)
                plots_list.extend([
                    mpf.make_addplot(df['norm_adx_up'], panel=3, color='darkgreen', width=1.2, 
                                    ylabel='Normalized ADX', secondary_y=False, label='Norm ADX (Up)'),
                    mpf.make_addplot(df['norm_adx_down'], panel=3, color='darkred', width=1.2, 
                                    secondary_y=False, label='Norm ADX (Down)'),
                    mpf.make_addplot(df['norm_adx_range'], panel=3, color='darkgray', width=1.0, 
                                    secondary_y=False, label='Norm ADX (Range)')
                ])
                
                # 標準偏差係数パネル (panel=4)
                plots_list.extend([
                    mpf.make_addplot(df['stddev_up'], panel=4, color='blue', width=1.2, 
                                    ylabel='StdDev Factor', secondary_y=False, label='StdDev (Up)'),
                    mpf.make_addplot(df['stddev_down'], panel=4, color='purple', width=1.2, 
                                    secondary_y=False, label='StdDev (Down)'),
                    mpf.make_addplot(df['stddev_range'], panel=4, color='lightgray', width=1.0, 
                                    secondary_y=False, label='StdDev (Range)')
                ])
                
                # ATRパネル (panel=5)
                plots_list.extend([
                    mpf.make_addplot(df['atr_up'], panel=5, color='cyan', width=1.2, 
                                    ylabel='ATR', secondary_y=False, label='ATR (Up)'),
                    mpf.make_addplot(df['atr_down'], panel=5, color='magenta', width=1.2, 
                                    secondary_y=False, label='ATR (Down)'),
                    mpf.make_addplot(df['atr_range'], panel=5, color='lightsteelblue', width=1.0, 
                                    secondary_y=False, label='ATR (Range)')
                ])
                
                # トレンド信号パネル (panel=6)
                plots_list.append(mpf.make_addplot(df['trend_signals'], panel=6, color='orange', width=1.5, 
                                                  ylabel='Trend Signal', secondary_y=False, label='Trend Signal', type='line'))
        else:
            if not df['dynamic_periods'].isna().all():
                kwargs['volume'] = False
                kwargs['panel_ratios'] = (4, 1.2, 1.2, 1, 1, 0.8, 0.8)  # メイン:ADX_VOL:正規化ADX:標準偏差:ATR:信号:期間
            else:
                kwargs['volume'] = False
                kwargs['panel_ratios'] = (4, 1.2, 1.2, 1, 1, 0.8)  # メイン:ADX_VOL:正規化ADX:標準偏差:ATR:信号
        
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
            
        # ADX_VOLパネルに固定しきい値の線を追加
        adx_vol_panel_idx = 1 + panel_offset
        axes[adx_vol_panel_idx].axhline(y=fixed_threshold, color='black', linestyle='--', alpha=0.7, label=f'Threshold ({fixed_threshold})')
        axes[adx_vol_panel_idx].axhline(y=0.0, color='black', linestyle='-', alpha=0.3)
        axes[adx_vol_panel_idx].axhline(y=1.0, color='black', linestyle='-', alpha=0.3)
        
        # 正規化ADXパネルに参照線を追加
        norm_adx_panel_idx = 2 + panel_offset
        axes[norm_adx_panel_idx].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
        axes[norm_adx_panel_idx].axhline(y=0.0, color='black', linestyle='-', alpha=0.3)
        axes[norm_adx_panel_idx].axhline(y=1.0, color='black', linestyle='-', alpha=0.3)
        
        # 標準偏差係数パネルに参照線を追加
        stddev_panel_idx = 3 + panel_offset
        axes[stddev_panel_idx].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        stddev_mean = df['stddev_factor'].mean()
        if not np.isnan(stddev_mean):
            axes[stddev_panel_idx].axhline(y=stddev_mean, color='black', linestyle='-', alpha=0.3)
        
        # ATRパネルに参照線を追加
        atr_panel_idx = 4 + panel_offset
        atr_mean = df['atr'].mean()
        if not np.isnan(atr_mean):
            axes[atr_panel_idx].axhline(y=atr_mean, color='black', linestyle='-', alpha=0.3)
        
        # トレンド信号パネルに参照線を追加
        trend_signal_panel_idx = 5 + panel_offset
        axes[trend_signal_panel_idx].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[trend_signal_panel_idx].axhline(y=1, color='green', linestyle='--', alpha=0.5)
        axes[trend_signal_panel_idx].axhline(y=-1, color='red', linestyle='--', alpha=0.5)
        
        # 動的期間パネルに参照線を追加（存在する場合）
        if not df['dynamic_periods'].isna().all():
            dynamic_panel_idx = 6 + panel_offset
            period_mean = df['dynamic_periods'].mean()
            if not np.isnan(period_mean):
                axes[dynamic_panel_idx].axhline(y=period_mean, color='black', linestyle='-', alpha=0.3)
        
        # 統計情報の表示
        print(f"\n=== ADX_VOL統計 ===")
        total_points = len(df[df['trend_signals'] != 0])
        uptrend_points = len(df[df['trend_signals'] == 1])
        downtrend_points = len(df[df['trend_signals'] == -1])
        range_points = len(df[df['trend_signals'] == 0])
        
        print(f"総データ点数: {len(df)}")
        print(f"有効トレンド点数: {total_points}")
        print(f"上昇トレンド: {uptrend_points} ({uptrend_points/len(df)*100:.1f}%)")
        print(f"下降トレンド: {downtrend_points} ({downtrend_points/len(df)*100:.1f}%)")
        print(f"レンジ状態: {range_points} ({range_points/len(df)*100:.1f}%)")
        
        # ADX_VOL値の統計
        valid_adx_vol = df['adx_vol'].dropna()
        if len(valid_adx_vol) > 0:
            print(f"ADX_VOL統計 - 平均: {valid_adx_vol.mean():.3f}, 中央値: {valid_adx_vol.median():.3f}")
            print(f"ADX_VOL範囲: {valid_adx_vol.min():.3f} - {valid_adx_vol.max():.3f}")
            print(f"しきい値({fixed_threshold})以上: {(valid_adx_vol >= fixed_threshold).sum()} ({(valid_adx_vol >= fixed_threshold).mean()*100:.1f}%)")
        
        # 正規化ADX値の統計
        valid_norm_adx = df['normalized_adx'].dropna()
        if len(valid_norm_adx) > 0:
            print(f"正規化ADX統計 - 平均: {valid_norm_adx.mean():.3f}, 範囲: {valid_norm_adx.min():.3f} - {valid_norm_adx.max():.3f}")
        
        # 標準偏差係数の統計
        valid_stddev = df['stddev_factor'].dropna()
        if len(valid_stddev) > 0:
            print(f"標準偏差係数統計 - 平均: {valid_stddev.mean():.3f}, 範囲: {valid_stddev.min():.3f} - {valid_stddev.max():.3f}")
        
        # ATRの統計
        valid_atr = df['atr'].dropna()
        if len(valid_atr) > 0:
            print(f"ATR統計 - 平均: {valid_atr.mean():.3f}, 範囲: {valid_atr.min():.3f} - {valid_atr.max():.3f}")
        
        # 動的期間の統計（存在する場合）
        if not df['dynamic_periods'].isna().all():
            valid_periods = df['dynamic_periods'].dropna()
            if len(valid_periods) > 0:
                print(f"動的ATR期間統計 - 平均: {valid_periods.mean():.1f}, 範囲: {valid_periods.min():.0f} - {valid_periods.max():.0f}")
        
        # 直近100本のデータを出力
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
        print(f"\n=== ADX_VOL 直近{num_bars}本のデータ ===")
        
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
        
        # ADX_VOL値
        if 'adx_vol' in recent_df.columns:
            display_columns.append(recent_df['adx_vol'].round(4))
            column_headers.append('ADX_VOL')
        
        # 正規化ADX値
        if 'normalized_adx' in recent_df.columns:
            display_columns.append(recent_df['normalized_adx'].round(4))
            column_headers.append('Norm_ADX')
        
        # 標準偏差係数
        if 'stddev_factor' in recent_df.columns:
            display_columns.append(recent_df['stddev_factor'].round(4))
            column_headers.append('StdDev_Factor')
        
        # ATR値
        if 'atr' in recent_df.columns:
            display_columns.append(recent_df['atr'].round(6))
            column_headers.append('ATR')
        
        # トレンド信号
        if 'trend_signals' in recent_df.columns:
            trend_labels = recent_df['trend_signals'].map({1: 'Up', -1: 'Down', 0: 'Range'})
            display_columns.append(trend_labels)
            column_headers.append('Trend_Signal')
        
        # 動的ATR期間（存在する場合）
        if 'dynamic_periods' in recent_df.columns and not recent_df['dynamic_periods'].isna().all():
            display_columns.append(recent_df['dynamic_periods'].round(1))
            column_headers.append('Dynamic_ATR_Period')
        
        # 固定しきい値との比較
        if 'adx_vol' in recent_df.columns:
            fixed_threshold = self.adx_vol.get_fixed_threshold()
            threshold_status = recent_df['adx_vol'].apply(
                lambda x: 'Above' if x >= fixed_threshold else 'Below' if not pd.isna(x) else 'NaN'
            )
            display_columns.append(threshold_status)
            column_headers.append(f'vs_Threshold({fixed_threshold})')
        
        # DataFrameとして整理
        display_df = pd.DataFrame(dict(zip(column_headers, display_columns)))
        
        # 表示設定
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 250)
        pd.set_option('display.max_colwidth', 20)
        
        print(display_df.to_string(index=False))
        
        # 直近の値のサマリー
        if 'adx_vol' in recent_df.columns:
            latest_adx_vol = recent_df['adx_vol'].iloc[-1]
            latest_trend = recent_df['trend_signals'].iloc[-1]
            trend_label = {1: '上昇', -1: '下降', 0: 'レンジ'}.get(latest_trend, '不明')
            
            print(f"\n【直近の状態】")
            print(f"ADX_VOL: {latest_adx_vol:.4f}")
            print(f"トレンド: {trend_label} ({latest_trend})")
            
            if 'normalized_adx' in recent_df.columns:
                latest_norm_adx = recent_df['normalized_adx'].iloc[-1]
                print(f"正規化ADX: {latest_norm_adx:.4f}")
            
            if 'stddev_factor' in recent_df.columns:
                latest_stddev = recent_df['stddev_factor'].iloc[-1]
                print(f"標準偏差係数: {latest_stddev:.4f}")
            
            if 'atr' in recent_df.columns:
                latest_atr = recent_df['atr'].iloc[-1]
                print(f"ATR: {latest_atr:.6f}")
            
            if 'dynamic_periods' in recent_df.columns and not recent_df['dynamic_periods'].isna().all():
                latest_period = recent_df['dynamic_periods'].iloc[-1]
                if not pd.isna(latest_period):
                    print(f"動的ATR期間: {latest_period:.1f}")
        
        # 設定をリセット
        pd.reset_option('display.max_rows')
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')
        pd.reset_option('display.max_colwidth')


def main():
    """メイン関数"""
    # コマンドライン引数を処理
    import argparse
    parser = argparse.ArgumentParser(description='ADX_VOLインジケーターの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--period', type=int, default=13, help='ADX期間')
    parser.add_argument('--dynamic-adx', action='store_true', help='動的ADX期間を使用')
    parser.add_argument('--atr-period', type=int, default=13, help='ATR期間')
    parser.add_argument('--atr-smoothing', type=str, default='alma', help='ATRスムージング方法')
    parser.add_argument('--dynamic-atr', action='store_true', help='動的ATR期間を使用')
    parser.add_argument('--threshold', type=float, default=0.25, help='固定しきい値')
    parser.add_argument('--slope-index', type=int, default=1, help='トレンド判定期間')
    parser.add_argument('--continuous-line', action='store_true', help='連続ライン表示を有効にする')
    parser.add_argument('--recent-bars', type=int, default=100, help='出力する直近のバー数')
    args = parser.parse_args()
    
    # チャートを作成
    chart = ADXVolChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        period=args.period,
        use_dynamic_period=args.dynamic_adx,
        atr_period=args.atr_period,
        atr_smoothing_method=args.atr_smoothing,
        use_dynamic_atr_period=args.dynamic_atr,
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