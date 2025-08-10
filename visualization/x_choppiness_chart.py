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
from indicators.trend_filter.x_choppiness import XChoppiness


class XChoppinessChart:
    """
    Xチョピネスを表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - Xチョピネス値（0-1の範囲、高い値=トレンド、低い値=レンジ）
    - ミッドライン
    - トレンド信号（1=トレンド、-1=レンジ）
    - STR値
    - 平滑化チョピネス値（オプション）
    - 動的期間値（動的期間適応が有効な場合）
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.x_choppiness = None
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
                            period: int = 55,
                            midline_period: int = 100,
                            # STRパラメータ
                            str_period: float = 20.0,
                            str_src_type: str = 'close',
                            # 平滑化オプション
                            use_smoothing: bool = True,
                            smoother_type: str = 'super_smoother',
                            smoother_period: int = 10,
                            smoother_src_type: str = 'close',
                            # エラーズ統合サイクル検出器パラメータ
                            use_dynamic_period: bool = True,
                            detector_type: str = 'hody_e',
                            lp_period: int = 13,
                            hp_period: int = 124,
                            cycle_part: float = 0.5,
                            max_cycle: int = 124,
                            min_cycle: int = 13,
                            max_output: int = 124,
                            min_output: int = 13,
                            # 統合カルマンフィルターパラメータ
                            use_kalman_filter: bool = True,
                            kalman_filter_type: str = 'unscented',
                            kalman_process_noise: float = 0.01,
                            kalman_observation_noise: float = 0.001,
                            # パーセンタイル分析パラメータ
                            enable_percentile_analysis: bool = True,
                            percentile_lookback_period: int = 50,
                            percentile_low_threshold: float = 0.25,
                            percentile_high_threshold: float = 0.75
                           ) -> None:
        """
        Xチョピネスを計算する
        
        Args:
            period: Xチョピネス計算期間
            midline_period: ミッドライン計算期間
            str_period: STR期間
            str_src_type: STRソースタイプ
            use_smoothing: 平滑化を使用するか
            smoother_type: 統合スムーサータイプ
            smoother_period: スムーサー期間
            smoother_src_type: スムーサーソースタイプ
            use_dynamic_period: 動的期間適応を使用するか
            detector_type: サイクル検出器タイプ
            lp_period: ローパスフィルター期間
            hp_period: ハイパスフィルター期間
            cycle_part: サイクル部分
            max_cycle: 最大サイクル期間
            min_cycle: 最小サイクル期間
            max_output: 最大出力値
            min_output: 最小出力値
            use_kalman_filter: カルマンフィルターを使用するか
            kalman_filter_type: カルマンフィルタータイプ
            kalman_process_noise: カルマンフィルタープロセスノイズ
            kalman_observation_noise: カルマンフィルター観測ノイズ
            enable_percentile_analysis: パーセンタイル分析を有効にするか
            percentile_lookback_period: パーセンタイル計算のルックバック期間
            percentile_low_threshold: パーセンタイル低閾値
            percentile_high_threshold: パーセンタイル高閾値
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\nXチョピネスを計算中...")
        
        # Xチョピネスを計算
        self.x_choppiness = XChoppiness(
            period=period,
            midline_period=midline_period,
            str_period=str_period,
            str_src_type=str_src_type,
            use_smoothing=use_smoothing,
            smoother_type=smoother_type,
            smoother_period=smoother_period,
            smoother_src_type=smoother_src_type,
            use_dynamic_period=use_dynamic_period,
            detector_type=detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            max_cycle=max_cycle,
            min_cycle=min_cycle,
            max_output=max_output,
            min_output=min_output,
            use_kalman_filter=use_kalman_filter,
            kalman_filter_type=kalman_filter_type,
            kalman_process_noise=kalman_process_noise,
            kalman_observation_noise=kalman_observation_noise,
            enable_percentile_analysis=enable_percentile_analysis,
            percentile_lookback_period=percentile_lookback_period,
            percentile_low_threshold=percentile_low_threshold,
            percentile_high_threshold=percentile_high_threshold
        )
        
        # Xチョピネスの計算
        print("計算を実行します...")
        result = self.x_choppiness.calculate(self.data)
        
        print(f"Xチョピネス計算完了 - 値: {len(result.values)}")
        
        # NaN値のチェック
        nan_count = np.isnan(result.values).sum()
        valid_count = (~np.isnan(result.values)).sum()
        trend_count = (result.trend_signal != 0).sum()
        print(f"NaN値: {nan_count}, 有効値: {valid_count}")
        print(f"トレンド信号 - 有効: {trend_count}, トレンド: {(result.trend_signal == 1).sum()}, レンジ: {(result.trend_signal == -1).sum()}")
        
        # 統計情報
        if valid_count > 0:
            valid_values = result.values[~np.isnan(result.values)]
            print(f"Xチョピネス統計 - 平均: {np.mean(valid_values):.4f}, 範囲: {np.min(valid_values):.4f} - {np.max(valid_values):.4f}")
        
        print("Xチョピネス計算完了")
            
    def plot(self, 
            title: str = "Xチョピネス", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (14, 12),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとXチョピネスを描画する
        
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
            
        if self.x_choppiness is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # Xチョピネスの値を取得
        print("Xチョピネスデータを取得中...")
        result = self.x_choppiness.calculate(self.data)
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'x_choppiness': result.values,
                'raw_choppiness': result.raw_choppiness,
                'smoothed_choppiness': result.smoothed_choppiness,
                'midline': result.midline,
                'trend_signal': result.trend_signal,
                'str_values': result.str_values,
                # パーセンタイル分析結果を追加
                'percentiles': result.percentiles,
                'trend_state': result.trend_state,
                'trend_intensity': result.trend_intensity
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        print(f"Xチョピネスデータ確認 - NaN: {df['x_choppiness'].isna().sum()}")
        
        # トレンド/レンジ状態に基づく色分け
        df['chop_trend'] = np.where(df['trend_signal'] == 1, df['x_choppiness'], np.nan)
        df['chop_range'] = np.where(df['trend_signal'] == -1, df['x_choppiness'], np.nan)
        
        # ミッドラインの色分け（トレンド状態に応じて）
        df['midline_trend'] = np.where(df['trend_signal'] == 1, df['midline'], np.nan)
        df['midline_range'] = np.where(df['trend_signal'] == -1, df['midline'], np.nan)
        
        # パーセンタイル分析による色分け（パーセンタイル状態に応じて）
        if result.percentiles is not None and not np.all(result.percentiles == 0):
            df['percentile_low'] = np.where(df['trend_state'] == -1.0, df['percentiles'], np.nan)   # レンジ状態
            df['percentile_mid'] = np.where(df['trend_state'] == 0.0, df['percentiles'], np.nan)    # 中状態
            df['percentile_high'] = np.where(df['trend_state'] == 1.0, df['percentiles'], np.nan)   # トレンド状態
            
            # トレンド強度による色分け
            df['intensity_low'] = np.where(df['trend_state'] == -1.0, df['trend_intensity'], np.nan)
            df['intensity_mid'] = np.where(df['trend_state'] == 0.0, df['trend_intensity'], np.nan)
            df['intensity_high'] = np.where(df['trend_state'] == 1.0, df['trend_intensity'], np.nan)
            
            # パーセンタイル判定によるミッドライン色分け（メインチャート用）
            df['midline_p_range'] = np.where(df['trend_state'] == -1.0, df['midline'], np.nan)  # レンジ相場（赤）
            df['midline_p_neutral'] = np.where(df['trend_state'] == 0.0, df['midline'], np.nan)  # 中間状態（グレー）
            df['midline_p_trend'] = np.where(df['trend_state'] == 1.0, df['midline'], np.nan)   # トレンド相場（緑）
        else:
            # パーセンタイル分析が無効な場合、空の列を作成
            df['percentile_low'] = np.nan
            df['percentile_mid'] = np.nan
            df['percentile_high'] = np.nan
            df['intensity_low'] = np.nan
            df['intensity_mid'] = np.nan
            df['intensity_high'] = np.nan
            df['midline_p_range'] = np.nan
            df['midline_p_neutral'] = np.nan
            df['midline_p_trend'] = np.nan
        
        # NaN値を含む行を出力（最初の5行のみ）
        nan_rows = df[df['x_choppiness'].isna() | df['midline'].isna()]
        if not nan_rows.empty:
            print(f"NaN値を含む行: {len(nan_rows)}行")
            if len(nan_rows) > 0:
                print(nan_rows.head())
        
        # mplfinanceでプロット用の設定
        # 1. メインチャート上のプロット（価格とボリューム用）
        main_plots = []
        
        # 1-1. 従来のミッドライン（薄く表示）
        if not df['midline_trend'].isna().all():
            main_plots.append(mpf.make_addplot(df['midline_trend'], color='lightgreen', width=1, 
                                             alpha=0.4, linestyle='--', label='Midline (Basic Trend)'))
        if not df['midline_range'].isna().all():
            main_plots.append(mpf.make_addplot(df['midline_range'], color='lightcoral', width=1, 
                                             alpha=0.4, linestyle='--', label='Midline (Basic Range)'))
        
        # 1-2. パーセンタイル判定ミッドライン（太く表示）
        if not df['midline_p_range'].isna().all():
            main_plots.append(mpf.make_addplot(df['midline_p_range'], color='red', width=2.5, 
                                             alpha=0.8, label='P-Midline (Range)'))
        if not df['midline_p_neutral'].isna().all():
            main_plots.append(mpf.make_addplot(df['midline_p_neutral'], color='gray', width=2.5, 
                                             alpha=0.8, label='P-Midline (Neutral)'))
        if not df['midline_p_trend'].isna().all():
            main_plots.append(mpf.make_addplot(df['midline_p_trend'], color='green', width=2.5, 
                                             alpha=0.8, label='P-Midline (Trend)'))
        
        # 2. Xチョピネスパネル
        chop_trend_plot = mpf.make_addplot(df['chop_trend'], panel=1, color='green', width=2, 
                                          ylabel='X-Choppiness', secondary_y=False, label='Trend')
        chop_range_plot = mpf.make_addplot(df['chop_range'], panel=1, color='red', width=2, 
                                          secondary_y=False, label='Range')
        midline_trend_plot = mpf.make_addplot(df['midline_trend'], panel=1, color='darkgreen', width=1, 
                                             alpha=0.7, secondary_y=False, label='Midline (Trend)')
        midline_range_plot = mpf.make_addplot(df['midline_range'], panel=1, color='darkred', width=1, 
                                             alpha=0.7, secondary_y=False, label='Midline (Range)')
        
        # 3. STR値パネル
        str_panel = mpf.make_addplot(df['str_values'], panel=2, color='blue', width=1.2, 
                                    ylabel='STR Values', secondary_y=False, label='STR')
        
        # 4. パーセンタイルパネル（パーセンタイル分析が有効な場合）
        percentile_plots = []
        if (result.percentiles is not None and 
            not np.all(result.percentiles == 0) and 
            not df['percentiles'].isna().all()):
            percentile_low_plot = mpf.make_addplot(df['percentile_low'], panel=3, color='red', width=2, 
                                                  ylabel='Percentiles', secondary_y=False, label='Low State')
            percentile_mid_plot = mpf.make_addplot(df['percentile_mid'], panel=3, color='gray', width=2, 
                                                  secondary_y=False, label='Mid State')
            percentile_high_plot = mpf.make_addplot(df['percentile_high'], panel=3, color='green', width=2, 
                                                   secondary_y=False, label='High State')
            percentile_plots = [percentile_low_plot, percentile_mid_plot, percentile_high_plot]
        
        # 5. トレンド強度パネル
        intensity_plots = []
        if (result.trend_intensity is not None and 
            not np.all(result.trend_intensity == 0) and 
            not df['trend_intensity'].isna().all()):
            intensity_low_plot = mpf.make_addplot(df['intensity_low'], panel=4, color='darkred', width=1.5, 
                                                 ylabel='Trend Intensity', secondary_y=False, label='Low Intensity')
            intensity_mid_plot = mpf.make_addplot(df['intensity_mid'], panel=4, color='darkgray', width=1.5, 
                                                 secondary_y=False, label='Mid Intensity')
            intensity_high_plot = mpf.make_addplot(df['intensity_high'], panel=4, color='darkgreen', width=1.5, 
                                                  secondary_y=False, label='High Intensity')
            intensity_plots = [intensity_low_plot, intensity_mid_plot, intensity_high_plot]
        
        # 6. トレンド信号パネル
        panel_idx = 5 if (percentile_plots and intensity_plots) else (4 if percentile_plots else 3)
        trend_panel = mpf.make_addplot(df['trend_signal'], panel=panel_idx, color='orange', width=1.5, 
                                      ylabel='Trend Signal', secondary_y=False, label='Signal', type='line')
        
        # 5. 平滑化チョピネス（使用している場合）
        smoothed_plots = []
        if self.x_choppiness.use_smoothing and not df['smoothed_choppiness'].isna().all():
            df['smoothed_trend'] = np.where(df['trend_signal'] == 1, df['smoothed_choppiness'], np.nan)
            df['smoothed_range'] = np.where(df['trend_signal'] == -1, df['smoothed_choppiness'], np.nan)
            
            smoothed_trend_plot = mpf.make_addplot(df['smoothed_trend'], panel=1, color='lightgreen', 
                                                  width=1.5, alpha=0.8, secondary_y=False, label='Smoothed (Trend)')
            smoothed_range_plot = mpf.make_addplot(df['smoothed_range'], panel=1, color='lightcoral', 
                                                  width=1.5, alpha=0.8, secondary_y=False, label='Smoothed (Range)')
            smoothed_plots = [smoothed_trend_plot, smoothed_range_plot]
        
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
            kwargs['volume'] = True
            # パネル比率を動的に調整
            base_ratios = [4, 1, 2, 1]  # メイン:出来高:Xチョピネス:STR
            if percentile_plots:
                base_ratios.append(1.5)  # パーセンタイル
            if intensity_plots:
                base_ratios.append(1)    # 強度
            base_ratios.append(1)        # トレンド信号
            kwargs['panel_ratios'] = tuple(base_ratios)
            # 出来高を表示する場合は、オシレーターのパネル番号を+1する
            chop_trend_plot = mpf.make_addplot(df['chop_trend'], panel=2, color='green', width=2, 
                                              ylabel='X-Choppiness', secondary_y=False, label='Trend')
            chop_range_plot = mpf.make_addplot(df['chop_range'], panel=2, color='red', width=2, 
                                              secondary_y=False, label='Range')
            midline_trend_plot = mpf.make_addplot(df['midline_trend'], panel=2, color='darkgreen', width=1, 
                                                 alpha=0.7, secondary_y=False, label='Midline (Trend)')
            midline_range_plot = mpf.make_addplot(df['midline_range'], panel=2, color='darkred', width=1, 
                                                 alpha=0.7, secondary_y=False, label='Midline (Range)')
            str_panel = mpf.make_addplot(df['str_values'], panel=3, color='blue', width=1.2, 
                                        ylabel='STR Values', secondary_y=False, label='STR')
            
            # パーセンタイル・強度パネルも更新
            panel_offset = 4
            if percentile_plots:
                percentile_plots = [
                    mpf.make_addplot(df['percentile_low'], panel=panel_offset, color='red', width=2, 
                                    ylabel='Percentiles', secondary_y=False, label='Low State'),
                    mpf.make_addplot(df['percentile_mid'], panel=panel_offset, color='gray', width=2, 
                                    secondary_y=False, label='Mid State'),
                    mpf.make_addplot(df['percentile_high'], panel=panel_offset, color='green', width=2, 
                                   secondary_y=False, label='High State')
                ]
                panel_offset += 1
                
            if intensity_plots:
                intensity_plots = [
                    mpf.make_addplot(df['intensity_low'], panel=panel_offset, color='darkred', width=1.5, 
                                   ylabel='Trend Intensity', secondary_y=False, label='Low Intensity'),
                    mpf.make_addplot(df['intensity_mid'], panel=panel_offset, color='darkgray', width=1.5, 
                                   secondary_y=False, label='Mid Intensity'),
                    mpf.make_addplot(df['intensity_high'], panel=panel_offset, color='darkgreen', width=1.5, 
                                  secondary_y=False, label='High Intensity')
                ]
                panel_offset += 1
                
            trend_panel = mpf.make_addplot(df['trend_signal'], panel=panel_offset, color='orange', width=1.5, 
                                          ylabel='Trend Signal', secondary_y=False, label='Signal', type='line')
            
            # 平滑化版も更新
            if smoothed_plots:
                smoothed_trend_plot = mpf.make_addplot(df['smoothed_trend'], panel=2, color='lightgreen', 
                                                      width=1.5, alpha=0.8, secondary_y=False, label='Smoothed (Trend)')
                smoothed_range_plot = mpf.make_addplot(df['smoothed_range'], panel=2, color='lightcoral', 
                                                      width=1.5, alpha=0.8, secondary_y=False, label='Smoothed (Range)')
                smoothed_plots = [smoothed_trend_plot, smoothed_range_plot]
        else:
            kwargs['volume'] = False
            # パネル比率を動的に調整（出来高なし）
            base_ratios = [4, 2, 1]  # メイン:Xチョピネス:STR
            if percentile_plots:
                base_ratios.append(1.5)  # パーセンタイル
            if intensity_plots:
                base_ratios.append(1)    # 強度
            base_ratios.append(1)        # トレンド信号
            kwargs['panel_ratios'] = tuple(base_ratios)
        
        # すべてのプロットを結合
        all_plots = main_plots + [chop_trend_plot, chop_range_plot, midline_trend_plot, midline_range_plot] + smoothed_plots + [str_panel] + percentile_plots + intensity_plots + [trend_panel]
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加（Xチョピネスパネル）
        chop_panel_idx = 2 if show_volume else 1
        legend_labels = ['X-Chop (Trend)', 'X-Chop (Range)', 'Midline (Trend)', 'Midline (Range)']
        if smoothed_plots:
            legend_labels.extend(['Smoothed (Trend)', 'Smoothed (Range)'])
        axes[chop_panel_idx].legend(legend_labels, loc='upper left', fontsize=8)
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加（動的インデックス計算）
        panel_offset = 2 if show_volume else 1
        
        # Xチョピネスパネル
        axes[panel_offset].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
        axes[panel_offset].axhline(y=0.0, color='black', linestyle='-', alpha=0.3)
        axes[panel_offset].axhline(y=1.0, color='black', linestyle='-', alpha=0.3)
        axes[panel_offset].set_ylim(-0.05, 1.05)
        panel_offset += 1
        
        # STR値パネル
        str_mean = df['str_values'].mean()
        axes[panel_offset].axhline(y=str_mean, color='black', linestyle='-', alpha=0.3)
        panel_offset += 1
        
        # パーセンタイルパネル
        if percentile_plots:
            axes[panel_offset].axhline(y=0.25, color='red', linestyle='--', alpha=0.5, label='Low Threshold')
            axes[panel_offset].axhline(y=0.5, color='black', linestyle='-', alpha=0.5)
            axes[panel_offset].axhline(y=0.75, color='green', linestyle='--', alpha=0.5, label='High Threshold')
            axes[panel_offset].axhline(y=0.0, color='black', linestyle='-', alpha=0.3)
            axes[panel_offset].axhline(y=1.0, color='black', linestyle='-', alpha=0.3)
            axes[panel_offset].set_ylim(-0.05, 1.05)
            panel_offset += 1
        
        # トレンド強度パネル
        if intensity_plots:
            axes[panel_offset].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
            axes[panel_offset].axhline(y=0.0, color='black', linestyle='-', alpha=0.3)
            axes[panel_offset].axhline(y=1.0, color='black', linestyle='-', alpha=0.3)
            axes[panel_offset].set_ylim(-0.05, 1.05)
            panel_offset += 1
        
        # トレンド信号パネル
        axes[panel_offset].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[panel_offset].axhline(y=1, color='green', linestyle='--', alpha=0.5)
        axes[panel_offset].axhline(y=-1, color='red', linestyle='--', alpha=0.5)
        axes[panel_offset].set_ylim(-1.5, 1.5)
        
        # 統計情報の表示
        print(f"\n=== Xチョピネス統計 ===")
        valid_mask = ~np.isnan(df['x_choppiness'])
        total_points = valid_mask.sum()
        trend_points = (df['trend_signal'] == 1).sum()
        range_points = (df['trend_signal'] == -1).sum()
        
        print(f"総データ点数: {total_points}")
        print(f"トレンド状態: {trend_points} ({trend_points/total_points*100:.1f}%)")
        print(f"レンジ状態: {range_points} ({range_points/total_points*100:.1f}%)")
        
        if total_points > 0:
            valid_chop = df['x_choppiness'][valid_mask]
            print(f"Xチョピネス - 平均: {valid_chop.mean():.4f}, 範囲: {valid_chop.min():.4f} - {valid_chop.max():.4f}")
            
        if not df['str_values'].isna().all():
            valid_str = df['str_values'][~np.isnan(df['str_values'])]
            print(f"STR - 平均: {valid_str.mean():.4f}, 範囲: {valid_str.min():.4f} - {valid_str.max():.4f}")
        
        # パーセンタイル分析統計
        if (result.percentiles is not None and 
            not np.all(result.percentiles == 0) and 
            not df['percentiles'].isna().all()):
            print(f"\n=== パーセンタイル分析統計 ===")
            valid_percentiles = df['percentiles'][~np.isnan(df['percentiles'])]
            print(f"パーセンタイル有効値数: {len(valid_percentiles)}")
            if len(valid_percentiles) > 0:
                print(f"パーセンタイル - 平均: {valid_percentiles.mean():.4f}, 範囲: {valid_percentiles.min():.4f} - {valid_percentiles.max():.4f}")
            
            if result.trend_state is not None:
                valid_state = df['trend_state'][~np.isnan(df['trend_state'])]
                if len(valid_state) > 0:
                    low_count = np.sum(valid_state == -1.0)
                    mid_count = np.sum(valid_state == 0.0)
                    high_count = np.sum(valid_state == 1.0)
                    total_state = len(valid_state)
                    
                    print(f"パーセンタイル状態分布:")
                    print(f"  低トレンド: {low_count}/{total_state} ({low_count/total_state:.1%})")
                    print(f"  中トレンド: {mid_count}/{total_state} ({mid_count/total_state:.1%})")
                    print(f"  高トレンド: {high_count}/{total_state} ({high_count/total_state:.1%})")
            
            if (result.trend_intensity is not None and 
                not np.all(result.trend_intensity == 0) and 
                not df['trend_intensity'].isna().all()):
                valid_intensity = df['trend_intensity'][~np.isnan(df['trend_intensity'])]
                if len(valid_intensity) > 0:
                    print(f"トレンド強度 - 平均: {valid_intensity.mean():.4f}, 範囲: {valid_intensity.min():.4f} - {valid_intensity.max():.4f}")
        
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
    parser = argparse.ArgumentParser(description='Xチョピネスの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--period', type=int, default=14, help='Xチョピネス計算期間')
    parser.add_argument('--midline-period', type=int, default=100, help='ミッドライン期間')
    parser.add_argument('--str-period', type=float, default=20.0, help='STR期間')
    parser.add_argument('--smooth', action='store_true', help='平滑化を有効にする')
    parser.add_argument('--smoother-type', type=str, default='ultimate_smoother', help='スムーサータイプ')
    parser.add_argument('--dynamic', action='store_true', help='動的期間適応を有効にする')
    parser.add_argument('--detector-type', type=str, default='phac_e', help='サイクル検出器タイプ')
    parser.add_argument('--kalman', action='store_true', help='カルマンフィルターを有効にする')
    parser.add_argument('--kalman-type', type=str, default='unscented', help='カルマンフィルタータイプ')
    # パーセンタイル分析オプション
    parser.add_argument('--percentile', action='store_true', help='パーセンタイル分析を有効にする')
    parser.add_argument('--percentile-lookback', type=int, default=50, help='パーセンタイルルックバック期間')
    parser.add_argument('--percentile-low', type=float, default=0.25, help='パーセンタイル低閾値')
    parser.add_argument('--percentile-high', type=float, default=0.75, help='パーセンタイル高閾値')
    args = parser.parse_args()
    
    # チャートを作成
    chart = XChoppinessChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        period=args.period,
        midline_period=args.midline_period,
        str_period=args.str_period,
        use_smoothing=args.smooth,
        smoother_type=args.smoother_type,
        use_dynamic_period=args.dynamic,
        detector_type=args.detector_type,
        use_kalman_filter=args.kalman,
        kalman_filter_type=args.kalman_type,
        enable_percentile_analysis=args.percentile,
        percentile_lookback_period=args.percentile_lookback,
        percentile_low_threshold=args.percentile_low,
        percentile_high_threshold=args.percentile_high
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output
    )


if __name__ == "__main__":
    main()