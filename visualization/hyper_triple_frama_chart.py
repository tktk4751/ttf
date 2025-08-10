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
from indicators.hyper_triple_frama import HyperTripleFRAMA


class HyperTripleFRAMAChart:
    """
    ハイパートリプルFRAMAを表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - 3本のFRAMA（1本目：高感度、2本目：中感度、3本目：低感度）
    - フラクタル次元
    - 各アルファ値
    - 複数のクロスオーバーシグナル
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.hyper_triple_frama = None
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
                            period: int = 16,
                            src_type: str = 'hl2',
                            fc: int = 1,
                            sc: int = 198,
                            alpha_multiplier1: float = 1.0,
                            alpha_multiplier2: float = 0.5,
                            alpha_multiplier3: float = 0.1,
                            # 動的期間パラメータ
                            period_mode: str = 'fixed',
                            cycle_detector_type: str = 'hody_e',
                            lp_period: int = 13,
                            hp_period: int = 124,
                            cycle_part: float = 0.5,
                            max_cycle: int = 89,
                            min_cycle: int = 8,
                            max_output: int = 124,
                            min_output: int = 8,
                            # 動的適応パラメータ
                            enable_indicator_adaptation: bool = False,
                            smoothing_mode: str = 'none'
                           ) -> None:
        """
        ハイパートリプルFRAMAを計算する
        
        Args:
            period: 期間（偶数である必要がある、デフォルト: 16）
            src_type: ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4')
            fc: Fast Constant（デフォルト: 1）
            sc: Slow Constant（デフォルト: 198）
            alpha_multiplier1: 1本目のアルファ調整係数（デフォルト: 1.0）
            alpha_multiplier2: 2本目のアルファ調整係数（デフォルト: 0.5）
            alpha_multiplier3: 3本目のアルファ調整係数（デフォルト: 0.1）
            period_mode: 期間モード ('fixed' または 'dynamic')
            cycle_detector_type: サイクル検出器タイプ
            lp_period: ローパスフィルター期間
            hp_period: ハイパスフィルター期間
            cycle_part: サイクル部分
            max_cycle: 最大サイクル期間
            min_cycle: 最小サイクル期間
            max_output: 最大出力値
            min_output: 最小出力値
            enable_indicator_adaptation: インジケーター動的適応を有効にするか
            smoothing_mode: 平滑化モード
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\nハイパートリプルFRAMAを計算中...")
        
        # ハイパートリプルFRAMAを計算
        self.hyper_triple_frama = HyperTripleFRAMA(
            period=period,
            src_type=src_type,
            fc=fc,
            sc=sc,
            alpha_multiplier1=alpha_multiplier1,
            alpha_multiplier2=alpha_multiplier2,
            alpha_multiplier3=alpha_multiplier3,
            period_mode=period_mode,
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            max_cycle=max_cycle,
            min_cycle=min_cycle,
            max_output=max_output,
            min_output=min_output,
            enable_indicator_adaptation=enable_indicator_adaptation,
            smoothing_mode=smoothing_mode
        )
        
        # ハイパートリプルFRAMAの計算
        print("計算を実行します...")
        result = self.hyper_triple_frama.calculate(self.data)
        
        # 結果の取得テスト
        frama1_values = self.hyper_triple_frama.get_frama_values()
        frama2_values = self.hyper_triple_frama.get_second_frama_values()
        frama3_values = self.hyper_triple_frama.get_third_frama_values()
        fractal_dim = self.hyper_triple_frama.get_fractal_dimension()
        alpha1_values = self.hyper_triple_frama.get_alpha()
        alpha2_values = self.hyper_triple_frama.get_second_alpha()
        alpha3_values = self.hyper_triple_frama.get_third_alpha()
        
        print(f"計算完了 - FRAMA1: {len(frama1_values)}, FRAMA2: {len(frama2_values)}, FRAMA3: {len(frama3_values)}")
        print(f"フラクタル次元: {len(fractal_dim)}, Alpha1: {len(alpha1_values)}, Alpha2: {len(alpha2_values)}, Alpha3: {len(alpha3_values)}")
        
        # NaN値のチェック
        nan_count_frama1 = np.isnan(frama1_values).sum()
        nan_count_frama2 = np.isnan(frama2_values).sum()
        nan_count_frama3 = np.isnan(frama3_values).sum()
        nan_count_dim = np.isnan(fractal_dim).sum()
        print(f"NaN値 - FRAMA1: {nan_count_frama1}, FRAMA2: {nan_count_frama2}, FRAMA3: {nan_count_frama3}, フラクタル次元: {nan_count_dim}")
        
        print("ハイパートリプルFRAMA計算完了")
            
    def plot(self, 
            title: str = "ハイパートリプルFRAMA", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (16, 14),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとハイパートリプルFRAMAを描画する
        
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
            
        if self.hyper_triple_frama is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # ハイパートリプルFRAMAの値を取得
        print("ハイパートリプルFRAMAデータを取得中...")
        frama1_values = self.hyper_triple_frama.get_frama_values()
        frama2_values = self.hyper_triple_frama.get_second_frama_values()
        frama3_values = self.hyper_triple_frama.get_third_frama_values()
        fractal_dim = self.hyper_triple_frama.get_fractal_dimension()
        alpha1_values = self.hyper_triple_frama.get_alpha()
        alpha2_values = self.hyper_triple_frama.get_second_alpha()
        alpha3_values = self.hyper_triple_frama.get_third_alpha()
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'frama1': frama1_values,
                'frama2': frama2_values,
                'frama3': frama3_values,
                'fractal_dimension': fractal_dim,
                'alpha1': alpha1_values,
                'alpha2': alpha2_values,
                'alpha3': alpha3_values
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        print(f"FRAMAデータ確認 - FRAMA1 NaN: {df['frama1'].isna().sum()}, FRAMA2 NaN: {df['frama2'].isna().sum()}, FRAMA3 NaN: {df['frama3'].isna().sum()}")
        
        # クロスオーバーの検出（NaN値を適切に処理）
        df['frama1_above_frama2'] = df['frama1'] > df['frama2']
        df['frama1_above_frama3'] = df['frama1'] > df['frama3']
        df['frama2_above_frama3'] = df['frama2'] > df['frama3']
        
        # シグナルの検出
        prev_frama1 = df['frama1'].shift(1)
        prev_frama2 = df['frama2'].shift(1)
        prev_frama3 = df['frama3'].shift(1)
        
        # FRAMA1とFRAMA2のクロスオーバー
        df['frama1_cross_above_frama2'] = (
            (df['frama1'] > df['frama2']) & 
            (prev_frama1 <= prev_frama2) &
            df['frama1'].notna() & df['frama2'].notna() &
            prev_frama1.notna() & prev_frama2.notna()
        )
        
        df['frama1_cross_below_frama2'] = (
            (df['frama1'] < df['frama2']) & 
            (prev_frama1 >= prev_frama2) &
            df['frama1'].notna() & df['frama2'].notna() &
            prev_frama1.notna() & prev_frama2.notna()
        )
        
        # FRAMA2とFRAMA3のクロスオーバー
        df['frama2_cross_above_frama3'] = (
            (df['frama2'] > df['frama3']) & 
            (prev_frama2 <= prev_frama3) &
            df['frama2'].notna() & df['frama3'].notna() &
            prev_frama2.notna() & prev_frama3.notna()
        )
        
        df['frama2_cross_below_frama3'] = (
            (df['frama2'] < df['frama3']) & 
            (prev_frama2 >= prev_frama3) &
            df['frama2'].notna() & df['frama3'].notna() &
            prev_frama2.notna() & prev_frama3.notna()
        )
        
        # 強力なシグナル: 3本すべてが同じ方向に並んだ時
        df['all_bullish'] = (df['frama1'] > df['frama2']) & (df['frama2'] > df['frama3'])
        df['all_bearish'] = (df['frama1'] < df['frama2']) & (df['frama2'] < df['frama3'])
        
        # トレンド転換シグナル
        prev_all_bullish = df['all_bullish'].shift(1)
        prev_all_bearish = df['all_bearish'].shift(1)
        
        df['trend_turn_bullish'] = df['all_bullish'] & ~prev_all_bullish.fillna(False)
        df['trend_turn_bearish'] = df['all_bearish'] & ~prev_all_bearish.fillna(False)
        
        # mplfinanceでプロット用の設定
        # 1. メインチャート上のプロット
        main_plots = []
        
        # 3本のFRAMAのプロット設定（異なる色とスタイル）
        main_plots.append(mpf.make_addplot(df['frama1'], color='#FF4444', width=2.5, label='FRAMA1 (Fast)', linestyle='-'))
        main_plots.append(mpf.make_addplot(df['frama2'], color='#4444FF', width=2.0, label='FRAMA2 (Medium)', linestyle='-'))
        main_plots.append(mpf.make_addplot(df['frama3'], color='#44AA44', width=1.5, label='FRAMA3 (Slow)', linestyle='-'))
        
        # クロスオーバーシグナルのプロット用データ準備
        # FRAMA1とFRAMA2のクロス
        frama12_cross_above = pd.Series(index=df.index, data=np.nan)
        frama12_cross_below = pd.Series(index=df.index, data=np.nan)
        
        frama12_cross_above.loc[df['frama1_cross_above_frama2']] = df.loc[df['frama1_cross_above_frama2'], 'frama1']
        frama12_cross_below.loc[df['frama1_cross_below_frama2']] = df.loc[df['frama1_cross_below_frama2'], 'frama1']
        
        # FRAMA2とFRAMA3のクロス
        frama23_cross_above = pd.Series(index=df.index, data=np.nan)
        frama23_cross_below = pd.Series(index=df.index, data=np.nan)
        
        frama23_cross_above.loc[df['frama2_cross_above_frama3']] = df.loc[df['frama2_cross_above_frama3'], 'frama2']
        frama23_cross_below.loc[df['frama2_cross_below_frama3']] = df.loc[df['frama2_cross_below_frama3'], 'frama2']
        
        # トレンド転換シグナル
        trend_bullish = pd.Series(index=df.index, data=np.nan)
        trend_bearish = pd.Series(index=df.index, data=np.nan)
        
        trend_bullish.loc[df['trend_turn_bullish']] = df.loc[df['trend_turn_bullish'], 'frama1']
        trend_bearish.loc[df['trend_turn_bearish']] = df.loc[df['trend_turn_bearish'], 'frama1']
        
        # シグナルが存在する場合のみプロットに追加
        if frama12_cross_above.notna().any():
            main_plots.append(mpf.make_addplot(frama12_cross_above, type='scatter', markersize=60, marker='^', color='orange', alpha=0.8))
        if frama12_cross_below.notna().any():
            main_plots.append(mpf.make_addplot(frama12_cross_below, type='scatter', markersize=60, marker='v', color='purple', alpha=0.8))
        
        if frama23_cross_above.notna().any():
            main_plots.append(mpf.make_addplot(frama23_cross_above, type='scatter', markersize=40, marker='o', color='lightgreen', alpha=0.7))
        if frama23_cross_below.notna().any():
            main_plots.append(mpf.make_addplot(frama23_cross_below, type='scatter', markersize=40, marker='o', color='lightcoral', alpha=0.7))
        
        if trend_bullish.notna().any():
            main_plots.append(mpf.make_addplot(trend_bullish, type='scatter', markersize=120, marker='*', color='gold', alpha=0.9))
        if trend_bearish.notna().any():
            main_plots.append(mpf.make_addplot(trend_bearish, type='scatter', markersize=120, marker='*', color='darkred', alpha=0.9))
        
        # 2. サブプロット
        # フラクタル次元パネル
        fractal_panel = mpf.make_addplot(df['fractal_dimension'], panel=1, color='purple', width=1.2, 
                                        ylabel='Fractal Dimension', secondary_y=False, label='Fractal Dim')
        
        # アルファ値パネル（3本すべて）
        alpha1_panel = mpf.make_addplot(df['alpha1'], panel=2, color='#FF4444', width=1.5, 
                                       ylabel='Alpha Values', secondary_y=False, label='Alpha1')
        alpha2_panel = mpf.make_addplot(df['alpha2'], panel=2, color='#4444FF', width=1.2, 
                                       secondary_y=False, label='Alpha2')
        alpha3_panel = mpf.make_addplot(df['alpha3'], panel=2, color='#44AA44', width=1.0, 
                                       secondary_y=False, label='Alpha3')
        
        # トレンド強度パネル（FRAMA間の相対差）
        df['trend_strength_12'] = (df['frama1'] - df['frama2']) / df['frama2'] * 100
        df['trend_strength_23'] = (df['frama2'] - df['frama3']) / df['frama3'] * 100
        df['trend_strength_13'] = (df['frama1'] - df['frama3']) / df['frama3'] * 100
        
        strength12_panel = mpf.make_addplot(df['trend_strength_12'], panel=3, color='#FF8844', width=1.5, 
                                           ylabel='Trend Strength (%)', secondary_y=False, label='FRAMA1-2')
        strength23_panel = mpf.make_addplot(df['trend_strength_23'], panel=3, color='#4488FF', width=1.2, 
                                           secondary_y=False, label='FRAMA2-3')
        strength13_panel = mpf.make_addplot(df['trend_strength_13'], panel=3, color='#88AA44', width=1.0, 
                                           secondary_y=False, label='FRAMA1-3')
        
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
            kwargs['panel_ratios'] = (4, 1, 1, 1, 1)  # メイン:出来高:フラクタル次元:アルファ:トレンド強度
            # 出来高を表示する場合は、オシレーターのパネル番号を+1する
            fractal_panel = mpf.make_addplot(df['fractal_dimension'], panel=2, color='purple', width=1.2, 
                                            ylabel='Fractal Dimension', secondary_y=False, label='Fractal Dim')
            alpha1_panel = mpf.make_addplot(df['alpha1'], panel=3, color='#FF4444', width=1.5, 
                                           ylabel='Alpha Values', secondary_y=False, label='Alpha1')
            alpha2_panel = mpf.make_addplot(df['alpha2'], panel=3, color='#4444FF', width=1.2, 
                                           secondary_y=False, label='Alpha2')
            alpha3_panel = mpf.make_addplot(df['alpha3'], panel=3, color='#44AA44', width=1.0, 
                                           secondary_y=False, label='Alpha3')
            strength12_panel = mpf.make_addplot(df['trend_strength_12'], panel=4, color='#FF8844', width=1.5, 
                                               ylabel='Trend Strength (%)', secondary_y=False, label='FRAMA1-2')
            strength23_panel = mpf.make_addplot(df['trend_strength_23'], panel=4, color='#4488FF', width=1.2, 
                                               secondary_y=False, label='FRAMA2-3')
            strength13_panel = mpf.make_addplot(df['trend_strength_13'], panel=4, color='#88AA44', width=1.0, 
                                               secondary_y=False, label='FRAMA1-3')
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (4, 1, 1, 1)  # メイン:フラクタル次元:アルファ:トレンド強度
        
        # すべてのプロットを結合
        all_plots = main_plots + [fractal_panel, alpha1_panel, alpha2_panel, alpha3_panel, 
                                 strength12_panel, strength23_panel, strength13_panel]
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加
        legend_labels = ['FRAMA1 (Fast)', 'FRAMA2 (Medium)', 'FRAMA3 (Slow)']
        if frama12_cross_above.notna().any() or frama12_cross_below.notna().any():
            legend_labels.extend(['FRAMA1-2 Cross Up', 'FRAMA1-2 Cross Down'])
        if frama23_cross_above.notna().any() or frama23_cross_below.notna().any():
            legend_labels.extend(['FRAMA2-3 Cross Up', 'FRAMA2-3 Cross Down'])
        if trend_bullish.notna().any() or trend_bearish.notna().any():
            legend_labels.extend(['Trend Turn Bullish', 'Trend Turn Bearish'])
        
        axes[0].legend(legend_labels[:min(len(legend_labels), 8)], loc='upper left', fontsize=8)
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        if show_volume:
            panel_offset = 1
        else:
            panel_offset = 0
        
        # フラクタル次元パネル
        fractal_panel_idx = 1 + panel_offset
        axes[fractal_panel_idx].axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Trend (1.0)')
        axes[fractal_panel_idx].axhline(y=1.5, color='black', linestyle='-', alpha=0.3)
        axes[fractal_panel_idx].axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='Range (2.0)')
        axes[fractal_panel_idx].legend(loc='upper right', fontsize=8)
        
        # アルファ値パネル
        alpha_panel_idx = 2 + panel_offset
        axes[alpha_panel_idx].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
        axes[alpha_panel_idx].axhline(y=0.01, color='red', linestyle='--', alpha=0.5, label='Min Alpha')
        axes[alpha_panel_idx].axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Max Alpha')
        axes[alpha_panel_idx].legend(loc='upper right', fontsize=8)
        
        # トレンド強度パネル
        strength_panel_idx = 3 + panel_offset
        axes[strength_panel_idx].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[strength_panel_idx].axhline(y=5, color='green', linestyle='--', alpha=0.3)
        axes[strength_panel_idx].axhline(y=-5, color='red', linestyle='--', alpha=0.3)
        axes[strength_panel_idx].axhline(y=2, color='lightgreen', linestyle=':', alpha=0.3)
        axes[strength_panel_idx].axhline(y=-2, color='lightcoral', linestyle=':', alpha=0.3)
        
        # 統計情報の表示
        print(f"\n=== ハイパートリプルFRAMA統計 ===")
        total_points = len(df.dropna())
        
        # シグナル統計
        frama12_cross_up = int(df['frama1_cross_above_frama2'].sum())
        frama12_cross_down = int(df['frama1_cross_below_frama2'].sum())
        frama23_cross_up = int(df['frama2_cross_above_frama3'].sum())
        frama23_cross_down = int(df['frama2_cross_below_frama3'].sum())
        trend_bullish_signals = int(df['trend_turn_bullish'].sum())
        trend_bearish_signals = int(df['trend_turn_bearish'].sum())
        
        print(f"総データ点数: {total_points}")
        print(f"FRAMA1-2クロス - 上: {frama12_cross_up}, 下: {frama12_cross_down}")
        print(f"FRAMA2-3クロス - 上: {frama23_cross_up}, 下: {frama23_cross_down}")
        print(f"トレンド転換 - 強気: {trend_bullish_signals}, 弱気: {trend_bearish_signals}")
        
        # NaN値を除いた統計計算
        fractal_clean = df['fractal_dimension'].dropna()
        alpha1_clean = df['alpha1'].dropna()
        alpha2_clean = df['alpha2'].dropna()
        alpha3_clean = df['alpha3'].dropna()
        trend_strength_12_clean = df['trend_strength_12'].dropna()
        trend_strength_23_clean = df['trend_strength_23'].dropna()
        trend_strength_13_clean = df['trend_strength_13'].dropna()
        
        if len(fractal_clean) > 0:
            print(f"フラクタル次元 - 平均: {fractal_clean.mean():.3f}, 範囲: {fractal_clean.min():.3f} - {fractal_clean.max():.3f}")
        if len(alpha1_clean) > 0:
            print(f"Alpha1値 - 平均: {alpha1_clean.mean():.3f}, 範囲: {alpha1_clean.min():.3f} - {alpha1_clean.max():.3f}")
        if len(alpha2_clean) > 0:
            print(f"Alpha2値 - 平均: {alpha2_clean.mean():.3f}, 範囲: {alpha2_clean.min():.3f} - {alpha2_clean.max():.3f}")
        if len(alpha3_clean) > 0:
            print(f"Alpha3値 - 平均: {alpha3_clean.mean():.3f}, 範囲: {alpha3_clean.min():.3f} - {alpha3_clean.max():.3f}")
        if len(trend_strength_12_clean) > 0:
            print(f"トレンド強度1-2 - 平均: {trend_strength_12_clean.mean():.2f}%, 範囲: {trend_strength_12_clean.min():.2f}% - {trend_strength_12_clean.max():.2f}%")
        if len(trend_strength_23_clean) > 0:
            print(f"トレンド強度2-3 - 平均: {trend_strength_23_clean.mean():.2f}%, 範囲: {trend_strength_23_clean.min():.2f}% - {trend_strength_23_clean.max():.2f}%")
        if len(trend_strength_13_clean) > 0:
            print(f"トレンド強度1-3 - 平均: {trend_strength_13_clean.mean():.2f}%, 範囲: {trend_strength_13_clean.min():.2f}% - {trend_strength_13_clean.max():.2f}%")
        
        # トレンド状態の統計
        all_bullish_count = int(df['all_bullish'].sum())
        all_bearish_count = int(df['all_bearish'].sum()) 
        neutral_count = total_points - all_bullish_count - all_bearish_count
        
        print(f"\nトレンド状態分布:")
        print(f"  強気配列 (FRAMA1>2>3): {all_bullish_count} ({all_bullish_count/total_points*100:.1f}%)")
        print(f"  弱気配列 (FRAMA1<2<3): {all_bearish_count} ({all_bearish_count/total_points*100:.1f}%)")
        print(f"  中性・混合: {neutral_count} ({neutral_count/total_points*100:.1f}%)")
        
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
    parser = argparse.ArgumentParser(description='ハイパートリプルFRAMAの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--period', type=int, default=16, help='期間（偶数である必要がある）')
    parser.add_argument('--src-type', type=str, default='hl2', help='ソースタイプ')
    parser.add_argument('--fc', type=int, default=1, help='Fast Constant')
    parser.add_argument('--sc', type=int, default=198, help='Slow Constant')
    parser.add_argument('--alpha1', type=float, default=1.0, help='1本目のアルファ調整係数')
    parser.add_argument('--alpha2', type=float, default=0.5, help='2本目のアルファ調整係数')
    parser.add_argument('--alpha3', type=float, default=0.1, help='3本目のアルファ調整係数')
    parser.add_argument('--period-mode', type=str, default='fixed', help='期間モード (fixed または dynamic)')
    parser.add_argument('--enable-adaptation', action='store_true', help='インジケーター動的適応を有効にする')
    parser.add_argument('--smoothing-mode', type=str, default='none', help='平滑化モード')
    args = parser.parse_args()
    
    # チャートを作成
    chart = HyperTripleFRAMAChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        period=args.period,
        src_type=args.src_type,
        fc=args.fc,
        sc=args.sc,
        alpha_multiplier1=args.alpha1,
        alpha_multiplier2=args.alpha2,
        alpha_multiplier3=args.alpha3,
        period_mode=args.period_mode,
        enable_indicator_adaptation=args.enable_adaptation,
        smoothing_mode=args.smoothing_mode
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output
    )


if __name__ == "__main__":
    main()