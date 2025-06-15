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
from indicators.z_adaptive_flow import ZAdaptiveFlow


class ZAdaptiveFlowChart:
    """
    Z Adaptive Flowを表示するローソク足チャートクラス
    
    表示内容:
    - メインパネル: ローソク足、Basis線、Level線、バンド、シグナル
    - サブパネル1: 出来高（オプション）
    - サブパネル2: トレンド状態
    - サブパネル3: ボラティリティ（生値と平滑化値）
    - サブパネル4: 動的乗数（スロー期間、ボラティリティ）
    - サブパネル5: ファスト・スローMA
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.z_adaptive_flow = None
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
                            length: int = 10,
                            smooth_length: int = 14,
                            src_type: str = 'hlc3',
                            # MAタイプ選択
                            ma_type: str = 'hma',
                            # ボラティリティタイプ選択
                            volatility_type: str = 'atr',
                            # 動的適応パラメータ
                            adaptive_trigger: str = 'chop_trend',
                            # MA固有パラメータ
                            hma_slope_index: int = 1,
                            hma_range_threshold: float = 0.005,
                            alma_offset: float = 0.85,
                            alma_sigma: float = 6,
                            z_adaptive_ma_fast_period: int = 2,
                            z_adaptive_ma_slow_period: int = 30,
                            zlema_slope_index: int = 1,
                            zlema_range_threshold: float = 0.005,
                            # ボラティリティ固有パラメータ
                            volatility_period_mode: str = 'fixed',
                            volatility_return_type: str = 'log',
                            volatility_smoother_type: str = 'hma',
                            atr_smoothing_method: str = 'alma',
                            # AdaptivePeriod共通パラメータ
                            adaptive_power: float = 1.0,
                            adaptive_invert: bool = False,
                            adaptive_reverse_mapping: bool = False,
                            # トリガーインジケーター用パラメータ
                            **trigger_params
                           ) -> None:
        """
        Z Adaptive Flowを計算する
        
        Args:
            length: メイン期間
            smooth_length: ボラティリティ平滑化期間
            src_type: 価格ソース
            ma_type: MAタイプ ('hma', 'alma', 'z_adaptive_ma', 'zlema')
            volatility_type: ボラティリティタイプ ('volatility', 'atr')
            adaptive_trigger: AdaptivePeriod用トリガーインジケーター
            hma_slope_index: HMA用スロープインデックス
            hma_range_threshold: HMA用range閾値
            alma_offset: ALMA用オフセット
            alma_sigma: ALMA用シグマ
            z_adaptive_ma_fast_period: ZAdaptiveMA用ファスト期間
            z_adaptive_ma_slow_period: ZAdaptiveMA用スロー期間
            zlema_slope_index: ZLEMA用スロープインデックス
            zlema_range_threshold: ZLEMA用range閾値
            volatility_period_mode: Volatility用期間モード
            volatility_return_type: Volatility用リターンタイプ
            volatility_smoother_type: Volatility用平滑化タイプ
            atr_smoothing_method: ATR用平滑化方法
            adaptive_power: AdaptivePeriod用べき乗値
            adaptive_invert: AdaptivePeriod用反転フラグ
            adaptive_reverse_mapping: AdaptivePeriod用逆マッピングフラグ
            **trigger_params: トリガーインジケーター用追加パラメータ
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print(f"\nZ Adaptive Flowを計算中...")
        print(f"設定: MA={ma_type}, Volatility={volatility_type}, Trigger={adaptive_trigger}")
        
        # Z Adaptive Flowインジケーターを初期化
        self.z_adaptive_flow = ZAdaptiveFlow(
            length=length,
            smooth_length=smooth_length,
            src_type=src_type,
            ma_type=ma_type,
            volatility_type=volatility_type,
            adaptive_trigger=adaptive_trigger,
            hma_slope_index=hma_slope_index,
            hma_range_threshold=hma_range_threshold,
            alma_offset=alma_offset,
            alma_sigma=alma_sigma,
            z_adaptive_ma_fast_period=z_adaptive_ma_fast_period,
            z_adaptive_ma_slow_period=z_adaptive_ma_slow_period,
            zlema_slope_index=zlema_slope_index,
            zlema_range_threshold=zlema_range_threshold,
            volatility_period_mode=volatility_period_mode,
            volatility_return_type=volatility_return_type,
            volatility_smoother_type=volatility_smoother_type,
            atr_smoothing_method=atr_smoothing_method,
            adaptive_power=adaptive_power,
            adaptive_invert=adaptive_invert,
            adaptive_reverse_mapping=adaptive_reverse_mapping,
            **trigger_params
        )
        
        # Z Adaptive Flowの計算
        print("計算を実行します...")
        result = self.z_adaptive_flow.calculate(self.data)
        
        print(f"計算完了:")
        print(f"  - Basis線: {len(result.basis)} ポイント")
        print(f"  - Level線: {len(result.level)} ポイント")
        print(f"  - バンド: Upper={len(result.upper)}, Lower={len(result.lower)}")
        print(f"  - トレンド状態: {len(result.trend_state)} ポイント")
        print(f"  - シグナル: Long={np.sum(result.long_signals)}, Short={np.sum(result.short_signals)}")
        
        # NaN値のチェック
        print(f"NaN値:")
        print(f"  - Basis: {np.isnan(result.basis).sum()}")
        print(f"  - Level: {np.isnan(result.level).sum()}")
        print(f"  - Upper: {np.isnan(result.upper).sum()}")
        print(f"  - Lower: {np.isnan(result.lower).sum()}")
        print(f"  - トレンド状態: {np.isnan(result.trend_state).sum()}")
        
        # 現在のトレンド状態表示
        current_trend = self.z_adaptive_flow.get_current_trend()
        print(f"現在のトレンド: {current_trend}")
        
        print("Z Adaptive Flow計算完了")
            
    def plot(self, 
            title: str = "Z Adaptive Flow", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            show_signals: bool = True,
            figsize: Tuple[int, int] = (16, 12),
            style: str = 'yahoo',
            savefig: Optional[str] = None,
            max_data_points: int = 5000) -> None:
        """
        ローソク足チャートとZ Adaptive Flowを描画する
        
        Args:
            title: チャートのタイトル
            start_date: 表示開始日（フォーマット: YYYY-MM-DD）
            end_date: 表示終了日（フォーマット: YYYY-MM-DD）
            show_volume: 出来高を表示するか
            show_signals: シグナルマーカーを表示するか
            figsize: 図のサイズ
            style: mplfinanceのスタイル
            savefig: 保存先のパス（指定しない場合は表示のみ）
            max_data_points: 最大データポイント数（この数を超える場合は最新データに制限）
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        if self.z_adaptive_flow is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
        
        # データポイント数制限
        if len(df) > max_data_points:
            print(f"データポイント数が{max_data_points}を超えています。最新{max_data_points}件に制限します。")
            df = df.tail(max_data_points)
            
        # Z Adaptive Flowの結果を取得
        print("Z Adaptive Flowデータを取得中...")
        result = self.z_adaptive_flow.get_detailed_result()
        
        # データの有効性チェック
        if len(result.basis) == 0:
            print("警告: Z Adaptive Flowの計算結果が空です。")
            return
        
        print(f"インジケーター結果のサイズ: {len(result.basis)}")
        print(f"チャート用データのサイズ: {len(df)}")
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'basis': result.basis,
                'level': result.level,
                'upper': result.upper,
                'lower': result.lower,
                'trend_state': result.trend_state,
                'volatility': result.volatility,
                'slow_multiplier': result.slow_multiplier,
                'volatility_multiplier': result.volatility_multiplier,
                'fast_ma': result.fast_ma,
                'slow_ma': result.slow_ma,
                'long_signals': result.long_signals,
                'short_signals': result.short_signals,
                'smoothed_volatility': result.smoothed_volatility
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        print(f"期間: {df.index.min()} → {df.index.max()}")
        print(f"データ確認:")
        print(f"  - Basis NaN: {df['basis'].isna().sum()}/{len(df)}")
        print(f"  - Level NaN: {df['level'].isna().sum()}/{len(df)}")
        print(f"  - Upper NaN: {df['upper'].isna().sum()}/{len(df)}")
        print(f"  - Lower NaN: {df['lower'].isna().sum()}/{len(df)}")
        print(f"  - Trend State NaN: {df['trend_state'].isna().sum()}/{len(df)}")
        print(f"  - シグナル: Long={df['long_signals'].sum()}, Short={df['short_signals'].sum()}")
        
        # サンプルデータ表示（デバッグ用）
        print(f"\n最初の10行のサンプルデータ:")
        sample_cols = ['basis', 'level', 'upper', 'lower', 'trend_state']
        for col in sample_cols:
            if col in df.columns:
                print(f"  {col}: {df[col].head(10).tolist()}")
        
        # 有効な値が存在するかチェックする関数
        def has_valid_data(series):
            if series is None or len(series) == 0:
                return False
            valid_count = len(series.dropna())
            total_count = len(series)
            # デバッグ出力は最初のチェック時のみ
            return valid_count > 0 and not series.isna().all()
        
        print(f"\n有効データチェック:")
        # データ有効性の確認（一度だけ）
        data_validity = {}
        for col in ['basis', 'level', 'upper', 'lower', 'fast_ma', 'slow_ma', 'trend_state', 'volatility', 'smoothed_volatility', 'slow_multiplier', 'volatility_multiplier']:
            if col in df.columns:
                valid_count = len(df[col].dropna())
                total_count = len(df[col])
                data_validity[col] = valid_count > 0 and not df[col].isna().all()
                print(f"    {col}: {valid_count}/{total_count} 有効値")
        
        # mplfinanceでプロット用の設定
        # 1. メインチャート上のプロット
        main_plots = []
        
        # Z Adaptive Flowのライン（有効なデータのみ追加）
        if data_validity.get('basis', False):
            main_plots.append(mpf.make_addplot(df['basis'], color='blue', width=2, label='Basis'))
        
        if data_validity.get('level', False):
            main_plots.append(mpf.make_addplot(df['level'], color='orange', width=2, label='Level'))
        
        if data_validity.get('upper', False):
            main_plots.append(mpf.make_addplot(df['upper'], color='green', width=1, alpha=0.7, label='Upper Band'))
        
        if data_validity.get('lower', False):
            main_plots.append(mpf.make_addplot(df['lower'], color='red', width=1, alpha=0.7, label='Lower Band'))
        
        # ファスト・スローMA（薄い色で表示、有効なデータのみ）
        if data_validity.get('fast_ma', False):
            main_plots.append(mpf.make_addplot(df['fast_ma'], color='lightblue', width=1, alpha=0.5, label='Fast MA'))
        
        if data_validity.get('slow_ma', False):
            main_plots.append(mpf.make_addplot(df['slow_ma'], color='lightcoral', width=1, alpha=0.5, label='Slow MA'))
        
        # シグナルマーカー（オプション）
        if show_signals:
            # ロングシグナル
            long_mask = df['long_signals'] == True
            if long_mask.any():
                long_signal_indices = df.index[long_mask]
                long_signals_y = df.loc[long_mask, 'low'] * 0.995  # 少し下に表示
                if len(long_signals_y) > 0 and len(long_signal_indices) == len(long_signals_y):
                    # DataFrame形式でプロット用データを作成
                    long_plot_data = pd.Series(index=df.index, dtype=float)
                    long_plot_data.loc[long_mask] = long_signals_y
                    main_plots.append(mpf.make_addplot(
                        long_plot_data, type='scatter', markersize=100, 
                        marker='^', color='green', alpha=0.8, label='Long Signal'
                    ))
            
            # ショートシグナル
            short_mask = df['short_signals'] == True
            if short_mask.any():
                short_signal_indices = df.index[short_mask]
                short_signals_y = df.loc[short_mask, 'high'] * 1.005  # 少し上に表示
                if len(short_signals_y) > 0 and len(short_signal_indices) == len(short_signals_y):
                    # DataFrame形式でプロット用データを作成
                    short_plot_data = pd.Series(index=df.index, dtype=float)
                    short_plot_data.loc[short_mask] = short_signals_y
                    main_plots.append(mpf.make_addplot(
                        short_plot_data, type='scatter', markersize=100, 
                        marker='v', color='red', alpha=0.8, label='Short Signal'
                    ))
        
        # サブパネル用のプロット
        sub_plots = []
        panel_index = 1  # 出来高パネルがある場合は調整
        
        # トレンド状態パネル
        if show_volume:
            current_panel = 2
        else:
            current_panel = 1
        
        if data_validity.get('trend_state', False):
            trend_panel = mpf.make_addplot(df['trend_state'], panel=current_panel, color='purple', width=2, 
                                         ylabel='Trend State', secondary_y=False, label='Trend')
            sub_plots.append(trend_panel)
            current_panel += 1
        
        # ボラティリティパネル
        vol_plots_added = False
        if data_validity.get('volatility', False):
            vol_panel = mpf.make_addplot(df['volatility'], panel=current_panel, color='brown', width=1.5, 
                                       ylabel='Volatility', secondary_y=False, label='Vol')
            sub_plots.append(vol_panel)
            vol_plots_added = True
        
        if data_validity.get('smoothed_volatility', False):
            smooth_vol_panel = mpf.make_addplot(df['smoothed_volatility'], panel=current_panel, color='orange', width=1.5, 
                                               secondary_y=False, label='Smooth Vol')
            sub_plots.append(smooth_vol_panel)
            vol_plots_added = True
        
        if vol_plots_added:
            current_panel += 1
        
        # 動的乗数パネル
        mult_plots_added = False
        if data_validity.get('slow_multiplier', False):
            slow_mult_panel = mpf.make_addplot(df['slow_multiplier'], panel=current_panel, color='blue', width=1.5, 
                                             ylabel='Multipliers', secondary_y=False, label='Slow Mult')
            sub_plots.append(slow_mult_panel)
            mult_plots_added = True
        
        if data_validity.get('volatility_multiplier', False):
            vol_mult_panel = mpf.make_addplot(df['volatility_multiplier'], panel=current_panel, color='red', width=1.5, 
                                            secondary_y=False, label='Vol Mult')
            sub_plots.append(vol_mult_panel)
            mult_plots_added = True
        
        # 何もプロットするものがない場合の警告
        if not main_plots and not sub_plots:
            print("警告: 表示可能なデータがありません。計算結果を確認してください。")
            return
        
        # mplfinanceの設定
        kwargs = dict(
            type='candle',
            figsize=figsize,
            title=title,
            style=style,
            datetime_format='%Y-%m-%d',
            xrotation=45,
            returnfig=True,
            warn_too_much_data=len(df) + 1000  # データが多い場合の警告を抑制
        )
        
        # パネル数の動的計算
        total_panels = 1  # メインパネル
        if show_volume:
            total_panels += 1
        if any('trend_state' in str(plot) for plot in sub_plots):
            total_panels += 1
        if vol_plots_added:
            total_panels += 1
        if mult_plots_added:
            total_panels += 1
        
        # パネル構成の設定
        if show_volume:
            kwargs['volume'] = True
            if total_panels == 2:  # メイン + 出来高のみ
                kwargs['panel_ratios'] = (5, 1)
            elif total_panels == 3:  # メイン + 出来高 + 1つのサブパネル
                kwargs['panel_ratios'] = (5, 1, 1)
            elif total_panels == 4:  # メイン + 出来高 + 2つのサブパネル
                kwargs['panel_ratios'] = (5, 1, 1, 1.5)
            else:  # メイン + 出来高 + 3つのサブパネル
                kwargs['panel_ratios'] = (5, 1, 1, 1.5, 1)
        else:
            kwargs['volume'] = False
            if total_panels == 1:  # メインのみ
                kwargs['panel_ratios'] = (5,)
            elif total_panels == 2:  # メイン + 1つのサブパネル
                kwargs['panel_ratios'] = (5, 1)
            elif total_panels == 3:  # メイン + 2つのサブパネル
                kwargs['panel_ratios'] = (5, 1, 1.5)
            else:  # メイン + 3つのサブパネル
                kwargs['panel_ratios'] = (5, 1, 1.5, 1)
        
        # すべてのプロットを結合
        all_plots = main_plots + sub_plots
        if all_plots:
            kwargs['addplot'] = all_plots
        
        try:
            # プロット実行
            fig, axes = mpf.plot(df, **kwargs)
            
            # 凡例の追加（メインパネル）
            if main_plots:
                legend_labels = []
                if data_validity.get('basis', False):
                    legend_labels.append('Basis')
                if data_validity.get('level', False):
                    legend_labels.append('Level')
                if data_validity.get('upper', False):
                    legend_labels.append('Upper Band')
                if data_validity.get('lower', False):
                    legend_labels.append('Lower Band')
                if data_validity.get('fast_ma', False):
                    legend_labels.append('Fast MA')
                if data_validity.get('slow_ma', False):
                    legend_labels.append('Slow MA')
                if show_signals:
                    if (df['long_signals'] == True).any():
                        legend_labels.append('Long Signal')
                    if (df['short_signals'] == True).any():
                        legend_labels.append('Short Signal')
                
                if legend_labels:
                    axes[0].legend(legend_labels, loc='upper left', fontsize=8)
            
            self.fig = fig
            self.axes = axes
            
            # 参照線の追加
            panel_offset = 1 if show_volume else 0
            
            # トレンド状態パネルの参照線
            if data_validity.get('trend_state', False):
                trend_panel_idx = 1 + panel_offset
                if trend_panel_idx < len(axes):
                    axes[trend_panel_idx].axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Bullish')
                    axes[trend_panel_idx].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                    axes[trend_panel_idx].axhline(y=-1, color='red', linestyle='--', alpha=0.5, label='Bearish')
                    axes[trend_panel_idx].set_ylim(-1.5, 1.5)
            
            # ボラティリティパネルの参照線
            if vol_plots_added:
                vol_panel_idx = 2 + panel_offset if data_validity.get('trend_state', False) else 1 + panel_offset
                if vol_panel_idx < len(axes):
                    vol_mean = df['volatility'].mean()
                    if not np.isnan(vol_mean):
                        axes[vol_panel_idx].axhline(y=vol_mean, color='black', linestyle='-', alpha=0.3, label='Vol Mean')
            
            # 動的乗数パネルの参照線
            if mult_plots_added:
                mult_panel_idx = len(axes) - 1  # 最後のパネル
                if mult_panel_idx < len(axes) and mult_panel_idx >= 0:
                    axes[mult_panel_idx].axhline(y=2, color='blue', linestyle='--', alpha=0.5, label='Min')
                    axes[mult_panel_idx].axhline(y=5, color='red', linestyle='--', alpha=0.5, label='Max')
                    axes[mult_panel_idx].axhline(y=3.5, color='gray', linestyle='-', alpha=0.3, label='Mid')
                    axes[mult_panel_idx].set_ylim(1.5, 5.5)
            
            # 保存または表示
            if savefig:
                try:
                    plt.tight_layout()
                except:
                    # tight_layoutが失敗した場合は調整せずに保存
                    pass
                plt.savefig(savefig, dpi=300, bbox_inches='tight')
                print(f"チャートを保存しました: {savefig}")
            else:
                try:
                    plt.tight_layout()
                except:
                    # tight_layoutが失敗した場合は手動で調整
                    plt.subplots_adjust(hspace=0.3, wspace=0.1)
                plt.show()
                
        except Exception as e:
            print(f"チャート描画中にエラーが発生しました: {str(e)}")
            print("デバッグ情報:")
            print(f"  - メインプロット数: {len(main_plots)}")
            print(f"  - サブプロット数: {len(sub_plots)}")
            print(f"  - 総パネル数: {total_panels}")
            
            # 基本的なチャートのみ表示を試行
            try:
                print("基本チャートのみで再試行...")
                basic_kwargs = dict(
                    type='candle',
                    figsize=figsize,
                    title=title,
                    style=style,
                    datetime_format='%Y-%m-%d',
                    xrotation=45,
                    returnfig=True,
                    warn_too_much_data=len(df) + 1000
                )
                
                if show_volume:
                    basic_kwargs['volume'] = True
                
                if main_plots:
                    basic_kwargs['addplot'] = main_plots
                
                fig, axes = mpf.plot(df, **basic_kwargs)
                self.fig = fig
                self.axes = axes
                
                if savefig:
                    try:
                        plt.tight_layout()
                    except:
                        pass
                    plt.savefig(savefig, dpi=300, bbox_inches='tight')
                    print(f"基本チャートを保存しました: {savefig}")
                else:
                    try:
                        plt.tight_layout()
                    except:
                        plt.subplots_adjust(hspace=0.3, wspace=0.1)
                    plt.show()
                    
            except Exception as e2:
                print(f"基本チャート描画も失敗しました: {str(e2)}")
                raise e

    def print_statistics(self) -> None:
        """
        Z Adaptive Flowの統計情報を表示
        """
        if self.z_adaptive_flow is None:
            print("インジケーターが計算されていません。")
            return
        
        result = self.z_adaptive_flow.get_detailed_result()
        
        print("\n=== Z Adaptive Flow 統計情報 ===")
        print(f"現在のトレンド: {self.z_adaptive_flow.get_current_trend()}")
        
        # 最新値
        if len(result.basis) > 0:
            print(f"\n最新値:")
            print(f"  - Basis: {result.basis[-1]:.4f}")
            print(f"  - Level: {result.level[-1]:.4f}")
            print(f"  - Upper Band: {result.upper[-1]:.4f}")
            print(f"  - Lower Band: {result.lower[-1]:.4f}")
            print(f"  - トレンド状態: {result.trend_state[-1]}")
            print(f"  - スロー乗数: {result.slow_multiplier[-1]:.2f}")
            print(f"  - ボラティリティ乗数: {result.volatility_multiplier[-1]:.2f}")
        
        # シグナル統計
        total_long = np.sum(result.long_signals)
        total_short = np.sum(result.short_signals)
        print(f"\nシグナル統計:")
        print(f"  - ロングシグナル: {total_long}回")
        print(f"  - ショートシグナル: {total_short}回")
        print(f"  - 総シグナル: {total_long + total_short}回")
        
        # トレンド状態統計
        bullish_count = np.sum(result.trend_state == 1)
        bearish_count = np.sum(result.trend_state == -1)
        total_count = len(result.trend_state)
        if total_count > 0:
            print(f"\nトレンド統計:")
            print(f"  - 強気期間: {bullish_count}期間 ({bullish_count/total_count*100:.1f}%)")
            print(f"  - 弱気期間: {bearish_count}期間 ({bearish_count/total_count*100:.1f}%)")
        
        # 乗数統計
        if len(result.slow_multiplier) > 0:
            print(f"\n乗数統計:")
            print(f"  - スロー乗数: 平均={np.nanmean(result.slow_multiplier):.2f}, "
                  f"範囲={np.nanmin(result.slow_multiplier):.2f}-{np.nanmax(result.slow_multiplier):.2f}")
            print(f"  - ボラティリティ乗数: 平均={np.nanmean(result.volatility_multiplier):.2f}, "
                  f"範囲={np.nanmin(result.volatility_multiplier):.2f}-{np.nanmax(result.volatility_multiplier):.2f}")


def main():
    """メイン関数"""
    # コマンドライン引数を処理
    import argparse
    parser = argparse.ArgumentParser(description='Z Adaptive Flowの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--ma-type', type=str, default='z_adaptive_ma', choices=['hma', 'alma', 'z_adaptive_ma', 'zlema'], 
                       help='MAタイプ')
    parser.add_argument('--volatility-type', type=str, default='volatility', choices=['volatility', 'atr'], 
                       help='ボラティリティタイプ')
    parser.add_argument('--adaptive-trigger', type=str, default='chop_trend', 
                       choices=['chop_trend', 'chop_er', 'adx_vol', 'normalized_adx', 'efficiency_ratio'],
                       help='適応トリガーインジケーター')
    parser.add_argument('--length', type=int, default=10, help='メイン期間')
    parser.add_argument('--smooth-length', type=int, default=14, help='ボラティリティ平滑化期間')
    parser.add_argument('--no-volume', action='store_true', help='出来高を非表示')
    parser.add_argument('--no-signals', action='store_true', help='シグナルマーカーを非表示')
    parser.add_argument('--stats', action='store_true', help='統計情報を表示')
    parser.add_argument('--max-points', type=int, default=1000, help='最大データポイント数（デフォルト: 1000）')
    args = parser.parse_args()
    
    # チャートを作成
    chart = ZAdaptiveFlowChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        length=args.length,
        smooth_length=args.smooth_length,
        ma_type=args.ma_type,
        volatility_type=args.volatility_type,
        adaptive_trigger=args.adaptive_trigger
    )
    
    # 統計情報表示（オプション）
    if args.stats:
        chart.print_statistics()
    
    # チャート描画
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        show_volume=not args.no_volume,
        show_signals=not args.no_signals,
        savefig=args.output,
        max_data_points=args.max_points
    )


if __name__ == "__main__":
    main() 