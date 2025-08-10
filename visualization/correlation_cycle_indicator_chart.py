#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from typing import Dict, Any, Optional, Tuple, List
from matplotlib.patches import Rectangle

# データ取得のための依存関係
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource

# インジケーター
from indicators.trend_filter.correlation_cycle_indicator import CorrelationCycleIndicator


class CorrelationCycleIndicatorChart:
    """
    Correlation Cycle Indicator を表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - Real成分とImaginary成分（直交成分）
    - フェーザー角度（-180°〜+180°）
    - 市場状態（+1: 上昇トレンド, 0: サイクル, -1: 下降トレンド）
    - 角度変化率（Rate of Change）
    - サイクルモード vs トレンドモード判定
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.cycle_indicator = None
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
        print("\\nデータを読み込み・処理中...")
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
                            period: int = 20,
                            src_type: str = 'close',
                            trend_threshold: float = 9.0,
                            use_theoretical_input: bool = False,
                            theoretical_period: int = 20
                           ) -> None:
        """
        Correlation Cycle Indicatorを計算する
        
        Args:
            period: 相関計算期間（デフォルト: 20）
            src_type: ソースタイプ（デフォルト: 'close'）
            trend_threshold: トレンド判定閾値（デフォルト: 9.0度）
            use_theoretical_input: 理論的入力を使用するか（デフォルト: False）
            theoretical_period: 理論的サイン波の周期（デフォルト: 20）
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\\nCorrelation Cycle Indicatorを計算中...")
        
        # Correlation Cycle Indicatorを計算
        self.cycle_indicator = CorrelationCycleIndicator(
            period=period,
            src_type=src_type,
            trend_threshold=trend_threshold,
            use_theoretical_input=use_theoretical_input,
            theoretical_period=theoretical_period
        )
        
        # インジケーターの計算
        print("計算を実行します...")
        result = self.cycle_indicator.calculate(self.data)
        
        # 結果の取得
        real_component = result.real_component
        imag_component = result.imag_component
        angle = result.angle
        state = result.state
        rate_of_change = result.rate_of_change
        cycle_mode = result.cycle_mode
        ri_mode = result.ri_mode
        
        print(f"Correlation Cycle Indicator計算完了 - データ点数: {len(angle)}")
        
        # 統計情報
        uptrend_count = (state == 1).sum()
        downtrend_count = (state == -1).sum()
        cycle_count = (state == 0).sum()
        
        cycle_mode_count = (cycle_mode == 1).sum()
        trend_mode_count = (cycle_mode == 0).sum()
        
        ri_trend_count = (ri_mode == 1).sum()
        ri_cycle_count = (ri_mode == 0).sum()
        
        print(f"状態統計:")
        print(f"  上昇トレンド: {uptrend_count} ({uptrend_count/len(state)*100:.1f}%)")
        print(f"  下降トレンド: {downtrend_count} ({downtrend_count/len(state)*100:.1f}%)")
        print(f"  サイクル: {cycle_count} ({cycle_count/len(state)*100:.1f}%)")
        print(f"モード統計（角度変化率ベース）:")
        print(f"  サイクルモード: {cycle_mode_count} ({cycle_mode_count/len(cycle_mode)*100:.1f}%)")
        print(f"  トレンドモード: {trend_mode_count} ({trend_mode_count/len(cycle_mode)*100:.1f}%)")
        print(f"モード統計（Real vs Imaginaryベース）:")
        print(f"  トレンドモード（|Real|>|Imag|）: {ri_trend_count} ({ri_trend_count/len(ri_mode)*100:.1f}%)")
        print(f"  サイクルモード（|Real|<=|Imag|）: {ri_cycle_count} ({ri_cycle_count/len(ri_mode)*100:.1f}%)")
        
        print(f"統計 - フェーザー角度平均: {np.nanmean(angle):.2f}°")
        print(f"角度範囲: {np.nanmin(angle):.2f}° - {np.nanmax(angle):.2f}°")
        print(f"Real成分平均: {np.nanmean(real_component):.4f}")
        print(f"Imaginary成分平均: {np.nanmean(imag_component):.4f}")
        print(f"角度変化率平均: {np.nanmean(rate_of_change):.2f}°/期間")
        
        print("Correlation Cycle Indicator計算完了")
    
    def _add_mode_background_colors(self, df: pd.DataFrame, main_axis) -> None:
        """
        メインチャートにトレンド・サイクルモードに基づくバックグラウンドカラーを追加する
        
        Args:
            df: チャートデータ
            main_axis: メインチャートの軸
        """
        try:
            # Y軸の範囲を取得
            y_min, y_max = main_axis.get_ylim()
            
            # モードの変化点を検出
            mode_changes = []
            current_mode = None
            
            for i, (_, row) in enumerate(df.iterrows()):
                if pd.notna(row['cycle_mode']):
                    if current_mode != row['cycle_mode']:
                        mode_changes.append((i, row['cycle_mode']))
                        current_mode = row['cycle_mode']
            
            # 最初と最後のポイントを追加
            if mode_changes:
                if mode_changes[0][0] > 0:
                    # 最初のポイントを追加
                    first_mode = df['cycle_mode'].iloc[0] if pd.notna(df['cycle_mode'].iloc[0]) else 1.0
                    mode_changes.insert(0, (0, first_mode))
                
                # 最後のポイントを追加
                if mode_changes[-1][0] < len(df) - 1:
                    mode_changes.append((len(df) - 1, mode_changes[-1][1]))
            
            # バックグラウンド矩形を描画
            for i in range(len(mode_changes) - 1):
                start_idx = mode_changes[i][0]
                end_idx = mode_changes[i + 1][0]
                mode = mode_changes[i][1]
                
                # モードに基づく色の設定
                if mode == 0.0:  # トレンドモード
                    color = 'lightgreen'
                    alpha = 0.15
                else:  # サイクルモード (mode == 1.0)
                    color = 'lightcoral'
                    alpha = 0.15
                
                # 矩形を描画
                width = end_idx - start_idx
                if width > 0:
                    rect = Rectangle(
                        (start_idx, y_min),
                        width,
                        y_max - y_min,
                        facecolor=color,
                        alpha=alpha,
                        edgecolor='none',
                        zorder=0  # 背景に配置
                    )
                    main_axis.add_patch(rect)
            
            # 凡例を追加
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='lightgreen', alpha=0.15, label='トレンドモード'),
                Patch(facecolor='lightcoral', alpha=0.15, label='サイクルモード')
            ]
            main_axis.legend(handles=legend_elements, loc='upper left', fontsize=8)
            
        except Exception as e:
            print(f"バックグラウンドカラー追加中にエラー: {e}")
            import traceback
            traceback.print_exc()
    
    def _add_ri_mode_background_colors(self, df: pd.DataFrame, main_axis) -> None:
        """
        メインチャートにReal vs Imaginaryモードに基づくバックグラウンドカラーを追加する
        
        Args:
            df: チャートデータ
            main_axis: メインチャートの軸
        """
        try:
            # モードの変化点を検出
            mode_changes = []
            current_mode = None
            
            for i, (_, row) in enumerate(df.iterrows()):
                if pd.notna(row['ri_mode']):
                    if current_mode != row['ri_mode']:
                        mode_changes.append((i, row['ri_mode']))
                        current_mode = row['ri_mode']
            
            # 最初と最後のポイントを追加
            if mode_changes:
                if mode_changes[0][0] > 0:
                    # 最初のポイントを追加
                    first_mode = df['ri_mode'].iloc[0] if pd.notna(df['ri_mode'].iloc[0]) else 1.0
                    mode_changes.insert(0, (0, first_mode))
                
                # 最後のポイントを追加
                if mode_changes[-1][0] < len(df) - 1:
                    mode_changes.append((len(df) - 1, mode_changes[-1][1]))
            
            # axvspanを使用してバックグラウンドを描画
            for i in range(len(mode_changes) - 1):
                start_idx = mode_changes[i][0]
                end_idx = mode_changes[i + 1][0]
                mode = mode_changes[i][1]
                
                # モードに基づく色の設定
                if mode == 1.0:  # トレンドモード（|Real| > |Imag|）
                    color = 'green'
                    alpha = 0.15
                else:  # サイクルモード（|Real| <= |Imag|）
                    color = 'red'
                    alpha = 0.15
                
                # axvspanで範囲を描画
                if end_idx > start_idx:
                    main_axis.axvspan(start_idx, end_idx, facecolor=color, alpha=alpha, zorder=0)
            
            # 凡例を追加
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', alpha=0.15, label='トレンドモード（|Real|>|Imag|）'),
                Patch(facecolor='red', alpha=0.15, label='サイクルモード（|Real|<=|Imag|）')
            ]
            main_axis.legend(handles=legend_elements, loc='upper left', fontsize=8)
            
        except Exception as e:
            print(f"Real vs Imaginaryバックグラウンドカラー追加中にエラー: {e}")
            import traceback
            traceback.print_exc()
            
    def plot(self, 
            title: str = "Correlation Cycle Indicator", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (16, 18),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとCorrelation Cycle Indicatorを描画する
        
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
            
        if self.cycle_indicator is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # Correlation Cycle Indicatorの値を取得
        print("Correlation Cycle Indicatorデータを取得中...")
        result = self.cycle_indicator.calculate(self.data)
        
        real_component = result.real_component
        imag_component = result.imag_component
        angle = result.angle
        state = result.state
        rate_of_change = result.rate_of_change
        cycle_mode = result.cycle_mode
        ri_mode = result.ri_mode
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'real_component': real_component,
                'imag_component': imag_component,
                'angle': angle,
                'state': state,
                'rate_of_change': rate_of_change,
                'cycle_mode': cycle_mode,
                'ri_mode': ri_mode
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        
        # 状態とモードに基づく色分け
        df['state_up'] = np.where(df['state'] == 1, df['state'], np.nan)
        df['state_down'] = np.where(df['state'] == -1, df['state'], np.nan)
        df['state_cycle'] = np.where(df['state'] == 0, df['state'], np.nan)
        
        df['cycle_mode_active'] = np.where(df['cycle_mode'] == 1, 1, np.nan)
        df['trend_mode_active'] = np.where(df['cycle_mode'] == 0, 0, np.nan)
        
        # Real vs Imaginaryモードの色分け
        df['ri_trend_mode'] = np.where(df['ri_mode'] == 1, 1, np.nan)
        df['ri_cycle_mode'] = np.where(df['ri_mode'] == 0, 0, np.nan)
        
        # フェーザー角度の色分け（正負で分ける）
        df['angle_positive'] = np.where(df['angle'] >= 0, df['angle'], np.nan)
        df['angle_negative'] = np.where(df['angle'] < 0, df['angle'], np.nan)
        
        # デバッグ情報の出力
        up_count = (~np.isnan(df['state_up'])).sum()
        down_count = (~np.isnan(df['state_down'])).sum()
        cycle_count = (~np.isnan(df['state_cycle'])).sum()
        cycle_mode_count = (~np.isnan(df['cycle_mode_active'])).sum()
        trend_mode_count = (~np.isnan(df['trend_mode_active'])).sum()
        print(f"State Up: {up_count}, Down: {down_count}, Cycle: {cycle_count}")
        print(f"Cycle Mode: {cycle_mode_count}, Trend Mode: {trend_mode_count}")
        
        # 全てがNaNの場合は、適切なダミー値を追加
        if up_count == 0:
            df.loc[df.index[0], 'state_up'] = 1
        if down_count == 0:
            df.loc[df.index[0], 'state_down'] = -1
        if cycle_count == 0:
            df.loc[df.index[0], 'state_cycle'] = 0
        
        # mplfinanceでプロット用の設定
        main_plots = []
        
        # Real成分とImaginary成分パネル
        real_panel = mpf.make_addplot(df['real_component'], panel=1, color='blue', width=2.0, 
                                     ylabel='Real & Imag\\nComponents', secondary_y=False, 
                                     label='Real', type='line')
        imag_panel = mpf.make_addplot(df['imag_component'], panel=1, color='red', width=2.0, 
                                     secondary_y=False, label='Imaginary', type='line')
        
        # フェーザー角度パネル
        angle_pos_panel = mpf.make_addplot(df['angle_positive'], panel=2, color='green', width=2.0, 
                                          ylabel='Phasor Angle\\n(-180° to +180°)', secondary_y=False, 
                                          label='Angle (+)', type='line')
        angle_neg_panel = mpf.make_addplot(df['angle_negative'], panel=2, color='red', width=2.0, 
                                          secondary_y=False, label='Angle (-)', type='line')
        
        # 市場状態パネル
        state_up_panel = mpf.make_addplot(df['state_up'], panel=3, color='green', width=4.0, 
                                         ylabel='Market State\\n(+1=Up, -1=Down, 0=Cycle)', secondary_y=False, 
                                         label='Up Trend', type='line')
        state_down_panel = mpf.make_addplot(df['state_down'], panel=3, color='red', width=4.0, 
                                           secondary_y=False, label='Down Trend', type='line')
        state_cycle_panel = mpf.make_addplot(df['state_cycle'], panel=3, color='gray', width=3.0, 
                                            secondary_y=False, label='Cycle', type='line')
        
        # 角度変化率パネル
        roc_panel = mpf.make_addplot(df['rate_of_change'], panel=4, color='orange', width=1.5, 
                                    ylabel='Rate of Change\\n(degrees/period)', secondary_y=False, 
                                    label='ROC')
        
        # サイクル・トレンドモードパネル（角度変化率ベース）
        cycle_mode_panel = mpf.make_addplot(df['cycle_mode_active'], panel=5, color='cyan', width=3.0, 
                                           ylabel='Angle ROC Mode\\n(1=Cycle, 0=Trend)', secondary_y=False, 
                                           label='Cycle Mode', type='line')
        trend_mode_panel = mpf.make_addplot(df['trend_mode_active'], panel=5, color='purple', width=3.0, 
                                           secondary_y=False, label='Trend Mode', type='line')
        
        # Real vs Imaginaryモードパネル
        ri_trend_panel = mpf.make_addplot(df['ri_trend_mode'], panel=6, color='green', width=3.0,
                                         ylabel='Real vs Imag Mode\\n(1=Trend, 0=Cycle)', secondary_y=False,
                                         label='RI Trend', type='line')
        ri_cycle_panel = mpf.make_addplot(df['ri_cycle_mode'], panel=6, color='red', width=3.0,
                                         secondary_y=False, label='RI Cycle', type='line')
        
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
            kwargs['panel_ratios'] = (5, 1, 1, 1, 1, 1, 1, 1)  # メイン:出来高:Real/Imag:Angle:State:ROC:AngleMode:RIMode
            # 出来高を表示する場合は、オシレーターのパネル番号を+1する
            real_panel = mpf.make_addplot(df['real_component'], panel=2, color='blue', width=2.0, 
                                         ylabel='Real & Imag Components', secondary_y=False, label='Real', type='line')
            imag_panel = mpf.make_addplot(df['imag_component'], panel=2, color='red', width=2.0, 
                                         secondary_y=False, label='Imaginary', type='line')
            angle_pos_panel = mpf.make_addplot(df['angle_positive'], panel=3, color='green', width=2.0, 
                                              ylabel='Phasor Angle', secondary_y=False, label='Angle (+)', type='line')
            angle_neg_panel = mpf.make_addplot(df['angle_negative'], panel=3, color='red', width=2.0, 
                                              secondary_y=False, label='Angle (-)', type='line')
            state_up_panel = mpf.make_addplot(df['state_up'], panel=4, color='green', width=4.0, 
                                             ylabel='Market State', secondary_y=False, label='Up Trend', type='line')
            state_down_panel = mpf.make_addplot(df['state_down'], panel=4, color='red', width=4.0, 
                                               secondary_y=False, label='Down Trend', type='line')
            state_cycle_panel = mpf.make_addplot(df['state_cycle'], panel=4, color='gray', width=3.0, 
                                                secondary_y=False, label='Cycle', type='line')
            roc_panel = mpf.make_addplot(df['rate_of_change'], panel=5, color='orange', width=1.5, 
                                        ylabel='Rate of Change', secondary_y=False, label='ROC')
            cycle_mode_panel = mpf.make_addplot(df['cycle_mode_active'], panel=6, color='cyan', width=3.0, 
                                               ylabel='Angle ROC Mode', secondary_y=False, label='Cycle Mode', type='line')
            trend_mode_panel = mpf.make_addplot(df['trend_mode_active'], panel=6, color='purple', width=3.0, 
                                               secondary_y=False, label='Trend Mode', type='line')
            ri_trend_panel = mpf.make_addplot(df['ri_trend_mode'], panel=7, color='green', width=3.0,
                                             ylabel='Real vs Imag Mode', secondary_y=False, label='RI Trend', type='line')
            ri_cycle_panel = mpf.make_addplot(df['ri_cycle_mode'], panel=7, color='red', width=3.0,
                                             secondary_y=False, label='RI Cycle', type='line')
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (5, 1, 1, 1, 1, 1, 1)  # メイン:Real/Imag:Angle:State:ROC:AngleMode:RIMode
        
        # すべてのプロットを結合
        all_plots = main_plots + [
            real_panel, imag_panel,
            angle_pos_panel, angle_neg_panel,
            state_up_panel, state_down_panel, state_cycle_panel,
            roc_panel,
            cycle_mode_panel, trend_mode_panel,
            ri_trend_panel, ri_cycle_panel
        ]
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        panel_offset = 2 if show_volume else 1
        
        # Real/Imaginaryパネル
        real_imag_axis = axes[panel_offset]
        real_imag_axis.set_ylim(-1.1, 1.1)
        real_imag_axis.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=1)
        real_imag_axis.axhline(y=1, color='black', linestyle='--', alpha=0.3, linewidth=1)
        real_imag_axis.axhline(y=-1, color='black', linestyle='--', alpha=0.3, linewidth=1)
        
        # フェーザー角度パネル
        angle_axis = axes[panel_offset + 1]
        angle_axis.set_ylim(-190, 190)
        angle_axis.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=1)
        angle_axis.axhline(y=90, color='green', linestyle='--', alpha=0.5, linewidth=1)
        angle_axis.axhline(y=-90, color='red', linestyle='--', alpha=0.5, linewidth=1)
        angle_axis.axhline(y=180, color='black', linestyle='-', alpha=0.3, linewidth=1)
        angle_axis.axhline(y=-180, color='black', linestyle='-', alpha=0.3, linewidth=1)
        
        # 市場状態パネル
        state_axis = axes[panel_offset + 2]
        state_axis.set_ylim(-1.5, 1.5)
        state_axis.set_yticks([-1, 0, 1])
        state_axis.set_yticklabels(['Down', 'Cycle', 'Up'])
        state_axis.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=1)
        state_axis.axhline(y=1, color='green', linestyle='--', alpha=0.6, linewidth=1)
        state_axis.axhline(y=-1, color='red', linestyle='--', alpha=0.6, linewidth=1)
        
        # 角度変化率パネル
        roc_axis = axes[panel_offset + 3]
        roc_axis.axhline(y=9, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Trend Threshold')
        roc_axis.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        roc_axis.set_ylim(0, max(50, df['rate_of_change'].max() * 1.1))
        
        # 角度変化率モードパネル
        mode_axis = axes[panel_offset + 4]
        mode_axis.set_ylim(-0.2, 1.2)
        mode_axis.set_yticks([0, 1])
        mode_axis.set_yticklabels(['Trend', 'Cycle'])
        mode_axis.axhline(y=0.5, color='black', linestyle='-', alpha=0.7, linewidth=1)
        
        # Real vs Imaginaryモードパネル
        ri_mode_axis = axes[panel_offset + 5]
        ri_mode_axis.set_ylim(-0.2, 1.2)
        ri_mode_axis.set_yticks([0, 1])
        ri_mode_axis.set_yticklabels(['Cycle', 'Trend'])
        ri_mode_axis.axhline(y=0.5, color='black', linestyle='-', alpha=0.7, linewidth=1)
        
        # 統計情報の表示
        print(f"\\n=== Correlation Cycle Indicator統計 ===")
        total_points = len(df)
        uptrend_points = len(df[df['state'] == 1])
        downtrend_points = len(df[df['state'] == -1])
        cycle_points = len(df[df['state'] == 0])
        cycle_mode_points = len(df[df['cycle_mode'] == 1])
        trend_mode_points = len(df[df['cycle_mode'] == 0])
        ri_trend_points = len(df[df['ri_mode'] == 1])
        ri_cycle_points = len(df[df['ri_mode'] == 0])
        
        print(f"総データ点数: {total_points}")
        print(f"上昇トレンド: {uptrend_points} ({uptrend_points/total_points*100:.1f}%)")
        print(f"下降トレンド: {downtrend_points} ({downtrend_points/total_points*100:.1f}%)")
        print(f"サイクル: {cycle_points} ({cycle_points/total_points*100:.1f}%)")
        print(f"サイクルモード（角度変化率ベース）: {cycle_mode_points} ({cycle_mode_points/total_points*100:.1f}%)")
        print(f"トレンドモード（角度変化率ベース）: {trend_mode_points} ({trend_mode_points/total_points*100:.1f}%)")
        print(f"トレンドモード（|Real|>|Imag|ベース）: {ri_trend_points} ({ri_trend_points/total_points*100:.1f}%)")
        print(f"サイクルモード（|Real|<=|Imag|ベース）: {ri_cycle_points} ({ri_cycle_points/total_points*100:.1f}%)")
        print(f"フェーザー角度 - 平均: {df['angle'].mean():.2f}°, 範囲: {df['angle'].min():.2f}° - {df['angle'].max():.2f}°")
        print(f"Real成分 - 平均: {df['real_component'].mean():.4f}, 範囲: {df['real_component'].min():.4f} - {df['real_component'].max():.4f}")
        print(f"Imaginary成分 - 平均: {df['imag_component'].mean():.4f}, 範囲: {df['imag_component'].min():.4f} - {df['imag_component'].max():.4f}")
        print(f"角度変化率 - 平均: {df['rate_of_change'].mean():.2f}°, 最大: {df['rate_of_change'].max():.2f}°")
        
        # モード遷移の分析
        state_changes = (df['state'] != df['state'].shift(1)).sum()
        mode_changes = (df['cycle_mode'] != df['cycle_mode'].shift(1)).sum()
        print(f"状態変更: {state_changes}回")
        print(f"モード変更: {mode_changes}回")
        
        
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
    parser = argparse.ArgumentParser(description='Correlation Cycle Indicatorの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--period', type=int, default=20, help='相関計算期間')
    parser.add_argument('--threshold', type=float, default=9.0, help='トレンド判定閾値（角度変化率）')
    parser.add_argument('--src-type', type=str, default='close', help='ソースタイプ')
    parser.add_argument('--theoretical', action='store_true', help='理論的入力を使用する')
    parser.add_argument('--theoretical-period', type=int, default=20, help='理論的サイン波の周期')
    args = parser.parse_args()
    
    # チャートを作成
    chart = CorrelationCycleIndicatorChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        period=args.period,
        src_type=args.src_type,
        trend_threshold=args.threshold,
        use_theoretical_input=args.theoretical,
        theoretical_period=args.theoretical_period
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output
    )


if __name__ == "__main__":
    main()