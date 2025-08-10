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

# プロジェクトルートをパスに追加
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# データ取得のための依存関係
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource

# インジケーター
from indicators.volatility.x_atr import XATR


class XATRChart:
    """
    X_ATRを表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - X_ATR値（ボラティリティ指標）
    - ミッドライン
    - ボラティリティ信号（1=低ボラ、-1=高ボラ）
    - True Range値
    - 動的期間値（動的期間適応が有効な場合）
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.x_atr = None
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
                            period: float = 20.0,
                            tr_method: str = 'atr',
                            smoother_type: str = 'super_smoother',
                            src_type: str = 'close',
                            enable_kalman: bool = False,
                            kalman_type: str = 'unscented',
                            period_mode: str = 'dynamic',
                            cycle_detector_type: str = 'absolute_ultimate',
                            cycle_detector_cycle_part: float = 1.0,
                            cycle_detector_max_cycle: int = 55,
                            cycle_detector_min_cycle: int = 5,
                            cycle_period_multiplier: float = 1.0,
                            cycle_detector_period_range: Tuple[int, int] = (5, 120),
                            midline_period: int = 100,
                            smoother_params: Optional[Dict[str, Any]] = None,
                            kalman_params: Optional[Dict[str, Any]] = None
                           ) -> None:
        """
        X_ATRを計算する
        
        Args:
            period: スムージング期間
            tr_method: True Range計算方法 ('atr' または 'str')
            smoother_type: スムージング手法
            src_type: プライスソース
            enable_kalman: カルマンフィルター使用フラグ
            kalman_type: カルマンフィルター種別
            period_mode: 期間モード ('fixed' または 'dynamic')
            cycle_detector_type: サイクル検出器タイプ
            cycle_detector_cycle_part: サイクル検出器のサイクル部分倍率
            cycle_detector_max_cycle: サイクル検出器の最大サイクル期間
            cycle_detector_min_cycle: サイクル検出器の最小サイクル期間
            cycle_period_multiplier: サイクル期間の乗数
            cycle_detector_period_range: サイクル検出器の周期範囲
            midline_period: ミッドライン計算期間
            smoother_params: スムーサー固有パラメータ
            kalman_params: カルマンフィルター固有パラメータ
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\nX_ATRを計算中...")
        
        # X_ATRを計算
        self.x_atr = XATR(
            period=period,
            tr_method=tr_method,
            smoother_type=smoother_type,
            src_type=src_type,
            enable_kalman=enable_kalman,
            kalman_type=kalman_type,
            period_mode=period_mode,
            cycle_detector_type=cycle_detector_type,
            cycle_detector_cycle_part=cycle_detector_cycle_part,
            cycle_detector_max_cycle=cycle_detector_max_cycle,
            cycle_detector_min_cycle=cycle_detector_min_cycle,
            cycle_period_multiplier=cycle_period_multiplier,
            cycle_detector_period_range=cycle_detector_period_range,
            midline_period=midline_period,
            smoother_params=smoother_params,
            kalman_params=kalman_params
        )
        
        # X_ATRの計算
        print("計算を実行します...")
        result = self.x_atr.calculate(self.data)
        
        print(f"X_ATR計算完了 - 値: {len(result.values)}")
        
        # NaN値のチェック
        nan_count = np.isnan(result.values).sum()
        valid_count = (~np.isnan(result.values)).sum()
        volatility_signal_count = (result.volatility_signal != 0).sum()
        print(f"NaN値: {nan_count}, 有効値: {valid_count}")
        print(f"ボラティリティ信号 - 有効: {volatility_signal_count}, 低ボラ: {(result.volatility_signal == 1).sum()}, 高ボラ: {(result.volatility_signal == -1).sum()}")
        
        # 統計情報
        if valid_count > 0:
            valid_values = result.values[~np.isnan(result.values)]
            print(f"X_ATR統計 - 平均: {np.mean(valid_values):.4f}, 範囲: {np.min(valid_values):.4f} - {np.max(valid_values):.4f}")
        
        print("X_ATR計算完了")
            
    def plot(self, 
            title: str = "X_ATR (Extended Average True Range)", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            show_percentage: bool = False,
            figsize: Tuple[int, int] = (14, 12),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとX_ATRを描画する
        
        Args:
            title: チャートのタイトル
            start_date: 表示開始日（フォーマット: YYYY-MM-DD）
            end_date: 表示終了日（フォーマット: YYYY-MM-DD）
            show_volume: 出来高を表示するか
            show_percentage: %ベース値を表示するか（False=金額ベース）
            figsize: 図のサイズ
            style: mplfinanceのスタイル
            savefig: 保存先のパス（指定しない場合は表示のみ）
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        if self.x_atr is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # X_ATRの値を取得
        print("X_ATRデータを取得中...")
        result = self.x_atr.calculate(self.data)
        
        # 表示する値を選択（金額ベースまたは%ベース）
        if show_percentage:
            atr_values = result.values_percentage
            tr_values = result.true_range_percentage
            midline_values = result.midline_percentage
            value_suffix = " (%)"
        else:
            atr_values = result.values
            tr_values = result.true_range
            midline_values = result.midline
            value_suffix = ""
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'x_atr': atr_values,
                'true_range': tr_values,
                'midline': midline_values,
                'volatility_signal': result.volatility_signal,
                'raw_high': result.raw_high,
                'raw_low': result.raw_low
            }
        )
        
        # カルマンフィルター結果があれば追加
        if result.filtered_high is not None and result.filtered_low is not None:
            full_df['filtered_high'] = result.filtered_high
            full_df['filtered_low'] = result.filtered_low
        
        # 動的期間があれば追加
        if result.dynamic_periods is not None:
            full_df['dynamic_periods'] = result.dynamic_periods
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        print(f"X_ATRデータ確認 - NaN: {df['x_atr'].isna().sum()}")
        
        # ボラティリティ状態に基づく色分け
        df['atr_low_vol'] = np.where(df['volatility_signal'] == 1, df['x_atr'], np.nan)
        df['atr_high_vol'] = np.where(df['volatility_signal'] == -1, df['x_atr'], np.nan)
        
        # ミッドラインの色分け（ボラティリティ状態に応じて）
        df['midline_low_vol'] = np.where(df['volatility_signal'] == 1, df['midline'], np.nan)
        df['midline_high_vol'] = np.where(df['volatility_signal'] == -1, df['midline'], np.nan)
        
        # NaN値を含む行を出力（最初の5行のみ）
        nan_rows = df[df['x_atr'].isna() | df['midline'].isna()]
        if not nan_rows.empty:
            print(f"NaN値を含む行: {len(nan_rows)}行")
            if len(nan_rows) > 0:
                print(nan_rows.head())
        
        # mplfinanceでプロット用の設定
        # 1. メインチャート上のプロット（価格とボリューム用）
        main_plots = []
        
        # 2. X_ATRパネル
        atr_low_vol_plot = mpf.make_addplot(df['atr_low_vol'], panel=1, color='blue', width=2, 
                                          ylabel=f'X-ATR{value_suffix}', secondary_y=False, label='Low Volatility')
        atr_high_vol_plot = mpf.make_addplot(df['atr_high_vol'], panel=1, color='red', width=2, 
                                          secondary_y=False, label='High Volatility')
        midline_low_vol_plot = mpf.make_addplot(df['midline_low_vol'], panel=1, color='darkblue', width=1, 
                                             alpha=0.7, secondary_y=False, label='Midline (Low Vol)')
        midline_high_vol_plot = mpf.make_addplot(df['midline_high_vol'], panel=1, color='darkred', width=1, 
                                             alpha=0.7, secondary_y=False, label='Midline (High Vol)')
        
        # 3. True Rangeパネル
        tr_panel = mpf.make_addplot(df['true_range'], panel=2, color='purple', width=1.2, 
                                    ylabel=f'True Range{value_suffix}', secondary_y=False, label='True Range')
        
        # 4. ボラティリティ信号パネル
        volatility_panel = mpf.make_addplot(df['volatility_signal'], panel=3, color='orange', width=1.5, 
                                      ylabel='Volatility Signal', secondary_y=False, label='Signal', type='line')
        
        # 5. 動的期間（使用している場合）
        dynamic_plots = []
        if 'dynamic_periods' in df.columns and not df['dynamic_periods'].isna().all():
            dynamic_plot = mpf.make_addplot(df['dynamic_periods'], panel=4, color='brown', 
                                          width=1.5, alpha=0.8, secondary_y=False, label='Dynamic Period')
            dynamic_plots = [dynamic_plot]
        
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
            if dynamic_plots:
                kwargs['volume'] = True
                kwargs['panel_ratios'] = (4, 1, 2, 1, 1, 1)  # メイン:出来高:X_ATR:TR:ボラ信号:動的期間
                # 出来高を表示する場合は、オシレーターのパネル番号を+1する
                atr_low_vol_plot = mpf.make_addplot(df['atr_low_vol'], panel=2, color='blue', width=2, 
                                                  ylabel=f'X-ATR{value_suffix}', secondary_y=False, label='Low Volatility')
                atr_high_vol_plot = mpf.make_addplot(df['atr_high_vol'], panel=2, color='red', width=2, 
                                                  secondary_y=False, label='High Volatility')
                midline_low_vol_plot = mpf.make_addplot(df['midline_low_vol'], panel=2, color='darkblue', width=1, 
                                                     alpha=0.7, secondary_y=False, label='Midline (Low Vol)')
                midline_high_vol_plot = mpf.make_addplot(df['midline_high_vol'], panel=2, color='darkred', width=1, 
                                                     alpha=0.7, secondary_y=False, label='Midline (High Vol)')
                tr_panel = mpf.make_addplot(df['true_range'], panel=3, color='purple', width=1.2, 
                                            ylabel=f'True Range{value_suffix}', secondary_y=False, label='True Range')
                volatility_panel = mpf.make_addplot(df['volatility_signal'], panel=4, color='orange', width=1.5, 
                                              ylabel='Volatility Signal', secondary_y=False, label='Signal', type='line')
                
                # 動的期間も更新
                dynamic_plot = mpf.make_addplot(df['dynamic_periods'], panel=5, color='brown', 
                                              width=1.5, alpha=0.8, secondary_y=False, label='Dynamic Period')
                dynamic_plots = [dynamic_plot]
            else:
                kwargs['volume'] = True
                kwargs['panel_ratios'] = (4, 1, 2, 1, 1)  # メイン:出来高:X_ATR:TR:ボラ信号
                # 出来高を表示する場合は、オシレーターのパネル番号を+1する
                atr_low_vol_plot = mpf.make_addplot(df['atr_low_vol'], panel=2, color='blue', width=2, 
                                                  ylabel=f'X-ATR{value_suffix}', secondary_y=False, label='Low Volatility')
                atr_high_vol_plot = mpf.make_addplot(df['atr_high_vol'], panel=2, color='red', width=2, 
                                                  secondary_y=False, label='High Volatility')
                midline_low_vol_plot = mpf.make_addplot(df['midline_low_vol'], panel=2, color='darkblue', width=1, 
                                                     alpha=0.7, secondary_y=False, label='Midline (Low Vol)')
                midline_high_vol_plot = mpf.make_addplot(df['midline_high_vol'], panel=2, color='darkred', width=1, 
                                                     alpha=0.7, secondary_y=False, label='Midline (High Vol)')
                tr_panel = mpf.make_addplot(df['true_range'], panel=3, color='purple', width=1.2, 
                                            ylabel=f'True Range{value_suffix}', secondary_y=False, label='True Range')
                volatility_panel = mpf.make_addplot(df['volatility_signal'], panel=4, color='orange', width=1.5, 
                                              ylabel='Volatility Signal', secondary_y=False, label='Signal', type='line')
        else:
            if dynamic_plots:
                kwargs['volume'] = False
                kwargs['panel_ratios'] = (4, 2, 1, 1, 1)  # メイン:X_ATR:TR:ボラ信号:動的期間
                # 動的期間も更新
                dynamic_plot = mpf.make_addplot(df['dynamic_periods'], panel=4, color='brown', 
                                              width=1.5, alpha=0.8, secondary_y=False, label='Dynamic Period')
                dynamic_plots = [dynamic_plot]
            else:
                kwargs['volume'] = False
                kwargs['panel_ratios'] = (4, 2, 1, 1)  # メイン:X_ATR:TR:ボラ信号
        
        # すべてのプロットを結合
        all_plots = main_plots + [atr_low_vol_plot, atr_high_vol_plot, midline_low_vol_plot, midline_high_vol_plot] + [tr_panel, volatility_panel] + dynamic_plots
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加（X_ATRパネル）
        atr_panel_idx = 2 if show_volume else 1
        legend_labels = ['X-ATR (Low Vol)', 'X-ATR (High Vol)', 'Midline (Low Vol)', 'Midline (High Vol)']
        axes[atr_panel_idx].legend(legend_labels, loc='upper left', fontsize=8)
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        if show_volume:
            # X_ATRパネル - 平均線
            atr_mean = df['x_atr'].mean()
            axes[2].axhline(y=atr_mean, color='black', linestyle='--', alpha=0.5)
            axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # True Rangeパネル
            tr_mean = df['true_range'].mean()
            axes[3].axhline(y=tr_mean, color='black', linestyle='-', alpha=0.3)
            
            # ボラティリティ信号パネル
            axes[4].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[4].axhline(y=1, color='blue', linestyle='--', alpha=0.5)
            axes[4].axhline(y=-1, color='red', linestyle='--', alpha=0.5)
            axes[4].set_ylim(-1.5, 1.5)
            
            # 動的期間パネル（存在する場合）
            if dynamic_plots:
                period_mean = df['dynamic_periods'].mean()
                axes[5].axhline(y=period_mean, color='black', linestyle='-', alpha=0.3)
        else:
            # X_ATRパネル - 平均線
            atr_mean = df['x_atr'].mean()
            axes[1].axhline(y=atr_mean, color='black', linestyle='--', alpha=0.5)
            axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # True Rangeパネル
            tr_mean = df['true_range'].mean()
            axes[2].axhline(y=tr_mean, color='black', linestyle='-', alpha=0.3)
            
            # ボラティリティ信号パネル
            axes[3].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[3].axhline(y=1, color='blue', linestyle='--', alpha=0.5)
            axes[3].axhline(y=-1, color='red', linestyle='--', alpha=0.5)
            axes[3].set_ylim(-1.5, 1.5)
            
            # 動的期間パネル（存在する場合）
            if dynamic_plots:
                period_mean = df['dynamic_periods'].mean()
                axes[4].axhline(y=period_mean, color='black', linestyle='-', alpha=0.3)
        
        # 統計情報の表示
        print(f"\n=== X_ATR統計 ===")
        valid_mask = ~np.isnan(df['x_atr'])
        total_points = valid_mask.sum()
        low_vol_points = (df['volatility_signal'] == 1).sum()
        high_vol_points = (df['volatility_signal'] == -1).sum()
        
        print(f"総データ点数: {total_points}")
        print(f"低ボラティリティ: {low_vol_points} ({low_vol_points/total_points*100:.1f}%)")
        print(f"高ボラティリティ: {high_vol_points} ({high_vol_points/total_points*100:.1f}%)")
        
        if total_points > 0:
            valid_atr = df['x_atr'][valid_mask]
            print(f"X_ATR - 平均: {valid_atr.mean():.4f}, 範囲: {valid_atr.min():.4f} - {valid_atr.max():.4f}")
            
        if not df['true_range'].isna().all():
            valid_tr = df['true_range'][~np.isnan(df['true_range'])]
            print(f"True Range - 平均: {valid_tr.mean():.4f}, 範囲: {valid_tr.min():.4f} - {valid_tr.max():.4f}")
        
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
    parser = argparse.ArgumentParser(description='X_ATRの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--period', type=float, default=25.0, help='X_ATRスムージング期間')
    parser.add_argument('--tr-method', type=str, default='str', choices=['atr', 'str'], help='True Range計算方法')
    parser.add_argument('--smoother', type=str, default='super_smoother', help='スムーサータイプ')
    parser.add_argument('--kalman', action='store_true', help='カルマンフィルターを有効にする')
    parser.add_argument('--kalman-type', type=str, default='unscented', help='カルマンフィルタータイプ')
    parser.add_argument('--period-mode', type=str, default='fixed', choices=['fixed', 'dynamic'], help='期間モード')
    parser.add_argument('--detector-type', type=str, default='absolute_ultimate', help='サイクル検出器タイプ')
    parser.add_argument('--midline-period', type=int, default=100, help='ミッドライン期間')
    parser.add_argument('--percentage', action='store_true', help='%ベース値を表示する（デフォルト: 金額ベース）')
    args = parser.parse_args()
    
    # チャートを作成
    chart = XATRChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        period=args.period,
        tr_method=args.tr_method,
        smoother_type=args.smoother,
        enable_kalman=args.kalman,
        kalman_type=args.kalman_type,
        period_mode=args.period_mode,
        cycle_detector_type=args.detector_type,
        midline_period=args.midline_period
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        show_percentage=args.percentage,
        savefig=args.output
    )


if __name__ == "__main__":
    main()