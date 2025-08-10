#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 **Hyper Adaptive Supertrend Chart - Hyperアダプティブスーパートレンドチャート** 🎯

Hyperアダプティブスーパートレンドを実際の相場データでチャートに描画：
- 設定ファイルからの相場データ取得
- ローソク足チャートとスーパートレンドライン
- ミッドライン・上限バンド・下限バンドの表示
- トレンド方向の色分け表示
- ATR値とカルマンフィルター効果の表示
- SuperTrendライクな視覚的表現

🌟 **表示要素:**
1. **メインチャート**: ローソク足 + スーパートレンドライン
2. **サブパネル1**: ミッドライン値
3. **サブパネル2**: ATR値とボラティリティ
4. **サブパネル3**: トレンド方向とフィルター効果
"""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta

# データ取得のための依存関係
try:
    from data.data_loader import DataLoader, CSVDataSource
    from data.data_processor import DataProcessor
    from data.binance_data_source import BinanceDataSource
except ImportError:
    # フォールバック
    import sys
    sys.path.append('.')
    from data.data_loader import DataLoader, CSVDataSource
    from data.data_processor import DataProcessor
    from data.binance_data_source import BinanceDataSource

# インジケーター
try:
    from indicators.hyper_adaptive_supertrend import HyperAdaptiveSupertrend
except ImportError:
    import sys
    sys.path.append('.')
    from indicators.hyper_adaptive_supertrend import HyperAdaptiveSupertrend


class HyperAdaptiveSupertrendChart:
    """
    Hyperアダプティブスーパートレンドを表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - Hyperアダプティブスーパートレンドのライン・バンド（SuperTrendライク表示）
    - トレンド方向のカラー表示
    - ミッドライン（統合スムーサー結果）
    - ATR値（X_ATR結果）
    - カルマンフィルター効果（使用時のみ）
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.hyper_supertrend = None
        self.result = None
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
                            # ATR/ボラティリティパラメータ
                            atr_period: float = 14.0,
                            multiplier: float = 2.0,
                            atr_method: str = 'atr',
                            atr_smoother_type: str = 'ultimate_smoother',
                            
                            # ミッドライン（統合スムーサー）パラメータ
                            midline_smoother_type: str = 'frama',
                            midline_period: float = 21.0,
                            
                            # ソース価格関連パラメータ
                            src_type: str = 'hlc3',
                            enable_kalman: bool = False,
                            kalman_alpha: float = 0.1,
                            kalman_beta: float = 2.0,
                            kalman_kappa: float = 0.0,
                            kalman_process_noise: float = 0.01,
                            
                            # 動的期間調整パラメータ
                            use_dynamic_period: bool = False,
                            cycle_part: float = 1.0,
                            detector_type: str = 'absolute_ultimate',
                            max_cycle: int = 233,
                            min_cycle: int = 13,
                            max_output: int = 144,
                            min_output: int = 13,
                            lp_period: int = 10,
                            hp_period: int = 48,
                            
                            # 追加パラメータ
                            midline_smoother_params: Optional[Dict] = None,
                            atr_smoother_params: Optional[Dict] = None,
                            atr_kalman_params: Optional[Dict] = None
                           ) -> None:
        """
        Hyperアダプティブスーパートレンドを計算する
        
        Args:
            atr_period: X_ATR期間
            multiplier: ATR乗数
            atr_method: X_ATRの計算方法（'atr' または 'str'）
            atr_smoother_type: X_ATRのスムーサータイプ
            midline_smoother_type: ミッドラインスムーサータイプ
            midline_period: ミッドライン期間
            src_type: ソースタイプ
            enable_kalman: カルマンフィルター使用フラグ
            kalman_alpha: UKFアルファパラメータ
            kalman_beta: UKFベータパラメータ
            kalman_kappa: UKFカッパパラメータ
            kalman_process_noise: UKFプロセスノイズ
            use_dynamic_period: 動的期間を使用するか
            cycle_part: サイクル部分の倍率
            detector_type: 検出器タイプ
            max_cycle: 最大サイクル期間
            min_cycle: 最小サイクル期間
            max_output: 最大出力値
            min_output: 最小出力値
            lp_period: ローパスフィルター期間
            hp_period: ハイパスフィルター期間
            midline_smoother_params: ミッドラインスムーサー固有パラメータ
            atr_smoother_params: ATRスムーサー固有パラメータ
            atr_kalman_params: ATR用カルマンパラメータ
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\nHyperアダプティブスーパートレンドを計算中...")
        
        # Hyperアダプティブスーパートレンドを初期化
        self.hyper_supertrend = HyperAdaptiveSupertrend(
            atr_period=atr_period,
            multiplier=multiplier,
            atr_method=atr_method,
            atr_smoother_type=atr_smoother_type,
            midline_smoother_type=midline_smoother_type,
            midline_period=midline_period,
            src_type=src_type,
            enable_kalman=enable_kalman,
            kalman_alpha=kalman_alpha,
            kalman_beta=kalman_beta,
            kalman_kappa=kalman_kappa,
            kalman_process_noise=kalman_process_noise,
            use_dynamic_period=use_dynamic_period,
            cycle_part=cycle_part,
            detector_type=detector_type,
            max_cycle=max_cycle,
            min_cycle=min_cycle,
            max_output=max_output,
            min_output=min_output,
            lp_period=lp_period,
            hp_period=hp_period,
            midline_smoother_params=midline_smoother_params or {},
            atr_smoother_params=atr_smoother_params or {},
            atr_kalman_params=atr_kalman_params or {}
        )
        
        # 計算実行
        print("計算を実行します...")
        self.result = self.hyper_supertrend.calculate(self.data)
        
        # 結果の確認
        print(f"計算完了 - スーパートレンド: {len(self.result.values)}, "
              f"バンド: {len(self.result.upper_band)}, "
              f"トレンド: {len(self.result.trend)}")
        
        # NaN値のチェック
        nan_count_values = np.isnan(self.result.values).sum()
        nan_count_upper = np.isnan(self.result.upper_band).sum()
        nan_count_lower = np.isnan(self.result.lower_band).sum()
        trend_count = (self.result.trend != 0).sum()
        
        print(f"NaN値 - スーパートレンド: {nan_count_values}, "
              f"上限: {nan_count_upper}, 下限: {nan_count_lower}")
        print(f"トレンド値 - 有効: {trend_count}, "
              f"上昇: {(self.result.trend == 1).sum()}, "
              f"下降: {(self.result.trend == -1).sum()}")
        
        # カルマンフィルター効果の確認
        if self.result.filtered_source is not None:
            print(f"カルマンフィルター効果確認:")
            print(f"  元の価格平均: {np.nanmean(self.result.raw_source):.4f}")
            print(f"  フィルター後平均: {np.nanmean(self.result.filtered_source):.4f}")
            print(f"  ミッドライン平均: {np.nanmean(self.result.midline):.4f}")
        
        print("Hyperアダプティブスーパートレンド計算完了")
            
    def plot(self, 
            title: str = "Hyper Adaptive Supertrend", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (16, 14),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとHyperアダプティブスーパートレンドを描画する
        
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
            
        if self.result is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # Hyperアダプティブスーパートレンドの値を取得
        print("インジケーターデータを取得中...")
        values = self.result.values
        upper_band = self.result.upper_band
        lower_band = self.result.lower_band
        trend = self.result.trend
        midline = self.result.midline
        atr_values = self.result.atr_values
        raw_source = self.result.raw_source
        filtered_source = self.result.filtered_source
        
        # 期間に対応するようにインデックスを調整
        start_idx = 0
        end_idx = len(self.data)
        
        if start_date:
            start_idx = max(0, self.data.index.searchsorted(pd.to_datetime(start_date)))
        if end_date:
            end_idx = min(len(self.data), self.data.index.searchsorted(pd.to_datetime(end_date)) + 1)
        
        # インジケーターデータを期間で切り取り
        values_slice = values[start_idx:end_idx]
        upper_band_slice = upper_band[start_idx:end_idx]
        lower_band_slice = lower_band[start_idx:end_idx]
        trend_slice = trend[start_idx:end_idx]
        midline_slice = midline[start_idx:end_idx]
        atr_slice = atr_values[start_idx:end_idx]
        raw_source_slice = raw_source[start_idx:end_idx]
        
        if filtered_source is not None:
            filtered_source_slice = filtered_source[start_idx:end_idx]
        else:
            filtered_source_slice = None
        
        # 全データの時系列データフレームを作成
        indicator_df = pd.DataFrame(
            index=df.index,
            data={
                'supertrend': values_slice,
                'upper_band': upper_band_slice,
                'lower_band': lower_band_slice,
                'trend_direction': trend_slice,
                'midline': midline_slice,
                'atr_values': atr_slice,
                'raw_source': raw_source_slice,
                'filtered_source': filtered_source_slice if filtered_source_slice is not None else np.nan
            }
        )
        
        # データフレームに結合
        df = df.join(indicator_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        print(f"スーパートレンドデータ確認 - NaN: {df['supertrend'].isna().sum()}")
        
        # SuperTrendライクな表示用データの準備
        # 上昇トレンドの時は下限バンドをサポートライン、下降トレンドの時は上限バンドをレジスタンスラインとして表示
        df['support_line'] = np.where(df['trend_direction'] == 1, df['lower_band'], np.nan)
        df['resistance_line'] = np.where(df['trend_direction'] == -1, df['upper_band'], np.nan)
        
        # トレンド方向に基づくスーパートレンドラインの色分け
        df['supertrend_uptrend'] = np.where(df['trend_direction'] == 1, df['supertrend'], np.nan)
        df['supertrend_downtrend'] = np.where(df['trend_direction'] == -1, df['supertrend'], np.nan)
        
        # ミッドラインの色分け
        df['midline_uptrend'] = np.where(df['trend_direction'] == 1, df['midline'], np.nan)
        df['midline_downtrend'] = np.where(df['trend_direction'] == -1, df['midline'], np.nan)
        
        # NaN値を含む行の確認
        nan_rows = df[df['supertrend'].isna() | df['midline'].isna()]
        if not nan_rows.empty:
            print(f"NaN値を含む行: {len(nan_rows)}行")
        
        # mplfinanceでプロット用の設定
        # 1. メインチャート上のプロット
        main_plots = []
        
        # Hyperアダプティブスーパートレンドのプロット設定
        main_plots.append(mpf.make_addplot(df['supertrend_uptrend'], color='lime', width=3, label='Supertrend (Up)'))
        main_plots.append(mpf.make_addplot(df['supertrend_downtrend'], color='red', width=3, label='Supertrend (Down)'))
        main_plots.append(mpf.make_addplot(df['support_line'], color='green', width=1.5, alpha=0.6, linestyle='--', label='Support'))
        main_plots.append(mpf.make_addplot(df['resistance_line'], color='red', width=1.5, alpha=0.6, linestyle='--', label='Resistance'))
        
        # ミッドライン（統合スムーサー）のプロット
        main_plots.append(mpf.make_addplot(df['midline_uptrend'], color='darkgreen', width=1.5, alpha=0.8, label='Midline (Up)'))
        main_plots.append(mpf.make_addplot(df['midline_downtrend'], color='darkred', width=1.5, alpha=0.8, label='Midline (Down)'))
        
        # 2. サブパネルのプロット
        # ミッドライン値パネル
        midline_panel = mpf.make_addplot(df['midline'], panel=1, color='blue', width=1.2, 
                                        ylabel='Midline Value', secondary_y=False, label='Midline')
        
        # ATR値とボラティリティパネル
        atr_panel = mpf.make_addplot(df['atr_values'], panel=2, color='orange', width=1.2, 
                                   ylabel='ATR Value', secondary_y=False, label='ATR')
        
        # トレンド方向とフィルター効果パネル
        trend_panel = mpf.make_addplot(df['trend_direction'], panel=3, color='purple', width=1.5, 
                                      ylabel='Trend / Filter', secondary_y=False, label='Trend', type='line')
        
        # カルマンフィルター効果があれば追加
        filter_plots = []
        if filtered_source_slice is not None and not np.all(np.isnan(df['filtered_source'])):
            # 元の価格とフィルター後の価格の差
            df['filter_effect'] = df['filtered_source'] - df['raw_source']
            filter_plots.append(mpf.make_addplot(df['filter_effect'], panel=3, color='cyan', width=1.0, 
                                               secondary_y=True, label='Filter Effect'))
        
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
            kwargs['panel_ratios'] = (5, 1, 1.5, 1.5, 1.5)  # メイン:出来高:ミッドライン:ATR:トレンド
            # 出来高を表示する場合は、パネル番号を+1する
            midline_panel = mpf.make_addplot(df['midline'], panel=2, color='blue', width=1.2, 
                                            ylabel='Midline Value', secondary_y=False, label='Midline')
            atr_panel = mpf.make_addplot(df['atr_values'], panel=3, color='orange', width=1.2, 
                                       ylabel='ATR Value', secondary_y=False, label='ATR')
            trend_panel = mpf.make_addplot(df['trend_direction'], panel=4, color='purple', width=1.5, 
                                          ylabel='Trend / Filter', secondary_y=False, label='Trend', type='line')
            
            if filter_plots:
                filter_plots = [mpf.make_addplot(df['filter_effect'], panel=4, color='cyan', width=1.0, 
                                               secondary_y=True, label='Filter Effect')]
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (5, 1.5, 1.5, 1.5)  # メイン:ミッドライン:ATR:トレンド
        
        # すべてのプロットを結合
        all_plots = main_plots + [midline_panel, atr_panel, trend_panel] + filter_plots
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加
        legend_labels = ['Supertrend (Up)', 'Supertrend (Down)', 'Support', 'Resistance', 'Midline (Up)', 'Midline (Down)']
        axes[0].legend(legend_labels, loc='upper left', fontsize=8)
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        panel_offset = 1 if show_volume else 0
        
        # ミッドラインパネル（価格レベル参照線）
        midline_mean = df['midline'].mean()
        axes[1 + panel_offset].axhline(y=midline_mean, color='black', linestyle='-', alpha=0.3, label='Mean')
        
        # ATRパネル（ボラティリティ参照線）
        atr_mean = df['atr_values'].mean()
        atr_std = df['atr_values'].std()
        axes[2 + panel_offset].axhline(y=atr_mean, color='black', linestyle='-', alpha=0.3)
        axes[2 + panel_offset].axhline(y=atr_mean + atr_std, color='gray', linestyle='--', alpha=0.3)
        axes[2 + panel_offset].axhline(y=atr_mean - atr_std, color='gray', linestyle='--', alpha=0.3)
        
        # トレンド方向パネル（トレンド参照線）
        axes[3 + panel_offset].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[3 + panel_offset].axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Uptrend')
        axes[3 + panel_offset].axhline(y=-1, color='red', linestyle='--', alpha=0.5, label='Downtrend')
        
        # 統計情報の表示
        print(f"\n=== Hyperアダプティブスーパートレンド統計 ===")
        total_points = len(df[df['trend_direction'] != 0])
        uptrend_points = len(df[df['trend_direction'] == 1])
        downtrend_points = len(df[df['trend_direction'] == -1])
        
        print(f"総データ点数: {total_points}")
        print(f"上昇トレンド: {uptrend_points} ({uptrend_points/total_points*100:.1f}%)")
        print(f"下降トレンド: {downtrend_points} ({downtrend_points/total_points*100:.1f}%)")
        print(f"ミッドライン - 平均: {midline_mean:.4f}, 範囲: {df['midline'].min():.4f} - {df['midline'].max():.4f}")
        print(f"ATR値 - 平均: {atr_mean:.4f}, 標準偏差: {atr_std:.4f}")
        
        # カルマンフィルター効果の統計
        if filtered_source_slice is not None and not np.all(np.isnan(df['filtered_source'])):
            filter_effect_mean = df['filter_effect'].mean()
            filter_effect_std = df['filter_effect'].std()
            print(f"カルマンフィルター効果 - 平均: {filter_effect_mean:.6f}, 標準偏差: {filter_effect_std:.6f}")
        
        # インジケーター設定情報の表示
        metadata = self.hyper_supertrend.get_metadata()
        print(f"\n=== インジケーター設定 ===")
        print(f"ミッドラインスムーサー: {metadata['components']['midline_smoother']}")
        print(f"ATR計算器: {metadata['components']['atr_calculator']}")
        print(f"カルマンフィルター: {metadata['components']['kalman_filter']}")
        print(f"動的期間調整: {metadata['features']['dynamic_periods']}")
        
        # 保存または表示
        if savefig:
            plt.savefig(savefig, dpi=300, bbox_inches='tight')
            print(f"\nチャートを保存しました: {savefig}")
        else:
            plt.tight_layout()
            plt.show()


def main():
    """メイン関数"""
    # コマンドライン引数を処理
    import argparse
    parser = argparse.ArgumentParser(description='Hyperアダプティブスーパートレンドの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    
    # Hyperアダプティブスーパートレンドのパラメータ
    parser.add_argument('--atr-period', type=float, default=14.0, help='ATR期間')
    parser.add_argument('--multiplier', type=float, default=2.0, help='ATR乗数')
    parser.add_argument('--atr-method', type=str, default='atr', help='ATR計算方法 (atr/str)')
    parser.add_argument('--atr-smoother', type=str, default='ultimate_smoother', help='ATRスムーサー')
    parser.add_argument('--midline-smoother', type=str, default='frama', help='ミッドラインスムーサー')
    parser.add_argument('--midline-period', type=float, default=22.0, help='ミッドライン期間（偶数に調整される）')
    parser.add_argument('--src-type', type=str, default='hlc3', help='ソースタイプ')
    parser.add_argument('--enable-kalman', action='store_true', help='カルマンフィルターを有効にする')
    parser.add_argument('--use-dynamic', action='store_true', help='動的期間調整を有効にする')
    
    args = parser.parse_args()
    
    try:
        # チャートを作成
        chart = HyperAdaptiveSupertrendChart()
        chart.load_data_from_config(args.config)
        chart.calculate_indicators(
            atr_period=args.atr_period,
            multiplier=args.multiplier,
            atr_method=args.atr_method,
            atr_smoother_type=args.atr_smoother,
            midline_smoother_type=args.midline_smoother,
            midline_period=args.midline_period,
            src_type=args.src_type,
            enable_kalman=args.enable_kalman,
            use_dynamic_period=args.use_dynamic
        )
        chart.plot(
            start_date=args.start,
            end_date=args.end,
            savefig=args.output
        )
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()