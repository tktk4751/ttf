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
from indicators.z_adaptive_trend import ZAdaptiveTrend


class ZAdaptiveTrendChart:
    """
    Zアダプティブトレンドを表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - Zアダプティブトレンドの中心線・上限バンド・下限バンド（SuperTrendライク表示）
    - トレンド方向のカラー表示
    - サイクル効率比（CER）またはトリガー値
    - 動的乗数値
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.z_adaptive_trend = None
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
                            # シンプルアジャストメント乗数パラメータ
                            max_multiplier: float = 8.0,
                            min_multiplier: float = 3.0,
                            # ソースタイプ選択
                            trend_src_type: str = 'z_adaptive_ma',
                            trigger_source: str = 'x_trend',
                            catr_src_type: str = 'hlc3',
                            cer_src_type: str = 'hlc3',
                            # CERパラメータ
                            detector_type: str = 'dudi_e',
                            cycle_part: float = 0.4,
                            lp_period: int = 5,
                            hp_period: int = 100,
                            max_cycle: int = 120,
                            min_cycle: int = 10,
                            max_output: int = 89,
                            min_output: int = 21,
                            use_kalman_filter: bool = False,
                            # Xトレンドインデックスパラメータ
                            x_detector_type: str = 'dudi_e',
                            x_cycle_part: float = 1.0,
                            x_max_cycle: int = 120,
                            x_min_cycle: int = 5,
                            x_max_output: int = 89,
                            x_min_output: int = 21,
                            x_smoother_type: str = 'alma',
                            fixed_threshold: float = 0.65,
                            # ZAdaptiveMA用パラメータ
                            fast_period: int = 2,
                            slow_period: int = 30
                           ) -> None:
        """
        Zアダプティブトレンドを計算する
        
        Args:
            max_multiplier: 最大乗数（トリガー値0の時に使用）
            min_multiplier: 最小乗数（トリガー値1の時に使用）
            trend_src_type: トレンド用ソースタイプ（中心線とトレンド判定に使用）
            trigger_source: トリガー値のソース
            catr_src_type: CATR計算用のソースタイプ
            cer_src_type: CER計算用のソースタイプ
            detector_type: CER用ドミナントサイクル検出器タイプ
            cycle_part: CER用サイクル部分
            lp_period: CER用ローパスフィルター期間
            hp_period: CER用ハイパスフィルター期間
            max_cycle: CER用最大サイクル期間
            min_cycle: CER用最小サイクル期間
            max_output: CER用最大出力値
            min_output: CER用最小出力値
            use_kalman_filter: CER用カルマンフィルター使用有無
            x_detector_type: Xトレンド用検出器タイプ
            x_cycle_part: Xトレンド用サイクル部分
            x_max_cycle: Xトレンド用最大サイクル期間
            x_min_cycle: Xトレンド用最小サイクル期間
            x_max_output: Xトレンド用最大出力値
            x_min_output: Xトレンド用最小出力値
            x_smoother_type: Xトレンド用平滑化タイプ
            fixed_threshold: 固定しきい値（XTrendIndex用）
            fast_period: 速い移動平均の期間（固定値）
            slow_period: 遅い移動平均の期間（固定値）
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\nZアダプティブトレンドを計算中...")
        
        # Zアダプティブトレンドを計算
        self.z_adaptive_trend = ZAdaptiveTrend(
            max_multiplier=max_multiplier,
            min_multiplier=min_multiplier,
            trend_src_type=trend_src_type,
            trigger_source=trigger_source,
            catr_src_type=catr_src_type,
            cer_src_type=cer_src_type,
            detector_type=detector_type,
            cycle_part=cycle_part,
            lp_period=lp_period,
            hp_period=hp_period,
            max_cycle=max_cycle,
            min_cycle=min_cycle,
            max_output=max_output,
            min_output=min_output,
            use_kalman_filter=use_kalman_filter,
            x_detector_type=x_detector_type,
            x_cycle_part=x_cycle_part,
            x_max_cycle=x_max_cycle,
            x_min_cycle=x_min_cycle,
            x_max_output=x_max_output,
            x_min_output=x_min_output,
            x_smoother_type=x_smoother_type,
            fixed_threshold=fixed_threshold,
            fast_period=fast_period,
            slow_period=slow_period
        )
        
        # ZAdaptiveTrendの計算
        print("計算を実行します...")
        self.z_adaptive_trend.calculate(self.data)
        
        # バンドの取得テスト
        middle, upper, lower = self.z_adaptive_trend.get_bands()
        trend = self.z_adaptive_trend.get_trend()
        print(f"バンド計算完了 - 中心線: {len(middle)}, 上限: {len(upper)}, 下限: {len(lower)}, トレンド: {len(trend)}")
        
        # NaN値のチェック
        nan_count_middle = np.isnan(middle).sum()
        nan_count_upper = np.isnan(upper).sum()
        nan_count_lower = np.isnan(lower).sum()
        trend_count = (trend != 0).sum()
        print(f"NaN値 - 中心線: {nan_count_middle}, 上限: {nan_count_upper}, 下限: {nan_count_lower}")
        print(f"トレンド値 - 有効: {trend_count}, 上昇: {(trend == 1).sum()}, 下降: {(trend == -1).sum()}")
        
        print("Zアダプティブトレンド計算完了")
            
    def plot(self, 
            title: str = "Zアダプティブトレンド", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (14, 12),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとZアダプティブトレンドを描画する
        
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
            
        if self.z_adaptive_trend is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # Zアダプティブトレンドの値を取得
        print("バンドデータを取得中...")
        middle, upper, lower = self.z_adaptive_trend.get_bands()
        trend = self.z_adaptive_trend.get_trend()
        trigger_values = self.z_adaptive_trend.get_trigger_values()
        dynamic_mult = self.z_adaptive_trend.get_dynamic_multiplier()
        cer = self.z_adaptive_trend.get_efficiency_ratio()
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'trend_middle': middle,
                'trend_upper': upper,
                'trend_lower': lower,
                'trend_direction': trend,
                'trigger_values': trigger_values,
                'dynamic_mult': dynamic_mult,
                'cer': cer
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        print(f"バンドデータ確認 - 中心線NaN: {df['trend_middle'].isna().sum()}, 上限NaN: {df['trend_upper'].isna().sum()}")
        
        # SuperTrendライクな表示用データの準備
        # 上昇トレンドの時は下限バンドを緑色、下降トレンドの時は上限バンドを赤色で表示
        df['support_line'] = np.where(df['trend_direction'] == 1, df['trend_lower'], np.nan)
        df['resistance_line'] = np.where(df['trend_direction'] == -1, df['trend_upper'], np.nan)
        
        # トレンド方向に基づく中心線の色分け
        df['middle_uptrend'] = np.where(df['trend_direction'] == 1, df['trend_middle'], np.nan)
        df['middle_downtrend'] = np.where(df['trend_direction'] == -1, df['trend_middle'], np.nan)
        
        # NaN値を含む行を出力（最初の5行のみ）
        nan_rows = df[df['trend_upper'].isna() | df['trend_middle'].isna() | df['trend_lower'].isna()]
        if not nan_rows.empty:
            print(f"NaN値を含む行: {len(nan_rows)}行")
            if len(nan_rows) > 0:
                print(nan_rows.head())
        
        # mplfinanceでプロット用の設定
        # 1. メインチャート上のプロット
        main_plots = []
        
        # Zアダプティブトレンドのプロット設定（SuperTrendライク）
        main_plots.append(mpf.make_addplot(df['middle_uptrend'], color='green', width=2, label='ZAT Middle (Up)'))
        main_plots.append(mpf.make_addplot(df['middle_downtrend'], color='red', width=2, label='ZAT Middle (Down)'))
        main_plots.append(mpf.make_addplot(df['support_line'], color='green', width=1.5, alpha=0.7, label='Support Line'))
        main_plots.append(mpf.make_addplot(df['resistance_line'], color='red', width=1.5, alpha=0.7, label='Resistance Line'))
        
        # 2. オシレータープロット
        # トリガー値とパネル配置を設定
        trigger_panel = mpf.make_addplot(df['trigger_values'], panel=1, color='purple', width=1.2, 
                                        ylabel='Trigger Values', secondary_y=False, label='Trigger')
        
        # 動的乗数パネル
        mult_panel = mpf.make_addplot(df['dynamic_mult'], panel=2, color='blue', width=1.2, 
                                     ylabel='Dynamic Mult', secondary_y=False, label='Multiplier')
        
        # トレンド方向パネル
        trend_panel = mpf.make_addplot(df['trend_direction'], panel=3, color='orange', width=1.5, 
                                      ylabel='Trend Direction', secondary_y=False, label='Trend', type='line')
        
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
            kwargs['panel_ratios'] = (4, 1, 1, 1, 1)  # メイン:出来高:トリガー:乗数:トレンド
            # 出来高を表示する場合は、オシレーターのパネル番号を+1する
            trigger_panel = mpf.make_addplot(df['trigger_values'], panel=2, color='purple', width=1.2, 
                                            ylabel='Trigger Values', secondary_y=False, label='Trigger')
            mult_panel = mpf.make_addplot(df['dynamic_mult'], panel=3, color='blue', width=1.2, 
                                         ylabel='Dynamic Mult', secondary_y=False, label='Multiplier')
            trend_panel = mpf.make_addplot(df['trend_direction'], panel=4, color='orange', width=1.5, 
                                          ylabel='Trend Direction', secondary_y=False, label='Trend', type='line')
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (4, 1, 1, 1)  # メイン:トリガー:乗数:トレンド
        
        # すべてのプロットを結合
        all_plots = main_plots + [trigger_panel, mult_panel, trend_panel]
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加
        axes[0].legend(['ZAT Middle (Up)', 'ZAT Middle (Down)', 'Support Line', 'Resistance Line'], 
                      loc='upper left')
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        if show_volume:
            # トリガー値パネル
            axes[2].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
            axes[2].axhline(y=0.0, color='black', linestyle='-', alpha=0.3)
            axes[2].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
            
            # 動的乗数パネル
            mult_mean = df['dynamic_mult'].mean()
            axes[3].axhline(y=mult_mean, color='black', linestyle='-', alpha=0.3)
            
            # トレンド方向パネル
            axes[4].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[4].axhline(y=1, color='green', linestyle='--', alpha=0.5)
            axes[4].axhline(y=-1, color='red', linestyle='--', alpha=0.5)
        else:
            # トリガー値パネル
            axes[1].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
            axes[1].axhline(y=0.0, color='black', linestyle='-', alpha=0.3)
            axes[1].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
            
            # 動的乗数パネル
            mult_mean = df['dynamic_mult'].mean()
            axes[2].axhline(y=mult_mean, color='black', linestyle='-', alpha=0.3)
            
            # トレンド方向パネル
            axes[3].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[3].axhline(y=1, color='green', linestyle='--', alpha=0.5)
            axes[3].axhline(y=-1, color='red', linestyle='--', alpha=0.5)
        
        # 統計情報の表示
        print(f"\n=== トレンド統計 ===")
        total_points = len(df[df['trend_direction'] != 0])
        uptrend_points = len(df[df['trend_direction'] == 1])
        downtrend_points = len(df[df['trend_direction'] == -1])
        
        print(f"総データ点数: {total_points}")
        print(f"上昇トレンド: {uptrend_points} ({uptrend_points/total_points*100:.1f}%)")
        print(f"下降トレンド: {downtrend_points} ({downtrend_points/total_points*100:.1f}%)")
        print(f"動的乗数 - 平均: {df['dynamic_mult'].mean():.2f}, 範囲: {df['dynamic_mult'].min():.2f} - {df['dynamic_mult'].max():.2f}")
        
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
    parser = argparse.ArgumentParser(description='Zアダプティブトレンドの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--trend-src', type=str, default='hlc3', help='トレンド用ソースタイプ')
    parser.add_argument('--trigger-src', type=str, default='x_trend', help='トリガーソース')
    parser.add_argument('--max-mult', type=float, default=6.0, help='最大乗数')
    parser.add_argument('--min-mult', type=float, default=1.0, help='最小乗数')
    args = parser.parse_args()
    
    # チャートを作成
    chart = ZAdaptiveTrendChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        trend_src_type=args.trend_src,
        trigger_source=args.trigger_src,
        max_multiplier=args.max_mult,
        min_multiplier=args.min_mult
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output
    )


if __name__ == "__main__":
    main() 