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
from indicators.p_mama import P_MAMA


class PMAMAChart:
    """
    P_MAMA (Phasor-based MAMA) を表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - P_MAMAライン（緑色）
    - P_FAMAライン（オレンジ色）  
    - フェーザー角度（度単位）
    - トレンド状態（+1: 上昇, 0: サイクリング, -1: 下降）
    - 瞬間周期
    - アルファ値
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.p_mama = None
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
                            period: int = 55,
                            fast_limit: float = 0.5,
                            slow_limit: float = 0.05,
                            src_type: str = 'close',
                            use_kalman_filter: bool = False,
                            kalman_filter_type: str = 'unscented',
                            kalman_process_noise: float = 0.01,
                            kalman_observation_noise: float = 0.001,
                            use_zero_lag: bool = True
                           ) -> None:
        """
        P_MAMAを計算する
        
        Args:
            period: フェーザー分析の固定周期（デフォルト: 28）
            fast_limit: 高速制限値（デフォルト: 0.5）
            slow_limit: 低速制限値（デフォルト: 0.05）
            src_type: ソースタイプ（デフォルト: 'close'）
            use_kalman_filter: カルマンフィルターを使用するか（デフォルト: False）
            kalman_filter_type: カルマンフィルタータイプ（デフォルト: 'unscented'）
            kalman_process_noise: プロセスノイズ（デフォルト: 0.01）
            kalman_observation_noise: 観測ノイズ（デフォルト: 0.001）
            use_zero_lag: ゼロラグ処理を使用するか（デフォルト: True）
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\\nP_MAMAを計算中...")
        
        # P_MAMAを計算
        self.p_mama = P_MAMA(
            period=period,
            fast_limit=fast_limit,
            slow_limit=slow_limit,
            src_type=src_type,
            use_kalman_filter=use_kalman_filter,
            kalman_filter_type=kalman_filter_type,
            kalman_process_noise=kalman_process_noise,
            kalman_observation_noise=kalman_observation_noise,
            use_zero_lag=use_zero_lag
        )
        
        # P_MAMAの計算
        print("計算を実行します...")
        result = self.p_mama.calculate(self.data)
        
        # 結果の取得テスト
        mama_values = result.mama_values
        fama_values = result.fama_values
        phase_values = result.phase_values
        state_values = result.state_values
        alpha_values = result.alpha_values
        instantaneous_period = result.instantaneous_period
        
        print(f"P_MAMA計算完了 - MAMA: {len(mama_values)}, FAMA: {len(fama_values)}, Phase: {len(phase_values)}")
        
        # NaN値のチェック
        nan_count_mama = np.isnan(mama_values).sum()
        nan_count_fama = np.isnan(fama_values).sum()
        nan_count_phase = np.isnan(phase_values).sum()
        state_count = (state_values != 0).sum()
        
        print(f"NaN値 - MAMA: {nan_count_mama}, FAMA: {nan_count_fama}, Phase: {nan_count_phase}")
        print(f"トレンド状態 - 有効: {state_count}, 上昇: {(state_values == 1).sum()}, 下降: {(state_values == -1).sum()}")
        
        # 統計情報
        print(f"統計 - MAMA平均: {np.nanmean(mama_values):.4f}, FAMA平均: {np.nanmean(fama_values):.4f}")
        print(f"Phase範囲: {np.nanmin(phase_values):.1f}° - {np.nanmax(phase_values):.1f}°")
        print(f"Alpha範囲: {np.nanmin(alpha_values):.4f} - {np.nanmax(alpha_values):.4f}")
        print(f"瞬間周期範囲: {np.nanmin(instantaneous_period):.1f} - {np.nanmax(instantaneous_period):.1f}")
        
        print("P_MAMA計算完了")
            
    def plot(self, 
            title: str = "P_MAMA (Phasor-based MAMA)", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (16, 14),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとP_MAMAを描画する
        
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
            
        if self.p_mama is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # P_MAMAの値を取得
        print("P_MAMAデータを取得中...")
        result = self.p_mama.calculate(self.data)
        
        mama_values = result.mama_values
        fama_values = result.fama_values
        phase_values = result.phase_values
        state_values = result.state_values
        alpha_values = result.alpha_values
        instantaneous_period = result.instantaneous_period
        real_values = result.real_values
        imag_values = result.imag_values
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'p_mama': mama_values,
                'p_fama': fama_values,
                'phase': phase_values,
                'state': state_values,
                'alpha': alpha_values,
                'inst_period': instantaneous_period,
                'real': real_values,
                'imag': imag_values
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        print(f"P_MAMAデータ確認 - MAMA NaN: {df['p_mama'].isna().sum()}, FAMA NaN: {df['p_fama'].isna().sum()}")
        
        # トレンド状態に基づく色分け
        df['mama_uptrend'] = np.where(df['state'] == 1, df['p_mama'], np.nan)
        df['mama_downtrend'] = np.where(df['state'] == -1, df['p_mama'], np.nan)
        df['mama_cycling'] = np.where(df['state'] == 0, df['p_mama'], np.nan)
        
        df['fama_uptrend'] = np.where(df['state'] == 1, df['p_fama'], np.nan)
        df['fama_downtrend'] = np.where(df['state'] == -1, df['p_fama'], np.nan)
        df['fama_cycling'] = np.where(df['state'] == 0, df['p_fama'], np.nan)
        
        # mplfinanceでプロット用の設定
        # 1. メインチャート上のプロット
        main_plots = []
        
        # P_MAMAのプロット設定（トレンド状態別）
        main_plots.append(mpf.make_addplot(df['mama_uptrend'], color='green', width=2.5, label='P_MAMA (Uptrend)'))
        main_plots.append(mpf.make_addplot(df['mama_downtrend'], color='red', width=2.5, label='P_MAMA (Downtrend)'))
        main_plots.append(mpf.make_addplot(df['mama_cycling'], color='gray', width=2.0, alpha=0.7, label='P_MAMA (Cycling)'))
        
        # P_FAMAのプロット設定（トレンド状態別）
        main_plots.append(mpf.make_addplot(df['fama_uptrend'], color='lime', width=1.8, alpha=0.8, label='P_FAMA (Uptrend)'))
        main_plots.append(mpf.make_addplot(df['fama_downtrend'], color='orange', width=1.8, alpha=0.8, label='P_FAMA (Downtrend)'))
        main_plots.append(mpf.make_addplot(df['fama_cycling'], color='lightgray', width=1.5, alpha=0.6, label='P_FAMA (Cycling)'))
        
        # 2. オシレータープロット
        # フェーザー角度パネル
        phase_panel = mpf.make_addplot(df['phase'], panel=1, color='purple', width=1.5, 
                                      ylabel='Phase Angle (°)', secondary_y=False, label='Phasor Angle')
        
        # トレンド状態パネル
        state_panel = mpf.make_addplot(df['state'], panel=2, color='orange', width=2.0, 
                                      ylabel='Trend State', secondary_y=False, label='State', type='line')
        
        # アルファ値パネル
        alpha_panel = mpf.make_addplot(df['alpha'], panel=3, color='blue', width=1.5, 
                                      ylabel='Alpha Value', secondary_y=False, label='Alpha')
        
        # 瞬間周期パネル
        period_panel = mpf.make_addplot(df['inst_period'], panel=4, color='brown', width=1.5, 
                                       ylabel='Instantaneous Period', secondary_y=False, label='Period')
        
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
            kwargs['panel_ratios'] = (5, 1, 1, 1, 1, 1)  # メイン:出来高:Phase:State:Alpha:Period
            # 出来高を表示する場合は、オシレーターのパネル番号を+1する
            phase_panel = mpf.make_addplot(df['phase'], panel=2, color='purple', width=1.5, 
                                          ylabel='Phase Angle (°)', secondary_y=False, label='Phasor Angle')
            state_panel = mpf.make_addplot(df['state'], panel=3, color='orange', width=2.0, 
                                          ylabel='Trend State', secondary_y=False, label='State', type='line')
            alpha_panel = mpf.make_addplot(df['alpha'], panel=4, color='blue', width=1.5, 
                                          ylabel='Alpha Value', secondary_y=False, label='Alpha')
            period_panel = mpf.make_addplot(df['inst_period'], panel=5, color='brown', width=1.5, 
                                           ylabel='Instantaneous Period', secondary_y=False, label='Period')
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (5, 1, 1, 1, 1)  # メイン:Phase:State:Alpha:Period
        
        # すべてのプロットを結合
        all_plots = main_plots + [phase_panel, state_panel, alpha_panel, period_panel]
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加
        axes[0].legend(['P_MAMA (Up)', 'P_MAMA (Down)', 'P_MAMA (Cycle)', 
                       'P_FAMA (Up)', 'P_FAMA (Down)', 'P_FAMA (Cycle)'], 
                      loc='upper left')
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        if show_volume:
            # フェーザー角度パネル
            axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[2].axhline(y=90, color='green', linestyle='--', alpha=0.5)
            axes[2].axhline(y=-90, color='red', linestyle='--', alpha=0.5)
            axes[2].axhline(y=180, color='black', linestyle='-', alpha=0.3)
            axes[2].axhline(y=-180, color='black', linestyle='-', alpha=0.3)
            
            # トレンド状態パネル
            axes[3].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[3].axhline(y=1, color='green', linestyle='--', alpha=0.5)
            axes[3].axhline(y=-1, color='red', linestyle='--', alpha=0.5)
            
            # アルファ値パネル
            alpha_mean = df['alpha'].mean()
            axes[4].axhline(y=alpha_mean, color='black', linestyle='-', alpha=0.3)
            axes[4].axhline(y=df['alpha'].max(), color='blue', linestyle='--', alpha=0.3)
            axes[4].axhline(y=df['alpha'].min(), color='blue', linestyle='--', alpha=0.3)
            
            # 瞬間周期パネル
            period_mean = df['inst_period'].mean()
            axes[5].axhline(y=period_mean, color='black', linestyle='-', alpha=0.3)
            axes[5].axhline(y=28, color='brown', linestyle='--', alpha=0.5)  # 基準周期
        else:
            # フェーザー角度パネル
            axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[1].axhline(y=90, color='green', linestyle='--', alpha=0.5)
            axes[1].axhline(y=-90, color='red', linestyle='--', alpha=0.5)
            axes[1].axhline(y=180, color='black', linestyle='-', alpha=0.3)
            axes[1].axhline(y=-180, color='black', linestyle='-', alpha=0.3)
            
            # トレンド状態パネル
            axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[2].axhline(y=1, color='green', linestyle='--', alpha=0.5)
            axes[2].axhline(y=-1, color='red', linestyle='--', alpha=0.5)
            
            # アルファ値パネル
            alpha_mean = df['alpha'].mean()
            axes[3].axhline(y=alpha_mean, color='black', linestyle='-', alpha=0.3)
            axes[3].axhline(y=df['alpha'].max(), color='blue', linestyle='--', alpha=0.3)
            axes[3].axhline(y=df['alpha'].min(), color='blue', linestyle='--', alpha=0.3)
            
            # 瞬間周期パネル
            period_mean = df['inst_period'].mean()
            axes[4].axhline(y=period_mean, color='black', linestyle='-', alpha=0.3)
            axes[4].axhline(y=28, color='brown', linestyle='--', alpha=0.5)  # 基準周期
        
        # 統計情報の表示
        print(f"\\n=== P_MAMA統計 ===")
        total_points = len(df[df['state'] != 0])
        uptrend_points = len(df[df['state'] == 1])
        downtrend_points = len(df[df['state'] == -1])
        cycling_points = len(df[df['state'] == 0])
        
        print(f"総データ点数: {len(df)}")
        print(f"上昇トレンド: {uptrend_points} ({uptrend_points/len(df)*100:.1f}%)")
        print(f"下降トレンド: {downtrend_points} ({downtrend_points/len(df)*100:.1f}%)")
        print(f"サイクリング: {cycling_points} ({cycling_points/len(df)*100:.1f}%)")
        print(f"アルファ値 - 平均: {df['alpha'].mean():.4f}, 範囲: {df['alpha'].min():.4f} - {df['alpha'].max():.4f}")
        print(f"瞬間周期 - 平均: {df['inst_period'].mean():.1f}, 範囲: {df['inst_period'].min():.1f} - {df['inst_period'].max():.1f}")
        
        # クロスオーバーシグナルの分析
        mama_above_fama = df['p_mama'] > df['p_fama']
        crossovers = (mama_above_fama != mama_above_fama.shift(1)).sum()
        print(f"MAMA/FAMAクロスオーバー: {crossovers}回")
        
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
    parser = argparse.ArgumentParser(description='P_MAMA (Phasor-based MAMA)の描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--period', type=int, default=28, help='フェーザー分析の固定周期')
    parser.add_argument('--fast-limit', type=float, default=0.5, help='高速制限値')
    parser.add_argument('--slow-limit', type=float, default=0.05, help='低速制限値')
    parser.add_argument('--src-type', type=str, default='close', help='ソースタイプ')
    parser.add_argument('--zero-lag', action='store_true', help='ゼロラグ処理を有効にする')
    parser.add_argument('--kalman', action='store_true', help='カルマンフィルターを有効にする')
    args = parser.parse_args()
    
    # チャートを作成
    chart = PMAMAChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        period=args.period,
        fast_limit=args.fast_limit,
        slow_limit=args.slow_limit,
        src_type=args.src_type,
        use_zero_lag=args.zero_lag,
        use_kalman_filter=args.kalman
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output
    )


if __name__ == "__main__":
    main() 