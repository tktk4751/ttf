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
from indicators.trend_filter.phasor_trend_filter import PhasorTrendFilter


class PhasorTrendFilterChart:
    """
    Phasor Trend Filter を表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - トレンド状態シグナル（買い・売り・中立）
    - フェーザー角度（度単位）
    - トレンド強度（0-1）
    - サイクル信頼度（0-1）
    - 瞬間周期
    - 位相角度変化率
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.phasor_filter = None
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
                            period: int = 28,
                            trend_threshold: float = 6.0,
                            src_type: str = 'close',
                            use_kalman_filter: bool = False,
                            kalman_filter_type: str = 'unscented',
                            kalman_process_noise: float = 0.01,
                            kalman_observation_noise: float = 0.001
                           ) -> None:
        """
        Phasor Trend Filterを計算する
        
        Args:
            period: フェーザー分析の固定周期（デフォルト: 28）
            trend_threshold: トレンド判定しきい値（デフォルト: 0.5）
            src_type: ソースタイプ（デフォルト: 'close'）
            use_kalman_filter: カルマンフィルターを使用するか（デフォルト: False）
            kalman_filter_type: カルマンフィルタータイプ（デフォルト: 'unscented'）
            kalman_process_noise: プロセスノイズ（デフォルト: 0.01）
            kalman_observation_noise: 観測ノイズ（デフォルト: 0.001）
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\nPhasor Trend Filterを計算中...")
        
        # Phasor Trend Filterを計算
        self.phasor_filter = PhasorTrendFilter(
            period=period,
            trend_threshold=trend_threshold,
            src_type=src_type,
            use_kalman_filter=use_kalman_filter,
            kalman_filter_type=kalman_filter_type,
            kalman_process_noise=kalman_process_noise,
            kalman_observation_noise=kalman_observation_noise
        )
        
        # フィルターの計算
        print("計算を実行します...")
        result = self.phasor_filter.calculate(self.data)
        
        # 結果の取得
        trend_state = result.state
        signal = result.signal
        phase_angle = result.phase_angle
        trend_strength = result.trend_strength
        cycle_confidence = result.cycle_confidence
        instantaneous_period = result.instantaneous_period
        real_values = result.real_component
        imag_values = result.imag_component
        
        print(f"Phasor Trend Filter計算完了 - データ点数: {len(trend_state)}")
        
        # 統計情報
        trend_count = (trend_state == 1).sum()
        range_count = (trend_state == 0).sum()
        buy_signals = (signal == 1).sum()
        sell_signals = (signal == -1).sum()
        neutral_signals = (signal == 0).sum()
        
        print(f"状態統計:")
        print(f"  トレンド状態: {trend_count} ({trend_count/len(trend_state)*100:.1f}%)")
        print(f"  レンジ状態: {range_count} ({range_count/len(trend_state)*100:.1f}%)")
        print(f"シグナル統計:")
        print(f"  買いシグナル: {buy_signals} ({buy_signals/len(signal)*100:.1f}%)")
        print(f"  売りシグナル: {sell_signals} ({sell_signals/len(signal)*100:.1f}%)")
        print(f"  中立シグナル: {neutral_signals} ({neutral_signals/len(signal)*100:.1f}%)")
        
        print(f"統計 - トレンド強度平均: {np.nanmean(trend_strength):.3f}")
        print(f"サイクル信頼度平均: {np.nanmean(cycle_confidence):.3f}")
        print(f"Phase角度範囲: {np.nanmin(phase_angle):.1f}° - {np.nanmax(phase_angle):.1f}°")
        print(f"瞬間周期範囲: {np.nanmin(instantaneous_period):.1f} - {np.nanmax(instantaneous_period):.1f}")
        
        print("Phasor Trend Filter計算完了")
            
    def plot(self, 
            title: str = "Phasor Trend Filter", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (16, 16),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとPhasor Trend Filterを描画する
        
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
            
        if self.phasor_filter is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # Phasor Trend Filterの値を取得
        print("Phasor Trend Filterデータを取得中...")
        result = self.phasor_filter.calculate(self.data)
        
        trend_state = result.state
        signal = result.signal
        phase_angle = result.phase_angle
        trend_strength = result.trend_strength
        cycle_confidence = result.cycle_confidence
        instantaneous_period = result.instantaneous_period
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'trend_state': trend_state,
                'signal': signal,
                'phase_angle': phase_angle,
                'trend_strength': trend_strength,
                'cycle_confidence': cycle_confidence,
                'inst_period': instantaneous_period
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        
        # シグナルに基づく色分けとマーカー
        df['buy_signal'] = np.where(df['signal'] == 1, df['close'], np.nan)
        df['sell_signal'] = np.where(df['signal'] == -1, df['close'], np.nan)
        
        # トレンド状態に基づく色分け（連続的な値を使用）
        # 各状態を連続的な値として設定
        df['trend_up'] = np.where(df['trend_state'] == 1, df['trend_state'], np.nan)
        df['trend_down'] = np.where(df['trend_state'] == -1, df['trend_state'], np.nan)
        df['trend_range'] = np.where(df['trend_state'] == 0, df['trend_state'], np.nan)
        
        # 補間による連続性の確保
        # Up trend用：値1のポイントを線で結ぶ
        trend_up_interpolated = df['trend_state'].copy()
        trend_up_interpolated[df['trend_state'] != 1] = np.nan
        
        # Down trend用：値-1のポイントを線で結ぶ
        trend_down_interpolated = df['trend_state'].copy()
        trend_down_interpolated[df['trend_state'] != -1] = np.nan
        
        # Range用：値0のポイントを線で結ぶ
        trend_range_interpolated = df['trend_state'].copy()
        trend_range_interpolated[df['trend_state'] != 0] = np.nan
        
        # 補間済みデータに更新
        df['trend_up'] = trend_up_interpolated
        df['trend_down'] = trend_down_interpolated
        df['trend_range'] = trend_range_interpolated
        
        # デバッグ情報の出力
        up_count = (~np.isnan(df['trend_up'])).sum()
        down_count = (~np.isnan(df['trend_down'])).sum()
        range_count = (~np.isnan(df['trend_range'])).sum()
        print(f"Trend Up points: {up_count}, Down points: {down_count}, Range points: {range_count}")
        
        # 全てがNaNの場合は、適切なダミー値を追加
        if up_count == 0:
            df.loc[df.index[0], 'trend_up'] = 1
        if down_count == 0:
            df.loc[df.index[0], 'trend_down'] = -1
        if range_count == 0:
            df.loc[df.index[0], 'trend_range'] = 0
        
        # mplfinanceでプロット用の設定
        main_plots = []
        
        # 買いシグナルのマーカー
        main_plots.append(mpf.make_addplot(df['buy_signal'], type='scatter', 
                                          markersize=100, marker='^', color='green', 
                                          alpha=0.8, label='Buy Signal'))
        
        # 売りシグナルのマーカー
        main_plots.append(mpf.make_addplot(df['sell_signal'], type='scatter', 
                                          markersize=100, marker='v', color='red', 
                                          alpha=0.8, label='Sell Signal'))
        
        # オシレータープロット
        # トレンド状態パネル - 状態別に色分けしたプロット
        trend_up_panel = mpf.make_addplot(df['trend_up'], panel=1, color='green', width=4.0, 
                                         ylabel='Trend State\n(1=Up, -1=Down, 0=Range)', secondary_y=False, 
                                         label='Trend Up', type='line')
        trend_down_panel = mpf.make_addplot(df['trend_down'], panel=1, color='red', width=4.0, 
                                           secondary_y=False, label='Trend Down', type='line')
        trend_range_panel = mpf.make_addplot(df['trend_range'], panel=1, color='gray', width=2.0, 
                                           secondary_y=False, label='Range', type='line')
        
        # シグナルパネル
        signal_panel = mpf.make_addplot(df['signal'], panel=2, color='purple', width=2.5, 
                                       ylabel='Signal\n(1=Buy, -1=Sell, 0=Neutral)', secondary_y=False, 
                                       label='Signal', type='line')
        
        # フェーザー角度パネル
        phase_panel = mpf.make_addplot(df['phase_angle'], panel=3, color='orange', width=1.5, 
                                      ylabel='Phase Angle (°)', secondary_y=False, label='Phase Angle')
        
        # トレンド強度パネル
        strength_panel = mpf.make_addplot(df['trend_strength'], panel=4, color='green', width=1.5, 
                                         ylabel='Trend Strength\n(0-1)', secondary_y=False, label='Trend Strength')
        
        # サイクル信頼度パネル
        confidence_panel = mpf.make_addplot(df['cycle_confidence'], panel=5, color='cyan', width=1.5, 
                                           ylabel='Cycle Confidence\n(0-1)', secondary_y=False, label='Cycle Confidence')
        
        # 瞬間周期パネル
        period_panel = mpf.make_addplot(df['inst_period'], panel=6, color='brown', width=1.5, 
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
            kwargs['panel_ratios'] = (5, 1, 1, 1, 1, 1, 1, 1)  # メイン:出来高:Trend:Signal:Phase:Strength:Confidence:Period
            # 出来高を表示する場合は、オシレーターのパネル番号を+1する
            trend_up_panel = mpf.make_addplot(df['trend_up'], panel=2, color='green', width=4.0, 
                                             ylabel='Trend State', secondary_y=False, label='Trend Up', type='line')
            trend_down_panel = mpf.make_addplot(df['trend_down'], panel=2, color='red', width=4.0, 
                                               secondary_y=False, label='Trend Down', type='line')
            trend_range_panel = mpf.make_addplot(df['trend_range'], panel=2, color='gray', width=2.0, 
                                               secondary_y=False, label='Range', type='line')
            signal_panel = mpf.make_addplot(df['signal'], panel=3, color='purple', width=2.5, 
                                           ylabel='Signal', secondary_y=False, label='Signal', type='line')
            phase_panel = mpf.make_addplot(df['phase_angle'], panel=4, color='orange', width=1.5, 
                                          ylabel='Phase Angle (°)', secondary_y=False, label='Phase Angle')
            strength_panel = mpf.make_addplot(df['trend_strength'], panel=5, color='green', width=1.5, 
                                             ylabel='Trend Strength', secondary_y=False, label='Trend Strength')
            confidence_panel = mpf.make_addplot(df['cycle_confidence'], panel=6, color='cyan', width=1.5, 
                                               ylabel='Cycle Confidence', secondary_y=False, label='Cycle Confidence')
            period_panel = mpf.make_addplot(df['inst_period'], panel=7, color='brown', width=1.5, 
                                           ylabel='Instantaneous Period', secondary_y=False, label='Period')
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (5, 1, 1, 1, 1, 1, 1)  # メイン:Trend:Signal:Phase:Strength:Confidence:Period
        
        # すべてのプロットを結合
        all_plots = main_plots + [trend_up_panel, trend_down_panel, trend_range_panel, signal_panel, phase_panel, strength_panel, confidence_panel, period_panel]
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加
        axes[0].legend(['Buy Signal', 'Sell Signal'], loc='upper left')
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        if show_volume:
            # トレンド状態パネル - y軸を強制的に設定
            trend_axis = axes[2]
            trend_axis.set_ylim(-1.5, 1.5)
            trend_axis.set_yticks([-1, 0, 1])
            trend_axis.set_yticklabels(['Down', 'Range', 'Up'])
            
            # 参照線を追加
            trend_axis.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=1)
            trend_axis.axhline(y=1, color='green', linestyle='--', alpha=0.6, linewidth=1)
            trend_axis.axhline(y=-1, color='red', linestyle='--', alpha=0.6, linewidth=1)
            
            # シグナルパネル
            axes[3].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[3].axhline(y=1, color='green', linestyle='--', alpha=0.5)
            axes[3].axhline(y=-1, color='red', linestyle='--', alpha=0.5)
            axes[3].set_ylim(-1.2, 1.2)
            
            # フェーザー角度パネル
            axes[4].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[4].axhline(y=90, color='green', linestyle='--', alpha=0.3)
            axes[4].axhline(y=-90, color='red', linestyle='--', alpha=0.3)
            axes[4].axhline(y=180, color='black', linestyle='-', alpha=0.2)
            axes[4].axhline(y=-180, color='black', linestyle='-', alpha=0.2)
            
            # トレンド強度パネル
            axes[5].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
            axes[5].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[5].axhline(y=1, color='black', linestyle='-', alpha=0.3)
            axes[5].set_ylim(0, 1)
            
            # サイクル信頼度パネル
            axes[6].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
            axes[6].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[6].axhline(y=1, color='black', linestyle='-', alpha=0.3)
            axes[6].set_ylim(0, 1)
            
            # 瞬間周期パネル
            period_mean = df['inst_period'].mean()
            axes[7].axhline(y=period_mean, color='black', linestyle='-', alpha=0.3)
            axes[7].axhline(y=28, color='brown', linestyle='--', alpha=0.5)  # 基準周期
        else:
            # トレンド状態パネル - y軸を強制的に設定
            trend_axis = axes[1]
            trend_axis.set_ylim(-1.5, 1.5)
            trend_axis.set_yticks([-1, 0, 1])
            trend_axis.set_yticklabels(['Down', 'Range', 'Up'])
            
            # 参照線を追加
            trend_axis.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=1)
            trend_axis.axhline(y=1, color='green', linestyle='--', alpha=0.6, linewidth=1)
            trend_axis.axhline(y=-1, color='red', linestyle='--', alpha=0.6, linewidth=1)
            
            # シグナルパネル
            axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[2].axhline(y=1, color='green', linestyle='--', alpha=0.5)
            axes[2].axhline(y=-1, color='red', linestyle='--', alpha=0.5)
            axes[2].set_ylim(-1.2, 1.2)
            
            # フェーザー角度パネル
            axes[3].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[3].axhline(y=90, color='green', linestyle='--', alpha=0.3)
            axes[3].axhline(y=-90, color='red', linestyle='--', alpha=0.3)
            axes[3].axhline(y=180, color='black', linestyle='-', alpha=0.2)
            axes[3].axhline(y=-180, color='black', linestyle='-', alpha=0.2)
            
            # トレンド強度パネル
            axes[4].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
            axes[4].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[4].axhline(y=1, color='black', linestyle='-', alpha=0.3)
            axes[4].set_ylim(0, 1)
            
            # サイクル信頼度パネル
            axes[5].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
            axes[5].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[5].axhline(y=1, color='black', linestyle='-', alpha=0.3)
            axes[5].set_ylim(0, 1)
            
            # 瞬間周期パネル
            period_mean = df['inst_period'].mean()
            axes[6].axhline(y=period_mean, color='black', linestyle='-', alpha=0.3)
            axes[6].axhline(y=28, color='brown', linestyle='--', alpha=0.5)  # 基準周期
        
        # 統計情報の表示
        print(f"\n=== Phasor Trend Filter統計 ===")
        total_points = len(df)
        trend_points = len(df[df['trend_state'] == 1])
        range_points = len(df[df['trend_state'] == 0])
        buy_signals = len(df[df['signal'] == 1])
        sell_signals = len(df[df['signal'] == -1])
        neutral_signals = len(df[df['signal'] == 0])
        
        print(f"総データ点数: {total_points}")
        print(f"トレンド状態: {trend_points} ({trend_points/total_points*100:.1f}%)")
        print(f"レンジ状態: {range_points} ({range_points/total_points*100:.1f}%)")
        print(f"買いシグナル: {buy_signals} ({buy_signals/total_points*100:.1f}%)")
        print(f"売りシグナル: {sell_signals} ({sell_signals/total_points*100:.1f}%)")
        print(f"中立シグナル: {neutral_signals} ({neutral_signals/total_points*100:.1f}%)")
        print(f"トレンド強度 - 平均: {df['trend_strength'].mean():.3f}, 範囲: {df['trend_strength'].min():.3f} - {df['trend_strength'].max():.3f}")
        print(f"サイクル信頼度 - 平均: {df['cycle_confidence'].mean():.3f}, 範囲: {df['cycle_confidence'].min():.3f} - {df['cycle_confidence'].max():.3f}")
        print(f"瞬間周期 - 平均: {df['inst_period'].mean():.1f}, 範囲: {df['inst_period'].min():.1f} - {df['inst_period'].max():.1f}")
        
        # シグナル遷移の分析
        signal_changes = (df['signal'] != df['signal'].shift(1)).sum()
        trend_changes = (df['trend_state'] != df['trend_state'].shift(1)).sum()
        print(f"シグナル変更: {signal_changes}回")
        print(f"トレンド状態変更: {trend_changes}回")
        
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
    parser = argparse.ArgumentParser(description='Phasor Trend Filterの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--period', type=int, default=28, help='フェーザー分析の固定周期')
    parser.add_argument('--threshold', type=float, default=6.0, help='トレンド判定しきい値')
    parser.add_argument('--src-type', type=str, default='close', help='ソースタイプ')
    parser.add_argument('--kalman', action='store_true', help='カルマンフィルターを有効にする')
    args = parser.parse_args()
    
    # チャートを作成
    chart = PhasorTrendFilterChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        period=args.period,
        trend_threshold=args.threshold,
        src_type=args.src_type,
        use_kalman_filter=args.kalman
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output
    )


if __name__ == "__main__":
    main()