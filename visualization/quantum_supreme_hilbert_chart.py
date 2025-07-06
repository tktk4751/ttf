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
from indicators.quantum_supreme_hilbert import QuantumSupremeHilbert


class QuantumSupremeHilbertChart:
    """
    量子Supreme版ヒルベルト変換を表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - 瞬時振幅（Instantaneous Amplitude）
    - 瞬時位相（Instantaneous Phase）
    - 瞬時周波数（Instantaneous Frequency）
    - 量子コヒーレンス（Quantum Coherence）
    - 市場状態（トレンドモード/サイクルモード判別）
    - トレンド強度とサイクル強度
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.quantum_hilbert = None
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
                            src_type: str = 'close',                    # 価格ソース
                            coherence_threshold: float = 0.7,           # コヒーレンス閾値
                            frequency_threshold: float = 0.1,           # 周波数閾値
                            amplitude_threshold: float = 0.5,           # 振幅閾値
                            analysis_window: int = 14,                  # 分析ウィンドウサイズ
                            min_periods: int = 32                       # 最小計算期間
                           ) -> None:
        """
        量子Supreme版ヒルベルト変換を計算する
        
        Args:
            src_type: 価格ソース ('close', 'hlc3', 'hl2', 'ohlc4')
            coherence_threshold: コヒーレンス閾値（トレンド判定用）
            frequency_threshold: 周波数閾値（トレンド判定用）
            amplitude_threshold: 振幅閾値（変動性判定用）
            analysis_window: 市場状態分析ウィンドウサイズ
            min_periods: 最小計算期間
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\n量子Supreme版ヒルベルト変換を計算中...")
        
        # 量子Supreme版ヒルベルト変換インディケーターを初期化
        self.quantum_hilbert = QuantumSupremeHilbert(
            src_type=src_type,
            coherence_threshold=coherence_threshold,
            frequency_threshold=frequency_threshold,
            amplitude_threshold=amplitude_threshold,
            analysis_window=analysis_window,
            min_periods=min_periods
        )
        
        # 計算を実行
        print("計算を実行します...")
        result = self.quantum_hilbert.calculate(self.data)
        
        # 結果のテスト取得
        amplitude = self.quantum_hilbert.get_amplitude()
        phase = self.quantum_hilbert.get_phase()
        frequency = self.quantum_hilbert.get_frequency()
        coherence = self.quantum_hilbert.get_quantum_coherence()
        trend_mode = self.quantum_hilbert.get_trend_mode()
        market_state = self.quantum_hilbert.get_market_state()
        cycle_strength = self.quantum_hilbert.get_cycle_strength()
        trend_strength = self.quantum_hilbert.get_trend_strength()
        
        print(f"計算完了:")
        print(f"  振幅: {len(amplitude)}, NaN値: {np.isnan(amplitude).sum()}")
        print(f"  位相: {len(phase)}, NaN値: {np.isnan(phase).sum()}")
        print(f"  周波数: {len(frequency)}, NaN値: {np.isnan(frequency).sum()}")
        print(f"  コヒーレンス: {len(coherence)}, NaN値: {np.isnan(coherence).sum()}")
        print(f"  トレンドモード: 有効データ数: {(trend_mode != 0).sum()}")
        print(f"  市場状態: レンジ: {(market_state == 0).sum()}, 弱トレンド: {(market_state == 1).sum()}, 強トレンド: {(market_state == 2).sum()}")
        
        # 現在の状態を表示
        current_state = self.quantum_hilbert.get_current_state()
        if current_state:
            print(f"\n現在の市場状態:")
            print(f"  振幅: {current_state['amplitude']:.4f}")
            print(f"  位相: {current_state['phase']:.4f}")
            print(f"  周波数: {current_state['frequency']:.4f}")
            print(f"  量子コヒーレンス: {current_state['quantum_coherence']:.4f}")
            print(f"  トレンドモード: {current_state['trend_mode']}")
            print(f"  市場状態: {current_state['market_state_str']}")
            print(f"  サイクル強度: {current_state['cycle_strength']:.4f}")
            print(f"  トレンド強度: {current_state['trend_strength']:.4f}")
        
        print("量子Supreme版ヒルベルト変換の計算完了")
            
    def plot(self, 
            title: str = "量子Supreme版ヒルベルト変換", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (16, 20),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートと量子Supreme版ヒルベルト変換を描画する
        
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
            
        if self.quantum_hilbert is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # 量子Supreme版ヒルベルト変換の値を取得
        print("ヒルベルト変換データを取得中...")
        amplitude = self.quantum_hilbert.get_amplitude()
        phase = self.quantum_hilbert.get_phase()
        frequency = self.quantum_hilbert.get_frequency()
        coherence = self.quantum_hilbert.get_quantum_coherence()
        trend_mode = self.quantum_hilbert.get_trend_mode()
        market_state = self.quantum_hilbert.get_market_state()
        cycle_strength = self.quantum_hilbert.get_cycle_strength()
        trend_strength = self.quantum_hilbert.get_trend_strength()
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'amplitude': amplitude,
                'phase': phase,
                'frequency': frequency,
                'coherence': coherence,
                'trend_mode': trend_mode,
                'market_state': market_state,
                'cycle_strength': cycle_strength,
                'trend_strength': trend_strength
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        print(f"データ確認 - 振幅NaN: {df['amplitude'].isna().sum()}, 位相NaN: {df['phase'].isna().sum()}")
        
        # 位相を度数に変換（表示用）
        df['phase_degrees'] = df['phase'] * 180 / np.pi
        
        # 正規化周波数をパーセント表示用に変換
        df['frequency_percent'] = df['frequency'] * 100
        
        # トレンドモード用の色分け用データ準備
        df['trend_mode_trend'] = np.where(df['trend_mode'] == 1, 1, np.nan)
        df['trend_mode_cycle'] = np.where(df['trend_mode'] == 0, 0, np.nan)
        
        # 市場状態用の色分け用データ準備
        df['market_range'] = np.where(df['market_state'] == 0, 0, np.nan)
        df['market_weak_trend'] = np.where(df['market_state'] == 1, 1, np.nan)
        df['market_strong_trend'] = np.where(df['market_state'] == 2, 2, np.nan)
        
        # mplfinanceでプロット用の設定
        main_plots = []
        
        # 振幅をメインチャート上にオーバーレイ表示（価格に正規化）
        price_range = df['high'].max() - df['low'].min()
        amplitude_normalized = df['low'].min() + (df['amplitude'] / df['amplitude'].max()) * price_range * 0.1
        main_plots.append(mpf.make_addplot(amplitude_normalized, color='cyan', width=1, alpha=0.6, label='Amplitude'))
        
        # 各種インジケーターパネルの設定
        panel_num = 1
        if show_volume:
            panel_num = 2  # 出来高パネルの分を考慮
        
        # パネル1: 瞬時振幅
        amplitude_panel = mpf.make_addplot(df['amplitude'], panel=panel_num, color='blue', width=1.5, 
                                          ylabel='Amplitude', label='Instantaneous Amplitude')
        
        # パネル2: 瞬時位相（度数表示）
        phase_panel = mpf.make_addplot(df['phase_degrees'], panel=panel_num+1, color='purple', width=1.5, 
                                      ylabel='Phase (deg)', label='Instantaneous Phase')
        
        # パネル3: 瞬時周波数（パーセント表示）
        frequency_panel = mpf.make_addplot(df['frequency_percent'], panel=panel_num+2, color='orange', width=1.5, 
                                          ylabel='Frequency (%)', label='Instantaneous Frequency')
        
        # パネル4: 量子コヒーレンス
        coherence_panel = mpf.make_addplot(df['coherence'], panel=panel_num+3, color='green', width=1.5, 
                                          ylabel='Coherence', label='Quantum Coherence')
        
        # パネル5: トレンドモード（散布図的に表示）
        trend_mode_panels = [
            mpf.make_addplot(df['trend_mode_trend'], panel=panel_num+4, color='red', width=2, 
                           ylabel='Trend Mode', label='Trend', type='scatter', markersize=20),
            mpf.make_addplot(df['trend_mode_cycle'], panel=panel_num+4, color='blue', width=2, 
                           label='Cycle', type='scatter', markersize=20)
        ]
        
        # パネル6: 市場状態
        market_state_panels = [
            mpf.make_addplot(df['market_range'], panel=panel_num+5, color='gray', width=2, 
                           ylabel='Market State', label='Range', type='scatter', markersize=15),
            mpf.make_addplot(df['market_weak_trend'], panel=panel_num+5, color='yellow', width=2, 
                           label='Weak Trend', type='scatter', markersize=15),
            mpf.make_addplot(df['market_strong_trend'], panel=panel_num+5, color='red', width=2, 
                           label='Strong Trend', type='scatter', markersize=15)
        ]
        
        # パネル7: サイクル強度とトレンド強度
        strength_panels = [
            mpf.make_addplot(df['cycle_strength'], panel=panel_num+6, color='blue', width=1.5, 
                           ylabel='Strength', label='Cycle Strength'),
            mpf.make_addplot(df['trend_strength'], panel=panel_num+6, color='red', width=1.5, 
                           label='Trend Strength', secondary_y=False)
        ]
        
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
        
        # 出来高とパネル比率の設定
        if show_volume:
            kwargs['volume'] = True
            kwargs['panel_ratios'] = (4, 1, 1.2, 1.2, 1.2, 1.2, 1, 1, 1.2)  # メイン:出来高:振幅:位相:周波数:コヒーレンス:トレンドモード:市場状態:強度
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (4, 1.2, 1.2, 1.2, 1.2, 1, 1, 1.2)  # メイン:振幅:位相:周波数:コヒーレンス:トレンドモード:市場状態:強度
        
        # すべてのプロットを結合
        all_plots = (main_plots + 
                    [amplitude_panel, phase_panel, frequency_panel, coherence_panel] +
                    trend_mode_panels + market_state_panels + strength_panels)
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加（メインチャート）
        axes[0].legend(['Amplitude (Normalized)'], loc='upper left')
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        axis_offset = 1 if show_volume else 0
        
        # 位相パネル: -180, 0, 180度の線
        phase_axis = axes[axis_offset + 2]
        phase_axis.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        phase_axis.axhline(y=180, color='black', linestyle='--', alpha=0.3)
        phase_axis.axhline(y=-180, color='black', linestyle='--', alpha=0.3)
        phase_axis.axhline(y=90, color='gray', linestyle=':', alpha=0.3)
        phase_axis.axhline(y=-90, color='gray', linestyle=':', alpha=0.3)
        
        # 周波数パネル: 基準線
        freq_axis = axes[axis_offset + 3]
        freq_axis.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='10%')  # 高周波の目安
        freq_axis.axhline(y=5, color='orange', linestyle='--', alpha=0.3, label='5%')
        
        # コヒーレンスパネル: 0.5, 0.7の閾値線
        coherence_axis = axes[axis_offset + 4]
        coherence_axis.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5)
        coherence_axis.axhline(y=0.7, color='red', linestyle='--', alpha=0.5)
        coherence_axis.axhline(y=1.0, color='green', linestyle='-', alpha=0.3)
        
        # トレンドモードパネル: 0.5の基準線
        trend_mode_axis = axes[axis_offset + 5]
        trend_mode_axis.axhline(y=0.5, color='black', linestyle='-', alpha=0.5)
        
        # 市場状態パネル: 各状態の基準線
        market_state_axis = axes[axis_offset + 6]
        market_state_axis.axhline(y=0, color='gray', linestyle='-', alpha=0.3, label='Range')
        market_state_axis.axhline(y=1, color='yellow', linestyle='-', alpha=0.3, label='Weak Trend')
        market_state_axis.axhline(y=2, color='red', linestyle='-', alpha=0.3, label='Strong Trend')
        
        # 強度パネル: 0.5の均衡線
        strength_axis = axes[axis_offset + 7]
        strength_axis.axhline(y=0.5, color='black', linestyle='-', alpha=0.5)
        strength_axis.axhline(y=0.8, color='red', linestyle='--', alpha=0.3)
        strength_axis.axhline(y=0.2, color='blue', linestyle='--', alpha=0.3)
        
        # 統計情報の表示
        print(f"\n=== ヒルベルト変換統計 ===")
        valid_data = df.dropna()
        total_points = len(valid_data)
        
        if total_points > 0:
            trend_mode_count = (valid_data['trend_mode'] == 1).sum()
            cycle_mode_count = (valid_data['trend_mode'] == 0).sum()
            
            range_count = (valid_data['market_state'] == 0).sum()
            weak_trend_count = (valid_data['market_state'] == 1).sum()
            strong_trend_count = (valid_data['market_state'] == 2).sum()
            
            print(f"総データ点数: {total_points}")
            print(f"トレンドモード: {trend_mode_count} ({trend_mode_count/total_points*100:.1f}%)")
            print(f"サイクルモード: {cycle_mode_count} ({cycle_mode_count/total_points*100:.1f}%)")
            print(f"レンジ状態: {range_count} ({range_count/total_points*100:.1f}%)")
            print(f"弱トレンド: {weak_trend_count} ({weak_trend_count/total_points*100:.1f}%)")
            print(f"強トレンド: {strong_trend_count} ({strong_trend_count/total_points*100:.1f}%)")
            print(f"平均振幅: {valid_data['amplitude'].mean():.4f}")
            print(f"平均周波数: {valid_data['frequency'].mean():.4f}")
            print(f"平均コヒーレンス: {valid_data['coherence'].mean():.4f}")
            print(f"平均トレンド強度: {valid_data['trend_strength'].mean():.4f}")
            print(f"平均サイクル強度: {valid_data['cycle_strength'].mean():.4f}")
        
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
    parser = argparse.ArgumentParser(description='量子Supreme版ヒルベルト変換の描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--src-type', type=str, default='close', help='価格ソースタイプ')
    parser.add_argument('--coherence-threshold', type=float, default=0.7, help='コヒーレンス閾値')
    parser.add_argument('--frequency-threshold', type=float, default=0.1, help='周波数閾値')
    parser.add_argument('--amplitude-threshold', type=float, default=0.5, help='振幅閾値')
    parser.add_argument('--analysis-window', type=int, default=14, help='分析ウィンドウサイズ')
    parser.add_argument('--min-periods', type=int, default=32, help='最小計算期間')
    args = parser.parse_args()
    
    # チャートを作成
    chart = QuantumSupremeHilbertChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        src_type=args.src_type,
        coherence_threshold=args.coherence_threshold,
        frequency_threshold=args.frequency_threshold,
        amplitude_threshold=args.amplitude_threshold,
        analysis_window=args.analysis_window,
        min_periods=args.min_periods
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output
    )


if __name__ == "__main__":
    main() 