#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mplfinance as mpf
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
import warnings
warnings.filterwarnings('ignore')

# データ取得のための依存関係
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource

# インジケーター
from indicators.quantum_wavelet_cycle import QuantumWaveletCycle


class QuantumWaveletCycleChart:
    """
    Quantum Wavelet Cycle Detectorを表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - ドミナントサイクル値の表示
    - 生の周期 vs 平滑化周期の比較
    - 統計情報とパラメータ表示
    - パワースペクトラム分析
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.quantum_wavelet_cycle = None
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

    def generate_synthetic_data(
        self, 
        length: int = 500, 
        base_price: float = 100.0,
        noise_level: float = 0.3,
        trend_strength: float = 0.02
    ) -> pd.DataFrame:
        """
        合成テストデータの生成
        
        Args:
            length: データ長
            base_price: ベース価格
            noise_level: ノイズレベル
            trend_strength: トレンド強度
            
        Returns:
            合成価格データのDataFrame
        """
        np.random.seed(42)
        time = np.arange(length)
        
        # 周波数の変化（40周期から15周期）
        freq = np.linspace(1/40., 1/15., length)
        phase = np.cumsum(freq * 2 * np.pi)
        
        # 複数のサイクル成分
        signal_part1 = 2 * np.sin(phase)
        signal_part2 = 0.5 * np.sin(phase * 2.5)
        signal_part3 = 0.3 * np.sin(phase * 0.7)  # 長周期成分
        
        # トレンドとノイズ
        trend = trend_strength * time
        noise = np.random.randn(length) * noise_level
        
        # 価格データ生成
        close = signal_part1 + signal_part2 + signal_part3 + trend + base_price + noise
        
        # OHLC生成（クローズ価格ベース）
        high = close + np.abs(np.random.randn(length) * 0.5)
        low = close - np.abs(np.random.randn(length) * 0.5)
        open_price = close + np.random.randn(length) * 0.2
        
        # 日時インデックス生成
        date_range = pd.date_range(start='2023-01-01', periods=length, freq='H')
        
        # ボリューム生成
        volume = np.abs(np.random.randn(length) * 1000 + 5000)
        
        df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
        }, index=date_range)
        
        self.data = df
        return df

    def calculate_indicators(self,
                            min_period: int = 8,
                            max_period: int = 60,
                            history_size: int = 100,
                            kalman_q: float = 0.05,
                            kalman_r: float = 2.0,
                            cycle_part: float = 0.5,
                            max_output: int = 34,
                            min_output: int = 1,
                            src_type: str = 'hlc3'
                           ) -> None:
        """
        Quantum Wavelet Cycle Detectorを計算する
        
        Args:
            min_period: 検索する最小サイクル期間
            max_period: 検索する最大サイクル期間
            history_size: 解析用に保持する価格バーの数
            kalman_q: カルマンフィルタープロセスノイズ
            kalman_r: カルマンフィルター測定ノイズ
            cycle_part: サイクル部分の倍率
            max_output: 最大出力値
            min_output: 最小出力値
            src_type: ソースタイプ
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()またはgenerate_synthetic_data()を先に実行してください。")
            
        print("\nQuantum Wavelet Cycle Detectorを計算中...")
        
        # Quantum Wavelet Cycle Detectorを計算
        self.quantum_wavelet_cycle = QuantumWaveletCycle(
            min_period=min_period,
            max_period=max_period,
            history_size=history_size,
            kalman_q=kalman_q,
            kalman_r=kalman_r,
            cycle_part=cycle_part,
            max_output=max_output,
            min_output=min_output,
            src_type=src_type
        )
        
        # 計算実行
        print("計算を実行します...")
        cycle_values = self.quantum_wavelet_cycle.calculate(self.data)
        
        print(f"QWCD計算完了 - サイクル値: {len(cycle_values)}")
        
        # NaN値のチェック
        nan_count = np.isnan(cycle_values).sum()
        valid_count = len(cycle_values) - nan_count
        
        print(f"NaN値: {nan_count}, 有効値: {valid_count}")
        
        if valid_count > 0:
            print(f"サイクル統計 - 平均: {np.nanmean(cycle_values):.2f}, 範囲: [{np.nanmin(cycle_values):.1f}, {np.nanmax(cycle_values):.1f}]")
        
        print("Quantum Wavelet Cycle Detector計算完了")

    def calculate_power_spectrum_over_time(
        self,
        window_size: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        時間経過に伴うパワースペクトラム計算
        
        Args:
            window_size: ウィンドウサイズ
            
        Returns:
            Tuple[time_points, periods, power_matrix]
        """
        if self.quantum_wavelet_cycle is None:
            return np.array([]), np.array([]), np.array([])
            
        price = self.quantum_wavelet_cycle.calculate_source_values(self.data, self.quantum_wavelet_cycle.src_type)
        periods = np.arange(self.quantum_wavelet_cycle.min_period, self.quantum_wavelet_cycle.max_period + 1)
        
        # 時間ポイント選択
        time_points = np.arange(self.quantum_wavelet_cycle.history_size, len(price), window_size)
        power_matrix = np.zeros((len(periods), len(time_points)))
        
        for i, t in enumerate(time_points):
            if t >= self.quantum_wavelet_cycle.history_size:
                # 履歴データ取得
                start_idx = t - self.quantum_wavelet_cycle.history_size + 1
                price_history = price[start_idx:t+1]
                
                # トレンド除去
                if len(price_history) >= 4:
                    from indicators.quantum_wavelet_cycle import detrend_series, calculate_cwt_power
                    try:
                        detrended = detrend_series(price_history)
                        power_spectrum = calculate_cwt_power(detrended, periods)
                        power_matrix[:, i] = power_spectrum
                    except:
                        pass  # エラーの場合はゼロのまま
        
        return time_points, periods, power_matrix
            
    def plot(self, 
            title: str = "Quantum Wavelet Cycle Detector", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            show_power_spectrum: bool = True,
            figsize: Tuple[int, int] = (16, 12),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとQuantum Wavelet Cycle Detectorを描画する
        
        Args:
            title: チャートのタイトル
            start_date: 表示開始日（フォーマット: YYYY-MM-DD）
            end_date: 表示終了日（フォーマット: YYYY-MM-DD）
            show_volume: 出来高を表示するか
            show_power_spectrum: パワースペクトラムを表示するか
            figsize: 図のサイズ
            style: mplfinanceのスタイル
            savefig: 保存先のパス（指定しない場合は表示のみ）
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。")
            
        if self.quantum_wavelet_cycle is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # Quantum Wavelet Cycle Detectorの値を取得
        print("QWCD データを取得中...")
        cycle_values = self.quantum_wavelet_cycle.calculate(self.data)
        
        # _resultから詳細データ取得
        result = self.quantum_wavelet_cycle._result
        if result is not None:
            raw_period = result.raw_period
            smooth_period = result.smooth_period
        else:
            raw_period = np.full_like(cycle_values, np.nan)
            smooth_period = np.full_like(cycle_values, np.nan)
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'cycle_values': cycle_values,
                'raw_period': raw_period,
                'smooth_period': smooth_period,
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        print(f"サイクルデータ確認 - NaN: {df['cycle_values'].isna().sum()}")
        
        # mplfinanceでプロット用の設定
        main_plots = []
        
        # サイクル値をメインチャートの右軸に表示
        cycle_plot = mpf.make_addplot(df['cycle_values'], panel=0, color='purple', width=2, 
                                     secondary_y=True, ylabel='Cycle Period')
        main_plots.append(cycle_plot)
        
        # 2. オシレータープロット
        # 生の周期 vs 平滑化周期
        raw_period_panel = mpf.make_addplot(df['raw_period'], panel=1, color='orange', width=1.2, 
                                           ylabel='Period', secondary_y=False, label='Raw Period')
        smooth_period_panel = mpf.make_addplot(df['smooth_period'], panel=1, color='red', width=2, 
                                              secondary_y=False, label='Smooth Period')
        
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
        
        # 出来高とパネルの設定
        if show_volume:
            kwargs['volume'] = True
            kwargs['panel_ratios'] = (4, 1, 1.5)  # メイン:出来高:周期比較
            # 出来高を表示する場合は、オシレーターのパネル番号を+1する
            raw_period_panel = mpf.make_addplot(df['raw_period'], panel=2, color='orange', width=1.2, 
                                               ylabel='Period', secondary_y=False, label='Raw Period')
            smooth_period_panel = mpf.make_addplot(df['smooth_period'], panel=2, color='red', width=2, 
                                                  secondary_y=False, label='Smooth Period')
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (4, 1.5)  # メイン:周期比較
        
        # すべてのプロットを結合
        all_plots = main_plots + [raw_period_panel, smooth_period_panel]
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加
        if show_volume:
            # 周期比較パネルに参照線追加
            period_ax = axes[2]
            period_ax.axhline(y=self.quantum_wavelet_cycle.min_period, color='black', linestyle='--', alpha=0.5, label=f'Min: {self.quantum_wavelet_cycle.min_period}')
            period_ax.axhline(y=self.quantum_wavelet_cycle.max_period, color='black', linestyle='--', alpha=0.5, label=f'Max: {self.quantum_wavelet_cycle.max_period}')
            period_ax.legend(['Raw Period', 'Smooth Period', f'Min: {self.quantum_wavelet_cycle.min_period}', f'Max: {self.quantum_wavelet_cycle.max_period}'], 
                            loc='upper left')
        else:
            # 周期比較パネルに参照線追加
            period_ax = axes[1]
            period_ax.axhline(y=self.quantum_wavelet_cycle.min_period, color='black', linestyle='--', alpha=0.5)
            period_ax.axhline(y=self.quantum_wavelet_cycle.max_period, color='black', linestyle='--', alpha=0.5)
            period_ax.legend(['Raw Period', 'Smooth Period', f'Min: {self.quantum_wavelet_cycle.min_period}', f'Max: {self.quantum_wavelet_cycle.max_period}'], 
                            loc='upper left')
        
        self.fig = fig
        self.axes = axes
        
        # 統計情報の表示
        print(f"\n=== Quantum Wavelet Cycle Detector 統計 ===")
        valid_cycles = df['cycle_values'].dropna()
        
        if len(valid_cycles) > 0:
            print(f"総データ点数: {len(df)}")
            print(f"有効サイクル点数: {len(valid_cycles)}")
            print(f"平均サイクル: {valid_cycles.mean():.2f}")
            print(f"標準偏差: {valid_cycles.std():.2f}")
            print(f"範囲: {valid_cycles.min():.2f} - {valid_cycles.max():.2f}")
            
            # パラメータ情報
            print(f"\n=== パラメータ ===")
            print(f"最小周期: {self.quantum_wavelet_cycle.min_period}")
            print(f"最大周期: {self.quantum_wavelet_cycle.max_period}")
            print(f"履歴サイズ: {self.quantum_wavelet_cycle.history_size}")
            print(f"カルマンQ: {self.quantum_wavelet_cycle.kalman_q}")
            print(f"カルマンR: {self.quantum_wavelet_cycle.kalman_r}")
            print(f"ソースタイプ: {self.quantum_wavelet_cycle.src_type}")
        
        # パワースペクトラム分析（オプション）
        if show_power_spectrum:
            self._plot_power_spectrum_analysis(df)
        
        # 保存または表示
        if savefig:
            plt.savefig(savefig, dpi=150, bbox_inches='tight')
            print(f"チャートを保存しました: {savefig}")
        else:
            plt.tight_layout()
            plt.show()

    def _plot_power_spectrum_analysis(self, df: pd.DataFrame) -> None:
        """
        パワースペクトラム分析の追加プロット
        
        Args:
            df: メインのデータフレーム
        """
        try:
            print("\nパワースペクトラム分析を実行中...")
            time_points, periods, power_matrix = self.calculate_power_spectrum_over_time(window_size=20)
            
            if power_matrix.size > 0 and np.max(power_matrix) > 0:
                # 新しいfigure作成
                fig_spectrum, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
                
                # 1. パワースペクトラムヒートマップ
                power_matrix_norm = power_matrix / np.max(power_matrix)
                im = ax1.imshow(power_matrix_norm, aspect='auto', origin='lower',
                               cmap='hot', interpolation='bilinear',
                               extent=[time_points[0], time_points[-1], 
                                      periods[0], periods[-1]])
                
                plt.colorbar(im, ax=ax1, label='正規化パワー')
                ax1.set_ylabel('周期', fontsize=12)
                ax1.set_title('時間-周波数パワースペクトラム', fontsize=14)
                ax1.set_xlabel('時間（データポイント）', fontsize=12)
                
                # 2. 平均パワースペクトラム
                mean_power = np.mean(power_matrix, axis=1)
                ax2.plot(periods, mean_power, 'b-', linewidth=2, label='平均パワー')
                ax2.set_xlabel('周期', fontsize=12)
                ax2.set_ylabel('パワー', fontsize=12)
                ax2.set_title('平均パワースペクトラム', fontsize=14)
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                
                # 支配的周期をマーク
                dominant_idx = np.argmax(mean_power)
                dominant_period = periods[dominant_idx]
                ax2.axvline(x=dominant_period, color='red', linestyle='--', alpha=0.7, 
                           label=f'支配的周期: {dominant_period:.1f}')
                ax2.legend()
                
                plt.tight_layout()
                plt.suptitle('Quantum Wavelet Cycle Detector - パワースペクトラム分析', y=1.02)
                
                print(f"支配的周期: {dominant_period:.1f}")
                
                plt.show()
            else:
                print("パワースペクトラムデータが不十分です")
        except Exception as e:
            print(f"パワースペクトラム分析エラー: {str(e)}")


def main():
    """メイン関数"""
    # コマンドライン引数を処理
    import argparse
    parser = argparse.ArgumentParser(description='Quantum Wavelet Cycle Detectorの描画')
    parser.add_argument('--config', '-c', type=str, help='設定ファイルのパス')
    parser.add_argument('--synthetic', '-syn', action='store_true', help='合成データを使用')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--src-type', type=str, default='hlc3', help='ソースタイプ')
    parser.add_argument('--min-period', type=int, default=8, help='最小周期')
    parser.add_argument('--max-period', type=int, default=60, help='最大周期')
    parser.add_argument('--history-size', type=int, default=100, help='履歴サイズ')
    parser.add_argument('--kalman-q', type=float, default=0.05, help='カルマンQ')
    parser.add_argument('--kalman-r', type=float, default=2.0, help='カルマンR')
    args = parser.parse_args()
    
    # チャートを作成
    chart = QuantumWaveletCycleChart()
    
    if args.synthetic:
        print("合成データを生成中...")
        chart.generate_synthetic_data(length=500, noise_level=0.3)
    elif args.config:
        chart.load_data_from_config(args.config)
    else:
        print("--configまたは--syntheticオプションを指定してください")
        return
    
    chart.calculate_indicators(
        src_type=args.src_type,
        min_period=args.min_period,
        max_period=args.max_period,
        history_size=args.history_size,
        kalman_q=args.kalman_q,
        kalman_r=args.kalman_r
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output
    )


if __name__ == "__main__":
    main() 