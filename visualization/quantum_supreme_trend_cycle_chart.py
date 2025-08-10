#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
量子最強トレンド・サイクル検出器のチャート表示
"""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mplfinance as mpf
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime

# データ取得のための依存関係
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource

# 量子最強トレンド・サイクル検出器
from indicators.trend_filter.quantum_supreme_trend_cycle_detector import QuantumSupremeTrendCycleDetector


class QuantumSupremeTrendCycleChart:
    """
    量子最強トレンド・サイクル検出器を表示するチャートクラス
    
    - ローソク足と出来高
    - トレンド・サイクルモードの色分け表示
    - 量子振幅・フラクタル次元・エントロピーのオシレーター
    - 売買シグナルのマーカー表示
    - 信頼度とアンサンブル統計
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.quantum_detector = None
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
                            # 量子パラメータ
                            quantum_levels: int = 20,
                            quantum_window: int = 100,
                            planck_constant: float = 1.0,
                            mass: float = 1.0,
                            # ウェーブレット解析パラメータ
                            wavelet_scales: Optional[List[int]] = None,
                            wavelet_type: int = 0,
                            # フラクタル解析パラメータ
                            fractal_window: int = 50,
                            fractal_max_lag: int = 10,
                            # エントロピー解析パラメータ
                            entropy_window: int = 50,
                            entropy_bins: int = 20,
                            tsallis_q: float = 2.0,
                            # アンサンブル学習パラメータ
                            ensemble_learning_rate: float = 0.01,
                            # 位相空間解析パラメータ
                            embedding_dim: int = 3,
                            time_delay: int = 1,
                            # 適応最適化パラメータ
                            adaptive_window: int = 20,
                            adaptive_learning_rate: float = 0.01,
                            # 統合パラメータ
                            use_kalman_filter: bool = False,
                            kalman_filter_type: str = 'unscented',
                            # パーセンタイル分析パラメータ
                            enable_percentile_analysis: bool = True,
                            percentile_lookback_period: int = 100,
                            percentile_low_threshold: float = 0.2,
                            percentile_high_threshold: float = 0.8,
                            # ソースタイプ
                            src_type: str = 'hl2'
                           ) -> None:
        """
        量子最強トレンド・サイクル検出器を計算する
        
        Args:
            quantum_levels: 量子レベル数
            quantum_window: 量子解析ウィンドウ
            planck_constant: プランク定数
            mass: 質量パラメータ
            wavelet_scales: ウェーブレットスケール
            wavelet_type: ウェーブレットタイプ
            fractal_window: フラクタル解析ウィンドウ
            fractal_max_lag: フラクタル解析最大ラグ
            entropy_window: エントロピー解析ウィンドウ
            entropy_bins: エントロピー計算ビン数
            tsallis_q: ツァリスエントロピーパラメータ
            ensemble_learning_rate: アンサンブル学習率
            embedding_dim: 埋め込み次元
            time_delay: 時間遅延
            adaptive_window: 適応ウィンドウ
            adaptive_learning_rate: 適応学習率
            use_kalman_filter: カルマンフィルター使用
            kalman_filter_type: カルマンフィルタータイプ
            enable_percentile_analysis: パーセンタイル分析有効
            percentile_lookback_period: パーセンタイル分析期間
            percentile_low_threshold: パーセンタイル低閾値
            percentile_high_threshold: パーセンタイル高閾値
            src_type: 価格ソースタイプ
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\n量子最強トレンド・サイクル検出器を計算中...")
        
        # デフォルト値の設定
        if wavelet_scales is None:
            wavelet_scales = [2, 4, 8, 16, 32, 64]
        
        # 量子最強検出器を初期化
        self.quantum_detector = QuantumSupremeTrendCycleDetector(
            src_type=src_type,
            quantum_levels=quantum_levels,
            quantum_window=quantum_window,
            planck_constant=planck_constant,
            mass=mass,
            wavelet_scales=wavelet_scales,
            wavelet_type=wavelet_type,
            fractal_window=fractal_window,
            fractal_max_lag=fractal_max_lag,
            entropy_window=entropy_window,
            entropy_bins=entropy_bins,
            tsallis_q=tsallis_q,
            ensemble_learning_rate=ensemble_learning_rate,
            embedding_dim=embedding_dim,
            time_delay=time_delay,
            adaptive_window=adaptive_window,
            adaptive_learning_rate=adaptive_learning_rate,
            use_kalman_filter=use_kalman_filter,
            kalman_filter_type=kalman_filter_type,
            enable_percentile_analysis=enable_percentile_analysis,
            percentile_lookback_period=percentile_lookback_period,
            percentile_low_threshold=percentile_low_threshold,
            percentile_high_threshold=percentile_high_threshold
        )
        
        # 計算実行
        print("計算を実行します...")
        start_time = datetime.now()
        self.result = self.quantum_detector.calculate(self.data)
        end_time = datetime.now()
        
        calculation_time = (end_time - start_time).total_seconds()
        
        # 結果の統計表示
        print(f"計算完了: {calculation_time:.2f}秒")
        print(f"品質スコア: {self.result.quality_score:.4f}")
        print(f"アルゴリズムバージョン: {self.result.algorithm_version}")
        
        # 統計情報
        trend_ratio = np.mean(self.result.trend_mode)
        cycle_ratio = np.mean(self.result.cycle_mode)
        avg_confidence = np.mean(self.result.confidence)
        
        print(f"\n=== 検出統計 ===")
        print(f"データポイント数: {len(self.result.trend_mode)}")
        print(f"トレンドモード比率: {trend_ratio:.2%}")
        print(f"サイクルモード比率: {cycle_ratio:.2%}")
        print(f"平均信頼度: {avg_confidence:.4f}")
        
        # 信号統計
        buy_signals = np.sum(self.result.signal > 0)
        sell_signals = np.sum(self.result.signal < 0)
        neutral_signals = np.sum(self.result.signal == 0)
        
        print(f"\n=== 信号統計 ===")
        print(f"買いシグナル: {buy_signals}")
        print(f"売りシグナル: {sell_signals}")
        print(f"中立シグナル: {neutral_signals}")
        
        # 高度統計
        fractal_dim_valid = self.result.fractal_dimension[~np.isnan(self.result.fractal_dimension)]
        hurst_valid = self.result.hurst_exponent[~np.isnan(self.result.hurst_exponent)]
        entropy_valid = self.result.shannon_entropy[~np.isnan(self.result.shannon_entropy)]
        quantum_amp_valid = self.result.quantum_amplitude[~np.isnan(self.result.quantum_amplitude)]
        
        print(f"\n=== 高度統計 ===")
        if len(fractal_dim_valid) > 0:
            print(f"平均フラクタル次元: {np.mean(fractal_dim_valid):.4f}")
        if len(hurst_valid) > 0:
            print(f"平均ハースト指数: {np.mean(hurst_valid):.4f}")
        if len(entropy_valid) > 0:
            print(f"平均シャノンエントロピー: {np.mean(entropy_valid):.4f}")
        if len(quantum_amp_valid) > 0:
            print(f"平均量子振幅: {np.mean(quantum_amp_valid):.4f}")
        
        print("量子最強検出器計算完了")
            
    def plot(self, 
            title: str = "量子最強トレンド・サイクル検出器", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (16, 14),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートと量子最強検出器を描画する
        
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
            
        # 量子検出器の結果を取得して対応するインデックスに合わせる
        print("量子検出器データを準備中...")
        
        # 結果をDataFrameとして整理
        results_df = pd.DataFrame(
            index=self.data.index,
            data={
                'trend_mode': self.result.trend_mode,
                'cycle_mode': self.result.cycle_mode,
                'signal': self.result.signal,
                'confidence': self.result.confidence,
                'quantum_amplitude': self.result.quantum_amplitude,
                'quantum_phase': self.result.quantum_phase,
                'fractal_dimension': self.result.fractal_dimension,
                'hurst_exponent': self.result.hurst_exponent,
                'shannon_entropy': self.result.shannon_entropy,
                'ensemble_trend': self.result.ensemble_trend,
                'ensemble_cycle': self.result.ensemble_cycle,
                'ensemble_confidence': self.result.ensemble_confidence
            }
        )
        
        # 絞り込み後のデータに結果を結合
        df = df.join(results_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        
        # トレンド・サイクルモードに基づく色分け
        df['trend_signal_up'] = np.where((df['trend_mode'] == 1) & (df['signal'] > 0), df['signal'], np.nan)
        df['trend_signal_down'] = np.where((df['trend_mode'] == 1) & (df['signal'] < 0), df['signal'], np.nan)
        df['cycle_signal_up'] = np.where((df['cycle_mode'] == 1) & (df['signal'] > 0), df['signal'], np.nan)
        df['cycle_signal_down'] = np.where((df['cycle_mode'] == 1) & (df['signal'] < 0), df['signal'], np.nan)
        
        # 背景色用の価格レベル設定
        df['trend_background'] = np.where(df['trend_mode'] == 1, df['high'] * 1.001, np.nan)
        df['cycle_background'] = np.where(df['cycle_mode'] == 1, df['low'] * 0.999, np.nan)
        
        # 信頼度による透明度調整
        confidence_normalized = (df['confidence'] - df['confidence'].min()) / (df['confidence'].max() - df['confidence'].min())
        
        # mplfinanceでプロット用の設定
        main_plots = []
        
        # トレンドモード背景（薄い緑）
        main_plots.append(mpf.make_addplot(df['trend_background'], type='line', color='lightgreen', 
                                          alpha=0.3, width=0.5, label='Trend Mode'))
        
        # サイクルモード背景（薄い青）
        main_plots.append(mpf.make_addplot(df['cycle_background'], type='line', color='lightblue', 
                                          alpha=0.3, width=0.5, label='Cycle Mode'))
        
        # 売買シグナルのプロット
        # トレンドモード信号
        trend_buy_signals = np.where(df['trend_signal_up'] > 0, df['low'] * 0.995, np.nan)
        trend_sell_signals = np.where(df['trend_signal_down'] < 0, df['high'] * 1.005, np.nan)
        
        # サイクルモード信号
        cycle_buy_signals = np.where(df['cycle_signal_up'] > 0, df['low'] * 0.995, np.nan)
        cycle_sell_signals = np.where(df['cycle_signal_down'] < 0, df['high'] * 1.005, np.nan)
        
        # 信号マーカー
        main_plots.append(mpf.make_addplot(trend_buy_signals, type='scatter', markersize=50, 
                                          marker='^', color='darkgreen', alpha=0.8, label='Trend Buy'))
        main_plots.append(mpf.make_addplot(trend_sell_signals, type='scatter', markersize=50, 
                                          marker='v', color='darkred', alpha=0.8, label='Trend Sell'))
        main_plots.append(mpf.make_addplot(cycle_buy_signals, type='scatter', markersize=30, 
                                          marker='o', color='blue', alpha=0.6, label='Cycle Buy'))
        main_plots.append(mpf.make_addplot(cycle_sell_signals, type='scatter', markersize=30, 
                                          marker='o', color='purple', alpha=0.6, label='Cycle Sell'))
        
        # オシレーターパネルの設定
        # 量子振幅（正規化）
        quantum_amp_norm = (df['quantum_amplitude'] - df['quantum_amplitude'].min()) / (df['quantum_amplitude'].max() - df['quantum_amplitude'].min())
        quantum_panel = mpf.make_addplot(quantum_amp_norm, panel=1, color='purple', width=1.5, 
                                        ylabel='Quantum Amplitude', label='Quantum')
        
        # フラクタル次元
        fractal_panel = mpf.make_addplot(df['fractal_dimension'], panel=2, color='orange', width=1.5, 
                                        ylabel='Fractal Dimension', label='Fractal')
        
        # エントロピー（正規化）
        entropy_norm = (df['shannon_entropy'] - df['shannon_entropy'].min()) / (df['shannon_entropy'].max() - df['shannon_entropy'].min())
        entropy_panel = mpf.make_addplot(entropy_norm, panel=3, color='red', width=1.5, 
                                        ylabel='Shannon Entropy', label='Entropy')
        
        # 信頼度
        confidence_panel = mpf.make_addplot(df['confidence'], panel=4, color='green', width=1.5, 
                                           ylabel='Confidence', label='Confidence')
        
        # トレンド・サイクル比率
        trend_ratio_panel = mpf.make_addplot(df['trend_mode'], panel=5, color='blue', width=2, 
                                            ylabel='Trend/Cycle', label='Mode')
        
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
            kwargs['panel_ratios'] = (4, 1, 1, 1, 1, 1, 1)  # メイン:出来高:量子:フラクタル:エントロピー:信頼度:モード
            # 出来高を表示する場合は、オシレーターのパネル番号を+1する
            quantum_panel = mpf.make_addplot(quantum_amp_norm, panel=2, color='purple', width=1.5, 
                                            ylabel='Quantum Amplitude', label='Quantum')
            fractal_panel = mpf.make_addplot(df['fractal_dimension'], panel=3, color='orange', width=1.5, 
                                            ylabel='Fractal Dimension', label='Fractal')
            entropy_panel = mpf.make_addplot(entropy_norm, panel=4, color='red', width=1.5, 
                                            ylabel='Shannon Entropy', label='Entropy')
            confidence_panel = mpf.make_addplot(df['confidence'], panel=5, color='green', width=1.5, 
                                               ylabel='Confidence', label='Confidence')
            trend_ratio_panel = mpf.make_addplot(df['trend_mode'], panel=6, color='blue', width=2, 
                                                ylabel='Trend/Cycle', label='Mode')
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (4, 1, 1, 1, 1, 1)  # メイン:量子:フラクタル:エントロピー:信頼度:モード
        
        # すべてのプロットを結合
        all_plots = main_plots + [quantum_panel, fractal_panel, entropy_panel, confidence_panel, trend_ratio_panel]
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加
        axes[0].legend(['Trend Mode', 'Cycle Mode', 'Trend Buy', 'Trend Sell', 'Cycle Buy', 'Cycle Sell'], 
                      loc='upper left', fontsize=8)
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        panel_offset = 1 if show_volume else 0
        
        # 量子振幅パネル
        axes[1 + panel_offset].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
        axes[1 + panel_offset].axhline(y=0.8, color='red', linestyle='--', alpha=0.3)
        
        # フラクタル次元パネル
        axes[2 + panel_offset].axhline(y=1.5, color='black', linestyle='--', alpha=0.5)
        axes[2 + panel_offset].axhline(y=2.0, color='red', linestyle='--', alpha=0.3)
        
        # エントロピーパネル
        axes[3 + panel_offset].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
        
        # 信頼度パネル
        axes[4 + panel_offset].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
        axes[4 + panel_offset].axhline(y=0.7, color='green', linestyle='--', alpha=0.3)
        
        # モードパネル
        axes[5 + panel_offset].axhline(y=0.5, color='black', linestyle='-', alpha=0.5)
        axes[5 + panel_offset].axhline(y=1, color='green', linestyle='--', alpha=0.5)
        axes[5 + panel_offset].axhline(y=0, color='blue', linestyle='--', alpha=0.5)
        
        # 統計情報の表示
        print(f"\n=== 描画統計 ===")
        valid_data = df.dropna(subset=['trend_mode', 'cycle_mode'])
        total_points = len(valid_data)
        trend_points = len(valid_data[valid_data['trend_mode'] == 1])
        cycle_points = len(valid_data[valid_data['cycle_mode'] == 1])
        
        print(f"総データ点数: {total_points}")
        print(f"トレンドモード: {trend_points} ({trend_points/total_points*100:.1f}%)")
        print(f"サイクルモード: {cycle_points} ({cycle_points/total_points*100:.1f}%)")
        print(f"平均信頼度: {df['confidence'].mean():.4f}")
        print(f"品質スコア: {self.result.quality_score:.4f}")
        
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
    parser = argparse.ArgumentParser(description='量子最強トレンド・サイクル検出器の描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--quantum-levels', type=int, default=15, help='量子レベル数')
    parser.add_argument('--quantum-window', type=int, default=80, help='量子解析ウィンドウ')
    parser.add_argument('--fractal-window', type=int, default=40, help='フラクタル解析ウィンドウ')
    parser.add_argument('--entropy-window', type=int, default=40, help='エントロピー解析ウィンドウ')
    parser.add_argument('--use-kalman', action='store_true', help='カルマンフィルター使用')
    parser.add_argument('--src-type', type=str, default='hl2', help='価格ソースタイプ')
    args = parser.parse_args()
    
    # チャートを作成
    chart = QuantumSupremeTrendCycleChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        quantum_levels=args.quantum_levels,
        quantum_window=args.quantum_window,
        fractal_window=args.fractal_window,
        entropy_window=args.entropy_window,
        use_kalman_filter=args.use_kalman,
        src_type=args.src_type,
        # 実用的なパラメータ設定
        wavelet_scales=[2, 4, 8, 16, 32],
        ensemble_learning_rate=0.02,
        adaptive_window=20
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output
    )


if __name__ == "__main__":
    main()