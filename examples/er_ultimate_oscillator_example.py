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
import warnings
warnings.filterwarnings('ignore')

# データ取得のための依存関係
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource

# インジケーター
from indicators.efficiency_ratio_v2 import ER_V2
from indicators.ultimate_oscillator import UltimateOscillator


class ERUltimateOscillatorChart:
    """
    ER_V2 + Ultimate Oscillator組み合わせチャートクラス
    
    - ローソク足チャート
    - ER_V2（効率比率V2）
    - Ultimate Oscillator（ER_V2をソースとして使用）
    - 組み合わせシグナル
    - シグナル強度
    - トレンド信号
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.er_v2 = None
        self.ultimate_oscillator = None
        self.results = None
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
                           # ER_V2 パラメータ
                           er_period: int = 21,
                           er_smoother_period: float = 5.0,
                           er_src_type: str = 'ukf_hlc3',
                           er_use_dynamic_period: bool = False,
                           # Ultimate Oscillator パラメータ
                           uo_edge: int = 30,
                           uo_width: int = 2,
                           uo_rms_period: int = 100,
                           # シグナル閾値
                           er_high_threshold: float = 0.618,
                           er_low_threshold: float = 0.382,
                           uo_high_threshold: float = 1.0,
                           uo_low_threshold: float = -1.0,
                           uo_neutral_threshold: float = 0.5
                           ) -> None:
        """
        ER_V2 + Ultimate Oscillatorの組み合わせ指標を計算する
        
        Args:
            er_period: ER_V2 期間
            er_smoother_period: ER_V2 平滑化期間
            er_src_type: ER_V2 ソースタイプ
            er_use_dynamic_period: ER_V2 動的期間使用
            uo_edge: Ultimate Oscillator エッジ期間
            uo_width: Ultimate Oscillator 幅乗数
            uo_rms_period: Ultimate Oscillator RMS期間
            er_high_threshold: ER高閾値
            er_low_threshold: ER低閾値
            uo_high_threshold: UO高閾値
            uo_low_threshold: UO低閾値
            uo_neutral_threshold: UO中立閾値
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\nER_V2 + Ultimate Oscillatorを計算中...")
        
        try:
            # Step 1: ER_V2の計算
            print("   ER_V2を計算中...")
            self.er_v2 = ER_V2(
                period=er_period,
                src_type=er_src_type,
                use_ultimate_smoother=True,
                smoother_period=er_smoother_period,
                use_dynamic_period=er_use_dynamic_period,
                detector_type='absolute_ultimate'
            )
            
            er_result = self.er_v2.calculate(self.data)
            print(f"   ER_V2計算完了: {len(er_result.smoothed_values)} 値")
            
            # ER_V2の値を検証
            valid_er = er_result.smoothed_values[~np.isnan(er_result.smoothed_values)]
            if len(valid_er) == 0:
                print("   警告: ER_V2の値がすべてNaNです")
                raise ValueError("ER_V2 calculation failed")
            
            # Step 2: ER_V2の平滑化値をUltimate Oscillatorのソースとして使用
            print("   Ultimate Oscillatorを計算中...")
            
            # ER_V2の平滑化値をクリーンアップ
            er_smoothed_clean = pd.Series(er_result.smoothed_values).bfill().ffill().values
            er_df = pd.DataFrame({'close': er_smoothed_clean}, index=self.data.index)
            
            # Ultimate Oscillatorを計算
            self.ultimate_oscillator = UltimateOscillator(
                edge=uo_edge,
                width=uo_width,
                rms_period=uo_rms_period,
                src_type='close'
            )
            
            uo_result = self.ultimate_oscillator.calculate(er_df)
            print(f"   Ultimate Oscillator計算完了: {len(uo_result.values)} 値")
            
            # Step 3: 組み合わせシグナルの生成
            print("   組み合わせシグナルを生成中...")
            combined_signals = self._analyze_combined_signals(
                er_result, uo_result,
                er_high_threshold, er_low_threshold,
                uo_high_threshold, uo_low_threshold, uo_neutral_threshold
            )
            
            # Step 4: シグナル強度の計算
            signal_strength = self._calculate_signal_strength(er_result, uo_result)
            
            # Step 5: パフォーマンス評価
            performance = self._evaluate_signals(combined_signals, signal_strength)
            
            # 結果を保存
            self.results = {
                'er_values': er_result.values,
                'er_smoothed': er_smoothed_clean,
                'er_trend': er_result.current_trend,
                'er_trend_signals': er_result.trend_signals,
                'er_dynamic_periods': er_result.dynamic_periods,
                'uo_values': uo_result.values,
                'uo_signals': uo_result.signals,
                'uo_rms': uo_result.rms_values,
                'uo_hp_short': uo_result.highpass_short,
                'uo_hp_long': uo_result.highpass_long,
                'combined_signals': combined_signals,
                'combined_strength': signal_strength,
                'performance': performance
            }
            
            print("   計算完了")
            
        except Exception as e:
            print(f"   エラー: {e}")
            raise ValueError(f"指標計算に失敗しました: {e}")
    
    def _analyze_combined_signals(self, er_result, uo_result,
                                 er_high_threshold, er_low_threshold,
                                 uo_high_threshold, uo_low_threshold, uo_neutral_threshold):
        """組み合わせシグナルの分析"""
        length = len(er_result.smoothed_values)
        signals = np.zeros(length, dtype=int)
        
        er_smoothed = er_result.smoothed_values
        uo_values = uo_result.values
        er_trend_signals = er_result.trend_signals
        
        for i in range(length):
            er_val = er_smoothed[i]
            uo_val = uo_values[i]
            er_trend = er_trend_signals[i]
            
            if np.isnan(er_val) or np.isnan(uo_val):
                signals[i] = 0
                continue
            
            # 強い買いシグナル
            if (er_val > er_high_threshold and 
                uo_val > uo_high_threshold and 
                er_trend == 1):
                signals[i] = 2  # 強い買い
            
            # 強い売りシグナル
            elif (er_val > er_high_threshold and 
                  uo_val < uo_low_threshold and 
                  er_trend == -1):
                signals[i] = -2  # 強い売り
            
            # 弱い買いシグナル
            elif (er_val > er_high_threshold and 
                  uo_val > uo_neutral_threshold and 
                  er_trend == 1):
                signals[i] = 1  # 弱い買い
            
            # 弱い売りシグナル
            elif (er_val > er_high_threshold and 
                  uo_val < -uo_neutral_threshold and 
                  er_trend == -1):
                signals[i] = -1  # 弱い売り
            
            # レンジ/ホールド
            elif er_val < er_low_threshold and abs(uo_val) < uo_neutral_threshold:
                signals[i] = 0  # ホールド/レンジ
            
            else:
                signals[i] = 0  # ニュートラル
        
        return signals
    
    def _calculate_signal_strength(self, er_result, uo_result):
        """シグナル強度の計算"""
        length = len(er_result.smoothed_values)
        strength = np.zeros(length)
        
        er_smoothed = er_result.smoothed_values
        uo_values = uo_result.values
        
        for i in range(length):
            if np.isnan(er_smoothed[i]) or np.isnan(uo_values[i]):
                strength[i] = 0
                continue
            
            # ERを0-100スケールに正規化
            er_strength = er_smoothed[i] * 100
            
            # UOを0-100スケールに正規化（絶対値、3σでキャップ）
            uo_strength = min(abs(uo_values[i]) * 20, 100)
            
            # 強度の結合（重み付け平均）
            strength[i] = (er_strength * 0.6 + uo_strength * 0.4)
        
        return strength
    
    def _evaluate_signals(self, signals, signal_strength):
        """シグナルパフォーマンスの評価"""
        prices = self.data['close'].values
        returns = np.diff(prices) / prices[:-1]
        
        # シグナルをリターンに合わせる
        signals_aligned = signals[1:]
        strength_aligned = signal_strength[1:]
        
        # シグナルベースのリターンを計算
        signal_returns = []
        signal_types = []
        
        for i in range(len(signals_aligned)):
            if signals_aligned[i] != 0:
                signal_return = returns[i] * np.sign(signals_aligned[i])
                signal_returns.append(signal_return)
                signal_types.append(signals_aligned[i])
        
        if len(signal_returns) == 0:
            return {
                'total_signals': 0, 
                'avg_return': 0, 
                'win_rate': 0, 
                'sharpe_ratio': 0,
                'strong_signals': 0,
                'weak_signals': 0,
                'strong_avg_return': 0,
                'weak_avg_return': 0
            }
        
        signal_returns = np.array(signal_returns)
        signal_types = np.array(signal_types)
        
        # メトリクスの計算
        total_signals = len(signal_returns)
        avg_return = np.mean(signal_returns)
        win_rate = np.sum(signal_returns > 0) / total_signals if total_signals > 0 else 0
        sharpe_ratio = np.mean(signal_returns) / np.std(signal_returns) if np.std(signal_returns) > 0 else 0
        
        # シグナルタイプ分析
        strong_signals = signal_returns[np.abs(signal_types) == 2]
        weak_signals = signal_returns[np.abs(signal_types) == 1]
        
        return {
            'total_signals': total_signals,
            'avg_return': avg_return,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'strong_signals': len(strong_signals),
            'weak_signals': len(weak_signals),
            'strong_avg_return': np.mean(strong_signals) if len(strong_signals) > 0 else 0,
            'weak_avg_return': np.mean(weak_signals) if len(weak_signals) > 0 else 0
        }
    
    def plot(self, 
            title: str = "ER_V2 + Ultimate Oscillator Combined Analysis", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (16, 20),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ER_V2 + Ultimate Oscillatorの組み合わせチャートを描画する
        
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
            
        if self.results is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # インジケーターデータの取得
        print("インジケーターデータを取得中...")
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'er_values': self.results['er_values'],
                'er_smoothed': self.results['er_smoothed'],
                'er_trend_signals': self.results['er_trend_signals'],
                'uo_values': self.results['uo_values'],
                'combined_signals': self.results['combined_signals'],
                'combined_strength': self.results['combined_strength']
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        
        # NaN値を処理
        df['er_values'] = df['er_values'].fillna(0.5)
        df['er_smoothed'] = df['er_smoothed'].fillna(0.5)
        df['uo_values'] = df['uo_values'].fillna(0.0)
        df['combined_strength'] = df['combined_strength'].fillna(50.0)
        
        # mplfinanceでプロット用の設定
        additional_plots = []
        
        # パネル番号を出来高の有無で調整
        if show_volume:
            # 出来高がある場合、パネル1は自動的に出来高用に使用される
            er_panel = 2
            uo_panel = 3  
            signal_panel = 4
            strength_panel = 5
        else:
            # 出来高がない場合、パネル1から使用可能
            er_panel = 1
            uo_panel = 2
            signal_panel = 3
            strength_panel = 4
        
        # 1. ER_V2 パネル
        er_raw_plot = mpf.make_addplot(df['er_values'], panel=er_panel, color='orange', 
                                      width=1, alpha=0.7, ylabel='ER_V2')
        er_smooth_plot = mpf.make_addplot(df['er_smoothed'], panel=er_panel, color='red', 
                                         width=2, label='ER_V2 Smoothed')
        additional_plots.extend([er_raw_plot, er_smooth_plot])
        
        # 2. Ultimate Oscillator パネル
        uo_plot = mpf.make_addplot(df['uo_values'], panel=uo_panel, color='purple', 
                                  width=2, ylabel='Ultimate Oscillator')
        additional_plots.append(uo_plot)
        
        # 3. 組み合わせシグナル パネル
        signal_plot = mpf.make_addplot(df['combined_signals'], panel=signal_panel, color='navy', 
                                      width=2, ylabel='Combined Signals', type='line')
        additional_plots.append(signal_plot)
        
        # 4. シグナル強度 パネル（メイン）
        strength_plot = mpf.make_addplot(df['combined_strength'], panel=strength_panel, color='darkorange', 
                                        width=2, ylabel='Signal Strength (%)')
        additional_plots.append(strength_plot)
        
        # 5. ER トレンドシグナル パネル（シグナル強度パネルのsecondary_y）
        trend_plot = mpf.make_addplot(df['er_trend_signals'], panel=strength_panel, color='green', 
                                     width=1, secondary_y=True, ylabel='ER Trend')
        additional_plots.append(trend_plot)
        
        # mplfinanceの設定
        kwargs = dict(
            type='candle',
            figsize=figsize,
            title=title,
            style=style,
            datetime_format='%Y-%m-%d',
            xrotation=45,
            returnfig=True,
            addplot=additional_plots
        )
        
        # 出来高とパネル比率の設定
        if show_volume:
            kwargs['volume'] = True
            # メイン:出来高:ER:UO:シグナル:強度 = 6パネル
            kwargs['panel_ratios'] = (4, 1, 1, 1, 1, 1)
        else:
            kwargs['volume'] = False
            # メイン:ER:UO:シグナル:強度 = 5パネル
            kwargs['panel_ratios'] = (4, 1, 1, 1, 1)
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        if show_volume:
            # 出来高がある場合のパネルインデックス
            er_panel_idx = 2
            uo_panel_idx = 3
            signal_panel_idx = 4
            strength_panel_idx = 5
        else:
            # 出来高がない場合のパネルインデックス
            er_panel_idx = 1
            uo_panel_idx = 2
            signal_panel_idx = 3
            strength_panel_idx = 4
        
        # ER_V2 パネル
        er_panel = axes[er_panel_idx]
        er_panel.axhline(y=0.618, color='green', linestyle='--', alpha=0.8, label='効率閾値 (0.618)')
        er_panel.axhline(y=0.382, color='brown', linestyle='--', alpha=0.8, label='非効率閾値 (0.382)')
        er_panel.set_ylim(0, 1)
        
        # Ultimate Oscillator パネル
        uo_panel = axes[uo_panel_idx]
        uo_panel.axhline(y=1.0, color='green', linestyle='--', alpha=0.8, label='買い閾値 (+1.0)')
        uo_panel.axhline(y=-1.0, color='red', linestyle='--', alpha=0.8, label='売り閾値 (-1.0)')
        uo_panel.axhline(y=0.5, color='lightgreen', linestyle=':', alpha=0.8, label='弱い買い (+0.5)')
        uo_panel.axhline(y=-0.5, color='lightcoral', linestyle=':', alpha=0.8, label='弱い売り (-0.5)')
        uo_panel.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 組み合わせシグナル パネル
        signal_panel = axes[signal_panel_idx]
        signal_panel.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        signal_panel.axhline(y=2, color='darkgreen', linestyle='--', alpha=0.8, label='強い買い')
        signal_panel.axhline(y=-2, color='darkred', linestyle='--', alpha=0.8, label='強い売り')
        signal_panel.axhline(y=1, color='green', linestyle=':', alpha=0.8, label='弱い買い')
        signal_panel.axhline(y=-1, color='red', linestyle=':', alpha=0.8, label='弱い売り')
        signal_panel.set_ylim(-2.5, 2.5)
        
        # シグナル強度 パネル - ERトレンドシグナルも統合
        strength_panel = axes[strength_panel_idx]
        strength_panel.axhline(y=80, color='red', linestyle='--', alpha=0.8, label='高強度 (80)')
        strength_panel.axhline(y=60, color='orange', linestyle='--', alpha=0.8, label='中強度 (60)')
        strength_panel.axhline(y=40, color='yellow', linestyle='--', alpha=0.8, label='低強度 (40)')
        strength_panel.set_ylim(0, 100)
        
        # ERトレンドシグナル用のsecondary_yを設定
        try:
            strength_panel_twin = strength_panel.twinx()
            strength_panel_twin.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            strength_panel_twin.axhline(y=1, color='green', linestyle='--', alpha=0.8, label='上昇トレンド')
            strength_panel_twin.axhline(y=-1, color='red', linestyle='--', alpha=0.8, label='下降トレンド')
            strength_panel_twin.set_ylim(-1.5, 1.5)
            strength_panel_twin.set_ylabel('ER Trend', color='green')
        except:
            pass  # twinxが失敗した場合は無視
        
        # 統計情報の表示
        self._print_statistics(df)
        
        # 保存または表示
        if savefig:
            plt.savefig(savefig, dpi=300, bbox_inches='tight')
            print(f"チャートを保存しました: {savefig}")
        else:
            plt.tight_layout()
            plt.show()
    
    def _print_statistics(self, df):
        """統計情報を表示"""
        print(f"\n=== 統計情報 ===")
        
        # パフォーマンス情報
        if 'performance' in self.results:
            perf = self.results['performance']
            print(f"総シグナル数: {perf['total_signals']}")
            print(f"平均リターン: {perf['avg_return']:.4f}")
            print(f"勝率: {perf['win_rate']:.2%}")
            print(f"シャープレシオ: {perf['sharpe_ratio']:.4f}")
            print(f"強いシグナル数: {perf['strong_signals']}")
            print(f"弱いシグナル数: {perf['weak_signals']}")
        
        # シグナル分布
        unique_signals, signal_counts = np.unique(self.results['combined_signals'], return_counts=True)
        signal_names = {-2: '強い売り', -1: '弱い売り', 0: 'ホールド', 1: '弱い買い', 2: '強い買い'}
        
        print(f"\n=== シグナル分布 ===")
        for signal, count in zip(unique_signals, signal_counts):
            percentage = count / len(self.results['combined_signals']) * 100
            print(f"{signal_names.get(signal, f'シグナル {signal}')}: {count} ({percentage:.1f}%)")
        
        # ER統計
        er_valid = self.results['er_smoothed'][~np.isnan(self.results['er_smoothed'])]
        if len(er_valid) > 0:
            print(f"\n=== ER_V2統計 ===")
            print(f"平均ER: {np.mean(er_valid):.3f}")
            print(f"ER範囲: {np.min(er_valid):.3f} - {np.max(er_valid):.3f}")
            
        # UO統計
        uo_valid = self.results['uo_values'][~np.isnan(self.results['uo_values'])]
        if len(uo_valid) > 0:
            print(f"\n=== Ultimate Oscillator統計 ===")
            print(f"平均UO: {np.mean(uo_valid):.3f}")
            print(f"UO範囲: {np.min(uo_valid):.3f} - {np.max(uo_valid):.3f}")


def main():
    """メイン関数"""
    # コマンドライン引数を処理
    import argparse
    parser = argparse.ArgumentParser(description='ER_V2 + Ultimate Oscillatorの組み合わせ分析')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--er-period', type=int, default=5, help='ER_V2期間')
    parser.add_argument('--er-smoother', type=float, default=10.0, help='ER_V2平滑化期間')
    parser.add_argument('--uo-edge', type=int, default=30, help='Ultimate Oscillatorエッジ期間')
    parser.add_argument('--uo-width', type=int, default=2, help='Ultimate Oscillator幅乗数')
    parser.add_argument('--uo-rms', type=int, default=100, help='Ultimate Oscillator RMS期間')
    args = parser.parse_args()
    
    # チャートを作成
    chart = ERUltimateOscillatorChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        er_period=args.er_period,
        er_smoother_period=args.er_smoother,
        uo_edge=args.uo_edge,
        uo_width=args.uo_width,
        uo_rms_period=args.uo_rms
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output
    )


if __name__ == "__main__":
    main() 