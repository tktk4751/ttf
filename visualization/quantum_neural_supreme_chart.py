#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# プロジェクトルートをパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# データ取得のための依存関係
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource

# QuantumNeuralSupremeインジケーター
from quantum_neural_supreme_trend_range_detector import QuantumNeuralSupremeTrendRangeDetector


class QuantumNeuralSupremeChart:
    """
    🌟 量子ニューラル至高検出器 可視化チャートクラス 🌟
    
    - ローソク足と出来高
    - 量子ニューラル至高検出器の各種指標
    - トレンド/レンジ判別結果
    - 信頼度とトレンド強度
    - 効率比とボラティリティ体制
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
        設定ファイルからデータを読み込む（z_adaptive_trend_chart.pyと同じ方式）
        
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
                            quantum_window: int = 50,
                            wavelet_scales: List[float] = None,
                            fractal_window: int = 30,
                            chaos_window: int = 50,
                            entropy_window: int = 30) -> None:
        """
        量子ニューラル至高検出器を計算する
        
        Args:
            quantum_window: 量子解析窓サイズ
            wavelet_scales: ウェーブレットスケール
            fractal_window: フラクタル解析窓サイズ
            chaos_window: カオス解析窓サイズ
            entropy_window: エントロピー解析窓サイズ
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\n🌟 次世代革命的検出器を計算中...")
        
        # 次世代革命的検出器を初期化
        self.quantum_detector = QuantumNeuralSupremeTrendRangeDetector(
            quantum_window=quantum_window,
            wavelet_scales=wavelet_scales,
            fractal_window=fractal_window,
            chaos_window=chaos_window,
            entropy_window=entropy_window
        )
        
        # 計算実行
        print("計算を実行します...")
        self.result = self.quantum_detector.calculate(self.data)
        
        # 結果の確認
        signals = self.result['signals']
        values = self.result['values']
        confidence = self.result['confidence_levels']
        trend_strength = self.result['trend_strengths']
        
        print(f"計算完了:")
        print(f"  シグナル範囲: {np.nanmin(signals):.3f} - {np.nanmax(signals):.3f}")
        print(f"  値範囲: {np.nanmin(values):.3f} - {np.nanmax(values):.3f}")
        print(f"  信頼度範囲: {np.nanmin(confidence):.3f} - {np.nanmax(confidence):.3f}")
        print(f"  トレンド強度範囲: {np.nanmin(trend_strength):.3f} - {np.nanmax(trend_strength):.3f}")
        
        # シグナル統計
        unique_signals, counts = np.unique(signals[~np.isnan(signals)], return_counts=True)
        print(f"  シグナル分布:")
        for sig, count in zip(unique_signals, counts):
            print(f"    {sig}: {count}回 ({count/len(signals)*100:.1f}%)")
        
        print("🌟 量子ニューラル至高検出器 計算完了")
            
    def plot(self, 
            title: str = "🌟 量子ニューラル至高検出器 チャート 🌟", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (16, 14),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートと量子ニューラル至高検出器を描画する
        
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
            
        # 量子ニューラル至高検出器の結果を取得
        print("量子検出器データを取得中...")
        
        # 対応するインデックスでインジケーターデータを抽出
        start_idx = 0
        end_idx = len(self.data)
        
        if start_date:
            start_idx = np.where(self.data.index >= pd.to_datetime(start_date))[0]
            start_idx = start_idx[0] if len(start_idx) > 0 else 0
        if end_date:
            end_idx = np.where(self.data.index <= pd.to_datetime(end_date))[0]
            end_idx = end_idx[-1] + 1 if len(end_idx) > 0 else len(self.data)
        
        # データの正規化と問題修正
        def safe_normalize(data, min_val=0, max_val=1):
            """安全な正規化関数"""
            data = np.array(data[start_idx:end_idx])
            data_clean = data[~np.isnan(data)]
            if len(data_clean) == 0:
                return np.full(len(data), 0.5)
            
            data_min = np.min(data_clean)
            data_max = np.max(data_clean)
            
            if data_max == data_min:
                return np.full(len(data), 0.5)
            
            normalized = (data - data_min) / (data_max - data_min)
            normalized = normalized * (max_val - min_val) + min_val
            return normalized
        
        # 正規化されたデータを作成
        qns_values_norm = safe_normalize(self.result['values'])
        qns_confidence_norm = safe_normalize(self.result['confidence_levels'])
        qns_trend_strength_norm = safe_normalize(self.result['trend_strengths'])
        qns_er_short_norm = safe_normalize(self.result['er_short'])
        qns_er_long_norm = safe_normalize(self.result['er_long'])
        qns_vol_price_norm = safe_normalize(self.result['vol_price'])
        qns_vol_return_norm = safe_normalize(self.result['vol_return'])
        qns_regime_scores = self.result['regime_scores'][start_idx:end_idx]
        qns_signals = self.result['signals'][start_idx:end_idx]
        
        # データフレームに追加
        df['qns_signals'] = qns_signals
        df['qns_values'] = qns_values_norm
        df['qns_confidence'] = qns_confidence_norm
        df['qns_trend_strength'] = qns_trend_strength_norm
        df['qns_regime_scores'] = qns_regime_scores
        df['qns_er_short'] = qns_er_short_norm
        df['qns_er_long'] = qns_er_long_norm
        df['qns_vol_price'] = qns_vol_price_norm
        df['qns_vol_return'] = qns_vol_return_norm
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        print(f"正規化後の値範囲: {np.nanmin(qns_values_norm):.3f} - {np.nanmax(qns_values_norm):.3f}")
        print(f"信頼度範囲: {np.nanmin(qns_confidence_norm):.3f} - {np.nanmax(qns_confidence_norm):.3f}")
        
        # シグナルベースの色分け用データ準備
        df['trend_signal'] = np.where(df['qns_signals'] == 1, df['close'], np.nan)
        df['range_signal'] = np.where(df['qns_signals'] == -1, df['close'], np.nan)
        df['neutral_signal'] = np.where(df['qns_signals'] == 0, df['close'], np.nan)
        
        # トレンド強度に基づくバンド
        df['upper_band'] = df['close'] * (1 + df['qns_trend_strength'] * 0.02)
        df['lower_band'] = df['close'] * (1 - df['qns_trend_strength'] * 0.02)
        
        # mplfinanceでプロット用の設定
        main_plots = []
        
        # 1. メインチャート上のプロット（シグナル点とバンド）
        main_plots.append(mpf.make_addplot(df['trend_signal'], type='scatter', markersize=50, 
                                          color='lime', alpha=0.8, label='Trend Signal'))
        main_plots.append(mpf.make_addplot(df['range_signal'], type='scatter', markersize=50, 
                                          color='red', alpha=0.8, label='Range Signal'))
        main_plots.append(mpf.make_addplot(df['neutral_signal'], type='scatter', markersize=8, 
                                          color='gray', alpha=0.3, label='Neutral'))
        
        # トレンド強度バンド（強度が高い場合のみ表示）
        strong_trend_mask = df['qns_trend_strength'] > 0.3
        df['upper_band_strong'] = np.where(strong_trend_mask, df['upper_band'], np.nan)
        df['lower_band_strong'] = np.where(strong_trend_mask, df['lower_band'], np.nan)
        
        main_plots.append(mpf.make_addplot(df['upper_band_strong'], color='cyan', width=1.5, alpha=0.7, label='Upper Band'))
        main_plots.append(mpf.make_addplot(df['lower_band_strong'], color='cyan', width=1.5, alpha=0.7, label='Lower Band'))
        
        # 2. シンプルな3パネル構成に変更
        if show_volume:
            panel_ratios = (4, 1, 1.5, 1.5, 1)  # メイン:出来高:値&信頼度:効率比:体制スコア
            panel_num = 2
        else:
            panel_ratios = (4, 1.5, 1.5, 1)  # メイン:値&信頼度:効率比:体制スコア
            panel_num = 1
        
        # 値と信頼度パネル（同じパネルに両方表示）
        values_panel = mpf.make_addplot(df['qns_values'], panel=panel_num, color='purple', width=2, 
                                       ylabel='Values & Confidence', label='Values')
        confidence_panel = mpf.make_addplot(df['qns_confidence'], panel=panel_num, color='blue', width=1.5, 
                                           secondary_y=False, alpha=0.7, label='Confidence')
        
        # 効率比パネル
        er_short_panel = mpf.make_addplot(df['qns_er_short'], panel=panel_num+1, color='green', width=2, 
                                         ylabel='Efficiency Ratio', label='ER Short')
        er_long_panel = mpf.make_addplot(df['qns_er_long'], panel=panel_num+1, color='red', width=2, 
                                        alpha=0.7, label='ER Long')
        
        # 体制スコアパネル
        regime_panel = mpf.make_addplot(df['qns_regime_scores'], panel=panel_num+2, color='gold', width=3, 
                                       ylabel='Regime Score', label='Regime')
        
        # mplfinanceの設定
        kwargs = dict(
            type='candle',
            figsize=figsize,
            title=title,
            style=style,
            datetime_format='%Y-%m-%d',
            xrotation=45,
            returnfig=True,
            panel_ratios=panel_ratios
        )
        
        # 出来高の設定
        kwargs['volume'] = show_volume
        
        # すべてのプロットを結合
        all_plots = main_plots + [
            values_panel, confidence_panel,
            er_short_panel, er_long_panel,
            regime_panel
        ]
        kwargs['addplot'] = all_plots
        
        # プロット実行
        try:
            fig, axes = mpf.plot(df, **kwargs)
        except Exception as e:
            print(f"描画エラー: {e}")
            print("シンプルな価格チャートのみを表示します...")
            
            # フォールバック: 簡単なmatplotlibプロット
            plt.figure(figsize=figsize)
            plt.subplot(3, 1, 1)
            plt.plot(df.index, df['close'], 'k-', label='Price')
            
            # シグナルをプロット
            trend_dates = df.index[df['qns_signals'] == 1]
            trend_prices = df['close'][df['qns_signals'] == 1]
            range_dates = df.index[df['qns_signals'] == -1]
            range_prices = df['close'][df['qns_signals'] == -1]
            
            plt.scatter(trend_dates, trend_prices, c='lime', s=50, alpha=0.8, label='Trend')
            plt.scatter(range_dates, range_prices, c='red', s=50, alpha=0.8, label='Range')
            plt.title(title)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 値パネル
            plt.subplot(3, 1, 2)
            plt.plot(df.index, df['qns_values'], 'purple', linewidth=2, label='Values')
            plt.plot(df.index, df['qns_confidence'], 'blue', alpha=0.7, label='Confidence')
            plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
            plt.ylabel('Values & Confidence')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 体制スコア
            plt.subplot(3, 1, 3)
            plt.plot(df.index, df['qns_regime_scores'], 'gold', linewidth=3, label='Regime Score')
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            plt.axhline(y=1, color='green', linestyle='--', alpha=0.5)
            plt.axhline(y=-1, color='red', linestyle='--', alpha=0.5)
            plt.ylabel('Regime Score')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if savefig:
                plt.savefig(savefig, dpi=150, bbox_inches='tight')
                print(f"チャートを保存しました: {savefig}")
            else:
                plt.show()
            return
        
        # 凡例の追加
        axes[0].legend(['Trend Signal', 'Range Signal', 'Neutral', 'Upper Band', 'Lower Band'], 
                      loc='upper left')
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        panel_offset = 2 if show_volume else 1
        
        # 値パネル（0-1の参照線）
        axes[panel_offset].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
        axes[panel_offset].axhline(y=0.0, color='black', linestyle='-', alpha=0.3)
        axes[panel_offset].axhline(y=1.0, color='black', linestyle='-', alpha=0.3)
        
        # 効率比パネル
        axes[panel_offset+1].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
        axes[panel_offset+1].axhline(y=0.7, color='green', linestyle='--', alpha=0.3)
        axes[panel_offset+1].axhline(y=0.3, color='red', linestyle='--', alpha=0.3)
        
        # 体制スコアパネル（-1, 0, 1の参照線）
        axes[panel_offset+2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[panel_offset+2].axhline(y=1, color='green', linestyle='--', alpha=0.5)
        axes[panel_offset+2].axhline(y=-1, color='red', linestyle='--', alpha=0.5)
        
        # 統計情報の表示
        print(f"\n=== 🌟 量子ニューラル至高検出器 統計 ===")
        total_points = len(df[~np.isnan(df['qns_signals'])])
        trend_points = len(df[df['qns_signals'] == 1])
        range_points = len(df[df['qns_signals'] == -1])
        neutral_points = len(df[df['qns_signals'] == 0])
        
        print(f"総データ点数: {total_points}")
        print(f"トレンド信号: {trend_points} ({trend_points/total_points*100:.1f}%)")
        print(f"レンジ信号: {range_points} ({range_points/total_points*100:.1f}%)")
        print(f"ニュートラル: {neutral_points} ({neutral_points/total_points*100:.1f}%)")
        
        print(f"値統計 - 平均: {df['qns_values'].mean():.3f}, 標準偏差: {df['qns_values'].std():.3f}")
        print(f"信頼度統計 - 平均: {df['qns_confidence'].mean():.3f}, 標準偏差: {df['qns_confidence'].std():.3f}")
        print(f"トレンド強度統計 - 平均: {df['qns_trend_strength'].mean():.3f}, 標準偏差: {df['qns_trend_strength'].std():.3f}")
        
        # 保存または表示
        if savefig:
            plt.savefig(savefig, dpi=150, bbox_inches='tight')
            print(f"チャートを保存しました: {savefig}")
        else:
            plt.tight_layout()
            plt.show()

    def analyze_performance(self, lookback_days: int = 30) -> Dict[str, float]:
        """
        パフォーマンス分析
        
        Args:
            lookback_days: 分析対象日数
            
        Returns:
            パフォーマンス指標
        """
        if self.data is None or self.result is None:
            return {}
        
        # 最新のデータを取得
        lookback_periods = lookback_days * 6  # 4時間足なら6倍
        recent_data = self.data.tail(lookback_periods)
        
        signals = self.result['signals'][-lookback_periods:]
        values = self.result['values'][-lookback_periods:]
        confidence = self.result['confidence_levels'][-lookback_periods:]
        
        # 基本統計
        signal_changes = np.sum(np.diff(signals) != 0)
        avg_confidence = np.nanmean(confidence)
        avg_values = np.nanmean(values)
        
        # シグナル分布
        unique_signals, counts = np.unique(signals[~np.isnan(signals)], return_counts=True)
        signal_distribution = dict(zip(unique_signals, counts))
        
        return {
            'period_days': lookback_days,
            'total_signals': len(signals),
            'signal_changes': signal_changes,
            'change_rate': signal_changes / len(signals) if len(signals) > 0 else 0,
            'avg_confidence': avg_confidence,
            'avg_values': avg_values,
            'signal_distribution': signal_distribution
        }


def main():
    """メイン関数"""
    # コマンドライン引数を処理
    import argparse
    parser = argparse.ArgumentParser(description='🌟 量子ニューラル至高検出器チャート描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--quantum-window', type=int, default=50, help='量子解析窓サイズ')
    parser.add_argument('--wavelet-scales', type=str, help='ウェーブレットスケール (カンマ区切り)')
    parser.add_argument('--fractal-window', type=int, default=30, help='フラクタル解析窓サイズ')
    parser.add_argument('--chaos-window', type=int, default=50, help='カオス解析窓サイズ')
    parser.add_argument('--entropy-window', type=int, default=30, help='エントロピー解析窓サイズ')
    parser.add_argument('--no-volume', action='store_true', help='出来高を非表示')
    args = parser.parse_args()
    
    # チャートを作成
    chart = QuantumNeuralSupremeChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        quantum_window=args.quantum_window,
        wavelet_scales=[float(scale) for scale in args.wavelet_scales.split(',')] if args.wavelet_scales else None,
        fractal_window=args.fractal_window,
        chaos_window=args.chaos_window,
        entropy_window=args.entropy_window
    )
    
    # パフォーマンス分析
    performance = chart.analyze_performance()
    print(f"\n📊 パフォーマンス分析 (過去{performance.get('period_days', 0)}日):")
    print(f"  シグナル変化率: {performance.get('change_rate', 0)*100:.2f}%")
    print(f"  平均信頼度: {performance.get('avg_confidence', 0):.3f}")
    print(f"  平均値: {performance.get('avg_values', 0):.3f}")
    print(f"  シグナル分布: {performance.get('signal_distribution', {})}")
    
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        show_volume=not args.no_volume,
        savefig=args.output
    )


if __name__ == "__main__":
    main() 