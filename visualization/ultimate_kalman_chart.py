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
from indicators.ultimate_kalman_filter import UltimateKalmanFilter


class UltimateKalmanChart:
    """
    アルティメットカルマンフィルターを表示するローソク足チャートクラス
    
    📊 **表示内容:**
    - ローソク足と出来高
    - 元の価格データ vs フィルター済み価格
    - 前方パス（Forward）と双方向パス（Bidirectional）の比較
    - 信頼度スコア・カルマンゲイン・予測誤差の推移
    - ボラティリティ推定値とノイズレベル
    - 包括的な統計情報とフィルター品質指標
    
    🎯 **分析機能:**
    - ノイズ除去効果の可視化
    - 適応的パラメータ調整の確認
    - 単方向 vs 双方向処理の効果比較
    - 市場状況別フィルター性能評価
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.ultimate_kalman = None
        self.kalman_result = None
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

    def calculate_filter(self,
                        bidirectional: bool = True,
                        base_process_noise: float = 1e-5,
                        base_observation_noise: float = 0.01,
                        volatility_window: int = 10,
                        src_type: str = 'hlc3') -> None:
        """
        アルティメットカルマンフィルターを計算する
        
        Args:
            bidirectional: 双方向処理を使用するか（True=高品質、False=高速）
            base_process_noise: 基本プロセスノイズ（デフォルト: 1e-5）
            base_observation_noise: 基本観測ノイズ（デフォルト: 0.01）
            volatility_window: ボラティリティ推定ウィンドウ（デフォルト: 10）
            src_type: 価格ソース ('close', 'hlc3', etc.)
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print(f"\nアルティメットカルマンフィルター計算中... モード: {'双方向' if bidirectional else '単方向'}")
        
        # アルティメットカルマンフィルターを初期化
        self.ultimate_kalman = UltimateKalmanFilter(
            bidirectional=bidirectional,
            base_process_noise=base_process_noise,
            base_observation_noise=base_observation_noise,
            volatility_window=volatility_window,
            src_type=src_type
        )
        
        # フィルター計算実行
        print("計算を実行します...")
        self.kalman_result = self.ultimate_kalman.calculate(self.data)
        
        # 結果の検証
        final_values = self.kalman_result.values
        print(f"フィルター計算完了 - データ数: {len(final_values)}")
        
        # NaN値のチェック
        nan_count = np.isnan(final_values).sum()
        print(f"NaN値: {nan_count}個")
        
        # 統計情報
        performance_stats = self.ultimate_kalman.get_performance_stats()
        print(f"ノイズ削減率: {performance_stats['noise_reduction_percentage']:.1f}%")
        print(f"平均信頼度: {performance_stats['average_confidence']:.3f}")
        print(f"平均カルマンゲイン: {performance_stats['average_kalman_gain']:.3f}")
        
        print("アルティメットカルマンフィルター計算完了")
            
    def plot(self, 
            title: str = "アルティメットカルマンフィルター分析", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            show_forward_backward_comparison: bool = True,
            figsize: Tuple[int, int] = (16, 14),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとアルティメットカルマンフィルターを描画する
        
        Args:
            title: チャートのタイトル
            start_date: 表示開始日（フォーマット: YYYY-MM-DD）
            end_date: 表示終了日（フォーマット: YYYY-MM-DD）
            show_volume: 出来高を表示するか
            show_forward_backward_comparison: 前方・双方向パスの比較を表示するか
            figsize: 図のサイズ
            style: mplfinanceのスタイル
            savefig: 保存先のパス（指定しない場合は表示のみ）
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        if self.kalman_result is None:
            raise ValueError("フィルターが計算されていません。calculate_filter()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # アルティメットカルマンフィルターの結果を取得
        print("フィルターデータを取得中...")
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'raw_values': self.kalman_result.raw_values,
                'final_values': self.kalman_result.values,
                'forward_values': self.kalman_result.forward_values,
                'backward_values': self.kalman_result.backward_values if self.kalman_result.is_bidirectional else np.nan,
                'confidence_scores': self.kalman_result.confidence_scores,
                'kalman_gains': self.kalman_result.kalman_gains,
                'prediction_errors': self.kalman_result.prediction_errors,
                'volatility_estimates': self.kalman_result.volatility_estimates,
                'process_noise': self.kalman_result.process_noise,
                'observation_noise': self.kalman_result.observation_noise
            }
        )
        
        # 絞り込み後のデータに対してフィルターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        print(f"フィルターデータ確認 - 最終値NaN: {df['final_values'].isna().sum()}")
        
        # mplfinanceでプロット用の設定
        main_plots = []
        
        # 1. メインチャート上のフィルター結果プロット
        main_plots.append(mpf.make_addplot(df['final_values'], color='blue', width=2.5, 
                                          label='Kalman Filtered', alpha=0.9))
        
        if show_forward_backward_comparison and self.kalman_result.is_bidirectional:
            main_plots.append(mpf.make_addplot(df['forward_values'], color='green', width=1.5, 
                                              label='Forward Only', alpha=0.7, linestyle='--'))
            main_plots.append(mpf.make_addplot(df['backward_values'], color='purple', width=1.5, 
                                              label='Bidirectional', alpha=0.8, linestyle='-'))
        
        # 2. 信頼度スコアパネル
        confidence_panel = mpf.make_addplot(df['confidence_scores'], panel=1, color='purple', width=1.5, 
                                           ylabel='Confidence', label='Confidence Score')
        
        # 3. カルマンゲインパネル
        gain_panel = mpf.make_addplot(df['kalman_gains'], panel=2, color='orange', width=1.5, 
                                     ylabel='Kalman Gain', label='Kalman Gain')
        
        # 4. 予測誤差パネル
        error_panel = mpf.make_addplot(df['prediction_errors'], panel=3, color='red', width=1.2, 
                                      ylabel='Prediction Error', label='Prediction Error')
        
        # 5. ボラティリティ・ノイズパネル
        vol_panel = mpf.make_addplot(df['volatility_estimates'], panel=4, color='brown', width=1.2, 
                                    ylabel='Volatility', label='Volatility')
        noise_panel = mpf.make_addplot(df['observation_noise'], panel=4, color='gray', width=1.0, 
                                      secondary_y=True, label='Obs Noise')
        
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
        
        # 出来高とパネル構成の設定
        if show_volume:
            kwargs['volume'] = True
            # メイン:出来高:信頼度:ゲイン:誤差:ボラティリティ
            kwargs['panel_ratios'] = (4, 1, 1, 1, 1, 1)
            # 出来高を表示する場合は、パネル番号を+1する
            confidence_panel = mpf.make_addplot(df['confidence_scores'], panel=2, color='purple', width=1.5, 
                                               ylabel='Confidence', label='Confidence Score')
            gain_panel = mpf.make_addplot(df['kalman_gains'], panel=3, color='orange', width=1.5, 
                                         ylabel='Kalman Gain', label='Kalman Gain')
            error_panel = mpf.make_addplot(df['prediction_errors'], panel=4, color='red', width=1.2, 
                                          ylabel='Prediction Error', label='Prediction Error')
            vol_panel = mpf.make_addplot(df['volatility_estimates'], panel=5, color='brown', width=1.2, 
                                        ylabel='Volatility', label='Volatility')
            noise_panel = mpf.make_addplot(df['observation_noise'], panel=5, color='gray', width=1.0, 
                                          secondary_y=True, label='Obs Noise')
        else:
            kwargs['volume'] = False
            # メイン:信頼度:ゲイン:誤差:ボラティリティ
            kwargs['panel_ratios'] = (4, 1, 1, 1, 1)
        
        # すべてのプロットを結合
        all_plots = main_plots + [confidence_panel, gain_panel, error_panel, vol_panel, noise_panel]
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加
        legend_labels = ['Kalman Filtered']
        if show_forward_backward_comparison and self.kalman_result.is_bidirectional:
            legend_labels.extend(['Forward Only', 'Bidirectional'])
        
        axes[0].legend(legend_labels, loc='upper left')
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        panel_offset = 1 if show_volume else 0
        
        # 信頼度スコアパネル
        conf_panel = panel_offset + 1
        axes[conf_panel].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
        axes[conf_panel].axhline(y=0.8, color='green', linestyle='--', alpha=0.3)
        axes[conf_panel].axhline(y=0.2, color='red', linestyle='--', alpha=0.3)
        
        # カルマンゲインパネル
        gain_panel_idx = panel_offset + 2
        gain_mean = df['kalman_gains'].mean()
        axes[gain_panel_idx].axhline(y=gain_mean, color='black', linestyle='-', alpha=0.3)
        axes[gain_panel_idx].axhline(y=0.1, color='green', linestyle='--', alpha=0.3)
        axes[gain_panel_idx].axhline(y=0.9, color='red', linestyle='--', alpha=0.3)
        
        # 予測誤差パネル
        error_panel_idx = panel_offset + 3
        error_mean = df['prediction_errors'].mean()
        axes[error_panel_idx].axhline(y=error_mean, color='black', linestyle='-', alpha=0.3)
        
        # 統計情報の表示
        print(f"\n=== アルティメットカルマンフィルター統計 ===")
        performance_stats = self.ultimate_kalman.get_performance_stats()
        comparison_stats = self.ultimate_kalman.get_comparison_with_components()
        
        print(f"処理モード: {performance_stats['processing_mode']}")
        print(f"ノイズ削減率: {performance_stats['noise_reduction_percentage']:.1f}%")
        print(f"平均信頼度: {performance_stats['average_confidence']:.3f}")
        print(f"平均カルマンゲイン: {performance_stats['average_kalman_gain']:.3f}")
        print(f"平均予測誤差: {performance_stats['average_prediction_error']:.6f}")
        print(f"平均ボラティリティ: {performance_stats['average_volatility']:.6f}")
        
        # フィルター特性
        filter_chars = performance_stats['filter_characteristics']
        print(f"\nフィルター特性:")
        print(f"基本プロセスノイズ: {filter_chars['base_process_noise']:.0e}")
        print(f"基本観測ノイズ: {filter_chars['base_observation_noise']:.3f}")
        print(f"適応的観測ノイズ範囲: {filter_chars['adaptive_noise_range'][0]:.4f} - {filter_chars['adaptive_noise_range'][1]:.4f}")
        
        # 品質指標
        quality = performance_stats['quality_indicators']
        print(f"\n品質指標:")
        print(f"生データとの相関: {quality['raw_filtered_correlation']:.3f}")
        print(f"平滑化ファクター: {quality['smoothness_factor']:.3f}")
        
        if self.kalman_result.is_bidirectional and quality['forward_backward_correlation'] is not None:
            print(f"前方・後方相関: {quality['forward_backward_correlation']:.3f}")
            
            # 双方向改善効果
            bidir_improvement = comparison_stats['noise_reduction_comparison']['bidirectional_improvement']
            print(f"双方向処理改善効果: {bidir_improvement:.1%}")
        
        # 保存または表示
        if savefig:
            plt.savefig(savefig, dpi=150, bbox_inches='tight')
            print(f"\nチャートを保存しました: {savefig}")
        else:
            plt.tight_layout()
            plt.show()

    def plot_comparison_analysis(self, 
                               title: str = "カルマンフィルター比較分析",
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None,
                               figsize: Tuple[int, int] = (16, 10),
                               savefig: Optional[str] = None) -> None:
        """
        カルマンフィルターの詳細比較分析チャートを作成
        
        Args:
            title: チャートのタイトル
            start_date: 表示開始日
            end_date: 表示終了日
            figsize: 図のサイズ
            savefig: 保存先のパス
        """
        if self.kalman_result is None:
            raise ValueError("フィルターが計算されていません。calculate_filter()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
        
        # カルマンフィルター結果を結合
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'raw': self.kalman_result.raw_values,
                'forward': self.kalman_result.forward_values,
                'final': self.kalman_result.values,
                'confidence': self.kalman_result.confidence_scores,
                'kalman_gain': self.kalman_result.kalman_gains,
                'volatility': self.kalman_result.volatility_estimates
            }
        )
        
        df = df.join(full_df).dropna()
        
        # サブプロット作成
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. 価格比較
        axes[0, 0].plot(df.index, df['close'], label='Original Price', color='gray', alpha=0.7)
        axes[0, 0].plot(df.index, df['forward'], label='Forward Pass', color='green', linewidth=2)
        if self.kalman_result.is_bidirectional:
            axes[0, 0].plot(df.index, df['final'], label='Bidirectional', color='blue', linewidth=2)
        axes[0, 0].set_title('価格比較')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. ノイズ除去効果
        raw_diff = np.diff(df['close'])
        forward_diff = np.diff(df['forward'])
        final_diff = np.diff(df['final'])
        
        axes[0, 1].hist(raw_diff, bins=50, alpha=0.5, label='Original', density=True)
        axes[0, 1].hist(forward_diff, bins=50, alpha=0.5, label='Forward', density=True)
        if self.kalman_result.is_bidirectional:
            axes[0, 1].hist(final_diff, bins=50, alpha=0.5, label='Bidirectional', density=True)
        axes[0, 1].set_title('価格変化分布（ノイズ除去効果）')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 信頼度スコア推移
        axes[1, 0].plot(df.index, df['confidence'], color='purple', linewidth=2)
        axes[1, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].axhline(y=0.8, color='green', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('信頼度スコア推移')
        axes[1, 0].set_ylabel('Confidence Score')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. カルマンゲイン推移
        axes[1, 1].plot(df.index, df['kalman_gain'], color='orange', linewidth=2)
        axes[1, 1].axhline(y=df['kalman_gain'].mean(), color='black', linestyle='-', alpha=0.5)
        axes[1, 1].set_title('カルマンゲイン推移')
        axes[1, 1].set_ylabel('Kalman Gain')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 5. ボラティリティ vs 信頼度散布図
        axes[2, 0].scatter(df['volatility'], df['confidence'], alpha=0.6, c=df['kalman_gain'], 
                          cmap='viridis', s=20)
        axes[2, 0].set_xlabel('Volatility')
        axes[2, 0].set_ylabel('Confidence')
        axes[2, 0].set_title('ボラティリティ vs 信頼度')
        axes[2, 0].grid(True, alpha=0.3)
        
        # 6. フィルター性能サマリー
        performance_stats = self.ultimate_kalman.get_performance_stats()
        comparison_stats = self.ultimate_kalman.get_comparison_with_components()
        
        stats_text = f"""フィルター性能サマリー:

処理モード: {performance_stats['processing_mode']}
ノイズ削減率: {performance_stats['noise_reduction_percentage']:.1f}%
平均信頼度: {performance_stats['average_confidence']:.3f}
平均ゲイン: {performance_stats['average_kalman_gain']:.3f}

品質指標:
生データ相関: {performance_stats['quality_indicators']['raw_filtered_correlation']:.3f}
平滑化係数: {performance_stats['quality_indicators']['smoothness_factor']:.3f}"""

        if self.kalman_result.is_bidirectional:
            bidir_corr = performance_stats['quality_indicators']['forward_backward_correlation']
            bidir_improvement = comparison_stats['noise_reduction_comparison']['bidirectional_improvement']
            stats_text += f"""
前方後方相関: {bidir_corr:.3f}
双方向改善: {bidir_improvement:.1%}"""
        
        axes[2, 1].text(0.05, 0.95, stats_text, transform=axes[2, 1].transAxes, 
                        verticalalignment='top', fontsize=10, fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        axes[2, 1].set_title('統計サマリー')
        axes[2, 1].axis('off')
        
        plt.tight_layout()
        
        if savefig:
            plt.savefig(savefig, dpi=150, bbox_inches='tight')
            print(f"比較分析チャートを保存しました: {savefig}")
        else:
            plt.show()


def main():
    """メイン関数"""
    # コマンドライン引数を処理
    import argparse
    parser = argparse.ArgumentParser(description='アルティメットカルマンフィルターの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--comparison-output', type=str, help='比較分析チャートの出力ファイルのパス')
    parser.add_argument('--src-type', type=str, default='hlc3', help='価格ソースタイプ')
    parser.add_argument('--bidirectional', action='store_true', help='双方向処理を使用')
    parser.add_argument('--process-noise', type=float, default=1e-5, help='基本プロセスノイズ')
    parser.add_argument('--observation-noise', type=float, default=0.01, help='基本観測ノイズ')
    parser.add_argument('--volatility-window', type=int, default=10, help='ボラティリティ推定ウィンドウ')
    args = parser.parse_args()
    
    # チャートを作成
    chart = UltimateKalmanChart()
    chart.load_data_from_config(args.config)
    chart.calculate_filter(
        bidirectional=args.bidirectional,
        base_process_noise=args.process_noise,
        base_observation_noise=args.observation_noise,
        volatility_window=args.volatility_window,
        src_type=args.src_type
    )
    
    # メインチャート
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output
    )
    
    # 比較分析チャート
    if args.comparison_output:
        chart.plot_comparison_analysis(
            start_date=args.start,
            end_date=args.end,
            savefig=args.comparison_output
        )


if __name__ == "__main__":
    main()