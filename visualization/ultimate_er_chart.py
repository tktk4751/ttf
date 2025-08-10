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
from indicators.ultimate_er import UltimateER
from indicators.efficiency_ratio import EfficiencyRatio


class UltimateERChart:
    """
    Ultimate Efficiency Ratioを表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - Traditional ER vs Ultimate ER
    - UKFフィルタリングの各段階
    - 信頼度スコア
    - トレンド信号
    - 動的期間（Dynamic mode時）
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.traditional_er = None
        self.ultimate_er = None
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
                           # Traditional ER パラメータ
                           traditional_period: int = 20,
                           traditional_smoothing: str = 'hma',
                           traditional_smoother_period: int = 13,
                           # Ultimate ER パラメータ（新ロジック）
                           er_period: float = 5.0,  # ER計算期間（固定5期間）
                           src_type: str = 'hlc3',
                           # UKFパラメータ
                           ukf_alpha: float = 0.001,
                           ukf_beta: float = 2.0,
                           ukf_kappa: float = 0.0,
                           ukf_process_noise: float = 0.001,
                           ukf_volatility_window: int = 10,
                           ukf_adaptive_noise: bool = True,
                           # Ultimate Smootherパラメータ（固定20期間）
                           smoother_period: float = 20.0,
                           # トレンド判定パラメータ
                           slope_index: int = 3,
                           range_threshold: float = 0.005,
                           # 信頼度重み付け
                           use_confidence_weighting: bool = True
                          ) -> None:
        """
        Traditional ERとUltimate ERを計算する
        
        Args:
            traditional_period: Traditional ERの期間
            traditional_smoothing: Traditional ERのスムージング方法
            traditional_smoother_period: Traditional ERのスムージング期間
            er_period: ER計算期間（固定5期間）
            src_type: 価格ソースタイプ
            ukf_alpha: UKFアルファパラメータ
            ukf_beta: UKFベータパラメータ
            ukf_kappa: UKFカッパパラメータ
            ukf_process_noise: UKFプロセスノイズスケール
            ukf_volatility_window: UKFボラティリティ計算窓
            ukf_adaptive_noise: UKF適応的ノイズ推定
            smoother_period: Ultimate Smoother期間（固定20期間）
            slope_index: トレンド判定期間
            range_threshold: レンジ判定閾値
            use_confidence_weighting: 信頼度重み付けを使用するか
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\nインジケーターを計算中...")
        
        # Traditional Efficiency Ratio
        print("Traditional Efficiency Ratioを計算中...")
        self.traditional_er = EfficiencyRatio(
            period=traditional_period,
            src_type=src_type,
            smoothing_method=traditional_smoothing,
            use_dynamic_period=False,
            slope_index=slope_index,
            range_threshold=range_threshold,
            smoother_period=traditional_smoother_period
        )
        self.traditional_result = self.traditional_er.calculate(self.data)
        
        # Ultimate Efficiency Ratio（新ロジック: 5期間ER + UKF + 20期間Ultimate Smoother）
        print("Ultimate Efficiency Ratio (5-period ER + UKF + 20-period Smoother)を計算中...")
        self.ultimate_er = UltimateER(
            er_period=er_period,
            src_type=src_type,
            ukf_alpha=ukf_alpha,
            ukf_beta=ukf_beta,
            ukf_kappa=ukf_kappa,
            ukf_process_noise=ukf_process_noise,
            ukf_volatility_window=ukf_volatility_window,
            ukf_adaptive_noise=ukf_adaptive_noise,
            smoother_period=smoother_period,
            slope_index=slope_index,
            range_threshold=range_threshold,
            use_confidence_weighting=use_confidence_weighting
        )
        self.ultimate_result = self.ultimate_er.calculate(self.data)
        
        print("インジケーター計算完了")
        
        # 統計情報の表示
        self._display_statistics()
    
    def _display_statistics(self) -> None:
        """統計情報を表示"""
        print("\n=== 統計情報 ===")
        
        # 平均ER値
        print(f"\n平均ER値:")
        print(f"Traditional: {np.nanmean(self.traditional_result.values):.4f}")
        print(f"Ultimate: {np.nanmean(self.ultimate_result.values):.4f}")
        
        # トレンド判定率
        traditional_trending = np.sum(self.traditional_result.trend_signals != 0) / len(self.traditional_result.trend_signals) * 100
        ultimate_trending = np.sum(self.ultimate_result.trend_signals != 0) / len(self.ultimate_result.trend_signals) * 100
        
        print(f"\nトレンド判定率:")
        print(f"Traditional: {traditional_trending:.2f}%")
        print(f"Ultimate: {ultimate_trending:.2f}%")
        
        # ノイズ除去効果
        if hasattr(self.ultimate_result, 'raw_er'):
            raw_std = np.nanstd(self.ultimate_result.raw_er)
            ultimate_std = np.nanstd(self.ultimate_result.values)
            print(f"\nノイズ除去効果（標準偏差）:")
            print(f"Raw ER: {raw_std:.4f}")
            print(f"Ultimate ER: {ultimate_std:.4f} (削減率: {(1-ultimate_std/raw_std)*100:.1f}%)")
        
        # 信頼度スコア
        print(f"\n信頼度スコア統計:")
        print(f"平均: {np.nanmean(self.ultimate_result.confidence_scores):.4f}")
        print(f"最小: {np.nanmin(self.ultimate_result.confidence_scores):.4f}")
        print(f"最大: {np.nanmax(self.ultimate_result.confidence_scores):.4f}")
            
    def plot(self, 
            title: str = "Ultimate Efficiency Ratio", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            show_processing_stages: bool = True,
            figsize: Tuple[int, int] = (16, 14),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとUltimate ERを描画する
        
        Args:
            title: チャートのタイトル
            start_date: 表示開始日（フォーマット: YYYY-MM-DD）
            end_date: 表示終了日（フォーマット: YYYY-MM-DD）
            show_volume: 出来高を表示するか
            show_processing_stages: 処理段階を表示するか
            figsize: 図のサイズ
            style: mplfinanceのスタイル
            savefig: 保存先のパス（指定しない場合は表示のみ）
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        if self.traditional_result is None or self.ultimate_result is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'traditional_er': self.traditional_result.values,
                'ultimate_er': self.ultimate_result.values,
                'raw_er': self.ultimate_result.raw_er,
                'ukf_filtered': self.ultimate_result.ukf_filtered,
                'confidence': self.ultimate_result.confidence_scores,
                'trend_traditional': self.traditional_result.trend_signals,
                'trend_ultimate': self.ultimate_result.trend_signals,
                'dynamic_periods': self.ultimate_result.dynamic_periods
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        
        # mplfinanceでプロット用の設定
        # メインチャート上のプロット
        main_plots = []
        
        # プロット設定
        if show_processing_stages:
            # 処理段階を表示
            panel_count = 6 if show_volume else 5
            panel_configs = []
            
            # 1. Traditional ER
            panel_configs.append(('traditional_er', 'Traditional ER', 'blue', 1))
            
            # 2. Ultimate ER Processing Stages
            panel_configs.append(('raw_er', 'Raw ER', 'gray', 2, 0.5))
            panel_configs.append(('ukf_filtered', 'UKF Filtered', 'purple', 2))
            panel_configs.append(('ultimate_er', 'Ultimate ER', 'darkgreen', 2, 2))
            
            # 3. 信頼度スコア
            panel_configs.append(('confidence', 'Confidence Score', 'orange', 3))
            
            # 4. トレンド信号
            panel_configs.append(('trend_ultimate', 'Trend Signal', 'red', 4))
            
            # 動的期間は使用しないのでスキップ
        else:
            # シンプル表示
            panel_count = 4 if show_volume else 3
            panel_configs = []
            
            # 1. ER比較
            panel_configs.append(('traditional_er', 'Traditional ER', 'blue', 1))
            panel_configs.append(('ultimate_er', 'Ultimate ER', 'darkgreen', 1, 2))
            
            # 2. 信頼度スコア
            panel_configs.append(('confidence', 'Confidence Score', 'orange', 2))
            
            # 3. トレンド信号
            panel_configs.append(('trend_ultimate', 'Trend Signal', 'red', 3))
        
        # パネルプロットを作成
        addplots = []
        panel_idx_offset = 1 if show_volume else 0
        
        for config in panel_configs:
            if len(config) == 4:
                col, label, color, panel = config
                width = 1.2
                alpha = 1.0
            else:
                col, label, color, panel, width = config
                alpha = 0.7 if width < 1 else 1.0
            
            if col in df.columns:
                addplots.append(mpf.make_addplot(
                    df[col],
                    panel=panel + panel_idx_offset,
                    color=color,
                    width=width,
                    alpha=alpha,
                    ylabel=label,
                    secondary_y=False,
                    label=label
                ))
        
        # mplfinanceの設定
        kwargs = dict(
            type='candle',
            figsize=figsize,
            title=title,
            style=style,
            datetime_format='%Y-%m-%d',
            xrotation=45,
            returnfig=True,
            addplot=addplots
        )
        
        # パネル比率の設定
        if show_volume:
            if show_processing_stages:
                kwargs['panel_ratios'] = (3, 1, 1.5, 1.5, 1, 1)
            else:
                kwargs['panel_ratios'] = (3, 1, 1.5, 1, 1)
            kwargs['volume'] = True
        else:
            if show_processing_stages:
                kwargs['panel_ratios'] = (3, 1.5, 1.5, 1, 1)
            else:
                kwargs['panel_ratios'] = (3, 1.5, 1, 1)
            kwargs['volume'] = False
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        panel_offset = 1 if show_volume else 0
        
        # ER値パネルに閾値線を追加
        er_panel_idx = 1 + panel_offset
        axes[er_panel_idx].axhline(y=0.618, color='green', linestyle='--', alpha=0.5, label='Strong Trend')
        axes[er_panel_idx].axhline(y=0.382, color='red', linestyle='--', alpha=0.5, label='Weak Trend')
        axes[er_panel_idx].set_ylim(0, 1)
        
        # 信頼度パネル
        if show_processing_stages:
            confidence_panel_idx = 3 + panel_offset
            axes[confidence_panel_idx].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
            axes[confidence_panel_idx].set_ylim(0, 1)
            
            # トレンド信号パネル
            trend_panel_idx = 4 + panel_offset
            axes[trend_panel_idx].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[trend_panel_idx].axhline(y=1, color='green', linestyle='--', alpha=0.5)
            axes[trend_panel_idx].axhline(y=-1, color='red', linestyle='--', alpha=0.5)
            axes[trend_panel_idx].set_ylim(-1.5, 1.5)
        else:
            confidence_panel_idx = 2 + panel_offset
            axes[confidence_panel_idx].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
            axes[confidence_panel_idx].set_ylim(0, 1)
            
            trend_panel_idx = 3 + panel_offset
            axes[trend_panel_idx].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[trend_panel_idx].set_ylim(-1.5, 1.5)
        
        # 統計情報の表示
        print(f"\n=== チャート期間の統計 ===")
        print(f"期間: {df.index[0].strftime('%Y-%m-%d')} から {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"データ点数: {len(df)}")
        
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
    parser = argparse.ArgumentParser(description='Ultimate Efficiency Ratioの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--src-type', type=str, default='hlc3', help='価格ソースタイプ')
    parser.add_argument('--er-period', type=float, default=5.0, help='ER計算期間')
    parser.add_argument('--ukf-alpha', type=float, default=0.001, help='UKFアルファパラメータ')
    parser.add_argument('--smoother-period', type=float, default=20.0, help='Ultimate Smoother期間')
    parser.add_argument('--show-stages', action='store_true', help='処理段階を表示')
    args = parser.parse_args()
    
    # チャートを作成
    chart = UltimateERChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        er_period=args.er_period,
        src_type=args.src_type,
        ukf_alpha=args.ukf_alpha,
        smoother_period=args.smoother_period
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output,
        show_processing_stages=args.show_stages
    )


if __name__ == "__main__":
    main()