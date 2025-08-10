#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
全エーラーズサイクル検出器期間比較チャート

このスクリプトは、ehlers_unified_dc.pyで利用可能な全てのサイクル検出器を使用して
実際の相場データでドミナントサイクル期間を計算し、比較チャートとして可視化します。

特徴:
- 設定ファイルから相場データを取得
- 全22種類のサイクル検出器を実行
- 各検出器の期間結果を同一チャートで比較表示
- 統計情報（平均、中央値、標準偏差）を表示
- インポートエラーがある検出器は自動的にスキップ
"""

import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import warnings

# プロジェクトルートを追加
sys.path.append(str(Path(__file__).parent))

# インポート
from data.data_loader import CSVDataSource, DataLoader
from data.binance_data_source import BinanceDataSource
from data.data_processor import DataProcessor
from indicators.cycle.ehlers_unified_dc import EhlersUnifiedDC
from logger import get_logger

# 警告を無視
warnings.filterwarnings('ignore')

class AllCycleDetectorsChart:
    """全サイクル検出器期間比較チャート"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        self.logger = get_logger(__name__)
        self.config = self._load_config(config_path)
        self.data_loader = self._setup_data_loader()
        
        # 利用可能な検出器を取得
        self.available_detectors = EhlersUnifiedDC.get_available_detectors()
        
        # カラーマップを設定
        self.colors = self._setup_colors()
        
        self.logger.info(f"利用可能な検出器数: {len(self.available_detectors)}")
        for detector_name, description in self.available_detectors.items():
            self.logger.info(f"  - {detector_name}: {description}")
    
    def _load_config(self, config_path: str) -> dict:
        """設定ファイルを読み込む"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"設定ファイルを読み込みました: {config_path}")
            return config
        except Exception as e:
            self.logger.error(f"設定ファイルの読み込みに失敗: {e}")
            raise
    
    def _setup_data_loader(self) -> DataLoader:
        """データローダーを設定"""
        try:
            # z_adaptive_trend_chart.pyと同じ方法でデータローダーを設定
            binance_config = self.config.get('binance_data', {})
            data_dir = binance_config.get('data_dir', 'data/binance')
            binance_data_source = BinanceDataSource(data_dir)
            
            # CSVデータソースはダミーとして渡す（Binanceデータソースのみを使用）
            dummy_csv_source = CSVDataSource("dummy")
            data_loader = DataLoader(
                data_source=dummy_csv_source,
                binance_data_source=binance_data_source
            )
            
            self.logger.info("データローダーを設定しました（Binanceデータソース使用）")
            return data_loader
            
        except Exception as e:
            self.logger.error(f"データローダーの設定に失敗: {e}")
            raise
    
    def _setup_colors(self) -> Dict[str, str]:
        """検出器ごとの色を設定"""
        # カテゴリ別に色を設定
        color_palette = {
            # コア検出器 - 青系
            'hody': '#1f77b4', 'phac': '#aec7e8', 'dudi': '#17becf',
            'dudi_e': '#9edae5', 'hody_e': '#2ca02c', 'phac_e': '#98df8a',
            
            # 基本サイクル検出器 - 緑系  
            'cycle_period': '#ff7f0e', 'cycle_period2': '#ffbb78',
            'bandpass_zero': '#d62728', 'autocorr_perio': '#ff9896',
            'dft_dominant': '#9467bd', 'multi_bandpass': '#c5b0d5',
            'absolute_ultimate': '#8c564b', 'ultra_supreme_stability': '#c49c94',
            'practical': '#e377c2',
            
            # 高度な検出器 - 紫・赤系
            'refined': '#f7b6d3', 'adaptive_ensemble': '#bcbd22',
            'adaptive_unified': '#dbdb8d', 'quantum_adaptive': '#17becf',
            'supreme_ultimate': '#ffbb78', 'ultimate': '#c7c7c7',
            'supreme': '#1f77b4'
        }
        
        return color_palette
    
    def load_market_data(self) -> pd.DataFrame:
        """市場データを読み込む"""
        try:
            # z_adaptive_trend_chart.pyと同じ方法でデータを読み込む
            self.logger.info("データを読み込み・処理中...")
            
            # データプロセッサーを初期化
            data_processor = DataProcessor()
            
            # データの読み込みと処理
            raw_data = self.data_loader.load_data_from_config(self.config)
            processed_data = {
                symbol: data_processor.process(df)
                for symbol, df in raw_data.items()
            }
            
            # 最初のシンボルのデータを取得
            first_symbol = next(iter(processed_data))
            data = processed_data[first_symbol]
            
            self.logger.info(f"データ読み込み完了: {first_symbol}")
            self.logger.info(f"期間: {data.index.min()} → {data.index.max()}")
            self.logger.info(f"データ数: {len(data)}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"データの読み込みに失敗: {e}")
            raise
    
    def calculate_all_detector_periods(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """全検出器でサイクル期間を計算"""
        results = {}
        successful_detectors = []
        failed_detectors = []
        
        for detector_name in self.available_detectors.keys():
            try:
                self.logger.info(f"計算中: {detector_name}")
                
                # 検出器を初期化
                detector = EhlersUnifiedDC(
                    detector_type=detector_name,
                    src_type='hlc3'  # 標準的な価格ソース
                )
                
                # サイクル期間を計算
                cycle_periods = detector.calculate(data)
                
                if cycle_periods is not None and len(cycle_periods) > 0:
                    # NaNを除去し、有効な値のみ保存
                    valid_periods = cycle_periods[~np.isnan(cycle_periods)]
                    if len(valid_periods) > 0:
                        results[detector_name] = cycle_periods
                        successful_detectors.append(detector_name)
                        
                        # 統計情報をログ出力
                        stats = {
                            'mean': np.nanmean(cycle_periods),
                            'median': np.nanmedian(cycle_periods),
                            'std': np.nanstd(cycle_periods),
                            'min': np.nanmin(cycle_periods),
                            'max': np.nanmax(cycle_periods)
                        }
                        self.logger.info(f"  {detector_name} 統計: 平均={stats['mean']:.2f}, "
                                       f"中央値={stats['median']:.2f}, 標準偏差={stats['std']:.2f}")
                    else:
                        self.logger.warning(f"  {detector_name}: 有効な期間データがありません")
                        failed_detectors.append(detector_name)
                else:
                    self.logger.warning(f"  {detector_name}: 計算結果が無効です")
                    failed_detectors.append(detector_name)
                    
            except Exception as e:
                self.logger.error(f"  {detector_name}の計算に失敗: {str(e)}")
                failed_detectors.append(detector_name)
        
        self.logger.info(f"成功した検出器数: {len(successful_detectors)}")
        self.logger.info(f"失敗した検出器数: {len(failed_detectors)}")
        
        if failed_detectors:
            self.logger.info(f"失敗した検出器: {', '.join(failed_detectors)}")
        
        return results
    
    def create_comparison_chart(
        self, 
        data: pd.DataFrame, 
        detector_results: Dict[str, np.ndarray],
        save_path: Optional[str] = None
    ) -> None:
        """比較チャートを作成"""
        
        if not detector_results:
            self.logger.error("描画可能な検出器結果がありません")
            return
        
        # 図のサイズを設定
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        fig.suptitle('全エーラーズサイクル検出器期間比較', fontsize=16, fontweight='bold')
        
        # 日付軸を準備
        dates = data.index
        
        # 1. 価格チャート
        ax1 = axes[0]
        ax1.plot(dates, data['close'], color='black', linewidth=1, label='終値')
        ax1.set_title('価格チャート', fontsize=12)
        ax1.set_ylabel('価格', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. サイクル期間チャート（全検出器）
        ax2 = axes[1] 
        legend_items = []
        
        for i, (detector_name, periods) in enumerate(detector_results.items()):
            color = self.colors.get(detector_name, f'C{i % 10}')
            
            # 期間をプロット
            ax2.plot(dates, periods, color=color, linewidth=1, alpha=0.7, label=detector_name)
            legend_items.append(detector_name)
        
        ax2.set_title(f'サイクル期間比較 ({len(detector_results)}種類の検出器)', fontsize=12)
        ax2.set_ylabel('期間', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # 3. 統計比較（ボックスプロット）
        ax3 = axes[2]
        
        # ボックスプロット用データを準備
        box_data = []
        box_labels = []
        
        for detector_name, periods in detector_results.items():
            valid_periods = periods[~np.isnan(periods)]
            if len(valid_periods) > 0:
                box_data.append(valid_periods)
                box_labels.append(detector_name)
        
        if box_data:
            bp = ax3.boxplot(box_data, labels=box_labels, patch_artist=True)
            
            # ボックスプロットに色を適用
            for patch, label in zip(bp['boxes'], box_labels):
                color = self.colors.get(label, 'lightblue')
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax3.set_title('期間分布比較（ボックスプロット）', fontsize=12)
        ax3.set_ylabel('期間', fontsize=10)
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 日付軸のフォーマット
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.tick_params(axis='x', rotation=45)
        
        # レイアウト調整
        plt.tight_layout()
        
        # 保存
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"チャートを保存しました: {save_path}")
        
        plt.show()
    
    def create_statistics_summary(self, detector_results: Dict[str, np.ndarray]) -> pd.DataFrame:
        """統計サマリーを作成"""
        stats_data = []
        
        for detector_name, periods in detector_results.items():  
            valid_periods = periods[~np.isnan(periods)]
            
            if len(valid_periods) > 0:
                stats = {
                    '検出器': detector_name,
                    '説明': self.available_detectors.get(detector_name, ''),
                    '平均期間': np.mean(valid_periods),
                    '中央値': np.median(valid_periods), 
                    '標準偏差': np.std(valid_periods),
                    '最小値': np.min(valid_periods),
                    '最大値': np.max(valid_periods),
                    '有効データ数': len(valid_periods),
                    '総データ数': len(periods)
                }
                stats_data.append(stats)
        
        df = pd.DataFrame(stats_data)
        
        # 平均期間でソート
        if not df.empty:
            df = df.sort_values('平均期間')
        
        return df
    
    def run_analysis(self, save_path: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        """分析を実行"""
        self.logger.info("=== 全サイクル検出器期間分析を開始 ===")
        
        # データを読み込む
        self.logger.info("1. 市場データを読み込み中...")
        data = self.load_market_data()
        
        # 全検出器で期間を計算
        self.logger.info("2. 全検出器でサイクル期間を計算中...")
        detector_results = self.calculate_all_detector_periods(data)
        
        if not detector_results:
            self.logger.error("計算できた検出器がありません。分析を終了します。")
            return pd.DataFrame(), {}
        
        # 統計サマリーを作成
        self.logger.info("3. 統計サマリーを作成中...")
        stats_df = self.create_statistics_summary(detector_results)
        
        self.logger.info("=== 統計サマリー ===")
        print(stats_df.to_string(index=False))
        
        # チャートを作成
        self.logger.info("4. 比較チャートを作成中...")
        
        # デフォルトの保存パスを設定
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"all_cycle_detectors_comparison_{timestamp}.png"
        
        self.create_comparison_chart(data, detector_results, save_path)
        
        self.logger.info("=== 分析完了 ===")
        
        return stats_df, detector_results


def main():
    """メイン実行関数"""
    try:
        # 分析を実行
        analyzer = AllCycleDetectorsChart()
        stats_df, detector_results = analyzer.run_analysis()
        
        # 結果をCSVで保存
        if not stats_df.empty:
            csv_path = f"cycle_detectors_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            stats_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"統計サマリーを保存しました: {csv_path}")
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"分析中にエラーが発生しました: {e}")
        raise


if __name__ == "__main__":
    main()