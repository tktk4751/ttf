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
import time
import warnings
warnings.filterwarnings('ignore')

# データ取得のための依存関係
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource

# インジケーター
from indicators.cycle.ehlers_unified_dc import EhlersUnifiedDC


class AllCycleDetectorsPerformanceTest:
    """
    EhlersUnifiedDCの全サイクル検出器の性能をテストするクラス
    
    - 実際の相場データを使用
    - 全ての利用可能なサイクル検出器をテスト
    - サイクル期間の分析と描画
    - 性能統計の収集
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.detectors = {}
        self.detector_results = {}
        self.detector_stats = {}
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
    
    def initialize_all_detectors(self) -> None:
        """
        利用可能な全てのサイクル検出器を初期化する
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
        
        print("\n全サイクル検出器を初期化中...")
        
        # 利用可能な検出器を取得
        available_detectors = EhlersUnifiedDC.get_available_detectors()
        
        # 各検出器を初期化
        for detector_name, description in available_detectors.items():
            try:
                print(f"初期化中: {detector_name} ({description})")
                
                # 共通パラメータ
                common_params = {
                    'detector_type': detector_name,
                    'src_type': 'hlc3',
                    'cycle_part': 0.5,
                    'max_cycle': 89,
                    'min_cycle': 8,
                    'max_output': 89,
                    'min_output': 8,
                    'use_kalman_filter': False,  # 基本テストではKalmanフィルターを無効
                }
                
                # 検出器固有のパラメータ調整
                if detector_name in ['dudi_e', 'hody_e', 'phac_e']:
                    common_params.update({
                        'lp_period': 13,
                        'hp_period': 124
                    })
                elif detector_name in ['cycle_period', 'cycle_period2']:
                    common_params.update({
                        'alpha': 0.07
                    })
                elif detector_name == 'bandpass_zero':
                    common_params.update({
                        'bandwidth': 0.6,
                        'center_period': 15.0
                    })
                elif detector_name == 'autocorr_perio':
                    common_params.update({
                        'avg_length': 3.0
                    })
                elif detector_name == 'dft_dominant':
                    common_params.update({
                        'window': 50
                    })
                elif detector_name in ['absolute_ultimate', 'ultra_supreme_stability']:
                    common_params.update({
                        'period_range': (8, 89)
                    })
                elif detector_name == 'ultra_supreme_dft':
                    common_params.update({
                        'window': 50
                    })
                elif detector_name == 'refined':
                    common_params.update({
                        'period_range': (8, 89),
                        'alpha': 0.07,
                        'ultimate_smoother_period': 20.0,
                        'use_ultimate_smoother': True
                    })
                
                # 検出器を作成
                detector = EhlersUnifiedDC(**common_params)
                self.detectors[detector_name] = detector
                
                print(f"✓ {detector_name} 初期化成功")
                
            except Exception as e:
                print(f"✗ {detector_name} 初期化失敗: {e}")
                continue
        
        print(f"\n初期化完了: {len(self.detectors)}/{len(available_detectors)} 検出器")
    
    def calculate_all_detectors(self) -> None:
        """
        全ての検出器でサイクルを計算する
        """
        if not self.detectors:
            raise ValueError("検出器が初期化されていません。initialize_all_detectors()を先に実行してください。")
        
        print("\n全検出器でサイクル計算を実行中...")
        
        for detector_name, detector in self.detectors.items():
            try:
                print(f"計算中: {detector_name}")
                start_time = time.time()
                
                # サイクル計算
                cycle_values = detector.calculate(self.data)
                calc_time = time.time() - start_time
                
                # 結果を保存
                self.detector_results[detector_name] = cycle_values
                
                # 統計を計算
                valid_values = cycle_values[~np.isnan(cycle_values)]
                if len(valid_values) > 0:
                    stats = {
                        'mean': np.mean(valid_values),
                        'std': np.std(valid_values),
                        'min': np.min(valid_values),
                        'max': np.max(valid_values),
                        'median': np.median(valid_values),
                        'nan_count': np.sum(np.isnan(cycle_values)),
                        'valid_count': len(valid_values),
                        'calc_time': calc_time,
                        'description': EhlersUnifiedDC.get_available_detectors().get(detector_name, detector_name)
                    }
                else:
                    stats = {
                        'mean': np.nan,
                        'std': np.nan,
                        'min': np.nan,
                        'max': np.nan,
                        'median': np.nan,
                        'nan_count': len(cycle_values),
                        'valid_count': 0,
                        'calc_time': calc_time,
                        'description': EhlersUnifiedDC.get_available_detectors().get(detector_name, detector_name)
                    }
                
                self.detector_stats[detector_name] = stats
                
                print(f"✓ {detector_name} 完了 ({calc_time:.3f}秒, 有効値: {stats['valid_count']}/{len(cycle_values)})")
                
            except Exception as e:
                print(f"✗ {detector_name} 計算失敗: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n計算完了: {len(self.detector_results)}/{len(self.detectors)} 検出器")
    
    def plot_comparison_chart(self, 
                            title: str = "全サイクル検出器比較", 
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None,
                            show_volume: bool = True,
                            figsize: Tuple[int, int] = (16, 20),
                            style: str = 'yahoo',
                            max_detectors_per_panel: int = 4,
                            savefig: Optional[str] = None) -> None:
        """
        全サイクル検出器の比較チャートを描画する
        
        Args:
            title: チャートのタイトル
            start_date: 表示開始日（フォーマット: YYYY-MM-DD）
            end_date: 表示終了日（フォーマット: YYYY-MM-DD）
            show_volume: 出来高を表示するか
            figsize: 図のサイズ
            style: mplfinanceのスタイル
            max_detectors_per_panel: パネルあたりの最大検出器数
            savefig: 保存先のパス（指定しない場合は表示のみ）
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。")
        
        if not self.detector_results:
            raise ValueError("検出器の結果がありません。calculate_all_detectors()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
        
        print(f"チャートデータ準備: {len(df)}件")
        
        # 検出器結果をデータフレームに結合
        for detector_name, cycle_values in self.detector_results.items():
            # データ長を合わせる
            if len(cycle_values) == len(self.data):
                full_series = pd.Series(cycle_values, index=self.data.index)
                df[f'cycle_{detector_name}'] = full_series
            else:
                print(f"警告: {detector_name}のデータ長が不一致 ({len(cycle_values)} vs {len(self.data)})")
        
        # 検出器をグループ分け
        detector_names = list(self.detector_results.keys())
        detector_groups = []
        for i in range(0, len(detector_names), max_detectors_per_panel):
            detector_groups.append(detector_names[i:i + max_detectors_per_panel])
        
        # 色のパレット
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 
                 'olive', 'cyan', 'magenta', 'yellow', 'navy', 'lime', 'teal', 'silver']
        
        # プロット設定
        addplots = []
        panel_ratios = [4]  # メインチャート
        
        if show_volume:
            panel_ratios.append(1)  # 出来高
        
        # 各検出器グループのパネルを追加
        for group_idx, group in enumerate(detector_groups):
            panel_ratios.append(2)  # サイクル検出器パネル
            
            for det_idx, detector_name in enumerate(group):
                col_name = f'cycle_{detector_name}'
                if col_name in df.columns:
                    color_idx = (group_idx * max_detectors_per_panel + det_idx) % len(colors)
                    panel_num = (2 if show_volume else 1) + group_idx
                    
                    addplots.append(
                        mpf.make_addplot(
                            df[col_name], 
                            panel=panel_num, 
                            color=colors[color_idx], 
                            width=1.5,
                            ylabel=f'Cycle Period (Group {group_idx + 1})',
                            label=detector_name
                        )
                    )
        
        # mplfinanceの設定
        kwargs = dict(
            type='candle',
            figsize=figsize,
            title=title,
            style=style,
            datetime_format='%Y-%m-%d',
            xrotation=45,
            returnfig=True,
            panel_ratios=panel_ratios,
            addplot=addplots if addplots else None
        )
        
        if show_volume:
            kwargs['volume'] = True
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 各サイクル検出器パネルに参照線を追加
        panel_start_idx = 2 if show_volume else 1
        for group_idx in range(len(detector_groups)):
            panel_idx = panel_start_idx + group_idx
            if panel_idx < len(axes):
                # 一般的なサイクル期間の参照線
                axes[panel_idx].axhline(y=20, color='gray', linestyle='--', alpha=0.5, label='Short Cycle (20)')
                axes[panel_idx].axhline(y=40, color='gray', linestyle='-', alpha=0.3, label='Medium Cycle (40)')
                axes[panel_idx].axhline(y=60, color='gray', linestyle='--', alpha=0.5, label='Long Cycle (60)')
                axes[panel_idx].legend(loc='upper right', fontsize=8)
                axes[panel_idx].set_ylim(5, 95)  # サイクル期間の表示範囲を制限
        
        self.fig = fig
        self.axes = axes
        
        # 保存または表示
        if savefig:
            plt.savefig(savefig, dpi=150, bbox_inches='tight')
            print(f"チャートを保存しました: {savefig}")
        else:
            plt.tight_layout()
            plt.show()
    
    def save_statistics_report(self, output_path: str = None) -> None:
        """
        統計レポートをCSVファイルに保存する
        
        Args:
            output_path: 出力ファイルのパス
        """
        if not self.detector_stats:
            raise ValueError("統計データがありません。calculate_all_detectors()を先に実行してください。")
        
        # CSVファイル名を生成
        if output_path is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"cycle_detectors_statistics_{timestamp}.csv"
        
        # 統計データをDataFrameに変換
        stats_data = []
        for detector_name, stats in self.detector_stats.items():
            stats_data.append({
                'Detector': detector_name,
                'Description': stats['description'],
                'Mean_Cycle': stats['mean'],
                'Std_Cycle': stats['std'],
                'Min_Cycle': stats['min'],
                'Max_Cycle': stats['max'],
                'Median_Cycle': stats['median'],
                'Valid_Count': stats['valid_count'],
                'NaN_Count': stats['nan_count'],
                'Calc_Time_Sec': stats['calc_time'],
                'Success_Rate': stats['valid_count'] / (stats['valid_count'] + stats['nan_count']) * 100 if (stats['valid_count'] + stats['nan_count']) > 0 else 0
            })
        
        stats_df = pd.DataFrame(stats_data)
        
        # パフォーマンスでソート（成功率、計算時間の組み合わせ）
        stats_df['Performance_Score'] = stats_df['Success_Rate'] / (stats_df['Calc_Time_Sec'] + 0.001)  # 0.001は0除算防止
        stats_df = stats_df.sort_values('Performance_Score', ascending=False)
        
        # CSVに保存
        stats_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"統計レポートを保存しました: {output_path}")
        
        return stats_df
    
    def print_performance_summary(self) -> None:
        """
        性能サマリーをコンソールに出力する
        """
        if not self.detector_stats:
            print("統計データがありません。")
            return
        
        print("\n" + "="*80)
        print("全サイクル検出器 性能サマリー")
        print("="*80)
        
        # 成功率でソート
        sorted_detectors = sorted(
            self.detector_stats.items(),
            key=lambda x: x[1]['valid_count'] / (x[1]['valid_count'] + x[1]['nan_count']) * 100 if (x[1]['valid_count'] + x[1]['nan_count']) > 0 else 0,
            reverse=True
        )
        
        print(f"{'検出器名':<25} {'成功率':<8} {'平均期間':<8} {'計算時間':<10} {'説明':<50}")
        print("-" * 120)
        
        for detector_name, stats in sorted_detectors:
            success_rate = stats['valid_count'] / (stats['valid_count'] + stats['nan_count']) * 100 if (stats['valid_count'] + stats['nan_count']) > 0 else 0
            mean_cycle = f"{stats['mean']:.1f}" if not np.isnan(stats['mean']) else "N/A"
            calc_time = f"{stats['calc_time']:.3f}s"
            description = stats['description'][:48] + "..." if len(stats['description']) > 48 else stats['description']
            
            print(f"{detector_name:<25} {success_rate:>6.1f}% {mean_cycle:>8} {calc_time:>10} {description:<50}")
        
        print("-" * 120)
        print(f"総検出器数: {len(self.detector_stats)}")
        print(f"平均計算時間: {np.mean([s['calc_time'] for s in self.detector_stats.values()]):.3f}秒")
        
        # 上位3つの推奨検出器
        print(f"\n🏆 推奨検出器 TOP 3:")
        for i, (detector_name, stats) in enumerate(sorted_detectors[:3]):
            success_rate = stats['valid_count'] / (stats['valid_count'] + stats['nan_count']) * 100 if (stats['valid_count'] + stats['nan_count']) > 0 else 0
            print(f"{i+1}. {detector_name} (成功率: {success_rate:.1f}%, 計算時間: {stats['calc_time']:.3f}s)")


def main():
    """メイン関数"""
    # コマンドライン引数を処理
    import argparse
    parser = argparse.ArgumentParser(description='全サイクル検出器の性能テスト')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output-chart', '-o', type=str, help='チャート出力ファイルのパス')
    parser.add_argument('--output-stats', type=str, help='統計レポート出力ファイルのパス')
    parser.add_argument('--max-detectors-per-panel', type=int, default=3, help='パネルあたりの最大検出器数')
    parser.add_argument('--no-chart', action='store_true', help='チャートを表示しない')
    args = parser.parse_args()
    
    # テストを実行
    tester = AllCycleDetectorsPerformanceTest()
    
    # データ読み込み
    tester.load_data_from_config(args.config)
    
    # 全検出器初期化
    tester.initialize_all_detectors()
    
    # 全検出器計算
    tester.calculate_all_detectors()
    
    # 性能サマリー表示
    tester.print_performance_summary()
    
    # 統計レポート保存
    stats_df = tester.save_statistics_report(args.output_stats)
    
    # チャート描画
    if not args.no_chart:
        chart_output = args.output_chart
        if chart_output is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            chart_output = f"all_cycle_detectors_comparison_{timestamp}.png"
        
        tester.plot_comparison_chart(
            start_date=args.start,
            end_date=args.end,
            max_detectors_per_panel=args.max_detectors_per_panel,
            savefig=chart_output
        )


if __name__ == "__main__":
    main()