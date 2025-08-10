#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
スムーザー比較チャートシステム

indicators/smoother/ディレクトリ内のすべてのスムーザーを
matplotlib で可視化・比較するシステム
"""

import os
import sys
import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import logging
import traceback

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# データ取得のための依存関係
try:
    from data.data_loader import DataLoader, CSVDataSource
    from data.data_processor import DataProcessor
    from data.binance_data_source import BinanceDataSource
except ImportError:
    print("Warning: Could not import data modules. Using fallback data generation.")
    DataLoader = None
    CSVDataSource = None
    DataProcessor = None
    BinanceDataSource = None


class SmootherComparisonChart:
    """
    すべてのスムーザーを比較するチャートクラス
    
    - データの読み込み
    - 各スムーザーの計算
    - matplotlib による可視化
    - パフォーマンス比較
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.smoothers = {}
        self.results = {}
        self.smoother_configs = {
            'FRAMA': {
                'module_file': 'frama.py',
                'class_name': 'FRAMA',
                'params': {'n': 16, 'fc': 1, 'sc': 200, 'src_type': 'close'},
                'color': 'blue',
                'linestyle': '-',
                'description': 'Fractal Adaptive MA'
            },
            'SuperSmoother': {
                'module_file': 'super_smoother.py',
                'class_name': 'SuperSmoother',
                'params': {'period': 20, 'src_type': 'close'},
                'color': 'green',
                'linestyle': '--',
                'description': 'Super Smoother'
            },
            'UltimateSmoother': {
                'module_file': 'ultimate_smoother.py',
                'class_name': 'UltimateSmoother',
                'params': {'period': 20, 'src_type': 'close'},
                'color': 'red',
                'linestyle': '-.',
                'description': 'Ultimate Smoother'
            },
            'AdaptiveKalman': {
                'module_file': 'adaptive_kalman.py',
                'class_name': 'AdaptiveKalman',
                'params': {'process_noise': 1e-5, 'src_type': 'close'},
                'color': 'purple',
                'linestyle': ':',
                'description': 'Adaptive Kalman Filter'
            },
            'UnscentedKalmanFilter': {
                'module_file': 'unscented_kalman_filter.py',
                'class_name': 'UnscentedKalmanFilter',
                'params': {'process_noise': 0.001, 'measurement_noise': 0.1, 'src_type': 'close'},
                'color': 'orange',
                'linestyle': '-',
                'description': 'Unscented Kalman Filter'
            }
        }
        
        # ログ設定
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _load_smoother_class(self, module_file: str, class_name: str):
        """
        スムーザークラスを動的にロード
        
        Args:
            module_file: モジュールファイル名
            class_name: クラス名
            
        Returns:
            ロードされたクラス
        """
        try:
            module_path = project_root / 'indicators' / 'smoother' / module_file
            
            # モジュールを動的にロード
            spec = importlib.util.spec_from_file_location(
                f"smoother_{class_name.lower()}", 
                module_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # クラスを取得
            smoother_class = getattr(module, class_name)
            self.logger.info(f"✓ {class_name} クラスをロードしました")
            return smoother_class
            
        except Exception as e:
            self.logger.error(f"✗ {class_name} のロードに失敗: {e}")
            return None
    
    def load_data_from_config(self, config_path: str = 'config.yaml') -> pd.DataFrame:
        """
        設定ファイルからデータを読み込む
        
        Args:
            config_path: 設定ファイルのパス
            
        Returns:
            処理済みのデータフレーム
        """
        if DataLoader is None:
            # フォールバックデータ生成
            return self._generate_sample_data()
        
        try:
            import yaml
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
            self.logger.info("データを読み込み・処理中...")
            raw_data = data_loader.load_data_from_config(config)
            processed_data = {
                symbol: data_processor.process(df)
                for symbol, df in raw_data.items()
            }
            
            # 最初のシンボルのデータを取得
            first_symbol = next(iter(processed_data))
            self.data = processed_data[first_symbol]
            
            # データを制限（計算速度のため）
            if len(self.data) > 1000:
                self.data = self.data.tail(1000)
            
            self.logger.info(f"データ読み込み完了: {first_symbol}")
            self.logger.info(f"期間: {self.data.index.min()} → {self.data.index.max()}")
            self.logger.info(f"データ数: {len(self.data)}")
            
            return self.data
            
        except Exception as e:
            self.logger.error(f"データ読み込みエラー: {e}")
            return self._generate_sample_data()
    
    def _generate_sample_data(self, n_points: int = 500) -> pd.DataFrame:
        """
        フォールバック用サンプルデータ生成
        
        Args:
            n_points: データポイント数
            
        Returns:
            サンプルデータフレーム
        """
        self.logger.info("サンプルデータを生成中...")
        
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=n_points, freq='1h')
        
        # トレンド + サイクル + ノイズの合成
        t = np.arange(n_points)
        base_price = 100
        trend = 0.02 * t
        cycle1 = 5 * np.sin(0.05 * t)
        cycle2 = 2 * np.sin(0.2 * t)
        noise = np.random.randn(n_points) * 1.0
        
        close = base_price + trend + cycle1 + cycle2 + noise
        high = close + np.abs(np.random.randn(n_points) * 0.5)
        low = close - np.abs(np.random.randn(n_points) * 0.5)
        open_price = close + np.random.randn(n_points) * 0.3
        volume = np.random.randint(1000, 5000, n_points)
        
        self.data = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=dates)
        
        self.logger.info(f"サンプルデータ生成完了: {len(self.data)}行")
        return self.data
    
    def calculate_all_smoothers(self) -> None:
        """
        すべてのスムーザーを計算
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
        
        self.logger.info("すべてのスムーザーを計算中...")
        
        for name, config in self.smoother_configs.items():
            try:
                # クラスをロード
                smoother_class = self._load_smoother_class(config['module_file'], config['class_name'])
                if smoother_class is None:
                    continue
                
                # インスタンスを作成
                smoother = smoother_class(**config['params'])
                self.smoothers[name] = smoother
                
                # 計算実行
                self.logger.info(f"  計算中: {name}")
                result = smoother.calculate(self.data)
                
                # 結果を保存（適切な値を取得）
                if hasattr(result, 'values'):
                    smoothed_values = result.values
                elif hasattr(result, 'filtered_signal'):
                    smoothed_values = result.filtered_signal
                elif hasattr(result, 'filtered_values'):
                    smoothed_values = result.filtered_values
                else:
                    # get_values() メソッドを試す
                    smoothed_values = smoother.get_values()
                
                if smoothed_values is not None and len(smoothed_values) > 0:
                    self.results[name] = {
                        'values': smoothed_values,
                        'config': config,
                        'result_object': result
                    }
                    self.logger.info(f"  ✓ {name} 計算完了 ({len(smoothed_values)} 値)")
                else:
                    self.logger.warning(f"  ✗ {name} 計算結果が無効")
                    
            except Exception as e:
                self.logger.error(f"  ✗ {name} 計算エラー: {e}")
                traceback.print_exc()
        
        self.logger.info(f"スムーザー計算完了: {len(self.results)}/{len(self.smoother_configs)} 成功")
    
    def _calculate_performance_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        各スムーザーのパフォーマンス指標を計算
        
        Returns:
            パフォーマンス指標の辞書
        """
        metrics = {}
        original_prices = self.data['close'].values
        
        for name, result_data in self.results.items():
            smoothed_values = result_data['values']
            
            # 有効な値のみを使用
            valid_indices = ~np.isnan(smoothed_values)
            if not np.any(valid_indices):
                continue
                
            valid_smoothed = smoothed_values[valid_indices]
            valid_original = original_prices[valid_indices]
            
            # ノイズ削減効果
            original_std = np.std(valid_original)
            smoothed_std = np.std(valid_smoothed)
            noise_reduction = (1 - smoothed_std / original_std) * 100 if original_std > 0 else 0
            
            # 遅延測定（相関のピーク位置）
            if len(valid_smoothed) > 10:
                correlation = np.correlate(valid_original, valid_smoothed, mode='full')
                lag = np.argmax(correlation) - len(valid_smoothed) + 1
            else:
                lag = 0
            
            # トレンド追従性（移動平均との相関）
            if len(valid_smoothed) > 20:
                ma20 = pd.Series(valid_original).rolling(20).mean().values
                valid_ma_indices = ~np.isnan(ma20)
                if np.any(valid_ma_indices):
                    trend_correlation = np.corrcoef(
                        valid_smoothed[valid_ma_indices], 
                        ma20[valid_ma_indices]
                    )[0, 1]
                else:
                    trend_correlation = 0
            else:
                trend_correlation = 0
            
            # 平滑度（変化率の標準偏差）
            smoothness = np.std(np.diff(valid_smoothed)) if len(valid_smoothed) > 1 else 0
            
            metrics[name] = {
                'noise_reduction': noise_reduction,
                'lag': abs(lag),
                'trend_correlation': trend_correlation if not np.isnan(trend_correlation) else 0,
                'smoothness': smoothness,
                'valid_points': len(valid_smoothed)
            }
        
        return metrics
    
    def plot_comparison(
        self, 
        title: str = "スムーザー比較チャート", 
        figsize: Tuple[int, int] = (16, 12),
        show_original: bool = True,
        show_performance: bool = True,
        save_path: Optional[str] = None
    ) -> None:
        """
        すべてのスムーザーを比較するチャートを描画
        
        Args:
            title: チャートのタイトル
            figsize: 図のサイズ
            show_original: 元の価格を表示するか
            show_performance: パフォーマンス指標を表示するか
            save_path: 保存先のパス（指定しない場合は表示のみ）
        """
        if not self.results:
            raise ValueError("計算結果がありません。calculate_all_smoothers()を先に実行してください。")
        
        # パフォーマンス指標を計算
        metrics = self._calculate_performance_metrics() if show_performance else {}
        
        # 図の設定
        n_subplots = 2 if show_performance else 1
        fig, axes = plt.subplots(n_subplots, 1, figsize=figsize)
        if n_subplots == 1:
            axes = [axes]
        
        # メインチャート
        ax_main = axes[0]
        
        # 元の価格データ
        dates = self.data.index
        original_prices = self.data['close'].values
        
        if show_original:
            ax_main.plot(dates, original_prices, 
                        color='gray', alpha=0.6, linewidth=1, 
                        label='Original Price', zorder=1)
        
        # 各スムーザーをプロット
        for name, result_data in self.results.items():
            config = result_data['config']
            values = result_data['values']
            
            # 有効な値のみをプロット
            valid_mask = ~np.isnan(values)
            if not np.any(valid_mask):
                continue
            
            ax_main.plot(dates[valid_mask], values[valid_mask],
                        color=config['color'], 
                        linestyle=config['linestyle'],
                        linewidth=2,
                        label=f"{name} ({config['description']})",
                        zorder=2)
        
        ax_main.set_title(title, fontsize=16, fontweight='bold')
        ax_main.set_xlabel('日時')
        ax_main.set_ylabel('価格')
        ax_main.legend(loc='upper left', fontsize=10)
        ax_main.grid(True, alpha=0.3)
        
        # 日付フォーマット
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax_main.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, len(dates)//10)))
        plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # パフォーマンス指標のチャート
        if show_performance and metrics:
            ax_perf = axes[1]
            
            names = list(metrics.keys())
            metrics_names = ['noise_reduction', 'trend_correlation', 'smoothness']
            metrics_labels = ['ノイズ削減(%)', 'トレンド相関', '平滑度']
            
            x = np.arange(len(names))
            width = 0.25
            
            for i, (metric_name, metric_label) in enumerate(zip(metrics_names, metrics_labels)):
                values = []
                for name in names:
                    val = metrics[name].get(metric_name, 0)
                    # 正規化
                    if metric_name == 'noise_reduction':
                        val = val / 100  # パーセントを0-1に
                    elif metric_name == 'smoothness':
                        val = 1 / (1 + val)  # 平滑度は逆数（小さいほど良い）
                    values.append(val)
                
                ax_perf.bar(x + i * width, values, width, 
                           label=metric_label, alpha=0.8)
            
            ax_perf.set_xlabel('スムーザー')
            ax_perf.set_ylabel('正規化スコア')
            ax_perf.set_title('パフォーマンス比較', fontsize=14)
            ax_perf.set_xticks(x + width)
            ax_perf.set_xticklabels(names, rotation=45, ha='right')
            ax_perf.legend()
            ax_perf.grid(True, alpha=0.3)
            ax_perf.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # 統計情報の表示
        print(f"\n=== スムーザー比較統計 ===")
        print(f"データ期間: {dates.min()} → {dates.max()}")
        print(f"データ数: {len(original_prices)}")
        print(f"元価格統計: 平均={np.mean(original_prices):.2f}, 標準偏差={np.std(original_prices):.2f}")
        
        for name, result_data in self.results.items():
            values = result_data['values']
            valid_values = values[~np.isnan(values)]
            if len(valid_values) > 0:
                print(f"\n{name}:")
                print(f"  有効値数: {len(valid_values)}")
                print(f"  平均: {np.mean(valid_values):.2f}")
                print(f"  標準偏差: {np.std(valid_values):.2f}")
                if name in metrics:
                    m = metrics[name]
                    print(f"  ノイズ削減: {m['noise_reduction']:.2f}%")
                    print(f"  遅延: {m['lag']} ポイント")
                    print(f"  トレンド相関: {m['trend_correlation']:.4f}")
        
        # 保存または表示
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nチャートを保存しました: {save_path}")
        
        plt.show()
    
    def reset(self) -> None:
        """
        すべての状態をリセット
        """
        self.data = None
        self.smoothers = {}
        self.results = {}


def main():
    """メイン関数"""
    import argparse
    parser = argparse.ArgumentParser(description='スムーザー比較チャートの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--no-original', action='store_true', help='元の価格を非表示')
    parser.add_argument('--no-performance', action='store_true', help='パフォーマンス指標を非表示')
    parser.add_argument('--sample-data', action='store_true', help='サンプルデータを使用')
    args = parser.parse_args()
    
    # チャートを作成
    chart = SmootherComparisonChart()
    
    if args.sample_data:
        chart._generate_sample_data()
    else:
        try:
            chart.load_data_from_config(args.config)
        except Exception as e:
            print(f"設定ファイルの読み込みに失敗。サンプルデータを使用します: {e}")
            chart._generate_sample_data()
    
    chart.calculate_all_smoothers()
    chart.plot_comparison(
        title="スムーザー総合比較 - indicators/smoother/ 全手法",
        show_original=not args.no_original,
        show_performance=not args.no_performance,
        save_path=args.output
    )


if __name__ == "__main__":
    main()