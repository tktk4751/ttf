#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🧠 **Four Adaptive UKF Methods - Real Market Data Chart** 🧠

実際の相場データで4つのAdaptive UKF手法を比較表示：
1. 標準UKF (基準)
2. 私の実装版AUKF (統計的監視・適応制御)
3. 論文版AUKF (Ge et al. 2019 - 相互相関理論)
4. Neural版AUKF (Levy & Klein 2025 - CNN ProcessNet)
"""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import warnings
import sys

# パス設定
project_root = Path(__file__).parent.parent
if project_root not in sys.path:
    sys.path.append(str(project_root))

# データ取得のための依存関係
try:
    from data.data_loader import DataLoader, CSVDataSource
    from data.data_processor import DataProcessor
    from data.binance_data_source import BinanceDataSource
except ImportError:
    print("⚠️ データソースモジュールが見つかりません。パスを調整してください。")

# 4つのカルマンフィルター
try:
    from indicators.unscented_kalman_filter import UnscentedKalmanFilter
    from indicators.adaptive_ukf import AdaptiveUnscentedKalmanFilter
    from indicators.academic_adaptive_ukf import AcademicAdaptiveUnscentedKalmanFilter
    from indicators.neural_adaptive_ukf import NeuralAdaptiveUnscentedKalmanFilter
except ImportError as e:
    print(f"⚠️ カルマンフィルターモジュールインポートエラー: {e}")

# 設定
warnings.filterwarnings('ignore')


class AdaptiveUKFComparisonChart:
    """
    4つのAdaptive UKF手法を実際の相場データで比較表示するチャートクラス
    
    特徴：
    - ローソク足と出来高
    - 4つのフィルター結果の重ね合わせ表示
    - 性能指標の比較パネル
    - 適応性指標の表示
    - リアルタイム相場データ対応
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.filters = {}
        self.results = {}
        self.performance_metrics = {}
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
        print("🧠 **Four Adaptive UKF Comparison Chart** 🧠")
        print("="*60)
        
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
        print("📊 データを読み込み・処理中...")
        raw_data = data_loader.load_data_from_config(config)
        processed_data = {
            symbol: data_processor.process(df)
            for symbol, df in raw_data.items()
        }
        
        # 最初のシンボルのデータを取得
        first_symbol = next(iter(processed_data))
        self.data = processed_data[first_symbol]
        
        print(f"✅ データ読み込み完了: {first_symbol}")
        print(f"   期間: {self.data.index.min()} → {self.data.index.max()}")
        print(f"   データ数: {len(self.data)}")
        print(f"   価格範囲: ${self.data['close'].min():.2f} - ${self.data['close'].max():.2f}")
        
        return self.data
    
    def load_simple_data(self, csv_path: str) -> pd.DataFrame:
        """
        簡単なCSVファイルからデータを読み込む（テスト用）
        
        Args:
            csv_path: CSVファイルのパス
            
        Returns:
            処理済みのデータフレーム
        """
        print("📊 **CSVデータ読み込み中...**")
        
        try:
            # CSVファイル読み込み
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            
            # 必要な列が存在するかチェック
            required_cols = ['open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"⚠️ 不足している列: {missing_cols}")
                # closeのみで疑似OHLCデータ作成
                if 'close' in df.columns:
                    df['open'] = df['close'].shift(1).fillna(df['close'])
                    df['high'] = df['close'] * 1.01
                    df['low'] = df['close'] * 0.99
                else:
                    raise ValueError("closeカラムが見つかりません")
            
            # volume列がない場合は作成
            if 'volume' not in df.columns:
                df['volume'] = 1000000  # ダミーボリューム
            
            self.data = df
            
            print(f"✅ データ読み込み完了")
            print(f"   期間: {self.data.index.min()} → {self.data.index.max()}")
            print(f"   データ数: {len(self.data)}")
            print(f"   価格範囲: ${self.data['close'].min():.2f} - ${self.data['close'].max():.2f}")
            
            return self.data
            
        except Exception as e:
            print(f"❌ データ読み込みエラー: {str(e)}")
            raise

    def calculate_filters(self,
                         src_type: str = 'close',
                         window_size: int = 100) -> None:
        """
        4つのAdaptive UKF手法を計算する
        
        Args:
            src_type: 価格ソースタイプ
            window_size: Neural版のウィンドウサイズ
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。")
            
        print("\n🔄 **4つのAdaptive UKF手法を計算中...**")
        
        # 価格データ抽出
        if src_type in self.data.columns:
            price_data = self.data[src_type].values
        else:
            price_data = self.data['close'].values
            print(f"   ⚠️ {src_type}が見つからないため、closeを使用")
        
        print(f"   📈 使用価格: {src_type}, データ数: {len(price_data)}")
        
        # 1. 標準UKF（基準）
        print("   1️⃣ 標準UKF計算中...")
        try:
            self.filters['Standard UKF'] = UnscentedKalmanFilter(src_type=src_type)
            self.results['Standard UKF'] = self.filters['Standard UKF'].calculate(self.data)
            print("      ✅ 標準UKF完了")
        except Exception as e:
            print(f"      ❌ 標準UKFエラー: {str(e)}")
            self.results['Standard UKF'] = None
        
        # 2. 私の実装版AUKF（統計的監視・適応制御）
        print("   2️⃣ 私の実装版AUKF計算中...")
        try:
            self.filters['My AUKF'] = AdaptiveUnscentedKalmanFilter(src_type=src_type)
            self.results['My AUKF'] = self.filters['My AUKF'].calculate(self.data)
            print("      ✅ 私の実装版AUKF完了")
        except Exception as e:
            print(f"      ❌ 私の実装版AUKFエラー: {str(e)}")
            self.results['My AUKF'] = None
        
        # 3. 論文版AUKF（Ge et al. 2019 - 相互相関理論）
        print("   3️⃣ 論文版AUKF (Ge et al. 2019) 計算中...")
        try:
            self.filters['Academic AUKF'] = AcademicAdaptiveUnscentedKalmanFilter(src_type=src_type)
            self.results['Academic AUKF'] = self.filters['Academic AUKF'].calculate(self.data)
            print("      ✅ 論文版AUKF完了")
        except Exception as e:
            print(f"      ❌ 論文版AUKFエラー: {str(e)}")
            self.results['Academic AUKF'] = None
        
        # 4. Neural版AUKF（Levy & Klein 2025 - CNN ProcessNet）
        print("   4️⃣ Neural版AUKF (Levy & Klein 2025) 計算中...")
        try:
            self.filters['Neural AUKF'] = NeuralAdaptiveUnscentedKalmanFilter(
                src_type=src_type, 
                window_size=window_size
            )
            self.results['Neural AUKF'] = self.filters['Neural AUKF'].calculate(self.data)
            print("      ✅ Neural版AUKF完了")
        except Exception as e:
            print(f"      ❌ Neural版AUKFエラー: {str(e)}")
            self.results['Neural AUKF'] = None
        
        # 性能指標計算
        self._calculate_performance_metrics(price_data)
        
        print("🎉 **4つのAdaptive UKF計算完了！**")
    
    def _calculate_performance_metrics(self, true_prices: np.ndarray) -> None:
        """性能指標を計算"""
        print("\n📊 **性能指標計算中...**")
        
        valid_results = {name: result for name, result in self.results.items() if result is not None}
        
        for method_name, result in valid_results.items():
            filtered_values = result.filtered_values
            
            # RMSE計算
            if len(filtered_values) == len(true_prices):
                rmse = np.sqrt(np.mean((filtered_values - true_prices) ** 2))
                mae = np.mean(np.abs(filtered_values - true_prices))
                
                # 遅延計算（相関のピーク位置）
                correlation = np.correlate(filtered_values, true_prices, mode='full')
                lag = np.argmax(correlation) - len(true_prices) + 1
                
                # 平滑性（二階差分の標準偏差）
                smoothness = np.std(np.diff(filtered_values, n=2))
                
                self.performance_metrics[method_name] = {
                    'RMSE': rmse,
                    'MAE': mae,
                    'Lag': abs(lag),
                    'Smoothness': smoothness,
                    'Filter_Type': self._get_filter_type(method_name)
                }
                
                print(f"   📈 {method_name}: RMSE={rmse:.4f}, MAE={mae:.4f}, Lag={abs(lag)}, Smoothness={smoothness:.4f}")
            else:
                print(f"   ⚠️ {method_name}: データ長不一致")
        
        # 基準（標準UKF）からの改善率計算
        if 'Standard UKF' in self.performance_metrics:
            baseline_rmse = self.performance_metrics['Standard UKF']['RMSE']
            for method_name, metrics in self.performance_metrics.items():
                if method_name != 'Standard UKF':
                    improvement = (baseline_rmse - metrics['RMSE']) / baseline_rmse * 100
                    metrics['Improvement'] = improvement
                    print(f"   🚀 {method_name}: 改善率 {improvement:+.1f}%")
        
        print("✅ 性能指標計算完了")
    
    def _get_filter_type(self, method_name: str) -> str:
        """フィルタータイプを取得"""
        if 'Standard' in method_name:
            return 'baseline'
        elif 'My' in method_name:
            return 'statistical_adaptive'
        elif 'Academic' in method_name:
            return 'mathematical_rigorous'
        elif 'Neural' in method_name:
            return 'neural_adaptive'
        else:
            return 'unknown'
    
    def plot(self, 
            title: str = "4つのAdaptive UKF比較 - 実際の相場データ", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (16, 14),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートと4つのAdaptive UKF手法を描画する
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。")
            
        if not self.results:
            raise ValueError("フィルターが計算されていません。calculate_filters()を先に実行してください。")
        
        print("\n🎨 **チャート描画中...**")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # フィルター結果をデータフレームに追加
        valid_results = {name: result for name, result in self.results.items() if result is not None}
        
        for method_name, result in valid_results.items():
            # フィルター値
            if len(result.filtered_values) == len(self.data):
                full_df = pd.DataFrame(
                    index=self.data.index,
                    data={f'{method_name}_filtered': result.filtered_values}
                )
                df = df.join(full_df)
                
                # 不確実性（利用可能な場合）
                if hasattr(result, 'uncertainty'):
                    uncertainty_df = pd.DataFrame(
                        index=self.data.index,
                        data={f'{method_name}_uncertainty': result.uncertainty}
                    )
                    df = df.join(uncertainty_df)
        
        print(f"   📊 チャートデータ準備完了 - 行数: {len(df)}")
        
        # フィルター表示用の色設定
        filter_colors = {
            'Standard UKF': 'gray',
            'My AUKF': 'blue',
            'Academic AUKF': 'red',
            'Neural AUKF': 'green'
        }
        
        # mplfinanceでプロット用の設定
        main_plots = []
        
        # 各フィルターの結果をメインチャートに追加
        for method_name in valid_results.keys():
            col_name = f'{method_name}_filtered'
            if col_name in df.columns:
                color = filter_colors.get(method_name, 'purple')
                alpha = 0.6 if 'Standard' in method_name else 0.8
                width = 1.5 if 'Standard' in method_name else 2.0
                
                main_plots.append(
                    mpf.make_addplot(df[col_name], color=color, width=width, alpha=alpha, label=method_name)
                )
        
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
        
        # 出来高設定
        if show_volume and 'volume' in df.columns:
            kwargs['volume'] = True
        else:
            kwargs['volume'] = False
        
        # プロット設定
        if main_plots:
            kwargs['addplot'] = main_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加
        if main_plots:
            legend_labels = list(valid_results.keys())
            axes[0].legend(legend_labels, loc='upper left', fontsize=9)
        
        self.fig = fig
        self.axes = axes
        
        # 統計情報の表示
        self._print_chart_statistics(df, valid_results)
        
        # 保存または表示
        if savefig:
            plt.savefig(savefig, dpi=150, bbox_inches='tight')
            print(f"📊 チャートを保存しました: {savefig}")
        else:
            plt.tight_layout()
            plt.show()
    
    def _print_chart_statistics(self, df: pd.DataFrame, valid_results: dict) -> None:
        """チャート統計情報を表示"""
        print(f"\n📊 **チャート統計情報**")
        print(f"   📅 表示期間: {df.index.min()} → {df.index.max()}")
        print(f"   📈 価格範囲: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        print(f"   📊 データ数: {len(df)}")
        
        print(f"\n🎯 **フィルター性能ランキング:**")
        if self.performance_metrics:
            # RMSE順でソート
            sorted_metrics = sorted(self.performance_metrics.items(), key=lambda x: x[1]['RMSE'])
            for i, (method_name, metrics) in enumerate(sorted_metrics, 1):
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}位"
                improvement = metrics.get('Improvement', 0)
                print(f"   {medal} {method_name}: RMSE={metrics['RMSE']:.4f} ({improvement:+.1f}%)")
    
    def get_performance_summary(self) -> pd.DataFrame:
        """性能指標のサマリーをDataFrameとして取得"""
        if not self.performance_metrics:
            return pd.DataFrame()
        
        return pd.DataFrame(self.performance_metrics).T


def create_test_config() -> str:
    """テスト用の設定ファイルを作成"""
    config_content = """
# 4つのAdaptive UKF比較テスト用設定

binance_data:
  data_dir: "data/binance"
  symbols:
    - "BTCUSDT"
  intervals:
    - "1h"
  start_date: "2024-01-01"
  end_date: "2024-06-01"
  limit: 1000

data_processing:
  remove_duplicates: true
  fill_missing: true
  min_volume: 0
"""
    
    config_path = "config_adaptive_ukf_test.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"✅ テスト用設定ファイルを作成: {config_path}")
    return config_path


def create_sample_data(filename: str = "sample_market_data.csv", n_points: int = 500) -> str:
    """サンプル相場データを作成"""
    print("📊 サンプル相場データ作成中...")
    
    # 時系列インデックス作成
    dates = pd.date_range(start='2024-01-01', periods=n_points, freq='1H')
    
    # 価格データ生成（ランダムウォーク + トレンド）
    np.random.seed(42)
    base_price = 50000  # 初期価格
    
    # トレンド成分
    trend = np.linspace(0, 0.2, n_points)
    
    # ランダムウォーク
    returns = np.random.normal(0, 0.02, n_points)
    log_prices = np.log(base_price) + np.cumsum(returns) + trend
    prices = np.exp(log_prices)
    
    # OHLC生成
    df = pd.DataFrame(index=dates)
    df['close'] = prices
    df['open'] = df['close'].shift(1).fillna(df['close'])
    
    # High/Low生成（Closeを基準に±1-3%の範囲）
    high_factor = 1 + np.random.uniform(0.001, 0.03, n_points)
    low_factor = 1 - np.random.uniform(0.001, 0.03, n_points)
    
    df['high'] = np.maximum(df['close'] * high_factor, np.maximum(df['close'], df['open']))
    df['low'] = np.minimum(df['close'] * low_factor, np.minimum(df['close'], df['open']))
    
    # ボリューム生成
    df['volume'] = np.random.uniform(100000, 1000000, n_points)
    
    # 保存
    df.to_csv(filename)
    print(f"✅ サンプルデータ保存: {filename}")
    return filename


def main():
    """メイン関数 - テスト実行"""
    import argparse
    
    parser = argparse.ArgumentParser(description='4つのAdaptive UKF手法の相場データ比較')
    parser.add_argument('--config', '-c', type=str, help='設定ファイルのパス（未指定時はテスト用設定を作成）')
    parser.add_argument('--csv', type=str, help='CSVファイルのパス（設定ファイルの代わりに使用）')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--src-type', type=str, default='close', help='価格ソースタイプ')
    parser.add_argument('--window-size', type=int, default=50, help='Neural版のウィンドウサイズ')
    parser.add_argument('--create-sample', action='store_true', help='サンプルデータ作成テスト')
    args = parser.parse_args()
    
    # サンプルデータ作成テスト
    if args.create_sample:
        sample_file = create_sample_data()
        args.csv = sample_file
    
    try:
        # チャートを作成
        chart = AdaptiveUKFComparisonChart()
        
        # データ読み込み
        if args.csv:
            # CSVから読み込み
            chart.load_simple_data(args.csv)
        elif args.config:
            # 設定ファイルから読み込み
            chart.load_data_from_config(args.config)
        else:
            # テスト用設定作成
            print("⚠️ データソースが指定されていません。サンプルデータを作成します...")
            sample_file = create_sample_data()
            chart.load_simple_data(sample_file)
        
        # フィルター計算
        chart.calculate_filters(
            src_type=args.src_type,
            window_size=args.window_size
        )
        
        # チャート描画
        chart.plot(
            start_date=args.start,
            end_date=args.end,
            savefig=args.output
        )
        
        # 性能サマリー保存
        performance_df = chart.get_performance_summary()
        if not performance_df.empty:
            summary_path = "output/adaptive_ukf_market_performance.csv"
            os.makedirs("output", exist_ok=True)
            performance_df.to_csv(summary_path)
            print(f"📊 性能サマリー保存: {summary_path}")
        
        print("\n🎉 **4つのAdaptive UKF相場データ比較完了！**")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {str(e)}")
        print("   - データディレクトリが存在することを確認してください")
        print("   - 設定ファイルが正しい形式であることを確認してください")
        print("   - 必要なモジュールがインストールされていることを確認してください")


if __name__ == "__main__":
    main() 