#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🚀 **4つのAdaptive UKF手法 - 実際の相場データテスト** 🚀

このテストファイルは以下の機能を提供します：
1. 実際の相場データでの4つのAdaptive UKF手法の比較
2. 複数のテストシナリオ
3. 詳細な性能分析
4. 可視化とレポート生成
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 必要なモジュールのインポート
try:
    from visualization.adaptive_ukf_comparison_chart import AdaptiveUKFComparisonChart, create_sample_data
    from indicators.unscented_kalman_filter import UnscentedKalmanFilter
    from indicators.adaptive_ukf import AdaptiveUnscentedKalmanFilter
    from indicators.academic_adaptive_ukf import AcademicAdaptiveUnscentedKalmanFilter
    from indicators.neural_adaptive_ukf import NeuralAdaptiveUnscentedKalmanFilter
except ImportError as e:
    print(f"⚠️ インポートエラー: {e}")
    print("   必要なモジュールがインストールされていることを確認してください。")

warnings.filterwarnings('ignore')


class AdaptiveUKFMarketTester:
    """
    4つのAdaptive UKF手法の相場データテスター
    """
    
    def __init__(self, output_dir: str = "output/market_tests"):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリ
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.test_results = {}
        self.charts = {}
        
        print("🚀 **Adaptive UKF Market Tester** 🚀")
        print("="*50)
    
    def run_comprehensive_test(self) -> None:
        """包括的なテストを実行"""
        print("\n🔥 **包括的テスト開始** 🔥")
        
        # テストシナリオ定義
        test_scenarios = [
            {
                'name': 'サンプルデータ（トレンド）',
                'type': 'sample',
                'points': 500,
                'description': '人工的なトレンドを含むサンプルデータ'
            },
            {
                'name': 'サンプルデータ（ボラティリティ）',
                'type': 'sample_volatile',
                'points': 800,
                'description': '高ボラティリティのサンプルデータ'
            },
            {
                'name': 'サンプルデータ（サイクリック）',
                'type': 'sample_cyclic',
                'points': 1000,
                'description': 'サイクリックなパターンのサンプルデータ'
            }
        ]
        
        # 各テストシナリオを実行
        for scenario in test_scenarios:
            print(f"\n📊 **テストシナリオ: {scenario['name']}**")
            print(f"   説明: {scenario['description']}")
            
            try:
                # データ作成
                data_file = self._create_test_data(scenario)
                
                # チャート作成・テスト実行
                self._run_single_test(scenario['name'], data_file)
                
                print(f"   ✅ {scenario['name']} テスト完了")
                
            except Exception as e:
                print(f"   ❌ {scenario['name']} テストエラー: {str(e)}")
                continue
        
        # 結果サマリー
        self._generate_summary_report()
        
        print("\n🎉 **包括的テスト完了！**")
    
    def _create_test_data(self, scenario: dict) -> str:
        """テストデータを作成"""
        scenario_type = scenario['type']
        points = scenario['points']
        
        filename = f"test_data_{scenario_type}.csv"
        filepath = self.output_dir / filename
        
        # 時系列インデックス作成
        dates = pd.date_range(start='2024-01-01', periods=points, freq='1H')
        
        if scenario_type == 'sample':
            # 標準的なトレンドデータ
            df = self._create_trend_data(dates, points)
        elif scenario_type == 'sample_volatile':
            # 高ボラティリティデータ
            df = self._create_volatile_data(dates, points)
        elif scenario_type == 'sample_cyclic':
            # サイクリックデータ
            df = self._create_cyclic_data(dates, points)
        else:
            # デフォルト
            df = self._create_trend_data(dates, points)
        
        # 保存
        df.to_csv(filepath)
        print(f"   📁 テストデータ保存: {filepath}")
        
        return str(filepath)
    
    def _create_trend_data(self, dates: pd.DatetimeIndex, points: int) -> pd.DataFrame:
        """トレンドデータを作成"""
        np.random.seed(42)
        base_price = 50000
        
        # トレンド成分
        trend = np.linspace(0, 0.3, points)
        
        # ランダムウォーク
        returns = np.random.normal(0, 0.015, points)
        log_prices = np.log(base_price) + np.cumsum(returns) + trend
        prices = np.exp(log_prices)
        
        return self._create_ohlc_from_prices(dates, prices)
    
    def _create_volatile_data(self, dates: pd.DatetimeIndex, points: int) -> pd.DataFrame:
        """高ボラティリティデータを作成"""
        np.random.seed(123)
        base_price = 45000
        
        # 高ボラティリティ
        volatility = np.random.uniform(0.02, 0.05, points)
        returns = np.random.normal(0, volatility)
        
        # 時々のスパイク
        spike_indices = np.random.choice(points, size=int(points * 0.1), replace=False)
        returns[spike_indices] *= np.random.choice([-1, 1], size=len(spike_indices)) * 3
        
        log_prices = np.log(base_price) + np.cumsum(returns)
        prices = np.exp(log_prices)
        
        return self._create_ohlc_from_prices(dates, prices)
    
    def _create_cyclic_data(self, dates: pd.DatetimeIndex, points: int) -> pd.DataFrame:
        """サイクリックデータを作成"""
        np.random.seed(456)
        base_price = 55000
        
        # サイクリック成分
        t = np.linspace(0, 4*np.pi, points)
        cycle1 = 0.1 * np.sin(t) + 0.05 * np.sin(2*t)
        cycle2 = 0.03 * np.sin(0.5*t)
        
        # ランダムウォーク
        returns = np.random.normal(0, 0.01, points)
        log_prices = np.log(base_price) + np.cumsum(returns) + cycle1 + cycle2
        prices = np.exp(log_prices)
        
        return self._create_ohlc_from_prices(dates, prices)
    
    def _create_ohlc_from_prices(self, dates: pd.DatetimeIndex, prices: np.ndarray) -> pd.DataFrame:
        """価格からOHLCデータを作成"""
        df = pd.DataFrame(index=dates)
        df['close'] = prices
        df['open'] = df['close'].shift(1).fillna(df['close'])
        
        # High/Low生成
        high_factor = 1 + np.random.uniform(0.001, 0.03, len(prices))
        low_factor = 1 - np.random.uniform(0.001, 0.03, len(prices))
        
        df['high'] = np.maximum(df['close'] * high_factor, np.maximum(df['close'], df['open']))
        df['low'] = np.minimum(df['close'] * low_factor, np.minimum(df['close'], df['open']))
        
        # ボリューム
        df['volume'] = np.random.uniform(100000, 1000000, len(prices))
        
        return df
    
    def _run_single_test(self, test_name: str, data_file: str) -> None:
        """単一テストを実行"""
        print(f"   🔄 フィルター計算中...")
        
        # チャート作成
        chart = AdaptiveUKFComparisonChart()
        chart.load_simple_data(data_file)
        
        # フィルター計算
        chart.calculate_filters(src_type='close', window_size=50)
        
        # チャート描画
        output_path = self.output_dir / f"{test_name.replace(' ', '_')}_chart.png"
        chart.plot(
            title=f"4つのAdaptive UKF比較 - {test_name}",
            savefig=str(output_path)
        )
        
        # 性能データ保存
        performance_df = chart.get_performance_summary()
        if not performance_df.empty:
            csv_path = self.output_dir / f"{test_name.replace(' ', '_')}_performance.csv"
            performance_df.to_csv(csv_path)
            print(f"   📊 性能データ保存: {csv_path}")
        
        # 結果保存
        self.test_results[test_name] = {
            'chart': chart,
            'performance': performance_df,
            'chart_path': str(output_path),
            'data_file': data_file
        }
        
        print(f"   📈 チャート保存: {output_path}")
    
    def _generate_summary_report(self) -> None:
        """サマリーレポートを生成"""
        print("\n📋 **サマリーレポート作成中...**")
        
        # 全テストの性能指標を統合
        all_performance = []
        
        for test_name, results in self.test_results.items():
            performance_df = results['performance']
            if not performance_df.empty:
                performance_df['Test_Scenario'] = test_name
                all_performance.append(performance_df)
        
        if all_performance:
            # 統合データフレーム作成
            combined_df = pd.concat(all_performance, axis=0)
            
            # サマリー保存
            summary_path = self.output_dir / "comprehensive_test_summary.csv"
            combined_df.to_csv(summary_path)
            
            # 統計サマリー
            stats_summary = self._calculate_summary_statistics(combined_df)
            stats_path = self.output_dir / "performance_statistics.csv"
            stats_summary.to_csv(stats_path)
            
            print(f"   📊 統合レポート保存: {summary_path}")
            print(f"   📈 統計サマリー保存: {stats_path}")
            
            # コンソール出力
            self._print_summary_statistics(stats_summary)
        
        # テスト結果のインデックスファイル作成
        self._create_index_file()
    
    def _calculate_summary_statistics(self, combined_df: pd.DataFrame) -> pd.DataFrame:
        """統計サマリーを計算"""
        # 手法別のグループ化
        method_stats = []
        
        for method in combined_df.index.unique():
            method_data = combined_df.loc[method]
            
            if isinstance(method_data, pd.Series):
                method_data = method_data.to_frame().T
            
            stats = {
                'Method': method,
                'Avg_RMSE': method_data['RMSE'].mean(),
                'Min_RMSE': method_data['RMSE'].min(),
                'Max_RMSE': method_data['RMSE'].max(),
                'Std_RMSE': method_data['RMSE'].std(),
                'Avg_MAE': method_data['MAE'].mean(),
                'Avg_Improvement': method_data.get('Improvement', pd.Series([0])).mean(),
                'Test_Count': len(method_data)
            }
            
            method_stats.append(stats)
        
        return pd.DataFrame(method_stats)
    
    def _print_summary_statistics(self, stats_df: pd.DataFrame) -> None:
        """統計サマリーをコンソールに出力"""
        print("\n📊 **全体統計サマリー**")
        print("="*60)
        
        # RMSE順でソート
        sorted_stats = stats_df.sort_values('Avg_RMSE')
        
        for i, row in sorted_stats.iterrows():
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}位"
            print(f"   {medal} {row['Method']}")
            print(f"      平均RMSE: {row['Avg_RMSE']:.4f} (±{row['Std_RMSE']:.4f})")
            print(f"      平均MAE: {row['Avg_MAE']:.4f}")
            print(f"      平均改善率: {row['Avg_Improvement']:+.1f}%")
            print(f"      テスト数: {row['Test_Count']}")
    
    def _create_index_file(self) -> None:
        """インデックスファイルを作成"""
        index_path = self.output_dir / "test_index.md"
        
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write("# 4つのAdaptive UKF手法 - 相場データテスト結果\n\n")
            f.write("## テスト概要\n")
            f.write("このテストは4つのAdaptive UKF手法を様々な相場データで比較評価したものです。\n\n")
            
            f.write("## 比較手法\n")
            f.write("1. **標準UKF** - 基準手法\n")
            f.write("2. **私の実装版AUKF** - 統計的監視・適応制御\n")
            f.write("3. **論文版AUKF** - Ge et al. (2019) 相互相関理論\n")
            f.write("4. **Neural版AUKF** - Levy & Klein (2025) CNN ProcessNet\n\n")
            
            f.write("## テスト結果\n")
            for test_name, results in self.test_results.items():
                f.write(f"### {test_name}\n")
                f.write(f"- チャート: [{test_name}_chart.png]({Path(results['chart_path']).name})\n")
                if not results['performance'].empty:
                    f.write(f"- 性能データ: [{test_name}_performance.csv]({test_name.replace(' ', '_')}_performance.csv)\n")
                f.write("\n")
            
            f.write("## 統合結果\n")
            f.write("- [統合レポート](comprehensive_test_summary.csv)\n")
            f.write("- [統計サマリー](performance_statistics.csv)\n")
        
        print(f"   📋 インデックスファイル作成: {index_path}")


def run_quick_test():
    """クイックテスト"""
    print("⚡ **クイックテスト開始** ⚡")
    
    try:
        # サンプルデータ作成
        sample_file = create_sample_data("quick_test_data.csv", 300)
        
        # チャート作成
        chart = AdaptiveUKFComparisonChart()
        chart.load_simple_data(sample_file)
        chart.calculate_filters(src_type='close', window_size=30)
        
        # チャート表示
        chart.plot(title="4つのAdaptive UKF - クイックテスト")
        
        # 性能サマリー表示
        performance_df = chart.get_performance_summary()
        if not performance_df.empty:
            print("\n📊 **性能サマリー**")
            print(performance_df.round(4))
        
        # クリーンアップ
        os.remove(sample_file)
        
        print("\n✅ クイックテスト完了！")
        
    except Exception as e:
        print(f"❌ クイックテストエラー: {str(e)}")


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='4つのAdaptive UKF手法の相場データテスト')
    parser.add_argument('--quick', action='store_true', help='クイックテスト実行')
    parser.add_argument('--comprehensive', action='store_true', help='包括的テスト実行')
    parser.add_argument('--output-dir', type=str, default='output/market_tests', help='出力ディレクトリ')
    args = parser.parse_args()
    
    if args.quick:
        run_quick_test()
    elif args.comprehensive:
        tester = AdaptiveUKFMarketTester(args.output_dir)
        tester.run_comprehensive_test()
    else:
        print("⚠️ テストモードを選択してください:")
        print("   --quick: クイックテスト")
        print("   --comprehensive: 包括的テスト")
        print("\n例:")
        print("   python adaptive_ukf_market_test.py --quick")
        print("   python adaptive_ukf_market_test.py --comprehensive")


if __name__ == "__main__":
    main() 