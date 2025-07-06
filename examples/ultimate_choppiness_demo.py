#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Ultimate Choppiness Index Demo
ジョン・エラーズのアルティメットチョピネスインデックス実演

従来のチョピネスインデックスとの比較分析:
- 遅延特性の比較
- 精度の比較
- 適応性の比較
- パフォーマンス測定
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time

# パスの設定
import sys
sys.path.append(str(Path(__file__).parent.parent))

from indicators.ultimate_choppiness_index import UltimateChoppinessIndex
from indicators.choppiness import ChoppinessIndex

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        plt.style.use('default')

class ChoppinessComparison:
    """チョピネスインデックス比較分析クラス"""
    
    def __init__(self, data: pd.DataFrame):
        """
        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV価格データ
        """
        self.data = data
        self.results = {}
        
        # インジケーター初期化
        self.ultimate_chop = UltimateChoppinessIndex(
            base_period=14,
            min_period=8,
            max_period=50,
            smoothing_period=3
        )
        
        self.classic_chop = ChoppinessIndex(period=14)
        
    def run_comparison(self):
        """比較分析を実行"""
        print("🚀 Ultimate Choppiness Index vs Classic Choppiness Index")
        print("=" * 60)
        
        # 1. 計算時間の比較
        self._benchmark_performance()
        
        # 2. 遅延特性の比較
        self._analyze_lag_characteristics()
        
        # 3. 精度の比較
        self._analyze_accuracy()
        
        # 4. 適応性の比較
        self._analyze_adaptivity()
        
        # 5. 視覚化
        self._create_visualizations()
        
        return self.results
        
    def _benchmark_performance(self):
        """パフォーマンス測定"""
        print("\n📊 Performance Benchmark")
        print("-" * 30)
        
        # Ultimate Choppiness Index
        start_time = time.time()
        ultimate_values = self.ultimate_chop.calculate(self.data)
        ultimate_time = time.time() - start_time
        
        # Classic Choppiness Index
        start_time = time.time()
        classic_values = self.classic_chop.calculate(self.data)
        classic_time = time.time() - start_time
        
        print(f"Ultimate Choppiness: {ultimate_time:.4f}秒")
        print(f"Classic Choppiness:  {classic_time:.4f}秒")
        print(f"速度比: {classic_time/ultimate_time:.1f}x faster")
        
        # 結果保存
        self.results['performance'] = {
            'ultimate_time': ultimate_time,
            'classic_time': classic_time,
            'speed_ratio': classic_time/ultimate_time,
            'ultimate_values': ultimate_values,
            'classic_values': classic_values
        }
        
    def _analyze_lag_characteristics(self):
        """遅延特性分析"""
        print("\n⚡ Lag Analysis")
        print("-" * 30)
        
        # 急激な価格変化ポイントを検出
        price_changes = np.abs(np.diff(self.data['close']))
        significant_changes = np.where(price_changes > np.percentile(price_changes, 95))[0]
        
        ultimate_values = self.results['performance']['ultimate_values']
        classic_values = self.results['performance']['classic_values']
        
        # 応答時間の計算
        ultimate_response_times = []
        classic_response_times = []
        
        for change_idx in significant_changes:
            if change_idx < len(ultimate_values) - 10:
                # 変化後の応答時間を測定
                ultimate_response = self._calculate_response_time(
                    ultimate_values[change_idx:change_idx+10]
                )
                classic_response = self._calculate_response_time(
                    classic_values[change_idx:change_idx+10]
                )
                
                ultimate_response_times.append(ultimate_response)
                classic_response_times.append(classic_response)
        
        avg_ultimate_lag = np.mean(ultimate_response_times)
        avg_classic_lag = np.mean(classic_response_times)
        
        print(f"Ultimate Choppiness平均遅延: {avg_ultimate_lag:.1f}バー")
        print(f"Classic Choppiness平均遅延:  {avg_classic_lag:.1f}バー")
        print(f"遅延改善: {(avg_classic_lag - avg_ultimate_lag)/avg_classic_lag*100:.1f}%")
        
        self.results['lag_analysis'] = {
            'ultimate_lag': avg_ultimate_lag,
            'classic_lag': avg_classic_lag,
            'improvement': (avg_classic_lag - avg_ultimate_lag)/avg_classic_lag*100
        }
        
    def _calculate_response_time(self, values):
        """応答時間計算"""
        if len(values) < 2:
            return 0
        
        initial_value = values[0]
        threshold = 0.1  # 10%の変化を検出
        
        for i, value in enumerate(values[1:], 1):
            if abs(value - initial_value) > threshold:
                return i
        
        return len(values)
        
    def _analyze_accuracy(self):
        """精度分析"""
        print("\n🎯 Accuracy Analysis")
        print("-" * 30)
        
        ultimate_result = self.ultimate_chop.get_result()
        
        # 信頼度の統計
        confidence_stats = {
            'mean': np.nanmean(ultimate_result.confidence),
            'std': np.nanstd(ultimate_result.confidence),
            'min': np.nanmin(ultimate_result.confidence),
            'max': np.nanmax(ultimate_result.confidence)
        }
        
        high_confidence_ratio = np.sum(ultimate_result.confidence > 0.7) / len(ultimate_result.confidence)
        
        print(f"平均信頼度: {confidence_stats['mean']:.3f}")
        print(f"高信頼度比率: {high_confidence_ratio*100:.1f}%")
        print(f"信頼度範囲: {confidence_stats['min']:.3f} - {confidence_stats['max']:.3f}")
        
        self.results['accuracy'] = {
            'confidence_stats': confidence_stats,
            'high_confidence_ratio': high_confidence_ratio
        }
        
    def _analyze_adaptivity(self):
        """適応性分析"""
        print("\n🔄 Adaptivity Analysis")
        print("-" * 30)
        
        ultimate_result = self.ultimate_chop.get_result()
        
        # 適応期間の統計
        period_stats = {
            'mean': np.nanmean(ultimate_result.adaptive_period),
            'std': np.nanstd(ultimate_result.adaptive_period),
            'min': np.nanmin(ultimate_result.adaptive_period),
            'max': np.nanmax(ultimate_result.adaptive_period)
        }
        
        # 効率比の統計
        efficiency_stats = {
            'mean': np.nanmean(ultimate_result.efficiency),
            'std': np.nanstd(ultimate_result.efficiency),
            'min': np.nanmin(ultimate_result.efficiency),
            'max': np.nanmax(ultimate_result.efficiency)
        }
        
        print(f"適応期間: {period_stats['mean']:.1f} ± {period_stats['std']:.1f}")
        print(f"効率比: {efficiency_stats['mean']:.3f} ± {efficiency_stats['std']:.3f}")
        
        self.results['adaptivity'] = {
            'period_stats': period_stats,
            'efficiency_stats': efficiency_stats
        }
        
    def _create_visualizations(self):
        """視覚化作成"""
        print("\n📈 Creating Visualizations...")
        
        # 図のセットアップ
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
        
        # 1. 価格とチョピネスインデックスの比較
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_price_and_choppiness(ax1)
        
        # 2. 遅延特性の比較
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_lag_comparison(ax2)
        
        # 3. 精度の比較
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_accuracy_comparison(ax3)
        
        # 4. 適応性の可視化
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_adaptivity(ax4)
        
        # 5. 信頼度の分布
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_confidence_distribution(ax5)
        
        # 6. パフォーマンス比較
        ax6 = fig.add_subplot(gs[3, :])
        self._plot_performance_comparison(ax6)
        
        plt.suptitle('🚀 Ultimate Choppiness Index vs Classic Choppiness Index', 
                    fontsize=16, fontweight='bold')
        
        # 保存
        output_dir = Path("examples/output")
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "ultimate_choppiness_comparison.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
    def _plot_price_and_choppiness(self, ax):
        """価格とチョピネス比較プロット"""
        dates = range(len(self.data))
        
        # 価格プロット
        ax2 = ax.twinx()
        ax2.plot(dates, self.data['close'], 'k-', alpha=0.7, linewidth=1, label='Price')
        ax2.set_ylabel('Price', color='black')
        
        # チョピネスプロット
        ultimate_values = self.results['performance']['ultimate_values']
        classic_values = self.results['performance']['classic_values']
        
        ax.plot(dates, ultimate_values, 'r-', linewidth=2, label='Ultimate Choppiness', alpha=0.8)
        ax.plot(dates, classic_values, 'b-', linewidth=1, label='Classic Choppiness', alpha=0.6)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Choppiness Index')
        ax.set_title('Price vs Choppiness Index Comparison')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
    def _plot_lag_comparison(self, ax):
        """遅延比較プロット"""
        categories = ['Ultimate', 'Classic']
        lag_values = [
            self.results['lag_analysis']['ultimate_lag'],
            self.results['lag_analysis']['classic_lag']
        ]
        
        bars = ax.bar(categories, lag_values, color=['red', 'blue'], alpha=0.7)
        ax.set_ylabel('Average Lag (bars)')
        ax.set_title('Response Lag Comparison')
        
        # 値をバーに表示
        for bar, value in zip(bars, lag_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.1f}', ha='center', va='bottom')
        
    def _plot_accuracy_comparison(self, ax):
        """精度比較プロット"""
        ultimate_result = self.ultimate_chop.get_result()
        confidence = ultimate_result.confidence
        
        # 信頼度のヒストグラム
        ax.hist(confidence[~np.isnan(confidence)], bins=20, alpha=0.7, color='red')
        ax.set_xlabel('Confidence Level')
        ax.set_ylabel('Frequency')
        ax.set_title('Confidence Distribution')
        ax.axvline(0.7, color='black', linestyle='--', alpha=0.5, label='High Confidence Threshold')
        ax.legend()
        
    def _plot_adaptivity(self, ax):
        """適応性プロット"""
        ultimate_result = self.ultimate_chop.get_result()
        dates = range(len(ultimate_result.adaptive_period))
        
        ax.plot(dates, ultimate_result.adaptive_period, 'g-', linewidth=1, alpha=0.7)
        ax.set_xlabel('Time')
        ax.set_ylabel('Adaptive Period')
        ax.set_title('Dynamic Period Adaptation')
        ax.grid(True, alpha=0.3)
        
    def _plot_confidence_distribution(self, ax):
        """信頼度分布プロット"""
        ultimate_result = self.ultimate_chop.get_result()
        
        # 効率比 vs 位相コヒーレンス
        valid_mask = ~(np.isnan(ultimate_result.efficiency) | np.isnan(ultimate_result.phase_coherence))
        
        scatter = ax.scatter(ultimate_result.efficiency[valid_mask], 
                           ultimate_result.phase_coherence[valid_mask],
                           c=ultimate_result.confidence[valid_mask], 
                           cmap='viridis', alpha=0.6)
        
        ax.set_xlabel('Efficiency Ratio')
        ax.set_ylabel('Phase Coherence')
        ax.set_title('Efficiency vs Phase Coherence')
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Confidence')
        
    def _plot_performance_comparison(self, ax):
        """パフォーマンス比較プロット"""
        metrics = ['Speed', 'Lag Improvement', 'Accuracy']
        values = [
            self.results['performance']['speed_ratio'],
            self.results['lag_analysis']['improvement'],
            self.results['accuracy']['high_confidence_ratio'] * 100
        ]
        
        bars = ax.bar(metrics, values, color=['green', 'orange', 'purple'], alpha=0.7)
        ax.set_ylabel('Performance Score')
        ax.set_title('Overall Performance Comparison')
        
        # 値をバーに表示
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.1f}', ha='center', va='bottom')


def load_sample_data():
    """サンプルOHLCデータをロードする（リアルな市場データを模擬）"""
    np.random.seed(42)  # 再現性のため
    n = 1000  # データポイント数
    
    # 基本的な価格トレンド（ランダムウォーク + トレンド）
    base = np.cumsum(np.random.normal(0, 1, n)) + np.linspace(0, 25, n)
    
    # 周期的な成分を追加（市場サイクルを模擬）
    cycles = 10 * np.sin(np.linspace(0, 8 * np.pi, n)) + 5 * np.sin(np.linspace(0, 20 * np.pi, n))
    
    # ボラティリティの変化（VIX指数のような変動）
    volatility = np.abs(np.sin(np.linspace(0, 6 * np.pi, n))) * 4 + 3
    
    # 終値を作成
    close = 100 + base + cycles
    
    # 高値、安値、始値を作成（リアルな価格変動）
    high = close + np.random.exponential(1, n) * volatility
    low = close - np.random.exponential(1, n) * volatility
    open_ = close + np.random.normal(0, 0.5, n) * volatility
    
    # 価格の整合性を確保
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    
    # DataFrameに変換
    df = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': np.random.randint(1000, 50000, n)
    })
    
    return df


def main():
    """メイン実行関数"""
    print("🚀 Ultimate Choppiness Index Demonstration")
    print("=" * 50)
    
    # サンプルデータロード
    try:
        data = load_sample_data()
        print(f"✅ データロード完了: {len(data)} rows")
    except Exception as e:
        print(f"❌ データロードエラー: {e}")
        # ダミーデータ作成
        print("📊 ダミーデータを生成中...")
        data = generate_sample_data()
    
    # 比較分析実行
    comparison = ChoppinessComparison(data)
    results = comparison.run_comparison()
    
    # 結果サマリー
    print("\n🎯 Final Results Summary")
    print("=" * 50)
    print(f"⚡ 遅延改善: {results['lag_analysis']['improvement']:.1f}%")
    print(f"🚀 処理速度: {results['performance']['speed_ratio']:.1f}x faster")
    print(f"🎯 高信頼度比率: {results['accuracy']['high_confidence_ratio']*100:.1f}%")
    print(f"📈 平均信頼度: {results['accuracy']['confidence_stats']['mean']:.3f}")
    print("\n✅ 分析完了！結果は examples/output/ultimate_choppiness_comparison.png に保存されました。")


def generate_sample_data(n_points=1000):
    """サンプルデータ生成"""
    np.random.seed(42)
    
    # 基本トレンド
    trend = np.cumsum(np.random.randn(n_points) * 0.01)
    
    # 価格データ生成
    dates = pd.date_range('2023-01-01', periods=n_points, freq='H')
    close = 100 + trend + np.random.randn(n_points) * 0.5
    
    # OHLC生成
    high = close + np.random.rand(n_points) * 2
    low = close - np.random.rand(n_points) * 2
    open_price = close + np.random.randn(n_points) * 0.3
    volume = np.random.randint(1000, 10000, n_points)
    
    return pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)


if __name__ == "__main__":
    main() 