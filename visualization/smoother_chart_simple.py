#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
シンプルなスムーザー比較チャート

indicators/smoother/ディレクトリ内のスムーザーを
matplotlib で可視化・比較する簡単なシステム
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from numba import njit
from typing import Dict, Any, Optional, Tuple
import logging


# Simple Numba implementations to avoid import issues
@njit(fastmath=True, cache=True)
def adaptive_kalman_filter_simple(signal: np.ndarray, process_noise: float = 1e-5) -> np.ndarray:
    """簡単な適応カルマンフィルタ実装"""
    length = len(signal)
    filtered_signal = np.zeros(length)
    
    if length > 0:
        state = signal[0]
        error_covariance = 1.0
        filtered_signal[0] = state
    
    for i in range(1, length):
        # 予測ステップ
        predicted_state = state
        predicted_covariance = error_covariance + process_noise
        
        # 適応的観測ノイズ推定
        if i > 5:
            recent_residuals = np.zeros(5)
            for j in range(5):
                recent_residuals[j] = abs(signal[i-j] - filtered_signal[i-j])
            observation_noise = np.var(recent_residuals) + 1e-6
        else:
            observation_noise = 1e-3
        
        # カルマンゲイン
        kalman_gain = predicted_covariance / (predicted_covariance + observation_noise)
        
        # 更新ステップ
        innovation = signal[i] - predicted_state
        state = predicted_state + kalman_gain * innovation
        error_covariance = (1 - kalman_gain) * predicted_covariance
        
        filtered_signal[i] = state
    
    return filtered_signal


@njit(fastmath=True, cache=True)
def super_smoother_simple(data: np.ndarray, length: int = 15) -> np.ndarray:
    """簡単なSuper Smoother実装"""
    n = len(data)
    result = np.zeros(n)
    
    # 初期値
    for i in range(min(length, n)):
        result[i] = data[i]
    
    # 2-pole Super Smoother approximation
    a1 = np.exp(-1.414 * 3.14159 / length)
    b1 = 2 * a1 * np.cos(1.414 * 3.14159 / length)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3
    
    for i in range(2, n):
        result[i] = c1 * data[i] + c2 * result[i-1] + c3 * result[i-2]
    
    return result


@njit(fastmath=True, cache=True)
def frama_simple(prices: np.ndarray, highs: np.ndarray, lows: np.ndarray, n: int = 16) -> np.ndarray:
    """簡単なFRAMA実装"""
    length = len(prices)
    result = np.zeros(length)
    
    # 初期値
    result[0] = prices[0] if length > 0 else 0
    
    for i in range(1, length):
        if i < n:
            result[i] = prices[i]
            continue
            
        # フラクタル次元の簡単な近似
        len1 = n // 2
        
        # 前半の高安値範囲
        h1 = np.max(highs[i-n:i-len1]) if i >= n else highs[i]
        l1 = np.min(lows[i-n:i-len1]) if i >= n else lows[i]
        
        # 後半の高安値範囲
        h2 = np.max(highs[i-len1:i]) if i >= len1 else highs[i]
        l2 = np.min(lows[i-len1:i]) if i >= len1 else lows[i]
        
        # 全体の高安値範囲
        h3 = np.max(highs[i-n:i]) if i >= n else highs[i]
        l3 = np.min(lows[i-n:i]) if i >= n else lows[i]
        
        # フラクタル次元の計算
        n1 = h1 - l1 if h1 > l1 else 0.0001
        n2 = h2 - l2 if h2 > l2 else 0.0001
        n3 = h3 - l3 if h3 > l3 else 0.0001
        
        if n1 > 0 and n2 > 0 and n3 > 0:
            dimen = (np.log(n1 + n2) - np.log(n3)) / np.log(2)
            alpha = np.exp(-4.6 * (dimen - 1))
            alpha = max(0.01, min(1.0, alpha))
        else:
            alpha = 0.1
        
        # FRAMA計算
        result[i] = alpha * prices[i] + (1 - alpha) * result[i-1]
    
    return result


@njit(fastmath=True, cache=True)
def simple_moving_average(data: np.ndarray, period: int = 20) -> np.ndarray:
    """シンプルな移動平均"""
    n = len(data)
    result = np.zeros(n)
    
    for i in range(n):
        if i < period - 1:
            result[i] = np.mean(data[:i+1])
        else:
            result[i] = np.mean(data[i-period+1:i+1])
    
    return result


@njit(fastmath=True, cache=True)
def exponential_moving_average(data: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """指数移動平均"""
    n = len(data)
    result = np.zeros(n)
    
    if n > 0:
        result[0] = data[0]
        for i in range(1, n):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
    
    return result


class SimpleSmootherChart:
    """シンプルなスムーザー比較チャート"""
    
    def __init__(self):
        self.data = None
        self.results = {}
        
        # スムーザー設定
        self.smoother_configs = {
            'Original': {
                'color': 'gray',
                'linestyle': '-',
                'alpha': 0.6,
                'description': 'Original Price'
            },
            'SMA_20': {
                'color': 'blue',
                'linestyle': '-',
                'alpha': 0.8,
                'description': 'Simple MA (20)'
            },
            'EMA_Fast': {
                'color': 'green', 
                'linestyle': '--',
                'alpha': 0.8,
                'description': 'EMA Fast (α=0.2)'
            },
            'EMA_Slow': {
                'color': 'red',
                'linestyle': '--',
                'alpha': 0.8,
                'description': 'EMA Slow (α=0.05)'
            },
            'SuperSmoother': {
                'color': 'purple',
                'linestyle': '-.',
                'alpha': 0.9,
                'description': 'Super Smoother'
            },
            'AdaptiveKalman': {
                'color': 'orange',
                'linestyle': ':',
                'alpha': 0.9,
                'description': 'Adaptive Kalman'
            },
            'FRAMA': {
                'color': 'brown',
                'linestyle': '-',
                'alpha': 0.9,
                'description': 'FRAMA'
            }
        }
    
    def generate_sample_data(self, n_points: int = 500) -> pd.DataFrame:
        """サンプルデータ生成"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=n_points, freq='1h')
        
        # トレンド + サイクル + ノイズの合成
        t = np.arange(n_points)
        base_price = 100
        trend = 0.02 * t
        cycle1 = 5 * np.sin(0.05 * t)
        cycle2 = 2 * np.sin(0.2 * t)
        noise = np.random.randn(n_points) * 1.5
        
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
        
        print(f"サンプルデータ生成完了: {len(self.data)}行")
        return self.data
    
    def calculate_all_smoothers(self) -> None:
        """すべてのスムーザーを計算"""
        if self.data is None:
            raise ValueError("データが生成されていません")
        
        print("スムーザーを計算中...")
        
        close_prices = self.data['close'].values
        high_prices = self.data['high'].values
        low_prices = self.data['low'].values
        
        # 各スムーザーを計算
        self.results['Original'] = close_prices
        
        # Simple Moving Average
        self.results['SMA_20'] = simple_moving_average(close_prices, 20)
        
        # Exponential Moving Average (Fast and Slow)
        self.results['EMA_Fast'] = exponential_moving_average(close_prices, 0.2)
        self.results['EMA_Slow'] = exponential_moving_average(close_prices, 0.05)
        
        # Super Smoother
        self.results['SuperSmoother'] = super_smoother_simple(close_prices, 15)
        
        # Adaptive Kalman Filter
        self.results['AdaptiveKalman'] = adaptive_kalman_filter_simple(close_prices, 1e-5)
        
        # FRAMA
        self.results['FRAMA'] = frama_simple(close_prices, high_prices, low_prices, 16)
        
        print(f"計算完了: {len(self.results)} スムーザー")
    
    def calculate_performance_metrics(self) -> Dict[str, Dict[str, float]]:
        """パフォーマンス指標を計算"""
        metrics = {}
        original_prices = self.results['Original']
        
        for name, smoothed_values in self.results.items():
            if name == 'Original':
                continue
            
            # ノイズ削減効果
            original_std = np.std(original_prices)
            smoothed_std = np.std(smoothed_values)
            noise_reduction = (1 - smoothed_std / original_std) * 100
            
            # 遅延測定（相関のピーク位置）
            correlation = np.correlate(original_prices, smoothed_values, mode='full')
            lag = abs(np.argmax(correlation) - len(smoothed_values) + 1)
            
            # 平滑度（変化率の標準偏差）
            smoothness = np.std(np.diff(smoothed_values))
            
            # トレンド追従性（移動平均との相関）
            ma20 = simple_moving_average(original_prices, 20)
            trend_correlation = np.corrcoef(smoothed_values[20:], ma20[20:])[0, 1]
            if np.isnan(trend_correlation):
                trend_correlation = 0
            
            metrics[name] = {
                'noise_reduction': noise_reduction,
                'lag': lag,
                'trend_correlation': trend_correlation,
                'smoothness': smoothness
            }
        
        return metrics
    
    def plot_comparison(
        self, 
        title: str = "スムーザー比較チャート",
        figsize: Tuple[int, int] = (16, 12),
        save_path: Optional[str] = None
    ) -> None:
        """比較チャートを描画"""
        if not self.results:
            raise ValueError("計算結果がありません")
        
        # パフォーマンス指標を計算
        metrics = self.calculate_performance_metrics()
        
        # 図の設定
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # メインチャート
        ax_main = axes[0]
        dates = self.data.index
        
        for name, values in self.results.items():
            config = self.smoother_configs[name]
            ax_main.plot(dates, values,
                        color=config['color'],
                        linestyle=config['linestyle'],
                        alpha=config['alpha'],
                        linewidth=2 if name != 'Original' else 1,
                        label=config['description'],
                        zorder=1 if name == 'Original' else 2)
        
        ax_main.set_title(title, fontsize=16, fontweight='bold')
        ax_main.set_xlabel('日時')
        ax_main.set_ylabel('価格')
        ax_main.legend(loc='upper left', fontsize=10)
        ax_main.grid(True, alpha=0.3)
        
        # 日付フォーマット
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax_main.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates)//20)))
        plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # パフォーマンス指標のチャート
        ax_perf = axes[1]
        
        names = [name for name in metrics.keys()]
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
                    val = 1 / (1 + val) if val > 0 else 0  # 平滑度は逆数
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
        original = self.results['Original']
        print(f"元価格統計: 平均={np.mean(original):.2f}, 標準偏差={np.std(original):.2f}")
        
        for name, values in self.results.items():
            if name == 'Original':
                continue
            print(f"\n{name}:")
            print(f"  平均: {np.mean(values):.2f}")
            print(f"  標準偏差: {np.std(values):.2f}")
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


def main():
    """メイン関数"""
    print("=== シンプルスムーザー比較チャート ===")
    
    # チャートを作成
    chart = SimpleSmootherChart()
    chart.generate_sample_data(400)
    chart.calculate_all_smoothers()
    chart.plot_comparison(
        title="スムーザー比較 - シンプル実装版",
        save_path="simple_smoother_comparison.png"
    )


if __name__ == "__main__":
    main()