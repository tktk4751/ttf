#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Kalman Integration Example - カルマンフィルター統合使用例

既存のインジケーターでカルマンフィルター統合クラスを使用する方法を示します。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, NamedTuple

from indicators.kalman_filter_unified import KalmanFilterUnified, KalmanFilterResult

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.style.use('default')


class UltimateMaWithUnifiedKalman:
    """
    Ultimate MAインジケーターのカルマンフィルター統合版
    
    従来のultimate_ma.pyのadaptive_kalman_filter_numbaを
    統合フィルターに置き換えた改良版
    """
    
    def __init__(
        self,
        kalman_filter_type: str = 'quantum_adaptive',
        super_smooth_period: int = 10,
        zero_lag_period: int = 21,
        src_type: str = 'hlc3'
    ):
        self.kalman_filter_type = kalman_filter_type
        self.super_smooth_period = super_smooth_period
        self.zero_lag_period = zero_lag_period
        self.src_type = src_type
        
        # 統合カルマンフィルター
        self.kalman_filter = KalmanFilterUnified(
            filter_type=kalman_filter_type,
            src_type=src_type,
            base_process_noise=0.01,
            base_measurement_noise=0.01,
            volatility_window=20
        )
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        """Ultimate MAを計算（統合カルマンフィルター使用）"""
        
        # 1. カルマンフィルター適用
        kalman_result = self.kalman_filter.calculate(data)
        kalman_values = kalman_result.filtered_values
        
        # 2. スーパースムーザーフィルター
        super_smooth_values = self._super_smoother_filter(kalman_values)
        
        # 3. ゼロラグEMA
        zero_lag_values = self._zero_lag_ema(super_smooth_values)
        
        # 4. トレンドシグナル計算
        trend_signals = self._calculate_trend_signals(zero_lag_values)
        
        return {
            'values': zero_lag_values,
            'kalman_values': kalman_values,
            'super_smooth_values': super_smooth_values,
            'trend_signals': trend_signals,
            'kalman_confidence': kalman_result.confidence_scores,
            'kalman_trend': kalman_result.trend_estimate,
            'filter_type': kalman_result.filter_type,
            'quantum_coherence': kalman_result.quantum_coherence
        }
    
    def _super_smoother_filter(self, prices: np.ndarray) -> np.ndarray:
        """スーパースムーザーフィルター"""
        n = len(prices)
        smoothed = np.zeros(n)
        
        if n < 4:
            return prices.copy()
        
        # 初期値設定
        for i in range(3):
            smoothed[i] = prices[i]
        
        # スーパースムーザー係数
        a1 = np.exp(-1.414 * np.pi / self.super_smooth_period)
        b1 = 2.0 * a1 * np.cos(1.414 * np.pi / self.super_smooth_period)
        c2 = b1
        c3 = -a1 * a1
        c1 = 1.0 - c2 - c3
        
        for i in range(3, n):
            smoothed[i] = (c1 * (prices[i] + prices[i-1]) / 2.0 + 
                          c2 * smoothed[i-1] + 
                          c3 * smoothed[i-2])
        
        return smoothed
    
    def _zero_lag_ema(self, prices: np.ndarray) -> np.ndarray:
        """ゼロラグEMA"""
        n = len(prices)
        zero_lag = np.zeros(n)
        
        if n < 2:
            return prices.copy()
        
        alpha = 2.0 / (self.zero_lag_period + 1.0)
        zero_lag[0] = prices[0]
        
        for i in range(1, n):
            # 標準EMA
            ema = alpha * prices[i] + (1 - alpha) * zero_lag[i-1]
            
            # ゼロラグ補正
            if i >= 2:
                momentum = prices[i] - prices[i-1]
                lag_correction = alpha * momentum
                zero_lag[i] = ema + lag_correction
            else:
                zero_lag[i] = ema
        
        return zero_lag
    
    def _calculate_trend_signals(self, values: np.ndarray) -> np.ndarray:
        """トレンドシグナル計算"""
        n = len(values)
        signals = np.zeros(n)
        
        for i in range(5, n):
            # 短期トレンド（5期間）
            short_trend = np.mean(np.diff(values[i-5:i]))
            
            # シグナル判定
            if short_trend > 0.001:
                signals[i] = 1  # 上昇トレンド
            elif short_trend < -0.001:
                signals[i] = -1  # 下降トレンド
            else:
                signals[i] = 0  # レンジ
        
        return signals


class VolatilityIndicatorWithUnifiedKalman:
    """
    ボラティリティインジケーターのカルマンフィルター統合版
    
    Ultimate Volatilityの量子適応カルマンフィルターを
    統合フィルターに置き換えた改良版
    """
    
    def __init__(
        self,
        kalman_filter_type: str = 'hyper_quantum',
        period: int = 14,
        src_type: str = 'hlc3'
    ):
        self.kalman_filter_type = kalman_filter_type
        self.period = period
        self.src_type = src_type
        
        # 統合カルマンフィルター
        self.kalman_filter = KalmanFilterUnified(
            filter_type=kalman_filter_type,
            src_type=src_type,
            base_process_noise=0.01,
            base_measurement_noise=0.01,
            volatility_window=period
        )
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        """ボラティリティインジケーターを計算"""
        
        # True Range計算
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        true_range = self._calculate_true_range(high, low, close)
        
        # カルマンフィルター適用（True Rangeに対して）
        # PriceSourceのために必要な列を作成（True Rangeベース）
        tr_data = pd.DataFrame({
            'open': true_range,
            'high': true_range * 1.01,  # 若干の変動を持たせる
            'low': true_range * 0.99,
            'close': true_range
        })
        kalman_result = self.kalman_filter.calculate(tr_data)
        
        # フィルター済みTrue Rangeを移動平均
        filtered_tr = kalman_result.filtered_values
        ultimate_volatility = self._moving_average(filtered_tr, self.period)
        
        # ボラティリティトレンド
        volatility_trend = self._calculate_volatility_trend(ultimate_volatility)
        
        return {
            'ultimate_volatility': ultimate_volatility,
            'raw_true_range': true_range,
            'filtered_true_range': filtered_tr,
            'volatility_trend': volatility_trend,
            'kalman_confidence': kalman_result.confidence_scores,
            'kalman_uncertainty': kalman_result.uncertainty,
            'filter_type': kalman_result.filter_type
        }
    
    def _calculate_true_range(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """True Range計算"""
        n = len(high)
        tr = np.zeros(n)
        
        tr[0] = high[0] - low[0]
        
        for i in range(1, n):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr[i] = max(tr1, tr2, tr3)
        
        return tr
    
    def _moving_average(self, values: np.ndarray, period: int) -> np.ndarray:
        """移動平均"""
        n = len(values)
        ma = np.zeros(n)
        
        for i in range(n):
            start_idx = max(0, i - period + 1)
            ma[i] = np.mean(values[start_idx:i+1])
        
        return ma
    
    def _calculate_volatility_trend(self, volatility: np.ndarray) -> np.ndarray:
        """ボラティリティトレンド計算"""
        n = len(volatility)
        trend = np.zeros(n)
        
        for i in range(10, n):
            # 10期間での線形回帰傾き
            x = np.arange(10)
            y = volatility[i-10:i]
            
            # 最小二乗法
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            
            numerator = np.sum((x - x_mean) * (y - y_mean))
            denominator = np.sum((x - x_mean) ** 2)
            
            if denominator > 0:
                slope = numerator / denominator
                trend[i] = slope
        
        return trend


class EnsembleIndicatorWithMultipleKalman:
    """
    複数のカルマンフィルターを使用するアンサンブルインジケーター
    
    複数のフィルタータイプを同時に実行し、
    動的に最適なフィルターを選択するシステム
    """
    
    def __init__(
        self,
        filter_types: List[str] = ['adaptive', 'quantum_adaptive', 'unscented', 'hyper_quantum'],
        src_type: str = 'hlc3'
    ):
        self.filter_types = filter_types
        self.src_type = src_type
        
        # 各フィルターを初期化
        self.filters = {}
        for filter_type in filter_types:
            self.filters[filter_type] = KalmanFilterUnified(
                filter_type=filter_type,
                src_type=src_type,
                base_process_noise=0.01,
                base_measurement_noise=0.01,
                volatility_window=20
            )
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        """アンサンブルインジケーターを計算"""
        
        # 各フィルターを実行
        filter_results = {}
        performance_scores = {}
        
        for filter_type in self.filter_types:
            try:
                result = self.filters[filter_type].calculate(data)
                filter_results[filter_type] = result
                
                # パフォーマンススコア計算
                score = self._calculate_performance_score(result, data['close'].values)
                performance_scores[filter_type] = score
                
            except Exception as e:
                print(f"Warning: {filter_type} filter failed: {e}")
                performance_scores[filter_type] = 0.0
        
        # 動的重み計算
        weights = self._calculate_dynamic_weights(performance_scores)
        
        # アンサンブル結果計算
        ensemble_result = self._calculate_ensemble(filter_results, weights)
        
        # 最適フィルター選択
        best_filter = max(performance_scores, key=performance_scores.get)
        
        return {
            'ensemble_values': ensemble_result['ensemble_values'],
            'ensemble_confidence': ensemble_result['ensemble_confidence'],
            'ensemble_trend': ensemble_result['ensemble_trend'],
            'filter_weights': weights,
            'performance_scores': performance_scores,
            'best_filter': best_filter,
            'individual_results': filter_results,
            'filter_ranking': sorted(performance_scores.items(), key=lambda x: x[1], reverse=True)
        }
    
    def _calculate_performance_score(self, kalman_result: KalmanFilterResult, original_prices: np.ndarray) -> float:
        """パフォーマンススコア計算"""
        if len(kalman_result.filtered_values) < 10:
            return 0.0
        
        # ノイズ除去効果
        original_noise = np.std(np.diff(original_prices))
        filtered_noise = np.std(np.diff(kalman_result.filtered_values))
        noise_reduction = max(0, 1 - filtered_noise / original_noise) if original_noise > 0 else 0
        
        # 追従性
        correlation = np.corrcoef(original_prices, kalman_result.filtered_values)[0, 1]
        correlation = max(0, correlation)
        
        # 信頼度
        avg_confidence = np.nanmean(kalman_result.confidence_scores)
        
        # 総合スコア
        score = (noise_reduction * 0.4 + correlation * 0.4 + avg_confidence * 0.2)
        return score
    
    def _calculate_dynamic_weights(self, performance_scores: Dict[str, float]) -> Dict[str, float]:
        """動的重み計算"""
        total_score = sum(performance_scores.values())
        
        if total_score == 0:
            # 均等重み
            n_filters = len(performance_scores)
            return {filter_type: 1.0/n_filters for filter_type in performance_scores.keys()}
        
        # 性能ベース重み
        weights = {}
        for filter_type, score in performance_scores.items():
            weights[filter_type] = score / total_score
        
        return weights
    
    def _calculate_ensemble(self, filter_results: Dict[str, KalmanFilterResult], weights: Dict[str, float]) -> Dict:
        """アンサンブル結果計算"""
        if not filter_results:
            return {'ensemble_values': np.array([]), 'ensemble_confidence': np.array([]), 'ensemble_trend': np.array([])}
        
        # 最初の結果の長さを取得
        first_result = next(iter(filter_results.values()))
        n = len(first_result.filtered_values)
        
        ensemble_values = np.zeros(n)
        ensemble_confidence = np.zeros(n)
        ensemble_trend = np.zeros(n)
        
        for i in range(n):
            weighted_value = 0.0
            weighted_confidence = 0.0
            weighted_trend = 0.0
            total_weight = 0.0
            
            for filter_type, result in filter_results.items():
                if filter_type in weights and i < len(result.filtered_values):
                    weight = weights[filter_type]
                    
                    weighted_value += weight * result.filtered_values[i]
                    weighted_confidence += weight * result.confidence_scores[i]
                    if result.trend_estimate is not None:
                        weighted_trend += weight * result.trend_estimate[i]
                    total_weight += weight
            
            if total_weight > 0:
                ensemble_values[i] = weighted_value / total_weight
                ensemble_confidence[i] = weighted_confidence / total_weight
                ensemble_trend[i] = weighted_trend / total_weight
        
        return {
            'ensemble_values': ensemble_values,
            'ensemble_confidence': ensemble_confidence,
            'ensemble_trend': ensemble_trend
        }


def generate_test_data(n_samples: int = 500) -> pd.DataFrame:
    """テスト用データ生成"""
    np.random.seed(42)
    
    t = np.arange(n_samples)
    trend = 100 + 0.05 * t
    cycle = 5 * np.sin(2 * np.pi * t / 50)
    noise = np.random.normal(0, 1, n_samples)
    
    close = trend + cycle + noise
    high = close + np.random.uniform(0, 2, n_samples)
    low = close - np.random.uniform(0, 2, n_samples)
    open_price = close + np.random.uniform(-1, 1, n_samples)
    
    return pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': np.random.randint(1000, 10000, n_samples)
    })


def demo_ultimate_ma_integration():
    """Ultimate MA統合デモ"""
    print("\n🎯 Ultimate MA カルマンフィルター統合デモ")
    print("=" * 60)
    
    # テストデータ生成
    data = generate_test_data(500)
    
    # 従来版vs統合版の比較
    filter_types = ['adaptive', 'quantum_adaptive', 'unscented', 'hyper_quantum']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Ultimate MA: カルマンフィルター統合比較', fontsize=16, fontweight='bold')
    
    original_prices = data['close'].values
    
    for i, filter_type in enumerate(filter_types):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        # 統合Ultimate MA
        uma_integrated = UltimateMaWithUnifiedKalman(
            kalman_filter_type=filter_type,
            super_smooth_period=10,
            zero_lag_period=21
        )
        
        result = uma_integrated.calculate(data)
        
        # プロット
        ax.plot(original_prices, label='元の価格', color='gray', alpha=0.7, linewidth=1)
        ax.plot(result['values'], label=f'UMA({filter_type})', color='blue', linewidth=2)
        ax.plot(result['kalman_values'], label='Kalmanフィルター', color='red', alpha=0.7, linewidth=1)
        
        # 信頼度背景
        confidence = result['kalman_confidence']
        ax.fill_between(range(len(confidence)), 
                       np.min(original_prices) + (np.max(original_prices) - np.min(original_prices)) * confidence,
                       np.min(original_prices), alpha=0.1, color='blue')
        
        ax.set_title(f'{filter_type.upper()} Filter')
        ax.set_ylabel('価格')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('examples/output/ultimate_ma_integration.png', dpi=300, bbox_inches='tight')
    plt.show()


def demo_volatility_integration():
    """ボラティリティ統合デモ"""
    print("\n📊 ボラティリティインジケーター統合デモ")
    print("=" * 60)
    
    data = generate_test_data(500)
    
    # 統合ボラティリティインジケーター
    vol_indicator = VolatilityIndicatorWithUnifiedKalman(
        kalman_filter_type='hyper_quantum',
        period=14
    )
    
    result = vol_indicator.calculate(data)
    
    # プロット
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('ボラティリティインジケーター: カルマンフィルター統合', fontsize=16, fontweight='bold')
    
    # 価格チャート
    ax1 = axes[0]
    ax1.plot(data['close'].values, label='価格', color='black', linewidth=1)
    ax1.set_title('価格チャート')
    ax1.set_ylabel('価格')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ボラティリティ比較
    ax2 = axes[1]
    ax2.plot(result['raw_true_range'], label='生True Range', color='gray', alpha=0.7, linewidth=1)
    ax2.plot(result['filtered_true_range'], label='フィルター済みTR', color='blue', linewidth=2)
    ax2.plot(result['ultimate_volatility'], label='Ultimate Volatility', color='red', linewidth=2)
    
    # 信頼度
    confidence = result['kalman_confidence']
    ax2_twin = ax2.twinx()
    ax2_twin.plot(confidence, label='信頼度', color='green', alpha=0.7)
    ax2_twin.set_ylabel('信頼度', color='green')
    ax2_twin.set_ylim(0, 1)
    
    ax2.set_title('ボラティリティ分析')
    ax2.set_xlabel('時間')
    ax2.set_ylabel('ボラティリティ')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('examples/output/volatility_integration.png', dpi=300, bbox_inches='tight')
    plt.show()


def demo_ensemble_integration():
    """アンサンブル統合デモ"""
    print("\n🎭 アンサンブルインジケーター統合デモ")
    print("=" * 60)
    
    data = generate_test_data(500)
    
    # アンサンブルインジケーター
    ensemble_indicator = EnsembleIndicatorWithMultipleKalman(
        filter_types=['adaptive', 'quantum_adaptive', 'unscented', 'hyper_quantum']
    )
    
    result = ensemble_indicator.calculate(data)
    
    # 結果表示
    print(f"🏆 最高性能フィルター: {result['best_filter']}")
    print(f"📊 フィルターランキング:")
    for i, (filter_type, score) in enumerate(result['filter_ranking']):
        print(f"  {i+1}. {filter_type}: {score:.3f}")
    
    print(f"\n⚖️ 動的重み:")
    for filter_type, weight in result['filter_weights'].items():
        print(f"  {filter_type}: {weight:.3f}")
    
    # プロット
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('アンサンブルインジケーター: マルチカルマンフィルター統合', fontsize=16, fontweight='bold')
    
    original_prices = data['close'].values
    
    # 1. 個別フィルター結果
    ax1 = axes[0, 0]
    ax1.plot(original_prices, label='元の価格', color='gray', alpha=0.7, linewidth=1)
    
    colors = ['red', 'blue', 'green', 'orange']
    for i, (filter_type, filter_result) in enumerate(result['individual_results'].items()):
        ax1.plot(filter_result.filtered_values, label=filter_type, 
                color=colors[i % len(colors)], linewidth=1.5, alpha=0.8)
    
    ax1.set_title('個別フィルター結果')
    ax1.set_ylabel('価格')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. アンサンブル結果
    ax2 = axes[0, 1]
    ax2.plot(original_prices, label='元の価格', color='gray', alpha=0.7, linewidth=1)
    ax2.plot(result['ensemble_values'], label='アンサンブル', color='purple', linewidth=3)
    
    # 信頼度
    confidence = result['ensemble_confidence']
    ax2.fill_between(range(len(confidence)), result['ensemble_values'] + confidence * 5,
                    result['ensemble_values'] - confidence * 5, alpha=0.2, color='purple')
    
    ax2.set_title('アンサンブル結果')
    ax2.set_ylabel('価格')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. パフォーマンススコア
    ax3 = axes[1, 0]
    filter_names = list(result['performance_scores'].keys())
    scores = list(result['performance_scores'].values())
    
    bars = ax3.bar(filter_names, scores, color=colors[:len(filter_names)])
    ax3.set_title('パフォーマンススコア')
    ax3.set_ylabel('スコア')
    ax3.set_xticklabels(filter_names, rotation=45, ha='right')
    
    # スコア値を表示
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    # 4. 動的重み
    ax4 = axes[1, 1]
    weights = list(result['filter_weights'].values())
    
    pie_colors = colors[:len(filter_names)]
    wedges, texts, autotexts = ax4.pie(weights, labels=filter_names, colors=pie_colors, 
                                      autopct='%1.1f%%', startangle=90)
    ax4.set_title('動的重み分布')
    
    plt.tight_layout()
    plt.savefig('examples/output/ensemble_integration.png', dpi=300, bbox_inches='tight')
    plt.show()


def show_migration_guide():
    """移行ガイドの表示"""
    print("\n📚 カルマンフィルター統合クラス移行ガイド")
    print("=" * 70)
    
    migration_examples = [
        {
            'title': '1. Ultimate MA のカルマンフィルター置き換え',
            'before': '''
# 従来のコード (ultimate_ma.py)
filtered_prices = adaptive_kalman_filter_numba(prices)
            ''',
            'after': '''
# 統合版
kalman_filter = KalmanFilterUnified(filter_type='adaptive')
result = kalman_filter.calculate(data)
filtered_prices = result.filtered_values
            '''
        },
        {
            'title': '2. Ultimate Breakout の量子カルマンフィルター置き換え',
            'before': '''
# 従来のコード (ultimate_breakout_channel.py)
filtered_prices, quantum_coherence = quantum_adaptive_kalman_filter(
    prices, amplitude, phase)
            ''',
            'after': '''
# 統合版
kalman_filter = KalmanFilterUnified(
    filter_type='quantum_adaptive', 
    enable_hilbert=True)
result = kalman_filter.calculate(data)
filtered_prices = result.filtered_values
quantum_coherence = result.quantum_coherence
            '''
        },
        {
            'title': '3. Ultimate Chop Trend の無香料カルマンフィルター置き換え',
            'before': '''
# 従来のコード (ultimate_chop_trend.py)
filtered_prices, trend_estimate, uncertainty = unscented_kalman_filter(
    prices, volatility, alpha=0.001, beta=2.0, kappa=0.0)
            ''',
            'after': '''
# 統合版
kalman_filter = KalmanFilterUnified(
    filter_type='unscented',
    ukf_alpha=0.001, ukf_beta=2.0, ukf_kappa=0.0)
result = kalman_filter.calculate(data)
filtered_prices = result.filtered_values
trend_estimate = result.trend_estimate
uncertainty = result.uncertainty
            '''
        }
    ]
    
    for example in migration_examples:
        print(f"\n{example['title']}")
        print("-" * len(example['title']))
        print(f"従来版:{example['before']}")
        print(f"統合版:{example['after']}")
    
    print(f"\n💡 統合の利点:")
    print(f"  • 一貫したインターフェース")
    print(f"  • パラメータの標準化")
    print(f"  • 複数フィルターの簡単切り替え")
    print(f"  • 統一されたエラーハンドリング")
    print(f"  • パフォーマンス指標の自動計算")
    print(f"  • キャッシュ機能による高速化")


def main():
    """メイン実行関数"""
    print("🚀 カルマンフィルター統合使用例デモを開始")
    print("=" * 70)
    
    # 出力ディレクトリ作成
    os.makedirs('examples/output', exist_ok=True)
    
    # 1. Ultimate MA統合デモ
    demo_ultimate_ma_integration()
    
    # 2. ボラティリティ統合デモ
    demo_volatility_integration()
    
    # 3. アンサンブル統合デモ
    demo_ensemble_integration()
    
    # 4. 移行ガイド表示
    show_migration_guide()
    
    print("\n✅ 統合使用例デモ完了！")
    print("   結果画像は examples/output/ に保存されました。")
    print("   これで既存のインジケーターでカルマンフィルター統合クラスを使用できます。")


if __name__ == "__main__":
    main() 