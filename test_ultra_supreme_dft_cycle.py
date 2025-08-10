#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🧪 **ULTRA SUPREME DFT CYCLE DETECTOR TEST** 🧪

EhlersUltraSupremeDFTCycle の性能テストとベンチマーク
- 従来のEhlersDFTDominantCycleとの比較
- 各種カルマンフィルターの性能評価
- リアルタイム処理速度測定
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# テスト対象のインポート
from indicators.cycle.ehlers_ultra_supreme_dft_cycle import EhlersUltraSupremeDFTCycle
from indicators.cycle.ehlers_dft_dominant_cycle import EhlersDFTDominantCycle

# データ生成用
from data.data_loader import DataLoader

def generate_synthetic_data(n_points: int = 1000) -> pd.DataFrame:
    """
    合成テストデータ生成
    - 複数の周期成分
    - ノイズ
    - トレンド変化
    - 相転移
    """
    np.random.seed(42)
    
    # 時間軸
    t = np.arange(n_points)
    
    # 基本価格レベル
    base_price = 100.0
    
    # 主要サイクル成分
    cycle_20 = 5.0 * np.sin(2 * np.pi * t / 20)  # 20期間サイクル
    cycle_15 = 3.0 * np.sin(2 * np.pi * t / 15 + np.pi/4)  # 15期間サイクル
    cycle_30 = 2.0 * np.sin(2 * np.pi * t / 30 + np.pi/2)  # 30期間サイクル
    
    # 動的サイクル（周期が変化）
    dynamic_freq = 25 + 10 * np.sin(2 * np.pi * t / 200)
    dynamic_cycle = 4.0 * np.sin(2 * np.pi * np.cumsum(1 / dynamic_freq))
    
    # トレンド成分
    trend = 0.02 * t + 10 * np.sin(2 * np.pi * t / 100)
    
    # 相転移（急激な変化）
    phase_transition = np.zeros(n_points)
    transition_points = [300, 600, 800]
    for tp in transition_points:
        if tp < n_points:
            phase_transition[tp:tp+50] = 15.0 * np.exp(-0.1 * np.arange(min(50, n_points-tp)))
    
    # ノイズ（時変分散）
    noise_variance = 1.0 + 2.0 * np.sin(2 * np.pi * t / 150) ** 2
    noise = np.random.normal(0, np.sqrt(noise_variance))
    
    # 合成価格
    price = (base_price + trend + cycle_20 + cycle_15 + cycle_30 + 
             dynamic_cycle + phase_transition + noise)
    
    # OHLC生成（簡易版）
    high = price + np.abs(np.random.normal(0, 0.5, n_points))
    low = price - np.abs(np.random.normal(0, 0.5, n_points))
    open_price = np.roll(price, 1)
    open_price[0] = price[0]
    close_price = price
    
    # ボリューム（ダミー）
    volume = np.random.lognormal(10, 0.5, n_points)
    
    return pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close_price,
        'volume': volume
    })

def benchmark_performance() -> Dict[str, Dict[str, float]]:
    """
    性能ベンチマークテスト
    """
    print("🏃‍♂️ 性能ベンチマーク開始...")
    
    # テストデータ生成
    test_sizes = [500, 1000, 2000]
    results = {}
    
    for size in test_sizes:
        print(f"\n📊 テストサイズ: {size}")
        data = generate_synthetic_data(size)
        
        size_results = {}
        
        # === 1. 従来のEhlersDFTDominantCycle ===
        print("  🔄 従来版テスト...")
        original_detector = EhlersDFTDominantCycle(
            window=50,
            cycle_part=0.5,
            src_type='hlc3'
        )
        
        start_time = time.time()
        original_result = original_detector.calculate(data)
        original_time = time.time() - start_time
        
        size_results['original'] = {
            'time': original_time,
            'result_length': len(original_result),
            'avg_value': np.mean(original_result),
            'std_value': np.std(original_result)
        }
        
        # === 2. Ultra Supreme DFT (カルマンなし) ===
        print("  🚀 Ultra Supreme (カルマンなし)テスト...")
        supreme_no_kalman = EhlersUltraSupremeDFTCycle(
            base_window=50,
            cycle_part=0.5,
            src_type='hlc3',
            use_kalman_filter=False,
            adaptive_window=True,
            prediction_enabled=True
        )
        
        start_time = time.time()
        supreme_no_kalman_result = supreme_no_kalman.calculate(data)
        supreme_no_kalman_time = time.time() - start_time
        
        size_results['supreme_no_kalman'] = {
            'time': supreme_no_kalman_time,
            'result_length': len(supreme_no_kalman_result),
            'avg_value': np.mean(supreme_no_kalman_result),
            'std_value': np.std(supreme_no_kalman_result),
            'performance_stats': supreme_no_kalman.get_performance_stats()
        }
        
        # === 3. Ultra Supreme DFT (Neural Supreme Kalman) ===
        print("  🧠 Ultra Supreme + Neural Supreme Kalman テスト...")
        supreme_neural = EhlersUltraSupremeDFTCycle(
            base_window=50,
            cycle_part=0.5,
            src_type='hlc3',
            use_kalman_filter=True,
            kalman_filter_type='neural_supreme',
            kalman_pre_filter=True,
            kalman_post_refinement=True
        )
        
        start_time = time.time()
        supreme_neural_result = supreme_neural.calculate(data)
        supreme_neural_time = time.time() - start_time
        
        size_results['supreme_neural'] = {
            'time': supreme_neural_time,
            'result_length': len(supreme_neural_result),
            'avg_value': np.mean(supreme_neural_result),
            'std_value': np.std(supreme_neural_result),
            'performance_stats': supreme_neural.get_performance_stats(),
            'kalman_metadata': supreme_neural.get_kalman_metadata()
        }
        
        # === 4. Ultra Supreme DFT (Market Adaptive UKF) ===
        print("  🎯 Ultra Supreme + Market Adaptive UKF テスト...")
        supreme_market = EhlersUltraSupremeDFTCycle(
            base_window=50,
            cycle_part=0.5,
            src_type='hlc3',
            use_kalman_filter=True,
            kalman_filter_type='market_adaptive_unscented',
            kalman_pre_filter=True,
            kalman_post_refinement=False  # UKFは重いので事後処理なし
        )
        
        start_time = time.time()
        supreme_market_result = supreme_market.calculate(data)
        supreme_market_time = time.time() - start_time
        
        size_results['supreme_market'] = {
            'time': supreme_market_time,
            'result_length': len(supreme_market_result),
            'avg_value': np.mean(supreme_market_result),
            'std_value': np.std(supreme_market_result),
            'performance_stats': supreme_market.get_performance_stats(),
            'kalman_metadata': supreme_market.get_kalman_metadata()
        }
        
        results[f'size_{size}'] = size_results
        
        # パフォーマンス比較出力
        print(f"    従来版: {original_time:.4f}秒")
        print(f"    Supreme (カルマンなし): {supreme_no_kalman_time:.4f}秒 ({supreme_no_kalman_time/original_time:.2f}x)")
        print(f"    Supreme + Neural: {supreme_neural_time:.4f}秒 ({supreme_neural_time/original_time:.2f}x)")
        print(f"    Supreme + Market UKF: {supreme_market_time:.4f}秒 ({supreme_market_time/original_time:.2f}x)")
    
    return results

def test_accuracy_comparison():
    """
    精度比較テスト
    """
    print("\n🎯 精度比較テスト開始...")
    
    # 既知のサイクルを持つテストデータ
    n_points = 1000
    t = np.arange(n_points)
    
    # 明確な20期間サイクル
    true_cycle_period = 20
    signal = 10.0 * np.sin(2 * np.pi * t / true_cycle_period)
    noise = np.random.normal(0, 1, n_points)
    
    test_data = pd.DataFrame({
        'open': signal + noise,
        'high': signal + noise + 0.5,
        'low': signal + noise - 0.5,
        'close': signal + noise,
        'volume': np.random.lognormal(8, 0.3, n_points)
    })
    
    # 各検出器でテスト
    detectors = {
        '従来版': EhlersDFTDominantCycle(window=50, src_type='close'),
        'Supreme基本': EhlersUltraSupremeDFTCycle(
            base_window=50, src_type='close', use_kalman_filter=False
        ),
        'Supreme+Neural': EhlersUltraSupremeDFTCycle(
            base_window=50, src_type='close', use_kalman_filter=True,
            kalman_filter_type='neural_supreme'
        ),
        'Supreme+Adaptive': EhlersUltraSupremeDFTCycle(
            base_window=50, src_type='close', use_kalman_filter=True,
            kalman_filter_type='adaptive'
        )
    }
    
    results = {}
    
    for name, detector in detectors.items():
        print(f"  📏 {name} 精度テスト...")
        
        detected_cycles = detector.calculate(test_data)
        
        # 安定期間での平均（最初と最後の100ポイントを除外）
        stable_period = detected_cycles[100:-100]
        avg_detected = np.mean(stable_period)
        std_detected = np.std(stable_period)
        
        # 真値との誤差
        error_abs = abs(avg_detected - true_cycle_period)
        error_rel = error_abs / true_cycle_period * 100
        
        # 安定性（変動係数）
        stability = std_detected / avg_detected if avg_detected > 0 else float('inf')
        
        results[name] = {
            'detected_avg': avg_detected,
            'detected_std': std_detected,
            'absolute_error': error_abs,
            'relative_error_pct': error_rel,
            'stability_cv': stability,
            'detection_range': (np.min(stable_period), np.max(stable_period))
        }
        
        print(f"    平均検出値: {avg_detected:.2f} (真値: {true_cycle_period})")
        print(f"    絶対誤差: {error_abs:.2f}")
        print(f"    相対誤差: {error_rel:.1f}%")
        print(f"    安定性(CV): {stability:.3f}")
    
    return results, test_data, true_cycle_period

def create_comparison_charts(accuracy_results: Dict, test_data: pd.DataFrame, true_cycle: int):
    """
    比較チャート作成
    """
    print("\n📊 比較チャート作成中...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('🚀 Ultra Supreme DFT Cycle Detector Performance Comparison', fontsize=16)
    
    # === 1. 時系列比較 ===
    ax1 = axes[0, 0]
    
    # 各検出器で再計算（チャート用）
    detectors = {
        '従来版': EhlersDFTDominantCycle(window=50, src_type='close'),
        'Supreme+Neural': EhlersUltraSupremeDFTCycle(
            base_window=50, src_type='close', use_kalman_filter=True,
            kalman_filter_type='neural_supreme'
        )
    }
    
    for name, detector in detectors.items():
        cycles = detector.calculate(test_data)
        ax1.plot(cycles, label=name, alpha=0.8)
    
    ax1.axhline(y=true_cycle, color='red', linestyle='--', label=f'True Cycle ({true_cycle})')
    ax1.set_title('Detected Cycles Over Time')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Cycle Period')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # === 2. 精度比較バー ===
    ax2 = axes[0, 1]
    
    names = list(accuracy_results.keys())
    errors = [accuracy_results[name]['relative_error_pct'] for name in names]
    
    bars = ax2.bar(names, errors, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'])
    ax2.set_title('Relative Error Comparison')
    ax2.set_ylabel('Relative Error (%)')
    ax2.tick_params(axis='x', rotation=45)
    
    # バーに値を表示
    for bar, error in zip(bars, errors):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{error:.1f}%', ha='center', va='bottom')
    
    # === 3. 安定性比較 ===
    ax3 = axes[1, 0]
    
    stabilities = [accuracy_results[name]['stability_cv'] for name in names]
    
    bars = ax3.bar(names, stabilities, color=['#ffa726', '#26a69a', '#42a5f5', '#66bb6a'])
    ax3.set_title('Stability Comparison (Lower = Better)')
    ax3.set_ylabel('Coefficient of Variation')
    ax3.tick_params(axis='x', rotation=45)
    
    for bar, stability in zip(bars, stabilities):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{stability:.3f}', ha='center', va='bottom')
    
    # === 4. 価格データ ===
    ax4 = axes[1, 1]
    
    ax4.plot(test_data['close'], label='Price', color='black', alpha=0.7)
    ax4.set_title('Test Data (Known 20-Period Cycle)')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Price')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    chart_filename = 'ultra_supreme_dft_comparison.png'
    plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
    print(f"📊 チャート保存: {chart_filename}")
    
    return chart_filename

def test_kalman_filters():
    """
    各種カルマンフィルターのテスト
    """
    print("\n🧠 カルマンフィルター比較テスト...")
    
    # テストデータ
    data = generate_synthetic_data(800)
    
    # 利用可能なカルマンフィルター
    kalman_types = [
        'adaptive',
        'quantum_adaptive', 
        'unscented',
        'hyper_quantum',
        'neural_supreme',
        'market_adaptive_unscented'
    ]
    
    results = {}
    
    for kalman_type in kalman_types:
        print(f"  🔬 {kalman_type} テスト...")
        
        try:
            detector = EhlersUltraSupremeDFTCycle(
                base_window=40,
                src_type='hlc3',
                use_kalman_filter=True,
                kalman_filter_type=kalman_type,
                kalman_pre_filter=True,
                kalman_post_refinement=False  # 統一条件
            )
            
            start_time = time.time()
            cycles = detector.calculate(data)
            execution_time = time.time() - start_time
            
            # 統計
            stats = {
                'execution_time': execution_time,
                'avg_cycle': np.mean(cycles),
                'std_cycle': np.std(cycles),
                'min_cycle': np.min(cycles),
                'max_cycle': np.max(cycles),
                'performance_stats': detector.get_performance_stats(),
                'kalman_metadata': detector.get_kalman_metadata()
            }
            
            results[kalman_type] = stats
            
            print(f"    実行時間: {execution_time:.4f}秒")
            print(f"    平均サイクル: {stats['avg_cycle']:.2f}")
            print(f"    標準偏差: {stats['std_cycle']:.2f}")
            
        except Exception as e:
            print(f"    ❌ エラー: {e}")
            results[kalman_type] = {'error': str(e)}
    
    return results

def main():
    """
    メインテスト実行
    """
    print("🚀🧠 ULTRA SUPREME DFT CYCLE DETECTOR TEST SUITE 🧠🚀")
    print("=" * 60)
    
    # === 1. 性能ベンチマーク ===
    performance_results = benchmark_performance()
    
    # === 2. 精度比較 ===
    accuracy_results, test_data, true_cycle = test_accuracy_comparison()
    
    # === 3. チャート作成 ===
    chart_file = create_comparison_charts(accuracy_results, test_data, true_cycle)
    
    # === 4. カルマンフィルター比較 ===
    kalman_results = test_kalman_filters()
    
    # === 5. 総合結果表示 ===
    print("\n" + "="*60)
    print("🏆 総合テスト結果サマリー")
    print("="*60)
    
    print("\n📊 精度ランキング (相対誤差ベース):")
    accuracy_ranking = sorted(accuracy_results.items(), 
                            key=lambda x: x[1]['relative_error_pct'])
    for i, (name, result) in enumerate(accuracy_ranking, 1):
        print(f"  {i}. {name}: {result['relative_error_pct']:.1f}% 誤差")
    
    print("\n⚡ カルマンフィルター性能:")
    kalman_ranking = [(k, v) for k, v in kalman_results.items() if 'error' not in v]
    kalman_ranking.sort(key=lambda x: x[1]['execution_time'])
    
    for i, (name, result) in enumerate(kalman_ranking, 1):
        print(f"  {i}. {name}: {result['execution_time']:.4f}秒")
    
    print(f"\n📊 比較チャート: {chart_file}")
    print("\n✅ 全テスト完了!")
    
    return {
        'performance': performance_results,
        'accuracy': accuracy_results,
        'kalman': kalman_results,
        'chart_file': chart_file
    }

if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        print(f"\n❌ テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()