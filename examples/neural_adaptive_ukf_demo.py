#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🧠 **Neural Adaptive UKF Demo** 🧠

論文「Adaptive Neural Unscented Kalman Filter」
by Amit Levy & Itzik Klein, arXiv:2503.05490v2 のデモ

🌟 **特徴:**
1. **ProcessNet**: CNNベース回帰ネットワーク
2. **リアルタイム学習**: センサー読み値のみでプロセスノイズ推定
3. **エンドツーエンド**: 完全自動適応システム
4. **AUV航行**: 自律水中航行体ナビゲーション応用
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
import sys

# モジュールパスの追加
project_root = Path(__file__).parent.parent
if project_root not in sys.path:
    sys.path.append(str(project_root))

try:
    from indicators.neural_adaptive_ukf import NeuralAdaptiveUnscentedKalmanFilter
    from indicators.unscented_kalman_filter import UnscentedKalmanFilter
except ImportError as e:
    print(f"⚠️ インポートエラー: {e}")
    sys.exit(1)

# 設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')


def generate_auv_navigation_data(n_points: int = 1000) -> tuple:
    """
    AUV航行データ模擬生成
    
    水中航行体の特徴を模擬：
    - 複雑な3次元軌道
    - センサーノイズ変動
    - 環境外乱
    - DVL信号断続
    """
    t = np.linspace(0, 100, n_points)  # 100秒間
    
    # 基本軌道（螺旋 + ランダムウォーク）
    spiral_x = 50 * np.cos(0.2 * t) + np.cumsum(np.random.normal(0, 0.1, n_points))
    spiral_y = 50 * np.sin(0.2 * t) + np.cumsum(np.random.normal(0, 0.1, n_points))
    spiral_z = -t * 0.5 + 10 * np.sin(0.1 * t)  # 深度変化
    
    # 複雑軌道
    true_position = spiral_x + spiral_y + spiral_z * 0.1
    
    # 環境ノイズ（深度・流れに依存）
    depth_factor = np.abs(spiral_z) / 50 + 0.1
    current_noise = depth_factor * np.random.normal(0, 1, n_points)
    
    # センサーノイズ（時変）
    sensor_quality = 1.0 + 0.5 * np.sin(0.05 * t)  # 周期的品質変化
    sensor_noise = np.random.normal(0, sensor_quality, n_points)
    
    # DVL信号断続模擬
    dvl_outage = np.zeros(n_points, dtype=bool)
    outage_periods = [(200, 220), (450, 480), (700, 730)]  # 信号断続期間
    for start, end in outage_periods:
        if start < n_points and end < n_points:
            dvl_outage[start:end] = True
    
    # 観測データ
    observed_position = true_position + current_noise + sensor_noise
    observed_position[dvl_outage] += np.random.normal(0, 5, np.sum(dvl_outage))  # 断続時の大きなノイズ
    
    return true_position, observed_position, sensor_quality, dvl_outage, depth_factor


def run_neural_adaptive_ukf_demo():
    """Neural Adaptive UKFデモ実行"""
    print("🧠 **Neural Adaptive UKF Demo (Levy & Klein 2025)** 🧠")
    print("="*60)
    
    # AUV航行データ生成
    print("🌊 AUV航行データ生成中...")
    true_pos, observed_pos, sensor_quality, dvl_outage, depth_factor = generate_auv_navigation_data(800)
    
    print(f"   - データ点数: {len(observed_pos)}")
    print(f"   - DVL信号断続期間: {np.sum(dvl_outage)}点 ({np.sum(dvl_outage)/len(observed_pos)*100:.1f}%)")
    
    # Neural Adaptive UKF実行
    print("\n🧠 Neural Adaptive UKF実行中...")
    neural_aukf = NeuralAdaptiveUnscentedKalmanFilter(window_size=100)
    neural_result = neural_aukf.calculate(observed_pos)
    
    # 標準UKF比較用
    print("📊 標準UKF比較実行中...")
    standard_ukf = UnscentedKalmanFilter()
    standard_result = standard_ukf.calculate(observed_pos)
    
    # 性能評価
    neural_rmse = np.sqrt(np.mean((neural_result.filtered_values - true_pos) ** 2))
    standard_rmse = np.sqrt(np.mean((standard_result.filtered_values - true_pos) ** 2))
    improvement = (standard_rmse - neural_rmse) / standard_rmse * 100
    
    print(f"\n📈 **性能結果:**")
    print(f"   🧠 Neural AUKF RMSE: {neural_rmse:.4f}")
    print(f"   📊 Standard UKF RMSE: {standard_rmse:.4f}")
    print(f"   🚀 改善率: {improvement:+.1f}%")
    
    # Neural固有情報
    neural_summary = neural_aukf.get_neural_summary()
    print(f"\n🧠 **Neural特徴:**")
    for key, value in neural_summary.items():
        if isinstance(value, list):
            print(f"   {key}: {', '.join(value)}")
        elif isinstance(value, (int, float)):
            print(f"   {key}: {value:.6f}")
        else:
            print(f"   {key}: {value}")
    
    # 可視化
    create_neural_visualization(
        true_pos, observed_pos, neural_result, standard_result, 
        sensor_quality, dvl_outage, depth_factor
    )
    
    print("\n🎉 Neural Adaptive UKFデモ完了！")


def create_neural_visualization(
    true_pos, observed_pos, neural_result, standard_result, 
    sensor_quality, dvl_outage, depth_factor
):
    """Neural特有の可視化"""
    
    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    
    # 1. メインフィルタリング結果
    ax1 = axes[0, 0]
    ax1.plot(true_pos, 'k-', linewidth=2, label='True Position', alpha=0.8)
    ax1.plot(observed_pos, color='lightgray', alpha=0.5, label='Observed', linewidth=0.5)
    ax1.plot(neural_result.filtered_values, 'g-', linewidth=2, label='Neural AUKF', alpha=0.9)
    ax1.plot(standard_result.filtered_values, 'b--', linewidth=2, label='Standard UKF', alpha=0.7)
    
    # DVL断続期間をハイライト
    dvl_periods = np.where(dvl_outage)[0]
    if len(dvl_periods) > 0:
        ax1.scatter(dvl_periods, observed_pos[dvl_periods], 
                   color='red', s=20, alpha=0.7, label='DVL Outage')
    
    ax1.set_title('🧠 Neural AUKF vs Standard UKF', fontweight='bold')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Position')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. ニューラル推定ノイズ
    ax2 = axes[0, 1]
    ax2.plot(neural_result.neural_process_noise, 'g-', linewidth=2, label='Process Noise', alpha=0.8)
    ax2.plot(neural_result.neural_observation_noise, 'r-', linewidth=2, label='Observation Noise', alpha=0.8)
    ax2.set_title('🔧 Neural Noise Estimation', fontweight='bold')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Noise Level')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 学習曲線と適応信号
    ax3 = axes[1, 0]
    ax3.plot(neural_result.learning_curves, 'purple', linewidth=2, label='Learning Curve', alpha=0.8)
    ax3.plot(neural_result.adaptation_signals, 'orange', linewidth=2, label='Adaptation Signal', alpha=0.8)
    ax3.set_title('📚 Learning Progress', fontweight='bold')
    ax3.set_xlabel('Time Steps')
    ax3.set_ylabel('Signal Strength')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. ネットワーク信頼度
    ax4 = axes[1, 1]
    ax4.plot(neural_result.network_confidence, 'cyan', linewidth=2, label='Network Confidence', alpha=0.8)
    ax4.fill_between(range(len(neural_result.network_confidence)), 
                     neural_result.network_confidence, alpha=0.3, color='cyan')
    ax4.set_title('🎯 Network Confidence', fontweight='bold')
    ax4.set_xlabel('Time Steps')
    ax4.set_ylabel('Confidence Score')
    ax4.set_ylim(0, 1.1)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 速度・加速度推定
    ax5 = axes[2, 0]
    ax5.plot(neural_result.velocity_estimates, 'g-', linewidth=2, label='Velocity', alpha=0.8)
    ax5.plot(neural_result.acceleration_estimates, 'r--', linewidth=2, label='Acceleration', alpha=0.8)
    ax5.set_title('🚀 Velocity & Acceleration Estimates', fontweight='bold')
    ax5.set_xlabel('Time Steps')
    ax5.set_ylabel('Derivative Values')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 環境要因とネットワーク出力
    ax6 = axes[2, 1]
    ax6.plot(sensor_quality, 'blue', linewidth=2, label='Sensor Quality', alpha=0.7)
    ax6.plot(depth_factor, 'brown', linewidth=2, label='Depth Factor', alpha=0.7)
    ax6.plot(neural_result.network_outputs / np.max(neural_result.network_outputs), 
             'green', linewidth=2, label='Network Output (norm)', alpha=0.8)
    ax6.set_title('🌊 Environmental Factors', fontweight='bold')
    ax6.set_xlabel('Time Steps')
    ax6.set_ylabel('Normalized Values')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    output_file = project_root / "output" / "neural_adaptive_ukf_demo.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n📊 Neural UKF可視化保存: {output_file}")
    
    plt.show()


if __name__ == "__main__":
    run_neural_adaptive_ukf_demo() 