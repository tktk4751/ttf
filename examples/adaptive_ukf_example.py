#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 **Adaptive UKF 使用例** 🎯

標準UKFを大幅に超える適応的フィルタリングのデモンストレーション
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from indicators.adaptive_ukf import AdaptiveUnscentedKalmanFilter
from indicators.unscented_kalman_filter import UnscentedKalmanFilter

def create_challenging_data(n_points: int = 1000) -> pd.DataFrame:
    """挑戦的なデータセット作成（ノイズ変化、異常値含む）"""
    np.random.seed(42)
    
    # 基本トレンド
    base_price = 100.0
    trend = np.linspace(0, 50, n_points)
    
    # 時変ノイズ（途中でノイズレベルが変化）
    noise1 = np.random.normal(0, 1.0, n_points // 3)      # 低ノイズ期間
    noise2 = np.random.normal(0, 5.0, n_points // 3)      # 高ノイズ期間
    noise3 = np.random.normal(0, 2.0, n_points - 2 * (n_points // 3))  # 中ノイズ期間
    noise = np.concatenate([noise1, noise2, noise3])
    
    # 基本価格
    prices = base_price + trend + noise
    
    # 異常値の挿入（5%の確率）
    outlier_indices = np.random.choice(n_points, size=int(n_points * 0.05), replace=False)
    for idx in outlier_indices:
        prices[idx] += np.random.normal(0, 20)  # 大きな外れ値
    
    # OHLC データ生成
    highs = prices + np.abs(np.random.normal(0, 1, n_points))
    lows = prices - np.abs(np.random.normal(0, 1, n_points))
    opens = prices + np.random.normal(0, 0.5, n_points)
    
    return pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': prices
    }), outlier_indices

def compare_filters():
    """フィルター比較デモ"""
    print("🎯 Adaptive UKF vs 標準UKF 性能比較")
    print("=" * 50)
    
    # 挑戦的データセット作成
    data, outlier_indices = create_challenging_data(800)
    print(f"📊 挑戦的データセット作成: {len(data)}ポイント")
    print(f"   - 時変ノイズ（低→高→中）")
    print(f"   - 異常値 {len(outlier_indices)}個 ({len(outlier_indices)/len(data)*100:.1f}%)")
    
    # フィルター初期化
    adaptive_ukf = AdaptiveUnscentedKalmanFilter(
        src_type='close',
        innovation_window=20,
        outlier_threshold=3.0,
        alpha_min=0.0001,
        alpha_max=0.1
    )
    
    standard_ukf = UnscentedKalmanFilter(
        src_type='close',
        alpha=0.001,
        beta=2.0,
        kappa=0.0
    )
    
    # フィルタリング実行
    print("\n🔧 フィルタリング実行中...")
    aukf_result = adaptive_ukf.calculate(data)
    ukf_result = standard_ukf.calculate(data)
    
    # 性能評価
    original_prices = data['close'].values
    
    # RMSE計算（異常値を除外）
    normal_indices = np.setdiff1d(np.arange(len(data)), outlier_indices)
    
    aukf_rmse = np.sqrt(np.mean((aukf_result.filtered_values[normal_indices] - 
                                original_prices[normal_indices]) ** 2))
    ukf_rmse = np.sqrt(np.mean((ukf_result.filtered_values[normal_indices] - 
                               original_prices[normal_indices]) ** 2))
    
    print(f"\n📈 性能評価結果:")
    print(f"   Adaptive UKF RMSE: {aukf_rmse:.4f}")
    print(f"   標準UKF RMSE:      {ukf_rmse:.4f}")
    print(f"   改善率:           {(ukf_rmse - aukf_rmse) / ukf_rmse * 100:.1f}%")
    
    # 適応機能の分析
    adaptation_summary = adaptive_ukf.get_adaptation_summary()
    print(f"\n🔍 適応機能分析:")
    print(f"   異常値検出率:     {adaptation_summary['outlier_detection_rate']:.1%}")
    print(f"   平均α値:         {adaptation_summary['avg_adaptive_alpha']:.6f}")
    print(f"   α調整範囲:       {adaptation_summary['alpha_range'][0]:.6f} - {adaptation_summary['alpha_range'][1]:.6f}")
    print(f"   フェージング率:   {adaptation_summary['fading_activation_rate']:.1%}")
    
    # 可視化
    plt.figure(figsize=(16, 12))
    
    # 1. フィルタリング結果比較
    plt.subplot(3, 2, 1)
    plt.plot(original_prices, label='元の価格', alpha=0.7, color='gray')
    plt.plot(aukf_result.filtered_values, label='Adaptive UKF', linewidth=2, color='blue')
    plt.plot(ukf_result.filtered_values, label='標準UKF', linewidth=1, color='red')
    plt.scatter(outlier_indices, original_prices[outlier_indices], 
               color='orange', s=30, label='異常値', zorder=5)
    plt.title('フィルタリング結果比較')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 適応的α値
    plt.subplot(3, 2, 2)
    plt.plot(aukf_result.adaptive_alpha, color='green', linewidth=2)
    plt.title('適応的α値の変化')
    plt.ylabel('α値')
    plt.grid(True, alpha=0.3)
    
    # 3. 異常値検出
    plt.subplot(3, 2, 3)
    plt.plot(aukf_result.outlier_flags, color='red', linewidth=1)
    plt.title('異常値検出フラグ')
    plt.ylabel('異常値フラグ')
    plt.grid(True, alpha=0.3)
    
    # 4. 適応的ノイズ
    plt.subplot(3, 2, 4)
    plt.plot(aukf_result.adaptive_observation_noise, color='purple', linewidth=1)
    plt.title('適応的観測ノイズ')
    plt.ylabel('ノイズ分散')
    plt.grid(True, alpha=0.3)
    
    # 5. 不確実性比較
    plt.subplot(3, 2, 5)
    plt.plot(aukf_result.uncertainty, label='Adaptive UKF', color='blue')
    plt.plot(ukf_result.uncertainty, label='標準UKF', color='red')
    plt.title('推定不確実性比較')
    plt.ylabel('不確実性')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. 共分散フェージング
    plt.subplot(3, 2, 6)
    plt.plot(aukf_result.covariance_fading, color='orange', linewidth=1)
    plt.title('共分散フェージング係数')
    plt.ylabel('フェージング係数')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/adaptive_ukf_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 比較チャートを保存: output/adaptive_ukf_comparison.png")
    
    return aukf_result, ukf_result, adaptation_summary

def demonstrate_adaptivity():
    """適応性のデモンストレーション"""
    print("\n🔄 適応性デモンストレーション")
    print("-" * 30)
    
    # 段階的ノイズ変化データ
    segments = 3
    points_per_segment = 200
    total_points = segments * points_per_segment
    
    prices = []
    base_price = 100.0
    
    for i in range(segments):
        # 各セグメントで異なるノイズレベル
        noise_levels = [0.5, 3.0, 1.0]  # 低→高→中
        noise_level = noise_levels[i]
        
        segment_prices = base_price + np.arange(points_per_segment) * 0.05
        segment_prices += np.random.normal(0, noise_level, points_per_segment)
        
        prices.extend(segment_prices)
        base_price = segment_prices[-1]
    
    data = pd.DataFrame({
        'open': prices,
        'high': np.array(prices) + 1,
        'low': np.array(prices) - 1,
        'close': prices
    })
    
    # Adaptive UKF適用
    aukf = AdaptiveUnscentedKalmanFilter(src_type='close')
    result = aukf.calculate(data)
    
    # 可視化
    plt.figure(figsize=(15, 10))
    
    # セグメント境界
    segment_boundaries = [points_per_segment * i for i in range(1, segments)]
    
    plt.subplot(2, 2, 1)
    plt.plot(prices, label='元の価格', alpha=0.7)
    plt.plot(result.filtered_values, label='Adaptive UKF', linewidth=2)
    for boundary in segment_boundaries:
        plt.axvline(x=boundary, color='red', linestyle='--', alpha=0.7)
    plt.title('段階的ノイズ変化への適応')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(result.adaptive_observation_noise)
    for boundary in segment_boundaries:
        plt.axvline(x=boundary, color='red', linestyle='--', alpha=0.7)
    plt.title('観測ノイズの適応的推定')
    plt.ylabel('推定ノイズ分散')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(result.adaptive_alpha)
    for boundary in segment_boundaries:
        plt.axvline(x=boundary, color='red', linestyle='--', alpha=0.7)
    plt.title('αパラメータの動的調整')
    plt.ylabel('α値')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.plot(result.confidence_scores)
    for boundary in segment_boundaries:
        plt.axvline(x=boundary, color='red', linestyle='--', alpha=0.7)
    plt.title('信頼度スコアの変化')
    plt.ylabel('信頼度')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/adaptive_ukf_adaptivity.png', dpi=300, bbox_inches='tight')
    print(f"📊 適応性デモチャートを保存: output/adaptive_ukf_adaptivity.png")

def main():
    """メイン実行"""
    # 出力ディレクトリ作成
    os.makedirs('output', exist_ok=True)
    
    try:
        # フィルター比較
        aukf_result, ukf_result, summary = compare_filters()
        
        # 適応性デモ
        demonstrate_adaptivity()
        
        print("\n🎉 Adaptive UKF デモンストレーション完了！")
        print("\n💡 Adaptive UKFの主な利点:")
        for feature in summary['adaptive_features']:
            print(f"   ✅ {feature}")
            
    except Exception as e:
        print(f"❌ エラー: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 