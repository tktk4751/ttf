#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🧠 **四種のAdaptive UKF比較デモ** 🧠

4つの手法を比較：
1. **標準UKF** (基準)
2. **私の実装版AUKF** (統計的監視・適応制御)
3. **論文版AUKF** (Ge et al. 2019 - 相互相関理論)
4. **Neural版AUKF** (Levy & Klein 2025 - CNN ProcessNet)

🌟 **比較項目:**
- フィルタリング精度 (RMSE)
- 適応性能
- 計算効率
- 理論的厳密性
- 実用性
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import time
from pathlib import Path
import sys

# モジュールパスの追加
project_root = Path(__file__).parent.parent
if project_root not in sys.path:
    sys.path.append(str(project_root))

try:
    from indicators.unscented_kalman_filter import UnscentedKalmanFilter
    from indicators.adaptive_ukf import AdaptiveUnscentedKalmanFilter
    from indicators.academic_adaptive_ukf import AcademicAdaptiveUnscentedKalmanFilter
    from indicators.neural_adaptive_ukf import NeuralAdaptiveUnscentedKalmanFilter
except ImportError as e:
    print(f"⚠️ インポートエラー: {e}")
    print("indicators/ディレクトリにUKFモジュールが存在することを確認してください。")
    sys.exit(1)

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 警告無効化
warnings.filterwarnings('ignore')


def generate_complex_test_data(n_points: int = 1000, noise_level: float = 0.5) -> tuple:
    """
    複雑なテストデータ生成
    
    特徴：
    - 複数のトレンド変化
    - 異常値を含む
    - 時変ノイズ
    - 周期性成分
    
    Returns:
        (true_signal, noisy_signal, outliers, noise_changes)
    """
    t = np.linspace(0, 10, n_points)
    
    # 真の信号（複雑な構造）
    trend1 = 100 + 10 * t
    trend2 = np.where(t > 3, 15 * (t - 3), 0)  # 3秒後からトレンド変化
    trend3 = np.where(t > 7, -20 * (t - 7), 0)  # 7秒後から下降トレンド
    
    # 周期成分
    seasonal = 5 * np.sin(2 * np.pi * t / 2) + 2 * np.cos(2 * np.pi * t / 0.5)
    
    # 真の信号
    true_signal = trend1 + trend2 + trend3 + seasonal
    
    # 時変ノイズ
    noise_phases = np.array([0.1, 0.3, 0.8, 0.4])  # 4段階のノイズレベル
    phase_boundaries = [0, 0.25, 0.5, 0.75, 1.0]
    
    noise = np.zeros(n_points)
    noise_changes = np.zeros(n_points)
    
    for i in range(len(noise_phases)):
        start_idx = int(phase_boundaries[i] * n_points)
        end_idx = int(phase_boundaries[i + 1] * n_points)
        
        phase_noise = noise_phases[i] * noise_level
        noise[start_idx:end_idx] = np.random.normal(0, phase_noise, end_idx - start_idx)
        noise_changes[start_idx:end_idx] = phase_noise
    
    # 異常値追加（全体の3%）
    n_outliers = int(n_points * 0.03)
    outlier_indices = np.random.choice(n_points, n_outliers, replace=False)
    outlier_values = np.random.normal(0, noise_level * 5, n_outliers)
    
    outliers = np.zeros(n_points, dtype=bool)
    outliers[outlier_indices] = True
    
    # ノイジー信号
    noisy_signal = true_signal + noise
    noisy_signal[outlier_indices] += outlier_values
    
    print(f"📊 **テストデータ生成完了**")
    print(f"   - データ点数: {n_points}")
    print(f"   - 異常値数: {n_outliers} ({n_outliers/n_points*100:.1f}%)")
    print(f"   - ノイズ段階: {len(noise_phases)}段階")
    print(f"   - ノイズレベル: {noise_phases}")
    
    return true_signal, noisy_signal, outliers, noise_changes


def run_four_methods_comparison(data: np.ndarray, true_values: np.ndarray) -> dict:
    """
    4つの手法でフィルタリング実行
    
    Args:
        data: 観測データ
        true_values: 真の値
    
    Returns:
        各手法の結果辞書
    """
    methods = {}
    
    print(f"\n🔄 **4手法比較実行中...**")
    
    # 1. 標準UKF（基準）
    print("   1️⃣ 標準UKF計算中...")
    start_time = time.time()
    
    try:
        ukf_standard = UnscentedKalmanFilter()
        ukf_result = ukf_standard.calculate(data)
        
        methods['Standard UKF'] = {
            'result': ukf_result,
            'filtered_values': ukf_result.filtered_values,
            'computation_time': time.time() - start_time,
            'type': 'baseline',
            'description': '標準UKF（固定パラメータ）'
        }
        print(f"      ✅ 完了 ({methods['Standard UKF']['computation_time']:.3f}秒)")
        
    except Exception as e:
        print(f"      ❌ エラー: {str(e)}")
        methods['Standard UKF'] = None
    
    # 2. 私の実装版AUKF（統計的監視・適応制御）
    print("   2️⃣ 私の実装版AUKF計算中...")
    start_time = time.time()
    
    try:
        aukf_mine = AdaptiveUnscentedKalmanFilter()
        aukf_result = aukf_mine.calculate(data)
        
        methods['My Implementation AUKF'] = {
            'result': aukf_result,
            'filtered_values': aukf_result.filtered_values,
            'computation_time': time.time() - start_time,
            'type': 'statistical_adaptive',
            'description': '統計的監視・適応制御版'
        }
        print(f"      ✅ 完了 ({methods['My Implementation AUKF']['computation_time']:.3f}秒)")
        
    except Exception as e:
        print(f"      ❌ エラー: {str(e)}")
        methods['My Implementation AUKF'] = None
    
    # 3. 論文版AUKF（Ge et al. 2019 - 相互相関理論）
    print("   3️⃣ 論文版AUKF (Ge et al. 2019) 計算中...")
    start_time = time.time()
    
    try:
        aukf_academic = AcademicAdaptiveUnscentedKalmanFilter()
        academic_result = aukf_academic.calculate(data)
        
        methods['Academic AUKF (Ge 2019)'] = {
            'result': academic_result,
            'filtered_values': academic_result.filtered_values,
            'computation_time': time.time() - start_time,
            'type': 'mathematical_rigorous',
            'description': '相互相関理論版（Ge et al.）'
        }
        print(f"      ✅ 完了 ({methods['Academic AUKF (Ge 2019)']['computation_time']:.3f}秒)")
        
    except Exception as e:
        print(f"      ❌ エラー: {str(e)}")
        methods['Academic AUKF (Ge 2019)'] = None
    
    # 4. Neural版AUKF（Levy & Klein 2025 - CNN ProcessNet）
    print("   4️⃣ Neural版AUKF (Levy & Klein 2025) 計算中...")
    start_time = time.time()
    
    try:
        aukf_neural = NeuralAdaptiveUnscentedKalmanFilter()
        neural_result = aukf_neural.calculate(data)
        
        methods['Neural AUKF (Levy 2025)'] = {
            'result': neural_result,
            'filtered_values': neural_result.filtered_values,
            'computation_time': time.time() - start_time,
            'type': 'neural_adaptive',
            'description': 'CNN ProcessNet版（Levy & Klein）'
        }
        print(f"      ✅ 完了 ({methods['Neural AUKF (Levy 2025)']['computation_time']:.3f}秒)")
        
    except Exception as e:
        print(f"      ❌ エラー: {str(e)}")
        methods['Neural AUKF (Levy 2025)'] = None
    
    return methods


def calculate_performance_metrics(methods: dict, true_values: np.ndarray, outliers: np.ndarray) -> pd.DataFrame:
    """
    性能指標計算
    
    Args:
        methods: 各手法の結果
        true_values: 真の値
        outliers: 異常値マスク
    
    Returns:
        性能指標のDataFrame
    """
    metrics_list = []
    
    for method_name, method_data in methods.items():
        if method_data is None:
            continue
        
        filtered_values = method_data['filtered_values']
        
        if len(filtered_values) != len(true_values):
            continue
        
        # 基本指標
        rmse_all = np.sqrt(np.mean((filtered_values - true_values) ** 2))
        mae_all = np.mean(np.abs(filtered_values - true_values))
        
        # クリーンデータでの性能（異常値除外）
        clean_mask = ~outliers
        if np.sum(clean_mask) > 0:
            rmse_clean = np.sqrt(np.mean((filtered_values[clean_mask] - true_values[clean_mask]) ** 2))
            mae_clean = np.mean(np.abs(filtered_values[clean_mask] - true_values[clean_mask]))
        else:
            rmse_clean = rmse_all
            mae_clean = mae_all
        
        # 計算時間
        comp_time = method_data['computation_time']
        
        # 手法タイプ
        method_type = method_data['type']
        description = method_data['description']
        
        metrics_list.append({
            'Method': method_name,
            'Type': method_type,
            'Description': description,
            'RMSE_All': rmse_all,
            'MAE_All': mae_all,
            'RMSE_Clean': rmse_clean,
            'MAE_Clean': mae_clean,
            'Computation_Time': comp_time,
            'Speed_Rank': 0  # 後で設定
        })
    
    metrics_df = pd.DataFrame(metrics_list)
    
    # 速度ランキング
    if len(metrics_df) > 0:
        metrics_df = metrics_df.sort_values('Computation_Time')
        metrics_df['Speed_Rank'] = range(1, len(metrics_df) + 1)
        
        # RMSE改善率計算（標準UKFを基準とする）
        baseline_rmse = None
        for _, row in metrics_df.iterrows():
            if 'Standard' in row['Method']:
                baseline_rmse = row['RMSE_All']
                break
        
        if baseline_rmse is not None:
            metrics_df['RMSE_Improvement'] = ((baseline_rmse - metrics_df['RMSE_All']) / baseline_rmse * 100)
        else:
            metrics_df['RMSE_Improvement'] = 0
    
    return metrics_df


def create_comprehensive_visualization(
    methods: dict,
    true_values: np.ndarray,
    noisy_data: np.ndarray,
    outliers: np.ndarray,
    metrics_df: pd.DataFrame
):
    """
    包括的可視化作成
    """
    # 手法の色とスタイル
    method_styles = {
        'Standard UKF': {'color': 'gray', 'linestyle': '-', 'alpha': 0.8},
        'My Implementation AUKF': {'color': 'blue', 'linestyle': '-', 'alpha': 0.9},
        'Academic AUKF (Ge 2019)': {'color': 'red', 'linestyle': '-', 'alpha': 0.9},
        'Neural AUKF (Levy 2025)': {'color': 'green', 'linestyle': '-', 'alpha': 0.9}
    }
    
    # メインプロット作成
    fig = plt.figure(figsize=(20, 16))
    
    # 1. メインフィルタリング結果比較
    ax1 = plt.subplot(3, 2, (1, 2))
    
    # 真の値
    plt.plot(true_values, 'k-', linewidth=2, label='True Signal', alpha=0.8)
    
    # ノイジーデータ
    plt.plot(noisy_data, color='lightgray', alpha=0.5, label='Noisy Data', linewidth=0.5)
    
    # 異常値をハイライト
    outlier_indices = np.where(outliers)[0]
    if len(outlier_indices) > 0:
        plt.scatter(outlier_indices, noisy_data[outlier_indices], 
                   color='orange', s=30, alpha=0.7, label='Outliers', zorder=5)
    
    # 各手法の結果
    for method_name, method_data in methods.items():
        if method_data is None:
            continue
        
        style = method_styles.get(method_name, {'color': 'purple', 'linestyle': '-', 'alpha': 0.8})
        plt.plot(method_data['filtered_values'], 
                label=method_name,
                color=style['color'],
                linestyle=style['linestyle'],
                linewidth=2,
                alpha=style['alpha'])
    
    plt.title('🧠 Four Adaptive UKF Methods Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 2. 性能指標バーチャート
    ax2 = plt.subplot(3, 2, 3)
    
    if len(metrics_df) > 0:
        methods_short = [name.split('(')[0].strip() for name in metrics_df['Method']]
        colors = [method_styles.get(full_name, {'color': 'purple'})['color'] 
                 for full_name in metrics_df['Method']]
        
        bars = plt.bar(methods_short, metrics_df['RMSE_All'], color=colors, alpha=0.7)
        
        # 値をバーの上に表示
        for bar, rmse in zip(bars, metrics_df['RMSE_All']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01,
                    f'{rmse:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.title('RMSE Comparison (All Data)', fontweight='bold')
        plt.ylabel('RMSE')
        plt.xticks(rotation=45)
    
    # 3. クリーンデータでのRMSE
    ax3 = plt.subplot(3, 2, 4)
    
    if len(metrics_df) > 0:
        bars = plt.bar(methods_short, metrics_df['RMSE_Clean'], color=colors, alpha=0.7)
        
        for bar, rmse in zip(bars, metrics_df['RMSE_Clean']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01,
                    f'{rmse:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.title('RMSE Comparison (Clean Data)', fontweight='bold')
        plt.ylabel('RMSE')
        plt.xticks(rotation=45)
    
    # 4. 計算時間比較
    ax4 = plt.subplot(3, 2, 5)
    
    if len(metrics_df) > 0:
        bars = plt.bar(methods_short, metrics_df['Computation_Time'], color=colors, alpha=0.7)
        
        for bar, time_val in zip(bars, metrics_df['Computation_Time']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01,
                    f'{time_val:.3f}s', ha='center', va='bottom', fontweight='bold')
        
        plt.title('Computation Time Comparison', fontweight='bold')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45)
    
    # 5. RMSE改善率
    ax5 = plt.subplot(3, 2, 6)
    
    if len(metrics_df) > 0 and 'RMSE_Improvement' in metrics_df.columns:
        colors_improvement = ['green' if x > 0 else 'red' for x in metrics_df['RMSE_Improvement']]
        bars = plt.bar(methods_short, metrics_df['RMSE_Improvement'], 
                      color=colors_improvement, alpha=0.7)
        
        for bar, improvement in zip(bars, metrics_df['RMSE_Improvement']):
            plt.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + (5 if bar.get_height() >= 0 else -8),
                    f'{improvement:+.1f}%', ha='center', va='bottom' if bar.get_height() >= 0 else 'top',
                    fontweight='bold')
        
        plt.title('RMSE Improvement vs Standard UKF', fontweight='bold')
        plt.ylabel('Improvement (%)')
        plt.xticks(rotation=45)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    output_file = project_root / "output" / "adaptive_ukf_four_methods_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n📊 **比較可視化保存**: {output_file}")
    
    plt.show()


def print_detailed_analysis(metrics_df: pd.DataFrame, methods: dict):
    """
    詳細分析結果出力
    """
    print("\n" + "="*80)
    print("🧠 **FOUR ADAPTIVE UKF METHODS - DETAILED ANALYSIS** 🧠")
    print("="*80)
    
    if len(metrics_df) == 0:
        print("❌ 有効な結果がありません")
        return
    
    # 基本統計
    print("\n📊 **基本性能統計:**")
    print("-"*60)
    
    for _, row in metrics_df.iterrows():
        print(f"🔹 **{row['Method']}**")
        print(f"   📝 説明: {row['Description']}")
        print(f"   📈 全体RMSE: {row['RMSE_All']:.6f}")
        print(f"   🧹 クリーンRMSE: {row['RMSE_Clean']:.6f}")
        print(f"   ⚡ 計算時間: {row['Computation_Time']:.4f}秒")
        
        if 'RMSE_Improvement' in row:
            improvement = row['RMSE_Improvement']
            if improvement > 0:
                print(f"   ✅ 改善率: +{improvement:.1f}% (改善)")
            elif improvement < 0:
                print(f"   ❌ 改善率: {improvement:.1f}% (悪化)")
            else:
                print(f"   ➖ 改善率: 0.0% (基準)")
        
        print()
    
    # ランキング
    print("\n🏆 **性能ランキング:**")
    print("-"*60)
    
    # RMSEランキング（全体）
    rmse_ranking = metrics_df.sort_values('RMSE_All')
    print("🎯 **RMSE (全体) ランキング:**")
    for i, (_, row) in enumerate(rmse_ranking.iterrows(), 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}位"
        print(f"   {medal} {row['Method']}: {row['RMSE_All']:.6f}")
    
    # RMSEランキング（クリーン）
    rmse_clean_ranking = metrics_df.sort_values('RMSE_Clean')
    print("\n🧹 **RMSE (クリーンデータ) ランキング:**")
    for i, (_, row) in enumerate(rmse_clean_ranking.iterrows(), 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}位"
        print(f"   {medal} {row['Method']}: {row['RMSE_Clean']:.6f}")
    
    # 速度ランキング
    speed_ranking = metrics_df.sort_values('Computation_Time')
    print("\n⚡ **計算速度ランキング:**")
    for i, (_, row) in enumerate(speed_ranking.iterrows(), 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}位"
        print(f"   {medal} {row['Method']}: {row['Computation_Time']:.4f}秒")
    
    # 理論的特性分析
    print("\n🔬 **理論的特性分析:**")
    print("-"*60)
    
    for _, row in metrics_df.iterrows():
        method_type = row['Type']
        method_name = row['Method']
        
        if method_type == 'baseline':
            print(f"📋 **{method_name}**: 標準実装 - 固定パラメータ、理論基準")
        elif method_type == 'statistical_adaptive':
            print(f"📊 **{method_name}**: 統計適応 - リアルタイム監視、異常値検出")
        elif method_type == 'mathematical_rigorous':
            print(f"🔬 **{method_name}**: 数学厳密 - 相互相関理論、線形行列方程式")
        elif method_type == 'neural_adaptive':
            print(f"🧠 **{method_name}**: ニューラル - CNN ProcessNet、深層学習")
    
    # 総合評価
    print("\n🎖️ **総合評価:**")
    print("-"*60)
    
    best_rmse = rmse_ranking.iloc[0]['Method']
    best_speed = speed_ranking.iloc[0]['Method']
    
    print(f"🏆 **最高精度**: {best_rmse}")
    print(f"⚡ **最高速度**: {best_speed}")
    
    # 改善率で評価
    if 'RMSE_Improvement' in metrics_df.columns:
        improvement_ranking = metrics_df.sort_values('RMSE_Improvement', ascending=False)
        best_improvement = improvement_ranking.iloc[0]
        print(f"📈 **最大改善**: {best_improvement['Method']} ({best_improvement['RMSE_Improvement']:+.1f}%)")
    
    print("\n" + "="*80)


def main():
    """メイン実行関数"""
    print("🧠 **FOUR ADAPTIVE UKF METHODS COMPARISON** 🧠")
    print("="*60)
    print("比較対象:")
    print("1. Standard UKF (基準)")
    print("2. My Implementation AUKF (統計的監視・適応制御)")
    print("3. Academic AUKF (Ge et al. 2019 - 相互相関理論)")
    print("4. Neural AUKF (Levy & Klein 2025 - CNN ProcessNet)")
    print("="*60)
    
    # テストデータ生成
    n_points = 800
    noise_level = 2.0
    
    true_values, noisy_data, outliers, noise_changes = generate_complex_test_data(
        n_points=n_points, 
        noise_level=noise_level
    )
    
    # 4手法比較実行
    methods = run_four_methods_comparison(noisy_data, true_values)
    
    # 有効な結果をフィルタリング
    valid_methods = {k: v for k, v in methods.items() if v is not None}
    
    if len(valid_methods) == 0:
        print("❌ 有効な結果がありません。")
        return
    
    # 性能指標計算
    metrics_df = calculate_performance_metrics(valid_methods, true_values, outliers)
    
    # 詳細分析出力
    print_detailed_analysis(metrics_df, valid_methods)
    
    # 可視化作成
    create_comprehensive_visualization(
        valid_methods,
        true_values,
        noisy_data,
        outliers,
        metrics_df
    )
    
    # 結果CSV保存
    output_csv = project_root / "output" / "adaptive_ukf_four_methods_metrics.csv"
    metrics_df.to_csv(output_csv, index=False)
    print(f"📊 **性能メトリクス保存**: {output_csv}")
    
    print("\n🎉 **四種Adaptive UKF比較完了！**")


if __name__ == "__main__":
    main() 