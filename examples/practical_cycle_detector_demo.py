#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ehlers Practical Cycle Detector Demo
実践的サイクル検出器のデモンストレーション

バックテスト結果で実証された拡張二重微分を含む
4つのコア技術を統合した実用性重視の検出器のデモ
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import logging

# プロジェクトのルートディレクトリを sys.path に追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from indicators.ehlers_practical_cycle_detector import EhlersPracticalCycleDetector
from indicators.ehlers_refined_cycle_detector import EhlersRefinedCycleDetector
from indicators.ehlers_absolute_ultimate_cycle import EhlersAbsoluteUltimateCycle

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_complex_test_data(n: int = 500) -> np.ndarray:
    """
    複雑なテストデータを生成
    
    現実的な相場データに近い特性を持つ複合サイクル信号
    """
    np.random.seed(42)
    t = np.linspace(0, 6*np.pi, n)
    
    # 複数のサイクル成分
    primary_cycle = 2.0 * np.sin(t/5)      # 主要サイクル（20期間）
    secondary_cycle = 1.5 * np.sin(t/8)    # 副次サイクル（32期間）
    short_cycle = 0.8 * np.sin(t/2.5)      # 短期サイクル（10期間）
    
    # トレンド成分
    trend = 0.02 * t
    
    # ノイズ成分（現実的レベル）
    noise = 0.3 * np.random.randn(n)
    
    # 市場ショック（急激な変化）
    shock = np.zeros(n)
    shock[n//3:n//3+5] = 2.0  # 急激な上昇
    shock[2*n//3:2*n//3+3] = -1.5  # 急激な下降
    
    # 複合信号
    signal = primary_cycle + secondary_cycle + short_cycle + trend + noise + shock
    
    # 価格データ風に変換（累積価格）
    price_data = 100 + np.cumsum(signal * 0.1)
    
    return price_data


def create_market_data_format(price_data: np.ndarray) -> pd.DataFrame:
    """
    価格データを市場データ形式に変換
    """
    n = len(price_data)
    
    # 各種価格を生成
    high = price_data + np.random.uniform(0, 0.5, n)
    low = price_data - np.random.uniform(0, 0.5, n)
    open_price = np.roll(price_data, 1)
    open_price[0] = price_data[0]
    
    # DataFrame作成
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': price_data
    })
    
    return df


def calculate_performance_metrics(periods: np.ndarray, true_periods: np.ndarray) -> dict:
    """
    性能メトリクスを計算
    """
    # 安定性（変動係数）
    stability = 1.0 - (np.std(periods) / (np.mean(periods) + 1e-10))
    
    # 真値近似度
    accuracy = 1.0 - (np.mean(np.abs(periods - true_periods)) / np.mean(true_periods))
    
    # 遅延計算（変化に対する応答速度）
    changes = np.diff(true_periods)
    responses = np.diff(periods)
    
    # 方向性一致度
    directional_accuracy = np.mean(np.sign(changes) == np.sign(responses))
    
    return {
        'stability': max(0, min(1, stability)),
        'accuracy': max(0, min(1, accuracy)),
        'directional_accuracy': directional_accuracy,
        'mean_period': np.mean(periods),
        'std_period': np.std(periods)
    }


def run_comprehensive_comparison():
    """
    包括的な比較デモを実行
    """
    logger.info("=== 実践的サイクル検出器 包括的比較デモ ===")
    
    # テストデータ生成
    logger.info("複雑なテストデータを生成中...")
    price_data = generate_complex_test_data(400)
    market_data = create_market_data_format(price_data)
    
    # 理論的サイクル期間（既知）
    true_periods = np.full(len(price_data), 20.0)  # 主要サイクル
    
    # 検出器の初期化
    logger.info("サイクル検出器を初期化中...")
    practical_detector = EhlersPracticalCycleDetector(min_period=6.0, max_period=50.0)
    refined_detector = EhlersRefinedCycleDetector(period_range=(6.0, 50.0))
    ultimate_detector = EhlersAbsoluteUltimateCycle(period_range=(6, 50))
    
    # 検出実行
    logger.info("サイクル検出を実行中...")
    practical_values = practical_detector.calculate(market_data)
    refined_values = refined_detector.calculate(market_data)
    ultimate_values = ultimate_detector.calculate(market_data)
    
    # 実践的検出器の詳細結果を取得
    practical_result = practical_detector.get_practical_result()
    
    # 性能評価
    logger.info("性能評価を実行中...")
    practical_metrics = calculate_performance_metrics(practical_values, true_periods)
    refined_metrics = calculate_performance_metrics(refined_values, true_periods)
    ultimate_metrics = calculate_performance_metrics(ultimate_values, true_periods)
    
    # 結果表示
    print("\n=== 性能比較結果 ===")
    print(f"{'検出器':<20} {'安定性':<10} {'精度':<10} {'方向性':<10} {'平均周期':<10}")
    print("-" * 70)
    print(f"{'実践的検出器':<20} {practical_metrics['stability']:.3f}     {practical_metrics['accuracy']:.3f}     {practical_metrics['directional_accuracy']:.3f}     {practical_metrics['mean_period']:.1f}")
    print(f"{'洗練検出器':<20} {refined_metrics['stability']:.3f}     {refined_metrics['accuracy']:.3f}     {refined_metrics['directional_accuracy']:.3f}     {refined_metrics['mean_period']:.1f}")
    print(f"{'絶対究極検出器':<20} {ultimate_metrics['stability']:.3f}     {ultimate_metrics['accuracy']:.3f}     {ultimate_metrics['directional_accuracy']:.3f}     {ultimate_metrics['mean_period']:.1f}")
    
    # 詳細メトリクス表示
    print("\n=== 詳細メトリクス（実践的検出器） ===")
    print(f"各手法の平均周期:")
    print(f"  ホモダイン判別機: {np.mean(practical_result.homodyne_period):.1f}")
    print(f"  ヒルベルト変換: {np.mean(practical_result.hilbert_period):.1f}")
    print(f"  拡張二重微分: {np.mean(practical_result.dual_diff_period):.1f}")
    print(f"平均信頼度: {np.mean(practical_result.confidence):.3f}")
    print(f"市場フェーズ分布:")
    print(f"  安定期: {np.mean(practical_result.market_phase > 0.7):.1%}")
    print(f"  移行期: {np.mean((practical_result.market_phase >= 0.3) & (practical_result.market_phase <= 0.7)):.1%}")
    print(f"  不安定期: {np.mean(practical_result.market_phase < 0.3):.1%}")
    
    # 可視化
    create_comparison_visualization(
        market_data, practical_values, refined_values, ultimate_values, practical_result, true_periods
    )
    
    return practical_values, refined_values, ultimate_values, practical_result


def create_comparison_visualization(market_data, practical_values, refined_values, ultimate_values, practical_result, true_periods):
    """
    比較可視化を作成
    """
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    # 価格データ
    axes[0].plot(market_data['close'], label='価格データ', color='black', linewidth=1)
    axes[0].set_title('価格データ', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('価格')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # サイクル期間比較
    axes[1].plot(practical_values, label='実践的検出器', color='red', linewidth=2)
    axes[1].plot(refined_values, label='洗練検出器', color='blue', linewidth=2)
    axes[1].plot(ultimate_values, label='絶対究極検出器', color='green', linewidth=2)
    axes[1].plot(true_periods, label='理論値', color='orange', linestyle='--', linewidth=1)
    axes[1].set_title('サイクル期間比較', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('サイクル期間')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 各手法の詳細（実践的検出器）
    axes[2].plot(practical_result.homodyne_period, label='ホモダイン判別機', color='purple', alpha=0.7)
    axes[2].plot(practical_result.hilbert_period, label='ヒルベルト変換', color='brown', alpha=0.7)
    axes[2].plot(practical_result.dual_diff_period, label='拡張二重微分', color='red', linewidth=2)
    axes[2].set_title('実践的検出器の各手法詳細', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('サイクル期間')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # 信頼度と市場フェーズ
    ax3_twin = axes[3].twinx()
    axes[3].plot(practical_result.confidence, label='信頼度', color='green', linewidth=2)
    ax3_twin.plot(practical_result.market_phase, label='市場フェーズ', color='orange', linewidth=2)
    axes[3].set_title('信頼度と市場フェーズ', fontsize=14, fontweight='bold')
    axes[3].set_ylabel('信頼度', color='green')
    ax3_twin.set_ylabel('市場フェーズ', color='orange')
    axes[3].set_xlabel('時間')
    axes[3].legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    output_dir = Path("examples/output")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "practical_cycle_detector_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n可視化結果を保存しました: {output_dir / 'practical_cycle_detector_comparison.png'}")


def demonstrate_real_world_usage():
    """
    実世界での使用例をデモ
    """
    logger.info("=== 実世界使用例デモ ===")
    
    # 仮想的な市場データ
    np.random.seed(123)
    n = 300
    
    # より現実的な価格データ
    returns = np.random.normal(0, 0.01, n)
    # トレンド
    trend = np.linspace(0, 0.2, n)
    returns += trend / 1000
    
    # サイクル的変動
    t = np.linspace(0, 4*np.pi, n)
    cycle_component = 0.005 * np.sin(t/3) + 0.003 * np.sin(t/7)
    returns += cycle_component
    
    # 累積価格
    price = 100 * np.exp(np.cumsum(returns))
    
    # 市場データ形式
    market_data = create_market_data_format(price)
    
    # 検出器実行
    detector = EhlersPracticalCycleDetector(min_period=8.0, max_period=40.0)
    values = detector.calculate(market_data)
    result = detector.get_practical_result()
    
    # 結果分析
    print("\n=== 実世界使用例結果 ===")
    print(f"平均サイクル期間: {np.mean(values):.1f} 期間")
    print(f"サイクル期間の標準偏差: {np.std(values):.1f}")
    print(f"平均信頼度: {np.mean(result.confidence):.3f}")
    print(f"最高信頼度: {np.max(result.confidence):.3f}")
    print(f"最低信頼度: {np.min(result.confidence):.3f}")
    
    # 市場状況分析
    stable_periods = np.sum(result.market_phase > 0.7)
    transition_periods = np.sum((result.market_phase >= 0.3) & (result.market_phase <= 0.7))
    unstable_periods = np.sum(result.market_phase < 0.3)
    
    print(f"\n市場状況分析:")
    print(f"  安定期: {stable_periods} 期間 ({stable_periods/len(values):.1%})")
    print(f"  移行期: {transition_periods} 期間 ({transition_periods/len(values):.1%})")
    print(f"  不安定期: {unstable_periods} 期間 ({unstable_periods/len(values):.1%})")
    
    # 実践的使用方法の提案
    print(f"\n=== 実践的使用方法 ===")
    current_cycle = values[-1]
    current_confidence = result.confidence[-1]
    current_phase = result.market_phase[-1]
    
    print(f"現在のサイクル期間: {current_cycle:.1f}")
    print(f"現在の信頼度: {current_confidence:.3f}")
    
    if current_phase > 0.7:
        market_status = "安定期"
        recommendation = "サイクル期間を基準とした戦略が有効"
    elif current_phase > 0.3:
        market_status = "移行期"
        recommendation = "慎重な観察が必要、リスク管理重視"
    else:
        market_status = "不安定期"
        recommendation = "サイクル分析は参考程度、他の指標も併用"
    
    print(f"市場状況: {market_status}")
    print(f"推奨: {recommendation}")
    
    # 可視化
    create_real_world_visualization(market_data, result)


def create_real_world_visualization(market_data, result):
    """
    実世界使用例の可視化
    """
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # 価格データ
    axes[0].plot(market_data['close'], label='価格', color='black', linewidth=1)
    axes[0].set_title('実世界使用例 - 価格データ', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('価格')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # サイクル期間と信頼度
    ax1_twin = axes[1].twinx()
    axes[1].plot(result.values, label='サイクル期間', color='blue', linewidth=2)
    ax1_twin.plot(result.confidence, label='信頼度', color='red', linewidth=2)
    axes[1].set_title('サイクル期間と信頼度', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('サイクル期間', color='blue')
    ax1_twin.set_ylabel('信頼度', color='red')
    axes[1].legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    # 市場フェーズ
    colors = ['red' if x < 0.3 else 'orange' if x < 0.7 else 'green' for x in result.market_phase]
    axes[2].scatter(range(len(result.market_phase)), result.market_phase, c=colors, alpha=0.7, s=30)
    axes[2].plot(result.market_phase, color='black', alpha=0.5, linewidth=1)
    axes[2].axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='不安定期閾値')
    axes[2].axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='安定期閾値')
    axes[2].set_title('市場フェーズ分析', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('市場フェーズ')
    axes[2].set_xlabel('時間')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    output_dir = Path("examples/output")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "practical_cycle_detector_real_world.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n可視化結果を保存しました: {output_dir / 'practical_cycle_detector_real_world.png'}")


def main():
    """
    メイン関数
    """
    print("Ehlers Practical Cycle Detector Demo")
    print("=" * 50)
    
    try:
        # 包括的比較デモ
        practical_values, refined_values, ultimate_values, practical_result = run_comprehensive_comparison()
        
        # 実世界使用例デモ
        demonstrate_real_world_usage()
        
        print("\n" + "=" * 50)
        print("デモが完了しました。")
        print("生成された可視化ファイルを確認してください。")
        
    except Exception as e:
        logger.error(f"デモ実行中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 