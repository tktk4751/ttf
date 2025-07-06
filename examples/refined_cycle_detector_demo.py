#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

# 必要なモジュールのインポート
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicators.ehlers_refined_cycle_detector import EhlersRefinedCycleDetector
from indicators.ehlers_absolute_ultimate_cycle import EhlersAbsoluteUltimateCycle
from indicators.ehlers_unified_dc import EhlersUnifiedDC


def create_test_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    テスト用のサンプルデータを生成する
    
    Args:
        n_samples: サンプル数
    
    Returns:
        OHLC価格データ
    """
    np.random.seed(42)
    
    # 基本トレンド
    base_trend = np.linspace(100, 120, n_samples)
    
    # 複数のサイクル成分
    cycle_20 = 8 * np.sin(2 * np.pi * np.arange(n_samples) / 20)
    cycle_35 = 5 * np.sin(2 * np.pi * np.arange(n_samples) / 35)
    cycle_50 = 3 * np.sin(2 * np.pi * np.arange(n_samples) / 50)
    
    # ノイズ
    noise = np.random.normal(0, 1, n_samples)
    
    # 合成価格
    close_price = base_trend + cycle_20 + cycle_35 + cycle_50 + noise
    
    # OHLC生成
    data = pd.DataFrame({
        'open': close_price + np.random.normal(0, 0.5, n_samples),
        'high': close_price + np.abs(np.random.normal(2, 1, n_samples)),
        'low': close_price - np.abs(np.random.normal(2, 1, n_samples)),
        'close': close_price,
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    return data


def compare_cycle_detectors(data: pd.DataFrame) -> Dict[str, Any]:
    """
    複数のサイクル検出器を比較する
    
    Args:
        data: 価格データ
    
    Returns:
        比較結果の辞書
    """
    print("🚀 サイクル検出器の比較分析を開始します...")
    
    # 1. 洗練されたサイクル検出器（新設計）
    print("\n1. 洗練されたサイクル検出器（Refined Cycle Detector）")
    refined_detector = EhlersRefinedCycleDetector(
        cycle_part=0.5,
        max_output=50,
        min_output=5,
        period_range=(6.0, 50.0),
        alpha=0.07,
        src_type='hlc3',
        ultimate_smoother_period=20.0,
        use_ultimate_smoother=True
    )
    
    refined_cycles = refined_detector.calculate(data)
    refined_confidence = refined_detector.confidence_scores
    refined_summary = refined_detector.get_analysis_summary()
    
    print(f"   計算完了: {len(refined_cycles)} 点")
    print(f"   平均周期: {np.mean(refined_cycles):.2f}")
    print(f"   平均信頼度: {np.mean(refined_confidence):.3f}")
    
    # 2. 絶対的究極サイクル検出器（比較用）
    print("\n2. 絶対的究極サイクル検出器（Absolute Ultimate Cycle）")
    absolute_detector = EhlersAbsoluteUltimateCycle(
        cycle_part=0.5,
        max_output=50,
        min_output=5,
        period_range=(6, 50),
        src_type='hlc3'
    )
    
    absolute_cycles = absolute_detector.calculate(data)
    absolute_confidence = absolute_detector.confidence_scores
    
    print(f"   計算完了: {len(absolute_cycles)} 点")
    print(f"   平均周期: {np.mean(absolute_cycles):.2f}")
    print(f"   平均信頼度: {np.mean(absolute_confidence):.3f}")
    
    # 3. 統合DC検出器（ホモダイン）
    print("\n3. 統合DC検出器（Homodyne Discriminator）")
    unified_detector = EhlersUnifiedDC(
        detector_type='hody',
        cycle_part=0.5,
        max_output=50,
        min_output=5,
        src_type='hlc3'
    )
    
    unified_cycles = unified_detector.calculate(data)
    
    print(f"   計算完了: {len(unified_cycles)} 点")
    print(f"   平均周期: {np.mean(unified_cycles):.2f}")
    
    # 結果の比較
    results = {
        'refined': {
            'cycles': refined_cycles,
            'confidence': refined_confidence,
            'summary': refined_summary,
            'name': 'Refined Cycle Detector',
            'color': '#2E86AB'  # 青
        },
        'absolute': {
            'cycles': absolute_cycles,
            'confidence': absolute_confidence,
            'name': 'Absolute Ultimate Cycle',
            'color': '#A23B72'  # 赤紫
        },
        'unified': {
            'cycles': unified_cycles,
            'confidence': None,
            'name': 'Unified DC (Homodyne)',
            'color': '#F18F01'  # オレンジ
        },
        'data': data
    }
    
    return results


def calculate_performance_metrics(results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    各検出器の性能メトリクスを計算する
    
    Args:
        results: 比較結果
    
    Returns:
        性能メトリクス
    """
    print("\n📊 性能メトリクスの計算...")
    
    metrics = {}
    
    # 真のサイクル（理論値）
    true_cycles = [20, 35, 50]  # 生成時に使用したサイクル
    
    for detector_name, detector_data in results.items():
        if detector_name == 'data':
            continue
            
        cycles = detector_data['cycles']
        confidence = detector_data['confidence']
        
        # 基本統計
        mean_cycle = np.mean(cycles)
        std_cycle = np.std(cycles)
        
        # 安定性（変動係数）
        stability = 1.0 - (std_cycle / mean_cycle) if mean_cycle > 0 else 0.0
        
        # 真値との近似度
        proximity_scores = []
        for true_cycle in true_cycles:
            proximity = 1.0 - np.abs(mean_cycle - true_cycle) / true_cycle
            proximity_scores.append(max(0, proximity))
        
        best_proximity = max(proximity_scores)
        
        # 信頼度平均
        avg_confidence = np.mean(confidence) if confidence is not None else 0.5
        
        # 計算速度（サンプル数で正規化）
        computation_speed = len(cycles) / 1000.0  # 相対的な速度指標
        
        # 総合スコア
        overall_score = (stability * 0.3 + 
                        best_proximity * 0.3 + 
                        avg_confidence * 0.2 + 
                        computation_speed * 0.2)
        
        metrics[detector_name] = {
            'mean_cycle': mean_cycle,
            'stability': stability,
            'proximity': best_proximity,
            'confidence': avg_confidence,
            'computation_speed': computation_speed,
            'overall_score': overall_score
        }
        
        print(f"\n{detector_data['name']}:")
        print(f"  平均周期: {mean_cycle:.2f}")
        print(f"  安定性: {stability:.3f}")
        print(f"  真値近似度: {best_proximity:.3f}")
        print(f"  信頼度: {avg_confidence:.3f}")
        print(f"  総合スコア: {overall_score:.3f}")
    
    return metrics


def visualize_results(results: Dict[str, Any], metrics: Dict[str, Dict[str, float]]):
    """
    結果を可視化する
    
    Args:
        results: 比較結果
        metrics: 性能メトリクス
    """
    print("\n📈 結果の可視化...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('🎯 洗練されたサイクル検出器 - 性能比較分析', fontsize=16, fontweight='bold')
    
    # 1. 価格データとサイクル検出結果
    ax1 = axes[0, 0]
    data = results['data']
    
    ax1.plot(data['close'], label='Close Price', color='black', alpha=0.7, linewidth=1)
    
    # 各検出器の結果をプロット
    for detector_name, detector_data in results.items():
        if detector_name == 'data':
            continue
        
        cycles = detector_data['cycles']
        # サイクル値を価格範囲にスケール
        scaled_cycles = ((cycles - np.min(cycles)) / (np.max(cycles) - np.min(cycles))) * 20 + np.min(data['close'])
        
        ax1.plot(scaled_cycles, label=detector_data['name'], 
                color=detector_data['color'], alpha=0.8, linewidth=2)
    
    ax1.set_title('価格データとサイクル検出結果')
    ax1.set_xlabel('時間')
    ax1.set_ylabel('価格 / スケール済みサイクル')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. サイクル周期の時系列
    ax2 = axes[0, 1]
    
    for detector_name, detector_data in results.items():
        if detector_name == 'data':
            continue
        
        cycles = detector_data['cycles']
        ax2.plot(cycles, label=detector_data['name'], 
                color=detector_data['color'], alpha=0.8, linewidth=2)
    
    # 真のサイクル周期をプロット
    ax2.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='True Cycle 20')
    ax2.axhline(y=35, color='red', linestyle='--', alpha=0.5, label='True Cycle 35')
    ax2.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='True Cycle 50')
    
    ax2.set_title('サイクル周期の時系列')
    ax2.set_xlabel('時間')
    ax2.set_ylabel('サイクル周期')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 信頼度スコア
    ax3 = axes[1, 0]
    
    for detector_name, detector_data in results.items():
        if detector_name == 'data' or detector_data['confidence'] is None:
            continue
        
        confidence = detector_data['confidence']
        ax3.plot(confidence, label=detector_data['name'], 
                color=detector_data['color'], alpha=0.8, linewidth=2)
    
    ax3.set_title('信頼度スコアの時系列')
    ax3.set_xlabel('時間')
    ax3.set_ylabel('信頼度')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 性能メトリクス比較
    ax4 = axes[1, 1]
    
    metric_names = ['stability', 'proximity', 'confidence', 'overall_score']
    metric_labels = ['安定性', '真値近似度', '信頼度', '総合スコア']
    
    x = np.arange(len(metric_labels))
    width = 0.25
    
    for i, (detector_name, detector_data) in enumerate(results.items()):
        if detector_name == 'data':
            continue
        
        values = [metrics[detector_name][metric] for metric in metric_names]
        ax4.bar(x + i * width, values, width, label=detector_data['name'], 
               color=detector_data['color'], alpha=0.8)
    
    ax4.set_title('性能メトリクス比較')
    ax4.set_xlabel('メトリクス')
    ax4.set_ylabel('スコア')
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(metric_labels)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/refined_cycle_detector_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 可視化が完了しました。'output/refined_cycle_detector_comparison.png'に保存されました。")


def main():
    """
    メイン実行関数
    """
    print("🎯 洗練されたサイクル検出器 - デモンストレーション")
    print("=" * 60)
    
    # 出力ディレクトリの作成
    os.makedirs('output', exist_ok=True)
    
    # 1. テストデータの生成
    print("\n📊 テストデータの生成...")
    data = create_test_data(n_samples=500)
    print(f"   生成完了: {len(data)} 点のOHLCデータ")
    print(f"   価格範囲: {data['close'].min():.2f} - {data['close'].max():.2f}")
    
    # 2. サイクル検出器の比較
    results = compare_cycle_detectors(data)
    
    # 3. 性能メトリクスの計算
    metrics = calculate_performance_metrics(results)
    
    # 4. 結果の可視化
    visualize_results(results, metrics)
    
    # 5. 詳細分析の表示
    print("\n🏆 最終結果分析:")
    print("=" * 40)
    
    # 最高スコアの検出器を特定
    best_detector = max(metrics.items(), key=lambda x: x[1]['overall_score'])
    print(f"🥇 最高性能: {results[best_detector[0]]['name']}")
    print(f"   総合スコア: {best_detector[1]['overall_score']:.3f}")
    
    # 洗練されたサイクル検出器の詳細情報
    if 'refined' in results:
        refined_summary = results['refined']['summary']
        print(f"\n📋 洗練されたサイクル検出器の詳細:")
        print(f"   アルゴリズム: {refined_summary['algorithm']}")
        print(f"   コア技術: {', '.join(refined_summary['core_technologies'])}")
        print(f"   特性: 遅延={refined_summary['characteristics']['latency']}, "
              f"精度={refined_summary['characteristics']['accuracy']}")
        print(f"   計算効率: {refined_summary['characteristics']['computation']}")
    
    print("\n✨ デモンストレーションが完了しました！")


if __name__ == "__main__":
    main() 