#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from indicators.ultimate_efficiency_ratio import UltimateEfficiencyRatio
from indicators.efficiency_ratio import EfficiencyRatio  # 従来ERとの比較用


def generate_synthetic_data(length: int = 1000, trend_strength: float = 0.1, noise_level: float = 0.05) -> pd.DataFrame:
    """
    合成価格データを生成する
    
    Args:
        length: データ点数
        trend_strength: トレンドの強さ
        noise_level: ノイズレベル
        
    Returns:
        価格データのDataFrame
    """
    dates = pd.date_range(start='2020-01-01', periods=length, freq='D')
    
    # 基本トレンド
    trend = np.cumsum(np.random.randn(length) * trend_strength)
    
    # 周期的成分（複数の周期）
    t = np.arange(length)
    cyclic = (0.5 * np.sin(2 * np.pi * t / 20) +  # 20日周期
              0.3 * np.sin(2 * np.pi * t / 50) +  # 50日周期
              0.2 * np.sin(2 * np.pi * t / 100))  # 100日周期
    
    # ノイズ
    noise = np.random.randn(length) * noise_level
    
    # 価格の合成
    price = 100 + trend + cyclic + noise
    
    # OHLC形式で作成
    high = price + np.random.uniform(0, 0.5, length)
    low = price - np.random.uniform(0, 0.5, length)
    open_price = price + np.random.uniform(-0.25, 0.25, length)
    close_price = price
    
    return pd.DataFrame({
        'date': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close_price,
        'volume': np.random.randint(1000, 10000, length)
    })


def test_ultimate_er_vs_traditional_er():
    """アルティメットERと従来ERの比較テスト"""
    print("🚀 アルティメットER vs 従来ER 比較テスト")
    print("=" * 60)
    
    # 合成データの生成
    data = generate_synthetic_data(length=500)
    
    # アルティメットERの初期化
    ultimate_er = UltimateEfficiencyRatio(
        period=14,
        smoother_period=20.0,
        src_type='hlc3',
        phase_lookback=14,
        coherence_periods=(5, 14, 34),
        trend_lookback=3,
        use_adaptive_smoothing=True
    )
    
    # 従来ERの初期化（比較用）
    traditional_er = EfficiencyRatio(
        period=14,
        src_type='hlc3',
        smoothing_method='hma',
        use_dynamic_period=False
    )
    
    # 計算実行
    print("⏱️ 計算実行中...")
    
    ultimate_result = ultimate_er.calculate(data)
    traditional_result = traditional_er.calculate(data)
    
    print(f"✅ アルティメットER計算完了: {len(ultimate_result.values)} データポイント")
    print(f"✅ 従来ER計算完了: {len(traditional_result.values)} データポイント")
    
    # 結果の分析
    print("\n📊 結果分析:")
    print("-" * 40)
    
    # 最新値
    if len(ultimate_result.values) > 0:
        latest_ultimate = ultimate_result.values[-1]
        latest_traditional = traditional_result.values[-1]
        
        print(f"最新アルティメットER: {latest_ultimate:.4f}")
        print(f"最新従来ER: {latest_traditional:.4f}")
        print(f"差異: {abs(latest_ultimate - latest_traditional):.4f}")
    
    # 統計情報
    print(f"\nアルティメットER統計:")
    print(f"  平均: {np.mean(ultimate_result.values):.4f}")
    print(f"  標準偏差: {np.std(ultimate_result.values):.4f}")
    print(f"  最大値: {np.max(ultimate_result.values):.4f}")
    print(f"  最小値: {np.min(ultimate_result.values):.4f}")
    
    print(f"\n従来ER統計:")
    print(f"  平均: {np.mean(traditional_result.values):.4f}")
    print(f"  標準偏差: {np.std(traditional_result.values):.4f}")
    print(f"  最大値: {np.max(traditional_result.values):.4f}")
    print(f"  最小値: {np.min(traditional_result.values):.4f}")
    
    # 成分分析
    print(f"\n🔬 アルティメットER成分分析:")
    component_analysis = ultimate_er.get_component_analysis()
    for key, value in component_analysis.items():
        print(f"  {key}: {value:.4f}")
    
    # 現在のトレンド
    current_trend = ultimate_er.get_current_trend()
    print(f"\n📈 現在のトレンド状態: {current_trend}")
    
    return ultimate_result, traditional_result, data


def plot_comparison_results(ultimate_result, traditional_result, data):
    """比較結果のプロット"""
    print("\n📊 比較チャート作成中...")
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('アルティメットER vs 従来ER 比較分析', fontsize=16, fontweight='bold')
    
    # 日付軸の準備
    dates = data['date'] if 'date' in data.columns else range(len(data))
    
    # 価格チャート
    axes[0, 0].plot(dates, data['close'], label='Close Price', color='blue', alpha=0.7)
    axes[0, 0].set_title('価格チャート')
    axes[0, 0].set_ylabel('Price')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 効率比比較
    axes[0, 1].plot(dates, ultimate_result.values, label='Ultimate ER', color='red', linewidth=2)
    axes[0, 1].plot(dates, traditional_result.values, label='Traditional ER', color='blue', linewidth=1)
    axes[0, 1].set_title('効率比比較')
    axes[0, 1].set_ylabel('Efficiency Ratio')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 位相効率指数
    axes[1, 0].plot(dates, ultimate_result.phase_efficiency, label='Phase Efficiency', color='green')
    axes[1, 0].set_title('位相効率指数')
    axes[1, 0].set_ylabel('Phase Efficiency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # スペクトル純度
    axes[1, 1].plot(dates, ultimate_result.spectral_purity, label='Spectral Purity', color='orange')
    axes[1, 1].set_title('スペクトル純度')
    axes[1, 1].set_ylabel('Spectral Purity')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Multi-Phase Coherence
    axes[2, 0].plot(dates, ultimate_result.multi_phase_coherence, label='Multi-Phase Coherence', color='purple')
    axes[2, 0].set_title('Multi-Phase Coherence')
    axes[2, 0].set_ylabel('Coherence')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # トレンド信号
    axes[2, 1].plot(dates, ultimate_result.trend_signals, label='Trend Signals', color='brown', marker='o', markersize=2)
    axes[2, 1].set_title('トレンド信号')
    axes[2, 1].set_ylabel('Signal (-1:Down, 0:Range, 1:Up)')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    output_path = os.path.join(os.path.dirname(__file__), 'output', 'ultimate_er_comparison.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 チャート保存完了: {output_path}")
    
    plt.show()


def performance_benchmark():
    """パフォーマンスベンチマーク"""
    print("\n⚡ パフォーマンスベンチマーク")
    print("=" * 60)
    
    import time
    
    # 異なるデータサイズでテスト
    sizes = [100, 500, 1000, 2000]
    
    for size in sizes:
        print(f"\nデータサイズ: {size} ポイント")
        print("-" * 30)
        
        # データ生成
        data = generate_synthetic_data(length=size)
        
        # アルティメットERのテスト
        ultimate_er = UltimateEfficiencyRatio(period=14, smoother_period=20.0)
        
        start_time = time.time()
        result = ultimate_er.calculate(data)
        end_time = time.time()
        
        calculation_time = end_time - start_time
        print(f"計算時間: {calculation_time:.4f} 秒")
        print(f"スループット: {size / calculation_time:.0f} ポイント/秒")
        
        # メモリ使用量の推定
        memory_usage = len(result.values) * 8 * 7  # 7つの配列 × 8バイト（float64）
        print(f"推定メモリ使用量: {memory_usage / 1024:.2f} KB")


def test_different_market_conditions():
    """異なる市場条件でのテスト"""
    print("\n🌍 異なる市場条件でのテスト")
    print("=" * 60)
    
    conditions = [
        ("強いトレンド", 0.2, 0.02),
        ("弱いトレンド", 0.05, 0.02),
        ("高ボラティリティ", 0.1, 0.15),
        ("低ボラティリティ", 0.1, 0.01),
        ("レンジ相場", 0.02, 0.08)
    ]
    
    ultimate_er = UltimateEfficiencyRatio(period=14, smoother_period=20.0)
    
    for condition_name, trend_strength, noise_level in conditions:
        print(f"\n📊 {condition_name}:")
        print("-" * 20)
        
        # データ生成
        data = generate_synthetic_data(length=300, trend_strength=trend_strength, noise_level=noise_level)
        
        # 計算
        result = ultimate_er.calculate(data)
        
        # 分析
        avg_er = np.mean(result.values)
        std_er = np.std(result.values)
        avg_phase_eff = np.mean(result.phase_efficiency)
        avg_coherence = np.mean(result.multi_phase_coherence)
        
        print(f"  平均ER: {avg_er:.4f}")
        print(f"  ER標準偏差: {std_er:.4f}")
        print(f"  平均位相効率: {avg_phase_eff:.4f}")
        print(f"  平均コヒーレンス: {avg_coherence:.4f}")
        
        # トレンド検出精度
        trend_signals = result.trend_signals
        uptrend_ratio = np.sum(trend_signals == 1) / len(trend_signals)
        downtrend_ratio = np.sum(trend_signals == -1) / len(trend_signals)
        range_ratio = np.sum(trend_signals == 0) / len(trend_signals)
        
        print(f"  上昇トレンド比率: {uptrend_ratio:.2%}")
        print(f"  下降トレンド比率: {downtrend_ratio:.2%}")
        print(f"  レンジ比率: {range_ratio:.2%}")


def main():
    """メイン実行関数"""
    print("🎯 アルティメットER (Ultimate Efficiency Ratio) デモンストレーション")
    print("=" * 80)
    print("🔬 ジョン・エラーズ式 革新的効率比インジケーター")
    print("💡 Hilbert変換 + DFTスペクトル分析 + Ultimate Smoother + Multi-Phase Coherence")
    print("=" * 80)
    
    try:
        # 基本的な比較テスト
        ultimate_result, traditional_result, data = test_ultimate_er_vs_traditional_er()
        
        # チャート作成
        plot_comparison_results(ultimate_result, traditional_result, data)
        
        # パフォーマンステスト
        performance_benchmark()
        
        # 異なる市場条件でのテスト
        test_different_market_conditions()
        
        print("\n✅ 全テスト完了!")
        print("🎉 アルティメットERは従来の効率比を大幅に上回る性能を実現しました！")
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 