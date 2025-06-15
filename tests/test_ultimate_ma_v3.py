#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from indicators.ultimate_ma_v3 import UltimateMAV3
from indicators.ultimate_ma import UltimateMA
import time

def generate_test_data(n_points=1000):
    """テスト用のOHLCデータを生成"""
    np.random.seed(42)
    
    # トレンドとノイズを含む価格データ
    trend = np.cumsum(np.random.randn(n_points) * 0.001)
    noise = np.random.randn(n_points) * 0.01
    base_price = 100 + trend + noise
    
    # OHLC作成
    high = base_price + np.abs(np.random.randn(n_points) * 0.5)
    low = base_price - np.abs(np.random.randn(n_points) * 0.5)
    open_price = base_price + np.random.randn(n_points) * 0.2
    close_price = base_price + np.random.randn(n_points) * 0.2
    
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close_price
    })
    
    return df

def test_ultimate_ma_v3():
    """UltimateMA V3のテスト"""
    print("🚀 UltimateMA V3 テスト開始...")
    
    # テストデータ生成
    data = generate_test_data(500)
    print(f"📊 テストデータ生成完了: {len(data)}個のデータポイント")
    
    # UltimateMA V3初期化
    uma_v3 = UltimateMAV3(
        super_smooth_period=10,
        zero_lag_period=21,
        realtime_window=89,
        quantum_window=21,
        fractal_window=21,
        entropy_window=21,
        src_type='hlc3',
        slope_index=1,
        base_threshold=0.003,
        min_confidence=0.3
    )
    
    # 計算実行
    start_time = time.time()
    result = uma_v3.calculate(data)
    calc_time = time.time() - start_time
    
    print(f"⚡ 計算時間: {calc_time:.3f}秒")
    print(f"🎯 現在のトレンド: {result.current_trend}")
    print(f"📈 トレンド値: {result.current_trend_value}")
    print(f"🔥 信頼度: {result.current_confidence:.3f}")
    
    # 統計情報
    up_signals = np.sum(result.trend_signals == 1)
    down_signals = np.sum(result.trend_signals == -1)
    range_signals = np.sum(result.trend_signals == 0)
    total_signals = len(result.trend_signals)
    
    print(f"\n📊 シグナル統計:")
    print(f"  📈 上昇: {up_signals} ({up_signals/total_signals*100:.1f}%)")
    print(f"  📉 下降: {down_signals} ({down_signals/total_signals*100:.1f}%)")
    print(f"  ➡️ レンジ: {range_signals} ({range_signals/total_signals*100:.1f}%)")
    
    # 信頼度統計
    avg_confidence = np.mean(result.trend_confidence[result.trend_confidence > 0])
    max_confidence = np.max(result.trend_confidence)
    
    print(f"\n🔥 信頼度統計:")
    print(f"  平均信頼度: {avg_confidence:.3f}")
    print(f"  最大信頼度: {max_confidence:.3f}")
    
    # 量子分析結果
    quantum_analysis = uma_v3.get_quantum_analysis()
    
    print(f"\n🌌 量子分析統計:")
    print(f"  量子状態範囲: [{np.min(quantum_analysis['quantum_state']):.3f}, {np.max(quantum_analysis['quantum_state']):.3f}]")
    print(f"  MTF合意度範囲: [{np.min(quantum_analysis['multi_timeframe_consensus']):.3f}, {np.max(quantum_analysis['multi_timeframe_consensus']):.3f}]")
    print(f"  フラクタル次元範囲: [{np.min(quantum_analysis['fractal_dimension']):.3f}, {np.max(quantum_analysis['fractal_dimension']):.3f}]")
    print(f"  エントロピー範囲: [{np.min(quantum_analysis['entropy_level']):.3f}, {np.max(quantum_analysis['entropy_level']):.3f}]")
    
    return result, data

def compare_versions():
    """V2とV3の比較テスト"""
    print("\n🆚 UltimateMA V2 vs V3 比較テスト...")
    
    data = generate_test_data(300)
    
    # V2（従来版）
    uma_v2 = UltimateMA(
        super_smooth_period=10,
        zero_lag_period=21,
        realtime_window=89,
        src_type='hlc3',
        slope_index=1,
        range_threshold=0.005
    )
    
    # V3（新版）
    uma_v3 = UltimateMAV3(
        super_smooth_period=10,
        zero_lag_period=21,
        realtime_window=89,
        quantum_window=21,
        src_type='hlc3',
        slope_index=1,
        base_threshold=0.003,
        min_confidence=0.3
    )
    
    # 計算時間比較
    start_time = time.time()
    result_v2 = uma_v2.calculate(data)
    time_v2 = time.time() - start_time
    
    start_time = time.time()
    result_v3 = uma_v3.calculate(data)
    time_v3 = time.time() - start_time
    
    print(f"⚡ 計算時間比較:")
    print(f"  V2: {time_v2:.3f}秒")
    print(f"  V3: {time_v3:.3f}秒")
    print(f"  比率: {time_v3/time_v2:.2f}x")
    
    # シグナル比較
    v2_up = np.sum(result_v2.trend_signals == 1)
    v2_down = np.sum(result_v2.trend_signals == -1)
    v2_range = np.sum(result_v2.trend_signals == 0)
    
    v3_up = np.sum(result_v3.trend_signals == 1)
    v3_down = np.sum(result_v3.trend_signals == -1)
    v3_range = np.sum(result_v3.trend_signals == 0)
    
    total = len(data)
    
    print(f"\n📊 シグナル分布比較:")
    print(f"  📈 上昇: V2={v2_up/total*100:.1f}% vs V3={v3_up/total*100:.1f}%")
    print(f"  📉 下降: V2={v2_down/total*100:.1f}% vs V3={v3_down/total*100:.1f}%")
    print(f"  ➡️ レンジ: V2={v2_range/total*100:.1f}% vs V3={v3_range/total*100:.1f}%")
    
    # 現在の判定比較
    print(f"\n🎯 現在の判定比較:")
    print(f"  V2: {result_v2.current_trend}")
    print(f"  V3: {result_v3.current_trend} (信頼度: {result_v3.current_confidence:.3f})")
    
    return result_v2, result_v3, data

def visualize_results(result_v3, data):
    """結果の可視化"""
    print("\n📈 結果可視化中...")
    
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    fig.suptitle('🚀 UltimateMA V3 - AI分析結果', fontsize=16, fontweight='bold')
    
    # 価格とMA
    axes[0].plot(data['close'], label='価格', alpha=0.7, color='gray')
    axes[0].plot(result_v3.values, label='Ultimate MA V3', color='red', linewidth=2)
    axes[0].set_title('💰 価格 vs Ultimate MA V3')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # トレンドシグナル
    colors = ['blue', 'red', 'green']
    labels = ['レンジ', '上昇', '下降']
    for i in range(3):
        mask = result_v3.trend_signals == (i-1)
        if np.any(mask):
            axes[1].scatter(np.where(mask)[0], result_v3.values[mask], 
                          c=colors[i], label=labels[i], alpha=0.7, s=20)
    axes[1].plot(result_v3.values, color='black', alpha=0.3)
    axes[1].set_title('🎯 トレンドシグナル')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 信頼度
    axes[2].fill_between(range(len(result_v3.trend_confidence)), 
                        result_v3.trend_confidence, alpha=0.6, color='orange')
    axes[2].set_title('🔥 トレンド信頼度')
    axes[2].set_ylabel('信頼度')
    axes[2].grid(True, alpha=0.3)
    
    # 量子分析
    axes[3].plot(result_v3.quantum_state, label='量子状態', color='purple')
    axes[3].plot(result_v3.multi_timeframe_consensus, label='MTF合意度', color='blue')
    axes[3].plot(result_v3.fractal_dimension - 1, label='フラクタル次元-1', color='green')
    axes[3].plot(result_v3.entropy_level, label='エントロピー', color='red')
    axes[3].set_title('🌌 量子分析指標')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # ファイル保存
    filename = "tests/ultimate_ma_v3_analysis.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ チャート保存完了: {filename}")
    
    plt.show()
    plt.close()

def main():
    """メイン実行関数"""
    print("🚀 UltimateMA V3 テストスイート")
    print("="*50)
    
    # V3単体テスト
    result_v3, data = test_ultimate_ma_v3()
    
    # バージョン比較テスト
    result_v2, result_v3_comp, data_comp = compare_versions()
    
    # 可視化
    visualize_results(result_v3, data)
    
    print("\n✅ 全テスト完了!")
    print("📊 生成されたチャートファイルをご確認ください。")

if __name__ == "__main__":
    main() 