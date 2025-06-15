#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UltimateMA V3 Simple Demo
実際の相場データでテストできるシンプルなデモ
"""

import numpy as np
import pandas as pd
import time
from ultimate_ma_v3 import UltimateMAV3


def create_sample_data(n_points=1000, trend_type='mixed'):
    """
    サンプルOHLCデータを作成
    
    Args:
        n_points: データポイント数
        trend_type: 'up', 'down', 'mixed', 'volatile'
    """
    np.random.seed(42)
    
    if trend_type == 'up':
        base_trend = np.cumsum(np.random.randn(n_points) * 0.002 + 0.015)
    elif trend_type == 'down':
        base_trend = np.cumsum(np.random.randn(n_points) * 0.002 - 0.015)
    elif trend_type == 'volatile':
        base_trend = np.cumsum(np.random.randn(n_points) * 0.01)
    else:  # mixed
        base_trend = np.cumsum(np.random.randn(n_points) * 0.003 + 
                              0.01 * np.sin(np.arange(n_points) / 50))
    
    # 基本価格
    base_price = 100 + base_trend
    noise = np.random.normal(0, 1.0, n_points)
    prices = base_price + noise
    
    # OHLC作成
    data = []
    for i, price in enumerate(prices):
        vol = 0.8
        high = price + np.random.uniform(0, vol)
        low = price - np.random.uniform(0, vol)
        open_price = price + np.random.normal(0, vol/3)
        
        low = min(low, price, open_price)
        high = max(high, price, open_price)
        
        data.append([open_price, high, low, price])
    
    df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
    return df


def run_uma_v3_test(data, description="テストデータ"):
    """UltimateMA V3のテスト実行"""
    
    print(f"\n🔧 UltimateMA V3初期化中...")
    uma_v3 = UltimateMAV3(
        super_smooth_period=8,
        zero_lag_period=16,
        realtime_window=34,
        quantum_window=16,
        fractal_window=16,
        entropy_window=16,
        src_type='hlc3',
        slope_index=2,
        base_threshold=0.002,
        min_confidence=0.15
    )
    
    print(f"⚡ 計算実行中... ({description})")
    start_time = time.time()
    result = uma_v3.calculate(data)
    calc_time = time.time() - start_time
    
    print(f"✅ 計算完了 (時間: {calc_time:.2f}秒, 速度: {len(data)/calc_time:.0f} データ/秒)")
    
    # 結果分析
    confident_signals = result.trend_confidence[result.trend_confidence > 0]
    high_conf_signals = result.trend_confidence[result.trend_confidence > 0.5]
    
    up_signals = np.sum(result.trend_signals == 1)
    down_signals = np.sum(result.trend_signals == -1)
    range_signals = np.sum(result.trend_signals == 0)
    
    print(f"\n📊 結果サマリー:")
    print(f"   現在のトレンド: {result.current_trend.upper()} (信頼度: {result.current_confidence:.3f})")
    print(f"   上昇シグナル: {up_signals} ({up_signals/len(data)*100:.1f}%)")
    print(f"   下降シグナル: {down_signals} ({down_signals/len(data)*100:.1f}%)")
    print(f"   レンジシグナル: {range_signals} ({range_signals/len(data)*100:.1f}%)")
    print(f"   平均信頼度: {np.mean(confident_signals):.3f}")
    print(f"   高信頼度シグナル: {len(high_conf_signals)}個")
    
    print(f"\n🌌 量子分析:")
    print(f"   量子状態: {np.mean(result.quantum_state):.3f}")
    print(f"   MTF合意度: {np.mean(result.multi_timeframe_consensus):.3f}")
    print(f"   フラクタル次元: {np.mean(result.fractal_dimension):.3f}")
    print(f"   エントロピー: {np.mean(result.entropy_level):.3f}")
    
    # ノイズ除去効果
    raw_vol = np.std(result.raw_values)
    filtered_vol = np.std(result.values)
    noise_reduction = (raw_vol - filtered_vol) / raw_vol * 100 if raw_vol > 0 else 0
    
    print(f"\n🔇 ノイズ除去効果: {noise_reduction:.1f}%")
    
    return result


def main():
    print("🚀 UltimateMA V3 - Simple Demo")
    print("量子ニューラル・フラクタル・エントロピー統合分析システム")
    print("="*80)
    
    # 異なるタイプのテストデータでテスト
    test_cases = [
        ('上昇トレンド', 'up', 800),
        ('下降トレンド', 'down', 800),
        ('ボラタイル相場', 'volatile', 1000),
        ('ミックス相場', 'mixed', 1200)
    ]
    
    results = {}
    
    for name, trend_type, n_points in test_cases:
        print(f"\n{'='*20} {name}テスト {'='*20}")
        
        # データ生成
        data = create_sample_data(n_points, trend_type)
        print(f"📊 {name}データ生成: {n_points}件")
        
        # テスト実行
        result = run_uma_v3_test(data, f"{name}データ")
        results[name] = result
    
    # 総合評価
    print(f"\n{'='*80}")
    print("🏆 総合評価")
    print("="*80)
    
    for name, result in results.items():
        confident_signals = result.trend_confidence[result.trend_confidence > 0]
        avg_confidence = np.mean(confident_signals) if len(confident_signals) > 0 else 0
        quantum_strength = np.mean(np.abs(result.quantum_state))
        
        print(f"\n{name}:")
        print(f"  トレンド判定: {result.current_trend.upper()}")
        print(f"  信頼度: {result.current_confidence:.3f}")
        print(f"  平均信頼度: {avg_confidence:.3f}")
        print(f"  量子強度: {quantum_strength:.3f}")
        
        # 評価スコア
        score = (avg_confidence * 0.4 + 
                min(quantum_strength * 5, 1.0) * 0.3 + 
                min(np.mean(result.multi_timeframe_consensus), 1.0) * 0.3)
        
        if score >= 0.7:
            evaluation = "🏆 EXCELLENT"
        elif score >= 0.5:
            evaluation = "🥈 GOOD"
        elif score >= 0.3:
            evaluation = "🥉 FAIR"
        else:
            evaluation = "📈 DEVELOPING"
        
        print(f"  評価: {evaluation} (スコア: {score:.3f})")
    
    print(f"\n✅ UltimateMA V3 Simple Demo 完了")
    print("🌟 全ての市場条件でテスト完了！")


if __name__ == "__main__":
    main() 