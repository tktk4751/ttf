#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from indicators.ultimate_breakout_channel import (
    quantum_enhanced_hilbert_transform,
    quantum_adaptive_kalman_filter,
    hyper_efficiency_ratio,
    wavelet_multiresolution_analysis,
    elite_dynamic_multiplier_system
)

def step_debug():
    print("🔍 段階別デバッグテスト")
    print("=" * 50)
    
    # シンプルなテストデータ
    np.random.seed(42)
    n = 50  # 小さなサイズでテスト
    prices = 50000 + np.cumsum(np.random.randn(n) * 100)
    
    print(f"テストデータ: {n}ポイント")
    print(f"価格レンジ: {prices.min():.0f} - {prices.max():.0f}")
    
    # 段階1: ヒルベルト変換
    print("\n🔥 段階1: ヒルベルト変換")
    hilbert_amplitude, hilbert_phase, trend_strength, quantum_entanglement = quantum_enhanced_hilbert_transform(prices)
    
    valid_trend = trend_strength[~np.isnan(trend_strength)]
    valid_quantum = quantum_entanglement[~np.isnan(quantum_entanglement)]
    
    print(f"  トレンド強度: {len(valid_trend)}個の有効値, 範囲={valid_trend.min():.3f}-{valid_trend.max():.3f}")
    print(f"  量子もつれ: {len(valid_quantum)}個の有効値, 範囲={valid_quantum.min():.3f}-{valid_quantum.max():.3f}")
    
    # 段階2: カルマンフィルター
    print("\n⚡ 段階2: カルマンフィルター")
    centerline, quantum_coherence = quantum_adaptive_kalman_filter(prices, hilbert_amplitude, hilbert_phase)
    
    valid_coherence = quantum_coherence[~np.isnan(quantum_coherence)]
    print(f"  量子コヒーレンス: {len(valid_coherence)}個の有効値, 範囲={valid_coherence.min():.3f}-{valid_coherence.max():.3f}")
    
    # 段階3: 効率率
    print("\n📊 段階3: ハイパー効率率")
    hyper_efficiency = hyper_efficiency_ratio(prices, 14)
    
    valid_efficiency = hyper_efficiency[~np.isnan(hyper_efficiency)]
    print(f"  効率率: {len(valid_efficiency)}個の有効値, 範囲={valid_efficiency.min():.3f}-{valid_efficiency.max():.3f}")
    
    # 段階4: ウェーブレット
    print("\n🌊 段階4: ウェーブレット解析")
    wavelet_trend, wavelet_cycle, market_regime = wavelet_multiresolution_analysis(prices)
    
    valid_wavelet = wavelet_trend[~np.isnan(wavelet_trend)]
    print(f"  ウェーブレットトレンド: {len(valid_wavelet)}個の有効値, 範囲={valid_wavelet.min():.3f}-{valid_wavelet.max():.3f}")
    
    # 段階5: ダミーボラティリティ
    print("\n💨 段階5: ダミーボラティリティ")
    dummy_volatility = np.full(n, 100.0)  # 固定値
    print(f"  ダミーボラティリティ: 全て100.0で統一")
    
    # 段階6: 動的乗数
    print("\n🎯 段階6: 動的乗数システム")
    dynamic_multiplier, confidence_score = elite_dynamic_multiplier_system(
        dummy_volatility, trend_strength, hyper_efficiency, 
        quantum_entanglement, 1.0, 6.0
    )
    
    valid_multiplier = dynamic_multiplier[~np.isnan(dynamic_multiplier)]
    valid_confidence = confidence_score[~np.isnan(confidence_score)]
    
    print(f"  動的乗数: {len(valid_multiplier)}個の有効値")
    if len(valid_multiplier) > 0:
        print(f"    範囲={valid_multiplier.min():.3f}-{valid_multiplier.max():.3f}")
        print(f"    平均={valid_multiplier.mean():.3f}")
    else:
        print(f"    全てがNaN - 問題あり！")
        
        # 詳細デバッグ
        print(f"\n🔍 詳細デバッグ:")
        print(f"  インデックス15からの状況:")
        for i in range(15, min(25, n)):
            ts = trend_strength[i] if not np.isnan(trend_strength[i]) else "NaN"
            he = hyper_efficiency[i] if not np.isnan(hyper_efficiency[i]) else "NaN"
            qe = quantum_entanglement[i] if not np.isnan(quantum_entanglement[i]) else "NaN"
            dm = dynamic_multiplier[i] if not np.isnan(dynamic_multiplier[i]) else "NaN"
            print(f"    i={i}: ts={ts}, he={he}, qe={qe}, dm={dm}")
    
    print(f"  信頼度スコア: {len(valid_confidence)}個の有効値")
    
    print("\n✅ 段階別デバッグ完了")

if __name__ == "__main__":
    step_debug() 