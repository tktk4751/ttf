#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from indicators.ultimate_breakout_channel import elite_dynamic_multiplier_system

def test_simple_multiplier():
    print("🎯 簡単な動的乗数テスト")
    print("=" * 40)
    
    # シンプルなテストデータ
    n = 30
    
    # 全て有効な値
    dummy_vol = np.full(n, 100.0)  # 固定ボラティリティ
    trend_strength = np.full(n, 0.5)  # 固定トレンド強度
    her_values = np.full(n, 0.7)  # 固定効率率
    quantum_entanglement = np.full(n, 0.3)  # 固定量子もつれ
    
    print(f"テストデータ:")
    print(f"  データ数: {n}")
    print(f"  ボラティリティ: 全て100.0")
    print(f"  トレンド強度: 全て0.5")
    print(f"  効率率: 全て0.7")
    print(f"  量子もつれ: 全て0.3")
    
    try:
        dynamic_multiplier, confidence_score = elite_dynamic_multiplier_system(
            dummy_vol, trend_strength, her_values, 
            quantum_entanglement, 1.0, 6.0
        )
        
        valid_mult = dynamic_multiplier[~np.isnan(dynamic_multiplier)]
        valid_conf = confidence_score[~np.isnan(confidence_score)]
        
        print(f"\n結果:")
        print(f"  動的乗数 有効数: {len(valid_mult)}")
        if len(valid_mult) > 0:
            print(f"  動的乗数 範囲: {valid_mult.min():.3f} - {valid_mult.max():.3f}")
            print(f"  動的乗数 平均: {valid_mult.mean():.3f}")
        else:
            print(f"  動的乗数: 全てNaN")
            
        print(f"  信頼度 有効数: {len(valid_conf)}")
        if len(valid_conf) > 0:
            print(f"  信頼度 範囲: {valid_conf.min():.3f} - {valid_conf.max():.3f}")
        
        # インデックス別詳細
        print(f"\nインデックス別詳細:")
        for i in range(10, min(20, n)):
            dm = dynamic_multiplier[i]
            cs = confidence_score[i]
            dm_str = f"{dm:.3f}" if not np.isnan(dm) else "NaN"
            cs_str = f"{cs:.3f}" if not np.isnan(cs) else "NaN"
            print(f"  i={i}: 動的乗数={dm_str}, 信頼度={cs_str}")
        
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()

def test_partial_nan():
    print("\n🎯 部分的NaNテスト")
    print("=" * 40)
    
    n = 30
    dummy_vol = np.full(n, 100.0)
    trend_strength = np.full(n, np.nan)  # 全てNaN
    her_values = np.full(n, 0.7)
    quantum_entanglement = np.full(n, 0.3)
    
    # いくつかの値を有効にする
    trend_strength[20:] = 0.5
    
    print(f"テストデータ:")
    print(f"  トレンド強度: インデックス20以降のみ有効（0.5）")
    print(f"  他は全て有効")
    
    try:
        dynamic_multiplier, confidence_score = elite_dynamic_multiplier_system(
            dummy_vol, trend_strength, her_values, 
            quantum_entanglement, 1.0, 6.0
        )
        
        valid_mult = dynamic_multiplier[~np.isnan(dynamic_multiplier)]
        print(f"\n結果:")
        print(f"  動的乗数 有効数: {len(valid_mult)}")
        if len(valid_mult) > 0:
            print(f"  動的乗数 平均: {valid_mult.mean():.3f}")
        
        # インデックス別詳細
        print(f"\nインデックス別詳細:")
        for i in range(15, n):
            ts = trend_strength[i] if not np.isnan(trend_strength[i]) else "NaN"
            dm = dynamic_multiplier[i] if not np.isnan(dynamic_multiplier[i]) else "NaN"
            print(f"  i={i}: ts={ts}, dm={dm}")
        
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()

def main():
    test_simple_multiplier()
    test_partial_nan()

if __name__ == "__main__":
    main() 