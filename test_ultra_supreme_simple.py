#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🚀 **ULTRA SUPREME DFT CYCLE 簡易テスト** 🚀

EhlersUnifiedDC経由での Ultra Supreme DFT の基本動作確認
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from indicators.cycle.ehlers_unified_dc import EhlersUnifiedDC

def main():
    print("🚀 Ultra Supreme DFT 簡易動作テスト")
    print("=" * 50)
    
    # テストデータ生成
    n_points = 200
    t = np.arange(n_points)
    base_price = 100.0
    cycle_signal = 10.0 * np.sin(2 * np.pi * t / 20)  # 20期間サイクル
    noise = np.random.normal(0, 2, n_points)
    
    test_data = pd.DataFrame({
        'open': base_price + cycle_signal + noise,
        'high': base_price + cycle_signal + noise + 1,
        'low': base_price + cycle_signal + noise - 1,
        'close': base_price + cycle_signal + noise,
        'volume': np.random.lognormal(8, 0.2, n_points)
    })
    
    # Ultra Supreme DFT 検出器テスト
    print("🧠 Ultra Supreme DFT 検出器初期化...")
    
    detector = EhlersUnifiedDC(
        detector_type='ultra_supreme_dft',
        cycle_part=0.5,
        max_output=40,
        min_output=8,
        src_type='close',
        use_kalman_filter=True,
        kalman_filter_type='adaptive',  # 軽量カルマンフィルター
        window=30
    )
    
    print("🔄 サイクル検出実行...")
    cycles = detector.calculate(test_data)
    
    # 結果分析
    stable_cycles = cycles[50:-20]  # 安定期間
    avg_cycle = np.mean(stable_cycles)
    std_cycle = np.std(stable_cycles)
    
    print(f"\n📊 結果:")
    print(f"  • データポイント: {len(cycles)}")
    print(f"  • 平均検出サイクル: {avg_cycle:.2f}")
    print(f"  • 標準偏差: {std_cycle:.2f}")
    print(f"  • 真のサイクル: 20.0")
    print(f"  • 絶対誤差: {abs(avg_cycle - 20.0):.2f}")
    print(f"  • 相対誤差: {abs(avg_cycle - 20.0) / 20.0 * 100:.1f}%")
    
    # サンプル値表示
    print(f"\n🔍 サンプル検出値 (最後の10個):")
    for i, cycle in enumerate(cycles[-10:], 1):
        print(f"  {i:2d}. {cycle:.1f}")
    
    print("\n✅ Ultra Supreme DFT 基本動作確認完了!")
    
    return cycles

if __name__ == "__main__":
    try:
        np.random.seed(42)
        cycles = main()
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback
        traceback.print_exc()