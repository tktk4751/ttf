#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from indicators.ultimate_breakout_channel import UltimateBreakoutChannel

def create_test_data(num_points=100):
    """シンプルなテストデータ"""
    np.random.seed(42)
    
    # より大きなボラティリティでテスト
    prices = 50000 + np.cumsum(np.random.randn(num_points) * 100)  # より大きな変動
    
    highs = prices + np.abs(np.random.randn(num_points)) * 200
    lows = prices - np.abs(np.random.randn(num_points)) * 200
    closes = prices + np.random.randn(num_points) * 50
    
    return pd.DataFrame({
        'open': prices,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': np.random.randint(100, 1000, num_points)
    })

def debug_volatility():
    print("🔍 ボラティリティデバッグテスト")
    print("=" * 50)
    
    data = create_test_data(100)
    print(f"テストデータサンプル:")
    print(f"価格レンジ: {data['close'].min():.0f} - {data['close'].max():.0f}")
    print(f"平均高値: {data['high'].mean():.0f}")
    print(f"平均安値: {data['low'].mean():.0f}")
    
    # ATR版テスト
    print("\n🔥 ATR版デバッグ")
    ubc_atr = UltimateBreakoutChannel(
        volatility_type='atr',
        min_multiplier=1.0,
        max_multiplier=6.0
    )
    
    result_atr = ubc_atr.calculate(data)
    
    # Ultimate版テスト
    print("\n⚡ Ultimate版デバッグ")
    ubc_ultimate = UltimateBreakoutChannel(
        volatility_type='ultimate',
        min_multiplier=1.0,
        max_multiplier=6.0
    )
    
    result_ultimate = ubc_ultimate.calculate(data)
    
    # 結果分析
    print("\n📊 結果詳細:")
    
    # ATR版
    atr_width = result_atr.dynamic_width[~np.isnan(result_atr.dynamic_width)]
    atr_mult = result_atr.dynamic_multiplier[~np.isnan(result_atr.dynamic_multiplier)]
    
    print(f"\nATR版:")
    print(f"  全動的乗数配列サイズ: {len(result_atr.dynamic_multiplier)}")
    print(f"  NaN以外の動的乗数数: {len(atr_mult)}")
    print(f"  チャネル幅統計: min={np.min(atr_width):.2f}, max={np.max(atr_width):.2f}, mean={np.mean(atr_width):.2f}")
    
    if len(atr_mult) > 0:
        print(f"  動的乗数統計: min={np.min(atr_mult):.2f}, max={np.max(atr_mult):.2f}, mean={np.mean(atr_mult):.2f}")
    else:
        print(f"  動的乗数: 全てがNaN")
        print(f"  動的乗数サンプル: {result_atr.dynamic_multiplier[-5:]}")
    print(f"  シグナル数: {int(np.sum(np.abs(result_atr.breakout_signals)))}")
    
    # Ultimate版
    ult_width = result_ultimate.dynamic_width[~np.isnan(result_ultimate.dynamic_width)]
    ult_mult = result_ultimate.dynamic_multiplier[~np.isnan(result_ultimate.dynamic_multiplier)]
    
    print(f"\nUltimate版:")
    print(f"  全動的乗数配列サイズ: {len(result_ultimate.dynamic_multiplier)}")
    print(f"  NaN以外の動的乗数数: {len(ult_mult)}")
    print(f"  チャネル幅統計: min={np.min(ult_width):.2f}, max={np.max(ult_width):.2f}, mean={np.mean(ult_width):.2f}")
    
    if len(ult_mult) > 0:
        print(f"  動的乗数統計: min={np.min(ult_mult):.2f}, max={np.max(ult_mult):.2f}, mean={np.mean(ult_mult):.2f}")
    else:
        print(f"  動的乗数: 全てがNaN")
        print(f"  動的乗数サンプル: {result_ultimate.dynamic_multiplier[-5:]}")
    print(f"  シグナル数: {int(np.sum(np.abs(result_ultimate.breakout_signals)))}")
    
    # チャネル幅の比較
    print(f"\n📈 チャネル幅比較:")
    if len(atr_width) > 0 and len(ult_width) > 0:
        print(f"  ATR版平均幅: {np.mean(atr_width):.2f}")
        print(f"  Ultimate版平均幅: {np.mean(ult_width):.2f}")
        print(f"  Ultimate版はATR版の{np.mean(ult_width)/np.mean(atr_width):.2f}倍")
    else:
        print(f"  チャネル幅比較: データ不足")
    
    # 簡単なチャート
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    x = range(len(data))
    
    # 価格とチャネル
    axes[0].plot(x, data['close'], label='価格', color='black')
    axes[0].plot(x, result_atr.upper_channel, label='ATR上限', color='blue', alpha=0.7)
    axes[0].plot(x, result_atr.lower_channel, label='ATR下限', color='blue', alpha=0.7)
    axes[0].plot(x, result_ultimate.upper_channel, label='Ultimate上限', color='red', alpha=0.7)
    axes[0].plot(x, result_ultimate.lower_channel, label='Ultimate下限', color='red', alpha=0.7)
    axes[0].set_title('価格チャネル比較')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 動的乗数
    axes[1].plot(x, result_atr.dynamic_multiplier, label='ATR動的乗数', color='blue')
    axes[1].plot(x, result_ultimate.dynamic_multiplier, label='Ultimate動的乗数', color='red')
    axes[1].set_title('動的乗数比較')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('examples/output/volatility_debug.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 デバッグチャート保存: examples/output/volatility_debug.png")
    plt.show()

if __name__ == "__main__":
    debug_volatility() 