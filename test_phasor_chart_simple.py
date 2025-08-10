#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from indicators.trend_filter.phasor_trend_filter import PhasorTrendFilter

def test_phasor_chart():
    """Phasor Trend Filterのシンプルなチャートテスト"""
    
    # テストデータ生成（トレンドとレンジが明確に分かれるデータ）
    np.random.seed(42)
    length = 500
    
    # より明確なトレンド・レンジパターンを生成
    prices = [100.0]
    for i in range(1, length):
        if i < 100:  # 強い上昇トレンド
            change = 0.005 + np.random.normal(0, 0.008)
        elif i < 200:  # レンジ相場
            change = np.random.normal(0, 0.003)
        elif i < 300:  # 強い下降トレンド
            change = -0.004 + np.random.normal(0, 0.008)
        elif i < 400:  # レンジ相場
            change = np.random.normal(0, 0.003)
        else:  # 上昇トレンド
            change = 0.003 + np.random.normal(0, 0.005)
        
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # OHLC データの生成
    data = []
    for i, close in enumerate(prices):
        daily_range = abs(np.random.normal(0, close * 0.01))
        
        high = close + daily_range * np.random.uniform(0.3, 1.0)
        low = close - daily_range * np.random.uniform(0.3, 1.0)
        
        if i == 0:
            open_price = close
        else:
            gap = np.random.normal(0, close * 0.002)
            open_price = prices[i-1] + gap
        
        # 論理的整合性の確保
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.uniform(1000, 10000)
        })
    
    df = pd.DataFrame(data)
    df.index = pd.date_range(start='2023-01-01', periods=len(df), freq='4H')
    
    print(f"テストデータ: {len(df)}ポイント")
    print(f"価格範囲: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    # Phasor Trend Filterを計算
    print("\nPhasor Trend Filterを計算中...")
    ptf = PhasorTrendFilter(
        period=28,
        trend_threshold=6.0,
        src_type='close',
        use_kalman_filter=False
    )
    
    result = ptf.calculate(df)
    
    print(f"計算完了 - データ点数: {len(result.state)}")
    
    # 統計情報
    up_count = (result.state == 1).sum()
    down_count = (result.state == -1).sum()
    range_count = (result.state == 0).sum()
    
    print(f"上昇トレンド: {up_count} ({up_count/len(result.state)*100:.1f}%)")
    print(f"下降トレンド: {down_count} ({down_count/len(result.state)*100:.1f}%)")
    print(f"レンジ状態: {range_count} ({range_count/len(result.state)*100:.1f}%)")
    
    # シンプルなMatplotlibチャートを作成
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # 価格チャート
    axes[0].plot(df.index, df['close'], label='Close Price', color='black', linewidth=1)
    axes[0].set_title('Price Chart')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # トレンド状態チャート
    # 状態別に色分けしてプロット
    trend_state_full = pd.Series(result.state, index=df.index)
    
    # 各状態を個別にプロット
    up_mask = trend_state_full == 1
    down_mask = trend_state_full == -1
    range_mask = trend_state_full == 0
    
    if up_mask.any():
        axes[1].plot(df.index[up_mask], trend_state_full[up_mask], 'g-', linewidth=3, label='Up Trend (+1)')
    if down_mask.any():
        axes[1].plot(df.index[down_mask], trend_state_full[down_mask], 'r-', linewidth=3, label='Down Trend (-1)')
    if range_mask.any():
        axes[1].plot(df.index[range_mask], trend_state_full[range_mask], 'gray', linewidth=2, label='Range (0)')
    
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[1].axhline(y=1, color='green', linestyle='--', alpha=0.5)
    axes[1].axhline(y=-1, color='red', linestyle='--', alpha=0.5)
    axes[1].set_ylim(-1.5, 1.5)
    axes[1].set_ylabel('Trend State')
    axes[1].set_title('Phasor Trend Filter - Trend State')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # フェーザー角度チャート
    phase_angle_full = pd.Series(result.phase_angle, index=df.index)
    axes[2].plot(df.index, phase_angle_full, color='orange', linewidth=1, label='Phase Angle')
    axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[2].axhline(y=90, color='green', linestyle='--', alpha=0.3)
    axes[2].axhline(y=-90, color='red', linestyle='--', alpha=0.3)
    axes[2].set_ylabel('Phase Angle (°)')
    axes[2].set_title('Phasor Phase Angle')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('phasor_trend_filter_simple_test.png', dpi=150, bbox_inches='tight')
    print(f"\nチャートを保存しました: phasor_trend_filter_simple_test.png")
    
    return result

if __name__ == "__main__":
    test_phasor_chart()