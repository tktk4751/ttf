#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from indicators.p_mama import P_MAMA

def generate_test_data(length: int = 200) -> pd.DataFrame:
    """テスト用のOHLCデータを生成"""
    np.random.seed(42)
    base_price = 100.0
    
    # トレンドとレンジが混在するデータを生成
    prices = [base_price]
    for i in range(1, length):
        if i < 50:  # トレンド相場
            change = 0.002 + np.random.normal(0, 0.01)
        elif i < 100:  # レンジ相場
            change = np.random.normal(0, 0.008)
        elif i < 150:  # 強いトレンド相場
            change = 0.004 + np.random.normal(0, 0.015)
        else:  # レンジ相場
            change = np.random.normal(0, 0.006)
        
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
            gap = np.random.normal(0, close * 0.005)
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
    
    return pd.DataFrame(data)

def test_p_mama():
    """P_MAMAのテスト"""
    print("=== P_MAMA インジケーターのテスト ===")
    
    # テストデータ生成
    df = generate_test_data(200)
    print(f"テストデータ: {len(df)}ポイント")
    print(f"価格範囲: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    # 基本版P_MAMAをテスト
    print("\\n基本版P_MAMAをテスト中...")
    p_mama_basic = P_MAMA(
        period=28,
        fast_limit=0.5,
        slow_limit=0.05,
        src_type='close',
        use_kalman_filter=False,
        use_zero_lag=False
    )
    
    try:
        result_basic = p_mama_basic.calculate(df)
        print(f"  P_MAMA結果の型: {type(result_basic)}")
        print(f"  MAMA配列の形状: {result_basic.mama_values.shape}")
        print(f"  FAMA配列の形状: {result_basic.fama_values.shape}")
        print(f"  Phase配列の形状: {result_basic.phase_values.shape}")
        print(f"  State配列の形状: {result_basic.state_values.shape}")
        
        valid_count = np.sum(~np.isnan(result_basic.mama_values))
        mean_mama = np.nanmean(result_basic.mama_values)
        mean_fama = np.nanmean(result_basic.fama_values)
        mean_phase = np.nanmean(result_basic.phase_values)
        uptrend_count = np.sum(result_basic.state_values == 1)
        downtrend_count = np.sum(result_basic.state_values == -1)
        
        print(f"  有効値数: {valid_count}/{len(df)}")
        print(f"  平均P_MAMA: {mean_mama:.4f}")
        print(f"  平均P_FAMA: {mean_fama:.4f}")
        print(f"  平均Phase: {mean_phase:.2f}°")
        print(f"  上昇トレンド: {uptrend_count}期間")
        print(f"  下降トレンド: {downtrend_count}期間")
        
        return result_basic
        
    except Exception as e:
        print(f"  エラー: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_zero_lag_p_mama():
    """ゼロラグ処理版P_MAMAのテスト"""
    print("\\nゼロラグ処理版P_MAMAをテスト中...")
    df = generate_test_data(200)
    
    p_mama_zero_lag = P_MAMA(
        period=28,
        fast_limit=0.5,
        slow_limit=0.05,
        src_type='close',
        use_kalman_filter=False,
        use_zero_lag=True
    )
    
    try:
        result_zero_lag = p_mama_zero_lag.calculate(df)
        
        valid_count_zero_lag = np.sum(~np.isnan(result_zero_lag.mama_values))
        mean_mama_zero_lag = np.nanmean(result_zero_lag.mama_values)
        mean_fama_zero_lag = np.nanmean(result_zero_lag.fama_values)
        
        print(f"  有効値数: {valid_count_zero_lag}/{len(df)}")
        print(f"  平均P_MAMA（ゼロラグ）: {mean_mama_zero_lag:.4f}")
        print(f"  平均P_FAMA（ゼロラグ）: {mean_fama_zero_lag:.4f}")
        
        return result_zero_lag
        
    except Exception as e:
        print(f"  ゼロラグ処理版でエラー: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_comparison(result_basic, result_zero_lag, df):
    """基本版とゼロラグ版の比較プロット"""
    if result_basic is None or result_zero_lag is None:
        print("結果がないため、プロットをスキップします")
        return
    
    print("\\n比較チャートを作成中...")
    
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    # 価格チャート
    axes[0].plot(df['close'], label='Close Price', color='black', alpha=0.7)
    axes[0].plot(result_basic.mama_values, label='P_MAMA (Basic)', color='blue', linewidth=2)
    axes[0].plot(result_basic.fama_values, label='P_FAMA (Basic)', color='orange', linewidth=1.5)
    axes[0].plot(result_zero_lag.mama_values, label='P_MAMA (Zero-lag)', color='green', linewidth=2, alpha=0.8)
    axes[0].plot(result_zero_lag.fama_values, label='P_FAMA (Zero-lag)', color='red', linewidth=1.5, alpha=0.8)
    axes[0].set_title('P_MAMA/P_FAMA Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # フェーザー角度
    axes[1].plot(result_basic.phase_values, label='Phase (Basic)', color='purple', linewidth=1.5)
    axes[1].plot(result_zero_lag.phase_values, label='Phase (Zero-lag)', color='magenta', linewidth=1.5, alpha=0.8)
    axes[1].axhline(y=90, color='green', linestyle='--', alpha=0.5)
    axes[1].axhline(y=-90, color='red', linestyle='--', alpha=0.5)
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1].set_title('Phasor Angle')
    axes[1].set_ylabel('Degrees')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # トレンド状態
    axes[2].plot(result_basic.state_values, label='State (Basic)', color='brown', linewidth=2, marker='o', markersize=2)
    axes[2].plot(result_zero_lag.state_values, label='State (Zero-lag)', color='darkgreen', linewidth=2, marker='s', markersize=2, alpha=0.8)
    axes[2].axhline(y=1, color='green', linestyle='--', alpha=0.5)
    axes[2].axhline(y=-1, color='red', linestyle='--', alpha=0.5)
    axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[2].set_title('Trend State')
    axes[2].set_ylabel('State (+1: Up, 0: Cycle, -1: Down)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # アルファ値
    axes[3].plot(result_basic.alpha_values, label='Alpha (Basic)', color='blue', linewidth=1.5)
    axes[3].plot(result_zero_lag.alpha_values, label='Alpha (Zero-lag)', color='cyan', linewidth=1.5, alpha=0.8)
    axes[3].set_title('Alpha Values')
    axes[3].set_ylabel('Alpha')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('p_mama_test_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("比較チャートを p_mama_test_result.png に保存しました")

if __name__ == "__main__":
    # テスト実行
    result_basic = test_p_mama()
    result_zero_lag = test_zero_lag_p_mama()
    
    # 比較プロット
    if result_basic is not None and result_zero_lag is not None:
        df = generate_test_data(200)
        plot_comparison(result_basic, result_zero_lag, df)
    
    print("\\n=== テスト完了 ===")