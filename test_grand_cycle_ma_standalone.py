#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Union, Optional, Dict, Tuple
from numba import njit
import math

@dataclass
class GrandCycleMAResult:
    """グランドサイクルMAの計算結果"""
    grand_mama_values: np.ndarray     # グランドサイクルMAMA値
    grand_fama_values: np.ndarray     # グランドサイクルFAMA値
    cycle_period: np.ndarray          # サイクル周期
    alpha_values: np.ndarray          # アルファ値
    phase_values: np.ndarray          # 位相値


@njit(fastmath=True, cache=True)
def calculate_grand_cycle_ma_core(
    price: np.ndarray,
    cycle_period: np.ndarray,
    fast_limit: float = 0.5,
    slow_limit: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    グランドサイクルMAを計算する（Numba最適化版）
    
    Args:
        price: 価格配列
        cycle_period: サイクル周期配列
        fast_limit: 速いリミット（デフォルト: 0.5）
        slow_limit: 遅いリミット（デフォルト: 0.05）
    
    Returns:
        Tuple[np.ndarray, ...]: グランドMAMA値, グランドFAMA値, Alpha値, Phase値
    """
    length = len(price)
    
    # 変数の初期化
    smooth = np.zeros(length, dtype=np.float64)
    phase = np.zeros(length, dtype=np.float64)
    delta_phase = np.zeros(length, dtype=np.float64)
    alpha = np.zeros(length, dtype=np.float64)
    grand_mama = np.zeros(length, dtype=np.float64)
    grand_fama = np.zeros(length, dtype=np.float64)
    
    # 初期値設定
    for i in range(min(5, length)):
        smooth[i] = price[i] if i < length else 100.0
        phase[i] = 0.0
        delta_phase[i] = 1.0
        alpha[i] = slow_limit
        grand_mama[i] = price[i] if i < length else 100.0
        grand_fama[i] = price[i] if i < length else 100.0
    
    # メインループ
    for i in range(5, length):
        # 価格のスムージング（MAMA方式）
        if i >= 3:
            smooth[i] = (4.0 * price[i] + 3.0 * price[i-1] + 2.0 * price[i-2] + price[i-3]) / 10.0
        else:
            smooth[i] = price[i]
        
        # サイクル周期を使った位相計算
        current_period = cycle_period[i] if not np.isnan(cycle_period[i]) and cycle_period[i] > 0 else 20.0
        
        # 位相の計算（簡略化されたHilbert変換）
        if i >= 4:
            phase[i] = math.atan2(
                smooth[i] - smooth[i-4],
                smooth[i-2]
            ) * 180.0 / math.pi
        else:
            phase[i] = phase[i-1] if i > 0 else 0.0
        
        # DeltaPhase計算
        if i > 0:
            delta_phase[i] = abs(phase[i-1] - phase[i])
            if delta_phase[i] < 1.0:
                delta_phase[i] = 1.0
        else:
            delta_phase[i] = 1.0
        
        # サイクル周期に基づいたAlpha調整
        # より長いサイクル周期の場合はより低速に、短い場合は高速に
        cycle_factor = 20.0 / current_period if current_period > 0 else 1.0
        adjusted_fast_limit = fast_limit * cycle_factor
        adjusted_fast_limit = min(adjusted_fast_limit, 1.0)
        adjusted_fast_limit = max(adjusted_fast_limit, slow_limit)
        
        # Alpha計算
        if delta_phase[i] > 0:
            alpha[i] = adjusted_fast_limit / delta_phase[i]
            if alpha[i] < slow_limit:
                alpha[i] = slow_limit
            elif alpha[i] > adjusted_fast_limit:
                alpha[i] = adjusted_fast_limit
        else:
            alpha[i] = slow_limit
        
        # グランドサイクルMAMA計算
        if i > 0 and not np.isnan(grand_mama[i-1]) and not np.isnan(alpha[i]):
            grand_mama[i] = alpha[i] * smooth[i] + (1.0 - alpha[i]) * grand_mama[i-1]
        else:
            grand_mama[i] = smooth[i]
        
        # グランドサイクルFAMA計算
        if i > 0 and not np.isnan(grand_fama[i-1]) and not np.isnan(grand_mama[i]) and not np.isnan(alpha[i]):
            grand_fama[i] = 0.5 * alpha[i] * grand_mama[i] + (1.0 - 0.5 * alpha[i]) * grand_fama[i-1]
        else:
            grand_fama[i] = grand_mama[i]
    
    return grand_mama, grand_fama, alpha, phase


def simple_cycle_detector(price: np.ndarray, period: int = 20) -> np.ndarray:
    """
    簡易サイクル検出器（テスト用）
    
    Args:
        price: 価格配列
        period: 基本周期
    
    Returns:
        np.ndarray: サイクル周期配列
    """
    length = len(price)
    cycle_period = np.full(length, float(period))
    
    # 単純な変動率ベースのサイクル調整
    for i in range(period, length):
        # 過去の価格変動から周期を推定
        recent_prices = price[i-period:i]
        volatility = np.std(recent_prices)
        mean_price = np.mean(recent_prices)
        
        # 変動率に基づいて周期を調整
        if volatility > 0:
            vol_ratio = volatility / (mean_price * 0.01)  # 1%基準
            if vol_ratio > 2.0:  # 高ボラティリティ
                cycle_period[i] = max(period * 0.7, 6)  # 短い周期
            elif vol_ratio < 0.5:  # 低ボラティリティ
                cycle_period[i] = min(period * 1.5, 50)  # 長い周期
            else:
                cycle_period[i] = period
        else:
            cycle_period[i] = period
    
    return cycle_period


def test_grand_cycle_ma_standalone():
    """スタンドアロンでのグランドサイクルMAテスト"""
    print("=== グランドサイクルMA スタンドアロンテスト開始 ===")
    
    # 1. テストデータの作成
    print("\n1. テストデータ作成中...")
    dates = pd.date_range('2024-01-01', periods=200, freq='1H')
    np.random.seed(42)
    
    # 複雑な価格データを生成（トレンド + 複数サイクル + ノイズ）
    t = np.linspace(0, 4*np.pi, 200)
    trend = np.linspace(100, 150, 200)
    
    # 複数のサイクル成分
    long_cycle = 8 * np.sin(t * 0.5)  # 長期サイクル
    short_cycle = 3 * np.sin(t * 2)   # 短期サイクル
    noise = np.random.normal(0, 1, 200)
    
    close_prices = trend + long_cycle + short_cycle + noise
    
    data = pd.DataFrame({
        'open': close_prices * 0.998,
        'high': close_prices * 1.003,
        'low': close_prices * 0.997,
        'close': close_prices,
        'volume': np.random.uniform(100000, 500000, 200)
    }, index=dates)
    
    print(f"✓ テストデータ作成完了: {len(data)}件")
    print(f"価格範囲: {data['close'].min():.2f} - {data['close'].max():.2f}")
    
    # 2. 簡易サイクル検出器でサイクル周期を計算
    print("\n2. サイクル周期計算中...")
    cycle_period = simple_cycle_detector(data['close'].values, period=20)
    print(f"✓ サイクル周期計算完了")
    print(f"周期範囲: {cycle_period.min():.1f} - {cycle_period.max():.1f}")
    
    # 3. グランドサイクルMAの計算
    print("\n3. グランドサイクルMA計算中...")
    
    grand_mama, grand_fama, alpha, phase = calculate_grand_cycle_ma_core(
        data['close'].values,
        cycle_period,
        fast_limit=0.5,
        slow_limit=0.05
    )
    
    # 結果オブジェクトの作成
    result = GrandCycleMAResult(
        grand_mama_values=grand_mama,
        grand_fama_values=grand_fama,
        cycle_period=cycle_period,
        alpha_values=alpha,
        phase_values=phase
    )
    
    print(f"✓ グランドサイクルMA計算完了")
    
    # 4. 結果の統計
    print("\n4. 結果統計:")
    
    valid_mama = grand_mama[~np.isnan(grand_mama)]
    valid_fama = grand_fama[~np.isnan(grand_fama)]
    valid_alpha = alpha[~np.isnan(alpha)]
    
    print(f"グランドMAMA:")
    print(f"  有効データ: {len(valid_mama)}/{len(grand_mama)}")
    if len(valid_mama) > 0:
        print(f"  平均値: {np.mean(valid_mama):.4f}")
        print(f"  標準偏差: {np.std(valid_mama):.4f}")
        print(f"  範囲: {np.min(valid_mama):.4f} - {np.max(valid_mama):.4f}")
    
    print(f"グランドFAMA:")
    print(f"  有効データ: {len(valid_fama)}/{len(grand_fama)}")
    if len(valid_fama) > 0:
        print(f"  平均値: {np.mean(valid_fama):.4f}")
        print(f"  標準偏差: {np.std(valid_fama):.4f}")
        print(f"  範囲: {np.min(valid_fama):.4f} - {np.max(valid_fama):.4f}")
    
    print(f"Alpha値:")
    print(f"  有効データ: {len(valid_alpha)}/{len(alpha)}")
    if len(valid_alpha) > 0:
        print(f"  平均値: {np.mean(valid_alpha):.4f}")
        print(f"  標準偏差: {np.std(valid_alpha):.4f}")
        print(f"  範囲: {np.min(valid_alpha):.4f} - {np.max(valid_alpha):.4f}")
    
    # 5. 可視化
    print("\n5. 結果の可視化中...")
    
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    # 価格とグランドサイクルMA
    axes[0].plot(data.index, data['close'], label='Close Price', alpha=0.7, color='black', linewidth=1)
    axes[0].plot(data.index, grand_mama, label='Grand MAMA', alpha=0.9, color='blue', linewidth=2)
    axes[0].plot(data.index, grand_fama, label='Grand FAMA', alpha=0.9, color='red', linewidth=2)
    
    axes[0].set_title('Price vs Grand Cycle MAMA/FAMA')
    axes[0].set_ylabel('Price')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # サイクル周期
    axes[1].plot(data.index, cycle_period, label='Cycle Period', alpha=0.8, color='green', linewidth=1.5)
    axes[1].axhline(y=20, color='gray', linestyle='--', alpha=0.5, label='Default Period')
    
    axes[1].set_title('Detected Cycle Period')
    axes[1].set_ylabel('Period')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # アルファ値
    axes[2].plot(data.index, alpha, label='Alpha Values', alpha=0.8, color='orange', linewidth=1.5)
    axes[2].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Fast Limit')
    axes[2].axhline(y=0.05, color='gray', linestyle='--', alpha=0.5, label='Slow Limit')
    
    axes[2].set_title('Alpha Values (Adaptation Speed)')
    axes[2].set_ylabel('Alpha')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # 位相値
    axes[3].plot(data.index, phase, label='Phase Values', alpha=0.8, color='purple', linewidth=1.5)
    
    axes[3].set_title('Phase Values')
    axes[3].set_ylabel('Phase (degrees)')
    axes[3].set_xlabel('Date')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 画像を保存
    output_file = 'grand_cycle_ma_standalone_test.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ グラフを保存しました: {output_file}")
    
    plt.show()
    
    # 6. パフォーマンステスト
    print("\n6. パフォーマンステスト:")
    import time
    
    # 10回実行して平均時間を測定
    times = []
    for i in range(10):
        start_time = time.time()
        calculate_grand_cycle_ma_core(
            data['close'].values,
            cycle_period,
            fast_limit=0.5,
            slow_limit=0.05
        )
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    print(f"  平均計算時間: {avg_time:.6f}秒")
    print(f"  データ長: {len(data)}件")
    print(f"  処理速度: {len(data)/avg_time:.0f}件/秒")
    
    # 7. 適応性テスト
    print("\n7. 適応性テスト:")
    
    # MAMAとFAMAの応答性を比較
    price_changes = np.diff(data['close'].values)
    mama_changes = np.diff(grand_mama[1:])  # NaNを除く
    fama_changes = np.diff(grand_fama[1:])  # NaNを除く
    
    # 相関係数
    if len(mama_changes) > 0 and len(price_changes) > 0:
        mama_corr = np.corrcoef(price_changes[-len(mama_changes):], mama_changes)[0, 1]
        fama_corr = np.corrcoef(price_changes[-len(fama_changes):], fama_changes)[0, 1]
        
        print(f"  価格変化との相関:")
        print(f"    Grand MAMA: {mama_corr:.4f}")
        print(f"    Grand FAMA: {fama_corr:.4f}")
        print(f"    (高い値ほど応答性が良い)")
    
    print("\n=== グランドサイクルMA スタンドアロンテスト完了 ===")
    
    return result


if __name__ == "__main__":
    result = test_grand_cycle_ma_standalone()