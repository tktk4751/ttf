#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
量子適応カルマンフィルターと他のカルマンフィルターの比較テスト
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from indicators.kalman.quantum_adaptive_kalman import QuantumAdaptiveKalman
from indicators.kalman.adaptive_kalman import AdaptiveKalman


def generate_test_data(length: int = 300) -> pd.DataFrame:
    """複雑なテストデータを生成"""
    np.random.seed(42)
    t = np.linspace(0, 4*np.pi, length)
    
    # 複雑な価格パターン
    base_signal = 100 + 10 * np.sin(t) + 5 * np.sin(3*t) + 2 * np.sin(7*t)
    trend = 0.05 * t
    regime_change = np.where(t > 2*np.pi, 15, 0)  # 構造変化
    
    # 動的ノイズ（ボラティリティクラスタリング）
    noise = np.zeros(length)
    volatility = 2.0
    for i in range(1, length):
        volatility = 0.95 * volatility + 0.05 * abs(np.random.normal(0, 3))
        noise[i] = np.random.normal(0, volatility)
    
    prices = base_signal + trend + regime_change + noise
    
    # OHLC データの生成
    data = []
    for i, close in enumerate(prices):
        daily_range = abs(np.random.normal(0, 1.5))
        high = close + daily_range * np.random.uniform(0.3, 1.0)
        low = close - daily_range * np.random.uniform(0.3, 1.0)
        open_price = close + np.random.normal(0, 0.5)
        
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
    start_date = datetime.now() - timedelta(days=length)
    df.index = [start_date + timedelta(hours=i) for i in range(length)]
    
    return df


def main():
    print("=== 量子適応カルマンフィルター vs 他のカルマンフィルター ===")
    
    # テストデータ生成
    data = generate_test_data(300)
    print(f"テストデータ: {len(data)}ポイント")
    print(f"価格範囲: {data['close'].min():.2f} - {data['close'].max():.2f}")
    
    # 各フィルターのテスト
    filters = {
        '量子適応カルマン': QuantumAdaptiveKalman(src_type='close'),
        '適応カルマン': AdaptiveKalman(src_type='close', process_noise=1e-5)
    }
    
    results = {}
    print("\nフィルター計算中...")
    
    for name, filter_obj in filters.items():
        try:
            result = filter_obj.calculate(data)
            results[name] = result
            
            if hasattr(result, 'values'):
                values = result.values
            else:
                values = result.filtered_signal
            
            print(f"\n{name}:")
            print(f"  フィルタリング値範囲: {np.nanmin(values):.2f} - {np.nanmax(values):.2f}")
            print(f"  有効値数: {np.sum(~np.isnan(values))}/{len(values)}")
            
            # 統計指標
            original_std = np.std(data['close'])
            filtered_std = np.nanstd(values)
            noise_reduction = (1 - filtered_std / original_std) * 100
            mae = np.nanmean(np.abs(values - data['close'].values))
            
            print(f"  ノイズ削減率: {noise_reduction:.2f}%")
            print(f"  追跡誤差(MAE): {mae:.4f}")
            
        except Exception as e:
            print(f"{name}でエラー: {e}")
            continue
    
    if not results:
        print("計算可能なフィルターがありません")
        return 1
    
    # 結果の可視化
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # 価格とフィルタリング結果
    axes[0].plot(data.index, data['close'], alpha=0.7, label='元の価格', color='blue', linewidth=1)
    
    colors = ['red', 'green']
    for i, (name, result) in enumerate(results.items()):
        if hasattr(result, 'values'):
            values = result.values
        else:
            values = result.filtered_signal
            
        axes[0].plot(data.index, values, label=f'{name}フィルター', 
                    color=colors[i], linewidth=2, alpha=0.8)
    
    axes[0].set_title('カルマンフィルター比較 - 価格フィルタリング')
    axes[0].set_ylabel('価格')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # フィルタリング誤差
    for i, (name, result) in enumerate(results.items()):
        if hasattr(result, 'values'):
            values = result.values
        else:
            values = result.filtered_signal
            
        error = values - data['close'].values
        axes[1].plot(data.index, error, label=f'{name}誤差', 
                    color=colors[i], linewidth=1.5, alpha=0.8)
    
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1].set_title('フィルタリング誤差')
    axes[1].set_ylabel('誤差')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 追加情報（量子コヒーレンスがある場合）
    if '量子適応カルマン' in results:
        quantum_result = results['量子適応カルマン']
        if hasattr(quantum_result, 'quantum_coherence'):
            axes[2].plot(data.index, quantum_result.quantum_coherence, 
                        color='purple', linewidth=2, label='量子コヒーレンス')
        if hasattr(quantum_result, 'kalman_gains'):
            axes[2].plot(data.index, quantum_result.kalman_gains, 
                        color='orange', linewidth=1.5, alpha=0.7, label='カルマンゲイン')
    
    if '適応カルマン' in results:
        adaptive_result = results['適応カルマン']
        if hasattr(adaptive_result, 'adaptive_gain'):
            axes[2].plot(data.index, adaptive_result.adaptive_gain, 
                        color='green', linewidth=1.5, alpha=0.7, label='適応ゲイン')
    
    axes[2].set_title('動的パラメータ')
    axes[2].set_ylabel('値')
    axes[2].set_xlabel('時間')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = "quantum_vs_other_kalman_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nチャートを保存しました: {save_path}")
    
    # 詳細な統計比較
    print("\n=== 詳細統計比較 ===")
    for name, result in results.items():
        if hasattr(result, 'values'):
            values = result.values
        else:
            values = result.filtered_signal
            
        # 統計指標
        original_prices = data['close'].values
        valid_mask = ~np.isnan(values)
        
        if np.any(valid_mask):
            valid_filtered = values[valid_mask]
            valid_original = original_prices[valid_mask]
            
            mae = np.mean(np.abs(valid_filtered - valid_original))
            rmse = np.sqrt(np.mean((valid_filtered - valid_original)**2))
            correlation = np.corrcoef(valid_filtered, valid_original)[0, 1]
            
            # スムージング効果
            original_volatility = np.std(valid_original)
            filtered_volatility = np.std(valid_filtered)
            smoothing_factor = 1 - (filtered_volatility / original_volatility)
            
            print(f"\n{name}:")
            print(f"  平均絶対誤差(MAE): {mae:.4f}")
            print(f"  二乗平均平方根誤差(RMSE): {rmse:.4f}")
            print(f"  相関係数: {correlation:.4f}")
            print(f"  スムージング係数: {smoothing_factor:.4f}")
    
    plt.show()
    
    print("\n=== 比較テスト完了 ===")
    return 0


if __name__ == "__main__":
    exit(main())