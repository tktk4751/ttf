#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
カルマンフィルター統合スムーサーのテスト

修正された処理フローのテスト:
価格データ → カルマンフィルター（オプション） → スムーサー
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from indicators.smoother.unified_smoother import UnifiedSmoother

def create_test_data(length=100):
    """テストデータの生成"""
    np.random.seed(42)
    
    # 真の信号（サイン波 + トレンド）
    t = np.linspace(0, 4*np.pi, length)
    true_signal = 100 + 10 * np.sin(t) + 0.1 * t
    
    # 観測値（ノイズ付き）
    noise = np.random.normal(0, 3, length)
    observations = true_signal + noise
    
    # OHLC データの生成
    data = []
    for i, close in enumerate(observations):
        high = close + abs(np.random.normal(0, 1))
        low = close - abs(np.random.normal(0, 1))
        open_price = observations[i-1] if i > 0 else close
        
        data.append({
            'open': open_price,
            'high': max(high, open_price, close),
            'low': min(low, open_price, close),
            'close': close,
            'volume': np.random.uniform(1000, 5000)
        })
    
    df = pd.DataFrame(data)
    start_date = datetime.now() - timedelta(days=length)
    df.index = [start_date + timedelta(hours=i) for i in range(length)]
    
    return df, true_signal

def test_kalman_integration():
    """カルマン統合テスト"""
    print("=== カルマン統合スムーサーのテスト ===")
    
    # テストデータ生成
    df, true_signal = create_test_data(100)
    print(f"テストデータ: {len(df)}ポイント")
    print(f"価格範囲: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    # テストケース
    test_cases = [
        # カルマンなし
        {
            'name': 'ALMA（カルマンなし）',
            'smoother_type': 'alma',
            'enable_kalman': False,
            'length': 9,
            'offset': 0.85,
            'sigma': 6.0
        },
        # カルマンあり（Simple）
        {
            'name': 'ALMA + Simple Kalman',
            'smoother_type': 'alma',
            'enable_kalman': True,
            'kalman_type': 'simple',
            'length': 9,
            'offset': 0.85,
            'sigma': 6.0,
            'kalman_R': 0.1,
            'kalman_Q': 0.01
        },
        # HMAカルマンなし
        {
            'name': 'HMA（カルマンなし）',
            'smoother_type': 'hma',
            'enable_kalman': False,
            'length': 14
        },
        # HMAカルマンあり
        {
            'name': 'HMA + Simple Kalman',
            'smoother_type': 'hma',
            'enable_kalman': True,
            'kalman_type': 'simple',
            'length': 14,
            'kalman_R': 0.1,
            'kalman_Q': 0.01
        }
    ]
    
    results = {}
    
    for case in test_cases:
        print(f"\n{case['name']} をテスト中...")
        
        try:
            # UnifiedSmootherインスタンス作成
            smoother = UnifiedSmoother(
                smoother_type=case['smoother_type'],
                src_type='close',
                enable_kalman=case['enable_kalman'],
                kalman_type=case.get('kalman_type', 'simple'),
                **{k: v for k, v in case.items() if k not in ['name', 'smoother_type', 'enable_kalman', 'kalman_type']}
            )
            
            # 計算実行
            result = smoother.calculate(df)
            
            # 統計計算
            valid_mask = ~np.isnan(result.values)
            if np.any(valid_mask):
                valid_smoothed = result.values[valid_mask]
                valid_true = true_signal[valid_mask]
                valid_raw = result.raw_values[valid_mask]
                
                mae_vs_true = np.mean(np.abs(valid_smoothed - valid_true))
                mae_vs_raw = np.mean(np.abs(valid_smoothed - valid_raw))
                correlation = np.corrcoef(valid_smoothed, valid_true)[0, 1]
                
                noise_reduction = (np.std(valid_raw - valid_true) - np.std(valid_smoothed - valid_true)) / np.std(valid_raw - valid_true) * 100
                
                results[case['name']] = {
                    'values': result.values,
                    'kalman_filtered': result.kalman_filtered_values,
                    'mae_vs_true': mae_vs_true,
                    'mae_vs_raw': mae_vs_raw,
                    'correlation': correlation,
                    'noise_reduction': noise_reduction,
                    'valid_count': np.sum(valid_mask),
                    'kalman_type': result.kalman_type,
                    'additional_data': result.additional_data
                }
                
                print(f"  ✓ 成功")
                print(f"    真の信号との相関: {correlation:.4f}")
                print(f"    真の信号とのMAE: {mae_vs_true:.4f}")
                print(f"    元データとのMAE: {mae_vs_raw:.4f}")
                print(f"    ノイズ削減率: {noise_reduction:.1f}%")
                print(f"    有効値数: {np.sum(valid_mask)}/{len(df)}")
                print(f"    カルマンフィルター値: {'あり' if result.kalman_filtered_values is not None else 'なし'}")
                print(f"    カルマンタイプ: {result.kalman_type}")
                print(f"    追加データ: {list(result.additional_data.keys()) if result.additional_data else 'なし'}")
                
            else:
                print(f"  ✗ エラー: 有効な値が得られませんでした")
                
        except Exception as e:
            print(f"  ✗ エラー: {e}")
    
    # 結果の可視化
    if results:
        print(f"\n=== 結果の比較 ===")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('カルマン統合スムーサーの比較', fontsize=16)
        
        x_axis = range(len(df))
        
        # ALMA比較
        axes[0, 0].plot(x_axis, df['close'].values, 'gray', alpha=0.5, label='元データ')
        axes[0, 0].plot(x_axis, true_signal, 'k--', label='真の信号')
        
        if 'ALMA（カルマンなし）' in results:
            axes[0, 0].plot(x_axis, results['ALMA（カルマンなし）']['values'], 'b-', label='ALMA単独')
        if 'ALMA + Simple Kalman' in results:
            axes[0, 0].plot(x_axis, results['ALMA + Simple Kalman']['values'], 'r-', label='ALMA+Kalman')
        
        axes[0, 0].set_title('ALMA比較')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # HMA比較
        axes[0, 1].plot(x_axis, df['close'].values, 'gray', alpha=0.5, label='元データ')
        axes[0, 1].plot(x_axis, true_signal, 'k--', label='真の信号')
        
        if 'HMA（カルマンなし）' in results:
            axes[0, 1].plot(x_axis, results['HMA（カルマンなし）']['values'], 'b-', label='HMA単独')
        if 'HMA + Simple Kalman' in results:
            axes[0, 1].plot(x_axis, results['HMA + Simple Kalman']['values'], 'r-', label='HMA+Kalman')
        
        axes[0, 1].set_title('HMA比較')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 相関係数比較
        names = list(results.keys())
        correlations = [results[name]['correlation'] for name in names]
        
        axes[1, 0].bar(range(len(names)), correlations)
        axes[1, 0].set_title('真の信号との相関係数')
        axes[1, 0].set_xticks(range(len(names)))
        axes[1, 0].set_xticklabels(names, rotation=45, ha='right')
        axes[1, 0].grid(True, axis='y')
        
        # ノイズ削減率比較
        noise_reductions = [results[name]['noise_reduction'] for name in names]
        
        axes[1, 1].bar(range(len(names)), noise_reductions)
        axes[1, 1].set_title('ノイズ削減率 (%)')
        axes[1, 1].set_xticks(range(len(names)))
        axes[1, 1].set_xticklabels(names, rotation=45, ha='right')
        axes[1, 1].grid(True, axis='y')
        
        plt.tight_layout()
        plt.savefig('kalman_smoother_integration_test.png', dpi=150, bbox_inches='tight')
        print(f"チャート保存: kalman_smoother_integration_test.png")
        
        # 統計表
        print(f"\n{'スムーサー':<25} {'相関':<8} {'MAE_真':<10} {'MAE_元':<10} {'ノイズ削減%':<12}")
        print("-" * 75)
        for name in names:
            r = results[name]
            print(f"{name:<25} {r['correlation']:<8.3f} {r['mae_vs_true']:<10.3f} {r['mae_vs_raw']:<10.3f} {r['noise_reduction']:<12.1f}")
    
    print("\n=== テスト完了 ===")
    return results

if __name__ == "__main__":
    test_kalman_integration()