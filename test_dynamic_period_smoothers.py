#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
動的期間対応スムーサーのテスト

全てのスムーサーがehlers_unified_dc.pyと連携して動的期間で計算できるかを検証
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from indicators.smoother.unified_smoother import UnifiedSmoother

def create_test_data(length=150):
    """テストデータの生成"""
    np.random.seed(42)
    
    # 複雑な信号（複数周期のサイン波 + トレンド + ノイズ）
    t = np.linspace(0, 8*np.pi, length)
    signal1 = 10 * np.sin(t)  # 長期サイクル
    signal2 = 5 * np.sin(4*t)  # 短期サイクル
    trend = 0.05 * t  # トレンド
    base_price = 100
    
    true_signal = base_price + signal1 + signal2 + trend
    
    # 観測値（ノイズ付き）
    noise = np.random.normal(0, 2, length)
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

def test_dynamic_period_smoothers():
    """動的期間スムーサーのテスト"""
    print("=== 動的期間対応スムーサーのテスト ===")
    
    # テストデータ生成
    df, true_signal = create_test_data(150)
    print(f"テストデータ: {len(df)}ポイント")
    print(f"価格範囲: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    # テストケース - 各スムーサーを固定期間と動的期間で比較
    smoothers_to_test = [
        'laguerre_filter',
        'alma', 
        'hma',
        'frama',
        'super_smoother',
        'zero_lag_ema',
        'ultimate_smoother'
    ]
    
    results = {}
    
    for smoother_type in smoothers_to_test:
        print(f"\n{smoother_type} をテスト中...")
        
        # 固定期間モード
        try:
            smoother_fixed = UnifiedSmoother(
                smoother_type=smoother_type,
                src_type='close',
                period_mode='fixed'
            )
            result_fixed = smoother_fixed.calculate(df)
            
            # 動的期間モード
            smoother_dynamic = UnifiedSmoother(
                smoother_type=smoother_type,
                src_type='close',
                period_mode='dynamic',
                cycle_detector_type='hody_e',
                cycle_part=0.5,
                max_output=50,
                min_output=8
            )
            result_dynamic = smoother_dynamic.calculate(df)
            
            # 統計計算
            valid_mask_fixed = ~np.isnan(result_fixed.values)
            valid_mask_dynamic = ~np.isnan(result_dynamic.values)
            
            if np.any(valid_mask_fixed) and np.any(valid_mask_dynamic):
                # 有効データでの比較
                valid_true_fixed = true_signal[valid_mask_fixed]
                valid_fixed = result_fixed.values[valid_mask_fixed]
                
                valid_true_dynamic = true_signal[valid_mask_dynamic]
                valid_dynamic = result_dynamic.values[valid_mask_dynamic]
                
                # MAE計算
                mae_fixed = np.mean(np.abs(valid_fixed - valid_true_fixed))
                mae_dynamic = np.mean(np.abs(valid_dynamic - valid_true_dynamic))
                
                # 相関係数計算
                corr_fixed = np.corrcoef(valid_fixed, valid_true_fixed)[0, 1]
                corr_dynamic = np.corrcoef(valid_dynamic, valid_true_dynamic)[0, 1]
                
                # ノイズ削減率計算
                raw_values = df['close'].values
                noise_reduction_fixed = (np.std(raw_values[valid_mask_fixed] - valid_true_fixed) - 
                                       np.std(valid_fixed - valid_true_fixed)) / np.std(raw_values[valid_mask_fixed] - valid_true_fixed) * 100
                
                noise_reduction_dynamic = (np.std(raw_values[valid_mask_dynamic] - valid_true_dynamic) - 
                                         np.std(valid_dynamic - valid_true_dynamic)) / np.std(raw_values[valid_mask_dynamic] - valid_true_dynamic) * 100
                
                results[smoother_type] = {
                    'fixed': {
                        'values': result_fixed.values,
                        'mae': mae_fixed,
                        'correlation': corr_fixed,
                        'noise_reduction': noise_reduction_fixed,
                        'valid_count': np.sum(valid_mask_fixed)
                    },
                    'dynamic': {
                        'values': result_dynamic.values,
                        'mae': mae_dynamic,
                        'correlation': corr_dynamic,
                        'noise_reduction': noise_reduction_dynamic,
                        'valid_count': np.sum(valid_mask_dynamic)
                    }
                }
                
                print(f"  ✓ 成功")
                print(f"    固定期間 - 相関: {corr_fixed:.4f}, MAE: {mae_fixed:.4f}, ノイズ削減: {noise_reduction_fixed:.1f}%")
                print(f"    動的期間 - 相関: {corr_dynamic:.4f}, MAE: {mae_dynamic:.4f}, ノイズ削減: {noise_reduction_dynamic:.1f}%")
                print(f"    有効値数 - 固定: {np.sum(valid_mask_fixed)}, 動的: {np.sum(valid_mask_dynamic)}")
                
                # 改善度の計算
                mae_improvement = ((mae_fixed - mae_dynamic) / mae_fixed) * 100
                corr_improvement = ((corr_dynamic - corr_fixed) / corr_fixed) * 100
                print(f"    改善度 - MAE: {mae_improvement:.1f}%, 相関: {corr_improvement:.1f}%")
                
            else:
                print(f"  ✗ エラー: 有効な値が得られませんでした")
                
        except Exception as e:
            print(f"  ✗ エラー: {e}")
    
    # 結果の可視化
    if results:
        print(f"\n=== 結果の比較 ===")
        
        # 各スムーサーの比較チャートを作成
        n_smoothers = len(results)
        fig, axes = plt.subplots(n_smoothers, 1, figsize=(15, 4*n_smoothers))
        if n_smoothers == 1:
            axes = [axes]
        
        fig.suptitle('動的期間vs固定期間スムーサーの比較', fontsize=16)
        
        x_axis = range(len(df))
        
        for i, (smoother_name, result) in enumerate(results.items()):
            ax = axes[i]
            
            # 元データと真の信号
            ax.plot(x_axis, df['close'].values, 'gray', alpha=0.3, label='元データ', linewidth=0.8)
            ax.plot(x_axis, true_signal, 'k--', label='真の信号', linewidth=1.5)
            
            # 固定期間と動的期間の結果
            ax.plot(x_axis, result['fixed']['values'], 'b-', label=f'固定期間 (相関:{result["fixed"]["correlation"]:.3f})', linewidth=1.2)
            ax.plot(x_axis, result['dynamic']['values'], 'r-', label=f'動的期間 (相関:{result["dynamic"]["correlation"]:.3f})', linewidth=1.2)
            
            ax.set_title(f'{smoother_name.upper()} - 固定期間 vs 動的期間')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # ファイル名に現在時刻を追加
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'dynamic_period_smoothers_comparison_{timestamp}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"チャート保存: {filename}")
        
        # 統計表
        print(f"\n{'スムーサー':<20} {'モード':<8} {'相関':<8} {'MAE':<10} {'ノイズ削減%':<12}")
        print("-" * 70)
        for smoother_name, result in results.items():
            for mode_name, mode_result in result.items():
                print(f"{smoother_name:<20} {mode_name:<8} {mode_result['correlation']:<8.3f} {mode_result['mae']:<10.3f} {mode_result['noise_reduction']:<12.1f}")
        
        # 改善度サマリー
        print(f"\n=== 動的期間による改善度 ===")
        print(f"{'スムーサー':<20} {'MAE改善%':<12} {'相関改善%':<12}")
        print("-" * 50)
        for smoother_name, result in results.items():
            mae_improvement = ((result['fixed']['mae'] - result['dynamic']['mae']) / result['fixed']['mae']) * 100
            corr_improvement = ((result['dynamic']['correlation'] - result['fixed']['correlation']) / result['fixed']['correlation']) * 100
            print(f"{smoother_name:<20} {mae_improvement:<12.1f} {corr_improvement:<12.1f}")
    
    print("\n=== テスト完了 ===")
    return results

if __name__ == "__main__":
    test_dynamic_period_smoothers()