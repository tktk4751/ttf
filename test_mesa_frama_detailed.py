#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from indicators.mesa_frama import MESA_FRAMA
from indicators.mama import MAMA
from indicators.smoother.frama import FRAMA

def generate_test_data(length=500):
    """テストデータを生成"""
    np.random.seed(42)
    base_price = 100.0
    
    # 複数の市場状況を含むデータを生成
    prices = [base_price]
    for i in range(1, length):
        if i < length // 5:  # トレンド相場
            change = 0.002 + np.random.normal(0, 0.01)
        elif i < 2 * length // 5:  # レンジ相場
            change = np.random.normal(0, 0.008)
        elif i < 3 * length // 5:  # 強いトレンド相場
            change = 0.004 + np.random.normal(0, 0.015)
        elif i < 4 * length // 5:  # ボラティリティの高いレンジ相場
            change = np.random.normal(0, 0.02)
        else:  # 安定したトレンド相場
            change = 0.001 + np.random.normal(0, 0.005)
        
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

def test_mesa_frama_detailed():
    """詳細なMESA_FRAMAテスト"""
    print("=== MESA_FRAMA 詳細テスト ===")
    
    # テストデータ生成
    df = generate_test_data(500)
    print(f"テストデータ: {len(df)}ポイント")
    print(f"価格範囲: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    # 1. 基本的なMESA_FRAMAテスト
    print("\n1. 基本的なMESA_FRAMAテスト")
    mesa_frama = MESA_FRAMA(
        base_period=16,
        src_type='hl2',
        fc=1,
        sc=198,
        mesa_fast_limit=0.5,
        mesa_slow_limit=0.05,
        use_zero_lag=False
    )
    
    result = mesa_frama.calculate(df)
    
    print(f"  MESA_FRAMA値: 有効数 {np.sum(~np.isnan(result.values))}, 平均 {np.nanmean(result.values):.4f}")
    print(f"  フラクタル次元: 平均 {np.nanmean(result.fractal_dimension):.4f}, 範囲 [{np.nanmin(result.fractal_dimension):.4f}, {np.nanmax(result.fractal_dimension):.4f}]")
    print(f"  動的期間: 平均 {np.nanmean(result.dynamic_periods):.2f}, 範囲 [{np.nanmin(result.dynamic_periods):.2f}, {np.nanmax(result.dynamic_periods):.2f}]")
    print(f"  MESAフェーズ: 平均 {np.nanmean(result.mesa_phase):.2f}, 範囲 [{np.nanmin(result.mesa_phase):.2f}, {np.nanmax(result.mesa_phase):.2f}]")
    
    # 2. ゼロラグ処理版のテスト
    print("\n2. ゼロラグ処理版テスト")
    mesa_frama_zl = MESA_FRAMA(
        base_period=16,
        src_type='hl2',
        use_zero_lag=True
    )
    
    result_zl = mesa_frama_zl.calculate(df)
    
    # 基本版とゼロラグ版の比較
    valid_mask = ~np.isnan(result.values) & ~np.isnan(result_zl.values)
    if np.sum(valid_mask) > 0:
        correlation = np.corrcoef(result.values[valid_mask], result_zl.values[valid_mask])[0, 1]
        print(f"  基本版とゼロラグ版の相関: {correlation:.4f}")
        
        # 応答性の比較（価格変化に対する反応速度）
        price_changes = np.diff(df['close'].to_numpy())
        mesa_changes = np.diff(result.values[~np.isnan(result.values)])
        mesa_zl_changes = np.diff(result_zl.values[~np.isnan(result_zl.values)])
        
        if len(mesa_changes) > 0 and len(mesa_zl_changes) > 0:
            print(f"  基本版の平均変化量: {np.nanmean(np.abs(mesa_changes)):.6f}")
            print(f"  ゼロラグ版の平均変化量: {np.nanmean(np.abs(mesa_zl_changes)):.6f}")
    
    # 3. 従来のFRAMAとの比較
    print("\n3. 従来のFRAMAとの比較")
    try:
        # 固定期間FRAMA
        frama_fixed = FRAMA(period=16, src_type='hl2')
        frama_result = frama_fixed.calculate(df)
        
        # MESA_FRAMAと固定期間FRAMAの比較
        valid_mask = ~np.isnan(result.values) & ~np.isnan(frama_result.values)
        if np.sum(valid_mask) > 0:
            correlation = np.corrcoef(result.values[valid_mask], frama_result.values[valid_mask])[0, 1]
            print(f"  MESA_FRAMAと固定期間FRAMAの相関: {correlation:.4f}")
            
            # 適応性の比較
            mesa_std = np.nanstd(result.values[valid_mask])
            frama_std = np.nanstd(frama_result.values[valid_mask])
            print(f"  MESA_FRAMA標準偏差: {mesa_std:.4f}")
            print(f"  固定期間FRAMA標準偏差: {frama_std:.4f}")
    except Exception as e:
        print(f"  FRAMAとの比較でエラー: {e}")
    
    # 4. MAMAとの期間比較
    print("\n4. MAMAとの期間比較")
    try:
        mama = MAMA(fast_limit=0.5, slow_limit=0.05, src_type='hl2')
        mama_result = mama.calculate(df)
        
        # MESA期間とMAMA期間の比較
        mama_periods = mama_result.period_values
        mesa_periods = result.dynamic_periods
        
        valid_mask = ~np.isnan(mama_periods) & ~np.isnan(mesa_periods)
        if np.sum(valid_mask) > 0:
            correlation = np.corrcoef(mama_periods[valid_mask], mesa_periods[valid_mask])[0, 1]
            print(f"  MESA期間とMAMA期間の相関: {correlation:.4f}")
            print(f"  MAMA平均期間: {np.nanmean(mama_periods):.2f}")
            print(f"  MESA平均期間: {np.nanmean(mesa_periods):.2f}")
    except Exception as e:
        print(f"  MAMAとの比較でエラー: {e}")
    
    # 5. パフォーマンステスト
    print("\n5. パフォーマンステスト")
    import time
    
    # 複数回実行して平均時間を測定
    times = []
    for _ in range(10):
        start_time = time.time()
        mesa_frama.calculate(df)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    print(f"  平均計算時間: {avg_time:.4f}秒 (10回平均)")
    print(f"  1ポイント当たり: {avg_time / len(df) * 1000:.4f}ms")
    
    # 6. 異なるパラメータでのテスト
    print("\n6. 異なるパラメータでのテスト")
    
    # 高速応答版
    mesa_fast = MESA_FRAMA(
        base_period=8,
        mesa_fast_limit=0.8,
        mesa_slow_limit=0.1,
        src_type='hl2'
    )
    result_fast = mesa_fast.calculate(df)
    
    # 低速応答版
    mesa_slow = MESA_FRAMA(
        base_period=32,
        mesa_fast_limit=0.3,
        mesa_slow_limit=0.02,
        src_type='hl2'
    )
    result_slow = mesa_slow.calculate(df)
    
    print(f"  高速版平均期間: {np.nanmean(result_fast.dynamic_periods):.2f}")
    print(f"  低速版平均期間: {np.nanmean(result_slow.dynamic_periods):.2f}")
    print(f"  基本版平均期間: {np.nanmean(result.dynamic_periods):.2f}")
    
    # 7. エラーハンドリングテスト
    print("\n7. エラーハンドリングテスト")
    
    # 短すぎるデータ
    short_df = df.head(5)
    result_short = mesa_frama.calculate(short_df)
    print(f"  短いデータ（5ポイント）: 有効値数 {np.sum(~np.isnan(result_short.values))}")
    
    # 空のデータ
    empty_df = df.head(0)
    result_empty = mesa_frama.calculate(empty_df)
    print(f"  空のデータ: 結果配列長 {len(result_empty.values)}")
    
    print("\n=== テスト完了 ===")
    
    return {
        'basic_result': result,
        'zero_lag_result': result_zl,
        'test_data': df
    }

if __name__ == "__main__":
    results = test_mesa_frama_detailed()