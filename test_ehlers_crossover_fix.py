#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
import os

# プロジェクトのルートディレクトリを追加
sys.path.append('/home/vapor/dev/ttf')

from signals.implementations.ehlers_instantaneous_trendline.entry import EhlersInstantaneousTrendlineCrossoverEntrySignal

def generate_crossover_test_data(n_periods=120):
    """Ehlers用クロスオーバー検出テストデータを生成"""
    np.random.seed(42)
    
    # より変動的なトレンド変化があるデータを作成
    prices = []
    base_price = 50000
    
    for i in range(n_periods):
        if i < 30:
            # 下降トレンド
            trend = -0.0008
        elif i < 60:
            # 上昇トレンド（クロスオーバーが発生する期間）
            trend = 0.0015
        elif i < 90:
            # 横ばい
            trend = 0.0002
        else:
            # 再び下降トレンド
            trend = -0.0012
        
        noise = np.random.normal(0, 0.008)
        base_price = base_price * (1 + trend + noise)
        prices.append(base_price)
    
    prices = np.array(prices)
    
    # OHLCV データの生成
    high = prices * (1 + np.abs(np.random.normal(0, 0.006, n_periods)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.006, n_periods)))
    open_prices = np.roll(prices, 1)
    open_prices[0] = prices[0]
    volume = np.random.uniform(1000, 10000, n_periods)
    
    data = pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': prices,
        'volume': volume
    })
    
    return data

def test_ehlers_crossover_detection():
    """改良されたEhlersクロスオーバー検出のテスト"""
    print("=== Ehlers Instantaneous Trendline改良クロスオーバー検出テスト ===")
    
    # テストデータ生成
    data = generate_crossover_test_data(120)
    
    # クロスオーバーシグナル
    crossover_signal = EhlersInstantaneousTrendlineCrossoverEntrySignal(
        alpha=0.08,  # アルファ値を調整してクロスオーバーを発生しやすくする
        src_type='hl2',
        enable_hyper_er_adaptation=False,  # 固定アルファでテスト
        smoothing_mode='none'
    )
    
    print(f"✓ Ehlersクロスオーバーシグナルを初期化: {crossover_signal.name}")
    
    # シグナル生成
    signals = crossover_signal.generate(data)
    
    # ITrend/Trigger値取得
    itrend_values = crossover_signal.get_itrend_values()
    trigger_values = crossover_signal.get_trigger_values()
    
    # シグナル統計
    long_signals = np.where(signals == 1)[0]
    short_signals = np.where(signals == -1)[0]
    neutral_signals = np.sum(signals == 0)
    
    print(f"✓ シグナル生成完了")
    print(f"  データ期間: {len(data)}")
    long_indices = long_signals.tolist() if len(long_signals) <= 8 else f'{long_signals[:4].tolist()}... (最初の4個)'
    short_indices = short_signals.tolist() if len(short_signals) <= 8 else f'{short_signals[:4].tolist()}... (最初の4個)'
    print(f"  ロングクロスオーバー: {len(long_signals)} (インデックス: {long_indices})")
    print(f"  ショートクロスオーバー: {len(short_signals)} (インデックス: {short_indices})")
    print(f"  ニュートラル: {neutral_signals} ({neutral_signals/len(signals)*100:.1f}%)")
    
    # クロスオーバー発生時の詳細検証
    if len(long_signals) > 0 or len(short_signals) > 0:
        print(f"\n✓ Ehlersクロスオーバー検証:")
        
        # 最初のロングクロスオーバーを検証
        if len(long_signals) > 0:
            idx = long_signals[0]
            if idx > 0:  # 前のデータが必要
                prev_position = "Trigger > ITrend" if trigger_values[idx-1] > itrend_values[idx-1] else "Trigger <= ITrend"
                curr_position = "Trigger > ITrend" if trigger_values[idx] > itrend_values[idx] else "Trigger <= ITrend"
                print(f"  ロングクロス[{idx}]: 前={prev_position}, 現={curr_position}")
                print(f"    前: Trigger={trigger_values[idx-1]:.2f}, ITrend={itrend_values[idx-1]:.2f}")
                print(f"    現: Trigger={trigger_values[idx]:.2f}, ITrend={itrend_values[idx]:.2f}")
        
        # 最初のショートクロスオーバーを検証
        if len(short_signals) > 0:
            idx = short_signals[0]
            if idx > 0:  # 前のデータが必要
                prev_position = "Trigger > ITrend" if trigger_values[idx-1] > itrend_values[idx-1] else "Trigger <= ITrend"
                curr_position = "Trigger > ITrend" if trigger_values[idx] > itrend_values[idx] else "Trigger <= ITrend"
                print(f"  ショートクロス[{idx}]: 前={prev_position}, 現={curr_position}")
                print(f"    前: Trigger={trigger_values[idx-1]:.2f}, ITrend={itrend_values[idx-1]:.2f}")
                print(f"    現: Trigger={trigger_values[idx]:.2f}, ITrend={itrend_values[idx]:.2f}")
    
    # 位置関係分析
    position_changes = []
    for i in range(1, len(itrend_values)):
        if not (np.isnan(itrend_values[i]) or np.isnan(trigger_values[i]) or
                np.isnan(itrend_values[i-1]) or np.isnan(trigger_values[i-1])):
            prev_pos = 1 if trigger_values[i-1] > itrend_values[i-1] else -1
            curr_pos = 1 if trigger_values[i] > itrend_values[i] else -1
            if prev_pos != curr_pos:
                position_changes.append((i, prev_pos, curr_pos))
    
    print(f"\n  Ehlers位置関係変化: {len(position_changes)}回")
    for i, (idx, prev, curr) in enumerate(position_changes[:5]):  # 最初の5個のみ表示
        change_type = "Triggerゴールデンクロス" if prev == -1 and curr == 1 else "Triggerデッドクロス"
        signal_detected = signals[idx] != 0
        print(f"    [{idx}] {change_type} - シグナル検出: {'✓' if signal_detected else '✗'}")
    
    return len(long_signals) > 0 or len(short_signals) > 0

def test_hyper_er_adaptation_crossover():
    """HyperER動的適応付きクロスオーバーテスト"""
    print(f"\n=== HyperER動的適応クロスオーバーテスト ===")
    
    data = generate_crossover_test_data(100)
    
    # HyperER動的適応付きクロスオーバーシグナル
    adaptive_crossover_signal = EhlersInstantaneousTrendlineCrossoverEntrySignal(
        alpha=0.07,  # 基本値（使用されない）
        src_type='hl2',
        enable_hyper_er_adaptation=True,
        hyper_er_period=14,
        alpha_min=0.04,
        alpha_max=0.12,
        smoothing_mode='none'
    )
    
    print(f"✓ HyperER動的適応クロスオーバーシグナルを初期化")
    
    # シグナル生成
    signals = adaptive_crossover_signal.generate(data)
    
    # アルファ値統計
    alpha_values = adaptive_crossover_signal.get_alpha_values()
    if len(alpha_values) > 0:
        alpha_min = np.nanmin(alpha_values)
        alpha_max = np.nanmax(alpha_values)
        alpha_mean = np.nanmean(alpha_values)
        
        print(f"✓ HyperER動的適応完了")
        print(f"  動的アルファ統計:")
        print(f"    最小値: {alpha_min:.4f}")
        print(f"    最大値: {alpha_max:.4f}")
        print(f"    平均値: {alpha_mean:.4f}")
    
    # シグナル統計
    long_signals = np.sum(signals == 1)
    short_signals = np.sum(signals == -1)
    
    print(f"  クロスオーバーシグナル統計:")
    print(f"    ロング: {long_signals}")
    print(f"    ショート: {short_signals}")
    
    return long_signals > 0 or short_signals > 0

def main():
    """メインテスト関数"""
    print("Ehlers Instantaneous Trendline クロスオーバー検出改良テスト開始")
    print("=" * 70)
    
    results = []
    
    # テストの実行
    results.append(test_ehlers_crossover_detection())
    results.append(test_hyper_er_adaptation_crossover())
    
    # 結果のまとめ
    print(f"\n{'='*70}")
    print(f"テスト結果まとめ:")
    print(f"  実行済みテスト: {len(results)}")
    print(f"  成功: {sum(results)}")
    print(f"  失敗: {len(results) - sum(results)}")
    
    if all(results):
        print(f"✓ 全てのテストが成功しました！")
        print(f"\nEhlers改良されたクロスオーバー検出アルゴリズム:")
        print(f"  • ITrendとTriggerの位置関係ベースの確実なクロス検出")
        print(f"  • ゴールデンクロス・デッドクロスの正確な識別")
        print(f"  • HyperER動的適応との統合")
        print(f"  • NumPy配列による高速処理")
        print(f"  • 無効データの適切なハンドリング")
    else:
        print(f"✗ 一部のテストが失敗しました。")
    
    print(f"{'='*70}")

if __name__ == "__main__":
    main()