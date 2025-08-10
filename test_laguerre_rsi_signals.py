#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ラゲールRSIシグナルのテスト統合スクリプト
"""

import numpy as np
import pandas as pd
import sys
import os

# パスの追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from signals.implementations.laguerre_rsi.trend_follow_entry import LaguerreRSITrendFollowEntrySignal
from signals.implementations.laguerre_rsi.trend_reversal_entry import LaguerreRSITrendReversalEntrySignal


def generate_test_data(length=200, data_type="mixed"):
    """
    テストデータを生成する
    
    Args:
        length: データポイント数
        data_type: データタイプ（"trend", "range", "mixed"）
        
    Returns:
        pd.DataFrame: OHLCV データ
    """
    np.random.seed(42)
    base_price = 100.0
    prices = [base_price]
    
    if data_type == "trend":
        # トレンドデータ: 上昇 → レンジ → 下降
        for i in range(1, length):
            if i < 60:  # 上昇トレンド
                change = 0.005 + np.random.normal(0, 0.01)
            elif i < 140:  # レンジ相場
                change = np.random.normal(0, 0.008)
            else:  # 下降トレンド
                change = -0.005 + np.random.normal(0, 0.01)
            
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
            
    elif data_type == "range":
        # レンジ相場データ: 振動の大きい横ばい
        for i in range(1, length):
            cycle_pos = (i / 20.0) * 2 * np.pi
            cycle_component = np.sin(cycle_pos) * 0.02  # 2%の周期変動
            random_component = np.random.normal(0, 0.015)  # 1.5%のランダム変動
            
            change = cycle_component + random_component
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
            
    else:  # mixed
        # 混合データ: トレンドとレンジが混在
        for i in range(1, length):
            if i < 50:  # 上昇トレンド
                change = 0.003 + np.random.normal(0, 0.008)
            elif i < 100:  # レンジ相場
                change = np.random.normal(0, 0.010)
            elif i < 150:  # 下降トレンド
                change = -0.003 + np.random.normal(0, 0.008)
            else:  # 再びレンジ相場
                change = np.random.normal(0, 0.010)
            
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
    
    # OHLC データの生成
    data = []
    for i, close in enumerate(prices):
        daily_range = abs(np.random.normal(0, close * 0.008))
        
        high = close + daily_range * np.random.uniform(0.3, 1.0)
        low = close - daily_range * np.random.uniform(0.3, 1.0)
        
        if i == 0:
            open_price = close
        else:
            gap = np.random.normal(0, close * 0.003)
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


def test_trend_follow_signals(df):
    """トレンドフォローシグナルのテスト"""
    print("\n" + "="*60)
    print("トレンドフォローエントリーシグナルのテスト")
    print("="*60)
    
    # ポジション維持モード（パインスクリプト仕様）
    print("\n--- ポジション維持モード ---")
    signal_pos = LaguerreRSITrendFollowEntrySignal(
        gamma=0.5,
        src_type='close',
        buy_band=0.8,
        sell_band=0.2,
        position_mode=True
    )
    
    signals_pos = signal_pos.generate(df)
    lrsi_values = signal_pos.get_lrsi_values()
    
    long_signals = np.sum(signals_pos == 1)
    short_signals = np.sum(signals_pos == -1)
    no_signals = np.sum(signals_pos == 0)
    
    print(f"データポイント: {len(df)}")
    print(f"ロングシグナル: {long_signals} ({long_signals/len(df)*100:.1f}%)")
    print(f"ショートシグナル: {short_signals} ({short_signals/len(df)*100:.1f}%)")
    print(f"シグナルなし: {no_signals} ({no_signals/len(df)*100:.1f}%)")
    print(f"平均ラゲールRSI: {np.nanmean(lrsi_values):.4f}")
    print(f"RSI範囲: {np.nanmin(lrsi_values):.4f} - {np.nanmax(lrsi_values):.4f}")
    
    # クロスオーバーモード
    print("\n--- クロスオーバーモード ---")
    signal_cross = LaguerreRSITrendFollowEntrySignal(
        gamma=0.5,
        src_type='close',
        buy_band=0.8,
        sell_band=0.2,
        position_mode=False
    )
    
    signals_cross = signal_cross.generate(df)
    
    long_cross = np.sum(signals_cross == 1)
    short_cross = np.sum(signals_cross == -1)
    no_cross = np.sum(signals_cross == 0)
    
    print(f"ロングシグナル: {long_cross}")
    print(f"ショートシグナル: {short_cross}")
    print(f"シグナルなし: {no_cross}")
    
    # 閾値別の統計
    print("\n--- 閾値別統計 ---")
    overbought_count = np.sum(lrsi_values > 0.8)
    oversold_count = np.sum(lrsi_values < 0.2)
    neutral_count = np.sum((lrsi_values >= 0.2) & (lrsi_values <= 0.8))
    
    print(f"買われすぎ (>0.8): {overbought_count} ({overbought_count/len(lrsi_values)*100:.1f}%)")
    print(f"売られすぎ (<0.2): {oversold_count} ({oversold_count/len(lrsi_values)*100:.1f}%)")
    print(f"中立 (0.2-0.8): {neutral_count} ({neutral_count/len(lrsi_values)*100:.1f}%)")
    
    return signals_pos, lrsi_values


def test_trend_reversal_signals(df):
    """トレンドリバーサルシグナルのテスト"""
    print("\n" + "="*60)
    print("トレンドリバーサルエントリーシグナルのテスト")
    print("="*60)
    
    # ポジション維持モード（リバーサル）
    print("\n--- ポジション維持モード（リバーサル） ---")
    signal_pos = LaguerreRSITrendReversalEntrySignal(
        gamma=0.5,
        src_type='close',
        buy_band=0.2,  # 売られすぎでロング
        sell_band=0.8,  # 買われすぎでショート
        position_mode=True
    )
    
    signals_pos = signal_pos.generate(df)
    lrsi_values = signal_pos.get_lrsi_values()
    
    long_signals = np.sum(signals_pos == 1)
    short_signals = np.sum(signals_pos == -1)
    no_signals = np.sum(signals_pos == 0)
    
    print(f"データポイント: {len(df)}")
    print(f"ロングシグナル: {long_signals} ({long_signals/len(df)*100:.1f}%)")
    print(f"ショートシグナル: {short_signals} ({short_signals/len(df)*100:.1f}%)")
    print(f"シグナルなし: {no_signals} ({no_signals/len(df)*100:.1f}%)")
    print(f"平均ラゲールRSI: {np.nanmean(lrsi_values):.4f}")
    
    # クロスオーバーモード（リバーサル）
    print("\n--- クロスオーバーモード（リバーサル） ---")
    signal_cross = LaguerreRSITrendReversalEntrySignal(
        gamma=0.5,
        src_type='close',
        buy_band=0.2,
        sell_band=0.8,
        position_mode=False
    )
    
    signals_cross = signal_cross.generate(df)
    
    long_cross = np.sum(signals_cross == 1)
    short_cross = np.sum(signals_cross == -1)
    no_cross = np.sum(signals_cross == 0)
    
    print(f"ロングシグナル: {long_cross}")
    print(f"ショートシグナル: {short_cross}")
    print(f"シグナルなし: {no_cross}")
    
    # 平均回帰モード
    print("\n--- 平均回帰モード ---")
    signal_meanrev = LaguerreRSITrendReversalEntrySignal(
        gamma=0.5,
        src_type='close',
        buy_band=0.3,
        sell_band=0.7,
        position_mode=True,
        mean_reversion_mode=True
    )
    
    signals_meanrev = signal_meanrev.generate(df)
    
    long_meanrev = np.sum(signals_meanrev == 1)
    short_meanrev = np.sum(signals_meanrev == -1)
    no_meanrev = np.sum(signals_meanrev == 0)
    
    print(f"ロングシグナル: {long_meanrev}")
    print(f"ショートシグナル: {short_meanrev}")
    print(f"シグナルなし: {no_meanrev}")
    
    # シグナル品質評価
    print("\n--- シグナル品質評価 ---")
    
    # ロングシグナル時のRSI分析
    long_positions = signals_pos == 1
    if np.sum(long_positions) > 0:
        long_rsi_avg = np.nanmean(lrsi_values[long_positions])
        print(f"ロングシグナル時の平均RSI: {long_rsi_avg:.4f}")
    
    # ショートシグナル時のRSI分析  
    short_positions = signals_pos == -1
    if np.sum(short_positions) > 0:
        short_rsi_avg = np.nanmean(lrsi_values[short_positions])
        print(f"ショートシグナル時の平均RSI: {short_rsi_avg:.4f}")
    
    return signals_pos, lrsi_values


def compare_signals(trend_signals, reversal_signals):
    """トレンドフォローvsリバーサルの比較"""
    print("\n" + "="*60)
    print("トレンドフォロー vs リバーサルシグナル比較")
    print("="*60)
    
    # シグナル一致率
    agreement = np.sum(trend_signals == reversal_signals)
    disagreement = np.sum(trend_signals != reversal_signals)
    opposite = np.sum(trend_signals == -reversal_signals)
    
    print(f"シグナル一致: {agreement} ({agreement/len(trend_signals)*100:.1f}%)")
    print(f"シグナル不一致: {disagreement} ({disagreement/len(trend_signals)*100:.1f}%)")
    print(f"反対シグナル: {opposite} ({opposite/len(trend_signals)*100:.1f}%)")
    
    # 同時発生分析
    both_long = np.sum((trend_signals == 1) & (reversal_signals == 1))
    both_short = np.sum((trend_signals == -1) & (reversal_signals == -1))
    trend_long_rev_short = np.sum((trend_signals == 1) & (reversal_signals == -1))
    trend_short_rev_long = np.sum((trend_signals == -1) & (reversal_signals == 1))
    
    print(f"両方ロング: {both_long}")
    print(f"両方ショート: {both_short}")
    print(f"トレンドロング・リバーサルショート: {trend_long_rev_short}")
    print(f"トレンドショート・リバーサルロング: {trend_short_rev_long}")


def test_parameter_sensitivity():
    """パラメータ感度のテスト"""
    print("\n" + "="*60)
    print("パラメータ感度テスト")
    print("="*60)
    
    # テストデータ
    df = generate_test_data(length=200, data_type="mixed")
    
    # ガンマ値の感度テスト
    print("\n--- ガンマ値感度テスト ---")
    gamma_values = [0.2, 0.5, 0.8]
    
    for gamma in gamma_values:
        signal = LaguerreRSITrendFollowEntrySignal(
            gamma=gamma,
            src_type='close',
            buy_band=0.8,
            sell_band=0.2,
            position_mode=True
        )
        
        signals = signal.generate(df)
        lrsi_values = signal.get_lrsi_values()
        
        long_count = np.sum(signals == 1)
        short_count = np.sum(signals == -1)
        avg_rsi = np.nanmean(lrsi_values)
        rsi_std = np.nanstd(lrsi_values)
        
        print(f"gamma={gamma}: Long={long_count}, Short={short_count}, "
              f"平均RSI={avg_rsi:.4f}, RSI標準偏差={rsi_std:.4f}")
    
    # 閾値感度テスト
    print("\n--- 閾値感度テスト ---")
    threshold_pairs = [(0.7, 0.3), (0.8, 0.2), (0.9, 0.1)]
    
    for buy_band, sell_band in threshold_pairs:
        signal = LaguerreRSITrendFollowEntrySignal(
            gamma=0.5,
            src_type='close',
            buy_band=buy_band,
            sell_band=sell_band,
            position_mode=True
        )
        
        signals = signal.generate(df)
        long_count = np.sum(signals == 1)
        short_count = np.sum(signals == -1)
        
        print(f"閾値({buy_band}, {sell_band}): Long={long_count}, Short={short_count}")


def main():
    """メイン関数"""
    print("=" * 80)
    print("ラゲールRSIエントリーシグナル総合テスト")
    print("=" * 80)
    
    # テストデータの種類別テスト
    data_types = ["trend", "range", "mixed"]
    
    for data_type in data_types:
        print(f"\n\n{'='*20} {data_type.upper()}データテスト {'='*20}")
        
        df = generate_test_data(length=200, data_type=data_type)
        print(f"価格範囲: {df['close'].min():.2f} - {df['close'].max():.2f}")
        print(f"価格標準偏差: {df['close'].std():.2f}")
        
        # トレンドフォローシグナルのテスト
        trend_signals, lrsi_trend = test_trend_follow_signals(df)
        
        # トレンドリバーサルシグナルのテスト
        reversal_signals, lrsi_reversal = test_trend_reversal_signals(df)
        
        # シグナル比較
        compare_signals(trend_signals, reversal_signals)
    
    # パラメータ感度テスト
    test_parameter_sensitivity()
    
    print("\n" + "="*80)
    print("テスト完了")
    print("="*80)


if __name__ == "__main__":
    main()