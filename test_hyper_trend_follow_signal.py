#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import yaml
import sys
import os

# プロジェクトのルートディレクトリを追加
sys.path.append('/home/vapor/dev/ttf')

from signals.implementations.hyper_trend_follow.hyper_trend_follow_signal import HyperTrendFollowSignal

def generate_test_data(n_periods=500):
    """テスト用のダミーデータを生成"""
    np.random.seed(42)
    
    dates = pd.date_range(start='2023-01-01', periods=n_periods, freq='4H')
    
    # トレンドのあるダミー価格データを生成
    price = 50000.0
    prices = []
    
    for i in range(n_periods):
        # トレンドとランダムウォーク
        trend = 0.0001 * (i - n_periods/2)  # 中間でトレンド転換
        noise = np.random.normal(0, 0.02)
        price = price * (1 + trend + noise)
        prices.append(price)
    
    prices = np.array(prices)
    
    # OHLCV データの生成
    high = prices * (1 + np.abs(np.random.normal(0, 0.01, n_periods)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.01, n_periods)))
    open_prices = np.roll(prices, 1)
    open_prices[0] = prices[0]
    volume = np.random.uniform(1000, 10000, n_periods)
    
    data = pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': prices,
        'volume': volume
    }, index=dates)
    
    return data

def main():
    """HyperTrendFollowSignalのテスト"""
    print("HyperTrendFollowSignal テスト開始")
    print("=" * 60)
    
    # テスト用データの生成
    try:
        data = generate_test_data(500)
        print(f"✓ テスト用データを生成しました: {len(data)} bars")
        print(f"  期間: {data.index[0]} ~ {data.index[-1]}")
    except Exception as e:
        print(f"✗ テスト用データ生成に失敗: {e}")
        return
    
    # HyperTrendFollowSignalのテスト
    try:
        print("\n--- HyperTrendFollowSignal テスト ---")
        
        # シグナルの作成（デフォルトパラメータ）
        signal = HyperTrendFollowSignal(
            # HyperFRAMAパラメータ
            hyper_frama_period=16,
            hyper_frama_src_type='hl2',
            hyper_frama_fc=1,
            hyper_frama_sc=198,
            hyper_frama_alpha_multiplier=0.5,
            
            # HyperFRAMAChannelパラメータ  
            channel_period=20,
            channel_hyper_frama_channel_src_type='hl2',
            channel_fixed_multiplier=2.0
        )
        
        # シグナル生成のテスト
        print(f"  シグナル生成中...")
        signals = signal.generate(data)
        
        # 結果の統計
        total_signals = len(signals)
        long_signals = np.sum(signals == 1)  
        short_signals = np.sum(signals == -1)
        neutral_signals = np.sum(signals == 0)
        
        print(f"  総シグナル数: {total_signals}")
        print(f"  ロングシグナル: {long_signals} ({long_signals/total_signals*100:.1f}%)")
        print(f"  ショートシグナル: {short_signals} ({short_signals/total_signals*100:.1f}%)")
        print(f"  ニュートラル: {neutral_signals} ({neutral_signals/total_signals*100:.1f}%)")
        
        # シグナルの変化点を確認
        signal_changes = np.diff(signals)
        entries = np.sum(np.abs(signal_changes) > 0)
        print(f"  シグナル変化回数: {entries}")
        
        # 最近の10個のシグナルを表示
        print(f"\n  最近のシグナル（最新10個）:")
        recent_data = data.tail(10)
        recent_signals = signals[-10:]
        
        for i, (idx, row) in enumerate(recent_data.iterrows()):
            signal_val = recent_signals[i]
            signal_str = "LONG" if signal_val == 1 else "SHORT" if signal_val == -1 else "NEUTRAL"
            print(f"    {idx.strftime('%Y-%m-%d %H:%M')} | Close: {row['close']:.4f} | Signal: {signal_str}")
        
        print(f"  ✓ HyperTrendFollowSignal テスト完了")
        
        # 個別シグナルの詳細テスト
        print(f"\n--- 個別シグナル詳細確認 ---")
        
        # Position Signal (HyperFRAMA)の確認
        position_signals = signal.position_signal.generate(data)
        pos_long = np.sum(position_signals == 1)
        pos_short = np.sum(position_signals == -1)
        print(f"  Position Signal - Long: {pos_long}, Short: {pos_short}")
        
        # Breakout Signal (HyperFRAMAChannel)の確認  
        breakout_signals = signal.breakout_signal.generate(data)
        bo_long = np.sum(breakout_signals == 1)
        bo_short = np.sum(breakout_signals == -1)
        print(f"  Breakout Signal - Long: {bo_long}, Short: {bo_short}")
        
        # 組み合わせロジックの確認
        combined_long = np.sum((position_signals == 1) & (breakout_signals == 1))
        combined_short = np.sum((position_signals == -1) & (breakout_signals == -1))
        print(f"  Combined Entry - Long: {combined_long}, Short: {combined_short}")
        
        # 決済ロジックの確認
        long_exits = np.sum((signals[:-1] == 1) & (breakout_signals[1:] == -1))
        short_exits = np.sum((signals[:-1] == -1) & (breakout_signals[1:] == 1))
        print(f"  Exit Signals - Long exits: {long_exits}, Short exits: {short_exits}")
        
    except Exception as e:
        print(f"✗ HyperTrendFollowSignal テストでエラー: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n{'='*60}")
    print(f"✓ 全てのテストが完了しました")

if __name__ == "__main__":
    main()