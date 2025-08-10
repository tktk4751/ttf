#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from strategies.implementations.hyper_frama_channel.strategy import HyperFRAMAChannelStrategy
from strategies.implementations.hyper_frama_channel.signal_generator import FilterType

def test_signals():
    """簡単なテストデータでシグナルをテスト"""
    
    # テストデータ作成（上昇トレンド -> 下降トレンド）
    np.random.seed(42)
    n_points = 200
    
    # 価格データ作成
    base_price = 100
    prices = []
    
    # 上昇トレンド (0-100)
    for i in range(100):
        base_price += np.random.normal(0.5, 1.0)  # 上昇バイアス
        prices.append(max(base_price, 1))  # 価格は1以上に保つ
    
    # 下降トレンド (100-200)
    for i in range(100):
        base_price += np.random.normal(-0.3, 1.0)  # 下降バイアス
        prices.append(max(base_price, 1))  # 価格は1以上に保つ
    
    # DataFrame作成
    data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': [1000] * n_points
    })
    
    # high, lowの調整
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    print(f"テストデータ作成完了: {len(data)}行")
    print(f"価格範囲: {data['close'].min():.2f} - {data['close'].max():.2f}")
    
    # ストラテジー初期化
    strategy = HyperFRAMAChannelStrategy(
        period=14,
        multiplier_mode="fixed",
        fixed_multiplier=1.5,
        src_type="oc2",
        short_hyper_frama_period=8,  # より短期
        short_hyper_frama_alpha_multiplier=1.0,
        long_hyper_frama_period=20,  # より長期
        long_hyper_frama_alpha_multiplier=0.3,
        hyper_frama_src_type='oc2',
        filter_type=FilterType.NONE
    )
    
    print("\nシグナル生成中...")
    
    try:
        # エントリーシグナル生成
        entry_signals = strategy.generate_entry(data)
        
        # 個別シグナル取得
        long_signals = strategy.get_long_signals(data)
        short_signals = strategy.get_short_signals(data)
        breakout_signals = strategy.get_breakout_signals(data)
        
        # FRAMA値取得
        short_frama, long_frama = strategy.get_frama_values(data)
        
        # チャネルバンド取得
        midline, upper_band, lower_band = strategy.get_channel_bands(data)
        
        print(f"\nシグナル統計:")
        print(f"エントリーシグナル - ロング: {np.sum(entry_signals == 1)}, ショート: {np.sum(entry_signals == -1)}")
        print(f"ロングシグナル: {np.sum(long_signals)}")
        print(f"ショートシグナル: {np.sum(short_signals)}")
        print(f"ブレイクアウトシグナル - ロング: {np.sum(breakout_signals == 1)}, ショート: {np.sum(breakout_signals == -1)}")
        
        # FRAMAクロス分析
        if short_frama is not None and long_frama is not None:
            valid_mask = ~(np.isnan(short_frama) | np.isnan(long_frama))
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_indices) > 1:
                frama_cross_up = 0
                frama_cross_down = 0
                
                for i in range(1, len(valid_indices)):
                    curr_idx = valid_indices[i]
                    prev_idx = valid_indices[i-1]
                    
                    if (short_frama[curr_idx] > long_frama[curr_idx] and 
                        short_frama[prev_idx] <= long_frama[prev_idx]):
                        frama_cross_up += 1
                    elif (short_frama[curr_idx] < long_frama[curr_idx] and 
                          short_frama[prev_idx] >= long_frama[prev_idx]):
                        frama_cross_down += 1
                
                print(f"FRAMAクロス - 上向き: {frama_cross_up}, 下向き: {frama_cross_down}")
                
                # 現在のFRAMA状態
                short_above_long = np.sum(short_frama[valid_mask] > long_frama[valid_mask])
                short_below_long = np.sum(short_frama[valid_mask] < long_frama[valid_mask])
                print(f"FRAMA状態 - 短期>長期: {short_above_long}, 短期<長期: {short_below_long}")
        
        # エントリーポイントの確認
        long_entry_indices = np.where(entry_signals == 1)[0]
        short_entry_indices = np.where(entry_signals == -1)[0]
        
        print(f"\nロングエントリー発生インデックス: {long_entry_indices[:5] if len(long_entry_indices) > 0 else 'なし'}")
        print(f"ショートエントリー発生インデックス: {short_entry_indices[:5] if len(short_entry_indices) > 0 else 'なし'}")
        
        # エグジット条件テスト
        if len(long_entry_indices) > 0:
            test_idx = long_entry_indices[0]
            print(f"\n最初のロングエントリー（インデックス {test_idx}）からのエグジットテスト:")
            
            exit_found = False
            for i in range(test_idx + 1, min(test_idx + 30, len(data))):
                exit_signal = strategy.generate_exit(data.iloc[:i+1], position=1, index=i)
                if exit_signal:
                    print(f"エグジットシグナル発生: インデックス {i} (エントリーから{i-test_idx}バー後)")
                    if short_frama is not None and long_frama is not None:
                        print(f"  短期FRAMA: {short_frama[i]:.4f}, 長期FRAMA: {long_frama[i]:.4f}")
                        print(f"  前の短期FRAMA: {short_frama[i-1]:.4f}, 前の長期FRAMA: {long_frama[i-1]:.4f}")
                    exit_found = True
                    break
            
            if not exit_found:
                print("30バー内でエグジットシグナル未発生")
        
        # 詳細データの出力（有効なデータの最初の10行）
        if short_frama is not None and long_frama is not None:
            valid_mask = ~(np.isnan(short_frama) | np.isnan(long_frama))
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_indices) > 0:
                print(f"\n詳細データ（有効データの最初の10行）:")
                print("Index | Close | Short_FRAMA | Long_FRAMA | S>L | Upper | Lower | Entry | Breakout")
                
                for j in range(min(10, len(valid_indices))):
                    i = valid_indices[j]
                    close_val = data.iloc[i]['close']
                    sf = short_frama[i]
                    lf = long_frama[i]
                    s_greater_l = "Y" if sf > lf else "N"
                    ub = upper_band[i] if upper_band is not None else np.nan
                    lb = lower_band[i] if lower_band is not None else np.nan
                    entry = entry_signals[i]
                    breakout = breakout_signals[i]
                    
                    print(f"{i:5d} | {close_val:6.2f} | {sf:10.4f} | {lf:10.4f} | {s_greater_l:3s} | {ub:6.2f} | {lb:6.2f} | {entry:5d} | {breakout:8d}")
    
    except Exception as e:
        print(f"エラー発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_signals()