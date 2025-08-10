#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from data.data_loader import DataLoader
from strategies.implementations.hyper_frama_channel.strategy import HyperFRAMAChannelStrategy
from strategies.implementations.hyper_frama_channel.signal_generator import FilterType

def debug_signals():
    """シグナルをデバッグ"""
    print("データロード中...")
    
    # データロード
    import yaml
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    from data.binance_data_source import BinanceDataSource
    data_source = BinanceDataSource()
    loader = DataLoader(data_source, config.get('data_processing', {}))
    raw_data = loader.load_data_from_config(config)
    data = raw_data['SOL_spot']
    print(f"データ数: {len(data)}")
    
    # 最初の300行のみを使用（デバッグ用）
    data_subset = data.iloc[:500].copy()
    print(f"デバッグ用データ数: {len(data_subset)}")
    
    # ストラテジー初期化
    strategy = HyperFRAMAChannelStrategy(
        period=14,
        multiplier_mode="fixed",
        fixed_multiplier=2.0,
        src_type="oc2",
        short_hyper_frama_period=16,
        short_hyper_frama_alpha_multiplier=1.0,
        long_hyper_frama_period=32,
        long_hyper_frama_alpha_multiplier=0.1,
        hyper_frama_src_type='oc2',
        filter_type=FilterType.NONE
    )
    
    print("\nシグナル生成中...")
    
    # エントリーシグナル生成
    entry_signals = strategy.generate_entry(data_subset)
    
    # 個別シグナル取得
    long_signals = strategy.get_long_signals(data_subset)
    short_signals = strategy.get_short_signals(data_subset)
    breakout_signals = strategy.get_breakout_signals(data_subset)
    
    # FRAMA値取得
    short_frama, long_frama = strategy.get_frama_values(data_subset)
    
    # チャネルバンド取得
    midline, upper_band, lower_band = strategy.get_channel_bands(data_subset)
    
    print(f"\nシグナル統計:")
    print(f"エントリーシグナル - ロング: {np.sum(entry_signals == 1)}, ショート: {np.sum(entry_signals == -1)}")
    print(f"ロングシグナル: {np.sum(long_signals)}")
    print(f"ショートシグナル: {np.sum(short_signals)}")
    print(f"ブレイクアウトシグナル - ロング: {np.sum(breakout_signals == 1)}, ショート: {np.sum(breakout_signals == -1)}")
    
    # FRAMAクロス分析
    if short_frama is not None and long_frama is not None:
        frama_cross_up = np.sum((short_frama[1:] > long_frama[1:]) & (short_frama[:-1] <= long_frama[:-1]))
        frama_cross_down = np.sum((short_frama[1:] < long_frama[1:]) & (short_frama[:-1] >= long_frama[:-1]))
        print(f"FRAMAクロス - 上向き: {frama_cross_up}, 下向き: {frama_cross_down}")
        
        # 現在のFRAMA状態
        short_above_long = np.sum(short_frama > long_frama)
        short_below_long = np.sum(short_frama < long_frama)
        print(f"FRAMA状態 - 短期>長期: {short_above_long}, 短期<長期: {short_below_long}")
    
    # 具体的なシグナル発生箇所を確認
    long_entry_indices = np.where(entry_signals == 1)[0]
    short_entry_indices = np.where(entry_signals == -1)[0]
    
    print(f"\nロングエントリー発生インデックス: {long_entry_indices[:10]}")  # 最初の10個
    print(f"ショートエントリー発生インデックス: {short_entry_indices[:10]}")  # 最初の10個
    
    # エグジット条件のチェック
    if len(long_entry_indices) > 0:
        test_index = long_entry_indices[0]
        print(f"\n最初のロングエントリー（インデックス {test_index}）のエグジット条件チェック:")
        
        for i in range(test_index + 1, min(test_index + 50, len(data_subset))):
            exit_signal = strategy.generate_exit(data_subset.iloc[:i+1], position=1, index=i)
            if exit_signal:
                print(f"エグジットシグナル発生: インデックス {i}")
                print(f"短期FRAMA: {short_frama[i]:.4f}, 長期FRAMA: {long_frama[i]:.4f}")
                print(f"前の短期FRAMA: {short_frama[i-1]:.4f}, 前の長期FRAMA: {long_frama[i-1]:.4f}")
                break
        else:
            print("最初の50バー内でエグジットシグナル未発生")
    
    # 詳細データの出力（最初の20行）
    print(f"\n詳細データ（最初の20行）:")
    print("Index | Close | Short_FRAMA | Long_FRAMA | Upper | Lower | Entry | Breakout")
    for i in range(min(20, len(data_subset))):
        close_val = data_subset.iloc[i]['close']
        sf = short_frama[i] if short_frama is not None else np.nan
        lf = long_frama[i] if long_frama is not None else np.nan
        ub = upper_band[i] if upper_band is not None else np.nan
        lb = lower_band[i] if lower_band is not None else np.nan
        entry = entry_signals[i]
        breakout = breakout_signals[i]
        
        print(f"{i:5d} | {close_val:6.2f} | {sf:10.4f} | {lf:10.4f} | {ub:6.2f} | {lb:6.2f} | {entry:5d} | {breakout:8d}")

if __name__ == "__main__":
    debug_signals()