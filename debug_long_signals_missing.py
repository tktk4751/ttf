#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
import yaml

# プロジェクトのルートディレクトリを追加
sys.path.append('/home/vapor/dev/ttf')

from strategies.implementations.hyper_frama.strategy import HyperFRAMAEnhancedStrategy
from data.binance_data_source import BinanceDataSource
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor

def analyze_missing_long_signals():
    """ロングシグナルがなぜ無視されるかを分析"""
    print("=== ロングシグナル無視原因分析 ===")
    
    # 設定とデータ読み込み
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    binance_config = config.get('binance_data', {})
    data_dir = binance_config.get('data_dir', 'data/binance')
    binance_data_source = BinanceDataSource(data_dir)
    
    dummy_csv_source = CSVDataSource("dummy")
    data_loader = DataLoader(
        data_source=dummy_csv_source,
        binance_data_source=binance_data_source
    )
    data_processor = DataProcessor()
    
    raw_data = data_loader.load_data_from_config(config)
    processed_data = {
        symbol: data_processor.process(df)
        for symbol, df in raw_data.items()
    }
    
    first_symbol = next(iter(processed_data))
    data = processed_data[first_symbol]
    
    # ストラテジー
    strategy = HyperFRAMAEnhancedStrategy()
    entry_signals = strategy.generate_entry(data)
    
    # シグナル分析
    long_signal_indices = np.where(entry_signals == 1)[0]
    short_signal_indices = np.where(entry_signals == -1)[0]
    all_signal_indices = sorted(list(long_signal_indices) + list(short_signal_indices))
    
    print(f"✓ 全シグナル: {len(all_signal_indices)}個")
    print(f"  ロングシグナル: {len(long_signal_indices)}個")
    print(f"  ショートシグナル: {len(short_signal_indices)}個")
    
    # ロングシグナルとショートシグナルの位置関係を分析
    print(f"\n=== シグナル位置関係分析 ===")
    
    warmup_bars = 100
    
    print(f"ウォームアップ後の有効シグナル:")
    valid_long_signals = [idx for idx in long_signal_indices if idx >= warmup_bars]
    valid_short_signals = [idx for idx in short_signal_indices if idx >= warmup_bars]
    
    print(f"  有効ロングシグナル: {len(valid_long_signals)}個")
    print(f"  有効ショートシグナル: {len(valid_short_signals)}個")
    
    # 最初の20個の有効シグナルを時系列で表示
    all_valid_signals = []
    for idx in valid_long_signals:
        all_valid_signals.append((idx, 'LONG', 1))
    for idx in valid_short_signals:
        all_valid_signals.append((idx, 'SHORT', -1))
    
    # インデックス順でソート
    all_valid_signals.sort(key=lambda x: x[0])
    
    print(f"\n=== 時系列シグナル順序（最初の20個） ===")
    for i, (idx, signal_type, signal_value) in enumerate(all_valid_signals[:20]):
        print(f"[{i+1:2d}] インデックス {idx:4d}: {signal_type:5s} | {data.index[idx]}")
    
    # ロングシグナルが連続する箇所を検索
    print(f"\n=== 連続するロングシグナル分析 ===")
    consecutive_longs = []
    
    for i in range(len(all_valid_signals) - 1):
        current_idx, current_type, _ = all_valid_signals[i]
        next_idx, next_type, _ = all_valid_signals[i + 1]
        
        if current_type == 'LONG' and next_type == 'LONG':
            consecutive_longs.append((current_idx, next_idx))
        elif current_type == 'LONG':
            # ロングシグナル直後に何が来るか
            gap = next_idx - current_idx
            print(f"ロング[{current_idx}] -> {next_type}[{next_idx}] (間隔: {gap}バー)")
    
    if consecutive_longs:
        print(f"連続するロングシグナル: {len(consecutive_longs)}組")
        for long1, long2 in consecutive_longs:
            print(f"  {long1} -> {long2} (間隔: {long2-long1}バー)")
    else:
        print("連続するロングシグナルはありません")
    
    # ロングシグナル直後のシグナルを分析
    print(f"\n=== ロングシグナル直後の状況 ===")
    long_followup_analysis = []
    
    for i, (idx, signal_type, _) in enumerate(all_valid_signals):
        if signal_type == 'LONG' and i < len(all_valid_signals) - 1:
            next_idx, next_type, _ = all_valid_signals[i + 1]
            gap = next_idx - idx
            long_followup_analysis.append((idx, next_idx, next_type, gap))
    
    print(f"ロングシグナル直後の分析（最初の10個）:")
    for i, (long_idx, next_idx, next_type, gap) in enumerate(long_followup_analysis[:10]):
        print(f"  ロング[{long_idx}] -> {next_type}[{next_idx}] (間隔: {gap}バー)")
        
        # 間隔が短い（即座に反対シグナル）場合は要注意
        if gap <= 3:
            print(f"    ⚠️ 短い間隔: {gap}バー以内にシグナル変更")
    
    # バックテスターのmax_positionsによる制限を確認
    print(f"\n=== バックテスター制限分析 ===")
    print(f"最大同時ポジション数: 1 (デフォルト)")
    print(f"これにより、ポジション保有中は新しいエントリーがブロックされます")
    
    # ロングシグナル発生時にショートポジションが既に存在するかもしれない
    print(f"\n=== ポジション競合分析 ===")
    position_states = []  # (index, action, position_type)
    
    # 簡略化されたポジション状態をシミュレート
    current_position_type = None
    
    for idx, signal_type, _ in all_valid_signals[:10]:  # 最初の10シグナルのみ
        if current_position_type is None:
            # ポジションなし -> エントリー
            current_position_type = signal_type
            position_states.append((idx, 'ENTRY', signal_type))
            print(f"[{idx}] エントリー: {signal_type}")
        else:
            # ポジションあり -> 反対シグナルならエグジット
            if signal_type != current_position_type:
                position_states.append((idx, 'EXIT', current_position_type))
                position_states.append((idx, 'ENTRY', signal_type))
                print(f"[{idx}] エグジット: {current_position_type} -> エントリー: {signal_type}")
                current_position_type = signal_type
            else:
                # 同じ方向のシグナル -> 無視
                print(f"[{idx}] 無視: {signal_type} (既に{current_position_type}ポジション保有中)")
    
    return all_valid_signals

def main():
    """メイン関数"""
    print("ロングシグナル無視原因の詳細分析")
    print("=" * 60)
    
    try:
        signals = analyze_missing_long_signals()
        
        print(f"\n{'='*60}")
        print(f"分析完了: {len(signals)}個のシグナルを分析")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"エラー: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()