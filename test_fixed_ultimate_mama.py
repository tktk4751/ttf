#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
修正されたUltimate MAMA のバックテスト
"""

import sys
import os
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource
from strategies.implementations.ultimate_mama.strategy import UltimateMAMAStrategy
from backtesting.backtester import Backtester
from position_sizing.supreme_position_sizing import SupremePositionSizing

def test_fixed_ultimate_mama():
    """修正されたUltimate MAMAのテスト"""
    print("=== 修正されたUltimate MAMA バックテスト ===")
    
    # 設定ファイルの読み込み
    config_path = Path('config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # データの準備
    binance_config = config.get('binance_data', {})
    data_dir = binance_config.get('data_dir', 'data/binance')
    binance_data_source = BinanceDataSource(data_dir)
    
    dummy_csv_source = CSVDataSource("dummy")
    data_loader = DataLoader(
        data_source=dummy_csv_source,
        binance_data_source=binance_data_source
    )
    data_processor = DataProcessor()
    
    # データの読み込み
    print("データを読み込み中...")
    raw_data = data_loader.load_data_from_config(config)
    processed_data = {
        symbol: data_processor.process(df)
        for symbol, df in raw_data.items()
    }
    
    # 修正されたUltimate MAMAストラテジー（非常に感度高く設定）
    strategy = UltimateMAMAStrategy(
        # 超感度設定
        fast_limit=0.8,
        slow_limit=0.08,
        src_type='hlc3',
        quantum_coherence_factor=0.5,
        quantum_entanglement_strength=0.5,
        mmae_models_count=3,
        vmd_modes_count=2,
        base_confidence_threshold=0.3,  # さらに低く
        minimum_signal_quality=0.1,    # さらに低く
        quantum_exit_threshold=0.15,   # さらに低く
        enable_adaptive_thresholds=True,
        enable_quantum_exit=True,
        enable_real_time_optimization=False,  # リアルタイム最適化を無効化
        ml_adaptation_enabled=False    # 機械学習を無効化
    )
    
    # ポジションサイジング
    position_sizing = SupremePositionSizing()
    
    # バックテスター
    backtester = Backtester(
        strategy=strategy,
        position_manager=position_sizing,
        initial_balance=10000,
        commission=0.001,
        verbose=True
    )
    
    print("バックテスト実行中...")
    trades = backtester.run(processed_data)
    
    print(f"\n=== バックテスト結果 ===")
    print(f"総トレード数: {len(trades)}")
    
    if len(trades) > 0:
        # トレード詳細
        long_trades = [t for t in trades if hasattr(t, 'direction') and t.direction == 'LONG']
        short_trades = [t for t in trades if hasattr(t, 'direction') and t.direction == 'SHORT']
        
        # directionがない場合はentry_typeを確認
        if len(long_trades) == 0 and len(short_trades) == 0:
            long_trades = [t for t in trades if hasattr(t, 'entry_type') and t.entry_type == 1]
            short_trades = [t for t in trades if hasattr(t, 'entry_type') and t.entry_type == -1]
        
        print(f"ロングトレード数: {len(long_trades)}")
        print(f"ショートトレード数: {len(short_trades)}")
        
        # 最初の5トレードを表示
        print(f"\n最初の5トレード:")
        for i, trade in enumerate(trades[:5]):
            # Tradeオブジェクトの属性を確認
            print(f"  Trade {i+1} attributes: {[attr for attr in dir(trade) if not attr.startswith('_')]}")
            
            # 可能な属性での表示
            entry_price = getattr(trade, 'entry_price', getattr(trade, 'open_price', 'N/A'))
            exit_price = getattr(trade, 'exit_price', getattr(trade, 'close_price', 'N/A'))
            pnl = getattr(trade, 'pnl', getattr(trade, 'profit_loss', 0))
            direction = getattr(trade, 'direction', getattr(trade, 'entry_type', 'Unknown'))
            
            print(f"  {i+1}: {direction} @ {entry_price} -> {exit_price} PnL: {pnl}")
        
        # PnL統計
        pnls = [getattr(trade, 'pnl', getattr(trade, 'profit_loss', 0)) for trade in trades]
        winning_trades = [t for t in trades if getattr(t, 'pnl', getattr(t, 'profit_loss', 0)) > 0]
        losing_trades = [t for t in trades if getattr(t, 'pnl', getattr(t, 'profit_loss', 0)) < 0]
        
        print(f"\nPnL統計:")
        print(f"  総PnL: {sum(pnls):.2f}")
        print(f"  勝ちトレード: {len(winning_trades)}回")
        print(f"  負けトレード: {len(losing_trades)}回")
        print(f"  勝率: {len(winning_trades)/len(trades)*100:.2f}%")
        
        if len(winning_trades) > 0:
            avg_win = np.mean([getattr(t, 'pnl', getattr(t, 'profit_loss', 0)) for t in winning_trades])
            print(f"  平均勝ちトレード: {avg_win:.2f}")
            
        if len(losing_trades) > 0:
            avg_loss = np.mean([getattr(t, 'pnl', getattr(t, 'profit_loss', 0)) for t in losing_trades])
            print(f"  平均負けトレード: {avg_loss:.2f}")
    else:
        print("❌ トレードが発生しませんでした")
        
        # シグナル分析
        first_symbol = list(processed_data.keys())[0]
        data = processed_data[first_symbol]
        
        print("\nシグナル分析:")
        entry_signals = strategy.generate_entry(data)
        signal_count = np.sum(entry_signals != 0)
        long_count = np.sum(entry_signals == 1)
        short_count = np.sum(entry_signals == -1)
        
        print(f"  総シグナル数: {signal_count}")
        print(f"  ロングシグナル数: {long_count}")
        print(f"  ショートシグナル数: {short_count}")
        print(f"  シグナル率: {signal_count/len(data)*100:.2f}%")
        
        # シグナル変化分析
        signal_changes = 0
        prev_signal = 0
        for signal in entry_signals:
            if signal != prev_signal:
                signal_changes += 1
            prev_signal = signal
        
        print(f"  シグナル変化回数: {signal_changes}")
        
    print("\n✅ 修正されたUltimate MAMA テスト完了")

if __name__ == "__main__":
    test_fixed_ultimate_mama()