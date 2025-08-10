#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
import os
import yaml

# プロジェクトのルートディレクトリを追加
sys.path.append('/home/vapor/dev/ttf')

from strategies.implementations.hyper_frama.strategy import HyperFRAMAEnhancedStrategy
from data.binance_data_source import BinanceDataSource
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from position_sizing.x_position_sizing import XATRPositionSizing
from backtesting.trade import Trade
from position_sizing.position_sizing import PositionSizingParams

def trace_backtester_execution():
    """バックテスターの実行を完全にトレース"""
    print("=== バックテスター実行完全トレース ===")
    
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
    
    # ストラテジーとポジションサイジング
    strategy = HyperFRAMAEnhancedStrategy()
    position_manager = XATRPositionSizing()
    
    # バックテスター設定
    dates = data.index
    closes = data['close'].values
    warmup_bars = 100
    initial_balance = 10000.0
    current_capital = initial_balance
    commission = 0.001
    max_positions = 1
    
    # バックテスター状態変数
    current_position = None
    pending_entry = None
    pending_exit = False
    trades = []
    
    # エントリーシグナル生成
    entry_signals = strategy.generate_entry(data)
    
    # シグナル統計
    long_signal_indices = np.where(entry_signals == 1)[0]
    short_signal_indices = np.where(entry_signals == -1)[0]
    all_signal_indices = sorted(list(long_signal_indices) + list(short_signal_indices))
    
    print(f"✓ 全シグナル: {len(all_signal_indices)}個")
    print(f"  ロング: {len(long_signal_indices)}個")
    print(f"  ショート: {len(short_signal_indices)}個")
    print(f"  最初の20シグナル: {all_signal_indices[:20]}")
    
    print(f"\n=== バックテスター実行トレース（最初の20シグナル） ===")
    
    # 最初の20シグナルでシミュレーション
    trace_count = 0
    for i in range(warmup_bars, len(data)):
        if trace_count >= 20:  # 最初の20シグナルのみ
            break
            
        date = dates[i]
        close = closes[i]
        
        # 保留中のエグジットの処理
        if pending_exit and current_position is not None:
            print(f"\n[{i}] === PENDING EXIT処理 ===")
            print(f"  日時: {date}")
            print(f"  価格: ${close:.2f}")
            print(f"  現在のポジション: {current_position.position_type}")
            
            # エグジット処理（簡略版）
            current_position.close(date, close, current_capital)
            trades.append(current_position)
            current_capital = current_position.balance
            print(f"  エグジット完了: PnL=${current_position.profit_loss:.2f}")
            print(f"  新しい資金: ${current_capital:.2f}")
            
            current_position = None
            pending_exit = False
        
        # 保留中のエントリーの処理
        if pending_entry is not None and current_position is None and position_manager.can_enter():
            position_type, position_size, entry_index = pending_entry
            print(f"\n[{i}] === PENDING ENTRY処理 ===")
            print(f"  日時: {date}")
            print(f"  価格: ${close:.2f}")
            print(f"  エントリータイプ: {position_type}")
            print(f"  ポジションサイズ: ${position_size:.2f}")
            print(f"  エントリーインデックス: {entry_index}")
            
            # エントリー処理
            current_position = Trade(
                position_type=position_type,
                position_size=position_size,
                commission_rate=commission,
                slippage_rate=0.001
            )
            current_position.symbol = first_symbol
            current_position.entry(date, close)
            print(f"  エントリー完了: {position_type} @ ${current_position.entry_price:.2f}")
            
            pending_entry = None
            trace_count += 1
        
        # 現在のポジションがある場合、エグジットシグナルをチェック
        if current_position is not None and not pending_exit:
            position_type = -1 if current_position.position_type == 'SHORT' else 1
            if strategy.generate_exit(data, position_type, i):
                print(f"\n[{i}] === EXIT SIGNAL検出 ===")
                print(f"  現在のポジション: {current_position.position_type}")
                print(f"  エグジット条件満たし -> pending_exit = True")
                pending_exit = True
        
        # 現在のポジションがない場合、エントリーシグナルをチェック
        if current_position is None and not pending_entry:
            signal_value = entry_signals[i]
            
            if signal_value != 0:  # シグナルがある場合
                signal_type = "LONG" if signal_value == 1 else "SHORT"
                print(f"\n[{i}] === ENTRY SIGNAL検出 ===")
                print(f"  日時: {date}")
                print(f"  価格: ${close:.2f}")
                print(f"  シグナル: {signal_type} ({signal_value})")
                
                # ポジションサイズ計算
                stop_loss_price = close * 0.95 if signal_value == 1 else close * 1.05
                lookback_start = max(0, i - warmup_bars)
                historical_data = data.iloc[lookback_start:i+1].copy()
                
                if len(historical_data) >= warmup_bars:
                    params = PositionSizingParams(
                        entry_price=close,
                        stop_loss_price=stop_loss_price,
                        capital=current_capital,
                        historical_data=historical_data
                    )
                    
                    sizing_result = position_manager.calculate(params)
                    position_size = sizing_result['position_size']
                    
                    print(f"  ポジションサイズ計算: ${position_size:.2f}")
                    print(f"  現在の資金: ${current_capital:.2f}")
                    print(f"  エントリー可能: {position_manager.can_enter()}")
                    
                    if position_size > 0:
                        pending_entry = (signal_type, position_size, i)
                        print(f"  ✅ pending_entry設定: {pending_entry}")
                    else:
                        print(f"  ❌ ポジションサイズ0につきスキップ")
                else:
                    print(f"  ❌ 履歴データ不足: {len(historical_data)} < {warmup_bars}")
    
    print(f"\n=== トレース完了 ===")
    print(f"実行されたトレード: {len(trades)}")
    
    for i, trade in enumerate(trades):
        print(f"  [{i+1}] {trade.position_type}: ${trade.profit_loss:.2f}")
    
    return trades

def main():
    """メイン関数"""
    print("バックテスター実行完全トレース")
    print("=" * 60)
    
    try:
        trades = trace_backtester_execution()
        
        long_trades = [t for t in trades if t.position_type == 'LONG']
        short_trades = [t for t in trades if t.position_type == 'SHORT']
        
        print(f"\n{'='*60}")
        print(f"最終結果:")
        print(f"  総トレード: {len(trades)}")
        print(f"  ロングトレード: {len(long_trades)}")
        print(f"  ショートトレード: {len(short_trades)}")
        
        if len(long_trades) == 0:
            print(f"  ⚠️ ロングトレードが実行されていません")
        else:
            print(f"  ✅ ロングトレードが正常に実行されました")
        
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"エラー: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()