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
from position_sizing.x_position_sizing import XATRPositionSizing
from position_sizing.position_sizing import PositionSizingParams
from backtesting.trade import Trade

def analyze_long_signal_timing():
    """ロングシグナル発生時のタイミングと制約を分析"""
    print("=== ロングシグナル発生タイミング分析 ===")
    
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
    position_manager = XATRPositionSizing()
    
    # エントリーシグナル生成
    entry_signals = strategy.generate_entry(data)
    
    # シグナル統計
    long_signal_indices = np.where(entry_signals == 1)[0]
    short_signal_indices = np.where(entry_signals == -1)[0]
    
    print(f"ロングシグナル: {len(long_signal_indices)}個")
    print(f"ショートシグナル: {len(short_signal_indices)}個")
    
    # 最初の有効なロングシグナルを特定
    warmup_bars = 100
    first_long_idx = None
    first_short_idx = None
    
    for idx in long_signal_indices:
        if idx >= warmup_bars:
            first_long_idx = idx
            break
    
    for idx in short_signal_indices:
        if idx >= warmup_bars:
            first_short_idx = idx
            break
    
    print(f"\n最初の有効シグナル:")
    print(f"  ロング: インデックス {first_long_idx} ({data.index[first_long_idx] if first_long_idx else 'なし'})")
    print(f"  ショート: インデックス {first_short_idx} ({data.index[first_short_idx] if first_short_idx else 'なし'})")
    
    if first_long_idx is None:
        print("❌ 有効なロングシグナルが見つかりません")
        return
    
    # バックテスターを最初のロングシグナルまで実行
    print(f"\n=== 最初のロングシグナル前の状況分析 ===")
    
    dates = data.index
    closes = data['close'].values
    initial_balance = 10000.0
    current_capital = initial_balance
    commission = 0.001
    
    # 簡化されたバックテスター状態
    current_position = None
    pending_entry = None
    pending_exit = False
    trades = []
    
    # 最初のロングシグナルまでの間に何が起こるかを追跡
    relevant_events = []
    
    for i in range(warmup_bars, first_long_idx + 1):
        date = dates[i]
        close = closes[i]
        signal = entry_signals[i]
        
        # イベントを記録
        if signal != 0:
            signal_type = "LONG" if signal == 1 else "SHORT"
            relevant_events.append({
                'index': i,
                'date': date,
                'price': close,
                'signal_type': signal_type,
                'signal_value': signal,
                'current_position': current_position.position_type if current_position else None
            })
        
        # 簡化されたバックテスターロジック
        # 1. 保留中のエグジットの処理
        if pending_exit and current_position is not None:
            current_position.close(date, close, current_capital)
            trades.append(current_position)
            current_capital = current_position.balance
            current_position = None
            pending_exit = False
            relevant_events[-1]['exit_executed'] = True if relevant_events else None
        
        # 2. 保留中のエントリーの処理
        if pending_entry is not None and current_position is None:
            position_type, position_size, entry_index = pending_entry
            
            if position_manager.can_enter() and position_size > 0:
                current_position = Trade(
                    position_type=position_type,
                    position_size=position_size,
                    commission_rate=commission,
                    slippage_rate=0.001
                )
                current_position.symbol = first_symbol
                current_position.entry(date, close)
                pending_entry = None
                
                if relevant_events:
                    relevant_events[-1]['entry_executed'] = True
                    relevant_events[-1]['position_size'] = position_size
        
        # 3. エグジットシグナルのチェック
        if current_position is not None and not pending_exit:
            position_direction = -1 if current_position.position_type == 'SHORT' else 1
            exit_signal = strategy.generate_exit(data, position_direction, i)
            
            if exit_signal:
                pending_exit = True
                if relevant_events:
                    relevant_events[-1]['exit_pending'] = True
        
        # 4. 新しいエントリーシグナルの処理
        if current_position is None and not pending_entry and signal != 0:
            signal_type = "LONG" if signal == 1 else "SHORT"
            
            # ポジションサイズ計算
            stop_loss_price = close * 0.95 if signal == 1 else close * 1.05
            lookback_start = max(0, i - warmup_bars)
            historical_data = data.iloc[lookback_start:i+1].copy()
            
            if len(historical_data) >= warmup_bars:
                try:
                    params = PositionSizingParams(
                        entry_price=close,
                        stop_loss_price=stop_loss_price,
                        capital=current_capital,
                        historical_data=historical_data
                    )
                    
                    sizing_result = position_manager.calculate(params)
                    position_size = sizing_result['position_size']
                    
                    if position_size > 0:
                        pending_entry = (signal_type, position_size, i)
                        if relevant_events:
                            relevant_events[-1]['pending_entry'] = (signal_type, position_size)
                except Exception as e:
                    if relevant_events:
                        relevant_events[-1]['sizing_error'] = str(e)
    
    # イベントログを表示
    print(f"\n=== 最初のロングシグナル前のイベント履歴 ===")
    for i, event in enumerate(relevant_events):
        print(f"\n[{i+1}] インデックス {event['index']}: {event['signal_type']}シグナル")
        print(f"    日時: {event['date']}")
        print(f"    価格: ${event['price']:.2f}")
        print(f"    現在のポジション: {event['current_position'] or 'なし'}")
        
        if 'pending_entry' in event:
            pos_type, pos_size = event['pending_entry']
            print(f"    → 保留エントリー設定: {pos_type}, ${pos_size:.2f}")
        
        if 'entry_executed' in event:
            print(f"    → ✅ エントリー実行: ${event.get('position_size', 0):.2f}")
        
        if 'exit_pending' in event:
            print(f"    → ⏳ エグジット保留設定")
        
        if 'exit_executed' in event:
            print(f"    → ✅ エグジット実行")
        
        if 'sizing_error' in event:
            print(f"    → ❌ サイジングエラー: {event['sizing_error']}")
    
    # 最初のロングシグナル時の状況分析
    print(f"\n=== 最初のロングシグナル時の詳細状況 ===")
    print(f"インデックス: {first_long_idx}")
    print(f"日時: {dates[first_long_idx]}")
    print(f"価格: ${closes[first_long_idx]:.2f}")
    print(f"現在のポジション: {current_position.position_type if current_position else 'なし'}")
    print(f"保留エントリー: {pending_entry}")
    print(f"保留エグジット: {pending_exit}")
    
    # ロングシグナルが無視される理由を分析
    if current_position is not None:
        print(f"\n❌ ロングシグナルが無視される理由:")
        print(f"  現在のポジション保有: {current_position.position_type}")
        print(f"  max_positions=1の制限により新しいエントリーがブロックされる")
        
        # このポジションはいつエグジットするか？
        print(f"\n🔍 現在のポジションのエグジット予測:")
        for j in range(first_long_idx + 1, min(first_long_idx + 50, len(data))):
            position_direction = -1 if current_position.position_type == 'SHORT' else 1
            exit_signal = strategy.generate_exit(data, position_direction, j)
            
            if exit_signal:
                print(f"  エグジットシグナル予測: インデックス {j} ({dates[j]})")
                print(f"  価格: ${closes[j]:.2f}")
                break
        else:
            print(f"  次の50バー以内にはエグジットシグナルが見つかりません")
    
    elif pending_entry is not None:
        print(f"\n❌ ロングシグナルが無視される理由:")
        print(f"  保留エントリーが存在: {pending_entry}")
        print(f"  新しいエントリーシグナルは保留エントリーがクリアされるまでブロックされる")
    
    else:
        print(f"\n✅ ロングシグナルが処理される可能性あり")
        print(f"  ポジションなし、保留エントリーなし")
        
        # ポジションサイズ計算をテスト
        stop_loss_price = closes[first_long_idx] * 0.95
        lookback_start = max(0, first_long_idx - warmup_bars)
        historical_data = data.iloc[lookback_start:first_long_idx+1].copy()
        
        try:
            params = PositionSizingParams(
                entry_price=closes[first_long_idx],
                stop_loss_price=stop_loss_price,
                capital=current_capital,
                historical_data=historical_data
            )
            
            sizing_result = position_manager.calculate(params)
            position_size = sizing_result['position_size']
            
            print(f"  計算されたポジションサイズ: ${position_size:.2f}")
            if position_size > 0:
                print(f"  ✅ ロングエントリーが実行される可能性が高い")
            else:
                print(f"  ❌ ポジションサイズ0のためスキップされる")
        except Exception as e:
            print(f"  ❌ ポジションサイジングエラー: {str(e)}")
    
    return relevant_events

def main():
    """メイン関数"""
    print("ロングシグナル発生タイミング詳細分析")
    print("=" * 60)
    
    try:
        events = analyze_long_signal_timing()
        
        print(f"\n{'='*60}")
        print(f"分析完了: {len(events)}個のイベントを分析")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"エラー: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()