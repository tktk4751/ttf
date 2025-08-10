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

def step_by_step_backtester_trace():
    """バックテスターの実行を1ステップずつ完全トレース"""
    print("=== バックテスター 1ステップずつ完全トレース ===")
    
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
    
    # バックテスター設定
    dates = data.index
    closes = data['close'].values
    warmup_bars = 100
    initial_balance = 10000.0
    current_capital = initial_balance
    commission = 0.001
    max_positions = 1
    
    print(f"✓ データ準備完了: {len(data)}行, シンボル: {first_symbol}")
    print(f"✓ 初期資金: ${initial_balance}")
    print(f"✓ 手数料: {commission}")
    print(f"✓ 最大同時ポジション: {max_positions}")
    
    # エントリーシグナル生成
    entry_signals = strategy.generate_entry(data)
    
    # シグナル統計
    long_signal_indices = np.where(entry_signals == 1)[0]
    short_signal_indices = np.where(entry_signals == -1)[0]
    
    print(f"\n✓ シグナル統計:")
    print(f"  ロングシグナル: {len(long_signal_indices)}個")
    print(f"  ショートシグナル: {len(short_signal_indices)}個")
    
    # 最初の有効なロングとショートシグナルを特定
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
    
    print(f"\n✓ 最初の有効シグナル:")
    print(f"  最初のロング: インデックス {first_long_idx}")
    print(f"  最初のショート: インデックス {first_short_idx}")
    
    # どちらが先に来るかを判定
    if first_long_idx is not None and first_short_idx is not None:
        if first_long_idx < first_short_idx:
            first_signal_idx = first_long_idx
            first_signal_type = "LONG"
            first_signal_value = 1
        else:
            first_signal_idx = first_short_idx
            first_signal_type = "SHORT"
            first_signal_value = -1
    elif first_long_idx is not None:
        first_signal_idx = first_long_idx
        first_signal_type = "LONG"
        first_signal_value = 1
    elif first_short_idx is not None:
        first_signal_idx = first_short_idx
        first_signal_type = "SHORT"
        first_signal_value = -1
    else:
        print("❌ 有効なシグナルが見つかりません")
        return
    
    print(f"\n=== 最初のシグナル詳細分析: {first_signal_type} @ インデックス {first_signal_idx} ===")
    
    # バックテスター状態変数
    current_position = None
    pending_entry = None
    pending_exit = False
    trades = []
    
    # 最初のシグナルまでバックテスターを実行
    print(f"\n=== バックテスターステップ実行 (インデックス {warmup_bars} から {first_signal_idx+5} まで) ===")
    
    executed_trades = []
    
    for i in range(warmup_bars, min(first_signal_idx + 10, len(data))):
        date = dates[i]
        close = closes[i]
        
        print(f"\n--- ステップ {i} ---")
        print(f"日時: {date}")
        print(f"価格: ${close:.2f}")
        print(f"エントリーシグナル: {entry_signals[i]}")
        print(f"現在のポジション: {current_position.position_type if current_position else 'なし'}")
        print(f"保留エントリー: {pending_entry}")
        print(f"保留エグジット: {pending_exit}")
        
        # 1. 保留中のエグジットの処理
        if pending_exit and current_position is not None:
            print(f"  🔸 保留エグジット処理開始")
            
            # エグジット実行
            current_position.close(date, close, current_capital)
            trades.append(current_position)
            current_capital = current_position.balance
            
            print(f"  ✅ エグジット完了: {current_position.position_type}")
            print(f"  💰 PnL: ${current_position.profit_loss:.2f}")
            print(f"  💼 新しい資金: ${current_capital:.2f}")
            
            executed_trades.append({
                'index': i,
                'type': 'EXIT',
                'position_type': current_position.position_type,
                'pnl': current_position.profit_loss,
                'capital': current_capital
            })
            
            current_position = None
            pending_exit = False
        
        # 2. 保留中のエントリーの処理
        if pending_entry is not None and current_position is None:
            position_type, position_size, entry_index = pending_entry
            
            print(f"  🔸 保留エントリー処理開始")
            print(f"    タイプ: {position_type}")
            print(f"    サイズ: ${position_size:.2f}")
            print(f"    エントリーインデックス: {entry_index}")
            
            # エントリー可能かチェック
            can_enter = position_manager.can_enter()
            print(f"    エントリー可能: {can_enter}")
            
            if can_enter and position_size > 0:
                # トレード作成とエントリー
                current_position = Trade(
                    position_type=position_type,
                    position_size=position_size,
                    commission_rate=commission,
                    slippage_rate=0.001
                )
                current_position.symbol = first_symbol
                current_position.entry(date, close)
                
                print(f"  ✅ エントリー完了: {position_type} @ ${current_position.entry_price:.2f}")
                print(f"  📊 ポジションサイズ: ${position_size:.2f}")
                
                executed_trades.append({
                    'index': i,
                    'type': 'ENTRY',
                    'position_type': position_type,
                    'entry_price': current_position.entry_price,
                    'position_size': position_size
                })
                
                pending_entry = None
            else:
                print(f"  ❌ エントリー失敗:")
                if not can_enter:
                    print(f"    理由: can_enter() = False")
                if position_size <= 0:
                    print(f"    理由: position_size = {position_size}")
        
        # 3. 現在のポジションがある場合、エグジットシグナルをチェック
        if current_position is not None and not pending_exit:
            position_direction = -1 if current_position.position_type == 'SHORT' else 1
            exit_signal = strategy.generate_exit(data, position_direction, i)
            
            if exit_signal:
                print(f"  🔸 エグジットシグナル検出")
                print(f"    現在のポジション: {current_position.position_type}")
                print(f"    エグジット条件満たし -> pending_exit = True")
                pending_exit = True
        
        # 4. 新しいエントリーシグナルの処理
        if current_position is None and not pending_entry:
            signal_value = entry_signals[i]
            
            if signal_value != 0:
                signal_type = "LONG" if signal_value == 1 else "SHORT"
                print(f"  🔸 新しいエントリーシグナル検出: {signal_type}")
                
                # ポジションサイズ計算
                stop_loss_price = close * 0.95 if signal_value == 1 else close * 1.05
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
                        
                        print(f"    ポジションサイズ計算: ${position_size:.2f}")
                        print(f"    現在の資金: ${current_capital:.2f}")
                        print(f"    計算詳細: {sizing_result}")
                        
                        if position_size > 0:
                            pending_entry = (signal_type, position_size, i)
                            print(f"  ✅ pending_entry設定: {pending_entry}")
                        else:
                            print(f"  ❌ ポジションサイズ0につきスキップ")
                    
                    except Exception as e:
                        print(f"  ❌ ポジションサイジングエラー: {str(e)}")
                else:
                    print(f"  ❌ 履歴データ不足: {len(historical_data)} < {warmup_bars}")
    
    print(f"\n=== ステップトレース完了 ===")
    print(f"実行されたトレードイベント: {len(executed_trades)}")
    
    for i, trade_event in enumerate(executed_trades):
        print(f"  [{i+1}] インデックス {trade_event['index']}: {trade_event['type']} - {trade_event['position_type']}")
        if trade_event['type'] == 'ENTRY':
            print(f"      エントリー価格: ${trade_event['entry_price']:.2f}")
            print(f"      ポジションサイズ: ${trade_event['position_size']:.2f}")
        elif trade_event['type'] == 'EXIT':
            print(f"      PnL: ${trade_event['pnl']:.2f}")
            print(f"      残り資金: ${trade_event['capital']:.2f}")
    
    # 実際のトレード結果確認
    long_trades = [t for t in trades if t.position_type == 'LONG']
    short_trades = [t for t in trades if t.position_type == 'SHORT']
    
    print(f"\n=== 最終トレード統計 ===")
    print(f"総トレード: {len(trades)}")
    print(f"ロングトレード: {len(long_trades)}")
    print(f"ショートトレード: {len(short_trades)}")
    
    if len(long_trades) == 0:
        print(f"⚠️ ロングトレードが実行されていません")
        print(f"最初のシグナルタイプ: {first_signal_type}")
        if first_signal_type == "LONG":
            print(f"最初がロングシグナルなのにロングトレードが実行されていません")
    
    return executed_trades, trades

def main():
    """メイン関数"""
    print("バックテスター ステップバイステップ 完全トレース")
    print("=" * 70)
    
    try:
        executed_trades, trades = step_by_step_backtester_trace()
        
        print(f"\n{'='*70}")
        print(f"分析完了")
        print(f"  実行イベント: {len(executed_trades)}")
        print(f"  完了トレード: {len(trades)}")
        print(f"{'='*70}")
        
    except Exception as e:
        print(f"エラー: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()