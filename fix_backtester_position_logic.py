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

def test_improved_backtester_logic():
    """改良されたバックテスターロジックをテスト"""
    print("=== 改良されたバックテスターロジック テスト ===")
    
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
    
    print(f"✓ データ準備完了: {len(data)}行")
    print(f"✓ 初期資金: ${initial_balance}")
    
    # エントリーシグナル生成
    entry_signals = strategy.generate_entry(data)
    
    # シグナル統計
    long_signal_indices = np.where(entry_signals == 1)[0]
    short_signal_indices = np.where(entry_signals == -1)[0]
    
    print(f"✓ シグナル統計:")
    print(f"  ロングシグナル: {len(long_signal_indices)}個")
    print(f"  ショートシグナル: {len(short_signal_indices)}個")
    
    # 改良されたバックテスターロジック
    print(f"\n=== 改良されたバックテスターロジック ===")
    print(f"戦略変更: 反対シグナルでポジション転換を許可")
    
    current_position = None
    trades = []
    
    # 最初の10シグナルでテスト
    all_signals = []
    for idx in long_signal_indices[:5]:
        if idx >= warmup_bars:
            all_signals.append((idx, 'LONG', 1))
    for idx in short_signal_indices[:5]:
        if idx >= warmup_bars:
            all_signals.append((idx, 'SHORT', -1))
    
    all_signals.sort(key=lambda x: x[0])
    
    print(f"\n=== シグナル処理テスト (最初の10シグナル) ===")
    
    for i, (idx, signal_type, signal_value) in enumerate(all_signals[:10]):
        print(f"\n[{i+1}] インデックス {idx}: {signal_type}シグナル")
        print(f"  日時: {dates[idx]}")
        print(f"  価格: ${closes[idx]:.2f}")
        print(f"  現在のポジション: {current_position.position_type if current_position else 'なし'}")
        
        # 改良されたロジック: 反対シグナルでポジション転換
        if current_position is None:
            # ポジションなし -> 新規エントリー
            print(f"  🆕 新規エントリー処理")
            
            # ポジションサイズ計算
            stop_loss_price = closes[idx] * 0.95 if signal_value == 1 else closes[idx] * 1.05
            lookback_start = max(0, idx - warmup_bars)
            historical_data = data.iloc[lookback_start:idx+1].copy()
            
            if len(historical_data) >= warmup_bars:
                try:
                    params = PositionSizingParams(
                        entry_price=closes[idx],
                        stop_loss_price=stop_loss_price,
                        capital=current_capital,
                        historical_data=historical_data
                    )
                    
                    sizing_result = position_manager.calculate(params)
                    position_size = sizing_result['position_size']
                    
                    if position_size > 0:
                        current_position = Trade(
                            position_type=signal_type,
                            position_size=position_size,
                            commission_rate=commission,
                            slippage_rate=0.001
                        )
                        current_position.symbol = first_symbol
                        current_position.entry(dates[idx], closes[idx])
                        
                        print(f"  ✅ エントリー成功: {signal_type} @ ${current_position.entry_price:.2f}")
                        print(f"  📊 ポジションサイズ: ${position_size:.2f}")
                    else:
                        print(f"  ❌ ポジションサイズ0のためスキップ")
                        
                except Exception as e:
                    print(f"  ❌ ポジションサイジングエラー: {str(e)}")
            else:
                print(f"  ❌ 履歴データ不足")
        
        else:
            # ポジション保有中
            if current_position.position_type == signal_type:
                # 同じ方向のシグナル -> 無視
                print(f"  ⚪ 同じ方向のシグナル -> 無視")
                
            else:
                # 反対方向のシグナル -> ポジション転換
                print(f"  🔄 反対シグナル -> ポジション転換処理")
                
                # 既存ポジションをクローズ
                current_position.close(dates[idx], closes[idx], current_capital)
                trades.append(current_position)
                current_capital = current_position.balance
                
                print(f"  💰 既存ポジションクローズ: {current_position.position_type}")
                print(f"    PnL: ${current_position.profit_loss:.2f}")
                print(f"    新しい資金: ${current_capital:.2f}")
                
                # 新しいポジションをオープン
                stop_loss_price = closes[idx] * 0.95 if signal_value == 1 else closes[idx] * 1.05
                lookback_start = max(0, idx - warmup_bars)
                historical_data = data.iloc[lookback_start:idx+1].copy()
                
                if len(historical_data) >= warmup_bars:
                    try:
                        params = PositionSizingParams(
                            entry_price=closes[idx],
                            stop_loss_price=stop_loss_price,
                            capital=current_capital,
                            historical_data=historical_data
                        )
                        
                        sizing_result = position_manager.calculate(params)
                        position_size = sizing_result['position_size']
                        
                        if position_size > 0:
                            current_position = Trade(
                                position_type=signal_type,
                                position_size=position_size,
                                commission_rate=commission,
                                slippage_rate=0.001
                            )
                            current_position.symbol = first_symbol
                            current_position.entry(dates[idx], closes[idx])
                            
                            print(f"  ✅ 新規エントリー成功: {signal_type} @ ${current_position.entry_price:.2f}")
                            print(f"  📊 ポジションサイズ: ${position_size:.2f}")
                        else:
                            print(f"  ❌ 新規ポジションサイズ0のため転換失敗")
                            current_position = None
                            
                    except Exception as e:
                        print(f"  ❌ 新規ポジションサイジングエラー: {str(e)}")
                        current_position = None
    
    # 最後に残ったポジションをクローズ
    if current_position is not None:
        final_idx = all_signals[-1][0] + 1 if all_signals else warmup_bars + 50
        if final_idx < len(data):
            current_position.close(dates[final_idx], closes[final_idx], current_capital)
            trades.append(current_position)
            current_capital = current_position.balance
            print(f"\n💰 最終ポジションクローズ: {current_position.position_type}")
            print(f"  PnL: ${current_position.profit_loss:.2f}")
    
    # 結果分析
    print(f"\n=== 改良されたロジックの結果 ===")
    print(f"総トレード数: {len(trades)}")
    
    long_trades = [t for t in trades if t.position_type == 'LONG']
    short_trades = [t for t in trades if t.position_type == 'SHORT']
    
    print(f"ロングトレード: {len(long_trades)}個")
    print(f"ショートトレード: {len(short_trades)}個")
    
    print(f"\n=== トレード詳細 ===")
    for i, trade in enumerate(trades):
        print(f"  [{i+1}] {trade.position_type}: PnL ${trade.profit_loss:.2f}")
    
    total_pnl = sum(t.profit_loss for t in trades)
    final_balance = initial_balance + total_pnl
    
    print(f"\n=== 最終結果 ===")
    print(f"総PnL: ${total_pnl:.2f}")
    print(f"最終資金: ${final_balance:.2f}")
    print(f"リターン: {(final_balance/initial_balance-1)*100:.2f}%")
    
    if len(long_trades) > 0:
        print(f"✅ ロングトレードが正常に実行されました！")
    else:
        print(f"❌ ロングトレードがまだ実行されていません")
    
    return trades

def main():
    """メイン関数"""
    print("改良されたバックテスターロジック テスト")
    print("=" * 60)
    
    try:
        trades = test_improved_backtester_logic()
        
        print(f"\n{'='*60}")
        print(f"テスト完了: {len(trades)}個のトレードを実行")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"エラー: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()