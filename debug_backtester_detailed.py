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

def debug_backtester_entry_logic():
    """バックテスターのエントリーロジックを詳細にデバッグ"""
    print("=== バックテスターエントリーロジック詳細デバッグ ===")
    
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
    
    print(f"✓ データ準備完了: {len(data)}行")
    print(f"✓ ストラテジー: {strategy.name}")
    print(f"✓ ポジションマネージャー: {type(position_manager).__name__}")
    
    # エントリーシグナルを生成
    entry_signals = strategy.generate_entry(data)
    
    # シグナル統計
    long_signal_indices = np.where(entry_signals == 1)[0]
    short_signal_indices = np.where(entry_signals == -1)[0]
    
    print(f"\n✓ シグナル統計:")
    print(f"  ロングシグナル: {len(long_signal_indices)} ({long_signal_indices[:10].tolist()}...)")
    print(f"  ショートシグナル: {len(short_signal_indices)} ({short_signal_indices[:10].tolist()}...)")
    
    # バックテスターのエントリーロジックをシミュレート
    dates = data.index
    closes = data['close'].values
    warmup_bars = 100
    initial_balance = 10000.0
    current_capital = initial_balance
    commission = 0.001
    
    print(f"\n=== バックテスターエントリーシミュレーション ===")
    print(f"ウォームアップ期間: {warmup_bars}バー")
    print(f"初期資金: ${initial_balance}")
    print(f"手数料: {commission}")
    
    # 最初の数個のロング・ショートシグナルを詳細分析
    analysis_indices = []
    
    # 最初の5個のロングシグナル
    for idx in long_signal_indices[:5]:
        if idx >= warmup_bars:
            analysis_indices.append((idx, 'LONG', 1))
    
    # 最初の5個のショートシグナル  
    for idx in short_signal_indices[:5]:
        if idx >= warmup_bars:
            analysis_indices.append((idx, 'SHORT', -1))
    
    # インデックス順でソート
    analysis_indices.sort(key=lambda x: x[0])
    
    print(f"\n=== 詳細エントリー分析 (最初の10シグナル) ===")
    
    for i, (idx, signal_type, signal_value) in enumerate(analysis_indices[:10]):
        print(f"\n[{i+1}] インデックス {idx}: {signal_type}シグナル")
        print(f"  日時: {dates[idx]}")
        print(f"  価格: ${closes[idx]:.2f}")
        print(f"  エントリーシグナル値: {entry_signals[idx]}")
        
        # ポジションサイジングをテスト
        try:
            # ストップロス価格の計算
            stop_loss_price = closes[idx] * 0.95 if signal_value == 1 else closes[idx] * 1.05
            
            # 過去データの準備
            lookback_start = max(0, idx - warmup_bars)
            historical_data = data.iloc[lookback_start:idx+1].copy()
            
            if len(historical_data) >= warmup_bars:
                # position_manager.calculateがある場合
                if hasattr(position_manager, 'calculate'):
                    from position_sizing.position_sizing import PositionSizingParams
                    
                    params = PositionSizingParams(
                        entry_price=closes[idx],
                        stop_loss_price=stop_loss_price,
                        capital=current_capital,
                        historical_data=historical_data
                    )
                    
                    sizing_result = position_manager.calculate(params)
                    position_size = sizing_result['position_size']
                    
                    print(f"  ポジションサイジング結果:")
                    print(f"    計算方法: calculate()メソッド")
                    print(f"    ポジションサイズ: ${position_size:.2f}")
                    print(f"    サイジング詳細: {sizing_result}")
                    
                    # エントリー条件の詳細確認
                    can_enter = position_manager.can_enter() if hasattr(position_manager, 'can_enter') else True
                    print(f"    エントリー可能: {can_enter}")
                    
                    if position_size > 0 and can_enter:
                        print(f"  ✅ エントリー条件満たし: pending_entry = ('{signal_type}', {position_size:.2f}, {idx})")
                    else:
                        print(f"  ❌ エントリー条件不満足:")
                        if position_size <= 0:
                            print(f"    理由: ポジションサイズが0以下")
                        if not can_enter:
                            print(f"    理由: エントリー不可状態")
                
                else:
                    # calculate_position_sizeメソッドがある場合
                    if hasattr(position_manager, 'calculate_position_size'):
                        position_size = position_manager.calculate_position_size(
                            current_capital, closes[idx], historical_data
                        )
                        print(f"  ポジションサイジング結果:")
                        print(f"    計算方法: calculate_position_size()メソッド")
                        print(f"    ポジションサイズ: ${position_size:.2f}")
                    else:
                        print(f"  ❌ ポジションサイジング方法が不明")
            else:
                print(f"  ❌ 履歴データ不足: {len(historical_data)} < {warmup_bars}")
        
        except Exception as e:
            print(f"  ❌ ポジションサイジングエラー: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # XATRPositionSizingの詳細調査
    print(f"\n=== XATRPositionSizing詳細調査 ===")
    print(f"クラス: {position_manager.__class__}")
    print(f"メソッド一覧: {[m for m in dir(position_manager) if not m.startswith('_')]}")
    
    # テスト用のシンプルなポジション計算
    test_idx = long_signal_indices[0] if len(long_signal_indices) > 0 else 500
    test_price = closes[test_idx]
    test_data = data.iloc[max(0, test_idx-warmup_bars):test_idx+1]
    
    print(f"\nテスト計算:")
    print(f"  インデックス: {test_idx}")
    print(f"  価格: ${test_price:.2f}")
    print(f"  履歴データ長: {len(test_data)}")
    
    if hasattr(position_manager, 'calculate_position_size'):
        test_size = position_manager.calculate_position_size(current_capital, test_price, test_data)
        print(f"  計算結果: ${test_size:.2f}")
    
    return analysis_indices

def main():
    """メイン関数"""
    print("バックテスターエントリーロジック詳細調査")
    print("=" * 60)
    
    try:
        analysis_indices = debug_backtester_entry_logic()
        
        print(f"\n{'='*60}")
        print(f"調査完了: {len(analysis_indices)}個のシグナルを分析")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"エラー: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()