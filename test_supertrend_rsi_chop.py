#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import yaml
import pandas as pd
import numpy as np
from datetime import datetime

from data.data_loader import DataLoader
from data.data_processor import DataProcessor
from strategies.supertrend_rsi_chopstrategy import SupertrendRsiChopStrategy

def load_config(file_path: str = 'config.yaml') -> dict:
    """設定ファイルを読み込む"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    """スーパートレンドRSIチョピネス戦略のテスト"""
    
    # 設定の読み込み
    config = load_config()
    
    # データの設定を取得
    data_config = config.get('data', {})
    data_dir = data_config.get('data_dir', 'data')
    symbol = data_config.get('symbol', 'BTCUSDT')
    timeframe = data_config.get('timeframe', '1h')
    start_date = data_config.get('start')
    end_date = data_config.get('end')
    
    # 日付文字列をdatetimeオブジェクトに変換
    start_dt = datetime.strptime(start_date, '%Y-%m-%d') if start_date else None
    end_dt = datetime.strptime(end_date, '%Y-%m-%d') if end_date else None
    
    # データの読み込み
    loader = DataLoader(data_dir)
    processor = DataProcessor()
    
    data = loader.load_data(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_dt,
        end_date=end_dt
    )
    data = processor.process(data)
    
    # 戦略の初期化（カスタムパラメータを設定可能）
    strategy = SupertrendRsiChopStrategy(
        supertrend_params={
            'period': 10,
            'multiplier': 3.0
        },
        rsi_entry_params={
            'period': 2,
            'solid': {
                'rsi_long_entry': 20,
                'rsi_short_entry': 80
            }
        },
        rsi_exit_params={
            'period': 14,
            'solid': {
                'rsi_long_exit_solid': 70,
                'rsi_short_exit_solid': 30
            }
        },
        chop_params={
            'period': 14,
            'solid': {
                'chop_solid': 50
            }
        }
    )
    
    # エントリーシグナルの生成
    entry_signals = strategy.generate_entry(data)
    
    # シミュレーション用の変数
    position = 0  # 0: ニュートラル, 1: ロング, -1: ショート
    positions = np.zeros(len(data))  # ポジション履歴
    entry_prices = np.zeros(len(data))  # エントリー価格履歴
    pnl = np.zeros(len(data))  # 損益履歴
    
    # ポジションのシミュレーション
    for i in range(len(data)):
        current_price = data['close'].iloc[i]
        
        # エグジットシグナルをチェック（現在のポジションがある場合）
        if position != 0 and strategy.generate_exit(data, position, i):
            # ポジションのクローズ
            if position == 1:  # ロング
                pnl[i] = (current_price - entry_prices[i-1]) / entry_prices[i-1] * 100
            else:  # ショート
                pnl[i] = (entry_prices[i-1] - current_price) / entry_prices[i-1] * 100
            position = 0
            entry_prices[i] = 0
        
        # エントリーシグナルをチェック（現在のポジションがない場合）
        elif position == 0 and entry_signals[i] != 0:
            position = entry_signals[i]
            entry_prices[i] = current_price
        
        # 現在のポジションを記録
        positions[i] = position
        
        # エントリー価格を引き継ぐ
        if i > 0 and entry_prices[i] == 0 and position != 0:
            entry_prices[i] = entry_prices[i-1]
    
    # 結果の表示用のDataFrame作成
    result = pd.DataFrame({
        'datetime': data.index,
        'close': data['close'],
        'entry_signal': entry_signals,
        'position': positions,
        'entry_price': entry_prices,
        'pnl': pnl
    })
    
    # シグナルまたはポジションが変化した箇所のみを抽出
    signal_changes = result[
        (result['entry_signal'] != 0) | 
        (result['position'] != result['position'].shift(1)) |
        (result['pnl'] != 0)
    ].copy()
    
    print("\n=== スーパートレンドRSIチョピネス戦略のシグナル ===")
    print("エントリーシグナル: 1=ロング, -1=ショート")
    print("ポジション: 1=ロング, -1=ショート, 0=ニュートラル")
    print(signal_changes.to_string())
    
    # シグナルの統計
    print("\n=== シグナル統計 ===")
    print(f"総データ数: {len(entry_signals)}")
    print(f"ロングエントリーシグナル数: {len(entry_signals[entry_signals == 1])}")
    print(f"ショートエントリーシグナル数: {len(entry_signals[entry_signals == -1])}")
    
    # ポジションの統計
    print("\n=== ポジション統計 ===")
    print(f"ロングポジション数: {len(positions[positions == 1])}")
    print(f"ショートポジション数: {len(positions[positions == -1])}")
    print(f"ニュートラル数: {len(positions[positions == 0])}")
    
    # 損益統計
    trades = signal_changes[signal_changes['pnl'] != 0]
    profitable_trades = trades[trades['pnl'] > 0]
    losing_trades = trades[trades['pnl'] < 0]
    
    if len(trades) > 0:
        print("\n=== 損益統計 ===")
        print(f"総トレード数: {len(trades)}")
        print(f"勝率: {len(profitable_trades) / len(trades) * 100:.2f}%")
        if len(profitable_trades) > 0:
            print(f"平均利益: {profitable_trades['pnl'].mean():.2f}%")
            print(f"最大利益: {profitable_trades['pnl'].max():.2f}%")
        if len(losing_trades) > 0:
            print(f"平均損失: {losing_trades['pnl'].mean():.2f}%")
            print(f"最大損失: {losing_trades['pnl'].min():.2f}%")
        print(f"総損益: {trades['pnl'].sum():.2f}%")
    else:
        print("\n=== 損益統計 ===")
        print("トレードが実行されていません")

if __name__ == '__main__':
    main() 