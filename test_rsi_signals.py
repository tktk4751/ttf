#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import yaml
import pandas as pd
import numpy as np
from datetime import datetime

from data.data_loader import DataLoader
from data.data_processor import DataProcessor
from signals.entry_signal import RSIEntrySignal
from signals.exit_signal import RSIExitSignal
from indicators.rsi import RSI

def load_config(file_path: str = 'config.yaml') -> dict:
    """設定ファイルを読み込む"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    """RSIシグナルのテスト"""
    
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
    
    # RSI値の計算
    rsi_2 = RSI(period=2)
    rsi_14 = RSI(period=14)
    rsi_2_values = rsi_2.calculate(data)
    rsi_14_values = rsi_14.calculate(data)
    
    # シグナルの生成
    entry_signal = RSIEntrySignal(period=2)
    exit_signal = RSIExitSignal(period=14)
    entry_signals = entry_signal.generate(data)
    exit_signals = exit_signal.generate(data)
    
    # 結果の表示用のDataFrame作成
    result = pd.DataFrame({
        'datetime': data.index,
        'close': data['close'],
        'RSI(2)': rsi_2_values,
        'RSI(14)': rsi_14_values,
        'entry_signal': entry_signals,
        'exit_signal': exit_signals
    })
    
    # シグナルが発生した箇所のみを抽出
    signal_points = result[
        (result['entry_signal'] != 0) | 
        (result['exit_signal'] != 0)
    ].copy()
    
    print("\n=== RSIシグナルの発生ポイント ===")
    print("エントリーシグナル: 1=ロング, -1=ショート")
    print("エグジットシグナル: 1=ロングエグジット, -1=ショートエグジット")
    print(signal_points.to_string())
    
    # シグナルの統計
    print("\n=== エントリーシグナルの統計 ===")
    print(f"総データ数: {len(entry_signals)}")
    print(f"ロングエントリー数: {len(entry_signals[entry_signals == 1])}")
    print(f"ショートエントリー数: {len(entry_signals[entry_signals == -1])}")
    
    print("\n=== エグジットシグナルの統計 ===")
    print(f"総データ数: {len(exit_signals)}")
    print(f"ロングエグジット数: {len(exit_signals[exit_signals == 1])}")
    print(f"ショートエグジット数: {len(exit_signals[exit_signals == -1])}")
    
    # RSIの統計
    print("\n=== RSI統計 ===")
    print("RSI(2):")
    print(f"最小値: {min(rsi_2_values):.2f}")
    print(f"最大値: {max(rsi_2_values):.2f}")
    print(f"平均値: {np.mean(rsi_2_values):.2f}")
    print("\nRSI(14):")
    print(f"最小値: {min(rsi_14_values):.2f}")
    print(f"最大値: {max(rsi_14_values):.2f}")
    print(f"平均値: {np.mean(rsi_14_values):.2f}")

if __name__ == '__main__':
    main() 