#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import yaml
import pandas as pd
from datetime import datetime

from data.data_loader import DataLoader
from data.data_processor import DataProcessor
from signals.direction_signal import SupertrendDirectionSignal

def load_config(file_path: str = 'config.yaml') -> dict:
    """設定ファイルを読み込む"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    """スーパートレンドシグナルのテスト"""
    
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
    
    # スーパートレンドシグナルの生成
    signal = SupertrendDirectionSignal(period=10, multiplier=3.0)
    signals = signal.generate(data)
    
    # 結果の表示
    result = pd.DataFrame({
        'datetime': data.index,
        'close': data['close'],
        'signal': signals
    })
    
    # シグナルが変化した箇所のみを表示
    signal_changes = result[result['signal'] != 0].copy()
    
    print("\n=== スーパートレンドシグナルの変化 ===")
    print("1: 上昇トレンド（買い）, -1: 下降トレンド（売り）")
    print(signal_changes.to_string())
    
    # シグナルの統計
    print("\n=== シグナルの統計 ===")
    print(f"総データ数: {len(signals)}")
    print(f"買いシグナル数: {len(signals[signals == 1])}")
    print(f"売りシグナル数: {len(signals[signals == -1])}")
    print(f"シグナルなし: {len(signals[signals == 0])}")

if __name__ == '__main__':
    main() 