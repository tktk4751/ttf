#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

# データ関連
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource

# ハイパーFRAMAシグナル
from signals.implementations.hyper_frama.entry import (
    HyperFRAMAPositionEntrySignal,
    HyperFRAMACrossoverEntrySignal
)


def load_test_data(config_path: str = 'config.yaml') -> pd.DataFrame:
    """テスト用データを読み込む"""
    with open(config_path, 'r', encoding='utf-8') as f:
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
    
    print("データを読み込み・処理中...")
    raw_data = data_loader.load_data_from_config(config)
    processed_data = {
        symbol: data_processor.process(df)
        for symbol, df in raw_data.items()
    }
    
    # 最初のシンボルのデータを取得
    first_symbol = next(iter(processed_data))
    data = processed_data[first_symbol]
    
    print(f"データ読み込み完了: {first_symbol}")
    print(f"期間: {data.index.min()} → {data.index.max()}")
    print(f"データ数: {len(data)}")
    
    return data


def test_position_signal(data: pd.DataFrame):
    """位置関係シグナルのテスト"""
    print("\n=== 位置関係シグナルのテスト ===")
    
    # 位置関係シグナルの作成（複数パラメータでテスト）
    signal_configs = [
        {'alpha_multiplier': 0.3, 'period': 16},
        {'alpha_multiplier': 0.5, 'period': 16},
        {'alpha_multiplier': 0.7, 'period': 20}
    ]
    
    for i, config in enumerate(signal_configs):
        print(f"\n--- 設定 {i+1}: alpha_mult={config['alpha_multiplier']}, period={config['period']} ---")
        
        signal = HyperFRAMAPositionEntrySignal(
            alpha_multiplier=config['alpha_multiplier'],
            period=config['period'],
            src_type='hl2'
        )
        
        # シグナル生成
        signals = signal.generate(data)
        
        # 統計の表示
        long_signals = (signals == 1).sum()
        short_signals = (signals == -1).sum()
        neutral_signals = (signals == 0).sum()
        
        print(f"総シグナル数: {len(signals)}")
        print(f"ロングシグナル: {long_signals} ({long_signals/len(signals)*100:.1f}%)")
        print(f"ショートシグナル: {short_signals} ({short_signals/len(signals)*100:.1f}%)")
        print(f"ニュートラル: {neutral_signals} ({neutral_signals/len(signals)*100:.1f}%)")
        
        # FRAMA値の統計
        frama_values = signal.get_frama_values()
        adjusted_frama_values = signal.get_adjusted_frama_values()
        
        if frama_values is not None and adjusted_frama_values is not None:
            valid_indices = ~(np.isnan(frama_values) | np.isnan(adjusted_frama_values))
            if valid_indices.any():
                frama_mean = np.mean(frama_values[valid_indices])
                adjusted_mean = np.mean(adjusted_frama_values[valid_indices])
                print(f"FRAMA平均: {frama_mean:.4f}")
                print(f"Adjusted FRAMA平均: {adjusted_mean:.4f}")
                print(f"平均差: {frama_mean - adjusted_mean:.4f}")


def test_crossover_signal(data: pd.DataFrame):
    """クロスオーバーシグナルのテスト"""
    print("\n=== クロスオーバーシグナルのテスト ===")
    
    # クロスオーバーシグナルの作成（複数パラメータでテスト）
    signal_configs = [
        {'alpha_multiplier': 0.3, 'period': 16},
        {'alpha_multiplier': 0.5, 'period': 16},
        {'alpha_multiplier': 0.7, 'period': 20}
    ]
    
    for i, config in enumerate(signal_configs):
        print(f"\n--- 設定 {i+1}: alpha_mult={config['alpha_multiplier']}, period={config['period']} ---")
        
        signal = HyperFRAMACrossoverEntrySignal(
            alpha_multiplier=config['alpha_multiplier'],
            period=config['period'],
            src_type='hl2'
        )
        
        # シグナル生成
        signals = signal.generate(data)
        
        # 統計の表示
        long_signals = (signals == 1).sum()
        short_signals = (signals == -1).sum()
        neutral_signals = (signals == 0).sum()
        
        print(f"総シグナル数: {len(signals)}")
        print(f"ロングクロス: {long_signals} ({long_signals/len(signals)*100:.1f}%)")
        print(f"ショートクロス: {short_signals} ({short_signals/len(signals)*100:.1f}%)")
        print(f"シグナルなし: {neutral_signals} ({neutral_signals/len(signals)*100:.1f}%)")
        
        # シグナル頻度の確認
        signal_points = long_signals + short_signals
        print(f"クロスオーバー頻度: {signal_points} / {len(signals)} ({signal_points/len(signals)*100:.2f}%)")


def create_comparison_chart(data: pd.DataFrame, save_path: str = "hyper_frama_signals_comparison.png"):
    """位置関係とクロスオーバーシグナルの比較チャート"""
    print("\n=== シグナル比較チャートの作成 ===")
    
    # 最近のデータのみを使用（表示のため）
    recent_data = data.tail(200)
    
    # シグナルの作成
    position_signal = HyperFRAMAPositionEntrySignal(
        alpha_multiplier=0.5,
        period=16,
        src_type='hl2'
    )
    
    crossover_signal = HyperFRAMACrossoverEntrySignal(
        alpha_multiplier=0.5,
        period=16,
        src_type='hl2'
    )
    
    # シグナル生成
    position_signals = position_signal.generate(recent_data)
    crossover_signals = crossover_signal.generate(recent_data)
    
    # FRAMA値の取得
    frama_values = position_signal.get_frama_values()
    adjusted_frama_values = position_signal.get_adjusted_frama_values()
    
    # 最近のデータに絞る
    frama_recent = frama_values[-len(recent_data):]
    adjusted_recent = adjusted_frama_values[-len(recent_data):]
    
    # チャートの作成
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # 1. 価格とFRAMAライン
    ax1.plot(recent_data.index, recent_data['close'], label='Close Price', color='black', alpha=0.7)
    ax1.plot(recent_data.index, frama_recent, label='FRAMA', color='blue', linewidth=2)
    ax1.plot(recent_data.index, adjusted_recent, label='Adjusted FRAMA', color='red', linewidth=2)
    ax1.set_title('HyperFRAMA Lines')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 位置関係シグナル
    ax2.plot(recent_data.index, position_signals, label='Position Signal', color='green', alpha=0.8)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.axhline(y=1, color='green', linestyle='--', alpha=0.5)
    ax2.axhline(y=-1, color='red', linestyle='--', alpha=0.5)
    ax2.set_title('Position Relationship Signals')
    ax2.set_ylabel('Signal')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. クロスオーバーシグナル
    # クロスオーバーポイントのみ表示
    long_cross_indices = np.where(crossover_signals == 1)[0]
    short_cross_indices = np.where(crossover_signals == -1)[0]
    
    ax3.scatter(recent_data.index[long_cross_indices], 
               np.ones(len(long_cross_indices)), 
               color='green', marker='^', s=100, label='Long Cross', alpha=0.8)
    ax3.scatter(recent_data.index[short_cross_indices], 
               -np.ones(len(short_cross_indices)), 
               color='red', marker='v', s=100, label='Short Cross', alpha=0.8)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.set_title('Crossover Signals')
    ax3.set_ylabel('Signal')
    ax3.set_xlabel('Date')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-1.5, 1.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"チャートを保存しました: {save_path}")
    
    # 統計情報
    print(f"\n最近200データポイントの統計:")
    print(f"ロング位置関係: {(position_signals == 1).sum()}")
    print(f"ショート位置関係: {(position_signals == -1).sum()}")
    print(f"ロングクロス: {len(long_cross_indices)}")
    print(f"ショートクロス: {len(short_cross_indices)}")


def main():
    """メイン関数"""
    try:
        # テストデータの読み込み
        data = load_test_data()
        
        # 位置関係シグナルのテスト
        test_position_signal(data)
        
        # クロスオーバーシグナルのテスト
        test_crossover_signal(data)
        
        # 比較チャートの作成
        create_comparison_chart(data)
        
        print("\n=== テスト完了 ===")
        
    except Exception as e:
        print(f"テスト中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()