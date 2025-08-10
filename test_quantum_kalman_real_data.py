#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
量子適応カルマンフィルターのリアルデータテスト
"""

import sys
import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from indicators.kalman.quantum_adaptive_kalman import QuantumAdaptiveKalman
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource


def load_config(config_path: str = "config.yaml") -> dict:
    """設定ファイルを読み込む"""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        return {}


def main():
    print("=== 量子適応カルマンフィルター - リアルデータテスト ===")
    
    # 設定ファイルの読み込み
    config = load_config()
    
    try:
        # データの準備
        binance_config = config.get('binance_data', {})
        data_dir = binance_config.get('data_dir', 'data/binance')
        binance_data_source = BinanceDataSource(data_dir)
        
        # CSVデータソースはダミーとして渡す
        dummy_csv_source = CSVDataSource("dummy")
        data_loader = DataLoader(
            data_source=dummy_csv_source,
            binance_data_source=binance_data_source
        )
        data_processor = DataProcessor()
        
        # データの読み込みと処理
        print("\nデータを読み込み・処理中...")
        raw_data = data_loader.load_data_from_config(config)
        processed_data = {
            symbol: data_processor.process(df)
            for symbol, df in raw_data.items()
        }
        
        # 最初のシンボルのデータを取得
        first_symbol = next(iter(processed_data))
        data = processed_data[first_symbol]
        
        # データ数を制限（最新の300ポイント）
        data = data.tail(300)
        
        print(f"データ読み込み完了: {first_symbol}")
        print(f"期間: {data.index.min()} → {data.index.max()}")
        print(f"データ数: {len(data)}")
        print(f"価格範囲: {data['close'].min():.2f} - {data['close'].max():.2f}")
        
        # 量子適応カルマンフィルターのテスト
        print("\n量子適応カルマンフィルターを実行中...")
        
        # 異なる設定でテスト
        configs = [
            {'base_process_noise': 0.001, 'amplitude_window': 14, 'name': 'デフォルト'},
            {'base_process_noise': 0.0005, 'amplitude_window': 10, 'name': '高感度'},
            {'base_process_noise': 0.002, 'amplitude_window': 20, 'name': '低感度'}
        ]
        
        results = {}
        for config_test in configs:
            name = config_test.pop('name')
            quantum_kalman = QuantumAdaptiveKalman(src_type='close', **config_test)
            result = quantum_kalman.calculate(data)
            results[name] = result
            
            print(f"\n{name}設定:")
            print(f"  フィルタリング値範囲: {np.nanmin(result.values):.2f} - {np.nanmax(result.values):.2f}")
            print(f"  量子コヒーレンス範囲: {np.nanmin(result.quantum_coherence):.4f} - {np.nanmax(result.quantum_coherence):.4f}")
            print(f"  平均カルマンゲイン: {np.nanmean(result.kalman_gains):.4f}")
            print(f"  平均信頼度スコア: {np.nanmean(result.confidence_scores):.4f}")
        
        # 結果の可視化
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        # 価格とフィルタリング結果の比較
        axes[0].plot(data.index, data['close'], alpha=0.7, label='実際の価格', color='blue', linewidth=1)
        
        colors = ['red', 'green', 'purple']
        for i, (name, result) in enumerate(results.items()):
            axes[0].plot(data.index, result.values, label=f'量子フィルター（{name}）', 
                        color=colors[i], linewidth=2, alpha=0.8)
        
        axes[0].set_title(f'{first_symbol} - 量子適応カルマンフィルター比較')
        axes[0].set_ylabel('価格')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 量子コヒーレンスの比較
        for i, (name, result) in enumerate(results.items()):
            axes[1].plot(data.index, result.quantum_coherence, label=f'コヒーレンス（{name}）', 
                        color=colors[i], linewidth=1.5, alpha=0.8)
        
        axes[1].set_title('量子コヒーレンス比較')
        axes[1].set_ylabel('コヒーレンス値')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # カルマンゲインの比較
        for i, (name, result) in enumerate(results.items()):
            axes[2].plot(data.index, result.kalman_gains, label=f'ゲイン（{name}）', 
                        color=colors[i], linewidth=1.5, alpha=0.8)
        
        axes[2].set_title('カルマンゲイン比較')
        axes[2].set_ylabel('ゲイン値')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # 信頼度スコアの比較
        for i, (name, result) in enumerate(results.items()):
            axes[3].plot(data.index, result.confidence_scores, label=f'信頼度（{name}）', 
                        color=colors[i], linewidth=1.5, alpha=0.8)
        
        axes[3].set_title('信頼度スコア比較')
        axes[3].set_ylabel('信頼度')
        axes[3].set_xlabel('時間')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        # レイアウト調整と保存
        plt.tight_layout()
        save_path = f"quantum_kalman_real_data_{first_symbol}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nチャートを保存しました: {save_path}")
        
        # 統計分析
        print("\n=== 統計分析 ===")
        for name, result in results.items():
            original_std = np.std(data['close'])
            filtered_std = np.nanstd(result.values)
            noise_reduction = (1 - filtered_std / original_std) * 100
            
            # 追跡精度（MAE）
            mae = np.nanmean(np.abs(result.values - data['close'].values))
            
            print(f"\n{name}設定:")
            print(f"  ノイズ削減率: {noise_reduction:.2f}%")
            print(f"  追跡誤差(MAE): {mae:.4f}")
            print(f"  平均量子コヒーレンス: {np.nanmean(result.quantum_coherence):.4f}")
        
        plt.show()
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n=== テスト完了 ===")
    return 0


if __name__ == "__main__":
    exit(main())