#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from indicators.grand_cycle_ma import GrandCycleMA
from data.binance_data_source import BinanceDataSource

def test_grand_cycle_ma():
    """グランドサイクルMAのテスト"""
    print("=== グランドサイクルMA テスト開始 ===")
    
    # テストデータの準備
    print("1. テストデータの準備中...")
    data_source = BinanceDataSource()
    
    try:
        # SOL/USDTの4時間足データを取得
        data = data_source.load_data(
            symbol='SOLUSDT',
            timeframe='4h',
            limit=500,
            data_type='spot'
        )
        print(f"データ取得成功: {len(data)}件")
        print(f"データ期間: {data.index[0]} ~ {data.index[-1]}")
        
    except Exception as e:
        print(f"データ取得エラー: {e}")
        # フォールバック: サンプルデータを作成
        print("サンプルデータを作成中...")
        dates = pd.date_range('2024-01-01', periods=200, freq='4H')
        np.random.seed(42)
        
        # トレンドとサイクルを含む価格データを生成
        trend = np.linspace(100, 150, 200)
        cycle = 10 * np.sin(np.linspace(0, 8*np.pi, 200))
        noise = np.random.normal(0, 2, 200)
        close_prices = trend + cycle + noise
        
        data = pd.DataFrame({
            'open': close_prices * 0.995,
            'high': close_prices * 1.005,
            'low': close_prices * 0.99,
            'close': close_prices,
            'volume': np.random.uniform(1000000, 5000000, 200)
        }, index=dates)
        print(f"サンプルデータ作成: {len(data)}件")
    
    # 2. 利用可能な検出器一覧の表示
    print("\n2. 利用可能なサイクル検出器:")
    available_detectors = GrandCycleMA.get_available_detectors()
    for detector, description in available_detectors.items():
        print(f"  - {detector}: {description}")
    
    # 3. 複数の検出器でテスト
    test_detectors = ['hody', 'phac', 'dudi', 'cycle_period', 'practical']
    results = {}
    
    print("\n3. 各検出器でのグランドサイクルMA計算テスト:")
    
    for detector in test_detectors:
        if detector in available_detectors:
            try:
                print(f"\n--- {detector} ({available_detectors[detector]}) ---")
                
                # グランドサイクルMAの作成
                grand_cycle_ma = GrandCycleMA(
                    detector_type=detector,
                    fast_limit=0.5,
                    slow_limit=0.05,
                    src_type='hlc3',
                    cycle_part=0.5,
                    max_cycle=50,
                    min_cycle=6
                )
                
                # 計算実行
                result = grand_cycle_ma.calculate(data)
                
                # 結果の保存
                results[detector] = {
                    'grand_mama': result.grand_mama_values,
                    'grand_fama': result.grand_fama_values,
                    'cycle_period': result.cycle_period,
                    'alpha': result.alpha_values,
                    'phase': result.phase_values
                }
                
                # 統計情報
                valid_mama = result.grand_mama_values[~np.isnan(result.grand_mama_values)]
                valid_fama = result.grand_fama_values[~np.isnan(result.grand_fama_values)]
                valid_cycle = result.cycle_period[~np.isnan(result.cycle_period)]
                
                print(f"  グランドMAMA - 有効データ数: {len(valid_mama)}")
                if len(valid_mama) > 0:
                    print(f"    平均値: {np.mean(valid_mama):.4f}")
                    print(f"    標準偏差: {np.std(valid_mama):.4f}")
                
                print(f"  グランドFAMA - 有効データ数: {len(valid_fama)}")
                if len(valid_fama) > 0:
                    print(f"    平均値: {np.mean(valid_fama):.4f}")
                    print(f"    標準偏差: {np.std(valid_fama):.4f}")
                
                print(f"  サイクル周期 - 有効データ数: {len(valid_cycle)}")
                if len(valid_cycle) > 0:
                    print(f"    平均周期: {np.mean(valid_cycle):.2f}")
                    print(f"    周期範囲: {np.min(valid_cycle):.2f} - {np.max(valid_cycle):.2f}")
                
                print(f"  ✓ {detector} 計算成功")
                
            except Exception as e:
                print(f"  ✗ {detector} 計算エラー: {e}")
                continue
        else:
            print(f"  ⚠ {detector} 検出器が利用できません")
    
    # 4. 視覚化
    if results:
        print("\n4. 結果の視覚化中...")
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # 元の価格データ
        axes[0].plot(data.index, data['close'], label='Close Price', alpha=0.7, color='black')
        
        # グランドサイクルMAMAの比較
        for i, (detector, result_data) in enumerate(results.items()):
            color = plt.cm.tab10(i)
            valid_indices = ~np.isnan(result_data['grand_mama'])
            if np.any(valid_indices):
                axes[0].plot(
                    data.index[valid_indices], 
                    result_data['grand_mama'][valid_indices],
                    label=f'Grand MAMA ({detector})',
                    alpha=0.8,
                    color=color
                )
        
        axes[0].set_title('Price vs Grand Cycle MAMA (各検出器比較)')
        axes[0].set_ylabel('Price')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # サイクル周期の比較
        for i, (detector, result_data) in enumerate(results.items()):
            color = plt.cm.tab10(i)
            valid_indices = ~np.isnan(result_data['cycle_period'])
            if np.any(valid_indices):
                axes[1].plot(
                    data.index[valid_indices],
                    result_data['cycle_period'][valid_indices],
                    label=f'Cycle Period ({detector})',
                    alpha=0.8,
                    color=color
                )
        
        axes[1].set_title('Cycle Period Comparison (サイクル周期比較)')
        axes[1].set_ylabel('Period')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # アルファ値の比較
        for i, (detector, result_data) in enumerate(results.items()):
            color = plt.cm.tab10(i)
            valid_indices = ~np.isnan(result_data['alpha'])
            if np.any(valid_indices):
                axes[2].plot(
                    data.index[valid_indices],
                    result_data['alpha'][valid_indices],
                    label=f'Alpha ({detector})',
                    alpha=0.8,
                    color=color
                )
        
        axes[2].set_title('Alpha Values Comparison (アルファ値比較)')
        axes[2].set_ylabel('Alpha')
        axes[2].set_xlabel('Date')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 画像を保存
        output_file = 'grand_cycle_ma_test_result.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  グラフを保存しました: {output_file}")
        
        plt.show()
    
    # 5. パフォーマンステスト
    print("\n5. パフォーマンステスト:")
    if results:
        import time
        
        # 最初に成功した検出器でパフォーマンステスト
        test_detector = list(results.keys())[0]
        grand_cycle_ma = GrandCycleMA(
            detector_type=test_detector,
            fast_limit=0.5,
            slow_limit=0.05,
            src_type='hlc3'
        )
        
        # 10回実行して平均時間を測定
        times = []
        for i in range(10):
            start_time = time.time()
            grand_cycle_ma.calculate(data)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        print(f"  平均計算時間 ({test_detector}): {avg_time:.4f}秒")
        print(f"  データ長: {len(data)}件")
        print(f"  処理速度: {len(data)/avg_time:.0f}件/秒")
    
    print("\n=== グランドサイクルMA テスト完了 ===")


def test_detector_parameters():
    """各検出器の固有パラメータテスト"""
    print("\n=== 検出器固有パラメータテスト ===")
    
    # サンプルデータ作成
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    np.random.seed(42)
    close_prices = 100 + np.cumsum(np.random.normal(0, 0.5, 100))
    
    data = pd.DataFrame({
        'open': close_prices * 0.999,
        'high': close_prices * 1.001,
        'low': close_prices * 0.998,
        'close': close_prices,
        'volume': np.random.uniform(100000, 500000, 100)
    }, index=dates)
    
    # 特定の検出器で特殊パラメータをテスト
    test_cases = [
        {
            'detector': 'bandpass_zero',
            'params': {'bandwidth': 0.3, 'center_period': 20.0},
            'description': 'バンドパスゼロクロッシング（狭帯域）'
        },
        {
            'detector': 'dft_dominant',
            'params': {'window': 30},
            'description': 'DFTドミナント（短いウィンドウ）'
        },
        {
            'detector': 'cycle_period',
            'params': {'alpha': 0.1},
            'description': 'サイクル周期（高アルファ）'
        }
    ]
    
    for test_case in test_cases:
        try:
            print(f"\n--- {test_case['description']} ---")
            
            grand_cycle_ma = GrandCycleMA(
                detector_type=test_case['detector'],
                **test_case['params']
            )
            
            result = grand_cycle_ma.calculate(data)
            
            valid_data = ~np.isnan(result.grand_mama_values)
            print(f"  有効データ率: {np.sum(valid_data)/len(valid_data)*100:.1f}%")
            
            if np.any(valid_data):
                print(f"  グランドMAMA範囲: {np.min(result.grand_mama_values[valid_data]):.2f} - {np.max(result.grand_mama_values[valid_data]):.2f}")
            
            print(f"  ✓ {test_case['detector']} パラメータテスト成功")
            
        except Exception as e:
            print(f"  ✗ {test_case['detector']} パラメータテストエラー: {e}")


if __name__ == "__main__":
    test_grand_cycle_ma()
    test_detector_parameters()