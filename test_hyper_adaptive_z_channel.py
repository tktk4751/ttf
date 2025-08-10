#!/usr/bin/env python3
"""
HyperAdaptiveChannelのz_adaptive_channel動的適応機能をテストするスクリプト
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# データ読み込み
from data.binance_data_source import BinanceDataSource
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor

# インジケーター
from indicators.hyper_adaptive_channel import HyperAdaptiveChannel

def load_test_data():
    """テスト用データを読み込む"""
    print("サンプルデータを生成中...")
    
    # サンプルデータを生成
    dates = pd.date_range(start='2024-01-01', end='2024-06-30', freq='4H')
    n = len(dates)
    
    # 価格データをシミュレート
    np.random.seed(42)  # 再現性のため
    price_base = 50000  # 基準価格
    
    # ランダムウォークで価格を生成
    returns = np.random.normal(0, 0.02, n)  # 2%の標準偏差
    cumulative_returns = np.cumsum(returns)
    prices = price_base * np.exp(cumulative_returns)
    
    # OHLCV データを生成
    high_factor = np.random.uniform(1.005, 1.02, n)  # 0.5%-2%の上振れ
    low_factor = np.random.uniform(0.98, 0.995, n)   # 0.5%-2%の下振れ
    
    data = pd.DataFrame({
        'open': prices,
        'high': prices * high_factor,
        'low': prices * low_factor,
        'close': prices,
        'volume': np.random.uniform(100, 1000, n)
    }, index=dates)
    
    print(f"サンプルデータ生成完了")
    print(f"期間: {data.index.min()} → {data.index.max()}")
    print(f"データ数: {len(data)}")
    print(f"価格範囲: {data['close'].min():.2f} - {data['close'].max():.2f}")
    
    return data

def test_z_adaptive_modes():
    """z_adaptive_channelの各モードをテスト"""
    
    print("\\n=== HyperAdaptiveChannel z_adaptive機能テスト ===")
    
    # テストデータ読み込み
    data = load_test_data()
    
    # テストケース
    test_cases = [
        {
            'name': 'Z_ADAPTIVE + CER',
            'multiplier_mode': 'z_adaptive',
            'z_multiplier_source': 'cer',
            'z_multiplier_method': 'adaptive'
        },
        {
            'name': 'Z_SIMPLE + CER + X_TREND',
            'multiplier_mode': 'z_simple',
            'z_multiplier_source': 'cer',
            'z_multiplier_method': 'simple'
        },
        {
            'name': 'Z_SIMPLE_ADJUSTMENT + X_TREND',
            'multiplier_mode': 'z_simple_adjustment',
            'z_multiplier_source': 'x_trend',
            'z_multiplier_method': 'simple_adjustment'
        }
    ]
    
    results = {}
    
    for test_case in test_cases:
        print(f"\\n--- {test_case['name']} をテスト中 ---")
        
        try:
            # HyperAdaptiveChannelを初期化
            hyper_channel = HyperAdaptiveChannel(
                period=14,
                midline_smoother="hyper_frama",
                multiplier_mode=test_case['multiplier_mode'],
                src_type="hlc3",
                
                # z_adaptive_channel固有パラメータ
                z_multiplier_method=test_case['z_multiplier_method'],
                z_multiplier_source=test_case['z_multiplier_source'],
                z_max_multiplier=3.0,
                z_min_multiplier=1.0,
                z_enable_multiplier_smoothing=True,
                z_multiplier_smoother_type='alma',
                z_multiplier_smoother_period=8
            )
            
            # 計算実行
            print("計算を実行中...")
            result = hyper_channel.calculate(data)
            
            # 結果確認
            midline = result.midline
            upper_band = result.upper_band
            lower_band = result.lower_band
            multiplier_values = result.multiplier_values
            
            # 統計情報
            valid_count = (~np.isnan(midline)).sum()
            multiplier_stats = pd.Series(multiplier_values[~np.isnan(multiplier_values)])
            
            print(f"✓ 計算成功")
            print(f"  有効データ数: {valid_count}/{len(data)}")
            print(f"  乗数統計 - 平均: {multiplier_stats.mean():.3f}, 範囲: {multiplier_stats.min():.3f} - {multiplier_stats.max():.3f}")
            print(f"  ミッドライン範囲: {pd.Series(midline).dropna().min():.2f} - {pd.Series(midline).dropna().max():.2f}")
            
            results[test_case['name']] = {
                'success': True,
                'result': result,
                'multiplier_stats': multiplier_stats.describe()
            }
            
        except Exception as e:
            print(f"✗ エラー: {e}")
            results[test_case['name']] = {
                'success': False,
                'error': str(e)
            }
    
    return results

def create_comparison_chart(data, results):
    """結果の比較チャート作成"""
    
    print("\\n比較チャートを作成中...")
    
    # 成功したテストケースのみを対象
    successful_results = {name: res for name, res in results.items() if res['success']}
    
    if len(successful_results) == 0:
        print("成功したテストケースがありません")
        return
    
    # データ期間を絞り込み（最新500本）
    plot_data = data.tail(500).copy()
    
    fig, axes = plt.subplots(len(successful_results) + 1, 1, figsize=(14, 6 * (len(successful_results) + 1)))
    if len(successful_results) == 0:
        axes = [axes]
    
    # 価格チャート
    ax_price = axes[0]
    ax_price.plot(plot_data.index, plot_data['close'], label='BTC Price', color='black', linewidth=1)
    ax_price.set_title('BTC/USDT 4H Price Chart')
    ax_price.set_ylabel('Price (USDT)')
    ax_price.legend()
    ax_price.grid(True, alpha=0.3)
    
    # 各テストケースのチャネル
    for i, (name, result_data) in enumerate(successful_results.items()):
        ax = axes[i + 1]
        
        result = result_data['result']
        
        # データを絞り込み
        start_idx = len(data) - len(plot_data)
        midline = result.midline[start_idx:]
        upper_band = result.upper_band[start_idx:]
        lower_band = result.lower_band[start_idx:]
        multiplier_values = result.multiplier_values[start_idx:]
        
        # 価格
        ax.plot(plot_data.index, plot_data['close'], label='Price', color='black', linewidth=1, alpha=0.7)
        
        # チャネル
        ax.plot(plot_data.index, midline, label='Midline', color='blue', linewidth=2)
        ax.plot(plot_data.index, upper_band, label='Upper Band', color='red', linewidth=1.5, alpha=0.8)
        ax.plot(plot_data.index, lower_band, label='Lower Band', color='green', linewidth=1.5, alpha=0.8)
        
        # 塗りつぶし
        ax.fill_between(plot_data.index, upper_band, lower_band, alpha=0.1, color='blue')
        
        ax.set_title(f'{name} - Dynamic Channel')
        ax.set_ylabel('Price (USDT)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 乗数プロット（右軸）
        ax2 = ax.twinx()
        ax2.plot(plot_data.index, multiplier_values, label='Multiplier', color='orange', linewidth=1, alpha=0.8)
        ax2.set_ylabel('Dynamic Multiplier', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
    
    plt.tight_layout()
    
    # 保存
    output_file = 'hyper_adaptive_z_channel_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"チャートを保存しました: {output_file}")
    
    plt.show()

if __name__ == "__main__":
    # テスト実行
    results = test_z_adaptive_modes()
    
    # 結果サマリー
    print("\\n=== テスト結果サマリー ===")
    for name, result in results.items():
        if result['success']:
            print(f"✓ {name}: 成功")
            print(f"  乗数統計: {result['multiplier_stats']['mean']:.3f} ± {result['multiplier_stats']['std']:.3f}")
        else:
            print(f"✗ {name}: 失敗 - {result['error']}")
    
    # チャート作成
    if any(res['success'] for res in results.values()):
        # テストデータを再取得
        data = load_test_data()
        create_comparison_chart(data, results)
    
    print("\\nテスト完了")