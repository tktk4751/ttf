#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from indicators.roc_persistence import ROCPersistence


def create_test_data(n_points: int = 1000) -> pd.DataFrame:
    """
    テスト用のサンプルデータを作成（トレンドシフトあり）
    
    Args:
        n_points: データポイント数
        
    Returns:
        OHLC価格データ
    """
    np.random.seed(42)
    
    time = np.arange(n_points)
    
    # 複数のトレンド期間を作成
    base_price = 100
    prices = []
    
    # セグメント1: 上昇トレンド（0-300）
    if n_points > 300:
        trend1 = np.linspace(0, 20, 300)  # 20ポイント上昇
        cycle1 = 3 * np.sin(2 * np.pi * np.arange(300) / 25)
        noise1 = np.random.normal(0, 1, 300)
        segment1 = base_price + trend1 + cycle1 + noise1
        prices.extend(segment1)
        
        # セグメント2: 下降トレンド（300-600）
        if n_points > 600:
            trend2 = np.linspace(20, -10, 300)  # 30ポイント下降
            cycle2 = 2 * np.sin(2 * np.pi * np.arange(300) / 30)
            noise2 = np.random.normal(0, 1, 300)
            segment2 = base_price + trend2 + cycle2 + noise2
            prices.extend(segment2)
            
            # セグメント3: 横ばい（600-800）
            if n_points > 800:
                trend3 = np.ones(200) * (-10)  # 横ばい
                cycle3 = 4 * np.sin(2 * np.pi * np.arange(200) / 20)
                noise3 = np.random.normal(0, 1.5, 200)
                segment3 = base_price + trend3 + cycle3 + noise3
                prices.extend(segment3)
                
                # セグメント4: 残りの期間（800-end）
                remaining = n_points - 800
                if remaining > 0:
                    trend4 = np.linspace(-10, 15, remaining)  # 再び上昇
                    cycle4 = 2.5 * np.sin(2 * np.pi * np.arange(remaining) / 35)
                    noise4 = np.random.normal(0, 1, remaining)
                    segment4 = base_price + trend4 + cycle4 + noise4
                    prices.extend(segment4)
    
    # 配列の長さを調整
    while len(prices) < n_points:
        prices.append(prices[-1] + np.random.normal(0, 1))
    prices = prices[:n_points]
    
    # OHLCデータを生成
    close_prices = np.array(prices)
    high_prices = close_prices + np.abs(np.random.normal(0, 0.5, n_points))
    low_prices = close_prices - np.abs(np.random.normal(0, 0.5, n_points))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]
    
    # 出来高を追加
    volume = np.random.randint(1000, 10000, n_points)
    
    # DataFrameを作成
    data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    })
    
    # 日付インデックスを追加
    start_date = pd.Timestamp('2023-01-01')
    data.index = pd.date_range(start=start_date, periods=n_points, freq='D')
    
    return data


def test_roc_persistence_basic():
    """基本的なROC継続性のテスト"""
    print("=== ROC継続性基本テスト ===")
    
    # テストデータを作成
    data = create_test_data(800)
    print(f"テストデータ作成完了: {len(data)}行")
    
    # ROC継続性インジケーターを作成
    roc_persistence = ROCPersistence(
        detector_type='dudi_e',
        max_persistence_periods=144,
        smooth_persistence=True,
        persistence_smooth_period=3,
        src_type='close',
        smooth_roc=True,
        roc_alma_period=5
    )
    
    # 計算実行
    print("ROC継続性を計算中...")
    persistence_values = roc_persistence.calculate(data)
    result = roc_persistence.get_result()
    
    print(f"計算完了: {len(persistence_values)}個の値")
    print(f"有効値数: {np.sum(~np.isnan(persistence_values))}")
    print(f"継続性値の範囲: {np.nanmin(persistence_values):.4f} ～ {np.nanmax(persistence_values):.4f}")
    print(f"継続性値の平均: {np.nanmean(persistence_values):.4f}")
    print(f"継続性値の標準偏差: {np.nanstd(persistence_values):.4f}")
    
    # 方向別の分析
    roc_directions = roc_persistence.get_roc_directions()
    positive_periods = np.sum(roc_directions == 1)
    negative_periods = np.sum(roc_directions == -1)
    zero_periods = np.sum(roc_directions == 0)
    print(f"正のROC期間: {positive_periods}回")
    print(f"負のROC期間: {negative_periods}回")
    print(f"ゼロのROC期間: {zero_periods}回")
    
    # 継続期間の分析
    persistence_periods = roc_persistence.get_persistence_periods()
    max_positive_persistence = 0
    max_negative_persistence = 0
    
    current_direction = 0
    current_count = 0
    
    for i, direction in enumerate(roc_directions):
        if direction == current_direction:
            current_count += 1
        else:
            if current_direction == 1:
                max_positive_persistence = max(max_positive_persistence, current_count)
            elif current_direction == -1:
                max_negative_persistence = max(max_negative_persistence, current_count)
            current_direction = direction
            current_count = 1
    
    print(f"最大正継続期間: {max_positive_persistence}")
    print(f"最大負継続期間: {max_negative_persistence}")
    
    return data, roc_persistence, persistence_values, result


def test_different_max_periods():
    """異なる最大期間のテスト"""
    print("\n=== 異なる最大期間のテスト ===")
    
    # テストデータを作成
    data = create_test_data(500)
    
    # 異なる最大期間でテスト
    max_periods = [50, 100, 144, 200]
    results = {}
    
    for max_period in max_periods:
        print(f"\n最大期間: {max_period}")
        
        roc_persistence = ROCPersistence(
            detector_type='dudi_e',
            max_persistence_periods=max_period,
            smooth_persistence=True,
            src_type='close'
        )
        
        persistence_values = roc_persistence.calculate(data)
        
        print(f"  継続性範囲: {np.nanmin(persistence_values):.4f} ～ {np.nanmax(persistence_values):.4f}")
        print(f"  継続性平均: {np.nanmean(persistence_values):.4f}")
        print(f"  継続性標準偏差: {np.nanstd(persistence_values):.4f}")
        
        # 極値（1や-1に近い値）の頻度
        extreme_positive = np.sum(persistence_values > 0.9)
        extreme_negative = np.sum(persistence_values < -0.9)
        print(f"  極大値(>0.9): {extreme_positive}回")
        print(f"  極小値(<-0.9): {extreme_negative}回")
        
        results[max_period] = {
            'values': persistence_values,
            'indicator': roc_persistence
        }
    
    return data, results


def test_different_smoothing():
    """異なる平滑化設定のテスト"""
    print("\n=== 異なる平滑化設定のテスト ===")
    
    # テストデータを作成
    data = create_test_data(400)
    
    # 異なる平滑化設定でテスト
    settings = [
        {'smooth_persistence': False, 'name': '平滑化なし'},
        {'smooth_persistence': True, 'persistence_smooth_period': 3, 'name': '3期間平滑化'},
        {'smooth_persistence': True, 'persistence_smooth_period': 5, 'name': '5期間平滑化'},
        {'smooth_persistence': True, 'persistence_smooth_period': 10, 'name': '10期間平滑化'}
    ]
    
    results = {}
    
    for setting in settings:
        print(f"\n設定: {setting['name']}")
        
        roc_persistence = ROCPersistence(
            detector_type='dudi_e',
            max_persistence_periods=100,
            smooth_persistence=setting['smooth_persistence'],
            persistence_smooth_period=setting.get('persistence_smooth_period', 3),
            src_type='close'
        )
        
        persistence_values = roc_persistence.calculate(data)
        
        print(f"  継続性平均: {np.nanmean(persistence_values):.4f}")
        print(f"  継続性標準偏差: {np.nanstd(persistence_values):.4f}")
        
        results[setting['name']] = {
            'values': persistence_values,
            'indicator': roc_persistence
        }
    
    return data, results


def plot_roc_persistence_analysis(data, roc_persistence, persistence_values):
    """ROC継続性の分析チャートを作成"""
    print("\n=== ROC継続性分析チャートを作成 ===")
    
    # チャートデータの準備
    result = roc_persistence.get_result()
    roc_values = result.roc_values
    persistence_periods = result.persistence_periods
    roc_directions = result.roc_directions
    
    # 直近300ポイントに絞る
    n_plot = min(300, len(data))
    plot_data = data.iloc[-n_plot:].copy()
    plot_persistence = persistence_values[-n_plot:]
    plot_roc = roc_values[-n_plot:]
    plot_periods = persistence_periods[-n_plot:]
    plot_directions = roc_directions[-n_plot:]
    
    # プロット作成
    fig, axes = plt.subplots(5, 1, figsize=(15, 14))
    fig.suptitle('ROC継続性分析', fontsize=16)
    
    # 1. 価格チャート
    axes[0].plot(plot_data.index, plot_data['close'], 'b-', linewidth=1, label='終値')
    axes[0].set_title('価格チャート')
    axes[0].set_ylabel('価格')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. ROC値
    axes[1].plot(plot_data.index, plot_roc, 'purple', linewidth=1, label='ROC')
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1].set_title('ROC値')
    axes[1].set_ylabel('ROC (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. ROC継続性（メイン指標）
    colors = ['red' if v < 0 else 'green' if v > 0 else 'gray' for v in plot_persistence]
    axes[2].plot(plot_data.index, plot_persistence, 'orange', linewidth=2, label='ROC継続性')
    axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[2].axhline(y=1, color='green', linestyle='--', alpha=0.5, label='最大正継続')
    axes[2].axhline(y=-1, color='red', linestyle='--', alpha=0.5, label='最大負継続')
    axes[2].fill_between(plot_data.index, 0, plot_persistence, 
                         where=np.array(plot_persistence) > 0, color='green', alpha=0.3)
    axes[2].fill_between(plot_data.index, 0, plot_persistence, 
                         where=np.array(plot_persistence) < 0, color='red', alpha=0.3)
    axes[2].set_title('ROC継続性 (-1から1)')
    axes[2].set_ylabel('継続性')
    axes[2].set_ylim(-1.1, 1.1)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # 4. 継続期間
    axes[3].plot(plot_data.index, plot_periods, 'green', linewidth=1, label='継続期間')
    axes[3].set_title('現在の継続期間')
    axes[3].set_ylabel('期間')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    # 5. ROC方向
    direction_colors = ['red' if d == -1 else 'green' if d == 1 else 'gray' for d in plot_directions]
    axes[4].scatter(plot_data.index, plot_directions, c=direction_colors, alpha=0.6, s=20)
    axes[4].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[4].set_title('ROC方向 (赤:負、緑:正、灰:ゼロ)')
    axes[4].set_ylabel('方向')
    axes[4].set_ylim(-1.5, 1.5)
    axes[4].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """メイン関数"""
    print("ROC継続性インジケーターのテストを開始します...\n")
    
    try:
        # 基本テスト
        data, roc_persistence, persistence_values, result = test_roc_persistence_basic()
        
        # 異なる最大期間のテスト
        test_different_max_periods()
        
        # 異なる平滑化設定のテスト
        test_different_smoothing()
        
        # 分析チャートの表示
        plot_roc_persistence_analysis(data, roc_persistence, persistence_values)
        
        print("\n=== テスト完了 ===")
        print("ROC継続性インジケーターは正常に動作しています。")
        print("\n主な特徴:")
        print("- ROCが正の領域に長くいると1に近づく")
        print("- ROCが負の領域に長くいると-1に近づく")
        print("- 144期間で飽和して自動的に±1になる")
        print("- サイクルROCベースで動的期間を使用")
        print("- Numba最適化で高速計算")
        
    except Exception as e:
        import traceback
        print(f"テスト中にエラーが発生しました: {str(e)}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main() 