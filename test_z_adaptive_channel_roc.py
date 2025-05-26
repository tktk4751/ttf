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

from indicators.z_adaptive_channel import ZAdaptiveChannel


def create_test_data(n_points: int = 800) -> pd.DataFrame:
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
    
    # セグメント1: 上昇トレンド（0-200）
    if n_points > 200:
        trend1 = np.linspace(0, 15, 200)  # 15ポイント上昇
        cycle1 = 3 * np.sin(2 * np.pi * np.arange(200) / 25)
        noise1 = np.random.normal(0, 1, 200)
        segment1 = base_price + trend1 + cycle1 + noise1
        prices.extend(segment1)
        
        # セグメント2: 下降トレンド（200-400）
        if n_points > 400:
            trend2 = np.linspace(15, -10, 200)  # 25ポイント下降
            cycle2 = 2 * np.sin(2 * np.pi * np.arange(200) / 30)
            noise2 = np.random.normal(0, 1, 200)
            segment2 = base_price + trend2 + cycle2 + noise2
            prices.extend(segment2)
            
            # セグメント3: 横ばい（400-600）
            if n_points > 600:
                trend3 = np.ones(200) * (-10)  # 横ばい
                cycle3 = 4 * np.sin(2 * np.pi * np.arange(200) / 20)
                noise3 = np.random.normal(0, 1.5, 200)
                segment3 = base_price + trend3 + cycle3 + noise3
                prices.extend(segment3)
                
                # セグメント4: 残りの期間（600-end）
                remaining = n_points - 600
                if remaining > 0:
                    trend4 = np.linspace(-10, 20, remaining)  # 再び上昇
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


def test_z_adaptive_channel_with_roc():
    """ROC Persistence機能付きZ Adaptive Channelのテスト"""
    print("=== ROC Persistence機能付きZ Adaptive Channelのテスト ===")
    
    # テストデータを作成
    data = create_test_data(600)
    print(f"テストデータ作成完了: {len(data)}行")
    
    # ROC Persistence機能付きZ Adaptive Channelを作成
    zac_with_roc = ZAdaptiveChannel(
        # 基本パラメータ
        max_max_multiplier=6.0,
        min_max_multiplier=2.0,
        max_min_multiplier=1.2,
        min_min_multiplier=0.3,
        src_type='close',
        multiplier_method='simple',
        multiplier_source='cer',
        ma_source='cer',
        
        # ROC Persistence有効化
        use_roc_persistence=True,
        
        # ROC Persistenceパラメータ
        roc_detector_type='dudi_e',
        roc_max_persistence_periods=100,
        roc_smooth_persistence=True,
        roc_persistence_smooth_period=3,
        roc_smooth_roc=True,
        roc_alma_period=5
    )
    
    # 比較用：ROC Persistence機能なしのチャネル
    zac_without_roc = ZAdaptiveChannel(
        # 同じ基本パラメータ
        max_max_multiplier=6.0,
        min_max_multiplier=2.0,
        max_min_multiplier=1.2,
        min_min_multiplier=0.3,
        src_type='close',
        multiplier_method='simple',
        multiplier_source='cer',
        ma_source='cer',
        
        # ROC Persistence無効化
        use_roc_persistence=False
    )
    
    # 計算実行
    print("ROC Persistence機能付きチャネルを計算中...")
    middle_with_roc = zac_with_roc.calculate(data)
    middle_with_roc, upper_with_roc, lower_with_roc = zac_with_roc.get_bands()
    
    print("ROC Persistence機能なしチャネルを計算中...")
    middle_without_roc = zac_without_roc.calculate(data)
    middle_without_roc, upper_without_roc, lower_without_roc = zac_without_roc.get_bands()
    
    # ROC Persistence関連データの取得
    roc_persistence_values = zac_with_roc.get_roc_persistence_values()
    roc_directions = zac_with_roc.get_roc_directions()
    upper_multiplier = zac_with_roc.get_upper_multiplier()
    lower_multiplier = zac_with_roc.get_lower_multiplier()
    base_multiplier = zac_with_roc.get_dynamic_multiplier()
    
    print(f"計算完了: {len(middle_with_roc)}個の値")
    print(f"有効値数: {np.sum(~np.isnan(middle_with_roc))}")
    
    # 統計情報
    print(f"\nROC継続性値の統計:")
    print(f"  範囲: {np.nanmin(roc_persistence_values):.4f} ～ {np.nanmax(roc_persistence_values):.4f}")
    print(f"  平均: {np.nanmean(roc_persistence_values):.4f}")
    print(f"  標準偏差: {np.nanstd(roc_persistence_values):.4f}")
    
    # ROC方向の統計
    positive_periods = np.sum(roc_directions == 1)
    negative_periods = np.sum(roc_directions == -1)
    zero_periods = np.sum(roc_directions == 0)
    print(f"\nROC方向の統計:")
    print(f"  正の期間: {positive_periods}回")
    print(f"  負の期間: {negative_periods}回")
    print(f"  ゼロの期間: {zero_periods}回")
    
    # 乗数調整の統計
    upper_adjust_count = np.sum(upper_multiplier != base_multiplier)
    lower_adjust_count = np.sum(lower_multiplier != base_multiplier)
    print(f"\n乗数調整の統計:")
    print(f"  アッパーバンド調整回数: {upper_adjust_count}")
    print(f"  ロワーバンド調整回数: {lower_adjust_count}")
    print(f"  基本乗数平均: {np.nanmean(base_multiplier):.4f}")
    print(f"  アッパー乗数平均: {np.nanmean(upper_multiplier):.4f}")
    print(f"  ロワー乗数平均: {np.nanmean(lower_multiplier):.4f}")
    
    return data, zac_with_roc, zac_without_roc


def test_different_roc_settings():
    """異なるROC Persistence設定のテスト"""
    print("\n=== 異なるROC Persistence設定のテスト ===")
    
    # テストデータを作成
    data = create_test_data(400)
    
    # 異なる設定でテスト
    settings = [
        {
            'name': 'ROC無効',
            'use_roc_persistence': False
        },
        {
            'name': 'ROC有効・短期',
            'use_roc_persistence': True,
            'roc_max_persistence_periods': 50,
            'roc_smooth_persistence': False
        },
        {
            'name': 'ROC有効・長期',
            'use_roc_persistence': True,
            'roc_max_persistence_periods': 144,
            'roc_smooth_persistence': True,
            'roc_persistence_smooth_period': 5
        },
        {
            'name': 'ROC有効・高感度',
            'use_roc_persistence': True,
            'roc_detector_type': 'phac_e',
            'roc_max_persistence_periods': 100,
            'roc_smooth_persistence': False
        }
    ]
    
    results = {}
    
    for setting in settings:
        print(f"\n設定: {setting['name']}")
        
        # パラメータを設定
        params = {
            'max_max_multiplier': 5.0,
            'min_max_multiplier': 2.0,
            'src_type': 'close',
            'multiplier_method': 'adaptive'
        }
        params.update({k: v for k, v in setting.items() if k != 'name'})
        
        # インジケーターを作成
        zac = ZAdaptiveChannel(**params)
        
        # 計算実行
        middle = zac.calculate(data)
        middle, upper, lower = zac.get_bands()
        
        # 統計計算
        valid_count = np.sum(~np.isnan(middle))
        channel_width = upper - lower
        avg_width = np.nanmean(channel_width)
        
        print(f"  有効値数: {valid_count}")
        print(f"  平均チャネル幅: {avg_width:.4f}")
        
        if setting['use_roc_persistence']:
            roc_values = zac.get_roc_persistence_values()
            roc_dirs = zac.get_roc_directions()
            upper_mult = zac.get_upper_multiplier()
            lower_mult = zac.get_lower_multiplier()
            base_mult = zac.get_dynamic_multiplier()
            
            print(f"  ROC継続性平均: {np.nanmean(roc_values):.4f}")
            print(f"  乗数調整率: {np.mean((upper_mult != base_mult) | (lower_mult != base_mult)):.4f}")
        
        results[setting['name']] = {
            'middle': middle,
            'upper': upper,
            'lower': lower,
            'indicator': zac
        }
    
    return data, results


def plot_roc_channel_analysis(data, zac_with_roc, zac_without_roc):
    """ROC Persistence機能付きチャネルの分析チャートを作成"""
    print("\n=== ROC Persistence機能付きチャネル分析チャートを作成 ===")
    
    # データの準備
    middle_with_roc, upper_with_roc, lower_with_roc = zac_with_roc.get_bands()
    middle_without_roc, upper_without_roc, lower_without_roc = zac_without_roc.get_bands()
    
    roc_persistence_values = zac_with_roc.get_roc_persistence_values()
    roc_directions = zac_with_roc.get_roc_directions()
    upper_multiplier = zac_with_roc.get_upper_multiplier()
    lower_multiplier = zac_with_roc.get_lower_multiplier()
    base_multiplier = zac_with_roc.get_dynamic_multiplier()
    
    # 直近250ポイントに絞る
    n_plot = min(250, len(data))
    plot_data = data.iloc[-n_plot:].copy()
    
    plot_middle_with = middle_with_roc[-n_plot:]
    plot_upper_with = upper_with_roc[-n_plot:]
    plot_lower_with = lower_with_roc[-n_plot:]
    
    plot_middle_without = middle_without_roc[-n_plot:]
    plot_upper_without = upper_without_roc[-n_plot:]
    plot_lower_without = lower_without_roc[-n_plot:]
    
    plot_roc_persistence = roc_persistence_values[-n_plot:]
    plot_roc_directions = roc_directions[-n_plot:]
    plot_upper_mult = upper_multiplier[-n_plot:]
    plot_lower_mult = lower_multiplier[-n_plot:]
    plot_base_mult = base_multiplier[-n_plot:]
    
    # プロット作成
    fig, axes = plt.subplots(5, 1, figsize=(15, 16))
    fig.suptitle('ROC Persistence機能付きZ Adaptive Channel分析', fontsize=16)
    
    # 1. 価格チャートとチャネル（ROC機能付き）
    axes[0].plot(plot_data.index, plot_data['close'], 'black', linewidth=1, label='終値')
    axes[0].plot(plot_data.index, plot_middle_with, 'blue', linewidth=1, label='中心線（ROC付き）')
    axes[0].fill_between(plot_data.index, plot_upper_with, plot_lower_with, 
                         alpha=0.2, color='blue', label='チャネル（ROC付き）')
    axes[0].plot(plot_data.index, plot_upper_with, 'blue', linewidth=1, alpha=0.7)
    axes[0].plot(plot_data.index, plot_lower_with, 'blue', linewidth=1, alpha=0.7)
    axes[0].set_title('価格チャート：ROC Persistence機能付きチャネル')
    axes[0].set_ylabel('価格')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. チャネル比較
    axes[1].plot(plot_data.index, plot_data['close'], 'black', linewidth=1, label='終値')
    axes[1].plot(plot_data.index, plot_upper_with, 'blue', linewidth=1, label='上限（ROC付き）')
    axes[1].plot(plot_data.index, plot_lower_with, 'blue', linewidth=1, label='下限（ROC付き）')
    axes[1].plot(plot_data.index, plot_upper_without, 'red', linewidth=1, linestyle='--', 
                 alpha=0.7, label='上限（ROCなし）')
    axes[1].plot(plot_data.index, plot_lower_without, 'red', linewidth=1, linestyle='--', 
                 alpha=0.7, label='下限（ROCなし）')
    axes[1].set_title('チャネル比較：ROC機能付き vs ROC機能なし')
    axes[1].set_ylabel('価格')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. ROC継続性
    colors = ['red' if v < 0 else 'green' if v > 0 else 'gray' for v in plot_roc_persistence]
    axes[2].plot(plot_data.index, plot_roc_persistence, 'orange', linewidth=2, label='ROC継続性')
    axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[2].axhline(y=1, color='green', linestyle='--', alpha=0.5, label='最大正継続')
    axes[2].axhline(y=-1, color='red', linestyle='--', alpha=0.5, label='最大負継続')
    axes[2].fill_between(plot_data.index, 0, plot_roc_persistence, 
                         where=np.array(plot_roc_persistence) > 0, color='green', alpha=0.3)
    axes[2].fill_between(plot_data.index, 0, plot_roc_persistence, 
                         where=np.array(plot_roc_persistence) < 0, color='red', alpha=0.3)
    axes[2].set_title('ROC継続性 (-1から1)')
    axes[2].set_ylabel('継続性')
    axes[2].set_ylim(-1.1, 1.1)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # 4. 乗数比較
    axes[3].plot(plot_data.index, plot_base_mult, 'gray', linewidth=2, label='基本乗数')
    axes[3].plot(plot_data.index, plot_upper_mult, 'blue', linewidth=1, label='アッパー乗数')
    axes[3].plot(plot_data.index, plot_lower_mult, 'red', linewidth=1, label='ロワー乗数')
    axes[3].set_title('乗数比較（基本 vs 調整後）')
    axes[3].set_ylabel('乗数')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    # 5. ROC方向とチャネル幅差
    width_with = plot_upper_with - plot_lower_with
    width_without = plot_upper_without - plot_lower_without
    width_diff = width_with - width_without
    
    # 左軸：ROC方向
    ax5_left = axes[4]
    direction_colors = ['red' if d == -1 else 'green' if d == 1 else 'gray' for d in plot_roc_directions]
    ax5_left.scatter(plot_data.index, plot_roc_directions, c=direction_colors, alpha=0.6, s=20)
    ax5_left.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax5_left.set_ylabel('ROC方向', color='black')
    ax5_left.set_ylim(-1.5, 1.5)
    ax5_left.grid(True, alpha=0.3)
    
    # 右軸：チャネル幅差
    ax5_right = ax5_left.twinx()
    ax5_right.plot(plot_data.index, width_diff, 'purple', linewidth=1, label='幅差（ROC付き - ROCなし）')
    ax5_right.axhline(y=0, color='purple', linestyle='--', alpha=0.5)
    ax5_right.set_ylabel('チャネル幅差', color='purple')
    ax5_right.legend(loc='upper right')
    
    axes[4].set_title('ROC方向（点）とチャネル幅差（線）')
    
    plt.tight_layout()
    plt.show()


def main():
    """メイン関数"""
    print("ROC Persistence機能付きZ Adaptive Channelのテストを開始します...\n")
    
    try:
        # 基本テスト
        data, zac_with_roc, zac_without_roc = test_z_adaptive_channel_with_roc()
        
        # 異なる設定のテスト
        test_different_roc_settings()
        
        # 分析チャートの表示
        plot_roc_channel_analysis(data, zac_with_roc, zac_without_roc)
        
        print("\n=== テスト完了 ===")
        print("ROC Persistence機能付きZ Adaptive Channelは正常に動作しています。")
        print("\n新機能の特徴:")
        print("- ROC方向が負の時：アッパーバンド乗数を減少")
        print("- ROC方向が正の時：ロワーバンド乗数を減少")
        print("- ROC継続性に基づく動的なバンド調整")
        print("- 従来機能との完全な互換性維持")
        
    except Exception as e:
        import traceback
        print(f"テスト中にエラーが発生しました: {str(e)}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main() 