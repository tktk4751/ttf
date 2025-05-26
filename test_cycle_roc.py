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

from indicators.cycle_roc import CycleROC
from indicators.roc import ROC  # 従来のROCと比較用


def create_test_data(n_points: int = 1000) -> pd.DataFrame:
    """
    テスト用のサンプルデータを作成
    
    Args:
        n_points: データポイント数
        
    Returns:
        OHLC価格データ
    """
    np.random.seed(42)
    
    # 基本的なトレンドとサイクル成分を持つ価格データを生成
    time = np.arange(n_points)
    
    # 長期トレンド
    trend = 100 + 0.01 * time
    
    # サイクル成分（複数の周期を組み合わせ）
    cycle1 = 5 * np.sin(2 * np.pi * time / 20)  # 20期間のサイクル
    cycle2 = 3 * np.sin(2 * np.pi * time / 50)  # 50期間のサイクル
    cycle3 = 2 * np.sin(2 * np.pi * time / 100) # 100期間のサイクル
    
    # ノイズ
    noise = np.random.normal(0, 1, n_points)
    
    # 基本価格
    base_price = trend + cycle1 + cycle2 + cycle3 + noise
    
    # OHLCデータを生成
    high = base_price + np.abs(np.random.normal(0, 0.5, n_points))
    low = base_price - np.abs(np.random.normal(0, 0.5, n_points))
    close = base_price + np.random.normal(0, 0.2, n_points)
    open_price = np.roll(close, 1)  # 前日の終値を基準に開始価格
    open_price[0] = close[0]
    
    # 出来高を追加
    volume = np.random.randint(1000, 10000, n_points)
    
    # DataFrameを作成
    data = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    # 日付インデックスを追加
    start_date = pd.Timestamp('2023-01-01')
    data.index = pd.date_range(start=start_date, periods=n_points, freq='D')
    
    return data


def test_cycle_roc_basic():
    """基本的なサイクルROCのテスト"""
    print("=== サイクルROC基本テスト ===")
    
    # テストデータを作成
    data = create_test_data(500)
    print(f"テストデータ作成完了: {len(data)}行")
    
    # サイクルROCインジケーターを作成
    cycle_roc = CycleROC(
        detector_type='dudi_e',
        cycle_part=0.5,
        max_cycle=50,
        min_cycle=5,
        src_type='close',
        smooth_roc=True,
        roc_alma_period=5,
        signal_threshold=1.0  # ±1%をシグナルしきい値とする
    )
    
    # 計算実行
    print("サイクルROCを計算中...")
    roc_values = cycle_roc.calculate(data)
    result = cycle_roc.get_result()
    
    print(f"計算完了: {len(roc_values)}個の値")
    print(f"有効値数: {np.sum(~np.isnan(roc_values))}")
    print(f"ROC値の範囲: {np.nanmin(roc_values):.4f} ～ {np.nanmax(roc_values):.4f}")
    print(f"ROC値の平均: {np.nanmean(roc_values):.4f}")
    print(f"ROC値の標準偏差: {np.nanstd(roc_values):.4f}")
    
    # サイクル期間の分析
    cycle_periods = cycle_roc.get_cycle_periods()
    print(f"サイクル期間の範囲: {np.nanmin(cycle_periods):.1f} ～ {np.nanmax(cycle_periods):.1f}")
    print(f"サイクル期間の平均: {np.nanmean(cycle_periods):.1f}")
    
    # ROCシグナルの分析
    roc_signals = cycle_roc.get_roc_signals()
    up_signals = np.sum(roc_signals == 1.0)
    down_signals = np.sum(roc_signals == -1.0)
    neutral_signals = np.sum(roc_signals == 0.0)
    print(f"上昇シグナル: {up_signals}回")
    print(f"下降シグナル: {down_signals}回")
    print(f"中立シグナル: {neutral_signals}回")
    
    return data, cycle_roc, roc_values, result


def test_comparison_with_standard_roc():
    """従来のROCとの比較テスト"""
    print("\n=== 従来ROCとの比較テスト ===")
    
    # テストデータを作成
    data = create_test_data(300)
    
    # サイクルROC（スムージングなし）
    cycle_roc = CycleROC(
        detector_type='dudi_e',
        src_type='close',
        smooth_roc=False,
        min_cycle=10,
        max_cycle=30
    )
    
    # 従来のROC（固定期間20）
    standard_roc = ROC(period=20)
    
    # 計算実行
    cycle_roc_values = cycle_roc.calculate(data)
    standard_roc_values = standard_roc.calculate(data)
    
    # 統計比較
    print(f"サイクルROC - 平均: {np.nanmean(cycle_roc_values):.4f}, 標準偏差: {np.nanstd(cycle_roc_values):.4f}")
    print(f"従来ROC - 平均: {np.nanmean(standard_roc_values):.4f}, 標準偏差: {np.nanstd(standard_roc_values):.4f}")
    
    # 相関係数
    valid_mask = ~(np.isnan(cycle_roc_values) | np.isnan(standard_roc_values))
    if np.sum(valid_mask) > 10:
        correlation = np.corrcoef(
            cycle_roc_values[valid_mask], 
            standard_roc_values[valid_mask]
        )[0, 1]
        print(f"相関係数: {correlation:.4f}")
    
    return data, cycle_roc_values, standard_roc_values


def test_different_detectors():
    """異なるサイクル検出器のテスト"""
    print("\n=== 異なるサイクル検出器のテスト ===")
    
    # テストデータを作成
    data = create_test_data(200)
    
    # 異なる検出器でテスト
    detector_types = ['dudi_e', 'hody_e', 'phac_e']
    results = {}
    
    for detector_type in detector_types:
        print(f"\n検出器: {detector_type}")
        
        cycle_roc = CycleROC(
            detector_type=detector_type,
            src_type='close',
            smooth_roc=True,
            roc_alma_period=3
        )
        
        roc_values = cycle_roc.calculate(data)
        cycle_periods = cycle_roc.get_cycle_periods()
        
        print(f"  ROC平均: {np.nanmean(roc_values):.4f}")
        print(f"  ROC標準偏差: {np.nanstd(roc_values):.4f}")
        print(f"  平均サイクル期間: {np.nanmean(cycle_periods):.1f}")
        
        results[detector_type] = {
            'roc_values': roc_values,
            'cycle_periods': cycle_periods,
            'indicator': cycle_roc
        }
    
    return data, results


def plot_cycle_roc_analysis(data, cycle_roc, roc_values):
    """サイクルROCの分析チャートを作成"""
    print("\n=== サイクルROC分析チャートを作成 ===")
    
    # チャートデータの準備
    result = cycle_roc.get_result()
    cycle_periods = result.cycle_periods
    roc_signals = result.roc_signals
    
    # 直近200ポイントに絞る
    n_plot = min(200, len(data))
    plot_data = data.iloc[-n_plot:].copy()
    plot_roc = roc_values[-n_plot:]
    plot_cycles = cycle_periods[-n_plot:]
    plot_signals = roc_signals[-n_plot:]
    
    # プロット作成
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    fig.suptitle('サイクルROC分析', fontsize=16)
    
    # 1. 価格チャート
    axes[0].plot(plot_data.index, plot_data['close'], 'b-', linewidth=1, label='終値')
    axes[0].set_title('価格チャート')
    axes[0].set_ylabel('価格')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. サイクルROC
    axes[1].plot(plot_data.index, plot_roc, 'purple', linewidth=1, label='サイクルROC')
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1].axhline(y=1, color='red', linestyle='--', alpha=0.5, label='上昇しきい値')
    axes[1].axhline(y=-1, color='red', linestyle='--', alpha=0.5, label='下降しきい値')
    axes[1].set_title('サイクルROC')
    axes[1].set_ylabel('ROC (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. 動的サイクル期間
    axes[2].plot(plot_data.index, plot_cycles, 'green', linewidth=1, label='サイクル期間')
    axes[2].set_title('動的サイクル期間')
    axes[2].set_ylabel('期間')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # 4. ROCシグナル
    colors = ['red' if s == -1 else 'green' if s == 1 else 'gray' for s in plot_signals]
    axes[3].scatter(plot_data.index, plot_signals, c=colors, alpha=0.6, s=20)
    axes[3].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[3].set_title('ROCシグナル (赤:下降, 緑:上昇, 灰:中立)')
    axes[3].set_ylabel('シグナル')
    axes[3].set_ylim(-1.5, 1.5)
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """メイン関数"""
    print("サイクルROCインジケーターのテストを開始します...\n")
    
    try:
        # 基本テスト
        data, cycle_roc, roc_values, result = test_cycle_roc_basic()
        
        # 従来ROCとの比較
        test_comparison_with_standard_roc()
        
        # 異なる検出器のテスト
        test_different_detectors()
        
        # 分析チャートの表示
        plot_cycle_roc_analysis(data, cycle_roc, roc_values)
        
        print("\n=== テスト完了 ===")
        print("サイクルROCインジケーターは正常に動作しています。")
        
    except Exception as e:
        import traceback
        print(f"テスト中にエラーが発生しました: {str(e)}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main() 