#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ultimate MAストラテジーの使用例

このファイルは、Ultimate MAストラテジーの基本的な使用方法を示します。
"""

import numpy as np
import pandas as pd
from strategy import UltimateMAStrategy


def create_sample_data(n_periods: int = 1000) -> pd.DataFrame:
    """
    サンプルのOHLCデータを生成する
    
    Args:
        n_periods: データポイント数
        
    Returns:
        pd.DataFrame: サンプルOHLCデータ
    """
    np.random.seed(42)
    
    # 基本価格のランダムウォーク
    base_price = 100.0
    returns = np.random.normal(0, 0.02, n_periods)
    prices = [base_price]
    
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices[1:])
    
    # OHLC生成
    data = []
    for i, price in enumerate(prices):
        # 日中変動を追加
        volatility = 0.01
        high = price * (1 + np.random.uniform(0, volatility))
        low = price * (1 - np.random.uniform(0, volatility))
        
        # 始値と終値
        if i == 0:
            open_price = price
        else:
            open_price = data[i-1]['close']
        
        close_price = price
        
        data.append({
            'open': open_price,
            'high': max(open_price, high, close_price),
            'low': min(open_price, low, close_price),
            'close': close_price
        })
    
    return pd.DataFrame(data)


def example_basic_usage():
    """基本的な使用例"""
    print("=== Ultimate MAストラテジー - 基本使用例 ===")
    
    # サンプルデータの生成
    data = create_sample_data(500)
    print(f"サンプルデータ生成完了: {len(data)}期間")
    
    # ストラテジーの初期化
    strategy = UltimateMAStrategy(
        super_smooth_period=10,
        zero_lag_period=21,
        realtime_window=5,
        src_type='hlc3',
        slope_index=1,
        range_threshold=0.005
    )
    
    print("ストラテジー初期化完了")
    
    # エントリーシグナルの生成
    entry_signals = strategy.generate_entry(data)
    print(f"エントリーシグナル数: ロング={np.sum(entry_signals == 1)}, ショート={np.sum(entry_signals == -1)}")
    
    # Ultimate MAの値を取得
    ultimate_ma_values = strategy.get_ultimate_ma_values(data)
    print(f"Ultimate MA最終値: {ultimate_ma_values[-1]:.4f}")
    
    # リアルタイムトレンドを取得
    realtime_trends = strategy.get_realtime_trends(data)
    print(f"最新リアルタイムトレンド: {realtime_trends[-1]:.4f}")
    
    # トレンドシグナルを取得
    trend_signals = strategy.get_trend_signals(data)
    current_trend = "上昇" if trend_signals[-1] == 1 else "下降" if trend_signals[-1] == -1 else "レンジ"
    print(f"現在のトレンド: {current_trend}")
    
    # ノイズ除去統計を取得
    noise_stats = strategy.get_noise_reduction_stats(data)
    if noise_stats:
        print(f"ノイズ除去効果: {noise_stats.get('noise_reduction_percentage', 0):.2f}%")


def example_detailed_analysis():
    """詳細な分析例"""
    print("\n=== Ultimate MAストラテジー - 詳細分析例 ===")
    
    # サンプルデータの生成
    data = create_sample_data(200)
    
    # ストラテジーの初期化（カスタムパラメータ）
    strategy = UltimateMAStrategy(
        super_smooth_period=8,
        zero_lag_period=15,
        realtime_window=7,
        src_type='hlc3',
        slope_index=2,
        range_threshold=0.003
    )
    
    # 全段階の結果を取得
    all_stages = strategy.get_all_ultimate_ma_stages(data)
    
    if all_stages:
        print("Ultimate MA各段階の最終値:")
        print(f"  生価格: {all_stages['raw_values'][-1]:.4f}")
        print(f"  カルマンフィルター後: {all_stages['kalman_values'][-1]:.4f}")
        print(f"  スーパースムーザー後: {all_stages['super_smooth_values'][-1]:.4f}")
        print(f"  ゼロラグEMA後: {all_stages['zero_lag_values'][-1]:.4f}")
        print(f"  最終値: {all_stages['final_values'][-1]:.4f}")
        print(f"  振幅: {all_stages['amplitude'][-1]:.4f}")
        print(f"  位相: {all_stages['phase'][-1]:.4f}")
        print(f"  現在のトレンド: {all_stages['current_trend']}")


def example_exit_signals():
    """決済シグナルの例"""
    print("\n=== Ultimate MAストラテジー - 決済シグナル例 ===")
    
    # サンプルデータの生成
    data = create_sample_data(100)
    
    # ストラテジーの初期化
    strategy = UltimateMAStrategy(
        super_smooth_period=10,
        zero_lag_period=21,
        realtime_window=5
    )
    
    # エントリーシグナルの取得
    entry_signals = strategy.generate_entry(data)
    
    # 最初のロングシグナルを探す
    long_entry_index = None
    for i, signal in enumerate(entry_signals):
        if signal == 1:
            long_entry_index = i
            break
    
    if long_entry_index is not None:
        print(f"ロングエントリー発生: インデックス {long_entry_index}")
        
        # そのポジションでの決済シグナルをチェック
        for i in range(long_entry_index + 1, len(data)):
            should_exit = strategy.generate_exit(data, position=1, index=i)
            if should_exit:
                print(f"ロング決済シグナル: インデックス {i} (保有期間: {i - long_entry_index}期間)")
                break
        else:
            print("決済シグナルなし（期間内）")
    else:
        print("ロングエントリーシグナルなし")


if __name__ == "__main__":
    try:
        example_basic_usage()
        example_detailed_analysis()
        example_exit_signals()
        print("\n✅ すべての例が正常に実行されました")
    except Exception as e:
        print(f"❌ エラーが発生しました: {str(e)}")
        import traceback
        print(traceback.format_exc()) 