#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple

# トゥルーモメンタムインジケーターとシグナルのインポート
from indicators.true_momentum import TrueMomentum
from signals.implementations.true_momentum import TrueMomentumEntrySignal, TrueMomentumDirectionSignal


def load_sample_data() -> pd.DataFrame:
    """
    サンプルデータをロードする
    
    Returns:
        サンプル価格データ
    """
    # ランダムな価格データを生成（実際の分析では実データを使用）
    np.random.seed(42)
    n = 500
    
    # 初期価格
    price = 100.0
    
    # 価格データの生成
    dates = pd.date_range(start='2023-01-01', periods=n)
    prices = []
    
    for i in range(n):
        # ランダムな価格変動を生成
        change = np.random.normal(0, 1)
        
        # トレンドを加える（100日周期）
        trend = 0.1 * np.sin(i / 50 * np.pi)
        
        # 価格を更新
        price += change + trend
        prices.append(price)
    
    # ボラティリティの高い期間を作成
    volatility_period = slice(200, 300)
    prices_array = np.array(prices)
    prices_array[volatility_period] += np.random.normal(0, 3, size=100)
    
    # トレンド期間を作成
    trend_period = slice(350, 450)
    trend_values = np.linspace(0, 15, 100)
    prices_array[trend_period] += trend_values
    
    # データフレームを作成
    df = pd.DataFrame({
        'date': dates,
        'open': prices_array,
        'high': prices_array + np.random.uniform(0.1, 1.0, size=n),
        'low': prices_array - np.random.uniform(0.1, 1.0, size=n),
        'close': prices_array,
        'volume': np.random.randint(1000, 10000, size=n)
    })
    
    df.set_index('date', inplace=True)
    return df


def analyze_true_momentum_signals(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    トゥルーモメンタムのシグナルを分析
    
    Args:
        data: 価格データ
    
    Returns:
        (entry_signals, direction_signals)のタプル
    """
    # トゥルーモメンタムインジケーターを初期化
    indicator = TrueMomentum(
        period=20,
        max_std_mult=2.0,
        min_std_mult=1.0,
        max_kama_slow=55,
        min_kama_slow=30,
        max_kama_fast=13,
        min_kama_fast=2,
        max_atr_period=120,
        min_atr_period=13,
        max_atr_mult=3.0,
        min_atr_mult=1.0,
        max_momentum_period=100,
        min_momentum_period=20
    )
    
    # インジケーターを計算
    momentum = indicator.calculate(data)
    
    # エントリーシグナルを初期化
    entry_signal = TrueMomentumEntrySignal(
        period=20,
        momentum_threshold=0.0  # モメンタムが0を超えたらシグナル発生
    )
    
    # 方向シグナルを初期化
    direction_signal = TrueMomentumDirectionSignal(
        period=20
    )
    
    # シグナルを生成
    entry_signals = entry_signal.generate(data)
    direction_signals = direction_signal.generate(data)
    
    return entry_signals, direction_signals


def plot_results(data: pd.DataFrame, momentum: np.ndarray, entry_signals: np.ndarray, direction_signals: np.ndarray):
    """
    結果をプロット
    
    Args:
        data: 価格データ
        momentum: モメンタム値
        entry_signals: エントリーシグナル
        direction_signals: 方向シグナル
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # 価格チャートをプロット
    axes[0].plot(data.index, data['close'], label='Close', color='blue')
    axes[0].set_title('価格チャート')
    axes[0].legend()
    
    # エントリーポイントをプロット
    buy_signals = np.where(entry_signals == 1)[0]
    sell_signals = np.where(entry_signals == -1)[0]
    
    axes[0].plot(data.index[buy_signals], data['close'].values[buy_signals], '^', 
                 markersize=10, color='green', label='買いシグナル')
    axes[0].plot(data.index[sell_signals], data['close'].values[sell_signals], 'v', 
                 markersize=10, color='red', label='売りシグナル')
    
    # モメンタムをプロット
    axes[1].plot(data.index, momentum, label='モメンタム', color='purple')
    axes[1].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    axes[1].set_title('トゥルーモメンタム')
    axes[1].legend()
    
    # 方向シグナルをプロット
    axes[2].fill_between(data.index, direction_signals, 0, where=direction_signals > 0, 
                         color='green', alpha=0.3, label='ロング方向')
    axes[2].fill_between(data.index, direction_signals, 0, where=direction_signals < 0, 
                         color='red', alpha=0.3, label='ショート方向')
    axes[2].set_title('方向シグナル')
    axes[2].set_ylim(-1.5, 1.5)
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # サンプルデータをロード
    data = load_sample_data()
    
    # トゥルーモメンタムインジケーターを初期化
    indicator = TrueMomentum(
        period=20,
        max_std_mult=2.0,
        min_std_mult=1.0,
        max_kama_slow=55,
        min_kama_slow=30,
        max_kama_fast=13,
        min_kama_fast=2,
        max_atr_period=120,
        min_atr_period=13,
        max_atr_mult=3.0,
        min_atr_mult=1.0,
        max_momentum_period=100,
        min_momentum_period=20
    )
    
    # インジケーターを計算
    momentum = indicator.calculate(data)
    
    # スクイーズ状態を取得
    sqz_on, sqz_off, no_sqz = indicator.get_squeeze_states()
    
    # 動的モメンタム期間を取得
    dynamic_period = indicator.get_dynamic_momentum_period()
    
    # シグナルを分析
    entry_signals, direction_signals = analyze_true_momentum_signals(data)
    
    # 結果をプロット
    plot_results(data, momentum, entry_signals, direction_signals)
    
    # 統計情報を表示
    print(f"買いシグナル数: {np.sum(entry_signals == 1)}")
    print(f"売りシグナル数: {np.sum(entry_signals == -1)}")
    print(f"スクイーズオン状態の発生回数: {np.sum(sqz_on)}")
    print(f"平均動的モメンタム期間: {np.nanmean(dynamic_period):.2f}") 