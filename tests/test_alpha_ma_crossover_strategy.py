#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from strategies.implementations.alpha_ma_crossover_strategy import AlphaMACrossoverStrategy


def test_alpha_ma_crossover_strategy():
    """アルファMAクロスオーバー戦略のテスト"""
    # テストデータの生成
    np.random.seed(42)  # 再現性のため
    n = 500  # データポイント数
    
    # 基本的な価格データの生成
    prices = np.zeros(n)
    prices[0] = 100.0
    
    # トレンドとレンジを交互に作成
    # 上昇トレンド
    for i in range(1, 100):
        prices[i] = prices[i-1] * (1 + np.random.normal(0.001, 0.005))
    
    # レンジ相場
    for i in range(100, 150):
        prices[i] = prices[i-1] * (1 + np.random.normal(0.0, 0.008))
    
    # 下降トレンド
    for i in range(150, 250):
        prices[i] = prices[i-1] * (1 + np.random.normal(-0.001, 0.005))
    
    # レンジ相場
    for i in range(250, 300):
        prices[i] = prices[i-1] * (1 + np.random.normal(0.0, 0.008))
    
    # 上昇トレンド
    for i in range(300, 400):
        prices[i] = prices[i-1] * (1 + np.random.normal(0.001, 0.005))
    
    # レンジ相場
    for i in range(400, 450):
        prices[i] = prices[i-1] * (1 + np.random.normal(0.0, 0.008))
    
    # 下降トレンド
    for i in range(450, n):
        prices[i] = prices[i-1] * (1 + np.random.normal(-0.001, 0.005))
    
    # データフレームの作成
    df = pd.DataFrame({
        'open': prices * (1 - np.random.normal(0, 0.002, n)),
        'high': prices * (1 + np.random.normal(0.005, 0.003, n)),
        'low': prices * (1 - np.random.normal(0.005, 0.003, n)),
        'close': prices
    })
    
    # 戦略の初期化
    strategy = AlphaMACrossoverStrategy(
        # AlphaMACrossoverEntrySignalのパラメータ
        er_period=21,
        short_max_kama_period=89,
        short_min_kama_period=5,
        long_max_kama_period=233,
        long_min_kama_period=21
    )
    
    # シグナルの生成
    entry_signals = strategy.get_entry_signals(df)
    ma_crossover_signals = strategy.get_ma_crossover_signals(df)
    
    # 結果の表示
    print(f"総シグナル数: {np.sum(np.abs(entry_signals))}")
    print(f"ロングシグナル数: {np.sum(entry_signals == 1)}")
    print(f"ショートシグナル数: {np.sum(entry_signals == -1)}")
    
    # 結果のプロット
    plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 1, height_ratios=[3, 1, 1])
    
    # 価格チャートとシグナル
    ax1 = plt.subplot(gs[0])
    ax1.plot(df.index, df['close'], label='価格', color='blue')
    
    # ロングシグナルとショートシグナルのプロット
    long_signals = np.where(entry_signals == 1)[0]
    short_signals = np.where(entry_signals == -1)[0]
    
    ax1.scatter(long_signals, df['close'].iloc[long_signals], marker='^', color='green', s=100, label='ロングエントリー')
    ax1.scatter(short_signals, df['close'].iloc[short_signals], marker='v', color='red', s=100, label='ショートエントリー')
    
    ax1.set_title('アルファMAクロスオーバー戦略テスト')
    ax1.set_ylabel('価格')
    ax1.legend()
    ax1.grid(True)
    
    # MAクロスオーバーシグナルプロット
    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax2.plot(df.index, ma_crossover_signals, label='MAクロスオーバーシグナル', color='purple')
    ax2.set_ylabel('MAシグナル')
    ax2.set_yticks([-1, 0, 1])
    ax2.grid(True)
    
    # エントリーシグナルプロット
    ax3 = plt.subplot(gs[2], sharex=ax1)
    ax3.plot(df.index, entry_signals, label='エントリーシグナル', color='blue')
    ax3.set_ylabel('エントリーシグナル')
    ax3.set_yticks([-1, 0, 1])
    ax3.set_xlabel('時間')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('alpha_ma_crossover_strategy_test.png')
    plt.close()
    
    print("テスト完了。結果は 'alpha_ma_crossover_strategy_test.png' に保存されました。")


if __name__ == "__main__":
    test_alpha_ma_crossover_strategy() 