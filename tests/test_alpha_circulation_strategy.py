#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from strategies.implementations.alpha_circulation_strategy import AlphaCirculationStrategy


def test_alpha_circulation_strategy():
    """アルファ循環戦略のテスト"""
    # テストデータの生成
    np.random.seed(42)  # 再現性のため
    n = 500  # データポイント数
    
    # 基本的な価格データの生成
    prices = np.zeros(n)
    prices[0] = 100.0
    
    # 6つのステージを持つ市場サイクルをシミュレート
    # ステージ1: 短期 > 中期 > 長期（安定上昇相場）
    for i in range(1, 100):
        prices[i] = prices[i-1] * (1 + np.random.normal(0.001, 0.005))
    
    # ステージ2: 中期 > 短期 > 長期（上昇相場の終焉）
    for i in range(100, 150):
        prices[i] = prices[i-1] * (1 + np.random.normal(0.0005, 0.008))
    
    # ステージ3: 中期 > 長期 > 短期（下降相場の入口）
    for i in range(150, 200):
        prices[i] = prices[i-1] * (1 + np.random.normal(-0.001, 0.007))
    
    # ステージ4: 長期 > 中期 > 短期（安定下降相場）
    for i in range(200, 300):
        prices[i] = prices[i-1] * (1 + np.random.normal(-0.001, 0.005))
    
    # ステージ5: 長期 > 短期 > 中期（下降相場の終焉）
    for i in range(300, 350):
        prices[i] = prices[i-1] * (1 + np.random.normal(-0.0005, 0.008))
    
    # ステージ6: 短期 > 長期 > 中期（上昇相場の入口）
    for i in range(350, 400):
        prices[i] = prices[i-1] * (1 + np.random.normal(0.001, 0.007))
    
    # 再びステージ1へ
    for i in range(400, n):
        prices[i] = prices[i-1] * (1 + np.random.normal(0.001, 0.005))
    
    # ダイバージェンスを作成するための調整
    # ステージ2の終わりに価格が上昇してもMACDが下降するようにする
    for i in range(130, 150):
        prices[i] = prices[i-1] * (1 + np.random.normal(0.002, 0.003))
    
    # ステージ5の終わりに価格が下降してもMACDが上昇するようにする
    for i in range(330, 350):
        prices[i] = prices[i-1] * (1 + np.random.normal(-0.002, 0.003))
    
    # データフレームの作成
    df = pd.DataFrame({
        'open': prices * (1 - np.random.normal(0, 0.002, n)),
        'high': prices * (1 + np.random.normal(0.005, 0.003, n)),
        'low': prices * (1 - np.random.normal(0.005, 0.003, n)),
        'close': prices
    })
    
    # 戦略の初期化
    strategy = AlphaCirculationStrategy(
        # AlphaMACirculationSignalのパラメータ
        er_period=21,
        short_max_kama_period=55,
        short_min_kama_period=3,
        middle_max_kama_period=144,
        middle_min_kama_period=21,
        long_max_kama_period=377,
        long_min_kama_period=55,
        
        # AlphaFilterSignalのパラメータ
        max_chop_period=55,
        min_chop_period=8,
        filter_threshold=0.5,
        
        # AlphaMACDDivergenceSignalのパラメータ
        fast_max_kama_period=89,
        fast_min_kama_period=8,
        slow_max_kama_period=144,
        slow_min_kama_period=21,
        lookback=30
    )
    
    # シグナルの生成
    entry_signals = strategy.get_entry_signals(df)
    stages = strategy.get_stages(df)
    filter_values = strategy.get_filter_values(df)
    divergence_values = strategy.get_divergence_values(df)
    
    # 結果の表示
    print(f"総シグナル数: {np.sum(np.abs(entry_signals))}")
    print(f"ロングシグナル数: {np.sum(entry_signals == 1)}")
    print(f"ショートシグナル数: {np.sum(entry_signals == -1)}")
    
    # 各ステージの数をカウント
    for i in range(1, 7):
        print(f"ステージ{i}の数: {np.sum(stages == i)}")
    
    # 結果のプロット
    plt.figure(figsize=(15, 12))
    gs = GridSpec(5, 1, height_ratios=[3, 1, 1, 1, 1])
    
    # 価格チャートとシグナル
    ax1 = plt.subplot(gs[0])
    ax1.plot(df.index, df['close'], label='価格', color='blue')
    
    # ロングシグナルとショートシグナルのプロット
    long_signals = np.where(entry_signals == 1)[0]
    short_signals = np.where(entry_signals == -1)[0]
    
    ax1.scatter(long_signals, df['close'].iloc[long_signals], marker='^', color='green', s=100, label='ロングエントリー')
    ax1.scatter(short_signals, df['close'].iloc[short_signals], marker='v', color='red', s=100, label='ショートエントリー')
    
    # ステージの背景色
    for i in range(len(stages)):
        if stages[i] == 1:
            ax1.axvspan(i-0.5, i+0.5, alpha=0.2, color='green')
        elif stages[i] == 2:
            ax1.axvspan(i-0.5, i+0.5, alpha=0.2, color='yellow')
        elif stages[i] == 3:
            ax1.axvspan(i-0.5, i+0.5, alpha=0.2, color='orange')
        elif stages[i] == 4:
            ax1.axvspan(i-0.5, i+0.5, alpha=0.2, color='red')
        elif stages[i] == 5:
            ax1.axvspan(i-0.5, i+0.5, alpha=0.2, color='yellow')
        elif stages[i] == 6:
            ax1.axvspan(i-0.5, i+0.5, alpha=0.2, color='green')
    
    ax1.set_title('アルファ循環戦略テスト')
    ax1.set_ylabel('価格')
    ax1.legend()
    ax1.grid(True)
    
    # ステージプロット
    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax2.plot(df.index, stages, label='ステージ', color='purple')
    ax2.set_ylabel('ステージ')
    ax2.set_yticks(range(1, 7))
    ax2.grid(True)
    
    # フィルター値プロット
    ax3 = plt.subplot(gs[2], sharex=ax1)
    ax3.plot(df.index, filter_values, label='フィルター値', color='orange')
    ax3.axhline(y=0.5, color='gray', linestyle='--')
    ax3.set_ylabel('フィルター値')
    ax3.grid(True)
    
    # MACDプロット
    ax4 = plt.subplot(gs[3], sharex=ax1)
    ax4.plot(df.index, divergence_values['macd'], label='MACD', color='blue')
    ax4.plot(df.index, divergence_values['signal'], label='シグナル', color='red')
    ax4.set_ylabel('MACD')
    ax4.legend()
    ax4.grid(True)
    
    # ヒストグラムプロット
    ax5 = plt.subplot(gs[4], sharex=ax1)
    ax5.bar(df.index, divergence_values['histogram'], label='ヒストグラム', color='green')
    ax5.set_ylabel('ヒストグラム')
    ax5.set_xlabel('時間')
    ax5.grid(True)
    
    plt.tight_layout()
    plt.savefig('alpha_circulation_strategy_test.png')
    plt.close()
    
    print("テスト完了。結果は 'alpha_circulation_strategy_test.png' に保存されました。")


if __name__ == "__main__":
    test_alpha_circulation_strategy() 