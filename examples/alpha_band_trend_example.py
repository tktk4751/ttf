#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.dates as mdates

# 親ディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.implementations.alpha_band_trend.strategy import AlphaBandTrendStrategy
from strategies.implementations.alpha_band_trend.signal_generator import AlphaBandTrendSignalGenerator



def main():
    """
    アルファバンド+アルファトレンドフィルター戦略のサンプル
    """
    # データの読み込み（例：日足）
    # data = load_ohlc_data("path/to/your/data.csv")
    
    # テスト用のランダムデータを生成
    np.random.seed(42)
    n = 500
    data = pd.DataFrame({
        'open': np.random.normal(100, 5, n),
        'high': np.zeros(n),
        'low': np.zeros(n),
        'close': np.zeros(n)
    })
    
    # トレンド相場とレンジ相場を含むサンプルデータを作成
    # はじめの部分をトレンド相場に
    data['close'][:n//3] = np.cumsum(np.random.normal(0.2, 1, n//3)) + data['open'][:n//3]
    
    # 中間部分をレンジ相場に
    data['close'][n//3:2*n//3] = data['open'][n//3:2*n//3] + np.random.normal(0, 2, n//3)
    
    # 後半部分を下降トレンドに
    data['close'][2*n//3:] = data['open'][2*n//3:] - np.cumsum(np.random.normal(0.2, 1, n-(2*n//3)))
    
    # 高値と安値を設定
    for i in range(n):
        data.loc[i, 'high'] = max(data.loc[i, 'open'], data.loc[i, 'close']) + np.random.uniform(0.5, 2.0)
        data.loc[i, 'low'] = min(data.loc[i, 'open'], data.loc[i, 'close']) - np.random.uniform(0.5, 2.0)
    
    # 日付インデックスの設定（視覚的に分かりやすく）
    data.index = pd.date_range(start='2022-01-01', periods=n)
    
    # シグナル生成器の初期化（デフォルトパラメータ）
    signal_generator = AlphaBandTrendSignalGenerator()
    
    # シグナルの計算
    entry_signals = signal_generator.get_entry_signals(data)
    band_signals = signal_generator.get_band_signals()
    filter_signals = signal_generator.get_filter_signals()
    
    # バンド値とフィルター値の取得
    center, upper, lower = signal_generator.get_band_values()
    filter_values = signal_generator.get_filter_values()
    
    # シミュレートされたトレード結果
    positions = np.zeros(len(data))
    equity = np.ones(len(data)) * 10000  # 初期資金10000
    for i in range(1, len(data)):
        # 前日のポジションを維持
        positions[i] = positions[i-1]
        
        # エントリーシグナル
        if entry_signals[i] == 1 and positions[i] == 0:  # ロングエントリー
            positions[i] = 1
            print(f"ロングエントリー: {data.index[i].date()} @ {data['close'][i]:.2f}")
        elif entry_signals[i] == -1 and positions[i] == 0:  # ショートエントリー
            positions[i] = -1
            print(f"ショートエントリー: {data.index[i].date()} @ {data['close'][i]:.2f}")
        
        # エグジットシグナルのチェック
        elif positions[i] == 1 and band_signals[i] == -1:  # ロングエグジット
            positions[i] = 0
            print(f"ロング決済: {data.index[i].date()} @ {data['close'][i]:.2f}")
        elif positions[i] == -1 and band_signals[i] == 1:  # ショートエグジット
            positions[i] = 0
            print(f"ショート決済: {data.index[i].date()} @ {data['close'][i]:.2f}")
        
        # 資産の更新（単純なシミュレーション）
        if i > 0:
            pnl = positions[i-1] * (data['close'][i] - data['close'][i-1])
            equity[i] = equity[i-1] + pnl * 100  # 100株単位
    
    # 結果の確認
    print("\n*** トレード結果 ***")
    print(f"初期資金: 10000")
    print(f"最終資金: {equity[-1]:.2f}")
    print(f"利益率: {(equity[-1]/10000-1)*100:.2f}%")
    print(f"最大ドローダウン: {(1 - min(equity)/max(equity[0:np.argmin(equity)]))*100:.2f}%")
    
    # 可視化
    plt.figure(figsize=(15, 10))
    
    # 価格チャートとバンド（上段）
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(data.index, data['close'], label='終値', color='blue', alpha=0.7)
    ax1.plot(data.index, center, label='中心線', color='purple', alpha=0.7)
    ax1.plot(data.index, upper, label='上限バンド', color='green', alpha=0.7)
    ax1.plot(data.index, lower, label='下限バンド', color='red', alpha=0.7)
    
    # ロングエントリー・ショートエントリーの表示
    long_entries = data.index[entry_signals == 1]
    short_entries = data.index[entry_signals == -1]
    long_prices = data.loc[entry_signals == 1, 'close']
    short_prices = data.loc[entry_signals == -1, 'close']
    
    ax1.scatter(long_entries, long_prices, marker='^', color='g', s=100, label='ロングエントリー')
    ax1.scatter(short_entries, short_prices, marker='v', color='r', s=100, label='ショートエントリー')
    
    # ロング決済・ショート決済の表示
    exits = []
    exit_prices = []
    exit_colors = []
    
    for i in range(1, len(positions)):
        if positions[i-1] == 1 and positions[i] == 0:  # ロング決済
            exits.append(data.index[i])
            exit_prices.append(data['close'][i])
            exit_colors.append('darkred')
        elif positions[i-1] == -1 and positions[i] == 0:  # ショート決済
            exits.append(data.index[i])
            exit_prices.append(data['close'][i])
            exit_colors.append('darkgreen')
    
    ax1.scatter(exits, exit_prices, marker='x', color=exit_colors, s=100, label='決済')
    
    # フィルター状態の表示（背景色）
    for i in range(len(data)):
        if filter_signals[i] == 1:  # トレンド相場
            ax1.axvspan(data.index[i], data.index[min(i+1, len(data)-1)], alpha=0.2, color='green')
        elif filter_signals[i] == -1:  # レンジ相場
            ax1.axvspan(data.index[i], data.index[min(i+1, len(data)-1)], alpha=0.2, color='red')
    
    ax1.set_title('アルファバンド+アルファトレンドフィルター戦略')
    ax1.set_ylabel('価格')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    
    # アルファトレンドフィルター値（中段）
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(data.index, filter_values, label='フィルター値', color='blue')
    
    # 動的しきい値
    threshold_values = signal_generator.alpha_filter_signal.get_threshold_values()
    ax2.plot(data.index, threshold_values, label='動的しきい値', color='purple', linestyle='--')
    
    # トレンド/レンジの状態を背景色で表示
    for i in range(len(data)):
        if filter_signals[i] == 1:  # トレンド相場
            ax2.axvspan(data.index[i], data.index[min(i+1, len(data)-1)], alpha=0.2, color='green')
        elif filter_signals[i] == -1:  # レンジ相場
            ax2.axvspan(data.index[i], data.index[min(i+1, len(data)-1)], alpha=0.2, color='red')
    
    ax2.set_title('アルファトレンドフィルター')
    ax2.set_ylabel('フィルター値')
    ax2.legend(loc='upper left')
    ax2.grid(True)
    
    # エクイティカーブ（下段）
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(data.index, equity, label='資産推移', color='blue')
    
    # ポジション状態を背景色で表示
    for i in range(len(data)):
        if positions[i] == 1:  # ロングポジション
            ax3.axvspan(data.index[i], data.index[min(i+1, len(data)-1)], alpha=0.2, color='green')
        elif positions[i] == -1:  # ショートポジション
            ax3.axvspan(data.index[i], data.index[min(i+1, len(data)-1)], alpha=0.2, color='red')
    
    ax3.set_title('資産推移')
    ax3.set_ylabel('資産額')
    ax3.set_xlabel('日付')
    ax3.legend(loc='upper left')
    ax3.grid(True)
    
    # X軸のフォーマット
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
    
    plt.tight_layout()
    plt.savefig('alpha_band_trend_example.png')
    plt.show()


if __name__ == "__main__":
    main() 