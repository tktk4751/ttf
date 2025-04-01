#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
# プロジェクトのルートディレクトリをPythonパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from indicators import AlphaVolatility


def main():
    # サンプルデータの生成（ランダムウォーク＋トレンド＋ボラティリティクラスター）
    np.random.seed(42)
    n = 500
    
    # 基本的なランダムウォーク
    price = 100.0
    prices = [price]
    volatility = 0.01
    
    # 異なるボラティリティ状態を作成
    volatility_states = np.ones(n) * volatility
    volatility_states[100:150] = volatility * 3  # 高ボラティリティ期間
    volatility_states[300:350] = volatility * 5  # 非常に高いボラティリティ期間
    
    # 価格系列の生成
    for i in range(1, n):
        # トレンド成分を追加
        trend = 0.0
        if i > 200 and i <= 350:
            trend = 0.05  # 上昇トレンド
        elif i > 350:
            trend = -0.03  # 下降トレンド
        
        # その時点でのボラティリティを使用
        current_vol = volatility_states[i-1]
        
        # 価格変動を計算
        change = np.random.normal(trend, current_vol)
        price *= (1 + change)
        prices.append(price)
    
    # DataFrame作成
    df = pd.DataFrame({'close': prices})
    
    # アルファボラティリティの計算
    alpha_vol = AlphaVolatility(
        er_period=21,
        max_vol_period=89,
        min_vol_period=13,
        smoothing_period=14
    )
    
    alpha_vol_values = alpha_vol.calculate(df)
    
    # 結果の取得
    er_values = alpha_vol.get_efficiency_ratio()
    std_values = alpha_vol.get_standard_deviation()
    dynamic_period = alpha_vol.get_dynamic_period()
    
    # 結果をDataFrameに追加
    df['alpha_volatility'] = alpha_vol_values
    df['efficiency_ratio'] = er_values
    df['std_dev'] = std_values
    df['dynamic_period'] = dynamic_period
    
    # グラフの描画
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(4, 1, height_ratios=[3, 1, 1, 1])
    
    # 価格チャート
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df.index, df['close'], label='Price', color='blue')
    ax1.set_title('Price Chart')
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # アルファボラティリティ
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(df.index, df['alpha_volatility'], label='Alpha Volatility', color='red')
    ax2.set_title('Alpha Volatility')
    ax2.set_ylabel('Value')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 標準偏差
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(df.index, df['std_dev'], label='Standard Deviation', color='green')
    ax3.set_title('Standard Deviation')
    ax3.set_ylabel('Value')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 効率比と動的期間
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.plot(df.index, df['efficiency_ratio'], label='Efficiency Ratio', color='purple')
    ax4_twin = ax4.twinx()
    ax4_twin.plot(df.index, df['dynamic_period'], label='Dynamic Period', color='orange', alpha=0.7)
    ax4.set_title('Efficiency Ratio and Dynamic Period')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Efficiency Ratio')
    ax4_twin.set_ylabel('Period')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('alpha_volatility_example.png')
    plt.show()
    
    # 統計情報の表示
    print("=== Alpha Volatility Statistics ===")
    print(f"Mean: {df['alpha_volatility'].mean():.6f}")
    print(f"Std: {df['alpha_volatility'].std():.6f}")
    print(f"Min: {df['alpha_volatility'].min():.6f}")
    print(f"Max: {df['alpha_volatility'].max():.6f}")
    print(f"Dynamic Period Range: {df['dynamic_period'].min():.0f} - {df['dynamic_period'].max():.0f}")


if __name__ == "__main__":
    main() 