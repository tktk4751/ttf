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


def generate_sample_data(n=500):
    """サンプルデータを生成する関数"""
    np.random.seed(42)
    
    # 基本的なランダムウォーク
    price = 100.0
    prices = [price]
    volatility = 0.01
    
    # 異なるボラティリティとトレンド状態を作成
    volatility_states = np.ones(n) * volatility
    volatility_states[100:150] = volatility * 3    # 高ボラティリティ期間
    volatility_states[300:350] = volatility * 5    # 非常に高いボラティリティ期間
    volatility_states[400:450] = volatility * 2.5  # 中程度のボラティリティ期間
    
    # シンプルなトレンド状態
    trend_states = np.zeros(n)
    trend_states[50:150] = 0.02     # 弱い上昇トレンド
    trend_states[200:300] = 0.06    # 強い上昇トレンド
    trend_states[350:400] = -0.04   # 下降トレンド
    
    # 価格系列の生成
    closes = [price]
    highs = [price]
    lows = [price]
    
    for i in range(1, n):
        # その時点でのボラティリティとトレンドを使用
        current_vol = volatility_states[i-1]
        current_trend = trend_states[i-1]
        
        # 価格変動を計算
        change = np.random.normal(current_trend, current_vol)
        
        # 終値の更新
        close_price = closes[-1] * (1 + change)
        closes.append(close_price)
        
        # 高値と安値の生成（ボラティリティに依存）
        high_price = close_price * (1 + np.random.uniform(0, current_vol * 1.5))
        low_price = close_price * (1 - np.random.uniform(0, current_vol * 1.5))
        
        # 高値は終値より大きく、安値は終値より小さいことを確認
        high_price = max(high_price, close_price)
        low_price = min(low_price, close_price)
        
        highs.append(high_price)
        lows.append(low_price)
    
    # DataFrameを作成
    df = pd.DataFrame({
        'high': highs,
        'low': lows,
        'close': closes
    })
    
    return df


def main():
    # サンプルデータの生成
    df = generate_sample_data(500)
    
    # アルファボラティリティの計算
    alpha_vol = AlphaVolatility(
        er_period=21,
        max_vol_period=89,
        min_vol_period=13,
        smoothing_period=14
    )
    
    # インジケーターの計算
    alpha_vol.calculate(df)
    
    # 結果の取得
    percent_volatility = alpha_vol.get_percent_volatility()
    absolute_volatility = alpha_vol.get_absolute_volatility()
    er_values = alpha_vol.get_efficiency_ratio()
    dynamic_period = alpha_vol.get_dynamic_period()
    
    # 結果をDataFrameに追加
    df['percent_volatility'] = percent_volatility
    df['absolute_volatility'] = absolute_volatility
    df['efficiency_ratio'] = er_values
    df['dynamic_period'] = dynamic_period
    
    # マルチプロットの設定
    fig = plt.figure(figsize=(15, 10))
    total_plots = 4
    
    # 価格チャート
    plt.subplot(total_plots, 1, 1)
    plt.plot(df.index, df['close'], label='Close', color='black')
    plt.fill_between(df.index, df['low'], df['high'], color='lightgray', alpha=0.3)
    plt.title('Price Chart')
    plt.ylabel('Price')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # %ベースのボラティリティ
    plt.subplot(total_plots, 1, 2)
    plt.plot(df.index, df['percent_volatility'] * 100, label='Volatility (%)', color='red')  # パーセント表示に変換
    plt.title('Percent-Based Volatility')
    plt.ylabel('Volatility (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 金額ベースのボラティリティ
    plt.subplot(total_plots, 1, 3)
    plt.plot(df.index, df['absolute_volatility'], label='Absolute Volatility (Currency)', color='green')
    plt.title('Price-Based Volatility')
    plt.ylabel('Volatility (Currency)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 効率比と動的期間
    plt.subplot(total_plots, 1, 4)
    plt.plot(df.index, df['efficiency_ratio'], label='Efficiency Ratio', color='purple')
    ax2 = plt.twinx()
    ax2.plot(df.index, df['dynamic_period'], label='Dynamic Period', color='orange', alpha=0.7)
    plt.title('Efficiency Ratio and Dynamic Period')
    plt.xlabel('Time')
    plt.ylabel('Efficiency Ratio')
    ax2.set_ylabel('Period')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('alpha_volatility_simple_example.png')
    plt.show()
    
    # 統計情報の表示
    print("=== Alpha Volatility Statistics ===")
    print(f"Percent Volatility Mean: {df['percent_volatility'].mean() * 100:.4f}%")
    print(f"Percent Volatility Max: {df['percent_volatility'].max() * 100:.4f}%")
    print(f"Absolute Volatility Mean: {df['absolute_volatility'].mean():.4f}")
    print(f"Absolute Volatility Max: {df['absolute_volatility'].max():.4f}")
    print(f"Average Dynamic Period: {df['dynamic_period'].mean():.1f}")
    
    # ボラティリティの乗数の例（リスク管理用）
    print("\n=== Volatility Multiples ===")
    multiples = [0.5, 1.0, 2.0, 3.0]
    
    # 最新のボラティリティ値
    latest_percent = df['percent_volatility'].iloc[-1] * 100  # パーセント表示
    latest_absolute = df['absolute_volatility'].iloc[-1]
    latest_price = df['close'].iloc[-1]
    
    print(f"Current Price: {latest_price:.2f}")
    print(f"Current Percent Volatility: {latest_percent:.4f}%")
    print(f"Current Absolute Volatility: {latest_absolute:.4f}")
    
    print("\nPercent-Based Volatility Multiples:")
    for mult in multiples:
        print(f"{mult}x Volatility: {latest_percent * mult:.4f}%")
    
    print("\nAbsolute Volatility Multiples:")
    for mult in multiples:
        print(f"{mult}x Volatility: {latest_absolute * mult:.4f}")
    
    # リスクに基づいたポジションサイジングの例
    print("\n=== Position Sizing Example ===")
    capital = 10000.0  # 資本金
    risk_percent = 0.01  # リスク許容度（資本金の1%）
    risk_amount = capital * risk_percent
    
    # 1ATRリスクに基づくポジションサイズ
    position_size = risk_amount / latest_absolute
    position_value = position_size * latest_price
    
    print(f"Capital: {capital:.2f}")
    print(f"Risk Amount (1% of capital): {risk_amount:.2f}")
    print(f"Position Size (shares/contracts): {position_size:.4f}")
    print(f"Position Value: {position_value:.2f}")
    print(f"This position risks approximately 1x the current volatility")


if __name__ == "__main__":
    main() 