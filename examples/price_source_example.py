#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
価格ソース（PriceSource）の例

異なる価格計算方法（HL2, HLC3, OHLC4など）を比較するサンプル
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# 親ディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 価格ソースインジケーターをインポート
from indicators import PriceSource


def generate_sample_data(length=200, noise_level=0.3):
    """サンプルのOHLCデータを生成する"""
    np.random.seed(42)  # 再現性のために乱数シードを固定
    
    # 価格の初期値
    base_price = 100.0
    
    # 日時の配列を生成
    dates = [datetime.now() + timedelta(days=i) for i in range(length)]
    
    # ランダムウォークのノイズを生成
    noise = np.cumsum(np.random.normal(0, noise_level, length))
    
    # 基本価格にトレンドとノイズを追加
    close = base_price + noise
    
    # 日中の変動を模擬
    daily_volatility = np.abs(np.random.normal(0, 1.5, length))
    
    # 始値、高値、安値を生成
    high = close + daily_volatility
    low = close - daily_volatility
    open_prices = close - np.random.normal(0, daily_volatility, length) * 0.5
    
    # 配列をNumPyに変換
    open_array = np.array(open_prices)
    high_array = np.array(high)
    low_array = np.array(low)
    close_array = np.array(close)
    
    # DataFrameを作成
    data = pd.DataFrame({
        'open': open_array,
        'high': high_array,
        'low': low_array,
        'close': close_array
    })
    
    # 日付インデックスを設定
    data.index = dates
    
    return data


def main():
    """メイン関数"""
    print("価格ソースのサンプルデータを生成中...")
    # サンプルデータの生成
    data = generate_sample_data(length=100)
    
    print("価格ソースを計算中...")
    # 価格ソースインジケーターのインスタンス化と計算
    price_source = PriceSource(weighted_close_factor=2.0)
    sources = price_source.calculate(data)
    
    # 各ソースの取得
    open_prices = price_source.get_open()
    high_prices = price_source.get_high()
    low_prices = price_source.get_low()
    close_prices = price_source.get_close()
    hl2_prices = price_source.get_hl2()
    hlc3_prices = price_source.get_hlc3()
    ohlc4_prices = price_source.get_ohlc4()
    hlcc4_prices = price_source.get_hlcc4()
    weighted_close_prices = price_source.get_weighted_close()
    
    # 統計情報の出力
    print("\n*** 価格ソース統計情報 ***")
    for source_name, source_data in [
        ('Open', open_prices),
        ('High', high_prices),
        ('Low', low_prices),
        ('Close', close_prices),
        ('HL2', hl2_prices),
        ('HLC3', hlc3_prices),
        ('OHLC4', ohlc4_prices),
        ('HLCC4', hlcc4_prices),
        ('Weighted Close', weighted_close_prices)
    ]:
        print(f"{source_name}:")
        print(f"  平均: {np.mean(source_data):.4f}")
        print(f"  標準偏差: {np.std(source_data):.4f}")
        print(f"  最小: {np.min(source_data):.4f}")
        print(f"  最大: {np.max(source_data):.4f}")
    
    print("\n結果をプロット中...")
    # プロットの設定
    plt.figure(figsize=(15, 10))
    plt.style.use('ggplot')
    
    # サブプロットの設定
    ax1 = plt.subplot(2, 1, 1)  # OHLC
    ax2 = plt.subplot(2, 1, 2)  # 価格ソース比較
    
    # OHLCチャートのプロット
    ax1.plot(data.index, open_prices, 'o-', color='blue', alpha=0.5, markersize=2, label='Open')
    ax1.plot(data.index, high_prices, '-', color='green', alpha=0.5, label='High')
    ax1.plot(data.index, low_prices, '-', color='red', alpha=0.5, label='Low')
    ax1.plot(data.index, close_prices, 'o-', color='black', alpha=0.8, markersize=2, label='Close')
    
    # 価格範囲表示（高値～安値）
    ax1.fill_between(data.index, low_prices, high_prices, color='gray', alpha=0.2)
    
    ax1.set_title('OHLC Price Chart', fontsize=14)
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 各価格ソースの比較プロット
    colors = {
        'HL2': 'purple',
        'HLC3': 'orange', 
        'OHLC4': 'brown',
        'HLCC4': 'teal',
        'Weighted Close': 'blue'
    }
    
    for source_name, source_data, color in [
        ('HL2', hl2_prices, colors['HL2']),
        ('HLC3', hlc3_prices, colors['HLC3']),
        ('OHLC4', ohlc4_prices, colors['OHLC4']),
        ('HLCC4', hlcc4_prices, colors['HLCC4']),
        ('Weighted Close', weighted_close_prices, colors['Weighted Close'])
    ]:
        ax2.plot(data.index, source_data, '-', color=color, linewidth=1.5, label=source_name)
    
    # 比較のためにCloseも表示
    ax2.plot(data.index, close_prices, '--', color='black', alpha=0.6, linewidth=1, label='Close')
    
    ax2.set_title('Price Source Comparison', fontsize=14)
    ax2.set_ylabel('Price')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # X軸の日付フォーマットを設定
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig('price_source_example.png', dpi=150)
    plt.show()
    
    print("処理完了！結果は 'price_source_example.png' に保存されました。")


if __name__ == "__main__":
    main() 