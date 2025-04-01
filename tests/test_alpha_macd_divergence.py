#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from signals.implementations.divergence.alpha_macd_divergence import AlphaMACDDivergenceSignal


def test_alpha_macd_divergence():
    """アルファMACDダイバージェンスシグナルのテスト"""
    
    # テストデータの生成
    np.random.seed(42)
    n = 200
    
    # 価格データの生成（トレンドとノイズの組み合わせ）
    trend = np.cumsum(np.random.normal(0.1, 0.1, n))
    noise = np.random.normal(0, 1, n)
    price = trend + noise
    
    # ダイバージェンスを作成するために価格を調整
    # 強気ダイバージェンス: 価格が下がるがオシレーターが上がる
    price[150:170] = price[150] - np.linspace(0, 5, 20)
    price[170:190] = price[170] - np.linspace(0, 3, 20)
    
    # 弱気ダイバージェンス: 価格が上がるがオシレーターが下がる
    price[50:70] = price[50] + np.linspace(0, 5, 20)
    price[70:90] = price[70] + np.linspace(0, 8, 20)
    
    # データフレームの作成
    df = pd.DataFrame({
        'open': price,
        'high': price + np.random.normal(0, 0.5, n),
        'low': price - np.random.normal(0, 0.5, n),
        'close': price
    })
    
    # アルファMACDダイバージェンスシグナルの初期化
    signal = AlphaMACDDivergenceSignal(
        er_period=21,
        fast_max_kama_period=89,
        fast_min_kama_period=8,
        slow_max_kama_period=144,
        slow_min_kama_period=21,
        signal_max_kama_period=55,
        signal_min_kama_period=5,
        lookback=30
    )
    
    # シグナルの生成
    signals = signal.generate(df)
    
    # アルファMACDの値を取得
    alpha_macd_values = signal.get_alpha_macd_values(df)
    
    # 効率比の値を取得
    er_values = signal.get_efficiency_ratio(df)
    
    # 動的な期間の値を取得
    dynamic_periods = signal.get_dynamic_periods(df)
    
    # 結果の表示
    print(f"シグナル数: {np.sum(signals != 0)}")
    print(f"ロングシグナル数: {np.sum(signals == 1)}")
    print(f"ショートシグナル数: {np.sum(signals == -1)}")
    
    # プロット
    plt.figure(figsize=(12, 10))
    
    # 価格チャート
    plt.subplot(3, 1, 1)
    plt.plot(df['close'], label='価格')
    plt.scatter(np.where(signals == 1)[0], df['close'][signals == 1], marker='^', color='g', s=100, label='ロングシグナル')
    plt.scatter(np.where(signals == -1)[0], df['close'][signals == -1], marker='v', color='r', s=100, label='ショートシグナル')
    plt.title('価格とシグナル')
    plt.legend()
    
    # アルファMACD
    plt.subplot(3, 1, 2)
    plt.plot(alpha_macd_values['macd'], label='MACD線')
    plt.plot(alpha_macd_values['signal'], label='シグナル線')
    plt.bar(range(len(alpha_macd_values['histogram'])), alpha_macd_values['histogram'], label='ヒストグラム')
    plt.title('アルファMACD')
    plt.legend()
    
    # 効率比と動的期間
    plt.subplot(3, 1, 3)
    plt.plot(er_values, label='効率比')
    plt.plot(dynamic_periods['kama_period'], label='動的KAMAピリオド')
    plt.title('効率比と動的期間')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('alpha_macd_divergence_test.png')
    plt.show()


if __name__ == "__main__":
    test_alpha_macd_divergence() 