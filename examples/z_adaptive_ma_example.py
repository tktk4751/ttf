#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from indicators.z_adaptive_ma import ZAdaptiveMA
from indicators.cycle_efficiency_ratio import CycleEfficiencyRatio

def generate_sample_data(days: int = 100) -> pd.DataFrame:
    """サンプルデータを生成する"""
    dates = [datetime.now() - timedelta(days=x) for x in range(days)]
    dates.reverse()
    
    # ランダムな価格データを生成
    np.random.seed(42)
    close = pd.Series(np.random.normal(100, 2, days).cumsum() + 1000)
    high = close + np.random.uniform(1, 3, days)
    low = close - np.random.uniform(1, 3, days)
    open_price = close.shift(1)
    open_price[0] = close[0] - np.random.uniform(1, 3)
    
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close
    }, index=dates)
    
    return df

def plot_z_adaptive_ma(data: pd.DataFrame, z_ma_values: np.ndarray, er_values: np.ndarray):
    """ZAdaptiveMAと効率比をプロットする"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # 価格とZAdaptiveMAのプロット
    ax1.plot(data.index, data['close'], label='Close', color='blue', alpha=0.5)
    ax1.plot(data.index, z_ma_values, label='ZAdaptiveMA', color='red', linewidth=2)
    ax1.set_title('ZAdaptiveMA Example')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    # 効率比のプロット
    ax2.plot(data.index, er_values, label='Efficiency Ratio', color='green')
    ax2.set_ylabel('Efficiency Ratio')
    ax2.set_ylim(0, 1)
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    # サンプルデータの生成
    data = generate_sample_data(days=100)
    
    # サイクル効率比の計算
    cer = CycleEfficiencyRatio(period=10)
    er_values = cer.calculate(data)
    
    # ZAdaptiveMAの計算
    z_ma = ZAdaptiveMA(fast_period=2, slow_period=30, src_type='close')
    z_ma_values = z_ma.calculate(data, external_er=er_values)
    
    # 結果のプロット
    plot_z_adaptive_ma(data, z_ma_values, er_values)

if __name__ == "__main__":
    main() 