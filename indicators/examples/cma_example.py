#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import yfinance as yf

# indicators モジュールをインポート
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from indicators.c_ma import CMA
from indicators.cycle_efficiency_ratio import CycleEfficiencyRatio

# データ取得
def get_data(symbol='SPY', period='1y', interval='1d'):
    """Yahoo Financeからデータを取得"""
    df = yf.download(symbol, period=period, interval=interval)
    return df

# メイン処理
def main():
    # データを取得
    df = get_data(symbol='BTC-USD', period='6mo', interval='1d')
    
    # サイクル効率比（CER）を計算
    cer = CycleEfficiencyRatio(
        detector_type='hody',  # ホモダイン判別機を使用
        cycle_part=0.5,
        max_cycle=144,
        min_cycle=5,
        max_output=89,
        min_output=5,
        src_type='hlc3'
    )
    cer_values = cer.calculate(df)
    
    # CMAを計算
    cma = CMA(
        detector_type='hody',  # ホモダイン判別機を使用
        cycle_part=0.5,
        max_cycle=144,
        min_cycle=5,
        max_output=34,
        min_output=8,
        fast_period=2,         # 固定値
        slow_period=30,        # 固定値
        src_type='hlc3'
    )
    cma_values = cma.calculate(df, cer_values)
    
    # ドミナントサイクルと効率比を取得
    dc_values = cma.get_dc_values()
    kama_period = cma.get_kama_period()
    er_values = cma.get_efficiency_ratio()
    sc_values = cma.get_smoothing_constants()
    
    # 結果をプロット
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(4, 1, height_ratios=[3, 1, 1, 1])
    
    # 価格とCMA
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df.index, df['Close'], label='Close', alpha=0.7, color='gray')
    ax1.plot(df.index, cma_values, label='CMA', linewidth=2, color='blue')
    ax1.set_title('BTC-USD CMAインジケーター')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ドミナントサイクル
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(df.index, dc_values, label='ドミナントサイクル', color='purple')
    ax2.set_ylabel('期間')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # サイクル効率比(CER)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(df.index, er_values, label='サイクル効率比', color='green')
    ax3.set_ylabel('効率比')
    ax3.set_ylim(-1.1, 1.1)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # スムージング定数
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.plot(df.index, sc_values, label='スムージング定数', color='red')
    ax4.set_ylabel('SC')
    ax4.set_ylim(0, 1.1)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 