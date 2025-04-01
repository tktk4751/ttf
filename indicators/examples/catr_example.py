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

from indicators.c_atr import CATR
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
    
    # CATRを計算
    catr = CATR(
        detector_type='hody',  # ホモダイン判別機を使用
        cycle_part=0.5,
        max_cycle=55,
        min_cycle=5,
        max_output=34,
        min_output=5,
        smoother_type='alma'   # 'alma'または'hyper'
    )
    catr_values = catr.calculate(df, cer_values)
    
    # 各種値を取得
    dc_values = catr.get_dc_values()
    atr_period = catr.get_atr_period()
    er_values = catr.get_efficiency_ratio()
    percent_atr = catr.get_percent_atr()  # パーセントベース（×100）
    absolute_atr = catr.get_absolute_atr()  # 金額ベース
    
    # 結果をプロット
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(4, 1, height_ratios=[3, 1, 1, 1])
    
    # 価格とATRチャネル
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df.index, df['Close'], label='Close', alpha=0.7, color='gray')
    
    # ATRチャネルを作成（終値 ± 2 ATR）
    upper_band = df['Close'] + absolute_atr * 2
    lower_band = df['Close'] - absolute_atr * 2
    ax1.fill_between(df.index, upper_band, lower_band, alpha=0.2, color='blue', label='ATRチャネル (±2ATR)')
    
    ax1.set_title('BTC-USD CATRインジケーター')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ATR値
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(df.index, percent_atr, label='%ATR (×100)', color='blue')
    ax2.set_ylabel('ATR (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # ドミナントサイクル（ATR期間）
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(df.index, atr_period, label='ATR期間', color='purple')
    ax3.set_ylabel('期間')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # サイクル効率比(CER)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.plot(df.index, er_values, label='サイクル効率比', color='green')
    ax4.set_ylabel('効率比')
    ax4.set_ylim(-1.1, 1.1)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 