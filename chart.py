#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import yaml
from pathlib import Path
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from data.data_loader import DataLoader , CSVDataSource
from data.data_processor import DataProcessor
from indicators.squeeze_momentum import SqueezeMomentum


def plot_squeeze_momentum(df: pd.DataFrame, val: pd.Series, sqz_on: pd.Series, sqz_off: pd.Series, no_sqz: pd.Series, ax=None, figsize=(15, 5)):
    """
    スクイーズモメンタムインディケーターをプロット
    
    Args:
        df: 価格データ
        val: モメンタム値
        sqz_on: スクイーズオン状態
        sqz_off: スクイーズオフ状態
        no_sqz: スクイーズなし状態
        ax: プロット先の軸（Noneの場合は新規作成）
        figsize: グラフのサイズ
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # モメンタム値の変化を計算
    val_diff = pd.Series(val).diff()
    
    # プラスとマイナスの値を分離
    pos_val = val.copy()
    neg_val = val.copy()
    pos_val[val_diff <= 0] = float('nan')
    neg_val[val_diff > 0] = float('nan')
    
    # モメンタム値をプロット
    ax.bar(df.index, pos_val, color='green', alpha=0.7, label='Increasing Momentum')
    ax.bar(df.index, neg_val, color='red', alpha=0.7, label='Decreasing Momentum')
    
    # スクイーズ状態をプロット
    ax.scatter(df.index[sqz_on], [0] * sum(sqz_on), color='red', marker='s', s=10, label='Squeeze On')
    ax.scatter(df.index[sqz_off], [0] * sum(sqz_off), color='green', marker='s', s=10, label='Squeeze Off')
    ax.scatter(df.index[no_sqz], [0] * sum(no_sqz), color='gray', marker='s', s=10, label='No Squeeze')
    
    # グラフの設定
    ax.set_title('Squeeze Momentum Indicator')
    ax.grid(True)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.legend()
    
    return ax


def plot_chart():
    """
    チャートをプロット
    """
    # 設定ファイルを読み込む
    config_path = Path("config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    print("\nデータを読み込んでいます...")
    
   # データの準備
    data_dir = config['data']['data_dir']
    data_loader = DataLoader(CSVDataSource(data_dir))
    data_processor = DataProcessor()

    
    # データの読み込みと処理
    print("\nLoading and processing data...")
    raw_data = data_loader.load_data_from_config(config)
    df = data_processor.process(raw_data)
    
    # スクイーズモメンタムを計算
    squeeze = SqueezeMomentum()
    val = squeeze.calculate(df)
    sqz_on, sqz_off, no_sqz = squeeze.get_squeeze_states()
    
    # プロット設定
    mc = mpf.make_marketcolors(
        up='red',
        down='blue',
        edge='inherit',
        wick='inherit',
        volume='in'
    )
    s = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle='--',
        y_on_right=True
    )
    
    # サブプロットの設定
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.1)
    
    # ローソク足チャートのプロット
    ax1 = fig.add_subplot(gs[0])
    mpf.plot(df, type='candle', style=s, ax=ax1)
    ax1.set_title('Price Chart')
    
    # スクイーズモメンタムのプロット
    ax2 = fig.add_subplot(gs[1])
    plot_squeeze_momentum(df, val, sqz_on, sqz_off, no_sqz, ax=ax2)
    
    plt.show()


if __name__ == "__main__":
    plot_chart()