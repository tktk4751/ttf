#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
import os
from datetime import datetime, timedelta
import matplotlib as mpl

# 日本語フォント設定
plt.rcParams['font.family'] = 'sans-serif'
# Ubuntuの場合
plt.rcParams['font.sans-serif'] = ['IPAGothic', 'IPAPGothic', 'VL Gothic', 'Noto Sans CJK JP', 'Takao']

# フォントが見つからない場合のフォールバック
import matplotlib.font_manager as fm
fonts = set([f.name for f in fm.fontManager.ttflist])
if not any(font in fonts for font in plt.rcParams['font.sans-serif']):
    # フォールバック: 日本語をASCIIで置き換える
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    
    # 警告を無効化
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    
    # 日本語のタイトルや軸ラベルの場合はASCIIに置き換える
    USE_ASCII_LABELS = True
else:
    USE_ASCII_LABELS = False

# 親ディレクトリをパスに追加してインポートできるようにする
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicators import AlphaTrendIndex


def generate_synthetic_data(days=200, seed=42):
    """
    異なる市場状態を持つ合成データを生成する
    
    Args:
        days: 生成する日数
        seed: 乱数シード
        
    Returns:
        pd.DataFrame: 生成されたOHLCデータ
    """
    np.random.seed(seed)
    
    # 日付インデックスの作成
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # 初期値
    base_price = 100.0
    
    # 異なる市場状態
    prices = []
    current_price = base_price
    
    # ランダムウォーク部分（最初の30%）
    random_days = int(days * 0.3)
    for _ in range(random_days):
        volatility = np.random.uniform(0.5, 1.5)
        change = np.random.normal(0, volatility)
        current_price += change
        prices.append(current_price)
    
    # トレンド部分（次の30%）
    trend_days = int(days * 0.3)
    trend_strength = np.random.choice([0.5, -0.5])  # 上昇または下降トレンド
    for _ in range(trend_days):
        volatility = np.random.uniform(0.3, 1.0)
        change = trend_strength + np.random.normal(0, volatility)
        current_price += change
        prices.append(current_price)
    
    # レンジ相場部分（次の20%）
    range_days = int(days * 0.2)
    range_center = current_price
    for _ in range(range_days):
        # レンジ相場では中心に引き戻される力が働く
        mean_reversion = (range_center - current_price) * 0.1
        volatility = np.random.uniform(0.3, 0.8)
        change = mean_reversion + np.random.normal(0, volatility)
        current_price += change
        prices.append(current_price)
    
    # 強いトレンド部分（残りの20%）
    strong_trend_days = days - random_days - trend_days - range_days
    strong_trend = np.random.choice([1.0, -1.0])  # 強い上昇または下降トレンド
    for _ in range(strong_trend_days):
        volatility = np.random.uniform(0.2, 0.7)
        change = strong_trend + np.random.normal(0, volatility)
        current_price += change
        prices.append(current_price)
    
    # リストからndarrayに変換
    prices = np.array(prices)
    
    # OHLCデータの生成
    data = pd.DataFrame(index=date_range[:len(prices)])
    data['close'] = prices
    
    # 高値と安値の生成
    daily_volatility = np.random.uniform(0.5, 2.0, size=len(prices))
    data['high'] = data['close'] + daily_volatility
    data['low'] = data['close'] - daily_volatility
    
    # 始値の生成（前日の終値からランダムに変動）
    open_changes = np.random.normal(0, 0.5, size=len(prices))
    data['open'] = data['close'].shift(1) + open_changes
    data.loc[data.index[0], 'open'] = data['close'].iloc[0] - np.random.uniform(0, 1)
    
    # 列の順序を調整
    data = data[['open', 'high', 'low', 'close']]
    
    # 市場状態ラベルの追加
    market_state = np.empty(len(prices), dtype=object)
    market_state[:random_days] = 'ランダム'
    market_state[random_days:random_days+trend_days] = '中程度トレンド'
    market_state[random_days+trend_days:random_days+trend_days+range_days] = 'レンジ相場'
    market_state[random_days+trend_days+range_days:] = '強いトレンド'
    
    data['market_state'] = market_state
    
    return data


def main():
    """メイン関数"""
    # 合成データの生成
    print("合成データを生成中...")
    df = generate_synthetic_data(days=200, seed=42)
    
    # 初期の値（最初の30日）を除外
    start_idx = 30
    
    # アルファトレンドインデックスの計算
    print("アルファトレンドインデックスを計算中...")
    # デフォルトパラメータ
    alpha_trend_index = AlphaTrendIndex(
        er_period=21,             # 効率比の計算期間
        max_chop_period=21,       # チョピネス期間の最大値
        min_chop_period=8,        # チョピネス期間の最小値
        max_atr_period=21,        # ATR期間の最大値
        min_atr_period=10,        # ATR期間の最小値
        max_stddev_period=21,     # 標準偏差期間の最大値
        min_stddev_period=14,     # 標準偏差期間の最小値
        max_lookback_period=14,   # ルックバック期間の最大値
        min_lookback_period=7     # ルックバック期間の最小値
    )
    result = alpha_trend_index.calculate(
        df['open'].values, 
        df['high'].values, 
        df['low'].values, 
        df['close'].values
    )
    trend_index_values = result.values
    
    # 必要に応じて平滑化を適用（例：5期間SMA）
    smooth_period = 5
    smoothed_values = np.zeros_like(trend_index_values)
    for i in range(len(trend_index_values)):
        if i < smooth_period:
            # 不完全なウィンドウでは利用可能なデータの平均を取る
            smoothed_values[i] = np.mean(trend_index_values[:i+1])
        else:
            # 完全なウィンドウ
            smoothed_values[i] = np.mean(trend_index_values[i-smooth_period+1:i+1])
    
    # プロットの作成
    print("\n結果をプロット中...")
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # 価格チャート (初期値を除外)
    ax[0].plot(df.index[start_idx:], df['close'][start_idx:], 'k-', linewidth=1.5, label='価格' if not USE_ASCII_LABELS else 'Price')
    ax[0].set_ylabel('価格' if not USE_ASCII_LABELS else 'Price')
    ax[0].set_title('価格チャート' if not USE_ASCII_LABELS else 'Price Chart')
    ax[0].grid(True)
    ax[0].legend(loc='upper left')
    
    # 市場状態に応じた背景色
    colors = {'ランダム': 'lightyellow', '中程度トレンド': 'lightgreen', 'レンジ相場': 'lightblue', '強いトレンド': 'lightcoral'}
    
    if USE_ASCII_LABELS:
        # 英語の市場状態ラベル
        color_map = {
            'ランダム': 'Random',
            '中程度トレンド': 'Moderate Trend',
            'レンジ相場': 'Range',
            '強いトレンド': 'Strong Trend'
        }
        # 色のマッピングを英語に変更
        new_colors = {}
        for k, v in colors.items():
            new_colors[color_map.get(k, k)] = v
        colors = new_colors
        
        # データフレームの市場状態ラベルも英語に変換
        df['market_state'] = df['market_state'].map(lambda x: color_map.get(x, x))
    
    current_state = df['market_state'].iloc[start_idx]
    state_start = df.index[start_idx]
    
    for i, (idx, row) in enumerate(df.iloc[start_idx:].iterrows()):
        if row['market_state'] != current_state or i == len(df.iloc[start_idx:]) - 1:
            ax[0].axvspan(state_start, idx, alpha=0.3, color=colors[current_state])
            current_state = row['market_state']
            state_start = idx
    
    # アルファトレンドインデックス (初期値を除外)
    ax[1].plot(df.index[start_idx:], trend_index_values[start_idx:], 'b-', linewidth=2, 
               label='アルファトレンドインデックス' if not USE_ASCII_LABELS else 'Alpha Trend Index')
    ax[1].plot(df.index[start_idx:], smoothed_values[start_idx:], 'r-', linewidth=1.5, 
               label=f'平滑化トレンドインデックス (SMA-{smooth_period})' if not USE_ASCII_LABELS else f'Smoothed Trend Index (SMA-{smooth_period})')
    ax[1].axhline(y=0.7, color='g', linestyle='--', alpha=0.7, 
                  label='トレンド閾値 (0.7)' if not USE_ASCII_LABELS else 'Trend Threshold (0.7)')
    ax[1].axhline(y=0.3, color='r', linestyle='--', alpha=0.7, 
                  label='レンジ閾値 (0.3)' if not USE_ASCII_LABELS else 'Range Threshold (0.3)')
    ax[1].set_ylabel('指標値' if not USE_ASCII_LABELS else 'Indicator Value')
    ax[1].set_ylim(0, 1)
    ax[1].grid(True)
    ax[1].legend(loc='upper left')
    
    # X軸の設定
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    
    # レイアウトの調整とタイトル
    plt.tight_layout()
    title = 'アルファトレンドインデックス - シンプル表示' if not USE_ASCII_LABELS else 'Alpha Trend Index - Simple Display'
    plt.suptitle(title, fontsize=16, y=1.02)
    
    # 保存と表示
    plt.savefig('alpha_trend_index_simple.png', bbox_inches='tight')
    # プロットを非ブロッキングモードで表示（またはコメントアウト）
    plt.show(block=False)
    # plt.show()
    
    print(f"処理が完了しました。結果は 'alpha_trend_index_simple.png' に保存されました。")


if __name__ == "__main__":
    main()
    
    # テスト用コード：直接アルファトレンドインデックスの値を計算して出力
    print("\n====== デバッグ情報 ======")
    df = generate_synthetic_data(days=200, seed=42)
    alpha_trend_index = AlphaTrendIndex(
        er_period=21,             # 効率比の計算期間
        max_chop_period=21,       # チョピネス期間の最大値
        min_chop_period=8,        # チョピネス期間の最小値
        max_atr_period=21,        # ATR期間の最大値
        min_atr_period=10,        # ATR期間の最小値
        max_stddev_period=21,     # 標準偏差期間の最大値
        min_stddev_period=14,     # 標準偏差期間の最小値
        max_lookback_period=14,   # ルックバック期間の最大値
        min_lookback_period=7     # ルックバック期間の最小値
    )
    result = alpha_trend_index.calculate(
        df['open'].values, 
        df['high'].values, 
        df['low'].values, 
        df['close'].values
    )
    trend_index_values = result.values
    
    # 平滑化の例（5期間SMA）
    smooth_period = 5
    smoothed_values = pd.Series(trend_index_values).rolling(smooth_period).mean().values
    
    # 値の統計を出力
    print("\nアルファトレンドインデックスの統計:")
    print(f"最小値: {np.min(trend_index_values):.4f}")
    print(f"最大値: {np.max(trend_index_values):.4f}")
    print(f"平均値: {np.mean(trend_index_values):.4f}")
    print(f"中央値: {np.median(trend_index_values):.4f}")
    
    print("\n平滑化したアルファトレンドインデックスの統計:")
    non_nan_values = smoothed_values[~np.isnan(smoothed_values)]
    print(f"最小値: {np.min(non_nan_values):.4f}")
    print(f"最大値: {np.max(non_nan_values):.4f}")
    print(f"平均値: {np.mean(non_nan_values):.4f}")
    print(f"中央値: {np.median(non_nan_values):.4f}")
    
    # 最初の20個の値を出力
    print("\nアルファトレンドインデックスの最初の20個の値:")
    for i in range(min(20, len(trend_index_values))):
        print(f"[{i}]: {trend_index_values[i]:.4f}")
    
    # 標準偏差係数の統計も出力
    stddev_factor = alpha_trend_index.get_stddev_factor()
    print("\n標準偏差係数の統計:")
    print(f"最小値: {np.min(stddev_factor):.4f}")
    print(f"最大値: {np.max(stddev_factor):.4f}")
    print(f"平均値: {np.mean(stddev_factor):.4f}")
    
    # チョピネス指数の統計
    chop = alpha_trend_index.get_choppiness_index()
    print("\nチョピネス指数の統計:")
    print(f"最小値: {np.min(chop):.4f}")
    print(f"最大値: {np.max(chop):.4f}")
    print(f"平均値: {np.mean(chop):.4f}")
    
    # Range Indexの統計
    range_index = alpha_trend_index.get_range_index()
    print("\nRange Indexの統計:")
    print(f"最小値: {np.min(range_index):.4f}")
    print(f"最大値: {np.max(range_index):.4f}")
    print(f"平均値: {np.mean(range_index):.4f}")
    print("==========================") 