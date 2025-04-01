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
from scipy.signal import find_peaks
from scipy import signal

from indicators import AlphaVIX, AlphaATR, AlphaVolatility


def generate_sample_data(n=500):
    """サンプルデータを生成する関数"""
    np.random.seed(42)
    
    # 基本的なランダムウォーク
    price = 100.0
    prices = [price]
    volatility = 0.01
    
    # 異なるボラティリティ状態を作成
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
    
    # 動的スムージングありのアルファVIX
    alpha_vix_dynamic = AlphaVIX(
        er_period=21,
        smoothing_period=14,
        use_dynamic_smoothing=True  # 動的スムージングを使用
    )
    
    # 動的スムージングなしのアルファVIX
    alpha_vix_fixed = AlphaVIX(
        er_period=21,
        smoothing_period=14,
        use_dynamic_smoothing=False  # 固定期間スムージングを使用
    )
    
    # 比較のため、個別のインジケーターも計算
    alpha_atr = AlphaATR(
        er_period=21,
        max_atr_period=89,
        min_atr_period=13
    )
    
    alpha_vol = AlphaVolatility(
        er_period=21,
        max_vol_period=89,
        min_vol_period=13,
        smoothing_period=14,
        use_dynamic_smoothing=True  # 動的スムージングを使用
    )
    
    # 各インジケーターの計算
    alpha_vix_dynamic_values = alpha_vix_dynamic.calculate(df)
    alpha_vix_fixed_values = alpha_vix_fixed.calculate(df)
    alpha_atr_values = alpha_atr.calculate(df)
    alpha_vol_values = alpha_vol.calculate(df)
    
    # 追加情報の取得
    er_values = alpha_vix_dynamic.get_efficiency_ratio()
    dynamic_weight = alpha_vix_dynamic.get_dynamic_weight()
    dynamic_period = alpha_vix_dynamic.get_dynamic_period()  # 動的期間を取得
    
    # 金額ベースのボラティリティ値を取得
    alpha_vix_absolute = alpha_vix_dynamic.get_absolute_values()
    alpha_vol_absolute = alpha_vol.get_absolute_standard_deviation()
    alpha_atr_absolute = alpha_atr.get_absolute_values()
    
    # 結果をDataFrameに追加
    df['alpha_vix_dynamic'] = alpha_vix_dynamic_values
    df['alpha_vix_fixed'] = alpha_vix_fixed_values
    df['alpha_atr'] = alpha_atr_values
    df['alpha_vol'] = alpha_vol_values  # すでに0-100にスケーリングされている
    df['er'] = er_values
    df['atr_weight'] = dynamic_weight
    df['vol_weight'] = 1.0 - dynamic_weight
    df['dynamic_period'] = dynamic_period  # 動的期間を追加
    df['vix_absolute'] = alpha_vix_absolute
    df['vol_absolute'] = alpha_vol_absolute
    df['atr_absolute'] = alpha_atr_absolute
    
    # グラフの描画
    fig = plt.figure(figsize=(15, 18))  # 高さを増やして新しいサブプロットを追加
    gs = GridSpec(8, 1, height_ratios=[3, 1.5, 1.5, 1.5, 1, 1, 1, 1.5])
    
    # 価格チャート
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df.index, df['close'], label='Close', color='black')
    ax1.fill_between(df.index, df['low'], df['high'], color='lightgray', alpha=0.3)
    ax1.set_title('Price Chart')
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 動的スムージングと固定スムージングの比較
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(df.index, df['alpha_vix_dynamic'], label='Dynamic Smoothing', color='red', linewidth=2)
    ax2.plot(df.index, df['alpha_vix_fixed'], label='Fixed Smoothing', color='blue', linewidth=1, alpha=0.8)
    ax2.set_title('Alpha VIX: Dynamic vs Fixed Smoothing')
    ax2.set_ylabel('Value')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # アルファVIX
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(df.index, df['alpha_vix_dynamic'], label='AlphaVIX', color='red', linewidth=2)
    ax3.set_title('Alpha VIX (Dynamic Smoothing)')
    ax3.set_ylabel('Value')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # コンポーネント比較
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.plot(df.index, df['alpha_vix_dynamic'], label='AlphaVIX', color='red', alpha=0.8)
    ax4.plot(df.index, df['alpha_atr'], label='AlphaATR', color='blue', alpha=0.6)
    ax4.plot(df.index, df['alpha_vol'], label='AlphaVolatility', color='green', alpha=0.6)
    ax4.set_title('Component Comparison')
    ax4.set_ylabel('Value')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # 効率比
    ax5 = fig.add_subplot(gs[4], sharex=ax1)
    ax5.plot(df.index, df['er'], label='Efficiency Ratio', color='purple')
    ax5.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax5.set_title('Efficiency Ratio')
    ax5.set_ylabel('Value')
    ax5.set_ylim(0, 1)
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # 動的重み
    ax6 = fig.add_subplot(gs[5], sharex=ax1)
    ax6.plot(df.index, df['atr_weight'], label='ATR Weight', color='blue')
    ax6.plot(df.index, df['vol_weight'], label='Volatility Weight', color='green')
    ax6.set_title('Dynamic Weights')
    ax6.set_ylabel('Weight')
    ax6.set_ylim(0, 1)
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    # 動的期間
    ax7 = fig.add_subplot(gs[6], sharex=ax1)
    ax7.plot(df.index, df['dynamic_period'], label='Dynamic Period', color='orange')
    ax7.set_title('Dynamic Smoothing Period')
    ax7.set_ylabel('Period')
    ax7.set_ylim(0, 100)  # 適切な範囲に調整
    ax7.grid(True, alpha=0.3)
    ax7.legend()
    
    # 金額ベースのボラティリティ
    ax8 = fig.add_subplot(gs[7], sharex=ax1)
    ax8.plot(df.index, df['vix_absolute'], label='Absolute AlphaVIX', color='red')
    ax8.plot(df.index, df['vol_absolute'], label='Absolute Volatility', color='green')
    ax8.plot(df.index, df['atr_absolute'], label='Absolute ATR', color='blue')
    ax8.set_title('Absolute (Price-Based) Volatility')
    ax8.set_xlabel('Time')
    ax8.set_ylabel('Value (Currency)')
    ax8.grid(True, alpha=0.3)
    ax8.legend()
    
    plt.tight_layout()
    plt.savefig('alpha_vix_example.png')
    plt.show()
    
    # 統計情報の表示
    print("=== Alpha VIX Statistics ===")
    print(f"Mean (Dynamic): {df['alpha_vix_dynamic'].mean():.6f}")
    print(f"Mean (Fixed): {df['alpha_vix_fixed'].mean():.6f}")
    print(f"Std (Dynamic): {df['alpha_vix_dynamic'].std():.6f}")
    print(f"Std (Fixed): {df['alpha_vix_fixed'].std():.6f}")
    print(f"Min (Dynamic): {df['alpha_vix_dynamic'].min():.6f}")
    print(f"Min (Fixed): {df['alpha_vix_fixed'].min():.6f}")
    print(f"Max (Dynamic): {df['alpha_vix_dynamic'].max():.6f}")
    print(f"Max (Fixed): {df['alpha_vix_fixed'].max():.6f}")
    print(f"ATR Weight Range: {df['atr_weight'].min():.2f} - {df['atr_weight'].max():.2f}")
    print(f"VOL Weight Range: {df['vol_weight'].min():.2f} - {df['vol_weight'].max():.2f}")
    print(f"Dynamic Period Range: {df['dynamic_period'].min():.2f} - {df['dynamic_period'].max():.2f}")
    
    # 反応速度の分析
    print("\n=== Response Speed Analysis ===")
    # ピークを見つける
    dynamic_peaks, _ = find_peaks(df['alpha_vix_dynamic'], height=df['alpha_vix_dynamic'].mean(), distance=10)
    fixed_peaks, _ = find_peaks(df['alpha_vix_fixed'], height=df['alpha_vix_fixed'].mean(), distance=10)
    
    # ピーク検出に基づいて反応速度を分析
    if len(dynamic_peaks) > 0 and len(fixed_peaks) > 0:
        # 価格変動からボラティリティピークまでの遅延を計算
        # (これは簡略化した例です。実際にはより複雑な分析が必要)
        print(f"Dynamic Smoothing Peaks: {len(dynamic_peaks)}")
        print(f"Fixed Smoothing Peaks: {len(fixed_peaks)}")
        
        # 相関関係の遅延を計算
        corr = signal.correlate(df['alpha_vix_dynamic'], df['alpha_vix_fixed'])
        lags = signal.correlation_lags(len(df['alpha_vix_dynamic']), len(df['alpha_vix_fixed']))
        lag = lags[np.argmax(corr)]
        
        if lag > 0:
            print(f"Dynamic smoothing leads fixed by {lag} periods")
        elif lag < 0:
            print(f"Fixed smoothing leads dynamic by {-lag} periods")
        else:
            print("No significant lead/lag relationship")
    
    # 動的スムージングと固定スムージングの相関
    smoothing_corr = df[['alpha_vix_dynamic', 'alpha_vix_fixed']].corr().iloc[0, 1]
    print(f"Correlation between Dynamic and Fixed Smoothing: {smoothing_corr:.6f}")
    
    # 金額ベースのボラティリティ統計
    print("\n=== Absolute Volatility Statistics ===")
    print(f"VIX Absolute Mean: {df['vix_absolute'].mean():.6f}")
    print(f"VIX Absolute Max: {df['vix_absolute'].max():.6f}")
    print(f"VOL Absolute Mean: {df['vol_absolute'].mean():.6f}")
    print(f"VOL Absolute Max: {df['vol_absolute'].max():.6f}")
    print(f"ATR Absolute Mean: {df['atr_absolute'].mean():.6f}")
    print(f"ATR Absolute Max: {df['atr_absolute'].max():.6f}")
    
    # 各インジケーターの相関
    corr_matrix = df[['alpha_vix_dynamic', 'alpha_vix_fixed', 'alpha_atr', 'alpha_vol', 'vix_absolute', 'vol_absolute', 'atr_absolute', 'dynamic_period']].corr()
    print("\n=== Correlation Matrix ===")
    print(corr_matrix)
    
    # 金額ベースの値同士の相関関係に特に注目
    print("\n=== Absolute Values Correlation ===")
    abs_corr = df[['vix_absolute', 'vol_absolute', 'atr_absolute']].corr()
    print(abs_corr)
    
    # 相対値と金額ベースの値の相関
    print("\n=== Relative vs Absolute Correlation ===")
    rel_abs_corr = pd.DataFrame({
        'Alpha VIX vs VIX Absolute': [df['alpha_vix_dynamic'].corr(df['vix_absolute'])],
        'Alpha VOL vs VOL Absolute': [df['alpha_vol'].corr(df['vol_absolute'])],
        'Alpha ATR vs ATR Absolute': [df['alpha_atr'].corr(df['atr_absolute'])],
    })
    print(rel_abs_corr)
    
    # 金額ベースの値のボラティリティ比較
    print("\n=== Absolute Values Volatility ===")
    print(f"VIX Absolute Std: {df['vix_absolute'].std():.6f}")
    print(f"VOL Absolute Std: {df['vol_absolute'].std():.6f}")
    print(f"ATR Absolute Std: {df['atr_absolute'].std():.6f}")
    
    # 金額ベースの値の変動係数 (標準偏差/平均) - ボラティリティの正規化比較
    print("\n=== Coefficient of Variation (Std/Mean) ===")
    vix_cv = df['vix_absolute'].std() / df['vix_absolute'].mean() if df['vix_absolute'].mean() != 0 else float('nan')
    vol_cv = df['vol_absolute'].std() / df['vol_absolute'].mean() if df['vol_absolute'].mean() != 0 else float('nan')
    atr_cv = df['atr_absolute'].std() / df['atr_absolute'].mean() if df['atr_absolute'].mean() != 0 else float('nan')
    
    print(f"VIX Absolute CV: {vix_cv:.6f}")
    print(f"VOL Absolute CV: {vol_cv:.6f}")
    print(f"ATR Absolute CV: {atr_cv:.6f}")
    
    # 各金額ベースの値が最大になるタイミングの一致度を分析
    vix_max_idx = df['vix_absolute'].idxmax()
    vol_max_idx = df['vol_absolute'].idxmax()
    atr_max_idx = df['atr_absolute'].idxmax()
    
    print("\n=== Timing of Maximum Values ===")
    print(f"VIX Absolute Max at index: {vix_max_idx}")
    print(f"VOL Absolute Max at index: {vol_max_idx}")
    print(f"ATR Absolute Max at index: {atr_max_idx}")
    print(f"Index difference VIX-VOL: {abs(vix_max_idx - vol_max_idx)}")
    print(f"Index difference VIX-ATR: {abs(vix_max_idx - atr_max_idx)}")
    print(f"Index difference VOL-ATR: {abs(vol_max_idx - atr_max_idx)}")
    
    # 動的スムージングの効果分析
    print("\n=== Dynamic Smoothing Analysis ===")
    # 効率比（ER）と動的期間の関係を分析
    corr_er_period = df[['er', 'dynamic_period']].corr().iloc[0, 1]
    print(f"Correlation between ER and Dynamic Period: {corr_er_period:.6f}")
    print(f"Average Dynamic Period: {df['dynamic_period'].mean():.2f}")
    
    # トレンド期間とレンジ期間での動的期間の違いを分析
    trend_mask = df['er'] > 0.6  # 強いトレンド
    range_mask = df['er'] < 0.4  # 弱いトレンド（レンジ）
    
    trend_period_avg = df.loc[trend_mask, 'dynamic_period'].mean() if trend_mask.any() else 0
    range_period_avg = df.loc[range_mask, 'dynamic_period'].mean() if range_mask.any() else 0
    
    print(f"Average Period during Strong Trends (ER > 0.6): {trend_period_avg:.2f}")
    print(f"Average Period during Ranges (ER < 0.4): {range_period_avg:.2f}")
    
    # 簡単な市場状態分析
    high_vol_periods = (df['alpha_vix_dynamic'] > df['alpha_vix_dynamic'].quantile(0.8)).sum()
    low_vol_periods = (df['alpha_vix_dynamic'] < df['alpha_vix_dynamic'].quantile(0.2)).sum()
    
    print("\n=== Market State Analysis ===")
    print(f"High Volatility Periods: {high_vol_periods} ({high_vol_periods/len(df)*100:.1f}%)")
    print(f"Low Volatility Periods: {low_vol_periods} ({low_vol_periods/len(df)*100:.1f}%)")
    
    # 金額ベースのボラティリティを使用した取引サイジングの例
    print("\n=== Position Sizing Example ===")
    capital = 10000.0  # 資金
    risk_percent = 0.01  # リスク率（資金の1%）
    
    # 金額ベースのボラティリティを使用してリスクに基づいたポジションサイズを計算
    # 例: 1標準偏差のムーブに対応するリスク
    risk_amount = capital * risk_percent
    last_price = df['close'].iloc[-1]
    last_vix_absolute = df['vix_absolute'].iloc[-1]
    
    # ボラティリティに基づくポジションサイズ（単位: 通貨）
    position_size = risk_amount / last_vix_absolute if last_vix_absolute > 0 else 0
    
    print(f"Current Price: {last_price:.2f}")
    print(f"Absolute VIX: {last_vix_absolute:.2f} (currency)")
    print(f"Capital: {capital:.2f}")
    print(f"Risk Amount (1% of capital): {risk_amount:.2f}")
    print(f"Position Size: {position_size:.2f} units")
    print(f"Position Value: {position_size * last_price:.2f}")


if __name__ == "__main__":
    main() 