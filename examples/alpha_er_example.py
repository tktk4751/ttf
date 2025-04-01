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

from indicators import AlphaER
from indicators.efficiency_ratio import EfficiencyRatio
from indicators.cumulative_er import CumulativeER


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
    
    # 各期間でのEfficiency Ratioとそのアルファバージョンを比較
    periods = [14, 21]
    
    # マルチプロットのセットアップ（修正済み）
    plt.figure(figsize=(15, 12))
    
    # サブプロット数の計算
    total_plots = 1 + len(periods) + 1  # 価格 + 各期間のER比較 + 正規化ER
    
    # 価格チャート
    plt.subplot(total_plots, 1, 1)
    plt.plot(df.index, df['close'], label='Close', color='black')
    plt.fill_between(df.index, df['low'], df['high'], color='lightgray', alpha=0.3)
    plt.title('Price Chart')
    plt.ylabel('Price')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 各期間でのEfficiency RatioとAlpha ERを計算・プロット
    for i, period in enumerate(periods):
        # 標準のEfficiency Ratio
        er = EfficiencyRatio(period=period)
        er_values = er.calculate(df)
        
        # 累積的Efficiency Ratio
        cumulative_er = CumulativeER(period=period)
        cumulative_er_values = cumulative_er.calculate(df)
        
        # Alpha ER
        alpha_er = AlphaER(period=period, normalize=False)  # 0-1スケールで比較
        alpha_er_values = alpha_er.calculate(df)
        raw_er = alpha_er.get_raw_efficiency_ratio()
        
        # DataFrameに結果を追加
        df[f'er_{period}'] = er_values
        df[f'cumulative_er_{period}'] = cumulative_er_values
        df[f'alpha_er_{period}'] = alpha_er_values
        df[f'raw_er_{period}'] = raw_er  # 参照用
        
        # プロット
        plt.subplot(total_plots, 1, 2 + i)
        plt.plot(df.index, df[f'er_{period}'], label=f'ER ({period})', color='blue')
        plt.plot(df.index, df[f'cumulative_er_{period}'], label=f'Cumulative ER ({period})', color='green')
        plt.plot(df.index, df[f'alpha_er_{period}'], label=f'Alpha ER ({period})', color='red', linewidth=2)
        
        # フィボナッチレベルの水平線
        plt.axhline(y=0.618, color='gray', linestyle='--', alpha=0.5)
        plt.axhline(y=0.50, color='gray', linestyle=':', alpha=0.5)
        plt.axhline(y=0.382, color='gray', linestyle='--', alpha=0.5)
        
        plt.title(f'Efficiency Ratio Comparison (Period: {period})')
        plt.ylabel('Value')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    # 正規化されたAlpha ER（0-100スケール）
    alpha_er_normalized = AlphaER(period=21, normalize=True)
    alpha_er_normalized_values = alpha_er_normalized.calculate(df)
    df['alpha_er_normalized'] = alpha_er_normalized_values
    
    plt.subplot(total_plots, 1, total_plots)
    plt.plot(df.index, df['alpha_er_normalized'], label='Alpha ER (Normalized 0-100)', color='purple', linewidth=2)
    
    # フィボナッチレベルの水平線（正規化版）
    plt.axhline(y=61.8, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=50.0, color='gray', linestyle=':', alpha=0.5)
    plt.axhline(y=38.2, color='gray', linestyle='--', alpha=0.5)
    
    plt.title('Alpha ER (Normalized 0-100 Scale)')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('alpha_er_example.png')
    plt.show()
    
    # 統計情報の表示
    print("=== Efficiency Ratio Statistics ===")
    for period in periods:
        print(f"\nPeriod: {period}")
        print(f"Standard ER Mean: {df[f'er_{period}'].mean():.6f}")
        print(f"Cumulative ER Mean: {df[f'cumulative_er_{period}'].mean():.6f}")
        print(f"Alpha ER Mean: {df[f'alpha_er_{period}'].mean():.6f}")
        
        print(f"Standard ER Std: {df[f'er_{period}'].std():.6f}")
        print(f"Cumulative ER Std: {df[f'cumulative_er_{period}'].std():.6f}")
        print(f"Alpha ER Std: {df[f'alpha_er_{period}'].std():.6f}")
    
    # 相関関係の分析
    correlation_matrix = df[[f'er_{periods[0]}', f'cumulative_er_{periods[0]}', f'alpha_er_{periods[0]}']].corr()
    print("\n=== Correlation Matrix ===")
    print(correlation_matrix)
    
    # ラグ分析（Alpha ERは通常のERより何期間遅れているか）
    from scipy import signal
    
    for period in periods:
        corr = signal.correlate(df[f'er_{period}'].fillna(0), 
                                df[f'alpha_er_{period}'].fillna(0), 
                                mode='full')
        lags = signal.correlation_lags(len(df[f'er_{period}']), len(df[f'alpha_er_{period}']))
        lag = lags[np.argmax(corr)]
        
        print(f"\nLag Analysis (Period {period}):")
        if lag > 0:
            print(f"Standard ER leads Alpha ER by {lag} periods")
        elif lag < 0:
            print(f"Alpha ER leads Standard ER by {-lag} periods")
        else:
            print("No significant lead/lag relationship")
    
    # 利用例：トレンド/レンジ状態の検出
    trend_threshold = 0.618  # トレンド状態の閾値
    range_threshold = 0.382  # レンジ状態の閾値
    
    # 標準ER
    standard_trend = (df[f'er_{periods[0]}'] >= trend_threshold).sum()
    standard_range = (df[f'er_{periods[0]}'] <= range_threshold).sum()
    
    # Alpha ER
    alpha_trend = (df[f'alpha_er_{periods[0]}'] >= trend_threshold).sum()
    alpha_range = (df[f'alpha_er_{periods[0]}'] <= range_threshold).sum()
    
    print("\n=== Market State Detection ===")
    print(f"Standard ER - Trend Periods: {standard_trend} ({standard_trend/len(df)*100:.1f}%)")
    print(f"Standard ER - Range Periods: {standard_range} ({standard_range/len(df)*100:.1f}%)")
    print(f"Alpha ER - Trend Periods: {alpha_trend} ({alpha_trend/len(df)*100:.1f}%)")
    print(f"Alpha ER - Range Periods: {alpha_range} ({alpha_range/len(df)*100:.1f}%)")
    
    # アルファERと標準ERのトレンド検出の違い
    both_trend = ((df[f'er_{periods[0]}'] >= trend_threshold) & 
                  (df[f'alpha_er_{periods[0]}'] >= trend_threshold)).sum()
    er_only_trend = ((df[f'er_{periods[0]}'] >= trend_threshold) & 
                     (df[f'alpha_er_{periods[0]}'] < trend_threshold)).sum()
    alpha_only_trend = ((df[f'er_{periods[0]}'] < trend_threshold) & 
                        (df[f'alpha_er_{periods[0]}'] >= trend_threshold)).sum()
    
    print("\n=== Trend Detection Comparison ===")
    print(f"Both Detected Trend: {both_trend} periods")
    print(f"Standard ER Only Detected Trend: {er_only_trend} periods")
    print(f"Alpha ER Only Detected Trend: {alpha_only_trend} periods")
    
    # ノイズフィルタリングの効果分析
    # 標準ERと比較してアルファERがどれだけノイズを減らしているか
    noise_reduction = 1.0 - (df[f'alpha_er_{periods[0]}'].std() / df[f'er_{periods[0]}'].std())
    print(f"\nNoise Reduction by Alpha ER: {noise_reduction:.2%}")


if __name__ == "__main__":
    main() 