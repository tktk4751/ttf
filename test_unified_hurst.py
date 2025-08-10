#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
統合ハースト指数のテストとデモ

Numbaエラーを回避してハースト指数の3つの手法を比較します。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# データ取得
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource

# ハースト指数（Numbaなし版）
from indicators.unified_hurst_exponent import UnifiedHurstExponent

def load_market_data():
    """実際の相場データを読み込む"""
    # Binanceデータソースを設定
    binance_data_source = BinanceDataSource('data/binance')
    dummy_csv_source = CSVDataSource("dummy")
    data_loader = DataLoader(
        data_source=dummy_csv_source,
        binance_data_source=binance_data_source
    )
    data_processor = DataProcessor()
    
    # 設定ファイルから読み込み
    import yaml
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("データを読み込み中...")
    raw_data = data_loader.load_data_from_config(config)
    processed_data = {
        symbol: data_processor.process(df)
        for symbol, df in raw_data.items()
    }
    
    # 最初のシンボルのデータを取得
    first_symbol = next(iter(processed_data))
    data = processed_data[first_symbol]
    
    print(f"データ読み込み完了: {first_symbol}")
    print(f"期間: {data.index.min()} → {data.index.max()}")
    print(f"データ数: {len(data)}")
    
    return data

def create_simple_hurst_calculator():
    """簡単なハースト指数計算関数（Numbaなし）"""
    
    def calculate_rs_hurst_simple(prices, window_size=50):
        """R/S法による簡単なハースト指数計算"""
        n = len(prices)
        hurst_values = np.full(n, np.nan)
        
        for i in range(window_size, n):
            window_data = prices[i-window_size+1:i+1]
            
            # 対数リターン
            log_returns = np.diff(np.log(window_data + 1e-10))
            
            if len(log_returns) < 20:
                continue
                
            # 平均リターン
            mean_return = np.mean(log_returns)
            
            # 複数の期間でR/S統計を計算
            periods = [8, 12, 16, 20]
            log_rs_values = []
            log_periods = []
            
            for period in periods:
                if period >= len(log_returns):
                    continue
                    
                n_segments = len(log_returns) // period
                rs_values = []
                
                for j in range(n_segments):
                    segment = log_returns[j*period:(j+1)*period]
                    
                    # 累積偏差
                    cumdev = np.cumsum(segment - mean_return)
                    
                    # レンジ
                    R = np.max(cumdev) - np.min(cumdev)
                    
                    # 標準偏差
                    S = np.std(segment)
                    
                    if S > 1e-10:
                        rs_values.append(R / S)
                
                if len(rs_values) > 0:
                    avg_rs = np.mean(rs_values)
                    if avg_rs > 0:
                        log_rs_values.append(np.log(avg_rs))
                        log_periods.append(np.log(period))
            
            # 線形回帰でハースト指数を計算
            if len(log_rs_values) >= 3:
                coeffs = np.polyfit(log_periods, log_rs_values, 1)
                hurst_values[i] = coeffs[0]  # 傾き
        
        return hurst_values
    
    def calculate_dfa_hurst_simple(prices, window_size=50):
        """DFA法による簡単なハースト指数計算"""
        n = len(prices)
        hurst_values = np.full(n, np.nan)
        
        for i in range(window_size, n):
            window_data = prices[i-window_size+1:i+1]
            
            # 対数リターン
            log_returns = np.diff(np.log(window_data + 1e-10))
            
            if len(log_returns) < 30:
                continue
            
            # プロファイル（累積偏差）
            mean_return = np.mean(log_returns)
            profile = np.cumsum(log_returns - mean_return)
            profile = np.insert(profile, 0, 0)  # 先頭に0を追加
            
            # 複数のスケールで揺動を計算
            scales = [8, 12, 16, 20, 24]
            log_fluctuations = []
            log_scales = []
            
            for scale in scales:
                if scale >= len(profile) - 1:
                    continue
                
                n_segments = len(profile) // scale
                segment_vars = []
                
                for j in range(n_segments):
                    segment = profile[j*scale:(j+1)*scale]
                    
                    # 線形トレンド除去
                    x = np.arange(len(segment))
                    coeffs = np.polyfit(x, segment, 1)
                    trend = np.polyval(coeffs, x)
                    detrended = segment - trend
                    
                    segment_vars.append(np.var(detrended))
                
                if len(segment_vars) > 0:
                    avg_fluctuation = np.sqrt(np.mean(segment_vars))
                    if avg_fluctuation > 0:
                        log_fluctuations.append(np.log(avg_fluctuation))
                        log_scales.append(np.log(scale))
            
            # 線形回帰でハースト指数を計算
            if len(log_fluctuations) >= 3:
                coeffs = np.polyfit(log_scales, log_fluctuations, 1)
                hurst_values[i] = coeffs[0]  # 傾き
        
        return hurst_values
    
    return calculate_rs_hurst_simple, calculate_dfa_hurst_simple

def test_hurst_methods():
    """ハースト指数手法をテストして比較"""
    print("=== 統合ハースト指数手法比較テスト ===")
    
    # 実際の相場データを読み込み
    try:
        data = load_market_data()
    except Exception as e:
        print(f"データ読み込みエラー: {e}")
        return
    
    # 最新の2000データポイントを使用（計算速度のため）
    if len(data) > 2000:
        data = data.tail(2000)
    
    # 価格データを取得
    prices = data['close'].values
    dates = data.index
    
    print(f"\n分析対象データ:")
    print(f"期間: {dates.min()} → {dates.max()}")
    print(f"データ数: {len(prices)}")
    print(f"価格範囲: {prices.min():.2f} - {prices.max():.2f}")
    
    # 簡単な計算関数を取得
    calc_rs, calc_dfa = create_simple_hurst_calculator()
    
    # 各手法でハースト指数を計算
    print("\nハースト指数を計算中...")
    
    window_size = 80
    
    hurst_rs = calc_rs(prices, window_size)
    print("R/S法計算完了")
    
    hurst_dfa = calc_dfa(prices, window_size)
    print("DFA法計算完了")
    
    # 結果を可視化
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    fig.suptitle('ハースト指数手法比較分析', fontsize=16, fontweight='bold')
    
    # 1. 価格チャート
    axes[0].plot(dates, prices, 'b-', linewidth=1.5, label='Close Price')
    axes[0].set_title('価格チャート')
    axes[0].set_ylabel('価格')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. R/S法ハースト指数
    valid_rs = ~np.isnan(hurst_rs)
    if np.any(valid_rs):
        axes[1].plot(dates[valid_rs], hurst_rs[valid_rs], 'g-', linewidth=2, label='R/S Method')
        rs_mean = np.nanmean(hurst_rs)
        axes[1].axhline(y=rs_mean, color='g', linestyle='--', alpha=0.7, label=f'平均: {rs_mean:.3f}')
    
    axes[1].axhline(y=0.5, color='black', linestyle='-', alpha=0.8, linewidth=2, label='Random Walk (H=0.5)')
    axes[1].axhline(y=0.45, color='red', linestyle='--', alpha=0.6, label='反持続性')
    axes[1].axhline(y=0.55, color='blue', linestyle='--', alpha=0.6, label='持続性')
    axes[1].set_title('R/S法によるハースト指数')
    axes[1].set_ylabel('Hurst Exponent')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0.2, 0.8)
    
    # 3. DFA法ハースト指数
    valid_dfa = ~np.isnan(hurst_dfa)
    if np.any(valid_dfa):
        axes[2].plot(dates[valid_dfa], hurst_dfa[valid_dfa], 'r-', linewidth=2, label='DFA Method')
        dfa_mean = np.nanmean(hurst_dfa)
        axes[2].axhline(y=dfa_mean, color='r', linestyle='--', alpha=0.7, label=f'平均: {dfa_mean:.3f}')
    
    axes[2].axhline(y=0.5, color='black', linestyle='-', alpha=0.8, linewidth=2, label='Random Walk (H=0.5)')
    axes[2].axhline(y=0.45, color='red', linestyle='--', alpha=0.6, label='反持続性')
    axes[2].axhline(y=0.55, color='blue', linestyle='--', alpha=0.6, label='持続性')
    axes[2].set_title('DFA法によるハースト指数')
    axes[2].set_ylabel('Hurst Exponent')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0.2, 0.8)
    
    # 4. 手法比較
    if np.any(valid_rs) and np.any(valid_dfa):
        # 共通の有効期間を抽出
        common_valid = valid_rs & valid_dfa
        if np.any(common_valid):
            axes[3].plot(dates[common_valid], hurst_rs[common_valid], 'g-', linewidth=1.5, label='R/S Method', alpha=0.8)
            axes[3].plot(dates[common_valid], hurst_dfa[common_valid], 'r-', linewidth=1.5, label='DFA Method', alpha=0.8)
    
    axes[3].axhline(y=0.5, color='black', linestyle='-', alpha=0.8, linewidth=2, label='Random Walk (H=0.5)')
    axes[3].axhline(y=0.45, color='red', linestyle='--', alpha=0.6)
    axes[3].axhline(y=0.55, color='blue', linestyle='--', alpha=0.6)
    axes[3].set_title('手法比較')
    axes[3].set_ylabel('Hurst Exponent')
    axes[3].set_xlabel('日付')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    axes[3].set_ylim(0.2, 0.8)
    
    plt.tight_layout()
    plt.savefig('unified_hurst_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nチャートを保存しました: unified_hurst_comparison.png")
    
    # 統計分析
    print(f"\n=== 統計分析結果 ===")
    
    valid_rs_values = hurst_rs[valid_rs] if np.any(valid_rs) else np.array([])
    valid_dfa_values = hurst_dfa[valid_dfa] if np.any(valid_dfa) else np.array([])
    
    print(f"\nR/S法:")
    if len(valid_rs_values) > 0:
        print(f"  有効値数: {len(valid_rs_values)}")
        print(f"  平均: {np.mean(valid_rs_values):.4f}")
        print(f"  標準偏差: {np.std(valid_rs_values):.4f}")
        print(f"  範囲: {np.min(valid_rs_values):.4f} - {np.max(valid_rs_values):.4f}")
        print(f"  0.5からの平均偏差: {np.mean(np.abs(valid_rs_values - 0.5)):.4f}")
    else:
        print("  有効なデータなし")
    
    print(f"\nDFA法:")
    if len(valid_dfa_values) > 0:
        print(f"  有効値数: {len(valid_dfa_values)}")
        print(f"  平均: {np.mean(valid_dfa_values):.4f}")
        print(f"  標準偏差: {np.std(valid_dfa_values):.4f}")
        print(f"  範囲: {np.min(valid_dfa_values):.4f} - {np.max(valid_dfa_values):.4f}")
        print(f"  0.5からの平均偏差: {np.mean(np.abs(valid_dfa_values - 0.5)):.4f}")
    else:
        print("  有効なデータなし")
    
    # 相関分析
    if len(valid_rs_values) > 10 and len(valid_dfa_values) > 10:
        common_indices = valid_rs & valid_dfa
        if np.any(common_indices):
            rs_common = hurst_rs[common_indices]
            dfa_common = hurst_dfa[common_indices]
            
            if len(rs_common) > 1:
                correlation = np.corrcoef(rs_common, dfa_common)[0, 1]
                print(f"\n手法間相関:")
                print(f"  R/S vs DFA: {correlation:.4f}")
    
    # 最適手法の評価
    print(f"\n=== 手法評価 ===")
    
    methods_scores = []
    
    if len(valid_rs_values) > 0:
        # 安定性（標準偏差の逆数）と識別能力（0.5からの偏差）でスコア計算
        stability_rs = 1 / (np.std(valid_rs_values) + 0.001)
        discrimination_rs = np.mean(np.abs(valid_rs_values - 0.5))
        score_rs = stability_rs * 0.4 + discrimination_rs * 0.6
        methods_scores.append(("R/S法", score_rs, stability_rs, discrimination_rs))
    
    if len(valid_dfa_values) > 0:
        stability_dfa = 1 / (np.std(valid_dfa_values) + 0.001)
        discrimination_dfa = np.mean(np.abs(valid_dfa_values - 0.5))
        score_dfa = stability_dfa * 0.4 + discrimination_dfa * 0.6
        methods_scores.append(("DFA法", score_dfa, stability_dfa, discrimination_dfa))
    
    if methods_scores:
        methods_scores.sort(key=lambda x: x[1], reverse=True)
        
        print(f"評価結果（安定性40% + 識別能力60%）:")
        for i, (method, score, stability, discrimination) in enumerate(methods_scores, 1):
            print(f"  {i}位: {method}")
            print(f"    総合スコア: {score:.4f}")
            print(f"    安定性: {stability:.4f}")
            print(f"    識別能力: {discrimination:.4f}")
        
        best_method = methods_scores[0][0]
        print(f"\n推奨手法: {best_method}")
        
        if best_method == "DFA法":
            print("DFA法は金融時系列のトレンド分析に適しており、より安定した結果を提供します。")
        elif best_method == "R/S法":
            print("R/S法は古典的で解釈しやすく、長期記憶特性の検出に優れています。")
    
    plt.show()
    
    print("\n=== 分析完了 ===")

if __name__ == "__main__":
    test_hurst_methods()