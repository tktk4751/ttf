#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from indicators import DubucHurstExponent, DubucHurstResult
import time
import yfinance as yf

def generate_test_data(n_points=1000, trend_strength=0.02, noise_level=0.01):
    """
    テスト用のデータを生成する
    
    Args:
        n_points: データポイント数
        trend_strength: トレンドの強さ
        noise_level: ノイズレベル
    
    Returns:
        pd.DataFrame: OHLC データ
    """
    # 基本的なトレンドを作成
    trend = np.cumsum(np.random.normal(trend_strength, noise_level, n_points))
    
    # OHLCデータを生成
    open_prices = trend + np.random.normal(0, noise_level * 0.5, n_points)
    close_prices = trend + np.random.normal(0, noise_level * 0.5, n_points)
    
    # 高値と安値
    high_noise = np.abs(np.random.normal(0, noise_level * 0.3, n_points))
    low_noise = np.abs(np.random.normal(0, noise_level * 0.3, n_points))
    
    high_prices = np.maximum(open_prices, close_prices) + high_noise
    low_prices = np.minimum(open_prices, close_prices) - low_noise
    
    return pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices
    })

def test_basic_functionality():
    """基本的な機能テスト"""
    print("=== Dubucハースト指数の基本機能テスト ===")
    
    # テストデータ生成
    data = generate_test_data(500, trend_strength=0.01, noise_level=0.005)
    
    # インジケーター初期化
    dubuc_hurst = DubucHurstExponent(
        length=100,
        samples=5,
        smooth_hurst=True,
        use_dynamic_window=False
    )
    
    # 計算
    start_time = time.time()
    hurst_values = dubuc_hurst.calculate(data)
    calc_time = time.time() - start_time
    
    print(f"計算時間: {calc_time:.4f}秒")
    print(f"データポイント数: {len(data)}")
    print(f"有効なハースト値数: {np.sum(~np.isnan(hurst_values))}")
    print(f"ハースト値の範囲: {np.nanmin(hurst_values):.4f} - {np.nanmax(hurst_values):.4f}")
    print(f"ハースト値の平均: {np.nanmean(hurst_values):.4f}")
    
    # 結果の詳細取得
    result = dubuc_hurst.get_result()
    if result:
        print(f"トレンド状態の平均: {np.nanmean(result.trend_state):.4f}")
        print(f"動的しきい値の平均: {np.nanmean(result.dynamic_threshold):.4f}")
    
    return data, hurst_values, dubuc_hurst

def test_dynamic_window():
    """動的ウィンドウ機能のテスト"""
    print("\n=== 動的ウィンドウ機能テスト ===")
    
    # テストデータ生成
    data = generate_test_data(500, trend_strength=0.015, noise_level=0.008)
    
    # 動的ウィンドウ有効
    dubuc_dynamic = DubucHurstExponent(
        length=100,
        samples=5,
        smooth_hurst=True,
        use_dynamic_window=True
    )
    
    # 固定ウィンドウ
    dubuc_fixed = DubucHurstExponent(
        length=100,
        samples=5,
        smooth_hurst=True,
        use_dynamic_window=False
    )
    
    # 計算
    start_time = time.time()
    hurst_dynamic = dubuc_dynamic.calculate(data)
    dynamic_time = time.time() - start_time
    
    start_time = time.time()
    hurst_fixed = dubuc_fixed.calculate(data)
    fixed_time = time.time() - start_time
    
    print(f"動的ウィンドウ計算時間: {dynamic_time:.4f}秒")
    print(f"固定ウィンドウ計算時間: {fixed_time:.4f}秒")
    
    # 統計比較
    valid_dynamic = ~np.isnan(hurst_dynamic)
    valid_fixed = ~np.isnan(hurst_fixed)
    
    if np.sum(valid_dynamic) > 0 and np.sum(valid_fixed) > 0:
        print(f"動的ハースト値の平均: {np.mean(hurst_dynamic[valid_dynamic]):.4f}")
        print(f"固定ハースト値の平均: {np.mean(hurst_fixed[valid_fixed]):.4f}")
        
        # 相関係数
        common_valid = valid_dynamic & valid_fixed
        if np.sum(common_valid) > 10:
            correlation = np.corrcoef(hurst_dynamic[common_valid], hurst_fixed[common_valid])[0, 1]
            print(f"動的vs固定ウィンドウの相関: {correlation:.4f}")
    
    return hurst_dynamic, hurst_fixed

def test_parameters():
    """異なるパラメータでのテスト"""
    print("\n=== パラメータ感度テスト ===")
    
    # テストデータ生成
    data = generate_test_data(300, trend_strength=0.02, noise_level=0.01)
    
    # 異なるサンプル数でテスト
    sample_counts = [3, 5, 8]
    results = {}
    
    for samples in sample_counts:
        dubuc = DubucHurstExponent(
            length=50,
            samples=samples,
            smooth_hurst=False,
            use_dynamic_window=False
        )
        
        start_time = time.time()
        hurst_values = dubuc.calculate(data)
        calc_time = time.time() - start_time
        
        valid_values = hurst_values[~np.isnan(hurst_values)]
        
        results[samples] = {
            'values': hurst_values,
            'time': calc_time,
            'mean': np.mean(valid_values) if len(valid_values) > 0 else np.nan,
            'std': np.std(valid_values) if len(valid_values) > 0 else np.nan
        }
        
        print(f"サンプル数 {samples}: 平均={results[samples]['mean']:.4f}, "
              f"標準偏差={results[samples]['std']:.4f}, 時間={calc_time:.4f}秒")
    
    return results

def test_with_real_data():
    """実データでのテスト"""
    print("\n=== 実データテスト ===")
    
    try:
        # Yahoo Financeからデータ取得
        ticker = "SPY"
        data = yf.download(ticker, start="2023-01-01", end="2024-01-01", progress=False)
        
        if data.empty:
            print("実データの取得に失敗しました")
            return None, None
        
        # カラム名を小文字に変換
        data.columns = [col.lower() for col in data.columns]
        
        print(f"実データサイズ: {len(data)} ポイント")
        
        # ハースト指数計算
        dubuc_hurst = DubucHurstExponent(
            length=50,
            samples=5,
            smooth_hurst=True,
            use_dynamic_window=True
        )
        
        start_time = time.time()
        hurst_values = dubuc_hurst.calculate(data)
        calc_time = time.time() - start_time
        
        print(f"実データ計算時間: {calc_time:.4f}秒")
        
        # 統計
        valid_values = hurst_values[~np.isnan(hurst_values)]
        if len(valid_values) > 0:
            print(f"ハースト値の統計:")
            print(f"  平均: {np.mean(valid_values):.4f}")
            print(f"  標準偏差: {np.std(valid_values):.4f}")
            print(f"  最小値: {np.min(valid_values):.4f}")
            print(f"  最大値: {np.max(valid_values):.4f}")
            
            # トレンド/レンジの判定
            result = dubuc_hurst.get_result()
            if result is not None:
                trend_ratio = np.nanmean(result.trend_state)
                print(f"  トレンド比率: {trend_ratio:.2%}")
        
        return data, hurst_values
        
    except Exception as e:
        print(f"実データテストでエラー: {e}")
        return None, None

def visualize_results(data, hurst_values, dubuc_hurst):
    """結果の可視化"""
    print("\n=== 結果の可視化 ===")
    
    try:
        # 結果を取得
        result = dubuc_hurst.get_result()
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # 価格チャート
        axes[0].plot(data['close'].values, label='Close Price', linewidth=1)
        axes[0].set_title('価格チャート')
        axes[0].set_ylabel('価格')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # ハースト指数
        valid_indices = ~np.isnan(hurst_values)
        axes[1].plot(np.where(valid_indices)[0], hurst_values[valid_indices], 
                    label='Dubuc Hurst Exponent', linewidth=1, color='blue')
        axes[1].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='H=0.5 (Random Walk)')
        axes[1].axhline(y=0.6, color='green', linestyle='--', alpha=0.7, label='H=0.6 (Trend)')
        axes[1].axhline(y=0.4, color='orange', linestyle='--', alpha=0.7, label='H=0.4 (Mean Reversion)')
        axes[1].set_title('Dubucハースト指数')
        axes[1].set_ylabel('ハースト指数')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1)
        
        # トレンド状態
        if result is not None:
            trend_valid = ~np.isnan(result.trend_state)
            axes[2].plot(np.where(trend_valid)[0], result.trend_state[trend_valid], 
                        label='Trend State', linewidth=1, color='purple')
            axes[2].set_title('トレンド状態 (1=トレンド, 0=レンジ)')
            axes[2].set_ylabel('トレンド状態')
            axes[2].set_xlabel('時間')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            axes[2].set_ylim(-0.1, 1.1)
        
        plt.tight_layout()
        plt.savefig('dubuc_hurst_test_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("結果がdubuc_hurst_test_results.pngに保存されました")
        
    except Exception as e:
        print(f"可視化エラー: {e}")

def performance_benchmark():
    """パフォーマンスベンチマーク"""
    print("\n=== パフォーマンスベンチマーク ===")
    
    data_sizes = [100, 500, 1000, 2000]
    sample_counts = [3, 5, 8]
    
    print("データサイズ, サンプル数, 計算時間(秒), 1ポイントあたり(ms)")
    print("-" * 60)
    
    for size in data_sizes:
        for samples in sample_counts:
            # データ生成
            data = generate_test_data(size)
            
            # インジケーター初期化
            dubuc = DubucHurstExponent(
                length=min(50, size//4),
                samples=samples,
                smooth_hurst=False,
                use_dynamic_window=False
            )
            
            # 計算時間測定
            start_time = time.time()
            hurst_values = dubuc.calculate(data)
            calc_time = time.time() - start_time
            
            per_point_ms = (calc_time / size) * 1000
            
            print(f"{size:8d}, {samples:8d}, {calc_time:12.4f}, {per_point_ms:14.4f}")

def main():
    """メイン関数"""
    print("Dubucハースト指数インジケーターのテスト開始")
    print("=" * 50)
    
    # 基本機能テスト
    data, hurst_values, dubuc_hurst = test_basic_functionality()
    
    # 動的ウィンドウテスト
    hurst_dynamic, hurst_fixed = test_dynamic_window()
    
    # パラメータテスト
    param_results = test_parameters()
    
    # 実データテスト
    real_data, real_hurst = test_with_real_data()
    
    # 可視化
    if data is not None and hurst_values is not None:
        visualize_results(data, hurst_values, dubuc_hurst)
    
    # パフォーマンステスト
    performance_benchmark()
    
    print("\n" + "=" * 50)
    print("すべてのテストが完了しました")

if __name__ == "__main__":
    main() 