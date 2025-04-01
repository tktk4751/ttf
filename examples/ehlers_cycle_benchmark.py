#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys
import os
from tabulate import tabulate
from tqdm import tqdm

# 親ディレクトリをパスに追加してインポートできるようにする
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicators import (
    EhlersHoDyDC,
    EhlersPhAcDC,
    EhlersDuDiDC,
    EhlersHoDyDCE,
    EhlersPhAcDCE,
    EhlersDuDiDCE,
    EhlersDFTDC
)


def generate_market_data(market_type, n_samples=1000, noise_level=0.2, seed=42):
    """
    様々な市場タイプのテストデータを生成する
    
    Parameters:
    -----------
    market_type : str
        'trending' - トレンド市場
        'sideways' - レンジ相場
        'volatile' - ボラティリティの高い市場
        'cyclical' - 周期的な市場
        'mixed' - 混合市場
    n_samples : int
        生成するサンプル数
    noise_level : float
        ノイズレベル
    seed : int
        乱数シード
    
    Returns:
    --------
    df : pd.DataFrame
        生成されたマーケットデータと実際の周期情報を含むDataFrame
    """
    np.random.seed(seed)
    
    # 時間軸
    t = np.linspace(0, 10, n_samples)
    
    # 初期化
    price = np.zeros(n_samples)
    actual_cycle = np.zeros(n_samples)
    
    if market_type == 'trending':
        # トレンド市場：上昇トレンドと下降トレンドを含む
        trend = np.cumsum(0.01 * np.random.randn(n_samples))
        cycles = 15 + 5 * np.sin(2 * np.pi * t / 200)  # 周期15-20
        cyclical_component = np.sin(2 * np.pi * t / cycles)
        price = trend + 0.2 * cyclical_component
        actual_cycle = cycles
        
    elif market_type == 'sideways':
        # レンジ相場：水平なトレンドと小さな振動
        baseline = 100
        cycles = 10 * np.ones_like(t)  # 固定周期10
        price = baseline + np.sin(2 * np.pi * t / cycles)
        actual_cycle = cycles
        
    elif market_type == 'volatile':
        # ボラティリティの高い市場：急速な変化とスパイク
        baseline = 100
        volatility = 1.0 + 0.5 * np.sin(2 * np.pi * t / 50)
        cycles = 8 + 4 * np.random.randn(n_samples)  # 非常に可変的な周期
        cycles = np.clip(cycles, 4, 20)  # 4〜20の範囲に制限
        
        # 前の値を使って滑らかにする
        smoothed_cycles = np.zeros_like(cycles)
        smoothed_cycles[0] = cycles[0]
        for i in range(1, n_samples):
            smoothed_cycles[i] = 0.9 * smoothed_cycles[i-1] + 0.1 * cycles[i]
        
        price_changes = volatility * np.random.randn(n_samples)
        price = baseline + np.cumsum(price_changes) / 10
        actual_cycle = smoothed_cycles
        
    elif market_type == 'cyclical':
        # 周期的な市場：明確な周期パターン
        cycles1 = 25 * np.ones(n_samples // 2)
        cycles2 = 10 * np.ones(n_samples - n_samples // 2)
        cycles = np.concatenate([cycles1, cycles2])
        
        phase = np.zeros(n_samples)
        for i in range(1, n_samples):
            phase[i] = phase[i-1] + 2 * np.pi / cycles[i]
        
        price = 100 + 10 * np.sin(phase)
        actual_cycle = cycles
        
    elif market_type == 'mixed':
        # 混合市場：様々なパターンを組み合わせる
        # トレンド成分
        trend = np.cumsum(0.005 * np.random.randn(n_samples))
        
        # 周期的成分（変化する周期）
        cycles = np.zeros(n_samples)
        cycles[:n_samples//3] = 20  # 最初の1/3: 周期20
        cycles[n_samples//3:2*n_samples//3] = 10  # 中間の1/3: 周期10
        cycles[2*n_samples//3:] = 30  # 最後の1/3: 周期30
        
        # サイン波の位相を計算
        phase = np.zeros(n_samples)
        for i in range(1, n_samples):
            phase[i] = phase[i-1] + 2 * np.pi / cycles[i]
        
        cyclical_component = np.sin(phase)
        
        # ボラティリティスパイク
        volatility = np.ones(n_samples)
        spike_indices = np.random.choice(n_samples, size=5, replace=False)
        for idx in spike_indices:
            volatility[idx:idx+20] = 3.0
        
        noise = noise_level * volatility * np.random.randn(n_samples)
        
        price = 100 + trend + 5 * cyclical_component + noise
        actual_cycle = cycles
    
    else:
        raise ValueError(f"不明な市場タイプ: {market_type}")
    
    # ノイズの追加
    price += noise_level * np.random.randn(n_samples)
    
    # DataFrameに変換
    df = pd.DataFrame({
        'time': t,
        'close': price,
        'actual_cycle': actual_cycle
    })
    
    return df


def evaluate_algorithm(detector, data, actual_cycle, name):
    """
    アルゴリズムの性能を評価する
    
    Parameters:
    -----------
    detector : object
        サイクル検出アルゴリズムのインスタンス
    data : pd.DataFrame
        価格データ
    actual_cycle : np.array
        実際の周期データ
    name : str
        アルゴリズムの名前
    
    Returns:
    --------
    dict
        性能評価指標を含む辞書
    """
    # 計算時間の測定
    start_time = time.time()
    detected_cycle = detector.calculate(data)
    end_time = time.time()
    
    # 最初の100サンプルはウォームアップ期間として除外
    warmup = 100
    if len(actual_cycle) > warmup:
        actual = actual_cycle[warmup:]
        detected = detected_cycle[warmup:]
    else:
        actual = actual_cycle
        detected = detected_cycle
    
    # 評価指標の計算
    mae = np.mean(np.abs(actual - detected))
    rmse = np.sqrt(np.mean((actual - detected) ** 2))
    
    # ノイズレベルの計算 (サイクル検出の滑らかさ)
    cycle_diff = np.diff(detected)
    noise_level = np.std(cycle_diff)
    
    return {
        'name': name,
        'mae': mae,
        'rmse': rmse,
        'noise': noise_level,
        'execution_time': end_time - start_time
    }


def run_benchmark(market_types=None, n_samples=1000, noise_levels=None):
    """
    様々な市場条件でベンチマークを実行
    
    Parameters:
    -----------
    market_types : list
        テストする市場タイプのリスト
    n_samples : int
        各テストで使用するサンプル数
    noise_levels : list
        テストするノイズレベルのリスト
    
    Returns:
    --------
    results : dict
        評価結果を含む辞書
    """
    if market_types is None:
        market_types = ['trending', 'sideways', 'volatile', 'cyclical', 'mixed']
    
    if noise_levels is None:
        noise_levels = [0.1, 0.3, 0.5]
    
    # アルゴリズムの初期化
    detectors = {
        "HoDy": EhlersHoDyDC(cycle_part=0.5),
        "PhAc": EhlersPhAcDC(cycle_part=0.5),
        "DuDi": EhlersDuDiDC(cycle_part=0.5),
        "HoDy-E": EhlersHoDyDCE(lp_period=10, hp_period=48, cycle_part=0.5),
        "PhAc-E": EhlersPhAcDCE(lp_period=10, hp_period=48, cycle_part=0.5),
        "DuDi-E": EhlersDuDiDCE(lp_period=10, hp_period=48, cycle_part=0.5),
        "DFT": EhlersDFTDC(window=50, cycle_part=0.5)
    }
    
    results = {}
    
    # 各市場タイプとノイズレベルの組み合わせでテスト
    for market_type in market_types:
        results[market_type] = {}
        
        for noise_level in tqdm(noise_levels, desc=f"市場タイプ: {market_type}", leave=False):
            # テストデータの生成
            df = generate_market_data(market_type, n_samples=n_samples, 
                                     noise_level=noise_level, seed=42)
            
            results[market_type][noise_level] = []
            
            # 各アルゴリズムの評価
            for name, detector in detectors.items():
                result = evaluate_algorithm(
                    detector, df, df['actual_cycle'].values, name
                )
                results[market_type][noise_level].append(result)
    
    return results


def print_results_table(results):
    """
    結果を表形式で表示
    
    Parameters:
    -----------
    results : dict
        ベンチマーク結果
    """
    for market_type, noise_levels in results.items():
        print(f"\n=== 市場タイプ: {market_type} ===")
        
        for noise_level, algos in noise_levels.items():
            print(f"\n-- ノイズレベル: {noise_level} --")
            
            table_data = []
            for result in algos:
                table_data.append([
                    result['name'],
                    f"{result['mae']:.2f}",
                    f"{result['rmse']:.2f}",
                    f"{result['noise']:.4f}",
                    f"{result['execution_time']*1000:.1f}"
                ])
            
            headers = ["アルゴリズム", "MAE", "RMSE", "ノイズ", "実行時間 (ms)"]
            print(tabulate(table_data, headers=headers, tablefmt="grid"))


def plot_benchmark_results(results):
    """
    ベンチマーク結果をグラフで表示
    
    Parameters:
    -----------
    results : dict
        ベンチマーク結果
    """
    market_types = list(results.keys())
    noise_levels = list(results[market_types[0]].keys())
    
    # 評価指標ごとのプロット
    metrics = ['mae', 'rmse', 'noise', 'execution_time']
    metric_names = ['平均絶対誤差 (MAE)', '二乗平均平方根誤差 (RMSE)', 
                   'ノイズレベル', '実行時間 (秒)']
    
    fig, axes = plt.subplots(len(metrics), len(market_types), 
                             figsize=(5*len(market_types), 4*len(metrics)))
    
    for m_idx, metric in enumerate(metrics):
        for mt_idx, market_type in enumerate(market_types):
            ax = axes[m_idx, mt_idx]
            
            # データの抽出
            for noise_level in noise_levels:
                x_data = []
                y_data = []
                labels = []
                
                for result in results[market_type][noise_level]:
                    x_data.append(noise_level)
                    y_data.append(result[metric])
                    labels.append(result['name'])
                
                # 散布図プロット
                for i, (x, y, label) in enumerate(zip(x_data, y_data, labels)):
                    ax.scatter(x, y, label=label if mt_idx == 0 and noise_level == noise_levels[0] else "")
            
            # 各アルゴリズムについて、ノイズレベルごとの線を描画
            for algo_idx in range(len(results[market_type][noise_levels[0]])):
                algo_name = results[market_type][noise_levels[0]][algo_idx]['name']
                x_values = noise_levels
                y_values = [results[market_type][nl][algo_idx][metric] for nl in noise_levels]
                ax.plot(x_values, y_values, 'o-', label='' if mt_idx == 0 else None)
            
            # グラフのタイトルと軸ラベル
            if m_idx == 0:
                ax.set_title(f'市場タイプ: {market_type}')
            
            if mt_idx == 0:
                ax.set_ylabel(metric_names[m_idx])
            
            if m_idx == len(metrics) - 1:
                ax.set_xlabel('ノイズレベル')
            
            ax.grid(True)
    
    # 凡例の追加（最初の列だけ）
    for m_idx in range(len(metrics)):
        handles, labels = axes[m_idx, 0].get_legend_handles_labels()
        axes[m_idx, 0].legend(handles, labels, loc='upper left', bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()
    plt.savefig('ehlers_cycle_benchmark_results.png')
    plt.show()


def main():
    print("エーラーズのドミナントサイクル検出アルゴリズムのベンチマークを開始します...")
    
    # 市場タイプとノイズレベルの設定
    market_types = ['trending', 'sideways', 'volatile', 'cyclical', 'mixed']
    noise_levels = [0.1, 0.2, 0.3]
    
    # ベンチマークの実行
    print("各市場タイプとノイズレベルでアルゴリズムを評価中...")
    results = run_benchmark(market_types, n_samples=1000, noise_levels=noise_levels)
    
    # 結果の表示
    print("\nベンチマーク結果:")
    print_results_table(results)
    
    # 結果のグラフ表示
    print("\n結果をグラフで表示中...")
    plot_benchmark_results(results)
    
    print("\n完了！結果が 'ehlers_cycle_benchmark_results.png' に保存されました")


if __name__ == "__main__":
    main() 