#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import os

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


def generate_test_data(n_samples=1000, noise_level=0.2):
    """合成テストデータを生成する"""
    # 時間軸
    t = np.linspace(0, 10, n_samples)
    
    # 基本の正弦波（周期が徐々に変化する）
    period1 = 20 + 10 * np.sin(t/5)  # 周期20〜30の範囲で変化
    sin_wave1 = np.sin(2 * np.pi * t / period1)
    
    # 第2の正弦波（短い周期）
    period2 = 10 * np.ones_like(t)  # 固定周期10
    sin_wave2 = 0.5 * np.sin(2 * np.pi * t / period2)
    
    # ノイズ
    noise = noise_level * np.random.randn(n_samples)
    
    # トレンド成分
    trend = 0.1 * t
    
    # 合成波形
    composite = sin_wave1 + sin_wave2 + noise + trend
    
    # DataFrameに変換
    df = pd.DataFrame({
        'time': t,
        'close': composite,
        'period1': period1,
        'period2': period2,
    })
    
    return df


def main():
    """メイン関数"""
    # テストデータの生成
    print("テストデータを生成中...")
    df = generate_test_data(n_samples=1000, noise_level=0.2)
    
    # 各アルゴリズムのインスタンス化
    print("ドミナントサイクル検出アルゴリズムを初期化中...")
    cycle_detectors = {
        "ホモダイン判別器 (HoDy)": EhlersHoDyDC(cycle_part=0.5),
        "位相累積法 (PhAc)": EhlersPhAcDC(cycle_part=0.5),
        "二重微分法 (DuDi)": EhlersDuDiDC(cycle_part=0.5),
        "拡張ホモダイン判別器 (HoDy-E)": EhlersHoDyDCE(lp_period=10, hp_period=48, cycle_part=0.5),
        "拡張位相累積法 (PhAc-E)": EhlersPhAcDCE(lp_period=10, hp_period=48, cycle_part=0.5),
        "拡張二重微分法 (DuDi-E)": EhlersDuDiDCE(lp_period=10, hp_period=48, cycle_part=0.5),
        "離散フーリエ変換 (DFT)": EhlersDFTDC(window=50, cycle_part=0.5)
    }
    
    # 各アルゴリズムでサイクル検出を実行
    print("各アルゴリズムでドミナントサイクルを検出中...")
    results = {}
    for name, detector in cycle_detectors.items():
        print(f"実行中: {name}")
        dom_cycle = detector.calculate(df)
        results[name] = dom_cycle
    
    # プロット
    print("結果をプロット中...")
    plt.figure(figsize=(15, 12))
    gs = GridSpec(4, 2, height_ratios=[2, 1, 1, 1])
    
    # 価格データとサイクル期間のプロット
    ax1 = plt.subplot(gs[0, :])
    ax1.plot(df['time'], df['close'], 'b-', label='合成価格データ')
    ax1.set_title('合成価格データ（正弦波 + ノイズ + トレンド）')
    ax1.set_ylabel('価格')
    ax1.grid(True)
    ax1.legend()
    
    # 実際の周期のプロット
    ax2 = plt.subplot(gs[1, :], sharex=ax1)
    ax2.plot(df['time'], df['period1'], 'r-', label='実際の主要周期')
    ax2.plot(df['time'], df['period2'], 'g-', label='実際の副次周期')
    ax2.set_title('実際の周期')
    ax2.set_ylabel('周期')
    ax2.grid(True)
    ax2.legend()
    
    # 基本アルゴリズムのプロット
    ax3 = plt.subplot(gs[2, :], sharex=ax1)
    ax3.plot(df['time'], results["ホモダイン判別器 (HoDy)"], 'r-', label='HoDy')
    ax3.plot(df['time'], results["位相累積法 (PhAc)"], 'g-', label='PhAc')
    ax3.plot(df['time'], results["二重微分法 (DuDi)"], 'b-', label='DuDi')
    ax3.set_title('基本アルゴリズムの結果')
    ax3.set_ylabel('検出周期')
    ax3.grid(True)
    ax3.legend()
    
    # 拡張アルゴリズムとDFTのプロット
    ax4 = plt.subplot(gs[3, :], sharex=ax1)
    ax4.plot(df['time'], results["拡張ホモダイン判別器 (HoDy-E)"], 'r-', label='HoDy-E')
    ax4.plot(df['time'], results["拡張位相累積法 (PhAc-E)"], 'g-', label='PhAc-E')
    ax4.plot(df['time'], results["拡張二重微分法 (DuDi-E)"], 'b-', label='DuDi-E')
    ax4.plot(df['time'], results["離散フーリエ変換 (DFT)"], 'k-', label='DFT')
    ax4.set_title('拡張アルゴリズムとDFTの結果')
    ax4.set_xlabel('時間')
    ax4.set_ylabel('検出周期')
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('ehlers_cycle_detection_results.png')
    plt.show()
    
    print("完了！結果が 'ehlers_cycle_detection_results.png' に保存されました")


if __name__ == "__main__":
    main() 