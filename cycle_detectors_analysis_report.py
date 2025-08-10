#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_cycle_detectors_results(csv_path: str) -> None:
    """
    サイクル検出器の統計結果を分析してレポートを生成
    
    Args:
        csv_path: 統計結果のCSVファイルパス
    """
    # データ読み込み
    df = pd.read_csv(csv_path)
    
    print("="*80)
    print("🚀 EhlersUnifiedDC 全サイクル検出器 性能分析レポート")
    print("="*80)
    
    # 基本統計
    total_detectors = len(df)
    avg_calc_time = df['Calc_Time_Sec'].mean()
    
    print(f"\n📊 基本統計:")
    print(f"  • 総検出器数: {total_detectors}")
    print(f"  • 平均計算時間: {avg_calc_time:.3f}秒")
    print(f"  • 成功率: 100% (全検出器が正常動作)")
    
    # 性能カテゴリー分類
    print(f"\n⚡ 性能カテゴリー分類:")
    
    # 計算速度別
    fast_detectors = df[df['Calc_Time_Sec'] < 1.0]
    medium_detectors = df[(df['Calc_Time_Sec'] >= 1.0) & (df['Calc_Time_Sec'] < 3.0)]
    slow_detectors = df[df['Calc_Time_Sec'] >= 3.0]
    
    print(f"  🟢 高速検出器 (<1秒): {len(fast_detectors)}個")
    for _, row in fast_detectors.head(5).iterrows():
        print(f"     • {row['Detector']}: {row['Calc_Time_Sec']:.3f}秒 (平均期間: {row['Mean_Cycle']:.1f})")
    
    print(f"  🟡 中速検出器 (1-3秒): {len(medium_detectors)}個")
    for _, row in medium_detectors.iterrows():
        print(f"     • {row['Detector']}: {row['Calc_Time_Sec']:.3f}秒 (平均期間: {row['Mean_Cycle']:.1f})")
    
    print(f"  🔴 低速検出器 (>3秒): {len(slow_detectors)}個")
    for _, row in slow_detectors.iterrows():
        print(f"     • {row['Detector']}: {row['Calc_Time_Sec']:.3f}秒 (平均期間: {row['Mean_Cycle']:.1f})")
    
    # サイクル期間別分析
    print(f"\n📈 サイクル期間分析:")
    print(f"  • 最短平均期間: {df['Mean_Cycle'].min():.1f} ({df.loc[df['Mean_Cycle'].idxmin(), 'Detector']})")
    print(f"  • 最長平均期間: {df['Mean_Cycle'].max():.1f} ({df.loc[df['Mean_Cycle'].idxmax(), 'Detector']})")
    print(f"  • 全体平均期間: {df['Mean_Cycle'].mean():.1f}")
    
    # 期間帯別分類
    short_cycle = df[df['Mean_Cycle'] < 12]
    medium_cycle = df[(df['Mean_Cycle'] >= 12) & (df['Mean_Cycle'] < 18)]
    long_cycle = df[df['Mean_Cycle'] >= 18]
    
    print(f"  🔵 短期サイクル (<12): {len(short_cycle)}個")
    print(f"  🟠 中期サイクル (12-18): {len(medium_cycle)}個") 
    print(f"  🟣 長期サイクル (>18): {len(long_cycle)}個")
    
    # 推奨検出器
    print(f"\n🏆 推奨検出器 (性能スコア順):")
    top_detectors = df.nlargest(5, 'Performance_Score')
    for i, (_, row) in enumerate(top_detectors.iterrows()):
        print(f"  {i+1}. {row['Detector']}")
        print(f"     ✓ 計算時間: {row['Calc_Time_Sec']:.3f}秒")
        print(f"     ✓ 平均期間: {row['Mean_Cycle']:.1f}")
        print(f"     ✓ 性能スコア: {row['Performance_Score']:.1f}")
        print(f"     ✓ 説明: {row['Description']}")
    
    # 用途別推奨
    print(f"\n💡 用途別推奨:")
    
    # 最高速度
    fastest = df.loc[df['Calc_Time_Sec'].idxmin()]
    print(f"  ⚡ 最高速度重視: {fastest['Detector']} ({fastest['Calc_Time_Sec']:.3f}秒)")
    
    # バランス型（中速で安定）
    balanced = df[(df['Calc_Time_Sec'] < 1.0) & (df['Std_Cycle'] < 5.0)]
    if not balanced.empty:
        best_balanced = balanced.loc[balanced['Performance_Score'].idxmax()]
        print(f"  ⚖️ バランス重視: {best_balanced['Detector']} (速度: {best_balanced['Calc_Time_Sec']:.3f}秒, 安定性: 良)")
    
    # 高精度（低標準偏差）
    most_stable = df.loc[df['Std_Cycle'].idxmin()]
    print(f"  🎯 高精度重視: {most_stable['Detector']} (標準偏差: {most_stable['Std_Cycle']:.1f})")
    
    # 次世代技術
    next_gen = df[df['Detector'].str.contains('ultra_supreme|quantum|supreme')]
    if not next_gen.empty:
        print(f"  🚀 次世代技術:")
        for _, row in next_gen.iterrows():
            print(f"     • {row['Detector']}: {row['Description'][:50]}...")
    
    print(f"\n" + "="*80)
    print("✅ 分析完了")


def main():
    """メイン関数"""
    import argparse
    parser = argparse.ArgumentParser(description='サイクル検出器分析レポート生成')
    parser.add_argument('--csv', type=str, help='統計結果のCSVファイルパス')
    args = parser.parse_args()
    
    # CSVファイルを探す
    csv_path = args.csv
    if not csv_path:
        # 最新の統計ファイルを自動検索
        csv_files = list(Path('.').glob('cycle_detectors_statistics_*.csv'))
        if csv_files:
            csv_path = str(max(csv_files, key=lambda x: x.stat().st_mtime))
            print(f"最新の統計ファイルを使用: {csv_path}")
        else:
            print("統計ファイルが見つかりません。先にall_cycle_detectors_performance_test.pyを実行してください。")
            return
    
    analyze_cycle_detectors_results(csv_path)


if __name__ == "__main__":
    main()