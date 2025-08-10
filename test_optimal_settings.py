#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import json
from pathlib import Path

def test_multiple_settings():
    """複数の設定でテストして最適な組み合わせを見つける"""
    
    test_configs = [
        # 名前, 期間, 閾値, ER重み, 動的期間
        ("Conservative", 25, 0.58, 0.55, True),
        ("Balanced", 20, 0.52, 0.60, True),
        ("Aggressive", 15, 0.48, 0.65, True),
        ("Ultra-Aggressive", 12, 0.45, 0.70, True),
        ("Fixed-Conservative", 25, 0.55, 0.60, False),
        ("Fixed-Balanced", 20, 0.50, 0.65, False),
        ("Chop-Heavy", 20, 0.52, 0.40, True),
        ("ER-Heavy", 20, 0.52, 0.75, True),
    ]
    
    results = []
    
    for name, period, threshold, er_weight, use_dynamic in test_configs:
        print(f"\n{'='*60}")
        print(f"テスト: {name}")
        print(f"期間: {period}, 閾値: {threshold}, ER重み: {er_weight}, 動的: {use_dynamic}")
        print('='*60)
        
        # コマンド構築
        cmd = [
            "python", "run_enhanced_trend_state.py",
            "--no-show",
            "--period", str(period),
            "--threshold", str(threshold),
            "--er-weight", str(er_weight)
        ]
        
        if not use_dynamic:
            cmd.append("--no-dynamic")
        
        try:
            # 実行
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            # 結果を解析
            output = result.stdout
            
            # トレンド期間を抽出
            trend_ratio = None
            switches = None
            avg_composite = None
            
            for line in output.split('\n'):
                if "トレンド期間:" in line and "%" in line:
                    try:
                        trend_ratio = float(line.split('(')[1].split('%')[0])
                    except:
                        pass
                elif "トレンド切り替え回数:" in line:
                    try:
                        switches = int(line.split(':')[1].strip())
                    except:
                        pass
                elif "平均複合スコア:" in line:
                    try:
                        avg_composite = float(line.split(':')[1].strip())
                    except:
                        pass
            
            results.append({
                "name": name,
                "period": period,
                "threshold": threshold,
                "er_weight": er_weight,
                "dynamic": use_dynamic,
                "trend_ratio": trend_ratio,
                "switches": switches,
                "avg_composite": avg_composite
            })
            
            print(f"✓ トレンド判定率: {trend_ratio}%")
            print(f"✓ 切り替え回数: {switches}")
            print(f"✓ 平均複合スコア: {avg_composite}")
            
        except subprocess.TimeoutExpired:
            print("✗ タイムアウト")
        except Exception as e:
            print(f"✗ エラー: {e}")
    
    # 結果をまとめて表示
    print(f"\n{'='*80}")
    print("テスト結果サマリー")
    print('='*80)
    print(f"{'設定名':<20} {'期間':>6} {'閾値':>6} {'ER重み':>8} {'動的':>6} {'トレンド%':>10} {'切替回数':>10}")
    print('-'*80)
    
    # トレンド判定率でソート
    results.sort(key=lambda x: x.get('trend_ratio', 0) or 0, reverse=True)
    
    for r in results:
        print(f"{r['name']:<20} {r['period']:>6} {r['threshold']:>6.2f} {r['er_weight']:>8.2f} "
              f"{'Yes' if r['dynamic'] else 'No':>6} {r.get('trend_ratio', 'N/A'):>10} "
              f"{r.get('switches', 'N/A'):>10}")
    
    # 推奨設定を判定
    print(f"\n{'='*60}")
    print("推奨設定")
    print('='*60)
    
    # 適度なトレンド判定率（5-15%）を持つ設定を探す
    optimal_configs = [r for r in results if r.get('trend_ratio') and 5 <= r['trend_ratio'] <= 15]
    
    if optimal_configs:
        # 切り替え回数が少ない順にソート
        optimal_configs.sort(key=lambda x: x.get('switches', 999))
        best = optimal_configs[0]
        
        print(f"最適設定: {best['name']}")
        print(f"  - 期間: {best['period']}")
        print(f"  - 閾値: {best['threshold']}")
        print(f"  - ER重み: {best['er_weight']}")
        print(f"  - 動的期間: {'有効' if best['dynamic'] else '無効'}")
        print(f"  - トレンド判定率: {best['trend_ratio']}%")
        print(f"  - 切り替え回数: {best['switches']}")
    else:
        print("適度なトレンド判定率の設定が見つかりませんでした。")
        print("閾値をさらに下げるか、パラメータの調整が必要です。")

if __name__ == "__main__":
    test_multiple_settings()