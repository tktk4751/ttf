#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
複数インジケーター付きローソク足チャートのデモ
- アルファMA
- アルファトレンド
- アルファチョピネス
- アルファフィルター
"""

import os
import sys
import argparse
from pathlib import Path

# プロジェクトのルートディレクトリをパスに追加
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from visualization.charts.candlestick_multi_indicator import MultiIndicatorChart


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='複数インジケーター付きローソク足チャートのデモ')
    parser.add_argument('--config', default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', help='表示開始日（YYYY-MM-DD）')
    parser.add_argument('--end', help='表示終了日（YYYY-MM-DD）')
    parser.add_argument('--save', help='チャート保存先のパス（指定しない場合は画面表示）')
    parser.add_argument('--er_period', type=int, default=21, help='効率比の計算期間')
    parser.add_argument('--max_kama', type=int, default=144, help='KAMAピリオドの最大値')
    parser.add_argument('--min_kama', type=int, default=10, help='KAMAピリオドの最小値')
    
    args = parser.parse_args()
    
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)
    
    print(f"設定ファイル: {config_path}")
    
    # チャートを作成
    chart = MultiIndicatorChart()
    
    # データの読み込み
    chart.load_data_from_config(config_path)
    
    # インジケーターの計算
    chart.calculate_indicators(
        er_period=args.er_period,
        max_kama_period=args.max_kama,
        min_kama_period=args.min_kama
    )
    
    # チャートの表示（すべての期間を表示）
    title = f"マルチインジケーターチャート (ER期間={args.er_period})"
    if args.start or args.end:
        print(f"期間指定: {args.start or '全期間開始'} → {args.end or '全期間終了'}")
    else:
        print("すべての期間を表示します")
    
    chart.plot(
        title=title,
        start_date=args.start,
        end_date=args.end,
        savefig=args.save,
        figsize=(16, 14)  # より大きめのサイズで表示
    )


if __name__ == "__main__":
    main() 