#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
アルファケルトナーチャネルとアルファATRのデモスクリプト
"""

import os
import sys
import argparse
from pathlib import Path

# プロジェクトのルートディレクトリをパスに追加
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from visualization.charts.keltner_atr_chart import KeltnerATRChart


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='アルファケルトナーチャネルとアルファATRのデモ')
    parser.add_argument('--config', default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', help='表示開始日（YYYY-MM-DD）')
    parser.add_argument('--end', help='表示終了日（YYYY-MM-DD）')
    parser.add_argument('--save', help='チャート保存先のパス（指定しない場合は画面表示）')
    parser.add_argument('--er_period', type=int, default=21, help='効率比の計算期間')
    parser.add_argument('--max_kama', type=int, default=55, help='KAMAピリオドの最大値')
    parser.add_argument('--min_kama', type=int, default=8, help='KAMAピリオドの最小値')
    parser.add_argument('--max_atr', type=int, default=55, help='ATR期間の最大値')
    parser.add_argument('--min_atr', type=int, default=8, help='ATR期間の最小値')
    parser.add_argument('--max_mult', type=float, default=3.0, help='ATR乗数の最大値')
    parser.add_argument('--min_mult', type=float, default=1.5, help='ATR乗数の最小値')
    parser.add_argument('--no_independent_atr', action='store_true', help='独立したアルファATRを使用しない')
    
    args = parser.parse_args()
    
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)
    
    print(f"設定ファイル: {config_path}")
    
    # チャートを作成
    chart = KeltnerATRChart()
    
    # データの読み込み
    chart.load_data_from_config(config_path)
    
    # インジケーターの計算
    chart.calculate_indicators(
        er_period=args.er_period,
        max_kama_period=args.max_kama,
        min_kama_period=args.min_kama,
        max_atr_period=args.max_atr,
        min_atr_period=args.min_atr,
        max_multiplier=args.max_mult,
        min_multiplier=args.min_mult,
        use_independent_atr=not args.no_independent_atr
    )
    
    # 設定情報をタイトルに追加
    title_parts = [
        f"アルファケルトナーチャネルとアルファATR",
        f"ER期間={args.er_period}",
        f"KAMA={args.min_kama}-{args.max_kama}",
        f"ATR={args.min_atr}-{args.max_atr}",
        f"乗数={args.min_mult}-{args.max_mult}"
    ]
    title = " | ".join(title_parts)
    
    # 期間情報を表示
    if args.start or args.end:
        period_info = f"期間指定: {args.start or '全期間開始'} → {args.end or '全期間終了'}"
        print(period_info)
    else:
        print("すべての期間を表示します")
    
    # チャートの表示
    chart.plot(
        title=title,
        start_date=args.start,
        end_date=args.end,
        savefig=args.save,
        figsize=(16, 12)  # より大きめのサイズで表示
    )


if __name__ == "__main__":
    main() 