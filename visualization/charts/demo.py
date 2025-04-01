#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ローソク足チャートとAlphaMAのデモ
"""

import os
import sys
import argparse
from pathlib import Path

# プロジェクトのルートディレクトリをパスに追加
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from visualization.charts.candlestick_alpha_ma import CandlestickAlphaMaChart


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='ローソク足チャートとAlphaMAのデモ')
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
    chart = CandlestickAlphaMaChart()
    
    # データの読み込み
    chart.load_data_from_config(config_path)
    
    # AlphaMAの計算
    chart.calculate_alpha_ma(
        er_period=args.er_period,
        max_kama_period=args.max_kama,
        min_kama_period=args.min_kama
    )
    
    # チャートの表示
    chart.plot(
        title=f"AlphaMAチャート (ER期間={args.er_period}, MaxKAMA={args.max_kama}, MinKAMA={args.min_kama})",
        start_date=args.start,
        end_date=args.end,
        savefig=args.save
    )


if __name__ == "__main__":
    main() 