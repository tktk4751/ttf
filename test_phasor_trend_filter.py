#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import pandas as pd

# パスを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from visualization.phasor_trend_filter_chart import PhasorTrendFilterChart


def main():
    """Phasor Trend Filterのテスト実行"""
    print("=== Phasor Trend Filter テスト ===")
    
    # チャートクラスのインスタンス作成
    chart = PhasorTrendFilterChart()
    
    try:
        # 設定ファイルからデータを読み込み
        print("\n1. データ読み込み...")
        chart.load_data_from_config('config.yaml')
        
        # インジケーターを計算
        print("\n2. Phasor Trend Filter計算...")
        chart.calculate_indicators(
            period=28,
            trend_threshold=0.5,
            src_type='close',
            use_kalman_filter=False
        )
        
        # チャートを描画・保存
        print("\n3. チャート描画...")
        output_filename = 'phasor_trend_filter_test.png'
        chart.plot(
            title="Phasor Trend Filter - Test Chart",
            savefig=output_filename,
            show_volume=True
        )
        
        print(f"\nテスト完了! チャートが '{output_filename}' に保存されました。")
        
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()