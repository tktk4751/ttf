#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QAFC Chart シンプルテストスクリプト
エラー処理を改善した版
"""

import subprocess
import sys
from datetime import datetime, timedelta

def run_qafc_chart_simple():
    """QAFCチャートを実行（シンプル版）"""
    print("=== QAFC Chart Simple Test ===\n")
    
    # 最近のデータで表示（過去30日）
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    # 出力ファイル名
    output_file = f"qafc_chart_test.png"
    
    # コマンド構築
    cmd = [
        sys.executable, 'visualization/qafc_chart.py',
        '--config', 'config.yaml',
        '--start', start_date,
        '--end', end_date,
        '--output', output_file,
        '--src-type', 'close',  # シンプルにclose価格を使用
        '--base-multiplier', '2.0',
        '--noise-window', '10',  # 小さめのウィンドウ
        '--prediction-lookback', '5',  # 小さめのlookback
        '--no-volume'  # 出来高非表示でシンプルに
    ]
    
    print(f"実行コマンド: {' '.join(cmd)}")
    print(f"\n期間: {start_date} から {end_date}")
    print(f"出力: {output_file}\n")
    
    # 実行
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # 出力表示
        if result.stdout:
            print("=== 実行結果 ===")
            print(result.stdout)
        
        if result.stderr:
            print("\n=== エラー情報 ===")
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"\n✓ チャートを正常に生成しました: {output_file}")
        else:
            print(f"\n✗ エラーが発生しました (リターンコード: {result.returncode})")
            
    except Exception as e:
        print(f"\n✗ 実行中にエラーが発生しました: {str(e)}")


if __name__ == "__main__":
    run_qafc_chart_simple()