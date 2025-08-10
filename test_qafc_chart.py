#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QAFC Chart テストスクリプト
実際の相場データでQAFCチャートを生成
"""

import subprocess
import sys
import os
from datetime import datetime, timedelta

def run_qafc_chart():
    """QAFCチャートを実行"""
    print("=== QAFC Chart Test ===\n")
    
    # 最近のデータで表示（過去3ヶ月）
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    
    # 出力ファイル名
    output_file = f"qafc_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    
    # コマンド構築
    cmd = [
        'python', 'visualization/qafc_chart.py',
        '--config', 'config.yaml',
        '--start', start_date,
        '--end', end_date,
        '--output', output_file,
        '--src-type', 'hlc3',
        '--base-multiplier', '2.0',
        '--noise-window', '20',
        '--prediction-lookback', '10'
    ]
    
    print(f"実行コマンド: {' '.join(cmd)}")
    print(f"\n期間: {start_date} から {end_date}")
    print(f"出力: {output_file}\n")
    
    # 実行
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # 出力表示
        if result.stdout:
            print("=== 標準出力 ===")
            print(result.stdout)
        
        if result.stderr:
            print("\n=== エラー出力 ===")
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"\n✓ チャートを正常に生成しました: {output_file}")
            
            # ファイルサイズ確認
            if os.path.exists(output_file):
                size = os.path.getsize(output_file)
                print(f"  ファイルサイズ: {size/1024:.1f} KB")
        else:
            print(f"\n✗ エラーが発生しました (リターンコード: {result.returncode})")
            
    except Exception as e:
        print(f"\n✗ 実行中にエラーが発生しました: {str(e)}")


def run_simple_test():
    """シンプルなテスト（表示のみ）"""
    print("\n\n=== シンプルテスト（表示のみ）===\n")
    
    cmd = [
        'python', 'visualization/qafc_chart.py',
        '--config', 'config.yaml',
        '--start', '2024-10-01',
        '--end', '2024-12-31',
        '--no-volume'  # 出来高非表示でシンプルに
    ]
    
    print(f"実行コマンド: {' '.join(cmd)}")
    
    try:
        # 表示のみ（保存しない）
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\nユーザーによって中断されました")
    except Exception as e:
        print(f"\n✗ エラー: {str(e)}")


if __name__ == "__main__":
    # ファイル保存テスト
    run_qafc_chart()
    
    # インタラクティブ表示テスト（オプション）
    response = input("\n\nインタラクティブな表示テストも実行しますか？ (y/n): ")
    if response.lower() == 'y':
        run_simple_test()