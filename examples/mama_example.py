#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MAMA (Mother of Adaptive Moving Average) / FAMA (Following Adaptive Moving Average) 使用例

このスクリプトはMAMAインジケーターの使用方法と結果の確認を示しています。
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# プロジェクトのルートディレクトリをPythonパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from indicators.mama import MAMA
    print("✓ MAMAインジケーターのインポートに成功しました")
except ImportError as e:
    print(f"✗ MAMAインジケーターのインポートに失敗しました: {e}")
    sys.exit(1)


def generate_sample_data(length: int = 200) -> pd.DataFrame:
    """
    テスト用のサンプル価格データを生成
    
    Args:
        length: データの長さ
        
    Returns:
        OHLC価格データのDataFrame
    """
    np.random.seed(42)  # 再現可能な結果のため
    
    # 基本価格トレンド
    base_price = 100.0
    trend = np.linspace(0, 20, length)  # 上昇トレンド
    
    # サイクリックな変動を追加
    cycle1 = 5 * np.sin(np.linspace(0, 4 * np.pi, length))  # 長いサイクル
    cycle2 = 2 * np.sin(np.linspace(0, 12 * np.pi, length))  # 短いサイクル
    
    # ランダムノイズ
    noise = np.random.normal(0, 1, length)
    
    # 終値の計算
    close = base_price + trend + cycle1 + cycle2 + noise
    
    # OHLC データの生成
    high = close + np.abs(np.random.normal(0, 0.5, length))
    low = close - np.abs(np.random.normal(0, 0.5, length))
    open_price = close + np.random.normal(0, 0.3, length)
    
    # DataFrameの作成
    dates = pd.date_range(start='2023-01-01', periods=length, freq='D')
    
    data = pd.DataFrame({
        'date': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close
    })
    
    return data


def test_mama_basic():
    """MAMAの基本機能テスト"""
    print("\n=== MAMA基本機能テスト ===")
    
    # サンプルデータの生成
    data = generate_sample_data(100)
    print(f"サンプルデータ生成: {len(data)} 行")
    
    # MAMAインジケーターの作成
    mama = MAMA(fast_limit=0.5, slow_limit=0.05, src_type='hl2')
    print(f"MAMAインジケーター作成: {mama.name}")
    
    # 計算実行
    try:
        result = mama.calculate(data)
        print("✓ MAMA計算成功")
        
        # 結果の確認
        print(f"MAMA値の長さ: {len(result.mama_values)}")
        print(f"FAMA値の長さ: {len(result.fama_values)}")
        print(f"Period値の長さ: {len(result.period_values)}")
        print(f"Alpha値の長さ: {len(result.alpha_values)}")
        
        # 有効値の数をチェック
        valid_mama = np.sum(~np.isnan(result.mama_values))
        valid_fama = np.sum(~np.isnan(result.fama_values))
        print(f"有効なMAMA値: {valid_mama}")
        print(f"有効なFAMA値: {valid_fama}")
        
        # 最後の10個の値を表示
        print("\n最後の10個の値:")
        for i in range(max(0, len(result.mama_values)-10), len(result.mama_values)):
            if not np.isnan(result.mama_values[i]):
                print(f"  [{i}] MAMA: {result.mama_values[i]:.4f}, "
                      f"FAMA: {result.fama_values[i]:.4f}, "
                      f"Period: {result.period_values[i]:.2f}, "
                      f"Alpha: {result.alpha_values[i]:.6f}")
        
        return result
        
    except Exception as e:
        print(f"✗ MAMA計算エラー: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_mama_methods():
    """MAMAの各メソッドテスト"""
    print("\n=== MAMAメソッドテスト ===")
    
    # サンプルデータの生成
    data = generate_sample_data(50)
    
    # MAMAインジケーターの作成と計算
    mama = MAMA(fast_limit=0.5, slow_limit=0.05)
    result = mama.calculate(data)
    
    # 各メソッドのテスト
    print("get_values():", "OK" if mama.get_values() is not None else "NG")
    print("get_mama_values():", "OK" if mama.get_mama_values() is not None else "NG")
    print("get_fama_values():", "OK" if mama.get_fama_values() is not None else "NG")
    print("get_period_values():", "OK" if mama.get_period_values() is not None else "NG")
    print("get_alpha_values():", "OK" if mama.get_alpha_values() is not None else "NG")
    print("get_phase_values():", "OK" if mama.get_phase_values() is not None else "NG")
    
    # InPhase/Quadrature取得テスト
    iq_result = mama.get_inphase_quadrature()
    print("get_inphase_quadrature():", "OK" if iq_result is not None else "NG")
    
    # リセットテスト
    mama.reset()
    print("reset():", "OK")


def test_mama_parameters():
    """MAMA異なるパラメータテスト"""
    print("\n=== MAMAパラメータテスト ===")
    
    # サンプルデータの生成
    data = generate_sample_data(100)
    
    # 異なるパラメータでのテスト
    test_cases = [
        {'fast_limit': 0.5, 'slow_limit': 0.05, 'src_type': 'hl2'},
        {'fast_limit': 0.7, 'slow_limit': 0.02, 'src_type': 'close'},
        {'fast_limit': 0.3, 'slow_limit': 0.1, 'src_type': 'hlc3'},
    ]
    
    for i, params in enumerate(test_cases):
        print(f"\nテストケース {i+1}: {params}")
        try:
            mama = MAMA(**params)
            result = mama.calculate(data)
            valid_count = np.sum(~np.isnan(result.mama_values))
            print(f"  ✓ 計算成功 - 有効値: {valid_count}")
            
            # 最後の有効値を表示
            last_valid_idx = -1
            for j in range(len(result.mama_values)-1, -1, -1):
                if not np.isnan(result.mama_values[j]):
                    last_valid_idx = j
                    break
            
            if last_valid_idx >= 0:
                print(f"  最終値 - MAMA: {result.mama_values[last_valid_idx]:.4f}, "
                      f"FAMA: {result.fama_values[last_valid_idx]:.4f}")
                
        except Exception as e:
            print(f"  ✗ エラー: {e}")


def create_visualization(result, data):
    """MAMA/FAMAの可視化"""
    print("\n=== MAMA/FAMA可視化 ===")
    
    try:
        plt.style.use('default')
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # メイン価格チャート
        ax1.plot(data['close'], label='Close Price', alpha=0.7, color='black')
        ax1.plot(result.mama_values, label='MAMA', color='blue', linewidth=2)
        ax1.plot(result.fama_values, label='FAMA', color='red', linewidth=2)
        ax1.set_title('MAMA/FAMA Adaptive Moving Averages')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Period値チャート
        ax2.plot(result.period_values, label='Adaptive Period', color='green', linewidth=1.5)
        ax2.set_title('Adaptive Period')
        ax2.set_ylabel('Period')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Alpha値チャート
        ax3.plot(result.alpha_values, label='Alpha (Smoothing Factor)', color='orange', linewidth=1.5)
        ax3.set_title('Alpha (Adaptive Smoothing Factor)')
        ax3.set_ylabel('Alpha')
        ax3.set_xlabel('Time Index')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # ファイル保存
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        filename = os.path.join(output_dir, "mama_fama_analysis.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ チャートを保存しました: {filename}")
        
        # 表示
        plt.show()
        
    except Exception as e:
        print(f"✗ 可視化エラー: {e}")


def main():
    """メイン実行関数"""
    print("MAMA (Mother of Adaptive Moving Average) / FAMA テスト開始")
    print("=" * 60)
    
    # 基本機能テスト
    result = test_mama_basic()
    
    if result is not None:
        # メソッドテスト
        test_mama_methods()
        
        # パラメータテスト
        test_mama_parameters()
        
        # 可視化（結果がある場合）
        data = generate_sample_data(100)
        create_visualization(result, data)
        
        print("\n" + "=" * 60)
        print("✓ 全てのテストが完了しました")
        
        # 統計情報の表示
        valid_mama = np.sum(~np.isnan(result.mama_values))
        valid_fama = np.sum(~np.isnan(result.fama_values))
        
        print(f"\n統計情報:")
        print(f"  有効なMAMA値: {valid_mama}")
        print(f"  有効なFAMA値: {valid_fama}")
        
        if valid_mama > 0:
            last_mama = result.mama_values[~np.isnan(result.mama_values)][-1]
            last_fama = result.fama_values[~np.isnan(result.fama_values)][-1]
            print(f"  最終MAMA値: {last_mama:.4f}")
            print(f"  最終FAMA値: {last_fama:.4f}")
            
            # トレンド方向の簡単な判定
            if last_mama > last_fama:
                print(f"  トレンド方向: 上昇（MAMA > FAMA）")
            else:
                print(f"  トレンド方向: 下降（MAMA < FAMA）")
    
    else:
        print("✗ 基本テストが失敗したため、他のテストをスキップしました")


if __name__ == "__main__":
    main() 