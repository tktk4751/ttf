#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_grand_cycle_ma_simple():
    """シンプルなグランドサイクルMAテスト（インポートエラー回避）"""
    print("=== グランドサイクルMA シンプルテスト開始 ===")
    
    try:
        # 直接インポートを試行
        from indicators.grand_cycle_ma import GrandCycleMA
        print("✓ GrandCycleMA インポート成功")
        
        # テストデータの作成
        print("\n1. テストデータ作成中...")
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        np.random.seed(42)
        
        # トレンドとサイクルを含む価格データを生成
        trend = np.linspace(100, 120, 100)
        cycle = 5 * np.sin(np.linspace(0, 4*np.pi, 100))
        noise = np.random.normal(0, 1, 100)
        close_prices = trend + cycle + noise
        
        data = pd.DataFrame({
            'open': close_prices * 0.998,
            'high': close_prices * 1.002,
            'low': close_prices * 0.996,
            'close': close_prices,
            'volume': np.random.uniform(100000, 500000, 100)
        }, index=dates)
        
        print(f"✓ テストデータ作成完了: {len(data)}件")
        
        # 2. 利用可能な検出器一覧
        print("\n2. 利用可能なサイクル検出器:")
        try:
            available_detectors = GrandCycleMA.get_available_detectors()
            for detector, description in available_detectors.items():
                print(f"  - {detector}: {description}")
        except Exception as e:
            print(f"  検出器一覧取得エラー: {e}")
        
        # 3. 基本的な検出器でテスト
        test_detectors = ['hody']  # 最もシンプルな検出器から開始
        
        print(f"\n3. {test_detectors[0]} 検出器でテスト:")
        
        try:
            # グランドサイクルMAの作成
            grand_cycle_ma = GrandCycleMA(
                detector_type=test_detectors[0],
                fast_limit=0.5,
                slow_limit=0.05,
                src_type='close'  # シンプルなソースタイプ
            )
            print(f"✓ GrandCycleMA({test_detectors[0]}) 作成成功")
            
            # 計算実行
            result = grand_cycle_ma.calculate(data)
            print("✓ 計算実行成功")
            
            # 結果の確認
            print(f"\n結果確認:")
            print(f"  グランドMAMA長: {len(result.grand_mama_values)}")
            print(f"  グランドFAMA長: {len(result.grand_fama_values)}")
            print(f"  サイクル周期長: {len(result.cycle_period)}")
            
            # 有効データの統計
            valid_mama = result.grand_mama_values[~np.isnan(result.grand_mama_values)]
            valid_fama = result.grand_fama_values[~np.isnan(result.grand_fama_values)]
            valid_cycle = result.cycle_period[~np.isnan(result.cycle_period)]
            
            print(f"  有効グランドMAMAデータ: {len(valid_mama)}/{len(result.grand_mama_values)}")
            if len(valid_mama) > 0:
                print(f"    平均値: {np.mean(valid_mama):.4f}")
                print(f"    範囲: {np.min(valid_mama):.4f} - {np.max(valid_mama):.4f}")
            
            print(f"  有効グランドFAMAデータ: {len(valid_fama)}/{len(result.grand_fama_values)}")
            if len(valid_fama) > 0:
                print(f"    平均値: {np.mean(valid_fama):.4f}")
                print(f"    範囲: {np.min(valid_fama):.4f} - {np.max(valid_fama):.4f}")
            
            print(f"  有効サイクル周期データ: {len(valid_cycle)}/{len(result.cycle_period)}")
            if len(valid_cycle) > 0:
                print(f"    平均周期: {np.mean(valid_cycle):.2f}")
                print(f"    周期範囲: {np.min(valid_cycle):.2f} - {np.max(valid_cycle):.2f}")
            
            # 4. 簡単な可視化
            print("\n4. 結果の可視化:")
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            # 価格とグランドサイクルMA
            axes[0].plot(data.index, data['close'], label='Close Price', alpha=0.7, color='black')
            
            if len(valid_mama) > 0:
                valid_indices = ~np.isnan(result.grand_mama_values)
                axes[0].plot(
                    data.index[valid_indices], 
                    result.grand_mama_values[valid_indices],
                    label='Grand MAMA',
                    alpha=0.8,
                    color='blue'
                )
            
            if len(valid_fama) > 0:
                valid_indices = ~np.isnan(result.grand_fama_values)
                axes[0].plot(
                    data.index[valid_indices], 
                    result.grand_fama_values[valid_indices],
                    label='Grand FAMA',
                    alpha=0.8,
                    color='red'
                )
            
            axes[0].set_title(f'Price vs Grand Cycle MA ({test_detectors[0]})')
            axes[0].set_ylabel('Price')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # サイクル周期
            if len(valid_cycle) > 0:
                valid_indices = ~np.isnan(result.cycle_period)
                axes[1].plot(
                    data.index[valid_indices],
                    result.cycle_period[valid_indices],
                    label='Cycle Period',
                    alpha=0.8,
                    color='green'
                )
            
            axes[1].set_title('Cycle Period')
            axes[1].set_ylabel('Period')
            axes[1].set_xlabel('Date')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 画像を保存
            output_file = 'grand_cycle_ma_simple_test.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"  ✓ グラフを保存しました: {output_file}")
            
            plt.show()
            
            print(f"\n✓ {test_detectors[0]} テスト完了")
            
        except Exception as e:
            import traceback
            print(f"✗ テストエラー: {e}")
            print(f"詳細: {traceback.format_exc()}")
            
    except ImportError as e:
        print(f"✗ インポートエラー: {e}")
        print("必要なモジュールが見つかりません")
        return False
    
    print("\n=== グランドサイクルMA シンプルテスト完了 ===")
    return True


def test_core_calculation():
    """コア計算関数の単体テスト"""
    print("\n=== コア計算関数テスト ===")
    
    try:
        from indicators.grand_cycle_ma import calculate_grand_cycle_ma_core
        
        # テストデータ
        price = np.array([100.0, 101.0, 102.0, 101.5, 100.5, 99.0, 98.5, 99.5, 101.0, 102.5], dtype=np.float64)
        cycle_period = np.array([20.0, 20.0, 18.0, 16.0, 15.0, 17.0, 19.0, 20.0, 21.0, 20.0], dtype=np.float64)
        
        # 計算実行
        grand_mama, grand_fama, alpha, phase = calculate_grand_cycle_ma_core(
            price, cycle_period, fast_limit=0.5, slow_limit=0.05
        )
        
        print(f"入力データ長: {len(price)}")
        print(f"出力データ長: {len(grand_mama)}")
        print(f"Grand MAMA範囲: {np.min(grand_mama):.4f} - {np.max(grand_mama):.4f}")
        print(f"Grand FAMA範囲: {np.min(grand_fama):.4f} - {np.max(grand_fama):.4f}")
        print(f"Alpha範囲: {np.min(alpha):.4f} - {np.max(alpha):.4f}")
        
        print("✓ コア計算関数テスト成功")
        
    except Exception as e:
        print(f"✗ コア計算関数テストエラー: {e}")


if __name__ == "__main__":
    # コア計算テスト
    test_core_calculation()
    
    # メインテスト
    test_grand_cycle_ma_simple()