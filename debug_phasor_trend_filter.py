#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import pandas as pd

# パスを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from indicators.trend_filter.phasor_trend_filter import PhasorTrendFilter


def debug_phasor_trend_filter():
    """Phasor Trend Filterの詳細デバッグ"""
    print("=== Phasor Trend Filter デバッグ ===")
    
    # テスト用の設定ファイルからデータを読み込み
    from visualization.phasor_trend_filter_chart import PhasorTrendFilterChart
    
    chart = PhasorTrendFilterChart()
    
    try:
        # データを読み込み
        print("\n1. データ読み込み...")
        data = chart.load_data_from_config('config.yaml')
        print(f"データ形状: {data.shape}")
        print(f"価格範囲: {data['close'].min():.2f} - {data['close'].max():.2f}")
        
        # インジケーターを計算
        print("\n2. インジケーター計算...")
        phasor_filter = PhasorTrendFilter(
            period=28,
            trend_threshold=6.0,  # デフォルト値
            src_type='close',
            use_kalman_filter=False
        )
        
        result = phasor_filter.calculate(data)
        
        # 詳細なデバッグ情報
        print(f"\n3. 計算結果デバッグ:")
        print(f"  values形状: {result.values.shape}")
        print(f"  state形状: {result.state.shape}")
        print(f"  phase_angle形状: {result.phase_angle.shape}")
        
        # Stateの詳細統計
        print(f"\n4. State詳細分析:")
        unique_states = np.unique(result.state)
        print(f"  ユニークなstate値: {unique_states}")
        
        for state_val in unique_states:
            count = np.sum(result.state == state_val)
            percentage = count / len(result.state) * 100
            print(f"  State {state_val}: {count}回 ({percentage:.1f}%)")
        
        # 最初の50個の値を確認
        print(f"\n5. 最初の50個のState値:")
        valid_idx = ~np.isnan(result.state)
        if np.any(valid_idx):
            first_valid_idx = np.where(valid_idx)[0][0]
            end_idx = min(first_valid_idx + 50, len(result.state))
            print(f"  有効データ開始: {first_valid_idx}")
            print(f"  State[{first_valid_idx}:{end_idx}]: {result.state[first_valid_idx:end_idx]}")
        else:
            print("  有効なStateデータがありません")
        
        # Phase angleの統計
        print(f"\n6. Phase Angle分析:")
        valid_angles = result.phase_angle[~np.isnan(result.phase_angle)]
        if len(valid_angles) > 0:
            print(f"  有効角度数: {len(valid_angles)}")
            print(f"  角度範囲: {np.min(valid_angles):.1f}° - {np.max(valid_angles):.1f}°")
            print(f"  角度平均: {np.mean(valid_angles):.1f}°")
            print(f"  角度標準偏差: {np.std(valid_angles):.1f}°")
        else:
            print("  有効な角度データがありません")
        
        # Trend strengthの統計
        print(f"\n7. Trend Strength分析:")
        valid_strength = result.trend_strength[~np.isnan(result.trend_strength)]
        if len(valid_strength) > 0:
            print(f"  有効強度数: {len(valid_strength)}")
            print(f"  強度範囲: {np.min(valid_strength):.3f} - {np.max(valid_strength):.3f}")
            print(f"  強度平均: {np.mean(valid_strength):.3f}")
        else:
            print("  有効な強度データがありません")
        
        # Real/Imaginary成分の統計
        print(f"\n8. Real/Imaginary成分分析:")
        valid_real = result.real_component[~np.isnan(result.real_component)]
        valid_imag = result.imag_component[~np.isnan(result.imag_component)]
        
        if len(valid_real) > 0:
            print(f"  Real成分範囲: {np.min(valid_real):.3f} - {np.max(valid_real):.3f}")
            print(f"  Real成分平均: {np.mean(valid_real):.3f}")
        
        if len(valid_imag) > 0:
            print(f"  Imag成分範囲: {np.min(valid_imag):.3f} - {np.max(valid_imag):.3f}")
            print(f"  Imag成分平均: {np.mean(valid_imag):.3f}")
        
        # しきい値のテスト
        print(f"\n9. しきい値調整テスト:")
        for threshold in [1.0, 3.0, 6.0, 10.0, 15.0]:
            print(f"\n  threshold = {threshold}:")
            test_filter = PhasorTrendFilter(
                period=28,
                trend_threshold=threshold,
                src_type='close',
                use_kalman_filter=False
            )
            
            test_result = test_filter.calculate(data)
            unique_test_states = np.unique(test_result.state)
            
            print(f"    ユニークなstate値: {unique_test_states}")
            for state_val in unique_test_states:
                count = np.sum(test_result.state == state_val)
                percentage = count / len(test_result.state) * 100
                print(f"    State {state_val}: {count}回 ({percentage:.1f}%)")
        
        print(f"\n=== デバッグ完了 ===")
        
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_phasor_trend_filter()