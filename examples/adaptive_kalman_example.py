#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AdaptiveKalman使用例

適応カルマンフィルタインジケーターの基本的な使い方を示します。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_sample_data(n_points=500):
    """サンプルデータの作成"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=n_points, freq='1h')
    
    # トレンド + サイクル + ノイズの合成
    t = np.arange(n_points)
    base_price = 100
    trend = 0.02 * t
    cycle1 = 5 * np.sin(0.1 * t)
    cycle2 = 2 * np.sin(0.3 * t)
    noise = np.random.randn(n_points) * 1.5
    
    close = base_price + trend + cycle1 + cycle2 + noise
    high = close + np.abs(np.random.randn(n_points) * 0.5)
    low = close - np.abs(np.random.randn(n_points) * 0.5)
    open_price = close + np.random.randn(n_points) * 0.3
    volume = np.random.randint(1000, 5000, n_points)
    
    return pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)

def adaptive_kalman_basic_example():
    """基本的な適応カルマンフィルタの使用例"""
    print("=== AdaptiveKalman基本使用例 ===")
    
    # サンプルデータの作成
    data = create_sample_data(300)
    print(f"サンプルデータ作成完了: {len(data)}行")
    
    try:
        # 必要なクラスを直接インポート（パッケージの問題を回避）
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "adaptive_kalman", 
            "indicators/smoother/adaptive_kalman.py"
        )
        adaptive_kalman_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adaptive_kalman_module)
        
        AdaptiveKalman = adaptive_kalman_module.AdaptiveKalman
        
        # 1. 基本的な設定
        kalman = AdaptiveKalman(
            process_noise=1e-5,
            src_type='close'
        )
        
        result = kalman.calculate(data)
        
        print(f"✓ 計算完了")
        print(f"  フィルタリング後信号長: {len(result.filtered_signal)}")
        print(f"  ノイズ削減効果: {(1 - np.std(result.filtered_signal)/np.std(data['close'])) * 100:.2f}%")
        print(f"  平均信頼度: {result.confidence_score.mean():.4f}")
        
        # 2. 異なるソースタイプでの比較
        print(f"\n=== 異なるソースタイプでの比較 ===")
        
        source_types = ['close', 'hlc3', 'hl2', 'ohlc4']
        results = {}
        
        for src_type in source_types:
            kalman_src = AdaptiveKalman(process_noise=1e-5, src_type=src_type)
            result_src = kalman_src.calculate(data)
            results[src_type] = result_src
            
            # 元の価格データを取得
            if src_type == 'close':
                original = data['close'].values
            elif src_type == 'hlc3':
                original = ((data['high'] + data['low'] + data['close']) / 3).values
            elif src_type == 'hl2':
                original = ((data['high'] + data['low']) / 2).values
            elif src_type == 'ohlc4':
                original = ((data['open'] + data['high'] + data['low'] + data['close']) / 4).values
            
            noise_reduction = (1 - np.std(result_src.filtered_signal)/np.std(original)) * 100
            avg_confidence = result_src.confidence_score.mean()
            
            print(f"  {src_type:6s}: ノイズ削減={noise_reduction:5.2f}%, 信頼度={avg_confidence:.4f}")
        
        # 3. 異なるプロセスノイズでの比較
        print(f"\n=== 異なるプロセスノイズでの比較 ===")
        
        process_noises = [1e-6, 1e-5, 1e-4, 1e-3]
        
        for pn in process_noises:
            kalman_pn = AdaptiveKalman(process_noise=pn, src_type='close')
            result_pn = kalman_pn.calculate(data)
            
            noise_reduction = (1 - np.std(result_pn.filtered_signal)/np.std(data['close'])) * 100
            avg_gain = result_pn.adaptive_gain.mean()
            responsiveness = np.mean(np.abs(np.diff(result_pn.filtered_signal)))
            
            print(f"  ProcNoise={pn:.0e}: ノイズ削減={noise_reduction:5.2f}%, 平均ゲイン={avg_gain:.4f}, 応答性={responsiveness:.4f}")
        
        print(f"\n✓ すべての例が正常に動作しました！")
        return True
        
    except Exception as e:
        print(f"✗ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def adaptive_kalman_visualization_example():
    """適応カルマンフィルタの可視化例"""
    print(f"\n=== AdaptiveKalman可視化例 ===")
    
    # シンプルなNumba関数を使って直接計算
    from numba import njit
    
    @njit(fastmath=True, cache=True)
    def simple_kalman_filter(signal, process_noise=1e-5):
        length = len(signal)
        filtered = np.zeros(length)
        gains = np.zeros(length)
        
        if length > 0:
            state = signal[0]
            error_cov = 1.0
            filtered[0] = state
            gains[0] = 0.5
        
        for i in range(1, length):
            predicted_state = state
            predicted_cov = error_cov + process_noise
            
            if i > 5:
                recent_residuals = np.zeros(5)
                for j in range(5):
                    recent_residuals[j] = abs(signal[i-j] - filtered[i-j])
                obs_noise = np.var(recent_residuals) + 1e-6
            else:
                obs_noise = 1e-3
            
            kalman_gain = predicted_cov / (predicted_cov + obs_noise)
            innovation = signal[i] - predicted_state
            state = predicted_state + kalman_gain * innovation
            error_cov = (1 - kalman_gain) * predicted_cov
            
            filtered[i] = state
            gains[i] = kalman_gain
        
        return filtered, gains
    
    # サンプルデータ
    data = create_sample_data(200)
    close_prices = data['close'].values
    
    # カルマンフィルタを適用
    filtered_prices, adaptive_gains = simple_kalman_filter(close_prices)
    
    print(f"可視化データ準備完了")
    print(f"元データ標準偏差: {np.std(close_prices):.4f}")
    print(f"フィルタ後標準偏差: {np.std(filtered_prices):.4f}")
    print(f"ノイズ削減効果: {(1 - np.std(filtered_prices)/np.std(close_prices)) * 100:.2f}%")
    
    # 簡単な統計表示
    print(f"適応ゲイン統計:")
    print(f"  平均: {adaptive_gains.mean():.6f}")
    print(f"  範囲: {adaptive_gains.min():.6f} - {adaptive_gains.max():.6f}")
    
    return True

if __name__ == "__main__":
    # 基本使用例
    success1 = adaptive_kalman_basic_example()
    
    # 可視化例
    success2 = adaptive_kalman_visualization_example()
    
    if success1 and success2:
        print(f"\n🎉 すべての例が成功しました！")
        print(f"\nAdaptiveKalmanの特徴:")
        print(f"  ✓ 適応的ノイズ推定")
        print(f"  ✓ 動的カルマンゲイン調整")
        print(f"  ✓ 複数のプライスソース対応")
        print(f"  ✓ 高いノイズ削減効果")
        print(f"  ✓ リアルタイム適用可能")
    else:
        print(f"\n❌ 一部の例でエラーが発生しました")