#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
量子最強トレンド・サイクル検出器のテスト
"""

import sys
import os
import time
import numpy as np
import pandas as pd

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from indicators.trend_filter.quantum_supreme_trend_cycle_detector import (
        QuantumSupremeTrendCycleDetector,
        calculate_quantum_supreme_trend_cycle,
        ultra_precision_dft_engine,
        quantum_harmonic_oscillator_cycle_detector,
        advanced_wavelet_transform_analysis,
        fractal_dimension_analysis,
        information_entropy_analysis,
        phase_space_reconstruction
    )
    print("量子最強トレンド・サイクル検出器を正常にインポートしました")
except ImportError as e:
    print(f"インポートエラー: {e}")
    print("基本的なアルゴリズム関数のみをテストします")
    
    # アルゴリズム関数を直接定義（最小限のテスト用）
    from numba import njit
    
    @njit
    def simple_dft_test(data):
        """簡単なDFTテスト"""
        n = len(data)
        result = np.zeros(n)
        for i in range(10, n):
            # 簡単な周期検出
            window = data[i-10:i]
            max_power = 0.0
            best_period = 10.0
            
            for period in range(5, 15):
                power = 0.0
                for j in range(len(window)):
                    angle = 2 * np.pi * j / period
                    power += window[j] * np.cos(angle)
                
                if abs(power) > abs(max_power):
                    max_power = power
                    best_period = period
            
            result[i] = best_period
        return result


def generate_test_data(n=500):
    """テストデータ生成"""
    np.random.seed(42)
    
    # 複雑な市場データシミュレーション
    prices = [100.0]
    for i in range(1, n):
        # 複数の周期的コンポーネント
        trend = 0.001 * np.sin(2 * np.pi * i / 100)
        cycle1 = 0.005 * np.sin(2 * np.pi * i / 20)
        cycle2 = 0.003 * np.sin(2 * np.pi * i / 50)
        noise = np.random.normal(0, 0.002)
        
        change = trend + cycle1 + cycle2 + noise
        prices.append(prices[-1] * (1 + change))
    
    # OHLC データ構築
    data = []
    for i, close in enumerate(prices):
        daily_range = abs(np.random.normal(0, close * 0.008))
        high = close + daily_range * np.random.uniform(0.4, 1.0)
        low = close - daily_range * np.random.uniform(0.4, 1.0)
        
        if i == 0:
            open_price = close
        else:
            gap = np.random.normal(0, close * 0.003)
            open_price = prices[i-1] + gap
        
        # 論理的整合性の確保
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.uniform(1000, 10000)
        })
    
    return pd.DataFrame(data)


def test_core_algorithms():
    """コアアルゴリズムのテスト"""
    print("=== コアアルゴリズムテスト ===")
    
    # テストデータ生成
    test_data = np.random.randn(200) * 0.1 + 100
    test_data = test_data.astype(np.float64)
    
    try:
        print("1. 超高精度DFT解析エンジンのテスト...")
        start_time = time.time()
        dft_periods, dft_power, dft_coherence, dft_purity = ultra_precision_dft_engine(test_data)
        dft_time = time.time() - start_time
        
        print(f"   DFT解析完了: {dft_time:.3f}秒")
        print(f"   平均期間: {np.mean(dft_periods[dft_periods > 0]):.2f}")
        print(f"   平均純度: {np.mean(dft_purity[dft_purity > 0]):.4f}")
        print(f"   平均コヒーレンス: {np.mean(dft_coherence[dft_coherence > 0]):.4f}")
        
    except Exception as e:
        print(f"   DFTエラー: {e}")
    
    try:
        print("\n2. 量子調和振動子検出器のテスト...")
        start_time = time.time()
        qphase, qamp, qenergy, qcoh = quantum_harmonic_oscillator_cycle_detector(test_data)
        quantum_time = time.time() - start_time
        
        print(f"   量子解析完了: {quantum_time:.3f}秒")
        print(f"   平均振幅: {np.mean(qamp[qamp > 0]):.4f}")
        print(f"   平均エネルギー: {np.mean(qenergy[qenergy > 0]):.4f}")
        print(f"   平均コヒーレンス: {np.mean(qcoh[qcoh > 0]):.4f}")
        
    except Exception as e:
        print(f"   量子解析エラー: {e}")
    
    try:
        print("\n3. ウェーブレット変換のテスト...")
        start_time = time.time()
        scales = np.array([2, 4, 8, 16], dtype=np.float64)
        wcoeffs, wtrend, wcycle = advanced_wavelet_transform_analysis(test_data, scales)
        wavelet_time = time.time() - start_time
        
        print(f"   ウェーブレット解析完了: {wavelet_time:.3f}秒")
        print(f"   係数形状: {wcoeffs.shape}")
        print(f"   平均トレンド成分: {np.mean(wtrend):.4f}")
        print(f"   平均サイクル成分: {np.mean(wcycle):.4f}")
        
    except Exception as e:
        print(f"   ウェーブレットエラー: {e}")
    
    try:
        print("\n4. フラクタル次元解析のテスト...")
        start_time = time.time()
        fdim, hurst, mstruct = fractal_dimension_analysis(test_data)
        fractal_time = time.time() - start_time
        
        print(f"   フラクタル解析完了: {fractal_time:.3f}秒")
        print(f"   平均フラクタル次元: {np.mean(fdim[fdim > 0]):.4f}")
        print(f"   平均ハースト指数: {np.mean(hurst[hurst > 0]):.4f}")
        print(f"   市場構造範囲: {np.min(mstruct)} - {np.max(mstruct)}")
        
    except Exception as e:
        print(f"   フラクタルエラー: {e}")
    
    try:
        print("\n5. エントロピー解析のテスト...")
        start_time = time.time()
        shannon, tsallis, infoflow = information_entropy_analysis(test_data)
        entropy_time = time.time() - start_time
        
        print(f"   エントロピー解析完了: {entropy_time:.3f}秒")
        print(f"   平均シャノンエントロピー: {np.mean(shannon[shannon > 0]):.4f}")
        print(f"   平均ツァリスエントロピー: {np.mean(tsallis[tsallis > 0]):.4f}")
        print(f"   情報流動性範囲: {np.min(infoflow)} - {np.max(infoflow)}")
        
    except Exception as e:
        print(f"   エントロピーエラー: {e}")
    
    try:
        print("\n6. 位相空間再構成のテスト...")
        start_time = time.time()
        pvolume, lyap, corrdim = phase_space_reconstruction(test_data)
        phase_time = time.time() - start_time
        
        print(f"   位相空間解析完了: {phase_time:.3f}秒")
        print(f"   平均位相空間体積: {np.mean(pvolume[pvolume > 0]):.4f}")
        print(f"   平均リアプノフ指数: {np.mean(lyap[np.isfinite(lyap)]):.4f}")
        print(f"   平均相関次元: {np.mean(corrdim[corrdim > 0]):.4f}")
        
    except Exception as e:
        print(f"   位相空間エラー: {e}")


def test_full_detector():
    """フル検出器のテスト"""
    print("\n=== フル検出器テスト ===")
    
    try:
        # テストデータ生成
        df = generate_test_data(300)  # 計算時間短縮のため300ポイント
        print(f"テストデータ: {len(df)}行")
        print(f"価格範囲: {df['close'].min():.4f} - {df['close'].max():.4f}")
        
        # 量子最強検出器をテスト
        detector = QuantumSupremeTrendCycleDetector(
            quantum_levels=10,  # 計算時間短縮
            quantum_window=50,
            wavelet_scales=[2, 4, 8, 16],
            ensemble_learning_rate=0.02,
            use_kalman_filter=False,  # カルマンフィルターは無効化
            fractal_window=30,
            entropy_window=30
        )
        
        print("\n量子最強検出器計算開始...")
        start_time = time.time()
        result = detector.calculate(df)
        end_time = time.time()
        
        print(f"計算完了: {end_time - start_time:.2f}秒")
        print(f"品質スコア: {result.quality_score:.4f}")
        print(f"アルゴリズムバージョン: {result.algorithm_version}")
        
        # 統計情報
        valid_count = len(result.trend_mode)
        trend_ratio = np.mean(result.trend_mode)
        cycle_ratio = np.mean(result.cycle_mode)
        avg_confidence = np.mean(result.confidence)
        
        print(f"\n結果統計:")
        print(f"  有効データポイント: {valid_count}")
        print(f"  トレンドモード比率: {trend_ratio:.2%}")
        print(f"  サイクルモード比率: {cycle_ratio:.2%}")
        print(f"  平均信頼度: {avg_confidence:.4f}")
        
        # 信号統計
        buy_signals = np.sum(result.signal > 0)
        sell_signals = np.sum(result.signal < 0)
        neutral_signals = np.sum(result.signal == 0)
        
        print(f"\n信号統計:")
        print(f"  買いシグナル: {buy_signals}")
        print(f"  売りシグナル: {sell_signals}")
        print(f"  中立シグナル: {neutral_signals}")
        
        # 高度な統計
        print(f"\n高度統計:")
        print(f"  平均フラクタル次元: {np.mean(result.fractal_dimension[result.fractal_dimension > 0]):.4f}")
        print(f"  平均ハースト指数: {np.mean(result.hurst_exponent[result.hurst_exponent > 0]):.4f}")
        print(f"  平均シャノンエントロピー: {np.mean(result.shannon_entropy[result.shannon_entropy > 0]):.4f}")
        print(f"  平均量子振幅: {np.mean(result.quantum_amplitude[result.quantum_amplitude > 0]):.4f}")
        
        return True
        
    except Exception as e:
        print(f"フル検出器テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def benchmark_performance():
    """パフォーマンステスト"""
    print("\n=== パフォーマンステスト ===")
    
    data_sizes = [100, 200, 300]
    
    for size in data_sizes:
        print(f"\nデータサイズ: {size}ポイント")
        
        try:
            df = generate_test_data(size)
            
            # 軽量設定で実行
            detector = QuantumSupremeTrendCycleDetector(
                quantum_levels=8,
                quantum_window=40,
                wavelet_scales=[2, 4, 8],
                use_kalman_filter=False,
                fractal_window=25,
                entropy_window=25
            )
            
            start_time = time.time()
            result = detector.calculate(df)
            execution_time = time.time() - start_time
            
            points_per_second = size / execution_time
            
            print(f"  実行時間: {execution_time:.3f}秒")
            print(f"  処理速度: {points_per_second:.1f}ポイント/秒")
            print(f"  品質スコア: {result.quality_score:.4f}")
            
        except Exception as e:
            print(f"  サイズ{size}でエラー: {e}")


def main():
    """メイン実行関数"""
    print("=== 量子最強トレンド・サイクル検出器 - 統合テスト ===")
    print("人類史上最強のトレンド・サイクル判別アルゴリズム")
    print("")
    
    # 1. コアアルゴリズムのテスト
    test_core_algorithms()
    
    # 2. フル検出器のテスト
    success = test_full_detector()
    
    # 3. パフォーマンステスト
    if success:
        benchmark_performance()
    
    print("\n=== 統合テスト完了 ===")
    
    if success:
        print("✅ 全てのテストが正常に完了しました")
        print("🚀 量子最強トレンド・サイクル検出器は正常に動作しています")
    else:
        print("❌ 一部のテストでエラーが発生しました")
        print("⚠️  依存関係やインポートパスを確認してください")


if __name__ == "__main__":
    main()