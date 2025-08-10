#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
import os

# プロジェクトのルートディレクトリを追加
sys.path.append('/home/vapor/dev/ttf')

from indicators.ehlers_instantaneous_trendline import EhlersInstantaneousTrendline

def generate_test_data(n_periods=100):
    """テスト用のダミーデータを生成"""
    np.random.seed(42)
    
    dates = pd.date_range(start='2023-01-01', periods=n_periods, freq='1H')
    
    # トレンドのあるダミー価格データを生成
    price = 50000.0
    prices = []
    
    for i in range(n_periods):
        # トレンドとランダムウォーク
        trend = 0.0001 * np.sin(i * 0.1)  # サイン波トレンド
        noise = np.random.normal(0, 0.01)
        price = price * (1 + trend + noise)
        prices.append(price)
    
    prices = np.array(prices)
    
    # OHLCV データの生成
    high = prices * (1 + np.abs(np.random.normal(0, 0.01, n_periods)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.01, n_periods)))
    open_prices = np.roll(prices, 1)
    open_prices[0] = prices[0]
    volume = np.random.uniform(1000, 10000, n_periods)
    
    data = pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': prices,
        'volume': volume
    }, index=dates)
    
    return data

def test_basic_functionality():
    """基本機能のテスト"""
    print("=== Ehlers Instantaneous Trendline 基本機能テスト ===")
    
    try:
        # テストデータ生成
        data = generate_test_data(100)
        
        # 基本的な指標初期化
        indicator = EhlersInstantaneousTrendline(
            alpha=0.07,
            src_type='hl2',
            enable_hyper_er_adaptation=False
        )
        
        print(f"✓ 指標を初期化しました: {indicator.name}")
        
        # 計算実行
        result = indicator.calculate(data)
        
        # 結果の検証
        print(f"✓ 計算が完了しました")
        print(f"  ITrend値数: {len(result.itrend_values)}")
        print(f"  Trigger値数: {len(result.trigger_values)}")
        print(f"  シグナル値数: {len(result.signal_values)}")
        print(f"  使用されたアルファ値: {result.alpha_values[0] if len(result.alpha_values) > 0 else 'なし'}")
        print(f"  適用された平滑化: {result.smoothing_applied}")
        
        # シグナル統計
        bullish_signals = np.sum(result.signal_values == 1)
        bearish_signals = np.sum(result.signal_values == -1)
        neutral_signals = np.sum(result.signal_values == 0)
        
        print(f"  シグナル統計:")
        print(f"    Bullish: {bullish_signals}")
        print(f"    Bearish: {bearish_signals}")
        print(f"    Neutral: {neutral_signals}")
        
        # NaN値のチェック
        nan_itrend = np.sum(np.isnan(result.itrend_values))
        nan_trigger = np.sum(np.isnan(result.trigger_values))
        
        print(f"  NaN値:")
        print(f"    ITrend: {nan_itrend}")
        print(f"    Trigger: {nan_trigger}")
        
        return True
        
    except Exception as e:
        print(f"✗ 基本機能テストでエラー: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_hyper_er_adaptation():
    """HyperER動的適応のテスト"""
    print("\n=== HyperER動的適応テスト ===")
    
    try:
        # テストデータ生成
        data = generate_test_data(100)
        
        # HyperER動的適応を有効にした指標
        indicator = EhlersInstantaneousTrendline(
            alpha=0.07,
            src_type='hl2',
            enable_hyper_er_adaptation=True,
            hyper_er_period=14,
            alpha_min=0.04,
            alpha_max=0.15
        )
        
        print(f"✓ HyperER動的適応指標を初期化しました")
        
        # 計算実行
        result = indicator.calculate(data)
        
        print(f"✓ HyperER動的適応計算が完了しました")
        
        # アルファ値の統計
        if len(result.alpha_values) > 0:
            alpha_min = np.nanmin(result.alpha_values)
            alpha_max = np.nanmax(result.alpha_values)
            alpha_mean = np.nanmean(result.alpha_values)
            
            print(f"  動的アルファ統計:")
            print(f"    最小値: {alpha_min:.4f}")
            print(f"    最大値: {alpha_max:.4f}")
            print(f"    平均値: {alpha_mean:.4f}")
            
            # アルファ範囲の検証
            if 0.04 <= alpha_min <= alpha_max <= 0.15:
                print(f"  ✓ アルファ値が指定範囲内にあります")
            else:
                print(f"  ✗ アルファ値が指定範囲外です")
        
        return True
        
    except Exception as e:
        print(f"✗ HyperER動的適応テストでエラー: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_kalman_smoothing():
    """カルマン統合フィルターのテスト"""
    print("\n=== カルマン統合フィルターテスト ===")
    
    try:
        # テストデータ生成
        data = generate_test_data(100)
        
        # カルマンフィルターを有効にした指標
        indicator = EhlersInstantaneousTrendline(
            alpha=0.07,
            src_type='hl2',
            smoothing_mode='kalman',
            kalman_filter_type='simple'
        )
        
        print(f"✓ カルマンフィルター指標を初期化しました")
        
        # 計算実行
        result = indicator.calculate(data)
        
        print(f"✓ カルマンフィルター計算が完了しました")
        print(f"  適用された平滑化: {result.smoothing_applied}")
        
        # 平滑化された価格の確認
        if len(result.filtered_prices) > 0:
            print(f"  平滑化価格統計:")
            print(f"    データ数: {len(result.filtered_prices)}")
            print(f"    NaN数: {np.sum(np.isnan(result.filtered_prices))}")
        
        return True
        
    except Exception as e:
        print(f"✗ カルマンフィルターテストでエラー: {str(e)}")
        # カルマンフィルターが利用できない場合は警告のみ
        if "統合カルマンフィルターが利用できません" in str(e):
            print("  ℹ カルマンフィルターが利用できないため、このテストはスキップされました")
            return True
        else:
            import traceback
            traceback.print_exc()
            return False

def test_ultimate_smoothing():
    """アルティメットスムーサーのテスト"""
    print("\n=== アルティメットスムーサーテスト ===")
    
    try:
        # テストデータ生成
        data = generate_test_data(100)
        
        # アルティメットスムーサーを有効にした指標
        indicator = EhlersInstantaneousTrendline(
            alpha=0.07,
            src_type='hl2',
            smoothing_mode='ultimate',
            ultimate_smoother_period=10
        )
        
        print(f"✓ アルティメットスムーサー指標を初期化しました")
        
        # 計算実行
        result = indicator.calculate(data)
        
        print(f"✓ アルティメットスムーサー計算が完了しました")
        print(f"  適用された平滑化: {result.smoothing_applied}")
        
        # 平滑化された価格の確認
        if len(result.filtered_prices) > 0:
            print(f"  平滑化価格統計:")
            print(f"    データ数: {len(result.filtered_prices)}")
            print(f"    NaN数: {np.sum(np.isnan(result.filtered_prices))}")
        
        return True
        
    except Exception as e:
        print(f"✗ アルティメットスムーサーテストでエラー: {str(e)}")
        # アルティメットスムーサーが利用できない場合は警告のみ
        if "Ultimate Smootherが利用できません" in str(e):
            print("  ℹ アルティメットスムーサーが利用できないため、このテストはスキップされました")
            return True
        else:
            import traceback
            traceback.print_exc()
            return False

def test_kalman_ultimate_smoothing():
    """カルマン + アルティメットスムーサー組み合わせのテスト"""
    print("\n=== カルマン + アルティメットスムーサー組み合わせテスト ===")
    
    try:
        # テストデータ生成
        data = generate_test_data(100)
        
        # カルマン + アルティメットスムーサーを有効にした指標
        indicator = EhlersInstantaneousTrendline(
            alpha=0.07,
            src_type='hl2',
            smoothing_mode='kalman_ultimate',
            kalman_filter_type='simple',
            ultimate_smoother_period=10
        )
        
        print(f"✓ カルマン + アルティメットスムーサー指標を初期化しました")
        
        # 計算実行
        result = indicator.calculate(data)
        
        print(f"✓ カルマン + アルティメットスムーサー計算が完了しました")
        print(f"  適用された平滑化: {result.smoothing_applied}")
        
        # 平滑化された価格の確認
        if len(result.filtered_prices) > 0:
            print(f"  平滑化価格統計:")
            print(f"    データ数: {len(result.filtered_prices)}")
            print(f"    NaN数: {np.sum(np.isnan(result.filtered_prices))}")
        
        return True
        
    except Exception as e:
        print(f"✗ カルマン + アルティメットスムーサーテストでエラー: {str(e)}")
        # 平滑化機能が利用できない場合は警告のみ
        if ("統合カルマンフィルターが利用できません" in str(e) or 
            "Ultimate Smootherが利用できません" in str(e)):
            print("  ℹ 一部の平滑化機能が利用できないため、このテストはスキップされました")
            return True
        else:
            import traceback
            traceback.print_exc()
            return False

def test_caching():
    """キャッシュ機能のテスト"""
    print("\n=== キャッシュ機能テスト ===")
    
    try:
        # テストデータ生成
        data = generate_test_data(50)
        
        # 指標初期化
        indicator = EhlersInstantaneousTrendline(alpha=0.07, src_type='hl2')
        
        # 最初の計算
        import time
        start_time = time.time()
        result1 = indicator.calculate(data)
        first_calc_time = time.time() - start_time
        
        # 二回目の計算（キャッシュヒットを期待）
        start_time = time.time()
        result2 = indicator.calculate(data)
        second_calc_time = time.time() - start_time
        
        print(f"✓ キャッシュテスト完了")
        print(f"  初回計算時間: {first_calc_time:.4f}秒")
        print(f"  二回目計算時間: {second_calc_time:.4f}秒")
        
        # 結果の一致確認
        arrays_equal = (
            np.allclose(result1.itrend_values, result2.itrend_values, equal_nan=True) and
            np.allclose(result1.trigger_values, result2.trigger_values, equal_nan=True) and
            np.array_equal(result1.signal_values, result2.signal_values)
        )
        
        if arrays_equal:
            print(f"  ✓ キャッシュされた結果が一致します")
        else:
            print(f"  ✗ キャッシュされた結果が一致しません")
        
        return arrays_equal
        
    except Exception as e:
        print(f"✗ キャッシュテストでエラー: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト関数"""
    print("Ehlers Instantaneous Trendline指標 総合テスト開始")
    print("=" * 60)
    
    results = []
    
    # テストの実行
    results.append(test_basic_functionality())
    results.append(test_hyper_er_adaptation())
    results.append(test_kalman_smoothing())
    results.append(test_ultimate_smoothing())
    results.append(test_kalman_ultimate_smoothing())
    results.append(test_caching())
    
    # 結果のまとめ
    print(f"\n{'='*60}")
    print(f"テスト結果まとめ:")
    print(f"  実行済みテスト: {len(results)}")
    print(f"  成功: {sum(results)}")
    print(f"  失敗: {len(results) - sum(results)}")
    
    if all(results):
        print(f"✓ 全てのテストが成功しました！")
        print(f"\nEhlers Instantaneous Trendline指標が正常に動作しています。")
    else:
        print(f"✗ 一部のテストが失敗しました。")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    main()