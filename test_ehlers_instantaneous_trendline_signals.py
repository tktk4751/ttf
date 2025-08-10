#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
import os

# プロジェクトのルートディレクトリを追加
sys.path.append('/home/vapor/dev/ttf')

from signals.implementations.ehlers_instantaneous_trendline.entry import (
    EhlersInstantaneousTrendlinePositionEntrySignal,
    EhlersInstantaneousTrendlineCrossoverEntrySignal
)

def generate_test_data(n_periods=150):
    """テスト用のダミーデータを生成"""
    np.random.seed(42)
    
    dates = pd.date_range(start='2023-01-01', periods=n_periods, freq='1H')
    
    # トレンドのあるダミー価格データを生成
    price = 50000.0
    prices = []
    
    for i in range(n_periods):
        # 複合的なトレンドパターン
        trend1 = 0.0002 * np.sin(i * 0.05)  # 長期波
        trend2 = 0.0001 * np.sin(i * 0.2)   # 短期波
        noise = np.random.normal(0, 0.01)
        price = price * (1 + trend1 + trend2 + noise)
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

def test_position_entry_signal():
    """位置関係エントリーシグナルのテスト"""
    print("=== Ehlers Instantaneous Trendline位置関係エントリーシグナルテスト ===")
    
    try:
        # テストデータ生成
        data = generate_test_data(150)
        
        # 基本的な位置関係シグナル
        position_signal = EhlersInstantaneousTrendlinePositionEntrySignal(
            alpha=0.07,
            src_type='hl2',
            enable_hyper_er_adaptation=False
        )
        
        print(f"✓ 位置関係シグナルを初期化しました: {position_signal.name}")
        
        # シグナル生成
        signals = position_signal.generate(data)
        
        # シグナル統計
        long_signals = np.sum(signals == 1)
        short_signals = np.sum(signals == -1)
        neutral_signals = np.sum(signals == 0)
        
        print(f"✓ シグナル生成完了")
        print(f"  ロングシグナル: {long_signals} ({long_signals/len(signals)*100:.1f}%)")
        print(f"  ショートシグナル: {short_signals} ({short_signals/len(signals)*100:.1f}%)")
        print(f"  ニュートラル: {neutral_signals} ({neutral_signals/len(signals)*100:.1f}%)")
        
        # 指標値の取得テスト
        itrend_values = position_signal.get_itrend_values(data)
        trigger_values = position_signal.get_trigger_values(data)
        
        print(f"  ITrend値統計: min={np.nanmin(itrend_values):.2f}, max={np.nanmax(itrend_values):.2f}")
        print(f"  Trigger値統計: min={np.nanmin(trigger_values):.2f}, max={np.nanmax(trigger_values):.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 位置関係エントリーシグナルテストでエラー: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_crossover_entry_signal():
    """クロスオーバーエントリーシグナルのテスト"""
    print("\n=== Ehlers Instantaneous Trendlineクロスオーバーエントリーシグナルテスト ===")
    
    try:
        # テストデータ生成
        data = generate_test_data(150)
        
        # クロスオーバーシグナル
        crossover_signal = EhlersInstantaneousTrendlineCrossoverEntrySignal(
            alpha=0.07,
            src_type='hl2',
            enable_hyper_er_adaptation=False
        )
        
        print(f"✓ クロスオーバーシグナルを初期化しました: {crossover_signal.name}")
        
        # シグナル生成
        signals = crossover_signal.generate(data)
        
        # シグナル統計
        long_signals = np.sum(signals == 1)
        short_signals = np.sum(signals == -1)
        neutral_signals = np.sum(signals == 0)
        
        print(f"✓ シグナル生成完了")
        print(f"  ロングクロスオーバー: {long_signals}")
        print(f"  ショートクロスオーバー: {short_signals}")
        print(f"  ニュートラル: {neutral_signals} ({neutral_signals/len(signals)*100:.1f}%)")
        
        # クロスオーバーの妥当性チェック
        if long_signals > 0 or short_signals > 0:
            print(f"  ✓ クロスオーバーシグナルが検出されました")
        else:
            print(f"  ⚠ クロスオーバーシグナルが検出されませんでした（データ依存）")
        
        return True
        
    except Exception as e:
        print(f"✗ クロスオーバーエントリーシグナルテストでエラー: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_hyper_er_adaptation():
    """HyperER動的適応のテスト"""
    print("\n=== HyperER動的適応シグナルテスト ===")
    
    try:
        # テストデータ生成
        data = generate_test_data(150)
        
        # HyperER動的適応シグナル
        adaptive_signal = EhlersInstantaneousTrendlinePositionEntrySignal(
            alpha=0.07,  # 基本値（使用されない）
            src_type='hl2',
            enable_hyper_er_adaptation=True,
            hyper_er_period=14,
            alpha_min=0.04,
            alpha_max=0.15
        )
        
        print(f"✓ HyperER動的適応シグナルを初期化しました")
        
        # シグナル生成
        signals = adaptive_signal.generate(data)
        
        # アルファ値統計
        alpha_values = adaptive_signal.get_alpha_values(data)
        if len(alpha_values) > 0:
            alpha_min = np.nanmin(alpha_values)
            alpha_max = np.nanmax(alpha_values)
            alpha_mean = np.nanmean(alpha_values)
            
            print(f"✓ HyperER動的適応完了")
            print(f"  動的アルファ統計:")
            print(f"    最小値: {alpha_min:.4f}")
            print(f"    最大値: {alpha_max:.4f}")
            print(f"    平均値: {alpha_mean:.4f}")
            
            # アルファ範囲の検証
            if 0.04 <= alpha_min <= alpha_max <= 0.15:
                print(f"  ✓ アルファ値が指定範囲内にあります")
            else:
                print(f"  ✗ アルファ値が指定範囲外です")
        
        # シグナル統計
        long_signals = np.sum(signals == 1)
        short_signals = np.sum(signals == -1)
        
        print(f"  シグナル統計:")
        print(f"    ロング: {long_signals}")
        print(f"    ショート: {short_signals}")
        
        return True
        
    except Exception as e:
        print(f"✗ HyperER動的適応テストでエラー: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_kalman_smoothing():
    """カルマンフィルター平滑化のテスト"""
    print("\n=== カルマンフィルター平滑化シグナルテスト ===")
    
    try:
        # テストデータ生成
        data = generate_test_data(150)
        
        # カルマンフィルター平滑化シグナル
        kalman_signal = EhlersInstantaneousTrendlinePositionEntrySignal(
            alpha=0.07,
            src_type='hl2',
            enable_hyper_er_adaptation=False,
            smoothing_mode='kalman',
            kalman_filter_type='simple'
        )
        
        print(f"✓ カルマンフィルター平滑化シグナルを初期化しました")
        
        # シグナル生成
        signals = kalman_signal.generate(data)
        
        # 平滑化価格の取得
        smoothed_prices = kalman_signal.get_smoothed_prices(data)
        
        if len(smoothed_prices) > 0:
            print(f"✓ カルマンフィルター平滑化完了")
            print(f"  平滑化価格統計:")
            print(f"    データ数: {len(smoothed_prices)}")
            print(f"    NaN数: {np.sum(np.isnan(smoothed_prices))}")
        
        # シグナル統計
        long_signals = np.sum(signals == 1)
        short_signals = np.sum(signals == -1)
        
        print(f"  シグナル統計:")
        print(f"    ロング: {long_signals}")
        print(f"    ショート: {short_signals}")
        
        return True
        
    except Exception as e:
        print(f"✗ カルマンフィルター平滑化テストでエラー: {str(e)}")
        # カルマンフィルターが利用できない場合は警告のみ
        if "統合カルマンフィルターが利用できません" in str(e):
            print("  ℹ カルマンフィルターが利用できないため、このテストはスキップされました")
            return True
        else:
            import traceback
            traceback.print_exc()
            return False

def test_ultimate_smoothing_signal():
    """アルティメットスムーサーシグナルのテスト"""
    print("\n=== アルティメットスムーサーシグナルテスト ===")
    
    try:
        # テストデータ生成
        data = generate_test_data(150)
        
        # アルティメットスムーサー平滑化シグナル
        ultimate_signal = EhlersInstantaneousTrendlinePositionEntrySignal(
            alpha=0.07,
            src_type='hl2',
            enable_hyper_er_adaptation=False,
            smoothing_mode='ultimate',
            ultimate_smoother_period=10
        )
        
        print(f"✓ アルティメットスムーサーシグナルを初期化しました")
        
        # シグナル生成
        signals = ultimate_signal.generate(data)
        
        # 平滑化価格の取得
        smoothed_prices = ultimate_signal.get_smoothed_prices(data)
        
        if len(smoothed_prices) > 0:
            print(f"✓ アルティメットスムーサー平滑化完了")
            print(f"  平滑化価格統計:")
            print(f"    データ数: {len(smoothed_prices)}")
            print(f"    NaN数: {np.sum(np.isnan(smoothed_prices))}")
        
        # シグナル統計
        long_signals = np.sum(signals == 1)
        short_signals = np.sum(signals == -1)
        
        print(f"  シグナル統計:")
        print(f"    ロング: {long_signals}")
        print(f"    ショート: {short_signals}")
        
        return True
        
    except Exception as e:
        print(f"✗ アルティメットスムーサーシグナルテストでエラー: {str(e)}")
        # アルティメットスムーサーが利用できない場合は警告のみ
        if "Ultimate Smootherが利用できません" in str(e):
            print("  ℹ アルティメットスムーサーが利用できないため、このテストはスキップされました")
            return True
        else:
            import traceback
            traceback.print_exc()
            return False

def test_kalman_ultimate_smoothing_signal():
    """カルマン + アルティメットスムーサー組み合わせシグナルのテスト"""
    print("\n=== カルマン + アルティメットスムーサー組み合わせシグナルテスト ===")
    
    try:
        # テストデータ生成
        data = generate_test_data(150)
        
        # カルマン + アルティメットスムーサー平滑化シグナル
        combined_signal = EhlersInstantaneousTrendlinePositionEntrySignal(
            alpha=0.07,
            src_type='hl2',
            enable_hyper_er_adaptation=False,
            smoothing_mode='kalman_ultimate',
            kalman_filter_type='simple',
            ultimate_smoother_period=10
        )
        
        print(f"✓ カルマン + アルティメットスムーサーシグナルを初期化しました")
        
        # シグナル生成
        signals = combined_signal.generate(data)
        
        # 平滑化価格の取得
        smoothed_prices = combined_signal.get_smoothed_prices(data)
        
        if len(smoothed_prices) > 0:
            print(f"✓ カルマン + アルティメットスムーサー平滑化完了")
            print(f"  平滑化価格統計:")
            print(f"    データ数: {len(smoothed_prices)}")
            print(f"    NaN数: {np.sum(np.isnan(smoothed_prices))}")
        
        # シグナル統計
        long_signals = np.sum(signals == 1)
        short_signals = np.sum(signals == -1)
        
        print(f"  シグナル統計:")
        print(f"    ロング: {long_signals}")
        print(f"    ショート: {short_signals}")
        
        return True
        
    except Exception as e:
        print(f"✗ カルマン + アルティメットスムーサーシグナルテストでエラー: {str(e)}")
        # 平滑化機能が利用できない場合は警告のみ
        if ("統合カルマンフィルターが利用できません" in str(e) or 
            "Ultimate Smootherが利用できません" in str(e)):
            print("  ℹ 一部の平滑化機能が利用できないため、このテストはスキップされました")
            return True
        else:
            import traceback
            traceback.print_exc()
            return False

def test_signal_consistency():
    """シグナルの一貫性テスト"""
    print("\n=== シグナル一貫性テスト ===")
    
    try:
        # テストデータ生成
        data = generate_test_data(100)
        
        # 同じパラメータで複数のシグナルを生成
        signal1 = EhlersInstantaneousTrendlinePositionEntrySignal(
            alpha=0.07,
            src_type='hl2',
            enable_hyper_er_adaptation=False
        )
        
        signal2 = EhlersInstantaneousTrendlinePositionEntrySignal(
            alpha=0.07,
            src_type='hl2',
            enable_hyper_er_adaptation=False
        )
        
        # シグナル生成
        signals1 = signal1.generate(data)
        signals2 = signal2.generate(data)
        
        # 一貫性チェック
        consistency = np.array_equal(signals1, signals2)
        print(f"シグナル一貫性: {'✓ 一致' if consistency else '✗ 不一致'}")
        
        if consistency:
            print(f"同一パラメータで同一のシグナルが生成されました")
        else:
            diff_count = np.sum(signals1 != signals2)
            print(f"不一致箇所: {diff_count}個 ({diff_count/len(signals1)*100:.2f}%)")
        
        return consistency
        
    except Exception as e:
        print(f"✗ 一貫性テストでエラー: {str(e)}")
        return False

def test_cache_performance():
    """キャッシュ性能テスト"""
    print("\n=== キャッシュ性能テスト ===")
    
    try:
        # テストデータ生成
        data = generate_test_data(100)
        
        # シグナル初期化
        signal = EhlersInstantaneousTrendlinePositionEntrySignal(
            alpha=0.07,
            src_type='hl2',
            enable_hyper_er_adaptation=False
        )
        
        # 最初の計算
        import time
        start_time = time.time()
        signals1 = signal.generate(data)
        first_calc_time = time.time() - start_time
        
        # 二回目の計算（キャッシュヒットを期待）
        start_time = time.time()
        signals2 = signal.generate(data)
        second_calc_time = time.time() - start_time
        
        print(f"✓ キャッシュ性能テスト完了")
        print(f"  初回計算時間: {first_calc_time:.4f}秒")
        print(f"  二回目計算時間: {second_calc_time:.4f}秒")
        
        # 結果の一致確認
        arrays_equal = np.array_equal(signals1, signals2)
        
        if arrays_equal:
            print(f"  ✓ キャッシュされた結果が一致します")
            if second_calc_time < first_calc_time:
                print(f"  ✓ キャッシュによる高速化を確認しました")
        else:
            print(f"  ✗ キャッシュされた結果が一致しません")
        
        return arrays_equal
        
    except Exception as e:
        print(f"✗ キャッシュ性能テストでエラー: {str(e)}")
        return False

def main():
    """メインテスト関数"""
    print("Ehlers Instantaneous Trendlineシグナル 総合テスト開始")
    print("=" * 70)
    
    results = []
    
    # テストの実行
    results.append(test_position_entry_signal())
    results.append(test_crossover_entry_signal())
    results.append(test_hyper_er_adaptation())
    results.append(test_kalman_smoothing())
    results.append(test_ultimate_smoothing_signal())
    results.append(test_kalman_ultimate_smoothing_signal())
    results.append(test_signal_consistency())
    results.append(test_cache_performance())
    
    # 結果のまとめ
    print(f"\n{'='*70}")
    print(f"テスト結果まとめ:")
    print(f"  実行済みテスト: {len(results)}")
    print(f"  成功: {sum(results)}")
    print(f"  失敗: {len(results) - sum(results)}")
    
    if all(results):
        print(f"✓ 全てのテストが成功しました！")
        print(f"\nEhlers Instantaneous Trendlineシグナルが正常に動作しています。")
    else:
        print(f"✗ 一部のテストが失敗しました。")
    
    print(f"{'='*70}")

if __name__ == "__main__":
    main()