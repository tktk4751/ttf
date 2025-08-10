#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hyper_ERインジケーターのテストスクリプト
"""

import numpy as np
import pandas as pd
import sys
import os

# インジケーターをインポートするためのパス設定
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from indicators.trend_filter.hyper_er import HyperER, calculate_hyper_er
import yaml


def load_config():
    """設定ファイルを読み込む"""
    config_path = 'config.yaml'
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"設定ファイル {config_path} が見つかりません。デフォルト設定を使用します。")
        return {
            'data': {
                'source': 'binance',
                'symbol': 'BTC',
                'interval': '4h',
                'limit': 1000
            }
        }


def create_test_data(length: int = 400) -> pd.DataFrame:
    """テスト用の価格データを生成"""
    np.random.seed(42)
    base_price = 50000.0  # BTCの基準価格
    
    # 複雑な市場データを生成（効率性とレンジの変化）
    prices = [base_price]
    for i in range(1, length):
        if i < 100:  # 効率的トレンド相場
            change = 0.002 + np.random.normal(0, 0.005)
        elif i < 200:  # 非効率的レンジ相場（高ノイズ）
            change = 0.0005 * np.sin(i * 0.15) + np.random.normal(0, 0.012)
        elif i < 300:  # 中程度の効率性トレンド相場
            change = -0.001 + np.random.normal(0, 0.008)
        else:  # 回復トレンド相場（高効率性）
            change = 0.003 + np.random.normal(0, 0.004)
        
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1.0))  # 負の価格を避ける
    
    # OHLC データの生成
    data = []
    for i, close in enumerate(prices):
        daily_range = abs(np.random.normal(0, close * 0.015))
        
        high = close + daily_range * np.random.uniform(0.4, 1.0)
        low = close - daily_range * np.random.uniform(0.4, 1.0)
        
        if i == 0:
            open_price = close
        else:
            gap = np.random.normal(0, close * 0.008)
            open_price = prices[i-1] + gap
        
        # 論理的整合性の確保
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.uniform(100, 1000)
        })
    
    return pd.DataFrame(data)


def analyze_efficiency_performance(values: np.ndarray, trend_signals: np.ndarray, data: pd.DataFrame) -> dict:
    """効率性パフォーマンスを分析する"""
    valid_values = values[~np.isnan(values)]
    valid_signals = trend_signals[~np.isnan(trend_signals)]
    
    if len(valid_values) == 0:
        return {
            'error': 'No valid values',
            'valid_count': 0,
            'total_length': len(values)
        }
    
    # 効率性統計
    mean_efficiency = np.mean(valid_values)
    std_efficiency = np.std(valid_values)
    min_efficiency = np.min(valid_values)
    max_efficiency = np.max(valid_values)
    
    # トレンド信号統計
    trend_periods = np.sum(valid_signals == 1)
    range_periods = np.sum(valid_signals == -1)
    total_signals = len(valid_signals)
    
    # 価格変動分析
    price_changes = []
    if len(data) > 1:
        prices = data['close'].values
        price_changes = np.diff(prices) / prices[:-1] * 100  # パーセント変化
    
    # 効率性レベル分析
    high_efficiency = np.sum(valid_values > 0.7)
    medium_efficiency = np.sum((valid_values >= 0.3) & (valid_values <= 0.7))
    low_efficiency = np.sum(valid_values < 0.3)
    
    return {
        'valid_count': len(valid_values),
        'total_length': len(values),
        'mean_efficiency': mean_efficiency,
        'std_efficiency': std_efficiency,
        'min_efficiency': min_efficiency,
        'max_efficiency': max_efficiency,
        'trend_periods': trend_periods,
        'range_periods': range_periods,
        'trend_ratio': trend_periods / total_signals * 100 if total_signals > 0 else 0,
        'range_ratio': range_periods / total_signals * 100 if total_signals > 0 else 0,
        'high_efficiency_count': high_efficiency,
        'medium_efficiency_count': medium_efficiency,
        'low_efficiency_count': low_efficiency,
        'high_efficiency_ratio': high_efficiency / len(valid_values) * 100,
        'medium_efficiency_ratio': medium_efficiency / len(valid_values) * 100,
        'low_efficiency_ratio': low_efficiency / len(valid_values) * 100,
        'avg_price_change': np.mean(np.abs(price_changes)) if len(price_changes) > 0 else 0,
        'price_volatility': np.std(price_changes) if len(price_changes) > 0 else 0
    }


def test_basic_hyper_er():
    """基本Hyper_ERのテスト"""
    print("=== 基本Hyper_ERテスト ===")
    
    df = create_test_data(200)
    print(f"テストデータ: {len(df)}ポイント")
    print(f"価格範囲: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    # 基本Hyper_ER（ルーフィングフィルター無し）
    hyper_er = HyperER(
        period=14,
        midline_period=50,
        er_period=13,
        er_src_type='hlc3',
        use_roofing_filter=False,
        use_dynamic_period=False,
        use_smoothing=False
    )
    
    try:
        result = hyper_er.calculate(df)
        analysis = analyze_efficiency_performance(result.values, result.trend_signal, df)
        
        print(f"\\n基本Hyper_ER分析:")
        print(f"  有効値数: {analysis['valid_count']}/{analysis['total_length']}")
        print(f"  平均効率性: {analysis['mean_efficiency']:.4f}")
        print(f"  効率性範囲: {analysis['min_efficiency']:.4f} - {analysis['max_efficiency']:.4f}")
        print(f"  標準偏差: {analysis['std_efficiency']:.4f}")
        print(f"  トレンド期間: {analysis['trend_periods']}回 ({analysis['trend_ratio']:.1f}%)")
        print(f"  レンジ期間: {analysis['range_periods']}回 ({analysis['range_ratio']:.1f}%)")
        print(f"  高効率性期間 (>0.7): {analysis['high_efficiency_count']}回 ({analysis['high_efficiency_ratio']:.1f}%)")
        print(f"  中効率性期間 (0.3-0.7): {analysis['medium_efficiency_count']}回 ({analysis['medium_efficiency_ratio']:.1f}%)")
        print(f"  低効率性期間 (<0.3): {analysis['low_efficiency_count']}回 ({analysis['low_efficiency_ratio']:.1f}%)")
        
        return True
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_roofing_filter_hyper_er():
    """ルーフィングフィルター版Hyper_ERのテスト"""
    print("\\n=== ルーフィングフィルター版Hyper_ERテスト ===")
    
    df = create_test_data(300)
    print(f"テストデータ: {len(df)}ポイント")
    
    # ルーフィングフィルター版Hyper_ER
    hyper_er_roofing = HyperER(
        period=14,
        midline_period=50,
        er_period=13,
        er_src_type='hlc3',
        use_roofing_filter=True,
        roofing_hp_cutoff=48.0,
        roofing_ss_band_edge=10.0,
        use_dynamic_period=False,
        use_smoothing=True,
        smoother_type='super_smoother',
        smoother_period=8
    )
    
    try:
        result = hyper_er_roofing.calculate(df)
        analysis = analyze_efficiency_performance(result.values, result.trend_signal, df)
        
        print(f"\\nルーフィングフィルター版Hyper_ER分析:")
        print(f"  有効値数: {analysis['valid_count']}/{analysis['total_length']}")
        
        if 'error' in analysis:
            print(f"  エラー: {analysis['error']}")
            return False
        
        print(f"  平均効率性: {analysis['mean_efficiency']:.4f}")
        print(f"  効率性範囲: {analysis['min_efficiency']:.4f} - {analysis['max_efficiency']:.4f}")
        print(f"  トレンド比率: {analysis['trend_ratio']:.1f}%")
        print(f"  高効率性比率: {analysis['high_efficiency_ratio']:.1f}%")
        
        # ルーフィングフィルター値の統計
        roofing_values = result.roofing_values
        valid_roofing = roofing_values[~np.isnan(roofing_values)]
        if len(valid_roofing) > 0:
            print(f"  ルーフィング値範囲: {np.min(valid_roofing):.6f} - {np.max(valid_roofing):.6f}")
            print(f"  ルーフィング値平均: {np.mean(valid_roofing):.6f}")
        
        return True
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dynamic_period_hyper_er():
    """動的期間版Hyper_ERのテスト"""
    print("\\n=== 動的期間版Hyper_ERテスト ===")
    
    df = create_test_data(400)
    print(f"テストデータ: {len(df)}ポイント")
    
    # 動的期間版Hyper_ER
    hyper_er_dynamic = HyperER(
        period=14,
        midline_period=50,
        er_period=13,
        er_src_type='hlc3',
        use_roofing_filter=True,
        roofing_hp_cutoff=48.0,
        roofing_ss_band_edge=10.0,
        use_dynamic_period=True,
        detector_type='phac_e',
        use_smoothing=True,
        smoother_type='super_smoother'
    )
    
    try:
        result = hyper_er_dynamic.calculate(df)
        analysis = analyze_efficiency_performance(result.values, result.trend_signal, df)
        
        print(f"\\n動的期間版Hyper_ER分析:")
        print(f"  有効値数: {analysis['valid_count']}/{analysis['total_length']}")
        
        if 'error' in analysis:
            print(f"  エラー: {analysis['error']}")
            return False
        
        print(f"  平均効率性: {analysis['mean_efficiency']:.4f}")
        print(f"  トレンド比率: {analysis['trend_ratio']:.1f}%")
        print(f"  高効率性比率: {analysis['high_efficiency_ratio']:.1f}%")
        
        # サイクル期間の統計
        cycle_periods = result.cycle_periods
        valid_cycles = cycle_periods[~np.isnan(cycle_periods) & (cycle_periods > 0)]
        if len(valid_cycles) > 0:
            print(f"  サイクル期間範囲: {np.min(valid_cycles):.1f} - {np.max(valid_cycles):.1f}")
            print(f"  平均サイクル期間: {np.mean(valid_cycles):.1f}")
        
        return True
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_comparison_hyper_er_vs_x_er():
    """Hyper_ER vs X_ERの比較テスト"""
    print("\\n=== Hyper_ER vs X_ER 比較テスト ===")
    
    try:
        from indicators.trend_filter.x_er import XER
    except ImportError:
        print("X_ERが利用できません。比較テストをスキップします。")
        return True
    
    df = create_test_data(300)
    
    try:
        # Hyper_ER（ルーフィングフィルター）
        hyper_er = HyperER(
            period=14,
            er_period=13,
            use_roofing_filter=True,
            use_dynamic_period=False,
            use_smoothing=False
        )
        hyper_result = hyper_er.calculate(df)
        
        # X_ER（カルマンフィルター）
        x_er = XER(
            period=14,
            er_period=13,
            use_kalman_filter=True,
            use_dynamic_period=False,
            use_smoothing=False
        )
        x_result = x_er.calculate(df)
        
        # 比較分析
        hyper_analysis = analyze_efficiency_performance(hyper_result.values, hyper_result.trend_signal, df)
        x_analysis = analyze_efficiency_performance(x_result.values, x_result.trend_signal, df)
        
        print(f"\\n比較結果:")
        print(f"  Hyper_ER平均効率性: {hyper_analysis['mean_efficiency']:.4f}")
        print(f"  X_ER平均効率性: {x_analysis['mean_efficiency']:.4f}")
        print(f"  Hyper_ERトレンド比率: {hyper_analysis['trend_ratio']:.1f}%")
        print(f"  X_ERトレンド比率: {x_analysis['trend_ratio']:.1f}%")
        
        # 相関分析
        hyper_valid = hyper_result.values[~np.isnan(hyper_result.values)]
        x_valid = x_result.values[~np.isnan(x_result.values)]
        
        if len(hyper_valid) > 0 and len(x_valid) > 0:
            min_len = min(len(hyper_valid), len(x_valid))
            if min_len > 10:
                correlation = np.corrcoef(
                    hyper_valid[-min_len:],
                    x_valid[-min_len:]
                )[0, 1]
                print(f"  Hyper_ER vs X_ER 相関: {correlation:.4f}")
        
        return True
    except Exception as e:
        print(f"エラー: {e}")
        return False


def test_convenience_function():
    """便利関数のテスト"""
    print("\\n=== 便利関数テスト ===")
    
    df = create_test_data(150)
    
    try:
        # 便利関数を使用
        hyper_er_values = calculate_hyper_er(
            df,
            period=14,
            er_period=13,
            use_roofing_filter=True,
            use_smoothing=True,
            smoother_type='super_smoother'
        )
        
        valid_count = np.sum(~np.isnan(hyper_er_values))
        mean_value = np.nanmean(hyper_er_values)
        
        print(f"便利関数結果:")
        print(f"  有効値数: {valid_count}/{len(df)}")
        print(f"  平均値: {mean_value:.4f}")
        print(f"  値域: {np.nanmin(hyper_er_values):.4f} - {np.nanmax(hyper_er_values):.4f}")
        
        return True
    except Exception as e:
        print(f"エラー: {e}")
        return False


def test_percentile_analysis():
    """パーセンタイル分析のテスト"""
    print("\\n=== パーセンタイル分析テスト ===")
    
    df = create_test_data(250)
    
    # パーセンタイル分析を有効にしたHyper_ER
    hyper_er = HyperER(
        period=14,
        er_period=13,
        use_roofing_filter=True,
        enable_percentile_analysis=True,
        percentile_lookback_period=50,
        percentile_low_threshold=0.25,
        percentile_high_threshold=0.75
    )
    
    try:
        result = hyper_er.calculate(df)
        
        print(f"パーセンタイル分析結果:")
        
        # パーセンタイル値
        percentiles = result.percentiles
        if percentiles is not None:
            valid_percentiles = percentiles[~np.isnan(percentiles)]
            print(f"  パーセンタイル値数: {len(valid_percentiles)}")
            if len(valid_percentiles) > 0:
                print(f"  パーセンタイル範囲: {np.min(valid_percentiles):.4f} - {np.max(valid_percentiles):.4f}")
        
        # トレンド状態
        trend_state = result.trend_state
        if trend_state is not None:
            valid_states = trend_state[~np.isnan(trend_state)]
            if len(valid_states) > 0:
                trend_count = np.sum(valid_states == 1)
                neutral_count = np.sum(valid_states == 0)
                range_count = np.sum(valid_states == -1)
                print(f"  トレンド状態: トレンド={trend_count}, 中立={neutral_count}, レンジ={range_count}")
        
        # トレンド強度
        trend_intensity = result.trend_intensity
        if trend_intensity is not None:
            valid_intensity = trend_intensity[~np.isnan(trend_intensity)]
            if len(valid_intensity) > 0:
                mean_intensity = np.mean(valid_intensity)
                print(f"  平均トレンド強度: {mean_intensity:.4f}")
        
        return True
    except Exception as e:
        print(f"エラー: {e}")
        return False


def main():
    """メイン実行関数"""
    print("Hyper_ER インジケーター テストスイート")
    print("=" * 50)
    
    test_results = []
    
    # 各テストを実行
    test_results.append(("基本Hyper_ER", test_basic_hyper_er()))
    test_results.append(("ルーフィングフィルター版", test_roofing_filter_hyper_er()))
    test_results.append(("動的期間版", test_dynamic_period_hyper_er()))
    test_results.append(("比較テスト", test_comparison_hyper_er_vs_x_er()))
    test_results.append(("便利関数", test_convenience_function()))
    test_results.append(("パーセンタイル分析", test_percentile_analysis()))
    
    # 結果サマリー
    print("\\n" + "=" * 50)
    print("テスト結果サマリー:")
    print("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\\n合計: {passed}/{total} テストパス ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\\n✅ すべてのテストが成功しました！")
        return 0
    else:
        print(f"\\n❌ {total - passed}個のテストが失敗しました。")
        return 1


if __name__ == "__main__":
    sys.exit(main())