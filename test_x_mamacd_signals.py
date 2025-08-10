#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
X_MAMACDシグナルのテストスクリプト
"""

import numpy as np
import pandas as pd
import sys
import os

# シグナルをインポートするためのパス設定
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from signals.implementations.x_mamacd import (
    XMAMACDCrossoverEntrySignal,
    XMAMACDZeroLineEntrySignal,
    XMAMACDTrendFollowEntrySignal
)
from data.binance_data_source import BinanceDataSource
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


def create_test_data(length: int = 300) -> pd.DataFrame:
    """テスト用の価格データを生成"""
    np.random.seed(42)
    base_price = 50000.0  # BTCの基準価格
    
    # 複雑な市場データを生成（明確なトレンドとサイクルを含む）
    prices = [base_price]
    for i in range(1, length):
        if i < 75:  # 上昇トレンド相場
            change = 0.003 + np.random.normal(0, 0.015)
        elif i < 150:  # レンジ相場
            change = 0.0005 * np.sin(i * 0.1) + np.random.normal(0, 0.012)  # サイクリックな動き
        elif i < 225:  # 下降トレンド相場
            change = -0.004 + np.random.normal(0, 0.018)
        else:  # 回復トレンド相場
            change = 0.0035 + np.random.normal(0, 0.014)
        
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1.0))  # 負の価格を避ける
    
    # OHLC データの生成
    data = []
    for i, close in enumerate(prices):
        daily_range = abs(np.random.normal(0, close * 0.02))
        
        high = close + daily_range * np.random.uniform(0.3, 1.0)
        low = close - daily_range * np.random.uniform(0.3, 1.0)
        
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


def load_real_data(config: dict) -> pd.DataFrame:
    """実際の市場データを読み込む"""
    try:
        data_config = config.get('data', {})
        source = data_config.get('source', 'binance')
        
        if source == 'binance':
            symbol = data_config.get('symbol', 'BTC')
            interval = data_config.get('interval', '4h')
            
            data_source = BinanceDataSource()
            df = data_source.load_data(
                symbol=symbol,
                timeframe=interval
            )
            print(f"実データを読み込みました: {symbol} {interval} {len(df)}ポイント")
            return df
        else:
            print(f"未対応のデータソース: {source}")
            return None
    except Exception as e:
        print(f"実データの読み込みに失敗: {e}")
        return None


def analyze_signals(signals: np.ndarray, signal_name: str) -> dict:
    """シグナルを分析する"""
    total_signals = len(signals)
    long_signals = np.sum(signals == 1)
    short_signals = np.sum(signals == -1)
    no_signals = np.sum(signals == 0)
    
    # シグナルの発生位置を取得
    long_positions = np.where(signals == 1)[0]
    short_positions = np.where(signals == -1)[0]
    
    return {
        'signal_name': signal_name,
        'total_length': total_signals,
        'long_signals': long_signals,
        'short_signals': short_signals,
        'no_signals': no_signals,
        'long_ratio': long_signals / total_signals * 100 if total_signals > 0 else 0,
        'short_ratio': short_signals / total_signals * 100 if total_signals > 0 else 0,
        'signal_ratio': (long_signals + short_signals) / total_signals * 100 if total_signals > 0 else 0,
        'long_positions': long_positions[-5:] if len(long_positions) > 0 else [],  # 最新5つ
        'short_positions': short_positions[-5:] if len(short_positions) > 0 else []  # 最新5つ
    }


def test_crossover_signal():
    """クロスオーバーシグナルのテスト"""
    print("=== XMAMACDクロスオーバーシグナルテスト ===")
    
    df = create_test_data(200)
    print(f"テストデータ: {len(df)}ポイント")
    print(f"価格範囲: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    # 基本クロスオーバーシグナル
    crossover_signal = XMAMACDCrossoverEntrySignal(
        fast_limit=0.5,
        slow_limit=0.05,
        src_type='hlc3',
        signal_period=9,
        use_adaptive_signal=True,
        use_zero_lag=True
    )
    
    try:
        signals = crossover_signal.generate(df)
        analysis = analyze_signals(signals, "MAMACDクロスオーバー")
        
        print(f"\\nシグナル分析:")
        print(f"  総シグナル数: {analysis['total_length']}")
        print(f"  ロングシグナル: {analysis['long_signals']}回 ({analysis['long_ratio']:.1f}%)")
        print(f"  ショートシグナル: {analysis['short_signals']}回 ({analysis['short_ratio']:.1f}%)")
        print(f"  シグナル発生率: {analysis['signal_ratio']:.1f}%")
        
        if analysis['long_positions'].size > 0:
            print(f"  最新ロング位置: {analysis['long_positions']}")
        if analysis['short_positions'].size > 0:
            print(f"  最新ショート位置: {analysis['short_positions']}")
            
        # MAMACD値も取得してみる
        mamacd_values = crossover_signal.get_mamacd_values(df)
        signal_values = crossover_signal.get_signal_values(df)
        histogram_values = crossover_signal.get_histogram_values(df)
        
        print(f"\\nMAMACD統計:")
        print(f"  MAMACD平均: {np.nanmean(mamacd_values):.6f}")
        print(f"  Signal平均: {np.nanmean(signal_values):.6f}")
        print(f"  Histogram平均: {np.nanmean(histogram_values):.6f}")
        
        return True
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_zero_line_signal():
    """ゼロラインクロスシグナルのテスト"""
    print("\\n=== XMAMACDゼロラインクロスシグナルテスト ===")
    
    df = create_test_data(200)
    
    zero_line_signal = XMAMACDZeroLineEntrySignal(
        fast_limit=0.5,
        slow_limit=0.05,
        src_type='hlc3',
        signal_period=9,
        use_adaptive_signal=True,
        use_zero_lag=True
    )
    
    try:
        signals = zero_line_signal.generate(df)
        analysis = analyze_signals(signals, "MAMACDゼロラインクロス")
        
        print(f"シグナル分析:")
        print(f"  総シグナル数: {analysis['total_length']}")
        print(f"  ロングシグナル: {analysis['long_signals']}回 ({analysis['long_ratio']:.1f}%)")
        print(f"  ショートシグナル: {analysis['short_signals']}回 ({analysis['short_ratio']:.1f}%)")
        print(f"  シグナル発生率: {analysis['signal_ratio']:.1f}%")
        
        if analysis['long_positions'].size > 0:
            print(f"  最新ロング位置: {analysis['long_positions']}")
        if analysis['short_positions'].size > 0:
            print(f"  最新ショート位置: {analysis['short_positions']}")
        
        return True
    except Exception as e:
        print(f"エラー: {e}")
        return False


def test_trend_follow_signal():
    """トレンドフォローシグナルのテスト"""
    print("\\n=== XMAMACDトレンドフォローシグナルテスト ===")
    
    df = create_test_data(200)
    
    # 基本トレンドフォローシグナル
    trend_signal = XMAMACDTrendFollowEntrySignal(
        fast_limit=0.5,
        slow_limit=0.05,
        src_type='hlc3',
        signal_period=9,
        use_adaptive_signal=True,
        trend_threshold=0.0,
        momentum_mode=False,
        use_zero_lag=True
    )
    
    try:
        signals = trend_signal.generate(df)
        analysis = analyze_signals(signals, "MAMACDトレンドフォロー")
        
        print(f"基本トレンドフォロー:")
        print(f"  ロングシグナル: {analysis['long_signals']}回 ({analysis['long_ratio']:.1f}%)")
        print(f"  ショートシグナル: {analysis['short_signals']}回 ({analysis['short_ratio']:.1f}%)")
        print(f"  シグナル発生率: {analysis['signal_ratio']:.1f}%")
        
        # モメンタムモードもテスト
        momentum_signal = XMAMACDTrendFollowEntrySignal(
            fast_limit=0.5,
            slow_limit=0.05,
            src_type='hlc3',
            signal_period=9,
            use_adaptive_signal=True,
            trend_threshold=0.0,
            momentum_mode=True,
            momentum_lookback=3,
            use_zero_lag=True
        )
        
        momentum_signals = momentum_signal.generate(df)
        momentum_analysis = analyze_signals(momentum_signals, "MAMACDモメンタム")
        
        print(f"\\nモメンタムモード:")
        print(f"  ロングシグナル: {momentum_analysis['long_signals']}回 ({momentum_analysis['long_ratio']:.1f}%)")
        print(f"  ショートシグナル: {momentum_analysis['short_signals']}回 ({momentum_analysis['short_ratio']:.1f}%)")
        print(f"  シグナル発生率: {momentum_analysis['signal_ratio']:.1f}%")
        
        return True
    except Exception as e:
        print(f"エラー: {e}")
        return False


def test_signal_comparison():
    """シグナル比較テスト"""
    print("\\n=== シグナル比較テスト ===")
    
    df = create_test_data(300)
    
    # 3つのシグナルを生成
    signals_list = []
    
    try:
        # 1. クロスオーバーシグナル
        crossover_signal = XMAMACDCrossoverEntrySignal()
        crossover_signals = crossover_signal.generate(df)
        signals_list.append(('クロスオーバー', crossover_signals))
        
        # 2. ゼロラインシグナル
        zero_line_signal = XMAMACDZeroLineEntrySignal()
        zero_line_signals = zero_line_signal.generate(df)
        signals_list.append(('ゼロラインクロス', zero_line_signals))
        
        # 3. トレンドフォローシグナル
        trend_signal = XMAMACDTrendFollowEntrySignal()
        trend_signals = trend_signal.generate(df)
        signals_list.append(('トレンドフォロー', trend_signals))
        
        print("シグナル比較結果:")
        print(f"{'シグナル名':<15} {'ロング':<8} {'ショート':<8} {'発生率':<8}")
        print("-" * 50)
        
        for signal_name, signals in signals_list:
            analysis = analyze_signals(signals, signal_name)
            print(f"{signal_name:<15} {analysis['long_signals']:<8} {analysis['short_signals']:<8} {analysis['signal_ratio']:<8.1f}%")
        
        # シグナル間の相関を計算
        if len(signals_list) >= 2:
            print("\\nシグナル間相関:")
            for i in range(len(signals_list)):
                for j in range(i+1, len(signals_list)):
                    name1, signals1 = signals_list[i]
                    name2, signals2 = signals_list[j]
                    
                    # ピアソン相関係数を計算
                    correlation = np.corrcoef(signals1, signals2)[0, 1]
                    print(f"  {name1} vs {name2}: {correlation:.3f}")
        
        return True
    except Exception as e:
        print(f"エラー: {e}")
        return False


def test_parameter_sensitivity():
    """パラメータ感度テスト"""
    print("\\n=== パラメータ感度テスト ===")
    
    df = create_test_data(200)
    
    try:
        # 異なるfast_limitでテスト
        fast_limits = [0.3, 0.5, 0.7]
        print("\\nfast_limit感度（クロスオーバーシグナル）:")
        
        for fast_limit in fast_limits:
            try:
                signal = XMAMACDCrossoverEntrySignal(
                    fast_limit=fast_limit,
                    slow_limit=0.05
                )
                signals = signal.generate(df)
                analysis = analyze_signals(signals, f"fast_limit={fast_limit}")
                
                print(f"  fast_limit={fast_limit}: ロング{analysis['long_signals']}回, ショート{analysis['short_signals']}回, 発生率{analysis['signal_ratio']:.1f}%")
            except Exception as e:
                print(f"  fast_limit={fast_limit}: エラー - {e}")
        
        # 異なるsignal_periodでテスト
        signal_periods = [6, 9, 12]
        print("\\nsignal_period感度（クロスオーバーシグナル）:")
        
        for period in signal_periods:
            try:
                signal = XMAMACDCrossoverEntrySignal(
                    signal_period=period
                )
                signals = signal.generate(df)
                analysis = analyze_signals(signals, f"signal_period={period}")
                
                print(f"  signal_period={period}: ロング{analysis['long_signals']}回, ショート{analysis['short_signals']}回, 発生率{analysis['signal_ratio']:.1f}%")
            except Exception as e:
                print(f"  signal_period={period}: エラー - {e}")
        
        return True
    except Exception as e:
        print(f"パラメータ感度テストでエラー: {e}")
        return False


def test_with_real_data():
    """実データでのテスト"""
    print("\\n=== 実データテスト ===")
    
    config = load_config()
    df = load_real_data(config)
    
    if df is None or len(df) < 50:
        print("実データが利用できません。テストデータを使用します。")
        df = create_test_data(500)
    
    print(f"データポイント数: {len(df)}")
    print(f"価格範囲: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    try:
        # フル機能版シグナルのテスト
        full_signal = XMAMACDCrossoverEntrySignal(
            fast_limit=0.5,
            slow_limit=0.05,
            src_type='hlc3',
            signal_period=9,
            use_adaptive_signal=True,
            use_zero_lag=True
        )
        
        signals = full_signal.generate(df)
        analysis = analyze_signals(signals, "実データクロスオーバー")
        
        print(f"\\n実データシグナル分析:")
        print(f"  ロングシグナル: {analysis['long_signals']}回")
        print(f"  ショートシグナル: {analysis['short_signals']}回")
        print(f"  シグナル発生率: {analysis['signal_ratio']:.1f}%")
        
        # 最新のシグナル位置を詳細表示
        if analysis['long_positions'].size > 0:
            print(f"  最新ロングシグナル位置: {analysis['long_positions']}")
        if analysis['short_positions'].size > 0:
            print(f"  最新ショートシグナル位置: {analysis['short_positions']}")
        
        # シグナルの品質評価（連続シグナルの確認）
        consecutive_analysis = analyze_consecutive_signals(signals)
        print(f"\\nシグナル品質:")
        print(f"  平均シグナル間隔: {consecutive_analysis['avg_interval']:.1f}バー")
        print(f"  最大連続期間: {consecutive_analysis['max_consecutive']}バー")
        
        return True
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def analyze_consecutive_signals(signals: np.ndarray) -> dict:
    """連続シグナルを分析する"""
    signal_positions = np.where(signals != 0)[0]
    
    if len(signal_positions) == 0:
        return {'avg_interval': 0, 'max_consecutive': 0}
    
    # シグナル間隔の計算
    intervals = []
    if len(signal_positions) > 1:
        intervals = np.diff(signal_positions)
    
    # 連続シグナル期間の計算
    consecutive_periods = []
    current_period = 1
    
    for i in range(1, len(signals)):
        if signals[i] != 0 and signals[i] == signals[i-1]:
            current_period += 1
        else:
            if current_period > 1:
                consecutive_periods.append(current_period)
            current_period = 1
    
    if current_period > 1:
        consecutive_periods.append(current_period)
    
    return {
        'avg_interval': np.mean(intervals) if len(intervals) > 0 else 0,
        'max_consecutive': max(consecutive_periods) if consecutive_periods else 1
    }


def main():
    """メイン実行関数"""
    print("X_MAMACD シグナル テストスイート")
    print("=" * 50)
    
    test_results = []
    
    # 各テストを実行
    test_results.append(("クロスオーバーシグナル", test_crossover_signal()))
    test_results.append(("ゼロラインクロスシグナル", test_zero_line_signal()))
    test_results.append(("トレンドフォローシグナル", test_trend_follow_signal()))
    test_results.append(("シグナル比較", test_signal_comparison()))
    test_results.append(("パラメータ感度", test_parameter_sensitivity()))
    test_results.append(("実データ", test_with_real_data()))
    
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