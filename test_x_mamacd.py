#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
X_MAMACDインジケーターのテストスクリプト
"""

import numpy as np
import pandas as pd
import sys
import os

# インジケーターをインポートするためのパス設定
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from indicators.x_mamacd import X_MAMACD, calculate_x_mamacd
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
                'symbol': 'BTCUSDT',
                'interval': '4h',
                'limit': 1000
            }
        }


def create_test_data(length: int = 300) -> pd.DataFrame:
    """テスト用の価格データを生成"""
    np.random.seed(42)
    base_price = 50000.0  # BTCの基準価格
    
    # 複雑な市場データを生成
    prices = [base_price]
    for i in range(1, length):
        if i < 75:  # 上昇トレンド相場
            change = 0.002 + np.random.normal(0, 0.015)
        elif i < 150:  # レンジ相場
            change = np.random.normal(0, 0.012)
        elif i < 225:  # 下降トレンド相場
            change = -0.003 + np.random.normal(0, 0.018)
        else:  # 回復トレンド相場
            change = 0.0025 + np.random.normal(0, 0.014)
        
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


def test_basic_functionality():
    """基本機能のテスト"""
    print("=== 基本機能テスト ===")
    
    # テストデータ生成
    df = create_test_data(200)
    print(f"テストデータ生成: {len(df)}ポイント")
    print(f"価格範囲: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    # 基本版X_MAMACDのテスト
    print("\\n1. 基本版X_MAMACD")
    x_mamacd_basic = X_MAMACD(
        fast_limit=0.5,
        slow_limit=0.05,
        src_type='hlc3',
        signal_period=9,
        use_adaptive_signal=False,
        use_kalman_filter=False,
        use_zero_lag=False
    )
    
    try:
        result = x_mamacd_basic.calculate(df)
        valid_count = np.sum(~np.isnan(result.mamacd))
        
        print(f"  有効値数: {valid_count}/{len(df)}")
        print(f"  MAMACD平均: {np.nanmean(result.mamacd):.6f}")
        print(f"  Signal平均: {np.nanmean(result.signal):.6f}")
        print(f"  Histogram平均: {np.nanmean(result.histogram):.6f}")
        
        # クロスオーバーシグナルのテスト
        bullish_cross, bearish_cross = x_mamacd_basic.get_crossover_signals()
        zero_bull, zero_bear = x_mamacd_basic.get_zero_line_crossover_signals()
        
        print(f"  MAMACDクロスオーバー: 強気{np.sum(bullish_cross)}回, 弱気{np.sum(bearish_cross)}回")
        print(f"  ゼロラインクロス: 強気{np.sum(zero_bull)}回, 弱気{np.sum(zero_bear)}回")
        
        return True
    except Exception as e:
        print(f"  エラー: {e}")
        return False


def test_adaptive_signal():
    """適応型シグナルラインのテスト"""
    print("\\n=== 適応型シグナルラインテスト ===")
    
    df = create_test_data(200)
    
    # 標準EMAシグナル版
    x_mamacd_ema = X_MAMACD(
        fast_limit=0.5,
        slow_limit=0.05,
        src_type='hlc3',
        signal_period=9,
        use_adaptive_signal=False,
        use_zero_lag=True
    )
    
    # 適応型シグナル版
    x_mamacd_adaptive = X_MAMACD(
        fast_limit=0.5,
        slow_limit=0.05,
        src_type='hlc3',
        signal_period=9,
        use_adaptive_signal=True,
        use_zero_lag=True
    )
    
    try:
        result_ema = x_mamacd_ema.calculate(df)
        result_adaptive = x_mamacd_adaptive.calculate(df)
        
        print("標準EMAシグナル:")
        print(f"  Signal平均: {np.nanmean(result_ema.signal):.6f}")
        print(f"  Signal標準偏差: {np.nanstd(result_ema.signal):.6f}")
        
        print("適応型シグナル:")
        print(f"  Signal平均: {np.nanmean(result_adaptive.signal):.6f}")
        print(f"  Signal標準偏差: {np.nanstd(result_adaptive.signal):.6f}")
        
        # シグナル間の相関
        valid_mask = ~(np.isnan(result_ema.signal) | np.isnan(result_adaptive.signal))
        if np.sum(valid_mask) > 10:
            correlation = np.corrcoef(
                result_ema.signal[valid_mask],
                result_adaptive.signal[valid_mask]
            )[0, 1]
            print(f"  シグナル間相関: {correlation:.4f}")
        
        return True
    except Exception as e:
        print(f"エラー: {e}")
        return False


def test_parameter_sensitivity():
    """パラメータ感度のテスト"""
    print("\\n=== パラメータ感度テスト ===")
    
    df = create_test_data(200)
    
    # 異なるfast_limitでテスト
    fast_limits = [0.3, 0.5, 0.7]
    print("\\nfast_limit感度:")
    
    for fast_limit in fast_limits:
        try:
            x_mamacd = X_MAMACD(
                fast_limit=fast_limit,
                slow_limit=0.05,
                src_type='hlc3',
                signal_period=9,
                use_adaptive_signal=True
            )
            result = x_mamacd.calculate(df)
            
            valid_count = np.sum(~np.isnan(result.mamacd))
            mean_mamacd = np.nanmean(result.mamacd)
            std_mamacd = np.nanstd(result.mamacd)
            
            print(f"  fast_limit={fast_limit}: 平均={mean_mamacd:.6f}, 標準偏差={std_mamacd:.6f}, 有効値={valid_count}")
        except Exception as e:
            print(f"  fast_limit={fast_limit}: エラー - {e}")
    
    # 異なるsignal_periodでテスト
    signal_periods = [6, 9, 12]
    print("\\nsignal_period感度:")
    
    for period in signal_periods:
        try:
            x_mamacd = X_MAMACD(
                fast_limit=0.5,
                slow_limit=0.05,
                src_type='hlc3',
                signal_period=period,
                use_adaptive_signal=False
            )
            result = x_mamacd.calculate(df)
            
            # クロスオーバー回数を計算
            bullish_cross, bearish_cross = x_mamacd.get_crossover_signals()
            total_crosses = np.sum(bullish_cross) + np.sum(bearish_cross)
            
            print(f"  signal_period={period}: クロスオーバー{total_crosses}回, Signal平均={np.nanmean(result.signal):.6f}")
        except Exception as e:
            print(f"  signal_period={period}: エラー - {e}")


def test_with_real_data():
    """実データでのテスト"""
    print("\\n=== 実データテスト ===")
    
    config = load_config()
    df = load_real_data(config)
    
    if df is None or len(df) < 50:
        print("実データが利用できません。テストデータを使用します。")
        df = create_test_data(300)
    
    print(f"データポイント数: {len(df)}")
    print(f"価格範囲: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    # フル機能版X_MAMACDのテスト
    x_mamacd_full = X_MAMACD(
        fast_limit=0.5,
        slow_limit=0.05,
        src_type='hlc3',
        signal_period=9,
        use_adaptive_signal=True,
        use_kalman_filter=False,  # カルマンフィルターは今回無効
        use_zero_lag=True
    )
    
    try:
        result = x_mamacd_full.calculate(df)
        
        valid_count = np.sum(~np.isnan(result.mamacd))
        print(f"有効データ: {valid_count}/{len(df)} ({valid_count/len(df)*100:.1f}%)")
        
        # 統計情報
        print(f"\\n統計情報:")
        print(f"  MAMACD - 平均: {np.nanmean(result.mamacd):.6f}, 標準偏差: {np.nanstd(result.mamacd):.6f}")
        print(f"  Signal - 平均: {np.nanmean(result.signal):.6f}, 標準偏差: {np.nanstd(result.signal):.6f}")
        print(f"  Histogram - 平均: {np.nanmean(result.histogram):.6f}, 標準偏差: {np.nanstd(result.histogram):.6f}")
        
        # 最新の値
        if valid_count > 0:
            last_valid_idx = -1
            for i in range(len(result.mamacd)-1, -1, -1):
                if not np.isnan(result.mamacd[i]):
                    last_valid_idx = i
                    break
            
            if last_valid_idx >= 0:
                print(f"\\n最新値:")
                print(f"  MAMACD: {result.mamacd[last_valid_idx]:.6f}")
                print(f"  Signal: {result.signal[last_valid_idx]:.6f}")
                print(f"  Histogram: {result.histogram[last_valid_idx]:.6f}")
        
        # シグナル分析
        bullish_cross, bearish_cross = x_mamacd_full.get_crossover_signals()
        zero_bull, zero_bear = x_mamacd_full.get_zero_line_crossover_signals()
        
        print(f"\\nシグナル分析:")
        print(f"  MAMACDクロスオーバー: 強気{np.sum(bullish_cross)}回, 弱気{np.sum(bearish_cross)}回")
        print(f"  ゼロラインクロス: 強気{np.sum(zero_bull)}回, 弱気{np.sum(zero_bear)}回")
        
        # 最近のシグナル
        recent_signals = []
        signal_window = min(20, len(result.mamacd))
        
        for i in range(max(0, len(result.mamacd) - signal_window), len(result.mamacd)):
            signals = []
            if bullish_cross[i]:
                signals.append("強気クロス")
            if bearish_cross[i]:
                signals.append("弱気クロス")
            if zero_bull[i]:
                signals.append("強気ゼロクロス")
            if zero_bear[i]:
                signals.append("弱気ゼロクロス")
            
            if signals:
                recent_signals.append(f"  インデックス{i}: {', '.join(signals)}")
        
        if recent_signals:
            print(f"\\n最近のシグナル:")
            for signal in recent_signals[-5:]:  # 最新5つまで表示
                print(signal)
        else:
            print(f"\\n最近のシグナル: なし")
        
        return True
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_convenience_function():
    """便利関数のテスト"""
    print("\\n=== 便利関数テスト ===")
    
    df = create_test_data(150)
    
    try:
        # 便利関数を使用
        mamacd, signal, histogram = calculate_x_mamacd(
            df,
            fast_limit=0.5,
            slow_limit=0.05,
            src_type='hlc3',
            signal_period=9,
            use_adaptive_signal=True,
            use_zero_lag=True
        )
        
        valid_count = np.sum(~np.isnan(mamacd))
        print(f"便利関数結果:")
        print(f"  配列形状: MAMACD{mamacd.shape}, Signal{signal.shape}, Histogram{histogram.shape}")
        print(f"  有効値数: {valid_count}/{len(df)}")
        print(f"  MAMACD平均: {np.nanmean(mamacd):.6f}")
        print(f"  Signal平均: {np.nanmean(signal):.6f}")
        print(f"  Histogram平均: {np.nanmean(histogram):.6f}")
        
        return True
    except Exception as e:
        print(f"エラー: {e}")
        return False


def main():
    """メイン実行関数"""
    print("X_MAMACD インジケーター テストスイート")
    print("=" * 50)
    
    test_results = []
    
    # 各テストを実行
    test_results.append(("基本機能", test_basic_functionality()))
    test_results.append(("適応型シグナルライン", test_adaptive_signal()))
    test_results.append(("パラメータ感度", test_parameter_sensitivity()))
    test_results.append(("実データ", test_with_real_data()))
    test_results.append(("便利関数", test_convenience_function()))
    
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