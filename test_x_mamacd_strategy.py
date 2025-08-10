#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
X_MAMACDストラテジーのテストスクリプト
"""

import numpy as np
import pandas as pd
import sys
import os

# ストラテジーをインポートするためのパス設定
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from strategies.implementations.x_mamacd import XMAMACDStrategy, XMAMACDSignalGenerator
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


def create_test_data(length: int = 400) -> pd.DataFrame:
    """テスト用の価格データを生成"""
    np.random.seed(42)
    base_price = 50000.0  # BTCの基準価格
    
    # 複雑な市場データを生成（明確なトレンドとサイクルを含む）
    prices = [base_price]
    for i in range(1, length):
        if i < 100:  # 上昇トレンド相場
            change = 0.003 + np.random.normal(0, 0.015)
        elif i < 200:  # レンジ相場
            change = 0.0005 * np.sin(i * 0.1) + np.random.normal(0, 0.012)  # サイクリックな動き
        elif i < 300:  # 下降トレンド相場
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


def analyze_strategy_performance(entry_signals: np.ndarray, data: pd.DataFrame) -> dict:
    """ストラテジーのパフォーマンスを分析する"""
    total_signals = len(entry_signals)
    long_signals = np.sum(entry_signals == 1)
    short_signals = np.sum(entry_signals == -1)
    no_signals = np.sum(entry_signals == 0)
    
    # シグナルの発生位置を取得
    long_positions = np.where(entry_signals == 1)[0]
    short_positions = np.where(entry_signals == -1)[0]
    
    # 価格変動の分析
    price_changes = []
    if len(data) > 1:
        prices = data['close'].values
        price_changes = np.diff(prices) / prices[:-1] * 100  # パーセント変化
    
    return {
        'total_length': total_signals,
        'long_signals': long_signals,
        'short_signals': short_signals,
        'no_signals': no_signals,
        'long_ratio': long_signals / total_signals * 100 if total_signals > 0 else 0,
        'short_ratio': short_signals / total_signals * 100 if total_signals > 0 else 0,
        'signal_ratio': (long_signals + short_signals) / total_signals * 100 if total_signals > 0 else 0,
        'long_positions': long_positions[-5:] if len(long_positions) > 0 else [],  # 最新5つ
        'short_positions': short_positions[-5:] if len(short_positions) > 0 else [],  # 最新5つ
        'avg_price_change': np.mean(np.abs(price_changes)) if len(price_changes) > 0 else 0,
        'price_volatility': np.std(price_changes) if len(price_changes) > 0 else 0
    }


def test_basic_strategy():
    """基本ストラテジーのテスト"""
    print("=== 基本X_MAMACDストラテジーテスト ===")
    
    df = create_test_data(200)
    print(f"テストデータ: {len(df)}ポイント")
    print(f"価格範囲: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    # 基本クロスオーバーストラテジー
    strategy = XMAMACDStrategy(
        fast_limit=0.5,
        slow_limit=0.05,
        src_type='hlc3',
        signal_period=9,
        use_adaptive_signal=True,
        signal_mode='crossover',
        use_zero_lag=True
    )
    
    try:
        entry_signals = strategy.generate_entry(df)
        analysis = analyze_strategy_performance(entry_signals, df)
        
        print(f"\\nストラテジー分析:")
        print(f"  総シグナル数: {analysis['total_length']}")
        print(f"  ロングシグナル: {analysis['long_signals']}回 ({analysis['long_ratio']:.1f}%)")
        print(f"  ショートシグナル: {analysis['short_signals']}回 ({analysis['short_ratio']:.1f}%)")
        print(f"  シグナル発生率: {analysis['signal_ratio']:.1f}%")
        
        if analysis['long_positions'].size > 0:
            print(f"  最新ロング位置: {analysis['long_positions']}")
        if analysis['short_positions'].size > 0:
            print(f"  最新ショート位置: {analysis['short_positions']}")
            
        # MAMACD値も取得してみる
        mamacd_values = strategy.get_all_mamacd_values(df)
        
        print(f"\\nMAMACD統計:")
        print(f"  MAMACD平均: {np.nanmean(mamacd_values['mamacd']):.6f}")
        print(f"  Signal平均: {np.nanmean(mamacd_values['signal']):.6f}")
        print(f"  Histogram平均: {np.nanmean(mamacd_values['histogram']):.6f}")
        
        # エグジットテスト
        test_exit_signals(strategy, df, entry_signals)
        
        return True
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_exit_signals(strategy, data, entry_signals):
    """エグジットシグナルのテスト"""
    print(f"\\nエグジットシグナルテスト:")
    
    # ロングポジションのエグジットテスト
    long_positions = np.where(entry_signals == 1)[0]
    long_exits = 0
    
    for pos_index in long_positions[:3]:  # 最初の3つのロングポジションをテスト
        # ポジション後の数バーでエグジット判定
        for i in range(pos_index + 1, min(pos_index + 10, len(data))):
            if strategy.generate_exit(data, 1, i):  # ロングポジション
                long_exits += 1
                break
    
    # ショートポジションのエグジットテスト
    short_positions = np.where(entry_signals == -1)[0]
    short_exits = 0
    
    for pos_index in short_positions[:3]:  # 最初の3つのショートポジションをテスト
        # ポジション後の数バーでエグジット判定
        for i in range(pos_index + 1, min(pos_index + 10, len(data))):
            if strategy.generate_exit(data, -1, i):  # ショートポジション
                short_exits += 1
                break
    
    print(f"  ロングエグジット: {long_exits}/{min(len(long_positions), 3)}")
    print(f"  ショートエグジット: {short_exits}/{min(len(short_positions), 3)}")


def test_signal_modes():
    """シグナルモードのテスト"""
    print("\\n=== シグナルモード比較テスト ===")
    
    df = create_test_data(300)
    
    # 3つのシグナルモードをテスト
    modes = ['crossover', 'zero_line', 'trend_follow']
    results = []
    
    for mode in modes:
        try:
            strategy = XMAMACDStrategy(
                fast_limit=0.5,
                slow_limit=0.05,
                signal_mode=mode,
                use_adaptive_signal=True,
                use_zero_lag=True
            )
            
            entry_signals = strategy.generate_entry(df)
            analysis = analyze_strategy_performance(entry_signals, df)
            results.append((mode, analysis))
            
        except Exception as e:
            print(f"  {mode}モードでエラー: {e}")
            results.append((mode, None))
    
    print("シグナルモード比較結果:")
    print(f"{'モード名':<15} {'ロング':<8} {'ショート':<8} {'発生率':<8}")
    print("-" * 50)
    
    for mode, analysis in results:
        if analysis:
            print(f"{mode:<15} {analysis['long_signals']:<8} {analysis['short_signals']:<8} {analysis['signal_ratio']:<8.1f}%")
        else:
            print(f"{mode:<15} {'エラー':<8} {'エラー':<8} {'エラー':<8}")
    
    return len([r for r in results if r[1] is not None]) == len(modes)


def test_signal_generator():
    """シグナルジェネレーターのテスト"""
    print("\\n=== シグナルジェネレーターテスト ===")
    
    df = create_test_data(200)
    
    try:
        # シグナルジェネレーターの直接テスト
        signal_gen = XMAMACDSignalGenerator(
            fast_limit=0.5,
            slow_limit=0.05,
            signal_mode='crossover',
            use_adaptive_signal=True
        )
        
        entry_signals = signal_gen.get_entry_signals(df)
        advanced_metrics = signal_gen.get_advanced_metrics(df)
        
        print(f"シグナルジェネレーター結果:")
        print(f"  エントリーシグナル数: {np.sum(entry_signals != 0)}")
        print(f"  利用可能メトリクス: {list(advanced_metrics.keys())}")
        
        # 各メトリクスの統計
        for metric_name, values in advanced_metrics.items():
            if len(values) > 0:
                print(f"  {metric_name}平均: {np.nanmean(values):.6f}")
        
        # エグジットシグナルのテスト
        long_positions = np.where(entry_signals == 1)[0]
        if len(long_positions) > 0:
            # 最初のロングポジションの数バー後でエグジット判定
            pos_index = long_positions[0]
            for i in range(pos_index + 1, min(pos_index + 10, len(df))):
                if signal_gen.get_exit_signals(df, 1, i):
                    print(f"  最初のロングポジション（{pos_index}）は{i}でエグジット")
                    break
        
        return True
    except Exception as e:
        print(f"エラー: {e}")
        return False


def test_parameter_optimization():
    """パラメータ最適化のテスト"""
    print("\\n=== パラメータ最適化テスト ===")
    
    try:
        import optuna
        
        # ダミートライアルを作成
        study = optuna.create_study()
        trial = study.ask()
        
        # 最適化パラメータの生成
        opt_params = XMAMACDStrategy.create_optimization_params(trial)
        strategy_params = XMAMACDStrategy.convert_params_to_strategy_format(opt_params)
        
        print(f"最適化パラメータ生成:")
        print(f"  生成パラメータ数: {len(opt_params)}")
        print(f"  変換パラメータ数: {len(strategy_params)}")
        
        # パラメータを使用してストラテジーを作成
        strategy = XMAMACDStrategy(**strategy_params)
        strategy_info = strategy.get_strategy_info()
        
        print(f"  ストラテジー名: {strategy_info['name']}")
        print(f"  機能数: {len(strategy_info['features'])}")
        print(f"  シグナルモード数: {len(strategy_info['signal_modes'])}")
        
        return True
    except ImportError:
        print("Optunaが利用できません。パラメータ最適化テストをスキップします。")
        return True
    except Exception as e:
        print(f"エラー: {e}")
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
        # フル機能版ストラテジーのテスト
        strategy = XMAMACDStrategy(
            fast_limit=0.5,
            slow_limit=0.05,
            src_type='hlc3',
            signal_period=9,
            use_adaptive_signal=True,
            signal_mode='crossover',
            use_zero_lag=True
        )
        
        entry_signals = strategy.generate_entry(df)
        analysis = analyze_strategy_performance(entry_signals, df)
        
        print(f"\\n実データストラテジー分析:")
        print(f"  ロングシグナル: {analysis['long_signals']}回")
        print(f"  ショートシグナル: {analysis['short_signals']}回")
        print(f"  シグナル発生率: {analysis['signal_ratio']:.1f}%")
        print(f"  平均価格変動: {analysis['avg_price_change']:.2f}%")
        print(f"  価格ボラティリティ: {analysis['price_volatility']:.2f}%")
        
        # MAMACD統計
        mamacd_values = strategy.get_all_mamacd_values(df)
        print(f"\\nMAMACD統計:")
        print(f"  MAMACD範囲: {np.nanmin(mamacd_values['mamacd']):.2f} - {np.nanmax(mamacd_values['mamacd']):.2f}")
        print(f"  Signal範囲: {np.nanmin(mamacd_values['signal']):.2f} - {np.nanmax(mamacd_values['signal']):.2f}")
        print(f"  Histogram範囲: {np.nanmin(mamacd_values['histogram']):.2f} - {np.nanmax(mamacd_values['histogram']):.2f}")
        
        # 最新のシグナル位置を詳細表示
        if analysis['long_positions'].size > 0:
            print(f"  最新ロングシグナル位置: {analysis['long_positions']}")
        if analysis['short_positions'].size > 0:
            print(f"  最新ショートシグナル位置: {analysis['short_positions']}")
        
        return True
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """メイン実行関数"""
    print("X_MAMACD ストラテジー テストスイート")
    print("=" * 50)
    
    test_results = []
    
    # 各テストを実行
    test_results.append(("基本ストラテジー", test_basic_strategy()))
    test_results.append(("シグナルモード", test_signal_modes()))
    test_results.append(("シグナルジェネレーター", test_signal_generator()))
    test_results.append(("パラメータ最適化", test_parameter_optimization()))
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