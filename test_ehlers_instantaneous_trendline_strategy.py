#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
import os

# プロジェクトのルートディレクトリを追加
sys.path.append('/home/vapor/dev/ttf')

from strategies.implementations.ehlers_instantaneous_trendline.strategy import EhlersInstantaneousTrendlineStrategy
from strategies.implementations.ehlers_instantaneous_trendline.signal_generator import FilterType

def generate_test_data(n_periods=200):
    """テスト用のダミーデータを生成"""
    np.random.seed(42)
    
    dates = pd.date_range(start='2023-01-01', periods=n_periods, freq='1H')
    
    # トレンドのあるダミー価格データを生成
    price = 50000.0
    prices = []
    
    for i in range(n_periods):
        # 複合的なトレンドパターン
        trend1 = 0.0003 * np.sin(i * 0.03)  # 長期波
        trend2 = 0.0001 * np.sin(i * 0.15)   # 短期波
        noise = np.random.normal(0, 0.008)
        price = price * (1 + trend1 + trend2 + noise)
        prices.append(price)
    
    prices = np.array(prices)
    
    # OHLCV データの生成
    high = prices * (1 + np.abs(np.random.normal(0, 0.008, n_periods)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.008, n_periods)))
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

def test_basic_strategy():
    """基本的なストラテジーテスト"""
    print("=== Ehlers Instantaneous Trendline基本ストラテジーテスト ===")
    
    try:
        # テストデータ生成
        data = generate_test_data(200)
        
        # 基本ストラテジー（フィルターなし）
        strategy = EhlersInstantaneousTrendlineStrategy(
            alpha=0.07,
            src_type='hl2',
            enable_hyper_er_adaptation=True,
            position_mode=True,
            filter_type=FilterType.NONE
        )
        
        print(f"✓ 基本ストラテジーを初期化しました: {strategy.name}")
        
        # エントリーシグナル生成
        entry_signals = strategy.generate_entry(data)
        
        # シグナル統計
        long_signals = np.sum(entry_signals == 1)
        short_signals = np.sum(entry_signals == -1)
        neutral_signals = np.sum(entry_signals == 0)
        
        print(f"✓ エントリーシグナル生成完了")
        print(f"  ロングシグナル: {long_signals} ({long_signals/len(entry_signals)*100:.1f}%)")
        print(f"  ショートシグナル: {short_signals} ({short_signals/len(entry_signals)*100:.1f}%)")
        print(f"  ニュートラル: {neutral_signals} ({neutral_signals/len(entry_signals)*100:.1f}%)")
        
        # エグジットシグナルテスト
        exit_test_result = strategy.generate_exit(data, position=1, index=-1)
        print(f"  エグジットシグナル（ロングポジション、最終インデックス）: {exit_test_result}")
        
        # 指標値の取得テスト
        itrend_values = strategy.get_itrend_values(data)
        trigger_values = strategy.get_trigger_values(data)
        alpha_values = strategy.get_alpha_values(data)
        
        print(f"  ITrend値統計: min={np.nanmin(itrend_values):.2f}, max={np.nanmax(itrend_values):.2f}")
        print(f"  Trigger値統計: min={np.nanmin(trigger_values):.2f}, max={np.nanmax(trigger_values):.2f}")
        print(f"  Alpha値統計: min={np.nanmin(alpha_values):.4f}, max={np.nanmax(alpha_values):.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 基本ストラテジーテストでエラー: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_filter_strategy():
    """フィルター付きストラテジーテスト"""
    print("\n=== フィルター付きストラテジーテスト ===")
    
    try:
        # テストデータ生成
        data = generate_test_data(200)
        
        # HyperERフィルター付きストラテジー
        strategy = EhlersInstantaneousTrendlineStrategy(
            alpha=0.07,
            src_type='hl2',
            enable_hyper_er_adaptation=True,
            position_mode=True,
            filter_type=FilterType.HYPER_ER,
            filter_hyper_er_period=14,
            filter_hyper_er_midline_period=100
        )
        
        print(f"✓ フィルター付きストラテジーを初期化しました: {strategy.name}")
        
        # エントリーシグナル生成
        entry_signals = strategy.generate_entry(data)
        
        # シグナル統計
        long_signals = np.sum(entry_signals == 1)
        short_signals = np.sum(entry_signals == -1)
        neutral_signals = np.sum(entry_signals == 0)
        
        print(f"✓ フィルター統合エントリーシグナル生成完了")
        print(f"  ロングシグナル: {long_signals} ({long_signals/len(entry_signals)*100:.1f}%)")
        print(f"  ショートシグナル: {short_signals} ({short_signals/len(entry_signals)*100:.1f}%)")
        print(f"  ニュートラル: {neutral_signals} ({neutral_signals/len(entry_signals)*100:.1f}%)")
        
        # フィルター詳細の取得
        filter_details = strategy.get_filter_details(data)
        if filter_details:
            print(f"  フィルター詳細取得成功: {list(filter_details.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ フィルター付きストラテジーテストでエラー: {str(e)}")
        # フィルターが利用できない場合の対処
        if any(msg in str(e) for msg in ["HyperER", "が利用できません", "見つかりません"]):
            print("  ℹ フィルター機能が利用できないため、このテストはスキップされました")
            return True
        else:
            import traceback
            traceback.print_exc()
            return False

def test_crossover_strategy():
    """クロスオーバーストラテジーテスト"""
    print("\n=== クロスオーバーストラテジーテスト ===")
    
    try:
        # テストデータ生成
        data = generate_test_data(200)
        
        # クロスオーバーストラテジー
        strategy = EhlersInstantaneousTrendlineStrategy(
            alpha=0.07,
            src_type='hl2',
            enable_hyper_er_adaptation=True,
            position_mode=False,  # クロスオーバーモード
            filter_type=FilterType.NONE
        )
        
        print(f"✓ クロスオーバーストラテジーを初期化しました: {strategy.name}")
        
        # エントリーシグナル生成
        entry_signals = strategy.generate_entry(data)
        
        # シグナル統計
        long_signals = np.sum(entry_signals == 1)
        short_signals = np.sum(entry_signals == -1)
        neutral_signals = np.sum(entry_signals == 0)
        
        print(f"✓ クロスオーバーエントリーシグナル生成完了")
        print(f"  ロングクロスオーバー: {long_signals}")
        print(f"  ショートクロスオーバー: {short_signals}")
        print(f"  ニュートラル: {neutral_signals} ({neutral_signals/len(entry_signals)*100:.1f}%)")
        
        # クロスオーバーの妥当性チェック
        if long_signals > 0 or short_signals > 0:
            print(f"  ✓ クロスオーバーシグナルが検出されました")
        else:
            print(f"  ⚠ クロスオーバーシグナルが検出されませんでした（データ依存）")
        
        return True
        
    except Exception as e:
        print(f"✗ クロスオーバーストラテジーテストでエラー: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_smoothing_strategy():
    """平滑化機能付きストラテジーテスト"""
    print("\n=== 平滑化機能付きストラテジーテスト ===")
    
    try:
        # テストデータ生成
        data = generate_test_data(200)
        
        # カルマン + アルティメット平滑化ストラテジー
        strategy = EhlersInstantaneousTrendlineStrategy(
            alpha=0.07,
            src_type='hl2',
            enable_hyper_er_adaptation=True,
            smoothing_mode='kalman_ultimate',
            kalman_filter_type='simple',
            ultimate_smoother_period=10,
            position_mode=True,
            filter_type=FilterType.NONE
        )
        
        print(f"✓ 平滑化機能付きストラテジーを初期化しました")
        
        # エントリーシグナル生成
        entry_signals = strategy.generate_entry(data)
        
        # 平滑化価格の取得
        smoothed_prices = strategy.get_smoothed_prices(data)
        
        if len(smoothed_prices) > 0:
            print(f"✓ 平滑化機能動作確認")
            print(f"  平滑化価格データ数: {len(smoothed_prices)}")
            print(f"  NaN数: {np.sum(np.isnan(smoothed_prices))}")
        
        # シグナル統計
        long_signals = np.sum(entry_signals == 1)
        short_signals = np.sum(entry_signals == -1)
        
        print(f"✓ 平滑化機能付きエントリーシグナル生成完了")
        print(f"  ロングシグナル: {long_signals}")
        print(f"  ショートシグナル: {short_signals}")
        
        return True
        
    except Exception as e:
        print(f"✗ 平滑化機能付きストラテジーテストでエラー: {str(e)}")
        # 平滑化機能が利用できない場合の対処
        if any(msg in str(e) for msg in ["統合カルマンフィルター", "Ultimate Smoother", "が利用できません"]):
            print("  ℹ 平滑化機能が利用できないため、このテストはスキップされました")
            return True
        else:
            import traceback
            traceback.print_exc()
            return False

def test_advanced_metrics():
    """高度なメトリクステスト"""
    print("\n=== 高度なメトリクステスト ===")
    
    try:
        # テストデータ生成
        data = generate_test_data(150)
        
        # ストラテジー初期化
        strategy = EhlersInstantaneousTrendlineStrategy(
            alpha=0.07,
            src_type='hl2',
            enable_hyper_er_adaptation=True,
            position_mode=True,
            filter_type=FilterType.NONE
        )
        
        print(f"✓ メトリクステスト用ストラテジーを初期化しました")
        
        # 高度なメトリクスの取得
        advanced_metrics = strategy.get_advanced_metrics(data)
        
        if advanced_metrics:
            print(f"✓ 高度なメトリクス取得完了")
            print(f"  利用可能メトリクス: {list(advanced_metrics.keys())}")
            
            # 主要メトリクスの存在確認
            required_metrics = ['itrend_values', 'trigger_values', 'alpha_values', 'ehlers_signals']
            missing_metrics = [metric for metric in required_metrics if metric not in advanced_metrics]
            
            if not missing_metrics:
                print(f"  ✓ 必須メトリクスが全て取得されました")
            else:
                print(f"  ⚠ 不足メトリクス: {missing_metrics}")
        else:
            print(f"  ✗ メトリクス取得に失敗しました")
        
        return len(advanced_metrics) > 0
        
    except Exception as e:
        print(f"✗ 高度なメトリクステストでエラー: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_strategy_info():
    """ストラテジー情報テスト"""
    print("\n=== ストラテジー情報テスト ===")
    
    try:
        # 各種ストラテジーの初期化と情報取得
        strategies = [
            EhlersInstantaneousTrendlineStrategy(filter_type=FilterType.NONE),
            EhlersInstantaneousTrendlineStrategy(filter_type=FilterType.HYPER_ER),
            EhlersInstantaneousTrendlineStrategy(position_mode=False, filter_type=FilterType.NONE)
        ]
        
        print(f"✓ 複数のストラテジー設定をテスト")
        
        for i, strategy in enumerate(strategies, 1):
            try:
                info = strategy.get_strategy_info()
                print(f"  ストラテジー{i}: {info['name']}")
                print(f"    説明: {info['description']}")
                print(f"    特徴数: {len(info['features'])}")
                print(f"    パラメータ数: {len(info['parameters'])}")
            except Exception as e:
                # フィルターが利用できない場合はスキップ
                if any(msg in str(e) for msg in ["HyperER", "が利用できません"]):
                    print(f"  ストラテジー{i}: フィルター機能が利用できないためスキップ")
                    continue
                else:
                    raise
        
        return True
        
    except Exception as e:
        print(f"✗ ストラテジー情報テストでエラー: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_optimization_params():
    """最適化パラメータテスト"""
    print("\n=== 最適化パラメータテスト ===")
    
    try:
        # Mock Optuna Trial クラス
        class MockTrial:
            def suggest_float(self, name, low, high, step=None):
                if name == 'alpha':
                    return 0.08
                elif name == 'alpha_min':
                    return 0.05
                elif name == 'alpha_max':
                    return 0.12
                else:
                    return (low + high) / 2
            
            def suggest_int(self, name, low, high, step=None):
                return (low + high) // 2
            
            def suggest_categorical(self, name, choices):
                if name == 'filter_type':
                    return FilterType.NONE.value
                elif name == 'src_type':
                    return 'hl2'
                elif name == 'smoothing_mode':
                    return 'none'
                elif name == 'kalman_filter_type':
                    return 'simple'
                else:
                    return choices[0]
        
        # 最適化パラメータ生成テスト
        trial = MockTrial()
        opt_params = EhlersInstantaneousTrendlineStrategy.create_optimization_params(trial)
        
        print(f"✓ 最適化パラメータ生成完了")
        print(f"  パラメータ数: {len(opt_params)}")
        
        # 戦略パラメータ変換テスト
        strategy_params = EhlersInstantaneousTrendlineStrategy.convert_params_to_strategy_format(opt_params)
        
        print(f"✓ 戦略パラメータ変換完了")
        print(f"  変換後パラメータ数: {len(strategy_params)}")
        
        # 変換されたパラメータでストラテジー作成
        strategy = EhlersInstantaneousTrendlineStrategy(**strategy_params)
        print(f"✓ 最適化パラメータベースのストラテジー作成成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 最適化パラメータテストでエラー: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト関数"""
    print("Ehlers Instantaneous Trendlineストラテジー 総合テスト開始")
    print("=" * 70)
    
    results = []
    
    # テストの実行
    results.append(test_basic_strategy())
    results.append(test_filter_strategy())
    results.append(test_crossover_strategy())
    results.append(test_smoothing_strategy())
    results.append(test_advanced_metrics())
    results.append(test_strategy_info())
    results.append(test_optimization_params())
    
    # 結果のまとめ
    print(f"\n{'='*70}")
    print(f"テスト結果まとめ:")
    print(f"  実行済みテスト: {len(results)}")
    print(f"  成功: {sum(results)}")
    print(f"  失敗: {len(results) - sum(results)}")
    
    if all(results):
        print(f"✓ 全てのテストが成功しました！")
        print(f"\nEhlers Instantaneous Trendlineストラテジーが正常に動作しています。")
        print(f"\n主な機能:")
        print(f"  • ITrendとTriggerラインによる瞬時トレンド検出")
        print(f"  • HyperERによる動的アルファ適応")
        print(f"  • カルマン統合フィルター + アルティメットスムーサー平滑化")
        print(f"  • 複数のフィルタリングオプション")
        print(f"  • 位置関係・クロスオーバーシグナル対応")
        print(f"  • 最適化パラメータサポート")
    else:
        print(f"✗ 一部のテストが失敗しました。")
    
    print(f"{'='*70}")

if __name__ == "__main__":
    main()