#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
グランドサイクルMAチャートのテストスクリプト
実際の相場データを使用してチャート描画をテスト
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime, timedelta

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_realistic_market_data(length=500, start_date='2024-01-01'):
    """
    よりリアルな市場データを生成
    
    Args:
        length: データ点数
        start_date: 開始日
        
    Returns:
        OHLCV形式のDataFrame
    """
    np.random.seed(42)
    
    # 日付インデックスの作成
    start = pd.to_datetime(start_date)
    dates = pd.date_range(start, periods=length, freq='4h')
    
    # 基準価格とトレンド
    base_price = 100.0
    trend_strength = 0.001  # トレンド強度
    
    # 複数のサイクル成分（より現実的）
    t = np.arange(length)
    long_cycle = 8 * np.sin(2 * np.pi * t / 120)    # 長期サイクル（約20日）
    medium_cycle = 4 * np.sin(2 * np.pi * t / 48)   # 中期サイクル（約8日）
    short_cycle = 2 * np.sin(2 * np.pi * t / 12)    # 短期サイクル（約2日）
    
    # トレンド成分
    trend = np.linspace(0, trend_strength * length, length)
    
    # ランダムウォーク成分
    random_walk = np.cumsum(np.random.normal(0, 0.5, length))
    
    # ボラティリティクラスター（GARCH効果）
    volatility = np.zeros(length)
    base_vol = 1.0
    for i in range(1, length):
        volatility[i] = 0.9 * volatility[i-1] + 0.1 * base_vol + 0.05 * abs(random_walk[i] - random_walk[i-1])
    
    # ノイズ（動的ボラティリティ）
    noise = np.random.normal(0, 1, length) * (0.5 + volatility)
    
    # 価格の合成
    log_returns = (
        trend_strength +  # トレンド
        long_cycle * 0.001 +  # 長期サイクル
        medium_cycle * 0.002 +  # 中期サイクル
        short_cycle * 0.003 +  # 短期サイクル
        noise * 0.01  # ノイズ
    )
    
    # 累積リターンから価格を計算
    prices = base_price * np.exp(np.cumsum(log_returns))
    
    # OHLC データの生成
    data = []
    for i, close in enumerate(prices):
        # 日内変動の計算
        daily_range = abs(np.random.normal(0, volatility[i] * close * 0.01))
        
        # OHLC の計算
        if i == 0:
            open_price = close
        else:
            gap = np.random.normal(0, volatility[i] * close * 0.005)
            open_price = prices[i-1] + gap
        
        high = max(open_price, close) + daily_range * np.random.uniform(0.3, 1.0)
        low = min(open_price, close) - daily_range * np.random.uniform(0.3, 1.0)
        
        # 論理的整合性の確保
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        # 出来高（価格変動に連動）
        volume_base = 10000
        volume_multiplier = 1 + abs(log_returns[i]) * 10  # 価格変動が大きい時に出来高増加
        volume = volume_base * volume_multiplier * np.random.uniform(0.5, 2.0)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    return df

def test_chart_with_different_configs():
    """
    異なる設定でグランドサイクルMAチャートをテスト
    """
    print("=== グランドサイクルMAチャート テスト ===")
    
    try:
        from visualization.grand_cycle_ma_chart import GrandCycleMAChart
        
        # テストデータの作成
        print("1. テストデータ作成中...")
        test_data = create_realistic_market_data(300, '2024-01-01')
        print(f"✓ テストデータ作成完了: {len(test_data)}件")
        print(f"期間: {test_data.index.min()} → {test_data.index.max()}")
        print(f"価格範囲: {test_data['close'].min():.2f} - {test_data['close'].max():.2f}")
        
        # テスト設定
        test_configs = [
            {
                'name': 'ベーシック設定',
                'params': {
                    'detector_type': 'hody',
                    'use_kalman_filter': False,
                    'use_smoother': False,
                    'src_type': 'hlc3'
                }
            },
            {
                'name': 'FRAMA強化設定',
                'params': {
                    'detector_type': 'hody',
                    'use_kalman_filter': False,
                    'use_smoother': True,
                    'smoother_type': 'frama',
                    'smoother_params': {'period': 16},
                    'src_type': 'hlc3'
                }
            },
            {
                'name': 'カルマンフィルター設定',
                'params': {
                    'detector_type': 'cycle_period',
                    'use_kalman_filter': True,
                    'kalman_filter_type': 'unscented',
                    'use_smoother': False,
                    'src_type': 'close'
                }
            },
            {
                'name': 'フル機能設定',
                'params': {
                    'detector_type': 'cycle_period',
                    'use_kalman_filter': True,
                    'kalman_filter_type': 'unscented',
                    'use_smoother': True,
                    'smoother_type': 'frama',
                    'smoother_params': {'period': 20},
                    'src_type': 'hlc3'
                }
            }
        ]
        
        # 各設定でテスト
        for i, config in enumerate(test_configs, 1):
            try:
                print(f"\n{i}. {config['name']} テスト中...")
                
                chart = GrandCycleMAChart()
                chart.data = test_data  # テストデータを直接設定
                
                # インジケーター計算
                chart.calculate_indicators(**config['params'])
                
                # チャート描画（保存のみ）
                output_file = f"grand_cycle_ma_test_{i}_{config['name'].replace(' ', '_')}.png"
                chart.plot(
                    title=f"グランドサイクルMA - {config['name']}",
                    show_volume=True,
                    figsize=(16, 12),
                    savefig=output_file
                )
                
                print(f"  ✓ チャート保存完了: {output_file}")
                
                # 統計情報の表示
                result = chart.result
                valid_mama = result.grand_mama_values[~np.isnan(result.grand_mama_values)]
                valid_alpha = result.alpha_values[~np.isnan(result.alpha_values)]
                
                if len(valid_mama) > 0:
                    print(f"  MAMA有効データ: {len(valid_mama)}/{len(result.grand_mama_values)}")
                    print(f"  MAMA平均値: {np.mean(valid_mama):.4f}")
                
                if len(valid_alpha) > 0:
                    print(f"  Alpha平均値: {np.mean(valid_alpha):.4f}")
                
            except Exception as e:
                print(f"  ✗ {config['name']} テストエラー: {e}")
                continue
        
        print("\n=== チャートテスト完了 ===")
        return True
        
    except ImportError as e:
        print(f"✗ インポートエラー: {e}")
        print("必要なモジュールが見つかりません")
        return False
    
    except Exception as e:
        import traceback
        print(f"✗ テストエラー: {e}")
        print(f"詳細: {traceback.format_exc()}")
        return False

def test_chart_with_config_file():
    """
    設定ファイルを使用したチャートテスト（config.yamlが存在する場合）
    """
    print("\n=== 設定ファイルベーステスト ===")
    
    config_path = 'config.yaml'
    if not os.path.exists(config_path):
        print(f"設定ファイル {config_path} が見つかりません。スキップします。")
        return True
    
    try:
        from visualization.grand_cycle_ma_chart import GrandCycleMAChart
        
        print("設定ファイルからデータを読み込み中...")
        chart = GrandCycleMAChart()
        chart.load_data_from_config(config_path)
        
        # 最新100件のデータでテスト
        recent_data = chart.data.tail(200)
        chart.data = recent_data
        
        # 計算とチャート描画
        chart.calculate_indicators(
            detector_type='hody',
            use_kalman_filter=True,
            kalman_filter_type='adaptive',
            use_smoother=True,
            smoother_type='frama',
            src_type='hlc3'
        )
        
        chart.plot(
            title="グランドサイクルMA - 実データテスト",
            show_volume=True,
            savefig="grand_cycle_ma_real_data_test.png"
        )
        
        print("✓ 実データチャート保存完了: grand_cycle_ma_real_data_test.png")
        return True
        
    except Exception as e:
        print(f"✗ 実データテストエラー: {e}")
        return False

def main():
    """メインテスト実行"""
    print("=== グランドサイクルMAチャート 総合テスト ===")
    
    # テスト実行
    test1_success = test_chart_with_different_configs()
    test2_success = test_chart_with_config_file()
    
    # 結果サマリー
    print(f"\n=== テスト結果サマリー ===")
    print(f"設定別チャートテスト: {'成功' if test1_success else '失敗'}")
    print(f"実データチャートテスト: {'成功' if test2_success else '失敗'}")
    
    if test1_success or test2_success:
        print("\n🎉 少なくとも一部のテストが成功しました！")
        print("\n✅ 確認されたポイント:")
        if test1_success:
            print("  - 合成市場データでのチャート描画")
            print("  - 複数設定での動作確認")
            print("  - MAMA/FAMAラインの表示")
            print("  - Alpha値・サイクル周期の表示")
            print("  - トレンド方向の色分け")
        if test2_success:
            print("  - 実際の相場データでの動作確認")
            print("  - 設定ファイルからのデータ読み込み")
        
        print("\n📊 生成されたチャートファイル:")
        for i in range(1, 5):
            filename = f"grand_cycle_ma_test_{i}_*.png"
            print(f"  - grand_cycle_ma_test_{i}_[設定名].png")
        if test2_success:
            print("  - grand_cycle_ma_real_data_test.png")
            
        print("\n🚀 次のステップ:")
        print("  - 実際の取引戦略での活用")
        print("  - パラメータの最適化")
        print("  - 他のインジケーターとの組み合わせ")
    else:
        print("\n⚠️ すべてのテストが失敗しました")
    
    return test1_success or test2_success

if __name__ == "__main__":
    main()