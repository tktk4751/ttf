#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HyperTripleFRAMAチャートのテスト
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from visualization.hyper_triple_frama_chart import HyperTripleFRAMAChart
from indicators.hyper_triple_frama import HyperTripleFRAMA


def create_test_data(length: int = 200, symbol: str = "TESTUSDT") -> pd.DataFrame:
    """テスト用のOHLCVデータを生成"""
    np.random.seed(42)
    
    # 開始日時
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(hours=i*4) for i in range(length)]  # 4時間足
    
    # 価格データの生成（トレンドとノイズを含む）
    base_price = 100.0
    prices = []
    
    for i in range(length):
        # トレンド成分（サイン波 + 長期上昇トレンド）
        trend_component = np.sin(i * 0.05) * 10 + i * 0.1
        
        # ノイズ成分
        noise = np.random.normal(0, 2.0)
        
        # 価格計算
        price = base_price + trend_component + noise
        
        # OHLC生成
        high = price + abs(np.random.normal(0, 1.5))
        low = price - abs(np.random.normal(0, 1.5))
        open_price = price + np.random.normal(0, 0.8)
        close_price = price + np.random.normal(0, 0.8)
        
        # 出来高（ランダム）
        volume = 1000 + np.random.randint(0, 2000)
        
        prices.append({
            'open': max(0.1, open_price),
            'high': max(0.1, high),
            'low': max(0.1, low),
            'close': max(0.1, close_price),
            'volume': volume
        })
    
    df = pd.DataFrame(prices, index=pd.DatetimeIndex(dates))
    return df


def test_chart_with_synthetic_data():
    """合成データでチャートをテスト"""
    print("=== HyperTripleFRAMAチャート テスト（合成データ） ===")
    
    # テストデータの生成
    test_data = create_test_data(200)
    print(f"テストデータ生成完了: {len(test_data)}行")
    print(f"価格範囲: {test_data['close'].min():.2f} - {test_data['close'].max():.2f}")
    
    # HyperTripleFRAMAの計算
    indicator = HyperTripleFRAMA(
        period=16,
        src_type='hl2',
        alpha_multiplier1=1.0,
        alpha_multiplier2=0.5,
        alpha_multiplier3=0.1,
        enable_indicator_adaptation=False,
        smoothing_mode='none'
    )
    
    print("\nインジケーターの計算中...")
    result = indicator.calculate(test_data)
    
    # 結果確認
    print(f"計算結果:")
    print(f"  FRAMA1有効値: {(~np.isnan(result.frama_values)).sum()}")
    print(f"  FRAMA2有効値: {(~np.isnan(result.second_frama_values)).sum()}")
    print(f"  FRAMA3有効値: {(~np.isnan(result.third_frama_values)).sum()}")
    
    # チャート用のクラスを使わずに直接プロット
    print("\nチャートの描画...")
    
    # データフレームにインジケーター結果を追加
    chart_data = test_data.copy()
    chart_data['frama1'] = result.frama_values
    chart_data['frama2'] = result.second_frama_values
    chart_data['frama3'] = result.third_frama_values
    chart_data['fractal_dim'] = result.fractal_dimension
    
    # 基本的なmatplotlibチャート
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # 価格とFRAMAライン
    axes[0].plot(chart_data.index, chart_data['close'], label='Close', color='black', linewidth=0.8, alpha=0.7)
    axes[0].plot(chart_data.index, chart_data['frama1'], label='FRAMA1 (Fast)', color='red', linewidth=2.0)
    axes[0].plot(chart_data.index, chart_data['frama2'], label='FRAMA2 (Medium)', color='blue', linewidth=1.5)
    axes[0].plot(chart_data.index, chart_data['frama3'], label='FRAMA3 (Slow)', color='green', linewidth=1.2)
    axes[0].set_title('HyperTripleFRAMA - 価格とFRAMAライン')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # フラクタル次元
    axes[1].plot(chart_data.index, chart_data['fractal_dim'], color='purple', linewidth=1.2)
    axes[1].axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Trend (1.0)')
    axes[1].axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='Range (2.0)')
    axes[1].set_title('フラクタル次元')
    axes[1].set_ylabel('Fractal Dimension')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # FRAMA間の差
    chart_data['diff_12'] = chart_data['frama1'] - chart_data['frama2']
    chart_data['diff_23'] = chart_data['frama2'] - chart_data['frama3']
    chart_data['diff_13'] = chart_data['frama1'] - chart_data['frama3']
    
    axes[2].plot(chart_data.index, chart_data['diff_12'], label='FRAMA1-2', color='orange', linewidth=1.2)
    axes[2].plot(chart_data.index, chart_data['diff_23'], label='FRAMA2-3', color='cyan', linewidth=1.0)
    axes[2].plot(chart_data.index, chart_data['diff_13'], label='FRAMA1-3', color='magenta', linewidth=0.8)
    axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[2].set_title('FRAMA間の差')
    axes[2].set_ylabel('Price Difference')
    axes[2].set_xlabel('Date')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 統計情報の表示
    print(f"\n=== 統計情報 ===")
    valid_data = chart_data.dropna()
    print(f"有効データ点数: {len(valid_data)}")
    
    if len(valid_data) > 0:
        # クロスオーバーの検出
        crossovers_12_up = ((valid_data['frama1'] > valid_data['frama2']) & 
                           (valid_data['frama1'].shift(1) <= valid_data['frama2'].shift(1))).sum()
        crossovers_12_down = ((valid_data['frama1'] < valid_data['frama2']) & 
                             (valid_data['frama1'].shift(1) >= valid_data['frama2'].shift(1))).sum()
        
        print(f"FRAMA1-2クロスオーバー - 上: {crossovers_12_up}, 下: {crossovers_12_down}")
        
        # フラクタル次元統計
        fractal_stats = valid_data['fractal_dim'].describe()
        print(f"フラクタル次元 - 平均: {fractal_stats['mean']:.3f}, 範囲: {fractal_stats['min']:.3f} - {fractal_stats['max']:.3f}")
    
    # チャート保存
    output_file = "hyper_triple_frama_test_chart.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"チャートを保存しました: {output_file}")
    
    # plt.show()  # 表示は無効化（保存のみ）
    
    return True


def test_different_parameters():
    """異なるパラメータでのテスト"""
    print("\n=== パラメータ別比較テスト ===")
    
    # テストデータ（短め）
    test_data = create_test_data(100)
    
    # パラメータ設定
    test_configs = [
        {
            'name': 'デフォルト設定',
            'params': {'alpha_multiplier1': 1.0, 'alpha_multiplier2': 0.5, 'alpha_multiplier3': 0.1}
        },
        {
            'name': '高感度設定',
            'params': {'alpha_multiplier1': 1.0, 'alpha_multiplier2': 0.8, 'alpha_multiplier3': 0.4}
        },
        {
            'name': '低感度設定',
            'params': {'alpha_multiplier1': 0.6, 'alpha_multiplier2': 0.3, 'alpha_multiplier3': 0.05}
        }
    ]
    
    fig, axes = plt.subplots(len(test_configs), 1, figsize=(14, 12), sharex=True)
    
    for i, config in enumerate(test_configs):
        print(f"\nテスト中: {config['name']}")
        
        indicator = HyperTripleFRAMA(
            period=12,
            enable_indicator_adaptation=False,
            smoothing_mode='none',
            **config['params']
        )
        
        result = indicator.calculate(test_data)
        
        # プロット
        ax = axes[i] if len(test_configs) > 1 else axes
        ax.plot(test_data.index, test_data['close'], label='Close', color='black', linewidth=0.8, alpha=0.7)
        ax.plot(test_data.index, result.frama_values, label='FRAMA1', color='red', linewidth=2.0)
        ax.plot(test_data.index, result.second_frama_values, label='FRAMA2', color='blue', linewidth=1.5)
        ax.plot(test_data.index, result.third_frama_values, label='FRAMA3', color='green', linewidth=1.2)
        
        ax.set_title(f'{config["name"]} - α1={config["params"]["alpha_multiplier1"]}, α2={config["params"]["alpha_multiplier2"]}, α3={config["params"]["alpha_multiplier3"]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 統計
        valid_count = (~np.isnan(result.frama_values)).sum()
        print(f"  有効値数: {valid_count}")
        
        if valid_count > 10:
            last_values = {
                'price': test_data['close'].iloc[-1],
                'frama1': result.frama_values[-1],
                'frama2': result.second_frama_values[-1],
                'frama3': result.third_frama_values[-1]
            }
            print(f"  最終値 - 価格: {last_values['price']:.2f}, FRAMA1: {last_values['frama1']:.2f}, FRAMA2: {last_values['frama2']:.2f}, FRAMA3: {last_values['frama3']:.2f}")
    
    plt.tight_layout()
    
    # チャート保存
    output_file = "hyper_triple_frama_params_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nパラメータ比較チャートを保存しました: {output_file}")
    
    # plt.show()  # 表示は無効化（保存のみ）
    
    return True


if __name__ == "__main__":
    print("HyperTripleFRAMAチャートテスト開始")
    
    try:
        # 基本テスト
        success1 = test_chart_with_synthetic_data()
        
        if success1:
            # パラメータテスト
            success2 = test_different_parameters()
            
            if success1 and success2:
                print("\n🎉 全てのチャートテスト完了!")
            else:
                print("\n❌ 一部のテストに失敗しました")
        else:
            print("\n❌ 基本テストに失敗しました")
            
    except Exception as e:
        print(f"\n❌ テスト中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()