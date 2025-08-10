#!/usr/bin/env python3
"""
HyperFRAMA Bollinger Bands Test

HyperFRAMAボリンジャーバンドの動作確認テスト
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# パスを追加
sys.path.append(str(Path(__file__).parent))

from indicators.hyper_frama_bollinger import HyperFRAMABollinger
from data.binance_data_source import BinanceDataSource
from indicators.price_source import PriceSource


def test_hyper_frama_bollinger_basic():
    """基本動作テスト"""
    print("=== HyperFRAMA Bollinger Bands 基本動作テスト ===")
    
    # テストデータ生成
    np.random.seed(42)
    n_points = 200
    
    # 価格データ生成（トレンドとノイズ）
    trend = np.linspace(100, 120, n_points)
    noise = np.random.normal(0, 2, n_points)
    close_prices = trend + noise
    
    # OHLCV データ作成
    data = pd.DataFrame({
        'open': close_prices * 0.999,
        'high': close_prices * 1.002,
        'low': close_prices * 0.998,
        'close': close_prices,
        'volume': np.random.uniform(1000, 5000, n_points)
    })
    
    # 固定シグマモードテスト
    print("\n1. 固定シグマモードテスト")
    indicator_fixed = HyperFRAMABollinger(
        period=20,
        sigma_mode="fixed",
        fixed_sigma=2.0
    )
    
    result_fixed = indicator_fixed.calculate(data)
    
    print(f"ミッドライン数: {np.sum(~np.isnan(result_fixed.midline))}")
    print(f"上バンド数: {np.sum(~np.isnan(result_fixed.upper_band))}")
    print(f"下バンド数: {np.sum(~np.isnan(result_fixed.lower_band))}")
    print(f"標準偏差数: {np.sum(~np.isnan(result_fixed.std_values))}")
    
    # 動的シグマモードテスト
    print("\n2. 動的シグマモードテスト")
    indicator_dynamic = HyperFRAMABollinger(
        period=20,
        sigma_mode="dynamic",
        sigma_min=1.0,
        sigma_max=2.5
    )
    
    result_dynamic = indicator_dynamic.calculate(data)
    
    print(f"ミッドライン数: {np.sum(~np.isnan(result_dynamic.midline))}")
    print(f"上バンド数: {np.sum(~np.isnan(result_dynamic.upper_band))}")
    print(f"下バンド数: {np.sum(~np.isnan(result_dynamic.lower_band))}")
    print(f"ER値数: {np.sum(~np.isnan(result_dynamic.er_values))}")
    print(f"シグマ値範囲: {np.nanmin(result_dynamic.sigma_values):.3f} - {np.nanmax(result_dynamic.sigma_values):.3f}")
    
    # パーセント B テスト
    print("\n3. パーセント B テスト")
    valid_pb = result_dynamic.percent_b[~np.isnan(result_dynamic.percent_b)]
    print(f"パーセント B 範囲: {np.min(valid_pb):.3f} - {np.max(valid_pb):.3f}")
    print(f"パーセント B > 1.0 の数: {np.sum(valid_pb > 1.0)}")
    print(f"パーセント B < 0.0 の数: {np.sum(valid_pb < 0.0)}")
    
    return result_fixed, result_dynamic


def test_hyper_frama_bollinger_real_data():
    """実データでのテスト"""
    print("\n=== 実データテスト ===")
    
    try:
        # Binanceデータ読み込み
        data_source = BinanceDataSource()
        data = data_source.get_data("BTCUSDT", "4h", limit=500)
        
        if data is None or len(data) < 100:
            print("実データの取得に失敗しました。合成データを使用します。")
            return test_synthetic_market_data()
        
        print(f"データサイズ: {len(data)}")
        print(f"価格範囲: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
        
        # 動的シグマモードで計算
        indicator = HyperFRAMABollinger(
            period=20,
            sigma_mode="dynamic",
            sigma_min=1.0,
            sigma_max=2.5,
            enable_signals=True,
            enable_percentile=True
        )
        
        result = indicator.calculate(data)
        
        # 統計表示
        print(f"\n計算結果統計:")
        print(f"ミッドライン有効数: {np.sum(~np.isnan(result.midline))}")
        print(f"バンド有効数: {np.sum(~np.isnan(result.upper_band))}")
        print(f"シグマ値範囲: {np.nanmin(result.sigma_values):.3f} - {np.nanmax(result.sigma_values):.3f}")
        print(f"ER値範囲: {np.nanmin(result.er_values):.3f} - {np.nanmax(result.er_values):.3f}")
        
        # シグナル統計
        if result.band_position is not None:
            upper_breaks = np.sum(result.band_position == 1.0)
            lower_breaks = np.sum(result.band_position == -1.0)
            inside_band = np.sum(result.band_position == 0.0)
            print(f"\nバンドブレイク統計:")
            print(f"上バンドブレイク: {upper_breaks}")
            print(f"下バンドブレイク: {lower_breaks}")
            print(f"バンド内: {inside_band}")
        
        if result.squeeze_signal is not None:
            squeeze_count = np.sum(result.squeeze_signal == 1.0)
            expansion_count = np.sum(result.expansion_signal == 1.0)
            print(f"スクイーズ回数: {squeeze_count}")
            print(f"エクスパンション回数: {expansion_count}")
        
        return result
        
    except Exception as e:
        print(f"実データテストエラー: {e}")
        return test_synthetic_market_data()


def test_synthetic_market_data():
    """合成マーケットデータテスト"""
    print("\n=== 合成マーケットデータテスト ===")
    
    # より現実的な価格データ生成
    np.random.seed(123)
    n_points = 300
    
    # ボラティリティクラスタリング付きの価格生成
    base_price = 50000
    returns = []
    volatilities = []
    
    # GARCH風のボラティリティ
    alpha = 0.1
    beta = 0.85
    omega = 0.001
    
    vol = 0.02
    for i in range(n_points):
        # ボラティリティ更新
        if i > 0:
            vol = np.sqrt(omega + alpha * (returns[-1]**2) + beta * (vol**2))
        volatilities.append(vol)
        
        # リターン生成
        ret = np.random.normal(0, vol)
        returns.append(ret)
    
    # 価格系列生成
    prices = [base_price]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices[:-1])  # 最後の要素を削除
    
    # OHLCV生成
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, n_points)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.003, n_points))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.003, n_points))),
        'close': prices,
        'volume': np.random.uniform(100, 1000, n_points)
    })
    
    # 動的シグマモードで計算
    indicator = HyperFRAMABollinger(
        period=20,
        sigma_mode="dynamic",
        sigma_min=1.0,
        sigma_max=2.5,
        enable_signals=True
    )
    
    result = indicator.calculate(data)
    
    print(f"価格範囲: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    print(f"シグマ値範囲: {np.nanmin(result.sigma_values):.3f} - {np.nanmax(result.sigma_values):.3f}")
    print(f"ER値範囲: {np.nanmin(result.er_values):.3f} - {np.nanmax(result.er_values):.3f}")
    
    return result


def create_visualization_chart(data, result, title="HyperFRAMA Bollinger Bands"):
    """可視化チャート作成"""
    
    plt.figure(figsize=(15, 10))
    
    # サブプロット1: 価格とボリンジャーバンド
    plt.subplot(3, 1, 1)
    plt.plot(data['close'], label='Close Price', color='black', linewidth=1)
    plt.plot(result.midline, label='HyperFRAMA Midline', color='blue', linewidth=1.5)
    plt.plot(result.upper_band, label='Upper Band', color='red', linewidth=1, alpha=0.8)
    plt.plot(result.lower_band, label='Lower Band', color='green', linewidth=1, alpha=0.8)
    plt.fill_between(range(len(result.upper_band)), result.upper_band, result.lower_band, 
                     alpha=0.1, color='gray', label='Band Area')
    plt.title(f'{title} - Price and Bands')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # サブプロット2: シグマ値とER値
    plt.subplot(3, 1, 2)
    plt.plot(result.sigma_values, label='Sigma Values', color='orange', linewidth=1.5)
    if result.er_values is not None:
        plt.plot(result.er_values, label='HyperER Values', color='purple', linewidth=1)
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Sigma Min')
    plt.axhline(y=2.5, color='red', linestyle='--', alpha=0.5, label='Sigma Max')
    plt.title('Dynamic Sigma and HyperER')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # サブプロット3: パーセント B
    plt.subplot(3, 1, 3)
    plt.plot(result.percent_b, label='Percent B', color='brown', linewidth=1)
    plt.axhline(y=0.0, color='green', linestyle='--', alpha=0.5, label='Lower Band')
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Upper Band')
    plt.axhline(y=0.5, color='blue', linestyle='--', alpha=0.5, label='Midline')
    plt.fill_between(range(len(result.percent_b)), 0, 1, alpha=0.1, color='gray')
    plt.title('Percent B')
    plt.ylabel('Percent B')
    plt.xlabel('Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hyper_frama_bollinger_test.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """メイン実行"""
    print("HyperFRAMA Bollinger Bands テスト開始\n")
    
    # 基本テスト
    result_fixed, result_dynamic = test_hyper_frama_bollinger_basic()
    
    # 実データテスト
    result_real = test_hyper_frama_bollinger_real_data()
    
    # 可視化（実データまたは合成データの結果を使用）
    if result_real is not None:
        # 実データまたは合成データ用の簡単なデータフレーム作成
        data = pd.DataFrame({
            'close': np.random.normal(100, 5, len(result_real.midline))  # ダミーデータ
        })
        create_visualization_chart(data, result_real)
    
    print("\nテスト完了！")
    print("チャート: hyper_frama_bollinger_test.png")


if __name__ == "__main__":
    main()