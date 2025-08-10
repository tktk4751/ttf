#!/usr/bin/env python3
"""
Hyper Adaptive Channel テストスクリプト

ハイパーアダプティブチャネルインジケーターの動作テスト
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# パス設定
sys.path.append('.')

from indicators.hyper_adaptive_channel import HyperAdaptiveChannel, MidlineSmootherType, MultiplierMode
from data.binance_data_source import BinanceDataSource


def test_basic_functionality():
    """基本機能テスト"""
    
    print("=== 基本機能テスト ===")
    
    # テストデータ生成
    dates = pd.date_range('2024-01-01', periods=200, freq='1H')
    np.random.seed(42)
    
    # 基本OHLCV生成
    close_prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
    high_prices = close_prices + np.abs(np.random.randn(200) * 0.3)
    low_prices = close_prices - np.abs(np.random.randn(200) * 0.3)
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]
    volume = np.random.randint(1000, 10000, 200)
    
    test_data = pd.DataFrame({
        'timestamp': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    })
    
    # 各スムーザータイプをテスト
    smoothers = [
        "hyper_frama",
        "ultimate_ma", 
        "laguerre_filter",
        "z_adaptive_ma",
        "super_smoother"
    ]
    
    results = {}
    
    for smoother in smoothers:
        print(f"\n--- {smoother} テスト ---")
        
        try:
            # 固定乗数モード
            indicator_fixed = HyperAdaptiveChannel(
                period=14,
                midline_smoother=smoother,
                multiplier_mode="fixed",
                fixed_multiplier=2.0
            )
            
            result_fixed = indicator_fixed.calculate(test_data)
            print(f"✓ {smoother} (固定乗数): OK")
            print(f"  - Midline有効値: {np.sum(~np.isnan(result_fixed.midline))}")
            print(f"  - Upper Band有効値: {np.sum(~np.isnan(result_fixed.upper_band))}")
            print(f"  - Lower Band有効値: {np.sum(~np.isnan(result_fixed.lower_band))}")
            
            # 動的乗数モード
            indicator_dynamic = HyperAdaptiveChannel(
                period=14,
                midline_smoother=smoother,
                multiplier_mode="dynamic",
                er_period=10
            )
            
            result_dynamic = indicator_dynamic.calculate(test_data)
            print(f"✓ {smoother} (動的乗数): OK")
            print(f"  - ER有効値: {np.sum(~np.isnan(result_dynamic.er_values))}")
            print(f"  - 乗数範囲: {np.nanmin(result_dynamic.multiplier_values):.2f} - {np.nanmax(result_dynamic.multiplier_values):.2f}")
            
            results[smoother] = {
                'fixed': result_fixed,
                'dynamic': result_dynamic
            }
            
        except Exception as e:
            print(f"✗ {smoother} エラー: {e}")


def test_extended_data():
    """拡張データテスト"""
    
    print("\n=== 拡張データテスト ===")
    
    try:
        # より複雑なテストデータ生成
        dates = pd.date_range('2024-01-01', periods=500, freq='4H')
        np.random.seed(42)
        
        # トレンドとボラティリティを含むデータ
        trend = np.linspace(100, 120, 500)
        noise = np.random.randn(500) * 2
        volatility_cycle = 1 + 0.5 * np.sin(np.arange(500) * 2 * np.pi / 100)
        
        close_prices = trend + (noise * volatility_cycle)
        high_prices = close_prices + np.abs(np.random.randn(500) * volatility_cycle * 0.5)
        low_prices = close_prices - np.abs(np.random.randn(500) * volatility_cycle * 0.5)
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = close_prices[0]
        volume = np.random.randint(1000, 10000, 500) * volatility_cycle
        
        extended_data = pd.DataFrame({
            'timestamp': dates,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume
        })
        
        print(f"拡張テストデータ生成: {len(extended_data)}件")
        
        # ハイパーアダプティブチャネル作成
        indicator = HyperAdaptiveChannel(
            period=14,
            midline_smoother="hyper_frama",
            multiplier_mode="dynamic",
            fixed_multiplier=2.5,
            enable_signals=True,
            enable_percentile=True
        )
        
        # 計算実行
        result = indicator.calculate(extended_data)
        
        print("✓ 拡張データ計算成功")
        print(f"  - Midline有効値: {np.sum(~np.isnan(result.midline))}")
        print(f"  - チャネル幅範囲: {np.nanmin(result.bandwidth):.4f} - {np.nanmax(result.bandwidth):.4f}")
        print(f"  - 乗数範囲: {np.nanmin(result.multiplier_values):.2f} - {np.nanmax(result.multiplier_values):.2f}")
        
        # シグナル統計
        if result.channel_position is not None:
            upper_breaks = np.sum(result.channel_position == 1.0)
            lower_breaks = np.sum(result.channel_position == -1.0)
            inside_channel = np.sum(result.channel_position == 0.0)
            
            print(f"  - チャネルシグナル統計:")
            print(f"    上バンドブレイク: {upper_breaks}回")
            print(f"    下バンドブレイク: {lower_breaks}回")
            print(f"    チャネル内: {inside_channel}回")
        
        if result.squeeze_signal is not None:
            squeeze_count = np.sum(result.squeeze_signal == 1.0)
            expansion_count = np.sum(result.expansion_signal == 1.0)
            
            print(f"    スクイーズ: {squeeze_count}回")
            print(f"    エクスパンション: {expansion_count}回")
        
        # チャート作成
        create_chart(extended_data, result, "hyper_adaptive_channel_extended_test")
        
        return True
        
    except Exception as e:
        print(f"✗ 拡張データテストエラー: {e}")
        return False


def create_chart(data, result, filename):
    """チャート作成"""
    
    try:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        # 価格データとチャネル
        ax1.plot(data['close'].values, label='Close', color='black', linewidth=1)
        ax1.plot(result.midline, label='Midline', color='blue', linewidth=1.5)
        ax1.plot(result.upper_band, label='Upper Band', color='red', linewidth=1, alpha=0.7)
        ax1.plot(result.lower_band, label='Lower Band', color='green', linewidth=1, alpha=0.7)
        
        # チャネル塗りつぶし
        valid_mask = ~(np.isnan(result.upper_band) | np.isnan(result.lower_band))
        if np.any(valid_mask):
            ax1.fill_between(
                range(len(result.upper_band)),
                result.upper_band,
                result.lower_band,
                alpha=0.1,
                color='gray'
            )
        
        ax1.set_title('Hyper Adaptive Channel - Real Data Test')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # ATRと乗数
        ax2_twin = ax2.twinx()
        ax2.plot(result.atr_values, label='ATR', color='orange')
        ax2_twin.plot(result.multiplier_values, label='Multiplier', color='purple')
        
        if result.er_values is not None:
            ax2_twin.plot(result.er_values, label='Efficiency Ratio', color='brown', alpha=0.7)
        
        ax2.set_ylabel('ATR', color='orange')
        ax2_twin.set_ylabel('Multiplier / ER', color='purple')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # シグナル
        if result.channel_position is not None:
            # チャネルポジション
            ax3.plot(result.channel_position, label='Channel Position', color='blue')
            ax3.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Upper Break')
            ax3.axhline(y=-1, color='green', linestyle='--', alpha=0.5, label='Lower Break')
            ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        if result.squeeze_signal is not None:
            # スクイーズ・エクスパンション (別軸)
            ax3_twin = ax3.twinx()
            squeeze_plot = result.squeeze_signal * 0.5
            expansion_plot = result.expansion_signal * -0.5
            
            ax3_twin.plot(squeeze_plot, label='Squeeze', color='red', alpha=0.7)
            ax3_twin.plot(expansion_plot, label='Expansion', color='green', alpha=0.7)
            ax3_twin.set_ylabel('Squeeze/Expansion')
            ax3_twin.legend(loc='upper right')
        
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Position')
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{filename}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ チャート保存: {filename}.png")
        
    except Exception as e:
        print(f"✗ チャート作成エラー: {e}")


def test_performance():
    """パフォーマンステスト"""
    
    print("\n=== パフォーマンステスト ===")
    
    import time
    
    # 大量データ生成
    dates = pd.date_range('2023-01-01', periods=2000, freq='1H')
    np.random.seed(42)
    
    close_prices = 100 + np.cumsum(np.random.randn(2000) * 0.5)
    high_prices = close_prices + np.abs(np.random.randn(2000) * 0.5)
    low_prices = close_prices - np.abs(np.random.randn(2000) * 0.5)
    open_prices = np.roll(close_prices, 1)
    volume = np.random.randint(1000, 50000, 2000)
    
    large_data = pd.DataFrame({
        'timestamp': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    })
    
    # パフォーマンステスト実行
    smoothers = ["hyper_frama", "ultimate_ma", "super_smoother"]
    
    for smoother in smoothers:
        print(f"\n--- {smoother} パフォーマンステスト ---")
        
        indicator = HyperAdaptiveChannel(
            period=14,
            midline_smoother=smoother,
            multiplier_mode="dynamic"
        )
        
        # 計算時間測定
        start_time = time.time()
        result = indicator.calculate(large_data)
        end_time = time.time()
        
        elapsed = end_time - start_time
        print(f"  計算時間: {elapsed:.3f}秒 ({len(large_data)}データポイント)")
        print(f"  処理速度: {len(large_data)/elapsed:.0f} データ/秒")
        
        # キャッシュテスト
        start_time = time.time()
        result_cached = indicator.calculate(large_data)
        end_time = time.time()
        
        cached_elapsed = end_time - start_time
        print(f"  キャッシュ計算時間: {cached_elapsed:.3f}秒")
        print(f"  高速化率: {elapsed/cached_elapsed:.1f}x")


def main():
    """メイン実行"""
    
    print("Hyper Adaptive Channel インジケーター テスト開始")
    print("="*50)
    
    # 基本機能テスト
    test_basic_functionality()
    
    # 拡張データテスト
    test_extended_data()
    
    # パフォーマンステスト
    test_performance()
    
    print("\n" + "="*50)
    print("テスト完了")


if __name__ == "__main__":
    main()