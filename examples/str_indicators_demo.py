#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STR（Smooth True Range）、Ultimate Channel、Ultimate Bandsインジケーターのデモンストレーション

John Ehlersの論文「ULTIMATE CHANNEL and ULTIMATE BANDS」に基づく実装のテストとデモ
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any

# インジケーターのインポート
from indicators.str import STR, UltimateChannel, UltimateBands
from indicators.supertrend import Supertrend


def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    テスト用のサンプルデータを生成
    
    Args:
        n_samples: サンプル数
    
    Returns:
        OHLC形式のDataFrame
    """
    np.random.seed(42)  # 再現性のため
    
    # 価格のランダムウォーク
    base_price = 100.0
    price_changes = np.random.normal(0, 0.02, n_samples)
    prices = base_price * np.exp(np.cumsum(price_changes))
    
    # OHLC生成
    data = []
    for i in range(n_samples):
        if i == 0:
            open_price = prices[i]
        else:
            open_price = close_price
        
        # 高値・安値をランダムに生成
        high_offset = np.random.uniform(0.001, 0.01)
        low_offset = np.random.uniform(-0.01, -0.001)
        
        high_price = open_price * (1 + high_offset)
        low_price = open_price * (1 + low_offset)
        close_price = prices[i]
        
        # 高値・安値の調整
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        data.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': np.random.randint(1000, 10000)
        })
    
    return pd.DataFrame(data)


def test_str_indicator():
    """STRインジケーターのテスト"""
    print("🧪 STRインジケーターのテスト開始...")
    
    # サンプルデータの生成
    data = generate_sample_data(500)
    
    # STRインジケーターの計算
    str_indicator = STR(period=20.0, src_type='hlc3')
    result = str_indicator.calculate(data)
    
    # 結果の検証
    print(f"✅ STR計算完了 - データ長: {len(result.values)}")
    print(f"   - STR値の範囲: {np.min(result.values):.4f} ~ {np.max(result.values):.4f}")
    print(f"   - True Range値の範囲: {np.min(result.true_range):.4f} ~ {np.max(result.true_range):.4f}")
    
    # 各値の取得テスト
    str_values = str_indicator.get_values()
    tr_values = str_indicator.get_true_range()
    th_values = str_indicator.get_true_high()
    tl_values = str_indicator.get_true_low()
    
    print(f"   - get_values()正常動作: {str_values is not None}")
    print(f"   - get_true_range()正常動作: {tr_values is not None}")
    print(f"   - get_true_high()正常動作: {th_values is not None}")
    print(f"   - get_true_low()正常動作: {tl_values is not None}")
    
    return data, result


def test_ultimate_channel():
    """Ultimate Channelインジケーターのテスト"""
    print("\n🧪 Ultimate Channelインジケーターのテスト開始...")
    
    # サンプルデータの生成
    data = generate_sample_data(500)
    
    # Ultimate Channelインジケーターの計算
    channel_indicator = UltimateChannel(
        length=20.0,
        str_length=20.0,
        num_strs=1.0,
        src_type='close'
    )
    result = channel_indicator.calculate(data)
    
    # 結果の検証
    print(f"✅ Ultimate Channel計算完了 - データ長: {len(result.upper_channel)}")
    print(f"   - 上側チャネル範囲: {np.min(result.upper_channel):.4f} ~ {np.max(result.upper_channel):.4f}")
    print(f"   - 下側チャネル範囲: {np.min(result.lower_channel):.4f} ~ {np.max(result.lower_channel):.4f}")
    print(f"   - 中心線範囲: {np.min(result.center_line):.4f} ~ {np.max(result.center_line):.4f}")
    
    # 各値の取得テスト
    center_values = channel_indicator.get_values()
    upper_values = channel_indicator.get_upper_channel()
    lower_values = channel_indicator.get_lower_channel()
    str_values = channel_indicator.get_str_values()
    
    print(f"   - get_values()正常動作: {center_values is not None}")
    print(f"   - get_upper_channel()正常動作: {upper_values is not None}")
    print(f"   - get_lower_channel()正常動作: {lower_values is not None}")
    print(f"   - get_str_values()正常動作: {str_values is not None}")
    
    return data, result


def test_ultimate_bands():
    """Ultimate Bandsインジケーターのテスト"""
    print("\n🧪 Ultimate Bandsインジケーターのテスト開始...")
    
    # サンプルデータの生成
    data = generate_sample_data(500)
    
    # Ultimate Bandsインジケーターの計算
    bands_indicator = UltimateBands(
        length=20.0,
        num_sds=1.0,
        src_type='close'
    )
    result = bands_indicator.calculate(data)
    
    # 結果の検証
    print(f"✅ Ultimate Bands計算完了 - データ長: {len(result.upper_band)}")
    print(f"   - 上側バンド範囲: {np.min(result.upper_band):.4f} ~ {np.max(result.upper_band):.4f}")
    print(f"   - 下側バンド範囲: {np.min(result.lower_band):.4f} ~ {np.max(result.lower_band):.4f}")
    print(f"   - 中心線範囲: {np.min(result.center_line):.4f} ~ {np.max(result.center_line):.4f}")
    print(f"   - 標準偏差範囲: {np.min(result.standard_deviation):.4f} ~ {np.max(result.standard_deviation):.4f}")
    
    # 各値の取得テスト
    center_values = bands_indicator.get_values()
    upper_values = bands_indicator.get_upper_band()
    lower_values = bands_indicator.get_lower_band()
    sd_values = bands_indicator.get_standard_deviation()
    
    print(f"   - get_values()正常動作: {center_values is not None}")
    print(f"   - get_upper_band()正常動作: {upper_values is not None}")
    print(f"   - get_lower_band()正常動作: {lower_values is not None}")
    print(f"   - get_standard_deviation()正常動作: {sd_values is not None}")
    
    return data, result


def create_comparison_chart():
    """比較チャートを作成"""
    print("\n📊 比較チャートを作成中...")
    
    # サンプルデータの生成
    data = generate_sample_data(300)
    
    # インジケーターの計算
    str_indicator = STR(period=20.0, src_type='hlc3')
    str_result = str_indicator.calculate(data)
    
    channel_indicator = UltimateChannel(length=20.0, str_length=20.0, num_strs=1.0, src_type='close')
    channel_result = channel_indicator.calculate(data)
    
    bands_indicator = UltimateBands(length=20.0, num_sds=1.0, src_type='close')
    bands_result = bands_indicator.calculate(data)
    
    # 比較用にSupertrendも計算
    supertrend_indicator = Supertrend(period=20, multiplier=1.0, src_type='close')
    supertrend_result = supertrend_indicator.calculate(data)
    
    # チャートの作成
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    
    # 1. 価格とUltimate Channel
    ax1 = axes[0]
    ax1.plot(data['close'], label='Close Price', color='black', linewidth=1)
    ax1.plot(channel_result.upper_channel, label='Ultimate Channel Upper', color='blue', alpha=0.7)
    ax1.plot(channel_result.lower_channel, label='Ultimate Channel Lower', color='blue', alpha=0.7)
    ax1.plot(channel_result.center_line, label='Ultimate Channel Center', color='red', alpha=0.7)
    ax1.fill_between(range(len(channel_result.upper_channel)), 
                     channel_result.lower_channel, 
                     channel_result.upper_channel, 
                     color='blue', alpha=0.1)
    ax1.set_title('Ultimate Channel vs Price', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 価格とUltimate Bands
    ax2 = axes[1]
    ax2.plot(data['close'], label='Close Price', color='black', linewidth=1)
    ax2.plot(bands_result.upper_band, label='Ultimate Bands Upper', color='green', alpha=0.7)
    ax2.plot(bands_result.lower_band, label='Ultimate Bands Lower', color='green', alpha=0.7)
    ax2.plot(bands_result.center_line, label='Ultimate Bands Center', color='red', alpha=0.7)
    ax2.fill_between(range(len(bands_result.upper_band)), 
                     bands_result.lower_band, 
                     bands_result.upper_band, 
                     color='green', alpha=0.1)
    ax2.set_title('Ultimate Bands vs Price', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. STRとSupertrend ATR比較
    ax3 = axes[2]
    ax3.plot(str_result.values, label='STR (Ultimate Smoother)', color='blue', linewidth=2)
    ax3.plot(supertrend_result.atr_values, label='ATR (Supertrend)', color='red', linewidth=2)
    ax3.set_title('STR vs ATR Comparison', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Ultimate ChannelとSupertrend比較
    ax4 = axes[3]
    ax4.plot(data['close'], label='Close Price', color='black', linewidth=1)
    ax4.plot(channel_result.upper_channel, label='Ultimate Channel Upper', color='blue', alpha=0.7)
    ax4.plot(channel_result.lower_channel, label='Ultimate Channel Lower', color='blue', alpha=0.7)
    ax4.plot(supertrend_result.values, label='Supertrend', color='red', linewidth=2)
    ax4.set_title('Ultimate Channel vs Supertrend', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 出力ディレクトリの作成
    output_dir = 'examples/output'
    os.makedirs(output_dir, exist_ok=True)
    
    # チャートの保存
    output_path = os.path.join(output_dir, 'str_indicators_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ チャートが保存されました: {output_path}")
    
    # 統計情報の表示
    print("\n📈 統計情報:")
    print(f"   - STR平均値: {np.mean(str_result.values):.4f}")
    print(f"   - ATR平均値: {np.mean(supertrend_result.atr_values):.4f}")
    print(f"   - STR標準偏差: {np.std(str_result.values):.4f}")
    print(f"   - ATR標準偏差: {np.std(supertrend_result.atr_values):.4f}")
    
    plt.show()


def performance_benchmark():
    """パフォーマンスベンチマークテスト"""
    print("\n⚡ パフォーマンスベンチマークテスト開始...")
    
    import time
    
    # 大きなデータセットの生成
    large_data = generate_sample_data(5000)
    
    # STRのベンチマーク
    str_indicator = STR(period=20.0, src_type='hlc3')
    
    start_time = time.time()
    str_result = str_indicator.calculate(large_data)
    str_time = time.time() - start_time
    
    # Ultimate Channelのベンチマーク
    channel_indicator = UltimateChannel(length=20.0, str_length=20.0, num_strs=1.0, src_type='close')
    
    start_time = time.time()
    channel_result = channel_indicator.calculate(large_data)
    channel_time = time.time() - start_time
    
    # Ultimate Bandsのベンチマーク
    bands_indicator = UltimateBands(length=20.0, num_sds=1.0, src_type='close')
    
    start_time = time.time()
    bands_result = bands_indicator.calculate(large_data)
    bands_time = time.time() - start_time
    
    # 比較用にSupertrendも計算
    supertrend_indicator = Supertrend(period=20, multiplier=1.0, src_type='close')
    
    start_time = time.time()
    supertrend_result = supertrend_indicator.calculate(large_data)
    supertrend_time = time.time() - start_time
    
    print(f"✅ パフォーマンスベンチマーク結果（データ数: {len(large_data)}）:")
    print(f"   - STR: {str_time:.4f}秒")
    print(f"   - Ultimate Channel: {channel_time:.4f}秒")
    print(f"   - Ultimate Bands: {bands_time:.4f}秒")
    print(f"   - Supertrend（比較用）: {supertrend_time:.4f}秒")
    
    # キャッシュ効果のテスト
    print("\n🔄 キャッシュ効果のテスト...")
    start_time = time.time()
    str_result_cached = str_indicator.calculate(large_data)  # 同じデータで再計算
    cached_time = time.time() - start_time
    
    print(f"   - 初回計算: {str_time:.4f}秒")
    print(f"   - キャッシュ利用: {cached_time:.6f}秒")
    print(f"   - 速度向上: {str_time/cached_time:.1f}倍")


def main():
    """メインデモンストレーション"""
    print("🚀 STRインジケーター群のデモンストレーション開始")
    print("=" * 60)
    
    try:
        # 各インジケーターの基本テスト
        test_str_indicator()
        test_ultimate_channel()
        test_ultimate_bands()
        
        # 比較チャートの作成
        create_comparison_chart()
        
        # パフォーマンスベンチマーク
        performance_benchmark()
        
        print("\n" + "=" * 60)
        print("✅ すべてのテストが正常に完了しました！")
        print("\n📝 実装されたインジケーター:")
        print("   1. STR（Smooth True Range）")
        print("   2. Ultimate Channel")
        print("   3. Ultimate Bands")
        print("\n🎯 特徴:")
        print("   - John Ehlersの論文に基づく正確な実装")
        print("   - Ultimate Smootherによる超低遅延")
        print("   - supertrendと同じインターフェース")
        print("   - 高速なNumba最適化")
        print("   - 効率的なキャッシュシステム")
        print("   - 包括的なエラーハンドリング")
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 