#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ultimate Breakout Channel ボラティリティタイプ選択機能テスト

このスクリプトは、Ultimate Breakout ChannelでATRとUltimate Volatilityの
両方のボラティリティタイプを使用して、その違いを比較テストします。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from indicators.ultimate_breakout_channel import UltimateBreakoutChannel
from data.binance_data_source import BinanceDataSource

def create_test_data(num_points=1000):
    """テスト用データを生成"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=num_points, freq='1H')
    
    # トレンドとボラティリティを含む価格データ生成
    trend = np.cumsum(np.random.randn(num_points) * 0.001)
    noise = np.random.randn(num_points) * 0.01
    prices = 50000 + trend * 1000 + noise * 100
    
    # OHLC生成
    opens = prices
    highs = prices + np.abs(np.random.randn(num_points)) * 50
    lows = prices - np.abs(np.random.randn(num_points)) * 50
    closes = prices + np.random.randn(num_points) * 20
    
    return pd.DataFrame({
        'timestamp': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': np.random.randint(100, 1000, num_points)
    })

def test_volatility_types():
    """ボラティリティタイプ比較テスト"""
    print("🚀 Ultimate Breakout Channel ボラティリティタイプ比較テスト")
    print("=" * 60)
    
    # テストデータ準備
    print("\n📊 テストデータ生成中...")
    test_data = create_test_data(500)
    print(f"データポイント数: {len(test_data)}")
    
    # ATR版テスト
    print("\n🔥 ATR版 Ultimate Breakout Channel テスト")
    print("-" * 40)
    
    ubc_atr = UltimateBreakoutChannel(
        atr_period=14,
        min_multiplier=1.0,
        max_multiplier=6.0,
        volatility_type='atr'  # ATR使用
    )
    
    result_atr = ubc_atr.calculate(test_data)
    print(f"✅ ATR版計算完了")
    print(f"シグナル数: {int(np.sum(np.abs(result_atr.breakout_signals)))}")
    print(f"平均品質: {np.nanmean(result_atr.signal_quality[result_atr.signal_quality > 0]):.3f}")
    print(f"平均動的乗数: {np.nanmean(result_atr.dynamic_multiplier[~np.isnan(result_atr.dynamic_multiplier)]):.2f}")
    
    # Ultimate Volatility版テスト
    print("\n⚡ Ultimate Volatility版 Ultimate Breakout Channel テスト")
    print("-" * 40)
    
    ubc_ultimate = UltimateBreakoutChannel(
        atr_period=14,
        min_multiplier=1.0,
        max_multiplier=6.0,
        volatility_type='ultimate'  # Ultimate Volatility使用
    )
    
    result_ultimate = ubc_ultimate.calculate(test_data)
    print(f"✅ Ultimate Volatility版計算完了")
    print(f"シグナル数: {int(np.sum(np.abs(result_ultimate.breakout_signals)))}")
    print(f"平均品質: {np.nanmean(result_ultimate.signal_quality[result_ultimate.signal_quality > 0]):.3f}")
    print(f"平均動的乗数: {np.nanmean(result_ultimate.dynamic_multiplier[~np.isnan(result_ultimate.dynamic_multiplier)]):.2f}")
    
    # 比較分析
    print("\n📈 比較分析結果")
    print("-" * 40)
    
    atr_signals = int(np.sum(np.abs(result_atr.breakout_signals)))
    ultimate_signals = int(np.sum(np.abs(result_ultimate.breakout_signals)))
    
    atr_avg_quality = np.nanmean(result_atr.signal_quality[result_atr.signal_quality > 0])
    ultimate_avg_quality = np.nanmean(result_ultimate.signal_quality[result_ultimate.signal_quality > 0])
    
    print(f"シグナル数比較:")
    print(f"  ATR版: {atr_signals}")
    print(f"  Ultimate版: {ultimate_signals}")
    if atr_signals > 0:
        print(f"  差分: {ultimate_signals - atr_signals} ({((ultimate_signals - atr_signals) / atr_signals * 100):+.1f}%)")
    else:
        print(f"  ATR版でシグナルが検出されませんでした")
    
    print(f"\n品質比較:")
    if not np.isnan(atr_avg_quality):
        print(f"  ATR版: {atr_avg_quality:.3f}")
    else:
        print(f"  ATR版: シグナル品質なし")
    
    if not np.isnan(ultimate_avg_quality):
        print(f"  Ultimate版: {ultimate_avg_quality:.3f}")
    else:
        print(f"  Ultimate版: シグナル品質なし")
    
    if not np.isnan(atr_avg_quality) and not np.isnan(ultimate_avg_quality) and atr_avg_quality > 0:
        print(f"  改善度: {((ultimate_avg_quality - atr_avg_quality) / atr_avg_quality * 100):+.1f}%")
    else:
        print(f"  品質比較: データ不足のため計算不可")
    
    # チャート生成
    create_comparison_chart(test_data, result_atr, result_ultimate)
    
    return result_atr, result_ultimate

def create_comparison_chart(data, result_atr, result_ultimate):
    """比較チャート生成"""
    print("\n📊 比較チャート生成中...")
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle('Ultimate Breakout Channel: ATR vs Ultimate Volatility 比較', 
                 fontsize=16, fontweight='bold')
    
    prices = data['close'].values
    x = range(len(prices))
    
    # 価格とチャネル比較
    axes[0].plot(x, prices, label='価格', color='black', linewidth=1)
    axes[0].plot(x, result_atr.upper_channel, label='ATR上部', color='blue', alpha=0.7)
    axes[0].plot(x, result_atr.lower_channel, label='ATR下部', color='blue', alpha=0.7)
    axes[0].fill_between(x, result_atr.upper_channel, result_atr.lower_channel, 
                        alpha=0.1, color='blue', label='ATRチャネル')
    
    axes[0].plot(x, result_ultimate.upper_channel, label='Ultimate上部', color='red', alpha=0.7)
    axes[0].plot(x, result_ultimate.lower_channel, label='Ultimate下部', color='red', alpha=0.7)
    axes[0].fill_between(x, result_ultimate.upper_channel, result_ultimate.lower_channel, 
                        alpha=0.1, color='red', label='Ultimateチャネル')
    
    axes[0].set_title('価格とチャネル比較')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 動的乗数比較
    axes[1].plot(x, result_atr.dynamic_multiplier, label='ATR動的乗数', color='blue')
    axes[1].plot(x, result_ultimate.dynamic_multiplier, label='Ultimate動的乗数', color='red')
    axes[1].set_title('動的乗数比較')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # シグナル品質比較
    axes[2].plot(x, result_atr.signal_quality, label='ATRシグナル品質', color='blue', alpha=0.7)
    axes[2].plot(x, result_ultimate.signal_quality, label='Ultimateシグナル品質', color='red', alpha=0.7)
    
    # ブレイクアウトシグナル
    signal_atr = np.where(result_atr.breakout_signals != 0)[0]
    signal_ultimate = np.where(result_ultimate.breakout_signals != 0)[0]
    
    if len(signal_atr) > 0:
        axes[2].scatter(signal_atr, [0.1] * len(signal_atr), 
                       color='blue', marker='^', s=50, label='ATRシグナル')
    
    if len(signal_ultimate) > 0:
        axes[2].scatter(signal_ultimate, [0.05] * len(signal_ultimate), 
                       color='red', marker='v', s=50, label='Ultimateシグナル')
    
    axes[2].set_title('シグナル品質とブレイクアウト比較')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, 1)
    
    plt.tight_layout()
    
    # 保存
    output_file = 'examples/output/ultimate_breakout_channel_volatility_comparison.png'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 比較チャート保存: {output_file}")
    
    plt.show()

def test_with_real_data():
    """実データテスト（オプション）"""
    print("\n🌐 実データテスト（BTC/USDT）")
    print("-" * 40)
    
    try:
        # Binanceデータソース使用
        data_source = BinanceDataSource()
        
        # 最近1000本のデータ取得
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1000)
        
        btc_data = data_source.fetch_ohlcv('BTCUSDT', '1h', start_time, end_time)
        
        if btc_data is not None and len(btc_data) > 100:
            print(f"実データ取得成功: {len(btc_data)}本")
            
            # ATR版
            ubc_atr = UltimateBreakoutChannel(volatility_type='atr')
            result_atr = ubc_atr.calculate(btc_data)
            
            # Ultimate版
            ubc_ultimate = UltimateBreakoutChannel(volatility_type='ultimate')
            result_ultimate = ubc_ultimate.calculate(btc_data)
            
            print(f"ATR版シグナル数: {int(np.sum(np.abs(result_atr.breakout_signals)))}")
            print(f"Ultimate版シグナル数: {int(np.sum(np.abs(result_ultimate.breakout_signals)))}")
            
            # 実データチャート生成
            create_comparison_chart(btc_data, result_atr, result_ultimate)
            
        else:
            print("⚠️  実データ取得に失敗（テストデータを使用）")
            
    except Exception as e:
        print(f"⚠️  実データテストエラー: {e}")
        print("テストデータで継続...")

def main():
    """メイン実行"""
    print("🚀 Ultimate Breakout Channel ボラティリティ選択機能テスト開始")
    print("=" * 70)
    
    try:
        # 基本テスト
        result_atr, result_ultimate = test_volatility_types()
        
        # レポート生成
        print("\n📋 最終レポート")
        print("=" * 40)
        
        # 安全な平均計算関数
        def safe_mean(values, condition=None):
            if condition is not None:
                filtered = values[condition]
            else:
                filtered = values[~np.isnan(values)]
            return float(np.mean(filtered)) if len(filtered) > 0 else 0.0
        
        atr_report = {
            'volatility_type': 'ATR',
            'signals': int(np.sum(np.abs(result_atr.breakout_signals))),
            'avg_quality': safe_mean(result_atr.signal_quality, result_atr.signal_quality > 0),
            'avg_multiplier': safe_mean(result_atr.dynamic_multiplier),
            'current_trend': result_atr.current_trend,
            'current_regime': result_atr.current_regime
        }
        
        ultimate_report = {
            'volatility_type': 'Ultimate Volatility',
            'signals': int(np.sum(np.abs(result_ultimate.breakout_signals))),
            'avg_quality': safe_mean(result_ultimate.signal_quality, result_ultimate.signal_quality > 0),
            'avg_multiplier': safe_mean(result_ultimate.dynamic_multiplier),
            'current_trend': result_ultimate.current_trend,
            'current_regime': result_ultimate.current_regime
        }
        
        for report in [atr_report, ultimate_report]:
            print(f"\n{report['volatility_type']}版:")
            print(f"  シグナル数: {report['signals']}")
            print(f"  平均品質: {report['avg_quality']:.3f}")
            print(f"  平均乗数: {report['avg_multiplier']:.2f}")
            print(f"  現在トレンド: {report['current_trend']}")
            print(f"  現在レジーム: {report['current_regime']}")
        
        # 推奨事項
        print("\n💡 使用推奨事項:")
        print("- 高精度が必要な場合: volatility_type='ultimate'")
        print("- 高速処理が必要な場合: volatility_type='atr'")
        print("- デフォルト推奨: volatility_type='ultimate'")
        
        print("\n✅ テスト完了！")
        
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 