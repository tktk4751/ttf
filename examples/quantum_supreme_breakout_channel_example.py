#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🌌 Quantum Supreme Breakout Channel V1.0 - 使用例
人類史上最強ボラティリティベースブレイクアウトチャネル サンプル実行

このスクリプトは、Quantum Supreme Breakout Channelインジケーターの基本的な使用方法を示します。
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from indicators.quantum_supreme_breakout_channel import QuantumSupremeBreakoutChannel


def generate_sample_data(n_points: int = 1000, trend_strength: float = 0.3, volatility: float = 0.02) -> pd.DataFrame:
    """
    テスト用のサンプル価格データを生成
    
    Args:
        n_points: データポイント数
        trend_strength: トレンド強度
        volatility: ボラティリティ
    
    Returns:
        OHLCV形式のDataFrame
    """
    print(f"📊 サンプルデータ生成中... (点数: {n_points}, トレンド強度: {trend_strength}, ボラティリティ: {volatility})")
    
    # ベースとなる価格系列を生成（トレンド + ランダムウォーク）
    np.random.seed(42)  # 再現性のため
    
    # 時間軸
    dates = pd.date_range(start='2023-01-01', periods=n_points, freq='1H')
    
    # トレンド成分
    trend = np.linspace(0, trend_strength * n_points, n_points)
    
    # ランダムウォーク成分
    random_walk = np.cumsum(np.random.normal(0, volatility, n_points))
    
    # サイクル成分（複数の周期を重ね合わせ）
    cycle1 = 0.5 * np.sin(2 * np.pi * np.arange(n_points) / 50)    # 50時間周期
    cycle2 = 0.3 * np.sin(2 * np.pi * np.arange(n_points) / 200)   # 200時間周期
    cycle3 = 0.2 * np.sin(2 * np.pi * np.arange(n_points) / 20)    # 20時間周期
    
    # ボラティリティクラスタリング
    volatility_multiplier = 1 + 0.5 * np.sin(2 * np.pi * np.arange(n_points) / 100)
    
    # 基準価格（100から開始）
    base_price = 100
    close_prices = base_price + trend + random_walk + cycle1 + cycle2 + cycle3
    
    # OHLC生成
    data = []
    for i in range(n_points):
        close = close_prices[i]
        
        # 日内変動を生成
        intraday_range = abs(np.random.normal(0, volatility * volatility_multiplier[i] * 10))
        
        high = close + intraday_range * np.random.uniform(0.3, 0.7)
        low = close - intraday_range * np.random.uniform(0.3, 0.7)
        
        # 前日終値から開始価格を決定
        if i == 0:
            open_price = close
        else:
            gap = np.random.normal(0, volatility * 0.5)
            open_price = data[i-1]['close'] + gap
        
        # 高値・安値の調整
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        # 出来高（価格変動に連動）
        volume = abs(np.random.normal(1000, 200)) * (1 + abs(close - (data[i-1]['close'] if i > 0 else close)) * 10)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    
    print(f"✅ サンプルデータ生成完了")
    print(f"   期間: {df.index.min()} → {df.index.max()}")
    print(f"   価格範囲: {df['close'].min():.2f} - {df['close'].max():.2f}")
    print(f"   平均出来高: {df['volume'].mean():.0f}")
    
    return df


def run_quantum_supreme_breakout_channel_example():
    """
    🌌 Quantum Supreme Breakout Channel の使用例を実行
    """
    print("🌌 Quantum Supreme Breakout Channel V1.0 - 使用例")
    print("=" * 60)
    
    # 1. サンプルデータ生成
    sample_data = generate_sample_data(
        n_points=500,           # 500時間分のデータ
        trend_strength=0.5,     # 中程度のトレンド
        volatility=0.03         # 3%のボラティリティ
    )
    
    # 2. Quantum Supreme Breakout Channel の初期化
    print("\n🚀 Quantum Supreme Breakout Channel 初期化...")
    qsbc = QuantumSupremeBreakoutChannel(
        # 基本設定
        analysis_period=21,
        src_type='hlc3',
        min_multiplier=1.5,
        max_multiplier=8.0,
        
        # 量子パラメータ
        quantum_coherence_threshold=0.75,
        entanglement_factor=0.618,
        
        # 適応パラメータ
        trend_sensitivity=0.85,
        range_sensitivity=0.75,
        ultra_low_latency=True,
        
        # アルゴリズム有効化（デモ用に一部無効化してパフォーマンス向上）
        enable_quantum_hilbert=True,
        enable_fractal_analysis=True,
        enable_kalman_quantum=False,    # Ultimate MAを無効化（依存関係エラー回避）
        enable_garch_volatility=False,  # Ultimate Volatilityを無効化
        enable_regime_switching=False,  # Ultimate Chop Trendを無効化
        enable_efficiency_ratio=False,  # Efficiency Ratioを無効化
        enable_spectral_analysis=False  # Ehlers Cycleを無効化
    )
    
    # 3. 計算実行
    print("\n🌊 計算実行中...")
    try:
        result = qsbc.calculate(sample_data)
        print("✅ 計算完了!")
        
        # 4. 結果の表示
        print("\n📊 === 結果サマリー ===")
        print(f"🎯 動的乗数範囲: {np.min(result.dynamic_multiplier):.2f} - {np.max(result.dynamic_multiplier):.2f}")
        print(f"🌀 現在のレジーム: {result.current_regime}")
        print(f"💪 現在のトレンド強度: {result.current_trend_strength:.3f}")
        print(f"🚀 現在のブレイクアウト確率: {result.current_breakout_probability:.1%}")
        print(f"🎛️ 現在の適応モード: {result.current_adaptation_mode}")
        
        # 5. 統計情報
        print(f"\n📈 === 統計情報 ===")
        print(f"データ点数: {len(result.dynamic_multiplier)}")
        
        # 市場レジーム分布
        regime_counts = np.bincount(result.market_regime.astype(int))
        total_points = len(result.market_regime)
        print(f"市場レジーム分布:")
        regime_names = ['レンジ', 'トレンド', 'ブレイクアウト']
        for i, count in enumerate(regime_counts):
            if count > 0:
                print(f"  {regime_names[i]}: {count} ({count/total_points*100:.1f}%)")
        
        # ブレイクアウトシグナル
        breakout_signals = result.breakout_signals
        total_breakouts = np.sum(breakout_signals != 0)
        up_breakouts = np.sum(breakout_signals == 1)
        down_breakouts = np.sum(breakout_signals == -1)
        print(f"ブレイクアウトシグナル:")
        print(f"  総数: {total_breakouts}")
        print(f"  上抜け: {up_breakouts}, 下抜け: {down_breakouts}")
        
        # 量子メトリクス
        print(f"量子メトリクス平均値:")
        print(f"  コヒーレンス: {np.mean(result.quantum_coherence):.3f}")
        print(f"  もつれ: {np.mean(result.quantum_entanglement):.3f}")
        print(f"  重ね合わせ: {np.mean(result.superposition_state):.3f}")
        
        # 6. 簡単なチャート表示
        print("\n🎨 チャート表示...")
        plot_simple_chart(sample_data, result)
        
        # 7. 個別データアクセス例
        print("\n🔍 === 個別データアクセス例 ===")
        upper_channel = qsbc.get_upper_channel()
        middle_line = qsbc.get_middle_line()
        lower_channel = qsbc.get_lower_channel()
        dynamic_multiplier = qsbc.get_dynamic_multiplier()
        
        print(f"最新チャネル値:")
        print(f"  上位: {upper_channel[-1]:.2f}")
        print(f"  中央: {middle_line[-1]:.2f}")
        print(f"  下位: {lower_channel[-1]:.2f}")
        print(f"  動的乗数: {dynamic_multiplier[-1]:.2f}")
        
        # 現在の状態
        current_status = qsbc.get_current_status()
        print(f"現在の状態: {current_status}")
        
        print("\n🌌 Quantum Supreme Breakout Channel 使用例完了!")
        
    except Exception as e:
        import traceback
        print(f"❌ エラーが発生しました: {e}")
        print(traceback.format_exc())


def plot_simple_chart(data: pd.DataFrame, result) -> None:
    """
    シンプルなチャートを表示
    
    Args:
        data: 価格データ
        result: QSBC計算結果
    """
    try:
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # 1. 価格とチャネル
        axes[0].plot(data.index, data['close'], label='Close Price', color='black', linewidth=1)
        axes[0].plot(data.index, result.upper_channel, label='Upper Channel', color='red', alpha=0.7)
        axes[0].plot(data.index, result.middle_line, label='Middle Line', color='blue', linewidth=2)
        axes[0].plot(data.index, result.lower_channel, label='Lower Channel', color='green', alpha=0.7)
        
        # ブレイクアウトシグナル
        breakout_up = np.where(result.breakout_signals == 1, data['high'] * 1.01, np.nan)
        breakout_down = np.where(result.breakout_signals == -1, data['low'] * 0.99, np.nan)
        axes[0].scatter(data.index, breakout_up, marker='^', color='red', s=100, alpha=0.8, label='Breakout Up')
        axes[0].scatter(data.index, breakout_down, marker='v', color='green', s=100, alpha=0.8, label='Breakout Down')
        
        axes[0].set_title('🌌 Quantum Supreme Breakout Channel')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. 動的乗数と市場レジーム
        axes[1].plot(data.index, result.dynamic_multiplier, label='Dynamic Multiplier', color='blue', linewidth=2)
        axes[1].axhline(y=1.5, color='green', linestyle='--', alpha=0.5, label='Min Multiplier')
        axes[1].axhline(y=8.0, color='red', linestyle='--', alpha=0.5, label='Max Multiplier')
        axes[1].axhline(y=4.75, color='black', linestyle='-', alpha=0.3, label='Neutral')
        
        # 市場レジーム（右軸）
        ax1_twin = axes[1].twinx()
        ax1_twin.plot(data.index, result.market_regime, label='Market Regime', color='orange', alpha=0.7)
        ax1_twin.set_ylabel('Market Regime (0:Range, 1:Trend, 2:Breakout)')
        
        axes[1].set_title('動的乗数 & 市場レジーム')
        axes[1].legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        
        # 3. ブレイクアウト確率と量子メトリクス
        axes[2].plot(data.index, result.breakout_probability, label='Breakout Probability', color='red', linewidth=2)
        axes[2].plot(data.index, result.quantum_coherence, label='Quantum Coherence', color='cyan', alpha=0.7)
        axes[2].plot(data.index, result.trend_strength, label='Trend Strength', color='purple', alpha=0.7)
        
        axes[2].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
        axes[2].axhline(y=0.8, color='red', linestyle='--', alpha=0.3)
        
        axes[2].set_title('ブレイクアウト確率 & 量子メトリクス')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        output_path = project_root / 'examples' / 'output' / 'quantum_supreme_breakout_channel_example.png'
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"💾 チャートを保存しました: {output_path}")
        
        # 表示
        plt.show()
        
    except Exception as e:
        print(f"⚠️ チャート表示エラー: {e}")


if __name__ == "__main__":
    run_quantum_supreme_breakout_channel_example() 