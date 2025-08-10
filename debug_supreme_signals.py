#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🌟 Supreme Adaptive Channel シグナル デバッグテスト
"""

import numpy as np
import pandas as pd
import sys
import os

# ストラテジーのパスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'strategies'))

from strategies.implementations.supreme_adaptive_channel import (
    SupremeAdaptiveChannelStrategy,
    SupremeAdaptiveChannelSignalGenerator
)

def create_strong_trending_data(n=200, trend_strength=0.05):
    """強いトレンドデータを作成"""
    np.random.seed(42)
    
    # 強いトレンドを作成
    base_trend = np.cumsum(np.random.randn(n) * 0.01 + trend_strength) + 100
    
    # 価格データを生成
    close = base_trend + np.random.randn(n) * 0.5
    high = close + np.abs(np.random.randn(n)) * 0.3
    low = close - np.abs(np.random.randn(n)) * 0.3
    open_prices = close + np.random.randn(n) * 0.1
    volume = np.random.randint(1000, 10000, n)
    
    # 明確なブレイクアウトを追加
    breakout_points = [50, 100, 150]
    for point in breakout_points:
        if point < n:
            close[point:point+10] += 3.0  # 強いブレイクアウト
            high[point:point+10] = close[point:point+10] + 0.5
            low[point:point+10] = close[point:point+10] - 0.5
    
    # DataFrameを作成
    data = pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    return data

def debug_signal_filtering():
    """シグナルフィルタリングのデバッグ"""
    print("🔍 Supreme Adaptive Channel シグナル フィルタリング デバッグ...")
    
    # 強いトレンドデータ作成
    test_data = create_strong_trending_data(200, trend_strength=0.08)
    print(f"📊 強いトレンドデータ作成完了: {len(test_data)} 本のキャンドル")
    
    # 最も緩い設定でテスト
    supreme_strategy = SupremeAdaptiveChannelStrategy(
        # Supreme Adaptive Channelパラメータ
        atr_period=14,          # 短期ATR
        base_multiplier=1.5,    # 狭いチャネル
        kalman_q=0.01,          # 敏感なカルマン
        kalman_r=0.01,          # 敏感なカルマン
        ultimate_period=10.0,   # 短期平滑化
        zero_lag_period=10,     # 短期EMA
        frama_period=8,         # 短期FRAMA
        phasor_window=10,       # 短期フェーザー
        src_type='close',       # シンプルな価格ソース
        
        # 極めて緩いフィルタリング設定
        min_confidence=0.01,           # 極めて低い信頼度閾値
        min_trend_strength=0.01,       # 極めて低いトレンド強度閾値
        max_fractal_dimension=1.99,    # 極めて緩い市場効率性閾値
        min_signal_strength=0.001,     # 極めて低いシグナル強度閾値
        
        # 決済パラメータ
        enable_exit_signals=True,
        exit_trend_threshold=0.01,
        exit_confidence_threshold=0.01,
        exit_fractal_threshold=1.95,
        exit_signal_threshold=0.001,
        
        # Supreme知能強化パラメータ
        enable_supreme_enhancement=False,  # 強化を無効化
        supreme_enhancement_threshold=0.01,
        require_strong_signals=False       # 強いシグナル要求を無効化
    )
    
    print("🔧 極めて緩い設定でSupreme Adaptive Channel Strategy 初期化完了")
    
    try:
        # エントリーシグナル生成実行
        entry_signals = supreme_strategy.generate_entry(test_data)
        print("✅ エントリーシグナル生成成功!")
        
        # シグナル分析
        long_signals = np.sum(entry_signals == 1)
        short_signals = np.sum(entry_signals == -1)
        total_signals = long_signals + short_signals
        
        print(f"\n📈 極めて緩い設定でのシグナル分析:")
        print(f"- ロングシグナル: {long_signals}")
        print(f"- ショートシグナル: {short_signals}")
        print(f"- 総エントリーシグナル: {total_signals}")
        print(f"- シグナル密度: {total_signals/len(test_data)*100:.1f}%")
        
        # Supreme知能レポート取得
        intelligence_report = supreme_strategy.get_supreme_intelligence_report(test_data)
        print(f"\n🧠 Supreme知能レポート（極めて緩い設定）:")
        print(f"- Supreme知能スコア: {intelligence_report['supreme_intelligence_score']:.6f}")
        print(f"- 現在のトレンドフェーズ: {intelligence_report['current_trend_phase']}")
        print(f"- 現在の市場効率性: {intelligence_report['current_market_efficiency']:.6f}")
        print(f"- 偽シグナル率: {intelligence_report['false_signal_rate']:.6f}")
        print(f"- 平均フラクタル次元: {intelligence_report['average_fractal_dimension']:.6f}")
        
        # Supreme指標の詳細分析
        supreme_indicators = supreme_strategy.get_supreme_indicators(test_data)
        print(f"\n🔍 Supreme指標詳細分析（極めて緩い設定）:")
        print(f"- 最終ブレイクアウト信頼度: {supreme_indicators['breakout_confidence'][-1]:.6f}")
        print(f"- 最終トレンド強度: {supreme_indicators['trend_strength'][-1]:.6f}")
        print(f"- 最終フラクタル次元: {supreme_indicators['fractal_dimension'][-1]:.6f}")
        print(f"- 最終シグナル強度: {supreme_indicators['signal_strength'][-1]:.6f}")
        print(f"- 最終市場効率性: {supreme_indicators['market_efficiency'][-1]:.6f}")
        
        # ブレイクアウトシグナルの詳細分析
        if hasattr(supreme_strategy.signal_generator, '_supreme_result') and supreme_strategy.signal_generator._supreme_result:
            result = supreme_strategy.signal_generator._supreme_result
            print(f"\n📊 生の計算結果分析:")
            print(f"- ブレイクアウトシグナル合計: {np.sum(np.abs(result.breakout_signals))}")
            print(f"- 信頼度 > 0 の数: {np.sum(result.breakout_confidence > 0)}")
            print(f"- 偽シグナルフィルター通過数: {np.sum(result.false_signal_filter)}")
            print(f"- トレンド強度 > 0.01 の数: {np.sum(np.abs(result.trend_strength) > 0.01)}")
            print(f"- フラクタル次元 < 1.99 の数: {np.sum(result.fractal_dimension < 1.99)}")
            print(f"- シグナル強度 > 0.001 の数: {np.sum(result.signal_strength > 0.001)}")
            
            # フィルタリングの段階的分析
            basic_signals = result.breakout_signals != 0
            confidence_pass = result.breakout_confidence >= 0.01
            trend_pass = np.abs(result.trend_strength) >= 0.01
            fractal_pass = result.fractal_dimension <= 1.99
            strength_pass = result.signal_strength >= 0.001
            filter_pass = result.false_signal_filter == 1
            
            print(f"\n🎯 フィルタリング段階分析:")
            print(f"- 基本シグナル: {np.sum(basic_signals)}")
            print(f"- 信頼度フィルター通過: {np.sum(basic_signals & confidence_pass)}")
            print(f"- トレンド強度フィルター通過: {np.sum(basic_signals & confidence_pass & trend_pass)}")
            print(f"- フラクタル次元フィルター通過: {np.sum(basic_signals & confidence_pass & trend_pass & fractal_pass)}")
            print(f"- シグナル強度フィルター通過: {np.sum(basic_signals & confidence_pass & trend_pass & fractal_pass & strength_pass)}")
            print(f"- 偽シグナルフィルター通過: {np.sum(basic_signals & filter_pass)}")
            print(f"- 最終シグナル: {total_signals}")
        
        return total_signals > 0
        
    except Exception as e:
        print(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインデバッグ関数"""
    print("=" * 80)
    print("🔍 Supreme Adaptive Channel V2.0 シグナル デバッグテスト")
    print("=" * 80)
    
    success = debug_signal_filtering()
    
    print("\n" + "=" * 80)
    if success:
        print("🎉 デバッグテスト完了！シグナルが生成されています。")
    else:
        print("⚠️ デバッグテスト完了。シグナル生成に問題があります。")
    print("=" * 80)

if __name__ == "__main__":
    main()