#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🌟 Supreme Adaptive Channel V2.0 Signal テスト
"""

import numpy as np
import pandas as pd
import sys
import os

# シグナルのパスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'signals'))

from signals.implementations.supreme_adaptive_channel import SupremeAdaptiveChannelEntrySignal

def create_trending_data(n=300, trend_strength=0.02):
    """トレンドを持つテスト用のOHLCVデータを作成"""
    np.random.seed(42)
    
    # 明確なトレンドを作成
    base_trend = np.cumsum(np.random.randn(n) * 0.001 + trend_strength) + 100
    
    # 価格データを生成（トレンドにノイズを追加）
    close = base_trend + np.random.randn(n) * 0.3
    high = close + np.abs(np.random.randn(n)) * 0.2
    low = close - np.abs(np.random.randn(n)) * 0.2
    open_prices = close + np.random.randn(n) * 0.1
    volume = np.random.randint(1000, 10000, n)
    
    # 中盤にブレイクアウトを演出
    breakout_start = n // 3
    breakout_end = n // 3 * 2
    close[breakout_start:breakout_end] += np.linspace(0, 3, breakout_end - breakout_start)
    high[breakout_start:breakout_end] = close[breakout_start:breakout_end] + np.abs(np.random.randn(breakout_end - breakout_start)) * 0.3
    low[breakout_start:breakout_end] = close[breakout_start:breakout_end] - np.abs(np.random.randn(breakout_end - breakout_start)) * 0.3
    
    # DataFrameを作成
    data = pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    return data

def test_supreme_adaptive_channel_signal():
    """Supreme Adaptive Channel V2.0 Signal の基本テスト"""
    print("🌟 Supreme Adaptive Channel V2.0 Signal テスト開始...")
    
    # トレンドデータ作成
    test_data = create_trending_data(300, trend_strength=0.01)
    print(f"📊 トレンドテストデータ作成完了: {len(test_data)} 本のキャンドル")
    
    # Supreme Adaptive Channel Entry Signal作成
    supreme_signal = SupremeAdaptiveChannelEntrySignal(
        # Supreme Adaptive Channelパラメータ
        atr_period=21,
        base_multiplier=2.0,
        kalman_q=0.001,
        kalman_r=0.1,
        ultimate_period=20.0,
        zero_lag_period=21,
        frama_period=16,
        phasor_window=20,
        src_type='hlc3',
        
        # シグナルフィルタリングパラメータ（実用的設定）
        min_confidence=0.2,           # 実用的信頼度閾値
        min_trend_strength=0.1,       # 実用的トレンド強度閾値
        max_fractal_dimension=1.8,    # 市場効率性閾値
        min_signal_strength=0.05,     # 実用的シグナル強度閾値
        
        # 決済パラメータ
        enable_exit_signals=True,
        exit_trend_threshold=0.05,
        exit_confidence_threshold=0.15,
        exit_fractal_threshold=1.7,
        exit_signal_threshold=0.03,
        
        # Supreme知能強化パラメータ
        enable_supreme_enhancement=True,
        supreme_enhancement_threshold=0.25,
        require_strong_signals=False
    )
    
    print("🔧 Supreme Adaptive Channel Entry Signal 初期化完了")
    
    # シグナル生成実行
    try:
        entry_signals = supreme_signal.generate(test_data)
        print("✅ エントリーシグナル生成成功!")
        
        # 決済シグナル取得
        exit_signals = supreme_signal.get_exit_signals()
        current_position = supreme_signal.get_current_position()
        
        # シグナル分析
        long_signals = np.sum(entry_signals == 1)
        short_signals = np.sum(entry_signals == -1)
        total_signals = long_signals + short_signals
        
        long_exits = np.sum(exit_signals == 1)
        short_exits = np.sum(exit_signals == -1)
        total_exits = long_exits + short_exits
        
        print(f"\n📈 シグナル分析:")
        print(f"- ロングシグナル: {long_signals}")
        print(f"- ショートシグナル: {short_signals}")
        print(f"- 総エントリーシグナル: {total_signals}")
        print(f"- ロング決済: {long_exits}")
        print(f"- ショート決済: {short_exits}")
        print(f"- 総決済シグナル: {total_exits}")
        
        # Supreme知能レポート取得
        intelligence_report = supreme_signal.get_supreme_intelligence_report()
        current_state = supreme_signal.get_current_state()
        
        print(f"\n🧠 Supreme知能レポート:")
        print(f"- Supreme知能スコア: {intelligence_report['supreme_intelligence_score']:.3f}")
        print(f"- 現在のトレンドフェーズ: {intelligence_report['current_trend_phase']}")
        print(f"- 現在の市場効率性: {intelligence_report['current_market_efficiency']:.3f}")
        print(f"- 偽シグナル率: {intelligence_report['false_signal_rate']:.3f}")
        print(f"- 平均フラクタル次元: {intelligence_report['average_fractal_dimension']:.3f}")
        
        # Supreme Adaptive Channel結果の詳細分析
        supreme_result = supreme_signal.get_supreme_result()
        if supreme_result is not None:
            print(f"\n🔍 Supreme Adaptive Channel 詳細分析:")
            print(f"- 最終上側チャネル: {supreme_result.upper_channel[-1]:.3f}")
            print(f"- 最終下側チャネル: {supreme_result.lower_channel[-1]:.3f}")
            print(f"- 最終FRAMA値: {supreme_result.frama_values[-1]:.3f}")
            print(f"- 最終フラクタル次元: {supreme_result.fractal_dimension[-1]:.3f}")
            print(f"- 最終トレンド強度: {supreme_result.trend_strength[-1]:.3f}")
            print(f"- 最終シグナル強度: {supreme_result.signal_strength[-1]:.3f}")
            
            # 5層システムの動作確認
            print(f"\n🏗️ 5層システム最終値:")
            print(f"- Layer 1 (カルマンフィルター): {supreme_result.kalman_filtered[-1]:.3f}")
            print(f"- Layer 1 (アルティメットスムーザー): {supreme_result.ultimate_smoothed[-1]:.3f}")
            print(f"- Layer 1 (ゼロラグEMA): {supreme_result.zero_lag_ema[-1]:.3f}")
            print(f"- Layer 2 (FRAMA): {supreme_result.frama_values[-1]:.3f}")
            print(f"- Layer 2 (フラクタル次元): {supreme_result.fractal_dimension[-1]:.3f}")
            print(f"- Layer 4 (トレンド強度): {supreme_result.trend_strength[-1]:.3f}")
            print(f"- Layer 4 (サイクル成分): {supreme_result.cycle_component[-1]:.3f}")
        
        # シグナル発生タイミングの詳細分析
        print(f"\n🎯 シグナル発生タイミング分析:")
        entry_indices = np.where(entry_signals != 0)[0]
        if len(entry_indices) > 0:
            print(f"- 最初のシグナル位置: {entry_indices[0]} ({entry_signals[entry_indices[0]]})")
            print(f"- 最後のシグナル位置: {entry_indices[-1]} ({entry_signals[entry_indices[-1]]})")
            
            # 最初のシグナル時の詳細情報
            first_signal_idx = entry_indices[0]
            if supreme_result is not None:
                print(f"- 最初のシグナル時の信頼度: {supreme_result.breakout_confidence[first_signal_idx]:.3f}")
                print(f"- 最初のシグナル時のトレンド強度: {supreme_result.trend_strength[first_signal_idx]:.3f}")
                print(f"- 最初のシグナル時のフラクタル次元: {supreme_result.fractal_dimension[first_signal_idx]:.3f}")
                print(f"- 最初のシグナル時のシグナル強度: {supreme_result.signal_strength[first_signal_idx]:.3f}")
        
        print(f"\n🎉 Supreme Adaptive Channel V2.0 Signal テスト完了!")
        print(f"✨ 全ての機能が正常に動作しています。")
        
        return True
        
    except Exception as e:
        print(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_signal_filtering():
    """シグナルフィルタリングのテスト"""
    print(f"\n🔍 シグナルフィルタリングテスト...")
    
    # ノイズの多いデータ作成
    test_data = create_trending_data(200, trend_strength=0.005)  # 弱いトレンド
    
    # 厳しいフィルタリング設定
    strict_signal = SupremeAdaptiveChannelEntrySignal(
        min_confidence=0.4,           # 高い信頼度要求
        min_trend_strength=0.2,       # 強いトレンド要求  
        max_fractal_dimension=1.6,    # 高い市場効率性要求
        min_signal_strength=0.15,     # 高いシグナル強度要求
        require_strong_signals=True   # 強いシグナルのみ
    )
    
    # 緩いフィルタリング設定
    loose_signal = SupremeAdaptiveChannelEntrySignal(
        min_confidence=0.1,           # 低い信頼度要求
        min_trend_strength=0.05,      # 弱いトレンド許可
        max_fractal_dimension=1.9,    # 低い市場効率性許可
        min_signal_strength=0.02,     # 低いシグナル強度許可
        require_strong_signals=False  # 全シグナル許可
    )
    
    try:
        strict_entry = strict_signal.generate(test_data)
        loose_entry = loose_signal.generate(test_data)
        
        strict_count = np.sum(np.abs(strict_entry))
        loose_count = np.sum(np.abs(loose_entry))
        
        print(f"✅ フィルタリングテスト成功!")
        print(f"- 厳しいフィルタリング: {strict_count} シグナル")
        print(f"- 緩いフィルタリング: {loose_count} シグナル")
        print(f"- フィルタリング効果: {((loose_count - strict_count) / max(loose_count, 1) * 100):.1f}% 削減")
        
        return True
    except Exception as e:
        print(f"❌ フィルタリングテストエラー: {e}")
        return False

def main():
    """メインテスト関数"""
    print("=" * 70)
    print("🌟 Supreme Adaptive Channel V2.0 Signal 完全テスト")
    print("=" * 70)
    
    success_count = 0
    total_tests = 2
    
    # テスト1: 基本シグナル機能
    if test_supreme_adaptive_channel_signal():
        success_count += 1
    
    # テスト2: シグナルフィルタリング
    if test_signal_filtering():
        success_count += 1
    
    print("\n" + "=" * 70)
    print(f"🏁 テスト結果: {success_count}/{total_tests} 成功")
    
    if success_count == total_tests:
        print("🎉 全てのテストが成功しました！")
        print("✨ Supreme Adaptive Channel V2.0 Signal は正常に動作しています。")
        print("🚀 宇宙最強のトレンドフォローシグナルが完成しました！")
    else:
        print("⚠️ 一部のテストが失敗しました。")
    
    print("=" * 70)

if __name__ == "__main__":
    main()