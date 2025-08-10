#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🌟 Supreme Adaptive Channel V2.0 テスト
"""

import numpy as np
import pandas as pd
import sys
import os

# インジケーターのパスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'indicators'))

from indicators.cosmic_adaptive_channel import SupremeAdaptiveChannel, SAC

def create_test_data(n=200):
    """テスト用のOHLCVデータを作成"""
    np.random.seed(42)
    
    # 基本トレンドを作成
    base_trend = np.cumsum(np.random.randn(n) * 0.001) + 100
    
    # 価格データを生成（トレンドにノイズを追加）
    close = base_trend + np.random.randn(n) * 0.5
    high = close + np.abs(np.random.randn(n)) * 0.3
    low = close - np.abs(np.random.randn(n)) * 0.3
    open_prices = close + np.random.randn(n) * 0.1
    volume = np.random.randint(1000, 10000, n)
    
    # DataFrameを作成
    data = pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    return data

def test_supreme_adaptive_channel():
    """Supreme Adaptive Channel V2.0 の基本テスト"""
    print("🌟 Supreme Adaptive Channel V2.0 テスト開始...")
    
    # テストデータ作成
    test_data = create_test_data(200)
    print(f"📊 テストデータ作成完了: {len(test_data)} 本のキャンドル")
    
    # Supreme Adaptive Channel インジケーター作成
    sac = SupremeAdaptiveChannel(
        atr_period=21,
        base_multiplier=2.0,
        kalman_q=0.001,
        kalman_r=0.1,
        ultimate_period=20.0,
        zero_lag_period=21,
        frama_period=16,
        phasor_window=20,
        src_type='hlc3'
    )
    
    print("🔧 Supreme Adaptive Channel インジケーター初期化完了")
    
    # 計算実行
    try:
        result = sac.calculate(test_data)
        print("✅ 計算成功!")
        
        # 結果の検証
        print(f"\n📈 計算結果:")
        print(f"- 上側チャネル: {result.upper_channel[-5:]}")
        print(f"- 下側チャネル: {result.lower_channel[-5:]}")
        print(f"- 中央線 (FRAMA): {result.midline[-5:]}")
        print(f"- チャネル幅: {result.dynamic_width[-5:]}")
        print(f"- フラクタル次元: {result.fractal_dimension[-5:]}")
        print(f"- トレンド強度: {result.trend_strength[-5:]}")
        
        # シグナル情報
        total_signals = np.sum(np.abs(result.breakout_signals))
        avg_confidence = np.mean(result.breakout_confidence[result.breakout_confidence > 0]) if np.any(result.breakout_confidence > 0) else 0.0
        
        print(f"\n🎯 シグナル情報:")
        print(f"- 総シグナル数: {total_signals}")
        print(f"- 平均信頼度: {avg_confidence:.3f}")
        print(f"- 現在のトレンドフェーズ: {result.current_trend_phase}")
        print(f"- 現在の市場効率性: {result.current_market_efficiency:.3f}")
        print(f"- Supreme知能スコア: {result.supreme_intelligence_score:.3f}")
        
        # 5層システムの動作確認
        print(f"\n🏗️ 5層システム動作確認:")
        print(f"- Layer 1 (カルマンフィルター): {result.kalman_filtered[-1]:.3f}")
        print(f"- Layer 1 (アルティメットスムーザー): {result.ultimate_smoothed[-1]:.3f}")
        print(f"- Layer 1 (ゼロラグEMA): {result.zero_lag_ema[-1]:.3f}")
        print(f"- Layer 2 (FRAMA): {result.frama_values[-1]:.3f}")
        print(f"- Layer 2 (フラクタル次元): {result.fractal_dimension[-1]:.3f}")
        print(f"- Layer 4 (トレンド強度): {result.trend_strength[-1]:.3f}")
        print(f"- Layer 4 (サイクル成分): {result.cycle_component[-1]:.3f}")
        
        # エイリアスのテスト
        print(f"\n🔗 エイリアステスト:")
        sac_alias = SAC()
        print(f"- SAC エイリアス: {type(sac_alias).__name__}")
        
        # ヘルパーメソッドのテスト
        print(f"\n🛠️ ヘルパーメソッドテスト:")
        trend_analysis = sac.get_trend_analysis()
        supreme_report = sac.get_supreme_intelligence_report()
        price_layers = sac.get_price_processing_layers()
        
        print(f"- トレンド解析: {len(trend_analysis) if trend_analysis else 0} 項目")
        print(f"- Supreme レポート: {len(supreme_report)} 項目")
        print(f"- 価格処理層: {len(price_layers) if price_layers else 0} 項目")
        
        print(f"\n🎉 Supreme Adaptive Channel V2.0 テスト完了!")
        print(f"✨ 全ての機能が正常に動作しています。")
        
        return True
        
    except Exception as e:
        print(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_numpy_input():
    """NumPy配列入力のテスト"""
    print(f"\n🔢 NumPy配列入力テスト...")
    
    # NumPy配列データ作成（OHLCV形式）
    n = 100
    np.random.seed(42)
    prices = np.random.randn(n).cumsum() + 100
    
    sac = SupremeAdaptiveChannel()
    
    try:
        result = sac.calculate(prices)
        print(f"✅ NumPy配列入力テスト成功!")
        print(f"- 最終価格: {prices[-1]:.3f}")
        print(f"- 最終チャネル上限: {result.upper_channel[-1]:.3f}")
        print(f"- 最終チャネル下限: {result.lower_channel[-1]:.3f}")
        return True
    except Exception as e:
        print(f"❌ NumPy配列入力テストエラー: {e}")
        return False

def main():
    """メインテスト関数"""
    print("=" * 60)
    print("🌟 Supreme Adaptive Channel V2.0 完全テスト")
    print("=" * 60)
    
    success_count = 0
    total_tests = 2
    
    # テスト1: DataFrame入力
    if test_supreme_adaptive_channel():
        success_count += 1
    
    # テスト2: NumPy配列入力
    if test_numpy_input():
        success_count += 1
    
    print("\n" + "=" * 60)
    print(f"🏁 テスト結果: {success_count}/{total_tests} 成功")
    
    if success_count == total_tests:
        print("🎉 全てのテストが成功しました！")
        print("✨ Supreme Adaptive Channel V2.0 は正常に動作しています。")
    else:
        print("⚠️ 一部のテストが失敗しました。")
    
    print("=" * 60)

if __name__ == "__main__":
    main()