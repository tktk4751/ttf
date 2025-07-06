#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🌌 Cosmic Adaptive Channel 実相場テストスクリプト 🌌

このスクリプトは以下を実行します:
1. config.yamlから実際の相場データを取得
2. Cosmic Adaptive Channelを計算
3. 詳細な統計解析を実行
4. 高度なチャートを生成
5. 戦略シミュレーションを実行
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualization.cosmic_adaptive_channel_chart import CosmicAdaptiveChannelChart
import pandas as pd
import numpy as np


def run_cosmic_channel_test():
    """🌌 Cosmic Adaptive Channelの包括的テストを実行"""
    
    print("🌌" * 30)
    print("🌌 COSMIC ADAPTIVE CHANNEL - 宇宙最強テストシステム 🌌")
    print("🌌" * 30)
    
    # チャートインスタンス作成
    chart = CosmicAdaptiveChannelChart()
    
    try:
        # Step 1: データ読み込み
        print("\n📡 STEP 1: 実相場データ取得")
        print("-" * 40)
        chart.load_data_from_config('config.yaml')
        
        # Step 2: インジケーター計算（最適化パラメータ）
        print("\n⚡ STEP 2: 8層ハイブリッドシステム計算")
        print("-" * 40)
        chart.calculate_indicators(
            atr_period=21,           # ATR期間
            base_multiplier=2.5,     # 基本チャネル倍率
            quantum_window=50,       # 量子解析ウィンドウ
            neural_window=100,       # 神経学習ウィンドウ
            volatility_window=30,    # ボラティリティ解析ウィンドウ
            src_type='hlc3'          # 価格ソースタイプ
        )
        
        # Step 3: 詳細パフォーマンス解析
        print("\n📊 STEP 3: 詳細パフォーマンス解析")
        print("-" * 40)
        analysis = chart.analyze_performance()
        
        # Step 4: 戦略シミュレーション（複数の信頼度レベル）
        print("\n💹 STEP 4: 戦略シミュレーション")
        print("-" * 40)
        
        confidence_levels = [0.3, 0.5, 0.7]
        strategy_results = {}
        
        for conf in confidence_levels:
            print(f"\n📈 信頼度≥{conf}での戦略テスト:")
            strategy_results[conf] = chart.simulate_strategy(min_confidence=conf)
        
        # Step 5: 高度なチャート生成
        print("\n🎨 STEP 5: 宇宙最強チャート生成")
        print("-" * 40)
        
        # 最近3ヶ月のデータでチャート生成
        end_date = chart.data.index.max()
        start_date = end_date - pd.Timedelta(days=90)
        
        chart.plot(
            title=f"🌌 Cosmic Adaptive Channel - {chart.symbol} ({chart.timeframe})",
            start_date=start_date.strftime('%Y-%m-%d'),
            show_volume=True,
            figsize=(18, 22),
            savefig=f'cosmic_channel_analysis_{chart.symbol}_{chart.timeframe}.png'
        )
        
        # Step 6: 宇宙知能レポート詳細出力
        print("\n🧠 STEP 6: 宇宙知能レポート")
        print("-" * 40)
        
        intel_report = analysis['intelligence_report']
        
        print(f"🌌 宇宙知能スコア: {intel_report['cosmic_intelligence_score']:.4f}")
        print(f"🧠 現在のトレンドフェーズ: {intel_report['current_trend_phase']}")
        print(f"🌊 ボラティリティレジーム: {intel_report['current_volatility_regime']}")
        print(f"🚀 ブレイクアウト確率: {intel_report['current_breakout_probability']:.3f}")
        print(f"⚛️ 量子コヒーレンス: {intel_report['current_quantum_coherence']:.3f}")
        print(f"🧬 神経適応スコア: {intel_report['current_neural_adaptation']:.3f}")
        print(f"🛡️ 偽シグナル防御率: {(1-intel_report['false_signal_rate'])*100:.1f}%")
        print(f"📊 チャネル効率度: {intel_report['current_channel_efficiency']:.3f}")
        
        # Step 7: 戦略比較テーブル
        print(f"\n📊 STEP 7: 戦略パフォーマンス比較")
        print("-" * 40)
        print(f"{'信頼度':<8} {'取引数':<8} {'総リターン':<12} {'勝率':<8} {'シャープ':<8}")
        print("-" * 50)
        
        for conf, stats in strategy_results.items():
            if stats:
                print(f"{conf:<8.1f} {stats['total_trades']:<8} {stats['total_return']:+<12.2%} "
                      f"{stats['win_rate']:<8.1%} {stats['sharpe_ratio']:<8.2f}")
        
        # Step 8: 最終評価とレコメンデーション
        print(f"\n🎯 STEP 8: 最終評価")
        print("=" * 60)
        
        best_strategy = max(strategy_results.values(), key=lambda x: x.get('sharpe_ratio', -999) if x else -999)
        best_conf = [k for k, v in strategy_results.items() if v == best_strategy][0] if best_strategy else None
        
        # 総合評価スコア計算
        intelligence_score = intel_report['cosmic_intelligence_score']
        channel_efficiency = analysis['channel_effectiveness']
        signal_quality = analysis['signal_quality']
        quantum_stability = analysis['quantum_stability']
        
        total_score = (intelligence_score * 0.3 + 
                      channel_efficiency * 0.25 + 
                      signal_quality * 0.25 + 
                      quantum_stability * 0.2)
        
        print(f"🌌 総合評価スコア: {total_score:.3f}/1.000")
        
        if total_score >= 0.8:
            grade = "🏆 COSMIC SUPREME (宇宙最強)"
        elif total_score >= 0.7:
            grade = "⭐ QUANTUM MASTER (量子マスター)"
        elif total_score >= 0.6:
            grade = "🚀 NEURAL EXPERT (神経エキスパート)"
        elif total_score >= 0.5:
            grade = "💫 ADAPTIVE PRO (適応プロ)"
        else:
            grade = "🌱 COSMIC ROOKIE (宇宙ルーキー)"
        
        print(f"🏅 等級: {grade}")
        
        if best_strategy:
            print(f"💎 推奨戦略: 信頼度≥{best_conf} (リターン: {best_strategy['total_return']:+.2%})")
        
        # レコメンデーション
        print(f"\n🎯 レコメンデーション:")
        
        if signal_quality < 0.5:
            print("📈 シグナル品質向上のため、ボラティリティウィンドウを調整することを推奨")
        
        if quantum_stability < 0.5:
            print("⚛️ 量子安定性向上のため、量子ウィンドウを拡大することを推奨")
        
        if channel_efficiency < 0.6:
            print("🌊 チャネル効率向上のため、基本倍率を調整することを推奨")
        
        if intel_report['false_signal_rate'] > 0.3:
            print("🛡️ 偽シグナル削減のため、神経学習ウィンドウを拡大することを推奨")
        
        print(f"\n🌌 Cosmic Adaptive Channel テスト完了! 🌌")
        print(f"📊 チャートファイル: cosmic_channel_analysis_{chart.symbol}_{chart.timeframe}.png")
        
        return {
            'analysis': analysis,
            'strategy_results': strategy_results,
            'intelligence_report': intel_report,
            'total_score': total_score,
            'grade': grade
        }
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_quick_test():
    """🚀 クイックテスト（軽量版）"""
    print("🚀 Cosmic Adaptive Channel - クイックテスト")
    print("=" * 50)
    
    chart = CosmicAdaptiveChannelChart()
    
    try:
        # データ読み込み
        chart.load_data_from_config('config.yaml')
        
        # 軽量パラメータで計算
        chart.calculate_indicators(
            atr_period=14,
            base_multiplier=2.0,
            quantum_window=30,
            neural_window=50,
            volatility_window=20
        )
        
        # 簡易解析
        analysis = chart.analyze_performance()
        strategy = chart.simulate_strategy(min_confidence=0.5)
        
        # 簡易レポート
        intel = analysis['intelligence_report']
        print(f"\n📊 クイック結果:")
        print(f"宇宙知能スコア: {intel['cosmic_intelligence_score']:.3f}")
        print(f"チャネル効率: {analysis['channel_effectiveness']:.3f}")
        print(f"シグナル品質: {analysis['signal_quality']:.3f}")
        
        if strategy:
            print(f"戦略リターン: {strategy['total_return']:+.2%}")
            print(f"勝率: {strategy['win_rate']:.1%}")
        
        print("✅ クイックテスト完了")
        
    except Exception as e:
        print(f"❌ エラー: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='🌌 Cosmic Adaptive Channel テストシステム')
    parser.add_argument('--quick', '-q', action='store_true', help='クイックテストモード')
    args = parser.parse_args()
    
    if args.quick:
        run_quick_test()
    else:
        run_cosmic_channel_test() 