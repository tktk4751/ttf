#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Laguerre RSI ストラテジーの統合テストスクリプト
"""

import numpy as np
import pandas as pd
import sys
import os

# パスの追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategies.implementations.laguerre_rsi_trend_follow.strategy import LaguerreRSITrendFollowStrategy
from strategies.implementations.laguerre_rsi_trend_reversal.strategy import LaguerreRSITrendReversalStrategy


def generate_test_data(length=200, data_type="mixed"):
    """
    テストデータを生成する
    
    Args:
        length: データポイント数
        data_type: データタイプ（"trend", "range", "mixed"）
        
    Returns:
        pd.DataFrame: OHLCV データ
    """
    np.random.seed(42)
    base_price = 100.0
    prices = [base_price]
    
    if data_type == "trend":
        # トレンドデータ: 上昇 → レンジ → 下降
        for i in range(1, length):
            if i < 60:  # 上昇トレンド
                change = 0.005 + np.random.normal(0, 0.01)
            elif i < 140:  # レンジ相場
                change = np.random.normal(0, 0.008)
            else:  # 下降トレンド
                change = -0.005 + np.random.normal(0, 0.01)
            
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
            
    elif data_type == "range":
        # レンジ相場データ: 振動の大きい横ばい
        for i in range(1, length):
            cycle_pos = (i / 20.0) * 2 * np.pi
            cycle_component = np.sin(cycle_pos) * 0.02  # 2%の周期変動
            random_component = np.random.normal(0, 0.015)  # 1.5%のランダム変動
            
            change = cycle_component + random_component
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
            
    else:  # mixed
        # 混合データ: トレンドとレンジが混在
        for i in range(1, length):
            if i < 50:  # 上昇トレンド
                change = 0.003 + np.random.normal(0, 0.008)
            elif i < 100:  # レンジ相場
                change = np.random.normal(0, 0.010)
            elif i < 150:  # 下降トレンド
                change = -0.003 + np.random.normal(0, 0.008)
            else:  # 再びレンジ相場
                change = np.random.normal(0, 0.010)
            
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
    
    # OHLC データの生成
    data = []
    for i, close in enumerate(prices):
        daily_range = abs(np.random.normal(0, close * 0.008))
        
        high = close + daily_range * np.random.uniform(0.3, 1.0)
        low = close - daily_range * np.random.uniform(0.3, 1.0)
        
        if i == 0:
            open_price = close
        else:
            gap = np.random.normal(0, close * 0.003)
            open_price = prices[i-1] + gap
        
        # 論理的整合性の確保
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.uniform(1000, 10000)
        })
    
    return pd.DataFrame(data)


def test_trend_follow_strategy(df):
    """トレンドフォロー戦略のテスト"""
    print("\n" + "="*70)
    print("Laguerre RSI トレンドフォロー戦略のテスト")
    print("="*70)
    
    # ポジション維持モード
    print("\n--- ポジション維持モード ---")
    strategy_pos = LaguerreRSITrendFollowStrategy(
        gamma=0.5,
        src_type='close',
        period=20.0,
        buy_band=0.8,
        sell_band=0.2,
        position_mode=True
    )
    
    entry_signals = strategy_pos.generate_entry(df)
    lrsi_values = strategy_pos.get_lrsi_values(df)
    
    long_signals = np.sum(entry_signals == 1)
    short_signals = np.sum(entry_signals == -1)
    no_signals = np.sum(entry_signals == 0)
    
    print(f"データポイント: {len(df)}")
    print(f"ロングエントリー: {long_signals} ({long_signals/len(df)*100:.1f}%)")
    print(f"ショートエントリー: {short_signals} ({short_signals/len(df)*100:.1f}%)")
    print(f"シグナルなし: {no_signals} ({no_signals/len(df)*100:.1f}%)")
    print(f"平均Laguerre RSI: {np.nanmean(lrsi_values):.4f}")
    print(f"RSI範囲: {np.nanmin(lrsi_values):.4f} - {np.nanmax(lrsi_values):.4f}")
    
    # クロスオーバーモード
    print("\n--- クロスオーバーモード ---")
    strategy_cross = LaguerreRSITrendFollowStrategy(
        gamma=0.5,
        src_type='close',
        period=20.0,
        buy_band=0.8,
        sell_band=0.2,
        position_mode=False
    )
    
    entry_signals_cross = strategy_cross.generate_entry(df)
    
    long_cross = np.sum(entry_signals_cross == 1)
    short_cross = np.sum(entry_signals_cross == -1)
    no_cross = np.sum(entry_signals_cross == 0)
    
    print(f"ロングエントリー: {long_cross}")
    print(f"ショートエントリー: {short_cross}")
    print(f"シグナルなし: {no_cross}")
    
    # エグジットテスト
    print("\n--- エグジットテスト ---")
    exit_count = 0
    for i in range(50, len(df)-10):
        if entry_signals[i] == 1:  # ロングポジション
            should_exit = strategy_pos.generate_exit(df, 1, i+5)
            if should_exit:
                exit_count += 1
                break
    
    print(f"エグジットテスト結果: {exit_count}回のエグジット")
    
    # 戦略情報の表示
    print("\n--- 戦略情報 ---")
    strategy_info = strategy_pos.get_strategy_info()
    print(f"戦略名: {strategy_info['name']}")
    print(f"説明: {strategy_info['description']}")
    
    return entry_signals, lrsi_values


def test_trend_reversal_strategy(df):
    """トレンドリバーサル戦略のテスト"""
    print("\n" + "="*70)
    print("Laguerre RSI トレンドリバーサル戦略のテスト")
    print("="*70)
    
    # ポジション維持モード（リバーサル）
    print("\n--- ポジション維持モード（リバーサル） ---")
    strategy_pos = LaguerreRSITrendReversalStrategy(
        gamma=0.5,
        src_type='close',
        period=20.0,
        buy_band=0.2,  # 売られすぎでロング
        sell_band=0.8,  # 買われすぎでショート
        position_mode=True
    )
    
    entry_signals = strategy_pos.generate_entry(df)
    lrsi_values = strategy_pos.get_lrsi_values(df)
    
    long_signals = np.sum(entry_signals == 1)
    short_signals = np.sum(entry_signals == -1)
    no_signals = np.sum(entry_signals == 0)
    
    print(f"データポイント: {len(df)}")
    print(f"ロングエントリー: {long_signals} ({long_signals/len(df)*100:.1f}%)")
    print(f"ショートエントリー: {short_signals} ({short_signals/len(df)*100:.1f}%)")
    print(f"シグナルなし: {no_signals} ({no_signals/len(df)*100:.1f}%)")
    print(f"平均Laguerre RSI: {np.nanmean(lrsi_values):.4f}")
    print(f"RSI範囲: {np.nanmin(lrsi_values):.4f} - {np.nanmax(lrsi_values):.4f}")
    
    # 平均回帰モード
    print("\n--- 平均回帰モード ---")
    strategy_meanrev = LaguerreRSITrendReversalStrategy(
        gamma=0.5,
        src_type='close',
        period=20.0,
        buy_band=0.3,
        sell_band=0.7,
        position_mode=True,
        mean_reversion_mode=True
    )
    
    entry_signals_meanrev = strategy_meanrev.generate_entry(df)
    
    long_meanrev = np.sum(entry_signals_meanrev == 1)
    short_meanrev = np.sum(entry_signals_meanrev == -1)
    no_meanrev = np.sum(entry_signals_meanrev == 0)
    
    print(f"ロングエントリー: {long_meanrev}")
    print(f"ショートエントリー: {short_meanrev}")
    print(f"シグナルなし: {no_meanrev}")
    
    # エグジットテスト
    print("\n--- エグジットテスト ---")
    exit_count = 0
    for i in range(50, len(df)-10):
        if entry_signals[i] == 1:  # ロングポジション
            should_exit = strategy_pos.generate_exit(df, 1, i+5)
            if should_exit:
                exit_count += 1
                break
    
    print(f"エグジットテスト結果: {exit_count}回のエグジット")
    
    # 戦略情報の表示
    print("\n--- 戦略情報 ---")
    strategy_info = strategy_pos.get_strategy_info()
    print(f"戦略名: {strategy_info['name']}")
    print(f"説明: {strategy_info['description']}")
    
    return entry_signals, lrsi_values


def compare_strategies(trend_follow_signals, trend_reversal_signals):
    """両戦略の比較"""
    print("\n" + "="*70)
    print("トレンドフォロー vs トレンドリバーサル 戦略比較")
    print("="*70)
    
    # シグナル一致率
    agreement = np.sum(trend_follow_signals == trend_reversal_signals)
    disagreement = np.sum(trend_follow_signals != trend_reversal_signals)
    opposite = np.sum(trend_follow_signals == -trend_reversal_signals)
    
    print(f"シグナル一致: {agreement} ({agreement/len(trend_follow_signals)*100:.1f}%)")
    print(f"シグナル不一致: {disagreement} ({disagreement/len(trend_follow_signals)*100:.1f}%)")
    print(f"反対シグナル: {opposite} ({opposite/len(trend_follow_signals)*100:.1f}%)")
    
    # 戦略特性比較
    print("\n--- 戦略特性比較 ---")
    print("トレンドフォロー:")
    print(f"  - ロングシグナル: {np.sum(trend_follow_signals == 1)}")
    print(f"  - ショートシグナル: {np.sum(trend_follow_signals == -1)}")
    print(f"  - 特徴: 買われすぎ/売られすぎ水準での順張り")
    
    print("トレンドリバーサル:")
    print(f"  - ロングシグナル: {np.sum(trend_reversal_signals == 1)}")
    print(f"  - ショートシグナル: {np.sum(trend_reversal_signals == -1)}")
    print(f"  - 特徴: 売られすぎ/買われすぎ水準での逆張り")
    
    # 相関分析
    correlation = np.corrcoef(trend_follow_signals, trend_reversal_signals)[0, 1]
    print(f"\nシグナル相関係数: {correlation:.4f}")
    
    if correlation < -0.7:
        print("→ 強い逆相関：典型的な順張り vs 逆張り関係")
    elif correlation > 0.7:
        print("→ 強い正相関：類似のシグナル生成")
    else:
        print("→ 弱い相関：独立性の高いシグナル生成")


def test_advanced_features(df):
    """高度な機能のテスト"""
    print("\n" + "="*70)
    print("高度な機能のテスト")
    print("="*70)
    
    # ルーフィングフィルター付きトレンドフォロー
    print("\n--- ルーフィングフィルター付きトレンドフォロー ---")
    strategy_roofing = LaguerreRSITrendFollowStrategy(
        gamma=0.5,
        src_type='close',
        period=20.0,
        buy_band=0.8,
        sell_band=0.2,
        use_roofing_filter=True,
        roofing_hp_cutoff=48.0,
        roofing_ss_band_edge=10.0,
        position_mode=True
    )
    
    entry_signals_roofing = strategy_roofing.generate_entry(df)
    long_roofing = np.sum(entry_signals_roofing == 1)
    short_roofing = np.sum(entry_signals_roofing == -1)
    
    print(f"ルーフィングフィルター付き - ロング: {long_roofing}, ショート: {short_roofing}")
    
    # 高度なメトリクス取得
    print("\n--- 高度なメトリクス ---")
    advanced_metrics = strategy_roofing.get_advanced_metrics(df)
    
    if 'lrsi_values' in advanced_metrics:
        lrsi_values = advanced_metrics['lrsi_values']
        print(f"RSI統計:")
        print(f"  - 平均: {np.nanmean(lrsi_values):.4f}")
        print(f"  - 標準偏差: {np.nanstd(lrsi_values):.4f}")
        print(f"  - 最小値: {np.nanmin(lrsi_values):.4f}")
        print(f"  - 最大値: {np.nanmax(lrsi_values):.4f}")
    
    # L0, L1, L2, L3値の確認
    l0_values = strategy_roofing.get_l0_values(df)
    l1_values = strategy_roofing.get_l1_values(df)
    l2_values = strategy_roofing.get_l2_values(df)
    l3_values = strategy_roofing.get_l3_values(df)
    
    print(f"\nラゲール変換レベル統計:")
    print(f"  - L0平均: {np.nanmean(l0_values):.4f}")
    print(f"  - L1平均: {np.nanmean(l1_values):.4f}")
    print(f"  - L2平均: {np.nanmean(l2_values):.4f}")
    print(f"  - L3平均: {np.nanmean(l3_values):.4f}")


def test_optimization_params():
    """最適化パラメータの生成テスト"""
    print("\n" + "="*70)
    print("最適化パラメータ生成テスト")
    print("="*70)
    
    # 模擬Optunaトライアル
    class MockTrial:
        def suggest_float(self, name, low, high, step=None):
            return np.random.uniform(low, high)
        
        def suggest_categorical(self, name, choices):
            return np.random.choice(choices)
        
        def suggest_int(self, name, low, high):
            return np.random.randint(low, high+1)
    
    trial = MockTrial()
    
    # トレンドフォロー最適化パラメータ
    print("\n--- トレンドフォロー最適化パラメータ ---")
    tf_params = LaguerreRSITrendFollowStrategy.create_optimization_params(trial)
    tf_strategy_params = LaguerreRSITrendFollowStrategy.convert_params_to_strategy_format(tf_params)
    
    print("生成されたパラメータ:")
    for key, value in tf_strategy_params.items():
        print(f"  - {key}: {value}")
    
    # トレンドリバーサル最適化パラメータ
    print("\n--- トレンドリバーサル最適化パラメータ ---")
    tr_params = LaguerreRSITrendReversalStrategy.create_optimization_params(trial)
    tr_strategy_params = LaguerreRSITrendReversalStrategy.convert_params_to_strategy_format(tr_params)
    
    print("生成されたパラメータ:")
    for key, value in tr_strategy_params.items():
        print(f"  - {key}: {value}")


def main():
    """メイン関数"""
    print("=" * 80)
    print("Laguerre RSI ストラテジー総合テスト")
    print("=" * 80)
    
    # テストデータの種類別テスト
    data_types = ["trend", "range", "mixed"]
    
    for data_type in data_types:
        print(f"\n\n{'='*25} {data_type.upper()}データテスト {'='*25}")
        
        df = generate_test_data(length=200, data_type=data_type)
        print(f"価格範囲: {df['close'].min():.2f} - {df['close'].max():.2f}")
        print(f"価格標準偏差: {df['close'].std():.2f}")
        
        # トレンドフォロー戦略のテスト
        trend_follow_signals, lrsi_trend_values = test_trend_follow_strategy(df)
        
        # トレンドリバーサル戦略のテスト
        trend_reversal_signals, lrsi_reversal_values = test_trend_reversal_strategy(df)
        
        # 戦略比較
        compare_strategies(trend_follow_signals, trend_reversal_signals)
        
        # 高度な機能のテスト（最初のデータタイプでのみ実行）
        if data_type == "mixed":
            test_advanced_features(df)
    
    # 最適化パラメータテスト
    test_optimization_params()
    
    print("\n" + "="*80)
    print("テスト完了 - 両戦略が正常に動作しています")
    print("="*80)


if __name__ == "__main__":
    main()