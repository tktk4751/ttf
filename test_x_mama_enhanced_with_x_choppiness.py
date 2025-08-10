#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from strategies.implementations.x_mama_enhanced.strategy import XMAMAEnhancedStrategy
from strategies.implementations.x_mama_enhanced.signal_generator import FilterType

def test_x_mama_enhanced_with_x_choppiness():
    """X-ChoppinessフィルターをテストするX-MAMA Enhanced戦略のテスト"""
    print("=== X-MAMA Enhanced戦略（X-Choppinessフィルター）のテスト ===")
    
    # テストデータ生成
    np.random.seed(42)
    length = 150
    base_price = 100.0
    
    # より長いトレンドとレンジを含むデータを生成
    prices = [base_price]
    for i in range(1, length):
        if i < 50:  # 上昇トレンド
            change = 0.004 + np.random.normal(0, 0.006)
        elif i < 100:  # レンジ相場
            change = np.random.normal(0, 0.003)
        else:  # 下降トレンド
            change = -0.003 + np.random.normal(0, 0.005)
        
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # OHLC データの生成
    data = []
    for i, close in enumerate(prices):
        daily_range = close * 0.008
        
        high = close + daily_range * np.random.uniform(0.3, 1.0)
        low = close - daily_range * np.random.uniform(0.3, 1.0)
        
        if i == 0:
            open_price = close
        else:
            gap = np.random.normal(0, close * 0.002)
            open_price = prices[i-1] + gap
        
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.uniform(1000, 10000)
        })
    
    df = pd.DataFrame(data)
    
    print(f"テストデータ: {len(df)}ポイント")
    print(f"価格範囲: {df['close'].min():.2f} - {df['close'].max():.2f}")
    print(f"総価格変化: {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:.2f}%")
    
    # X-MAMA Enhanced戦略（X-Choppinessフィルター付き）のテスト
    print("\\n--- X-MAMA Enhanced戦略（X-Choppinessフィルター）のテスト ---")
    
    try:
        strategy = XMAMAEnhancedStrategy(
            # X_MAMAパラメータ
            fast_limit=0.5,
            slow_limit=0.05,
            src_type='hlc3',
            use_kalman_filter=False,
            use_zero_lag=True,
            position_mode=False,  # クロスオーバーモード
            # フィルター設定
            filter_type=FilterType.X_CHOPPINESS,
            # X-Choppinessフィルターパラメータ（サイクル検出器関連のみ）
            x_choppiness_detector_type='hody_e',
            x_choppiness_lp_period=10,
            x_choppiness_hp_period=80,
            x_choppiness_cycle_part=0.5,
            x_choppiness_max_cycle=80,
            x_choppiness_min_cycle=10,
            x_choppiness_max_output=70,
            x_choppiness_min_output=8
        )
        
        print(f"戦略名: {strategy.name}")
        print("戦略情報:")
        info = strategy.get_strategy_info()
        print(f"  説明: {info['description']}")
        print(f"  X-Choppinessフィルター機能: {info['filter_capabilities']['x_choppiness']}")
        
        # エントリーシグナルの生成
        entry_signals = strategy.generate_entry(df)
        print(f"\\nエントリーシグナルの統計:")
        print(f"  シグナル配列の形状: {entry_signals.shape}")
        
        valid_signals = entry_signals[~np.isnan(entry_signals)]
        if len(valid_signals) > 0:
            long_count = np.sum(valid_signals == 1)
            short_count = np.sum(valid_signals == -1)
            total_signals = long_count + short_count
            
            print(f"  有効シグナル数: {len(valid_signals)}/{len(entry_signals)}")
            print(f"  ロングシグナル: {long_count} ({long_count/len(valid_signals)*100:.1f}%)")
            print(f"  ショートシグナル: {short_count} ({short_count/len(valid_signals)*100:.1f}%)")
            print(f"  総シグナル数: {total_signals}")
            
            if total_signals > 0:
                print(f"  シグナル頻度: {total_signals/len(valid_signals)*100:.1f}%")
        else:
            print("  有効なエントリーシグナルが生成されませんでした")
        
        # X_MAMAとフィルターの詳細情報を取得
        print("\\n--- X_MAMAとフィルターの詳細情報 ---")
        
        # X_MAMA値
        mama_values = strategy.get_mama_values(df)
        fama_values = strategy.get_fama_values(df)
        if len(mama_values) > 0 and len(fama_values) > 0:
            print(f"X_MAMA値: {np.nanmean(mama_values):.4f} (平均)")
            print(f"X_FAMA値: {np.nanmean(fama_values):.4f} (平均)")
        
        # X_MAMAシグナル
        x_mama_signals = strategy.get_x_mama_signals(df)
        valid_x_mama = x_mama_signals[~np.isnan(x_mama_signals)]
        if len(valid_x_mama) > 0:
            x_mama_long = np.sum(valid_x_mama == 1)
            x_mama_short = np.sum(valid_x_mama == -1)
            print(f"X_MAMAシグナル: ロング{x_mama_long}, ショート{x_mama_short}")
        
        # フィルターシグナル
        filter_signals = strategy.get_filter_signals(df)
        valid_filter = filter_signals[~np.isnan(filter_signals)]
        if len(valid_filter) > 0:
            filter_trend = np.sum(valid_filter == 1)
            filter_range = np.sum(valid_filter == -1)
            print(f"X-Choppinessフィルター: トレンド{filter_trend}, レンジ{filter_range}")
        
        # フィルター詳細情報
        filter_details = strategy.get_filter_details(df)
        if filter_details:
            print("\\nX-Choppinessフィルターの詳細:")
            for key, values in filter_details.items():
                if isinstance(values, np.ndarray) and len(values) > 0:
                    valid_values = values[~np.isnan(values)]
                    if len(valid_values) > 0:
                        print(f"  {key}: 平均={np.mean(valid_values):.4f}, 標準偏差={np.std(valid_values):.4f}")
        
        # 時系列での動作確認（3つの期間に分けて）
        print("\\n--- 時系列でのシグナル分析 ---")
        period_length = len(valid_signals) // 3
        periods = ['期間1 (上昇トレンド)', '期間2 (レンジ)', '期間3 (下降トレンド)']
        
        for p in range(3):
            start_idx = p * period_length
            end_idx = (p + 1) * period_length if p < 2 else len(valid_signals)
            period_signals = valid_signals[start_idx:end_idx] if len(valid_signals) > end_idx else valid_signals[start_idx:]
            
            if len(period_signals) > 0:
                period_long = np.sum(period_signals == 1)
                period_short = np.sum(period_signals == -1)
                period_total = period_long + period_short
                
                print(f"  {periods[p]}: ロング{period_long}, ショート{period_short}, 総シグナル{period_total}")
        
        # エグジットシグナルのテスト
        print("\\n--- エグジットシグナルのテスト ---")
        # 最後の数ポイントでエグジット条件をテスト
        for position in [1, -1]:  # ロング・ショート両方
            position_name = "ロング" if position == 1 else "ショート"
            exit_count = 0
            
            for i in range(max(0, len(df) - 20), len(df)):
                should_exit = strategy.generate_exit(df, position, i)
                if should_exit:
                    exit_count += 1
            
            print(f"  {position_name}ポジション: 最後20ポイントで{exit_count}回のエグジットシグナル")
        
        print("\\n=== X-MAMA Enhanced戦略（X-Choppinessフィルター）テスト完了 ===")
        
    except Exception as e:
        print(f"テスト中にエラーが発生: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_x_mama_enhanced_with_x_choppiness()