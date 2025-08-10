#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
X Position Sizing 使用例とデモンストレーション
"""

import numpy as np
import pandas as pd
import sys
import os

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from position_sizing.x_position_sizing import XATRPositionSizing
from position_sizing.position_sizing import PositionSizingParams


def create_demo_data() -> pd.DataFrame:
    """デモ用のマーケットデータを作成"""
    np.random.seed(123)
    length = 100
    base_price = 50000.0  # BTC価格想定
    
    # より現実的な価格変動を生成
    returns = np.random.normal(0, 0.03, length)  # 日次変動率 平均3%
    log_returns = np.cumsum(returns)
    prices = base_price * np.exp(log_returns)
    
    data = []
    for i, close in enumerate(prices):
        # より現実的なOHLCデータの生成
        daily_volatility = abs(np.random.normal(0, 0.02))
        
        high = close * (1 + daily_volatility * np.random.uniform(0.3, 1.0))
        low = close * (1 - daily_volatility * np.random.uniform(0.3, 1.0))
        
        if i == 0:
            open_price = close
        else:
            gap = np.random.normal(0, 0.005)
            open_price = prices[i-1] * (1 + gap)
        
        # 論理的整合性の確保
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(days=i),
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.uniform(10000, 100000)
        })
    
    return pd.DataFrame(data)


def demonstrate_x_position_sizing():
    """X Position Sizingのデモンストレーション"""
    print("=== X Position Sizing デモンストレーション ===")
    print("BTCトレーディングシナリオを想定")
    
    # デモデータの作成
    market_data = create_demo_data()
    current_price = market_data['close'].iloc[-1]
    capital = 100000.0  # $100,000の資金
    
    print(f"\n現在のBTC価格: ${current_price:,.2f}")
    print(f"利用可能資金: ${capital:,.2f}")
    
    print("\n" + "="*60)
    print("シナリオ1: Hyper ER を使用した動的ポジションサイジング")
    print("="*60)
    
    # Hyper ER を使用したポジションサイジング
    sizing_hyper_er = XATRPositionSizing(
        base_risk_ratio=0.02,           # 基本リスク2%
        unit=1.0,
        max_position_percent=0.4,       # 最大40%のポジション
        leverage=1.0,
        trigger_type='hyper_er',        # Hyper ERを使用
        apply_dynamic_adjustment=True,
        
        # X_ATRパラメータ（より保守的な設定）
        x_atr_period=14.0,
        x_atr_tr_method='atr',
        x_atr_smoother_type='frama',
        x_atr_period_mode='dynamic',
        
        # 動的調整パラメータ
        max_multiplier=4.0,
        min_multiplier=2.0,
        max_risk_ratio=0.03,            # 最大3%リスク
        min_risk_ratio=0.01,            # 最小1%リスク
        
        # Hyper ER パラメータ
        hyper_er_period=14,
        hyper_er_use_roofing_filter=True
    )
    
    params = PositionSizingParams(
        entry_price=current_price,
        stop_loss_price=None,  # X_ATRが内部計算するため不要
        capital=capital,
        leverage=1.0,
        risk_per_trade=0.02,
        historical_data=market_data
    )
    
    try:
        result_hyper_er = sizing_hyper_er.calculate(params)
        
        print(f"ポジションサイズ: ${result_hyper_er['position_size']:,.2f}")
        print(f"BTC数量: {result_hyper_er['asset_quantity']:.6f} BTC")
        print(f"リスク金額: ${result_hyper_er['risk_amount']:,.2f}")
        print(f"リスク率: {(result_hyper_er['risk_amount']/capital)*100:.2f}%")
        print(f"X_ATR値: ${result_hyper_er['x_atr_value']:,.2f}")
        print(f"ATR乗数: {result_hyper_er['atr_multiplier']:.2f}")
        print(f"Hyper ER値: {result_hyper_er['trigger_value']:.4f}")
        print(f"調整係数: {result_hyper_er['trigger_factor']:.4f}")
        
    except Exception as e:
        print(f"計算エラー: {e}")
    
    print("\n" + "="*60)
    print("シナリオ2: Hyper Trend Index を使用した動的ポジションサイジング")
    print("="*60)
    
    # Hyper Trend Index を使用したポジションサイジング
    sizing_hyper_trend = XATRPositionSizing(
        base_risk_ratio=0.02,
        unit=1.0,
        max_position_percent=0.4,
        leverage=1.0,
        trigger_type='hyper_trend_index',  # Hyper Trend Indexを使用
        apply_dynamic_adjustment=True,
        
        # 動的調整パラメータ
        max_multiplier=4.0,
        min_multiplier=2.0,
        max_risk_ratio=0.03,
        min_risk_ratio=0.01,
        
        # Hyper Trend Index パラメータ
        hyper_trend_period=14,
        hyper_trend_use_kalman_filter=True,
        hyper_trend_use_dynamic_period=True,
        hyper_trend_use_roofing_filter=True
    )
    
    try:
        result_hyper_trend = sizing_hyper_trend.calculate(params)
        
        print(f"ポジションサイズ: ${result_hyper_trend['position_size']:,.2f}")
        print(f"BTC数量: {result_hyper_trend['asset_quantity']:.6f} BTC")
        print(f"リスク金額: ${result_hyper_trend['risk_amount']:,.2f}")
        print(f"リスク率: {(result_hyper_trend['risk_amount']/capital)*100:.2f}%")
        print(f"X_ATR値: ${result_hyper_trend['x_atr_value']:,.2f}")
        print(f"ATR乗数: {result_hyper_trend['atr_multiplier']:.2f}")
        print(f"Hyper Trend Index値: {result_hyper_trend['trigger_value']:.4f}")
        print(f"調整係数: {result_hyper_trend['trigger_factor']:.4f}")
        
    except Exception as e:
        print(f"計算エラー: {e}")
    
    print("\n" + "="*60)
    print("シナリオ3: 固定リスクでのポジションサイジング")
    print("="*60)
    
    try:
        # 固定リスクでのポジションサイジング
        result_fixed = sizing_hyper_er.calculate_position_size_with_fixed_risk(
            entry_price=current_price,
            capital=capital,
            historical_data=market_data,
            is_long=True
        )
        
        print(f"ポジションサイズ: ${result_fixed['position_size']:,.2f}")
        print(f"BTC数量: {result_fixed['asset_quantity']:.6f} BTC")
        print(f"固定リスク金額: ${result_fixed['risk_amount']:,.2f}")
        print(f"ストップロス価格: ${result_fixed['stop_loss_price']:,.2f}")
        print(f"ストップロス距離: ${result_fixed['stop_loss_distance']:,.2f}")
        print(f"エントリー価格: ${current_price:,.2f}")
        
        # ストップロス%を計算
        stop_loss_percent = (result_fixed['stop_loss_distance'] / current_price) * 100
        print(f"ストップロス率: {stop_loss_percent:.2f}%")
        
    except Exception as e:
        print(f"計算エラー: {e}")
    
    print("\n" + "="*60)
    print("比較分析")
    print("="*60)
    
    try:
        if 'result_hyper_er' in locals() and 'result_hyper_trend' in locals():
            print(f"Hyper ER vs Hyper Trend Index:")
            print(f"  ポジションサイズ差: ${abs(result_hyper_er['position_size'] - result_hyper_trend['position_size']):,.2f}")
            print(f"  リスク比率差: {abs(result_hyper_er['risk_ratio'] - result_hyper_trend['risk_ratio']):.4f}")
            print(f"  ATR乗数差: {abs(result_hyper_er['atr_multiplier'] - result_hyper_trend['atr_multiplier']):.2f}")
            
            # より積極的なトリガーを特定
            if result_hyper_er['position_size'] > result_hyper_trend['position_size']:
                print(f"  Hyper ERがより積極的なポジショニングを推奨")
            else:
                print(f"  Hyper Trend Indexがより積極的なポジショニングを推奨")
    except:
        print("比較分析をスキップ（一部の計算でエラーが発生）")
    
    print("\n=== デモンストレーション完了 ===")


if __name__ == "__main__":
    demonstrate_x_position_sizing()