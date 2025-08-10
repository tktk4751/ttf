#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
動的調整の分析とビジュアライゼーション
"""

import numpy as np
import pandas as pd
import sys
import os

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from position_sizing.x_position_sizing import XATRPositionSizing, calculate_dynamic_multiplier_vec, calculate_dynamic_risk_ratio
from position_sizing.position_sizing import PositionSizingParams


def create_analysis_data():
    """分析用データの作成"""
    np.random.seed(42)
    length = 100
    base_price = 50000.0
    
    # 様々な市場状況を模擬
    market_phases = [
        {"start": 0, "end": 25, "trend": 0.002, "volatility": 0.015, "name": "強いトレンド"},
        {"start": 25, "end": 50, "trend": 0.0005, "volatility": 0.025, "name": "弱いトレンド"},
        {"start": 50, "end": 75, "trend": 0.0, "volatility": 0.035, "name": "レンジ相場"},
        {"start": 75, "end": 100, "trend": 0.003, "volatility": 0.012, "name": "非常に強いトレンド"}
    ]
    
    returns = np.zeros(length)
    for phase in market_phases:
        phase_length = phase["end"] - phase["start"]
        phase_returns = np.random.normal(phase["trend"], phase["volatility"], phase_length)
        returns[phase["start"]:phase["end"]] = phase_returns
    
    log_returns = np.cumsum(returns)
    prices = base_price * np.exp(log_returns)
    
    data = []
    for i, close in enumerate(prices):
        daily_volatility = abs(np.random.normal(0, 0.02))
        high = close * (1 + daily_volatility * np.random.uniform(0.3, 1.0))
        low = close * (1 - daily_volatility * np.random.uniform(0.3, 1.0))
        
        if i == 0:
            open_price = close
        else:
            gap = np.random.normal(0, 0.005)
            open_price = prices[i-1] * (1 + gap)
        
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(days=i),
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.uniform(10000, 100000),
            'phase': next(p["name"] for p in market_phases if p["start"] <= i < p["end"])
        })
    
    return pd.DataFrame(data)


def analyze_dynamic_adjustment():
    """動的調整の詳細分析"""
    print("=== X Position Sizing 動的調整分析 ===")
    
    # 分析用データの作成
    market_data = create_analysis_data()
    capital = 100000.0
    
    print(f"分析データ: {len(market_data)}日間")
    print(f"市場フェーズ:")
    for phase in market_data['phase'].unique():
        count = len(market_data[market_data['phase'] == phase])
        print(f"  {phase}: {count}日")
    
    # 2つのトリガータイプで比較分析
    configurations = [
        {
            "name": "Hyper ER",
            "trigger_type": "hyper_er",
            "color": "🔵"
        },
        {
            "name": "Hyper Trend Index", 
            "trigger_type": "hyper_trend_index",
            "color": "🟢"
        }
    ]
    
    results = {}
    
    for config in configurations:
        print(f"\n{config['color']} {config['name']} 分析中...")
        
        # ポジションサイジングインスタンス作成
        sizing = XATRPositionSizing(
            base_risk_ratio=0.02,
            max_position_percent=0.4,
            trigger_type=config["trigger_type"],
            apply_dynamic_adjustment=True,
            max_multiplier=6.0,
            min_multiplier=2.0,
            max_risk_ratio=0.03,
            min_risk_ratio=0.01
        )
        
        # 各日の結果を計算
        daily_results = []
        
        for i in range(20, len(market_data)):  # 最初の20日はスキップ（インディケーター計算のため）
            try:
                current_data = market_data.iloc[:i+1].copy()
                current_price = current_data['close'].iloc[-1]
                
                params = PositionSizingParams(
                    entry_price=current_price,
                    stop_loss_price=None,
                    capital=capital,
                    leverage=1.0,
                    risk_per_trade=0.02,
                    historical_data=current_data
                )
                
                result = sizing.calculate(params)
                
                daily_results.append({
                    'date': current_data['timestamp'].iloc[-1],
                    'price': current_price,
                    'phase': current_data['phase'].iloc[-1],
                    'position_size': result['position_size'],
                    'risk_amount': result['risk_amount'],
                    'risk_ratio': result['risk_ratio'],
                    'x_atr_value': result['x_atr_value'],
                    'atr_multiplier': result['atr_multiplier'],
                    'trigger_value': result['trigger_value'],
                    'trigger_factor': result['trigger_factor']
                })
                
            except Exception as e:
                print(f"  日付 {i} でエラー: {e}")
                continue
        
        results[config["name"]] = pd.DataFrame(daily_results)
        
        # 統計サマリー
        df = results[config["name"]]
        print(f"  計算成功日数: {len(df)}日")
        print(f"  平均ポジションサイズ: ${df['position_size'].mean():,.2f}")
        print(f"  平均リスク比率: {df['risk_ratio'].mean():.4f}")
        print(f"  平均ATR乗数: {df['atr_multiplier'].mean():.2f}")
        print(f"  平均トリガー値: {df['trigger_value'].mean():.4f}")
        
        # フェーズ別分析
        print(f"  フェーズ別分析:")
        for phase in df['phase'].unique():
            phase_data = df[df['phase'] == phase]
            if len(phase_data) > 0:
                print(f"    {phase}:")
                print(f"      平均ポジションサイズ: ${phase_data['position_size'].mean():,.2f}")
                print(f"      平均リスク比率: {phase_data['risk_ratio'].mean():.4f}")
                print(f"      平均ATR乗数: {phase_data['atr_multiplier'].mean():.2f}")
                print(f"      平均トリガー値: {phase_data['trigger_value'].mean():.4f}")
    
    # 比較分析
    if len(results) == 2:
        print(f"\n🔄 比較分析")
        
        hyper_er_df = results["Hyper ER"]
        hyper_trend_df = results["Hyper Trend Index"]
        
        # 日付でマージ（共通の日付のみ）
        merged = pd.merge(hyper_er_df, hyper_trend_df, on='date', suffixes=('_er', '_trend'))
        
        if len(merged) > 0:
            print(f"  比較可能日数: {len(merged)}日")
            
            # ポジションサイズの相関
            correlation = merged['position_size_er'].corr(merged['position_size_trend'])
            print(f"  ポジションサイズ相関: {correlation:.4f}")
            
            # 平均差異
            pos_size_diff = merged['position_size_trend'].mean() - merged['position_size_er'].mean()
            risk_ratio_diff = merged['risk_ratio_trend'].mean() - merged['risk_ratio_er'].mean()
            atr_mult_diff = merged['atr_multiplier_trend'].mean() - merged['atr_multiplier_er'].mean()
            
            print(f"  平均ポジションサイズ差: ${pos_size_diff:,.2f}")
            print(f"  平均リスク比率差: {risk_ratio_diff:.4f}")
            print(f"  平均ATR乗数差: {atr_mult_diff:.2f}")
            
            # より積極的/保守的な傾向
            if pos_size_diff > 0:
                print(f"  Hyper Trend Indexがより積極的")
            else:
                print(f"  Hyper ERがより積極的")
    
    print(f"\n=== 動的調整効果の確認 ===")
    
    # 動的無しの場合との比較
    sizing_static = XATRPositionSizing(
        base_risk_ratio=0.02,
        trigger_type="hyper_er",
        apply_dynamic_adjustment=False  # 動的調整OFF
    )
    
    try:
        # 最新のデータで比較
        current_price = market_data['close'].iloc[-1]
        params = PositionSizingParams(
            entry_price=current_price,
            stop_loss_price=None,
            capital=capital,
            leverage=1.0,
            risk_per_trade=0.02,
            historical_data=market_data
        )
        
        dynamic_result = sizing.calculate(params)
        static_result = sizing_static.calculate(params)
        
        print(f"動的調整あり:")
        print(f"  ポジションサイズ: ${dynamic_result['position_size']:,.2f}")
        print(f"  リスク比率: {dynamic_result['risk_ratio']:.4f}")
        print(f"  ATR乗数: {dynamic_result['atr_multiplier']:.2f}")
        
        print(f"動的調整なし:")
        print(f"  ポジションサイズ: ${static_result['position_size']:,.2f}")
        print(f"  リスク比率: {static_result['risk_ratio']:.4f}")
        print(f"  ATR乗数: {static_result.get('atr_multiplier', 'N/A')}")
        
        improvement = ((dynamic_result['position_size'] - static_result['position_size']) 
                      / static_result['position_size'] * 100)
        print(f"動的調整による改善: {improvement:+.1f}%")
        
    except Exception as e:
        print(f"比較計算でエラー: {e}")
    
    print(f"\n=== 分析完了 ===")


if __name__ == "__main__":
    analyze_dynamic_adjustment()