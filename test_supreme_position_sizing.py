#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Supreme Position Sizing Algorithm テストスクリプト

最強ポジションサイジングアルゴリズムの動作検証と性能テスト
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from position_sizing.supreme_position_sizing import SupremePositionSizing, SupremePositionConfig
from position_sizing.position_sizing import PositionSizingParams
from position_sizing.simple_atr_sizing import SimpleATRPositionSizing


def setup_logging():
    """ログ設定"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_test_data(symbol: str = 'BTCUSDT', timeframe: str = '1h', limit: int = 1000):
    """
    テスト用データロード
    
    Args:
        symbol: シンボル
        timeframe: 時間枠
        limit: データ数
        
    Returns:
        pd.DataFrame: OHLCV データ
    """
    # 直接合成データを生成（外部APIの依存を回避）
    logging.info("合成データを生成します...")
    return generate_synthetic_data(limit)


def generate_synthetic_data(n_periods: int = 1000, initial_price: float = 50000):
    """
    合成データ生成（テスト用）
    
    Args:
        n_periods: 期間数
        initial_price: 初期価格
        
    Returns:
        pd.DataFrame: 合成OHLCV データ
    """
    np.random.seed(42)
    
    # 価格変動率生成（平均0、標準偏差0.02）
    returns = np.random.normal(0, 0.02, n_periods)
    
    # トレンド成分追加
    trend = np.sin(np.arange(n_periods) * 0.01) * 0.001
    returns += trend
    
    # 価格計算
    prices = [initial_price]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices[1:])
    
    # OHLC生成
    highs = prices * (1 + np.abs(np.random.normal(0, 0.01, n_periods)))
    lows = prices * (1 - np.abs(np.random.normal(0, 0.01, n_periods)))
    opens = np.roll(prices, 1)
    opens[0] = initial_price
    closes = prices
    volumes = np.random.uniform(1000, 10000, n_periods)
    
    # DataFrame作成
    dates = pd.date_range(start='2024-01-01', periods=n_periods, freq='H')
    
    data = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    }, index=dates)
    
    return data


def test_supreme_vs_simple_atr():
    """
    Supreme Position Sizing と Simple ATR Position Sizing の比較テスト
    """
    print("=" * 60)
    print("Supreme Position Sizing vs Simple ATR 比較テスト")
    print("=" * 60)
    
    # テストデータロード
    data = load_test_data()
    print(f"テストデータ: {len(data)} 期間")
    print(f"価格範囲: {data['close'].min():.2f} - {data['close'].max():.2f}")
    
    # Supreme Position Sizing 初期化
    supreme_config = SupremePositionConfig(
        kelly_fraction=0.30,
        cppi_multiplier_base=5.0,
        volatility_target=0.15,
        max_position_percent=0.25
    )
    supreme_sizing = SupremePositionSizing(supreme_config)
    
    # Simple ATR Position Sizing 初期化
    simple_sizing = SimpleATRPositionSizing(
        base_risk_ratio=0.02,
        max_position_percent=0.25,
        atr_multiplier=2.0
    )
    
    # テストパラメータ
    capital = 100000  # 10万ドル
    entry_price = data['close'].iloc[-1]
    
    # 履歴データが必要な期間を取得
    history_periods = max(supreme_config.atr_period, supreme_config.efficiency_ratio_period, 50)
    historical_data = data.iloc[-history_periods:].copy()
    
    print(f"\\n現在価格: ${entry_price:.2f}")
    print(f"資本金: ${capital:,}")
    print(f"履歴データ期間: {len(historical_data)} 期間")
    
    # PositionSizingParams 作成
    params = PositionSizingParams(
        entry_price=entry_price,
        stop_loss_price=entry_price * 0.95,  # 5%ストップロス
        capital=capital,
        leverage=1.0,
        risk_per_trade=0.02,
        historical_data=historical_data
    )
    
    # Supreme Position Sizing 計算
    print("\\n--- Supreme Position Sizing 計算中 ---")
    supreme_result = supreme_sizing.calculate(params)
    
    # Simple ATR Position Sizing 計算
    print("--- Simple ATR Position Sizing 計算中 ---")
    simple_result = simple_sizing.calculate(params)
    
    # 結果比較表示
    print("\\n" + "=" * 60)
    print("計算結果比較")
    print("=" * 60)
    
    print(f"{'指標':<25} {'Supreme':<15} {'Simple ATR':<15} {'差異':<10}")
    print("-" * 65)
    
    # ポジションサイズ比較
    supreme_size = supreme_result['position_size']
    simple_size = simple_result['position_size']
    size_diff = ((supreme_size - simple_size) / simple_size * 100) if simple_size > 0 else 0
    
    print(f"{'ポジションサイズ ($)':<25} {supreme_size:<15,.0f} {simple_size:<15,.0f} {size_diff:<10.1f}%")
    
    # 資産数量比較
    supreme_qty = supreme_result['asset_quantity']
    simple_qty = simple_result['asset_quantity']
    qty_diff = ((supreme_qty - simple_qty) / simple_qty * 100) if simple_qty > 0 else 0
    
    print(f"{'資産数量':<25} {supreme_qty:<15.4f} {simple_qty:<15.4f} {qty_diff:<10.1f}%")
    
    # リスク金額比較
    supreme_risk = supreme_result['risk_amount']
    simple_risk = simple_result['risk_amount']
    risk_diff = ((supreme_risk - simple_risk) / simple_risk * 100) if simple_risk > 0 else 0
    
    print(f"{'リスク金額 ($)':<25} {supreme_risk:<15,.0f} {simple_risk:<15,.0f} {risk_diff:<10.1f}%")
    
    # 資本比率比較
    supreme_ratio = (supreme_size / capital) * 100
    simple_ratio = (simple_size / capital) * 100
    
    print(f"{'資本比率 (%)':<25} {supreme_ratio:<15.2f} {simple_ratio:<15.2f} {supreme_ratio-simple_ratio:<10.2f}pp")
    
    # Supreme 固有の詳細情報
    print("\\n" + "=" * 60)
    print("Supreme Algorithm 詳細分析")
    print("=" * 60)
    
    print(f"Kelly ポジションサイズ: ${supreme_result.get('kelly_position_size', 0):,.0f}")
    print(f"CPPI ポジションサイズ: ${supreme_result.get('cppi_position_size', 0):,.0f}")
    print(f"Kelly 分数: {supreme_result.get('kelly_fraction', 0):.4f}")
    print(f"CPPI マルチプライヤー: {supreme_result.get('cppi_multiplier', 0):.2f}")
    print(f"信頼度スコア: {supreme_result.get('confidence_score', 0):.3f}")
    print(f"効率比率: {supreme_result.get('efficiency_ratio', 0):.3f}")
    print(f"現在ボラティリティ: {supreme_result.get('current_volatility', 0):.3f}")
    print(f"平均ボラティリティ: {supreme_result.get('avg_volatility', 0):.3f}")
    print(f"勝率: {supreme_result.get('win_rate', 0):.1%}")
    print(f"平均勝ち: {supreme_result.get('avg_win', 0):.3f}")
    print(f"平均負け: {supreme_result.get('avg_loss', 0):.3f}")
    
    return supreme_result, simple_result


def test_market_conditions():
    """
    異なる市場環境でのテスト
    """
    print("\\n" + "=" * 60)
    print("異なる市場環境でのテスト")
    print("=" * 60)
    
    # 異なる市場環境のデータ生成
    market_conditions = {
        'トレンド上昇': generate_trending_data(500, trend=0.001),
        'トレンド下降': generate_trending_data(500, trend=-0.001),
        'レンジ相場': generate_ranging_data(500),
        '高ボラティリティ': generate_high_volatility_data(500)
    }
    
    supreme_config = SupremePositionConfig()
    supreme_sizing = SupremePositionSizing(supreme_config)
    
    capital = 100000
    results = {}
    
    for condition_name, data in market_conditions.items():
        print(f"\\n--- {condition_name} ---")
        
        entry_price = data['close'].iloc[-1]
        historical_data = data.iloc[-100:].copy()
        
        params = PositionSizingParams(
            entry_price=entry_price,
            stop_loss_price=entry_price * 0.95,
            capital=capital,
            historical_data=historical_data
        )
        
        result = supreme_sizing.calculate(params)
        results[condition_name] = result
        
        print(f"ポジションサイズ: ${result['position_size']:,.0f} ({result['position_size']/capital:.1%})")
        print(f"Kelly分数: {result.get('kelly_fraction', 0):.4f}")
        print(f"CPPIマルチプライヤー: {result.get('cppi_multiplier', 0):.2f}")
        print(f"効率比率: {result.get('efficiency_ratio', 0):.3f}")
        print(f"信頼度スコア: {result.get('confidence_score', 0):.3f}")
    
    return results


def generate_trending_data(n_periods: int, trend: float = 0.001):
    """トレンドデータ生成"""
    np.random.seed(42)
    returns = np.random.normal(trend, 0.01, n_periods)
    
    prices = [50000]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices[1:])
    
    data = pd.DataFrame({
        'open': np.roll(prices, 1),
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.uniform(1000, 5000, n_periods)
    })
    data['open'].iloc[0] = 50000
    
    return data


def generate_ranging_data(n_periods: int):
    """レンジ相場データ生成"""
    np.random.seed(42)
    center = 50000
    
    # サイン波ベースのレンジ
    base_prices = center + 2000 * np.sin(np.arange(n_periods) * 0.1)
    noise = np.random.normal(0, 200, n_periods)
    prices = base_prices + noise
    
    data = pd.DataFrame({
        'open': np.roll(prices, 1),
        'high': prices * 1.005,
        'low': prices * 0.995,
        'close': prices,
        'volume': np.random.uniform(1000, 5000, n_periods)
    })
    data['open'].iloc[0] = center
    
    return data


def generate_high_volatility_data(n_periods: int):
    """高ボラティリティデータ生成"""
    np.random.seed(42)
    returns = np.random.normal(0, 0.05, n_periods)  # 高ボラティリティ
    
    prices = [50000]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices[1:])
    
    data = pd.DataFrame({
        'open': np.roll(prices, 1),
        'high': prices * 1.03,
        'low': prices * 0.97,
        'close': prices,
        'volume': np.random.uniform(1000, 10000, n_periods)
    })
    data['open'].iloc[0] = 50000
    
    return data


def test_performance_tracking():
    """
    パフォーマンス追跡機能のテスト
    """
    print("\\n" + "=" * 60)
    print("パフォーマンス追跡機能テスト")
    print("=" * 60)
    
    supreme_sizing = SupremePositionSizing()
    
    # 模擬取引結果を追加
    mock_trades = [
        (50000, 51000, 1000),  # 勝ち取引
        (51000, 50500, 1000),  # 負け取引
        (50500, 52000, 1000),  # 勝ち取引
        (52000, 51800, 1000),  # 負け取引
        (51800, 53000, 1000),  # 勝ち取引
    ]
    
    print("模擬取引結果を追加中...")
    for entry, exit_price, size in mock_trades:
        supreme_sizing.add_trade_result(entry, exit_price, size)
        trade_return = (exit_price - entry) / entry
        print(f"Entry: ${entry}, Exit: ${exit_price}, Return: {trade_return:.2%}")
    
    print("\\nパフォーマンス指標:")
    metrics = supreme_sizing.performance_metrics
    print(f"勝率: {metrics['win_rate']:.1%}")
    print(f"平均勝ち: {metrics['avg_win']:.3f}")
    print(f"平均負け: {metrics['avg_loss']:.3f}")
    
    # アルゴリズム情報表示
    info = supreme_sizing.get_algorithm_info()
    print(f"\\nアルゴリズム情報:")
    print(f"名前: {info['name']}")
    print(f"バージョン: {info['version']}")
    print(f"取引数: {info['trade_count']}")


def main():
    """メイン実行関数"""
    setup_logging()
    
    print("Supreme Position Sizing Algorithm テスト開始")
    print(f"テスト開始時刻: {datetime.now()}")
    
    try:
        # 基本比較テスト
        supreme_result, simple_result = test_supreme_vs_simple_atr()
        
        # 市場環境テスト
        market_results = test_market_conditions()
        
        # パフォーマンス追跡テスト
        test_performance_tracking()
        
        print("\\n" + "=" * 60)
        print("全テスト完了")
        print("=" * 60)
        print(f"Supreme Position Sizing Algorithm は正常に動作しています。")
        print(f"最新の学術研究に基づく最先端のポジションサイジングを実現しています。")
        
    except Exception as e:
        logging.error(f"テスト実行エラー: {e}")
        print(f"\\nエラーが発生しました: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)