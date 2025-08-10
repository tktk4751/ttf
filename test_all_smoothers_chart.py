#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 **統合スムーサー全種類チャートテスト** 🎯

indicators/smoother/unified_smoother.py にある全てのスムーサーを
チャートに描画してパフォーマンスと特性を比較するテストコード

📊 **機能:**
- 全スムーサーの自動取得と実行
- リアルタイムチャート描画
- 統計情報とパフォーマンス分析
- エラーハンドリングと詳細ログ
- スムージング効果の定量評価

🔧 **対象スムーサー:**
- FRAMA (フラクタル適応移動平均)
- Super Smoother (エーラーズ2極フィルター)
- Ultimate Smoother (究極スムーサー)
- Zero Lag EMA (ゼロラグ指数移動平均)
- Laguerre Filter (ラゲールフィルター)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import traceback
import time
from typing import Dict, List, Tuple, Any
import warnings

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

# 警告を抑制
warnings.filterwarnings('ignore')

try:
    from indicators.smoother import UnifiedSmoother
    from indicators.price_source import PriceSource
except ImportError as e:
    print(f"インポートエラー: {e}")
    print("プロジェクトルートから実行してください")
    exit(1)


def generate_test_data(length: int = 200, complexity: str = 'medium') -> pd.DataFrame:
    """
    テストデータ生成
    
    Args:
        length: データ長
        complexity: 複雑さ ('simple', 'medium', 'complex')
        
    Returns:
        OHLC データフレーム
    """
    print(f"テストデータ生成中... (長さ: {length}, 複雑さ: {complexity})")
    
    np.random.seed(42)  # 再現性のため
    
    # 複雑さに応じたパラメータ設定
    complexity_params = {
        'simple': {'trend': 0.0005, 'volatility': 0.015, 'noise_factor': 0.1, 'jump_probability': 0.01},
        'medium': {'trend': 0.001, 'volatility': 0.025, 'noise_factor': 0.3, 'jump_probability': 0.02},
        'complex': {'trend': 0.002, 'volatility': 0.04, 'noise_factor': 0.5, 'jump_probability': 0.05}
    }
    
    params = complexity_params.get(complexity, complexity_params['medium'])
    
    base_price = 100.0
    trend = params['trend']
    volatility = params['volatility']
    noise_factor = params['noise_factor']
    jump_prob = params['jump_probability']
    
    # 価格系列生成
    prices = [base_price]
    
    for i in range(1, length):
        # 基本トレンド
        base_change = trend
        
        # 周期的変動（複数の周期）
        cycle1 = 0.001 * np.sin(2 * np.pi * i / 20)  # 20期間周期
        cycle2 = 0.0005 * np.sin(2 * np.pi * i / 50)  # 50期間周期
        cycle3 = 0.0003 * np.sin(2 * np.pi * i / 100)  # 100期間周期
        
        # ランダムノイズ
        noise = np.random.normal(0, volatility * noise_factor)
        
        # 時々発生するジャンプ
        if np.random.random() < jump_prob:
            jump = np.random.choice([-1, 1]) * volatility * 3
        else:
            jump = 0
        
        # 総変化量
        total_change = base_change + cycle1 + cycle2 + cycle3 + noise + jump
        
        # ボラティリティクラスタリング効果
        if i > 10:
            recent_vol = np.std(np.diff(prices[-10:]))
            vol_adjust = recent_vol / volatility
            total_change *= (0.7 + 0.6 * vol_adjust)  # ボラティリティの持続性
        
        new_price = prices[-1] * (1 + total_change)
        prices.append(max(new_price, 0.1))  # 負の価格を防ぐ
    
    # OHLC データ構築
    data = []
    for i, close in enumerate(prices):
        # 日中変動の生成
        daily_range = abs(np.random.normal(0, volatility * close * 0.4))
        
        high = close + daily_range * np.random.uniform(0.2, 1.0)
        low = close - daily_range * np.random.uniform(0.2, 1.0)
        
        if i == 0:
            open_price = close
        else:
            # ギャップの生成
            gap = np.random.normal(0, volatility * close * 0.15)
            open_price = prices[i-1] + gap
        
        # OHLC の論理的整合性確保
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.uniform(1000, 20000)
        })
    
    df = pd.DataFrame(data)
    
    # タイムスタンプ設定
    start_date = datetime.now() - timedelta(days=length)
    df.index = [start_date + timedelta(hours=i*4) for i in range(length)]  # 4時間足
    
    print(f"データ生成完了: {len(df)}ポイント")
    print(f"価格範囲: {df['close'].min():.2f} - {df['close'].max():.2f}")
    print(f"平均ボラティリティ: {np.std(df['close'].pct_change().dropna())*100:.2f}%")
    
    return df


def calculate_smoother_statistics(original: np.ndarray, smoothed: np.ndarray) -> Dict[str, float]:
    """
    スムーサーの統計指標を計算
    
    Args:
        original: 元データ
        smoothed: スムーズ済みデータ
        
    Returns:
        統計指標の辞書
    """
    valid_mask = ~(np.isnan(original) | np.isnan(smoothed))
    if np.sum(valid_mask) < 2:
        return {'error': True}
    
    orig_valid = original[valid_mask]
    smooth_valid = smoothed[valid_mask]
    
    # 基本統計
    correlation = np.corrcoef(orig_valid, smooth_valid)[0, 1] if len(orig_valid) > 1 else 0
    
    # スムージング効果 (ボラティリティ削減率)
    orig_vol = np.std(np.diff(orig_valid)) if len(orig_valid) > 1 else 0
    smooth_vol = np.std(np.diff(smooth_valid)) if len(smooth_valid) > 1 else 0
    smoothing_effect = (1 - smooth_vol / orig_vol) * 100 if orig_vol > 0 else 0
    
    # 遅延推定 (クロスコリレーション)
    if len(orig_valid) > 10:
        max_lag = min(20, len(orig_valid) // 4)
        correlations = []
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                corr = np.corrcoef(orig_valid[:lag], smooth_valid[-lag:])[0, 1]
            elif lag > 0:
                corr = np.corrcoef(orig_valid[lag:], smooth_valid[:-lag])[0, 1]
            else:
                corr = np.corrcoef(orig_valid, smooth_valid)[0, 1]
            correlations.append((lag, corr))
        
        best_lag = max(correlations, key=lambda x: abs(x[1]))[0]
        lag_estimate = best_lag
    else:
        lag_estimate = 0
    
    # 平均絶対誤差
    mae = np.mean(np.abs(orig_valid - smooth_valid))
    
    # 最大偏差
    max_deviation = np.max(np.abs(orig_valid - smooth_valid))
    
    return {
        'correlation': correlation,
        'smoothing_effect': smoothing_effect,
        'lag_estimate': lag_estimate,
        'mae': mae,
        'max_deviation': max_deviation,
        'valid_ratio': np.sum(valid_mask) / len(original) * 100,
        'error': False
    }


def test_all_smoothers(data: pd.DataFrame) -> Dict[str, Any]:
    """
    全スムーサーでテスト実行
    
    Args:
        data: OHLCデータ
        
    Returns:
        結果辞書
    """
    print("\n=== 全スムーサーテスト実行 ===")
    
    # 利用可能なスムーサーを取得
    available_smoothers = UnifiedSmoother.get_available_smoothers()
    
    # エイリアスを除外してユニークなスムーサーのみ取得
    unique_smoothers = {}
    seen_classes = set()
    
    for name, description in available_smoothers.items():
        smoother_key = (description, name)
        if smoother_key not in seen_classes:
            unique_smoothers[name] = description
            seen_classes.add(smoother_key)
    
    print(f"テスト対象スムーサー: {len(unique_smoothers)}種類")
    for name, desc in unique_smoothers.items():
        print(f"  {name}: {desc}")
    
    results = {}
    src_prices = PriceSource.calculate_source(data, 'close')
    
    for smoother_name in unique_smoothers.keys():
        print(f"\n{smoother_name} を実行中...")
        start_time = time.time()
        
        try:
            # スムーサー作成
            smoother = UnifiedSmoother(smoother_type=smoother_name, src_type='close')
            
            # 計算実行
            result = smoother.calculate(data)
            
            # 統計計算
            stats = calculate_smoother_statistics(src_prices, result.values)
            
            execution_time = time.time() - start_time
            
            results[smoother_name] = {
                'values': result.values,
                'description': unique_smoothers[smoother_name],
                'parameters': result.parameters,
                'additional_data': result.additional_data,
                'statistics': stats,
                'execution_time': execution_time,
                'success': True,
                'error_message': None
            }
            
            print(f"  ✓ 成功 ({execution_time:.3f}秒)")
            if not stats.get('error', False):
                print(f"    相関係数: {stats['correlation']:.3f}")
                print(f"    スムージング効果: {stats['smoothing_effect']:.1f}%")
                print(f"    推定遅延: {stats['lag_estimate']:.0f}期間")
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            
            results[smoother_name] = {
                'values': np.full(len(data), np.nan),
                'description': unique_smoothers[smoother_name],
                'parameters': {},
                'additional_data': {},
                'statistics': {'error': True},
                'execution_time': execution_time,
                'success': False,
                'error_message': error_msg
            }
            
            print(f"  ✗ エラー: {error_msg}")
    
    # 成功したスムーサーの数
    successful_count = sum(1 for r in results.values() if r['success'])
    print(f"\n実行結果: {successful_count}/{len(unique_smoothers)} 成功")
    
    return results


def create_comprehensive_chart(data: pd.DataFrame, smoother_results: Dict[str, Any]) -> None:
    """
    包括的なチャート作成
    
    Args:
        data: 元のOHLCデータ
        smoother_results: スムーサー結果
    """
    print("\n=== チャート作成中 ===")
    
    # 成功したスムーサーのみ抽出
    successful_results = {k: v for k, v in smoother_results.items() if v['success']}
    
    if not successful_results:
        print("成功したスムーサーがありません")
        return
    
    # 図のサイズと色設定
    fig_width = 16
    fig_height = 12
    colors = plt.cm.tab10(np.linspace(0, 1, len(successful_results)))
    
    # メインチャート作成
    fig, axes = plt.subplots(3, 1, figsize=(fig_width, fig_height))
    fig.suptitle('統合スムーサー全種類比較テスト', fontsize=16, fontweight='bold')
    
    # 時間軸
    time_index = data.index
    
    # 1. 価格とスムーサー比較
    ax1 = axes[0]
    
    # 元価格（薄いグレー）
    ax1.plot(time_index, data['close'], color='lightgray', linewidth=1, alpha=0.7, label='元価格')
    
    # 各スムーサー
    for i, (name, result) in enumerate(successful_results.items()):
        ax1.plot(time_index, result['values'], color=colors[i], linewidth=2, 
                label=f"{name}", alpha=0.8)
    
    ax1.set_title('価格とスムーサー比較', fontsize=14, fontweight='bold')
    ax1.set_ylabel('価格')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. スムージング効果比較 (標準偏差比較)
    ax2 = axes[1]
    
    smoother_names = []
    smoothing_effects = []
    correlations = []
    
    for name, result in successful_results.items():
        stats = result['statistics']
        if not stats.get('error', False):
            smoother_names.append(name)
            smoothing_effects.append(stats['smoothing_effect'])
            correlations.append(stats['correlation'])
    
    if smoother_names:
        # スムージング効果バー
        bars = ax2.bar(range(len(smoother_names)), smoothing_effects, 
                      color=colors[:len(smoother_names)], alpha=0.7)
        ax2.set_title('スムージング効果 (ボラティリティ削減率)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('削減率 (%)')
        ax2.set_xticks(range(len(smoother_names)))
        ax2.set_xticklabels(smoother_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # 値をバーの上に表示
        for i, (bar, effect) in enumerate(zip(bars, smoothing_effects)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{effect:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # 3. 相関係数と遅延
    ax3 = axes[2]
    
    if smoother_names:
        # 相関係数
        ax3_twin = ax3.twinx()
        
        line1 = ax3.plot(range(len(smoother_names)), correlations, 
                        'o-', color='blue', linewidth=2, markersize=8, label='相関係数')
        ax3.set_ylabel('相関係数', color='blue')
        ax3.tick_params(axis='y', labelcolor='blue')
        
        # 遅延
        lags = [successful_results[name]['statistics']['lag_estimate'] 
               for name in smoother_names if not successful_results[name]['statistics'].get('error', False)]
        
        if len(lags) == len(smoother_names):
            line2 = ax3_twin.plot(range(len(smoother_names)), lags, 
                                 's-', color='red', linewidth=2, markersize=8, label='推定遅延')
            ax3_twin.set_ylabel('推定遅延 (期間)', color='red')
            ax3_twin.tick_params(axis='y', labelcolor='red')
        
        ax3.set_title('相関係数と推定遅延', fontsize=14, fontweight='bold')
        ax3.set_xticks(range(len(smoother_names)))
        ax3.set_xticklabels(smoother_names, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
    
    # レイアウト調整
    plt.tight_layout()
    
    # ファイル保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"all_smoothers_comparison_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"チャート保存: {filename}")
    
    # 表示
    plt.show()


def print_detailed_statistics(smoother_results: Dict[str, Any]) -> None:
    """
    詳細統計情報の表示
    
    Args:
        smoother_results: スムーサー結果
    """
    print("\n" + "="*80)
    print("📊 詳細統計情報")
    print("="*80)
    
    successful_results = {k: v for k, v in smoother_results.items() if v['success']}
    
    if not successful_results:
        print("成功したスムーサーがありません")
        return
    
    # テーブルヘッダー
    print(f"{'スムーサー':<20} {'相関':<8} {'効果%':<8} {'遅延':<6} {'MAE':<10} {'時間(s)':<8}")
    print("-" * 80)
    
    # 各スムーサーの統計
    for name, result in successful_results.items():
        stats = result['statistics']
        
        if not stats.get('error', False):
            correlation = stats['correlation']
            smoothing = stats['smoothing_effect']
            lag = stats['lag_estimate']
            mae = stats['mae']
            exec_time = result['execution_time']
            
            print(f"{name:<20} {correlation:<8.3f} {smoothing:<8.1f} {lag:<6.0f} {mae:<10.4f} {exec_time:<8.3f}")
        else:
            print(f"{name:<20} {'ERROR':<8} {'ERROR':<8} {'ERROR':<6} {'ERROR':<10} {result['execution_time']:<8.3f}")
    
    print("-" * 80)
    
    # サマリー統計
    valid_stats = [r['statistics'] for r in successful_results.values() 
                  if not r['statistics'].get('error', False)]
    
    if valid_stats:
        avg_correlation = np.mean([s['correlation'] for s in valid_stats])
        avg_smoothing = np.mean([s['smoothing_effect'] for s in valid_stats])
        avg_lag = np.mean([s['lag_estimate'] for s in valid_stats])
        total_time = sum(r['execution_time'] for r in successful_results.values())
        
        print(f"平均統計値:")
        print(f"  相関係数: {avg_correlation:.3f}")
        print(f"  スムージング効果: {avg_smoothing:.1f}%")
        print(f"  平均遅延: {avg_lag:.1f}期間")
        print(f"  総実行時間: {total_time:.3f}秒")
    
    # エラーがあった場合の情報
    failed_results = {k: v for k, v in smoother_results.items() if not v['success']}
    if failed_results:
        print(f"\n❌ エラーが発生したスムーサー ({len(failed_results)}個):")
        for name, result in failed_results.items():
            print(f"  {name}: {result['error_message']}")


def main():
    """メイン実行関数"""
    print("🎯 統合スムーサー全種類チャートテスト")
    print("=" * 60)
    
    try:
        # テストデータ生成
        data = generate_test_data(length=150, complexity='medium')
        
        # 全スムーサーテスト
        smoother_results = test_all_smoothers(data)
        
        # 詳細統計表示
        print_detailed_statistics(smoother_results)
        
        # チャート作成
        create_comprehensive_chart(data, smoother_results)
        
        print("\n✅ テスト完了")
        
    except Exception as e:
        print(f"\n❌ テスト中にエラーが発生しました: {e}")
        print("\nスタックトレース:")
        traceback.print_exc()


if __name__ == "__main__":
    main()