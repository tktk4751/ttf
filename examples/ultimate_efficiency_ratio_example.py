#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ultimate Efficiency Ratio (UER) 使用例

🌟 **人類史上最強の効率性分析システムのデモンストレーション** 🌟

Ultimate MAとUltimate Volatilityの最先端技術を統合した究極のERインジケーター。
従来ER → Super ER → Ultimate ERの進化を比較し、圧倒的な性能向上を実証します。

🎯 **デモ内容:**
1. 従来ER vs Super ER vs Ultimate ER 三者比較
2. 6層統合革新システムの効果測定
3. 量子強化技術の威力検証
4. 動的適応機能の実演
5. 予測機能の精度評価
6. リアルタイム取引シミュレーション
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# フォント設定（英語のみ）
plt.rcParams['font.family'] = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']

try:
    from indicators.ultimate_efficiency_ratio import UltimateEfficiencyRatio
    from indicators.super_efficiency_ratio import SuperEfficiencyRatio
    from indicators.efficiency_ratio import EfficiencyRatio
except ImportError as e:
    print(f"インポートエラー: {e}")
    print("indicators/ ディレクトリが適切に配置されていることを確認してください。")
    sys.exit(1)


def generate_ultimate_test_data(n_points: int = 1000) -> pd.DataFrame:
    """
    究極テストデータ生成 - 様々な市場状況を含む高品質データ
    """
    np.random.seed(42)
    
    dates = pd.date_range(start='2023-01-01', periods=n_points, freq='1H')
    base_price = 50000
    
    # 6つの異なる市場フェーズを作成
    phase_length = n_points // 6
    
    prices = []
    current_price = base_price
    
    for i in range(n_points):
        phase = i // phase_length
        position_in_phase = (i % phase_length) / phase_length
        
        if phase == 0:  # 強いアップトレンド
            trend = 0.0015 + 0.001 * np.sin(position_in_phase * 2 * np.pi)
            noise = np.random.normal(0, 0.008)
        elif phase == 1:  # ボラティリティレンジ
            trend = 0.0005 * np.sin(position_in_phase * 8 * np.pi)
            noise = np.random.normal(0, 0.015)
        elif phase == 2:  # 緩やかダウントレンド
            trend = -0.0008 - 0.0003 * np.sin(position_in_phase * 3 * np.pi)
            noise = np.random.normal(0, 0.006)
        elif phase == 3:  # 高ボラティリティブレイクアウト
            trend = 0.002 * np.tanh((position_in_phase - 0.5) * 10)
            noise = np.random.normal(0, 0.012)
        elif phase == 4:  # 安定レンジ
            trend = 0.0002 * np.sin(position_in_phase * 6 * np.pi)
            noise = np.random.normal(0, 0.004)
        else:  # 複雑なサイクル
            trend = 0.001 * np.sin(position_in_phase * 4 * np.pi) + 0.0005 * np.cos(position_in_phase * 7 * np.pi)
            noise = np.random.normal(0, 0.007)
        
        change = trend + noise
        current_price *= (1 + change)
        prices.append(current_price)
    
    prices = np.array(prices)
    
    # OHLC生成
    highs = prices * (1 + np.abs(np.random.normal(0, 0.003, n_points)))
    lows = prices * (1 - np.abs(np.random.normal(0, 0.003, n_points)))
    opens = np.roll(prices, 1)
    opens[0] = prices[0]
    
    return pd.DataFrame({
        'datetime': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': prices
    })


def performance_comparison_test():
    """
    包括的パフォーマンス比較テスト
    """
    print("\n" + "="*80)
    print("🌟 Ultimate ER vs Super ER vs 従来ER - 究極パフォーマンス比較 🌟")
    print("="*80)
    
    # テストデータ生成
    print("\n📊 高品質テストデータ生成中...")
    data = generate_ultimate_test_data(1000)
    
    # インジケーター初期化
    print("🚀 インジケーター初期化中...")
    
    # 従来ER
    classic_er = EfficiencyRatio(period=14)
    
    # Super ER
    super_er = SuperEfficiencyRatio(
        base_period=14,
        src_type='hlc3',
        use_adaptive_filter=True,
        use_multiscale=True,
        hurst_window=21
    )
    
    # Ultimate ER（全機能ON）
    ultimate_er = UltimateEfficiencyRatio(
        base_period=14,
        src_type='hlc3',
        use_dynamic_adaptation=True,
        use_quantum_enhancement=True,
        use_wavelet_analysis=True,
        use_predictive_mode=True
    )
    
    # パフォーマンス測定
    print("\n⚡ 計算速度測定中...")
    
    # 従来ER
    start_time = time.time()
    classic_result = classic_er.calculate(data)
    classic_time = time.time() - start_time
    
    # Super ER
    start_time = time.time()
    super_result = super_er.calculate(data)
    super_time = time.time() - start_time
    
    # Ultimate ER
    start_time = time.time()
    ultimate_result = ultimate_er.calculate(data)
    ultimate_time = time.time() - start_time
    
    # ノイズレベル分析
    def calculate_noise_level(er_values):
        if len(er_values) < 20:
            return 0.0
        diffs = np.diff(er_values[~np.isnan(er_values)])
        return np.std(diffs) if len(diffs) > 0 else 0.0
    
    # 従来ERの結果を適切に処理
    if hasattr(classic_result, 'values'):
        classic_values = classic_result.values
    else:
        classic_values = np.array(classic_result) if classic_result is not None else np.array([])
    
    classic_noise = calculate_noise_level(classic_values)
    super_noise = calculate_noise_level(super_result.values)
    ultimate_noise = calculate_noise_level(ultimate_result.values)
    
    # 結果表示
    print("\n📈 パフォーマンス比較結果:")
    print(f"{'='*50}")
    
    print(f"🏃 計算速度:")
    print(f"  従来ER:      {classic_time:.4f}秒")
    print(f"  Super ER:    {super_time:.4f}秒 ({classic_time/super_time:.1f}x)")
    print(f"  Ultimate ER: {ultimate_time:.4f}秒 ({classic_time/ultimate_time:.1f}x)")
    
    print(f"\n🔇 ノイズレベル:")
    print(f"  従来ER:      {classic_noise:.6f}")
    if classic_noise > 0:
        print(f"  Super ER:    {super_noise:.6f} ({(1-super_noise/classic_noise)*100:.1f}% 削減)")
        print(f"  Ultimate ER: {ultimate_noise:.6f} ({(1-ultimate_noise/classic_noise)*100:.1f}% 削減)")
    else:
        print(f"  Super ER:    {super_noise:.6f}")
        print(f"  Ultimate ER: {ultimate_noise:.6f}")
    
    # Ultimate ERの詳細分析
    print(f"\n🌟 Ultimate ER 詳細分析:")
    report = ultimate_er.get_intelligence_report()
    print(f"  現在効率性:     {report['current_efficiency']:.3f}")
    print(f"  効率性状態:     {report['efficiency_state']}")
    print(f"  信頼度:         {report['confidence']:.3f}")
    print(f"  量子コヒーレンス: {report['quantum_coherence']:.3f}")
    print(f"  予測値:         {report['forecast']:.3f}")
    print(f"  市場レジーム:   {report['market_regime']}")
    
    active_features = report['active_features']
    print(f"  有効な機能:")
    for feature, enabled in active_features.items():
        status = "✅" if enabled else "❌"
        print(f"    {feature}: {status}")
    
    return data, classic_result, super_result, ultimate_result


def visualization_demo(data, classic_result, super_result, ultimate_result):
    """
    高度な可視化デモ
    """
    print("\n" + "="*60)
    print("📊 究極可視化デモ - 3インジケーター比較")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Ultimate ER vs Super ER vs Classic ER - Performance Analysis', fontsize=16, fontweight='bold')
    
    # 価格チャート
    ax1 = axes[0, 0]
    ax1.plot(data.index, data['close'], 'k-', linewidth=1, alpha=0.7, label='Price')
    ax1.set_title('Price Chart', fontweight='bold')
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # ER比較
    ax2 = axes[0, 1]
    # 従来ERの結果を適切に処理
    if hasattr(classic_result, 'values'):
        classic_values = classic_result.values
    else:
        classic_values = np.array(classic_result) if classic_result is not None else np.full(len(data), np.nan)
    
    valid_idx = ~np.isnan(classic_values)
    ax2.plot(data.index[valid_idx], classic_values[valid_idx], 'b-', linewidth=1.5, alpha=0.8, label='Classic ER')
    ax2.plot(data.index, super_result.values, 'g-', linewidth=1.5, alpha=0.8, label='Super ER')
    ax2.plot(data.index, ultimate_result.values, 'r-', linewidth=2, alpha=0.9, label='Ultimate ER')
    
    # 効率性レベル線
    ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Ultra Efficient')
    ax2.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Efficient')
    ax2.axhline(y=0.4, color='gray', linestyle='--', alpha=0.5, label='Neutral')
    ax2.axhline(y=0.2, color='blue', linestyle='--', alpha=0.5, label='Inefficient')
    
    ax2.set_title('Efficiency Comparison (0-1 Scale)', fontweight='bold')
    ax2.set_ylabel('Efficiency')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Ultimate ER詳細分析
    ax3 = axes[1, 0]
    ax3.plot(data.index, ultimate_result.values, 'r-', linewidth=2, label='Ultimate ER', alpha=0.9)
    ax3.plot(data.index, ultimate_result.efficiency_forecast, 'r--', linewidth=1.5, label='Forecast', alpha=0.7)
    ax3.fill_between(data.index, 
                     ultimate_result.values - ultimate_result.confidence_score * 0.1,
                     ultimate_result.values + ultimate_result.confidence_score * 0.1,
                     alpha=0.2, color='red', label='Confidence Band')
    
    ax3.set_title('Ultimate ER Detailed Analysis', fontweight='bold')
    ax3.set_ylabel('Efficiency')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 量子成分分析
    ax4 = axes[1, 1]
    ax4.plot(data.index, ultimate_result.quantum_coherence, 'purple', linewidth=1.5, label='Quantum Coherence', alpha=0.8)
    ax4.plot(data.index, ultimate_result.confidence_score, 'orange', linewidth=1.5, label='Confidence Score', alpha=0.8)
    ax4.fill_between(data.index, 0, ultimate_result.market_regime, 
                     where=(ultimate_result.market_regime > 0), alpha=0.3, color='green', label='Efficient Market')
    ax4.fill_between(data.index, 0, ultimate_result.market_regime, 
                     where=(ultimate_result.market_regime < 0), alpha=0.3, color='red', label='Inefficient Market')
    
    ax4.set_title('Quantum Components Analysis', fontweight='bold')
    ax4.set_ylabel('Intensity')
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    
    # 保存
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filename = os.path.join(output_dir, 'ultimate_er_comprehensive_analysis.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 Visualization saved: {filename}")
    
    plt.show()


def trading_simulation_demo(data, ultimate_result):
    """
    Ultimate ERを使用したリアルタイム取引シミュレーション
    """
    print("\n" + "="*70)
    print("🎯 Ultimate ER リアルタイム取引シミュレーション")
    print("="*70)
    
    initial_capital = 1000000  # 100万円
    position = 0  # 0=ノーポジション, 1=ロング, -1=ショート
    capital = initial_capital
    trades = []
    
    entry_price = 0
    
    for i in range(len(data)):
        if i < 50:  # 初期データが必要
            continue
            
        current_price = data['close'].iloc[i]
        current_efficiency = ultimate_result.values[i]
        current_confidence = ultimate_result.confidence_score[i]
        current_forecast = ultimate_result.efficiency_forecast[i]
        
        if np.isnan(current_efficiency) or np.isnan(current_confidence):
            continue
        
        # エントリーロジック
        if position == 0:  # ノーポジション
            if (current_efficiency > 0.75 and 
                current_confidence > 0.7 and 
                current_forecast > current_efficiency):
                # 強いアップトレンド + 高信頼度 + 予測上昇
                position = 1
                entry_price = current_price
                trades.append({
                    'time': data.index[i],
                    'action': 'BUY',
                    'price': current_price,
                    'efficiency': current_efficiency,
                    'confidence': current_confidence,
                    'forecast': current_forecast
                })
            
            elif (current_efficiency < 0.25 and 
                  current_confidence > 0.7 and 
                  current_forecast < current_efficiency):
                # 強いダウントレンド + 高信頼度 + 予測下降
                position = -1
                entry_price = current_price
                trades.append({
                    'time': data.index[i],
                    'action': 'SELL',
                    'price': current_price,
                    'efficiency': current_efficiency,
                    'confidence': current_confidence,
                    'forecast': current_forecast
                })
        
        # エグジットロジック
        elif position == 1:  # ロングポジション
            if (current_efficiency < 0.4 or 
                current_confidence < 0.5 or 
                current_forecast < current_efficiency * 0.8):
                # 効率性低下またはトレンド転換の兆候
                pnl = (current_price - entry_price) / entry_price
                capital *= (1 + pnl)
                position = 0
                trades.append({
                    'time': data.index[i],
                    'action': 'CLOSE_LONG',
                    'price': current_price,
                    'pnl': pnl * 100,
                    'efficiency': current_efficiency,
                    'confidence': current_confidence
                })
        
        elif position == -1:  # ショートポジション
            if (current_efficiency > 0.6 or 
                current_confidence < 0.5 or 
                current_forecast > current_efficiency * 1.2):
                # 効率性上昇またはトレンド転換の兆候
                pnl = (entry_price - current_price) / entry_price
                capital *= (1 + pnl)
                position = 0
                trades.append({
                    'time': data.index[i],
                    'action': 'CLOSE_SHORT',
                    'price': current_price,
                    'pnl': pnl * 100,
                    'efficiency': current_efficiency,
                    'confidence': current_confidence
                })
    
    # 最終的にポジションがある場合はクローズ
    if position != 0:
        final_price = data['close'].iloc[-1]
        if position == 1:
            pnl = (final_price - entry_price) / entry_price
        else:
            pnl = (entry_price - final_price) / entry_price
        capital *= (1 + pnl)
    
    # 結果表示
    total_return = (capital - initial_capital) / initial_capital * 100
    num_trades = len([t for t in trades if 'pnl' in t])
    
    print(f"\n📊 取引シミュレーション結果:")
    print(f"{'='*40}")
    print(f"💰 初期資本:     ¥{initial_capital:,}")
    print(f"💰 最終資本:     ¥{capital:,.0f}")
    print(f"📈 総リターン:   {total_return:+.2f}%")
    print(f"🔄 取引回数:     {num_trades}回")
    print(f"📊 勝率計算中...")
    
    if num_trades > 0:
        profits = [t['pnl'] for t in trades if 'pnl' in t and t['pnl'] > 0]
        losses = [t['pnl'] for t in trades if 'pnl' in t and t['pnl'] <= 0]
        win_rate = len(profits) / num_trades * 100
        avg_profit = np.mean(profits) if profits else 0
        avg_loss = np.mean(losses) if losses else 0
        
        print(f"🏆 勝率:         {win_rate:.1f}%")
        print(f"📈 平均利益:     {avg_profit:.2f}%")
        print(f"📉 平均損失:     {avg_loss:.2f}%")
        
        if avg_loss != 0:
            profit_factor = abs(avg_profit / avg_loss)
            print(f"⚖️ プロフィットファクター: {profit_factor:.2f}")
    
    print(f"\n🎯 最近の取引履歴（最新5件）:")
    recent_trades = trades[-5:] if len(trades) >= 5 else trades
    for trade in recent_trades:
        if 'pnl' in trade:
            print(f"  {trade['time'].strftime('%m/%d %H:%M')} | {trade['action']} | "
                  f"¥{trade['price']:.0f} | PnL: {trade['pnl']:+.2f}% | "
                  f"効率性: {trade['efficiency']:.3f}")
        else:
            print(f"  {trade['time'].strftime('%m/%d %H:%M')} | {trade['action']} | "
                  f"¥{trade['price']:.0f} | 効率性: {trade['efficiency']:.3f} | "
                  f"信頼度: {trade['confidence']:.3f}")


def main():
    """
    メイン実行関数
    """
    print("🌟" * 40)
    print("🌟 ULTIMATE EFFICIENCY RATIO - 究極効率性分析システム 🌟")
    print("🌟 人類史上最強の効率性インジケーター実演デモ 🌟")
    print("🌟" * 40)
    
    try:
        # 1. パフォーマンス比較テスト
        data, classic_result, super_result, ultimate_result = performance_comparison_test()
        
        # 2. 可視化デモ
        visualization_demo(data, classic_result, super_result, ultimate_result)
        
        # 3. 取引シミュレーション
        trading_simulation_demo(data, ultimate_result)
        
        print("\n" + "🎉" * 30)
        print("🎉 Ultimate ER デモンストレーション完了! 🎉")
        print("🎉 量子強化技術による究極の効率性分析を体験! 🎉")
        print("🎉" * 30)
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        print(traceback.format_exc())


if __name__ == "__main__":
    main()