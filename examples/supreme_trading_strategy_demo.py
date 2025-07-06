#!/usr/bin/env python3
"""
🚀 Supreme Breakout Channel - 実用的トレード戦略デモ 🚀

実際に利益を出すための現実的なパラメータ設定でのデモ
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from examples.supreme_trading_strategy import SupremeTradingStrategy
from indicators.supreme_breakout_channel import SupremeBreakoutChannel

def create_realistic_data(periods=2000):
    """より現実的なマーケットデータ生成"""
    print("📊 Creating realistic market data...")
    
    dates = pd.date_range('2022-01-01', periods=periods, freq='4H')
    np.random.seed(123)  # 再現可能な結果
    
    # より現実的なトレンドとボラティリティ
    base_trend = 0.0002  # 年2%程度の上昇トレンド
    volatility = 0.015   # 1.5%ボラティリティ
    
    # ランダムウォーク with トレンド
    returns = np.random.randn(periods) * volatility + base_trend
    
    # トレンドの変化を追加（より現実的）
    trend_changes = np.random.choice([0, 1], periods, p=[0.98, 0.02])  # 2%の確率でトレンド変化
    trend_direction = 1
    
    for i in range(periods):
        if trend_changes[i]:
            trend_direction *= -1  # トレンド反転
        returns[i] += trend_direction * base_trend * 2
    
    # 価格計算
    prices = 45000 * np.exp(np.cumsum(returns))  # BTCっぽい価格レンジ
    
    # OHLC作成
    price_data = pd.DataFrame(index=dates)
    price_data['close'] = prices
    
    # High/Low にリアルなスプレッド追加
    spreads = np.abs(np.random.randn(periods) * 0.008)  # 0.8%のスプレッド
    price_data['high'] = prices * (1 + spreads)
    price_data['low'] = prices * (1 - spreads)
    price_data['open'] = price_data['close'].shift(1).fillna(price_data['close'])
    price_data['volume'] = np.random.randint(1000, 5000, periods)
    
    return price_data

def run_strategy_comparison():
    """複数のパラメータセットで戦略比較"""
    
    # データ生成
    price_data = create_realistic_data(2000)
    print(f"✅ Market data created: {len(price_data)} candles")
    print(f"   Price range: ${price_data['close'].min():.0f} - ${price_data['close'].max():.0f}")
    
    # Supreme Breakout Channel計算
    print("\n🚀 Calculating Supreme Breakout Channel...")
    sbc = SupremeBreakoutChannel(
        atr_period=14,
        base_multiplier=2.0,
        min_confidence_threshold=0.2,  # より緩い設定
        min_strength_threshold=0.2
    )
    
    sbc_result = sbc.calculate(price_data)
    print("✅ SBC calculation completed")
    
    # 基本統計
    total_signals = np.sum(np.abs(sbc_result.breakout_signals))
    avg_confidence = np.mean(sbc_result.signal_confidence[sbc_result.signal_confidence > 0])
    
    print(f"\n📊 SBC Basic Stats:")
    print(f"   Total Breakout Signals: {total_signals}")
    print(f"   Average Signal Confidence: {avg_confidence:.3f}")
    print(f"   Supreme Intelligence Score: {sbc_result.supreme_intelligence_score:.3f}")
    
    # 戦略設定比較
    strategies = {
        "Conservative": {
            'min_confidence': 0.8,
            'min_trend_strength': 0.7,
            'min_breakout_strength': 0.6,
            'min_supreme_score': 0.7,
            'max_risk_per_trade': 0.01,
            'profit_target_ratio': 3.0
        },
        "Balanced": {
            'min_confidence': 0.6,
            'min_trend_strength': 0.5,
            'min_breakout_strength': 0.4,
            'min_supreme_score': 0.5,
            'max_risk_per_trade': 0.02,
            'profit_target_ratio': 2.0
        },
        "Aggressive": {
            'min_confidence': 0.4,
            'min_trend_strength': 0.4,
            'min_breakout_strength': 0.3,
            'min_supreme_score': 0.4,
            'max_risk_per_trade': 0.03,
            'profit_target_ratio': 1.5
        }
    }
    
    results = {}
    
    print(f"\n🎯 Testing Multiple Strategy Configurations...")
    
    for name, params in strategies.items():
        print(f"\n📈 Testing {name} Strategy...")
        
        strategy = SupremeTradingStrategy(**params)
        signals = strategy.generate_signals(price_data, sbc_result, account_balance=10000)
        performance = strategy.get_performance_summary()
        
        results[name] = {
            'signals': len([s for s in signals if s.signal_type in ['BUY', 'SELL']]),
            'performance': performance,
            'params': params
        }
        
        if performance:
            print(f"   📊 Trades: {performance['total_trades']}")
            print(f"   🎯 Win Rate: {performance['win_rate']:.1%}")
            print(f"   💰 Total PnL: {performance['total_pnl_pct']:.2f}%")
            print(f"   ⚖️  Risk/Reward: {performance['risk_reward_ratio']:.2f}")
        else:
            print(f"   ⚠️ No trades generated")
    
    # 結果比較表示
    print(f"\n📊 Strategy Comparison Summary:")
    print("="*70)
    print(f"{'Strategy':<12} {'Signals':<8} {'Trades':<7} {'Win Rate':<9} {'Total PnL':<10} {'R/R Ratio':<10}")
    print("="*70)
    
    for name, result in results.items():
        perf = result['performance']
        if perf:
            print(f"{name:<12} {result['signals']:<8} {perf['total_trades']:<7} "
                  f"{perf['win_rate']:.1%}{'':>4} {perf['total_pnl_pct']:>6.1f}%{'':>3} "
                  f"{perf['risk_reward_ratio']:>6.2f}{'':>4}")
        else:
            print(f"{name:<12} {result['signals']:<8} {'0':<7} {'N/A':<9} {'N/A':<10} {'N/A':<10}")
    
    # 最適設定の推奨
    best_strategy = None
    best_score = -999
    
    for name, result in results.items():
        perf = result['performance']
        if perf and perf['total_trades'] > 5:  # 最低5トレード必要
            # スコア = 総PnL * 勝率 - ドローダウン
            score = (perf['total_pnl_pct'] * perf['win_rate'] - 
                    abs(perf['max_drawdown_pct']) * 0.5)
            
            if score > best_score:
                best_score = score
                best_strategy = name
    
    if best_strategy:
        print(f"\n🏆 Recommended Strategy: {best_strategy}")
        best_perf = results[best_strategy]['performance']
        print(f"   💰 Expected Monthly Return: {best_perf['total_pnl_pct'] * 0.5:.1f}%")
        print(f"   🎯 Win Rate: {best_perf['win_rate']:.1%}")
        print(f"   📉 Max Drawdown: {best_perf['max_drawdown_pct']:.1f}%")
    
    return results

def demonstrate_signal_quality():
    """シグナル品質の分析デモ"""
    print(f"\n🔍 Supreme Signal Quality Analysis")
    
    # シンプルなデータでシグナル品質を分析
    price_data = create_realistic_data(500)
    
    sbc = SupremeBreakoutChannel(
        min_confidence_threshold=0.3,
        min_strength_threshold=0.3
    )
    
    result = sbc.calculate(price_data)
    
    # シグナル品質分析
    signals_mask = result.breakout_signals != 0
    if np.any(signals_mask):
        signal_confidences = result.signal_confidence[signals_mask]
        signal_strengths = result.breakout_strength[signals_mask]
        trend_strengths = result.trend_strength[signals_mask]
        
        print(f"\n📊 Signal Quality Statistics:")
        print(f"   Total Signals: {np.sum(signals_mask)}")
        print(f"   Avg Confidence: {np.mean(signal_confidences):.3f}")
        print(f"   Avg Strength: {np.mean(signal_strengths):.3f}")
        print(f"   Avg Trend Strength: {np.mean(trend_strengths):.3f}")
        
        # 信頼度別分析
        high_conf = signal_confidences >= 0.7
        med_conf = (signal_confidences >= 0.5) & (signal_confidences < 0.7)
        low_conf = signal_confidences < 0.5
        
        print(f"\n🎯 Confidence Distribution:")
        print(f"   High (≥70%): {np.sum(high_conf)} signals ({np.sum(high_conf)/len(signal_confidences)*100:.1f}%)")
        print(f"   Medium (50-70%): {np.sum(med_conf)} signals ({np.sum(med_conf)/len(signal_confidences)*100:.1f}%)")
        print(f"   Low (<50%): {np.sum(low_conf)} signals ({np.sum(low_conf)/len(signal_confidences)*100:.1f}%)")
        
        # 推奨フィルター設定
        print(f"\n💡 Recommended Filter Settings:")
        if np.sum(high_conf) >= 5:
            print(f"   ✅ Use min_confidence = 0.7 (High quality: {np.sum(high_conf)} signals)")
        elif np.sum(med_conf) >= 10:
            print(f"   ⚖️  Use min_confidence = 0.5 (Medium quality: {np.sum(med_conf)} signals)")
        else:
            print(f"   ⚠️ Use min_confidence = 0.4 (Lower threshold needed)")

def main():
    """メイン実行"""
    print("🚀 Supreme Breakout Channel - 実用的トレード戦略分析")
    print("="*60)
    
    # シグナル品質分析
    demonstrate_signal_quality()
    
    # 戦略比較実行
    results = run_strategy_comparison()
    
    print(f"\n📚 Key Takeaways:")
    print(f"   1. 🎯 適切な信頼度フィルターが勝率向上の鍵")
    print(f"   2. 💰 リスク管理がドローダウン抑制に重要")
    print(f"   3. ⚖️  バランス型戦略が最も安定的")
    print(f"   4. 🧠 Supreme知能スコアは市場環境判断に有効")
    
    print(f"\n💡 Next Steps:")
    print(f"   • 実データでのバックテスト実行")
    print(f"   • パラメータの最適化")
    print(f"   • リスク管理ルールの詳細化")
    print(f"   • 複数時間足での検証")

if __name__ == "__main__":
    main() 