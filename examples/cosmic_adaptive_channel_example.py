#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🌌 **Cosmic Adaptive Channel (CAC) - 宇宙最強ブレイクアウト戦略デモ** 🌌

🎯 **革命的8層ハイブリッドシステムのデモンストレーション:**
- **量子統計フュージョン**: 量子コヒーレンス + 統計回帰の融合
- **ヒルベルト位相解析**: 瞬時トレンド検出 + 位相遅延ゼロ
- **神経適応学習**: 市場パターン自動学習 + 動的重み調整
- **動的チャネル幅**: トレンド強度反比例 + 偽シグナル防御
- **超低遅延処理**: ゼロラグ + 予測補正システム
- **ボラティリティレジーム**: リアルタイム市場状態検出
- **超追従適応**: 瞬時相場変化対応 + 学習型最適化
- **ブレイクアウト予測**: 突破確率 + タイミング予測

🏆 **トレンドフォロー最適化:**
- トレンド強い → チャネル幅縮小 → 早期エントリー
- トレンド弱い → チャネル幅拡大 → 偽シグナル回避
- ボラティリティ高 → 適応調整 → 安定性確保
- 相場転換 → 瞬時検出 → 即座対応
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from indicators.cosmic_adaptive_channel import CosmicAdaptiveChannel
from api.binance_data_fetcher import BinanceDataFetcher


def create_sample_data(length: int = 1000) -> pd.DataFrame:
    """サンプル価格データを生成（リアルな相場パターンをシミュレート）"""
    np.random.seed(42)
    
    # 基本トレンド + ノイズ + ボラティリティクラスター
    trend_changes = np.random.choice([-1, 0, 1], length//50, p=[0.3, 0.4, 0.3])
    trend_periods = [length//50] * 50
    
    prices = [100.0]
    volatility_regime = 1
    
    for i in range(1, length):
        # トレンド期間の更新
        period_idx = min(i // (length//50), len(trend_changes)-1)
        trend_direction = trend_changes[period_idx]
        
        # ボラティリティレジーム（時々変化）
        if np.random.random() < 0.02:
            volatility_regime = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.2, 0.4, 0.2, 0.1])
        
        # ボラティリティ調整
        base_vol = {1: 0.005, 2: 0.01, 3: 0.015, 4: 0.025, 5: 0.04}[volatility_regime]
        
        # 価格変動計算
        trend_component = trend_direction * 0.001
        random_component = np.random.normal(0, base_vol)
        
        # Mean reversion component（レンジ相場時）
        if trend_direction == 0:
            mean_reversion = (100 - prices[-1]) * 0.001
            change = trend_component + random_component + mean_reversion
        else:
            change = trend_component + random_component
        
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 0.1))  # 価格の下限
    
    # OHLC データ作成
    data = []
    for i in range(len(prices)):
        if i == 0:
            open_price = close_price = high_price = low_price = prices[i]
        else:
            close_price = prices[i]
            open_price = prices[i-1]
            
            # 高値・安値の生成
            volatility = abs(close_price - open_price) * np.random.uniform(1.2, 2.5)
            high_price = max(open_price, close_price) + volatility * np.random.uniform(0, 0.8)
            low_price = min(open_price, close_price) - volatility * np.random.uniform(0, 0.8)
        
        timestamp = datetime.now() - timedelta(minutes=(len(prices)-i))
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': np.random.uniform(1000, 10000)
        })
    
    return pd.DataFrame(data)


def analyze_cosmic_adaptive_channel(data: pd.DataFrame, symbol: str = "SAMPLE") -> dict:
    """宇宙最強適応チャネルの詳細解析"""
    print(f"\n🌌 Cosmic Adaptive Channel 解析開始 - {symbol}")
    print("=" * 80)
    
    # Cosmic Adaptive Channel を計算
    cac = CosmicAdaptiveChannel(
        atr_period=21,
        base_multiplier=2.0,
        quantum_window=50,
        neural_window=100,
        volatility_window=30,
        src_type='hlc3'
    )
    
    result = cac.calculate(data)
    
    # 宇宙知能レポートを取得
    intelligence_report = cac.get_cosmic_intelligence_report()
    
    print(f"🎯 現在のトレンドフェーズ: {intelligence_report['current_trend_phase']}")
    print(f"🌊 現在のボラティリティレジーム: {intelligence_report['current_volatility_regime']}")
    print(f"🚀 現在のブレイクアウト確率: {intelligence_report['current_breakout_probability']:.3f}")
    print(f"🧠 宇宙知能スコア: {intelligence_report['cosmic_intelligence_score']:.3f}")
    print(f"📊 総ブレイクアウトシグナル数: {intelligence_report['total_breakout_signals']}")
    print(f"🎯 平均信頼度: {intelligence_report['average_confidence']:.3f}")
    print(f"⚡ 偽シグナル率: {intelligence_report['false_signal_rate']:.3f}")
    print(f"🔄 チャネル効率度: {intelligence_report['channel_efficiency']:.3f}")
    print(f"🧠 神経適応度: {intelligence_report['neural_adaptation']:.3f}")
    print(f"⚛️ 量子コヒーレンス: {intelligence_report['quantum_coherence']:.3f}")
    
    # シグナル解析
    signals = result.breakout_signals
    confidences = result.breakout_confidence
    
    up_signals = np.sum(signals == 1)
    down_signals = np.sum(signals == -1)
    
    print(f"\n📈 上昇ブレイクアウト: {up_signals}回")
    print(f"📉 下降ブレイクアウト: {down_signals}回")
    
    # トレンド解析
    trend_analysis = cac.get_trend_analysis()
    if trend_analysis:
        current_trend_strength = trend_analysis['trend_strength'][-1] if len(trend_analysis['trend_strength']) > 0 else 0
        current_momentum = trend_analysis['trend_momentum'][-1] if len(trend_analysis['trend_momentum']) > 0 else 0
        current_continuation = trend_analysis['continuation_strength'][-1] if len(trend_analysis['continuation_strength']) > 0 else 0
        current_reversal = trend_analysis['reversal_probability'][-1] if len(trend_analysis['reversal_probability']) > 0 else 0
        
        print(f"\n🎯 現在のトレンド強度: {current_trend_strength:.3f}")
        print(f"⚡ 現在のトレンド勢い: {current_momentum:.3f}")
        print(f"📈 継続強度: {current_continuation:.3f}")
        print(f"🔄 反転確率: {current_reversal:.3f}")
    
    return {
        'result': result,
        'intelligence_report': intelligence_report,
        'signals_up': up_signals,
        'signals_down': down_signals,
        'cac_indicator': cac
    }


def create_comprehensive_chart(data: pd.DataFrame, analysis_result: dict, symbol: str = "SAMPLE"):
    """宇宙最強チャートを作成"""
    result = analysis_result['result']
    
    # 4つのサブプロットを作成
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(f'🌌 Cosmic Adaptive Channel - 宇宙最強ブレイクアウト解析 ({symbol})', fontsize=16, fontweight='bold')
    
    # 価格とチャネル
    ax1.plot(data.index, data['close'], label='価格', color='black', linewidth=1, alpha=0.8)
    ax1.plot(data.index, result.upper_channel, label='上側チャネル', color='lime', linewidth=2, alpha=0.7)
    ax1.plot(data.index, result.lower_channel, label='下側チャネル', color='red', linewidth=2, alpha=0.7)
    ax1.plot(data.index, result.midline, label='宇宙フィルタ中央線', color='blue', linewidth=1.5, alpha=0.8)
    
    # チャネルエリア塗りつぶし
    ax1.fill_between(data.index, result.upper_channel, result.lower_channel, 
                     alpha=0.1, color='purple', label='チャネルエリア')
    
    # ブレイクアウトシグナル
    up_signals = np.where(result.breakout_signals == 1)[0]
    down_signals = np.where(result.breakout_signals == -1)[0]
    
    if len(up_signals) > 0:
        ax1.scatter(up_signals, data['close'].iloc[up_signals], 
                   color='lime', marker='^', s=100, label=f'上昇ブレイクアウト ({len(up_signals)})', zorder=5)
    if len(down_signals) > 0:
        ax1.scatter(down_signals, data['close'].iloc[down_signals], 
                   color='red', marker='v', s=100, label=f'下降ブレイクアウト ({len(down_signals)})', zorder=5)
    
    ax1.set_title('🚀 価格 & 宇宙最強適応チャネル & ブレイクアウトシグナル')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 量子統計解析
    ax2.plot(data.index, result.quantum_coherence, label='量子コヒーレンス', color='purple', linewidth=2)
    ax2.plot(data.index, result.statistical_trend, label='統計トレンド', color='orange', linewidth=2)
    ax2.plot(data.index, result.trend_strength, label='統合トレンド強度', color='red', linewidth=2)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.set_title('⚛️ 量子統計フュージョン解析')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1, 1)
    
    # 神経適応システム
    ax3.plot(data.index, result.neural_weights, label='神経重み', color='green', linewidth=2)
    ax3.plot(data.index, result.adaptation_score, label='適応スコア', color='blue', linewidth=2)
    ax3.plot(data.index, result.memory_state, label='記憶状態', color='purple', linewidth=2)
    ax3.plot(data.index, result.learning_velocity, label='学習速度', color='orange', linewidth=1)
    ax3.set_title('🧠 神経適応学習システム')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # ボラティリティレジーム & チャネル効率
    ax4_twin = ax4.twinx()
    ax4.plot(data.index, result.volatility_regime, label='ボラティリティレジーム', color='red', linewidth=2, marker='.')
    ax4.plot(data.index, result.regime_stability, label='レジーム安定度', color='orange', linewidth=2)
    ax4_twin.plot(data.index, result.channel_efficiency, label='チャネル効率度', color='green', linewidth=2)
    ax4_twin.plot(data.index, result.breakout_confidence, label='ブレイクアウト信頼度', color='blue', linewidth=1, alpha=0.7)
    
    ax4.set_title('🌊 ボラティリティレジーム & チャネル効率度')
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 6)
    ax4_twin.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # 保存
    output_path = f'examples/output/cosmic_adaptive_channel_{symbol.lower()}_analysis.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 チャートを保存しました: {output_path}")
    
    plt.show()


def create_strategy_performance_chart(data: pd.DataFrame, analysis_result: dict, symbol: str = "SAMPLE"):
    """戦略パフォーマンスチャートを作成"""
    result = analysis_result['result']
    
    # シンプルなトレンドフォロー戦略をシミュレート
    signals = result.breakout_signals
    confidences = result.breakout_confidence
    prices = data['close'].values
    
    # 戦略リターン計算
    position = 0
    returns = []
    trades = []
    entry_price = 0
    
    for i in range(len(signals)):
        if signals[i] != 0 and confidences[i] > 0.5:  # 高信頼度シグナルのみ
            if signals[i] == 1 and position <= 0:  # 上昇ブレイクアウト
                if position < 0:  # ショートポジションクローズ
                    ret = (entry_price - prices[i]) / entry_price
                    returns.append(ret)
                    trades.append({'type': 'close_short', 'price': prices[i], 'return': ret, 'index': i})
                
                position = 1
                entry_price = prices[i]
                trades.append({'type': 'open_long', 'price': prices[i], 'index': i})
                
            elif signals[i] == -1 and position >= 0:  # 下降ブレイクアウト
                if position > 0:  # ロングポジションクローズ
                    ret = (prices[i] - entry_price) / entry_price
                    returns.append(ret)
                    trades.append({'type': 'close_long', 'price': prices[i], 'return': ret, 'index': i})
                
                position = -1
                entry_price = prices[i]
                trades.append({'type': 'open_short', 'price': prices[i], 'index': i})
    
    # 最終ポジションクローズ
    if position != 0:
        final_price = prices[-1]
        if position > 0:
            ret = (final_price - entry_price) / entry_price
        else:
            ret = (entry_price - final_price) / entry_price
        returns.append(ret)
        trades.append({'type': 'final_close', 'price': final_price, 'return': ret, 'index': len(prices)-1})
    
    # 累積リターン計算
    if returns:
        cumulative_returns = np.cumprod([1 + r for r in returns])
        total_return = cumulative_returns[-1] - 1
        win_rate = len([r for r in returns if r > 0]) / len(returns)
        
        print(f"\n📊 戦略パフォーマンス分析:")
        print(f"🎯 総取引数: {len(returns)}")
        print(f"💰 総リターン: {total_return:.2%}")
        print(f"🏆 勝率: {win_rate:.2%}")
        print(f"📈 平均リターン: {np.mean(returns):.2%}")
        print(f"📊 リターン標準偏差: {np.std(returns):.2%}")
        if np.std(returns) > 0:
            sharpe_approx = np.mean(returns) / np.std(returns)
            print(f"⚡ シャープレシオ (概算): {sharpe_approx:.2f}")
    
    # パフォーマンスチャート作成
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle(f'🎯 Cosmic Adaptive Channel - トレンドフォロー戦略パフォーマンス ({symbol})', fontsize=14, fontweight='bold')
    
    # 価格とトレード
    ax1.plot(data.index, data['close'], label='価格', color='black', linewidth=1)
    ax1.plot(data.index, result.upper_channel, color='lime', alpha=0.5, linewidth=1)
    ax1.plot(data.index, result.lower_channel, color='red', alpha=0.5, linewidth=1)
    
    # トレードマーク
    for trade in trades:
        idx = trade['index']
        price = trade['price']
        if trade['type'] == 'open_long':
            ax1.scatter(idx, price, color='lime', marker='^', s=100, zorder=5)
        elif trade['type'] == 'open_short':
            ax1.scatter(idx, price, color='red', marker='v', s=100, zorder=5)
        elif trade['type'] in ['close_long', 'close_short']:
            color = 'green' if trade['return'] > 0 else 'red'
            ax1.scatter(idx, price, color=color, marker='x', s=80, zorder=5)
    
    ax1.set_title('💹 価格チャート & トレードポイント')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 累積リターン
    if returns:
        cum_ret_indices = [trades[i]['index'] for i in range(len(returns))]
        ax2.plot(cum_ret_indices, [(c-1)*100 for c in cumulative_returns], 
                color='blue', linewidth=2, label=f'累積リターン ({total_return:.1%})')
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # 個別トレードリターン
        trade_returns = [r*100 for r in returns]
        colors = ['green' if r > 0 else 'red' for r in returns]
        ax2.bar(cum_ret_indices, trade_returns, alpha=0.6, color=colors, label='個別トレードリターン')
    
    ax2.set_title('📈 戦略パフォーマンス')
    ax2.set_ylabel('リターン (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    output_path = f'examples/output/cosmic_adaptive_channel_{symbol.lower()}_strategy.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 戦略チャートを保存しました: {output_path}")
    
    plt.show()


def main():
    """メイン実行関数"""
    print("🌌" * 30)
    print("🌌 Cosmic Adaptive Channel (CAC) - 宇宙最強ブレイクアウト戦略デモ 🌌")
    print("🌌" * 30)
    
    # データ取得オプション
    use_real_data = input("\n実際の市場データを使用しますか? (y/n, デフォルト: n): ").lower().strip()
    
    if use_real_data == 'y':
        # 実際の市場データを取得
        try:
            print("\n📡 Binanceから実際の市場データを取得中...")
            fetcher = BinanceDataFetcher()
            symbol = input("シンボルを入力してください (デフォルト: BTCUSDT): ").strip() or "BTCUSDT"
            interval = input("時間軸を入力してください (デフォルト: 1h): ").strip() or "1h"
            days = int(input("過去何日分のデータを取得しますか? (デフォルト: 30): ").strip() or "30")
            
            data = fetcher.fetch_historical_data(symbol, interval, days)
            if data is None or len(data) == 0:
                raise Exception("データ取得に失敗しました")
            
            print(f"✅ {symbol} の {len(data)}本のローソク足データを取得しました")
            
        except Exception as e:
            print(f"❌ リアルデータ取得エラー: {e}")
            print("📊 サンプルデータを使用します...")
            data = create_sample_data(1000)
            symbol = "SAMPLE"
    else:
        # サンプルデータを使用
        print("\n📊 高品質サンプルデータを生成中...")
        data = create_sample_data(1000)
        symbol = "SAMPLE"
    
    print(f"📈 データ期間: {len(data)}期間")
    print(f"💰 価格範囲: {data['close'].min():.2f} - {data['close'].max():.2f}")
    
    # Cosmic Adaptive Channel 解析
    analysis_result = analyze_cosmic_adaptive_channel(data, symbol)
    
    # チャート作成
    print(f"\n🎨 宇宙最強チャートを作成中...")
    create_comprehensive_chart(data, analysis_result, symbol)
    
    # 戦略パフォーマンス解析
    print(f"\n📊 戦略パフォーマンスを解析中...")
    create_strategy_performance_chart(data, analysis_result, symbol)
    
    print(f"\n✅ Cosmic Adaptive Channel デモ完了!")
    print(f"🎯 トレンド強度に応じた動的チャネル幅調整")
    print(f"⚡ 超低遅延ブレイクアウト検出")
    print(f"🧠 神経適応学習による偽シグナル防御")
    print(f"🌌 量子統計フュージョンによる高精度予測")
    
    print(f"\n🌌 宇宙最強のトレンドフォロー戦略をお楽しみください! 🌌")


if __name__ == "__main__":
    main() 