#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🌌 Cosmic Adaptive Channel Signal Demo - 宇宙最強シグナルデモ 🌌

実際の相場データを使用したCosmic Adaptive Channel Entryシグナルの
包括的テストとデモンストレーション

機能:
- リアルタイム市場データでのシグナル生成
- エントリー・決済シグナルの詳細解析
- 宇宙知能レポート出力
- パフォーマンス統計表示
- シグナル可視化
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

# データ読み込み関連
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource

# Cosmic Adaptive Channel Signal
from signals.implementations.cosmic_adaptive_channel import CosmicAdaptiveChannelEntrySignal


def load_sample_data() -> pd.DataFrame:
    """サンプルデータを読み込む"""
    try:
        # config.yamlから実際のデータを読み込み
        import yaml
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        binance_config = config.get('binance_data', {})
        data_dir = binance_config.get('data_dir', 'data/binance')
        binance_data_source = BinanceDataSource(data_dir)
        
        dummy_csv_source = CSVDataSource("dummy")
        data_loader = DataLoader(
            data_source=dummy_csv_source,
            binance_data_source=binance_data_source
        )
        data_processor = DataProcessor()
        
        print("📡 実相場データを読み込み中...")
        raw_data = data_loader.load_data_from_config(config)
        processed_data = {
            symbol: data_processor.process(df)
            for symbol, df in raw_data.items()
        }
        
        first_symbol = next(iter(processed_data))
        data = processed_data[first_symbol]
        
        print(f"✅ データ読み込み完了: {first_symbol}")
        print(f"📊 期間: {data.index.min()} → {data.index.max()}")
        print(f"📈 データ数: {len(data)}")
        
        return data
        
    except Exception as e:
        print(f"❌ データ読み込みエラー: {e}")
        # フォールバック: ダミーデータ生成
        print("🔄 ダミーデータを生成します...")
        return generate_dummy_data()


def generate_dummy_data(length: int = 1000) -> pd.DataFrame:
    """ダミーデータを生成する"""
    np.random.seed(42)
    
    # トレンドとボラティリティを持つ価格データ生成
    dates = pd.date_range('2023-01-01', periods=length, freq='4H')
    
    # 基本価格トレンド
    trend = np.cumsum(np.random.randn(length) * 0.001) + 100
    
    # ボラティリティ
    volatility = 0.02 + 0.01 * np.sin(np.arange(length) * 0.01)
    
    # OHLC生成
    close = trend + np.random.randn(length) * volatility * trend
    high = close + np.abs(np.random.randn(length)) * volatility * trend * 0.5
    low = close - np.abs(np.random.randn(length)) * volatility * trend * 0.5
    open_price = np.roll(close, 1)
    open_price[0] = close[0]
    
    # 出来高生成
    volume = np.random.exponential(1000, length)
    
    data = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    return data


def test_cosmic_signal_basic():
    """🌌 基本シグナルテスト"""
    print("\n🚀 基本シグナルテスト開始")
    print("=" * 50)
    
    # データ準備
    data = load_sample_data()
    
    # シグナル初期化（標準設定）
    signal = CosmicAdaptiveChannelEntrySignal(
        min_confidence=0.5,
        min_trend_strength=0.3,
        min_quantum_coherence=0.4,
        enable_cosmic_enhancement=True
    )
    
    # シグナル生成
    print("⚡ 宇宙最強シグナル計算中...")
    entry_signals = signal.generate(data)
    
    # 基本統計
    total_signals = np.sum(np.abs(entry_signals))
    long_signals = np.sum(entry_signals == 1)
    short_signals = np.sum(entry_signals == -1)
    
    print(f"📊 シグナル統計:")
    print(f"  総シグナル数: {total_signals}")
    print(f"  ロングシグナル: {long_signals}")
    print(f"  ショートシグナル: {short_signals}")
    print(f"  シグナル密度: {total_signals/len(data)*100:.2f}%")
    
    # 決済シグナルも取得
    exit_signals = signal.get_exit_signals()
    exit_count = np.sum(np.abs(exit_signals))
    print(f"  決済シグナル数: {exit_count}")
    
    return signal, entry_signals, exit_signals, data


def test_cosmic_signal_advanced():
    """🌌 高度シグナルテスト"""
    print("\n🌟 高度シグナルテスト開始")
    print("=" * 50)
    
    # データ準備
    data = load_sample_data()
    
    # 複数の設定でテスト
    configurations = [
        {
            'name': '🏆 宇宙最強設定',
            'params': {
                'min_confidence': 0.7,
                'min_trend_strength': 0.5,
                'min_quantum_coherence': 0.6,
                'enable_cosmic_enhancement': True,
                'require_strong_signals': True
            }
        },
        {
            'name': '⚡ バランス設定',
            'params': {
                'min_confidence': 0.5,
                'min_trend_strength': 0.3,
                'min_quantum_coherence': 0.4,
                'enable_cosmic_enhancement': True,
                'require_strong_signals': False
            }
        },
        {
            'name': '🚀 高感度設定',
            'params': {
                'min_confidence': 0.3,
                'min_trend_strength': 0.2,
                'min_quantum_coherence': 0.3,
                'enable_cosmic_enhancement': False,
                'require_strong_signals': False
            }
        }
    ]
    
    results = {}
    
    for config in configurations:
        print(f"\n🎯 {config['name']}でテスト中...")
        
        signal = CosmicAdaptiveChannelEntrySignal(**config['params'])
        entry_signals = signal.generate(data)
        
        # 統計計算
        total_signals = np.sum(np.abs(entry_signals))
        long_signals = np.sum(entry_signals == 1)
        short_signals = np.sum(entry_signals == -1)
        signal_density = total_signals / len(data) * 100
        
        # 宇宙知能レポート取得
        cosmic_report = signal.get_cosmic_intelligence_report()
        
        results[config['name']] = {
            'total_signals': total_signals,
            'long_signals': long_signals,
            'short_signals': short_signals,
            'signal_density': signal_density,
            'cosmic_intelligence_score': cosmic_report['cosmic_intelligence_score'],
            'false_signal_rate': cosmic_report['false_signal_rate']
        }
        
        print(f"  シグナル数: {total_signals} (L:{long_signals}, S:{short_signals})")
        print(f"  密度: {signal_density:.2f}%")
        print(f"  宇宙知能: {cosmic_report['cosmic_intelligence_score']:.3f}")
        print(f"  偽シグナル率: {cosmic_report['false_signal_rate']:.1%}")
    
    # 比較テーブル表示
    print(f"\n📊 設定比較テーブル:")
    print(f"{'設定':<15} {'シグナル数':<10} {'密度':<8} {'宇宙知能':<10} {'偽シグナル率':<12}")
    print("-" * 70)
    
    for name, stats in results.items():
        print(f"{name:<15} {stats['total_signals']:<10} {stats['signal_density']:<8.2f}% "
              f"{stats['cosmic_intelligence_score']:<10.3f} {stats['false_signal_rate']:<12.1%}")
    
    return results


def analyze_cosmic_components(signal: CosmicAdaptiveChannelEntrySignal, data: pd.DataFrame):
    """🌌 宇宙コンポーネント解析"""
    print("\n🔬 宇宙コンポーネント解析開始")
    print("=" * 50)
    
    # Cosmic Adaptive Channelの結果取得
    cosmic_result = signal.get_cosmic_result()
    
    if cosmic_result is None:
        print("❌ 宇宙結果が取得できませんでした")
        return
    
    # 各コンポーネントの統計
    components = {
        'ブレイクアウト信頼度': cosmic_result.breakout_confidence,
        'トレンド強度': cosmic_result.trend_strength,
        '量子コヒーレンス': cosmic_result.quantum_coherence,
        'チャネル効率度': cosmic_result.channel_efficiency,
        '神経適応スコア': cosmic_result.adaptation_score,
        'ボラティリティレジーム': cosmic_result.volatility_regime
    }
    
    print(f"🧬 宇宙コンポーネント統計:")
    for name, values in components.items():
        if len(values) > 0:
            mean_val = np.nanmean(values)
            std_val = np.nanstd(values)
            min_val = np.nanmin(values)
            max_val = np.nanmax(values)
            
            print(f"  {name}:")
            print(f"    平均: {mean_val:.3f} ± {std_val:.3f}")
            print(f"    範囲: {min_val:.3f} → {max_val:.3f}")
    
    # 現在状態の詳細
    current_state = signal.get_current_state()
    print(f"\n🌌 現在の宇宙状態:")
    cosmic_intel = current_state['cosmic_intelligence']
    
    print(f"  トレンドフェーズ: {cosmic_intel['current_trend_phase']}")
    print(f"  ボラティリティレジーム: {cosmic_intel['current_volatility_regime']}")
    print(f"  ブレイクアウト確率: {cosmic_intel['current_breakout_probability']:.3f}")
    print(f"  宇宙知能スコア: {cosmic_intel['cosmic_intelligence_score']:.3f}")
    print(f"  偽シグナル防御率: {(1-cosmic_intel['false_signal_rate'])*100:.1f}%")


def simulate_cosmic_strategy(signal: CosmicAdaptiveChannelEntrySignal, 
                           entry_signals: np.ndarray, 
                           exit_signals: np.ndarray, 
                           data: pd.DataFrame) -> Dict[str, Any]:
    """🌌 宇宙戦略シミュレーション"""
    print("\n💹 宇宙戦略シミュレーション開始")
    print("=" * 50)
    
    prices = data['close'].values
    position = 0  # 0=なし, 1=ロング, -1=ショート
    entry_price = 0
    trades = []
    returns = []
    
    for i in range(len(entry_signals)):
        # エントリーシグナル処理
        if entry_signals[i] == 1 and position != 1:  # ロングエントリー
            if position == -1:  # 既存ショートクローズ
                ret = (entry_price - prices[i]) / entry_price
                returns.append(ret)
                trades.append({'type': 'close_short', 'price': prices[i], 'return': ret, 'index': i})
            
            position = 1
            entry_price = prices[i]
            trades.append({'type': 'open_long', 'price': prices[i], 'index': i})
        
        elif entry_signals[i] == -1 and position != -1:  # ショートエントリー
            if position == 1:  # 既存ロングクローズ
                ret = (prices[i] - entry_price) / entry_price
                returns.append(ret)
                trades.append({'type': 'close_long', 'price': prices[i], 'return': ret, 'index': i})
            
            position = -1
            entry_price = prices[i]
            trades.append({'type': 'open_short', 'price': prices[i], 'index': i})
        
        # 決済シグナル処理
        if exit_signals[i] != 0 and position != 0:
            if position == 1 and exit_signals[i] == 1:  # ロング決済
                ret = (prices[i] - entry_price) / entry_price
                returns.append(ret)
                trades.append({'type': 'exit_long', 'price': prices[i], 'return': ret, 'index': i})
                position = 0
            
            elif position == -1 and exit_signals[i] == -1:  # ショート決済
                ret = (entry_price - prices[i]) / entry_price
                returns.append(ret)
                trades.append({'type': 'exit_short', 'price': prices[i], 'return': ret, 'index': i})
                position = 0
    
    # 最終ポジションクローズ
    if position != 0:
        final_price = prices[-1]
        if position == 1:
            ret = (final_price - entry_price) / entry_price
        else:
            ret = (entry_price - final_price) / entry_price
        returns.append(ret)
        trades.append({'type': 'final_close', 'price': final_price, 'return': ret, 'index': len(prices)-1})
    
    # 戦略統計計算
    if returns:
        total_return = np.prod([1 + r for r in returns]) - 1
        win_trades = [r for r in returns if r > 0]
        lose_trades = [r for r in returns if r <= 0]
        
        strategy_stats = {
            'total_trades': len(returns),
            'total_return': total_return,
            'win_rate': len(win_trades) / len(returns),
            'average_return': np.mean(returns),
            'average_win': np.mean(win_trades) if win_trades else 0,
            'average_loss': np.mean(lose_trades) if lose_trades else 0,
            'max_return': max(returns),
            'min_return': min(returns),
            'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
            'trades': trades,
            'returns': returns
        }
        
        print(f"🎯 宇宙戦略成績:")
        print(f"  総取引数: {strategy_stats['total_trades']}")
        print(f"  総リターン: {strategy_stats['total_return']:+.2%}")
        print(f"  勝率: {strategy_stats['win_rate']:.1%}")
        print(f"  平均リターン: {strategy_stats['average_return']:+.2%}")
        print(f"  平均利益: {strategy_stats['average_win']:+.2%}")
        print(f"  平均損失: {strategy_stats['average_loss']:+.2%}")
        print(f"  最大利益: {strategy_stats['max_return']:+.2%}")
        print(f"  最大損失: {strategy_stats['min_return']:+.2%}")
        print(f"  シャープレシオ: {strategy_stats['sharpe_ratio']:.2f}")
        
        return strategy_stats
    else:
        print("❌ 取引が発生しませんでした")
        return {}


def visualize_cosmic_signals(signal: CosmicAdaptiveChannelEntrySignal, 
                            entry_signals: np.ndarray, 
                            data: pd.DataFrame, 
                            show_last_periods: int = 200):
    """🌌 宇宙シグナル可視化"""
    print(f"\n🎨 宇宙シグナル可視化（最新{show_last_periods}期間）")
    print("=" * 50)
    
    # 最新データに絞り込み
    data_subset = data.tail(show_last_periods).copy()
    signals_subset = entry_signals[-show_last_periods:]
    
    # Cosmic結果取得
    cosmic_result = signal.get_cosmic_result()
    if cosmic_result is None:
        print("❌ 可視化データが取得できませんでした")
        return
    
    # プロット作成
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    fig.suptitle('🌌 Cosmic Adaptive Channel - 宇宙最強シグナル', fontsize=16)
    
    # 1. 価格とシグナル
    ax1 = axes[0]
    ax1.plot(data_subset.index, data_subset['close'], label='価格', color='blue', alpha=0.7)
    
    # エントリーシグナルをプロット
    long_indices = data_subset.index[signals_subset == 1]
    short_indices = data_subset.index[signals_subset == -1]
    
    if len(long_indices) > 0:
        ax1.scatter(long_indices, data_subset.loc[long_indices, 'close'], 
                   color='green', marker='^', s=100, zorder=5, label=f'ロング ({len(long_indices)})')
    
    if len(short_indices) > 0:
        ax1.scatter(short_indices, data_subset.loc[short_indices, 'close'], 
                   color='red', marker='v', s=100, zorder=5, label=f'ショート ({len(short_indices)})')
    
    ax1.set_title('価格とエントリーシグナル')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. ブレイクアウト信頼度
    ax2 = axes[1]
    confidence_subset = cosmic_result.breakout_confidence[-show_last_periods:]
    ax2.plot(data_subset.index, confidence_subset, label='ブレイクアウト信頼度', color='purple')
    ax2.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='しきい値')
    ax2.set_title('ブレイクアウト信頼度')
    ax2.set_ylabel('信頼度')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. トレンド強度
    ax3 = axes[2]
    trend_subset = cosmic_result.trend_strength[-show_last_periods:]
    ax3.plot(data_subset.index, trend_subset, label='トレンド強度', color='orange')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.axhline(y=0.3, color='green', linestyle='--', alpha=0.5, label='上昇しきい値')
    ax3.axhline(y=-0.3, color='red', linestyle='--', alpha=0.5, label='下降しきい値')
    ax3.set_title('統合トレンド強度')
    ax3.set_ylabel('強度')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 量子コヒーレンス
    ax4 = axes[3]
    quantum_subset = cosmic_result.quantum_coherence[-show_last_periods:]
    ax4.plot(data_subset.index, quantum_subset, label='量子コヒーレンス', color='cyan')
    ax4.axhline(y=0.4, color='black', linestyle='--', alpha=0.5, label='しきい値')
    ax4.set_title('量子コヒーレンス指数')
    ax4.set_ylabel('コヒーレンス')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cosmic_adaptive_channel_signals.png', dpi=300, bbox_inches='tight')
    print("📊 チャートを保存しました: cosmic_adaptive_channel_signals.png")
    plt.show()


def main():
    """メイン関数"""
    print("🌌" * 30)
    print("🌌 COSMIC ADAPTIVE CHANNEL SIGNAL DEMO 🌌")
    print("🌌" * 30)
    
    try:
        # 基本テスト
        signal, entry_signals, exit_signals, data = test_cosmic_signal_basic()
        
        # 高度テスト
        test_cosmic_signal_advanced()
        
        # コンポーネント解析
        analyze_cosmic_components(signal, data)
        
        # 戦略シミュレーション
        strategy_stats = simulate_cosmic_strategy(signal, entry_signals, exit_signals, data)
        
        # 可視化
        visualize_cosmic_signals(signal, entry_signals, data)
        
        # 最終まとめ
        print(f"\n🎯 宇宙最強シグナルデモ完了!")
        print(f"✅ 全ての宇宙機能が正常に動作しました")
        
        if strategy_stats:
            print(f"💫 推奨戦略リターン: {strategy_stats['total_return']:+.2%}")
            print(f"🏆 推奨戦略勝率: {strategy_stats['win_rate']:.1%}")
        
        print(f"🌌 宇宙の力があなたのトレードと共にあります! 🌌")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()