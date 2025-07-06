#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🌌 Cosmic Adaptive Channel Strategy Demo - 宇宙最強戦略デモ 🌌

実際の相場データを使用したCosmic Adaptive Channel Strategyの
包括的テストとデモンストレーション

機能:
- リアルタイム市場データでの戦略テスト
- エントリー・エグジットシグナルの詳細解析
- 宇宙知能レポート出力
- バックテスト実行
- 戦略可視化
- Optuna最適化デモ
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

# データ読み込み関連
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource

# Cosmic Adaptive Channel Strategy
from strategies.implementations.cosmic_adaptive_channel import CosmicAdaptiveChannelStrategy


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


def test_cosmic_strategy_basic():
    """🌌 基本戦略テスト"""
    print("\n🚀 基本戦略テスト開始")
    print("=" * 50)
    
    # データ準備
    data = load_sample_data()
    
    # 戦略初期化（ショート対応超緩和設定）
    strategy = CosmicAdaptiveChannelStrategy(
        min_confidence=0.1,  # 信頼度を大幅緩和
        min_trend_strength=0.05,  # トレンド強度要求を大幅緩和
        min_quantum_coherence=0.1,  # 量子コヒーレンス要求を大幅緩和
        enable_cosmic_enhancement=False,  # 強化を無効にしてシグナルを増やす
        require_strong_signals=False  # 強いシグナル要求を無効
    )
    
    # エントリーシグナル生成
    print("⚡ 宇宙最強戦略シグナル計算中...")
    entry_signals = strategy.generate_entry(data)
    
    # 基本統計
    total_signals = np.sum(np.abs(entry_signals))
    long_signals = np.sum(entry_signals == 1)
    short_signals = np.sum(entry_signals == -1)
    
    print(f"📊 戦略シグナル統計:")
    print(f"  総エントリーシグナル数: {total_signals}")
    print(f"  ロングエントリー: {long_signals}")
    print(f"  ショートエントリー: {short_signals}")
    print(f"  シグナル密度: {total_signals/len(data)*100:.2f}%")
    
    # ショートシグナル詳細解析
    if short_signals == 0:
        print(f"⚠️  ショートシグナルが生成されていません。詳細解析中...")
        
        # Cosmic結果の取得
        cosmic_indicators = strategy.get_cosmic_indicators(data)
        breakout_confidence = cosmic_indicators['breakout_confidence']
        trend_strength = cosmic_indicators['trend_strength']
        quantum_coherence = cosmic_indicators['quantum_coherence']
        
        # トレンド強度の分析
        negative_trend_count = np.sum(trend_strength < 0)
        print(f"  下降トレンド期間: {negative_trend_count} / {len(trend_strength)} ({negative_trend_count/len(trend_strength)*100:.1f}%)")
        
        # 信頼度の分析
        high_confidence_count = np.sum(breakout_confidence >= 0.3)
        print(f"  高信頼度期間: {high_confidence_count} / {len(breakout_confidence)} ({high_confidence_count/len(breakout_confidence)*100:.1f}%)")
        
        # 量子コヒーレンスの分析
        high_quantum_count = np.sum(quantum_coherence >= 0.3)
        print(f"  高量子コヒーレンス期間: {high_quantum_count} / {len(quantum_coherence)} ({high_quantum_count/len(quantum_coherence)*100:.1f}%)")
    
    # エグジットテスト（修正版）
    exit_count = 0
    current_position = 0
    for i in range(len(data)):
        current_signal = entry_signals[i]
        
        # ポジション追跡
        if current_signal == 1:
            current_position = 1
        elif current_signal == -1:
            current_position = -1
        
        # エグジットテスト
        if current_position != 0:
            exit_signal = strategy.generate_exit(data, current_position, i)
            if exit_signal:
                exit_count += 1
                current_position = 0  # エグジット後はポジションなし
    
    print(f"  エグジット機会: {exit_count}")
    
    return strategy, entry_signals, data


def test_cosmic_strategy_configurations():
    """🌌 複数設定戦略テスト"""
    print("\n🌟 複数設定戦略テスト開始")
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
        
        strategy = CosmicAdaptiveChannelStrategy(**config['params'])
        entry_signals = strategy.generate_entry(data)
        
        # 統計計算
        total_signals = np.sum(np.abs(entry_signals))
        long_signals = np.sum(entry_signals == 1)
        short_signals = np.sum(entry_signals == -1)
        signal_density = total_signals / len(data) * 100
        
        # 宇宙知能レポート取得
        cosmic_report = strategy.get_cosmic_intelligence_report(data)
        
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


def run_cosmic_backtest(strategy: CosmicAdaptiveChannelStrategy, 
                       data: pd.DataFrame, 
                       initial_capital: float = 10000.0) -> Dict[str, Any]:
    """🌌 宇宙戦略バックテスト実行"""
    print("\n💹 宇宙戦略バックテスト開始")
    print("=" * 50)
    
    # バックテスト設定
    capital = initial_capital
    position = 0  # 0=なし, 1=ロング, -1=ショート
    entry_price = 0
    trades = []
    returns = []
    equity_curve = [capital]
    
    # エントリーシグナル取得
    entry_signals = strategy.generate_entry(data)
    prices = data['close'].values
    
    for i in range(len(data)):
        current_price = prices[i]
        current_signal = entry_signals[i]
        
        # シグナルベースエントリー・エグジット処理（修正版）
        if current_signal == 1:  # ロングシグナル
            if position == -1:  # 既存ショートクローズ
                ret = (entry_price - current_price) / entry_price
                capital *= (1 + ret)
                returns.append(ret)
                trades.append({
                    'type': 'close_short',
                    'price': current_price,
                    'return': ret,
                    'index': i,
                    'capital': capital
                })
            
            if position != 1:  # ロングエントリー（新規またはショートからの切り替え）
                position = 1
                entry_price = current_price
                trades.append({
                    'type': 'open_long',
                    'price': current_price,
                    'index': i,
                    'capital': capital
                })
        
        elif current_signal == -1:  # ショートシグナル
            if position == 1:  # 既存ロングクローズ
                ret = (current_price - entry_price) / entry_price
                capital *= (1 + ret)
                returns.append(ret)
                trades.append({
                    'type': 'close_long',
                    'price': current_price,
                    'return': ret,
                    'index': i,
                    'capital': capital
                })
            
            if position != -1:  # ショートエントリー（新規またはロングからの切り替え）
                position = -1
                entry_price = current_price
                trades.append({
                    'type': 'open_short',
                    'price': current_price,
                    'index': i,
                    'capital': capital
                })
        
        equity_curve.append(capital)
    
    # 最終ポジションクローズ
    if position != 0:
        final_price = prices[-1]
        if position == 1:
            ret = (final_price - entry_price) / entry_price
        else:
            ret = (entry_price - final_price) / entry_price
        capital *= (1 + ret)
        returns.append(ret)
        trades.append({
            'type': 'final_close',
            'price': final_price,
            'return': ret,
            'index': len(prices)-1,
            'capital': capital
        })
    
    # バックテスト統計計算
    if returns:
        total_return = (capital - initial_capital) / initial_capital
        win_trades = [r for r in returns if r > 0]
        lose_trades = [r for r in returns if r <= 0]
        
        backtest_stats = {
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_return': total_return,
            'total_trades': len(returns),
            'win_trades': len(win_trades),
            'lose_trades': len(lose_trades),
            'win_rate': len(win_trades) / len(returns) if returns else 0,
            'average_return': np.mean(returns),
            'average_win': np.mean(win_trades) if win_trades else 0,
            'average_loss': np.mean(lose_trades) if lose_trades else 0,
            'max_return': max(returns) if returns else 0,
            'min_return': min(returns) if returns else 0,
            'volatility': np.std(returns) if returns else 0,
            'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
            'max_drawdown': calculate_max_drawdown(equity_curve),
            'trades': trades,
            'returns': returns,
            'equity_curve': equity_curve
        }
        
        print(f"🎯 宇宙戦略バックテスト成績:")
        print(f"  初期資本: ${backtest_stats['initial_capital']:,.2f}")
        print(f"  最終資本: ${backtest_stats['final_capital']:,.2f}")
        print(f"  総リターン: {backtest_stats['total_return']:+.2%}")
        print(f"  総取引数: {backtest_stats['total_trades']}")
        print(f"  勝率: {backtest_stats['win_rate']:.1%}")
        print(f"  平均リターン: {backtest_stats['average_return']:+.2%}")
        print(f"  平均利益: {backtest_stats['average_win']:+.2%}")
        print(f"  平均損失: {backtest_stats['average_loss']:+.2%}")
        print(f"  最大ドローダウン: {backtest_stats['max_drawdown']:.2%}")
        print(f"  シャープレシオ: {backtest_stats['sharpe_ratio']:.2f}")
        
        return backtest_stats
    else:
        print("❌ 取引が発生しませんでした")
        return {}


def calculate_max_drawdown(equity_curve: List[float]) -> float:
    """最大ドローダウンを計算"""
    peak = equity_curve[0]
    max_drawdown = 0
    
    for value in equity_curve:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    return max_drawdown * 100


def visualize_cosmic_strategy(strategy: CosmicAdaptiveChannelStrategy, 
                            entry_signals: np.ndarray, 
                            data: pd.DataFrame, 
                            show_last_periods: int = 200):
    """🌌 宇宙戦略可視化"""
    print(f"\n🎨 宇宙戦略可視化（最新{show_last_periods}期間）")
    print("=" * 50)
    
    # 最新データに絞り込み
    data_subset = data.tail(show_last_periods).copy()
    signals_subset = entry_signals[-show_last_periods:]
    
    # Cosmic指標取得
    cosmic_indicators = strategy.get_cosmic_indicators(data)
    cosmic_bands = strategy.get_cosmic_band_values(data)
    
    # プロット作成
    fig, axes = plt.subplots(5, 1, figsize=(15, 16))
    fig.suptitle('🌌 Cosmic Adaptive Channel Strategy - 宇宙最強戦略', fontsize=16)
    
    # 1. 価格とチャネル・シグナル
    ax1 = axes[0]
    ax1.plot(data_subset.index, data_subset['close'], label='価格', color='blue', alpha=0.7)
    
    # Cosmicチャネル
    if len(cosmic_bands['center_line']) > 0:
        center_subset = cosmic_bands['center_line'][-show_last_periods:]
        upper_subset = cosmic_bands['upper_channel'][-show_last_periods:]
        lower_subset = cosmic_bands['lower_channel'][-show_last_periods:]
        
        ax1.plot(data_subset.index, center_subset, label='中心線', color='orange', alpha=0.8)
        ax1.plot(data_subset.index, upper_subset, label='上限チャネル', color='red', alpha=0.6)
        ax1.plot(data_subset.index, lower_subset, label='下限チャネル', color='green', alpha=0.6)
        ax1.fill_between(data_subset.index, upper_subset, lower_subset, alpha=0.1, color='gray')
    
    # エントリーシグナルをプロット
    long_indices = data_subset.index[signals_subset == 1]
    short_indices = data_subset.index[signals_subset == -1]
    
    if len(long_indices) > 0:
        ax1.scatter(long_indices, data_subset.loc[long_indices, 'close'], 
                   color='green', marker='^', s=100, zorder=5, label=f'ロング ({len(long_indices)})')
    
    if len(short_indices) > 0:
        ax1.scatter(short_indices, data_subset.loc[short_indices, 'close'], 
                   color='red', marker='v', s=100, zorder=5, label=f'ショート ({len(short_indices)})')
    
    ax1.set_title('価格・チャネル・エントリーシグナル')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. ブレイクアウト信頼度
    ax2 = axes[1]
    if len(cosmic_indicators['breakout_confidence']) > 0:
        confidence_subset = cosmic_indicators['breakout_confidence'][-show_last_periods:]
        ax2.plot(data_subset.index, confidence_subset, label='ブレイクアウト信頼度', color='purple')
        ax2.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='しきい値')
    ax2.set_title('ブレイクアウト信頼度')
    ax2.set_ylabel('信頼度')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. トレンド強度
    ax3 = axes[2]
    if len(cosmic_indicators['trend_strength']) > 0:
        trend_subset = cosmic_indicators['trend_strength'][-show_last_periods:]
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
    if len(cosmic_indicators['quantum_coherence']) > 0:
        quantum_subset = cosmic_indicators['quantum_coherence'][-show_last_periods:]
        ax4.plot(data_subset.index, quantum_subset, label='量子コヒーレンス', color='cyan')
        ax4.axhline(y=0.4, color='black', linestyle='--', alpha=0.5, label='しきい値')
    ax4.set_title('量子コヒーレンス指数')
    ax4.set_ylabel('コヒーレンス')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. チャネル効率度
    ax5 = axes[4]
    if len(cosmic_indicators['channel_efficiency']) > 0:
        efficiency_subset = cosmic_indicators['channel_efficiency'][-show_last_periods:]
        ax5.plot(data_subset.index, efficiency_subset, label='チャネル効率度', color='magenta')
        ax5.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='しきい値')
    ax5.set_title('チャネル効率度')
    ax5.set_ylabel('効率度')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cosmic_adaptive_channel_strategy.png', dpi=300, bbox_inches='tight')
    print("📊 戦略チャートを保存しました: cosmic_adaptive_channel_strategy.png")
    plt.show()


def main():
    """メイン関数"""
    print("🌌" * 30)
    print("🌌 COSMIC ADAPTIVE CHANNEL STRATEGY DEMO 🌌")
    print("🌌" * 30)
    
    try:
        # 基本戦略テスト
        strategy, entry_signals, data = test_cosmic_strategy_basic()
        
        # 複数設定テスト
        test_cosmic_strategy_configurations()
        
        # バックテスト実行
        backtest_results = run_cosmic_backtest(strategy, data)
        
        # 戦略可視化
        visualize_cosmic_strategy(strategy, entry_signals, data)
        
        # 戦略サマリー表示
        strategy_summary = strategy.get_strategy_summary(data)
        print(f"\n📋 宇宙戦略サマリー:")
        print(f"  戦略名: {strategy_summary['strategy_name']}")
        print(f"  戦略バージョン: {strategy_summary['strategy_version']}")
        print(f"  戦略タイプ: {strategy_summary['strategy_type']}")
        
        if 'signal_statistics' in strategy_summary:
            stats = strategy_summary['signal_statistics']
            print(f"  総シグナル数: {stats['total_signals']}")
            print(f"  ロング/ショート比: {stats['long_short_ratio']:.2f}")
        
        cosmic_intel = strategy_summary.get('cosmic_intelligence', {})
        print(f"  宇宙知能スコア: {cosmic_intel.get('cosmic_intelligence_score', 0):.3f}")
        
        # 最終まとめ
        print(f"\n🎯 宇宙最強戦略デモ完了!")
        print(f"✅ 全ての宇宙機能が正常に動作しました")
        
        if backtest_results:
            print(f"💫 推奨戦略リターン: {backtest_results['total_return']:+.2%}")
            print(f"🏆 推奨戦略勝率: {backtest_results['win_rate']:.1%}")
        
        print(f"🌌 宇宙の力があなたのトレードと共にあります! 🌌")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()