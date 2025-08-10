#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
信頼度ベース・コンセンサス戦略のテストとデモンストレーション

階層的適応型コンセンサス法の動作確認とパフォーマンス評価を行います。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 戦略のインポート
from strategies.implementations.confidence_consensus_strategy import (
    ConfidenceConsensusStrategy,
    create_confidence_consensus_strategy
)


def generate_realistic_market_data(length: int = 500, base_price: float = 100.0) -> pd.DataFrame:
    """
    リアルな市場データを生成する
    
    Args:
        length: データ数
        base_price: 基準価格
        
    Returns:
        OHLCV DataFrame
    """
    np.random.seed(42)
    
    # 複数の市場フェーズを含むデータ
    prices = [base_price]
    
    for i in range(1, length):
        # 市場フェーズによる変動パターン
        if i < length // 5:  # フェーズ1: 強い上昇トレンド
            trend = 0.004
            vol = 0.008
        elif i < 2 * length // 5:  # フェーズ2: 高ボラティリティレンジ
            trend = 0.0
            vol = 0.020
        elif i < 3 * length // 5:  # フェーズ3: 穏やかな上昇
            trend = 0.001
            vol = 0.006
        elif i < 4 * length // 5:  # フェーズ4: 下降トレンド
            trend = -0.003
            vol = 0.012
        else:  # フェーズ5: 低ボラティリティレンジ
            trend = 0.0005
            vol = 0.004
        
        # ランダムウォーク with ドリフト
        change = trend + np.random.normal(0, vol)
        
        # たまに大きなショックを入れる
        if np.random.random() < 0.005:  # 0.5%の確率
            shock = np.random.normal(0, 0.05)  # 5%の大きなショック
            change += shock
        
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 0.1))  # 価格が負にならないよう制限
    
    # OHLC データの生成
    ohlcv_data = []
    for i, close in enumerate(prices):
        # 日中レンジの計算
        daily_vol = abs(np.random.normal(0, close * 0.008))
        
        # Open価格（前日終値からのギャップ）
        if i == 0:
            open_price = close
        else:
            gap = np.random.normal(0, close * 0.003)
            open_price = max(prices[i-1] + gap, 0.1)
        
        # High/Low価格
        intraday_range = daily_vol * np.random.uniform(0.5, 1.5)
        high = max(open_price, close) + intraday_range * np.random.uniform(0.2, 0.8)
        low = min(open_price, close) - intraday_range * np.random.uniform(0.2, 0.8)
        
        # 論理的整合性の確保
        high = max(high, open_price, close, low + 0.01)
        low = min(low, open_price, close, high - 0.01)
        low = max(low, 0.1)  # 最低価格制限
        
        # Volume（価格変動に連動）
        price_change_pct = abs(close - prices[i-1]) / prices[i-1] if i > 0 else 0
        base_volume = np.random.uniform(5000, 15000)
        volume_multiplier = 1 + price_change_pct * 5  # 価格変動が大きいほど出来高も大きく
        volume = base_volume * volume_multiplier
        
        ohlcv_data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(ohlcv_data)


def analyze_strategy_performance(strategy, data: pd.DataFrame) -> dict:
    """
    戦略のパフォーマンスを分析する
    
    Args:
        strategy: 戦略インスタンス
        data: 価格データ
        
    Returns:
        パフォーマンス統計辞書
    """
    try:
        # シグナル生成
        signals = strategy.generate_signals(data)
        
        entry_signals = signals['entry']
        confidence = signals['confidence']
        
        # 基本統計
        long_entries = np.sum(entry_signals == 1)
        short_entries = np.sum(entry_signals == -1)
        total_entries = long_entries + short_entries
        no_signal_periods = np.sum(entry_signals == 0)
        
        # 信頼度統計
        valid_confidence = confidence[~np.isnan(confidence)]
        if len(valid_confidence) > 0:
            conf_mean = np.mean(valid_confidence)
            conf_std = np.std(valid_confidence)
            conf_min = np.min(valid_confidence)
            conf_max = np.max(valid_confidence)
            
            # 信頼度分布
            high_conf_periods = np.sum(np.abs(valid_confidence) > 0.5)
            very_high_conf_periods = np.sum(np.abs(valid_confidence) > 0.8)
        else:
            conf_mean = conf_std = conf_min = conf_max = np.nan
            high_conf_periods = very_high_conf_periods = 0
        
        # エントリー効率（シグナルが出た時の信頼度）
        entry_mask = entry_signals != 0
        if np.any(entry_mask):
            entry_confidence = confidence[entry_mask]
            avg_entry_confidence = np.nanmean(np.abs(entry_confidence))
        else:
            avg_entry_confidence = np.nan
        
        # レイヤー別統計
        filter_consensus = signals.get('filter_consensus', np.full(len(data), np.nan))
        directional_strength = signals.get('directional_strength', np.full(len(data), np.nan))
        momentum_factor = signals.get('momentum_factor', np.full(len(data), np.nan))
        
        layer_stats = {
            'filter_consensus': {
                'mean': np.nanmean(filter_consensus),
                'std': np.nanstd(filter_consensus),
                'positive_ratio': np.sum(filter_consensus > 0) / len(filter_consensus)
            },
            'directional_strength': {
                'mean': np.nanmean(directional_strength),
                'std': np.nanstd(directional_strength),
                'positive_ratio': np.sum(directional_strength > 0) / len(directional_strength)
            },
            'momentum_factor': {
                'mean': np.nanmean(momentum_factor),
                'std': np.nanstd(momentum_factor),
                'positive_ratio': np.sum(momentum_factor > 0) / len(momentum_factor)
            }
        }
        
        return {
            'data_length': len(data),
            'long_entries': long_entries,
            'short_entries': short_entries,
            'total_entries': total_entries,
            'entry_rate': total_entries / len(data),
            'no_signal_periods': no_signal_periods,
            'confidence_stats': {
                'mean': conf_mean,
                'std': conf_std,
                'min': conf_min,
                'max': conf_max,
                'high_confidence_periods': high_conf_periods,
                'very_high_confidence_periods': very_high_conf_periods
            },
            'avg_entry_confidence': avg_entry_confidence,
            'layer_stats': layer_stats,
            'strategy_info': strategy.get_strategy_info()
        }
        
    except Exception as e:
        print(f"パフォーマンス分析中にエラー: {e}")
        return {'error': str(e)}


def plot_strategy_results(data: pd.DataFrame, signals: dict, title: str = "戦略結果"):
    """
    戦略結果をプロットする
    
    Args:
        data: 価格データ
        signals: シグナル辞書
        title: プロットタイトル
    """
    try:
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # 価格とエントリーポイント
        ax1 = axes[0]
        ax1.plot(data['close'], label='Close Price', color='black', linewidth=1)
        
        entry_signals = signals['entry']
        long_entries = np.where(entry_signals == 1)[0]
        short_entries = np.where(entry_signals == -1)[0]
        
        if len(long_entries) > 0:
            ax1.scatter(long_entries, data['close'].iloc[long_entries], 
                       color='green', marker='^', s=50, label='ロングエントリー', zorder=5)
        
        if len(short_entries) > 0:
            ax1.scatter(short_entries, data['close'].iloc[short_entries], 
                       color='red', marker='v', s=50, label='ショートエントリー', zorder=5)
        
        ax1.set_ylabel('価格')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title('価格とエントリーポイント')
        
        # 信頼度
        ax2 = axes[1]
        confidence = signals['confidence']
        ax2.plot(confidence, label='信頼度', color='blue', linewidth=1.5)
        ax2.axhline(y=0.6, color='green', linestyle='--', alpha=0.7, label='ロング閾値')
        ax2.axhline(y=-0.6, color='red', linestyle='--', alpha=0.7, label='ショート閾値')
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax2.set_ylabel('信頼度')
        ax2.set_ylim(-1.1, 1.1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_title('信頼度推移')
        
        # レイヤー別信頼度
        ax3 = axes[2]
        filter_consensus = signals.get('filter_consensus', np.full(len(data), np.nan))
        directional_strength = signals.get('directional_strength', np.full(len(data), np.nan))
        momentum_factor = signals.get('momentum_factor', np.full(len(data), np.nan))
        
        ax3.plot(filter_consensus, label='フィルター合意度', color='orange', alpha=0.8)
        ax3.plot(directional_strength, label='方向性強度', color='purple', alpha=0.8)
        ax3.plot(momentum_factor, label='モメンタム係数', color='brown', alpha=0.8)
        
        ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax3.set_ylabel('レイヤー値')
        ax3.set_ylim(-1.1, 1.1)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_title('各レイヤーの推移')
        
        # ボラティリティ補正
        ax4 = axes[3]
        volatility_correction = signals.get('volatility_correction', np.ones(len(data)))
        ax4.plot(volatility_correction, label='ボラティリティ補正', color='teal', linewidth=1.5)
        ax4.axhline(y=1.0, color='gray', linestyle='-', alpha=0.5)
        ax4.set_ylabel('補正倍率')
        ax4.set_xlabel('時間')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_title('ボラティリティ補正要素')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"プロット中にエラー: {e}")


def main():
    """メイン実行関数"""
    print("=== 信頼度ベース・コンセンサス戦略 - 総合テスト ===\n")
    
    # 1. テストデータ生成
    print("1. リアルな市場データを生成中...")
    data = generate_realistic_market_data(length=400, base_price=100.0)
    print(f"   データ生成完了: {len(data)}期間")
    print(f"   価格範囲: {data['close'].min():.2f} - {data['close'].max():.2f}")
    print(f"   価格変動率: {((data['close'].iloc[-1] / data['close'].iloc[0]) - 1) * 100:.2f}%")
    
    # 2. 戦略インスタンス作成
    print("\n2. 戦略インスタンス作成中...")
    
    # デフォルト設定
    strategy_default = create_confidence_consensus_strategy(
        long_threshold=0.6,
        short_threshold=-0.6,
        donchian_periods=(60, 120, 240),
        laguerre_gamma=0.8
    )
    
    # 厳しい設定（高い閾値）
    strategy_strict = create_confidence_consensus_strategy(
        long_threshold=0.8,
        short_threshold=-0.8,
        donchian_periods=(60, 120, 240),
        laguerre_gamma=0.8
    )
    
    # 緩い設定（低い閾値）
    strategy_loose = create_confidence_consensus_strategy(
        long_threshold=0.4,
        short_threshold=-0.4,
        donchian_periods=(60, 120, 240),
        laguerre_gamma=0.8
    )
    
    strategies = {
        'デフォルト設定 (±0.6)': strategy_default,
        '厳しい設定 (±0.8)': strategy_strict,
        '緩い設定 (±0.4)': strategy_loose
    }
    
    print(f"   {len(strategies)}つの戦略設定を作成完了")
    
    # 3. 各戦略のテスト
    print("\n3. 戦略パフォーマンス分析中...")
    
    results = {}
    for name, strategy in strategies.items():
        print(f"\n--- {name} ---")
        try:
            # パフォーマンス分析
            perf = analyze_strategy_performance(strategy, data)
            results[name] = perf
            
            if 'error' not in perf:
                print(f"エントリー回数: ロング {perf['long_entries']}, ショート {perf['short_entries']}")
                print(f"エントリー率: {perf['entry_rate']:.3f}")
                print(f"平均エントリー信頼度: {perf['avg_entry_confidence']:.3f}")
                print(f"信頼度統計: 平均 {perf['confidence_stats']['mean']:.3f}, 標準偏差 {perf['confidence_stats']['std']:.3f}")
                print(f"高信頼度期間: {perf['confidence_stats']['high_confidence_periods']} / {perf['data_length']}")
                
                # レイヤー統計
                layer_stats = perf['layer_stats']
                print("\nレイヤー統計:")
                for layer_name, stats in layer_stats.items():
                    print(f"  {layer_name}: 平均 {stats['mean']:.3f}, ポジティブ比率 {stats['positive_ratio']:.3f}")
            else:
                print(f"エラー: {perf['error']}")
                
        except Exception as e:
            print(f"テスト中にエラー: {e}")
            results[name] = {'error': str(e)}
    
    # 4. 詳細分析（デフォルト設定）
    if 'デフォルト設定 (±0.6)' in results and 'error' not in results['デフォルト設定 (±0.6)']:
        print("\n4. 詳細分析（デフォルト設定）...")
        
        try:
            signals = strategy_default.generate_signals(data)
            confidence_result = strategy_default.calculate_confidence(data)
            
            # 信頼度分布の分析
            confidence = signals['confidence']
            valid_conf = confidence[~np.isnan(confidence)]
            
            if len(valid_conf) > 0:
                print(f"\n信頼度分布分析:")
                print(f"  範囲: {np.min(valid_conf):.3f} to {np.max(valid_conf):.3f}")
                print(f"  四分位: Q1={np.percentile(valid_conf, 25):.3f}, "
                      f"Q2={np.percentile(valid_conf, 50):.3f}, Q3={np.percentile(valid_conf, 75):.3f}")
                
                # 閾値到達頻度
                above_long_threshold = np.sum(valid_conf >= 0.6)
                below_short_threshold = np.sum(valid_conf <= -0.6)
                total_periods = len(valid_conf)
                
                print(f"  閾値到達頻度:")
                print(f"    ロング閾値以上 (≥0.6): {above_long_threshold} / {total_periods} ({above_long_threshold/total_periods*100:.1f}%)")
                print(f"    ショート閾値以下 (≤-0.6): {below_short_threshold} / {total_periods} ({below_short_threshold/total_periods*100:.1f}%)")
            
            # 可視化（簡易版）
            print("\n5. 結果可視化...")
            plot_strategy_results(data, signals, "信頼度ベース・コンセンサス戦略 - デフォルト設定")
            
        except Exception as e:
            print(f"詳細分析中にエラー: {e}")
    
    # 6. 比較サマリー
    print("\n6. 戦略比較サマリー")
    print("=" * 60)
    print(f"{'設定':<15} {'エントリー率':<10} {'平均信頼度':<12} {'高信頼度期間':<12}")
    print("-" * 60)
    
    for name, perf in results.items():
        if 'error' not in perf:
            entry_rate = f"{perf['entry_rate']:.3f}"
            avg_conf = f"{perf['avg_entry_confidence']:.3f}"
            high_conf = f"{perf['confidence_stats']['high_confidence_periods']}"
            print(f"{name:<15} {entry_rate:<10} {avg_conf:<12} {high_conf:<12}")
        else:
            print(f"{name:<15} {'ERROR':<10} {'ERROR':<12} {'ERROR':<12}")
    
    print("\n=== テスト完了 ===")
    
    # 戦略情報の表示
    print("\n戦略設計の特徴:")
    print("• フィルター合意度レイヤー (40%): 3つのハイパーインジケーターによる市場環境判定（全て-1→0変換）")
    print("• 方向性強度レイヤー (30%): 3期間ドンチャンFRAMAによる方向性確認")
    print("• モメンタム係数レイヤー (20%): ラゲールRSIによるモメンタム評価")
    print("• ボラティリティ補正レイヤー (10%): XATRによる市場環境補正")
    print("• 方向性が負の場合の符号逆転による適応的調整")
    print("• 高い信頼度閾値による質の高いエントリー選別")
    print("• フィルターシグナルの-1を0として扱い、方向性を示さない状態を正確に反映")
    print("• 信頼度ベースの論理的エグジット: ロング≤0、ショート≥0")


if __name__ == "__main__":
    main()