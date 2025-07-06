#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🌌 Cosmic Adaptive Channel Chart - 宇宙最強ブレイクアウトチャネル可視化 🌌

実際の相場データを使用したCosmic Adaptive Channelの包括的テストと可視化システム
- リアルタイム市場データ取得
- 8層ハイブリッドシステムの詳細解析
- 高度な統計解析とパフォーマンス評価
- インタラクティブチャート表示
"""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# データ取得のための依存関係
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource

# Cosmic Adaptive Channel インジケーター
from indicators.cosmic_adaptive_channel import CosmicAdaptiveChannel


class CosmicAdaptiveChannelChart:
    """
    🌌 Cosmic Adaptive Channel可視化クラス
    
    機能:
    - 実際の市場データ取得・処理
    - 8層ハイブリッドシステムの計算
    - 多パネル高度チャート表示
    - 詳細統計解析・パフォーマンス評価
    - ブレイクアウト戦略シミュレーション
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.cac_indicator = None
        self.cac_result = None
        self.fig = None
        self.axes = None
        self.symbol = None
        self.timeframe = None
    
    def load_data_from_config(self, config_path: str) -> pd.DataFrame:
        """
        設定ファイルからデータを読み込む
        
        Args:
            config_path: 設定ファイルのパス
            
        Returns:
            処理済みのデータフレーム
        """
        print("🌌 Cosmic Adaptive Channel - データ読み込み開始")
        print("=" * 60)
        
        # 設定ファイルの読み込み
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        # データの準備
        binance_config = config.get('binance_data', {})
        data_dir = binance_config.get('data_dir', 'data/binance')
        self.symbol = binance_config.get('symbol', 'BTC')
        self.timeframe = binance_config.get('timeframe', '4h')
        
        binance_data_source = BinanceDataSource(data_dir)
        
        # CSVデータソースはダミーとして渡す
        dummy_csv_source = CSVDataSource("dummy")
        data_loader = DataLoader(
            data_source=dummy_csv_source,
            binance_data_source=binance_data_source
        )
        data_processor = DataProcessor()
        
        # データの読み込みと処理
        print(f"📡 {self.symbol} ({self.timeframe}) データを読み込み中...")
        raw_data = data_loader.load_data_from_config(config)
        processed_data = {
            symbol: data_processor.process(df)
            for symbol, df in raw_data.items()
        }
        
        # 最初のシンボルのデータを取得
        first_symbol = next(iter(processed_data))
        self.data = processed_data[first_symbol]
        
        print(f"✅ データ読み込み完了: {first_symbol}")
        print(f"📊 期間: {self.data.index.min()} → {self.data.index.max()}")
        print(f"📈 データ数: {len(self.data)}")
        print(f"💰 価格範囲: {self.data['close'].min():.2f} - {self.data['close'].max():.2f}")
        
        return self.data

    def calculate_indicators(self,
                            atr_period: int = 21,
                            base_multiplier: float = 2.0,
                            quantum_window: int = 50,
                            neural_window: int = 100,
                            volatility_window: int = 30,
                            src_type: str = 'hlc3') -> None:
        """
        🌌 Cosmic Adaptive Channelを計算する
        
        Args:
            atr_period: ATR計算期間
            base_multiplier: 基本チャネル幅倍率
            quantum_window: 量子解析ウィンドウ
            neural_window: 神経学習ウィンドウ
            volatility_window: ボラティリティ解析ウィンドウ
            src_type: 価格ソースタイプ
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\n🚀 Cosmic Adaptive Channel計算開始...")
        print("=" * 60)
        
        # Cosmic Adaptive Channel初期化
        self.cac_indicator = CosmicAdaptiveChannel(
            atr_period=atr_period,
            base_multiplier=base_multiplier,
            quantum_window=quantum_window,
            neural_window=neural_window,
            volatility_window=volatility_window,
            src_type=src_type
        )
        
        # 計算実行
        print("⚡ 8層ハイブリッドシステム計算中...")
        self.cac_result = self.cac_indicator.calculate(self.data)
        
        # 結果検証
        print(f"✅ 計算完了 - データ長: {len(self.cac_result.upper_channel)}")
        print(f"🎯 ブレイクアウトシグナル数: {np.sum(np.abs(self.cac_result.breakout_signals))}")
        
        # NaN値チェック
        nan_count = {
            'upper_channel': np.isnan(self.cac_result.upper_channel).sum(),
            'lower_channel': np.isnan(self.cac_result.lower_channel).sum(),
            'midline': np.isnan(self.cac_result.midline).sum(),
            'trend_strength': np.isnan(self.cac_result.trend_strength).sum()
        }
        
        print(f"🔍 NaN値確認:")
        for key, count in nan_count.items():
            print(f"  {key}: {count}個")
        
        # 宇宙知能レポート取得
        intelligence_report = self.cac_indicator.get_cosmic_intelligence_report()
        print(f"\n🌌 宇宙知能レポート:")
        print(f"🧠 現在のトレンドフェーズ: {intelligence_report['current_trend_phase']}")
        print(f"🌊 ボラティリティレジーム: {intelligence_report['current_volatility_regime']}")
        print(f"🚀 ブレイクアウト確率: {intelligence_report['current_breakout_probability']:.3f}")
        print(f"⚛️ 宇宙知能スコア: {intelligence_report['cosmic_intelligence_score']:.3f}")
        
        print("✅ Cosmic Adaptive Channel計算完了")
            
    def analyze_performance(self) -> Dict[str, Any]:
        """
        📊 詳細パフォーマンス解析
        
        Returns:
            解析結果の辞書
        """
        if self.cac_result is None:
            raise ValueError("インジケーターが計算されていません。")
        
        print("\n📊 パフォーマンス解析開始...")
        print("=" * 60)
        
        # 基本統計
        signals = self.cac_result.breakout_signals
        confidences = self.cac_result.breakout_confidence
        trend_strength = self.cac_result.trend_strength
        
        # シグナル統計
        total_signals = np.sum(np.abs(signals))
        up_signals = np.sum(signals == 1)
        down_signals = np.sum(signals == -1)
        high_confidence_signals = np.sum(confidences > 0.5)
        
        # トレンド解析
        strong_trend_periods = np.sum(np.abs(trend_strength) > 0.7)
        weak_trend_periods = np.sum(np.abs(trend_strength) < 0.3)
        trend_consistency = np.std(trend_strength)
        
        # 偽シグナル分析
        false_signals = np.sum(self.cac_result.false_signal_filter == 0)
        signal_quality = 1 - (false_signals / max(total_signals, 1))
        
        # ボラティリティレジーム分析
        regime_distribution = {}
        for regime in [1, 2, 3, 4, 5]:
            count = np.sum(self.cac_result.volatility_regime == regime)
            regime_distribution[f"regime_{regime}"] = count
        
        # 神経適応学習評価
        adaptation_improvement = np.mean(self.cac_result.adaptation_score[-100:]) - np.mean(self.cac_result.adaptation_score[:100])
        learning_stability = 1 - np.std(self.cac_result.learning_velocity)
        
        # 量子コヒーレンス解析
        quantum_stability = np.mean(self.cac_result.quantum_coherence)
        quantum_consistency = 1 - np.std(self.cac_result.quantum_coherence)
        
        # チャネル効率度
        channel_effectiveness = np.mean(self.cac_result.channel_efficiency)
        
        analysis = {
            # シグナル解析
            'total_signals': int(total_signals),
            'up_signals': int(up_signals),
            'down_signals': int(down_signals),
            'high_confidence_signals': int(high_confidence_signals),
            'signal_quality': signal_quality,
            
            # トレンド解析
            'strong_trend_periods': int(strong_trend_periods),
            'weak_trend_periods': int(weak_trend_periods),
            'trend_consistency': trend_consistency,
            
            # システム性能
            'adaptation_improvement': adaptation_improvement,
            'learning_stability': learning_stability,
            'quantum_stability': quantum_stability,
            'quantum_consistency': quantum_consistency,
            'channel_effectiveness': channel_effectiveness,
            
            # レジーム分布
            'regime_distribution': regime_distribution,
            
            # 宇宙知能レポート
            'intelligence_report': self.cac_indicator.get_cosmic_intelligence_report()
        }
        
        # 結果表示
        print(f"🎯 シグナル統計:")
        print(f"  総シグナル数: {analysis['total_signals']}")
        print(f"  上昇シグナル: {analysis['up_signals']}")
        print(f"  下降シグナル: {analysis['down_signals']}")
        print(f"  高信頼シグナル: {analysis['high_confidence_signals']}")
        print(f"  シグナル品質: {analysis['signal_quality']:.3f}")
        
        print(f"\n📈 トレンド解析:")
        print(f"  強トレンド期間: {analysis['strong_trend_periods']}")
        print(f"  弱トレンド期間: {analysis['weak_trend_periods']}")
        print(f"  トレンド一貫性: {analysis['trend_consistency']:.3f}")
        
        print(f"\n🧠 システム性能:")
        print(f"  適応改善度: {analysis['adaptation_improvement']:+.3f}")
        print(f"  学習安定性: {analysis['learning_stability']:.3f}")
        print(f"  量子安定性: {analysis['quantum_stability']:.3f}")
        print(f"  量子一貫性: {analysis['quantum_consistency']:.3f}")
        print(f"  チャネル効率: {analysis['channel_effectiveness']:.3f}")
        
        print(f"\n🌊 ボラティリティレジーム分布:")
        for regime, count in analysis['regime_distribution'].items():
            percentage = count / len(self.data) * 100
            print(f"  {regime}: {count} ({percentage:.1f}%)")
        
        return analysis
    
    def simulate_strategy(self, min_confidence: float = 0.5) -> Dict[str, Any]:
        """
        💹 トレンドフォロー戦略シミュレーション
        
        Args:
            min_confidence: 最小信頼度しきい値
            
        Returns:
            戦略結果の辞書
        """
        if self.cac_result is None:
            raise ValueError("インジケーターが計算されていません。")
        
        print(f"\n💹 戦略シミュレーション開始 (信頼度≥{min_confidence})")
        print("=" * 60)
        
        signals = self.cac_result.breakout_signals
        confidences = self.cac_result.breakout_confidence
        prices = self.data['close'].values
        
        position = 0  # 0=ポジションなし, 1=ロング, -1=ショート
        entry_price = 0
        trades = []
        returns = []
        
        for i in range(len(signals)):
            signal = signals[i]
            confidence = confidences[i]
            current_price = prices[i]
            
            # 高信頼度シグナルのみ処理
            if signal != 0 and confidence >= min_confidence:
                
                # 上昇ブレイクアウト
                if signal == 1 and position != 1:
                    # 既存ショートポジションクローズ
                    if position == -1:
                        ret = (entry_price - current_price) / entry_price
                        returns.append(ret)
                        trades.append({
                            'type': 'close_short',
                            'price': current_price,
                            'return': ret,
                            'confidence': confidence,
                            'index': i
                        })
                    
                    # ロングオープン
                    position = 1
                    entry_price = current_price
                    trades.append({
                        'type': 'open_long',
                        'price': current_price,
                        'confidence': confidence,
                        'index': i
                    })
                
                # 下降ブレイクアウト
                elif signal == -1 and position != -1:
                    # 既存ロングポジションクローズ
                    if position == 1:
                        ret = (current_price - entry_price) / entry_price
                        returns.append(ret)
                        trades.append({
                            'type': 'close_long',
                            'price': current_price,
                            'return': ret,
                            'confidence': confidence,
                            'index': i
                        })
                    
                    # ショートオープン
                    position = -1
                    entry_price = current_price
                    trades.append({
                        'type': 'open_short',
                        'price': current_price,
                        'confidence': confidence,
                        'index': i
                    })
        
        # 最終ポジションクローズ
        if position != 0:
            final_price = prices[-1]
            if position == 1:
                ret = (final_price - entry_price) / entry_price
            else:
                ret = (entry_price - final_price) / entry_price
            returns.append(ret)
            trades.append({
                'type': 'final_close',
                'price': final_price,
                'return': ret,
                'index': len(prices)-1
            })
        
        # 戦略統計計算
        strategy_stats = {}
        
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
                'return_std': np.std(returns),
                'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
                'trades': trades,
                'returns': returns
            }
            
            # 結果表示
            print(f"📊 戦略成績:")
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
    
    def plot(self, 
            title: Optional[str] = None,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (16, 20),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        🎨 宇宙最強チャートを描画する
        
        Args:
            title: チャートのタイトル
            start_date: 表示開始日（フォーマット: YYYY-MM-DD）
            end_date: 表示終了日（フォーマット: YYYY-MM-DD）
            show_volume: 出来高を表示するか
            figsize: 図のサイズ
            style: mplfinanceのスタイル
            savefig: 保存先のパス（指定しない場合は表示のみ）
        """
        if self.data is None or self.cac_result is None:
            raise ValueError("データまたはインジケーターが計算されていません。")
        
        print("\n🎨 宇宙最強チャート描画開始...")
        print("=" * 60)
        
        # タイトル設定
        if title is None:
            title = f"🌌 Cosmic Adaptive Channel - {self.symbol} ({self.timeframe})"
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
        
        # インジケーターデータをDataFrameに結合
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'upper_channel': self.cac_result.upper_channel,
                'lower_channel': self.cac_result.lower_channel,
                'midline': self.cac_result.midline,
                'breakout_signals': self.cac_result.breakout_signals,
                'breakout_confidence': self.cac_result.breakout_confidence,
                'trend_strength': self.cac_result.trend_strength,
                'quantum_coherence': self.cac_result.quantum_coherence,
                'neural_weights': self.cac_result.neural_weights,
                'adaptation_score': self.cac_result.adaptation_score,
                'volatility_regime': self.cac_result.volatility_regime,
                'channel_efficiency': self.cac_result.channel_efficiency,
                'trend_momentum': self.cac_result.trend_momentum,
                'reversal_probability': self.cac_result.reversal_probability
            }
        )
        
        # 絞り込み後のデータに結合
        df = df.join(full_df)
        
        # ブレイクアウトポイントの準備
        up_breakouts = np.where(df['breakout_signals'] == 1)[0]
        down_breakouts = np.where(df['breakout_signals'] == -1)[0]
        
        print(f"📊 チャートデータ準備完了:")
        print(f"  期間: {df.index.min()} → {df.index.max()}")
        print(f"  データ数: {len(df)}")
        print(f"  上昇ブレイクアウト: {len(up_breakouts)}個")
        print(f"  下降ブレイクアウト: {len(down_breakouts)}個")
        
        # mplfinanceプロット設定
        main_plots = []
        
        # 1. チャネルライン
        main_plots.append(mpf.make_addplot(df['upper_channel'], color='lime', width=2, alpha=0.8))
        main_plots.append(mpf.make_addplot(df['lower_channel'], color='red', width=2, alpha=0.8))
        main_plots.append(mpf.make_addplot(df['midline'], color='blue', width=1.5, alpha=0.9))
        
        # 2. チャネルエリア塗りつぶし
        # mpfinanceでは直接fill_betweenはできないので、別途matplotlibで描画
        
        # 3. 各種パネルの設定
        panel_num = 1 if show_volume else 0
        
        # トレンド強度パネル
        trend_panel = mpf.make_addplot(df['trend_strength'], panel=panel_num+1, color='purple', width=2,
                                     ylabel='Trend Strength', secondary_y=False)
        
        # 量子コヒーレンスパネル
        quantum_panel = mpf.make_addplot(df['quantum_coherence'], panel=panel_num+2, color='cyan', width=2,
                                       ylabel='Quantum Coherence', secondary_y=False)
        
        # 神経適応スコアパネル
        neural_panel = mpf.make_addplot(df['adaptation_score'], panel=panel_num+3, color='green', width=2,
                                      ylabel='Neural Adaptation', secondary_y=False)
        
        # ボラティリティレジームパネル
        regime_panel = mpf.make_addplot(df['volatility_regime'], panel=panel_num+4, color='orange', width=2,
                                       ylabel='Volatility Regime', secondary_y=False, type='line')
        
        # ブレイクアウト信頼度パネル
        confidence_panel = mpf.make_addplot(df['breakout_confidence'], panel=panel_num+5, color='magenta', width=1.5,
                                          ylabel='Breakout Confidence', secondary_y=False)
        
        # すべてのプロットを結合
        all_plots = main_plots + [trend_panel, quantum_panel, neural_panel, regime_panel, confidence_panel]
        
        # mplfinanceの設定
        kwargs = dict(
            type='candle',
            figsize=figsize,
            title=title,
            style=style,
            datetime_format='%Y-%m-%d',
            xrotation=45,
            returnfig=True,
            addplot=all_plots
        )
        
        # パネル比率の設定
        if show_volume:
            kwargs['volume'] = True
            kwargs['panel_ratios'] = (6, 1, 1.5, 1.5, 1.5, 1.5, 1.5)  # メイン:出来高:各パネル
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (6, 1.5, 1.5, 1.5, 1.5, 1.5)  # メイン:各パネル
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # ブレイクアウトポイントを手動でプロット
        if len(up_breakouts) > 0:
            axes[0].scatter(up_breakouts, df['close'].iloc[up_breakouts], 
                          color='lime', marker='^', s=100, zorder=5, alpha=0.8,
                          label=f'上昇ブレイクアウト ({len(up_breakouts)})')
        
        if len(down_breakouts) > 0:
            axes[0].scatter(down_breakouts, df['close'].iloc[down_breakouts], 
                          color='red', marker='v', s=100, zorder=5, alpha=0.8,
                          label=f'下降ブレイクアウト ({len(down_breakouts)})')
        
        # チャネルエリア塗りつぶし
        axes[0].fill_between(df.index, df['upper_channel'], df['lower_channel'], 
                           alpha=0.1, color='purple', label='チャネルエリア')
        
        # 凡例追加
        axes[0].legend(loc='upper left')
        
        # 各パネルに参照線を追加
        panel_start = 2 if show_volume else 1
        
        # トレンド強度パネル
        axes[panel_start].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[panel_start].axhline(y=0.7, color='green', linestyle='--', alpha=0.5)
        axes[panel_start].axhline(y=-0.7, color='red', linestyle='--', alpha=0.5)
        
        # 量子コヒーレンスパネル
        axes[panel_start+1].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
        axes[panel_start+1].axhline(y=0.4, color='red', linestyle=':', alpha=0.5)  # 偽シグナルしきい値
        
        # 神経適応スコアパネル
        axes[panel_start+2].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
        
        # ボラティリティレジームパネル
        axes[panel_start+3].axhline(y=3, color='black', linestyle='-', alpha=0.5)  # 中ボラティリティ
        
        # ブレイクアウト信頼度パネル
        axes[panel_start+4].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)  # 信頼度しきい値
        
        self.fig = fig
        self.axes = axes
        
        # 保存または表示
        if savefig:
            plt.savefig(savefig, dpi=300, bbox_inches='tight')
            print(f"📊 チャートを保存しました: {savefig}")
        else:
            plt.tight_layout()
            plt.show()
        
        print("✅ チャート描画完了")


def main():
    """メイン関数"""
    import argparse
    parser = argparse.ArgumentParser(description='🌌 Cosmic Adaptive Channel可視化テスト')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--atr-period', type=int, default=21, help='ATR期間')
    parser.add_argument('--base-mult', type=float, default=2.0, help='基本倍率')
    parser.add_argument('--quantum-window', type=int, default=50, help='量子解析ウィンドウ')
    parser.add_argument('--neural-window', type=int, default=100, help='神経学習ウィンドウ')
    parser.add_argument('--min-confidence', type=float, default=0.5, help='戦略シミュレーション最小信頼度')
    args = parser.parse_args()
    
    # チャートを作成
    print("🌌" * 20)
    print("🌌 Cosmic Adaptive Channel - 宇宙最強テストシステム 🌌")
    print("🌌" * 20)
    
    chart = CosmicAdaptiveChannelChart()
    
    try:
        # データ読み込み
        chart.load_data_from_config(args.config)
        
        # インジケーター計算
        chart.calculate_indicators(
            atr_period=args.atr_period,
            base_multiplier=args.base_mult,
            quantum_window=args.quantum_window,
            neural_window=args.neural_window
        )
        
        # パフォーマンス解析
        analysis = chart.analyze_performance()
        
        # 戦略シミュレーション
        strategy_stats = chart.simulate_strategy(min_confidence=args.min_confidence)
        
        # チャート描画
        chart.plot(
            start_date=args.start,
            end_date=args.end,
            savefig=args.output
        )
        
        print(f"\n🎯 最終評価:")
        print(f"宇宙知能スコア: {analysis['intelligence_report']['cosmic_intelligence_score']:.3f}")
        print(f"チャネル効率度: {analysis['channel_effectiveness']:.3f}")
        print(f"偽シグナル防御率: {(1-analysis['intelligence_report']['false_signal_rate']):.1%}")
        if strategy_stats:
            print(f"戦略リターン: {strategy_stats['total_return']:+.2%}")
            print(f"戦略勝率: {strategy_stats['win_rate']:.1%}")
        
        print(f"\n🌌 Cosmic Adaptive Channel テスト完了! 🌌")
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()