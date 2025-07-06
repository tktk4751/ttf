#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🌌 Real Market UQAVC Analyzer - リアル相場データ分析ツール

🎯 **機能:**
- config.yamlから実際の相場データ取得
- Ultra Quantum Adaptive Volatility Channel 完全分析
- 4層詳細チャート描画
- パフォーマンス比較分析
- リアルタイム市場分析レポート
- トレーディングシグナル生成
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import yaml
import warnings
import os
import sys
import time
from typing import Optional, Dict, Any

warnings.filterwarnings('ignore')

# プロジェクトのルートディレクトリを追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicators.quantum_adaptive_volatility_channel import QuantumAdaptiveVolatilityChannel

# データローダーのインポート
try:
    from data.data_loader import DataLoader, CSVDataSource
    from data.data_processor import DataProcessor
    from data.binance_data_source import BinanceDataSource
except ImportError as e:
    print(f"⚠️ データローダーモジュールが見つかりません: {e}")
    print("data/フォルダのモジュールを確認してください。")


class RealMarketDataFetcher:
    """🌐 リアル相場データ取得クラス"""
    
    def __init__(self):
        pass
    
    def load_real_data_from_config(self, config_path: str = "config.yaml") -> Optional[pd.DataFrame]:
        """
        🔄 設定ファイルから実際の相場データを読み込む
        
        Args:
            config_path: 設定ファイルのパス
            
        Returns:
            処理済みのOHLCVデータフレーム
        """
        print(f"📡 設定ファイルから実データを読み込み中: {config_path}")
        
        try:
            # 設定ファイルの読み込み
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            # データの準備
            binance_config = config.get('binance_data', {})
            data_dir = binance_config.get('data_dir', 'data/binance')
            binance_data_source = BinanceDataSource(data_dir)
            
            # CSVデータソースはダミーとして渡す（Binanceデータソースのみを使用）
            dummy_csv_source = CSVDataSource("dummy")
            data_loader = DataLoader(
                data_source=dummy_csv_source,
                binance_data_source=binance_data_source
            )
            data_processor = DataProcessor()
            
            # データの読み込みと処理
            print("📊 データを読み込み・処理中...")
            raw_data = data_loader.load_data_from_config(config)
            processed_data = {
                symbol: data_processor.process(df)
                for symbol, df in raw_data.items()
            }
            
            # 最初のシンボルのデータを取得
            first_symbol = next(iter(processed_data))
            real_data = processed_data[first_symbol]
            
            print(f"✅ データ読み込み完了: {first_symbol}")
            print(f"📅 期間: {real_data.index.min()} → {real_data.index.max()}")
            print(f"📈 データ数: {len(real_data)}")
            
            # 必要なカラムの確認と修正
            if 'close' not in real_data.columns:
                real_data['close'] = real_data['Close'] if 'Close' in real_data.columns else real_data.iloc[:, 3]
            if 'open' not in real_data.columns:
                real_data['open'] = real_data['Open'] if 'Open' in real_data.columns else real_data.iloc[:, 0]
            if 'high' not in real_data.columns:
                real_data['high'] = real_data['High'] if 'High' in real_data.columns else real_data.iloc[:, 1]
            if 'low' not in real_data.columns:
                real_data['low'] = real_data['Low'] if 'Low' in real_data.columns else real_data.iloc[:, 2]
            
            # timestampカラムの追加（インデックスから）
            real_data['timestamp'] = real_data.index
            
            return real_data
            
        except Exception as e:
            print(f"❌ 実データ読み込みエラー: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_csv_data(self, csv_path: str) -> Optional[pd.DataFrame]:
        """
        📁 CSVファイルから相場データを読み込み
        """
        try:
            print(f"📁 CSVファイル読み込み中: {csv_path}")
            
            df = pd.read_csv(csv_path)
            
            # 必要な列の確認
            required_columns = ['timestamp', 'open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_columns):
                print(f"❌ 必要な列が不足: {required_columns}")
                print(f"利用可能な列: {list(df.columns)}")
                return None
            
            # タイムスタンプを日時型に変換
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            print(f"✅ CSV読み込み完了: {len(df)} 件")
            print(f"📅 期間: {df['timestamp'].iloc[0]} ～ {df['timestamp'].iloc[-1]}")
            print(f"💰 価格範囲: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
            
            return df
            
        except Exception as e:
            print(f"❌ CSV読み込みエラー: {e}")
            return None


class UQAVCMarketAnalyzer:
    """🌌 UQAVC市場分析エンジン"""
    
    def __init__(self, 
                 volatility_period: int = 21,
                 base_multiplier: float = 2.0,
                 src_type: str = 'hlc3'):
        """
        初期化
        
        Args:
            volatility_period: ボラティリティ計算期間
            base_multiplier: 基本チャネル幅倍率
            src_type: 価格ソースタイプ
        """
        self.uqavc = QuantumAdaptiveVolatilityChannel(
            volatility_period=volatility_period,
            base_multiplier=base_multiplier,
            src_type=src_type
        )
        
        self.data = None
        self.result = None
        self.analysis_summary = None
    
    def analyze_market_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        🎯 市場データの完全分析を実行
        """
        print("\n🌌 UQAVC市場分析開始...")
        
        self.data = data.copy()
        
        # UQAVC計算
        start_time = time.time()
        self.result = self.uqavc.calculate(data)
        calculation_time = time.time() - start_time
        
        print(f"⚡ UQAVC計算完了: {calculation_time:.4f}秒")
        
        # 詳細分析を実行
        self.analysis_summary = self._perform_detailed_analysis()
        
        return self.analysis_summary
    
    def _perform_detailed_analysis(self) -> Dict[str, Any]:
        """📊 詳細分析を実行"""
        
        if self.result is None or self.data is None:
            return {}
        
        close_prices = self.data['close'].values
        
        # 基本統計
        basic_stats = {
            'データ期間': f"{self.data['timestamp'].iloc[0]} ～ {self.data['timestamp'].iloc[-1]}",
            'データ数': len(self.data),
            '価格範囲': f"${close_prices.min():.2f} - ${close_prices.max():.2f}",
            '最新価格': f"${close_prices[-1]:.2f}",
            '期間リターン': f"{((close_prices[-1] / close_prices[0]) - 1) * 100:.2f}%"
        }
        
        # UQAVC分析
        uqavc_analysis = self._analyze_uqavc_signals()
        
        # トレンド分析
        trend_analysis = self._analyze_trend_patterns()
        
        # ボラティリティ分析
        volatility_analysis = self._analyze_volatility_patterns()
        
        # 市場レジーム分析
        regime_analysis = self._analyze_market_regimes()
        
        # トレーディングシグナル
        trading_signals = self._generate_trading_signals()
        
        return {
            '基本統計': basic_stats,
            'UQAVC分析': uqavc_analysis,
            'トレンド分析': trend_analysis,
            'ボラティリティ分析': volatility_analysis,
            '市場レジーム分析': regime_analysis,
            'トレーディングシグナル': trading_signals
        }
    
    def _analyze_uqavc_signals(self) -> Dict[str, Any]:
        """🎯 UQAVCシグナル分析"""
        
        breakout_signals = self.result.breakout_signals
        signal_strength = getattr(self.result, 'signal_strength', np.ones_like(breakout_signals) * 0.5)
        
        # シグナル統計
        total_signals = np.sum(np.abs(breakout_signals))
        buy_signals = np.sum(breakout_signals == 1)
        sell_signals = np.sum(breakout_signals == -1)
        
        # 信頼度分析
        high_confidence_signals = np.sum(signal_strength > 0.7)
        avg_confidence = np.mean(signal_strength[signal_strength > 0]) if np.any(signal_strength > 0) else 0.0
        
        # チャネル効率
        channel_width = np.mean(self.result.upper_channel - self.result.lower_channel)
        price_range = np.max(self.data['close']) - np.min(self.data['close'])
        channel_efficiency = channel_width / price_range if price_range > 0 else 0.0
        
        return {
            '総シグナル数': int(total_signals),
            '買いシグナル': int(buy_signals),
            '売りシグナル': int(sell_signals),
            '平均信頼度': round(avg_confidence, 3),
            '高信頼度シグナル': int(high_confidence_signals),
            'チャネル効率': round(channel_efficiency, 3),
            'シグナル頻度': f"{total_signals / len(self.data) * 100:.1f}%"
        }
    
    def _analyze_trend_patterns(self) -> Dict[str, Any]:
        """📈 トレンド分析"""
        
        trend_probability = getattr(self.result, 'trend_probability', np.full(len(self.data), 0.5))
        quantum_state = getattr(self.result, 'quantum_state', np.full(len(self.data), 0.5))
        
        # トレンド強度分析
        strong_trend_periods = np.sum(trend_probability > 0.7)
        weak_trend_periods = np.sum(trend_probability < 0.3)
        
        # 現在のトレンド状態
        current_trend_strength = trend_probability[-1] if len(trend_probability) > 0 else 0.5
        current_quantum_state = quantum_state[-1] if len(quantum_state) > 0 else 0.5
        
        return {
            '現在のトレンド強度': round(current_trend_strength, 3),
            '現在の量子状態': round(current_quantum_state, 3),
            '強いトレンド期間': f"{strong_trend_periods / len(self.data) * 100:.1f}%",
            '弱いトレンド期間': f"{weak_trend_periods / len(self.data) * 100:.1f}%",
            'トレンド一貫性': round(np.std(trend_probability), 3),
            '現在のレジーム': getattr(self.result, 'current_regime', 'unknown')
        }
    
    def _analyze_volatility_patterns(self) -> Dict[str, Any]:
        """📊 ボラティリティ分析"""
        
        volatility_forecast = getattr(self.result, 'volatility_forecast', np.full(len(self.data), 0.02))
        dynamic_width = self.result.dynamic_width
        
        # ボラティリティ統計
        avg_volatility = np.mean(volatility_forecast)
        current_volatility = volatility_forecast[-1] if len(volatility_forecast) > 0 else 0.02
        volatility_trend = np.corrcoef(np.arange(len(volatility_forecast)), volatility_forecast)[0, 1]
        
        # 動的幅の効果
        avg_dynamic_width = np.mean(dynamic_width)
        width_adaptation = np.std(dynamic_width) / np.mean(dynamic_width) if np.mean(dynamic_width) > 0 else 0
        
        return {
            '平均ボラティリティ': round(avg_volatility, 4),
            '現在のボラティリティ': round(current_volatility, 4),
            'ボラティリティトレンド': round(volatility_trend, 3),
            '平均チャネル幅': round(avg_dynamic_width, 2),
            '幅適応性': round(width_adaptation, 3),
            'ボラティリティレベル': getattr(self.result, 'current_volatility_level', 'medium')
        }
    
    def _analyze_market_regimes(self) -> Dict[str, Any]:
        """🎭 市場レジーム分析"""
        
        regime_state = getattr(self.result, 'regime_state', np.zeros(len(self.data)))
        
        # レジーム分布
        range_periods = np.sum(regime_state == 0)
        trend_periods = np.sum(regime_state == 1)
        breakout_periods = np.sum(regime_state == 2)
        crash_periods = np.sum(regime_state == 3)
        
        total_periods = len(self.data)
        
        return {
            'レンジ相場': f"{range_periods / total_periods * 100:.1f}%",
            'トレンド相場': f"{trend_periods / total_periods * 100:.1f}%",
            'ブレイクアウト相場': f"{breakout_periods / total_periods * 100:.1f}%",
            'クラッシュ相場': f"{crash_periods / total_periods * 100:.1f}%",
            '現在のレジーム': getattr(self.result, 'current_regime', 'unknown'),
            'レジーム安定性': round(1 - np.std(regime_state) / (np.mean(regime_state) + 1e-8), 3)
        }
    
    def _generate_trading_signals(self) -> Dict[str, Any]:
        """⚡ トレーディングシグナル生成"""
        
        close_prices = self.data['close'].values
        breakout_signals = self.result.breakout_signals
        upper_channel = self.result.upper_channel
        lower_channel = self.result.lower_channel
        
        # 現在の市場状況
        current_price = close_prices[-1]
        current_upper = upper_channel[-1]
        current_lower = lower_channel[-1]
        channel_position = (current_price - current_lower) / (current_upper - current_lower)
        
        # 最新シグナル
        latest_signal = breakout_signals[-1] if len(breakout_signals) > 0 else 0
        signal_strength = getattr(self.result, 'signal_strength', np.ones_like(breakout_signals))
        latest_strength = signal_strength[-1] if len(signal_strength) > 0 else 0.5
        
        # アクション推奨
        if latest_signal == 1 and latest_strength > 0.6:
            action = "🟢 強い買いシグナル"
        elif latest_signal == 1 and latest_strength > 0.3:
            action = "🟡 弱い買いシグナル"
        elif latest_signal == -1 and latest_strength > 0.6:
            action = "🔴 強い売りシグナル"
        elif latest_signal == -1 and latest_strength > 0.3:
            action = "🟠 弱い売りシグナル"
        elif channel_position > 0.8:
            action = "⚠️ 上側チャネル接近 - 売り警戒"
        elif channel_position < 0.2:
            action = "⚠️ 下側チャネル接近 - 買い準備"
        else:
            action = "⚪ 中立 - 様子見"
        
        return {
            '推奨アクション': action,
            'チャネル内位置': f"{channel_position:.1%}",
            '最新シグナル強度': round(latest_strength, 3),
            '上側チャネル距離': f"{((current_upper / current_price) - 1) * 100:.2f}%",
            '下側チャネル距離': f"{(1 - (current_lower / current_price)) * 100:.2f}%",
            '市場知能指数': round(getattr(self.result, 'market_intelligence', 0.5), 3)
        }
    
    def create_detailed_visualization(self, save_path: Optional[str] = None):
        """🎨 詳細なチャート可視化"""
        
        if self.data is None or self.result is None:
            print("❌ 分析データがありません")
            return
        
        print("\n🎨 詳細チャート作成中...")
        
        fig, axes = plt.subplots(4, 1, figsize=(16, 20))
        fig.suptitle('🌌 Ultra Quantum Adaptive Volatility Channel - リアル市場分析', 
                    fontsize=16, fontweight='bold')
        
        timestamps = self.data['timestamp'].values
        close_prices = self.data['close'].values
        
        # チャート1: 価格 + UQAVCチャネル + シグナル
        self._plot_price_and_channel(axes[0], timestamps, close_prices)
        
        # チャート2: 量子・トレンド分析
        self._plot_quantum_trend_analysis(axes[1], timestamps)
        
        # チャート3: ボラティリティ・フラクタル分析
        self._plot_volatility_fractal_analysis(axes[2], timestamps)
        
        # チャート4: 予測・レジーム分析
        self._plot_prediction_regime_analysis(axes[3], timestamps)
        
        plt.tight_layout()
        
        # 保存
        if save_path is None:
            save_path = f"uqavc_real_market_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ チャートを '{save_path}' に保存しました")
        
        plt.show()
        
        return save_path
    
    def _plot_price_and_channel(self, ax, timestamps, close_prices):
        """チャート1: 価格とチャネル"""
        ax.plot(timestamps, close_prices, label='終値', color='black', linewidth=1.5, alpha=0.8)
        ax.plot(timestamps, self.result.upper_channel, label='UQAVC上側チャネル', color='red', alpha=0.7)
        ax.plot(timestamps, self.result.lower_channel, label='UQAVC下側チャネル', color='green', alpha=0.7)
        ax.plot(timestamps, self.result.midline, label='量子中央線', color='blue', alpha=0.8)
        
        # チャネル間を塗りつぶし
        ax.fill_between(timestamps, self.result.upper_channel, self.result.lower_channel,
                       alpha=0.1, color='purple', label='適応チャネル')
        
        # ブレイクアウトシグナルをマーク
        buy_signals = np.where(self.result.breakout_signals == 1)[0]
        sell_signals = np.where(self.result.breakout_signals == -1)[0]
        
        if len(buy_signals) > 0:
            ax.scatter(timestamps[buy_signals], close_prices[buy_signals], 
                      color='green', marker='^', s=100, label='買いシグナル', zorder=5)
        
        if len(sell_signals) > 0:
            ax.scatter(timestamps[sell_signals], close_prices[sell_signals], 
                      color='red', marker='v', s=100, label='売りシグナル', zorder=5)
        
        ax.set_title('🌌 Ultra Quantum Adaptive Volatility Channel')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    
    def _plot_quantum_trend_analysis(self, ax, timestamps):
        """チャート2: 量子・トレンド分析"""
        trend_probability = getattr(self.result, 'trend_probability', np.full(len(self.data), 0.5))
        quantum_state = getattr(self.result, 'quantum_state', np.full(len(self.data), 0.5))
        signal_strength = getattr(self.result, 'signal_strength', np.full(len(self.data), 0.5))
        
        ax.plot(timestamps, trend_probability, label='トレンド確率', color='blue', linewidth=2)
        ax.plot(timestamps, quantum_state, label='量子状態', color='purple', alpha=0.7)
        ax.plot(timestamps, signal_strength, label='シグナル強度', color='orange', alpha=0.7)
        
        # 閾値ライン
        ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='超強力 (0.8+)')
        ax.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='強力 (0.6+)')
        ax.axhline(y=0.4, color='yellow', linestyle='--', alpha=0.5, label='中程度 (0.4+)')
        ax.axhline(y=0.2, color='lightblue', linestyle='--', alpha=0.5, label='弱い (0.2+)')
        
        ax.set_title('📊 量子・トレンド分析')
        ax.set_ylabel('確率/強度')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
    def _plot_volatility_fractal_analysis(self, ax, timestamps):
        """チャート3: ボラティリティ・フラクタル分析"""
        volatility_forecast = getattr(self.result, 'volatility_forecast', np.full(len(self.data), 0.02))
        fractal_dimension = getattr(self.result, 'fractal_dimension', np.full(len(self.data), 1.5))
        dynamic_width = self.result.dynamic_width
        
        ax2 = ax.twinx()
        
        # ボラティリティ（左軸）
        line1 = ax.plot(timestamps, volatility_forecast * 100, label='GARCH ボラティリティ (%)', 
                       color='red', linewidth=2)[0]
        line2 = ax.plot(timestamps, (dynamic_width / np.mean(dynamic_width)) * 2, 
                       label='動的チャネル幅 (正規化)', color='blue', alpha=0.7)[0]
        
        # フラクタル次元（右軸）
        line3 = ax2.plot(timestamps, fractal_dimension, label='フラクタル次元', 
                        color='green', linewidth=1.5)[0]
        
        # フラクタル次元の閾値
        ax2.axhline(y=1.3, color='green', linestyle='--', alpha=0.3, label='単純市場 (1.3)')
        ax2.axhline(y=1.7, color='red', linestyle='--', alpha=0.3, label='複雑市場 (1.7)')
        
        ax.set_title('📈 ボラティリティ・フラクタル分析')
        ax.set_ylabel('ボラティリティ・幅')
        ax2.set_ylabel('フラクタル次元')
        
        # 凡例を統合
        lines = [line1, line2, line3]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
    def _plot_prediction_regime_analysis(self, ax, timestamps):
        """チャート4: 予測・レジーム分析"""
        regime_state = getattr(self.result, 'regime_state', np.zeros(len(self.data)))
        breakout_probability = getattr(self.result, 'breakout_probability', np.full(len(self.data), 0.5))
        confidence_level = getattr(self.result, 'confidence_level', np.full(len(self.data), 0.5))
        
        # レジーム状態を色分け表示
        colors = ['blue', 'green', 'orange', 'red']  # レンジ、トレンド、ブレイクアウト、クラッシュ
        regime_colors = [colors[int(state)] for state in regime_state]
        
        ax.scatter(timestamps, regime_state, c=regime_colors, alpha=0.6, s=20, label='市場レジーム')
        ax.plot(timestamps, breakout_probability, label='ブレイクアウト確率', color='purple', linewidth=2)
        ax.plot(timestamps, confidence_level, label='予測信頼度', color='cyan', alpha=0.7)
        
        # レジーム説明
        ax.axhline(y=0, color='blue', linestyle='-', alpha=0.3, label='レンジ相場')
        ax.axhline(y=1, color='green', linestyle='-', alpha=0.3, label='トレンド相場')
        ax.axhline(y=2, color='orange', linestyle='-', alpha=0.3, label='ブレイクアウト相場')
        ax.axhline(y=3, color='red', linestyle='-', alpha=0.3, label='クラッシュ相場')
        
        ax.set_title('🔮 予測・レジーム分析')
        ax.set_ylabel('レジーム/確率')
        ax.set_xlabel('時間')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.5, 3.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
    def print_analysis_report(self):
        """📋 分析レポートを出力"""
        if not self.analysis_summary:
            print("❌ 分析データがありません")
            return
        
        print("\n" + "="*80)
        print("🌌 ULTRA QUANTUM ADAPTIVE VOLATILITY CHANNEL - 市場分析レポート")
        print("="*80)
        
        for category, data in self.analysis_summary.items():
            print(f"\n📊 {category}:")
            print("-" * 50)
            for key, value in data.items():
                print(f"  {key}: {value}")
        
        print("\n" + "="*80)
        print("🎯 分析完了 - 次世代トレーディング技術による市場洞察")
        print("="*80)


def main():
    """🚀 メイン実行関数"""
    print("🌌 Real Market UQAVC Analyzer - リアル相場データ分析開始")
    print("="*80)
    
    # データ取得器を初期化
    data_fetcher = RealMarketDataFetcher()
    
    # データ取得方法を選択
    print("\n📡 データ取得方法を選択してください:")
    print("1. config.yamlから設定読み込み (推奨)")
    print("2. CSVファイル読み込み")
    
    try:
        choice = input("\n選択 (1-2): ").strip() or "1"
        
        if choice == "1":
            # config.yamlから取得
            config_path = input("設定ファイルパス (例: config.yaml): ").strip() or "config.yaml"
            
            market_data = data_fetcher.load_real_data_from_config(config_path)
            
            if market_data is None:
                print("❌ config.yamlからのデータ取得に失敗しました")
                return
        
        elif choice == "2":
            # CSVファイルから読み込み
            csv_path = input("CSVファイルパス: ").strip()
            market_data = data_fetcher.load_csv_data(csv_path)
            
            if market_data is None:
                print("❌ CSV読み込みに失敗しました")
                return
        
        else:
            print("❌ 無効な選択です")
            return
        
        if market_data is None or len(market_data) == 0:
            print("❌ データ取得に失敗しました")
            return
        
        # データの表示期間を制限（可視化とパフォーマンスのため）
        if len(market_data) > 2000:
            print(f"⚠️ データが多すぎます({len(market_data)}件)、最新2000件に制限します")
            market_data = market_data.tail(2000).copy()
        
        # UQAVC分析器を初期化
        print("\n🔧 UQAVC分析器を初期化中...")
        analyzer = UQAVCMarketAnalyzer(
            volatility_period=21,
            base_multiplier=2.0,
            src_type='hlc3'
        )
        
        # 市場分析を実行
        analysis_result = analyzer.analyze_market_data(market_data)
        
        # 分析レポート出力
        analyzer.print_analysis_report()
        
        # チャート作成
        print("\n🎨 詳細チャート作成中...")
        chart_path = analyzer.create_detailed_visualization()
        
        print(f"\n✅ 分析完了! チャートは {chart_path} に保存されました")
        
        # パフォーマンス比較（オプション）
        performance_comparison = input("\n📊 詳細パフォーマンス分析を実行しますか？ (y/n): ").strip().lower()
        if performance_comparison == 'y':
            perform_detailed_performance_analysis(analyzer, market_data)
    
    except KeyboardInterrupt:
        print("\n\n⚠️ ユーザーによる中断")
    except Exception as e:
        print(f"\n❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()


def perform_detailed_performance_analysis(analyzer, market_data):
    """📊 詳細パフォーマンス分析"""
    print("\n📊 詳細パフォーマンス分析実行中...")
    
    result = analyzer.result
    
    # シグナル精度分析
    breakout_signals = result.breakout_signals
    signal_indices = np.where(breakout_signals != 0)[0]
    
    if len(signal_indices) > 0:
        print(f"\n🎯 シグナル精度分析:")
        print(f"総シグナル数: {len(signal_indices)}")
        
        # 各シグナル後のパフォーマンス
        close_prices = market_data['close'].values
        signal_returns = []
        
        for i, signal_idx in enumerate(signal_indices):
            if signal_idx < len(close_prices) - 5:  # 5期間後まで確認
                signal_direction = breakout_signals[signal_idx]
                entry_price = close_prices[signal_idx]
                exit_price = close_prices[signal_idx + 5]
                
                if signal_direction == 1:  # 買いシグナル
                    return_rate = (exit_price - entry_price) / entry_price
                else:  # 売りシグナル
                    return_rate = (entry_price - exit_price) / entry_price
                
                signal_returns.append(return_rate)
        
        if signal_returns:
            avg_return = np.mean(signal_returns)
            win_rate = np.sum(np.array(signal_returns) > 0) / len(signal_returns)
            
            print(f"平均リターン: {avg_return:.4f} ({avg_return*100:.2f}%)")
            print(f"勝率: {win_rate:.3f} ({win_rate*100:.1f}%)")
            print(f"最大利益: {np.max(signal_returns):.4f}")
            print(f"最大損失: {np.min(signal_returns):.4f}")
    
    # チャネル効率分析
    upper_channel = result.upper_channel
    lower_channel = result.lower_channel
    close_prices = market_data['close'].values
    
    # チャネル内滞在率
    in_channel = ((close_prices >= lower_channel) & (close_prices <= upper_channel))
    channel_stay_rate = np.sum(in_channel) / len(close_prices)
    
    print(f"\n📊 チャネル効率分析:")
    print(f"チャネル内滞在率: {channel_stay_rate:.3f} ({channel_stay_rate*100:.1f}%)")
    
    # 偽ブレイクアウト率
    breakout_count = np.sum(np.abs(breakout_signals))
    if breakout_count > 0:
        false_breakout_count = 0
        for i in np.where(breakout_signals != 0)[0]:
            if i < len(close_prices) - 3:
                # 3期間後に元のチャネル内に戻ったかチェック
                future_price = close_prices[i + 3]
                if lower_channel[i] <= future_price <= upper_channel[i]:
                    false_breakout_count += 1
        
        false_breakout_rate = false_breakout_count / breakout_count
        print(f"偽ブレイクアウト率: {false_breakout_rate:.3f} ({false_breakout_rate*100:.1f}%)")
    
    print("\n✅ パフォーマンス分析完了!")


if __name__ == "__main__":
    main()