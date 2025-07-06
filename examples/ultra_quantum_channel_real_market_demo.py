#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🌌 **Ultra Quantum Adaptive Volatility Channel (UQAVC) - リアル相場データ検証** 🌌

🎯 **実際の相場データでUQAVC革新機能を完全検証:**
- **15層フィルタリング**: ウェーブレット + 量子コヒーレンス + 神経回路網
- **17指標統合幅調整**: 超知能適応システム
- **量子もつれ検出**: 市場の量子的相関解析
- **神経学習システム**: リアルタイム適応
- **4層統合可視化**: 包括的市場分析チャート
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import yaml
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

from indicators.ultra_quantum_adaptive_channel import UltraQuantumAdaptiveVolatilityChannel
from data.data_loader import DataLoader, CSVDataSource
from data.binance_data_source import BinanceDataSource
from data.data_processor import DataProcessor


class UQAVCRealMarketTester:
    """UQAVCリアル市場テスター"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        self.config_path = config_path
        self.config = self._load_config()
        
        # データローダー初期化
        self.csv_data_source = CSVDataSource("dummy")
        self.binance_data_source = BinanceDataSource()
        self.data_processor = DataProcessor()
        self.data_loader = DataLoader(
            data_source=self.csv_data_source,
            binance_data_source=self.binance_data_source
        )
        
        # UQAVC初期化
        self.uqavc = UltraQuantumAdaptiveVolatilityChannel(
            volatility_period=21,
            base_multiplier=2.0,
            quantum_window=50,
            neural_window=100,
            src_type='hlc3'
        )
        
        self.data = None
        self.results = None
    
    def _load_config(self) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"⚠️ 設定ファイルが見つかりません: {self.config_path}")
            return {
                'symbol': 'BTCUSDT',
                'timeframe': '1h',
                'limit': 2000
            }
    
    def fetch_market_data(self) -> pd.DataFrame:
        """市場データ取得"""
        symbol = self.config.get('symbol', 'BTCUSDT')
        timeframe = self.config.get('timeframe', '1h')
        limit = self.config.get('limit', 2000)
        
        print(f"📊 市場データ取得: {symbol} ({timeframe}) - {limit}件")
        
        try:
            data = self.data_loader.load_market_data(
                symbol=symbol,
                timeframe=timeframe
            )
            
            # limitで制限
            if len(data) > limit:
                data = data.tail(limit)
                
            # timestampカラムを追加（DataLoaderではindexになるため）
            if 'timestamp' not in data.columns:
                data = data.reset_index()
                if 'index' in data.columns:
                    data = data.rename(columns={'index': 'timestamp'})
            
            if data is None or data.empty:
                print("⚠️ リアルデータ取得失敗、サンプルデータ生成中...")
                return self._generate_sample_data(limit)
            
            print(f"✅ リアルデータ取得完了: {len(data)}件")
            return data
            
        except Exception as e:
            print(f"❌ データ取得エラー: {e}")
            print("📈 サンプルデータを生成します...")
            return self._generate_sample_data(limit)
    
    def _generate_sample_data(self, limit: int) -> pd.DataFrame:
        """サンプルデータ生成"""
        np.random.seed(42)
        base_price = 50000.0
        
        dates = pd.date_range(start='2024-01-01', periods=limit, freq='H')
        prices = [base_price]
        
        for i in range(1, limit):
            # 複雑なトレンド成分
            trend = 0.0002 * np.sin(i * 0.01) + 0.0001 * np.cos(i * 0.005)
            volatility = 0.02 * (1 + 0.5 * np.sin(i * 0.05))
            random_change = np.random.normal(trend, volatility)
            
            new_price = prices[-1] * (1 + random_change)
            prices.append(max(1000, new_price))
        
        data = []
        for i, price in enumerate(prices):
            noise = np.random.normal(0, price * 0.005)
            high = price + abs(noise)
            low = price - abs(noise)
            open_price = price + np.random.normal(0, price * 0.002)
            close_price = price + np.random.normal(0, price * 0.002)
            
            data.append({
                'timestamp': dates[i],
                'open': open_price,
                'high': max(open_price, high, close_price),
                'low': min(open_price, low, close_price),
                'close': close_price,
                'volume': np.random.uniform(100, 1000)
            })
        
        return pd.DataFrame(data)
    
    def run_uqavc_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """UQAVC完全分析実行"""
        print("🌌 UQAVC完全分析開始...")
        
        self.data = data.copy()
        self.results = self.uqavc.calculate(data)
        
        if not self.results:
            print("❌ UQAVC計算失敗")
            return {}
        
        # 分析結果生成
        analysis = {
            'data_info': self._analyze_data_info(),
            'quantum_metrics': self._analyze_quantum_metrics(),
            'neural_performance': self._analyze_neural_performance(),
            'channel_efficiency': self._analyze_channel_efficiency(),
            'signal_quality': self._analyze_signal_quality(),
            'market_intelligence': self._get_market_intelligence()
        }
        
        print("✅ UQAVC分析完了")
        return analysis
    
    def _analyze_data_info(self) -> Dict[str, Any]:
        """データ情報分析"""
        close = self.data['close']
        return {
            'period': f"{self.data['timestamp'].iloc[0]} ～ {self.data['timestamp'].iloc[-1]}",
            'samples': len(self.data),
            'price_range': f"${close.min():.2f} - ${close.max():.2f}",
            'volatility': f"{(close.std() / close.mean() * 100):.2f}%"
        }
    
    def _analyze_quantum_metrics(self) -> Dict[str, Any]:
        """量子メトリクス分析"""
        return {
            'coherence_avg': f"{np.mean(self.results.quantum_coherence):.3f}",
            'entanglement_max': f"{np.max(self.results.entanglement_strength):.3f}",
            'tunnel_prob_avg': f"{np.mean(self.results.tunnel_probability):.3f}",
            'current_coherence': f"{self.results.current_coherence:.3f}"
        }
    
    def _analyze_neural_performance(self) -> Dict[str, Any]:
        """神経パフォーマンス分析"""
        return {
            'adaptation_avg': f"{np.mean(self.results.adaptation_score):.3f}",
            'learning_efficiency': f"{np.mean(self.results.learning_rate):.4f}",
            'memory_stability': f"{np.std(self.results.memory_state):.3f}",
            'market_intelligence': f"{self.results.market_intelligence:.3f}"
        }
    
    def _analyze_channel_efficiency(self) -> Dict[str, Any]:
        """チャネル効率分析"""
        close = self.data['close']
        upper = self.results.upper_channel
        lower = self.results.lower_channel
        
        in_channel = np.sum((close >= lower) & (close <= upper))
        efficiency = in_channel / len(close) * 100
        
        return {
            'efficiency': f"{efficiency:.1f}%",
            'avg_width': f"${np.mean(upper - lower):.2f}",
            'width_adaptation': f"{np.std(self.results.dynamic_width) / np.mean(self.results.dynamic_width) * 100:.1f}%"
        }
    
    def _analyze_signal_quality(self) -> Dict[str, Any]:
        """シグナル品質分析"""
        signals = self.results.breakout_signals
        confidence = self.results.entry_confidence
        
        total_signals = np.sum(np.abs(signals))
        high_conf_signals = np.sum(confidence > 0.7)
        
        return {
            'total_signals': int(total_signals),
            'high_confidence': int(high_conf_signals),
            'quality_rate': f"{(high_conf_signals / total_signals * 100 if total_signals > 0 else 0):.1f}%",
            'avg_confidence': f"{np.mean(confidence[confidence > 0]):.3f}" if np.any(confidence > 0) else "0.000"
        }
    
    def _get_market_intelligence(self) -> Dict[str, Any]:
        """市場知能取得"""
        return self.uqavc.get_market_intelligence_report()
    
    def create_4layer_chart(self, save_path: str = None):
        """4層統合チャート作成"""
        if not self.results or self.data is None:
            print("❌ 分析結果がありません")
            return
        
        print("📊 4層統合チャート作成中...")
        
        # データ準備
        dates = pd.to_datetime(self.data['timestamp'])
        close = self.data['close'].values
        
        # スタイル設定
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(4, 1, height_ratios=[3, 2, 2, 2], hspace=0.3)
        
        # カラー設定
        colors = {
            'price': '#00BFFF', 'upper': '#FF6B35', 'lower': '#32CD32',
            'midline': '#FF1493', 'buy': '#00FF00', 'sell': '#FF0000',
            'quantum': '#9370DB', 'neural': '#FFD700'
        }
        
        # 1層: 価格 + UQAVCチャネル
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(dates, close, color=colors['price'], linewidth=2, label='価格')
        ax1.plot(dates, self.results.upper_channel, color=colors['upper'], linewidth=2, label='上側チャネル')
        ax1.plot(dates, self.results.lower_channel, color=colors['lower'], linewidth=2, label='下側チャネル')
        ax1.plot(dates, self.results.midline, color=colors['midline'], linewidth=1.5, label='量子中央線')
        
        # チャネル塗りつぶし
        ax1.fill_between(dates, self.results.upper_channel, self.results.lower_channel, 
                        color='white', alpha=0.1)
        
        # シグナル
        buy_mask = self.results.breakout_signals == 1
        sell_mask = self.results.breakout_signals == -1
        
        if np.any(buy_mask):
            ax1.scatter(dates[buy_mask], close[buy_mask], color=colors['buy'], 
                       marker='^', s=100, label=f'買い({np.sum(buy_mask)})', zorder=5)
        
        if np.any(sell_mask):
            ax1.scatter(dates[sell_mask], close[sell_mask], color=colors['sell'], 
                       marker='v', s=100, label=f'売り({np.sum(sell_mask)})', zorder=5)
        
        ax1.set_title('🌌 Ultra Quantum Adaptive Volatility Channel - 実相場検証', 
                     fontsize=16, fontweight='bold', color='white')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2層: 量子解析
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(dates, self.results.quantum_coherence, color=colors['quantum'], 
                linewidth=2, label='量子コヒーレンス')
        ax2.plot(dates, self.results.entanglement_strength, color='orange', 
                linewidth=2, label='量子もつれ強度')
        ax2.plot(dates, self.results.tunnel_probability, color='red', 
                linewidth=1.5, label='トンネル確率')
        
        ax2.axhline(y=0.7, color='red', linestyle='--', alpha=0.5)
        ax2.axhline(y=0.3, color='blue', linestyle='--', alpha=0.5)
        
        ax2.set_title('⚛️ 量子解析 - コヒーレンス・もつれ・トンネル効果', 
                     fontsize=14, fontweight='bold', color='white')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # 3層: ウェーブレット・トレンド解析
        ax3 = fig.add_subplot(gs[2])
        ax3.plot(dates, self.results.short_term_trend, color='lightgreen', 
                linewidth=1.5, label='短期トレンド')
        ax3.plot(dates, self.results.medium_term_trend, color='skyblue', 
                linewidth=1.5, label='中期トレンド')
        ax3.plot(dates, self.results.long_term_trend, color='orchid', 
                linewidth=1.5, label='長期トレンド')
        
        # フラクタル複雑度（右軸）
        ax3_twin = ax3.twinx()
        ax3_twin.plot(dates, self.results.fractal_complexity, color='brown', 
                     linewidth=2, alpha=0.7, label='フラクタル複雑度')
        ax3_twin.set_ylabel('フラクタル複雑度', color='brown')
        
        ax3.set_title('🌊 ウェーブレット・フラクタル解析', 
                     fontsize=14, fontweight='bold', color='white')
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        # 4層: 神経回路網・予測
        ax4 = fig.add_subplot(gs[3])
        ax4.plot(dates, self.results.adaptation_score, color=colors['neural'], 
                linewidth=2, label='適応スコア')
        ax4.plot(dates, self.results.memory_state, color='cyan', 
                linewidth=1.5, label='記憶状態')
        ax4.plot(dates, self.results.future_direction, color='magenta', 
                linewidth=1.5, label='未来方向予測')
        
        # 信頼度エリア
        ax4.fill_between(dates, 0, self.results.entry_confidence, 
                        color='lightblue', alpha=0.3, label='エントリー信頼度')
        
        ax4.set_title('🧠 神経回路網・予測システム', 
                     fontsize=14, fontweight='bold', color='white')
        ax4.legend(loc='upper left')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(-0.1, 1.1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='black', edgecolor='none')
            print(f"📁 チャート保存: {save_path}")
        
        plt.show()
        print("✅ 4層統合チャート完成")
    
    def print_analysis_report(self, analysis: Dict[str, Any]):
        """分析レポート出力"""
        print("\n" + "="*80)
        print("🌌 UQAVC (Ultra Quantum Adaptive Volatility Channel) 実相場検証レポート")
        print("="*80)
        
        print(f"\n📊 データ情報:")
        for key, value in analysis['data_info'].items():
            print(f"  • {key}: {value}")
        
        print(f"\n⚛️ 量子メトリクス:")
        for key, value in analysis['quantum_metrics'].items():
            print(f"  • {key}: {value}")
        
        print(f"\n🧠 神経パフォーマンス:")
        for key, value in analysis['neural_performance'].items():
            print(f"  • {key}: {value}")
        
        print(f"\n📈 チャネル効率:")
        for key, value in analysis['channel_efficiency'].items():
            print(f"  • {key}: {value}")
        
        print(f"\n🎯 シグナル品質:")
        for key, value in analysis['signal_quality'].items():
            print(f"  • {key}: {value}")
        
        print(f"\n🤖 市場知能:")
        for key, value in analysis['market_intelligence'].items():
            print(f"  • {key}: {value}")
        
        print("\n" + "="*80)


def main():
    """メイン実行"""
    print("🌌 Ultra Quantum Adaptive Volatility Channel - 実相場データ検証")
    print("="*80)
    
    try:
        # 1. テスター初期化
        tester = UQAVCRealMarketTester('config.yaml')
        
        # 2. データ取得
        print("\n📊 Step 1: 市場データ取得")
        data = tester.fetch_market_data()
        
        # 3. UQAVC分析
        print(f"\n🌌 Step 2: UQAVC分析実行 ({len(data)}件)")
        analysis = tester.run_uqavc_analysis(data)
        
        # 4. レポート出力
        print("\n📋 Step 3: 分析レポート")
        tester.print_analysis_report(analysis)
        
        # 5. チャート作成
        print("\n📊 Step 4: 4層統合チャート")
        chart_path = "examples/output/uqavc_real_market_chart.png"
        tester.create_4layer_chart(save_path=chart_path)
        
        print(f"\n🎉 UQAVC実相場検証完了!")
        print(f"📁 チャート: {chart_path}")
        
    except Exception as e:
        import traceback
        print(f"\n❌ エラー: {e}")
        print(f"詳細: {traceback.format_exc()}")


if __name__ == "__main__":
    main() 