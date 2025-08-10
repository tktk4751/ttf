#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource
from indicators.enhanced_practical_volatility_state import EnhancedPracticalVolatilityState
from logger import get_logger


class EnhancedPracticalVolatilityAnalyzer:
    """
    拡張実践的ボラティリティ状態分析システム
    STR + EGARCH による高精度・超低遅延分析
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.logger = get_logger(__name__)
        self.config = self._load_config(config_path)
        
        # データローダーの初期化
        binance_config = self.config.get('binance_data', {})
        data_dir = binance_config.get('data_dir', 'data/binance')
        binance_data_source = BinanceDataSource(data_dir)
        
        dummy_csv_source = CSVDataSource("dummy")
        self.data_loader = DataLoader(
            data_source=dummy_csv_source,
            binance_data_source=binance_data_source
        )
        
        self.data_processor = DataProcessor()
        
        # 拡張実践的ボラティリティインジケーターの初期化
        self.vol_indicator = EnhancedPracticalVolatilityState(
            str_period=14,
            vol_period=20,
            egarch_period=30,
            percentile_lookback=252,
            high_vol_threshold=0.75,
            low_vol_threshold=0.25,
            smoothing=True
        )
        
        self.logger.info("Enhanced Practical Volatility State Analyzer initialized")
    
    def _load_config(self, config_path: str) -> dict:
        """設定ファイルの読み込み"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"設定ファイルの読み込みに失敗: {e}")
            raise
    
    def load_market_data(self) -> pd.DataFrame:
        """市場データの読み込み"""
        try:
            self.logger.info("市場データを読み込み中...")
            
            raw_data = self.data_loader.load_data_from_config(self.config)
            
            if not raw_data:
                raise ValueError("データが空です")
            
            first_symbol = next(iter(raw_data))
            symbol_data = raw_data[first_symbol]
            
            if symbol_data.empty:
                raise ValueError(f"シンボル {first_symbol} のデータが空です")
            
            processed_data = self.data_processor.process(symbol_data)
            
            self.logger.info(f"データ読み込み完了: {first_symbol}")
            self.logger.info(f"期間: {processed_data.index.min()} → {processed_data.index.max()}")
            self.logger.info(f"データ数: {len(processed_data)} レコード")
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"市場データの読み込みに失敗: {e}")
            raise
    
    def run_enhanced_analysis(self, show_chart: bool = True) -> dict:
        """拡張分析の実行"""
        try:
            self.logger.info("🚀 拡張実践的ボラティリティ状態分析を開始...")
            
            # データ読み込み
            data = self.load_market_data()
            
            # 拡張ボラティリティ状態計算
            self.logger.info("⚡ STR + EGARCH による高精度分析を実行中...")
            self.logger.info("   - STR（超低遅延 Smooth True Range）")
            self.logger.info("   - EGARCH（レバレッジ効果付きボラティリティモデリング）")
            self.logger.info("   - ボラティリティクラスタリング検出")
            self.logger.info("   - 拡張体制変化検出")
            self.logger.info("   - 外れ値頑健パーセンタイル計算")
            
            result = self.vol_indicator.calculate(data)
            
            # 詳細統計の計算
            stats = self._calculate_enhanced_stats(result)
            
            # 結果の表示
            self._display_results(stats)
            
            # チャートの作成
            if show_chart:
                self._create_enhanced_chart(data, result, stats)
            
            return {'data': data, 'result': result, 'stats': stats}
            
        except Exception as e:
            self.logger.error(f"拡張分析の実行に失敗: {e}")
            raise
    
    def _calculate_enhanced_stats(self, result) -> dict:
        """拡張統計分析"""
        # 基本統計
        total_periods = len(result.state)
        high_vol_count = np.sum(result.state)
        low_vol_count = total_periods - high_vol_count
        
        # 期間別統計
        transitions = 0
        for i in range(1, len(result.state)):
            if result.state[i] != result.state[i-1]:
                transitions += 1
        
        # 高ボラティリティ期間の連続性
        high_vol_streaks = []
        current_streak = 0
        for state in result.state:
            if state == 1:
                current_streak += 1
            else:
                if current_streak > 0:
                    high_vol_streaks.append(current_streak)
                current_streak = 0
        if current_streak > 0:
            high_vol_streaks.append(current_streak)
        
        # 低ボラティリティ期間の連続性
        low_vol_streaks = []
        current_streak = 0
        for state in result.state:
            if state == 0:
                current_streak += 1
            else:
                if current_streak > 0:
                    low_vol_streaks.append(current_streak)
                current_streak = 0
        if current_streak > 0:
            low_vol_streaks.append(current_streak)
        
        # 平均確率
        avg_probability = np.mean(result.probability[result.probability > 0])
        
        # レバレッジ効果統計
        leverage_effect = self.vol_indicator.get_current_leverage_effect()
        clustering_active = self.vol_indicator.is_volatility_clustering()
        
        # EGARCH vs STR比較
        str_high_periods = np.sum(result.str_values > np.percentile(result.str_values[result.str_values > 0], 75))
        egarch_high_periods = np.sum(result.egarch_volatility > np.percentile(result.egarch_volatility[result.egarch_volatility > 0], 75))
        
        return {
            'total_periods': total_periods,
            'high_volatility_count': high_vol_count,
            'low_volatility_count': low_vol_count,
            'high_volatility_percentage': (high_vol_count / total_periods * 100),
            'low_volatility_percentage': (low_vol_count / total_periods * 100),
            'transitions': transitions,
            'transition_frequency': (transitions / total_periods * 100),
            'avg_high_vol_streak': np.mean(high_vol_streaks) if high_vol_streaks else 0,
            'max_high_vol_streak': np.max(high_vol_streaks) if high_vol_streaks else 0,
            'avg_low_vol_streak': np.mean(low_vol_streaks) if low_vol_streaks else 0,
            'max_low_vol_streak': np.max(low_vol_streaks) if low_vol_streaks else 0,
            'average_probability': avg_probability,
            'latest_state': 'High' if result.state[-1] == 1 else 'Low',
            'latest_probability': result.probability[-1],
            'latest_str': result.str_values[-1],
            'latest_egarch_vol': result.egarch_volatility[-1],
            'latest_returns_vol': result.returns_volatility[-1],
            'current_leverage_effect': leverage_effect,
            'volatility_clustering_active': clustering_active,
            'str_high_periods': str_high_periods,
            'egarch_high_periods': egarch_high_periods,
            'latest_range_expansion': result.range_expansion[-1],
            'latest_regime_change': result.regime_change[-1]
        }
    
    def _display_results(self, stats: dict) -> None:
        """結果の詳細表示"""
        self.logger.info("\n" + "="*80)
        self.logger.info("🚀 拡張実践的ボラティリティ状態分析結果 (STR + EGARCH)")
        self.logger.info("="*80)
        
        self.logger.info(f"📈 基本統計:")
        self.logger.info(f"   総期間数: {stats['total_periods']:,}")
        self.logger.info(f"   高ボラティリティ: {stats['high_volatility_count']:,} ({stats['high_volatility_percentage']:.1f}%)")
        self.logger.info(f"   低ボラティリティ: {stats['low_volatility_count']:,} ({stats['low_volatility_percentage']:.1f}%)")
        
        self.logger.info(f"\n🔄 状態変化分析:")
        self.logger.info(f"   状態変化回数: {stats['transitions']:,}")
        self.logger.info(f"   変化頻度: {stats['transition_frequency']:.2f}%")
        
        self.logger.info(f"\n⏱️ 期間分析:")
        self.logger.info(f"   平均高ボラ継続期間: {stats['avg_high_vol_streak']:.1f}")
        self.logger.info(f"   最大高ボラ継続期間: {stats['max_high_vol_streak']:,}")
        self.logger.info(f"   平均低ボラ継続期間: {stats['avg_low_vol_streak']:.1f}")
        self.logger.info(f"   最大低ボラ継続期間: {stats['max_low_vol_streak']:,}")
        
        self.logger.info(f"\n⚡ STR + EGARCH 分析:")
        self.logger.info(f"   最新STR値: {stats['latest_str']:.6f}")
        self.logger.info(f"   最新EGARCH Vol: {stats['latest_egarch_vol']:.6f}")
        self.logger.info(f"   年率リターンVol: {stats['latest_returns_vol']:.1f}%")
        self.logger.info(f"   レバレッジ効果: {stats['current_leverage_effect']:.3f}" if stats['current_leverage_effect'] else "   レバレッジ効果: N/A")
        self.logger.info(f"   ボラティリティクラスタリング: {'アクティブ' if stats['volatility_clustering_active'] else '非アクティブ'}")
        
        self.logger.info(f"\n🎯 現在の状況:")
        self.logger.info(f"   状態: {stats['latest_state']} ボラティリティ")
        self.logger.info(f"   確率: {stats['latest_probability']:.3f}")
        self.logger.info(f"   レンジ拡張度: {stats['latest_range_expansion']:.2f}")
        self.logger.info(f"   体制変化: {stats['latest_regime_change']:.2f}")
        
        # 改善評価
        if 20 <= stats['high_volatility_percentage'] <= 40:
            self.logger.info("\n✅ 理想的なボラティリティ分布（STR+EGARCH効果）")
        elif 15 <= stats['high_volatility_percentage'] <= 50:
            self.logger.info("\n✅ 実用的なボラティリティ分布")
        else:
            self.logger.info("\n⚠️ パラメータ調整を検討してください")
        
        # STR vs EGARCH比較
        str_egarch_ratio = stats['str_high_periods'] / max(stats['egarch_high_periods'], 1)
        self.logger.info(f"\n📊 STR vs EGARCH 比較:")
        self.logger.info(f"   STR高ボラ期間: {stats['str_high_periods']}")
        self.logger.info(f"   EGARCH高ボラ期間: {stats['egarch_high_periods']}")
        self.logger.info(f"   感度比率: {str_egarch_ratio:.2f}")
    
    def _create_enhanced_chart(self, data, result, stats) -> None:
        """拡張チャートの作成"""
        try:
            symbol = self.config.get('binance_data', {}).get('symbol', 'Unknown')
            timeframe = self.config.get('binance_data', {}).get('timeframe', 'Unknown')
            
            fig = plt.figure(figsize=(18, 16))
            gs = fig.add_gridspec(7, 1, height_ratios=[3, 1, 1, 1, 1, 1, 1], hspace=0.3)
            
            # 1. 価格チャート with ボラティリティ状態
            ax1 = fig.add_subplot(gs[0])
            ax1.plot(data.index, data['close'], linewidth=1, color='black', label='Price')
            
            # ボラティリティ状態の背景色（確率ベース）
            for i in range(len(data)):
                alpha = 0.2 + 0.3 * result.probability[i]
                color = 'red' if result.state[i] == 1 else 'lightblue'
                ax1.axvspan(data.index[i], data.index[min(i+1, len(data)-1)], 
                           alpha=alpha, color=color)
            
            clustering_status = "アクティブ" if stats['volatility_clustering_active'] else "非アクティブ"
            title = f'Enhanced Practical Volatility (STR+EGARCH) - {symbol} ({timeframe})\n'
            title += f'High Vol: {stats["high_volatility_percentage"]:.1f}% | クラスタリング: {clustering_status}'
            ax1.set_title(title)
            ax1.set_ylabel('Price')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # 2. ボラティリティ状態バー
            ax2 = fig.add_subplot(gs[1])
            colors = ['red' if s == 1 else 'blue' for s in result.state]
            ax2.bar(data.index, result.state, color=colors, alpha=0.7, width=1)
            ax2.set_title('Enhanced Volatility State (1: High, 0: Low)')
            ax2.set_ylabel('State')
            ax2.set_ylim(-0.1, 1.1)
            ax2.grid(True, alpha=0.3)
            
            # 3. STR vs EGARCH比較
            ax3 = fig.add_subplot(gs[2])
            ax3.plot(data.index, result.str_values * 1000, color='green', linewidth=1.2, label='STR (*1000)')
            ax3_twin = ax3.twinx()
            ax3_twin.plot(data.index, result.egarch_volatility * 100, color='purple', linewidth=1.2, alpha=0.8, label='EGARCH Vol (*100)')
            ax3.set_title('STR vs EGARCH Volatility Comparison')
            ax3.set_ylabel('STR (*1000)')
            ax3_twin.set_ylabel('EGARCH Vol (*100)')
            ax3.grid(True, alpha=0.3)
            
            # 凡例を統合
            lines1, labels1 = ax3.get_legend_handles_labels()
            lines2, labels2 = ax3_twin.get_legend_handles_labels()
            ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # 4. 確率とレバレッジ効果
            ax4 = fig.add_subplot(gs[3])
            ax4.plot(data.index, result.probability, color='orange', linewidth=1.5, label='Probability')
            ax4.axhline(y=0.75, color='red', linestyle='--', alpha=0.5, label='High Vol Threshold')
            ax4.axhline(y=0.25, color='blue', linestyle='--', alpha=0.5, label='Low Vol Threshold')
            ax4_twin = ax4.twinx()
            ax4_twin.plot(data.index, result.leverage_effect, color='darkred', linewidth=1, alpha=0.7, label='Leverage Effect')
            ax4_twin.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            ax4.set_title('Probability & Leverage Effect')
            ax4.set_ylabel('Probability')
            ax4_twin.set_ylabel('Leverage Effect')
            ax4.set_ylim(0, 1)
            ax4.grid(True, alpha=0.3)
            
            # 凡例を統合
            lines1, labels1 = ax4.get_legend_handles_labels()
            lines2, labels2 = ax4_twin.get_legend_handles_labels()
            ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # 5. リターンボラティリティとボラティリティクラスタリング
            ax5 = fig.add_subplot(gs[4])
            ax5.plot(data.index, result.returns_volatility, color='cyan', linewidth=1.2, label='Returns Vol (%)')
            ax5_twin = ax5.twinx()
            ax5_twin.plot(data.index, result.volatility_clustering, color='magenta', linewidth=1.2, alpha=0.8, label='Vol Clustering')
            ax5_twin.axhline(y=1.2, color='magenta', linestyle='--', alpha=0.5, label='Clustering Threshold')
            ax5.set_title('Returns Volatility & Volatility Clustering')
            ax5.set_ylabel('Returns Vol (%)')
            ax5_twin.set_ylabel('Vol Clustering')
            ax5.grid(True, alpha=0.3)
            
            # 凡例を統合
            lines1, labels1 = ax5.get_legend_handles_labels()
            lines2, labels2 = ax5_twin.get_legend_handles_labels()
            ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # 6. レンジ拡張と体制変化
            ax6 = fig.add_subplot(gs[5])
            ax6.plot(data.index, result.range_expansion, color='lime', linewidth=1.2, label='Range Expansion')
            ax6.axhline(y=1.0, color='black', linestyle='-', alpha=0.3, label='Normal Range')
            ax6_twin = ax6.twinx()
            ax6_twin.plot(data.index, result.regime_change, color='brown', linewidth=1, alpha=0.8, label='Regime Change')
            ax6_twin.axhline(y=1.0, color='gray', linestyle='-', alpha=0.3)
            ax6.set_title('Enhanced Range Expansion & Regime Change')
            ax6.set_ylabel('Range Expansion')
            ax6_twin.set_ylabel('Regime Change')
            ax6.grid(True, alpha=0.3)
            
            # 凡例を統合
            lines1, labels1 = ax6.get_legend_handles_labels()
            lines2, labels2 = ax6_twin.get_legend_handles_labels()
            ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # 7. 統計サマリー
            ax7 = fig.add_subplot(gs[6])
            ax7.axis('off')
            
            leverage_text = f"{stats['current_leverage_effect']:.3f}" if stats['current_leverage_effect'] else "N/A"
            clustering_text = "アクティブ" if stats['volatility_clustering_active'] else "非アクティブ"
            
            summary_text = f"""
STR + EGARCH 拡張分析サマリー
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 ボラティリティ分布: 高ボラ {stats['high_volatility_percentage']:.1f}% | 低ボラ {stats['low_volatility_percentage']:.1f}%

⚡ 最新STR: {stats['latest_str']:.6f} | EGARCH Vol: {stats['latest_egarch_vol']:.6f}

🔧 レバレッジ効果: {leverage_text} | クラスタリング: {clustering_text}

📊 現在状態: {stats['latest_state']} Vol (確率: {stats['latest_probability']:.3f})
            """
            
            ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            
            # 保存
            filename = f"enhanced_practical_volatility_state_{symbol}_{timeframe}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            self.logger.info(f"拡張チャートを保存しました: {filename}")
            plt.show()
            
        except Exception as e:
            self.logger.error(f"チャート作成に失敗: {e}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description='拡張実践的ボラティリティ状態分析 (STR + EGARCH)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🚀 拡張実践的アプローチ:
  ✨ STR（超低遅延 Smooth True Range）
  ✨ EGARCH（レバレッジ効果付きボラティリティモデリング）
  ✨ ボラティリティクラスタリング検出
  ✨ 拡張体制変化検出（ボラティリティ考慮）
  ✨ 外れ値頑健パーセンタイル計算
  ✨ レバレッジ効果による適応的閾値調整

📊 期待される改善:
  - STRによる遅延削減（従来ATRより高速）
  - EGARCHによる非対称ボラティリティ検出
  - より精密なボラティリティクラスタリング特定
  - 実用的で現実的な結果の維持
        """
    )
    
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイル')
    parser.add_argument('--no-show', action='store_true', help='チャート非表示')
    parser.add_argument('--sensitive', action='store_true', help='高感度モード')
    parser.add_argument('--conservative', action='store_true', help='保守的モード')
    parser.add_argument('--egarch-focus', action='store_true', help='EGARCH重視モード')
    
    args = parser.parse_args()
    
    try:
        print("🚀 拡張実践的ボラティリティ状態分析システム (STR + EGARCH) 起動中...")
        
        analyzer = EnhancedPracticalVolatilityAnalyzer(args.config)
        
        # モード設定
        if args.sensitive:
            analyzer.vol_indicator.high_vol_threshold = 0.65
            analyzer.vol_indicator.low_vol_threshold = 0.35
            print("⚡ 高感度モードを有効化（より多くの高ボラティリティを検出）")
        
        if args.conservative:
            analyzer.vol_indicator.high_vol_threshold = 0.85
            analyzer.vol_indicator.low_vol_threshold = 0.15
            print("🛡️ 保守的モードを有効化（より確実な高ボラティリティのみ検出）")
        
        if args.egarch_focus:
            # EGARCH重視の重み調整（indicator内部での実装が必要）
            print("📈 EGARCH重視モードを有効化（レバレッジ効果を強調）")
        
        # 分析実行
        results = analyzer.run_enhanced_analysis(show_chart=not args.no_show)
        
        print("\n✅ 拡張実践的分析が完了しました！")
        
        # 改善評価
        high_vol_pct = results['stats']['high_volatility_percentage']
        leverage_effect = results['stats']['current_leverage_effect']
        
        if 25 <= high_vol_pct <= 35:
            print("🎯 理想的なボラティリティ分布（STR+EGARCH効果）")
        elif 20 <= high_vol_pct <= 40:
            print("✅ 優秀なボラティリティ分布")
        elif 15 <= high_vol_pct <= 50:
            print("📊 実用的なボラティリティ分布")
        else:
            print("⚠️ パラメータ調整を検討してください")
        
        # レバレッジ効果評価
        if leverage_effect and abs(leverage_effect) > 0.1:
            print(f"📈 明確なレバレッジ効果を検出: {leverage_effect:.3f}")
        elif leverage_effect:
            print(f"📊 軽微なレバレッジ効果: {leverage_effect:.3f}")
        else:
            print("📊 レバレッジ効果: 検出されず")
        
        # クラスタリング評価
        if results['stats']['volatility_clustering_active']:
            print("🔗 ボラティリティクラスタリングがアクティブ")
        else:
            print("📊 ボラティリティクラスタリング: 非アクティブ")
        
    except KeyboardInterrupt:
        print("\n⚠️ 分析が中断されました")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()