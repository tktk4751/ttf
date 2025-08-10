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
from indicators.dual_str_volatility_state import DualSTRVolatilityState
from logger import get_logger


class DualSTRVolatilityAnalyzer:
    """
    Dual STR ボラティリティ状態分析システム
    短期STR vs 長期STR による超低遅延ボラティリティ判定
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
        
        # Dual STRインジケーターの初期化
        self.vol_indicator = DualSTRVolatilityState(
            short_period=20,
            long_period=100,
            lookback_period=50,
            trend_period=10,
            ratio_weight=0.6,
            difference_weight=0.25,
            trend_weight=0.15,
            smoothing=True
        )
        
        self.logger.info("Dual STR Volatility State Analyzer initialized")
    
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
    
    def run_dual_str_analysis(self, show_chart: bool = True) -> dict:
        """Dual STR分析の実行"""
        try:
            self.logger.info("🎯 Dual STR ボラティリティ状態分析を開始...")
            
            # データ読み込み
            data = self.load_market_data()
            
            # Dual STRボラティリティ状態計算
            self.logger.info("⚡ Dual STR による超低遅延分析を実行中...")
            self.logger.info("   - 短期STR（20期間）- 短期ボラティリティ変化")
            self.logger.info("   - 長期STR（100期間）- 長期ボラティリティベースライン")
            self.logger.info("   - STR比率判定 - 短期 > 長期 = 高ボラティリティ")
            self.logger.info("   - 適応的閾値 - 市場状況に応じた動的調整")
            self.logger.info("   - トレンド強度 - 判定の安定性評価")
            
            result = self.vol_indicator.calculate(data)
            
            # 詳細統計の計算
            stats = self._calculate_dual_str_stats(result)
            
            # 結果の表示
            self._display_results(stats)
            
            # チャートの作成
            if show_chart:
                self._create_dual_str_chart(data, result, stats)
            
            return {'data': data, 'result': result, 'stats': stats}
            
        except Exception as e:
            self.logger.error(f"Dual STR分析の実行に失敗: {e}")
            raise
    
    def _calculate_dual_str_stats(self, result) -> dict:
        """Dual STR統計分析"""
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
        
        # STR統計
        current_short_str = self.vol_indicator.get_current_short_str()
        current_long_str = self.vol_indicator.get_current_long_str()
        current_str_ratio = self.vol_indicator.get_current_str_ratio()
        current_str_difference = self.vol_indicator.get_current_str_difference()
        current_trend_strength = self.vol_indicator.get_current_trend_strength()
        
        # STR比率統計
        str_ratios = result.str_ratio[result.str_ratio > 0]
        ratio_above_1 = np.sum(str_ratios > 1.0)
        ratio_above_1_pct = (ratio_above_1 / len(str_ratios) * 100) if len(str_ratios) > 0 else 0
        
        # STR差分統計
        str_differences = result.str_difference
        positive_differences = np.sum(str_differences > 0)
        positive_diff_pct = (positive_differences / len(str_differences) * 100) if len(str_differences) > 0 else 0
        
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
            'current_short_str': current_short_str,
            'current_long_str': current_long_str,
            'current_str_ratio': current_str_ratio,
            'current_str_difference': current_str_difference,
            'current_trend_strength': current_trend_strength,
            'ratio_above_1_percentage': ratio_above_1_pct,
            'positive_difference_percentage': positive_diff_pct,
            'avg_str_ratio': np.mean(str_ratios) if len(str_ratios) > 0 else 0,
            'std_str_ratio': np.std(str_ratios) if len(str_ratios) > 0 else 0,
            'latest_short_str': result.short_str[-1],
            'latest_long_str': result.long_str[-1],
            'latest_str_ratio': result.str_ratio[-1],
            'latest_str_difference': result.str_difference[-1],
            'latest_trend_strength': result.trend_strength[-1]
        }
    
    def _display_results(self, stats: dict) -> None:
        """結果の詳細表示"""
        self.logger.info("\n" + "="*80)
        self.logger.info("🎯 Dual STR ボラティリティ状態分析結果")
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
        
        self.logger.info(f"\n⚡ STR分析:")
        self.logger.info(f"   最新短期STR（20期間）: {stats['latest_short_str']:.6f}")
        self.logger.info(f"   最新長期STR（100期間）: {stats['latest_long_str']:.6f}")
        self.logger.info(f"   最新STR比率: {stats['latest_str_ratio']:.3f}")
        self.logger.info(f"   最新STR差分: {stats['latest_str_difference']:.6f}")
        self.logger.info(f"   平均STR比率: {stats['avg_str_ratio']:.3f}")
        self.logger.info(f"   STR比率標準偏差: {stats['std_str_ratio']:.3f}")
        
        self.logger.info(f"\n📊 STR統計:")
        self.logger.info(f"   STR比率 > 1.0 の期間: {stats['ratio_above_1_percentage']:.1f}%")
        self.logger.info(f"   STR差分 > 0 の期間: {stats['positive_difference_percentage']:.1f}%")
        
        self.logger.info(f"\n🎯 現在の状況:")
        self.logger.info(f"   状態: {stats['latest_state']} ボラティリティ")
        self.logger.info(f"   確率: {stats['latest_probability']:.3f}")
        self.logger.info(f"   トレンド強度: {stats['latest_trend_strength']:.3f}")
        
        # 判定ロジック評価
        str_ratio_check = "✅" if stats['latest_str_ratio'] > 1.0 else "❌"
        str_diff_check = "✅" if stats['latest_str_difference'] > 0 else "❌"
        
        self.logger.info(f"\n🔍 判定ロジック確認:")
        self.logger.info(f"   短期STR > 長期STR: {str_ratio_check} (比率: {stats['latest_str_ratio']:.3f})")
        self.logger.info(f"   STR差分 > 0: {str_diff_check} (差分: {stats['latest_str_difference']:.6f})")
        
        # 実用性評価
        if 20 <= stats['high_volatility_percentage'] <= 40:
            self.logger.info("\n✅ 理想的なボラティリティ分布")
        elif 15 <= stats['high_volatility_percentage'] <= 50:
            self.logger.info("\n✅ 実用的なボラティリティ分布")
        else:
            self.logger.info("\n⚠️ パラメータ調整を検討してください")
    
    def _create_dual_str_chart(self, data, result, stats) -> None:
        """Dual STRチャートの作成"""
        try:
            symbol = self.config.get('binance_data', {}).get('symbol', 'Unknown')
            timeframe = self.config.get('binance_data', {}).get('timeframe', 'Unknown')
            
            fig = plt.figure(figsize=(18, 20))
            gs = fig.add_gridspec(8, 1, height_ratios=[3, 1, 1, 1, 1, 1, 1, 1], hspace=0.3)
            
            # 1. 価格チャート with ボラティリティ状態
            ax1 = fig.add_subplot(gs[0])
            ax1.plot(data.index, data['close'], linewidth=1, color='black', label='Price')
            
            # ボラティリティ状態の背景色（確率ベース）
            for i in range(len(data)):
                alpha = 0.2 + 0.3 * result.probability[i]
                color = 'red' if result.state[i] == 1 else 'lightblue'
                ax1.axvspan(data.index[i], data.index[min(i+1, len(data)-1)], 
                           alpha=alpha, color=color)
            
            title = f'Dual STR Volatility Analysis - {symbol} ({timeframe})\n'
            title += f'High Vol: {stats["high_volatility_percentage"]:.1f}% | STR Ratio > 1.0: {stats["ratio_above_1_percentage"]:.1f}%'
            ax1.set_title(title)
            ax1.set_ylabel('Price')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # 2. ボラティリティ状態バー
            ax2 = fig.add_subplot(gs[1])
            colors = ['red' if s == 1 else 'blue' for s in result.state]
            ax2.bar(data.index, result.state, color=colors, alpha=0.7, width=1)
            ax2.set_title('Dual STR Volatility State (1: High, 0: Low)')
            ax2.set_ylabel('State')
            ax2.set_ylim(-0.1, 1.1)
            ax2.grid(True, alpha=0.3)
            
            # 3. 短期STR vs 長期STR
            ax3 = fig.add_subplot(gs[2])
            ax3.plot(data.index, result.short_str, color='red', linewidth=1.5, label='Short STR (20)')
            ax3.plot(data.index, result.long_str, color='blue', linewidth=1.5, label='Long STR (100)')
            ax3.set_title('Short STR vs Long STR Comparison')
            ax3.set_ylabel('STR Values')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            # 4. STR比率（重要指標）
            ax4 = fig.add_subplot(gs[3])
            ax4.plot(data.index, result.str_ratio, color='purple', linewidth=1.5, label='STR Ratio')
            ax4.axhline(y=1.0, color='black', linestyle='-', alpha=0.5, label='Neutral Line')
            ax4.axhline(y=1.1, color='red', linestyle='--', alpha=0.5, label='High Vol Threshold')
            ax4.axhline(y=0.9, color='blue', linestyle='--', alpha=0.5, label='Low Vol Threshold')
            ax4.set_title('STR Ratio (Short/Long) - Main Signal')
            ax4.set_ylabel('Ratio')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            
            # 5. STR差分
            ax5 = fig.add_subplot(gs[4])
            ax5.plot(data.index, result.str_difference, color='green', linewidth=1.2, label='STR Difference')
            ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3, label='Zero Line')
            # 正の差分を赤、負の差分を青で塗りつぶし
            ax5.fill_between(data.index, 0, result.str_difference, 
                           where=(result.str_difference > 0), color='red', alpha=0.3, label='Positive')
            ax5.fill_between(data.index, 0, result.str_difference, 
                           where=(result.str_difference <= 0), color='blue', alpha=0.3, label='Negative')
            ax5.set_title('STR Difference (Short - Long)')
            ax5.set_ylabel('Difference')
            ax5.grid(True, alpha=0.3)
            ax5.legend()
            
            # 6. 確率とトレンド強度
            ax6 = fig.add_subplot(gs[5])
            ax6.plot(data.index, result.probability, color='orange', linewidth=1.5, label='Probability')
            ax6.axhline(y=0.65, color='red', linestyle='--', alpha=0.5, label='High Threshold')
            ax6.axhline(y=0.35, color='blue', linestyle='--', alpha=0.5, label='Low Threshold')
            ax6_twin = ax6.twinx()
            ax6_twin.plot(data.index, result.trend_strength, color='cyan', linewidth=1.2, alpha=0.8, label='Trend Strength')
            ax6.set_title('Probability & Trend Strength')
            ax6.set_ylabel('Probability')
            ax6_twin.set_ylabel('Trend Strength')
            ax6.set_ylim(0, 1)
            ax6_twin.set_ylim(0, 1)
            ax6.grid(True, alpha=0.3)
            
            # 凡例を統合
            lines1, labels1 = ax6.get_legend_handles_labels()
            lines2, labels2 = ax6_twin.get_legend_handles_labels()
            ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # 7. 生スコアと重み付き成分
            ax7 = fig.add_subplot(gs[6])
            ax7.plot(data.index, result.raw_score, color='darkred', linewidth=1.5, label='Raw Score')
            ax7.axhline(y=0.5, color='black', linestyle='-', alpha=0.3, label='Neutral')
            ax7.axhline(y=0.65, color='red', linestyle='--', alpha=0.5)
            ax7.axhline(y=0.35, color='blue', linestyle='--', alpha=0.5)
            ax7.set_title('Raw Volatility Score')
            ax7.set_ylabel('Score')
            ax7.set_ylim(0, 1)
            ax7.grid(True, alpha=0.3)
            ax7.legend()
            
            # 8. 統計サマリー
            ax8 = fig.add_subplot(gs[7])
            ax8.axis('off')
            
            summary_text = f"""
Dual STR ボラティリティ分析サマリー
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 ボラティリティ分布: 高ボラ {stats['high_volatility_percentage']:.1f}% | 低ボラ {stats['low_volatility_percentage']:.1f}%

⚡ STR分析: 短期STR {stats['latest_short_str']:.6f} | 長期STR {stats['latest_long_str']:.6f}

📊 現在比率: {stats['latest_str_ratio']:.3f} | 差分: {stats['latest_str_difference']:.6f}

🔍 判定: 短期 > 長期 = {"✅" if stats['latest_str_ratio'] > 1.0 else "❌"} | 現在状態: {stats['latest_state']} Vol (確率: {stats['latest_probability']:.3f})

📈 統計: 平均比率 {stats['avg_str_ratio']:.3f} | 比率>1.0期間 {stats['ratio_above_1_percentage']:.1f}% | 変化頻度 {stats['transition_frequency']:.2f}%
            """
            
            ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            
            # 保存
            filename = f"dual_str_volatility_state_{symbol}_{timeframe}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            self.logger.info(f"Dual STRチャートを保存しました: {filename}")
            plt.show()
            
        except Exception as e:
            self.logger.error(f"チャート作成に失敗: {e}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description='Dual STR ボラティリティ状態分析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎯 Dual STR アプローチ:
  ✨ 短期STR（20期間）- 短期ボラティリティ変化検出
  ✨ 長期STR（100期間）- 長期ボラティリティベースライン
  ✨ STR比率判定 - 短期 > 長期 = 高ボラティリティ
  ✨ 適応的閾値 - 市場状況に応じた動的調整
  ✨ トレンド強度 - 判定の安定性・信頼性評価
  ✨ 超低遅延 - STRベースの高速反応

📊 判定ロジック:
  - 基本: 短期STR > 長期STR → 高ボラティリティ
  - 比率: STR比率 > 1.0 → 高ボラティリティ傾向
  - 差分: STR差分 > 0 → 短期的ボラティリティ上昇
  - 安定性: トレンド強度で判定の信頼性を評価
        """
    )
    
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイル')
    parser.add_argument('--no-show', action='store_true', help='チャート非表示')
    parser.add_argument('--sensitive', action='store_true', help='高感度モード（短期重視）')
    parser.add_argument('--stable', action='store_true', help='安定モード（長期重視）')
    parser.add_argument('--balanced', action='store_true', help='バランスモード（均等重み）')
    
    args = parser.parse_args()
    
    try:
        print("🎯 Dual STR ボラティリティ状態分析システム起動中...")
        
        analyzer = DualSTRVolatilityAnalyzer(args.config)
        
        # モード設定
        if args.sensitive:
            analyzer.vol_indicator.ratio_weight = 0.7
            analyzer.vol_indicator.difference_weight = 0.2
            analyzer.vol_indicator.trend_weight = 0.1
            print("⚡ 高感度モードを有効化（短期STR重視）")
        
        if args.stable:
            analyzer.vol_indicator.ratio_weight = 0.4
            analyzer.vol_indicator.difference_weight = 0.2
            analyzer.vol_indicator.trend_weight = 0.4
            print("🛡️ 安定モードを有効化（トレンド強度重視）")
        
        if args.balanced:
            analyzer.vol_indicator.ratio_weight = 0.5
            analyzer.vol_indicator.difference_weight = 0.3
            analyzer.vol_indicator.trend_weight = 0.2
            print("⚖️ バランスモードを有効化（均等重み配分）")
        
        # 分析実行
        results = analyzer.run_dual_str_analysis(show_chart=not args.no_show)
        
        print("\n✅ Dual STR分析が完了しました！")
        
        # 判定ロジック評価
        high_vol_pct = results['stats']['high_volatility_percentage']
        str_ratio = results['stats']['latest_str_ratio']
        str_diff = results['stats']['latest_str_difference']
        
        if 25 <= high_vol_pct <= 35:
            print("🎯 理想的なボラティリティ分布")
        elif 20 <= high_vol_pct <= 40:
            print("✅ 優秀なボラティリティ分布")
        elif 15 <= high_vol_pct <= 50:
            print("📊 実用的なボラティリティ分布")
        else:
            print("⚠️ パラメータ調整を検討してください")
        
        # STRロジック評価
        print(f"\n🔍 Dual STR判定結果:")
        if str_ratio > 1.0 and str_diff > 0:
            print("✅ 短期・長期両方で高ボラティリティを示唆")
        elif str_ratio > 1.0:
            print("📊 STR比率は高ボラティリティを示唆")
        elif str_diff > 0:
            print("📈 STR差分は短期的な上昇を示唆")
        else:
            print("📉 低ボラティリティ環境")
        
        print(f"   STR比率: {str_ratio:.3f} | STR差分: {str_diff:.6f}")
        
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