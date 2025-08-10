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
from indicators.str_kaufman_efficiency_volatility_state import STRKaufmanEfficiencyVolatilityState
from logger import get_logger


class STRKaufmanEfficiencyVolatilityAnalyzer:
    """
    STR + カウフマン効率比 ボラティリティ状態分析システム
    STRをソースとした効率比による高精度ボラティリティ判定
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
        
        # STR + カウフマン効率比インジケーターの初期化
        self.vol_indicator = STRKaufmanEfficiencyVolatilityState(
            str_period=14,
            efficiency_period=10,
            trend_period=5,
            lookback_period=100,
            efficiency_weight=0.6,
            trend_weight=0.25,
            strength_weight=0.15,
            base_threshold=0.5,
            smoothing=True
        )
        
        self.logger.info("STR + Kaufman Efficiency Volatility State Analyzer initialized")
    
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
    
    def run_kaufman_efficiency_analysis(self, show_chart: bool = True) -> dict:
        """STR + カウフマン効率比分析の実行"""
        try:
            self.logger.info("🎯 STR + カウフマン効率比 ボラティリティ状態分析を開始...")
            
            # データ読み込み
            data = self.load_market_data()
            
            # STR + カウフマン効率比ボラティリティ状態計算
            self.logger.info("⚡ STR + カウフマン効率比による分析を実行中...")
            self.logger.info("   - STRをソースとしたカウフマン効率比計算")
            self.logger.info("   - 効率比 > 0.5 = 高ボラティリティ（強いトレンド）")
            self.logger.info("   - 効率比 <= 0.5 = 低ボラティリティ（ノイズ優勢）")
            self.logger.info("   - 効率比トレンドによる平滑化")
            self.logger.info("   - シグナル強度による信頼性評価")
            
            result = self.vol_indicator.calculate(data)
            
            # 詳細統計の計算
            stats = self._calculate_efficiency_stats(result)
            
            # 結果の表示
            self._display_results(stats)
            
            # チャートの作成
            if show_chart:
                self._create_efficiency_chart(data, result, stats)
            
            return {'data': data, 'result': result, 'stats': stats}
            
        except Exception as e:
            self.logger.error(f"STR + カウフマン効率比分析の実行に失敗: {e}")
            raise
    
    def _calculate_efficiency_stats(self, result) -> dict:
        """STR + カウフマン効率比統計分析"""
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
        
        # カウフマン効率比統計
        efficiency_stats = self.vol_indicator.get_efficiency_statistics()
        current_efficiency = self.vol_indicator.get_current_efficiency_ratio()
        current_trend = self.vol_indicator.get_current_efficiency_trend()
        current_strength = self.vol_indicator.get_current_signal_strength()
        
        # 効率比分布
        valid_efficiency = result.kaufman_efficiency[result.kaufman_efficiency > 0]
        above_05 = np.sum(valid_efficiency > 0.5)
        above_05_pct = (above_05 / len(valid_efficiency) * 100) if len(valid_efficiency) > 0 else 0
        
        above_07 = np.sum(valid_efficiency > 0.7)
        above_07_pct = (above_07 / len(valid_efficiency) * 100) if len(valid_efficiency) > 0 else 0
        
        below_03 = np.sum(valid_efficiency < 0.3)
        below_03_pct = (below_03 / len(valid_efficiency) * 100) if len(valid_efficiency) > 0 else 0
        
        # 方向性 vs ボラティリティ比率
        directional_avg = np.mean(result.directional_movement[result.directional_movement > 0])
        volatility_avg = np.mean(result.volatility_movement[result.volatility_movement > 0])
        direction_vol_ratio = directional_avg / volatility_avg if volatility_avg > 0 else 0
        
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
            'current_efficiency': current_efficiency,
            'current_trend': current_trend,
            'current_strength': current_strength,
            'efficiency_stats': efficiency_stats,
            'above_05_percentage': above_05_pct,
            'above_07_percentage': above_07_pct,
            'below_03_percentage': below_03_pct,
            'direction_vol_ratio': direction_vol_ratio,
            'latest_str': result.str_values[-1],
            'latest_efficiency': result.kaufman_efficiency[-1],
            'latest_trend': result.efficiency_trend[-1],
            'latest_strength': result.signal_strength[-1],
            'latest_directional': result.directional_movement[-1],
            'latest_volatility': result.volatility_movement[-1]
        }
    
    def _display_results(self, stats: dict) -> None:
        """結果の詳細表示"""
        self.logger.info("\n" + "="*80)
        self.logger.info("🎯 STR + カウフマン効率比 ボラティリティ状態分析結果")
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
        
        self.logger.info(f"\n⚡ カウフマン効率比分析:")
        self.logger.info(f"   最新STR値: {stats['latest_str']:.6f}")
        self.logger.info(f"   最新効率比: {stats['latest_efficiency']:.3f}")
        self.logger.info(f"   最新効率比トレンド: {stats['latest_trend']:.3f}")
        self.logger.info(f"   最新シグナル強度: {stats['latest_strength']:.3f}")
        
        if stats['efficiency_stats']:
            eff_stats = stats['efficiency_stats']
            self.logger.info(f"\n📊 効率比統計:")
            self.logger.info(f"   平均効率比: {eff_stats['mean']:.3f}")
            self.logger.info(f"   効率比標準偏差: {eff_stats['std']:.3f}")
            self.logger.info(f"   効率比範囲: {eff_stats['min']:.3f} - {eff_stats['max']:.3f}")
            self.logger.info(f"   効率比 > 0.5の期間: {eff_stats['above_threshold']:.1f}%")
        
        self.logger.info(f"\n📈 効率比分布:")
        self.logger.info(f"   効率比 > 0.5: {stats['above_05_percentage']:.1f}%")
        self.logger.info(f"   効率比 > 0.7: {stats['above_07_percentage']:.1f}%")
        self.logger.info(f"   効率比 < 0.3: {stats['below_03_percentage']:.1f}%")
        
        self.logger.info(f"\n🔍 方向性分析:")
        self.logger.info(f"   最新方向性動き: {stats['latest_directional']:.6f}")
        self.logger.info(f"   最新ボラティリティ動き: {stats['latest_volatility']:.6f}")
        self.logger.info(f"   方向性/ボラティリティ比率: {stats['direction_vol_ratio']:.3f}")
        
        self.logger.info(f"\n🎯 現在の状況:")
        self.logger.info(f"   状態: {stats['latest_state']} ボラティリティ")
        self.logger.info(f"   確率: {stats['latest_probability']:.3f}")
        
        # 効率比判定ロジック評価
        efficiency_check = "✅" if stats['latest_efficiency'] > 0.5 else "❌"
        trend_check = "✅" if stats['latest_trend'] > 0.5 else "❌"
        
        self.logger.info(f"\n🔍 判定ロジック確認:")
        self.logger.info(f"   効率比 > 0.5: {efficiency_check} (値: {stats['latest_efficiency']:.3f})")
        self.logger.info(f"   トレンド > 0.5: {trend_check} (値: {stats['latest_trend']:.3f})")
        
        # 実用性評価
        if 20 <= stats['high_volatility_percentage'] <= 40:
            self.logger.info("\n✅ 理想的なボラティリティ分布")
        elif 15 <= stats['high_volatility_percentage'] <= 50:
            self.logger.info("\n✅ 実用的なボラティリティ分布")
        else:
            self.logger.info("\n⚠️ パラメータ調整を検討してください")
    
    def _create_efficiency_chart(self, data, result, stats) -> None:
        """STR + カウフマン効率比チャートの作成"""
        try:
            symbol = self.config.get('binance_data', {}).get('symbol', 'Unknown')
            timeframe = self.config.get('binance_data', {}).get('timeframe', 'Unknown')
            
            fig = plt.figure(figsize=(18, 22))
            gs = fig.add_gridspec(9, 1, height_ratios=[3, 1, 1, 1, 1, 1, 1, 1, 1], hspace=0.3)
            
            # 1. 価格チャート with ボラティリティ状態
            ax1 = fig.add_subplot(gs[0])
            ax1.plot(data.index, data['close'], linewidth=1, color='black', label='Price')
            
            # ボラティリティ状態の背景色（確率ベース）
            for i in range(len(data)):
                alpha = 0.2 + 0.3 * result.probability[i]
                color = 'red' if result.state[i] == 1 else 'lightblue'
                ax1.axvspan(data.index[i], data.index[min(i+1, len(data)-1)], 
                           alpha=alpha, color=color)
            
            title = f'STR + Kaufman Efficiency Volatility Analysis - {symbol} ({timeframe})\n'
            title += f'High Vol: {stats["high_volatility_percentage"]:.1f}% | Efficiency > 0.5: {stats["above_05_percentage"]:.1f}%'
            ax1.set_title(title)
            ax1.set_ylabel('Price')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # 2. ボラティリティ状態バー
            ax2 = fig.add_subplot(gs[1])
            colors = ['red' if s == 1 else 'blue' for s in result.state]
            ax2.bar(data.index, result.state, color=colors, alpha=0.7, width=1)
            ax2.set_title('STR + Kaufman Efficiency Volatility State (1: High, 0: Low)')
            ax2.set_ylabel('State')
            ax2.set_ylim(-0.1, 1.1)
            ax2.grid(True, alpha=0.3)
            
            # 3. STR値
            ax3 = fig.add_subplot(gs[2])
            ax3.plot(data.index, result.str_values, color='green', linewidth=1.2, label='STR Values')
            ax3.set_title('STR (Source for Kaufman Efficiency)')
            ax3.set_ylabel('STR')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            # 4. カウフマン効率比（メインシグナル）
            ax4 = fig.add_subplot(gs[3])
            ax4.plot(data.index, result.kaufman_efficiency, color='purple', linewidth=1.5, label='Kaufman Efficiency')
            ax4.axhline(y=0.5, color='black', linestyle='-', alpha=0.7, label='Threshold (0.5)')
            ax4.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='High Efficiency')
            ax4.axhline(y=0.3, color='blue', linestyle='--', alpha=0.5, label='Low Efficiency')
            # 効率比に基づく背景色
            ax4.fill_between(data.index, 0, 1, where=(result.kaufman_efficiency > 0.5), 
                           color='red', alpha=0.1, label='High Vol Zone')
            ax4.fill_between(data.index, 0, 1, where=(result.kaufman_efficiency <= 0.5), 
                           color='blue', alpha=0.1, label='Low Vol Zone')
            ax4.set_title('Kaufman Efficiency Ratio - Main Signal')
            ax4.set_ylabel('Efficiency')
            ax4.set_ylim(0, 1)
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            
            # 5. 効率比トレンド
            ax5 = fig.add_subplot(gs[4])
            ax5.plot(data.index, result.efficiency_trend, color='orange', linewidth=1.3, label='Efficiency Trend')
            ax5.axhline(y=0.5, color='black', linestyle='-', alpha=0.5)
            ax5.set_title('Efficiency Trend (Smoothed)')
            ax5.set_ylabel('Trend')
            ax5.set_ylim(0, 1)
            ax5.grid(True, alpha=0.3)
            ax5.legend()
            
            # 6. 方向性動き vs ボラティリティ動き
            ax6 = fig.add_subplot(gs[5])
            ax6.plot(data.index, result.directional_movement, color='cyan', linewidth=1.2, alpha=0.8, label='Directional Movement')
            ax6_twin = ax6.twinx()
            ax6_twin.plot(data.index, result.volatility_movement, color='magenta', linewidth=1, alpha=0.7, label='Volatility Movement')
            ax6.set_title('Directional vs Volatility Movement')
            ax6.set_ylabel('Directional')
            ax6_twin.set_ylabel('Volatility')
            ax6.grid(True, alpha=0.3)
            
            # 凡例を統合
            lines1, labels1 = ax6.get_legend_handles_labels()
            lines2, labels2 = ax6_twin.get_legend_handles_labels()
            ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # 7. シグナル強度と確率
            ax7 = fig.add_subplot(gs[6])
            ax7.plot(data.index, result.signal_strength, color='darkgreen', linewidth=1.3, label='Signal Strength')
            ax7_twin = ax7.twinx()
            ax7_twin.plot(data.index, result.probability, color='red', linewidth=1.2, alpha=0.8, label='Probability')
            ax7_twin.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            ax7.set_title('Signal Strength & Probability')
            ax7.set_ylabel('Strength')
            ax7_twin.set_ylabel('Probability')
            ax7.set_ylim(0, 1)
            ax7_twin.set_ylim(0, 1)
            ax7.grid(True, alpha=0.3)
            
            # 凡例を統合
            lines1, labels1 = ax7.get_legend_handles_labels()
            lines2, labels2 = ax7_twin.get_legend_handles_labels()
            ax7.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # 8. 生スコア
            ax8 = fig.add_subplot(gs[7])
            ax8.plot(data.index, result.raw_score, color='darkred', linewidth=1.5, label='Raw Score')
            ax8.axhline(y=0.5, color='black', linestyle='-', alpha=0.5, label='Neutral')
            ax8.axhline(y=0.6, color='red', linestyle='--', alpha=0.5)
            ax8.axhline(y=0.4, color='blue', linestyle='--', alpha=0.5)
            ax8.set_title('Raw Volatility Score')
            ax8.set_ylabel('Score')
            ax8.set_ylim(0, 1)
            ax8.grid(True, alpha=0.3)
            ax8.legend()
            
            # 9. 統計サマリー
            ax9 = fig.add_subplot(gs[8])
            ax9.axis('off')
            
            summary_text = f"""
STR + カウフマン効率比 ボラティリティ分析サマリー
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 ボラティリティ分布: 高ボラ {stats['high_volatility_percentage']:.1f}% | 低ボラ {stats['low_volatility_percentage']:.1f}%

⚡ 効率比分析: 現在 {stats['latest_efficiency']:.3f} | トレンド {stats['latest_trend']:.3f} | 強度 {stats['latest_strength']:.3f}

📊 効率比統計: >0.5期間 {stats['above_05_percentage']:.1f}% | >0.7期間 {stats['above_07_percentage']:.1f}% | <0.3期間 {stats['below_03_percentage']:.1f}%

🔍 判定: 効率比>0.5 = {"✅" if stats['latest_efficiency'] > 0.5 else "❌"} | 現在状態: {stats['latest_state']} Vol (確率: {stats['latest_probability']:.3f})

📈 動き: 方向性 {stats['latest_directional']:.6f} | ボラティリティ {stats['latest_volatility']:.6f} | 比率 {stats['direction_vol_ratio']:.3f}
            """
            
            ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
            
            # 保存
            filename = f"str_kaufman_efficiency_volatility_state_{symbol}_{timeframe}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            self.logger.info(f"STR + カウフマン効率比チャートを保存しました: {filename}")
            plt.show()
            
        except Exception as e:
            self.logger.error(f"チャート作成に失敗: {e}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description='STR + カウフマン効率比 ボラティリティ状態分析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎯 STR + カウフマン効率比 アプローチ:
  ✨ STRをソースとしたカウフマン効率比計算
  ✨ 効率比 > 0.5 = 高ボラティリティ（強いトレンド性）
  ✨ 効率比 <= 0.5 = 低ボラティリティ（ノイズ優勢）
  ✨ 効率比トレンドによる平滑化
  ✨ シグナル強度による信頼性評価
  ✨ 適応的閾値で市場適応

📊 カウフマン効率比の特徴:
  - 方向性動き / ボラティリティ動き の比率
  - 0に近い: ノイズが多い（非効率、横ばい）
  - 1に近い: トレンドが強い（効率的、方向性明確）
  
🔍 判定ロジック:
  - 基本: 効率比 > 0.5 → 高ボラティリティ
  - トレンド: 効率比の移動平均による平滑化
  - 強度: STRレベル × 効率比 × 安定性
        """
    )
    
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイル')
    parser.add_argument('--no-show', action='store_true', help='チャート非表示')
    parser.add_argument('--sensitive', action='store_true', help='高感度モード（閾値0.4）')
    parser.add_argument('--conservative', action='store_true', help='保守的モード（閾値0.6）')
    parser.add_argument('--trend-focus', action='store_true', help='トレンド重視モード')
    
    args = parser.parse_args()
    
    try:
        print("🎯 STR + カウフマン効率比 ボラティリティ状態分析システム起動中...")
        
        analyzer = STRKaufmanEfficiencyVolatilityAnalyzer(args.config)
        
        # モード設定
        if args.sensitive:
            analyzer.vol_indicator.base_threshold = 0.4
            print("⚡ 高感度モードを有効化（閾値: 0.4）")
        
        if args.conservative:
            analyzer.vol_indicator.base_threshold = 0.6
            print("🛡️ 保守的モードを有効化（閾値: 0.6）")
        
        if args.trend_focus:
            analyzer.vol_indicator.efficiency_weight = 0.4
            analyzer.vol_indicator.trend_weight = 0.4
            analyzer.vol_indicator.strength_weight = 0.2
            print("📈 トレンド重視モードを有効化")
        
        # 分析実行
        results = analyzer.run_kaufman_efficiency_analysis(show_chart=not args.no_show)
        
        print("\n✅ STR + カウフマン効率比分析が完了しました！")
        
        # 判定ロジック評価
        high_vol_pct = results['stats']['high_volatility_percentage']
        efficiency = results['stats']['latest_efficiency']
        trend = results['stats']['latest_trend']
        
        if 25 <= high_vol_pct <= 35:
            print("🎯 理想的なボラティリティ分布")
        elif 20 <= high_vol_pct <= 40:
            print("✅ 優秀なボラティリティ分布")
        elif 15 <= high_vol_pct <= 50:
            print("📊 実用的なボラティリティ分布")
        else:
            print("⚠️ パラメータ調整を検討してください")
        
        # 効率比評価
        print(f"\n🔍 カウフマン効率比判定結果:")
        if efficiency > 0.7:
            print("✅ 非常に高い効率比（強いトレンド）")
        elif efficiency > 0.5:
            print("📈 高い効率比（トレンド傾向）")
        elif efficiency > 0.3:
            print("📊 中程度の効率比（混合状態）")
        else:
            print("📉 低い効率比（ノイズ優勢）")
        
        print(f"   効率比: {efficiency:.3f} | トレンド: {trend:.3f}")
        
        # 効率比統計
        if results['stats']['efficiency_stats']:
            eff_stats = results['stats']['efficiency_stats']
            print(f"   平均効率比: {eff_stats['mean']:.3f} | >閾値期間: {eff_stats['above_threshold']:.1f}%")
        
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