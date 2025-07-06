#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 **Ultimate Efficiency Ratio V3.0 - 実用最強版デモ** 🎯

実際のトレードで使える実用的効率比の分析・チャート表示システム

🔧 **V3.0実用機能:**
1. **実用的範囲可視化**: 0.05-0.95での適切な変動表示
2. **ダイナミック期間分析**: 市場状況に応じた期間調整
3. **トレード判断支援**: エントリー・エグジット判断に使える分析
4. **シンプル理解**: 複雑さを隠したわかりやすい表示

💎 **実用性重視:**
実際のトレードで使える感度と安定性のバランスを確認
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import mplfinance as mpf
from typing import Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# パス設定
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from indicators.ultimate_efficiency_ratio import UltimateEfficiencyRatio
    from data.binance_data_source import BinanceDataSource
    from logger.logger import Logger
except ImportError as e:
    print(f"インポートエラー: {e}")
    print(f"プロジェクトルート: {project_root}")
    sys.exit(1)


class UltimateERAnalyzer:
    """Ultimate ER V3.0 実用分析システム"""
    
    def __init__(self, logger: Optional[Logger] = None):
        self.logger = logger or Logger("UltimateERAnalyzer")
        
        # 実用的チャート設定
        plt.style.use('dark_background')
        self.colors = {
            'ultimate_er': '#00FF41',      # 明るいグリーン（実用ER）
            'standard_er': '#FF6B6B',      # 赤（標準ER）
            'high_efficiency': '#00FF41',   # 高効率（緑）
            'medium_efficiency': '#FFA500', # 中効率（オレンジ）
            'low_efficiency': '#FF4444',    # 低効率（赤）
            'dynamic_period': '#00BFFF',    # 期間調整（スカイブルー）
            'background': '#1E1E1E',        # 背景
            'grid': '#333333'               # グリッド
        }
    
    def analyze_and_visualize(
        self,
        symbol: str = "SOLUSDT",
        interval: str = "4h",
        limit: int = 1000,
        period: int = 14,
        sensitivity: float = 0.25,
        show_chart: bool = True,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        🚀 Ultimate ER V3.0 総合分析実行
        
        実際のトレードで使える分析結果を生成
        """
        self.logger.info(f"🎯 Ultimate ER V3.0分析開始: {symbol} ({interval})")
        
        try:
            # データ取得
            data_source = BinanceDataSource()
            data = data_source.get_historical_data(symbol, interval, limit)
            
            if data is None or len(data) == 0:
                raise ValueError(f"データ取得失敗: {symbol}")
            
            # Ultimate ER V3.0計算
            ultimate_er = UltimateEfficiencyRatio(
                period=period,
                sensitivity=sensitivity
            )
            
            result = ultimate_er.calculate(data)
            
            # 分析実行
            analysis_results = self._perform_comprehensive_analysis(
                data, result, symbol, interval
            )
            
            # チャート表示
            if show_chart:
                output_path = self._create_visualization(
                    data, result, analysis_results, 
                    symbol, interval, save_path
                )
                analysis_results['chart_path'] = output_path
            
            # レポート生成
            self._generate_analysis_report(
                analysis_results, symbol, interval, save_path
            )
            
            self.logger.info("✅ Ultimate ER V3.0分析完了")
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"❌ 分析エラー: {str(e)}")
            raise
    
    def _perform_comprehensive_analysis(
        self,
        data: pd.DataFrame,
        result,
        symbol: str,
        interval: str
    ) -> Dict[str, Any]:
        """🔍 包括的分析実行"""
        
        # 基本統計
        ultimate_values = result.values[result.values > 0]
        standard_values = result.raw_er[result.raw_er > 0]
        
        # 実用性分析
        practical_stats = {
            'ultimate_er_mean': np.mean(ultimate_values),
            'ultimate_er_std': np.std(ultimate_values),
            'standard_er_mean': np.mean(standard_values),
            'standard_er_std': np.std(standard_values),
            'range_min': np.min(ultimate_values),
            'range_max': np.max(ultimate_values),
            'current_efficiency': result.current_efficiency,
            'market_state': result.market_efficiency_state
        }
        
        # ノイズ除去効果
        noise_reduction = ((practical_stats['standard_er_std'] - practical_stats['ultimate_er_std']) / 
                          practical_stats['standard_er_std']) if practical_stats['standard_er_std'] > 0 else 0.0
        
        # 効率性レベル分布（実用版）
        high_efficiency = np.sum(ultimate_values > 0.7)
        medium_efficiency = np.sum((ultimate_values >= 0.3) & (ultimate_values <= 0.7))
        low_efficiency = np.sum(ultimate_values < 0.3)
        total_samples = len(ultimate_values)
        
        # ダイナミック期間分析
        dynamic_periods = result.dynamic_periods[result.dynamic_periods > 0]
        period_stats = {
            'avg_period': np.mean(dynamic_periods),
            'min_period': np.min(dynamic_periods),
            'max_period': np.max(dynamic_periods),
            'period_std': np.std(dynamic_periods)
        }
        
        # トレンド感応性
        trend_sensitivity = np.mean(np.abs(result.efficiency_trend))
        
        return {
            'symbol': symbol,
            'interval': interval,
            'total_samples': total_samples,
            'practical_stats': practical_stats,
            'noise_reduction_ratio': noise_reduction,
            'efficiency_distribution': {
                'high_efficiency': (high_efficiency, high_efficiency / total_samples * 100),
                'medium_efficiency': (medium_efficiency, medium_efficiency / total_samples * 100),
                'low_efficiency': (low_efficiency, low_efficiency / total_samples * 100)
            },
            'period_analysis': period_stats,
            'trend_sensitivity': trend_sensitivity,
            'improvement_score': noise_reduction * 100,
            'quality_rating': self._determine_quality_rating(noise_reduction),
            'raw_result': result
        }
    
    def _create_visualization(
        self,
        data: pd.DataFrame,
        result,
        analysis: Dict[str, Any],
        symbol: str,
        interval: str,
        save_path: Optional[str] = None
    ) -> str:
        """📈 実用的チャート作成"""
        
        # データ準備
        plot_data = data.copy()
        plot_data.index = pd.to_datetime(plot_data.index)
        
        # 5パネル構成
        fig = plt.figure(figsize=(16, 12))
        fig.patch.set_facecolor(self.colors['background'])
        
        # === パネル1: 価格チャート ===
        ax1 = plt.subplot(5, 1, 1)
        ax1.set_facecolor(self.colors['background'])
        
        # ローソク足（簡易版）
        for i in range(len(plot_data)):
            color = 'green' if plot_data['close'].iloc[i] > plot_data['open'].iloc[i] else 'red'
            ax1.plot([i, i], [plot_data['low'].iloc[i], plot_data['high'].iloc[i]], 
                    color=color, linewidth=1, alpha=0.7)
            ax1.plot([i, i], [plot_data['open'].iloc[i], plot_data['close'].iloc[i]], 
                    color=color, linewidth=3)
        
        ax1.set_title(f"{symbol} Price Chart ({interval})", color='white', fontsize=12, pad=10)
        ax1.grid(True, color=self.colors['grid'], alpha=0.3)
        ax1.tick_params(colors='white')
        
        # === パネル2: 出来高 ===
        ax2 = plt.subplot(5, 1, 2)
        ax2.set_facecolor(self.colors['background'])
        ax2.bar(range(len(plot_data)), plot_data['volume'], 
                color='skyblue', alpha=0.6, width=0.8)
        ax2.set_title("Volume", color='white', fontsize=10)
        ax2.grid(True, color=self.colors['grid'], alpha=0.3)
        ax2.tick_params(colors='white')
        
        # === パネル3: Ultimate ER vs Standard ER ===
        ax3 = plt.subplot(5, 1, 3)
        ax3.set_facecolor(self.colors['background'])
        
        x_range = range(len(result.values))
        ax3.plot(x_range, result.values, 
                color=self.colors['ultimate_er'], linewidth=2, 
                label=f'Ultimate ER V3.0 (avg: {analysis["practical_stats"]["ultimate_er_mean"]:.3f})')
        ax3.plot(x_range, result.raw_er, 
                color=self.colors['standard_er'], linewidth=1, alpha=0.7,
                label=f'Standard ER (avg: {analysis["practical_stats"]["standard_er_mean"]:.3f})')
        
        # 効率性レベル帯域
        ax3.axhspan(0.7, 1.0, alpha=0.1, color=self.colors['high_efficiency'], label='高効率域')
        ax3.axhspan(0.3, 0.7, alpha=0.1, color=self.colors['medium_efficiency'], label='中効率域')
        ax3.axhspan(0.0, 0.3, alpha=0.1, color=self.colors['low_efficiency'], label='低効率域')
        
        ax3.set_title("Efficiency Ratio Comparison", color='white', fontsize=10)
        ax3.set_ylabel("Efficiency", color='white')
        ax3.legend(loc='upper right', fontsize=8)
        ax3.grid(True, color=self.colors['grid'], alpha=0.3)
        ax3.tick_params(colors='white')
        ax3.set_ylim(0, 1)
        
        # === パネル4: ダイナミック期間 ===
        ax4 = plt.subplot(5, 1, 4)
        ax4.set_facecolor(self.colors['background'])
        
        ax4.plot(x_range, result.dynamic_periods, 
                color=self.colors['dynamic_period'], linewidth=1.5,
                label=f'Dynamic Period (avg: {analysis["period_analysis"]["avg_period"]:.1f})')
        ax4.axhline(y=analysis["period_analysis"]["avg_period"], 
                   color='yellow', linestyle='--', alpha=0.7, label='Average')
        
        ax4.set_title("Dynamic Period Adjustment", color='white', fontsize=10)
        ax4.set_ylabel("Period", color='white')
        ax4.legend(loc='upper right', fontsize=8)
        ax4.grid(True, color=self.colors['grid'], alpha=0.3)
        ax4.tick_params(colors='white')
        
        # === パネル5: 効率性トレンド ===
        ax5 = plt.subplot(5, 1, 5)
        ax5.set_facecolor(self.colors['background'])
        
        ax5.plot(x_range, result.efficiency_trend, 
                color='cyan', linewidth=1.5, label='Efficiency Trend')
        ax5.axhline(y=0, color='white', linestyle='-', alpha=0.5)
        ax5.fill_between(x_range, 0, result.efficiency_trend, 
                        where=(result.efficiency_trend > 0), 
                        color='green', alpha=0.3, label='Rising')
        ax5.fill_between(x_range, 0, result.efficiency_trend, 
                        where=(result.efficiency_trend < 0), 
                        color='red', alpha=0.3, label='Falling')
        
        ax5.set_title("Efficiency Trend", color='white', fontsize=10)
        ax5.set_ylabel("Trend", color='white')
        ax5.set_xlabel("Time", color='white')
        ax5.legend(loc='upper right', fontsize=8)
        ax5.grid(True, color=self.colors['grid'], alpha=0.3)
        ax5.tick_params(colors='white')
        
        plt.tight_layout()
        
        # 保存
        if save_path:
            output_path = f"{save_path}/ultimate_er_v3_live_analysis.png"
        else:
            output_dir = project_root / "examples" / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / "ultimate_er_v3_live_analysis.png")
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor=self.colors['background'])
        
        if plt.get_backend() != 'Agg':
            plt.show()
        
        return output_path
    
    def _generate_analysis_report(
        self,
        analysis: Dict[str, Any],
        symbol: str,
        interval: str,
        save_path: Optional[str] = None
    ) -> str:
        """📋 実用分析レポート生成"""
        
        # レポート作成
        report_lines = [
            "🎯 === Ultimate ER V3.0 - 実用分析レポート ===",
            f"📊 シンボル: {symbol} ({interval})",
            f"📈 総サンプル数: {analysis['total_samples']:,}",
            "",
            "🔧 === 実用性能指標 ===",
            f"🚀 Ultimate ER V3.0 - 平均: {analysis['practical_stats']['ultimate_er_mean']:.4f}, 標準偏差: {analysis['practical_stats']['ultimate_er_std']:.4f}",
            f"📊 標準ER - 平均: {analysis['practical_stats']['standard_er_mean']:.4f}, 標準偏差: {analysis['practical_stats']['standard_er_std']:.4f}",
            f"🧬 ノイズ除去比率: {analysis['noise_reduction_ratio']*100:.2f}%",
            f"⭐ {analysis['quality_rating']} - {self._get_quality_description(analysis['noise_reduction_ratio'])}",
            "",
            "📈 効率性レベル分布:",
            f"  🟢 高効率 (>0.7): {analysis['efficiency_distribution']['high_efficiency'][0]} ({analysis['efficiency_distribution']['high_efficiency'][1]:.1f}%)",
            f"  🟠 中効率 (0.3-0.7): {analysis['efficiency_distribution']['medium_efficiency'][0]} ({analysis['efficiency_distribution']['medium_efficiency'][1]:.1f}%)",
            f"  🔴 低効率 (<0.3): {analysis['efficiency_distribution']['low_efficiency'][0]} ({analysis['efficiency_distribution']['low_efficiency'][1]:.1f}%)",
            "",
            "📊 === ダイナミック期間分析 ===",
            f"📏 平均期間: {analysis['period_analysis']['avg_period']:.1f}",
            f"📐 期間範囲: {analysis['period_analysis']['min_period']:.0f} - {analysis['period_analysis']['max_period']:.0f}",
            f"📊 期間変動: ±{analysis['period_analysis']['period_std']:.1f}",
            "",
            "🎯 === 現在状態 ===",
            f"💎 現在の効率性: {analysis['practical_stats']['current_efficiency']:.4f}",
            f"🏛️ 市場効率性状態: {analysis['practical_stats']['market_state']}",
            f"📈 トレンド感応性: {analysis['trend_sensitivity']:.3f}",
            "",
            "⚡ === 実用性評価 ===",
            f"🔧 実用的変動範囲: {analysis['practical_stats']['range_min']:.3f} - {analysis['practical_stats']['range_max']:.3f}",
            f"📊 改善スコア: {analysis['improvement_score']:.2f}%",
            f"🎯 トレード適用性: {self._evaluate_trading_suitability(analysis)}",
            "",
            "🎯 Ultimate ER V3.0 - 実用最強版分析完了",
            "🔧 4つの実用技術により最適なトレード判断を支援"
        ]
        
        # レポート出力
        report_text = "\n".join(report_lines)
        print(report_text)
        
        # ファイル保存
        if save_path:
            report_path = f"{save_path}/ultimate_er_v3_live_report.txt"
        else:
            output_dir = project_root / "examples" / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            report_path = str(output_dir / "ultimate_er_v3_live_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"📋 V3.0詳細レポートを保存しました: {report_path}")
        
        return report_path
    
    def _determine_quality_rating(self, noise_reduction: float) -> str:
        """品質評価判定"""
        if noise_reduction > 0.3:
            return "優秀"
        elif noise_reduction > 0.1:
            return "良好"
        elif noise_reduction > 0.0:
            return "普通"
        else:
            return "要調整"
    
    def _get_quality_description(self, noise_reduction: float) -> str:
        """品質説明"""
        if noise_reduction > 0.3:
            return "大幅なノイズ除去効果"
        elif noise_reduction > 0.1:
            return "適度なノイズ除去効果"
        elif noise_reduction > 0.0:
            return "軽微なノイズ除去効果"
        else:
            return "ノイズ除去効果なし"
    
    def _evaluate_trading_suitability(self, analysis: Dict[str, Any]) -> str:
        """トレード適用性評価"""
        range_span = analysis['practical_stats']['range_max'] - analysis['practical_stats']['range_min']
        trend_sensitivity = analysis['trend_sensitivity']
        
        if range_span > 0.4 and trend_sensitivity > 0.1:
            return "非常に適している"
        elif range_span > 0.2 and trend_sensitivity > 0.05:
            return "適している"
        elif range_span > 0.1:
            return "部分的に適している"
        else:
            return "適用注意"


def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ultimate ER V3.0 実用分析システム')
    parser.add_argument('--symbol', default='SOLUSDT', help='取引シンボル')
    parser.add_argument('--interval', default='4h', help='時間軸')
    parser.add_argument('--limit', type=int, default=1000, help='データ数')
    parser.add_argument('--period', type=int, default=14, help='基本期間')
    parser.add_argument('--sensitivity', type=float, default=0.25, help='感応性')
    parser.add_argument('--no-chart', action='store_true', help='チャート表示無効')
    parser.add_argument('--save-path', help='保存パス')
    
    args = parser.parse_args()
    
    try:
        analyzer = UltimateERAnalyzer()
        results = analyzer.analyze_and_visualize(
            symbol=args.symbol,
            interval=args.interval,
            limit=args.limit,
            period=args.period,
            sensitivity=args.sensitivity,
            show_chart=not args.no_chart,
            save_path=args.save_path
        )
        
        print(f"\n✅ Ultimate ER V3.0分析完了: {args.symbol}")
        return results
        
    except Exception as e:
        print(f"❌ エラー: {str(e)}")
        return None


if __name__ == "__main__":
    main()