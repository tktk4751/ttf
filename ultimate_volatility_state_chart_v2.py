#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import mplfinance as mpf

from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource
from indicators.ultimate_volatility_state_v2 import UltimateVolatilityStateV2
from logger import get_logger


class UltimateVolatilityStateChartV2:
    """
    Ultimate Volatility State V2 インジケーターのチャート表示クラス
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        self.logger = get_logger(__name__)
        self.config = self._load_config(config_path)
        
        # データローダーの初期化（Binanceデータソース対応）
        binance_config = self.config.get('binance_data', {})
        data_dir = binance_config.get('data_dir', 'data/binance')
        binance_data_source = BinanceDataSource(data_dir)
        
        # CSVデータソースはダミーとして設定
        dummy_csv_source = CSVDataSource("dummy")
        self.data_loader = DataLoader(
            data_source=dummy_csv_source,
            binance_data_source=binance_data_source
        )
        
        # データプロセッサーの初期化
        self.data_processor = DataProcessor()
        
        # Ultimate Volatility State V2 インジケーターの初期化
        uvs_config = self.config.get('ultimate_volatility_state', {})
        self.uvs_indicator = UltimateVolatilityStateV2(
            period=uvs_config.get('period', 21),
            threshold=uvs_config.get('threshold', 0.5),
            zscore_period=uvs_config.get('zscore_period', 50),
            src_type=uvs_config.get('src_type', 'hlc3'),
            smoother_period=uvs_config.get('smoother_period', 3),  # V2では短縮
            adaptive_threshold=uvs_config.get('adaptive_threshold', True),
            confidence_threshold=0.7
        )
        
        self.logger.info("Ultimate Volatility State Chart V2 initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """設定ファイルを読み込み"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"設定ファイルの読み込みに失敗: {e}")
            raise
    
    def load_market_data(self) -> pd.DataFrame:
        """市場データを読み込み（Binanceデータソース使用）"""
        try:
            self.logger.info("市場データを読み込み中...")
            
            # 設定ファイルからデータを読み込み
            raw_data = self.data_loader.load_data_from_config(self.config)
            
            if not raw_data:
                raise ValueError("データが空です")
            
            # 最初のシンボルのデータを取得
            first_symbol = next(iter(raw_data))
            symbol_data = raw_data[first_symbol]
            
            if symbol_data.empty:
                raise ValueError(f"シンボル {first_symbol} のデータが空です")
            
            # データの前処理
            processed_data = self.data_processor.process(symbol_data)
            
            self.logger.info(f"データ読み込み完了: {first_symbol}")
            self.logger.info(f"期間: {processed_data.index.min()} → {processed_data.index.max()}")
            self.logger.info(f"データ数: {len(processed_data)} レコード")
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"市場データの読み込みに失敗: {e}")
            raise
    
    def calculate_volatility_state(self, data: pd.DataFrame) -> Dict[str, Any]:
        """ボラティリティ状態を計算（V2高精度版）"""
        try:
            self.logger.info("V2ボラティリティ状態の計算を開始...")
            
            # インジケーターの計算
            result = self.uvs_indicator.calculate(data)
            
            # 統計情報の計算
            high_vol_count = np.sum(result.state)
            low_vol_count = np.sum(result.state == 0)
            total_count = len(result.state)
            
            high_vol_percentage = (high_vol_count / total_count * 100) if total_count > 0 else 0
            low_vol_percentage = (low_vol_count / total_count * 100) if total_count > 0 else 0
            
            # 信頼度統計
            avg_confidence = np.mean(result.confidence[result.confidence > 0])
            high_confidence_count = np.sum(result.confidence > 0.7)
            high_confidence_percentage = (high_confidence_count / total_count * 100) if total_count > 0 else 0
            
            # 確率統計
            avg_probability = np.mean(result.probability[result.probability > 0])
            
            # タイムフレーム別統計
            timeframe_stats = {}
            if result.timeframe_analysis:
                for tf_name, tf_data in result.timeframe_analysis.items():
                    timeframe_stats[tf_name] = {
                        'mean': np.mean(tf_data[tf_data > 0]),
                        'std': np.std(tf_data[tf_data > 0]),
                        'max': np.max(tf_data)
                    }
            
            stats = {
                'total_periods': total_count,
                'high_volatility_count': high_vol_count,
                'low_volatility_count': low_vol_count,
                'high_volatility_percentage': high_vol_percentage,
                'low_volatility_percentage': low_vol_percentage,
                'average_probability': avg_probability,
                'average_confidence': avg_confidence,
                'high_confidence_count': high_confidence_count,
                'high_confidence_percentage': high_confidence_percentage,
                'latest_state': 'High' if result.state[-1] == 1 else 'Low',
                'latest_probability': result.probability[-1],
                'latest_confidence': result.confidence[-1],
                'timeframe_stats': timeframe_stats
            }
            
            self.logger.info(f"V2計算完了 - 高ボラティリティ: {high_vol_percentage:.1f}%, 信頼度: {avg_confidence:.3f}")
            
            return {
                'result': result,
                'stats': stats
            }
            
        except Exception as e:
            self.logger.error(f"V2ボラティリティ状態の計算に失敗: {e}")
            raise
    
    def create_chart_v2(self, data: pd.DataFrame, volatility_data: Dict[str, Any], show_chart: bool = True) -> None:
        """V2チャートを作成（高度な分析表示）"""
        try:
            result = volatility_data['result']
            stats = volatility_data['stats']
            uvs_config = self.config.get('ultimate_volatility_state', {})
            
            # 図のサイズとレイアウト設定（より多くのパネル）
            fig = plt.figure(figsize=(18, 16))
            
            # グリッドレイアウト: 6行1列
            gs = fig.add_gridspec(6, 1, height_ratios=[3, 1, 1, 1, 1, 1], hspace=0.4)
            
            # 1. 価格チャート + ボラティリティ状態 + 信頼度
            ax1 = fig.add_subplot(gs[0])
            self._plot_price_with_volatility_and_confidence(ax1, data, result, uvs_config)
            
            # 2. ボラティリティ状態と信頼度
            ax2 = fig.add_subplot(gs[1])
            self._plot_state_and_confidence(ax2, data, result, uvs_config)
            
            # 3. 確率ライン（改良版）
            ax3 = fig.add_subplot(gs[2])
            self._plot_enhanced_probability(ax3, data, result, uvs_config)
            
            # 4. タイムフレーム別ボラティリティ
            ax4 = fig.add_subplot(gs[3])
            self._plot_timeframe_volatility(ax4, data, result)
            
            # 5. 高度なコンポーネント分析
            ax5 = fig.add_subplot(gs[4])
            self._plot_advanced_components(ax5, data, result)
            
            # 6. 統計サマリー
            ax6 = fig.add_subplot(gs[5])
            self._plot_statistics_summary(ax6, stats)
            
            # タイトルと統計情報（詳細版）
            symbol = self.config.get('binance_data', {}).get('symbol', 'Unknown')
            timeframe = self.config.get('binance_data', {}).get('timeframe', 'Unknown')
            
            title = f"Ultimate Volatility State V2 Analysis - {symbol} ({timeframe})\n"
            title += f"High Vol: {stats['high_volatility_percentage']:.1f}% | "
            title += f"Low Vol: {stats['low_volatility_percentage']:.1f}% | "
            title += f"Avg Confidence: {stats['average_confidence']:.3f} | "
            title += f"High Confidence: {stats['high_confidence_percentage']:.1f}%\n"
            title += f"Latest: {stats['latest_state']} (Prob: {stats['latest_probability']:.3f}, Conf: {stats['latest_confidence']:.3f})"
            
            fig.suptitle(title, fontsize=12, fontweight='bold')
            
            # チャートの保存
            if uvs_config.get('save_chart', True):
                filename = uvs_config.get('chart_filename', 'ultimate_volatility_state_analysis')
                output_path = f"{filename}_v2_{symbol}_{timeframe}.png"
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"V2チャートを保存しました: {output_path}")
            
            if show_chart:
                plt.show()
            else:
                plt.close(fig)
            
        except Exception as e:
            self.logger.error(f"V2チャートの作成に失敗: {e}")
            raise
    
    def _plot_price_with_volatility_and_confidence(self, ax, data: pd.DataFrame, result, config: Dict):
        """価格チャートとボラティリティ状態、信頼度を表示"""
        # 価格チャート
        ax.plot(data.index, data['close'], linewidth=1, color='black', label='Price')
        
        # 信頼度に応じてボラティリティ状態の透明度を調整
        high_vol_periods = result.state == 1
        low_vol_periods = result.state == 0
        
        for i in range(len(data)):
            alpha = 0.1 + 0.3 * result.confidence[i]  # 信頼度に応じた透明度
            
            if high_vol_periods[i]:
                ax.axvspan(data.index[i], data.index[min(i+1, len(data)-1)], 
                          alpha=alpha, color=config.get('high_vol_color', 'red'))
            elif low_vol_periods[i]:
                ax.axvspan(data.index[i], data.index[min(i+1, len(data)-1)], 
                          alpha=alpha, color=config.get('low_vol_color', 'blue'))
        
        ax.set_title('Price Chart with Volatility States (opacity = confidence)')
        ax.set_ylabel('Price')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_state_and_confidence(self, ax, data: pd.DataFrame, result, config: Dict):
        """ボラティリティ状態と信頼度を表示"""
        # 状態をバーで表示
        colors = [config.get('high_vol_color', 'red') if state == 1 
                 else config.get('low_vol_color', 'blue') for state in result.state]
        
        ax.bar(data.index, result.state, color=colors, alpha=0.6, width=1, label='State')
        
        # 信頼度をラインで重ねて表示
        ax2 = ax.twinx()
        ax2.plot(data.index, result.confidence, color='green', linewidth=1.5, alpha=0.8, label='Confidence')
        ax2.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='High Confidence Threshold')
        
        ax.set_title('Volatility State & Confidence')
        ax.set_ylabel('State (1: High, 0: Low)')
        ax2.set_ylabel('Confidence')
        ax.set_ylim(-0.1, 1.1)
        ax2.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # 凡例を統合
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    def _plot_enhanced_probability(self, ax, data: pd.DataFrame, result, config: Dict):
        """拡張確率表示"""
        # 基本確率ライン
        ax.plot(data.index, result.probability, 
               color=config.get('probability_color', 'orange'), 
               linewidth=1.5, label='Probability')
        
        # Raw scoreも表示（正規化）
        normalized_score = result.raw_score / np.max(result.raw_score) if np.max(result.raw_score) > 0 else result.raw_score
        ax.plot(data.index, normalized_score, 
               color='purple', linewidth=1, alpha=0.7, label='Raw Score (normalized)')
        
        # 閾値ライン
        threshold = self.uvs_indicator.threshold
        ax.axhline(y=threshold, color='gray', linestyle='--', alpha=0.5, label=f'Threshold ({threshold})')
        
        # 高確率・高信頼度領域をハイライト
        high_prob_conf = (result.probability > 0.7) & (result.confidence > 0.7)
        for i in range(len(data)):
            if high_prob_conf[i]:
                ax.axvspan(data.index[i], data.index[min(i+1, len(data)-1)], 
                          alpha=0.2, color='gold')
        
        ax.set_title('Enhanced Probability Analysis')
        ax.set_ylabel('Probability / Score')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_timeframe_volatility(self, ax, data: pd.DataFrame, result):
        """タイムフレーム別ボラティリティ表示"""
        timeframe_analysis = result.timeframe_analysis
        
        if not timeframe_analysis:
            ax.text(0.5, 0.5, 'No timeframe data available', 
                   transform=ax.transAxes, ha='center', va='center')
            return
        
        colors = ['red', 'orange', 'blue', 'green']
        labels = ['Short Term', 'Medium Term', 'Long Term', 'Trend']
        
        for i, (tf_name, tf_data) in enumerate(timeframe_analysis.items()):
            if i < len(colors):
                # 正規化して表示
                max_val = np.max(tf_data) if np.max(tf_data) > 0 else 1
                normalized_data = tf_data / max_val
                ax.plot(data.index, normalized_data, 
                       color=colors[i], linewidth=1.2, alpha=0.8, 
                       label=f"{labels[i] if i < len(labels) else tf_name}")
        
        ax.set_title('Multi-Timeframe Volatility Analysis')
        ax.set_ylabel('Normalized Volatility')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_advanced_components(self, ax, data: pd.DataFrame, result):
        """高度なコンポーネント分析表示"""
        components = result.components
        
        if not components:
            ax.text(0.5, 0.5, 'No component data available', 
                   transform=ax.transAxes, ha='center', va='center')
            return
        
        # 重要なコンポーネントを選択
        key_components = ['garch_volatility', 'spectral_entropy', 'hurst_exponent']
        colors = ['red', 'blue', 'green']
        
        for i, comp_name in enumerate(key_components):
            if comp_name in components and i < len(colors):
                comp_data = components[comp_name]
                # 正規化
                if np.max(np.abs(comp_data)) > 0:
                    if comp_name == 'hurst_exponent':
                        # Hurst指数は0.5からの偏差を強調
                        normalized_data = np.abs(comp_data - 0.5) * 2
                    else:
                        normalized_data = np.abs(comp_data) / np.max(np.abs(comp_data))
                    
                    ax.plot(data.index, normalized_data, 
                           color=colors[i], alpha=0.8, linewidth=1, 
                           label=comp_name.replace('_', ' ').title())
        
        ax.set_title('Advanced Component Analysis')
        ax.set_ylabel('Normalized Value')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_statistics_summary(self, ax, stats: Dict):
        """統計サマリー表示"""
        ax.axis('off')  # 軸を非表示
        
        # 統計情報をテキストで表示
        summary_text = f"""
V2 VOLATILITY STATE ANALYSIS SUMMARY

High Volatility Periods: {stats['high_volatility_count']} ({stats['high_volatility_percentage']:.1f}%)
Low Volatility Periods: {stats['low_volatility_count']} ({stats['low_volatility_percentage']:.1f}%)

Average Probability: {stats['average_probability']:.3f}
Average Confidence: {stats['average_confidence']:.3f}
High Confidence Periods: {stats['high_confidence_count']} ({stats['high_confidence_percentage']:.1f}%)

Latest State: {stats['latest_state']}
Latest Probability: {stats['latest_probability']:.3f}
Latest Confidence: {stats['latest_confidence']:.3f}
        """
        
        # タイムフレーム統計を追加
        if 'timeframe_stats' in stats and stats['timeframe_stats']:
            summary_text += "\nTIMEFRAME ANALYSIS:\n"
            for tf_name, tf_stats in stats['timeframe_stats'].items():
                summary_text += f"{tf_name}: Mean={tf_stats['mean']:.4f}, Max={tf_stats['max']:.4f}\n"
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    def run_analysis_v2(self, show_chart: bool = True) -> None:
        """V2分析の実行"""
        try:
            self.logger.info("Ultimate Volatility State V2 分析を開始...")
            
            # 市場データの読み込み
            data = self.load_market_data()
            
            # V2ボラティリティ状態の計算
            volatility_data = self.calculate_volatility_state(data)
            
            # 統計情報の表示
            stats = volatility_data['stats']
            self.logger.info("=== V2分析結果 ===")
            self.logger.info(f"総期間数: {stats['total_periods']}")
            self.logger.info(f"高ボラティリティ期間: {stats['high_volatility_count']} ({stats['high_volatility_percentage']:.1f}%)")
            self.logger.info(f"低ボラティリティ期間: {stats['low_volatility_count']} ({stats['low_volatility_percentage']:.1f}%)")
            self.logger.info(f"平均確率: {stats['average_probability']:.3f}")
            self.logger.info(f"平均信頼度: {stats['average_confidence']:.3f}")
            self.logger.info(f"高信頼度期間: {stats['high_confidence_count']} ({stats['high_confidence_percentage']:.1f}%)")
            self.logger.info(f"最新状態: {stats['latest_state']} (確率: {stats['latest_probability']:.3f}, 信頼度: {stats['latest_confidence']:.3f})")
            
            # V2チャートの作成
            self.create_chart_v2(data, volatility_data, show_chart)
            
            self.logger.info("V2分析が完了しました")
            
        except Exception as e:
            self.logger.error(f"V2分析の実行に失敗: {e}")
            raise


def main():
    """メイン関数"""
    try:
        # V2チャート分析の実行
        chart_analyzer = UltimateVolatilityStateChartV2()
        chart_analyzer.run_analysis_v2()
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"実行に失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()