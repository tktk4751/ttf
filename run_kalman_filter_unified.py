#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 **Kalman Filter Unified Analyzer - 統合カルマンフィルター分析システム** 🎯

kalman_filter_unified.py の全カルマンフィルターを実際の相場データで分析し、
包括的な比較チャートを作成します。

🚀 **分析対象フィルター:**
- adaptive: 基本適応カルマンフィルター
- quantum_adaptive: 量子適応カルマンフィルター  
- unscented: 無香料カルマンフィルター（UKF）
- extended: 拡張カルマンフィルター（EKF）
- hyper_quantum: ハイパー量子適応カルマンフィルター
- triple_ensemble: 三重アンサンブルカルマンフィルター
- neural_supreme: 🧠🚀 Neural Adaptive Quantum Supreme
- market_adaptive_unscented: 🎯 市場適応無香料カルマンフィルター

📊 **包括的分析:**
- 各フィルターの性能比較
- 信頼度スコア分析
- カルマンゲイン動向
- 予測精度評価
- 市場レジーム検出（対応フィルター）
"""

import argparse
import sys
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource
from indicators.kalman_filter_unified import KalmanFilterUnified
from logger import get_logger

# 色設定
FILTER_COLORS = {
    'adaptive': '#2E86AB',           # 青
    'quantum_adaptive': '#A23B72',   # 紫
    'unscented': '#F18F01',          # オレンジ
    'extended': '#C73E1D',           # 赤
    'hyper_quantum': '#7209B7',      # 紫
    'triple_ensemble': '#2A9D8F',    # 青緑
    'neural_supreme': '#E63946',     # 赤
    'market_adaptive_unscented': '#FF6B35'  # オレンジ赤
}

class KalmanFilterUnifiedAnalyzer:
    """
    🎯 統合カルマンフィルター分析システム
    
    全カルマンフィルターを実際の相場データで分析し、
    包括的な比較チャートを作成します。
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
        
        # 全フィルターを初期化
        self.filters = {}
        available_filters = KalmanFilterUnified.get_available_filters()
        
        for filter_type in available_filters.keys():
            try:
                self.filters[filter_type] = KalmanFilterUnified(
                    filter_type=filter_type,
                    src_type='hlc3',
                    ukf_alpha=0.001,
                    ukf_beta=2.0,
                    ukf_kappa=0.0,
                    quantum_scale=0.5
                )
                self.logger.info(f"✅ {filter_type} フィルターを初期化")
            except Exception as e:
                self.logger.error(f"❌ {filter_type} フィルターの初期化に失敗: {e}")
        
        self.logger.info(f"🎯 KalmanFilterUnifiedAnalyzer initialized with {len(self.filters)} filters")
    
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
            self.logger.info("📊 市場データを読み込み中...")
            
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
    
    def run_comprehensive_analysis(self, show_chart: bool = True) -> Dict[str, Any]:
        """包括的カルマンフィルター分析の実行"""
        try:
            self.logger.info("🚀 統合カルマンフィルター分析を開始...")
            
            # データ読み込み
            data = self.load_market_data()
            
            # 各フィルターの計算
            filter_results = {}
            filter_metadata = {}
            
            for filter_name, filter_obj in self.filters.items():
                try:
                    self.logger.info(f"🔍 {filter_name} フィルターを計算中...")
                    result = filter_obj.calculate(data)
                    filter_results[filter_name] = result
                    filter_metadata[filter_name] = filter_obj.get_filter_metadata()
                    self.logger.info(f"✅ {filter_name} 計算完了")
                except Exception as e:
                    self.logger.error(f"❌ {filter_name} フィルターの計算に失敗: {e}")
            
            # 統計分析
            stats = self._calculate_comprehensive_stats(data, filter_results)
            
            # 結果の表示
            self._display_comprehensive_results(stats, filter_metadata)
            
            # チャートの作成
            if show_chart:
                self._create_comprehensive_charts(data, filter_results, filter_metadata, stats)
            
            return {
                'data': data,
                'filter_results': filter_results,
                'filter_metadata': filter_metadata,
                'stats': stats
            }
            
        except Exception as e:
            self.logger.error(f"統合カルマンフィルター分析の実行に失敗: {e}")
            raise
    
    def _calculate_comprehensive_stats(self, data: pd.DataFrame, filter_results: Dict) -> Dict[str, Any]:
        """包括的統計分析"""
        
        def safe_mean(arr):
            if arr is None or len(arr) == 0:
                return 0.0
            valid_values = arr[np.isfinite(arr)]
            return np.mean(valid_values) if len(valid_values) > 0 else 0.0
        
        def safe_std(arr):
            if arr is None or len(arr) == 0:
                return 0.0
            valid_values = arr[np.isfinite(arr)]
            return np.std(valid_values) if len(valid_values) > 0 else 0.0
        
        def calculate_tracking_error(filtered_values, actual_values):
            """追跡誤差を計算"""
            if len(filtered_values) == 0 or len(actual_values) == 0:
                return float('inf')
            
            min_len = min(len(filtered_values), len(actual_values))
            filtered_values = filtered_values[:min_len]
            actual_values = actual_values[:min_len]
            
            valid_mask = np.isfinite(filtered_values) & np.isfinite(actual_values)
            if np.sum(valid_mask) == 0:
                return float('inf')
            
            error = np.sqrt(np.mean((filtered_values[valid_mask] - actual_values[valid_mask])**2))
            return float(error)
        
        def calculate_prediction_accuracy(filtered_values, actual_values):
            """予測精度を計算（方向性の一致率）"""
            if len(filtered_values) < 2 or len(actual_values) < 2:
                return 0.0
            
            min_len = min(len(filtered_values), len(actual_values))
            filtered_values = filtered_values[:min_len]
            actual_values = actual_values[:min_len]
            
            # 方向性の計算
            filtered_direction = np.diff(filtered_values) > 0
            actual_direction = np.diff(actual_values) > 0
            
            valid_mask = np.isfinite(filtered_direction) & np.isfinite(actual_direction)
            if np.sum(valid_mask) == 0:
                return 0.0
            
            accuracy = np.mean(filtered_direction[valid_mask] == actual_direction[valid_mask])
            return float(accuracy)
        
        # 価格データの取得 - 利用可能なカラムを確認
        if 'hlc3' in data.columns:
            price_values = data['hlc3'].values
        elif 'close' in data.columns:
            price_values = data['close'].values
        else:
            # フォールバック: 最初の数値カラムを使用
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                price_values = data[numeric_columns[0]].values
            else:
                raise ValueError("価格データが見つかりません")
        stats = {}
        
        # 各フィルターの統計計算
        for filter_name, result in filter_results.items():
            try:
                filtered_values = result.filtered_values
                confidence_scores = result.confidence_scores
                kalman_gains = result.kalman_gains
                innovations = result.innovation
                
                filter_stats = {
                    'tracking_error': calculate_tracking_error(filtered_values, price_values),
                    'prediction_accuracy': calculate_prediction_accuracy(filtered_values, price_values),
                    'avg_confidence': safe_mean(confidence_scores),
                    'std_confidence': safe_std(confidence_scores),
                    'avg_kalman_gain': safe_mean(kalman_gains),
                    'std_kalman_gain': safe_std(kalman_gains),
                    'avg_innovation': safe_mean(np.abs(innovations)),
                    'std_innovation': safe_std(innovations),
                    'smoothness': safe_std(np.diff(filtered_values)) if len(filtered_values) > 1 else 0.0,
                    'responsiveness': safe_mean(np.abs(np.diff(filtered_values))) if len(filtered_values) > 1 else 0.0
                }
                
                # 高度フィルター用の追加統計
                if hasattr(result, 'quantum_coherence') and result.quantum_coherence is not None:
                    filter_stats['avg_quantum_coherence'] = safe_mean(result.quantum_coherence)
                    filter_stats['std_quantum_coherence'] = safe_std(result.quantum_coherence)
                
                if hasattr(result, 'uncertainty') and result.uncertainty is not None:
                    filter_stats['avg_uncertainty'] = safe_mean(result.uncertainty)
                    filter_stats['std_uncertainty'] = safe_std(result.uncertainty)
                
                if hasattr(result, 'trend_estimate') and result.trend_estimate is not None:
                    filter_stats['avg_trend_estimate'] = safe_mean(result.trend_estimate)
                    filter_stats['std_trend_estimate'] = safe_std(result.trend_estimate)
                
                stats[filter_name] = filter_stats
                
            except Exception as e:
                self.logger.error(f"フィルター {filter_name} の統計計算に失敗: {e}")
                stats[filter_name] = {
                    'tracking_error': float('inf'),
                    'prediction_accuracy': 0.0,
                    'avg_confidence': 0.0,
                    'avg_kalman_gain': 0.0,
                    'smoothness': 0.0,
                    'responsiveness': 0.0
                }
        
        # 総合ランキング
        ranking = self._calculate_filter_ranking(stats)
        stats['ranking'] = ranking
        
        return stats
    
    def _calculate_filter_ranking(self, stats: Dict) -> Dict[str, int]:
        """フィルターランキングを計算"""
        ranking_scores = {}
        
        for filter_name, filter_stats in stats.items():
            if filter_name == 'ranking':
                continue
            
            # スコア計算（低い方が良い指標は逆転）
            score = 0
            
            # 追跡誤差（低い方が良い）
            if filter_stats['tracking_error'] != float('inf'):
                score += (1.0 / (1.0 + filter_stats['tracking_error'])) * 30
            
            # 予測精度（高い方が良い）
            score += filter_stats['prediction_accuracy'] * 25
            
            # 信頼度（高い方が良い）
            score += filter_stats['avg_confidence'] * 20
            
            # 平滑性（適度が良い）
            smoothness = filter_stats['smoothness']
            if smoothness > 0:
                score += min(1.0 / smoothness, 10.0) * 15
            
            # 応答性（適度が良い）
            responsiveness = filter_stats['responsiveness']
            if responsiveness > 0:
                score += min(responsiveness, 10.0) * 10
            
            ranking_scores[filter_name] = score
        
        # ランキング作成
        sorted_filters = sorted(ranking_scores.items(), key=lambda x: x[1], reverse=True)
        ranking = {filter_name: rank + 1 for rank, (filter_name, _) in enumerate(sorted_filters)}
        
        return ranking
    
    def _display_comprehensive_results(self, stats: Dict, metadata: Dict) -> None:
        """包括的結果の表示"""
        self.logger.info("\n" + "="*100)
        self.logger.info("🎯 統合カルマンフィルター包括分析結果")
        self.logger.info("="*100)
        
        # ランキング表示
        ranking = stats.get('ranking', {})
        if ranking:
            self.logger.info(f"\n🏆 フィルター性能ランキング:")
            for filter_name, rank in sorted(ranking.items(), key=lambda x: x[1]):
                description = KalmanFilterUnified.get_available_filters().get(filter_name, filter_name)
                self.logger.info(f"   {rank}位: {filter_name} - {description}")
        
        # 各フィルターの詳細結果
        for filter_name in sorted(stats.keys()):
            if filter_name == 'ranking':
                continue
            
            filter_stats = stats[filter_name]
            rank = ranking.get(filter_name, 0)
            
            self.logger.info(f"\n📊 {filter_name} (ランク: {rank}位):")
            self.logger.info(f"   追跡誤差: {filter_stats['tracking_error']:.6f}")
            self.logger.info(f"   予測精度: {filter_stats['prediction_accuracy']:.3f} ({filter_stats['prediction_accuracy']*100:.1f}%)")
            self.logger.info(f"   平均信頼度: {filter_stats['avg_confidence']:.3f}")
            self.logger.info(f"   平均カルマンゲイン: {filter_stats['avg_kalman_gain']:.3f}")
            self.logger.info(f"   平滑性: {filter_stats['smoothness']:.6f}")
            self.logger.info(f"   応答性: {filter_stats['responsiveness']:.6f}")
            
            # 高度フィルター用の追加情報
            if 'avg_quantum_coherence' in filter_stats:
                self.logger.info(f"   平均量子コヒーレンス: {filter_stats['avg_quantum_coherence']:.3f}")
            if 'avg_uncertainty' in filter_stats:
                self.logger.info(f"   平均不確実性: {filter_stats['avg_uncertainty']:.3f}")
            if 'avg_trend_estimate' in filter_stats:
                self.logger.info(f"   平均トレンド推定: {filter_stats['avg_trend_estimate']:.3f}")
        
        # 総合評価
        if ranking:
            top_filter = min(ranking.items(), key=lambda x: x[1])[0]
            self.logger.info(f"\n🎯 総合評価: {top_filter} フィルターが最高性能を示しました！")
    
    def _create_comprehensive_charts(self, data: pd.DataFrame, filter_results: Dict, 
                                   filter_metadata: Dict, stats: Dict) -> None:
        """包括的比較チャートの作成"""
        try:
            symbol = self.config.get('binance_data', {}).get('symbol', 'Unknown')
            timeframe = self.config.get('binance_data', {}).get('timeframe', 'Unknown')
            
            # フィルター結果の準備
            valid_filters = {name: result for name, result in filter_results.items() 
                           if len(result.filtered_values) > 0}
            
            if not valid_filters:
                self.logger.error("有効なフィルター結果がありません")
                return
            
            # 1. メイン比較チャート
            self._create_main_comparison_chart(data, valid_filters, symbol, timeframe)
            
            # 2. 詳細分析チャート
            self._create_detailed_analysis_chart(data, valid_filters, stats, symbol, timeframe)
            
            # 3. 統計比較チャート
            self._create_statistical_comparison_chart(stats, symbol, timeframe)
            
            # 4. 高度フィルター分析チャート
            self._create_advanced_filters_chart(data, valid_filters, symbol, timeframe)
            
        except Exception as e:
            self.logger.error(f"チャート作成に失敗: {e}")
    
    def _create_main_comparison_chart(self, data: pd.DataFrame, filter_results: Dict, 
                                    symbol: str, timeframe: str) -> None:
        """メイン比較チャートの作成"""
        fig, axes = plt.subplots(2, 1, figsize=(20, 12))
        
        # 上段：価格とフィルター結果
        ax1 = axes[0]
        # 価格データの取得 - 利用可能なカラムを確認
        if 'hlc3' in data.columns:
            price_data = data['hlc3']
            price_label = 'Price (HLC3)'
        elif 'close' in data.columns:
            price_data = data['close']
            price_label = 'Price (Close)'
        else:
            # フォールバック: 最初の数値カラムを使用
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                price_data = data[numeric_columns[0]]
                price_label = f'Price ({numeric_columns[0]})'
            else:
                raise ValueError("価格データが見つかりません")
        
        ax1.plot(data.index, price_data, linewidth=1, color='black', label=price_label, alpha=0.7)
        
        for filter_name, result in filter_results.items():
            color = FILTER_COLORS.get(filter_name, '#666666')
            ax1.plot(data.index, result.filtered_values, 
                    linewidth=1.5, color=color, label=f'{filter_name}', alpha=0.8)
        
        ax1.set_title(f'🎯 統合カルマンフィルター比較 - {symbol} ({timeframe})')
        ax1.set_ylabel('Price')
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 下段：信頼度スコア
        ax2 = axes[1]
        for filter_name, result in filter_results.items():
            if hasattr(result, 'confidence_scores') and result.confidence_scores is not None:
                color = FILTER_COLORS.get(filter_name, '#666666')
                ax2.plot(data.index, result.confidence_scores, 
                        linewidth=1, color=color, label=f'{filter_name}', alpha=0.7)
        
        ax2.set_title('信頼度スコア比較')
        ax2.set_ylabel('Confidence Score')
        ax2.set_xlabel('Time')
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        filename = f"kalman_unified_main_{symbol}_{timeframe}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        self.logger.info(f"📊 メイン比較チャートを保存: {filename}")
        plt.show()
    
    def _create_detailed_analysis_chart(self, data: pd.DataFrame, filter_results: Dict, 
                                      stats: Dict, symbol: str, timeframe: str) -> None:
        """詳細分析チャートの作成"""
        fig = plt.figure(figsize=(24, 16))
        gs = GridSpec(4, 2, height_ratios=[2, 1, 1, 1], hspace=0.3, wspace=0.3)
        
        # 1. 価格チャート（上部全体）
        ax1 = fig.add_subplot(gs[0, :])
        # 価格データの取得
        if 'hlc3' in data.columns:
            price_data = data['hlc3']
            price_label = 'Price'
        elif 'close' in data.columns:
            price_data = data['close']
            price_label = 'Price'
        else:
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                price_data = data[numeric_columns[0]]
                price_label = 'Price'
            else:
                raise ValueError("価格データが見つかりません")
        
        ax1.plot(data.index, price_data, linewidth=2, color='black', label=price_label, alpha=0.8)
        
        # 上位3フィルターのみ表示
        ranking = stats.get('ranking', {})
        top_filters = sorted(ranking.items(), key=lambda x: x[1])[:3]
        
        for filter_name, rank in top_filters:
            if filter_name in filter_results:
                result = filter_results[filter_name]
                color = FILTER_COLORS.get(filter_name, '#666666')
                ax1.plot(data.index, result.filtered_values, 
                        linewidth=2, color=color, label=f'{filter_name} (#{rank})', alpha=0.9)
        
        ax1.set_title(f'🏆 トップ3フィルター詳細分析 - {symbol} ({timeframe})')
        ax1.set_ylabel('Price')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. カルマンゲイン比較（左下）
        ax2 = fig.add_subplot(gs[1, 0])
        for filter_name, rank in top_filters:
            if filter_name in filter_results:
                result = filter_results[filter_name]
                if hasattr(result, 'kalman_gains') and result.kalman_gains is not None:
                    color = FILTER_COLORS.get(filter_name, '#666666')
                    ax2.plot(data.index, result.kalman_gains, 
                            linewidth=1, color=color, label=f'{filter_name}', alpha=0.8)
        
        ax2.set_title('カルマンゲイン比較')
        ax2.set_ylabel('Kalman Gain')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. イノベーション比較（右下）
        ax3 = fig.add_subplot(gs[1, 1])
        for filter_name, rank in top_filters:
            if filter_name in filter_results:
                result = filter_results[filter_name]
                if hasattr(result, 'innovation') and result.innovation is not None:
                    color = FILTER_COLORS.get(filter_name, '#666666')
                    ax3.plot(data.index, np.abs(result.innovation), 
                            linewidth=1, color=color, label=f'{filter_name}', alpha=0.8)
        
        ax3.set_title('イノベーション（予測誤差）比較')
        ax3.set_ylabel('|Innovation|')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. 追跡誤差ヒートマップ（下部左）
        ax4 = fig.add_subplot(gs[2, 0])
        tracking_errors = []
        filter_names = []
        
        for filter_name in sorted(stats.keys()):
            if filter_name != 'ranking':
                tracking_errors.append(stats[filter_name]['tracking_error'])
                filter_names.append(filter_name)
        
        y_pos = np.arange(len(filter_names))
        bars = ax4.barh(y_pos, tracking_errors, color=[FILTER_COLORS.get(name, '#666666') for name in filter_names])
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(filter_names)
        ax4.set_xlabel('Tracking Error')
        ax4.set_title('追跡誤差比較')
        ax4.grid(True, alpha=0.3)
        
        # 5. 予測精度比較（下部右）
        ax5 = fig.add_subplot(gs[2, 1])
        accuracies = []
        
        for filter_name in filter_names:
            accuracies.append(stats[filter_name]['prediction_accuracy'] * 100)
        
        bars = ax5.barh(y_pos, accuracies, color=[FILTER_COLORS.get(name, '#666666') for name in filter_names])
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(filter_names)
        ax5.set_xlabel('Prediction Accuracy (%)')
        ax5.set_title('予測精度比較')
        ax5.grid(True, alpha=0.3)
        
        # 6. 総合スコア比較（最下部）
        ax6 = fig.add_subplot(gs[3, :])
        ranking_scores = []
        
        for filter_name in filter_names:
            rank = ranking.get(filter_name, len(filter_names))
            ranking_scores.append(len(filter_names) + 1 - rank)  # 逆順スコア
        
        bars = ax6.bar(filter_names, ranking_scores, color=[FILTER_COLORS.get(name, '#666666') for name in filter_names])
        ax6.set_ylabel('Performance Score')
        ax6.set_title('総合性能スコア')
        ax6.grid(True, alpha=0.3)
        plt.setp(ax6.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        filename = f"kalman_unified_detailed_{symbol}_{timeframe}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        self.logger.info(f"📊 詳細分析チャートを保存: {filename}")
        plt.show()
    
    def _create_statistical_comparison_chart(self, stats: Dict, symbol: str, timeframe: str) -> None:
        """統計比較チャートの作成"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        filter_names = [name for name in stats.keys() if name != 'ranking']
        
        # 1. 追跡誤差 vs 予測精度
        ax1 = axes[0, 0]
        for filter_name in filter_names:
            filter_stats = stats[filter_name]
            x = filter_stats['tracking_error']
            y = filter_stats['prediction_accuracy']
            color = FILTER_COLORS.get(filter_name, '#666666')
            ax1.scatter(x, y, s=100, color=color, alpha=0.7, label=filter_name)
        
        ax1.set_xlabel('Tracking Error')
        ax1.set_ylabel('Prediction Accuracy')
        ax1.set_title('追跡誤差 vs 予測精度')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. 信頼度 vs カルマンゲイン
        ax2 = axes[0, 1]
        for filter_name in filter_names:
            filter_stats = stats[filter_name]
            x = filter_stats['avg_confidence']
            y = filter_stats['avg_kalman_gain']
            color = FILTER_COLORS.get(filter_name, '#666666')
            ax2.scatter(x, y, s=100, color=color, alpha=0.7, label=filter_name)
        
        ax2.set_xlabel('Average Confidence')
        ax2.set_ylabel('Average Kalman Gain')
        ax2.set_title('信頼度 vs カルマンゲイン')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. 平滑性 vs 応答性
        ax3 = axes[1, 0]
        for filter_name in filter_names:
            filter_stats = stats[filter_name]
            x = filter_stats['smoothness']
            y = filter_stats['responsiveness']
            color = FILTER_COLORS.get(filter_name, '#666666')
            ax3.scatter(x, y, s=100, color=color, alpha=0.7, label=filter_name)
        
        ax3.set_xlabel('Smoothness')
        ax3.set_ylabel('Responsiveness')
        ax3.set_title('平滑性 vs 応答性')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. 総合評価レーダーチャート
        ax4 = axes[1, 1]
        ranking = stats.get('ranking', {})
        top_3_filters = sorted(ranking.items(), key=lambda x: x[1])[:3]
        
        categories = ['Accuracy', 'Confidence', 'Smoothness', 'Responsiveness']
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # 円を閉じる
        
        for filter_name, rank in top_3_filters:
            filter_stats = stats[filter_name]
            values = [
                filter_stats['prediction_accuracy'],
                filter_stats['avg_confidence'],
                min(1.0, 1.0 / (filter_stats['smoothness'] + 0.001)),
                min(1.0, filter_stats['responsiveness'] / 10.0)
            ]
            values += values[:1]  # 円を閉じる
            
            color = FILTER_COLORS.get(filter_name, '#666666')
            ax4.plot(angles, values, 'o-', linewidth=2, color=color, label=f'{filter_name} (#{rank})')
            ax4.fill(angles, values, alpha=0.25, color=color)
        
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(categories)
        ax4.set_ylim(0, 1)
        ax4.set_title('トップ3フィルター総合評価')
        ax4.grid(True)
        ax4.legend()
        
        plt.tight_layout()
        filename = f"kalman_unified_stats_{symbol}_{timeframe}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        self.logger.info(f"📊 統計比較チャートを保存: {filename}")
        plt.show()
    
    def _create_advanced_filters_chart(self, data: pd.DataFrame, filter_results: Dict, 
                                     symbol: str, timeframe: str) -> None:
        """高度フィルター分析チャートの作成"""
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(5, 2, height_ratios=[2, 1, 1, 1, 1], hspace=0.3, wspace=0.3)
        
        # 1. 価格チャート（上部全体）
        ax1 = fig.add_subplot(gs[0, :])
        # 価格データの取得
        if 'hlc3' in data.columns:
            price_data = data['hlc3']
            price_label = 'Price'
        elif 'close' in data.columns:
            price_data = data['close']
            price_label = 'Price'
        else:
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                price_data = data[numeric_columns[0]]
                price_label = 'Price'
            else:
                raise ValueError("価格データが見つかりません")
        
        ax1.plot(data.index, price_data, linewidth=2, color='black', label=price_label, alpha=0.8)
        
        # 高度フィルター（neural_supreme, market_adaptive_unscented）
        advanced_filters = ['neural_supreme', 'market_adaptive_unscented']
        for filter_name in advanced_filters:
            if filter_name in filter_results:
                result = filter_results[filter_name]
                color = FILTER_COLORS.get(filter_name, '#666666')
                ax1.plot(data.index, result.filtered_values, 
                        linewidth=2, color=color, label=filter_name, alpha=0.9)
        
        ax1.set_title(f'🚀 高度フィルター分析 - {symbol} ({timeframe})')
        ax1.set_ylabel('Price')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. Neural Supreme の量子位相（左）
        ax2 = fig.add_subplot(gs[1, 0])
        if 'neural_supreme' in filter_results:
            result = filter_results['neural_supreme']
            if hasattr(result, 'quantum_coherence') and result.quantum_coherence is not None:
                ax2.plot(data.index, result.quantum_coherence, 
                        linewidth=1, color=FILTER_COLORS.get('neural_supreme', '#666666'), alpha=0.8)
        ax2.set_title('Neural Supreme: 量子位相')
        ax2.set_ylabel('Quantum Phase')
        ax2.grid(True, alpha=0.3)
        
        # 3. Market Adaptive の市場レジーム（右）
        ax3 = fig.add_subplot(gs[1, 1])
        if 'market_adaptive_unscented' in filter_results:
            result = filter_results['market_adaptive_unscented']
            if hasattr(result, 'quantum_coherence') and result.quantum_coherence is not None:
                ax3.plot(data.index, result.quantum_coherence, 
                        linewidth=1, color=FILTER_COLORS.get('market_adaptive_unscented', '#666666'), alpha=0.8)
                ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
                ax3.axhline(y=-0.5, color='gray', linestyle='--', alpha=0.5)
        ax3.set_title('Market Adaptive: 市場レジーム')
        ax3.set_ylabel('Market Regime')
        ax3.grid(True, alpha=0.3)
        
        # 4. 不確実性比較（左）
        ax4 = fig.add_subplot(gs[2, 0])
        for filter_name in advanced_filters:
            if filter_name in filter_results:
                result = filter_results[filter_name]
                if hasattr(result, 'uncertainty') and result.uncertainty is not None:
                    color = FILTER_COLORS.get(filter_name, '#666666')
                    ax4.plot(data.index, result.uncertainty, 
                            linewidth=1, color=color, label=filter_name, alpha=0.8)
        ax4.set_title('不確実性推定比較')
        ax4.set_ylabel('Uncertainty')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # 5. トレンド推定比較（右）
        ax5 = fig.add_subplot(gs[2, 1])
        for filter_name in advanced_filters:
            if filter_name in filter_results:
                result = filter_results[filter_name]
                if hasattr(result, 'trend_estimate') and result.trend_estimate is not None:
                    color = FILTER_COLORS.get(filter_name, '#666666')
                    ax5.plot(data.index, result.trend_estimate, 
                            linewidth=1, color=color, label=filter_name, alpha=0.8)
        ax5.set_title('トレンド推定比較')
        ax5.set_ylabel('Trend Estimate')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        
        # 6. 量子コヒーレンス比較（左）
        ax6 = fig.add_subplot(gs[3, 0])
        quantum_filters = ['quantum_adaptive', 'hyper_quantum', 'neural_supreme']
        for filter_name in quantum_filters:
            if filter_name in filter_results:
                result = filter_results[filter_name]
                if hasattr(result, 'quantum_coherence') and result.quantum_coherence is not None:
                    color = FILTER_COLORS.get(filter_name, '#666666')
                    ax6.plot(data.index, result.quantum_coherence, 
                            linewidth=1, color=color, label=filter_name, alpha=0.8)
        ax6.set_title('量子コヒーレンス比較')
        ax6.set_ylabel('Quantum Coherence')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
        
        # 7. プロセスノイズ比較（右）
        ax7 = fig.add_subplot(gs[3, 1])
        for filter_name in advanced_filters:
            if filter_name in filter_results:
                result = filter_results[filter_name]
                if hasattr(result, 'process_noise') and result.process_noise is not None:
                    color = FILTER_COLORS.get(filter_name, '#666666')
                    ax7.plot(data.index, result.process_noise, 
                            linewidth=1, color=color, label=filter_name, alpha=0.8)
        ax7.set_title('プロセスノイズ比較')
        ax7.set_ylabel('Process Noise')
        ax7.grid(True, alpha=0.3)
        ax7.legend()
        
        # 8. 全フィルターのカルマンゲイン分布（下部全体）
        ax8 = fig.add_subplot(gs[4, :])
        
        # ヒストグラム用データ準備
        gain_data = []
        labels = []
        colors = []
        
        for filter_name, result in filter_results.items():
            if hasattr(result, 'kalman_gains') and result.kalman_gains is not None:
                valid_gains = result.kalman_gains[np.isfinite(result.kalman_gains)]
                if len(valid_gains) > 0:
                    gain_data.append(valid_gains)
                    labels.append(filter_name)
                    colors.append(FILTER_COLORS.get(filter_name, '#666666'))
        
        if gain_data:
            ax8.hist(gain_data, bins=50, alpha=0.7, label=labels, color=colors, density=True)
            ax8.set_xlabel('Kalman Gain')
            ax8.set_ylabel('Density')
            ax8.set_title('カルマンゲイン分布比較')
            ax8.grid(True, alpha=0.3)
            ax8.legend()
        
        plt.tight_layout()
        filename = f"kalman_unified_advanced_{symbol}_{timeframe}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        self.logger.info(f"📊 高度フィルター分析チャートを保存: {filename}")
        plt.show()


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description='🎯 統合カルマンフィルター分析システム',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🚀 **分析対象フィルター:**
- adaptive: 基本適応カルマンフィルター
- quantum_adaptive: 量子適応カルマンフィルター  
- unscented: 無香料カルマンフィルター（UKF）
- extended: 拡張カルマンフィルター（EKF）
- hyper_quantum: ハイパー量子適応カルマンフィルター
- triple_ensemble: 三重アンサンブルカルマンフィルター
- neural_supreme: 🧠🚀 Neural Adaptive Quantum Supreme
- market_adaptive_unscented: 🎯 市場適応無香料カルマンフィルター

📊 **包括的分析:**
- 各フィルターの性能比較
- 追跡誤差・予測精度評価
- 信頼度スコア・カルマンゲイン分析
- 市場レジーム・量子コヒーレンス可視化
- 統計的性能ランキング
        """
    )
    
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイル')
    parser.add_argument('--no-show', action='store_true', help='チャート非表示')
    
    args = parser.parse_args()
    
    try:
        print("🎯 統合カルマンフィルター分析システム起動中...")
        print("   📊 全カルマンフィルターを実際の相場データで包括分析")
        
        analyzer = KalmanFilterUnifiedAnalyzer(args.config)
        
        # 分析実行
        results = analyzer.run_comprehensive_analysis(show_chart=not args.no_show)
        
        print("\n✅ 統合カルマンフィルター分析が完了しました！")
        
        # 最終評価
        ranking = results['stats'].get('ranking', {})
        if ranking:
            top_filter = min(ranking.items(), key=lambda x: x[1])[0]
            print(f"🏆 最高性能: {top_filter} フィルター")
            print("📊 詳細な分析結果とチャートが生成されました。")
        
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