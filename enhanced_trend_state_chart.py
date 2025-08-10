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
from indicators.enhanced_trend_state_v2 import EnhancedTrendStateV2 as EnhancedTrendState
from logger import get_logger


class EnhancedTrendStateChart:
    """
    Enhanced Trend State インジケーターのチャート表示クラス
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
        
        # Enhanced Trend State インジケーターの初期化
        ets_config = self.config.get('enhanced_trend_state', {})
        self.ets_indicator = EnhancedTrendState(
            base_period=ets_config.get('base_period', 20),
            threshold=ets_config.get('threshold', 0.4),
            src_type=ets_config.get('src_type', 'hlc3'),
            use_dynamic_period=ets_config.get('use_dynamic_period', True),
            volatility_adjustment=ets_config.get('volatility_adjustment', True),
            atr_smoothing=ets_config.get('atr_smoothing', True),
            er_weight=ets_config.get('er_weight', 0.6),
            chop_weight=ets_config.get('chop_weight', 0.4),
            detector_type=ets_config.get('detector_type', 'absolute_ultimate'),
            max_cycle=ets_config.get('max_cycle', 50),
            min_cycle=ets_config.get('min_cycle', 8)
        )
        
        self.logger.info("Enhanced Trend State Chart initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """設定ファイルを読み込み"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"設定ファイルの読み込みに失敗: {e}")
            # デフォルト設定を返す
            return {
                'data': {
                    'symbol': 'BTCUSDT',
                    'timeframe': '4h',
                    'limit': 500
                },
                'enhanced_trend_state': {
                    'base_period': 20,
                    'threshold': 0.6,
                    'use_dynamic_period': True,
                    'volatility_adjustment': True,
                    'atr_smoothing': True,
                    'show_components': True,
                    'save_chart': True,
                    'chart_title': 'Enhanced Trend State Analysis'
                },
                'binance_data': {
                    'data_dir': 'data/binance'
                }
            }
    
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
    
    def calculate_trend_state(self, data: pd.DataFrame) -> Dict[str, Any]:
        """トレンド状態を計算"""
        try:
            self.logger.info("トレンド状態の計算を開始...")
            
            # インジケーターの計算
            result = self.ets_indicator.calculate(data)
            
            # 統計情報の計算
            trend_count = np.sum(result.trend_state)
            range_count = np.sum(result.trend_state == 0)
            total_count = len(result.trend_state)
            
            trend_percentage = (trend_count / total_count * 100) if total_count > 0 else 0
            range_percentage = (range_count / total_count * 100) if total_count > 0 else 0
            
            # 有効な値のみで平均を計算
            valid_confidence = result.confidence[~np.isnan(result.confidence)]
            valid_composite = result.composite_score[~np.isnan(result.composite_score)]
            valid_er = result.efficiency_ratio[~np.isnan(result.efficiency_ratio)]
            valid_chop = result.choppiness_index[~np.isnan(result.choppiness_index)]
            
            avg_confidence = np.mean(valid_confidence) if len(valid_confidence) > 0 else 0
            avg_composite = np.mean(valid_composite) if len(valid_composite) > 0 else 0
            avg_er = np.mean(valid_er) if len(valid_er) > 0 else 0
            avg_chop = np.mean(valid_chop) if len(valid_chop) > 0 else 0
            
            # トレンド切り替え回数を計算
            trend_switches = np.sum(np.diff(result.trend_state.astype(int)) != 0)
            
            stats = {
                'total_periods': total_count,
                'trend_count': trend_count,
                'range_count': range_count,
                'trend_percentage': trend_percentage,
                'range_percentage': range_percentage,
                'average_confidence': avg_confidence,
                'average_composite_score': avg_composite,
                'average_efficiency_ratio': avg_er,
                'average_choppiness_index': avg_chop,
                'trend_switches': trend_switches,
                'latest_state': 'トレンド' if result.trend_state[-1] == 1 else 'レンジ',
                'latest_confidence': result.confidence[-1] if not np.isnan(result.confidence[-1]) else 0,
                'latest_composite_score': result.composite_score[-1] if not np.isnan(result.composite_score[-1]) else 0
            }
            
            self.logger.info(f"計算完了 - トレンド: {trend_percentage:.1f}%, レンジ: {range_percentage:.1f}%")
            
            return {
                'result': result,
                'stats': stats
            }
            
        except Exception as e:
            self.logger.error(f"トレンド状態の計算に失敗: {e}")
            raise
    
    def create_chart(self, data: pd.DataFrame, trend_data: Dict[str, Any], show_chart: bool = True) -> None:
        """チャートを作成"""
        try:
            result = trend_data['result']
            stats = trend_data['stats']
            ets_config = self.config.get('enhanced_trend_state', {})
            
            # 図のサイズとレイアウト設定
            fig = plt.figure(figsize=(18, 14))
            
            # グリッドレイアウト: 5行1列
            gs = fig.add_gridspec(5, 1, height_ratios=[3, 0.8, 1.2, 1.2, 1.2], hspace=0.35)
            
            # 1. 価格チャート + トレンド状態
            ax1 = fig.add_subplot(gs[0])
            self._plot_price_with_trend_state(ax1, data, result, ets_config)
            
            # 2. トレンド状態バー
            ax2 = fig.add_subplot(gs[1])
            self._plot_trend_state_bar(ax2, data, result, ets_config)
            
            # 3. 複合スコアと信頼度
            ax3 = fig.add_subplot(gs[2])
            self._plot_composite_and_confidence(ax3, data, result, ets_config)
            
            # 4. 効率比とチョピネス指数
            ax4 = fig.add_subplot(gs[3])
            self._plot_er_and_choppiness(ax4, data, result, ets_config)
            
            # 5. 動的期間とボラティリティ係数
            if ets_config.get('show_components', True):
                ax5 = fig.add_subplot(gs[4])
                self._plot_dynamic_components(ax5, data, result)
            
            # タイトルと統計情報
            symbol = self.config.get('data', {}).get('symbol', 'Unknown')
            timeframe = self.config.get('data', {}).get('timeframe', 'Unknown')
            
            title = f"{ets_config.get('chart_title', 'Enhanced Trend State Analysis')} - {symbol} ({timeframe})\n"
            title += f"トレンド: {stats['trend_percentage']:.1f}% | レンジ: {stats['range_percentage']:.1f}% | "
            title += f"切替回数: {stats['trend_switches']} | 現在: {stats['latest_state']} "
            title += f"(スコア: {stats['latest_composite_score']:.3f}, 信頼度: {stats['latest_confidence']:.3f})"
            
            fig.suptitle(title, fontsize=14, fontweight='bold')
            
            # チャートの保存
            if ets_config.get('save_chart', True):
                filename = ets_config.get('chart_filename', 'enhanced_trend_state_analysis')
                output_path = f"{filename}_{symbol}_{timeframe}.png"
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"チャートを保存しました: {output_path}")
            
            if show_chart:
                plt.show()
            else:
                plt.close(fig)
            
        except Exception as e:
            self.logger.error(f"チャートの作成に失敗: {e}")
            raise
    
    def _plot_price_with_trend_state(self, ax, data: pd.DataFrame, result, config: Dict):
        """価格チャートとトレンド状態を表示"""
        # 価格チャート（ローソク足風）
        ax.plot(data.index, data['close'], linewidth=1.2, color='black', alpha=0.8, label='Close Price')
        
        # トレンド期間をハイライト
        trend_periods = result.trend_state == 1
        if np.any(trend_periods):
            # トレンド期間を緑色でハイライト
            ax.fill_between(data.index, data['close'].min(), data['close'].max(),
                           where=trend_periods, alpha=0.15, color='green', label='トレンド期間')
        
        # レンジ期間をハイライト
        range_periods = result.trend_state == 0
        if np.any(range_periods):
            # レンジ期間を赤色でハイライト
            ax.fill_between(data.index, data['close'].min(), data['close'].max(),
                           where=range_periods, alpha=0.10, color='red', label='レンジ期間')
        
        ax.set_title('価格チャート + トレンド状態', fontsize=12, fontweight='bold')
        ax.set_ylabel('価格', fontsize=10)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # X軸の日付フォーマット
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(data)//10)))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_trend_state_bar(self, ax, data: pd.DataFrame, result, config: Dict):
        """トレンド状態バーを表示"""
        # トレンド状態をバーとして表示
        colors = ['red' if state == 0 else 'green' for state in result.trend_state]
        
        ax.bar(range(len(result.trend_state)), result.trend_state, 
               color=colors, alpha=0.7, width=1.0)
        
        ax.set_title('トレンド状態 (緑=トレンド, 赤=レンジ)', fontsize=12, fontweight='bold')
        ax.set_ylabel('状態', fontsize=10)
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['レンジ', 'トレンド'])
        ax.grid(True, alpha=0.3)
    
    def _plot_composite_and_confidence(self, ax, data: pd.DataFrame, result, config: Dict):
        """複合スコアと信頼度を表示"""
        # 複合スコア
        ax.plot(data.index, result.composite_score, label='複合スコア', 
                color='purple', linewidth=1.5, alpha=0.8)
        
        # 信頼度
        ax.plot(data.index, result.confidence, label='信頼度', 
                color='orange', linewidth=1.5, alpha=0.8)
        
        # 閾値ライン
        threshold = self.ets_indicator.threshold
        ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, 
                   label=f'閾値 ({threshold:.2f})')
        
        # トレンド期間をハイライト
        trend_periods = result.trend_state == 1
        if np.any(trend_periods):
            ax.fill_between(data.index, 0, 1.2, where=trend_periods, 
                           alpha=0.1, color='green', label='トレンド期間')
        
        ax.set_title('複合スコア & 信頼度', fontsize=12, fontweight='bold')
        ax.set_ylabel('スコア', fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _plot_er_and_choppiness(self, ax, data: pd.DataFrame, result, config: Dict):
        """効率比とチョピネス指数を表示"""
        # 効率比
        ax.plot(data.index, result.efficiency_ratio, label='効率比', 
                color='blue', linewidth=1.3, alpha=0.8)
        
        # チョピネス指数（0-1に正規化）
        normalized_chop = result.choppiness_index / 100.0
        ax.plot(data.index, normalized_chop, label='チョピネス指数 (正規化)', 
                color='red', linewidth=1.3, alpha=0.8)
        
        # ボラティリティ係数
        ax.plot(data.index, result.volatility_factor, label='ボラティリティ係数', 
                color='brown', linewidth=1.1, alpha=0.7)
        
        # 基準ライン
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='基準線 (0.5)')
        ax.axhline(y=0.618, color='gold', linestyle=':', alpha=0.7, label='黄金比 (0.618)')
        
        ax.set_title('効率比 & チョピネス指数', fontsize=12, fontweight='bold')
        ax.set_ylabel('値', fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _plot_dynamic_components(self, ax, data: pd.DataFrame, result):
        """動的コンポーネントを表示"""
        # 動的期間
        ax.plot(data.index, result.dynamic_periods, label='動的期間', 
                color='purple', linewidth=1.3, alpha=0.8)
        
        # 基本期間ライン
        base_period = self.ets_indicator.base_period
        ax.axhline(y=base_period, color='gray', linestyle='--', alpha=0.7,
                   label=f'基本期間 ({base_period})')
        
        # 最大・最小サイクルライン
        ax.axhline(y=self.ets_indicator.max_cycle, color='red', linestyle=':', alpha=0.5,
                   label=f'最大サイクル ({self.ets_indicator.max_cycle})')
        ax.axhline(y=self.ets_indicator.min_cycle, color='blue', linestyle=':', alpha=0.5,
                   label=f'最小サイクル ({self.ets_indicator.min_cycle})')
        
        ax.set_title('動的期間', fontsize=12, fontweight='bold')
        ax.set_ylabel('期間', fontsize=10)
        ax.set_xlabel('日付', fontsize=10)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # X軸の日付フォーマット
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(data)//10)))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def run_analysis_optimized(self, show_chart: bool = True) -> None:
        """最適化された分析実行"""
        try:
            self.logger.info("Enhanced Trend State 分析を開始...")
            
            # データ読み込み
            market_data = self.load_market_data()
            
            # トレンド状態計算
            trend_analysis = self.calculate_trend_state(market_data)
            
            # 結果の表示
            stats = trend_analysis['stats']
            
            print("\n" + "="*60)
            print("Enhanced Trend State 分析結果")
            print("="*60)
            print(f"対象期間: {market_data.index.min().strftime('%Y-%m-%d')} ～ {market_data.index.max().strftime('%Y-%m-%d')}")
            print(f"総期間数: {stats['total_periods']}")
            print(f"トレンド期間: {stats['trend_count']} ({stats['trend_percentage']:.1f}%)")
            print(f"レンジ期間: {stats['range_count']} ({stats['range_percentage']:.1f}%)")
            print(f"トレンド切り替え回数: {stats['trend_switches']}")
            print(f"平均複合スコア: {stats['average_composite_score']:.3f}")
            print(f"平均信頼度: {stats['average_confidence']:.3f}")
            print(f"平均効率比: {stats['average_efficiency_ratio']:.3f}")
            print(f"平均チョピネス指数: {stats['average_choppiness_index']:.1f}")
            print(f"現在の状態: {stats['latest_state']}")
            print(f"現在の複合スコア: {stats['latest_composite_score']:.3f}")
            print(f"現在の信頼度: {stats['latest_confidence']:.3f}")
            print("="*60)
            
            # チャート作成
            self.create_chart(market_data, trend_analysis, show_chart)
            
            self.logger.info("分析が完了しました")
            
        except Exception as e:
            self.logger.error(f"分析実行中にエラー: {e}")
            raise


def main():
    """メイン実行関数"""
    try:
        chart_analyzer = EnhancedTrendStateChart()
        chart_analyzer.run_analysis_optimized(show_chart=True)
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()