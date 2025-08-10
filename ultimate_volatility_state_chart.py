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
from indicators.ultimate_volatility_state import UltimateVolatilityState
from logger import get_logger


class UltimateVolatilityStateChart:
    """
    Ultimate Volatility State インジケーターのチャート表示クラス
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
        
        # Ultimate Volatility State インジケーターの初期化
        uvs_config = self.config.get('ultimate_volatility_state', {})
        self.uvs_indicator = UltimateVolatilityState(
            period=uvs_config.get('period', 21),
            threshold=uvs_config.get('threshold', 0.5),
            zscore_period=uvs_config.get('zscore_period', 50),
            src_type=uvs_config.get('src_type', 'hlc3'),
            smoother_period=uvs_config.get('smoother_period', 5),
            adaptive_threshold=uvs_config.get('adaptive_threshold', True)
        )
        
        self.logger.info("Ultimate Volatility State Chart initialized")
    
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
        """ボラティリティ状態を計算"""
        try:
            self.logger.info("ボラティリティ状態の計算を開始...")
            
            # インジケーターの計算
            result = self.uvs_indicator.calculate(data)
            
            # 統計情報の計算
            high_vol_count = np.sum(result.state)
            low_vol_count = np.sum(result.state == 0)
            total_count = len(result.state)
            
            high_vol_percentage = (high_vol_count / total_count * 100) if total_count > 0 else 0
            low_vol_percentage = (low_vol_count / total_count * 100) if total_count > 0 else 0
            
            avg_probability = np.mean(result.probability[result.probability > 0])
            
            stats = {
                'total_periods': total_count,
                'high_volatility_count': high_vol_count,
                'low_volatility_count': low_vol_count,
                'high_volatility_percentage': high_vol_percentage,
                'low_volatility_percentage': low_vol_percentage,
                'average_probability': avg_probability,
                'latest_state': 'High' if result.state[-1] == 1 else 'Low',
                'latest_probability': result.probability[-1]
            }
            
            self.logger.info(f"計算完了 - 高ボラティリティ: {high_vol_percentage:.1f}%, 低ボラティリティ: {low_vol_percentage:.1f}%")
            
            return {
                'result': result,
                'stats': stats
            }
            
        except Exception as e:
            self.logger.error(f"ボラティリティ状態の計算に失敗: {e}")
            raise
    
    def create_chart(self, data: pd.DataFrame, volatility_data: Dict[str, Any], show_chart: bool = True) -> None:
        """チャートを作成"""
        try:
            result = volatility_data['result']
            stats = volatility_data['stats']
            uvs_config = self.config.get('ultimate_volatility_state', {})
            
            # 図のサイズとレイアウト設定
            fig = plt.figure(figsize=(16, 12))
            
            # グリッドレイアウト: 4行1列
            gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.3)
            
            # 1. 価格チャート + ボラティリティ状態
            ax1 = fig.add_subplot(gs[0])
            self._plot_price_with_volatility_state(ax1, data, result, uvs_config)
            
            # 2. ボラティリティ状態バー
            ax2 = fig.add_subplot(gs[1])
            self._plot_volatility_state_bar(ax2, data, result, uvs_config)
            
            # 3. 確率ライン
            ax3 = fig.add_subplot(gs[2])
            self._plot_probability_line(ax3, data, result, uvs_config)
            
            # 4. コンポーネント分析
            if uvs_config.get('show_components', True):
                ax4 = fig.add_subplot(gs[3])
                self._plot_components(ax4, data, result)
            
            # タイトルと統計情報
            title = f"{uvs_config.get('chart_title', 'Ultimate Volatility State Analysis')} - {self.config['data']['symbol']}\n"
            title += f"High Vol: {stats['high_volatility_percentage']:.1f}% | Low Vol: {stats['low_volatility_percentage']:.1f}% | "
            title += f"Latest: {stats['latest_state']} ({stats['latest_probability']:.3f})"
            
            fig.suptitle(title, fontsize=14, fontweight='bold')
            
            # チャートの保存
            if uvs_config.get('save_chart', True):
                filename = uvs_config.get('chart_filename', 'ultimate_volatility_state_analysis')
                output_path = f"{filename}_{self.config['data']['symbol']}_{self.config['data']['timeframe']}.png"
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"チャートを保存しました: {output_path}")
            
            if show_chart:
                plt.show()
            else:
                plt.close(fig)
            
        except Exception as e:
            self.logger.error(f"チャートの作成に失敗: {e}")
            raise
    
    def _plot_price_with_volatility_state(self, ax, data: pd.DataFrame, result, config: Dict):
        """価格チャートとボラティリティ状態を表示"""
        # 価格チャート
        ax.plot(data.index, data['close'], linewidth=1, color='black', label='Price')
        
        # ボラティリティ状態に応じて背景色を変更
        high_vol_periods = result.state == 1
        low_vol_periods = result.state == 0
        
        # 高ボラティリティ期間
        for i in range(len(data)):
            if high_vol_periods[i]:
                ax.axvspan(data.index[i], data.index[min(i+1, len(data)-1)], 
                          alpha=0.2, color=config.get('high_vol_color', 'red'))
        
        # 低ボラティリティ期間
        for i in range(len(data)):
            if low_vol_periods[i]:
                ax.axvspan(data.index[i], data.index[min(i+1, len(data)-1)], 
                          alpha=0.1, color=config.get('low_vol_color', 'blue'))
        
        ax.set_title('Price Chart with Volatility States')
        ax.set_ylabel('Price')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_volatility_state_bar(self, ax, data: pd.DataFrame, result, config: Dict):
        """ボラティリティ状態をバーで表示"""
        colors = [config.get('high_vol_color', 'red') if state == 1 
                 else config.get('low_vol_color', 'blue') for state in result.state]
        
        ax.bar(data.index, result.state, color=colors, alpha=0.7, width=1)
        ax.set_title('Volatility State (1: High, 0: Low)')
        ax.set_ylabel('State')
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)
    
    def _plot_probability_line(self, ax, data: pd.DataFrame, result, config: Dict):
        """確率ラインを表示"""
        ax.plot(data.index, result.probability, 
               color=config.get('probability_color', 'orange'), 
               linewidth=1.5, label='Probability')
        
        # 閾値ライン
        threshold = self.uvs_indicator.threshold
        ax.axhline(y=threshold, color='gray', linestyle='--', alpha=0.5, label=f'Threshold ({threshold})')
        
        ax.set_title('Volatility State Probability')
        ax.set_ylabel('Probability')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_components(self, ax, data: pd.DataFrame, result):
        """コンポーネント分析を表示"""
        components = result.components
        
        if not components:
            ax.text(0.5, 0.5, 'No component data available', 
                   transform=ax.transAxes, ha='center', va='center')
            return
        
        # 主要コンポーネントを選択して表示
        main_components = ['str_zscore', 'vol_zscore', 'acceleration']
        colors = ['blue', 'green', 'red']
        
        for i, comp_name in enumerate(main_components):
            if comp_name in components:
                comp_data = components[comp_name]
                # 正規化して表示
                normalized_data = np.abs(comp_data) / (np.max(np.abs(comp_data)) + 1e-8)
                ax.plot(data.index, normalized_data, 
                       color=colors[i % len(colors)], 
                       alpha=0.7, linewidth=1, label=comp_name)
        
        ax.set_title('Component Analysis (Normalized)')
        ax.set_ylabel('Normalized Value')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def run_analysis(self) -> None:
        """完全な分析を実行"""
        self.run_analysis_optimized(show_chart=True)
    
    def run_analysis_optimized(self, show_chart: bool = True) -> None:
        """最適化された分析実行（表示オプション付き）"""
        try:
            self.logger.info("Ultimate Volatility State 分析を開始...")
            
            # 市場データの読み込み
            data = self.load_market_data()
            
            # ボラティリティ状態の計算
            volatility_data = self.calculate_volatility_state(data)
            
            # 統計情報の表示
            stats = volatility_data['stats']
            self.logger.info("=== 分析結果 ===")
            self.logger.info(f"総期間数: {stats['total_periods']}")
            self.logger.info(f"高ボラティリティ期間: {stats['high_volatility_count']} ({stats['high_volatility_percentage']:.1f}%)")
            self.logger.info(f"低ボラティリティ期間: {stats['low_volatility_count']} ({stats['low_volatility_percentage']:.1f}%)")
            self.logger.info(f"平均確率: {stats['average_probability']:.3f}")
            self.logger.info(f"最新状態: {stats['latest_state']} (確率: {stats['latest_probability']:.3f})")
            
            # チャートの作成
            self.create_chart(data, volatility_data, show_chart)
            
            self.logger.info("分析が完了しました")
            
        except Exception as e:
            self.logger.error(f"分析の実行に失敗: {e}")
            raise


def main():
    """メイン関数"""
    try:
        # チャート分析の実行
        chart_analyzer = UltimateVolatilityStateChart()
        chart_analyzer.run_analysis()
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"実行に失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()