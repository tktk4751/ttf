#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🚀 Ultra Adaptive ATR Channel Chart Visualizer 🚀

Ultra Adaptive ATR Channelインジケーターを実際の相場データで検証し、
美しく高度なチャート描画を行うビジュアライザー

🌟 **表示内容:**
1. **ローソク足チャート**: OHLC価格データ
2. **🧠 Neural Supreme Kalman中心線**: 超低遅延中心線
3. **Supreme適応的バンド**: 上下バンド（トレンド色分け）
4. **🌌 Cosmic Trend**: 宇宙レベルトレンド成分
5. **💥 ブレイクアウトシグナル**: エントリーポイント表示
6. **Supreme改良ATR**: 革新的ATR値
7. **量子コヒーレンス**: Neural Supreme品質指標
8. **統計パネル**: パフォーマンス分析
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# データ取得のための依存関係
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource

# Ultra Adaptive ATR Channelインジケーター
from indicators.ultra_adaptive_atr_channel import UltraAdaptiveATRChannel


class UltraAdaptiveATRChart:
    """
    🚀 Ultra Adaptive ATR Channel Chart Visualizer
    
    Supreme技術統合による究極のチャート描画クラス
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.ultra_atr = None
        self.result = None
        self.fig = None
        self.axes = None
        
        # チャートスタイル設定
        plt.style.use('dark_background')
        
    def load_data_from_config(self, config_path: str) -> pd.DataFrame:
        """
        設定ファイルからデータを読み込む
        
        Args:
            config_path: 設定ファイルのパス
            
        Returns:
            処理済みのデータフレーム
        """
        print("🚀 Supreme Data Loading initiated...")
        
        # 設定ファイルの読み込み
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        # データの準備
        binance_config = config.get('binance_data', {})
        data_dir = binance_config.get('data_dir', 'data/binance')
        binance_data_source = BinanceDataSource(data_dir)
        
        # CSVデータソースはダミーとして渡す
        dummy_csv_source = CSVDataSource("dummy")
        data_loader = DataLoader(
            data_source=dummy_csv_source,
            binance_data_source=binance_data_source
        )
        data_processor = DataProcessor()
        
        # データの読み込みと処理
        print("📊 Loading and processing market data...")
        raw_data = data_loader.load_data_from_config(config)
        processed_data = {
            symbol: data_processor.process(df)
            for symbol, df in raw_data.items()
        }
        
        # 最初のシンボルのデータを取得
        first_symbol = next(iter(processed_data))
        self.data = processed_data[first_symbol]
        
        # データ検証とカラム名の確認
        print(f"📋 Data columns: {list(self.data.columns)}")
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        
        if missing_columns:
            print(f"⚠️ Missing columns: {missing_columns}")
            # カラム名の標準化
            column_mapping = {}
            for col in self.data.columns:
                col_lower = col.lower()
                if 'open' in col_lower:
                    column_mapping[col] = 'open'
                elif 'high' in col_lower:
                    column_mapping[col] = 'high'
                elif 'low' in col_lower:
                    column_mapping[col] = 'low'
                elif 'close' in col_lower:
                    column_mapping[col] = 'close'
                elif 'volume' in col_lower:
                    column_mapping[col] = 'volume'
            
            if column_mapping:
                self.data = self.data.rename(columns=column_mapping)
                print(f"🔧 Renamed columns: {column_mapping}")
        
        print(f"✅ Data loaded successfully: {first_symbol}")
        print(f"📅 Period: {self.data.index.min()} → {self.data.index.max()}")
        print(f"📈 Data points: {len(self.data)}")
        print(f"📋 Final columns: {list(self.data.columns)}")
        
        return self.data

    def calculate_supreme_indicators(self,
                                   # Supreme Core Parameters
                                   price_source: str = 'close',  # hlc3→closeに変更
                                   atr_period: int = 14,
                                   band_multiplier: float = 2.0,
                                   adaptation_factor: float = 0.7,
                                   trend_sensitivity: float = 1.2,
                                   min_trend_strength: float = 0.3,
                                   # 🧠 Neural Supreme Kalman Parameters
                                   kalman_base_process_noise: float = 0.0001,
                                   kalman_base_measurement_noise: float = 0.001,
                                   # 🌌 Ultimate Cosmic Wavelet Parameters
                                   cosmic_power_level: float = 1.5,
                                   # ヒルベルト変換パラメータ
                                   hilbert_algorithm: str = 'quantum_enhanced',
                                   # 適応パラメータ
                                   adaptation_range: float = 1.0,
                                   warmup_periods: Optional[int] = None
                                   ) -> None:
        """
        🌟 Supreme Ultra Adaptive ATR Channelを計算
        """
        if self.data is None:
            raise ValueError("❌ Data not loaded. Run load_data_from_config() first.")
            
        print("\n🧠🌌 Calculating Supreme Ultra Adaptive ATR Channel...")
        print("🔧 Integrating Neural Supreme Kalman + Ultimate Cosmic Wavelet...")
        
        # 🚀 Ultra Adaptive ATR Channel初期化
        self.ultra_atr = UltraAdaptiveATRChannel(
            price_source=price_source,
            atr_period=atr_period,
            band_multiplier=band_multiplier,
            adaptation_factor=adaptation_factor,
            trend_sensitivity=trend_sensitivity,
            min_trend_strength=min_trend_strength,
            # 🧠 Neural Supreme Kalman Parameters
            kalman_base_process_noise=kalman_base_process_noise,
            kalman_base_measurement_noise=kalman_base_measurement_noise,
            # 🌌 Ultimate Cosmic Wavelet Parameters
            cosmic_power_level=cosmic_power_level,
            # ヒルベルト変換パラメータ
            hilbert_algorithm=hilbert_algorithm,
            # 適応パラメータ
            adaptation_range=adaptation_range,
            warmup_periods=warmup_periods
        )
        
        # Supreme計算実行
        print("⚡ Executing Supreme calculation...")
        try:
            self.result = self.ultra_atr.calculate(self.data)
            
            # 結果検証
            center_line = self.ultra_atr.get_center_line()
            bands = self.ultra_atr.get_bands()
            enhanced_atr = self.ultra_atr.get_enhanced_atr()
            trend_info = self.ultra_atr.get_trend_info()
            signals = self.ultra_atr.get_breakout_signals()
            cosmic_trend = self.ultra_atr.get_cosmic_trend()
            quantum_coherence = self.ultra_atr.get_quantum_coherence()
            neural_weights = self.ultra_atr.get_neural_weights()
            
            print(f"✅ Supreme calculation completed!")
            print(f"🧠 Center line points: {len(center_line) if center_line is not None else 0}")
            print(f"📊 Bands calculated: {bands is not None}")
            print(f"⚡ Enhanced ATR points: {len(enhanced_atr) if enhanced_atr is not None else 0}")
            print(f"🧭 Trend analysis: {trend_info is not None}")
            print(f"💥 Breakout signals: {np.sum(np.abs(signals)) if signals is not None else 0}")
            print(f"🌌 Cosmic trend integration: {cosmic_trend is not None}")
            print(f"🔬 Quantum coherence: {quantum_coherence is not None}")
            print(f"🧠 Neural weights: {neural_weights is not None}")
            
            # NaN値のチェック
            if center_line is not None:
                nan_count = np.isnan(center_line).sum()
                print(f"📉 Center line NaN count: {nan_count}")
            
            if bands is not None:
                upper_nan = np.isnan(bands[0]).sum()
                lower_nan = np.isnan(bands[1]).sum()
                print(f"📊 Band NaN count - Upper: {upper_nan}, Lower: {lower_nan}")
            
            if signals is not None:
                long_signals = np.sum(signals == 1)
                short_signals = np.sum(signals == -1)
                print(f"💥 Signals - Long: {long_signals}, Short: {short_signals}")
            
            print("🌟 Supreme indicators calculation completed!")
            
        except Exception as e:
            print(f"❌ Supreme calculation error: {e}")
            import traceback
            traceback.print_exc()
            raise
            
    def plot_supreme_chart(self, 
                          title: str = "🚀 Ultra Adaptive ATR Channel - Supreme Analysis",
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None,
                          show_volume: bool = True,
                          figsize: Tuple[int, int] = (16, 14),
                          style: str = 'nightclouds',
                          savefig: Optional[str] = None,
                          show_signals: bool = True,
                          show_statistics: bool = True) -> None:
        """
        🎨 Supreme Ultra Adaptive ATR Channel Chart描画
        """
        if self.data is None:
            raise ValueError("❌ Data not loaded. Run load_data_from_config() first.")
            
        if self.ultra_atr is None or self.result is None:
            raise ValueError("❌ Indicators not calculated. Run calculate_supreme_indicators() first.")
        
        print("🎨 Generating Supreme chart visualization...")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        print(f"📊 Chart data prepared - Rows: {len(df)}")
        
        # Supreme指標値を取得
        center_line = self.ultra_atr.get_center_line()
        bands = self.ultra_atr.get_bands()
        enhanced_atr = self.ultra_atr.get_enhanced_atr()
        trend_info = self.ultra_atr.get_trend_info()
        signals = self.ultra_atr.get_breakout_signals()
        confidence = self.ultra_atr.get_confidence_scores()
        cosmic_trend = self.ultra_atr.get_cosmic_trend()
        quantum_coherence = self.ultra_atr.get_quantum_coherence()
        neural_weights = self.ultra_atr.get_neural_weights()
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(index=self.data.index)
        
        if center_line is not None:
            full_df['center_line'] = center_line
        if bands is not None:
            full_df['upper_band'] = bands[0]
            full_df['lower_band'] = bands[1]
        if enhanced_atr is not None:
            full_df['enhanced_atr'] = enhanced_atr
        if trend_info is not None:
            full_df['trend_direction'] = trend_info[0]
            full_df['trend_strength'] = trend_info[1]
        if signals is not None:
            full_df['signals'] = signals
        if confidence is not None:
            full_df['confidence'] = confidence
        if cosmic_trend is not None:
            full_df['cosmic_trend'] = cosmic_trend
        if quantum_coherence is not None:
            full_df['quantum_coherence'] = quantum_coherence
        if neural_weights is not None:
            full_df['neural_weights'] = neural_weights
        
        # 絞り込み後のデータに結合
        df = df.join(full_df)
        
        # Supreme表示用データの準備
        # トレンド方向に基づく色分け
        df['center_uptrend'] = np.where(df['trend_direction'] == 1, df['center_line'], np.nan)
        df['center_downtrend'] = np.where(df['trend_direction'] == -1, df['center_line'], np.nan)
        df['center_neutral'] = np.where(df['trend_direction'] == 0, df['center_line'], np.nan)
        
        # バンドの色分け（トレンド方向に基づく）
        df['upper_uptrend'] = np.where(df['trend_direction'] == 1, df['upper_band'], np.nan)
        df['upper_downtrend'] = np.where(df['trend_direction'] == -1, df['upper_band'], np.nan)
        df['lower_uptrend'] = np.where(df['trend_direction'] == 1, df['lower_band'], np.nan)
        df['lower_downtrend'] = np.where(df['trend_direction'] == -1, df['lower_band'], np.nan)
        
        # シグナルポイントの準備
        if show_signals and 'signals' in df.columns:
            df['long_signals'] = np.where(df['signals'] == 1, df['close'], np.nan)
            df['short_signals'] = np.where(df['signals'] == -1, df['close'], np.nan)
        
        print(f"🎯 Chart data validation:")
        print(f"  - Center line valid points: {df['center_line'].notna().sum()}")
        if 'upper_band' in df.columns:
            print(f"  - Upper band valid points: {df['upper_band'].notna().sum()}")
            print(f"  - Lower band valid points: {df['lower_band'].notna().sum()}")
        if 'signals' in df.columns:
            print(f"  - Long signals: {(df['signals'] == 1).sum()}")
            print(f"  - Short signals: {(df['signals'] == -1).sum()}")
        
        # mplfinanceプロット設定
        main_plots = []
        
        # 🧠 Neural Supreme Kalman中心線（トレンド色分け）
        if 'center_uptrend' in df.columns:
            main_plots.append(mpf.make_addplot(df['center_uptrend'], color='#00FF88', width=2.5, label='🧠 Neural Center (Up)'))
            main_plots.append(mpf.make_addplot(df['center_downtrend'], color='#FF4444', width=2.5, label='🧠 Neural Center (Down)'))
            main_plots.append(mpf.make_addplot(df['center_neutral'], color='#FFAA00', width=2, label='🧠 Neural Center (Neutral)'))
        
        # Supreme適応的バンド
        if 'upper_band' in df.columns:
            # NaN値のチェックと処理
            upper_uptrend_valid = df['upper_uptrend'].dropna()
            upper_downtrend_valid = df['upper_downtrend'].dropna()
            lower_uptrend_valid = df['lower_uptrend'].dropna()
            lower_downtrend_valid = df['lower_downtrend'].dropna()
            
            if len(upper_uptrend_valid) > 0:
                main_plots.append(mpf.make_addplot(df['upper_uptrend'], color='#00FF88', width=1.5, alpha=0.7, label='Upper Band (Up)'))
            if len(upper_downtrend_valid) > 0:
                main_plots.append(mpf.make_addplot(df['upper_downtrend'], color='#FF4444', width=1.5, alpha=0.7, label='Upper Band (Down)'))
            if len(lower_uptrend_valid) > 0:
                main_plots.append(mpf.make_addplot(df['lower_uptrend'], color='#00FF88', width=1.5, alpha=0.7, label='Lower Band (Up)'))
            if len(lower_downtrend_valid) > 0:
                main_plots.append(mpf.make_addplot(df['lower_downtrend'], color='#FF4444', width=1.5, alpha=0.7, label='Lower Band (Down)'))
        
        # 💥 ブレイクアウトシグナル（有効な場合のみ）
        if show_signals and 'signals' in df.columns:
            long_signals_valid = df['long_signals'].dropna()
            short_signals_valid = df['short_signals'].dropna()
            
            if len(long_signals_valid) > 0:
                main_plots.append(mpf.make_addplot(df['long_signals'], type='scatter', markersize=100, 
                                                  marker='^', color='#00FF00', alpha=0.8, label='💥 Long Signal'))
            if len(short_signals_valid) > 0:
                main_plots.append(mpf.make_addplot(df['short_signals'], type='scatter', markersize=100, 
                                                  marker='v', color='#FF0000', alpha=0.8, label='💥 Short Signal'))
        
        # サブプロット設定
        subplot_panels = []
        panel_idx = 1 if show_volume else 0
        
        # 🌌 Cosmic Trend & Trend Strength
        if 'cosmic_trend' in df.columns and 'trend_strength' in df.columns:
            panel_idx += 1
            cosmic_valid = df['cosmic_trend'].dropna()
            strength_valid = df['trend_strength'].dropna()
            
            if len(cosmic_valid) > 0:
                cosmic_plot = mpf.make_addplot(df['cosmic_trend'], panel=panel_idx, color='#9966FF', width=2, 
                                              ylabel='🌌 Cosmic Trend', secondary_y=False, label='Cosmic Trend')
                subplot_panels.append(cosmic_plot)
            
            if len(strength_valid) > 0:
                strength_plot = mpf.make_addplot(df['trend_strength'], panel=panel_idx, color='#66FFFF', width=1.5, 
                                               secondary_y=True, label='Trend Strength')
                subplot_panels.append(strength_plot)
        
        # 🔬 Quantum Coherence & Neural Weights
        if 'quantum_coherence' in df.columns and 'neural_weights' in df.columns:
            panel_idx += 1
            quantum_valid = df['quantum_coherence'].dropna()
            neural_valid = df['neural_weights'].dropna()
            
            if len(quantum_valid) > 0:
                quantum_plot = mpf.make_addplot(df['quantum_coherence'], panel=panel_idx, color='#FF66FF', width=2, 
                                               ylabel='🔬 Quantum Coherence', secondary_y=False, label='Quantum Coherence')
                subplot_panels.append(quantum_plot)
            
            if len(neural_valid) > 0:
                neural_plot = mpf.make_addplot(df['neural_weights'], panel=panel_idx, color='#FFFF66', width=1.5, 
                                              secondary_y=True, label='Neural Weights')
                subplot_panels.append(neural_plot)
        
        # ⚡ Enhanced ATR & Confidence
        if 'enhanced_atr' in df.columns and 'confidence' in df.columns:
            panel_idx += 1
            atr_valid = df['enhanced_atr'].dropna()
            confidence_valid = df['confidence'].dropna()
            
            if len(atr_valid) > 0:
                atr_plot = mpf.make_addplot(df['enhanced_atr'], panel=panel_idx, color='#66FF66', width=2, 
                                           ylabel='⚡ Enhanced ATR', secondary_y=False, label='Enhanced ATR')
                subplot_panels.append(atr_plot)
            
            if len(confidence_valid) > 0:
                confidence_plot = mpf.make_addplot(df['confidence'], panel=panel_idx, color='#6666FF', width=1.5, 
                                                 secondary_y=True, label='Confidence')
                subplot_panels.append(confidence_plot)
        
        # すべてのプロットを結合（空でない場合のみ）
        all_plots = main_plots
        if subplot_panels:
            all_plots.extend(subplot_panels)
        
        # パネル比率の設定
        num_subplots = len(subplot_panels) // 2 if len(subplot_panels) > 0 else 0
        
        if show_volume:
            if num_subplots == 3:
                panel_ratios = (6, 1, 1.5, 1.5, 1.5)
            elif num_subplots == 2:
                panel_ratios = (6, 1, 1.5, 1.5)
            elif num_subplots == 1:
                panel_ratios = (6, 1, 1.5)
            else:
                panel_ratios = (4, 1)
        else:
            if num_subplots == 3:
                panel_ratios = (6, 1.5, 1.5, 1.5)
            elif num_subplots == 2:
                panel_ratios = (6, 1.5, 1.5)
            elif num_subplots == 1:
                panel_ratios = (6, 1.5)
            else:
                panel_ratios = (4,)
        
        # mplfinance設定
        kwargs = dict(
            type='candle',
            figsize=figsize,
            title=title,
            style=style,
            datetime_format='%Y-%m-%d',
            xrotation=45,
            volume=show_volume,
            returnfig=True,
            warn_too_much_data=1000
        )
        
        # プロットが存在する場合のみ追加
        if all_plots:
            kwargs['addplot'] = all_plots
            kwargs['panel_ratios'] = panel_ratios
        
        try:
            # プロット実行
            print("🖼️ Rendering Supreme chart...")
            fig, axes = mpf.plot(df, **kwargs)
            
            # 軸の調整とグリッドの追加
            axes[0].grid(True, alpha=0.3)
            if all_plots:  # プロットがある場合のみ凡例を追加
                axes[0].legend(loc='upper left', fontsize=8)
            
            # サブプロットの調整
            if len(axes) > 1:
                for i, ax in enumerate(axes[1:], 1):
                    ax.grid(True, alpha=0.3)
                    
                    # 参照線の追加
                    if i == (1 + (1 if show_volume else 0)) and 'cosmic_trend' in df.columns:
                        ax.axhline(y=0.5, color='white', linestyle='--', alpha=0.5)
                        ax.axhline(y=0.0, color='white', linestyle='-', alpha=0.3)
                        ax.axhline(y=1.0, color='white', linestyle='--', alpha=0.5)
                    
                    if i == (2 + (1 if show_volume else 0)) and 'quantum_coherence' in df.columns:
                        ax.axhline(y=0.5, color='white', linestyle='--', alpha=0.5)
                        ax.axhline(y=0.8, color='green', linestyle=':', alpha=0.7)
                        ax.axhline(y=0.2, color='red', linestyle=':', alpha=0.7)
            
            self.fig = fig
            self.axes = axes
            
            # 統計情報の表示
            if show_statistics:
                self._display_supreme_statistics(df)
            
            # 保存または表示
            if savefig:
                plt.savefig(savefig, dpi=300, bbox_inches='tight', facecolor='black')
                print(f"💾 Supreme chart saved: {savefig}")
            else:
                plt.tight_layout()
                plt.show()
                
            print("🌟 Supreme chart visualization completed!")
            
        except Exception as e:
            print(f"❌ Chart rendering error: {e}")
            import traceback
            traceback.print_exc()
            # 簡易バックアップチャート
            self._render_simple_chart(df, title, figsize, savefig)
    
    def _render_simple_chart(self, df: pd.DataFrame, title: str, figsize: Tuple[int, int], savefig: Optional[str]):
        """簡易チャート描画（エラー時のバックアップ）"""
        print("🔧 Rendering simple backup chart...")
        try:
            fig, ax = plt.subplots(figsize=figsize, facecolor='black')
            ax.set_facecolor('black')
            
            # 基本的なローソク足チャート
            from matplotlib.patches import Rectangle
            from matplotlib.lines import Line2D
            
            for i, (timestamp, row) in enumerate(df.iterrows()):
                if i % 50 == 0:  # 間引いて表示
                    open_price = row['open']
                    high_price = row['high']
                    low_price = row['low']
                    close_price = row['close']
                    
                    color = 'green' if close_price >= open_price else 'red'
                    ax.plot([i, i], [low_price, high_price], color=color, linewidth=1)
                    ax.plot([i, i], [open_price, close_price], color=color, linewidth=3)
            
            # 中心線の追加
            if 'center_line' in df.columns:
                center_valid = df['center_line'].dropna()
                if len(center_valid) > 0:
                    ax.plot(range(len(df)), df['center_line'], color='yellow', linewidth=2, label='Neural Center')
            
            ax.set_title(title, color='white', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if savefig:
                plt.savefig(savefig, dpi=150, bbox_inches='tight', facecolor='black')
                print(f"💾 Simple chart saved: {savefig}")
            else:
                plt.show()
                
        except Exception as e:
            print(f"❌ Simple chart error: {e}")
    
    def _display_supreme_statistics(self, df: pd.DataFrame) -> None:
        """📊 Supreme統計情報を表示"""
        print(f"\n{'='*60}")
        print(f"📊 SUPREME ULTRA ADAPTIVE ATR CHANNEL STATISTICS 📊")
        print(f"{'='*60}")
        
        # 基本統計
        total_points = len(df)
        valid_signals = df['signals'].notna().sum() if 'signals' in df.columns else 0
        
        print(f"📈 Total data points: {total_points}")
        print(f"⚡ Valid signal points: {valid_signals}")
        
        # トレンド統計
        if 'trend_direction' in df.columns:
            uptrend_points = (df['trend_direction'] == 1).sum()
            downtrend_points = (df['trend_direction'] == -1).sum()
            neutral_points = (df['trend_direction'] == 0).sum()
            
            print(f"\n🧭 TREND ANALYSIS:")
            print(f"  📈 Uptrend: {uptrend_points} ({uptrend_points/total_points*100:.1f}%)")
            print(f"  📉 Downtrend: {downtrend_points} ({downtrend_points/total_points*100:.1f}%)")
            print(f"  ➡️  Neutral: {neutral_points} ({neutral_points/total_points*100:.1f}%)")
        
        # シグナル統計
        if 'signals' in df.columns:
            long_signals = (df['signals'] == 1).sum()
            short_signals = (df['signals'] == -1).sum()
            
            print(f"\n💥 BREAKOUT SIGNALS:")
            print(f"  🟢 Long signals: {long_signals}")
            print(f"  🔴 Short signals: {short_signals}")
            if (long_signals + short_signals) > 0:
                signal_frequency = (long_signals + short_signals) / total_points * 100
                print(f"  📊 Signal frequency: {signal_frequency:.2f}%")
        
        # Supreme指標統計
        if 'enhanced_atr' in df.columns:
            atr_mean = df['enhanced_atr'].mean()
            atr_std = df['enhanced_atr'].std()
            print(f"\n⚡ ENHANCED ATR:")
            print(f"  📊 Average: {atr_mean:.4f}")
            print(f"  📈 Std Dev: {atr_std:.4f}")
        
        if 'quantum_coherence' in df.columns:
            qc_mean = df['quantum_coherence'].mean()
            qc_min = df['quantum_coherence'].min()
            qc_max = df['quantum_coherence'].max()
            print(f"\n🔬 QUANTUM COHERENCE:")
            print(f"  📊 Average: {qc_mean:.3f}")
            print(f"  📉 Range: {qc_min:.3f} - {qc_max:.3f}")
        
        if 'cosmic_trend' in df.columns:
            ct_mean = df['cosmic_trend'].mean()
            ct_trend = "🔼 Bullish" if ct_mean > 0.5 else "🔽 Bearish" if ct_mean < 0.5 else "➡️ Neutral"
            print(f"\n🌌 COSMIC TREND:")
            print(f"  📊 Average: {ct_mean:.3f} {ct_trend}")
        
        if 'neural_weights' in df.columns:
            nw_mean = df['neural_weights'].mean()
            print(f"\n🧠 NEURAL WEIGHTS:")
            print(f"  📊 Average: {nw_mean:.3f}")
        
        if 'confidence' in df.columns:
            conf_mean = df['confidence'].mean()
            conf_high = (df['confidence'] > 0.7).sum()
            print(f"\n🎯 CONFIDENCE SCORES:")
            print(f"  📊 Average: {conf_mean:.3f}")
            print(f"  ✨ High confidence points: {conf_high} ({conf_high/total_points*100:.1f}%)")
        
        # Supreme解析サマリー
        try:
            summary = self.ultra_atr.get_supreme_analysis_summary()
            if summary:
                print(f"\n🚀 SUPREME ANALYSIS SUMMARY:")
                print(f"  🎯 Algorithm: {summary.get('algorithm', 'N/A')}")
                print(f"  ⚡ Status: {summary.get('status', 'N/A')}")
                
                metrics = summary.get('performance_metrics', {})
                if metrics:
                    print(f"  📊 Performance Metrics:")
                    for key, value in metrics.items():
                        print(f"    - {key}: {value:.4f}")
        except Exception as e:
            print(f"  ⚠️ Summary error: {e}")
        
        print(f"{'='*60}\n")
    
    def get_supreme_summary(self) -> Dict[str, Any]:
        """🌟 Supreme解析サマリーを取得"""
        if self.ultra_atr is None:
            return {}
        
        try:
            return self.ultra_atr.get_supreme_analysis_summary()
        except:
            return {}


def main():
    """🚀 Supreme メイン関数"""
    import argparse
    parser = argparse.ArgumentParser(description='🚀 Ultra Adaptive ATR Channel Supreme Visualization')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='Configuration file path')
    parser.add_argument('--start', '-s', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='Output file path')
    parser.add_argument('--price-source', type=str, default='close', help='Price source type')
    parser.add_argument('--atr-period', type=int, default=14, help='ATR period')
    parser.add_argument('--band-multiplier', type=float, default=2.0, help='Band multiplier')
    parser.add_argument('--adaptation-factor', type=float, default=0.7, help='Adaptation factor')
    parser.add_argument('--cosmic-power', type=float, default=1.5, help='🌌 Cosmic power level')
    parser.add_argument('--trend-sensitivity', type=float, default=1.2, help='Trend sensitivity')
    parser.add_argument('--no-volume', action='store_true', help='Hide volume')
    parser.add_argument('--no-signals', action='store_true', help='Hide signals')
    parser.add_argument('--no-stats', action='store_true', help='Hide statistics')
    args = parser.parse_args()
    
    print("🚀🧠🌌 SUPREME ULTRA ADAPTIVE ATR CHANNEL ANALYZER 🌌🧠🚀")
    print("=" * 70)
    
    try:
        # Supreme Chart作成
        chart = UltraAdaptiveATRChart()
        chart.load_data_from_config(args.config)
        chart.calculate_supreme_indicators(
            price_source=args.price_source,
            atr_period=args.atr_period,
            band_multiplier=args.band_multiplier,
            adaptation_factor=args.adaptation_factor,
            cosmic_power_level=args.cosmic_power,
            trend_sensitivity=args.trend_sensitivity
        )
        chart.plot_supreme_chart(
            start_date=args.start,
            end_date=args.end,
            show_volume=not args.no_volume,
            show_signals=not args.no_signals,
            show_statistics=not args.no_stats,
            savefig=args.output
        )
        
        # Supreme Summary表示
        summary = chart.get_supreme_summary()
        if summary:
            print(f"\n🌟 SUPREME ANALYSIS COMPLETED SUCCESSFULLY! 🌟")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 