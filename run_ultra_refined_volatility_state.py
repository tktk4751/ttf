#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource
from indicators.ultra_refined_volatility_state import UltraRefinedVolatilityState
from logger import get_logger


class UltraRefinedVolatilityAnalyzer:
    """
    超洗練されたボラティリティ状態分析システム
    最先端のデジタル信号処理技術による高精度ボラティリティ判定
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
        
        # 超洗練されたボラティリティ状態インジケーターの初期化
        self.vol_indicator = UltraRefinedVolatilityState(
            str_period=14,
            lookback_period=100,
            hilbert_smooth=4,
            wavelet_scales=2,
            entropy_window=16,
            fractal_k=8,
            sensitivity=2.0,
            confidence_threshold=0.7,
            src_type='hlc3',
            smoothing=True
        )
        
        self.logger.info("Ultra-Refined Volatility State Analyzer initialized")
        self.logger.info("🧠 Advanced DSP Features:")
        self.logger.info("   - STR-based ultra-low latency measurement")
        self.logger.info("   - Hilbert transform for envelope/phase analysis")
        self.logger.info("   - Discrete wavelet transform for multi-resolution analysis")
        self.logger.info("   - Spectral entropy for complexity measurement")
        self.logger.info("   - Fractal dimension for self-similarity analysis")
        self.logger.info("   - Adaptive Kalman filtering for noise reduction")
        self.logger.info("   - Adaptive threshold with dynamic market adjustment")
        self.logger.info("   - Confidence-based judgment system")
    
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
    
    def run_ultra_refined_analysis(self, show_chart: bool = True) -> dict:
        """超洗練されたボラティリティ分析の実行"""
        try:
            self.logger.info("🚀 超洗練されたボラティリティ状態分析を開始...")
            
            # データ読み込み
            data = self.load_market_data()
            
            # 超洗練されたボラティリティ状態計算
            self.logger.info("🧠 最先端DSP技術による分析を実行中...")
            self.logger.info("   ⚡ STRベース超低遅延測定")
            self.logger.info("   🌊 ヒルベルト変換による包絡線・位相解析")
            self.logger.info("   📊 ウェーブレット変換による多解像度解析")
            self.logger.info("   🔍 スペクトラルエントロピーによる複雑性測定")
            self.logger.info("   📐 フラクタル次元による自己相似性分析")
            self.logger.info("   🎯 適応カルマンフィルタによるノイズ除去")
            self.logger.info("   🎛️ 適応的閾値による動的市場調整")
            self.logger.info("   ✨ 信頼度ベース判定システム")
            
            result = self.vol_indicator.calculate(data)
            
            # 詳細統計の計算
            stats = self._calculate_ultra_refined_stats(result)
            
            # 結果の表示
            self._display_results(stats, result)
            
            # チャートの作成
            if show_chart:
                self._create_ultra_refined_chart(data, result, stats)
            
            return {'data': data, 'result': result, 'stats': stats}
            
        except Exception as e:
            self.logger.error(f"超洗練されたボラティリティ分析の実行に失敗: {e}")
            raise
    
    def _calculate_ultra_refined_stats(self, result) -> dict:
        """超洗練された統計分析"""
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
        
        # 確率と信頼度統計
        valid_prob = result.probability[result.probability > 0]
        valid_conf = result.confidence[result.confidence > 0]
        
        # DSP特徴量統計
        valid_hilbert = result.hilbert_envelope[result.hilbert_envelope > 0]
        valid_wavelet = result.wavelet_energy[result.wavelet_energy > 0]
        valid_entropy = result.spectral_entropy[result.spectral_entropy > 0]
        valid_fractal = result.fractal_dimension[result.fractal_dimension > 0]
        valid_freq = result.instantaneous_frequency[result.instantaneous_frequency > 0]
        
        # 信頼度統計
        high_confidence_count = np.sum(result.confidence > 0.8)
        medium_confidence_count = np.sum((result.confidence > 0.6) & (result.confidence <= 0.8))
        low_confidence_count = np.sum(result.confidence <= 0.6)
        
        # 適応的閾値統計
        valid_threshold = result.adaptive_threshold[result.adaptive_threshold > 0]
        
        # 品質指標の取得
        quality_metrics = self.vol_indicator.get_signal_quality_metrics()
        current_confidence = self.vol_indicator.get_current_confidence()
        
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
            'average_probability': np.mean(valid_prob) if len(valid_prob) > 0 else 0,
            'average_confidence': np.mean(valid_conf) if len(valid_conf) > 0 else 0,
            'latest_state': 'High' if result.state[-1] == 1 else 'Low',
            'latest_probability': result.probability[-1],
            'latest_confidence': result.confidence[-1],
            'current_confidence': current_confidence,
            'quality_metrics': quality_metrics,
            
            # DSP特徴量統計
            'hilbert_envelope_avg': np.mean(valid_hilbert) if len(valid_hilbert) > 0 else 0,
            'hilbert_envelope_std': np.std(valid_hilbert) if len(valid_hilbert) > 0 else 0,
            'wavelet_energy_avg': np.mean(valid_wavelet) if len(valid_wavelet) > 0 else 0,
            'wavelet_energy_std': np.std(valid_wavelet) if len(valid_wavelet) > 0 else 0,
            'spectral_entropy_avg': np.mean(valid_entropy) if len(valid_entropy) > 0 else 0,
            'spectral_entropy_std': np.std(valid_entropy) if len(valid_entropy) > 0 else 0,
            'fractal_dimension_avg': np.mean(valid_fractal) if len(valid_fractal) > 0 else 0,
            'fractal_dimension_std': np.std(valid_fractal) if len(valid_fractal) > 0 else 0,
            'instantaneous_freq_avg': np.mean(valid_freq) if len(valid_freq) > 0 else 0,
            'instantaneous_freq_std': np.std(valid_freq) if len(valid_freq) > 0 else 0,
            'adaptive_threshold_avg': np.mean(valid_threshold) if len(valid_threshold) > 0 else 0,
            'adaptive_threshold_std': np.std(valid_threshold) if len(valid_threshold) > 0 else 0,
            
            # 信頼度分布
            'high_confidence_percentage': (high_confidence_count / total_periods * 100),
            'medium_confidence_percentage': (medium_confidence_count / total_periods * 100),
            'low_confidence_percentage': (low_confidence_count / total_periods * 100),
            
            # 最新値
            'latest_str': result.str_values[-1],
            'latest_hilbert_envelope': result.hilbert_envelope[-1],
            'latest_hilbert_phase': result.hilbert_phase[-1],
            'latest_instantaneous_freq': result.instantaneous_frequency[-1],
            'latest_wavelet_energy': result.wavelet_energy[-1],
            'latest_spectral_entropy': result.spectral_entropy[-1],
            'latest_fractal_dimension': result.fractal_dimension[-1],
            'latest_adaptive_threshold': result.adaptive_threshold[-1],
            'latest_adaptive_gain': result.adaptive_gain[-1],
            'latest_raw_score': result.raw_score[-1]
        }
    
    def _display_results(self, stats: dict, result) -> None:
        """結果の詳細表示"""
        self.logger.info("\n" + "="*100)
        self.logger.info("🚀 超洗練されたボラティリティ状態分析結果")
        self.logger.info("="*100)
        
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
        
        self.logger.info(f"\n✨ 信頼度分析:")
        self.logger.info(f"   平均信頼度: {stats['average_confidence']:.3f}")
        self.logger.info(f"   最新信頼度: {stats['latest_confidence']:.3f}")
        self.logger.info(f"   高信頼度(>0.8): {stats['high_confidence_percentage']:.1f}%")
        self.logger.info(f"   中信頼度(0.6-0.8): {stats['medium_confidence_percentage']:.1f}%")
        self.logger.info(f"   低信頼度(<=0.6): {stats['low_confidence_percentage']:.1f}%")
        
        self.logger.info(f"\n🧠 DSP特徴量分析:")
        self.logger.info(f"   最新STR値: {stats['latest_str']:.6f}")
        self.logger.info(f"   ヒルベルト包絡線: {stats['latest_hilbert_envelope']:.6f} (平均: {stats['hilbert_envelope_avg']:.6f})")
        self.logger.info(f"   瞬間周波数: {stats['latest_instantaneous_freq']:.6f} (平均: {stats['instantaneous_freq_avg']:.6f})")
        self.logger.info(f"   ウェーブレットエネルギー: {stats['latest_wavelet_energy']:.6f} (平均: {stats['wavelet_energy_avg']:.6f})")
        self.logger.info(f"   スペクトラルエントロピー: {stats['latest_spectral_entropy']:.6f} (平均: {stats['spectral_entropy_avg']:.6f})")
        self.logger.info(f"   フラクタル次元: {stats['latest_fractal_dimension']:.6f} (平均: {stats['fractal_dimension_avg']:.6f})")
        
        self.logger.info(f"\n🎛️ 適応システム:")
        self.logger.info(f"   最新適応閾値: {stats['latest_adaptive_threshold']:.3f}")
        self.logger.info(f"   適応ゲイン: {stats['latest_adaptive_gain']:.3f}")
        self.logger.info(f"   生スコア: {stats['latest_raw_score']:.3f}")
        
        self.logger.info(f"\n🎯 現在の状況:")
        self.logger.info(f"   状態: {stats['latest_state']} ボラティリティ")
        self.logger.info(f"   確率: {stats['latest_probability']:.3f}")
        
        if stats['quality_metrics']:
            qm = stats['quality_metrics']
            self.logger.info(f"\n📊 品質指標:")
            self.logger.info(f"   平均信頼度: {qm['avg_confidence']:.3f}")
            self.logger.info(f"   信頼度安定性: {qm['confidence_stability']:.3f}")
            self.logger.info(f"   平均複雑性: {qm['avg_complexity']:.3f}")
            self.logger.info(f"   平均フラクタル次元: {qm['avg_fractal_dimension']:.3f}")
            self.logger.info(f"   高信頼度比率: {qm['high_confidence_ratio']:.3f}")
        
        # 実用性評価
        if 20 <= stats['high_volatility_percentage'] <= 40:
            self.logger.info("\n✅ 理想的なボラティリティ分布")
        elif 15 <= stats['high_volatility_percentage'] <= 50:
            self.logger.info("\n✅ 実用的なボラティリティ分布")
        else:
            self.logger.info("\n⚠️ パラメータ調整を検討してください")
        
        # 信頼度評価
        if stats['high_confidence_percentage'] > 60:
            self.logger.info("✅ 優秀な信頼度分布")
        elif stats['high_confidence_percentage'] > 40:
            self.logger.info("📊 良好な信頼度分布")
        else:
            self.logger.info("⚠️ 信頼度向上が必要")
    
    def _create_ultra_refined_chart(self, data, result, stats) -> None:
        """超洗練されたチャートの作成"""
        try:
            symbol = self.config.get('binance_data', {}).get('symbol', 'Unknown')
            timeframe = self.config.get('binance_data', {}).get('timeframe', 'Unknown')
            
            fig = plt.figure(figsize=(20, 28))
            gs = fig.add_gridspec(12, 1, height_ratios=[3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], hspace=0.4)
            
            # 1. 価格チャート with ボラティリティ状態
            ax1 = fig.add_subplot(gs[0])
            ax1.plot(data.index, data['close'], linewidth=1, color='black', label='Price')
            
            # 信頼度ベースの背景色
            for i in range(len(data)):
                confidence = result.confidence[i]
                alpha = 0.1 + 0.4 * confidence
                if result.state[i] == 1:
                    color = 'red' if confidence > 0.7 else 'orange'
                else:
                    color = 'blue' if confidence > 0.7 else 'lightblue'
                ax1.axvspan(data.index[i], data.index[min(i+1, len(data)-1)], 
                           alpha=alpha, color=color)
            
            title = f'Ultra-Refined DSP Volatility Analysis - {symbol} ({timeframe})\n'
            title += f'High Vol: {stats["high_volatility_percentage"]:.1f}% | High Confidence: {stats["high_confidence_percentage"]:.1f}%'
            ax1.set_title(title, fontsize=14, fontweight='bold')
            ax1.set_ylabel('Price')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # 2. ボラティリティ状態バー
            ax2 = fig.add_subplot(gs[1])
            colors = ['red' if s == 1 else 'blue' for s in result.state]
            ax2.bar(data.index, result.state, color=colors, alpha=0.7, width=1)
            ax2.set_title('Ultra-Refined Volatility State (1: High, 0: Low)')
            ax2.set_ylabel('State')
            ax2.set_ylim(-0.1, 1.1)
            ax2.grid(True, alpha=0.3)
            
            # 3. STR値（基本特徴量）
            ax3 = fig.add_subplot(gs[2])
            ax3.plot(data.index, result.str_values, color='green', linewidth=1.2, label='STR Values')
            ax3.set_title('STR - Ultra-Low Latency Base Feature')
            ax3.set_ylabel('STR')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            # 4. ヒルベルト包絡線と瞬間周波数
            ax4 = fig.add_subplot(gs[3])
            ax4.plot(data.index, result.hilbert_envelope, color='purple', linewidth=1.3, label='Hilbert Envelope')
            ax4_twin = ax4.twinx()
            ax4_twin.plot(data.index, result.instantaneous_frequency, color='orange', linewidth=1, alpha=0.7, label='Instantaneous Frequency')
            ax4.set_title('Hilbert Transform - Envelope & Instantaneous Frequency')
            ax4.set_ylabel('Envelope')
            ax4_twin.set_ylabel('Frequency')
            ax4.grid(True, alpha=0.3)
            
            # 凡例を統合
            lines1, labels1 = ax4.get_legend_handles_labels()
            lines2, labels2 = ax4_twin.get_legend_handles_labels()
            ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # 5. ウェーブレットエネルギー
            ax5 = fig.add_subplot(gs[4])
            ax5.plot(data.index, result.wavelet_energy, color='darkred', linewidth=1.3, label='Wavelet Energy')
            ax5.set_title('Discrete Wavelet Transform - Multi-Resolution Energy')
            ax5.set_ylabel('Energy')
            ax5.grid(True, alpha=0.3)
            ax5.legend()
            
            # 6. スペクトラルエントロピー
            ax6 = fig.add_subplot(gs[5])
            ax6.plot(data.index, result.spectral_entropy, color='darkblue', linewidth=1.3, label='Spectral Entropy')
            ax6.set_title('Spectral Entropy - Signal Complexity')
            ax6.set_ylabel('Entropy')
            ax6.set_ylim(0, 1)
            ax6.grid(True, alpha=0.3)
            ax6.legend()
            
            # 7. フラクタル次元
            ax7 = fig.add_subplot(gs[6])
            ax7.plot(data.index, result.fractal_dimension, color='brown', linewidth=1.3, label='Fractal Dimension')
            ax7.axhline(y=1.5, color='gray', linestyle='--', alpha=0.5, label='Mid-level')
            ax7.set_title('Fractal Dimension - Self-Similarity Analysis')
            ax7.set_ylabel('Dimension')
            ax7.set_ylim(1.0, 2.0)
            ax7.grid(True, alpha=0.3)
            ax7.legend()
            
            # 8. 適応的閾値とカルマンゲイン
            ax8 = fig.add_subplot(gs[7])
            ax8.plot(data.index, result.adaptive_threshold, color='magenta', linewidth=1.3, label='Adaptive Threshold')
            ax8_twin = ax8.twinx()
            ax8_twin.plot(data.index, result.adaptive_gain, color='cyan', linewidth=1, alpha=0.7, label='Kalman Gain')
            ax8.set_title('Adaptive Threshold & Kalman Gain')
            ax8.set_ylabel('Threshold')
            ax8_twin.set_ylabel('Gain')
            ax8.grid(True, alpha=0.3)
            
            # 凡例を統合
            lines1, labels1 = ax8.get_legend_handles_labels()
            lines2, labels2 = ax8_twin.get_legend_handles_labels()
            ax8.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # 9. 信頼度と確率
            ax9 = fig.add_subplot(gs[8])
            ax9.plot(data.index, result.confidence, color='darkgreen', linewidth=1.3, label='Confidence')
            ax9_twin = ax9.twinx()
            ax9_twin.plot(data.index, result.probability, color='red', linewidth=1.2, alpha=0.8, label='Probability')
            ax9.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='High Confidence')
            ax9_twin.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            ax9.set_title('Confidence & Probability')
            ax9.set_ylabel('Confidence')
            ax9_twin.set_ylabel('Probability')
            ax9.set_ylim(0, 1)
            ax9_twin.set_ylim(0, 1)
            ax9.grid(True, alpha=0.3)
            
            # 凡例を統合
            lines1, labels1 = ax9.get_legend_handles_labels()
            lines2, labels2 = ax9_twin.get_legend_handles_labels()
            ax9.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # 10. 生スコア
            ax10 = fig.add_subplot(gs[9])
            ax10.plot(data.index, result.raw_score, color='darkred', linewidth=1.5, label='Raw Score')
            ax10.axhline(y=0.5, color='black', linestyle='-', alpha=0.5, label='Neutral')
            ax10.fill_between(data.index, 0, 1, where=(result.raw_score > result.adaptive_threshold), 
                             color='red', alpha=0.1, label='Above Threshold')
            ax10.fill_between(data.index, 0, 1, where=(result.raw_score <= result.adaptive_threshold), 
                             color='blue', alpha=0.1, label='Below Threshold')
            ax10.set_title('Raw Volatility Score vs Adaptive Threshold')
            ax10.set_ylabel('Score')
            ax10.set_ylim(0, 1)
            ax10.grid(True, alpha=0.3)
            ax10.legend()
            
            # 11. DSP特徴量の統合表示
            ax11 = fig.add_subplot(gs[10])
            # 正規化して表示
            norm_envelope = result.hilbert_envelope / np.max(result.hilbert_envelope) if np.max(result.hilbert_envelope) > 0 else result.hilbert_envelope
            norm_wavelet = result.wavelet_energy / np.max(result.wavelet_energy) if np.max(result.wavelet_energy) > 0 else result.wavelet_energy
            norm_freq = result.instantaneous_frequency / np.max(result.instantaneous_frequency) if np.max(result.instantaneous_frequency) > 0 else result.instantaneous_frequency
            
            ax11.plot(data.index, norm_envelope, color='purple', alpha=0.7, linewidth=1, label='Hilbert Envelope (norm)')
            ax11.plot(data.index, norm_wavelet, color='red', alpha=0.7, linewidth=1, label='Wavelet Energy (norm)')
            ax11.plot(data.index, norm_freq, color='orange', alpha=0.7, linewidth=1, label='Inst. Frequency (norm)')
            ax11.plot(data.index, result.spectral_entropy, color='blue', alpha=0.7, linewidth=1, label='Spectral Entropy')
            ax11.set_title('DSP Features Normalized Comparison')
            ax11.set_ylabel('Normalized Value')
            ax11.set_ylim(0, 1)
            ax11.grid(True, alpha=0.3)
            ax11.legend()
            
            # 12. 統計サマリー
            ax12 = fig.add_subplot(gs[11])
            ax12.axis('off')
            
            summary_text = f"""
超洗練されたボラティリティ分析サマリー - 最先端DSP技術
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 ボラティリティ分布: 高ボラ {stats['high_volatility_percentage']:.1f}% | 低ボラ {stats['low_volatility_percentage']:.1f}%

✨ 信頼度分析: 高信頼度 {stats['high_confidence_percentage']:.1f}% | 平均信頼度 {stats['average_confidence']:.3f}

🧠 DSP特徴量: ヒルベルト包絡線 {stats['latest_hilbert_envelope']:.6f} | ウェーブレット {stats['latest_wavelet_energy']:.6f} | エントロピー {stats['latest_spectral_entropy']:.3f}

📐 フラクタル次元: {stats['latest_fractal_dimension']:.3f} | 瞬間周波数: {stats['latest_instantaneous_freq']:.6f}

🎛️ 適応システム: 閾値 {stats['latest_adaptive_threshold']:.3f} | ゲイン {stats['latest_adaptive_gain']:.3f}

🎯 現在状態: {stats['latest_state']} Vol (確率: {stats['latest_probability']:.3f}, 信頼度: {stats['latest_confidence']:.3f})

📊 品質指標: 平均信頼度 {stats['quality_metrics']['avg_confidence']:.3f} | 安定性 {stats['quality_metrics']['confidence_stability']:.3f} | 複雑性 {stats['quality_metrics']['avg_complexity']:.3f}
            """
            
            ax12.text(0.05, 0.95, summary_text, transform=ax12.transAxes, 
                    fontsize=9, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
            
            # 保存
            filename = f"ultra_refined_volatility_state_{symbol}_{timeframe}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            self.logger.info(f"超洗練されたボラティリティチャートを保存しました: {filename}")
            plt.show()
            
        except Exception as e:
            self.logger.error(f"チャート作成に失敗: {e}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description='超洗練されたボラティリティ状態分析システム',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🚀 超洗練されたボラティリティ分析の特徴:
  🧠 最先端デジタル信号処理技術
  ⚡ STRベース超低遅延測定
  🌊 ヒルベルト変換による包絡線・位相解析
  📊 ウェーブレット変換による多解像度解析
  🔍 スペクトラルエントロピーによる複雑性測定
  📐 フラクタル次元による自己相似性分析
  🎯 適応カルマンフィルタによるノイズ除去
  🎛️ 適応的閾値による動的市場調整
  ✨ 信頼度ベース判定システム

🔬 DSP技術の応用:
  - ヒルベルト変換: 信号の包絡線と瞬間位相を抽出
  - ウェーブレット変換: 時間-周波数領域での多解像度解析
  - スペクトラルエントロピー: 信号の複雑性・不規則性を定量化
  - フラクタル次元: 時系列の自己相似性・複雑性を測定
  - 適応カルマンフィルタ: 動的ノイズ除去と平滑化
  - 適応的閾値: 市場状況に応じた動的判定基準
        """
    )
    
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイル')
    parser.add_argument('--no-show', action='store_true', help='チャート非表示')
    parser.add_argument('--sensitive', action='store_true', help='高感度モード（sensitivity=3.0）')
    parser.add_argument('--conservative', action='store_true', help='保守的モード（confidence=0.8）')
    parser.add_argument('--high-precision', action='store_true', help='高精度モード（全パラメータ最適化）')
    parser.add_argument('--debug', action='store_true', help='デバッグモード')
    
    args = parser.parse_args()
    
    try:
        print("🚀 超洗練されたボラティリティ状態分析システム起動中...")
        print("🧠 最先端デジタル信号処理技術を初期化中...")
        
        analyzer = UltraRefinedVolatilityAnalyzer(args.config)
        
        # モード設定
        if args.sensitive:
            analyzer.vol_indicator.sensitivity = 3.0
            analyzer.vol_indicator.confidence_threshold = 0.6
            print("⚡ 高感度モードを有効化（sensitivity: 3.0, confidence: 0.6）")
        
        if args.conservative:
            analyzer.vol_indicator.sensitivity = 1.5
            analyzer.vol_indicator.confidence_threshold = 0.8
            print("🛡️ 保守的モードを有効化（sensitivity: 1.5, confidence: 0.8）")
        
        if args.high_precision:
            analyzer.vol_indicator.str_period = 10
            analyzer.vol_indicator.entropy_window = 20
            analyzer.vol_indicator.fractal_k = 12
            analyzer.vol_indicator.sensitivity = 2.5
            analyzer.vol_indicator.confidence_threshold = 0.75
            print("🎯 高精度モードを有効化（全パラメータ最適化）")
        
        # 分析実行
        results = analyzer.run_ultra_refined_analysis(show_chart=not args.no_show)
        
        print("\n✅ 超洗練されたボラティリティ分析が完了しました！")
        
        # 結果評価
        high_vol_pct = results['stats']['high_volatility_percentage']
        high_conf_pct = results['stats']['high_confidence_percentage']
        
        if 25 <= high_vol_pct <= 35:
            print("🎯 理想的なボラティリティ分布")
        elif 20 <= high_vol_pct <= 40:
            print("✅ 優秀なボラティリティ分布")
        elif 15 <= high_vol_pct <= 50:
            print("📊 実用的なボラティリティ分布")
        else:
            print("⚠️ パラメータ調整を検討してください")
        
        if high_conf_pct > 60:
            print("✅ 優秀な信頼度分布")
        elif high_conf_pct > 40:
            print("📊 良好な信頼度分布")
        else:
            print("⚠️ 信頼度向上が必要")
        
        # DSP特徴量評価
        print(f"\n🧠 DSP特徴量評価:")
        if results['stats']['quality_metrics']:
            qm = results['stats']['quality_metrics']
            print(f"   平均信頼度: {qm['avg_confidence']:.3f}")
            print(f"   信頼度安定性: {qm['confidence_stability']:.3f}")
            print(f"   信号複雑性: {qm['avg_complexity']:.3f}")
            print(f"   フラクタル次元: {qm['avg_fractal_dimension']:.3f}")
        
    except KeyboardInterrupt:
        print("\n⚠️ 分析が中断されました")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()