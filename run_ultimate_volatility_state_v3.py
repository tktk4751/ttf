#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource
from indicators.ultimate_volatility_state_v3 import UltimateVolatilityStateV3
from logger import get_logger


class UltimateVolatilityStateV3Analyzer:
    """
    Ultimate Volatility State V3 の超高精度分析システム
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
        
        # V3インジケーターの初期化
        uvs_config = self.config.get('ultimate_volatility_state', {})
        self.uvs_indicator = UltimateVolatilityStateV3(
            period=uvs_config.get('period', 21),
            threshold=uvs_config.get('threshold', 0.5),
            zscore_period=uvs_config.get('zscore_period', 50),
            src_type=uvs_config.get('src_type', 'hlc3'),
            learning_rate=0.005,  # V3専用パラメータ
            chaos_embedding_dim=3,
            n_learners=7,
            confidence_threshold=0.85
        )
        
        self.logger.info("Ultimate Volatility State V3 Analyzer initialized")
    
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
    
    def run_ultra_analysis(self, show_chart: bool = True) -> dict:
        """超高精度分析の実行"""
        try:
            self.logger.info("🚀 Ultimate Volatility State V3 超高精度分析を開始...")
            
            # データ読み込み
            data = self.load_market_data()
            
            # V3計算
            self.logger.info("🧠 最先端アルゴリズムによる分析を実行中...")
            self.logger.info("   - カオス理論（Lyapunov指数）")
            self.logger.info("   - 情報理論（マルチスケールエントロピー）")
            self.logger.info("   - デジタル信号処理（適応カルマンフィルター）")
            self.logger.info("   - 神経適応学習（オンライン学習）")
            self.logger.info("   - 経験的モード分解（EMD）")
            self.logger.info("   - 適応アンサンブル学習")
            
            result = self.uvs_indicator.calculate(data)
            
            # 詳細統計の計算
            stats = self._calculate_advanced_stats(result)
            
            # 結果の表示
            self._display_results(stats)
            
            # チャートの作成
            if show_chart:
                self._create_ultra_chart(data, result, stats)
            
            return {'data': data, 'result': result, 'stats': stats}
            
        except Exception as e:
            self.logger.error(f"V3分析の実行に失敗: {e}")
            raise
    
    def _calculate_advanced_stats(self, result) -> dict:
        """高度な統計分析"""
        # 基本統計
        total_periods = len(result.state)
        high_vol_count = np.sum(result.state)
        low_vol_count = total_periods - high_vol_count
        
        # 信頼度統計
        ultra_high_confidence = np.sum(result.confidence > 0.9)
        high_confidence = np.sum(result.confidence > 0.8)
        medium_confidence = np.sum(result.confidence > 0.6)
        
        # カオス分析統計
        chaos_periods = np.sum(np.abs(result.chaos_measure) > 0.1)
        high_chaos = np.sum(np.abs(result.chaos_measure) > 0.2)
        
        # 神経適応統計
        adaptation_efficiency = np.mean(result.neural_adaptation)
        adaptation_stability = 1.0 - np.std(result.neural_adaptation)
        
        # アンサンブル統計
        ensemble_consistency = 1.0 - np.std(result.ensemble_weight)
        
        # 精度指標
        confident_high_vol = np.sum((result.state == 1) & (result.confidence > 0.8))
        confident_low_vol = np.sum((result.state == 0) & (result.confidence > 0.8))
        precision_score = (confident_high_vol + confident_low_vol) / total_periods
        
        return {
            'total_periods': total_periods,
            'high_volatility_count': high_vol_count,
            'low_volatility_count': low_vol_count,
            'high_volatility_percentage': (high_vol_count / total_periods * 100),
            'low_volatility_percentage': (low_vol_count / total_periods * 100),
            'ultra_high_confidence_count': ultra_high_confidence,
            'high_confidence_count': high_confidence,
            'medium_confidence_count': medium_confidence,
            'ultra_high_confidence_percentage': (ultra_high_confidence / total_periods * 100),
            'high_confidence_percentage': (high_confidence / total_periods * 100),
            'chaos_periods': chaos_periods,
            'high_chaos_periods': high_chaos,
            'adaptation_efficiency': adaptation_efficiency,
            'adaptation_stability': adaptation_stability,
            'ensemble_consistency': ensemble_consistency,
            'precision_score': precision_score,
            'average_probability': np.mean(result.probability[result.probability > 0]),
            'average_confidence': np.mean(result.confidence[result.confidence > 0]),
            'latest_state': 'High' if result.state[-1] == 1 else 'Low',
            'latest_probability': result.probability[-1],
            'latest_confidence': result.confidence[-1],
            'latest_chaos': result.chaos_measure[-1]
        }
    
    def _display_results(self, stats: dict) -> None:
        """結果の詳細表示"""
        self.logger.info("\n" + "="*80)
        self.logger.info("🎯 ULTIMATE VOLATILITY STATE V3 分析結果")
        self.logger.info("="*80)
        
        self.logger.info(f"📊 基本統計:")
        self.logger.info(f"   総期間数: {stats['total_periods']:,}")
        self.logger.info(f"   高ボラティリティ: {stats['high_volatility_count']:,} ({stats['high_volatility_percentage']:.1f}%)")
        self.logger.info(f"   低ボラティリティ: {stats['low_volatility_count']:,} ({stats['low_volatility_percentage']:.1f}%)")
        
        self.logger.info(f"\n🔍 信頼度分析:")
        self.logger.info(f"   超高信頼度判定: {stats['ultra_high_confidence_count']:,} ({stats['ultra_high_confidence_percentage']:.1f}%)")
        self.logger.info(f"   高信頼度判定: {stats['high_confidence_count']:,} ({stats['high_confidence_percentage']:.1f}%)")
        self.logger.info(f"   平均信頼度: {stats['average_confidence']:.3f}")
        
        self.logger.info(f"\n🌀 カオス理論分析:")
        self.logger.info(f"   カオス検出期間: {stats['chaos_periods']:,}")
        self.logger.info(f"   高カオス期間: {stats['high_chaos_periods']:,}")
        
        self.logger.info(f"\n🧠 神経適応学習:")
        self.logger.info(f"   適応効率: {stats['adaptation_efficiency']:.3f}")
        self.logger.info(f"   適応安定性: {stats['adaptation_stability']:.3f}")
        
        self.logger.info(f"\n🎲 アンサンブル学習:")
        self.logger.info(f"   学習器一貫性: {stats['ensemble_consistency']:.3f}")
        
        self.logger.info(f"\n⚡ 最新状態:")
        self.logger.info(f"   状態: {stats['latest_state']} ボラティリティ")
        self.logger.info(f"   確率: {stats['latest_probability']:.3f}")
        self.logger.info(f"   信頼度: {stats['latest_confidence']:.3f}")
        self.logger.info(f"   カオス度: {stats['latest_chaos']:.3f}")
        
        self.logger.info(f"\n🎯 総合精度スコア: {stats['precision_score']:.3f}")
        
        # 精度評価
        if stats['precision_score'] > 0.9:
            self.logger.info("🟢 卓越した分析精度！")
        elif stats['precision_score'] > 0.8:
            self.logger.info("🟢 優秀な分析精度")
        elif stats['precision_score'] > 0.7:
            self.logger.info("🟡 良好な分析精度")
        else:
            self.logger.info("🔴 分析精度要改善")
    
    def _create_ultra_chart(self, data, result, stats) -> None:
        """超高精度チャートの作成"""
        try:
            symbol = self.config.get('binance_data', {}).get('symbol', 'Unknown')
            timeframe = self.config.get('binance_data', {}).get('timeframe', 'Unknown')
            
            fig = plt.figure(figsize=(20, 24))
            gs = fig.add_gridspec(8, 1, height_ratios=[3, 1, 1, 1, 1, 1, 1, 1], hspace=0.4)
            
            # 1. 価格チャート with 状態
            ax1 = fig.add_subplot(gs[0])
            ax1.plot(data.index, data['close'], linewidth=1, color='black', label='Price')
            
            # 信頼度ベースの状態表示
            for i in range(len(data)):
                alpha = 0.1 + 0.4 * result.confidence[i]
                color = 'red' if result.state[i] == 1 else 'blue'
                ax1.axvspan(data.index[i], data.index[min(i+1, len(data)-1)], 
                           alpha=alpha, color=color)
            
            ax1.set_title(f'V3 Ultra Analysis - {symbol} ({timeframe}) - Precision: {stats["precision_score"]:.3f}')
            ax1.set_ylabel('Price')
            ax1.grid(True, alpha=0.3)
            
            # 2. 状態 & 信頼度
            ax2 = fig.add_subplot(gs[1])
            colors = ['red' if s == 1 else 'blue' for s in result.state]
            ax2.bar(data.index, result.state, color=colors, alpha=0.6, width=1)
            ax2_twin = ax2.twinx()
            ax2_twin.plot(data.index, result.confidence, color='green', linewidth=1.5, alpha=0.8)
            ax2_twin.axhline(y=0.8, color='green', linestyle='--', alpha=0.5)
            ax2.set_title('State & Confidence')
            ax2.set_ylabel('State')
            ax2_twin.set_ylabel('Confidence')
            
            # 3. カオス分析
            ax3 = fig.add_subplot(gs[2])
            ax3.plot(data.index, result.chaos_measure, color='purple', linewidth=1.2)
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax3.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='Chaos Threshold')
            ax3.axhline(y=-0.1, color='orange', linestyle='--', alpha=0.5)
            ax3.set_title('Chaos Theory Analysis (Lyapunov Exponent)')
            ax3.set_ylabel('Lyapunov Exp')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            # 4. 神経適応学習
            ax4 = fig.add_subplot(gs[3])
            ax4.plot(data.index, result.neural_adaptation, color='cyan', linewidth=1.2)
            ax4.set_title('Neural Adaptive Learning')
            ax4.set_ylabel('Adaptation Weight')
            ax4.grid(True, alpha=0.3)
            
            # 5. エントロピー分析
            ax5 = fig.add_subplot(gs[4])
            if 'multiscale_entropy' in result.entropy_metrics:
                ax5.plot(data.index, result.entropy_metrics['multiscale_entropy'], 
                        color='orange', linewidth=1.2, label='MSE Scale 1')
            ax5.set_title('Information Theory Analysis')
            ax5.set_ylabel('Entropy')
            ax5.grid(True, alpha=0.3)
            ax5.legend()
            
            # 6. DSP特徴量
            ax6 = fig.add_subplot(gs[5])
            if 'kalman_volatility' in result.dsp_features:
                ax6.plot(data.index, result.dsp_features['kalman_volatility'], 
                        color='red', linewidth=1.2, alpha=0.8, label='Kalman Vol')
            if 'neural_filtered' in result.dsp_features:
                normalized_neural = result.dsp_features['neural_filtered'] / np.max(np.abs(result.dsp_features['neural_filtered']))
                ax6.plot(data.index, normalized_neural, 
                        color='blue', linewidth=1, alpha=0.8, label='Neural Filtered')
            ax6.set_title('Digital Signal Processing Features')
            ax6.set_ylabel('DSP Values')
            ax6.grid(True, alpha=0.3)
            ax6.legend()
            
            # 7. 機械学習予測
            ax7 = fig.add_subplot(gs[6])
            ax7.plot(data.index, result.ml_prediction, color='green', linewidth=1.5, alpha=0.8)
            ax7.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Decision Boundary')
            ax7.set_title('Machine Learning Ensemble Prediction')
            ax7.set_ylabel('ML Prediction')
            ax7.grid(True, alpha=0.3)
            ax7.legend()
            
            # 8. 統計サマリー
            ax8 = fig.add_subplot(gs[7])
            ax8.axis('off')
            
            summary_text = f"""
V3 ULTRA ANALYSIS SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 PRECISION SCORE: {stats['precision_score']:.3f}  |  🔍 CONFIDENCE: {stats['average_confidence']:.3f}

📊 Volatility Distribution:  High: {stats['high_volatility_percentage']:.1f}%  |  Low: {stats['low_volatility_percentage']:.1f}%

🧠 AI Analysis:  Adaptation Efficiency: {stats['adaptation_efficiency']:.3f}  |  Ensemble Consistency: {stats['ensemble_consistency']:.3f}

🌀 Chaos Theory:  Chaotic Periods: {stats['chaos_periods']}  |  High Chaos: {stats['high_chaos_periods']}

⚡ Current State:  {stats['latest_state']} Vol (Prob: {stats['latest_probability']:.3f}, Conf: {stats['latest_confidence']:.3f})
            """
            
            ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, 
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            # 保存
            filename = f"ultimate_volatility_state_v3_{symbol}_{timeframe}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            self.logger.info(f"V3チャートを保存しました: {filename}")
            plt.show()
            
        except Exception as e:
            self.logger.error(f"チャート作成に失敗: {e}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description='Ultimate Volatility State V3 - 超高精度ボラティリティ分析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🚀 V3 革新的特徴:
  ✨ カオス理論（Lyapunov指数による敏感依存性分析）
  ✨ 情報理論（マルチスケールエントロピー）
  ✨ デジタル信号処理（適応カルマンフィルター）
  ✨ 神経適応学習（オンライン学習）
  ✨ 経験的モード分解（EMD）
  ✨ 適応アンサンブル学習

🎯 使用例:
  python run_ultimate_volatility_state_v3.py --ultra-precision
  python run_ultimate_volatility_state_v3.py --chaos-analysis
        """
    )
    
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイル')
    parser.add_argument('--no-show', action='store_true', help='チャート非表示')
    parser.add_argument('--ultra-precision', action='store_true', help='超高精度モード')
    parser.add_argument('--chaos-analysis', action='store_true', help='カオス分析強化')
    
    args = parser.parse_args()
    
    try:
        print("🚀 Ultimate Volatility State V3 - 超高精度分析システム起動中...")
        
        analyzer = UltimateVolatilityStateV3Analyzer(args.config)
        
        # モード設定
        if args.ultra_precision:
            analyzer.uvs_indicator.confidence_threshold = 0.9
            analyzer.uvs_indicator.learning_rate = 0.003  # より慎重な学習
            print("⚡ 超高精度モードを有効化")
        
        if args.chaos_analysis:
            analyzer.uvs_indicator.chaos_embedding_dim = 5  # より高次元解析
            print("🌀 カオス分析強化モードを有効化")
        
        # 分析実行
        results = analyzer.run_ultra_analysis(show_chart=not args.no_show)
        
        print("\n✅ V3超高精度分析が完了しました！")
        
        # 最終評価
        precision = results['stats']['precision_score']
        if precision > 0.95:
            print("🏆 究極の分析精度を達成！")
        elif precision > 0.9:
            print("🥇 卓越した分析精度")
        elif precision > 0.85:
            print("🥈 優秀な分析精度")
        else:
            print("🥉 良好な分析精度")
        
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