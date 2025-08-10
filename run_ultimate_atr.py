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
from indicators.ultimate_atr import UltimateATR
from logger import get_logger


class UltimateATRAnalyzer:
    """
    アルティメットATR分析システム
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
        
        # アルティメットATRインジケーターの初期化（UKF+Ultimate Smoother）
        self.ultimate_atr = UltimateATR(
            ultimate_smoother_period=20.0,       # Ultimate Smoother期間（デフォルト：20）
            src_type='hlc3',                      # 価格ソース
            period_mode='fixed',                  # 期間モード
            cycle_detector_type='absolute_ultimate'
        )
        
        # 比較用に動的期間版も作成
        self.ultimate_atr_dynamic = UltimateATR(
            ultimate_smoother_period=20.0,       # Ultimate Smoother期間（デフォルト：20）
            src_type='hlc3',                      # 価格ソース
            period_mode='dynamic',                # 動的期間モード
            cycle_detector_type='absolute_ultimate'
        )
        
        self.logger.info("Ultimate ATR Analyzer initialized")
    
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
    
    def run_analysis(self, show_chart: bool = True) -> dict:
        """アルティメットATR分析の実行"""
        try:
            self.logger.info("📊 アルティメットATR分析を開始...")
            
            # データ読み込み
            data = self.load_market_data()
            
            # 固定期間版の計算
            self.logger.info("🔍 固定期間アルティメットATRを計算中...")
            fixed_result = self.ultimate_atr.calculate(data)
            
            # 動的期間版の計算
            self.logger.info("🔍 動的期間アルティメットATRを計算中...")
            dynamic_result = self.ultimate_atr_dynamic.calculate(data)
            
            # 詳細統計の計算
            fixed_stats = self._calculate_statistics(fixed_result, "固定期間")
            dynamic_stats = self._calculate_statistics(dynamic_result, "動的期間")
            
            # 結果の表示
            self._display_results(fixed_stats, dynamic_stats)
            
            # チャートの作成
            if show_chart:
                self._create_comparison_chart(data, fixed_result, dynamic_result, fixed_stats, dynamic_stats)
            
            return {
                'data': data,
                'fixed_result': fixed_result,
                'dynamic_result': dynamic_result,
                'fixed_stats': fixed_stats,
                'dynamic_stats': dynamic_stats
            }
            
        except Exception as e:
            self.logger.error(f"アルティメットATR分析の実行に失敗: {e}")
            raise
    
    def _calculate_statistics(self, result, version_name: str) -> dict:
        """統計分析"""
        # NaN値を処理する関数
        def safe_mean(arr):
            valid_values = arr[np.isfinite(arr) & (arr > 0)]
            return np.mean(valid_values) if len(valid_values) > 0 else 0.0
        
        def safe_std(arr):
            valid_values = arr[np.isfinite(arr) & (arr > 0)]
            return np.std(valid_values) if len(valid_values) > 0 else 0.0
        
        def safe_median(arr):
            valid_values = arr[np.isfinite(arr) & (arr > 0)]
            return np.median(valid_values) if len(valid_values) > 0 else 0.0
        
        def safe_last_value(arr):
            for i in range(len(arr) - 1, -1, -1):
                if np.isfinite(arr[i]):
                    return arr[i]
            return 0.0
        
        # 基本統計
        ultimate_atr_mean = safe_mean(result.values)
        ultimate_atr_std = safe_std(result.values)
        ultimate_atr_median = safe_median(result.values)
        
        raw_atr_mean = safe_mean(result.raw_atr)
        raw_atr_std = safe_std(result.raw_atr)
        raw_atr_median = safe_median(result.raw_atr)
        
        ultimate_smoothed_mean = safe_mean(result.ultimate_smoothed)
        ultimate_smoothed_std = safe_std(result.ultimate_smoothed)
        
        tr_mean = safe_mean(result.true_range)
        tr_std = safe_std(result.true_range)
        
        # 最新値
        latest_ultimate = safe_last_value(result.values)
        latest_raw = safe_last_value(result.raw_atr)
        latest_ultimate_smoothed = safe_last_value(result.ultimate_smoothed)
        latest_tr = safe_last_value(result.true_range)
        
        # 比較メトリクス
        smoothing_effectiveness = raw_atr_std / ultimate_atr_std if ultimate_atr_std > 0 else 0
        stage1_smoothing_effectiveness = raw_atr_std / ultimate_smoothed_std if ultimate_smoothed_std > 0 else 0
        stage2_smoothing_effectiveness = ultimate_smoothed_std / ultimate_atr_std if ultimate_atr_std > 0 else 0
        ultimate_vs_raw_ratio = latest_ultimate / latest_raw if latest_raw > 0 else 0
        
        # 相関係数
        valid_ultimate = result.values[np.isfinite(result.values)]
        valid_raw = result.raw_atr[np.isfinite(result.raw_atr)]
        if len(valid_ultimate) > 1 and len(valid_raw) > 1 and len(valid_ultimate) == len(valid_raw):
            correlation = np.corrcoef(valid_ultimate, valid_raw)[0, 1]
        else:
            correlation = 0.0
        
        return {
            'version': version_name,
            'ultimate_atr_mean': ultimate_atr_mean,
            'ultimate_atr_std': ultimate_atr_std,
            'ultimate_atr_median': ultimate_atr_median,
            'raw_atr_mean': raw_atr_mean,
            'raw_atr_std': raw_atr_std,
            'raw_atr_median': raw_atr_median,
            'ultimate_smoothed_mean': ultimate_smoothed_mean,
            'ultimate_smoothed_std': ultimate_smoothed_std,
            'tr_mean': tr_mean,
            'tr_std': tr_std,
            'latest_ultimate': latest_ultimate,
            'latest_raw': latest_raw,
            'latest_ultimate_smoothed': latest_ultimate_smoothed,
            'latest_tr': latest_tr,
            'smoothing_effectiveness': smoothing_effectiveness,
            'stage1_smoothing_effectiveness': stage1_smoothing_effectiveness,
            'stage2_smoothing_effectiveness': stage2_smoothing_effectiveness,
            'ultimate_vs_raw_ratio': ultimate_vs_raw_ratio,
            'correlation': correlation
        }
    
    def _display_results(self, fixed_stats: dict, dynamic_stats: dict) -> None:
        """結果の詳細表示"""
        self.logger.info("\n" + "="*70)
        self.logger.info("📊 アルティメットATR分析結果")
        self.logger.info("="*70)
        
        # 固定期間版の結果
        self.logger.info(f"\n🔧 {fixed_stats['version']}版:")
        self.logger.info(f"   アルティメットATR（最終値）: {fixed_stats['latest_ultimate']:.4f}")
        self.logger.info(f"   UKFフィルター済み（中間値）: {fixed_stats['latest_ultimate_smoothed']:.4f}")
        self.logger.info(f"   通常のATR: {fixed_stats['latest_raw']:.4f}")
        self.logger.info(f"   True Range: {fixed_stats['latest_tr']:.4f}")
        self.logger.info(f"   アルティメット/通常 比率: {fixed_stats['ultimate_vs_raw_ratio']:.3f}")
        self.logger.info(f"   全体平滑化効果: {fixed_stats['smoothing_effectiveness']:.3f}")
        self.logger.info(f"   UKFフィルター効果: {fixed_stats['stage1_smoothing_effectiveness']:.3f}")
        self.logger.info(f"   Ultimate Smoother効果: {fixed_stats['stage2_smoothing_effectiveness']:.3f}")
        self.logger.info(f"   相関係数: {fixed_stats['correlation']:.3f}")
        
        # 動的期間版の結果
        self.logger.info(f"\n⚡ {dynamic_stats['version']}版:")
        self.logger.info(f"   アルティメットATR（最終値）: {dynamic_stats['latest_ultimate']:.4f}")
        self.logger.info(f"   UKFフィルター済み（中間値）: {dynamic_stats['latest_ultimate_smoothed']:.4f}")
        self.logger.info(f"   通常のATR: {dynamic_stats['latest_raw']:.4f}")
        self.logger.info(f"   True Range: {dynamic_stats['latest_tr']:.4f}")
        self.logger.info(f"   アルティメット/通常 比率: {dynamic_stats['ultimate_vs_raw_ratio']:.3f}")
        self.logger.info(f"   全体平滑化効果: {dynamic_stats['smoothing_effectiveness']:.3f}")
        self.logger.info(f"   UKFフィルター効果: {dynamic_stats['stage1_smoothing_effectiveness']:.3f}")
        self.logger.info(f"   Ultimate Smoother効果: {dynamic_stats['stage2_smoothing_effectiveness']:.3f}")
        self.logger.info(f"   相関係数: {dynamic_stats['correlation']:.3f}")
        
        # 統計比較
        self.logger.info(f"\n📈 統計比較:")
        self.logger.info(f"   固定期間 アルティメットATR平均: {fixed_stats['ultimate_atr_mean']:.4f} ± {fixed_stats['ultimate_atr_std']:.4f}")
        self.logger.info(f"   動的期間 アルティメットATR平均: {dynamic_stats['ultimate_atr_mean']:.4f} ± {dynamic_stats['ultimate_atr_std']:.4f}")
        self.logger.info(f"   固定期間 通常ATR平均: {fixed_stats['raw_atr_mean']:.4f} ± {fixed_stats['raw_atr_std']:.4f}")
        self.logger.info(f"   動的期間 通常ATR平均: {dynamic_stats['raw_atr_mean']:.4f} ± {dynamic_stats['raw_atr_std']:.4f}")
        
        # 効果評価
        better_smoothing = "動的期間" if dynamic_stats['smoothing_effectiveness'] > fixed_stats['smoothing_effectiveness'] else "固定期間"
        self.logger.info(f"\n✅ より良い平滑化効果: {better_smoothing}版")
    
    def _create_comparison_chart(self, data, fixed_result, dynamic_result, fixed_stats, dynamic_stats) -> None:
        """比較チャートの作成"""
        try:
            symbol = self.config.get('binance_data', {}).get('symbol', 'Unknown')
            timeframe = self.config.get('binance_data', {}).get('timeframe', 'Unknown')
            
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(4, 1, height_ratios=[2, 1, 1, 1], hspace=0.3)
            
            # 1. 価格チャート
            ax1 = fig.add_subplot(gs[0])
            ax1.plot(data.index, data['close'], linewidth=1, color='black', label='Price')
            ax1.set_title(f'Ultimate ATR Analysis - {symbol} ({timeframe})')
            ax1.set_ylabel('Price')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # 2. ATR比較（UKF+Ultimate Smoother）
            ax2 = fig.add_subplot(gs[1])
            ax2.plot(data.index, fixed_result.values, linewidth=1.5, color='blue', label='Ultimate ATR (Fixed)')
            ax2.plot(data.index, dynamic_result.values, linewidth=1.5, color='red', label='Ultimate ATR (Dynamic)')
            ax2.plot(data.index, fixed_result.ultimate_smoothed, linewidth=1, color='green', alpha=0.7, label='UKF Filtered')
            ax2.plot(data.index, fixed_result.raw_atr, linewidth=1, color='gray', alpha=0.7, label='Standard ATR')
            ax2.set_title('ATR Comparison (UKF + Ultimate Smoother)')
            ax2.set_ylabel('ATR Value')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # 3. True Range
            ax3 = fig.add_subplot(gs[2])
            ax3.plot(data.index, fixed_result.true_range, linewidth=1, color='orange', alpha=0.7, label='True Range')
            ax3.set_title('True Range')
            ax3.set_ylabel('True Range')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            # 4. アルティメット/通常 比率
            ax4 = fig.add_subplot(gs[3])
            fixed_ratio = fixed_result.values / fixed_result.raw_atr
            dynamic_ratio = dynamic_result.values / dynamic_result.raw_atr
            
            # 無限大やNaNを除去
            fixed_ratio = np.where(np.isfinite(fixed_ratio), fixed_ratio, 1.0)
            dynamic_ratio = np.where(np.isfinite(dynamic_ratio), dynamic_ratio, 1.0)
            
            ax4.plot(data.index, fixed_ratio, linewidth=1, color='blue', alpha=0.7, label='Ultimate/Raw Ratio (Fixed)')
            ax4.plot(data.index, dynamic_ratio, linewidth=1, color='red', alpha=0.7, label='Ultimate/Raw Ratio (Dynamic)')
            ax4.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Ratio = 1.0')
            ax4.set_title('Ultimate ATR / Standard ATR Ratio')
            ax4.set_ylabel('Ratio')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            
            # 保存
            filename = f"ultimate_atr_analysis_{symbol}_{timeframe}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            self.logger.info(f"アルティメットATR比較チャートを保存しました: {filename}")
            plt.show()
            
        except Exception as e:
            self.logger.error(f"チャート作成に失敗: {e}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description='アルティメットATR分析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎯 アルティメットATR特徴（UKF+Ultimate Smoother版）:
  ✨ 通常のATR計算式を使用（高-低、高-前終値、低-前終値の最大値）
  ✨ 第1段階：UKF（無香料カルマンフィルター）でTrue Rangeをフィルタリング
  ✨ 第2段階：Ultimate Smootherで最終平滑化（デフォルト：20期間）
  ✨ 固定期間と動的期間の両方に対応
  ✨ 従来のATRよりも大幅にノイズを軽減
  ✨ True Range、UKFフィルター済み、最終ATRの比較表示

📊 分析結果:
  UKF+Ultimate Smootherによる高精度ATR分析
        """
    )
    
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイル')
    parser.add_argument('--no-show', action='store_true', help='チャート非表示')
    parser.add_argument('--period', type=float, default=10.0, help='Ultimate Smoother期間（第1段階）')
    
    args = parser.parse_args()
    
    try:
        print("📊 アルティメットATR分析システム起動中...")
        
        analyzer = UltimateATRAnalyzer(args.config)
        
        # 期間の設定（Ultimate Smoother期間を変更）
        if args.period != 10.0:
            analyzer.ultimate_atr.ultimate_smoother_period = args.period
            analyzer.ultimate_atr_dynamic.ultimate_smoother_period = args.period
            print(f"⚙️ Ultimate Smoother期間を {args.period} に設定")
        
        # 分析実行
        results = analyzer.run_analysis(show_chart=not args.no_show)
        
        print("\n✅ アルティメットATR分析が完了しました！")
        
        # 効果評価
        fixed_smoothing = results['fixed_stats']['smoothing_effectiveness']
        dynamic_smoothing = results['dynamic_stats']['smoothing_effectiveness']
        
        if dynamic_smoothing > fixed_smoothing:
            print("🎯 動的期間版がより効果的な平滑化を実現")
        else:
            print("🎯 固定期間版が安定した平滑化を提供")
        
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