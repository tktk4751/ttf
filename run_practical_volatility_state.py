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
from indicators.practical_volatility_state import PracticalVolatilityState
from logger import get_logger


class PracticalVolatilityAnalyzer:
    """
    実践的ボラティリティ状態分析システム
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
        
        # 実践的ボラティリティインジケーターの初期化（バランス版）
        self.vol_indicator = PracticalVolatilityState(
            str_period=20.0,
            str_threshold=0.75,                  # STRパーセンタイル閾値（25%が高ボラ）
            returns_threshold=0.75,              # リターンボラパーセンタイル閾値
            zscore_threshold=1.0,                # Z-スコア閾値（緩めに設定）
            percentile_window=120,               # パーセンタイル計算ウィンドウ
            zscore_window=60,                    # Z-スコア計算ウィンドウ
            velocity_period=3,
            acceleration_period=3,
            returns_period=20,
            src_type='ukf_hlc3',
            smoothing=True,
            dynamic_adaptation=True,
            cycle_detector_type='absolute_ultimate'
        )
        
        self.logger.info("Practical Volatility State Analyzer initialized")
    
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
    
    def run_practical_analysis(self, show_chart: bool = True) -> dict:
        """実践的分析の実行"""
        try:
            self.logger.info("📊 実践的ボラティリティ状態分析を開始...")
            
            # データ読み込み
            data = self.load_market_data()
            
            # ボラティリティ状態計算
            self.logger.info("🔍 ボラティリティ状態を計算中...")
            result = self.vol_indicator.calculate(data)
            
            # 詳細統計の計算
            stats = self._calculate_practical_stats(result)
            
            # 結果の表示
            self._display_results(stats)
            
            # チャートの作成
            if show_chart:
                self._create_practical_chart(data, result, stats)
            
            return {'data': data, 'result': result, 'stats': stats}
            
        except Exception as e:
            self.logger.error(f"実践的分析の実行に失敗: {e}")
            raise
    
    def _calculate_practical_stats(self, result) -> dict:
        """実践的統計分析"""
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
        
        # STR関連統計
        str_metrics = self.vol_indicator.get_str_metrics()
        volatility_strength = self.vol_indicator.get_volatility_strength()
        current_regime = self.vol_indicator.get_current_regime()
        is_expanding = self.vol_indicator.is_volatility_expanding()
        is_contracting = self.vol_indicator.is_volatility_contracting()
        
        # STR値の統計
        str_mean = np.mean(result.str_values[result.str_values > 0])
        str_std = np.std(result.str_values[result.str_values > 0])
        str_percentile_mean = np.mean(result.str_percentile[result.str_percentile > 0])
        
        # 速度と加速度の統計
        velocity_mean = np.mean(np.abs(result.str_velocity[~np.isnan(result.str_velocity)]))
        acceleration_mean = np.mean(np.abs(result.str_acceleration[~np.isnan(result.str_acceleration)]))
        
        # リターンボラティリティ統計
        returns_vol_mean = np.mean(result.returns_volatility[result.returns_volatility > 0])
        returns_vol_std = np.std(result.returns_volatility[result.returns_volatility > 0])
        returns_percentile_mean = np.mean(result.returns_percentile[result.returns_percentile > 0])
        
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
            'current_regime': current_regime,
            'volatility_strength': volatility_strength,
            'is_expanding': is_expanding,
            'is_contracting': is_contracting,
            # STR関連統計
            'str_metrics': str_metrics,
            'str_mean': str_mean,
            'str_std': str_std,
            'str_percentile_mean': str_percentile_mean,
            'velocity_mean': velocity_mean,
            'acceleration_mean': acceleration_mean,
            # 最新値（STRベース）
            'latest_str': result.str_values[-1] if len(result.str_values) > 0 else 0,
            'latest_str_percentile': result.str_percentile[-1] if len(result.str_percentile) > 0 else 0,
            'latest_str_velocity': result.str_velocity[-1] if len(result.str_velocity) > 0 else 0,
            'latest_str_acceleration': result.str_acceleration[-1] if len(result.str_acceleration) > 0 else 0,
            # リターンボラティリティ関連統計
            'returns_vol_mean': returns_vol_mean,
            'returns_vol_std': returns_vol_std,
            'returns_percentile_mean': returns_percentile_mean,
            'latest_returns_vol': result.returns_volatility[-1] if len(result.returns_volatility) > 0 else 0,
            'latest_returns_percentile': result.returns_percentile[-1] if len(result.returns_percentile) > 0 else 0
        }
    
    def _display_results(self, stats: dict) -> None:
        """結果の詳細表示"""
        self.logger.info("\n" + "="*70)
        self.logger.info("📊 実践的ボラティリティ状態分析結果")
        self.logger.info("="*70)
        
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
        
        self.logger.info(f"\n⚡ 現在の状況:")
        self.logger.info(f"   状態: {stats['latest_state']} ボラティリティ ({stats['volatility_strength']})")
        self.logger.info(f"   確率: {stats['latest_probability']:.3f}")
        self.logger.info(f"   市場体制: {stats['current_regime']}")
        if stats['is_expanding']:
            self.logger.info(f"   📈 ボラティリティ拡大中")
        elif stats['is_contracting']:
            self.logger.info(f"   📉 ボラティリティ収縮中")
        
        self.logger.info(f"\n🎯 STR分析:")
        self.logger.info(f"   現在STR値: {stats['latest_str']:.4f}")
        self.logger.info(f"   STRパーセンタイル: {stats['latest_str_percentile']:.3f}")
        self.logger.info(f"   STR変化率: {stats['latest_str_velocity']:+.4f}")
        self.logger.info(f"   STR加速度: {stats['latest_str_acceleration']:+.4f}")
        self.logger.info(f"   平均STR: {stats['str_mean']:.4f} ± {stats['str_std']:.4f}")
        
        self.logger.info(f"\n📊 リターンボラティリティ分析:")
        self.logger.info(f"   現在年率ボラティリティ: {stats['latest_returns_vol']:.1f}%")
        self.logger.info(f"   リターンボラパーセンタイル: {stats['latest_returns_percentile']:.3f}")
        self.logger.info(f"   平均年率ボラティリティ: {stats['returns_vol_mean']:.1f}% ± {stats['returns_vol_std']:.1f}%")
        
        # 実用性評価
        if 20 <= stats['high_volatility_percentage'] <= 40:
            self.logger.info("\n✅ 現実的なボラティリティ分布です")
        elif 15 <= stats['high_volatility_percentage'] <= 50:
            self.logger.info("\n✅ 実用的なボラティリティ分布です")
        else:
            self.logger.info("\n⚠️ ボラティリティ分布を確認してください")
    
    def _create_practical_chart(self, data, result, stats) -> None:
        """実践的チャートの作成"""
        try:
            symbol = self.config.get('binance_data', {}).get('symbol', 'Unknown')
            timeframe = self.config.get('binance_data', {}).get('timeframe', 'Unknown')
            
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(5, 1, height_ratios=[3, 1, 1, 1, 1], hspace=0.3)
            
            # 1. 価格チャート with ボラティリティ状態
            ax1 = fig.add_subplot(gs[0])
            ax1.plot(data.index, data['close'], linewidth=1, color='black', label='Price')
            
            # ボラティリティ状態の背景色
            for i in range(len(data)):
                # 確率値を0-1の範囲にクリップしてalphaを計算
                prob_clipped = max(0.0, min(1.0, result.probability[i]))
                alpha = 0.1 + 0.3 * prob_clipped  # 0.1-0.4の範囲
                color = 'red' if result.state[i] == 1 else 'lightblue'
                ax1.axvspan(data.index[i], data.index[min(i+1, len(data)-1)], 
                           alpha=alpha, color=color)
            
            ax1.set_title(f'Practical Volatility Analysis - {symbol} ({timeframe})\nHigh Vol: {stats["high_volatility_percentage"]:.1f}% | Transitions: {stats["transitions"]}')
            ax1.set_ylabel('Price')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # 2. ボラティリティ状態バー
            ax2 = fig.add_subplot(gs[1])
            colors = ['red' if s == 1 else 'blue' for s in result.state]
            ax2.bar(data.index, result.state, color=colors, alpha=0.7, width=1)
            ax2.set_title('Volatility State (1: High, 0: Low)')
            ax2.set_ylabel('State')
            ax2.set_ylim(-0.1, 1.1)
            ax2.grid(True, alpha=0.3)
            
            # 3. 確率とATR
            ax3 = fig.add_subplot(gs[2])
            ax3.plot(data.index, result.probability, color='orange', linewidth=1.5, label='Probability')
            ax3.axhline(y=0.75, color='red', linestyle='--', alpha=0.5, label='High Vol Threshold')
            ax3.axhline(y=0.25, color='blue', linestyle='--', alpha=0.5, label='Low Vol Threshold')
            ax3.set_title('Volatility Probability')
            ax3.set_ylabel('Probability')
            ax3.set_ylim(0, 1)
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            # 4. STR値とリターンボラティリティ
            ax4 = fig.add_subplot(gs[3])
            ax4.plot(data.index, result.str_values, color='green', linewidth=1.2, label='STR Values')
            ax4_twin = ax4.twinx()
            ax4_twin.plot(data.index, result.returns_volatility, color='purple', linewidth=1.2, alpha=0.8, label='Returns Vol (%)')
            ax4.set_title('STR Values & Returns Volatility')
            ax4.set_ylabel('STR Values')
            ax4_twin.set_ylabel('Returns Volatility (%)')
            ax4.grid(True, alpha=0.3)
            
            # 凡例を統合
            lines1, labels1 = ax4.get_legend_handles_labels()
            lines2, labels2 = ax4_twin.get_legend_handles_labels()
            ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # 5. パーセンタイル比較
            ax5 = fig.add_subplot(gs[4])
            ax5.plot(data.index, result.str_percentile, color='cyan', linewidth=1.2, label='STR Percentile')
            ax5.plot(data.index, result.returns_percentile, color='orange', linewidth=1.2, label='Returns Percentile')
            ax5.axhline(y=0.80, color='red', linestyle='--', alpha=0.5, label='High Vol Threshold')
            ax5.axhline(y=0.20, color='blue', linestyle='--', alpha=0.5, label='Low Vol Threshold')
            ax5.axhline(y=0.5, color='gray', linestyle='-', alpha=0.3, label='Median')
            ax5.set_title('Percentile Comparison')
            ax5.set_ylabel('Percentile')
            ax5.set_ylim(0, 1)
            ax5.grid(True, alpha=0.3)
            ax5.legend(loc='upper left')
            
            # 保存
            filename = f"practical_volatility_state_balanced_{symbol}_{timeframe}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            self.logger.info(f"実践的バランス型ボラティリティチャートを保存しました: {filename}")
            plt.show()
            
        except Exception as e:
            self.logger.error(f"チャート作成に失敗: {e}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description='実践的ボラティリティ状態分析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎯 シンプル統計型アプローチ:
  ✨ ローリングZ-スコアによる外れ値検出（閾値1.5 = 上位約7%）
  ✨ ローリングパーセンタイルランクによる相対評価
  ✨ 連続性要件による一時的スパイク除去
  ✨ 統計的に意味のある期間のみを高ボラ判定
  ✨ 理解しやすい計算ロジック
  ✨ 計算効率の最適化

📊 統計的に正確な結果:
  通常7-15%程度の高ボラティリティ期間を検出
        """
    )
    
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイル')
    parser.add_argument('--no-show', action='store_true', help='チャート非表示')
    parser.add_argument('--sensitive', action='store_true', help='高感度モード')
    parser.add_argument('--conservative', action='store_true', help='保守的モード')
    
    args = parser.parse_args()
    
    try:
        print("📊 シンプル統計型ボラティリティ状態分析システム起動中...")
        
        analyzer = PracticalVolatilityAnalyzer(args.config)
        
        # モード設定
        if args.sensitive:
            analyzer.vol_indicator.high_vol_zscore_threshold = 1.2  # より緩い（上位約11%）
            analyzer.vol_indicator.extreme_percentile_threshold = 0.85
            print("⚡ 高感度モードを有効化（Z-スコア1.2以上で判定）")
        
        if args.conservative:
            analyzer.vol_indicator.high_vol_zscore_threshold = 2.0  # より厳格（上位約2.5%）
            analyzer.vol_indicator.extreme_percentile_threshold = 0.95
            print("🛡️ 超保守的モードを有効化（Z-スコア2.0以上で判定）")
        
        # 分析実行
        results = analyzer.run_practical_analysis(show_chart=not args.no_show)
        
        print("\n✅ シンプル統計型分析が完了しました！")
        
        # 実用性評価
        high_vol_pct = results['stats']['high_volatility_percentage']
        if 25 <= high_vol_pct <= 35:
            print("🎯 理想的なボラティリティ分布")
        elif 20 <= high_vol_pct <= 40:
            print("✅ 現実的なボラティリティ分布")
        elif 15 <= high_vol_pct <= 50:
            print("📊 実用的なボラティリティ分布")
        else:
            print("⚠️ パラメータ調整を検討してください")
        
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