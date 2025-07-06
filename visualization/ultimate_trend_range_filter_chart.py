#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultimate Trend Range Filter チャート視覚化システム

トレンド/レンジの明確な視覚表現を提供：
- チャート背景色でトレンド/レンジ状態を表示
- 多層的な分析ビュー
- リアルタイム状態表示
- 詳細な統計情報
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import seaborn as sns
from typing import Optional, Tuple, Dict, Any
import sys
import os
import yaml
import mplfinance as mpf
from pathlib import Path

# パス追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicators.ultimate_trend_range_filter import UltimateTrendRangeFilter, UltimateTrendRangeResult
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource


class UltimateTrendRangeFilterChart:
    """
    Ultimate Trend Range Filterを表示するローソク足チャートクラス
    
    - ローソク足と出来高
    - トレンド/レンジ状態の背景色表示
    - トレンド強度・信頼度・各種メトリクスの表示
    - 実践的な視覚化による判定結果の確認
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.ultimate_trend_range_filter = None
        self.fig = None
        self.axes = None
    
    def load_data_from_config(self, config_path: str) -> pd.DataFrame:
        """
        設定ファイルからデータを読み込む
        
        Args:
            config_path: 設定ファイルのパス
            
        Returns:
            処理済みのデータフレーム
        """
        # 設定ファイルの読み込み
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        # データの準備
        binance_config = config.get('binance_data', {})
        data_dir = binance_config.get('data_dir', 'data/binance')
        binance_data_source = BinanceDataSource(data_dir)
        
        # CSVデータソースはダミーとして渡す（Binanceデータソースのみを使用）
        dummy_csv_source = CSVDataSource("dummy")
        data_loader = DataLoader(
            data_source=dummy_csv_source,
            binance_data_source=binance_data_source
        )
        data_processor = DataProcessor()
        
        # データの読み込みと処理
        print("\n🔄 実際の相場データを読み込み・処理中...")
        raw_data = data_loader.load_data_from_config(config)
        processed_data = {
            symbol: data_processor.process(df)
            for symbol, df in raw_data.items()
        }
        
        # 最初のシンボルのデータを取得
        first_symbol = next(iter(processed_data))
        self.data = processed_data[first_symbol]
        
        print(f"✅ データ読み込み完了: {first_symbol}")
        print(f"📅 期間: {self.data.index.min()} → {self.data.index.max()}")
        print(f"📊 データ数: {len(self.data)}")
        print(f"💰 価格範囲: ${self.data['close'].min():.2f} - ${self.data['close'].max():.2f}")
        
        return self.data

    def calculate_indicators(self,
                            # コアパラメータ
                            analysis_period: int = 20,
                            ensemble_window: int = 50,
                            
                            # 判定しきい値
                            trend_threshold: float = 0.4,        # 実践的な値
                            strong_trend_threshold: float = 0.7, # 実践的な値
                            
                            # アルゴリズム有効化
                            enable_advanced_trend: bool = True,
                            enable_range_analysis: bool = True,
                            enable_multi_scale: bool = True,
                            enable_noise_suppression: bool = True,
                            enable_harmonic_patterns: bool = True,
                            enable_volatility_regime: bool = True,
                            enable_ml_features: bool = True,
                            
                            # 高度設定
                            multi_scale_periods: Optional[np.ndarray] = None,
                            noise_adaptation_factor: float = 0.1,
                            harmonic_detection_period: int = 30
                           ) -> None:
        """
        Ultimate Trend Range Filterを計算する
        
        Args:
            analysis_period: 基本分析期間
            ensemble_window: アンサンブル統合ウィンドウ
            trend_threshold: トレンド判定しきい値
            strong_trend_threshold: 強トレンド判定しきい値
            enable_*: 各アルゴリズムの有効化フラグ
            multi_scale_periods: マルチスケール解析期間
            noise_adaptation_factor: ノイズ適応係数
            harmonic_detection_period: ハーモニック検出期間
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\n🔬 Ultimate Trend Range Filterを計算中...")
        
        # Ultimate Trend Range Filterを計算
        self.ultimate_trend_range_filter = UltimateTrendRangeFilter(
            analysis_period=analysis_period,
            ensemble_window=ensemble_window,
            enable_advanced_trend=enable_advanced_trend,
            enable_range_analysis=enable_range_analysis,
            enable_multi_scale=enable_multi_scale,
            enable_noise_suppression=enable_noise_suppression,
            enable_harmonic_patterns=enable_harmonic_patterns,
            enable_volatility_regime=enable_volatility_regime,
            enable_ml_features=enable_ml_features,
            trend_threshold=trend_threshold,
            strong_trend_threshold=strong_trend_threshold,
            multi_scale_periods=multi_scale_periods,
            noise_adaptation_factor=noise_adaptation_factor,
            harmonic_detection_period=harmonic_detection_period
        )
        
        # 計算実行
        print("⚡ 計算を実行します...")
        result = self.ultimate_trend_range_filter.calculate(self.data)
        
        print(f"✅ 計算完了")
        print(f"   トレンド強度範囲: {np.nanmin(result.trend_strength):.3f} - {np.nanmax(result.trend_strength):.3f}")
        print(f"   トレンド判定率: {np.sum(result.trend_classification == 1.0)/len(result.trend_classification)*100:.1f}%")
        print(f"   平均信頼度: {np.nanmean(result.confidence_score):.3f}")
        print(f"   現在状態: {result.current_state}")
        
        print("🎯 Ultimate Trend Range Filter計算完了")
            
    def plot(self, 
            title: str = "Ultimate Trend Range Filter", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            figsize: Tuple[int, int] = (16, 12),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        ローソク足チャートとUltimate Trend Range Filterを描画する
        
        Args:
            title: チャートのタイトル
            start_date: 表示開始日（フォーマット: YYYY-MM-DD）
            end_date: 表示終了日（フォーマット: YYYY-MM-DD）
            show_volume: 出来高を表示するか
            figsize: 図のサイズ
            style: mplfinanceのスタイル
            savefig: 保存先のパス（指定しない場合は表示のみ）
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        if self.ultimate_trend_range_filter is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        # Ultimate Trend Range Filterの結果を取得
        print("📊 結果データを取得中...")
        result = self.ultimate_trend_range_filter.get_result()
        
        if result is None:
            raise ValueError("インジケーター結果が取得できません")
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'trend_strength': result.trend_strength,
                'trend_classification': result.trend_classification,
                'range_probability': result.range_probability,
                'trend_probability': result.trend_probability,
                'confidence_score': result.confidence_score,
                'signal_quality': result.signal_quality,
                'directional_movement': result.directional_movement,
                'consolidation_index': result.consolidation_index,
                'volatility_regime': result.volatility_regime,
                'harmonic_strength': result.harmonic_strength
            }
        )
        
        # 絞り込み後のデータに対してインジケーターデータを結合
        df = df.join(full_df)
        
        print(f"📈 チャートデータ準備完了 - 行数: {len(df)}")
        print(f"   トレンド強度NaN: {df['trend_strength'].isna().sum()}")
        print(f"   信頼度NaN: {df['confidence_score'].isna().sum()}")
        
        # トレンド/レンジ状態に基づく背景色の準備
        # NaN値や未定義値を適切に処理
        trend_classification_clean = df['trend_classification'].fillna(0.0)  # NaNはレンジとして扱う
        
        # 完全な2値分類を保証（トレンドかレンジのどちらか必ず）
        df['trend_background'] = np.where(trend_classification_clean == 1.0, 1, 0)
        df['range_background'] = np.where(trend_classification_clean == 0.0, 1, 0)
        
        # 安全性チェック：すべての期間がトレンドかレンジのどちらかに分類されていることを確認
        total_classified = df['trend_background'].sum() + df['range_background'].sum()
        if total_classified != len(df):
            print(f"⚠️  警告: 未分類期間が {len(df) - total_classified} 個あります。レンジとして処理します。")
            # 未分類期間をレンジとして強制分類
            unclassified_mask = (df['trend_background'] == 0) & (df['range_background'] == 0)
            df.loc[unclassified_mask, 'range_background'] = 1
        
        # 強度レベルに基づく表示データの準備
        df['strong_trend'] = np.where(df['trend_strength'] > 0.7, df['trend_strength'], np.nan)
        df['medium_trend'] = np.where((df['trend_strength'] >= 0.4) & (df['trend_strength'] <= 0.7), df['trend_strength'], np.nan)
        df['weak_signal'] = np.where(df['trend_strength'] < 0.4, df['trend_strength'], np.nan)
        
        # 信頼度レベルに基づく表示データの準備
        df['high_confidence'] = np.where(df['confidence_score'] > 0.8, df['confidence_score'], np.nan)
        df['medium_confidence'] = np.where((df['confidence_score'] >= 0.5) & (df['confidence_score'] <= 0.8), df['confidence_score'], np.nan)
        df['low_confidence'] = np.where(df['confidence_score'] < 0.5, df['confidence_score'], np.nan)
        
        # mplfinanceでプロット用の設定
        main_plots = []
        
        # 1. トレンド/レンジ背景（メインチャート上のフィル）
        # バックグラウンド表示用の高さを計算（チャートの全高を使用）
        chart_height = df['high'].max() - df['low'].min()
        background_height = chart_height * 1.1  # チャート全体をカバーする高さ
        background_base = df['low'].min() - chart_height * 0.05  # ベースライン
        
        # トレンド期間の緑背景
        trend_background_values = np.where(df['trend_background'] == 1, background_height, 0)
        main_plots.append(mpf.make_addplot(trend_background_values, 
                                          type='bar', color='green', alpha=0.15, panel=0, 
                                          secondary_y=False, label='Trend Period'))
        
        # レンジ期間のグレー背景
        range_background_values = np.where(df['range_background'] == 1, background_height, 0)
        main_plots.append(mpf.make_addplot(range_background_values, 
                                          type='bar', color='gray', alpha=0.1, panel=0, 
                                          secondary_y=False, label='Range Period'))
        
        # 2. トレンド強度プロット（パネル1）
        trend_strength_panel = mpf.make_addplot(df['strong_trend'], panel=1, color='darkgreen', width=2.5, 
                                               ylabel='Trend Strength', secondary_y=False, label='Strong Trend')
        medium_trend_panel = mpf.make_addplot(df['medium_trend'], panel=1, color='orange', width=2, 
                                             secondary_y=False, label='Medium Trend')
        weak_signal_panel = mpf.make_addplot(df['weak_signal'], panel=1, color='gray', width=1.5, 
                                            secondary_y=False, label='Weak Signal')
        
        # 3. 信頼度プロット（パネル2）
        high_conf_panel = mpf.make_addplot(df['high_confidence'], panel=2, color='blue', width=2.5, 
                                          ylabel='Confidence Score', secondary_y=False, label='High Confidence')
        medium_conf_panel = mpf.make_addplot(df['medium_confidence'], panel=2, color='purple', width=2, 
                                            secondary_y=False, label='Medium Confidence')
        low_conf_panel = mpf.make_addplot(df['low_confidence'], panel=2, color='red', width=1.5, 
                                         secondary_y=False, label='Low Confidence')
        
        # 4. 確率プロット（パネル3）
        trend_prob_panel = mpf.make_addplot(df['trend_probability'], panel=3, color='green', width=2, 
                                           ylabel='Probabilities', secondary_y=False, label='Trend Prob')
        range_prob_panel = mpf.make_addplot(df['range_probability'], panel=3, color='red', width=2, 
                                           secondary_y=False, label='Range Prob')
        
        # 5. 成分分析プロット（パネル4）
        directional_panel = mpf.make_addplot(df['directional_movement'], panel=4, color='cyan', width=1.5, 
                                            ylabel='Components', secondary_y=False, label='Directional')
        consolidation_panel = mpf.make_addplot(df['consolidation_index'], panel=4, color='magenta', width=1.5, 
                                              secondary_y=False, label='Consolidation')
        
        # mplfinanceの設定
        kwargs = dict(
            type='candle',
            figsize=figsize,
            title=title,
            style=style,
            datetime_format='%Y-%m-%d',
            xrotation=45,
            returnfig=True
        )
        
        # 出来高とパネル配置の設定
        if show_volume:
            kwargs['volume'] = True
            kwargs['panel_ratios'] = (4, 1, 1, 1, 1, 1)  # メイン:出来高:強度:信頼度:確率:成分
            # 出来高を表示する場合は、パネル番号を+1する
            trend_strength_panel = mpf.make_addplot(df['strong_trend'], panel=2, color='darkgreen', width=2.5, 
                                                   ylabel='Trend Strength', secondary_y=False, label='Strong Trend')
            medium_trend_panel = mpf.make_addplot(df['medium_trend'], panel=2, color='orange', width=2, 
                                                 secondary_y=False, label='Medium Trend')
            weak_signal_panel = mpf.make_addplot(df['weak_signal'], panel=2, color='gray', width=1.5, 
                                                secondary_y=False, label='Weak Signal')
            
            high_conf_panel = mpf.make_addplot(df['high_confidence'], panel=3, color='blue', width=2.5, 
                                              ylabel='Confidence Score', secondary_y=False, label='High Confidence')
            medium_conf_panel = mpf.make_addplot(df['medium_confidence'], panel=3, color='purple', width=2, 
                                                secondary_y=False, label='Medium Confidence')
            low_conf_panel = mpf.make_addplot(df['low_confidence'], panel=3, color='red', width=1.5, 
                                             secondary_y=False, label='Low Confidence')
            
            trend_prob_panel = mpf.make_addplot(df['trend_probability'], panel=4, color='green', width=2, 
                                               ylabel='Probabilities', secondary_y=False, label='Trend Prob')
            range_prob_panel = mpf.make_addplot(df['range_probability'], panel=4, color='red', width=2, 
                                               secondary_y=False, label='Range Prob')
            
            directional_panel = mpf.make_addplot(df['directional_movement'], panel=5, color='cyan', width=1.5, 
                                                ylabel='Components', secondary_y=False, label='Directional')
            consolidation_panel = mpf.make_addplot(df['consolidation_index'], panel=5, color='magenta', width=1.5, 
                                                  secondary_y=False, label='Consolidation')
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (4, 1, 1, 1, 1)  # メイン:強度:信頼度:確率:成分
        
        # すべてのプロットを結合
        all_plots = main_plots + [
            trend_strength_panel, medium_trend_panel, weak_signal_panel,
            high_conf_panel, medium_conf_panel, low_conf_panel,
            trend_prob_panel, range_prob_panel,
            directional_panel, consolidation_panel
        ]
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        self.fig = fig
        self.axes = axes
        
        # 各パネルに参照線を追加
        panel_offset = 1 if show_volume else 0
        
        # トレンド強度パネル（0.4と0.7のしきい値線）
        strength_panel_idx = 1 + panel_offset
        axes[strength_panel_idx].axhline(y=0.4, color='orange', linestyle='--', alpha=0.7, label='Trend Threshold')
        axes[strength_panel_idx].axhline(y=0.7, color='green', linestyle='--', alpha=0.7, label='Strong Trend Threshold')
        axes[strength_panel_idx].axhline(y=0.0, color='black', linestyle='-', alpha=0.3)
        axes[strength_panel_idx].axhline(y=1.0, color='black', linestyle='-', alpha=0.3)
        
        # 信頼度パネル（0.5と0.8のしきい値線）
        confidence_panel_idx = 2 + panel_offset
        axes[confidence_panel_idx].axhline(y=0.5, color='purple', linestyle='--', alpha=0.7, label='Medium Confidence')
        axes[confidence_panel_idx].axhline(y=0.8, color='blue', linestyle='--', alpha=0.7, label='High Confidence')
        axes[confidence_panel_idx].axhline(y=0.0, color='black', linestyle='-', alpha=0.3)
        axes[confidence_panel_idx].axhline(y=1.0, color='black', linestyle='-', alpha=0.3)
        
        # 確率パネル（0.5の中央線）
        probability_panel_idx = 3 + panel_offset
        axes[probability_panel_idx].axhline(y=0.5, color='black', linestyle='-', alpha=0.5)
        axes[probability_panel_idx].axhline(y=0.0, color='black', linestyle='-', alpha=0.3)
        axes[probability_panel_idx].axhline(y=1.0, color='black', linestyle='-', alpha=0.3)
        
        # 成分分析パネル（中央線）
        components_panel_idx = 4 + panel_offset
        comp_mean = np.nanmean([np.nanmean(df['directional_movement']), np.nanmean(df['consolidation_index'])])
        axes[components_panel_idx].axhline(y=comp_mean, color='black', linestyle='-', alpha=0.3)
        
        # 統計情報の表示
        print(f"\n📈 チャート統計:")
        total_points = len(df)
        trend_points = np.sum(df['trend_background'] == 1)
        range_points = np.sum(df['range_background'] == 1)
        
        print(f"   総データ点数: {total_points}")
        print(f"   トレンド期間: {trend_points} ({trend_points/total_points*100:.1f}%)")
        print(f"   レンジ期間: {range_points} ({range_points/total_points*100:.1f}%)")
        print(f"   分類完了率: {(trend_points + range_points)/total_points*100:.1f}%")
        print(f"   平均トレンド強度: {np.nanmean(df['trend_strength']):.3f}")
        print(f"   平均信頼度: {np.nanmean(df['confidence_score']):.3f}")
        
        # 強度分布
        strong_periods = np.sum(df['trend_strength'] > 0.7)
        medium_periods = np.sum((df['trend_strength'] >= 0.4) & (df['trend_strength'] <= 0.7))
        weak_periods = np.sum(df['trend_strength'] < 0.4)
        
        print(f"   強度分布:")
        print(f"     強いシグナル (>0.7): {strong_periods} ({strong_periods/total_points*100:.1f}%)")
        print(f"     中程度シグナル (0.4-0.7): {medium_periods} ({medium_periods/total_points*100:.1f}%)")
        print(f"     弱いシグナル (<0.4): {weak_periods} ({weak_periods/total_points*100:.1f}%)")
        
        # 保存または表示
        if savefig:
            plt.savefig(savefig, dpi=150, bbox_inches='tight')
            print(f"📁 チャートを保存しました: {savefig}")
        else:
            plt.tight_layout()
            plt.show()


def main():
    """メイン関数"""
    # コマンドライン引数を処理
    import argparse
    parser = argparse.ArgumentParser(description='Ultimate Trend Range Filterの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--trend-threshold', type=float, default=0.4, help='トレンド判定しきい値')
    parser.add_argument('--strong-threshold', type=float, default=0.7, help='強トレンド判定しきい値')
    parser.add_argument('--analysis-period', type=int, default=20, help='分析期間')
    args = parser.parse_args()
    
    # チャートを作成
    chart = UltimateTrendRangeFilterChart()
    chart.load_data_from_config(args.config)
    chart.calculate_indicators(
        analysis_period=args.analysis_period,
        trend_threshold=args.trend_threshold,
        strong_trend_threshold=args.strong_threshold
    )
    chart.plot(
        start_date=args.start,
        end_date=args.end,
        savefig=args.output
    )


if __name__ == "__main__":
    main() 