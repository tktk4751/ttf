#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import time

# データ取得のための依存関係
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource

# 比較するカルマンフィルター
from indicators.ultimate_ma import UltimateMA
from indicators.ehlers_absolute_ultimate_cycle import EhlersAbsoluteUltimateCycle
from indicators.hyper_adaptive_kalman import HyperAdaptiveKalmanFilter


class KalmanFiltersComparisonChart:
    """
    3つのカルマンフィルターを比較するチャートクラス
    
    - Ultimate MA (adaptive_kalman_filter_numba)
    - Ehlers Absolute Ultimate (ultimate_kalman_smoother) 
    - HyperAdaptiveKalmanFilter (hyper_realtime_kalman + hyper_bidirectional_kalman)
    
    機能:
    - 3つのフィルター結果の同時表示
    - ノイズ除去効果の比較
    - 遅延性能の比較
    - 追従性の比較
    - 詳細な統計情報の出力
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.ultimate_ma = None
        self.ehlers_cycle = None
        self.hyper_kalman = None
        self.fig = None
        self.axes = None
        self.performance_stats = {}
    
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
        print("\nデータを読み込み・処理中...")
        raw_data = data_loader.load_data_from_config(config)
        processed_data = {
            symbol: data_processor.process(df)
            for symbol, df in raw_data.items()
        }
        
        # 最初のシンボルのデータを取得
        first_symbol = next(iter(processed_data))
        self.data = processed_data[first_symbol]
        
        print(f"データ読み込み完了: {first_symbol}")
        print(f"期間: {self.data.index.min()} → {self.data.index.max()}")
        print(f"データ数: {len(self.data)}")
        
        return self.data

    def calculate_all_filters(self,
                            # Ultimate MA パラメータ
                            ultimate_ma_super_smooth_period: int = 10,
                            ultimate_ma_zero_lag_period: int = 21,
                            ultimate_ma_src_type: str = 'hlc3',
                            # Ehlers Absolute Ultimate パラメータ  
                            ehlers_cycle_part: float = 1.0,
                            ehlers_max_output: int = 120,
                            ehlers_min_output: int = 5,
                            ehlers_period_range: Tuple[int, int] = (5, 120),
                            ehlers_src_type: str = 'hlc3',
                            # HyperAdaptiveKalman パラメータ
                            hyper_processing_mode: str = 'adaptive',
                            hyper_market_regime_window: int = 20,
                            hyper_base_process_noise: float = 1e-6,
                            hyper_base_observation_noise: float = 0.001,
                            hyper_src_type: str = 'hlc3'
                           ) -> None:
        """
        3つのカルマンフィルターを計算する
        
        Args:
            ultimate_ma_super_smooth_period: Ultimate MA スーパースムーザー期間
            ultimate_ma_zero_lag_period: Ultimate MA ゼロラグEMA期間
            ultimate_ma_src_type: Ultimate MA ソースタイプ
            ehlers_cycle_part: Ehlers サイクル部分倍率
            ehlers_max_output: Ehlers 最大出力値
            ehlers_min_output: Ehlers 最小出力値
            ehlers_period_range: Ehlers 周期範囲
            ehlers_src_type: Ehlers ソースタイプ
            hyper_processing_mode: Hyper 処理モード
            hyper_market_regime_window: Hyper 市場体制検出ウィンドウ
            hyper_base_process_noise: Hyper 基本プロセスノイズ
            hyper_base_observation_noise: Hyper 基本観測ノイズ
            hyper_src_type: Hyper ソースタイプ
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\n🚀 3つのカルマンフィルターを計算中...")
        
        # 1. Ultimate MA（適応的カルマンフィルター使用）
        print("⚡ Ultimate MA 計算中...")
        start_time = time.time()
        self.ultimate_ma = UltimateMA(
            super_smooth_period=ultimate_ma_super_smooth_period,
            zero_lag_period=ultimate_ma_zero_lag_period,
            src_type=ultimate_ma_src_type,
            zero_lag_period_mode='fixed',  # 固定モードで比較
            realtime_window_mode='fixed'
        )
        ultimate_result = self.ultimate_ma.calculate(self.data)
        ultimate_time = time.time() - start_time
        
        # 2. Ehlers Absolute Ultimate（究極のカルマンスムーザー使用）
        print("🌀 Ehlers Absolute Ultimate 計算中...")
        start_time = time.time()
        self.ehlers_cycle = EhlersAbsoluteUltimateCycle(
            cycle_part=ehlers_cycle_part,
            max_output=ehlers_max_output,
            min_output=ehlers_min_output,
            period_range=ehlers_period_range,
            src_type=ehlers_src_type
        )
        ehlers_result = self.ehlers_cycle.calculate(self.data)
        ehlers_time = time.time() - start_time
        
        # 3. HyperAdaptiveKalmanFilter（ハイブリッドカルマンフィルター）
        print("🎯 HyperAdaptiveKalmanFilter 計算中...")
        start_time = time.time()
        self.hyper_kalman = HyperAdaptiveKalmanFilter(
            processing_mode=hyper_processing_mode,
            market_regime_window=hyper_market_regime_window,
            base_process_noise=hyper_base_process_noise,
            base_observation_noise=hyper_base_observation_noise,
            src_type=hyper_src_type
        )
        hyper_result = self.hyper_kalman.calculate(self.data)
        hyper_time = time.time() - start_time
        
        # パフォーマンス統計の計算
        self.performance_stats = {
            'processing_times': {
                'ultimate_ma': ultimate_time,
                'ehlers_absolute': ehlers_time,
                'hyper_adaptive': hyper_time
            },
            'ultimate_ma_stats': self.ultimate_ma.get_noise_reduction_stats(),
            'hyper_kalman_stats': self.hyper_kalman.get_performance_stats(),
            'hyper_comparison': self.hyper_kalman.get_comparison_with_originals()
        }
        
        print("✅ 全フィルター計算完了")
        print(f"処理時間 - Ultimate MA: {ultimate_time:.3f}s, Ehlers: {ehlers_time:.3f}s, Hyper: {hyper_time:.3f}s")
        
    def analyze_performance(self) -> Dict[str, Any]:
        """
        🏆 詳細なパフォーマンス分析を実行する
        
        Returns:
            詳細な分析結果の辞書
        """
        if not all([self.ultimate_ma, self.ehlers_cycle, self.hyper_kalman]):
            raise ValueError("全てのフィルターが計算されていません。calculate_all_filters()を先に実行してください。")
        
        print("\n📊 詳細パフォーマンス分析実行中...")
        
        # 元の価格データ（HLC3）
        raw_prices = (self.data['high'] + self.data['low'] + self.data['close']) / 3
        
        # 各フィルターの結果取得
        ultimate_values = self.ultimate_ma.get_kalman_values()  # カルマンフィルター段階の値
        ehlers_cycles = self.ehlers_cycle.calculate(self.data)  # サイクル値だが内部でカルマンスムーザー使用
        hyper_realtime = self.hyper_kalman.get_realtime_values()
        hyper_bidirectional = self.hyper_kalman.get_bidirectional_values()
        hyper_adaptive = self.hyper_kalman.get_adaptive_values()
        
        # 正規化（同じ価格データに基づく比較のため）
        ehlers_normalized = ehlers_cycles / np.nanmax(ehlers_cycles) * np.nanmax(raw_prices)
        
        # 1. ノイズ除去効果の比較
        raw_volatility = np.nanstd(raw_prices)
        ultimate_volatility = np.nanstd(ultimate_values)
        hyper_rt_volatility = np.nanstd(hyper_realtime)
        hyper_bi_volatility = np.nanstd(hyper_bidirectional)
        hyper_ad_volatility = np.nanstd(hyper_adaptive)
        
        noise_reduction = {
            'raw_volatility': raw_volatility,
            'ultimate_ma_reduction': (raw_volatility - ultimate_volatility) / raw_volatility * 100,
            'hyper_realtime_reduction': (raw_volatility - hyper_rt_volatility) / raw_volatility * 100,
            'hyper_bidirectional_reduction': (raw_volatility - hyper_bi_volatility) / raw_volatility * 100,
            'hyper_adaptive_reduction': (raw_volatility - hyper_ad_volatility) / raw_volatility * 100
        }
        
        # 2. 遅延性の比較（相関分析）
        def calculate_lag_correlation(original, filtered, max_lag=10):
            """遅延相関を計算"""
            correlations = []
            for lag in range(max_lag + 1):
                if lag == 0:
                    corr = np.corrcoef(original[:-1], filtered[:-1])[0, 1]
                else:
                    corr = np.corrcoef(original[:-lag-1], filtered[lag:-1])[0, 1]
                correlations.append(corr)
            return correlations
        
        # 各フィルターの遅延分析
        ultimate_lag_corr = calculate_lag_correlation(raw_prices, ultimate_values)
        hyper_rt_lag_corr = calculate_lag_correlation(raw_prices, hyper_realtime)
        hyper_bi_lag_corr = calculate_lag_correlation(raw_prices, hyper_bidirectional)
        hyper_ad_lag_corr = calculate_lag_correlation(raw_prices, hyper_adaptive)
        
        lag_analysis = {
            'ultimate_ma_peak_correlation': max(ultimate_lag_corr),
            'ultimate_ma_optimal_lag': ultimate_lag_corr.index(max(ultimate_lag_corr)),
            'hyper_realtime_peak_correlation': max(hyper_rt_lag_corr),
            'hyper_realtime_optimal_lag': hyper_rt_lag_corr.index(max(hyper_rt_lag_corr)),
            'hyper_bidirectional_peak_correlation': max(hyper_bi_lag_corr),
            'hyper_bidirectional_optimal_lag': hyper_bi_lag_corr.index(max(hyper_bi_lag_corr)),
            'hyper_adaptive_peak_correlation': max(hyper_ad_lag_corr),
            'hyper_adaptive_optimal_lag': hyper_ad_lag_corr.index(max(hyper_ad_lag_corr))
        }
        
        # 3. 追従性の比較（トレンド追従能力）
        def calculate_trend_following(original, filtered):
            """トレンド追従性を計算"""
            original_diff = np.diff(original)
            filtered_diff = np.diff(filtered)
            
            # 方向一致率
            direction_matches = np.sum((original_diff > 0) == (filtered_diff > 0))
            direction_accuracy = direction_matches / len(original_diff) * 100
            
            # 変化量の相関
            change_correlation = np.corrcoef(original_diff, filtered_diff)[0, 1]
            
            return {
                'direction_accuracy': direction_accuracy,
                'change_correlation': change_correlation
            }
        
        ultimate_trend = calculate_trend_following(raw_prices, ultimate_values)
        hyper_rt_trend = calculate_trend_following(raw_prices, hyper_realtime)
        hyper_bi_trend = calculate_trend_following(raw_prices, hyper_bidirectional)
        hyper_ad_trend = calculate_trend_following(raw_prices, hyper_adaptive)
        
        trend_following = {
            'ultimate_ma': ultimate_trend,
            'hyper_realtime': hyper_rt_trend,
            'hyper_bidirectional': hyper_bi_trend,
            'hyper_adaptive': hyper_ad_trend
        }
        
        # 4. 全体的な品質指標
        def calculate_quality_score(noise_reduction_pct, lag, direction_accuracy, change_correlation):
            """総合品質スコアを計算"""
            # 重み付きスコア（0-100点）
            noise_score = min(noise_reduction_pct, 50) * 2  # 最大100点
            lag_score = max(0, 100 - lag * 10)  # 遅延ペナルティ
            direction_score = direction_accuracy  # 方向一致率
            correlation_score = change_correlation * 100  # 相関スコア
            
            total_score = (noise_score * 0.3 + lag_score * 0.3 + 
                          direction_score * 0.2 + correlation_score * 0.2)
            
            return {
                'total_score': total_score,
                'noise_score': noise_score,
                'lag_score': lag_score,
                'direction_score': direction_score,
                'correlation_score': correlation_score
            }
        
        quality_scores = {
            'ultimate_ma': calculate_quality_score(
                noise_reduction['ultimate_ma_reduction'],
                lag_analysis['ultimate_ma_optimal_lag'],
                ultimate_trend['direction_accuracy'],
                ultimate_trend['change_correlation']
            ),
            'hyper_realtime': calculate_quality_score(
                noise_reduction['hyper_realtime_reduction'],
                lag_analysis['hyper_realtime_optimal_lag'],
                hyper_rt_trend['direction_accuracy'],
                hyper_rt_trend['change_correlation']
            ),
            'hyper_bidirectional': calculate_quality_score(
                noise_reduction['hyper_bidirectional_reduction'],
                lag_analysis['hyper_bidirectional_optimal_lag'],
                hyper_bi_trend['direction_accuracy'],
                hyper_bi_trend['change_correlation']
            ),
            'hyper_adaptive': calculate_quality_score(
                noise_reduction['hyper_adaptive_reduction'],
                lag_analysis['hyper_adaptive_optimal_lag'],
                hyper_ad_trend['direction_accuracy'],
                hyper_ad_trend['change_correlation']
            )
        }
        
        # 総合分析結果
        analysis_result = {
            'noise_reduction': noise_reduction,
            'lag_analysis': lag_analysis,
            'trend_following': trend_following,
            'quality_scores': quality_scores,
            'processing_times': self.performance_stats['processing_times'],
            'winner_analysis': self._determine_winners(quality_scores, noise_reduction, lag_analysis, trend_following)
        }
        
        return analysis_result
    
    def _determine_winners(self, quality_scores, noise_reduction, lag_analysis, trend_following):
        """各カテゴリーの勝者を決定"""
        winners = {}
        
        # 総合品質スコア
        best_overall = max(quality_scores.keys(), key=lambda k: quality_scores[k]['total_score'])
        winners['overall'] = best_overall
        
        # ノイズ除去効果
        noise_keys = ['ultimate_ma_reduction', 'hyper_realtime_reduction', 
                     'hyper_bidirectional_reduction', 'hyper_adaptive_reduction']
        best_noise = max(noise_keys, key=lambda k: noise_reduction[k])
        winners['noise_reduction'] = best_noise.replace('_reduction', '')
        
        # 低遅延性
        lag_keys = ['ultimate_ma_optimal_lag', 'hyper_realtime_optimal_lag',
                   'hyper_bidirectional_optimal_lag', 'hyper_adaptive_optimal_lag']
        best_lag = min(lag_keys, key=lambda k: lag_analysis[k])
        winners['low_latency'] = best_lag.replace('_optimal_lag', '')
        
        # トレンド追従性
        trend_keys = ['ultimate_ma', 'hyper_realtime', 'hyper_bidirectional', 'hyper_adaptive']
        best_trend = max(trend_keys, key=lambda k: trend_following[k]['direction_accuracy'])
        winners['trend_following'] = best_trend
        
        return winners

    def print_detailed_analysis(self, analysis_result: Dict[str, Any]) -> None:
        """詳細分析結果を美しく出力する"""
        print("\n" + "="*80)
        print("🏆 カルマンフィルター詳細比較分析結果")
        print("="*80)
        
        # 1. 処理時間
        print("\n⚡ 処理速度比較:")
        times = analysis_result['processing_times']
        for name, time_val in times.items():
            print(f"  {name:20}: {time_val:.4f}秒")
        fastest = min(times.keys(), key=lambda k: times[k])
        print(f"  🥇 最速: {fastest}")
        
        # 2. ノイズ除去効果
        print("\n🔇 ノイズ除去効果比較:")
        noise = analysis_result['noise_reduction']
        print(f"  元データ標準偏差: {noise['raw_volatility']:.6f}")
        for key, value in noise.items():
            if 'reduction' in key:
                name = key.replace('_reduction', '').replace('_', ' ').title()
                print(f"  {name:20}: {value:.2f}%")
        
        # 3. 遅延性分析
        print("\n⚡ 遅延性分析:")
        lag = analysis_result['lag_analysis']
        filters = ['ultimate_ma', 'hyper_realtime', 'hyper_bidirectional', 'hyper_adaptive']
        for filter_name in filters:
            lag_key = f"{filter_name}_optimal_lag"
            corr_key = f"{filter_name}_peak_correlation"
            if lag_key in lag and corr_key in lag:
                name = filter_name.replace('_', ' ').title()
                print(f"  {name:20}: 最適遅延 {lag[lag_key]}期間, 相関 {lag[corr_key]:.4f}")
        
        # 4. トレンド追従性
        print("\n📈 トレンド追従性:")
        trend = analysis_result['trend_following']
        for filter_name, stats in trend.items():
            name = filter_name.replace('_', ' ').title()
            print(f"  {name:20}: 方向一致率 {stats['direction_accuracy']:.2f}%, "
                  f"変化相関 {stats['change_correlation']:.4f}")
        
        # 5. 総合品質スコア
        print("\n🏆 総合品質スコア (0-100点):")
        quality = analysis_result['quality_scores']
        sorted_scores = sorted(quality.items(), key=lambda x: x[1]['total_score'], reverse=True)
        
        for i, (filter_name, scores) in enumerate(sorted_scores):
            rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}位"
            name = filter_name.replace('_', ' ').title()
            print(f"  {rank} {name:18}: {scores['total_score']:.1f}点")
            print(f"      ├ ノイズ除去: {scores['noise_score']:.1f}点")
            print(f"      ├ 低遅延性: {scores['lag_score']:.1f}点")
            print(f"      ├ 方向一致: {scores['direction_score']:.1f}点")
            print(f"      └ 変化相関: {scores['correlation_score']:.1f}点")
        
        # 6. カテゴリー別勝者
        print("\n🏅 カテゴリー別勝者:")
        winners = analysis_result['winner_analysis']
        categories = {
            'overall': '総合品質',
            'noise_reduction': 'ノイズ除去',
            'low_latency': '低遅延性',
            'trend_following': 'トレンド追従'
        }
        
        for category, japanese_name in categories.items():
            winner = winners[category].replace('_', ' ').title()
            print(f"  {japanese_name:12}: 🏆 {winner}")
        
        # 7. 推奨事項
        print("\n💡 用途別推奨:")
        overall_winner = winners['overall'].replace('_', ' ').title()
        latency_winner = winners['low_latency'].replace('_', ' ').title()
        quality_winner = winners['noise_reduction'].replace('_', ' ').title()
        
        print(f"  🚀 リアルタイム取引: {latency_winner}")
        print(f"  📊 高品質分析: {quality_winner}")
        print(f"  ⚖️  バランス重視: {overall_winner}")
        
        print("\n" + "="*80)

    def plot_comparison(self, 
                       title: str = "カルマンフィルター比較", 
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       show_volume: bool = True,
                       figsize: Tuple[int, int] = (16, 14),
                       style: str = 'yahoo',
                       savefig: Optional[str] = None) -> None:
        """
        3つのカルマンフィルターの比較チャートを描画する
        
        Args:
            title: チャートのタイトル
            start_date: 表示開始日（フォーマット: YYYY-MM-DD）
            end_date: 表示終了日（フォーマット: YYYY-MM-DD）
            show_volume: 出来高を表示するか
            figsize: 図のサイズ
            style: mplfinanceのスタイル
            savefig: 保存先のパス（指定しない場合は表示のみ）
        """
        if not all([self.ultimate_ma, self.ehlers_cycle, self.hyper_kalman]):
            raise ValueError("全てのフィルターが計算されていません。calculate_all_filters()を先に実行してください。")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
        
        # 各フィルターの値を取得
        ultimate_kalman = self.ultimate_ma.get_kalman_values()
        ultimate_final = self.ultimate_ma.get_values()
        
        hyper_realtime = self.hyper_kalman.get_realtime_values()
        hyper_bidirectional = self.hyper_kalman.get_bidirectional_values()
        hyper_adaptive = self.hyper_kalman.get_adaptive_values()
        hyper_regimes = self.hyper_kalman.get_market_regimes()
        hyper_confidence = self.hyper_kalman.get_confidence_scores()
        
        # 元価格データ
        raw_hlc3 = (self.data['high'] + self.data['low'] + self.data['close']) / 3
        
        # 全データの時系列データフレームを作成
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                'raw_hlc3': raw_hlc3,
                'ultimate_kalman': ultimate_kalman,
                'ultimate_final': ultimate_final,
                'hyper_realtime': hyper_realtime,
                'hyper_bidirectional': hyper_bidirectional,
                'hyper_adaptive': hyper_adaptive,
                'market_regimes': hyper_regimes,
                'confidence_scores': hyper_confidence
            }
        )
        
        # 絞り込み後のデータに結合
        df = df.join(full_df)
        
        print(f"チャートデータ準備完了 - 行数: {len(df)}")
        
        # mplfinanceでプロット用の設定
        main_plots = []
        
        # 元データ（薄いグレー）
        main_plots.append(mpf.make_addplot(df['raw_hlc3'], color='lightgray', width=1, alpha=0.5, label='Raw HLC3'))
        
        # Ultimate MA（青系）
        main_plots.append(mpf.make_addplot(df['ultimate_kalman'], color='blue', width=1.5, label='Ultimate MA (Kalman)'))
        main_plots.append(mpf.make_addplot(df['ultimate_final'], color='darkblue', width=1.0, alpha=0.7, label='Ultimate MA (Final)'))
        
        # HyperAdaptiveKalman（緑・赤・紫系）
        main_plots.append(mpf.make_addplot(df['hyper_realtime'], color='green', width=1.5, label='Hyper (Realtime)'))
        main_plots.append(mpf.make_addplot(df['hyper_bidirectional'], color='red', width=1.5, label='Hyper (Bidirectional)'))
        main_plots.append(mpf.make_addplot(df['hyper_adaptive'], color='purple', width=2, label='Hyper (Adaptive)'))
        
        # サブプロット：市場体制と信頼度
        regime_panel = mpf.make_addplot(df['market_regimes'], panel=1, color='orange', width=1.2, 
                                       ylabel='Market Regime', secondary_y=False, label='Regime')
        
        confidence_panel = mpf.make_addplot(df['confidence_scores'], panel=2, color='brown', width=1.2, 
                                          ylabel='Confidence', secondary_y=False, label='Confidence')
        
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
        
        # 出来高とパネル設定
        if show_volume:
            kwargs['volume'] = True
            kwargs['panel_ratios'] = (5, 1, 1, 1)  # メイン:出来高:体制:信頼度
            # パネル番号を調整
            regime_panel = mpf.make_addplot(df['market_regimes'], panel=2, color='orange', width=1.2, 
                                           ylabel='Market Regime', secondary_y=False, label='Regime')
            confidence_panel = mpf.make_addplot(df['confidence_scores'], panel=3, color='brown', width=1.2, 
                                              ylabel='Confidence', secondary_y=False, label='Confidence')
        else:
            kwargs['volume'] = False
            kwargs['panel_ratios'] = (5, 1, 1)  # メイン:体制:信頼度
        
        # 全プロットを結合
        all_plots = main_plots + [regime_panel, confidence_panel]
        kwargs['addplot'] = all_plots
        
        # プロット実行
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例の追加
        axes[0].legend(['Raw HLC3', 'Ultimate MA (Kalman)', 'Ultimate MA (Final)', 
                       'Hyper (Realtime)', 'Hyper (Bidirectional)', 'Hyper (Adaptive)'], 
                      loc='upper left', fontsize=8)
        
        self.fig = fig
        self.axes = axes
        
        # 参照線の追加
        if show_volume:
            # 市場体制パネル（0=レンジング, 1=トレンディング, 2=高ボラティリティ）
            axes[2].axhline(y=0.5, color='blue', linestyle='--', alpha=0.5)
            axes[2].axhline(y=1.5, color='orange', linestyle='--', alpha=0.5)
            axes[2].set_ylim(-0.5, 2.5)
            
            # 信頼度パネル
            axes[3].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
            axes[3].axhline(y=0.8, color='green', linestyle='--', alpha=0.5)
            axes[3].set_ylim(0, 1)
        else:
            # 市場体制パネル
            axes[1].axhline(y=0.5, color='blue', linestyle='--', alpha=0.5)
            axes[1].axhline(y=1.5, color='orange', linestyle='--', alpha=0.5)
            axes[1].set_ylim(-0.5, 2.5)
            
            # 信頼度パネル
            axes[2].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
            axes[2].axhline(y=0.8, color='green', linestyle='--', alpha=0.5)
            axes[2].set_ylim(0, 1)
        
        # 保存または表示
        if savefig:
            plt.savefig(savefig, dpi=150, bbox_inches='tight')
            print(f"チャートを保存しました: {savefig}")
        else:
            plt.tight_layout()
            plt.show()


def main():
    """メイン関数"""
    import argparse
    parser = argparse.ArgumentParser(description='カルマンフィルター比較チャートの描画')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    parser.add_argument('--hyper-mode', type=str, default='adaptive', 
                       choices=['realtime', 'high_quality', 'adaptive'], 
                       help='HyperAdaptiveKalmanの処理モード')
    args = parser.parse_args()
    
    # 比較チャートを作成
    chart = KalmanFiltersComparisonChart()
    
    try:
        # データ読み込み
        chart.load_data_from_config(args.config)
        
        # 全フィルター計算
        chart.calculate_all_filters(hyper_processing_mode=args.hyper_mode)
        
        # 詳細分析実行
        analysis_result = chart.analyze_performance()
        
        # 分析結果出力
        chart.print_detailed_analysis(analysis_result)
        
        # チャート描画
        chart.plot_comparison(
            start_date=args.start,
            end_date=args.end,
            savefig=args.output
        )
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 