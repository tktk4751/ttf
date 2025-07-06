#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🌌 Quantum Supreme Breakout Channel Chart Visualizer V1.0
人類史上最強ボラティリティベースブレイクアウトチャネル チャート描画システム

設定ファイルから実際の相場データを取得し、Quantum Supreme Breakout Channelを美しく描画
"""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# データ取得のための依存関係
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource

# Quantum Supreme Breakout Channel インジケーター
from indicators.quantum_supreme_breakout_channel import QuantumSupremeBreakoutChannel


class QuantumSupremeBreakoutChannelChart:
    """
    🌌 Quantum Supreme Breakout Channel を表示するローソク足チャートクラス
    
    主要機能:
    - ローソク足と出来高表示
    - 動的適応チャネル（上位・中央・下位）
    - 市場レジーム表示（トレンド・レンジ・ブレイクアウト）
    - 動的乗数（1.5-8.0）のリアルタイム表示
    - 量子メトリクス（コヒーレンス・もつれ・重ね合わせ）
    - ブレイクアウト確率・シグナル
    - トレンド強度・効率性スコア
    """
    
    def __init__(self):
        """初期化"""
        self.data = None
        self.qsbc = None
        self.result = None
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
        print("\n🌌 Quantum Supreme Breakout Channel - データ読み込み中...")
        raw_data = data_loader.load_data_from_config(config)
        processed_data = {
            symbol: data_processor.process(df)
            for symbol, df in raw_data.items()
        }
        
        # 最初のシンボルのデータを取得
        first_symbol = next(iter(processed_data))
        self.data = processed_data[first_symbol]
        
        print(f"✅ データ読み込み完了: {first_symbol}")
        print(f"📊 期間: {self.data.index.min()} → {self.data.index.max()}")
        print(f"📈 データ数: {len(self.data)}")
        
        return self.data

    def calculate_indicators(self,
                            # 基本設定
                            analysis_period: int = 21,
                            src_type: str = 'hlc3',
                                        min_multiplier: float = 1.0,
            max_multiplier: float = 6.0,
                            smoothing_alpha: float = 0.25,
                            
                            # 量子パラメータ
                            quantum_coherence_threshold: float = 0.75,
                            entanglement_factor: float = 0.618,
                            superposition_weight: float = 0.5,
                            
                            # 適応パラメータ
                            trend_sensitivity: float = 0.85,
                            range_sensitivity: float = 0.75,
                            adaptation_speed: float = 0.12,
                            memory_decay: float = 0.95,
                            multiplier_smoothing_mode: str = 'adaptive',
                            ultra_low_latency: bool = True,
                            smooth_transition_threshold: float = 0.3,
                            
                            # アルゴリズム有効化
                            enable_quantum_hilbert: bool = True,
                            enable_fractal_analysis: bool = True,
                            enable_wavelet_decomp: bool = True,
                            enable_kalman_quantum: bool = True,
                            enable_garch_volatility: bool = True,
                            enable_regime_switching: bool = True,
                            enable_spectral_analysis: bool = True,
                            enable_entropy_analysis: bool = True,
                            enable_chaos_theory: bool = True,
                            enable_efficiency_ratio: bool = True,
                            enable_x_trend_index: bool = True,
                            enable_roc_persistence: bool = True
                           ) -> None:
        """
        🌌 Quantum Supreme Breakout Channel を計算する
        
        Args:
            analysis_period: 基本分析期間
            src_type: 価格ソース ('hlc3', 'close', 'ohlc4', etc.)
            min_multiplier: 最小乗数（トレンド時）
            max_multiplier: 最大乗数（レンジ時）
            smoothing_alpha: スムージング係数
            quantum_coherence_threshold: 量子コヒーレンス閾値
            entanglement_factor: もつれファクター
            superposition_weight: 重ね合わせ重み
            trend_sensitivity: トレンド感度
            range_sensitivity: レンジ感度
            adaptation_speed: 適応速度
            memory_decay: メモリー減衰
            multiplier_smoothing_mode: 乗数スムージングモード
            ultra_low_latency: 超低遅延モード
            smooth_transition_threshold: スムーズ遷移閾値
            enable_*: 各アルゴリズムの有効化フラグ
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        print("\n🚀 Quantum Supreme Breakout Channel 計算開始...")
        
        # Quantum Supreme Breakout Channel を初期化
        self.qsbc = QuantumSupremeBreakoutChannel(
            analysis_period=analysis_period,
            src_type=src_type,
            min_multiplier=min_multiplier,
            max_multiplier=max_multiplier,
            smoothing_alpha=smoothing_alpha,
            quantum_coherence_threshold=quantum_coherence_threshold,
            entanglement_factor=entanglement_factor,
            superposition_weight=superposition_weight,
            trend_sensitivity=trend_sensitivity,
            range_sensitivity=range_sensitivity,
            adaptation_speed=adaptation_speed,
            memory_decay=memory_decay,
            multiplier_smoothing_mode=multiplier_smoothing_mode,
            ultra_low_latency=ultra_low_latency,
            smooth_transition_threshold=smooth_transition_threshold,
            enable_quantum_hilbert=enable_quantum_hilbert,
            enable_fractal_analysis=enable_fractal_analysis,
            enable_wavelet_decomp=enable_wavelet_decomp,
            enable_kalman_quantum=enable_kalman_quantum,
            enable_garch_volatility=enable_garch_volatility,
            enable_regime_switching=enable_regime_switching,
            enable_spectral_analysis=enable_spectral_analysis,
            enable_entropy_analysis=enable_entropy_analysis,
            enable_chaos_theory=enable_chaos_theory,
            enable_efficiency_ratio=enable_efficiency_ratio,
            enable_x_trend_index=enable_x_trend_index,
            enable_roc_persistence=enable_roc_persistence
        )
        
        # インジケーター状態のデバッグ情報
        print(f"🔧 インジケーター設定:")
        print(f"   enable_kalman_quantum: {self.qsbc.enable_kalman_quantum}")
        print(f"   enable_regime_switching: {self.qsbc.enable_regime_switching}")
        print(f"   enable_garch_volatility: {self.qsbc.enable_garch_volatility}")
        print(f"   enable_efficiency_ratio: {self.qsbc.enable_efficiency_ratio}")
        print(f"   quantum_hyper_ma 存在: {hasattr(self.qsbc, 'quantum_hyper_ma')}")
        print(f"   chop_trend 存在: {hasattr(self.qsbc, 'chop_trend')}")
        print(f"   ultimate_volatility 存在: {hasattr(self.qsbc, 'ultimate_volatility')}")
        print(f"   efficiency_ratio 存在: {hasattr(self.qsbc, 'efficiency_ratio')}")
        
        # 計算実行
        print("🌊 量子強化価格分析エンジン実行中...")
        self.result = self.qsbc.calculate(self.data)
        
        # 結果検証
        print("✅ 計算完了 - 結果検証中...")
        print(f"📊 チャネルデータ: 上位 {len(self.result.upper_channel)}, 中央 {len(self.result.middle_line)}, 下位 {len(self.result.lower_channel)}")
        print(f"🎯 動的乗数範囲: {np.min(self.result.dynamic_multiplier):.2f} - {np.max(self.result.dynamic_multiplier):.2f}")
        print(f"🌀 現在のレジーム: {self.result.current_regime}")
        print(f"💪 現在のトレンド強度: {self.result.current_trend_strength:.3f}")
        print(f"🚀 現在のブレイクアウト確率: {self.result.current_breakout_probability:.1%}")
        print(f"🎛️ 現在の適応モード: {self.result.current_adaptation_mode}")
        
        # ミッドラインの値の範囲をチェック
        middle_min = np.nanmin(self.result.middle_line)
        middle_max = np.nanmax(self.result.middle_line)
        middle_mean = np.nanmean(self.result.middle_line)
        print(f"🎯 ミッドライン統計: 範囲 {middle_min:.2f} - {middle_max:.2f}, 平均 {middle_mean:.2f}")
        
        # 価格データとミッドラインの比較
        price_min = np.nanmin(self.data['close'])
        price_max = np.nanmax(self.data['close'])
        price_mean = np.nanmean(self.data['close'])
        print(f"📈 価格データ統計: 範囲 {price_min:.2f} - {price_max:.2f}, 平均 {price_mean:.2f}")
        
        # チャネル位置関係の検証
        upper_min = np.nanmin(self.result.upper_channel)
        upper_max = np.nanmax(self.result.upper_channel)
        lower_min = np.nanmin(self.result.lower_channel)
        lower_max = np.nanmax(self.result.lower_channel)
        print(f"🔺 上位チャネル統計: 範囲 {upper_min:.2f} - {upper_max:.2f}")
        print(f"🔻 下位チャネル統計: 範囲 {lower_min:.2f} - {lower_max:.2f}")
        
        # 位置関係の検証（サンプルポイント）
        sample_indices = [len(self.result.middle_line)//4, len(self.result.middle_line)//2, len(self.result.middle_line)*3//4, -1]
        print(f"🔍 位置関係検証（サンプルポイント）:")
        for i, idx in enumerate(sample_indices):
            if idx < len(self.result.middle_line):
                upper = self.result.upper_channel[idx]
                middle = self.result.middle_line[idx]
                lower = self.result.lower_channel[idx]
                print(f"   ポイント{i+1}: 上位={upper:.2f}, 中央={middle:.2f}, 下位={lower:.2f}")
                print(f"   　　　　　　　正順序? 上位>中央: {upper > middle}, 中央>下位: {middle > lower}")
        
        # NaN値チェック
        nan_upper = np.isnan(self.result.upper_channel).sum()
        nan_middle = np.isnan(self.result.middle_line).sum()
        nan_lower = np.isnan(self.result.lower_channel).sum()
        print(f"🔍 NaN値チェック - 上位: {nan_upper}, 中央: {nan_middle}, 下位: {nan_lower}")
        
        # 市場レジーム統計
        regime_counts = np.bincount(self.result.market_regime.astype(int))
        total_points = len(self.result.market_regime)
        print(f"📈 市場レジーム統計:")
        
        # 安全な配列アクセス
        range_count = regime_counts[0] if len(regime_counts) > 0 else 0
        trend_count = regime_counts[1] if len(regime_counts) > 1 else 0
        breakout_count = regime_counts[2] if len(regime_counts) > 2 else 0
        
        print(f"   レンジ(0): {range_count} ({range_count/total_points*100:.1f}%)")
        print(f"   トレンド(1): {trend_count} ({trend_count/total_points*100:.1f}%)")
        print(f"   ブレイクアウト(2): {breakout_count} ({breakout_count/total_points*100:.1f}%)")
        
        print("🌌 Quantum Supreme Breakout Channel 計算完了!")
            
    def plot(self, 
            title: str = "🌌 Quantum Supreme Breakout Channel V1.0", 
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            show_volume: bool = True,
            show_quantum_metrics: bool = True,
            show_regime_analysis: bool = True,
            show_breakout_analysis: bool = True,
            figsize: Tuple[int, int] = (16, 14),
            style: str = 'yahoo',
            savefig: Optional[str] = None) -> None:
        """
        🌌 ローソク足チャートとQuantum Supreme Breakout Channelを描画する
        
        Args:
            title: チャートのタイトル
            start_date: 表示開始日（フォーマット: YYYY-MM-DD）
            end_date: 表示終了日（フォーマット: YYYY-MM-DD）
            show_volume: 出来高を表示するか
            show_quantum_metrics: 量子メトリクスを表示するか
            show_regime_analysis: レジーム分析を表示するか
            show_breakout_analysis: ブレイクアウト分析を表示するか
            figsize: 図のサイズ
            style: mplfinanceのスタイル
            savefig: 保存先のパス（指定しない場合は表示のみ）
        """
        if self.data is None:
            raise ValueError("データが読み込まれていません。load_data_from_config()を先に実行してください。")
            
        if self.result is None:
            raise ValueError("インジケーターが計算されていません。calculate_indicators()を先に実行してください。")
        
        print("\n🎨 チャート描画開始...")
        
        # データの期間絞り込み
        df = self.data.copy()
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
            
        print(f"📊 描画期間: {df.index.min()} → {df.index.max()} ({len(df)}本)")
        
        # Quantum Supreme Breakout Channel の全データを時系列データフレームに変換
        full_df = pd.DataFrame(
            index=self.data.index,
            data={
                # メインチャネル
                'qsbc_upper': self.result.upper_channel,
                'qsbc_middle': self.result.middle_line,
                'qsbc_lower': self.result.lower_channel,
                
                # 市場状態
                'market_regime': self.result.market_regime,
                'trend_strength': self.result.trend_strength,
                'volatility_regime': self.result.volatility_regime,
                'efficiency_score': self.result.efficiency_score,
                
                # 量子メトリクス
                'quantum_coherence': self.result.quantum_coherence,
                'quantum_entanglement': self.result.quantum_entanglement,
                'superposition_state': self.result.superposition_state,
                
                # 動的適応
                'dynamic_multiplier': self.result.dynamic_multiplier,
                'channel_width_ratio': self.result.channel_width_ratio,
                'adaptation_confidence': self.result.adaptation_confidence,
                
                # 予測・分析
                'breakout_probability': self.result.breakout_probability,
                'trend_persistence': self.result.trend_persistence,
                'volatility_forecast': self.result.volatility_forecast,
                
                # シグナル
                'breakout_signals': self.result.breakout_signals,
                'trend_signals': self.result.trend_signals,
                'regime_change_signals': self.result.regime_change_signals
            }
        )
        
        # 絞り込み後のデータに結合
        df = df.join(full_df)
        
        print(f"🔍 チャートデータ準備完了 - NaN値: 上位 {df['qsbc_upper'].isna().sum()}, 中央 {df['qsbc_middle'].isna().sum()}, 下位 {df['qsbc_lower'].isna().sum()}")
        
        # シンプルなチャネル表示用データ準備
        # 3つの基本ライン
        df['upper_channel'] = df['qsbc_upper']      # 上位チャネル
        df['middle_channel'] = df['qsbc_middle']    # 中央チャネル
        df['lower_channel'] = df['qsbc_lower']      # 下位チャネル
        
        # ブレイクアウトシグナル表示用
        df['breakout_up'] = np.where(df['breakout_signals'] == 1, df['high'] * 1.01, np.nan)    # 上抜けシグナル
        df['breakout_down'] = np.where(df['breakout_signals'] == -1, df['low'] * 0.99, np.nan)  # 下抜けシグナル
        
        # mplfinanceプロット設定
        main_plots = []
        
        # 🌌 Quantum Supreme Breakout Channel メインプロット（シンプル版）
        # 3つの基本チャネルライン
        if not df['upper_channel'].isna().all():
            main_plots.append(mpf.make_addplot(df['upper_channel'], color='red', width=1.8, alpha=0.8, label='Upper Channel'))
        if not df['middle_channel'].isna().all():
            main_plots.append(mpf.make_addplot(df['middle_channel'], color='orange', width=2.0, alpha=0.9, label='Middle Channel'))
        if not df['lower_channel'].isna().all():
            main_plots.append(mpf.make_addplot(df['lower_channel'], color='green', width=1.8, alpha=0.8, label='Lower Channel'))
        
        # ブレイクアウトシグナル
        if not df['breakout_up'].isna().all():
            main_plots.append(mpf.make_addplot(df['breakout_up'], type='scatter', markersize=80, marker='^', color='red', alpha=0.9, label='Breakout Up'))
        if not df['breakout_down'].isna().all():
            main_plots.append(mpf.make_addplot(df['breakout_down'], type='scatter', markersize=80, marker='v', color='green', alpha=0.9, label='Breakout Down'))
        
        # サブプロット設定（パネル番号を正確に管理）
        current_panel = 1 if show_volume else 0
        
        # 🎛️ 動的乗数パネル（常に表示）
        multiplier_panel = mpf.make_addplot(df['dynamic_multiplier'], panel=current_panel, color='blue', width=2.0, 
                                          ylabel='Dynamic Multiplier', secondary_y=False, label='Multiplier')
        current_panel += 1
        
        # 📊 市場レジーム＆トレンド強度パネル（レジーム分析）
        regime_panels = []
        if show_regime_analysis:
            if not df['market_regime'].isna().all():
                regime_panels.append(mpf.make_addplot(df['market_regime'], panel=current_panel, color='orange', width=1.5, 
                                                    ylabel='Market Regime', secondary_y=False, label='Regime'))
            if not df['trend_strength'].isna().all():
                regime_panels.append(mpf.make_addplot(df['trend_strength'], panel=current_panel, color='purple', width=1.2, 
                                                    secondary_y=True, label='Trend Strength'))
            if regime_panels:  # パネルにプロットが追加された場合のみ
                current_panel += 1
        
        # 🚀 ブレイクアウト分析パネル
        breakout_panels = []
        if show_breakout_analysis:
            if not df['breakout_probability'].isna().all():
                breakout_panels.append(mpf.make_addplot(df['breakout_probability'], panel=current_panel, color='red', width=1.8, 
                                                       ylabel='Breakout Probability', secondary_y=False, label='BO Probability'))
            if not df['efficiency_score'].isna().all():
                breakout_panels.append(mpf.make_addplot(df['efficiency_score'], panel=current_panel, color='green', width=1.2, 
                                                       secondary_y=True, label='Efficiency Score'))
            if breakout_panels:  # パネルにプロットが追加された場合のみ
                current_panel += 1
        
        # 🌀 量子メトリクスパネル
        quantum_panels = []
        if show_quantum_metrics:
            if not df['quantum_coherence'].isna().all():
                quantum_panels.append(mpf.make_addplot(df['quantum_coherence'], panel=current_panel, color='cyan', width=1.5, 
                                                      ylabel='Quantum Metrics', secondary_y=False, label='Coherence'))
            if not df['quantum_entanglement'].isna().all():
                quantum_panels.append(mpf.make_addplot(df['quantum_entanglement'], panel=current_panel, color='magenta', width=1.2, 
                                                      secondary_y=False, label='Entanglement'))
            if quantum_panels:  # パネルにプロットが追加された場合のみ
                current_panel += 1
        
        # 実際のパネル数を計算（mplfinanceが認識するパネル数）
        actual_panels = current_panel  # 最後に使用したパネル番号 + 1 = 総パネル数
        
        # パネル比率の設定（実際のパネル数に合わせる）
        panel_ratios = [5]  # メインチャート（パネル0）
        if show_volume:
            panel_ratios.append(1)  # 出来高（パネル1）
        panel_ratios.append(1.2)  # 動的乗数（常に表示）
        if show_regime_analysis and regime_panels:
            panel_ratios.append(1.2)  # レジーム分析（プロットがある場合のみ）
        if show_breakout_analysis and breakout_panels:
            panel_ratios.append(1.2)  # ブレイクアウト分析（プロットがある場合のみ）
        if show_quantum_metrics and quantum_panels:
            panel_ratios.append(1.0)  # 量子メトリクス（プロットがある場合のみ）
        
        # デバッグ情報
        print(f"🔧 パネル設定: 実際のパネル数 {actual_panels}, 比率数 {len(panel_ratios)}, 比率 {panel_ratios}")
        print(f"🔧 プロット詳細: regime_panels={len(regime_panels)}, breakout_panels={len(breakout_panels)}, quantum_panels={len(quantum_panels)}")
        
        # 全プロットを結合
        all_plots = main_plots + [multiplier_panel] + regime_panels + breakout_panels + quantum_panels
        
        print(f"🔧 プロット数: main={len(main_plots)}, mult=1, regime={len(regime_panels)}, breakout={len(breakout_panels)}, quantum={len(quantum_panels)}, 総計={len(all_plots)}")
        
        # 各プロットの有効性を確認
        print(f"🔍 データ有効性チェック:")
        print(f"   dynamic_multiplier: {not df['dynamic_multiplier'].isna().all()}")
        if show_regime_analysis:
            print(f"   market_regime: {not df['market_regime'].isna().all()}")
            print(f"   trend_strength: {not df['trend_strength'].isna().all()}")
        if show_breakout_analysis:
            print(f"   breakout_probability: {not df['breakout_probability'].isna().all()}")
            print(f"   efficiency_score: {not df['efficiency_score'].isna().all()}")
        if show_quantum_metrics:
            print(f"   quantum_coherence: {not df['quantum_coherence'].isna().all()}")
            print(f"   quantum_entanglement: {not df['quantum_entanglement'].isna().all()}")
        
        # mplfinanceの設定
        kwargs = dict(
            type='candle',
            figsize=figsize,
            title=title,
            style=style,
            datetime_format='%Y-%m-%d',
            xrotation=45,
            volume=show_volume,
            addplot=all_plots,
            returnfig=True,
            warn_too_much_data=False  # 大量データ警告を無効化
        )
        
        # パネル比率が正しい場合のみ設定
        if len(panel_ratios) == actual_panels:
            kwargs['panel_ratios'] = tuple(panel_ratios)
            print(f"✅ パネル比率を設定: {panel_ratios}")
        else:
            print(f"⚠️ パネル比率をスキップ: 期待値 {actual_panels}, 実際 {len(panel_ratios)}")
        
        # プロット実行
        print("🎨 チャート描画実行中...")
        fig, axes = mpf.plot(df, **kwargs)
        
        # 凡例とスタイル調整
        axes[0].legend(['Upper Channel', 'Middle Channel', 'Lower Channel', 'Breakout Up', 'Breakout Down'], 
                      loc='upper left', fontsize=10)
        
        self.fig = fig
        self.axes = axes
        
        # 参照線の追加
        axis_idx = 1 if show_volume else 0
        
        # 動的乗数パネルの参照線
        axis_idx += 1
        axes[axis_idx].axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Min Multiplier')
        axes[axis_idx].axhline(y=6.0, color='red', linestyle='--', alpha=0.5, label='Max Multiplier')
        axes[axis_idx].axhline(y=3.5, color='black', linestyle='-', alpha=0.3, label='Neutral')
        
        # レジーム分析パネルの参照線
        if show_regime_analysis:
            axis_idx += 1
            axes[axis_idx].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            axes[axis_idx].axhline(y=1, color='blue', linestyle='--', alpha=0.5)
            axes[axis_idx].axhline(y=2, color='purple', linestyle='--', alpha=0.5)
        
        # ブレイクアウト分析パネルの参照線
        if show_breakout_analysis:
            axis_idx += 1
            axes[axis_idx].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
            axes[axis_idx].axhline(y=0.8, color='red', linestyle='--', alpha=0.3)
        
        # 量子メトリクスパネルの参照線
        if show_quantum_metrics:
            axis_idx += 1
            axes[axis_idx].axhline(y=0.5, color='black', linestyle='-', alpha=0.3)
            axes[axis_idx].axhline(y=0.75, color='cyan', linestyle='--', alpha=0.3)
        
        # 統計情報の表示
        print(f"\n🌌 === Quantum Supreme Breakout Channel 統計 ===")
        print(f"📊 総データ点数: {len(df)}")
        print(f"🎛️ 動的乗数統計:")
        print(f"   平均: {df['dynamic_multiplier'].mean():.2f}")
        print(f"   範囲: {df['dynamic_multiplier'].min():.2f} - {df['dynamic_multiplier'].max():.2f}")
        print(f"   標準偏差: {df['dynamic_multiplier'].std():.2f}")
        
        if show_regime_analysis:
            print(f"📈 市場レジーム分布:")
            regime_counts = df['market_regime'].value_counts().sort_index()
            total = len(df)
            regime_names = ['Range', 'Trend', 'Breakout']
            for regime, count in regime_counts.items():
                regime_idx = int(regime)
                if 0 <= regime_idx < len(regime_names):
                    regime_name = regime_names[regime_idx]
                    print(f"   {regime_name}: {count} ({count/total*100:.1f}%)")
                else:
                    print(f"   Unknown({regime_idx}): {count} ({count/total*100:.1f}%)")
        
        if show_breakout_analysis:
            breakout_count = (df['breakout_signals'] != 0).sum()
            up_breakouts = (df['breakout_signals'] == 1).sum()
            down_breakouts = (df['breakout_signals'] == -1).sum()
            print(f"🚀 ブレイクアウト統計:")
            print(f"   総ブレイクアウト: {breakout_count}")
            print(f"   上抜け: {up_breakouts}, 下抜け: {down_breakouts}")
            print(f"   平均ブレイクアウト確率: {df['breakout_probability'].mean():.1%}")
        
        if show_quantum_metrics:
            print(f"🌀 量子メトリクス統計:")
            print(f"   平均コヒーレンス: {df['quantum_coherence'].mean():.3f}")
            print(f"   平均もつれ: {df['quantum_entanglement'].mean():.3f}")
            print(f"   平均重ね合わせ: {df['superposition_state'].mean():.3f}")
        
        # 保存または表示
        if savefig:
            plt.savefig(savefig, dpi=150, bbox_inches='tight')
            print(f"💾 チャートを保存しました: {savefig}")
        else:
            plt.tight_layout()
            plt.show()
        
        print("✅ チャート描画完了!")


def main():
    """メイン関数"""
    import argparse
    parser = argparse.ArgumentParser(description='🌌 Quantum Supreme Breakout Channel チャート描画')
    
    # 基本設定
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--start', '-s', type=str, help='表示開始日 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='表示終了日 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, help='出力ファイルのパス')
    
    # インジケーターパラメータ
    parser.add_argument('--period', type=int, default=21, help='分析期間')
    parser.add_argument('--src-type', type=str, default='hlc3', help='価格ソースタイプ')
    parser.add_argument('--min-mult', type=float, default=1.0, help='最小乗数')
    parser.add_argument('--max-mult', type=float, default=6.0, help='最大乗数')
    parser.add_argument('--trend-sensitivity', type=float, default=0.85, help='トレンド感度')
    parser.add_argument('--range-sensitivity', type=float, default=0.75, help='レンジ感度')
    
    # 表示設定
    parser.add_argument('--no-volume', action='store_true', help='出来高を非表示')
    parser.add_argument('--no-quantum', action='store_true', help='量子メトリクスを非表示')
    parser.add_argument('--no-regime', action='store_true', help='レジーム分析を非表示')
    parser.add_argument('--no-breakout', action='store_true', help='ブレイクアウト分析を非表示')
    parser.add_argument('--figsize', nargs=2, type=int, default=[16, 14], help='図のサイズ')
    
    # アルゴリズム有効化
    parser.add_argument('--disable-quantum-hilbert', action='store_true', help='量子ヒルベルト変換を無効化')
    parser.add_argument('--disable-fractal', action='store_true', help='フラクタル分析を無効化')
    parser.add_argument('--disable-kalman', action='store_true', help='カルマンフィルターを無効化')
    parser.add_argument('--disable-garch', action='store_true', help='GARCHボラティリティを無効化')
    parser.add_argument('--disable-regime-switching', action='store_true', help='レジーム切り替えを無効化')
    parser.add_argument('--disable-entropy', action='store_true', help='エントロピー分析を無効化')
    
    args = parser.parse_args()
    
    print("🌌 Quantum Supreme Breakout Channel Chart V1.0")
    print("=" * 60)
    
    try:
        # チャートを作成
        chart = QuantumSupremeBreakoutChannelChart()
        
        # データ読み込み
        chart.load_data_from_config(args.config)
        
        # インジケーター計算
        chart.calculate_indicators(
            analysis_period=args.period,
            src_type=args.src_type,
            min_multiplier=args.min_mult,
            max_multiplier=args.max_mult,
            trend_sensitivity=args.trend_sensitivity,
            range_sensitivity=args.range_sensitivity,
            enable_quantum_hilbert=not args.disable_quantum_hilbert,
            enable_fractal_analysis=not args.disable_fractal,
            enable_kalman_quantum=not args.disable_kalman,
            enable_garch_volatility=not args.disable_garch,
            enable_regime_switching=not args.disable_regime_switching,
            enable_entropy_analysis=not args.disable_entropy
        )
        
        # チャート描画
        chart.plot(
            start_date=args.start,
            end_date=args.end,
            show_volume=not args.no_volume,
            show_quantum_metrics=not args.no_quantum,
            show_regime_analysis=not args.no_regime,
            show_breakout_analysis=not args.no_breakout,
            figsize=tuple(args.figsize),
            savefig=args.output
        )
        
    except Exception as e:
        import traceback
        print(f"❌ エラーが発生しました: {e}")
        print(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 