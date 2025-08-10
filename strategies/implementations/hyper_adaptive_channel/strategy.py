#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, List, Optional, Tuple
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import HyperAdaptiveChannelSignalGenerator


class HyperAdaptiveChannelStrategy(BaseStrategy):
    """
    HyperAdaptiveChannelストラテジー
    
    特徴:
    - HyperER（ハイパー効率比）に基づく動的パラメータ最適化
    - HyperAdaptiveChannelによる高精度なエントリーポイント検出
    - 複数のスムーザー（HyperFRAMA、UltimateMA、Laguerre、SuperSmoother、ZAdaptive）対応
    - X_ATRによる高精度なボラティリティ測定
    - 段階的最適化による効率的なパラメータ調整
    
    エントリー条件:
    - ロング: HyperAdaptiveChannelの買いシグナル
    - ショート: HyperAdaptiveChannelの売りシグナル
    
    エグジット条件:
    - ロング: HyperAdaptiveChannelの売りシグナル
    - ショート: HyperAdaptiveChannelの買いシグナル
    """
    
    def __init__(
        self,
        # 基本パラメータ
        band_lookback: int = 1,
        period: int = 14,
        midline_smoother: str = "laguerre_filter", # super_smoother ultimate_smoother laguerre_filter hyper_frama z_adaptive_ma ultimate_ma
        multiplier_mode: str = "dynamic",
        fixed_multiplier: float = 2.5,
        src_type: str = "hlc3",
        
        # === HyperFRAMA パラメータ ===
        # 基本パラメータ
        hyper_frama_period: int = 16,
        hyper_frama_src_type: str = 'oc2',
        hyper_frama_fc: int = 1,
        hyper_frama_sc: int = 198,
        hyper_frama_alpha_multiplier: float = 0.5,
        # 動的期間パラメータ
        hyper_frama_period_mode: str = 'fixed',
        hyper_frama_cycle_detector_type: str = 'hody_e',
        hyper_frama_lp_period: int = 10,
        hyper_frama_hp_period: int = 60,
        hyper_frama_cycle_part: float = 0.5,
        hyper_frama_max_cycle: int = 89,
        hyper_frama_min_cycle: int = 8,
        hyper_frama_max_output: int = 75,
        hyper_frama_min_output: int = 5,
        # 動的適応パラメータ
        hyper_frama_enable_indicator_adaptation: bool = True,
        hyper_frama_adaptation_indicator: str = 'hyper_er',
        hyper_frama_hyper_er_period: int = 14,
        hyper_frama_hyper_er_midline_period: int = 100,
        hyper_frama_hyper_adx_period: int = 14,
        hyper_frama_hyper_adx_midline_period: int = 100,
        hyper_frama_hyper_trend_index_period: int = 14,
        hyper_frama_hyper_trend_index_midline_period: int = 100,
        hyper_frama_fc_min: float = 1.0,
        hyper_frama_fc_max: float = 13.0,
        hyper_frama_sc_min: float = 50.0,
        hyper_frama_sc_max: float = 250.0,
        hyper_frama_period_min: int = 4,
        hyper_frama_period_max: int = 60,  # 偶数
        
        # === UltimateMA パラメータ ===
        ultimate_ma_ultimate_smoother_period: float = 20.0,
        ultimate_ma_zero_lag_period: int = 21,
        ultimate_ma_realtime_window: int = 89,
        ultimate_ma_src_type: str = 'oc2',
        ultimate_ma_slope_index: int = 1,
        ultimate_ma_range_threshold: float = 0.005,
        # 適応的カルマンフィルターパラメータ
        ultimate_ma_use_adaptive_kalman: bool = True,
        ultimate_ma_kalman_process_variance: float = 1e-5,
        ultimate_ma_kalman_measurement_variance: float = 0.01,
        ultimate_ma_kalman_volatility_window: int = 5,
        # 動的適応パラメータ
        ultimate_ma_zero_lag_period_mode: str = 'dynamic',
        ultimate_ma_realtime_window_mode: str = 'dynamic',
        # ゼロラグ用サイクル検出器パラメータ
        ultimate_ma_zl_cycle_detector_type: str = 'dft_dominant',
        ultimate_ma_zl_cycle_detector_cycle_part: float = 1.0,
        ultimate_ma_zl_cycle_detector_max_cycle: int = 120,
        ultimate_ma_zl_cycle_detector_min_cycle: int = 5,
        ultimate_ma_zl_cycle_period_multiplier: float = 1.0,
        # リアルタイムウィンドウ用サイクル検出器パラメータ
        ultimate_ma_rt_cycle_detector_type: str = 'dft_dominant',
        ultimate_ma_rt_cycle_detector_cycle_part: float = 0.5,
        ultimate_ma_rt_cycle_detector_max_cycle: int = 120,
        ultimate_ma_rt_cycle_detector_min_cycle: int = 5,
        ultimate_ma_rt_cycle_period_multiplier: float = 0.5,
        # period_rangeパラメータ
        ultimate_ma_zl_cycle_detector_period_range: Tuple[int, int] = (5, 120),
        ultimate_ma_rt_cycle_detector_period_range: Tuple[int, int] = (5, 120),
        
        # === LaguerreFilter パラメータ ===
        laguerre_gamma: float = 0.5,
        laguerre_order: int = 4,
        laguerre_coefficients: Optional[List[float]] = None,
        laguerre_src_type: str = 'oc2',
        laguerre_period: int = 4,
        laguerre_period_mode: str = 'fixed',
        laguerre_cycle_detector_type: str = 'practical',
        laguerre_cycle_part: float = 0.5,
        laguerre_max_cycle: int = 124,
        laguerre_min_cycle: int = 13,
        laguerre_max_output: int = 124,
        laguerre_min_output: int = 13,
        laguerre_lp_period: int = 13,
        laguerre_hp_period: int = 124,
        
        # === ZAdaptiveMA パラメータ ===
        z_adaptive_fast_period: int = 2,
        z_adaptive_slow_period: int = 120,
        z_adaptive_src_type: str = 'hlc3',
        z_adaptive_slope_index: int = 1,
        z_adaptive_range_threshold: float = 0.005,
        
        # === SuperSmoother パラメータ ===
        super_smoother_length: int = 15,
        super_smoother_num_poles: int = 2,
        super_smoother_src_type: str = 'oc2',
        # 動的期間パラメータ
        super_smoother_period_mode: str = 'dynamic',
        super_smoother_cycle_detector_type: str = 'practical',
        super_smoother_lp_period: int = 10,
        super_smoother_hp_period: int = 60,
        super_smoother_cycle_part: float = 0.5,
        super_smoother_max_cycle: int = 89,
        super_smoother_min_cycle: int = 8,
        super_smoother_max_output: int = 75,
        super_smoother_min_output: int = 5,
        
        # === X_ATR パラメータ ===
        x_atr_period: float = 16.0,
        x_atr_tr_method: str = 'atr',
        x_atr_smoother_type: str = 'frama',
        x_atr_src_type: str = 'close',
        x_atr_enable_kalman: bool = True,
        x_atr_kalman_type: str = 'unscented',
        # 動的適応パラメータ
        x_atr_period_mode: str = 'dynamic',
        x_atr_cycle_detector_type: str = 'practical',
        x_atr_cycle_detector_cycle_part: float = 0.5,
        x_atr_cycle_detector_max_cycle: int = 55,
        x_atr_cycle_detector_min_cycle: int = 5,
        x_atr_cycle_period_multiplier: float = 0.5,
        x_atr_cycle_detector_period_range: Tuple[int, int] = (5, 120),
        # ミッドラインパラメータ
        x_atr_midline_period: int = 100,
        # パーセンタイルベースボラティリティ分析パラメータ
        x_atr_enable_percentile_analysis: bool = True,
        x_atr_percentile_lookback_period: int = 50,
        x_atr_percentile_low_threshold: float = 0.25,
        x_atr_percentile_high_threshold: float = 0.75,
        # スムーサーパラメータ
        x_atr_smoother_params: Optional[Dict[str, Any]] = None,
        # カルマンフィルターパラメータ
        x_atr_kalman_params: Optional[Dict[str, Any]] = None,
        
        # === HyperER パラメータ ===
        hyper_er_period: int = 8,
        hyper_er_midline_period: int = 100,
        # ERパラメータ
        hyper_er_er_period: int = 13,
        hyper_er_er_src_type: str = 'oc2',
        # 統合カルマンフィルターパラメータ
        hyper_er_use_kalman_filter: bool = False,
        hyper_er_kalman_filter_type: str = 'simple',
        hyper_er_kalman_process_noise: float = 1e-5,
        hyper_er_kalman_min_observation_noise: float = 1e-6,
        hyper_er_kalman_adaptation_window: int = 5,
        # ルーフィングフィルターパラメータ
        hyper_er_use_roofing_filter: bool = True,
        hyper_er_roofing_hp_cutoff: float = 55.0,
        hyper_er_roofing_ss_band_edge: float = 10.0,
        # ラゲールフィルターパラメータ（後方互換性のため残す）
        hyper_er_use_laguerre_filter: bool = False,
        hyper_er_laguerre_gamma: float = 0.5,
        # 平滑化オプション
        hyper_er_use_smoothing: bool = True,
        hyper_er_smoother_type: str = 'frama',
        hyper_er_smoother_period: int = 12,
        hyper_er_smoother_src_type: str = 'close',
        # エラーズ統合サイクル検出器パラメータ
        hyper_er_use_dynamic_period: bool = True,
        hyper_er_detector_type: str = 'hody',
        hyper_er_lp_period: int = 13,
        hyper_er_hp_period: int = 124,
        hyper_er_cycle_part: float = 0.4,
        hyper_er_max_cycle: int = 124,
        hyper_er_min_cycle: int = 13,
        hyper_er_max_output: int = 89,
        hyper_er_min_output: int = 5,
        # パーセンタイルベーストレンド分析パラメータ
        hyper_er_enable_percentile_analysis: bool = True,
        hyper_er_percentile_lookback_period: int = 50,
        hyper_er_percentile_low_threshold: float = 0.25,
        hyper_er_percentile_high_threshold: float = 0.75,
        
        # === ハイパーアダプティブチャネル独自パラメータ ===
        enable_signals: bool = True,
        enable_percentile: bool = True,
        percentile_period: int = 100,
        
        # === フィルターパラメータ ===
        # フィルター選択
        filter_type: str = "none",  # none, hyper_er, hyper_trend_index, hyper_adx, consensus
        # HyperER フィルターパラメータ（追加）
        filter_hyper_er_period: int = 14,
        filter_hyper_er_midline_period: int = 100,
        # HyperTrendIndex フィルターパラメータ
        filter_hyper_trend_index_period: int = 14,
        filter_hyper_trend_index_midline_period: int = 100,
        # HyperADX フィルターパラメータ
        filter_hyper_adx_period: int = 14,
        filter_hyper_adx_midline_period: int = 100
    ):
        """
        初期化
        
        全150パラメータを受け取って、HyperAdaptiveChannelシステム全体を構築
        フィルター機能を含む完全版
        """
        super().__init__(f"HyperAdaptiveChannel_{filter_type}")
        
        # パラメータの設定（全ての引数をそのまま辞書にする）
        self._parameters = locals().copy()
        # 不要なキーを削除
        self._parameters.pop('self', None)
        self._parameters.pop('__class__', None)
        
        # シグナル生成器の初期化（全パラメータを渡す）
        self.signal_generator = HyperAdaptiveChannelSignalGenerator(**self._parameters)
    
    
    def generate_entry(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナルを生成する
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: エントリーシグナル
        """
        try:
            return self.signal_generator.get_entry_signals(data)
        except Exception as e:
            self.logger.error(f"エントリーシグナル生成中にエラー: {str(e)}")
            return np.zeros(len(data), dtype=np.int8)
    
    def generate_exit(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """
        エグジットシグナルを生成する
        
        Args:
            data: 価格データ
            position: 現在のポジション（1: ロング、-1: ショート）
            index: データのインデックス（デフォルト: -1）
            
        Returns:
            bool: エグジットすべきかどうか
        """
        try:
            return self.signal_generator.get_exit_signals(data, position, index)
        except Exception as e:
            self.logger.error(f"エグジットシグナル生成中にエラー: {str(e)}")
            return False
    
    @classmethod
    def create_optimization_params(cls, trial: optuna.Trial, optimization_level: str = "full") -> Dict[str, Any]:
        """
        効率的な最適化パラメータを生成
        
        Args:
            trial: Optunaのトライアル
            optimization_level: 最適化レベル
                - "basic": 基本パラメータのみ（10-15個）
                - "balanced": バランス型（25-35個）
                - "comprehensive": 包括的（50-70個）
                - "full": 全パラメータ（140個以上）
            
        Returns:
            Dict[str, Any]: 最適化パラメータ
        """
        
        # 基本パラメータ（すべてのレベルで含む）
        params = {
            # 主要パラメータ
            'band_lookback': trial.suggest_int('band_lookback', 1, 5),
            'period': trial.suggest_int('period', 8, 30),
            'src_type': trial.suggest_categorical('src_type', ['close', 'hlc3', 'hl2', 'ohlc4']),
            'midline_smoother': trial.suggest_categorical('midline_smoother', 
                                                        ['hyper_frama', 'ultimate_ma', 'laguerre_filter', 
                                                         'super_smoother', 'z_adaptive_ma']),
            'multiplier_mode': trial.suggest_categorical('multiplier_mode', ['fixed', 'dynamic']),
            'fixed_multiplier': trial.suggest_float('fixed_multiplier', 1.5, 4.0, step=0.1),
            
            # チャネル独自パラメータ
            'enable_signals': trial.suggest_categorical('enable_signals', [True, False]),
            'enable_percentile': trial.suggest_categorical('enable_percentile', [True, False]),
            'percentile_period': trial.suggest_int('percentile_period', 50, 200),
            
            # フィルターパラメータ
            'filter_type': trial.suggest_categorical('filter_type', ['none', 'hyper_er', 'hyper_trend_index', 'hyper_adx', 'consensus']),
            'filter_hyper_er_period': trial.suggest_int('filter_hyper_er_period', 8, 25),
            'filter_hyper_er_midline_period': trial.suggest_int('filter_hyper_er_midline_period', 50, 200),
            'filter_hyper_trend_index_period': trial.suggest_int('filter_hyper_trend_index_period', 8, 25),
            'filter_hyper_trend_index_midline_period': trial.suggest_int('filter_hyper_trend_index_midline_period', 50, 200),
            'filter_hyper_adx_period': trial.suggest_int('filter_hyper_adx_period', 8, 25),
            'filter_hyper_adx_midline_period': trial.suggest_int('filter_hyper_adx_midline_period', 50, 200),
        }
        
        if optimization_level in ["balanced", "comprehensive", "full"]:
            # 各スムーザーの主要パラメータ
            params.update({
                # HyperFRAMAの主要パラメータ（偶数のみ）
                'hyper_frama_period': trial.suggest_int('hyper_frama_period', 4, 25) * 2,
                'hyper_frama_src_type': trial.suggest_categorical('hyper_frama_src_type', ['close', 'hlc3', 'hl2', 'ohlc4']),
                'hyper_frama_fc': trial.suggest_int('hyper_frama_fc', 1, 8),
                'hyper_frama_sc': trial.suggest_int('hyper_frama_sc', 50, 300),
                'hyper_frama_alpha_multiplier': trial.suggest_float('hyper_frama_alpha_multiplier', 0.1, 1.0, step=0.1),
                
                # UltimateMAの主要パラメータ
                'ultimate_ma_ultimate_smoother_period': trial.suggest_int('ultimate_ma_ultimate_smoother_period', 2, 7) * 2.0,
                'ultimate_ma_zero_lag_period': trial.suggest_int('ultimate_ma_zero_lag_period', 8, 50),
                'ultimate_ma_realtime_window': trial.suggest_int('ultimate_ma_realtime_window', 30, 150),
                'ultimate_ma_src_type': trial.suggest_categorical('ultimate_ma_src_type', ['close', 'hlc3', 'hl2', 'ohlc4','oc2']),
                'ultimate_ma_use_adaptive_kalman': trial.suggest_categorical('ultimate_ma_use_adaptive_kalman', [True, False]),
                
                # LaguerreFilterパラメータ
                'laguerre_gamma': trial.suggest_float('laguerre_gamma', 0.1, 0.9, step=0.1),
                'laguerre_order': trial.suggest_int('laguerre_order', 2, 8),
                'laguerre_src_type': trial.suggest_categorical('laguerre_src_type', ['close', 'hlc3', 'hl2', 'ohlc4']),
                'laguerre_period': trial.suggest_int('laguerre_period', 2, 20),
                
                # SuperSmootherパラメータ
                'super_smoother_length': trial.suggest_int('super_smoother_length', 8, 30),
                'super_smoother_num_poles': trial.suggest_int('super_smoother_num_poles', 2, 3),
                'super_smoother_src_type': trial.suggest_categorical('super_smoother_src_type', ['close', 'hlc3', 'hl2', 'ohlc4', 'oc2']),
                
                # ZAdaptiveMAパラメータ
                'z_adaptive_fast_period': trial.suggest_int('z_adaptive_fast_period', 1, 8),
                'z_adaptive_slow_period': trial.suggest_int('z_adaptive_slow_period', 50, 200),
                'z_adaptive_src_type': trial.suggest_categorical('z_adaptive_src_type', ['close', 'hlc3', 'hl2', 'ohlc4']),
                
                # X_ATRパラメータ
                'x_atr_period': trial.suggest_int('x_atr_period', 4, 12) * 2.0,
                'x_atr_tr_method': trial.suggest_categorical('x_atr_tr_method', ['atr', 'str']),
                'x_atr_smoother_type': trial.suggest_categorical('x_atr_smoother_type', ['frama', 'super_smoother', 'ultimate_smoother','laguerre']),
                'x_atr_src_type': trial.suggest_categorical('x_atr_src_type', ['close', 'hlc3', 'hl2', 'ohlc4']),
                'x_atr_enable_kalman': trial.suggest_categorical('x_atr_enable_kalman', [True, False]),
                
                # HyperERの主要パラメータ
                'hyper_er_period': trial.suggest_int('hyper_er_period', 5, 25),
                'hyper_er_midline_period': trial.suggest_int('hyper_er_midline_period', 50, 200),
                'hyper_er_er_period': trial.suggest_int('hyper_er_er_period', 8, 30),
                'hyper_er_er_src_type': trial.suggest_categorical('hyper_er_er_src_type', ['close', 'hlc3', 'hl2', 'ohlc4', 'oc2']),
            })
        
        if optimization_level in ["comprehensive", "full"]:
            # より詳細なパラメータ
            params.update({
                # HyperFRAMA詳細パラメータ
                'hyper_frama_period_mode': trial.suggest_categorical('hyper_frama_period_mode', ['fixed', 'dynamic']),
                'hyper_frama_cycle_detector_type': trial.suggest_categorical('hyper_frama_cycle_detector_type', 
                    ['hody_e', 'dft_dominant', 'absolute_ultimate', 'phac']),
                'hyper_frama_lp_period': trial.suggest_int('hyper_frama_lp_period', 8, 25),
                'hyper_frama_hp_period': trial.suggest_int('hyper_frama_hp_period', 80, 200),
                'hyper_frama_cycle_part': trial.suggest_float('hyper_frama_cycle_part', 0.2, 0.8, step=0.1),
                'hyper_frama_max_cycle': trial.suggest_int('hyper_frama_max_cycle', 30, 75) * 2,
                'hyper_frama_min_cycle': trial.suggest_int('hyper_frama_min_cycle', 1, 7) * 2,
                'hyper_frama_max_output': trial.suggest_int('hyper_frama_max_output', 40, 100) * 2,
                'hyper_frama_min_output': trial.suggest_int('hyper_frama_min_output', 1, 7) * 2,
                
                # UltimateMA詳細パラメータ
                'ultimate_ma_slope_index': trial.suggest_int('ultimate_ma_slope_index', 1, 3),
                'ultimate_ma_range_threshold': trial.suggest_float('ultimate_ma_range_threshold', 0.001, 0.01, step=0.001),
                'ultimate_ma_kalman_process_variance': trial.suggest_float('ultimate_ma_kalman_process_variance', 1e-6, 1e-4, log=True),
                'ultimate_ma_kalman_measurement_variance': trial.suggest_float('ultimate_ma_kalman_measurement_variance', 0.001, 0.1, log=True),
                'ultimate_ma_kalman_volatility_window': trial.suggest_int('ultimate_ma_kalman_volatility_window', 3, 10),
                
                # X_ATR詳細パラメータ
                'x_atr_period_mode': trial.suggest_categorical('x_atr_period_mode', ['fixed', 'dynamic']),
                'x_atr_cycle_detector_type': trial.suggest_categorical('x_atr_cycle_detector_type', 
                    ['absolute_ultimate', 'hody_e', 'dft_dominant']),
                'x_atr_cycle_detector_cycle_part': trial.suggest_float('x_atr_cycle_detector_cycle_part', 0.2, 0.8, step=0.1),
                'x_atr_cycle_detector_max_cycle': trial.suggest_int('x_atr_cycle_detector_max_cycle', 30, 80),
                'x_atr_cycle_detector_min_cycle': trial.suggest_int('x_atr_cycle_detector_min_cycle', 3, 10),
                'x_atr_cycle_period_multiplier': trial.suggest_float('x_atr_cycle_period_multiplier', 0.5, 2.0, step=0.1),
                'x_atr_midline_period': trial.suggest_int('x_atr_midline_period', 50, 200),
                'x_atr_enable_percentile_analysis': trial.suggest_categorical('x_atr_enable_percentile_analysis', [True, False]),
                'x_atr_percentile_lookback_period': trial.suggest_int('x_atr_percentile_lookback_period', 20, 100),
                'x_atr_percentile_low_threshold': trial.suggest_float('x_atr_percentile_low_threshold', 0.1, 0.4, step=0.05),
                'x_atr_percentile_high_threshold': trial.suggest_float('x_atr_percentile_high_threshold', 0.6, 0.9, step=0.05),
                
                # HyperER詳細パラメータ
                'hyper_er_use_kalman_filter': trial.suggest_categorical('hyper_er_use_kalman_filter', [True, False]),
                'hyper_er_kalman_filter_type': trial.suggest_categorical('hyper_er_kalman_filter_type', ['unscented', 'extended', 'standard']),
                'hyper_er_kalman_process_noise': trial.suggest_float('hyper_er_kalman_process_noise', 1e-6, 1e-4, log=True),
                'hyper_er_kalman_min_observation_noise': trial.suggest_float('hyper_er_kalman_min_observation_noise', 1e-7, 1e-5, log=True),
                'hyper_er_kalman_adaptation_window': trial.suggest_int('hyper_er_kalman_adaptation_window', 3, 10),
                'hyper_er_use_roofing_filter': trial.suggest_categorical('hyper_er_use_roofing_filter', [True, False]),
                'hyper_er_roofing_hp_cutoff': trial.suggest_float('hyper_er_roofing_hp_cutoff', 30.0, 80.0, step=5.0),
                'hyper_er_roofing_ss_band_edge': trial.suggest_float('hyper_er_roofing_ss_band_edge', 5.0, 20.0, step=1.0),
                'hyper_er_use_smoothing': trial.suggest_categorical('hyper_er_use_smoothing', [True, False]),
                'hyper_er_smoother_type': trial.suggest_categorical('hyper_er_smoother_type', ['laguerre', 'frama', 'super_smoother']),
                'hyper_er_smoother_period': trial.suggest_int('hyper_er_smoother_period', 5, 25),
                'hyper_er_smoother_src_type': trial.suggest_categorical('hyper_er_smoother_src_type', ['close', 'hlc3', 'hl2', 'ohlc4']),
            })
        
        if optimization_level == "full":
            # 全パラメータ（フラグ系・高度なパラメータも含む）
            params.update({
                # HyperFRAMA高度なパラメータ
                'hyper_frama_enable_indicator_adaptation': trial.suggest_categorical('hyper_frama_enable_indicator_adaptation', [True, False]),
                'hyper_frama_adaptation_indicator': trial.suggest_categorical('hyper_frama_adaptation_indicator', 
                    ['hyper_er', 'hyper_adx', 'hyper_trend_index']),
                'hyper_frama_hyper_er_period': trial.suggest_int('hyper_frama_hyper_er_period', 8, 25),
                'hyper_frama_hyper_er_midline_period': trial.suggest_int('hyper_frama_hyper_er_midline_period', 50, 200),
                'hyper_frama_hyper_adx_period': trial.suggest_int('hyper_frama_hyper_adx_period', 8, 25),
                'hyper_frama_hyper_adx_midline_period': trial.suggest_int('hyper_frama_hyper_adx_midline_period', 50, 200),
                'hyper_frama_hyper_trend_index_period': trial.suggest_int('hyper_frama_hyper_trend_index_period', 8, 25),
                'hyper_frama_hyper_trend_index_midline_period': trial.suggest_int('hyper_frama_hyper_trend_index_midline_period', 50, 200),
                'hyper_frama_fc_min': trial.suggest_float('hyper_frama_fc_min', 0.5, 2.0, step=0.1),
                'hyper_frama_fc_max': trial.suggest_float('hyper_frama_fc_max', 4.0, 15.0, step=0.5),
                'hyper_frama_sc_min': trial.suggest_float('hyper_frama_sc_min', 30.0, 80.0, step=5.0),
                'hyper_frama_sc_max': trial.suggest_float('hyper_frama_sc_max', 150.0, 300.0, step=10.0),
                'hyper_frama_period_min': trial.suggest_int('hyper_frama_period_min', 1, 4) * 2,
                'hyper_frama_period_max': trial.suggest_int('hyper_frama_period_max', 30, 60) * 2,
                
                # UltimateMA高度なパラメータ
                'ultimate_ma_zero_lag_period_mode': trial.suggest_categorical('ultimate_ma_zero_lag_period_mode', ['fixed', 'dynamic']),
                'ultimate_ma_realtime_window_mode': trial.suggest_categorical('ultimate_ma_realtime_window_mode', ['fixed', 'dynamic']),
                'ultimate_ma_zl_cycle_detector_type': trial.suggest_categorical('ultimate_ma_zl_cycle_detector_type', 
                    ['absolute_ultimate', 'hody_e', 'dft_dominant', 'phac']),
                'ultimate_ma_zl_cycle_detector_cycle_part': trial.suggest_float('ultimate_ma_zl_cycle_detector_cycle_part', 0.5, 1.5, step=0.1),
                'ultimate_ma_zl_cycle_detector_max_cycle': trial.suggest_int('ultimate_ma_zl_cycle_detector_max_cycle', 80, 200),
                'ultimate_ma_zl_cycle_detector_min_cycle': trial.suggest_int('ultimate_ma_zl_cycle_detector_min_cycle', 3, 10),
                'ultimate_ma_zl_cycle_period_multiplier': trial.suggest_float('ultimate_ma_zl_cycle_period_multiplier', 0.5, 2.0, step=0.1),
                'ultimate_ma_rt_cycle_detector_type': trial.suggest_categorical('ultimate_ma_rt_cycle_detector_type', 
                    ['absolute_ultimate', 'hody_e', 'dft_dominant', 'phac']),
                'ultimate_ma_rt_cycle_detector_cycle_part': trial.suggest_float('ultimate_ma_rt_cycle_detector_cycle_part', 0.2, 0.8, step=0.1),
                'ultimate_ma_rt_cycle_detector_max_cycle': trial.suggest_int('ultimate_ma_rt_cycle_detector_max_cycle', 80, 200),
                'ultimate_ma_rt_cycle_detector_min_cycle': trial.suggest_int('ultimate_ma_rt_cycle_detector_min_cycle', 3, 10),
                'ultimate_ma_rt_cycle_period_multiplier': trial.suggest_float('ultimate_ma_rt_cycle_period_multiplier', 0.2, 1.0, step=0.1),
                
                # Laguerre高度なパラメータ
                'laguerre_period_mode': trial.suggest_categorical('laguerre_period_mode', ['fixed', 'dynamic']),
                'laguerre_cycle_detector_type': trial.suggest_categorical('laguerre_cycle_detector_type', 
                    ['hody_e', 'dft_dominant', 'absolute_ultimate']),
                'laguerre_cycle_part': trial.suggest_float('laguerre_cycle_part', 0.2, 0.8, step=0.1),
                'laguerre_max_cycle': trial.suggest_int('laguerre_max_cycle', 80, 200),
                'laguerre_min_cycle': trial.suggest_int('laguerre_min_cycle', 8, 20),
                'laguerre_max_output': trial.suggest_int('laguerre_max_output', 80, 200),
                'laguerre_min_output': trial.suggest_int('laguerre_min_output', 8, 20),
                'laguerre_lp_period': trial.suggest_int('laguerre_lp_period', 8, 25),
                'laguerre_hp_period': trial.suggest_int('laguerre_hp_period', 80, 200),
                
                # ZAdaptiveMA高度なパラメータ
                'z_adaptive_slope_index': trial.suggest_int('z_adaptive_slope_index', 1, 3),
                'z_adaptive_range_threshold': trial.suggest_float('z_adaptive_range_threshold', 0.001, 0.01, step=0.001),
                
                # SuperSmoother高度なパラメータ
                'super_smoother_period_mode': trial.suggest_categorical('super_smoother_period_mode', ['fixed', 'dynamic']),
                'super_smoother_cycle_detector_type': trial.suggest_categorical('super_smoother_cycle_detector_type', 
                    ['hody_e', 'dft_dominant', 'absolute_ultimate']),
                'super_smoother_lp_period': trial.suggest_int('super_smoother_lp_period', 8, 25),
                'super_smoother_hp_period': trial.suggest_int('super_smoother_hp_period', 80, 200),
                'super_smoother_cycle_part': trial.suggest_float('super_smoother_cycle_part', 0.2, 0.8, step=0.1),
                'super_smoother_max_cycle': trial.suggest_int('super_smoother_max_cycle', 80, 200),
                'super_smoother_min_cycle': trial.suggest_int('super_smoother_min_cycle', 8, 20),
                'super_smoother_max_output': trial.suggest_int('super_smoother_max_output', 80, 200),
                'super_smoother_min_output': trial.suggest_int('super_smoother_min_output', 8, 20),
                
                # X_ATR高度なパラメータ  
                'x_atr_kalman_type': trial.suggest_categorical('x_atr_kalman_type', ['unscented', 'extended', 'standard']),
                
                # HyperER高度なパラメータ
                'hyper_er_use_laguerre_filter': trial.suggest_categorical('hyper_er_use_laguerre_filter', [True, False]),
                'hyper_er_laguerre_gamma': trial.suggest_float('hyper_er_laguerre_gamma', 0.1, 0.9, step=0.1),
                'hyper_er_use_dynamic_period': trial.suggest_categorical('hyper_er_use_dynamic_period', [True, False]),
                'hyper_er_detector_type': trial.suggest_categorical('hyper_er_detector_type', 
                    ['dft_dominant', 'hody_e', 'absolute_ultimate', 'phac']),
                'hyper_er_lp_period': trial.suggest_int('hyper_er_lp_period', 8, 25),
                'hyper_er_hp_period': trial.suggest_int('hyper_er_hp_period', 80, 200),
                'hyper_er_cycle_part': trial.suggest_float('hyper_er_cycle_part', 0.2, 0.6, step=0.1),
                'hyper_er_max_cycle': trial.suggest_int('hyper_er_max_cycle', 80, 200),
                'hyper_er_min_cycle': trial.suggest_int('hyper_er_min_cycle', 8, 20),
                'hyper_er_max_output': trial.suggest_int('hyper_er_max_output', 60, 150),
                'hyper_er_min_output': trial.suggest_int('hyper_er_min_output', 3, 10),
                'hyper_er_enable_percentile_analysis': trial.suggest_categorical('hyper_er_enable_percentile_analysis', [True, False]),
                'hyper_er_percentile_lookback_period': trial.suggest_int('hyper_er_percentile_lookback_period', 20, 100),
                'hyper_er_percentile_low_threshold': trial.suggest_float('hyper_er_percentile_low_threshold', 0.1, 0.4, step=0.05),
                'hyper_er_percentile_high_threshold': trial.suggest_float('hyper_er_percentile_high_threshold', 0.6, 0.9, step=0.05),
            })
        
        return params
    
    @classmethod
    def create_conditional_optimization_params(cls, trial: optuna.Trial, midline_smoother: str = None, optimization_level: str = "balanced") -> Dict[str, Any]:
        """
        ミッドラインスムーザー別の条件付き最適化パラメータ生成
        
        Args:
            trial: Optunaのトライアル
            midline_smoother: 指定されたミッドラインスムーザー（Noneなら自動選択）
            optimization_level: 最適化レベル
            
        Returns:
            Dict[str, Any]: 条件付き最適化パラメータ
        """
        
        # 基本パラメータ
        params = {
            'band_lookback': trial.suggest_int('band_lookback', 1, 5),
            'period': trial.suggest_int('period', 8, 30),
            'src_type': trial.suggest_categorical('src_type', ['close', 'hlc3', 'hl2', 'ohlc4']),
            'multiplier_mode': trial.suggest_categorical('multiplier_mode', ['fixed', 'dynamic']),
            'fixed_multiplier': trial.suggest_float('fixed_multiplier', 1.5, 4.0, step=0.1),
            
            # チャネル独自パラメータ
            'enable_signals': trial.suggest_categorical('enable_signals', [True, False]),
            'enable_percentile': trial.suggest_categorical('enable_percentile', [True, False]),
            'percentile_period': trial.suggest_int('percentile_period', 50, 200),
        }
        
        # ミッドラインスムーザーの選択
        if midline_smoother is None:
            midline_smoother = trial.suggest_categorical('midline_smoother', 
                ['hyper_frama', 'ultimate_ma', 'laguerre_filter', 'super_smoother', 'z_adaptive_ma'])
        params['midline_smoother'] = midline_smoother
        
        # 共通パラメータ（全スムーザーで使用）
        params.update({
            # X_ATRパラメータ（全スムーザーで共通）
            'x_atr_period': trial.suggest_int('x_atr_period', 4, 12) * 2.0,
            'x_atr_tr_method': trial.suggest_categorical('x_atr_tr_method', ['atr', 'str']),
            'x_atr_smoother_type': trial.suggest_categorical('x_atr_smoother_type', ['frama', 'super_smoother', 'ultimate_smoother']),
            'x_atr_src_type': trial.suggest_categorical('x_atr_src_type', ['close', 'hlc3', 'hl2', 'ohlc4']),
            'x_atr_enable_kalman': trial.suggest_categorical('x_atr_enable_kalman', [True, False]),
            
            # HyperERパラメータ（全スムーザーで共通）
            'hyper_er_period': trial.suggest_int('hyper_er_period', 5, 25),
            'hyper_er_midline_period': trial.suggest_int('hyper_er_midline_period', 50, 200),
            'hyper_er_er_period': trial.suggest_int('hyper_er_er_period', 8, 30),
            'hyper_er_er_src_type': trial.suggest_categorical('hyper_er_er_src_type', ['close', 'hlc3', 'hl2', 'ohlc4', 'oc2']),
        })
        
        # スムーザー別パラメータの最適化
        if midline_smoother == 'hyper_frama':
            params.update(cls._get_hyper_frama_params(trial, optimization_level))
        elif midline_smoother == 'ultimate_ma':
            params.update(cls._get_ultimate_ma_params(trial, optimization_level))
        elif midline_smoother == 'laguerre_filter':
            params.update(cls._get_laguerre_params(trial, optimization_level))
        elif midline_smoother == 'super_smoother':
            params.update(cls._get_super_smoother_params(trial, optimization_level))
        elif midline_smoother == 'z_adaptive_ma':
            params.update(cls._get_z_adaptive_params(trial, optimization_level))
        
        # 詳細レベルでの追加パラメータ
        if optimization_level in ["comprehensive", "full"]:
            params.update(cls._get_detailed_common_params(trial, optimization_level))
        
        return params
    
    @classmethod
    def _get_hyper_frama_params(cls, trial: optuna.Trial, optimization_level: str) -> Dict[str, Any]:
        """HyperFRAMAパラメータを取得"""
        params = {
            'hyper_frama_period': trial.suggest_int('hyper_frama_period', 4, 25) * 2,
            'hyper_frama_src_type': trial.suggest_categorical('hyper_frama_src_type', ['close', 'hlc3', 'hl2', 'ohlc4']),
            'hyper_frama_fc': trial.suggest_int('hyper_frama_fc', 1, 8),
            'hyper_frama_sc': trial.suggest_int('hyper_frama_sc', 50, 300),
            'hyper_frama_alpha_multiplier': trial.suggest_float('hyper_frama_alpha_multiplier', 0.1, 1.0, step=0.1),
        }
        
        if optimization_level in ["comprehensive", "full"]:
            params.update({
                'hyper_frama_period_mode': trial.suggest_categorical('hyper_frama_period_mode', ['fixed', 'dynamic']),
                'hyper_frama_cycle_detector_type': trial.suggest_categorical('hyper_frama_cycle_detector_type', 
                    ['hody_e', 'dft_dominant', 'absolute_ultimate', 'phac']),
                'hyper_frama_lp_period': trial.suggest_int('hyper_frama_lp_period', 8, 25),
                'hyper_frama_hp_period': trial.suggest_int('hyper_frama_hp_period', 80, 200),
                'hyper_frama_cycle_part': trial.suggest_float('hyper_frama_cycle_part', 0.2, 0.8, step=0.1),
                'hyper_frama_max_cycle': trial.suggest_int('hyper_frama_max_cycle', 30, 75) * 2,
                'hyper_frama_min_cycle': trial.suggest_int('hyper_frama_min_cycle', 1, 7) * 2,
            })
        
        if optimization_level == "full":
            params.update({
                'hyper_frama_enable_indicator_adaptation': trial.suggest_categorical('hyper_frama_enable_indicator_adaptation', [True, False]),
                'hyper_frama_adaptation_indicator': trial.suggest_categorical('hyper_frama_adaptation_indicator', 
                    ['hyper_er', 'hyper_adx', 'hyper_trend_index']),
                'hyper_frama_fc_min': trial.suggest_float('hyper_frama_fc_min', 0.5, 2.0, step=0.1),
                'hyper_frama_fc_max': trial.suggest_float('hyper_frama_fc_max', 4.0, 15.0, step=0.5),
                'hyper_frama_sc_min': trial.suggest_float('hyper_frama_sc_min', 30.0, 80.0, step=5.0),
                'hyper_frama_sc_max': trial.suggest_float('hyper_frama_sc_max', 150.0, 300.0, step=10.0),
                'hyper_frama_period_min': trial.suggest_int('hyper_frama_period_min', 1, 4) * 2,
                'hyper_frama_period_max': trial.suggest_int('hyper_frama_period_max', 30, 60) * 2,
            })
        
        return params
    
    @classmethod
    def _get_ultimate_ma_params(cls, trial: optuna.Trial, optimization_level: str) -> Dict[str, Any]:
        """UltimateMAパラメータを取得"""
        params = {
            'ultimate_ma_ultimate_smoother_period': trial.suggest_int('ultimate_ma_ultimate_smoother_period', 2, 7) * 2.0,
            'ultimate_ma_zero_lag_period': trial.suggest_int('ultimate_ma_zero_lag_period', 8, 50),
            'ultimate_ma_realtime_window': trial.suggest_int('ultimate_ma_realtime_window', 30, 150),
            'ultimate_ma_src_type': trial.suggest_categorical('ultimate_ma_src_type', ['close', 'hlc3', 'hl2', 'ohlc4']),
            'ultimate_ma_use_adaptive_kalman': trial.suggest_categorical('ultimate_ma_use_adaptive_kalman', [True, False]),
        }
        
        if optimization_level in ["comprehensive", "full"]:
            params.update({
                'ultimate_ma_slope_index': trial.suggest_int('ultimate_ma_slope_index', 1, 3),
                'ultimate_ma_range_threshold': trial.suggest_float('ultimate_ma_range_threshold', 0.001, 0.01, step=0.001),
                'ultimate_ma_kalman_process_variance': trial.suggest_float('ultimate_ma_kalman_process_variance', 1e-6, 1e-4, log=True),
                'ultimate_ma_kalman_measurement_variance': trial.suggest_float('ultimate_ma_kalman_measurement_variance', 0.001, 0.1, log=True),
                'ultimate_ma_kalman_volatility_window': trial.suggest_int('ultimate_ma_kalman_volatility_window', 3, 10),
            })
        
        if optimization_level == "full":
            params.update({
                'ultimate_ma_zero_lag_period_mode': trial.suggest_categorical('ultimate_ma_zero_lag_period_mode', ['fixed', 'dynamic']),
                'ultimate_ma_realtime_window_mode': trial.suggest_categorical('ultimate_ma_realtime_window_mode', ['fixed', 'dynamic']),
                'ultimate_ma_zl_cycle_detector_type': trial.suggest_categorical('ultimate_ma_zl_cycle_detector_type', 
                    ['absolute_ultimate', 'hody_e', 'dft_dominant', 'phac']),
                'ultimate_ma_zl_cycle_detector_cycle_part': trial.suggest_float('ultimate_ma_zl_cycle_detector_cycle_part', 0.5, 1.5, step=0.1),
                'ultimate_ma_rt_cycle_detector_type': trial.suggest_categorical('ultimate_ma_rt_cycle_detector_type', 
                    ['absolute_ultimate', 'hody_e', 'dft_dominant', 'phac']),
            })
        
        return params
    
    @classmethod
    def _get_laguerre_params(cls, trial: optuna.Trial, optimization_level: str) -> Dict[str, Any]:
        """Laguerreパラメータを取得"""
        params = {
            'laguerre_gamma': trial.suggest_float('laguerre_gamma', 0.1, 0.9, step=0.1),
            'laguerre_order': trial.suggest_int('laguerre_order', 2, 8),
            'laguerre_src_type': trial.suggest_categorical('laguerre_src_type', ['close', 'hlc3', 'hl2', 'ohlc4']),
            'laguerre_period': trial.suggest_int('laguerre_period', 2, 20),
        }
        
        if optimization_level == "full":
            params.update({
                'laguerre_period_mode': trial.suggest_categorical('laguerre_period_mode', ['fixed', 'dynamic']),
                'laguerre_cycle_detector_type': trial.suggest_categorical('laguerre_cycle_detector_type', 
                    ['hody_e', 'dft_dominant', 'absolute_ultimate']),
                'laguerre_cycle_part': trial.suggest_float('laguerre_cycle_part', 0.2, 0.8, step=0.1),
            })
        
        return params
    
    @classmethod
    def _get_super_smoother_params(cls, trial: optuna.Trial, optimization_level: str) -> Dict[str, Any]:
        """SuperSmootherパラメータを取得"""
        params = {
            'super_smoother_length': trial.suggest_int('super_smoother_length', 8, 30),
            'super_smoother_num_poles': trial.suggest_int('super_smoother_num_poles', 2, 3),
            'super_smoother_src_type': trial.suggest_categorical('super_smoother_src_type', ['close', 'hlc3', 'hl2', 'ohlc4', 'oc2']),
        }
        
        if optimization_level == "full":
            params.update({
                'super_smoother_period_mode': trial.suggest_categorical('super_smoother_period_mode', ['fixed', 'dynamic']),
                'super_smoother_cycle_detector_type': trial.suggest_categorical('super_smoother_cycle_detector_type', 
                    ['hody_e', 'dft_dominant', 'absolute_ultimate']),
                'super_smoother_cycle_part': trial.suggest_float('super_smoother_cycle_part', 0.2, 0.8, step=0.1),
            })
        
        return params
    
    @classmethod
    def _get_z_adaptive_params(cls, trial: optuna.Trial, optimization_level: str) -> Dict[str, Any]:
        """ZAdaptiveMAパラメータを取得"""
        params = {
            'z_adaptive_fast_period': trial.suggest_int('z_adaptive_fast_period', 1, 8),
            'z_adaptive_slow_period': trial.suggest_int('z_adaptive_slow_period', 50, 200),
            'z_adaptive_src_type': trial.suggest_categorical('z_adaptive_src_type', ['close', 'hlc3', 'hl2', 'ohlc4']),
        }
        
        if optimization_level == "full":
            params.update({
                'z_adaptive_slope_index': trial.suggest_int('z_adaptive_slope_index', 1, 3),
                'z_adaptive_range_threshold': trial.suggest_float('z_adaptive_range_threshold', 0.001, 0.01, step=0.001),
            })
        
        return params
    
    @classmethod
    def _get_detailed_common_params(cls, trial: optuna.Trial, optimization_level: str) -> Dict[str, Any]:
        """詳細な共通パラメータを取得"""
        params = {
            # X_ATR詳細パラメータ
            'x_atr_period_mode': trial.suggest_categorical('x_atr_period_mode', ['fixed', 'dynamic']),
            'x_atr_cycle_detector_type': trial.suggest_categorical('x_atr_cycle_detector_type', 
                ['absolute_ultimate', 'hody_e', 'dft_dominant']),
            'x_atr_cycle_detector_cycle_part': trial.suggest_float('x_atr_cycle_detector_cycle_part', 0.2, 0.8, step=0.1),
            'x_atr_midline_period': trial.suggest_int('x_atr_midline_period', 50, 200),
            'x_atr_enable_percentile_analysis': trial.suggest_categorical('x_atr_enable_percentile_analysis', [True, False]),
            
            # HyperER詳細パラメータ
            'hyper_er_use_kalman_filter': trial.suggest_categorical('hyper_er_use_kalman_filter', [True, False]),
            'hyper_er_kalman_filter_type': trial.suggest_categorical('hyper_er_kalman_filter_type', ['unscented', 'extended', 'standard']),
            'hyper_er_use_roofing_filter': trial.suggest_categorical('hyper_er_use_roofing_filter', [True, False]),
            'hyper_er_use_smoothing': trial.suggest_categorical('hyper_er_use_smoothing', [True, False]),
            'hyper_er_smoother_type': trial.suggest_categorical('hyper_er_smoother_type', ['laguerre', 'frama', 'super_smoother']),
            'hyper_er_smoother_period': trial.suggest_int('hyper_er_smoother_period', 5, 25),
        }
        
        if optimization_level == "full":
            params.update({
                # X_ATR高度なパラメータ
                'x_atr_percentile_lookback_period': trial.suggest_int('x_atr_percentile_lookback_period', 20, 100),
                'x_atr_percentile_low_threshold': trial.suggest_float('x_atr_percentile_low_threshold', 0.1, 0.4, step=0.05),
                'x_atr_percentile_high_threshold': trial.suggest_float('x_atr_percentile_high_threshold', 0.6, 0.9, step=0.05),
                'x_atr_kalman_type': trial.suggest_categorical('x_atr_kalman_type', ['unscented', 'extended', 'standard']),
                
                # HyperER高度なパラメータ
                'hyper_er_use_dynamic_period': trial.suggest_categorical('hyper_er_use_dynamic_period', [True, False]),
                'hyper_er_detector_type': trial.suggest_categorical('hyper_er_detector_type', 
                    ['dft_dominant', 'hody_e', 'absolute_ultimate', 'phac']),
                'hyper_er_enable_percentile_analysis': trial.suggest_categorical('hyper_er_enable_percentile_analysis', [True, False]),
                'hyper_er_percentile_lookback_period': trial.suggest_int('hyper_er_percentile_lookback_period', 20, 100),
            })
        
        return params

    @classmethod
    def create_staged_optimization_params(cls, trial: optuna.Trial, stage: int = 1) -> Dict[str, Any]:
        """
        段階的最適化パラメータ生成
        
        Args:
            trial: Optunaのトライアル
            stage: 最適化段階
                1: 基本パラメータ（10-15個）
                2: スムーザー最適化（20-30個）
                3: ATR・ER最適化（30-40個）
                4: 詳細調整（50-70個）
            
        Returns:
            Dict[str, Any]: 段階別最適化パラメータ
        """
        
        if stage == 1:
            # 第1段階: 基本構造の最適化
            return {
                'period': trial.suggest_int('period', 8, 30),
                'midline_smoother': trial.suggest_categorical('midline_smoother', 
                                                            ['hyper_frama', 'ultimate_ma', 'laguerre_filter', 
                                                             'super_smoother', 'z_adaptive_ma']),
                'multiplier_mode': trial.suggest_categorical('multiplier_mode', ['fixed', 'dynamic']),
                'fixed_multiplier': trial.suggest_float('fixed_multiplier', 1.5, 4.0, step=0.1),
                'band_lookback': trial.suggest_int('band_lookback', 1, 5),
                'src_type': trial.suggest_categorical('src_type', ['close', 'hlc3', 'hl2', 'ohlc4']),
            }
        
        elif stage == 2:
            # 第2段階: スムーザーパラメータの最適化
            base_params = cls.create_staged_optimization_params(trial, 1)
            
            # 選択されたスムーザーに応じて最適化
            smoother = base_params['midline_smoother']
            
            if smoother == 'hyper_frama':
                base_params.update({
                    'hyper_frama_period': trial.suggest_int('hyper_frama_period', 4, 25) * 2,
                    'hyper_frama_alpha_multiplier': trial.suggest_float('hyper_frama_alpha_multiplier', 0.1, 1.0, step=0.1),
                    'hyper_frama_fc_min': trial.suggest_float('hyper_frama_fc_min', 0.5, 2.0, step=0.1),
                    'hyper_frama_fc_max': trial.suggest_float('hyper_frama_fc_max', 4.0, 15.0, step=0.5),
                })
            elif smoother == 'ultimate_ma':
                base_params.update({
                    'ultimate_ma_ultimate_smoother_period': trial.suggest_int('ultimate_ma_ultimate_smoother_period', 2, 7) * 2.0,
                    'ultimate_ma_zero_lag_period': trial.suggest_int('ultimate_ma_zero_lag_period', 8, 50),
                    'ultimate_ma_realtime_window': trial.suggest_int('ultimate_ma_realtime_window', 30, 150),
                    'ultimate_ma_use_adaptive_kalman': trial.suggest_categorical('ultimate_ma_use_adaptive_kalman', [True, False]),
                })
            elif smoother == 'laguerre_filter':
                base_params.update({
                    'laguerre_gamma': trial.suggest_float('laguerre_gamma', 0.1, 0.9, step=0.1),
                    'laguerre_order': trial.suggest_int('laguerre_order', 2, 8),
                })
            elif smoother == 'super_smoother':
                base_params.update({
                    'super_smoother_length': trial.suggest_int('super_smoother_length', 8, 30),
                    'super_smoother_num_poles': trial.suggest_int('super_smoother_num_poles', 2, 3),
                })
            elif smoother == 'z_adaptive_ma':
                base_params.update({
                    'z_adaptive_fast_period': trial.suggest_int('z_adaptive_fast_period', 1, 8),
                    'z_adaptive_slow_period': trial.suggest_int('z_adaptive_slow_period', 50, 200),
                })
            
            return base_params
        
        elif stage == 3:
            # 第3段階: ATR・ER最適化
            base_params = cls.create_staged_optimization_params(trial, 2)
            base_params.update({
                # X_ATR最適化
                'x_atr_period': trial.suggest_int('x_atr_period', 4, 12) * 2.0,
                'x_atr_tr_method': trial.suggest_categorical('x_atr_tr_method', ['atr', 'str']),
                'x_atr_enable_kalman': trial.suggest_categorical('x_atr_enable_kalman', [True, False]),
                'x_atr_smoother_type': trial.suggest_categorical('x_atr_smoother_type', ['frama', 'super_smoother', 'ultimate_smoother']),
                
                # HyperER最適化
                'hyper_er_period': trial.suggest_int('hyper_er_period', 5, 25),
                'hyper_er_smooth_method': trial.suggest_categorical('hyper_er_smooth_method', ['frama', 'ema', 'sma']),
                'hyper_er_smooth_period': trial.suggest_int('hyper_er_smooth_period', 5, 30),
                'hyper_er_differential_period': trial.suggest_int('hyper_er_differential_period', 10, 50),
                'hyper_er_volatility_period': trial.suggest_int('hyper_er_volatility_period', 8, 30),
            })
            return base_params
        
        elif stage == 4:
            # 第4段階: 詳細調整
            base_params = cls.create_staged_optimization_params(trial, 3)
            base_params.update({
                # 詳細パラメータ
                'x_atr_percentile_window': trial.suggest_int('x_atr_percentile_window', 10, 50),
                'x_atr_percentile_low': trial.suggest_float('x_atr_percentile_low', 10.0, 30.0, step=5.0),
                'x_atr_percentile_high': trial.suggest_float('x_atr_percentile_high', 70.0, 90.0, step=5.0),
                
                'hyper_er_trend_period': trial.suggest_int('hyper_er_trend_period', 10, 50),
                'hyper_er_cycle_period': trial.suggest_int('hyper_er_cycle_period', 10, 40),
                'hyper_er_regime_period': trial.suggest_int('hyper_er_regime_period', 50, 200),
                'hyper_er_regime_threshold': trial.suggest_float('hyper_er_regime_threshold', 0.1, 0.6, step=0.05),
                
                # フラグ系
                'hyper_er_use_volume': trial.suggest_categorical('hyper_er_use_volume', [True, False]),
                'hyper_er_use_trend_filter': trial.suggest_categorical('hyper_er_use_trend_filter', [True, False]),
                'hyper_er_use_cycle_component': trial.suggest_categorical('hyper_er_use_cycle_component', [True, False]),
                'hyper_er_enable_regime_detection': trial.suggest_categorical('hyper_er_enable_regime_detection', [True, False]),
            })
            return base_params
        
        else:
            raise ValueError(f"無効な段階: {stage}. 1-4を指定してください。")
    
    @classmethod
    def convert_params_to_strategy_format(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        最適化パラメータを戦略パラメータに変換
        
        Args:
            params: 最適化パラメータ
            
        Returns:
            Dict[str, Any]: 戦略パラメータ
        """
        strategy_params = {}
        
        # 基本的な型変換
        for key, value in params.items():
            if isinstance(value, (int, np.integer)):
                strategy_params[key] = int(value)
            elif isinstance(value, (float, np.floating)):
                strategy_params[key] = float(value)
            elif isinstance(value, (bool, np.bool_)):
                strategy_params[key] = bool(value)
            else:
                strategy_params[key] = value
        
        return strategy_params
    
    @classmethod
    def get_optimization_groups(cls) -> Dict[str, List[str]]:
        """
        パラメータグループを取得（グループ別最適化用）
        
        Returns:
            Dict[str, List[str]]: パラメータグループ
        """
        return {
            'basic': [
                'period', 'band_lookback', 'src_type', 'midline_smoother', 
                'multiplier_mode', 'fixed_multiplier', 'enable_signals', 'enable_percentile', 'percentile_period'
            ],
            'hyper_frama': [
                'hyper_frama_src_type', 'hyper_frama_fc', 'hyper_frama_sc',
                'hyper_frama_alpha_multiplier', 'hyper_frama_period_mode', 'hyper_frama_cycle_detector_type',
                'hyper_frama_lp_period', 'hyper_frama_hp_period', 'hyper_frama_cycle_part',
                'hyper_frama_max_cycle', 'hyper_frama_min_cycle', 'hyper_frama_max_output', 'hyper_frama_min_output',
                'hyper_frama_enable_indicator_adaptation', 'hyper_frama_adaptation_indicator',
                'hyper_frama_hyper_er_period', 'hyper_frama_hyper_er_midline_period',
                'hyper_frama_hyper_adx_period', 'hyper_frama_hyper_adx_midline_period',
                'hyper_frama_hyper_trend_index_period', 'hyper_frama_hyper_trend_index_midline_period',
                'hyper_frama_fc_min', 'hyper_frama_fc_max', 'hyper_frama_sc_min', 'hyper_frama_sc_max',
                'hyper_frama_period_min', 'hyper_frama_period_max'
            ],
            'ultimate_ma': [
                'ultimate_ma_ultimate_smoother_period', 'ultimate_ma_zero_lag_period', 'ultimate_ma_realtime_window',
                'ultimate_ma_src_type', 'ultimate_ma_slope_index', 'ultimate_ma_range_threshold',
                'ultimate_ma_use_adaptive_kalman', 'ultimate_ma_kalman_process_variance',
                'ultimate_ma_kalman_measurement_variance', 'ultimate_ma_kalman_volatility_window',
                'ultimate_ma_zero_lag_period_mode', 'ultimate_ma_realtime_window_mode',
                'ultimate_ma_zl_cycle_detector_type', 'ultimate_ma_zl_cycle_detector_cycle_part',
                'ultimate_ma_zl_cycle_detector_max_cycle', 'ultimate_ma_zl_cycle_detector_min_cycle',
                'ultimate_ma_zl_cycle_period_multiplier', 'ultimate_ma_rt_cycle_detector_type',
                'ultimate_ma_rt_cycle_detector_cycle_part', 'ultimate_ma_rt_cycle_detector_max_cycle',
                'ultimate_ma_rt_cycle_detector_min_cycle', 'ultimate_ma_rt_cycle_period_multiplier'
            ],
            'laguerre': [
                'laguerre_gamma', 'laguerre_order', 'laguerre_src_type', 'laguerre_period',
                'laguerre_period_mode', 'laguerre_cycle_detector_type', 'laguerre_cycle_part',
                'laguerre_max_cycle', 'laguerre_min_cycle', 'laguerre_max_output', 'laguerre_min_output',
                'laguerre_lp_period', 'laguerre_hp_period'
            ],
            'super_smoother': [
                'super_smoother_length', 'super_smoother_num_poles', 'super_smoother_src_type',
                'super_smoother_period_mode', 'super_smoother_cycle_detector_type',
                'super_smoother_lp_period', 'super_smoother_hp_period', 'super_smoother_cycle_part',
                'super_smoother_max_cycle', 'super_smoother_min_cycle',
                'super_smoother_max_output', 'super_smoother_min_output'
            ],
            'z_adaptive': [
                'z_adaptive_fast_period', 'z_adaptive_slow_period', 'z_adaptive_src_type',
                'z_adaptive_slope_index', 'z_adaptive_range_threshold'
            ],
            'x_atr': [
                'x_atr_period', 'x_atr_tr_method', 'x_atr_smoother_type', 'x_atr_src_type',
                'x_atr_enable_kalman', 'x_atr_kalman_type', 'x_atr_period_mode',
                'x_atr_cycle_detector_type', 'x_atr_cycle_detector_cycle_part',
                'x_atr_cycle_detector_max_cycle', 'x_atr_cycle_detector_min_cycle',
                'x_atr_cycle_period_multiplier', 'x_atr_midline_period',
                'x_atr_enable_percentile_analysis', 'x_atr_percentile_lookback_period',
                'x_atr_percentile_low_threshold', 'x_atr_percentile_high_threshold'
            ],
            'hyper_er': [
                'hyper_er_period', 'hyper_er_midline_period', 'hyper_er_er_period', 'hyper_er_er_src_type',
                'hyper_er_use_kalman_filter', 'hyper_er_kalman_filter_type',
                'hyper_er_kalman_process_noise', 'hyper_er_kalman_min_observation_noise',
                'hyper_er_kalman_adaptation_window', 'hyper_er_use_roofing_filter',
                'hyper_er_roofing_hp_cutoff', 'hyper_er_roofing_ss_band_edge',
                'hyper_er_use_laguerre_filter', 'hyper_er_laguerre_gamma',
                'hyper_er_use_smoothing', 'hyper_er_smoother_type',
                'hyper_er_smoother_period', 'hyper_er_smoother_src_type',
                'hyper_er_use_dynamic_period', 'hyper_er_detector_type',
                'hyper_er_lp_period', 'hyper_er_hp_period', 'hyper_er_cycle_part',
                'hyper_er_max_cycle', 'hyper_er_min_cycle', 'hyper_er_max_output', 'hyper_er_min_output',
                'hyper_er_enable_percentile_analysis', 'hyper_er_percentile_lookback_period',
                'hyper_er_percentile_low_threshold', 'hyper_er_percentile_high_threshold'
            ]
        }
    
    @classmethod 
    def create_efficient_optimization_params(cls, trial: optuna.Trial, midline_smoother: str, optimization_focus: str = "performance") -> Dict[str, Any]:
        """
        効率的な最適化パラメータ生成（無駄な計算を排除）
        
        Args:
            trial: Optunaのトライアル
            midline_smoother: 使用するミッドラインスムーザー
            optimization_focus: 最適化の焦点
                - "performance": パフォーマンス重視（最小パラメータ）
                - "balance": バランス重視（中程度パラメータ）
                - "comprehensive": 包括的（多くのパラメータ）
                
        Returns:
            Dict[str, Any]: 効率的な最適化パラメータ
        """
        
        # 基本パラメータ（全てのスムーザーで共通）
        params = {
            'band_lookback': trial.suggest_int('band_lookback', 1, 3),
            'period': trial.suggest_int('period', 10, 25),
            'midline_smoother': midline_smoother,
            'src_type': trial.suggest_categorical('src_type', ['hlc3', 'hl2']),  # 効率的な選択肢のみ
            'multiplier_mode': 'dynamic',  # 最も効果的なモードに固定
            'enable_signals': True,  # 常に有効
            'enable_percentile': trial.suggest_categorical('enable_percentile', [True, False]),
        }
        
        # 効率的な共通パラメータ
        if optimization_focus in ["balance", "comprehensive"]:
            params.update({
                'fixed_multiplier': trial.suggest_float('fixed_multiplier', 2.0, 3.0, step=0.2),
                'percentile_period': trial.suggest_int('percentile_period', 80, 120),
                
                # X_ATRの効率的パラメータ
                'x_atr_period': trial.suggest_int('x_atr_period', 5, 10) * 2.0,
                'x_atr_tr_method': 'str',  # 最も効果的な方法に固定
                'x_atr_smoother_type': 'frama',  # 最も効果的なスムーザーに固定
                'x_atr_enable_kalman': trial.suggest_categorical('x_atr_enable_kalman', [False, True]),
                
                # HyperERの効率的パラメータ
                'hyper_er_period': trial.suggest_int('hyper_er_period', 8, 20),
                'hyper_er_midline_period': trial.suggest_int('hyper_er_midline_period', 80, 150),
                'hyper_er_use_kalman_filter': trial.suggest_categorical('hyper_er_use_kalman_filter', [True, False]),
                'hyper_er_use_roofing_filter': True,  # 効果的なフィルターを有効
            })
        
        # スムーザー別の効率的パラメータ
        if midline_smoother == 'hyper_frama':
            params.update({
                'hyper_frama_period': trial.suggest_int('hyper_frama_period', 6, 15) * 2,
                'hyper_frama_alpha_multiplier': trial.suggest_float('hyper_frama_alpha_multiplier', 0.3, 0.8, step=0.1),
                'hyper_frama_fc': trial.suggest_int('hyper_frama_fc', 1, 4),
                'hyper_frama_sc': trial.suggest_int('hyper_frama_sc', 100, 200),
            })
            
            if optimization_focus == "comprehensive":
                params.update({
                    'hyper_frama_period_mode': trial.suggest_categorical('hyper_frama_period_mode', ['fixed', 'dynamic']),
                    'hyper_frama_enable_indicator_adaptation': trial.suggest_categorical('hyper_frama_enable_indicator_adaptation', [True, False]),
                })
                
        elif midline_smoother == 'ultimate_ma':
            params.update({
                'ultimate_ma_ultimate_smoother_period': trial.suggest_int('ultimate_ma_ultimate_smoother_period', 2, 5) * 2.0,
                'ultimate_ma_zero_lag_period': trial.suggest_int('ultimate_ma_zero_lag_period', 15, 35),
                'ultimate_ma_realtime_window': trial.suggest_int('ultimate_ma_realtime_window', 60, 120),
                'ultimate_ma_use_adaptive_kalman': trial.suggest_categorical('ultimate_ma_use_adaptive_kalman', [True, False]),
            })
            
        elif midline_smoother == 'laguerre_filter':
            params.update({
                'laguerre_gamma': trial.suggest_float('laguerre_gamma', 0.3, 0.7, step=0.1),
                'laguerre_order': trial.suggest_int('laguerre_order', 3, 6),
                'laguerre_period': trial.suggest_int('laguerre_period', 4, 12),
            })
            
        elif midline_smoother == 'super_smoother':
            params.update({
                'super_smoother_length': trial.suggest_int('super_smoother_length', 12, 25),
                'super_smoother_num_poles': trial.suggest_int('super_smoother_num_poles', 2, 3),
            })
            
        elif midline_smoother == 'z_adaptive_ma':
            params.update({
                'z_adaptive_fast_period': trial.suggest_int('z_adaptive_fast_period', 2, 5),
                'z_adaptive_slow_period': trial.suggest_int('z_adaptive_slow_period', 80, 150),
            })
        
        return params
    
    @classmethod
    def get_parameter_count_estimate(cls, optimization_method: str, **kwargs) -> int:
        """
        最適化方法別のパラメータ数推定
        
        Args:
            optimization_method: 最適化方法名
            **kwargs: 追加引数
            
        Returns:
            int: 推定パラメータ数
        """
        estimates = {
            # 従来の方法
            'create_optimization_params': {
                'basic': 9,
                'balanced': 25,
                'comprehensive': 55,
                'full': 140
            },
            # 新しい条件付き方法
            'create_conditional_optimization_params': {
                'basic': 12,
                'balanced': 18,
                'comprehensive': 35,
                'full': 65
            },
            # 効率的方法
            'create_efficient_optimization_params': {
                'performance': 8,
                'balance': 15,
                'comprehensive': 25
            },
            # 段階的方法
            'create_staged_optimization_params': {
                1: 6,
                2: 12,
                3: 20,
                4: 35
            }
        }
        
        if optimization_method in estimates:
            method_estimates = estimates[optimization_method]
            level = kwargs.get('optimization_level', kwargs.get('optimization_focus', kwargs.get('stage', 'balanced')))
            return method_estimates.get(level, method_estimates.get('balanced', 20))
        
        return 20  # デフォルト推定値