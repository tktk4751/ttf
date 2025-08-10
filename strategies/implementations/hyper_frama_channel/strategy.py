#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import HyperFRAMAChannelSignalGenerator, FilterType


class HyperFRAMAChannelStrategy(BaseStrategy):
    """
    HyperFRAMAChannel ストラテジー（フィルタリング機能強化版）
    
    特徴:
    - HyperFRAMAChannelを使用したブレイクアウト戦略
    - 包括的なパラメータサポート（100+パラメータ）
    - シンプルなブレイクアウトロジック（HyperFRAMAトレンド判定なし）
    - Optuna最適化対応
    - 4つの高度なフィルターから選択可能:
      1. HyperER Filter - 効率性比率ベースの高精度トレンド判定
      2. HyperTrendIndex Filter - 高度なトレンドインデックスによる判定
      3. HyperADX Filter - 方向性移動インデックスによる判定
      4. Consensus Filter - 3つのフィルターの合意判定（3つのうち2つが1を出力）
    - フィルターなしオプションも提供
    
    エントリー条件:
    - ロング: 価格がHyperFRAMAChannel上限バンドをブレイクアウト かつ フィルターシグナル=1（フィルター有効時）
    - ショート: 価格がHyperFRAMAChannel下限バンドをブレイクアウト かつ フィルターシグナル=1（フィルター有効時）
    
    エグジット条件:
    - exit_mode=1: 逆方向ブレイクアウト（上限→下限、下限→上限）
    - exit_mode=2: 中心線クロス
    
    革新的な優位性:
    - チャネルブレイクアウトと高度なフィルタリングの組み合わせ
    - 複数のEhlersアルゴリズムの統合による高精度判定
    - 適応的フィルタリングによる誤判定の大幅削減
    - 市場状態に応じた自動フィルター調整
    - Numba JIT最適化による高速処理
    """
    
    def __init__(
        self,
        # 基本パラメータ
        band_lookback: int = 1,
        exit_mode: int = 1,  # 1: 逆ブレイクアウト, 2: 中心線クロス
        src_type: str = 'hlc3',
        
        # === HyperFRAMAChannel 基本パラメータ ===
        channel_period: int = 14,
        channel_multiplier_mode: str = "dynamic",
        channel_fixed_multiplier: float = 2.5,
        channel_src_type: str = "hlc3",
        
        # === HyperFRAMA パラメータ ===
        # 基本パラメータ
        channel_hyper_frama_period: int = 16,
        channel_hyper_frama_src_type: str = 'oc2',
        channel_hyper_frama_fc: int = 1,
        channel_hyper_frama_sc: int = 198,
        channel_hyper_frama_alpha_multiplier: float = 0.5,
        # 動的期間パラメータ
        channel_hyper_frama_period_mode: str = 'fixed',
        channel_hyper_frama_cycle_detector_type: str = 'hody_e',
        channel_hyper_frama_lp_period: int = 13,
        channel_hyper_frama_hp_period: int = 124,
        channel_hyper_frama_cycle_part: float = 0.5,
        channel_hyper_frama_max_cycle: int = 89,
        channel_hyper_frama_min_cycle: int = 8,
        channel_hyper_frama_max_output: int = 124,
        channel_hyper_frama_min_output: int = 8,
        # 動的適応パラメータ
        channel_hyper_frama_enable_indicator_adaptation: bool = True,
        channel_hyper_frama_adaptation_indicator: str = 'hyper_er',
        channel_hyper_frama_hyper_er_period: int = 14,
        channel_hyper_frama_hyper_er_midline_period: int = 100,
        channel_hyper_frama_hyper_adx_period: int = 14,
        channel_hyper_frama_hyper_adx_midline_period: int = 100,
        channel_hyper_frama_hyper_trend_index_period: int = 14,
        channel_hyper_frama_hyper_trend_index_midline_period: int = 100,
        channel_hyper_frama_fc_min: float = 1.0,
        channel_hyper_frama_fc_max: float = 8.0,
        channel_hyper_frama_sc_min: float = 50.0,
        channel_hyper_frama_sc_max: float = 250.0,
        channel_hyper_frama_period_min: int = 4,
        channel_hyper_frama_period_max: int = 44,
        
        # === X_ATR パラメータ ===
        channel_x_atr_period: float = 12.0,
        channel_x_atr_tr_method: str = 'atr',
        channel_x_atr_smoother_type: str = 'frama',
        channel_x_atr_src_type: str = 'close',
        channel_x_atr_enable_kalman: bool = False,
        channel_x_atr_kalman_type: str = 'unscented',
        # 動的適応パラメータ
        channel_x_atr_period_mode: str = 'dynamic',
        channel_x_atr_cycle_detector_type: str = 'dft_dominant',
        channel_x_atr_cycle_detector_cycle_part: float = 0.5,
        channel_x_atr_cycle_detector_max_cycle: int = 55,
        channel_x_atr_cycle_detector_min_cycle: int = 5,
        channel_x_atr_cycle_period_multiplier: float = 1.0,
        channel_x_atr_cycle_detector_period_range: tuple = (5, 120),
        # ミッドラインパラメータ
        channel_x_atr_midline_period: int = 100,
        # パーセンタイルベースボラティリティ分析パラメータ
        channel_x_atr_enable_percentile_analysis: bool = True,
        channel_x_atr_percentile_lookback_period: int = 50,
        channel_x_atr_percentile_low_threshold: float = 0.25,
        channel_x_atr_percentile_high_threshold: float = 0.75,
        # スムーサーパラメータ
        channel_x_atr_smoother_params: dict = None,
        # カルマンフィルターパラメータ
        channel_x_atr_kalman_params: dict = None,
        
        # === HyperER パラメータ ===
        channel_hyper_er_period: int = 8,
        channel_hyper_er_midline_period: int = 100,
        # ERパラメータ
        channel_hyper_er_er_period: int = 13,
        channel_hyper_er_er_src_type: str = 'oc2',
        # 統合カルマンフィルターパラメータ
        channel_hyper_er_use_kalman_filter: bool = True,
        channel_hyper_er_kalman_filter_type: str = 'simple',
        channel_hyper_er_kalman_process_noise: float = 1e-5,
        channel_hyper_er_kalman_min_observation_noise: float = 1e-6,
        channel_hyper_er_kalman_adaptation_window: int = 5,
        # ルーフィングフィルターパラメータ
        channel_hyper_er_use_roofing_filter: bool = True,
        channel_hyper_er_roofing_hp_cutoff: float = 55.0,
        channel_hyper_er_roofing_ss_band_edge: float = 10.0,
        # ラゲールフィルターパラメータ（後方互換性のため残す）
        channel_hyper_er_use_laguerre_filter: bool = False,
        channel_hyper_er_laguerre_gamma: float = 0.5,
        # 平滑化オプション
        channel_hyper_er_use_smoothing: bool = True,
        channel_hyper_er_smoother_type: str = 'frama',
        channel_hyper_er_smoother_period: int = 16,
        channel_hyper_er_smoother_src_type: str = 'close',
        # エラーズ統合サイクル検出器パラメータ
        channel_hyper_er_use_dynamic_period: bool = True,
        channel_hyper_er_detector_type: str = 'dft_dominant',
        channel_hyper_er_lp_period: int = 13,
        channel_hyper_er_hp_period: int = 124,
        channel_hyper_er_cycle_part: float = 0.4,
        channel_hyper_er_max_cycle: int = 124,
        channel_hyper_er_min_cycle: int = 13,
        channel_hyper_er_max_output: int = 89,
        channel_hyper_er_min_output: int = 5,
        # パーセンタイルベーストレンド分析パラメータ
        channel_hyper_er_enable_percentile_analysis: bool = True,
        channel_hyper_er_percentile_lookback_period: int = 50,
        channel_hyper_er_percentile_low_threshold: float = 0.25,
        channel_hyper_er_percentile_high_threshold: float = 0.75,
        
        # === HyperFRAMAチャネル独自パラメータ ===
        channel_enable_signals: bool = True,
        channel_enable_percentile: bool = True,
        channel_percentile_period: int = 100,
        
        # === フィルター設定 ===
        filter_type: FilterType = FilterType.NONE,  # フィルタータイプ
        # HyperER フィルターパラメータ
        filter_hyper_er_period: int = 14,
        filter_hyper_er_midline_period: int = 100,
        # HyperTrendIndex フィルターパラメータ
        filter_hyper_trend_index_period: int = 14,
        filter_hyper_trend_index_midline_period: int = 100,
        # HyperADX フィルターパラメータ
        filter_hyper_adx_period: int = 14,
        filter_hyper_adx_midline_period: int = 100
    ):
        """初期化"""
        filter_name = filter_type.value if isinstance(filter_type, FilterType) else str(filter_type)
        super().__init__(f"HyperFRAMAChannel_{filter_name}")
        
        # パラメータの設定（全パラメータ版）
        self._parameters = {
            # 基本パラメータ
            'band_lookback': band_lookback,
            'exit_mode': exit_mode,
            'src_type': src_type,
            
            # HyperFRAMAChannel 基本パラメータ
            'channel_period': channel_period,
            'channel_multiplier_mode': channel_multiplier_mode,
            'channel_fixed_multiplier': channel_fixed_multiplier,
            'channel_src_type': channel_src_type,
            
            # HyperFRAMAパラメータ
            'channel_hyper_frama_period': channel_hyper_frama_period,
            'channel_hyper_frama_src_type': channel_hyper_frama_src_type,
            'channel_hyper_frama_fc': channel_hyper_frama_fc,
            'channel_hyper_frama_sc': channel_hyper_frama_sc,
            'channel_hyper_frama_alpha_multiplier': channel_hyper_frama_alpha_multiplier,
            'channel_hyper_frama_period_mode': channel_hyper_frama_period_mode,
            'channel_hyper_frama_cycle_detector_type': channel_hyper_frama_cycle_detector_type,
            'channel_hyper_frama_lp_period': channel_hyper_frama_lp_period,
            'channel_hyper_frama_hp_period': channel_hyper_frama_hp_period,
            'channel_hyper_frama_cycle_part': channel_hyper_frama_cycle_part,
            'channel_hyper_frama_max_cycle': channel_hyper_frama_max_cycle,
            'channel_hyper_frama_min_cycle': channel_hyper_frama_min_cycle,
            'channel_hyper_frama_max_output': channel_hyper_frama_max_output,
            'channel_hyper_frama_min_output': channel_hyper_frama_min_output,
            'channel_hyper_frama_enable_indicator_adaptation': channel_hyper_frama_enable_indicator_adaptation,
            'channel_hyper_frama_adaptation_indicator': channel_hyper_frama_adaptation_indicator,
            'channel_hyper_frama_hyper_er_period': channel_hyper_frama_hyper_er_period,
            'channel_hyper_frama_hyper_er_midline_period': channel_hyper_frama_hyper_er_midline_period,
            'channel_hyper_frama_hyper_adx_period': channel_hyper_frama_hyper_adx_period,
            'channel_hyper_frama_hyper_adx_midline_period': channel_hyper_frama_hyper_adx_midline_period,
            'channel_hyper_frama_hyper_trend_index_period': channel_hyper_frama_hyper_trend_index_period,
            'channel_hyper_frama_hyper_trend_index_midline_period': channel_hyper_frama_hyper_trend_index_midline_period,
            'channel_hyper_frama_fc_min': channel_hyper_frama_fc_min,
            'channel_hyper_frama_fc_max': channel_hyper_frama_fc_max,
            'channel_hyper_frama_sc_min': channel_hyper_frama_sc_min,
            'channel_hyper_frama_sc_max': channel_hyper_frama_sc_max,
            'channel_hyper_frama_period_min': channel_hyper_frama_period_min,
            'channel_hyper_frama_period_max': channel_hyper_frama_period_max,
            
            # X_ATRパラメータ
            'channel_x_atr_period': channel_x_atr_period,
            'channel_x_atr_tr_method': channel_x_atr_tr_method,
            'channel_x_atr_smoother_type': channel_x_atr_smoother_type,
            'channel_x_atr_src_type': channel_x_atr_src_type,
            'channel_x_atr_enable_kalman': channel_x_atr_enable_kalman,
            'channel_x_atr_kalman_type': channel_x_atr_kalman_type,
            'channel_x_atr_period_mode': channel_x_atr_period_mode,
            'channel_x_atr_cycle_detector_type': channel_x_atr_cycle_detector_type,
            'channel_x_atr_cycle_detector_cycle_part': channel_x_atr_cycle_detector_cycle_part,
            'channel_x_atr_cycle_detector_max_cycle': channel_x_atr_cycle_detector_max_cycle,
            'channel_x_atr_cycle_detector_min_cycle': channel_x_atr_cycle_detector_min_cycle,
            'channel_x_atr_cycle_period_multiplier': channel_x_atr_cycle_period_multiplier,
            'channel_x_atr_cycle_detector_period_range': channel_x_atr_cycle_detector_period_range,
            'channel_x_atr_midline_period': channel_x_atr_midline_period,
            'channel_x_atr_enable_percentile_analysis': channel_x_atr_enable_percentile_analysis,
            'channel_x_atr_percentile_lookback_period': channel_x_atr_percentile_lookback_period,
            'channel_x_atr_percentile_low_threshold': channel_x_atr_percentile_low_threshold,
            'channel_x_atr_percentile_high_threshold': channel_x_atr_percentile_high_threshold,
            'channel_x_atr_smoother_params': channel_x_atr_smoother_params,
            'channel_x_atr_kalman_params': channel_x_atr_kalman_params,
            
            # HyperERパラメータ
            'channel_hyper_er_period': channel_hyper_er_period,
            'channel_hyper_er_midline_period': channel_hyper_er_midline_period,
            'channel_hyper_er_er_period': channel_hyper_er_er_period,
            'channel_hyper_er_er_src_type': channel_hyper_er_er_src_type,
            'channel_hyper_er_use_kalman_filter': channel_hyper_er_use_kalman_filter,
            'channel_hyper_er_kalman_filter_type': channel_hyper_er_kalman_filter_type,
            'channel_hyper_er_kalman_process_noise': channel_hyper_er_kalman_process_noise,
            'channel_hyper_er_kalman_min_observation_noise': channel_hyper_er_kalman_min_observation_noise,
            'channel_hyper_er_kalman_adaptation_window': channel_hyper_er_kalman_adaptation_window,
            'channel_hyper_er_use_roofing_filter': channel_hyper_er_use_roofing_filter,
            'channel_hyper_er_roofing_hp_cutoff': channel_hyper_er_roofing_hp_cutoff,
            'channel_hyper_er_roofing_ss_band_edge': channel_hyper_er_roofing_ss_band_edge,
            'channel_hyper_er_use_laguerre_filter': channel_hyper_er_use_laguerre_filter,
            'channel_hyper_er_laguerre_gamma': channel_hyper_er_laguerre_gamma,
            'channel_hyper_er_use_smoothing': channel_hyper_er_use_smoothing,
            'channel_hyper_er_smoother_type': channel_hyper_er_smoother_type,
            'channel_hyper_er_smoother_period': channel_hyper_er_smoother_period,
            'channel_hyper_er_smoother_src_type': channel_hyper_er_smoother_src_type,
            'channel_hyper_er_use_dynamic_period': channel_hyper_er_use_dynamic_period,
            'channel_hyper_er_detector_type': channel_hyper_er_detector_type,
            'channel_hyper_er_lp_period': channel_hyper_er_lp_period,
            'channel_hyper_er_hp_period': channel_hyper_er_hp_period,
            'channel_hyper_er_cycle_part': channel_hyper_er_cycle_part,
            'channel_hyper_er_max_cycle': channel_hyper_er_max_cycle,
            'channel_hyper_er_min_cycle': channel_hyper_er_min_cycle,
            'channel_hyper_er_max_output': channel_hyper_er_max_output,
            'channel_hyper_er_min_output': channel_hyper_er_min_output,
            'channel_hyper_er_enable_percentile_analysis': channel_hyper_er_enable_percentile_analysis,
            'channel_hyper_er_percentile_lookback_period': channel_hyper_er_percentile_lookback_period,
            'channel_hyper_er_percentile_low_threshold': channel_hyper_er_percentile_low_threshold,
            'channel_hyper_er_percentile_high_threshold': channel_hyper_er_percentile_high_threshold,
            
            # HyperFRAMAチャネル独自パラメータ
            'channel_enable_signals': channel_enable_signals,
            'channel_enable_percentile': channel_enable_percentile,
            'channel_percentile_period': channel_percentile_period,
            
            # フィルター設定
            'filter_type': filter_type,
            # HyperER フィルターパラメータ
            'filter_hyper_er_period': filter_hyper_er_period,
            'filter_hyper_er_midline_period': filter_hyper_er_midline_period,
            # HyperTrendIndex フィルターパラメータ
            'filter_hyper_trend_index_period': filter_hyper_trend_index_period,
            'filter_hyper_trend_index_midline_period': filter_hyper_trend_index_midline_period,
            # HyperADX フィルターパラメータ
            'filter_hyper_adx_period': filter_hyper_adx_period,
            'filter_hyper_adx_midline_period': filter_hyper_adx_midline_period
        }
        
        # シグナル生成器の初期化（全パラメータ版）
        self.signal_generator = HyperFRAMAChannelSignalGenerator(**self._parameters)
    
    def generate_entry(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エントリーシグナルを生成する"""
        try:
            return self.signal_generator.get_entry_signals(data)
        except Exception as e:
            self.logger.error(f"エントリーシグナル生成中にエラー: {str(e)}")
            return np.zeros(len(data), dtype=np.int8)
    
    def generate_exit(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """エグジットシグナルを生成する"""
        try:
            return self.signal_generator.get_exit_signals(data, position, index)
        except Exception as e:
            self.logger.error(f"エグジットシグナル生成中にエラー: {str(e)}")
            return False
    
    def get_channel_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """HyperFRAMAChannelシグナル取得"""
        try:
            return self.signal_generator.get_channel_signals(data)
        except Exception as e:
            self.logger.error(f"チャネルシグナル取得中にエラー: {str(e)}")
            return np.zeros(len(data), dtype=np.int8)
    
    def get_filter_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """フィルターシグナル取得"""
        try:
            return self.signal_generator.get_filter_signals(data)
        except Exception as e:
            self.logger.error(f"フィルターシグナル取得中にエラー: {str(e)}")
            return np.zeros(len(data), dtype=np.int8)
    
    def get_filter_details(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, np.ndarray]:
        """フィルター詳細情報を取得"""
        try:
            return self.signal_generator.get_filter_details(data)
        except Exception as e:
            self.logger.error(f"フィルター詳細取得中にエラー: {str(e)}")
            return {}
    
    def get_advanced_metrics(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, np.ndarray]:
        """全ての高度なメトリクスを取得"""
        try:
            return self.signal_generator.get_advanced_metrics(data)
        except Exception as e:
            self.logger.error(f"高度なメトリクス取得中にエラー: {str(e)}")
            return {}
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """ストラテジー情報を取得"""
        filter_type = self._parameters.get('filter_type', FilterType.NONE)
        filter_name = filter_type.value if isinstance(filter_type, FilterType) else str(filter_type)
        
        return {
            'name': 'HyperFRAMAChannel Strategy with Filtering',
            'description': f'HyperFRAMAChannelを使用したブレイクアウト戦略 with {filter_name} Filter Integration',
            'parameters': self._parameters.copy(),
            'features': [
                '包括的なパラメータサポート（100+パラメータ）',
                'HyperFRAMA, X_ATR, HyperERの完全統合',
                'シンプルなブレイクアウトロジック',
                '2つのエグジットモード（逆ブレイクアウト、中心線クロス）',
                f'Advanced {filter_name} filtering',
                'Optuna最適化フル対応（95+パラメータ）',
                '高性能Numba最適化',
                '適応的フィルタリングによる誤判定削減'
            ],
            'filter_capabilities': {
                'hyper_er': 'HyperER efficiency ratio-based trend filtering',
                'hyper_trend_index': 'HyperTrendIndex advanced trend detection',
                'hyper_adx': 'HyperADX directional movement filtering',
                'consensus': '3-filter consensus (2 out of 3 agreement)',
                'none': 'Pure HyperFRAMAChannel signals without filtering'
            },
            'channel_features': {
                'breakout_detection': 'Price breakout from HyperFRAMA channel bands',
                'dynamic_multipliers': 'HyperER-based adaptive channel multipliers',
                'exit_modes': 'Reverse breakout or centerline cross exits',
                'comprehensive_parameters': '100+ parameters for fine-tuning'
            }
        }
    
    def reset(self) -> None:
        """ストラテジーの状態をリセット"""
        super().reset()
        if hasattr(self.signal_generator, 'reset'):
            self.signal_generator.reset()
        
        # フィルターのリセット
        if hasattr(self.signal_generator, 'hyper_er_filter') and self.signal_generator.hyper_er_filter is not None:
            if hasattr(self.signal_generator.hyper_er_filter, 'reset'):
                self.signal_generator.hyper_er_filter.reset()
        if hasattr(self.signal_generator, 'hyper_trend_index_filter') and self.signal_generator.hyper_trend_index_filter is not None:
            if hasattr(self.signal_generator.hyper_trend_index_filter, 'reset'):
                self.signal_generator.hyper_trend_index_filter.reset()
        if hasattr(self.signal_generator, 'hyper_adx_filter') and self.signal_generator.hyper_adx_filter is not None:
            if hasattr(self.signal_generator.hyper_adx_filter, 'reset'):
                self.signal_generator.hyper_adx_filter.reset()
    
    @classmethod
    def create_optimization_params(cls, trial: optuna.Trial) -> Dict[str, Any]:
        """最適化パラメータを生成（全パラメータ版）"""
        params = {
            # 基本パラメータ
            'band_lookback': trial.suggest_int('band_lookback', 1, 5),
            'exit_mode': trial.suggest_categorical('exit_mode', [1, 2]),
            'src_type': trial.suggest_categorical('src_type', ['close', 'hlc3', 'hl2', 'ohlc4']),
            
            # HyperFRAMAChannel 基本パラメータ
            'channel_period': trial.suggest_int('channel_period', 5, 34),
            'channel_multiplier_mode': trial.suggest_categorical('channel_multiplier_mode', ['fixed', 'dynamic']),
            'channel_fixed_multiplier': trial.suggest_float('channel_fixed_multiplier', 1.0, 4.0, step=0.1),
            'channel_src_type': trial.suggest_categorical('channel_src_type', ['close', 'hlc3', 'hl2', 'ohlc4']),
            
            # HyperFRAMAパラメータ（基本）
            'channel_hyper_frama_period': trial.suggest_int('channel_hyper_frama_period', 8, 54, step=2),
            'channel_hyper_frama_src_type': trial.suggest_categorical('channel_hyper_frama_src_type', ['close', 'hlc3', 'hl2', 'ohlc4']),
            'channel_hyper_frama_fc': trial.suggest_int('channel_hyper_frama_fc', 1, 8),
            'channel_hyper_frama_sc': trial.suggest_int('channel_hyper_frama_sc', 50, 300),
            'channel_hyper_frama_alpha_multiplier': trial.suggest_float('channel_hyper_frama_alpha_multiplier', 0.1, 2.0, step=0.1),
            
            # HyperFRAMAパラメータ（動的期間）
            'channel_hyper_frama_period_mode': trial.suggest_categorical('channel_hyper_frama_period_mode', ['fixed', 'dynamic']),
            'channel_hyper_frama_cycle_detector_type': trial.suggest_categorical('channel_hyper_frama_cycle_detector_type', ['hody_e', 'phac_e', 'dudi_e']),
            'channel_hyper_frama_lp_period': trial.suggest_int('channel_hyper_frama_lp_period', 5, 34),
            'channel_hyper_frama_hp_period': trial.suggest_int('channel_hyper_frama_hp_period', 55, 200),
            'channel_hyper_frama_cycle_part': trial.suggest_float('channel_hyper_frama_cycle_part', 0.2, 0.8, step=0.1),
            'channel_hyper_frama_max_cycle': trial.suggest_int('channel_hyper_frama_max_cycle', 55, 144),
            'channel_hyper_frama_min_cycle': trial.suggest_int('channel_hyper_frama_min_cycle', 3, 21),
            'channel_hyper_frama_max_output': trial.suggest_int('channel_hyper_frama_max_output', 89, 200),
            'channel_hyper_frama_min_output': trial.suggest_int('channel_hyper_frama_min_output', 3, 21),
            
            # HyperFRAMAパラメータ（動的適応）
            'channel_hyper_frama_enable_indicator_adaptation': trial.suggest_categorical('channel_hyper_frama_enable_indicator_adaptation', [True, False]),
            'channel_hyper_frama_adaptation_indicator': trial.suggest_categorical('channel_hyper_frama_adaptation_indicator', ['hyper_er', 'hyper_adx', 'hyper_trend_index']),
            'channel_hyper_frama_hyper_er_period': trial.suggest_int('channel_hyper_frama_hyper_er_period', 8, 34),
            'channel_hyper_frama_hyper_er_midline_period': trial.suggest_int('channel_hyper_frama_hyper_er_midline_period', 50, 200),
            'channel_hyper_frama_fc_min': trial.suggest_float('channel_hyper_frama_fc_min', 0.5, 3.0, step=0.5),
            'channel_hyper_frama_fc_max': trial.suggest_float('channel_hyper_frama_fc_max', 5.0, 13.0, step=1.0),
            'channel_hyper_frama_sc_min': trial.suggest_float('channel_hyper_frama_sc_min', 21, 89, step=5),
            'channel_hyper_frama_sc_max': trial.suggest_float('channel_hyper_frama_sc_max', 144, 377, step=10),
            'channel_hyper_frama_period_min': trial.suggest_int('channel_hyper_frama_period_min', 2, 8, step=2),
            'channel_hyper_frama_period_max': trial.suggest_int('channel_hyper_frama_period_max', 34, 89, step=2),
            
            # X_ATRパラメータ（基本）
            'channel_x_atr_period': trial.suggest_float('channel_x_atr_period', 4.0, 34.0, step=2.0),
            'channel_x_atr_tr_method': trial.suggest_categorical('channel_x_atr_tr_method', ['atr', 'str']),
            'channel_x_atr_smoother_type': trial.suggest_categorical('channel_x_atr_smoother_type', ['frama', 'super_smoother', 'ultimate_smoother', 'alma']),
            'channel_x_atr_src_type': trial.suggest_categorical('channel_x_atr_src_type', ['close', 'hlc3', 'hl2', 'ohlc4']),
            'channel_x_atr_enable_kalman': trial.suggest_categorical('channel_x_atr_enable_kalman', [True, False]),
            'channel_x_atr_kalman_type': trial.suggest_categorical('channel_x_atr_kalman_type', ['simple', 'unscented']),
            
            # X_ATRパラメータ（動的適応）
            'channel_x_atr_period_mode': trial.suggest_categorical('channel_x_atr_period_mode', ['fixed', 'dynamic']),
            'channel_x_atr_cycle_detector_type': trial.suggest_categorical('channel_x_atr_cycle_detector_type', ['practical', 'dft_dominant', 'hody_e']),
            'channel_x_atr_cycle_detector_cycle_part': trial.suggest_float('channel_x_atr_cycle_detector_cycle_part', 0.2, 0.8, step=0.1),
            'channel_x_atr_cycle_detector_max_cycle': trial.suggest_int('channel_x_atr_cycle_detector_max_cycle', 34, 89),
            'channel_x_atr_cycle_detector_min_cycle': trial.suggest_int('channel_x_atr_cycle_detector_min_cycle', 3, 13),
            'channel_x_atr_cycle_period_multiplier': trial.suggest_float('channel_x_atr_cycle_period_multiplier', 0.5, 2.0, step=0.1),
            'channel_x_atr_midline_period': trial.suggest_int('channel_x_atr_midline_period', 50, 200),
            
            # X_ATRパラメータ（パーセンタイル分析）
            'channel_x_atr_enable_percentile_analysis': trial.suggest_categorical('channel_x_atr_enable_percentile_analysis', [True, False]),
            'channel_x_atr_percentile_lookback_period': trial.suggest_int('channel_x_atr_percentile_lookback_period', 21, 144),
            'channel_x_atr_percentile_low_threshold': trial.suggest_float('channel_x_atr_percentile_low_threshold', 0.1, 0.4, step=0.05),
            'channel_x_atr_percentile_high_threshold': trial.suggest_float('channel_x_atr_percentile_high_threshold', 0.6, 0.9, step=0.05),
            
            # HyperERパラメータ（基本）
            'channel_hyper_er_period': trial.suggest_int('channel_hyper_er_period', 3, 21),
            'channel_hyper_er_midline_period': trial.suggest_int('channel_hyper_er_midline_period', 50, 200),
            'channel_hyper_er_er_period': trial.suggest_int('channel_hyper_er_er_period', 8, 34),
            'channel_hyper_er_er_src_type': trial.suggest_categorical('channel_hyper_er_er_src_type', ['close', 'hlc3', 'hl2', 'ohlc4', 'oc2']),
            
            # HyperERパラメータ（カルマンフィルター）
            'channel_hyper_er_use_kalman_filter': trial.suggest_categorical('channel_hyper_er_use_kalman_filter', [True, False]),
            'channel_hyper_er_kalman_filter_type': trial.suggest_categorical('channel_hyper_er_kalman_filter_type', ['simple', 'unscented', 'extended']),
            'channel_hyper_er_kalman_process_noise': trial.suggest_float('channel_hyper_er_kalman_process_noise', 1e-6, 1e-3, log=True),
            'channel_hyper_er_kalman_min_observation_noise': trial.suggest_float('channel_hyper_er_kalman_min_observation_noise', 1e-7, 1e-4, log=True),
            'channel_hyper_er_kalman_adaptation_window': trial.suggest_int('channel_hyper_er_kalman_adaptation_window', 3, 13),
            
            # HyperERパラメータ（フィルター）
            'channel_hyper_er_use_roofing_filter': trial.suggest_categorical('channel_hyper_er_use_roofing_filter', [True, False]),
            'channel_hyper_er_roofing_hp_cutoff': trial.suggest_float('channel_hyper_er_roofing_hp_cutoff', 34.0, 89.0, step=5.0),
            'channel_hyper_er_roofing_ss_band_edge': trial.suggest_float('channel_hyper_er_roofing_ss_band_edge', 5.0, 21.0, step=1.0),
            'channel_hyper_er_use_laguerre_filter': trial.suggest_categorical('channel_hyper_er_use_laguerre_filter', [True, False]),
            'channel_hyper_er_laguerre_gamma': trial.suggest_float('channel_hyper_er_laguerre_gamma', 0.1, 0.9, step=0.1),
            
            # HyperERパラメータ（平滑化）
            'channel_hyper_er_use_smoothing': trial.suggest_categorical('channel_hyper_er_use_smoothing', [True, False]),
            'channel_hyper_er_smoother_type': trial.suggest_categorical('channel_hyper_er_smoother_type', ['frama', 'super_smoother', 'laguerre']),
            'channel_hyper_er_smoother_period': trial.suggest_int('channel_hyper_er_smoother_period', 8, 34, step=2),
            'channel_hyper_er_smoother_src_type': trial.suggest_categorical('channel_hyper_er_smoother_src_type', ['close', 'hlc3', 'hl2', 'ohlc4']),
            
            # HyperERパラメータ（動的期間検出）
            'channel_hyper_er_use_dynamic_period': trial.suggest_categorical('channel_hyper_er_use_dynamic_period', [True, False]),
            'channel_hyper_er_detector_type': trial.suggest_categorical('channel_hyper_er_detector_type', ['dft_dominant', 'autocorrelation', 'bandpass_zero']),
            'channel_hyper_er_lp_period': trial.suggest_int('channel_hyper_er_lp_period', 5, 34),
            'channel_hyper_er_hp_period': trial.suggest_int('channel_hyper_er_hp_period', 55, 200),
            'channel_hyper_er_cycle_part': trial.suggest_float('channel_hyper_er_cycle_part', 0.2, 0.8, step=0.1),
            'channel_hyper_er_max_cycle': trial.suggest_int('channel_hyper_er_max_cycle', 89, 200),
            'channel_hyper_er_min_cycle': trial.suggest_int('channel_hyper_er_min_cycle', 8, 34),
            'channel_hyper_er_max_output': trial.suggest_int('channel_hyper_er_max_output', 55, 144),
            'channel_hyper_er_min_output': trial.suggest_int('channel_hyper_er_min_output', 3, 13),
            
            # HyperERパラメータ（パーセンタイル分析）
            'channel_hyper_er_enable_percentile_analysis': trial.suggest_categorical('channel_hyper_er_enable_percentile_analysis', [True, False]),
            'channel_hyper_er_percentile_lookback_period': trial.suggest_int('channel_hyper_er_percentile_lookback_period', 21, 144),
            'channel_hyper_er_percentile_low_threshold': trial.suggest_float('channel_hyper_er_percentile_low_threshold', 0.1, 0.4, step=0.05),
            'channel_hyper_er_percentile_high_threshold': trial.suggest_float('channel_hyper_er_percentile_high_threshold', 0.6, 0.9, step=0.05),
            
            # HyperFRAMAチャネル独自パラメータ
            'channel_enable_signals': trial.suggest_categorical('channel_enable_signals', [True, False]),
            'channel_enable_percentile': trial.suggest_categorical('channel_enable_percentile', [True, False]),
            'channel_percentile_period': trial.suggest_int('channel_percentile_period', 50, 200),
            
            # フィルター設定
            'filter_type': trial.suggest_categorical('filter_type', [
                FilterType.NONE.value,
                FilterType.HYPER_ER.value,
                FilterType.HYPER_TREND_INDEX.value,
                FilterType.HYPER_ADX.value,
                FilterType.CONSENSUS.value
            ]),
            
            # フィルターパラメータ
            'filter_hyper_er_period': trial.suggest_int('filter_hyper_er_period', 5, 30),
            'filter_hyper_er_midline_period': trial.suggest_int('filter_hyper_er_midline_period', 50, 200, step=10),
            'filter_hyper_trend_index_period': trial.suggest_int('filter_hyper_trend_index_period', 5, 30),
            'filter_hyper_trend_index_midline_period': trial.suggest_int('filter_hyper_trend_index_midline_period', 50, 200, step=10),
            'filter_hyper_adx_period': trial.suggest_int('filter_hyper_adx_period', 5, 30),
            'filter_hyper_adx_midline_period': trial.suggest_int('filter_hyper_adx_midline_period', 50, 200, step=10)
        }
        return params
    
    @classmethod
    def convert_params_to_strategy_format(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """最適化パラメータを戦略パラメータに変換（全パラメータ版）"""
        strategy_params = {}
        
        # 基本パラメータ
        strategy_params.update({
            'band_lookback': int(params.get('band_lookback', 1)),
            'exit_mode': int(params.get('exit_mode', 1)),
            'src_type': params.get('src_type', 'hlc3'),
        })
        
        # HyperFRAMAChannel 基本パラメータ
        strategy_params.update({
            'channel_period': int(params.get('channel_period', 14)),
            'channel_multiplier_mode': params.get('channel_multiplier_mode', 'dynamic'),
            'channel_fixed_multiplier': float(params.get('channel_fixed_multiplier', 2.5)),
            'channel_src_type': params.get('channel_src_type', 'hlc3'),
        })
        
        # HyperFRAMAパラメータ（基本）
        strategy_params.update({
            'channel_hyper_frama_period': int(params.get('channel_hyper_frama_period', 16)),
            'channel_hyper_frama_src_type': params.get('channel_hyper_frama_src_type', 'hl2'),
            'channel_hyper_frama_fc': int(params.get('channel_hyper_frama_fc', 1)),
            'channel_hyper_frama_sc': int(params.get('channel_hyper_frama_sc', 198)),
            'channel_hyper_frama_alpha_multiplier': float(params.get('channel_hyper_frama_alpha_multiplier', 0.5)),
        })
        
        # HyperFRAMAパラメータ（動的期間）
        strategy_params.update({
            'channel_hyper_frama_period_mode': params.get('channel_hyper_frama_period_mode', 'fixed'),
            'channel_hyper_frama_cycle_detector_type': params.get('channel_hyper_frama_cycle_detector_type', 'hody_e'),
            'channel_hyper_frama_lp_period': int(params.get('channel_hyper_frama_lp_period', 13)),
            'channel_hyper_frama_hp_period': int(params.get('channel_hyper_frama_hp_period', 124)),
            'channel_hyper_frama_cycle_part': float(params.get('channel_hyper_frama_cycle_part', 0.5)),
            'channel_hyper_frama_max_cycle': int(params.get('channel_hyper_frama_max_cycle', 89)),
            'channel_hyper_frama_min_cycle': int(params.get('channel_hyper_frama_min_cycle', 8)),
            'channel_hyper_frama_max_output': int(params.get('channel_hyper_frama_max_output', 124)),
            'channel_hyper_frama_min_output': int(params.get('channel_hyper_frama_min_output', 8)),
        })
        
        # HyperFRAMAパラメータ（動的適応）
        strategy_params.update({
            'channel_hyper_frama_enable_indicator_adaptation': bool(params.get('channel_hyper_frama_enable_indicator_adaptation', True)),
            'channel_hyper_frama_adaptation_indicator': params.get('channel_hyper_frama_adaptation_indicator', 'hyper_er'),
            'channel_hyper_frama_hyper_er_period': int(params.get('channel_hyper_frama_hyper_er_period', 14)),
            'channel_hyper_frama_hyper_er_midline_period': int(params.get('channel_hyper_frama_hyper_er_midline_period', 100)),
            'channel_hyper_frama_hyper_adx_period': int(params.get('channel_hyper_frama_hyper_adx_period', 14)),
            'channel_hyper_frama_hyper_adx_midline_period': int(params.get('channel_hyper_frama_hyper_adx_midline_period', 100)),
            'channel_hyper_frama_hyper_trend_index_period': int(params.get('channel_hyper_frama_hyper_trend_index_period', 14)),
            'channel_hyper_frama_hyper_trend_index_midline_period': int(params.get('channel_hyper_frama_hyper_trend_index_midline_period', 100)),
            'channel_hyper_frama_fc_min': float(params.get('channel_hyper_frama_fc_min', 1.0)),
            'channel_hyper_frama_fc_max': float(params.get('channel_hyper_frama_fc_max', 8.0)),
            'channel_hyper_frama_sc_min': float(params.get('channel_hyper_frama_sc_min', 50.0)),
            'channel_hyper_frama_sc_max': float(params.get('channel_hyper_frama_sc_max', 250.0)),
            'channel_hyper_frama_period_min': int(params.get('channel_hyper_frama_period_min', 4)),
            'channel_hyper_frama_period_max': int(params.get('channel_hyper_frama_period_max', 44)),
        })
        
        # X_ATRパラメータ（基本）
        strategy_params.update({
            'channel_x_atr_period': float(params.get('channel_x_atr_period', 12.0)),
            'channel_x_atr_tr_method': params.get('channel_x_atr_tr_method', 'atr'),
            'channel_x_atr_smoother_type': params.get('channel_x_atr_smoother_type', 'frama'),
            'channel_x_atr_src_type': params.get('channel_x_atr_src_type', 'close'),
            'channel_x_atr_enable_kalman': bool(params.get('channel_x_atr_enable_kalman', False)),
            'channel_x_atr_kalman_type': params.get('channel_x_atr_kalman_type', 'unscented'),
        })
        
        # X_ATRパラメータ（動的適応）
        strategy_params.update({
            'channel_x_atr_period_mode': params.get('channel_x_atr_period_mode', 'dynamic'),
            'channel_x_atr_cycle_detector_type': params.get('channel_x_atr_cycle_detector_type', 'practical'),
            'channel_x_atr_cycle_detector_cycle_part': float(params.get('channel_x_atr_cycle_detector_cycle_part', 0.5)),
            'channel_x_atr_cycle_detector_max_cycle': int(params.get('channel_x_atr_cycle_detector_max_cycle', 55)),
            'channel_x_atr_cycle_detector_min_cycle': int(params.get('channel_x_atr_cycle_detector_min_cycle', 5)),
            'channel_x_atr_cycle_period_multiplier': float(params.get('channel_x_atr_cycle_period_multiplier', 1.0)),
            'channel_x_atr_cycle_detector_period_range': params.get('channel_x_atr_cycle_detector_period_range', (5, 120)),
            'channel_x_atr_midline_period': int(params.get('channel_x_atr_midline_period', 100)),
        })
        
        # X_ATRパラメータ（パーセンタイル分析）
        strategy_params.update({
            'channel_x_atr_enable_percentile_analysis': bool(params.get('channel_x_atr_enable_percentile_analysis', True)),
            'channel_x_atr_percentile_lookback_period': int(params.get('channel_x_atr_percentile_lookback_period', 50)),
            'channel_x_atr_percentile_low_threshold': float(params.get('channel_x_atr_percentile_low_threshold', 0.25)),
            'channel_x_atr_percentile_high_threshold': float(params.get('channel_x_atr_percentile_high_threshold', 0.75)),
            'channel_x_atr_smoother_params': params.get('channel_x_atr_smoother_params', None),
            'channel_x_atr_kalman_params': params.get('channel_x_atr_kalman_params', None),
        })
        
        # HyperERパラメータ（基本）
        strategy_params.update({
            'channel_hyper_er_period': int(params.get('channel_hyper_er_period', 8)),
            'channel_hyper_er_midline_period': int(params.get('channel_hyper_er_midline_period', 100)),
            'channel_hyper_er_er_period': int(params.get('channel_hyper_er_er_period', 13)),
            'channel_hyper_er_er_src_type': params.get('channel_hyper_er_er_src_type', 'oc2'),
        })
        
        # HyperERパラメータ（カルマンフィルター）
        strategy_params.update({
            'channel_hyper_er_use_kalman_filter': bool(params.get('channel_hyper_er_use_kalman_filter', True)),
            'channel_hyper_er_kalman_filter_type': params.get('channel_hyper_er_kalman_filter_type', 'simple'),
            'channel_hyper_er_kalman_process_noise': float(params.get('channel_hyper_er_kalman_process_noise', 1e-5)),
            'channel_hyper_er_kalman_min_observation_noise': float(params.get('channel_hyper_er_kalman_min_observation_noise', 1e-6)),
            'channel_hyper_er_kalman_adaptation_window': int(params.get('channel_hyper_er_kalman_adaptation_window', 5)),
        })
        
        # HyperERパラメータ（フィルター）
        strategy_params.update({
            'channel_hyper_er_use_roofing_filter': bool(params.get('channel_hyper_er_use_roofing_filter', True)),
            'channel_hyper_er_roofing_hp_cutoff': float(params.get('channel_hyper_er_roofing_hp_cutoff', 55.0)),
            'channel_hyper_er_roofing_ss_band_edge': float(params.get('channel_hyper_er_roofing_ss_band_edge', 10.0)),
            'channel_hyper_er_use_laguerre_filter': bool(params.get('channel_hyper_er_use_laguerre_filter', False)),
            'channel_hyper_er_laguerre_gamma': float(params.get('channel_hyper_er_laguerre_gamma', 0.5)),
        })
        
        # HyperERパラメータ（平滑化）
        strategy_params.update({
            'channel_hyper_er_use_smoothing': bool(params.get('channel_hyper_er_use_smoothing', True)),
            'channel_hyper_er_smoother_type': params.get('channel_hyper_er_smoother_type', 'frama'),
            'channel_hyper_er_smoother_period': int(params.get('channel_hyper_er_smoother_period', 16)),
            'channel_hyper_er_smoother_src_type': params.get('channel_hyper_er_smoother_src_type', 'close'),
        })
        
        # HyperERパラメータ（動的期間検出）
        strategy_params.update({
            'channel_hyper_er_use_dynamic_period': bool(params.get('channel_hyper_er_use_dynamic_period', True)),
            'channel_hyper_er_detector_type': params.get('channel_hyper_er_detector_type', 'dft_dominant'),
            'channel_hyper_er_lp_period': int(params.get('channel_hyper_er_lp_period', 13)),
            'channel_hyper_er_hp_period': int(params.get('channel_hyper_er_hp_period', 124)),
            'channel_hyper_er_cycle_part': float(params.get('channel_hyper_er_cycle_part', 0.4)),
            'channel_hyper_er_max_cycle': int(params.get('channel_hyper_er_max_cycle', 124)),
            'channel_hyper_er_min_cycle': int(params.get('channel_hyper_er_min_cycle', 13)),
            'channel_hyper_er_max_output': int(params.get('channel_hyper_er_max_output', 89)),
            'channel_hyper_er_min_output': int(params.get('channel_hyper_er_min_output', 5)),
        })
        
        # HyperERパラメータ（パーセンタイル分析）
        strategy_params.update({
            'channel_hyper_er_enable_percentile_analysis': bool(params.get('channel_hyper_er_enable_percentile_analysis', True)),
            'channel_hyper_er_percentile_lookback_period': int(params.get('channel_hyper_er_percentile_lookback_period', 50)),
            'channel_hyper_er_percentile_low_threshold': float(params.get('channel_hyper_er_percentile_low_threshold', 0.25)),
            'channel_hyper_er_percentile_high_threshold': float(params.get('channel_hyper_er_percentile_high_threshold', 0.75)),
        })
        
        # HyperFRAMAチャネル独自パラメータ
        strategy_params.update({
            'channel_enable_signals': bool(params.get('channel_enable_signals', True)),
            'channel_enable_percentile': bool(params.get('channel_enable_percentile', True)),
            'channel_percentile_period': int(params.get('channel_percentile_period', 100)),
            
            # フィルター設定
            'filter_type': FilterType(params.get('filter_type', FilterType.NONE.value)),
            # HyperER フィルターパラメータ
            'filter_hyper_er_period': int(params.get('filter_hyper_er_period', 14)),
            'filter_hyper_er_midline_period': int(params.get('filter_hyper_er_midline_period', 100)),
            # HyperTrendIndex フィルターパラメータ
            'filter_hyper_trend_index_period': int(params.get('filter_hyper_trend_index_period', 14)),
            'filter_hyper_trend_index_midline_period': int(params.get('filter_hyper_trend_index_midline_period', 100)),
            # HyperADX フィルターパラメータ
            'filter_hyper_adx_period': int(params.get('filter_hyper_adx_period', 14)),
            'filter_hyper_adx_midline_period': int(params.get('filter_hyper_adx_midline_period', 100))
        })
        
        return strategy_params