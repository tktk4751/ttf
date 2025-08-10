#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import HyperFRAMABollingerSignalGenerator, SignalType, FilterType


class HyperFRAMABollingerStrategy(BaseStrategy):
    """
    HyperFRAMA Bollinger Strategy
    
    特徴:
    - HyperFRAMAボリンジャーバンドベースの高度なトレードシステム
    - 動的シグマ適応（HyperER）による市場環境適応
    - ブレイクアウト/リバーサル戦略の選択可能
    - 4つの高度なフィルターから選択可能:
      1. HyperER Filter - 効率性比率ベースの高精度トレンド判定
      2. HyperTrendIndex Filter - 高度なトレンドインデックスによる判定
      3. HyperADX Filter - 方向性移動インデックスによる判定
      4. Consensus Filter - 3つのフィルターの合意判定（3つのうち2つが1を出力）
    - フィルターなしオプションも提供
    
    エントリー戦略:
    ブレイクアウト:
    - ロング: 上バンドブレイクアウト かつ フィルター許可（フィルター有効時）
    - ショート: 下バンドブレイクアウト かつ フィルター許可（フィルター有効時）
    
    リバーサル:
    - ロング: 下バンド付近からの反発 かつ フィルター許可（フィルター有効時）
    - ショート: 上バンド付近からの反落 かつ フィルター許可（フィルター有効時）
    
    エグジット戦略:
    - モード1: 逆ブレイクアウト（反対側バンド突破）
    - モード2: ミッドラインクロス（HyperFRAMAミッドライン突破）
    - モード3: パーセントB反転（パーセントBの反転）
    
    革新的な優位性:
    - HyperFRAMAによる適応的ミッドライン
    - 動的シグマ適応による市場効率性対応
    - パーセントBによる精密なポジション管理
    - 複数のEhlersアルゴリズムの統合による高精度判定
    - 適応的フィルタリングによる誤判定の大幅削減
    - Numba JIT最適化による高速処理
    """
    
    def __init__(
        self,
        # 基本シグナル設定
        signal_type: SignalType = SignalType.BREAKOUT,
        lookback: int = 1,
        exit_mode: int = 1,  # 1: 逆ブレイクアウト, 2: ミッドラインクロス, 3: パーセントB反転
        src_type: str = 'oc2',
        
        # === HyperFRAMABollinger パラメータ ===
        bollinger_period: int = 20,
        bollinger_sigma_mode: str = "fixed",
        bollinger_fixed_sigma: float = 1.5,
        bollinger_src_type: str = "oc2",
        
        # === HyperFRAMA パラメータ ===
        bollinger_hyper_frama_period: int = 16,
        bollinger_hyper_frama_src_type: str = 'oc2',
        bollinger_hyper_frama_fc: int = 1,
        bollinger_hyper_frama_sc: int = 198,
        bollinger_hyper_frama_alpha_multiplier: float = 0.5,
        # 動的期間パラメータ
        bollinger_hyper_frama_period_mode: str = 'fixed',
        bollinger_hyper_frama_cycle_detector_type: str = 'hody_e',
        bollinger_hyper_frama_lp_period: int = 13,
        bollinger_hyper_frama_hp_period: int = 124,
        bollinger_hyper_frama_cycle_part: float = 0.5,
        bollinger_hyper_frama_max_cycle: int = 89,
        bollinger_hyper_frama_min_cycle: int = 8,
        bollinger_hyper_frama_max_output: int = 124,
        bollinger_hyper_frama_min_output: int = 8,
        # 動的適応パラメータ
        bollinger_hyper_frama_enable_indicator_adaptation: bool = True,
        bollinger_hyper_frama_adaptation_indicator: str = 'hyper_er',
        bollinger_hyper_frama_hyper_er_period: int = 14,
        bollinger_hyper_frama_hyper_er_midline_period: int = 100,
        bollinger_hyper_frama_hyper_adx_period: int = 14,
        bollinger_hyper_frama_hyper_adx_midline_period: int = 100,
        bollinger_hyper_frama_hyper_trend_index_period: int = 14,
        bollinger_hyper_frama_hyper_trend_index_midline_period: int = 100,
        bollinger_hyper_frama_fc_min: float = 1.0,
        bollinger_hyper_frama_fc_max: float = 8.0,
        bollinger_hyper_frama_sc_min: float = 50.0,
        bollinger_hyper_frama_sc_max: float = 250.0,
        bollinger_hyper_frama_period_min: int = 4,
        bollinger_hyper_frama_period_max: int = 44,
        
        # === HyperER パラメータ ===
        bollinger_hyper_er_period: int = 8,
        bollinger_hyper_er_midline_period: int = 100,
        bollinger_hyper_er_er_period: int = 13,
        bollinger_hyper_er_er_src_type: str = 'oc2',
        bollinger_hyper_er_use_kalman_filter: bool = True,
        bollinger_hyper_er_kalman_filter_type: str = 'simple',
        bollinger_hyper_er_kalman_process_noise: float = 1e-5,
        bollinger_hyper_er_kalman_min_observation_noise: float = 1e-6,
        bollinger_hyper_er_kalman_adaptation_window: int = 5,
        bollinger_hyper_er_use_roofing_filter: bool = True,
        bollinger_hyper_er_roofing_hp_cutoff: float = 55.0,
        bollinger_hyper_er_roofing_ss_band_edge: float = 10.0,
        bollinger_hyper_er_use_laguerre_filter: bool = False,
        bollinger_hyper_er_laguerre_gamma: float = 0.5,
        bollinger_hyper_er_use_smoothing: bool = True,
        bollinger_hyper_er_smoother_type: str = 'frama',
        bollinger_hyper_er_smoother_period: int = 16,
        bollinger_hyper_er_smoother_src_type: str = 'close',
        bollinger_hyper_er_use_dynamic_period: bool = True,
        bollinger_hyper_er_detector_type: str = 'dft_dominant',
        bollinger_hyper_er_lp_period: int = 13,
        bollinger_hyper_er_hp_period: int = 124,
        bollinger_hyper_er_cycle_part: float = 0.4,
        bollinger_hyper_er_max_cycle: int = 124,
        bollinger_hyper_er_min_cycle: int = 13,
        bollinger_hyper_er_max_output: int = 89,
        bollinger_hyper_er_min_output: int = 5,
        bollinger_hyper_er_enable_percentile_analysis: bool = True,
        bollinger_hyper_er_percentile_lookback_period: int = 50,
        bollinger_hyper_er_percentile_low_threshold: float = 0.25,
        bollinger_hyper_er_percentile_high_threshold: float = 0.75,
        
        # シグマ範囲設定
        bollinger_sigma_min: float = 1.0,
        bollinger_sigma_max: float = 2.5,
        bollinger_enable_signals: bool = True,
        bollinger_enable_percentile: bool = True,
        bollinger_percentile_period: int = 100,
        
        # フィルター選択
        filter_type: FilterType = FilterType.NONE,
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
        """
        初期化
        
        Args:
            signal_type: シグナルタイプ（ブレイクアウト or リバーサル）
            lookback: ルックバック期間
            exit_mode: エグジットモード
            src_type: 価格ソースタイプ
            その他: HyperFRAMAボリンジャーと各フィルターのパラメータ
        """
        signal_name = signal_type.value if isinstance(signal_type, SignalType) else str(signal_type)
        filter_name = filter_type.value if isinstance(filter_type, FilterType) else str(filter_type)
        exit_mode_str = {1: "逆ブレイク", 2: "ミッドライン", 3: "パーセントB"}.get(exit_mode, "不明")
        
        super().__init__(f"HyperFRAMABollinger_{signal_name}_{filter_name}_{exit_mode_str}")
        
        # パラメータの設定
        self._parameters = {
            # 基本シグナル設定
            'signal_type': signal_type,
            'lookback': lookback,
            'exit_mode': exit_mode,
            'src_type': src_type,
            # HyperFRAMAボリンジャーパラメータ
            'bollinger_period': bollinger_period,
            'bollinger_sigma_mode': bollinger_sigma_mode,
            'bollinger_fixed_sigma': bollinger_fixed_sigma,
            'bollinger_src_type': bollinger_src_type,
            'bollinger_sigma_min': bollinger_sigma_min,
            'bollinger_sigma_max': bollinger_sigma_max,
            # HyperFRAMAパラメータ
            'bollinger_hyper_frama_period': bollinger_hyper_frama_period,
            'bollinger_hyper_frama_src_type': bollinger_hyper_frama_src_type,
            'bollinger_hyper_frama_fc': bollinger_hyper_frama_fc,
            'bollinger_hyper_frama_sc': bollinger_hyper_frama_sc,
            'bollinger_hyper_frama_alpha_multiplier': bollinger_hyper_frama_alpha_multiplier,
            # フィルター設定
            'filter_type': filter_type,
            'filter_hyper_er_period': filter_hyper_er_period,
            'filter_hyper_er_midline_period': filter_hyper_er_midline_period,
            'filter_hyper_trend_index_period': filter_hyper_trend_index_period,
            'filter_hyper_trend_index_midline_period': filter_hyper_trend_index_midline_period,
            'filter_hyper_adx_period': filter_hyper_adx_period,
            'filter_hyper_adx_midline_period': filter_hyper_adx_midline_period
        }
        
        # シグナル生成器の初期化
        self.signal_generator = HyperFRAMABollingerSignalGenerator(
            # 基本シグナル設定
            signal_type=signal_type,
            lookback=lookback,
            exit_mode=exit_mode,
            src_type=src_type,
            
            # HyperFRAMABollingerパラメータ
            bollinger_period=bollinger_period,
            bollinger_sigma_mode=bollinger_sigma_mode,
            bollinger_fixed_sigma=bollinger_fixed_sigma,
            bollinger_src_type=bollinger_src_type,
            
            # HyperFRAMAパラメータ
            bollinger_hyper_frama_period=bollinger_hyper_frama_period,
            bollinger_hyper_frama_src_type=bollinger_hyper_frama_src_type,
            bollinger_hyper_frama_fc=bollinger_hyper_frama_fc,
            bollinger_hyper_frama_sc=bollinger_hyper_frama_sc,
            bollinger_hyper_frama_alpha_multiplier=bollinger_hyper_frama_alpha_multiplier,
            bollinger_hyper_frama_period_mode=bollinger_hyper_frama_period_mode,
            bollinger_hyper_frama_cycle_detector_type=bollinger_hyper_frama_cycle_detector_type,
            bollinger_hyper_frama_lp_period=bollinger_hyper_frama_lp_period,
            bollinger_hyper_frama_hp_period=bollinger_hyper_frama_hp_period,
            bollinger_hyper_frama_cycle_part=bollinger_hyper_frama_cycle_part,
            bollinger_hyper_frama_max_cycle=bollinger_hyper_frama_max_cycle,
            bollinger_hyper_frama_min_cycle=bollinger_hyper_frama_min_cycle,
            bollinger_hyper_frama_max_output=bollinger_hyper_frama_max_output,
            bollinger_hyper_frama_min_output=bollinger_hyper_frama_min_output,
            bollinger_hyper_frama_enable_indicator_adaptation=bollinger_hyper_frama_enable_indicator_adaptation,
            bollinger_hyper_frama_adaptation_indicator=bollinger_hyper_frama_adaptation_indicator,
            bollinger_hyper_frama_hyper_er_period=bollinger_hyper_frama_hyper_er_period,
            bollinger_hyper_frama_hyper_er_midline_period=bollinger_hyper_frama_hyper_er_midline_period,
            bollinger_hyper_frama_hyper_adx_period=bollinger_hyper_frama_hyper_adx_period,
            bollinger_hyper_frama_hyper_adx_midline_period=bollinger_hyper_frama_hyper_adx_midline_period,
            bollinger_hyper_frama_hyper_trend_index_period=bollinger_hyper_frama_hyper_trend_index_period,
            bollinger_hyper_frama_hyper_trend_index_midline_period=bollinger_hyper_frama_hyper_trend_index_midline_period,
            bollinger_hyper_frama_fc_min=bollinger_hyper_frama_fc_min,
            bollinger_hyper_frama_fc_max=bollinger_hyper_frama_fc_max,
            bollinger_hyper_frama_sc_min=bollinger_hyper_frama_sc_min,
            bollinger_hyper_frama_sc_max=bollinger_hyper_frama_sc_max,
            bollinger_hyper_frama_period_min=bollinger_hyper_frama_period_min,
            bollinger_hyper_frama_period_max=bollinger_hyper_frama_period_max,
            
            # HyperERパラメータ
            bollinger_hyper_er_period=bollinger_hyper_er_period,
            bollinger_hyper_er_midline_period=bollinger_hyper_er_midline_period,
            bollinger_hyper_er_er_period=bollinger_hyper_er_er_period,
            bollinger_hyper_er_er_src_type=bollinger_hyper_er_er_src_type,
            bollinger_hyper_er_use_kalman_filter=bollinger_hyper_er_use_kalman_filter,
            bollinger_hyper_er_kalman_filter_type=bollinger_hyper_er_kalman_filter_type,
            bollinger_hyper_er_kalman_process_noise=bollinger_hyper_er_kalman_process_noise,
            bollinger_hyper_er_kalman_min_observation_noise=bollinger_hyper_er_kalman_min_observation_noise,
            bollinger_hyper_er_kalman_adaptation_window=bollinger_hyper_er_kalman_adaptation_window,
            bollinger_hyper_er_use_roofing_filter=bollinger_hyper_er_use_roofing_filter,
            bollinger_hyper_er_roofing_hp_cutoff=bollinger_hyper_er_roofing_hp_cutoff,
            bollinger_hyper_er_roofing_ss_band_edge=bollinger_hyper_er_roofing_ss_band_edge,
            bollinger_hyper_er_use_laguerre_filter=bollinger_hyper_er_use_laguerre_filter,
            bollinger_hyper_er_laguerre_gamma=bollinger_hyper_er_laguerre_gamma,
            bollinger_hyper_er_use_smoothing=bollinger_hyper_er_use_smoothing,
            bollinger_hyper_er_smoother_type=bollinger_hyper_er_smoother_type,
            bollinger_hyper_er_smoother_period=bollinger_hyper_er_smoother_period,
            bollinger_hyper_er_smoother_src_type=bollinger_hyper_er_smoother_src_type,
            bollinger_hyper_er_use_dynamic_period=bollinger_hyper_er_use_dynamic_period,
            bollinger_hyper_er_detector_type=bollinger_hyper_er_detector_type,
            bollinger_hyper_er_lp_period=bollinger_hyper_er_lp_period,
            bollinger_hyper_er_hp_period=bollinger_hyper_er_hp_period,
            bollinger_hyper_er_cycle_part=bollinger_hyper_er_cycle_part,
            bollinger_hyper_er_max_cycle=bollinger_hyper_er_max_cycle,
            bollinger_hyper_er_min_cycle=bollinger_hyper_er_min_cycle,
            bollinger_hyper_er_max_output=bollinger_hyper_er_max_output,
            bollinger_hyper_er_min_output=bollinger_hyper_er_min_output,
            bollinger_hyper_er_enable_percentile_analysis=bollinger_hyper_er_enable_percentile_analysis,
            bollinger_hyper_er_percentile_lookback_period=bollinger_hyper_er_percentile_lookback_period,
            bollinger_hyper_er_percentile_low_threshold=bollinger_hyper_er_percentile_low_threshold,
            bollinger_hyper_er_percentile_high_threshold=bollinger_hyper_er_percentile_high_threshold,
            
            # シグマ範囲設定
            bollinger_sigma_min=bollinger_sigma_min,
            bollinger_sigma_max=bollinger_sigma_max,
            bollinger_enable_signals=bollinger_enable_signals,
            bollinger_enable_percentile=bollinger_enable_percentile,
            bollinger_percentile_period=bollinger_percentile_period,
            
            # フィルター設定
            filter_type=filter_type,
            filter_hyper_er_period=filter_hyper_er_period,
            filter_hyper_er_midline_period=filter_hyper_er_midline_period,
            filter_hyper_trend_index_period=filter_hyper_trend_index_period,
            filter_hyper_trend_index_midline_period=filter_hyper_trend_index_midline_period,
            filter_hyper_adx_period=filter_hyper_adx_period,
            filter_hyper_adx_midline_period=filter_hyper_adx_midline_period
        )
    
    def generate_entry(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナルを生成する
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: エントリーシグナル（ロング=1、ショート=-1、なし=0）
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
    
    def get_bollinger_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> tuple:
        """ボリンジャーバンド値を取得"""
        try:
            return self.signal_generator.get_bollinger_values(data)
        except Exception as e:
            self.logger.error(f"ボリンジャーバンド値取得中にエラー: {str(e)}")
            n = len(data) if data is not None else 100
            return tuple(np.full(n, np.nan) for _ in range(5))
    
    def get_bollinger_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """ボリンジャーエントリーシグナル取得"""
        try:
            return self.signal_generator.get_bollinger_entry_signals(data)
        except Exception as e:
            self.logger.error(f"ボリンジャーエントリーシグナル取得中にエラー: {str(e)}")
            return np.zeros(len(data), dtype=np.int8)
    
    def get_bollinger_exit_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """ボリンジャーエグジットシグナル取得"""
        try:
            return self.signal_generator.get_bollinger_exit_signals(data)
        except Exception as e:
            self.logger.error(f"ボリンジャーエグジットシグナル取得中にエラー: {str(e)}")
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
    
    @classmethod
    def create_optimization_params(cls, trial: optuna.Trial) -> Dict[str, Any]:
        """
        最適化パラメータを生成
        
        Args:
            trial: Optunaのトライアル
            
        Returns:
            Dict[str, Any]: 最適化パラメータ
        """
        # シグナルタイプの選択
        signal_type = trial.suggest_categorical('signal_type', [
            SignalType.BREAKOUT.value,
            SignalType.REVERSAL.value
        ])
        
        # フィルタータイプの選択
        filter_type = trial.suggest_categorical('filter_type', [
            FilterType.NONE.value,
            FilterType.HYPER_ER.value,
            FilterType.HYPER_TREND_INDEX.value,
            FilterType.HYPER_ADX.value,
            FilterType.CONSENSUS.value
        ])
        
        params = {
            # 基本シグナル設定
            'signal_type': signal_type,
            'lookback': trial.suggest_int('lookback', 1, 5),
            'exit_mode': trial.suggest_int('exit_mode', 1, 3),
            'src_type': trial.suggest_categorical('src_type', ['close', 'hlc3', 'hl2', 'ohlc4', 'oc2']),
            
            # HyperFRAMAボリンジャーパラメータ
            'bollinger_period': trial.suggest_int('bollinger_period', 10, 30),
            'bollinger_sigma_mode': trial.suggest_categorical('bollinger_sigma_mode', ['fixed', 'dynamic']),
            'bollinger_fixed_sigma': trial.suggest_float('bollinger_fixed_sigma', 1.5, 3.0, step=0.1),
            'bollinger_src_type': trial.suggest_categorical('bollinger_src_type', ['close', 'hlc3', 'hl2', 'ohlc4']),
            
            # HyperFRAMAパラメータ
            'bollinger_hyper_frama_period': trial.suggest_int('bollinger_hyper_frama_period', 4, 32, step=2),
            'bollinger_hyper_frama_src_type': trial.suggest_categorical('bollinger_hyper_frama_src_type', ['hl2', 'hlc3', 'close']),
            'bollinger_hyper_frama_fc': trial.suggest_int('bollinger_hyper_frama_fc', 1, 8),
            'bollinger_hyper_frama_sc': trial.suggest_int('bollinger_hyper_frama_sc', 50, 300, step=10),
            'bollinger_hyper_frama_alpha_multiplier': trial.suggest_float('bollinger_hyper_frama_alpha_multiplier', 0.1, 1.0, step=0.05),
            
            # フィルター設定
            'filter_type': filter_type,
        }
        
        return params
    
    @classmethod
    def convert_params_to_strategy_format(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        最適化パラメータを戦略パラメータに変換
        
        Args:
            params: 最適化パラメータ
            
        Returns:
            Dict[str, Any]: 戦略パラメータ
        """
        strategy_params = {
            # 基本シグナル設定
            'signal_type': SignalType(params.get('signal_type', SignalType.BREAKOUT.value)),
            'lookback': int(params.get('lookback', 1)),
            'exit_mode': int(params.get('exit_mode', 2)),
            'src_type': params.get('src_type', 'close'),
            
            # HyperFRAMAボリンジャーパラメータ
            'bollinger_period': int(params.get('bollinger_period', 20)),
            'bollinger_sigma_mode': params.get('bollinger_sigma_mode', 'dynamic'),
            'bollinger_fixed_sigma': float(params.get('bollinger_fixed_sigma', 2.0)),
            'bollinger_src_type': params.get('bollinger_src_type', 'close'),
            
            # HyperFRAMAパラメータ
            'bollinger_hyper_frama_period': int(params.get('bollinger_hyper_frama_period', 16)),
            'bollinger_hyper_frama_src_type': params.get('bollinger_hyper_frama_src_type', 'hl2'),
            'bollinger_hyper_frama_fc': int(params.get('bollinger_hyper_frama_fc', 1)),
            'bollinger_hyper_frama_sc': int(params.get('bollinger_hyper_frama_sc', 198)),
            'bollinger_hyper_frama_alpha_multiplier': float(params.get('bollinger_hyper_frama_alpha_multiplier', 0.5)),
            
            # フィルター設定
            'filter_type': FilterType(params.get('filter_type', FilterType.NONE.value))
        }
        
        return strategy_params
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """ストラテジー情報を取得"""
        signal_type = self._parameters.get('signal_type', SignalType.BREAKOUT)
        filter_type = self._parameters.get('filter_type', FilterType.NONE)
        signal_name = signal_type.value if isinstance(signal_type, SignalType) else str(signal_type)
        filter_name = filter_type.value if isinstance(filter_type, FilterType) else str(filter_type)
        exit_mode = self._parameters.get('exit_mode', 2)
        exit_mode_str = {1: "逆ブレイクアウト", 2: "ミッドラインクロス", 3: "パーセントB反転"}.get(exit_mode, "不明")
        
        return {
            'name': 'HyperFRAMA Bollinger Strategy',
            'description': f'HyperFRAMA Bollinger Bands with {signal_name} signals, {filter_name} filtering, and {exit_mode_str} exit',
            'parameters': self._parameters.copy(),
            'features': [
                'HyperFRAMA adaptive midline with fractal dimension control',
                'Dynamic sigma adaptation via HyperER efficiency ratio',
                f'{signal_name.capitalize()} entry strategy',
                f'{exit_mode_str} exit strategy',
                'Percent B position management',
                f'Advanced {filter_name} filtering' if filter_type != FilterType.NONE else 'No filtering (pure Bollinger signals)',
                'Configurable sigma range (1.0-2.5)',
                'Market efficiency responsive band width',
                'Optimized with Numba JIT compilation'
            ],
            'signal_capabilities': {
                'breakout': 'Band breakthrough entry with percent B confirmation',
                'reversal': 'Band bounce entry with percent B reversal detection'
            },
            'exit_capabilities': {
                'reverse_breakout': 'Exit on opposite band breakthrough',
                'midline_cross': 'Exit on HyperFRAMA midline cross',
                'percent_b_reversal': 'Exit on percent B reversal signals'
            },
            'filter_capabilities': {
                'hyper_er': 'HyperER efficiency ratio-based trend filtering',
                'hyper_trend_index': 'HyperTrendIndex advanced trend detection',
                'hyper_adx': 'HyperADX directional movement filtering',
                'consensus': '3-filter consensus (2 out of 3 agreement)',
                'none': 'Pure HyperFRAMA Bollinger signals without filtering'
            },
            'adaptive_features': {
                'dynamic_sigma': 'HyperER-based sigma adaptation (1.0-2.5 range)',
                'fractal_midline': 'HyperFRAMA adaptive midline calculation',
                'percent_b_tracking': 'Precise band position tracking',
                'market_efficiency_response': 'Automatic band width adjustment to market conditions'
            }
        }
    
    def reset(self) -> None:
        """ストラテジーの状態をリセット"""
        super().reset()
        if hasattr(self.signal_generator, 'reset'):
            self.signal_generator.reset()