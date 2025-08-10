#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import HyperMAMAEnhancedSignalGenerator, FilterType


class HyperMAMAEnhancedStrategy(BaseStrategy):
    """
    HyperMAMA Enhanced ストラテジー
    
    特徴:
    - HyperMAMA (Hyper Mother of Adaptive Moving Average) ベースの高度なトレードシステム
    - HyperERインジケーターによる効率性ベースの動的適応制御
    - fastlimit: 0.1〜0.5、slowlimit: 0.01〜0.05 の動的調整
    - 4つの高度なフィルターから選択可能:
      1. HyperER Filter - 効率性比率ベースの高精度トレンド判定
      2. HyperTrendIndex Filter - 高度なトレンドインデックスによる判定
      3. HyperADX Filter - 方向性移動インデックスによる判定
      4. Consensus Filter - 3つのフィルターの合意判定（3つのうち2つが1を出力）
    - フィルターなしオプションも提供
    
    エントリー条件:
    - ロング: HyperMAMAシグナル=1 かつ フィルターシグナル=1（フィルター有効時）
    - ショート: HyperMAMAシグナル=-1 かつ フィルターシグナル=-1（フィルター有効時）
    - フィルターシグナル=0または逆方向の場合はスルー
    
    エグジット条件:
    - ロング: HyperMAMAシグナル=-1
    - ショート: HyperMAMAシグナル=1
    
    革新的な優位性:
    - HyperER効率性による市場適応制御
    - 複数のEhlersアルゴリズムの統合による高精度判定
    - 適応的フィルタリングによる誤判定の大幅削減
    - 市場状態に応じた自動フィルター調整
    - Numba JIT最適化による高速処理
    """
    
    def __init__(
        self,
        # 動的適応のトリガータイプ
        trigger_type: str = 'hyper_er',          # 'hyper_er' または 'hyper_trend_index'
        
        # HyperMAMAパラメータ
        hyper_er_period: int = 14,               # HyperER計算期間
        hyper_er_midline_period: int = 100,      # HyperERミッドライン計算期間
        fast_max: float = 0.8,                   # fastlimitの最大値
        fast_min: float = 0.01,                   # fastlimitの最小値
        slow_max: float = 0.08,                  # slowlimitの最大値
        slow_min: float = 0.001,                  # slowlimitの最小値
        # HyperERの詳細パラメータ
        hyper_er_er_period: int = 13,            # HyperER ER計算期間
        hyper_er_src_type: str = 'oc2',          # HyperER ソースタイプ
        # HyperER ルーフィングフィルターパラメータ
        hyper_er_use_roofing_filter: bool = True,  # ルーフィングフィルターを使用するか
        hyper_er_roofing_hp_cutoff: float = 55.0,  # ルーフィングフィルターのHighPassカットオフ
        hyper_er_roofing_ss_band_edge: float = 10.0,  # ルーフィングフィルターのSuperSmootherバンドエッジ
        # HyperER ラゲールフィルターパラメータ
        hyper_er_use_laguerre_filter: bool = False,  # ラゲールフィルターを使用するか
        hyper_er_laguerre_gamma: float = 0.9,  # ラゲールフィルターのガンマパラメータ
        # HyperER 平滑化オプション
        hyper_er_use_smoothing: bool = True,     # 平滑化を使用するか
        hyper_er_smoother_type: str = 'ultimate_smoother',   # 統合スムーサータイプ
        hyper_er_smoother_period: int = 16,      # スムーサー期間
        hyper_er_smoother_src_type: str = 'close',  # スムーサーソースタイプ
        # HyperER エラーズ統合サイクル検出器パラメータ
        hyper_er_use_dynamic_period: bool = True,   # 動的期間適応を使用するか
        hyper_er_detector_type: str = 'dft_dominant',     # サイクル検出器タイプ
        hyper_er_lp_period: int = 10,               # ローパスフィルター期間
        hyper_er_hp_period: int = 124,              # ハイパスフィルター期間
        hyper_er_cycle_part: float = 0.4,           # サイクル部分
        hyper_er_max_cycle: int = 89,              # 最大サイクル期間
        hyper_er_min_cycle: int = 10,               # 最小サイクル期間
        hyper_er_max_output: int = 55,              # 最大出力値
        hyper_er_min_output: int = 5,               # 最小出力値
        # HyperER パーセンタイルベーストレンド分析パラメータ
        hyper_er_enable_percentile_analysis: bool = True,  # パーセンタイル分析を有効にするか
        hyper_er_percentile_lookback_period: int = 50,     # パーセンタイル分析のルックバック期間
        hyper_er_percentile_low_threshold: float = 0.25,   # パーセンタイル分析の低閾値
        hyper_er_percentile_high_threshold: float = 0.75,  # パーセンタイル分析の高閾値
        
        # HyperTrendIndex関連パラメータ
        hyper_trend_period: int = 14,                    # HyperTrendIndex計算期間
        hyper_trend_midline_period: int = 100,           # HyperTrendIndexミッドライン計算期間
        hyper_trend_src_type: str = 'hlc3',              # HyperTrendIndexソースタイプ
        hyper_trend_use_kalman_filter: bool = True,      # HyperTrendIndexカルマンフィルターを使用するか
        hyper_trend_kalman_filter_type: str = 'unscented',  # HyperTrendIndexカルマンフィルタータイプ
        hyper_trend_use_dynamic_period: bool = True,     # HyperTrendIndex動的期間適応を使用するか
        hyper_trend_detector_type: str = 'dft_dominant',  # HyperTrendIndexサイクル検出器タイプ
        hyper_trend_use_roofing_filter: bool = True,     # HyperTrendIndexルーフィングフィルターを使用するか
        hyper_trend_roofing_hp_cutoff: float = 55.0,     # HyperTrendIndexルーフィングフィルターHighPassカットオフ
        hyper_trend_roofing_ss_band_edge: float = 10.0,  # HyperTrendIndexルーフィングフィルターSuperSmootherバンドエッジ
        # 動的適応パラメータ
        er_high_threshold: float = 0.8,          # HyperERの高閾値
        er_low_threshold: float = 0.2,           # HyperERの低閾値
        src_type: str = 'hlc3',                   # ソースタイプ
        # カルマンフィルターパラメータ（HyperMAMA用）
        use_kalman_filter: bool = True,         # カルマンフィルターを使用するか
        kalman_filter_type: str = 'unscented',   # カルマンフィルタータイプ
        kalman_process_noise: float = 0.01,      # プロセスノイズ
        kalman_observation_noise: float = 0.001, # 観測ノイズ
        # ゼロラグ処理パラメータ（HyperMAMA用）
        use_zero_lag: bool = False,              # ゼロラグ処理を使用するか
        # シグナル設定（HyperMAMA用）
        position_mode: bool = True,              # 位置関係シグナル(True)またはクロスオーバーシグナル(False)
        # フィルター選択
        filter_type: FilterType = FilterType.NONE,  # フィルタータイプ
        # HyperER フィルターパラメータ（2つ目）
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
            hyper_er_period: HyperER計算期間（デフォルト: 14）
            hyper_er_midline_period: HyperERミッドライン計算期間（デフォルト: 100）
            fast_max: fastlimitの最大値（デフォルト: 0.5）
            fast_min: fastlimitの最小値（デフォルト: 0.1）
            slow_max: slowlimitの最大値（デフォルト: 0.05）
            slow_min: slowlimitの最小値（デフォルト: 0.01）
            er_high_threshold: HyperERの高閾値（デフォルト: 0.8）
            er_low_threshold: HyperERの低閾値（デフォルト: 0.2）
            src_type: ソースタイプ（デフォルト: 'oc2'）
            use_kalman_filter: HyperMAMA用カルマンフィルター使用（デフォルト: False）
            kalman_filter_type: カルマンフィルタータイプ（デフォルト: 'unscented'）
            kalman_process_noise: プロセスノイズ（デフォルト: 0.01）
            kalman_observation_noise: 観測ノイズ（デフォルト: 0.001）
            use_zero_lag: ゼロラグ処理使用（デフォルト: False）
            position_mode: 位置関係シグナル(True)またはクロスオーバーシグナル(False)
            filter_type: フィルタータイプ（デフォルト: FilterType.NONE）
            その他: 各フィルターのパラメータ
        """
        filter_name = filter_type.value if isinstance(filter_type, FilterType) else str(filter_type)
        signal_type = "Position" if position_mode else "Crossover"
        kalman_str = f"_kalman({kalman_filter_type})" if use_kalman_filter else ""
        zero_lag_str = "_zero_lag" if use_zero_lag else ""
        
        super().__init__(f"HyperMAMA_Enhanced_{signal_type}_{filter_name}{kalman_str}{zero_lag_str}")
        
        # パラメータの設定
        self._parameters = {
            # HyperMAMAパラメータ
            'hyper_er_period': hyper_er_period,
            'hyper_er_midline_period': hyper_er_midline_period,
            # HyperERの詳細パラメータ
            'hyper_er_er_period': hyper_er_er_period,
            'hyper_er_src_type': hyper_er_src_type,
            # HyperER ルーフィングフィルターパラメータ
            'hyper_er_use_roofing_filter': hyper_er_use_roofing_filter,
            'hyper_er_roofing_hp_cutoff': hyper_er_roofing_hp_cutoff,
            'hyper_er_roofing_ss_band_edge': hyper_er_roofing_ss_band_edge,
            # HyperER ラゲールフィルターパラメータ
            'hyper_er_use_laguerre_filter': hyper_er_use_laguerre_filter,
            'hyper_er_laguerre_gamma': hyper_er_laguerre_gamma,
            # HyperER 平滑化オプション
            'hyper_er_use_smoothing': hyper_er_use_smoothing,
            'hyper_er_smoother_type': hyper_er_smoother_type,
            'hyper_er_smoother_period': hyper_er_smoother_period,
            'hyper_er_smoother_src_type': hyper_er_smoother_src_type,
            # HyperER エラーズ統合サイクル検出器パラメータ
            'hyper_er_use_dynamic_period': hyper_er_use_dynamic_period,
            'hyper_er_detector_type': hyper_er_detector_type,
            'hyper_er_lp_period': hyper_er_lp_period,
            'hyper_er_hp_period': hyper_er_hp_period,
            'hyper_er_cycle_part': hyper_er_cycle_part,
            'hyper_er_max_cycle': hyper_er_max_cycle,
            'hyper_er_min_cycle': hyper_er_min_cycle,
            'hyper_er_max_output': hyper_er_max_output,
            'hyper_er_min_output': hyper_er_min_output,
            # HyperER パーセンタイルベーストレンド分析パラメータ
            'hyper_er_enable_percentile_analysis': hyper_er_enable_percentile_analysis,
            'hyper_er_percentile_lookback_period': hyper_er_percentile_lookback_period,
            'hyper_er_percentile_low_threshold': hyper_er_percentile_low_threshold,
            'hyper_er_percentile_high_threshold': hyper_er_percentile_high_threshold,
            # HyperTrendIndexパラメータ
            'trigger_type': trigger_type,
            'hyper_trend_period': hyper_trend_period,
            'hyper_trend_midline_period': hyper_trend_midline_period,
            'hyper_trend_src_type': hyper_trend_src_type,
            'hyper_trend_use_kalman_filter': hyper_trend_use_kalman_filter,
            'hyper_trend_kalman_filter_type': hyper_trend_kalman_filter_type,
            'hyper_trend_use_dynamic_period': hyper_trend_use_dynamic_period,
            'hyper_trend_detector_type': hyper_trend_detector_type,
            'hyper_trend_use_roofing_filter': hyper_trend_use_roofing_filter,
            'hyper_trend_roofing_hp_cutoff': hyper_trend_roofing_hp_cutoff,
            'hyper_trend_roofing_ss_band_edge': hyper_trend_roofing_ss_band_edge,
            # 動的適応パラメータ
            'fast_max': fast_max,
            'fast_min': fast_min,
            'slow_max': slow_max,
            'slow_min': slow_min,
            'er_high_threshold': er_high_threshold,
            'er_low_threshold': er_low_threshold,
            'src_type': src_type,
            'use_kalman_filter': use_kalman_filter,
            'kalman_filter_type': kalman_filter_type,
            'kalman_process_noise': kalman_process_noise,
            'kalman_observation_noise': kalman_observation_noise,
            'use_zero_lag': use_zero_lag,
            'position_mode': position_mode,
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
        
        # シグナル生成器の初期化
        self.signal_generator = HyperMAMAEnhancedSignalGenerator(
            # トリガータイプ
            trigger_type=trigger_type,
            # HyperMAMAパラメータ
            hyper_er_period=hyper_er_period,
            hyper_er_midline_period=hyper_er_midline_period,
            # HyperERの詳細パラメータ
            hyper_er_er_period=hyper_er_er_period,
            hyper_er_src_type=hyper_er_src_type,
            # HyperER ルーフィングフィルターパラメータ
            hyper_er_use_roofing_filter=hyper_er_use_roofing_filter,
            hyper_er_roofing_hp_cutoff=hyper_er_roofing_hp_cutoff,
            hyper_er_roofing_ss_band_edge=hyper_er_roofing_ss_band_edge,
            # HyperER ラゲールフィルターパラメータ
            hyper_er_use_laguerre_filter=hyper_er_use_laguerre_filter,
            hyper_er_laguerre_gamma=hyper_er_laguerre_gamma,
            # HyperER 平滑化オプション
            hyper_er_use_smoothing=hyper_er_use_smoothing,
            hyper_er_smoother_type=hyper_er_smoother_type,
            hyper_er_smoother_period=hyper_er_smoother_period,
            hyper_er_smoother_src_type=hyper_er_smoother_src_type,
            # HyperER エラーズ統合サイクル検出器パラメータ
            hyper_er_use_dynamic_period=hyper_er_use_dynamic_period,
            hyper_er_detector_type=hyper_er_detector_type,
            hyper_er_lp_period=hyper_er_lp_period,
            hyper_er_hp_period=hyper_er_hp_period,
            hyper_er_cycle_part=hyper_er_cycle_part,
            hyper_er_max_cycle=hyper_er_max_cycle,
            hyper_er_min_cycle=hyper_er_min_cycle,
            hyper_er_max_output=hyper_er_max_output,
            hyper_er_min_output=hyper_er_min_output,
            # HyperER パーセンタイルベーストレンド分析パラメータ
            hyper_er_enable_percentile_analysis=hyper_er_enable_percentile_analysis,
            hyper_er_percentile_lookback_period=hyper_er_percentile_lookback_period,
            hyper_er_percentile_low_threshold=hyper_er_percentile_low_threshold,
            hyper_er_percentile_high_threshold=hyper_er_percentile_high_threshold,
            # HyperTrendIndexパラメータ
            hyper_trend_period=hyper_trend_period,
            hyper_trend_midline_period=hyper_trend_midline_period,
            hyper_trend_src_type=hyper_trend_src_type,
            hyper_trend_use_kalman_filter=hyper_trend_use_kalman_filter,
            hyper_trend_kalman_filter_type=hyper_trend_kalman_filter_type,
            hyper_trend_use_dynamic_period=hyper_trend_use_dynamic_period,
            hyper_trend_detector_type=hyper_trend_detector_type,
            hyper_trend_use_roofing_filter=hyper_trend_use_roofing_filter,
            hyper_trend_roofing_hp_cutoff=hyper_trend_roofing_hp_cutoff,
            hyper_trend_roofing_ss_band_edge=hyper_trend_roofing_ss_band_edge,
            # 動的適応パラメータ
            fast_max=fast_max,
            fast_min=fast_min,
            slow_max=slow_max,
            slow_min=slow_min,
            er_high_threshold=er_high_threshold,
            er_low_threshold=er_low_threshold,
            src_type=src_type,
            use_kalman_filter=use_kalman_filter,
            kalman_filter_type=kalman_filter_type,
            kalman_process_noise=kalman_process_noise,
            kalman_observation_noise=kalman_observation_noise,
            use_zero_lag=use_zero_lag,
            position_mode=position_mode,
            # フィルター設定
            filter_type=filter_type,
            # HyperER フィルターパラメータ
            filter_hyper_er_period=filter_hyper_er_period,
            filter_hyper_er_midline_period=filter_hyper_er_midline_period,
            # HyperTrendIndex フィルターパラメータ
            filter_hyper_trend_index_period=filter_hyper_trend_index_period,
            filter_hyper_trend_index_midline_period=filter_hyper_trend_index_midline_period,
            # HyperADX フィルターパラメータ
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
    
    def get_mama_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """HyperMAMA値を取得"""
        try:
            return self.signal_generator.get_mama_values(data)
        except Exception as e:
            self.logger.error(f"HyperMAMA値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_fama_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """HyperFAMA値を取得"""
        try:
            return self.signal_generator.get_fama_values(data)
        except Exception as e:
            self.logger.error(f"HyperFAMA値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_hyper_er_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """HyperER値を取得"""
        try:
            return self.signal_generator.get_hyper_er_values(data)
        except Exception as e:
            self.logger.error(f"HyperER値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_adaptive_limits(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Optional[tuple]:
        """動的適応されたfastlimitとslowlimitを取得"""
        try:
            return self.signal_generator.get_adaptive_limits(data)
        except Exception as e:
            self.logger.error(f"動的適応制限取得中にエラー: {str(e)}")
            return None
    
    def get_long_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """ロングエントリーシグナル取得"""
        try:
            return self.signal_generator.get_long_signals(data)
        except Exception as e:
            self.logger.error(f"ロングシグナル取得中にエラー: {str(e)}")
            return np.zeros(len(data), dtype=np.int8)
    
    def get_short_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """ショートエントリーシグナル取得"""
        try:
            return self.signal_generator.get_short_signals(data)
        except Exception as e:
            self.logger.error(f"ショートシグナル取得中にエラー: {str(e)}")
            return np.zeros(len(data), dtype=np.int8)
    
    def get_hyper_mama_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """HyperMAMAシグナル取得"""
        try:
            return self.signal_generator.get_hyper_mama_signals(data)
        except Exception as e:
            self.logger.error(f"HyperMAMAシグナル取得中にエラー: {str(e)}")
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
        # フィルタータイプの選択
        filter_type = trial.suggest_categorical('filter_type', [
            FilterType.NONE.value,
            FilterType.HYPER_ER.value,
            FilterType.HYPER_TREND_INDEX.value,
            FilterType.HYPER_ADX.value,
            FilterType.CONSENSUS.value
        ])
        
        params = {
            # トリガータイプの選択
            'trigger_type': trial.suggest_categorical('trigger_type', ['hyper_er', 'hyper_trend_index']),
            
            # HyperERの詳細パラメータ最適化
            'hyper_er_src_type': trial.suggest_categorical('hyper_er_src_type', ['close', 'hlc3', 'hl2', 'ohlc4', 'oc2']),

            # HyperER ルーフィングフィルターパラメータ
            'hyper_er_use_roofing_filter': trial.suggest_categorical('hyper_er_use_roofing_filter', [True, False]),
            'hyper_er_roofing_hp_cutoff': trial.suggest_float('hyper_er_roofing_hp_cutoff', 20.0, 100.0, step=4.0),
            'hyper_er_roofing_ss_band_edge': trial.suggest_float('hyper_er_roofing_ss_band_edge', 5.0, 19.0, step=1.0),
            # HyperER ラゲールフィルターパラメータ
            'hyper_er_use_laguerre_filter': trial.suggest_categorical('hyper_er_use_laguerre_filter', [True, False]),
            'hyper_er_laguerre_gamma': trial.suggest_float('hyper_er_laguerre_gamma', 0.1, 0.9, step=0.1),
            # HyperER 平滑化オプション
            'hyper_er_smoother_type': trial.suggest_categorical('hyper_er_smoother_type', ['super_smoother', 'laguerre', 'ultimate_smoother','alma','hma','zlema']),
            'hyper_er_smoother_period': trial.suggest_int('hyper_er_smoother_period', 5, 25),
            # HyperER エラーズ統合サイクル検出器パラメータ
            'hyper_er_detector_type': trial.suggest_categorical('hyper_er_detector_type', [
                # コア検出器
                'hody', 'phac', 'dudi', 'dudi_e', 'hody_e', 'phac_e',
                # 基本サイクル検出器
                'cycle_period', 'cycle_period2', 'bandpass_zero', 'autocorr_perio', 
                'dft_dominant', 'multi_bandpass', 'absolute_ultimate'
            ]),
            'hyper_er_lp_period': trial.suggest_int('hyper_er_lp_period', 5, 20),
            'hyper_er_hp_period': trial.suggest_int('hyper_er_hp_period', 40, 150),
            'hyper_er_cycle_part': trial.suggest_float('hyper_er_cycle_part', 0.2, 1.0, step=0.1),
            'hyper_er_max_cycle': trial.suggest_int('hyper_er_max_cycle', 40, 150),
            'hyper_er_min_cycle': trial.suggest_int('hyper_er_min_cycle', 5, 20),
            'hyper_er_max_output': trial.suggest_int('hyper_er_max_output', 30, 100),
            'hyper_er_min_output': trial.suggest_int('hyper_er_min_output', 3, 8),
            
            # HyperTrendIndexパラメータ最適化
            'hyper_trend_period': trial.suggest_int('hyper_trend_period', 10, 30),
            'hyper_trend_midline_period': trial.suggest_int('hyper_trend_midline_period', 50, 200),
            'hyper_trend_src_type': trial.suggest_categorical('hyper_trend_src_type', ['close', 'hlc3', 'hl2', 'ohlc4']),
            'hyper_trend_use_kalman_filter': trial.suggest_categorical('hyper_trend_use_kalman_filter', [True, False]),
            'hyper_trend_kalman_filter_type': trial.suggest_categorical('hyper_trend_kalman_filter_type', ['simple', 'unscented', 'quantum_adaptive']),
            'hyper_trend_use_dynamic_period': trial.suggest_categorical('hyper_trend_use_dynamic_period', [True, False]),
            'hyper_trend_detector_type': trial.suggest_categorical('hyper_trend_detector_type', [
                'dft_dominant', 'hody', 'phac', 'dudi', 'cycle_period', 'bandpass_zero'
            ]),
            'hyper_trend_use_roofing_filter': trial.suggest_categorical('hyper_trend_use_roofing_filter', [True, False]),
            'hyper_trend_roofing_hp_cutoff': trial.suggest_float('hyper_trend_roofing_hp_cutoff', 20.0, 100.0, step=5.0),
            'hyper_trend_roofing_ss_band_edge': trial.suggest_float('hyper_trend_roofing_ss_band_edge', 5.0, 20.0, step=1.0),

            'src_type': trial.suggest_categorical('src_type', ['close', 'hlc3', 'hl2', 'ohlc4', 'oc2']),
            
            # カルマンフィルターパラメータ
            'use_kalman_filter': trial.suggest_categorical('use_kalman_filter', [True, False]),
            'kalman_filter_type': trial.suggest_categorical('kalman_filter_type', ['quantum_adaptive', 'simple', 'unscented']),
            
            # ゼロラグ処理パラメータ
            'use_zero_lag': trial.suggest_categorical('use_zero_lag', [True, False]),
            
            # フィルター設定
            'filter_type': filter_type,
        }
        
        # フィルタータイプに応じたパラメータ
        if filter_type == FilterType.HYPER_ER.value:
            params.update({
                'filter_hyper_er_period': trial.suggest_int('filter_hyper_er_period', 10, 30),
                'filter_hyper_er_midline_period': trial.suggest_int('filter_hyper_er_midline_period', 50, 200)
            })
        elif filter_type == FilterType.HYPER_TREND_INDEX.value:
            params.update({
                'filter_hyper_trend_index_period': trial.suggest_int('filter_hyper_trend_index_period', 10, 30),
                'filter_hyper_trend_index_midline_period': trial.suggest_int('filter_hyper_trend_index_midline_period', 50, 200)
            })
        elif filter_type == FilterType.HYPER_ADX.value:
            params.update({
                'filter_hyper_adx_period': trial.suggest_int('filter_hyper_adx_period', 10, 30),
                'filter_hyper_adx_midline_period': trial.suggest_int('filter_hyper_adx_midline_period', 50, 200)
            })
        elif filter_type == FilterType.CONSENSUS.value:
            params.update({
                'filter_hyper_er_period': trial.suggest_int('filter_hyper_er_period', 10, 30),
                'filter_hyper_er_midline_period': trial.suggest_int('filter_hyper_er_midline_period', 50, 200),
                'filter_hyper_trend_index_period': trial.suggest_int('filter_hyper_trend_index_period', 10, 30),
                'filter_hyper_trend_index_midline_period': trial.suggest_int('filter_hyper_trend_index_midline_period', 50, 200),
                'filter_hyper_adx_period': trial.suggest_int('filter_hyper_adx_period', 10, 30),
                'filter_hyper_adx_midline_period': trial.suggest_int('filter_hyper_adx_midline_period', 50, 200)
            })
        
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
            # トリガータイプ
            'trigger_type': params.get('trigger_type', 'hyper_er'),
            
            # HyperTrendIndexパラメータ
            'hyper_trend_period': int(params.get('hyper_trend_period', 14)),
            'hyper_trend_midline_period': int(params.get('hyper_trend_midline_period', 100)),
            'hyper_trend_src_type': params.get('hyper_trend_src_type', 'hlc3'),
            'hyper_trend_use_kalman_filter': bool(params.get('hyper_trend_use_kalman_filter', True)),
            'hyper_trend_kalman_filter_type': params.get('hyper_trend_kalman_filter_type', 'simple'),
            'hyper_trend_use_dynamic_period': bool(params.get('hyper_trend_use_dynamic_period', True)),
            'hyper_trend_detector_type': params.get('hyper_trend_detector_type', 'dft_dominant'),
            'hyper_trend_use_roofing_filter': bool(params.get('hyper_trend_use_roofing_filter', True)),
            'hyper_trend_roofing_hp_cutoff': float(params.get('hyper_trend_roofing_hp_cutoff', 55.0)),
            'hyper_trend_roofing_ss_band_edge': float(params.get('hyper_trend_roofing_ss_band_edge', 10.0)),
            
            'hyper_er_src_type': params.get('hyper_er_src_type', 'oc2'),
            # HyperER ルーフィングフィルターパラメータ
            'hyper_er_use_roofing_filter': bool(params.get('hyper_er_use_roofing_filter', True)),
            'hyper_er_roofing_hp_cutoff': float(params.get('hyper_er_roofing_hp_cutoff', 48.0)),
            'hyper_er_roofing_ss_band_edge': float(params.get('hyper_er_roofing_ss_band_edge', 10.0)),
            # HyperER ラゲールフィルターパラメータ
            'hyper_er_use_laguerre_filter': bool(params.get('hyper_er_use_laguerre_filter', False)),
            'hyper_er_laguerre_gamma': float(params.get('hyper_er_laguerre_gamma', 0.8)),
            # HyperER 平滑化オプション
            'hyper_er_smoother_type': params.get('hyper_er_smoother_type', 'frama'),
            'hyper_er_smoother_period': int(params.get('hyper_er_smoother_period', 12)),
            # HyperER エラーズ統合サイクル検出器パラメータ
            'hyper_er_detector_type': params.get('hyper_er_detector_type', 'hody_e'),
            'hyper_er_lp_period': int(params.get('hyper_er_lp_period', 13)),
            'hyper_er_hp_period': int(params.get('hyper_er_hp_period', 124)),
            'hyper_er_cycle_part': float(params.get('hyper_er_cycle_part', 0.5)),
            'hyper_er_max_cycle': int(params.get('hyper_er_max_cycle', 124)),
            'hyper_er_min_cycle': int(params.get('hyper_er_min_cycle', 13)),
            'hyper_er_max_output': int(params.get('hyper_er_max_output', 89)),
            'hyper_er_min_output': int(params.get('hyper_er_min_output', 5)),
            'src_type': params['src_type'],
            'use_kalman_filter': bool(params['use_kalman_filter']),
            'kalman_filter_type': params['kalman_filter_type'],
            'use_zero_lag': bool(params['use_zero_lag']),
            'filter_type': FilterType(params['filter_type'])
        }
        
        # フィルタータイプに応じたパラメータの追加
        filter_type = params['filter_type']
        if filter_type == FilterType.HYPER_ER.value:
            strategy_params.update({
                'filter_hyper_er_period': int(params.get('filter_hyper_er_period', 14)),
                'filter_hyper_er_midline_period': int(params.get('filter_hyper_er_midline_period', 100))
            })
        elif filter_type == FilterType.HYPER_TREND_INDEX.value:
            strategy_params.update({
                'filter_hyper_trend_index_period': int(params.get('filter_hyper_trend_index_period', 14)),
                'filter_hyper_trend_index_midline_period': int(params.get('filter_hyper_trend_index_midline_period', 100))
            })
        elif filter_type == FilterType.HYPER_ADX.value:
            strategy_params.update({
                'filter_hyper_adx_period': int(params.get('filter_hyper_adx_period', 14)),
                'filter_hyper_adx_midline_period': int(params.get('filter_hyper_adx_midline_period', 100))
            })
        elif filter_type == FilterType.CONSENSUS.value:
            strategy_params.update({
                'filter_hyper_er_period': int(params.get('filter_hyper_er_period', 14)),
                'filter_hyper_er_midline_period': int(params.get('filter_hyper_er_midline_period', 100)),
                'filter_hyper_trend_index_period': int(params.get('filter_hyper_trend_index_period', 14)),
                'filter_hyper_trend_index_midline_period': int(params.get('filter_hyper_trend_index_midline_period', 100)),
                'filter_hyper_adx_period': int(params.get('filter_hyper_adx_period', 14)),
                'filter_hyper_adx_midline_period': int(params.get('filter_hyper_adx_midline_period', 100))
            })
        
        return strategy_params
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """ストラテジー情報を取得"""
        filter_type = self._parameters.get('filter_type', FilterType.NONE)
        filter_name = filter_type.value if isinstance(filter_type, FilterType) else str(filter_type)
        
        return {
            'name': 'HyperMAMA Enhanced Strategy',
            'description': f'Hyper Mother of Adaptive Moving Average with {filter_name} Filter Integration and HyperER Dynamic Adaptation',
            'parameters': self._parameters.copy(),
            'features': [
                'HyperER efficiency-based dynamic adaptation',
                'Adaptive fastlimit (0.1-0.5) and slowlimit (0.01-0.05) control',
                'Market cycle-responsive moving average system',
                'Multiple Ehlers algorithm integration',
                f'Advanced {filter_name} filtering',
                'Kalman Filter integration for noise reduction',
                'Zero-lag processing for faster response',
                'Configurable crossover or position-based signals',
                'Optimized with Numba JIT compilation',
                'High-precision trend and cycle detection with efficiency weighting'
            ],
            'filter_capabilities': {
                'hyper_er': 'HyperER efficiency ratio-based trend filtering',
                'hyper_trend_index': 'HyperTrendIndex advanced trend detection',
                'hyper_adx': 'HyperADX directional movement filtering',
                'consensus': '3-filter consensus (2 out of 3 agreement)',
                'none': 'Pure HyperMAMA signals without filtering'
            },
            'adaptive_features': {
                'hyper_er_dynamic_adaptation': 'Efficiency ratio-based parameter adjustment',
                'kaufman_scaling_approach': 'Smooth parameter transitions based on market efficiency',
                'adaptive_range_control': 'fastlimit: 0.1-0.5, slowlimit: 0.01-0.05'
            }
        }
    
    def reset(self) -> None:
        """ストラテジーの状態をリセット"""
        super().reset()
        if hasattr(self.signal_generator, 'reset'):
            self.signal_generator.reset()