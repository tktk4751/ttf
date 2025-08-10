#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import EhlersInstantaneousTrendlineSignalGenerator, FilterType


class EhlersInstantaneousTrendlineStrategy(BaseStrategy):
    """
    Ehlers Instantaneous Trendline ストラテジー
    
    特徴:
    - Ehlers Instantaneous Trendlineベースの高度なトレードシステム
    - ITrendとTriggerラインによる瞬時トレンド検出
    - HyperERによる動的アルファ適応（0.04〜0.15の範囲）
    - カルマン統合フィルター + アルティメットスムーサーによる平滑化対応
    - 4つの高度なフィルターから選択可能:
      1. HyperER Filter - 効率性比率ベースの高精度トレンド判定
      2. HyperTrendIndex Filter - 高度なトレンドインデックスによる判定
      3. HyperADX Filter - 方向性移動インデックスによる判定
      4. Consensus Filter - 3つのフィルターの合意判定（3つのうち2つが1を出力）
    - フィルターなしオプションも提供
    
    エントリー条件:
    - ロング: Ehlersシグナル=1 かつ フィルターシグナル=1（フィルター有効時）
    - ショート: Ehlersシグナル=-1 かつ フィルターシグナル=1（フィルター有効時）
    - フィルターシグナル=0または逆方向の場合はスルー
    
    エグジット条件:
    - ロング: Ehlersシグナル=-1
    - ショート: Ehlersシグナル=1
    
    革新的な優位性:
    - 瞬時トレンドラインによる市場の即座な変化検出
    - HyperERによる適応的アルファ調整で市場状況に最適化
    - 複数のEhlersアルゴリズムの統合による高精度判定
    - 適応的フィルタリングによる誤判定の大幅削減
    - 市場状態に応じた自動フィルター調整
    - Numba JIT最適化による高速処理
    """
    
    def __init__(
        self,
        # Ehlers Instantaneous Trendlineパラメータ
        alpha: float = 0.07,
        src_type: str = 'oc2',
        # HyperER動的適応パラメータ
        enable_hyper_er_adaptation: bool = False,
        hyper_er_period: int = 14,
        hyper_er_midline_period: int = 100,
        alpha_min: float = 0.07,
        alpha_max: float = 0.15,
        # 平滑化モード設定
        smoothing_mode: str = 'none',
        # 統合カルマンフィルターパラメータ
        kalman_filter_type: str = 'simple',
        kalman_process_noise: float = 1e-5,
        kalman_min_observation_noise: float = 1e-6,
        kalman_adaptation_window: int = 5,
        # Ultimate Smootherパラメータ
        ultimate_smoother_period: int = 10,
        # シグナル設定
        position_mode: bool = True,              # 位置関係シグナル(True)またはクロスオーバーシグナル(False)
        # フィルター選択
        filter_type: FilterType = FilterType.HYPER_ADX,  # フィルタータイプ
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
            alpha: アルファ値（0.01-1.0の範囲、デフォルト: 0.07）
            src_type: ソースタイプ（デフォルト: 'hl2'）
            enable_hyper_er_adaptation: HyperER動的適応を有効にするか（デフォルト: True）
            hyper_er_period: HyperER計算期間（デフォルト: 14）
            hyper_er_midline_period: HyperERミッドライン期間（デフォルト: 100）
            alpha_min: アルファ最小値（HyperER低い時）（デフォルト: 0.04）
            alpha_max: アルファ最大値（HyperER高い時）（デフォルト: 0.15）
            smoothing_mode: 平滑化モード（デフォルト: 'none'） - 'none', 'kalman', 'ultimate', 'kalman_ultimate'
            kalman_filter_type: カルマンフィルタータイプ（'simple', 'unscented', 'unscented_v2', 'adaptive', 'multivariate', 'quantum_adaptive'）（デフォルト: 'simple'）
            kalman_process_noise: カルマンフィルター プロセスノイズ（デフォルト: 1e-5）
            kalman_min_observation_noise: カルマンフィルター 最小観測ノイズ（デフォルト: 1e-6）
            kalman_adaptation_window: カルマンフィルター 適応ウィンドウ（デフォルト: 5）
            ultimate_smoother_period: Ultimate Smoother 期間（デフォルト: 10）
            position_mode: 位置関係シグナル(True)またはクロスオーバーシグナル(False)
            filter_type: フィルタータイプ（デフォルト: FilterType.NONE）
            その他: 各フィルターのパラメータ
        """
        filter_name = filter_type.value if isinstance(filter_type, FilterType) else str(filter_type)
        signal_type = "Position" if position_mode else "Crossover"
        
        super().__init__(f"EhlersInstantaneousTrendline_{signal_type}_{filter_name}")
        
        # パラメータの設定
        self._parameters = {
            # Ehlers Instantaneous Trendlineパラメータ
            'alpha': alpha,
            'src_type': src_type,
            'enable_hyper_er_adaptation': enable_hyper_er_adaptation,
            'hyper_er_period': hyper_er_period,
            'hyper_er_midline_period': hyper_er_midline_period,
            'alpha_min': alpha_min,
            'alpha_max': alpha_max,
            # 平滑化パラメータ
            'smoothing_mode': smoothing_mode,
            'kalman_filter_type': kalman_filter_type,
            'kalman_process_noise': kalman_process_noise,
            'kalman_min_observation_noise': kalman_min_observation_noise,
            'kalman_adaptation_window': kalman_adaptation_window,
            'ultimate_smoother_period': ultimate_smoother_period,
            # シグナル設定
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
        self.signal_generator = EhlersInstantaneousTrendlineSignalGenerator(
            # Ehlers Instantaneous Trendlineパラメータ
            alpha=alpha,
            src_type=src_type,
            enable_hyper_er_adaptation=enable_hyper_er_adaptation,
            hyper_er_period=hyper_er_period,
            hyper_er_midline_period=hyper_er_midline_period,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            # 平滑化パラメータ
            smoothing_mode=smoothing_mode,
            kalman_filter_type=kalman_filter_type,
            kalman_process_noise=kalman_process_noise,
            kalman_min_observation_noise=kalman_min_observation_noise,
            kalman_adaptation_window=kalman_adaptation_window,
            ultimate_smoother_period=ultimate_smoother_period,
            # シグナル設定
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
    
    def get_itrend_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """ITrend値を取得"""
        try:
            return self.signal_generator.get_itrend_values(data)
        except Exception as e:
            self.logger.error(f"ITrend値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_trigger_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """Trigger値を取得"""
        try:
            return self.signal_generator.get_trigger_values(data)
        except Exception as e:
            self.logger.error(f"Trigger値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_alpha_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """Alpha値を取得"""
        try:
            return self.signal_generator.get_alpha_values(data)
        except Exception as e:
            self.logger.error(f"Alpha値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_smoothed_prices(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """平滑化後の価格を取得"""
        try:
            return self.signal_generator.get_smoothed_prices(data)
        except Exception as e:
            self.logger.error(f"平滑化価格取得中にエラー: {str(e)}")
            return np.array([])
    
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
    
    def get_ehlers_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Ehlers Instantaneous Trendlineシグナル取得"""
        try:
            return self.signal_generator.get_ehlers_signals(data)
        except Exception as e:
            self.logger.error(f"Ehlersシグナル取得中にエラー: {str(e)}")
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
            # Ehlers Instantaneous Trendlineパラメータ
            'alpha': trial.suggest_float('alpha', 0.01, 0.20, step=0.01),
            'src_type': trial.suggest_categorical('src_type', ['close', 'hlc3', 'hl2', 'ohlc4','oc2']),
            'enable_hyper_er_adaptation': trial.suggest_categorical('enable_hyper_er_adaptation', [True, False]),
            'alpha_min': trial.suggest_float('alpha_min', 0.02, 0.08, step=0.01),
            'alpha_max': trial.suggest_float('alpha_max', 0.10, 0.25, step=0.01),
            
            # 平滑化モード
            'smoothing_mode': trial.suggest_categorical('smoothing_mode', ['none', 'kalman', 'ultimate', 'kalman_ultimate']),
            'kalman_filter_type': trial.suggest_categorical('kalman_filter_type', ['simple', 'unscented', 'adaptive']),
            'ultimate_smoother_period': trial.suggest_int('ultimate_smoother_period', 5, 20),
            
            # シグナル設定
            'position_mode': trial.suggest_categorical('position_mode', [True, False]),
            
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
            # Ehlers Instantaneous Trendlineパラメータ
            'alpha': float(params.get('alpha', 0.07)),
            'src_type': params.get('src_type', 'hl2'),
            'enable_hyper_er_adaptation': bool(params.get('enable_hyper_er_adaptation', True)),
            'alpha_min': float(params.get('alpha_min', 0.04)),
            'alpha_max': float(params.get('alpha_max', 0.15)),
            
            # 平滑化パラメータ
            'smoothing_mode': params.get('smoothing_mode', 'none'),
            'kalman_filter_type': params.get('kalman_filter_type', 'simple'),
            'ultimate_smoother_period': int(params.get('ultimate_smoother_period', 10)),
            
            # シグナル設定
            'position_mode': bool(params.get('position_mode', True)),

            # フィルター設定
            'filter_type': FilterType(params['filter_type'])
        }
        
        return strategy_params
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """ストラテジー情報を取得"""
        filter_type = self._parameters.get('filter_type', FilterType.NONE)
        filter_name = filter_type.value if isinstance(filter_type, FilterType) else str(filter_type)
        
        return {
            'name': 'Ehlers Instantaneous Trendline Strategy',
            'description': f'Ehlers Instantaneous Trendline with {filter_name} Filter Integration',
            'parameters': self._parameters.copy(),
            'features': [
                'Instantaneous trendline-based market change detection',
                'HyperER dynamic alpha adaptation (0.04-0.15 range)',
                'ITrend and Trigger line analysis',
                'Configurable smoothing modes (Kalman, Ultimate Smoother)',
                'Multiple Ehlers algorithm integration',
                f'Advanced {filter_name} filtering',
                'Configurable position or crossover-based signals',
                'Optimized with Numba JIT compilation',
                'High-precision instantaneous trend detection'
            ],
            'filter_capabilities': {
                'hyper_er': 'HyperER efficiency ratio-based trend filtering',
                'hyper_trend_index': 'HyperTrendIndex advanced trend detection',
                'hyper_adx': 'HyperADX directional movement filtering',
                'consensus': '3-filter consensus (2 out of 3 agreement)',
                'none': 'Pure Ehlers Instantaneous Trendline signals without filtering'
            },
            'adaptive_features': {
                'hyper_er_alpha_adaptation': 'HyperER-based dynamic alpha adjustment',
                'smoothing_integration': 'Kalman and Ultimate Smoother price smoothing',
                'instantaneous_response': 'Real-time market trend change detection'
            }
        }
    
    def reset(self) -> None:
        """ストラテジーの状態をリセット"""
        super().reset()
        if hasattr(self.signal_generator, 'reset'):
            self.signal_generator.reset()