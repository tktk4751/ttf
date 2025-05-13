#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import ZAdaptiveMACrossoverSignalGenerator


class ZAdaptiveMACrossoverStrategy(BaseStrategy):
    """
    ZAdaptiveMACrossover戦略
    
    特徴:
    - 2本のZAdaptiveMAのクロスオーバーに基づくシンプルかつ効果的なトレード戦略
    - 短期・長期それぞれに最適化されたサイクル効率比（CER）を使用
    - Numbaによる高速処理
    
    エントリー条件:
    - ロング: 短期ZMA > 長期ZMA (ゴールデンクロス)
    - ショート: 短期ZMA < 長期ZMA (デッドクロス)
    
    エグジット条件:
    - ロング: 短期ZMA < 長期ZMA (デッドクロス)
    - ショート: 短期ZMA > 長期ZMA (ゴールデンクロス)
    """
    
    def __init__(
        self,
        # 短期ZMAパラメータ
        short_fast_period: int = 2,
        short_slow_period: int = 15,
        # 長期ZMAパラメータ
        long_fast_period: int = 2,
        long_slow_period: int = 30,
        # 共通パラメータ
        src_type: str = 'hlc3',
        # 短期サイクルERパラメータ
        short_detector_type: str = 'dudi',
        short_lp_period: int = 5,
        short_hp_period: int = 89,
        short_cycle_part: float = 0.4,
        short_cycle_max: int = 144,
        short_cycle_min: int = 5,
        short_max_output: int = 55,
        short_min_output: int = 5,
        short_use_kalman_filter: bool = False,
        short_kalman_measurement_noise: float = 1.0,
        short_kalman_process_noise: float = 0.01,
        short_kalman_n_states: int = 5,
        short_smooth_er: bool = True,
        short_er_alma_period: int = 5,
        short_er_alma_offset: float = 0.85,
        short_er_alma_sigma: float = 6,
        short_self_adaptive: bool = False,
        # 長期サイクルERパラメータ
        long_detector_type: str = 'dudi',
        long_lp_period: int = 5,
        long_hp_period: int = 89,
        long_cycle_part: float = 0.4,
        long_cycle_max: int = 144,
        long_cycle_min: int = 5,
        long_max_output: int = 55,
        long_min_output: int = 5,
        long_use_kalman_filter: bool = False,
        long_kalman_measurement_noise: float = 1.0,
        long_kalman_process_noise: float = 0.01,
        long_kalman_n_states: int = 5,
        long_smooth_er: bool = True,
        long_er_alma_period: int = 5,
        long_er_alma_offset: float = 0.85,
        long_er_alma_sigma: float = 6,
        long_self_adaptive: bool = False
    ):
        """
        初期化
        
        Args:
            # 短期ZMAパラメータ
            short_fast_period: 短期ZMAの速い移動平均期間
            short_slow_period: 短期ZMAの遅い移動平均期間
            # 長期ZMAパラメータ
            long_fast_period: 長期ZMAの速い移動平均期間
            long_slow_period: 長期ZMAの遅い移動平均期間
            # 共通パラメータ
            src_type: 価格ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4')
            
            # 短期サイクルERパラメータ
            short_detector_type: 短期サイクル検出器タイプ ('hody', 'phac', 'dudi', etc.)
            short_lp_period: 短期サイクル用ローパスフィルター期間
            short_hp_period: 短期サイクル用ハイパスフィルター期間
            short_cycle_part: 短期サイクル部分比率
            short_cycle_max: 短期サイクルの最大期間
            short_cycle_min: 短期サイクルの最小期間
            short_max_output: 短期サイクルの最大出力値
            short_min_output: 短期サイクルの最小出力値
            short_use_kalman_filter: 短期サイクルでカルマンフィルターを使用するか
            short_kalman_measurement_noise: 短期カルマンフィルターの測定ノイズ
            short_kalman_process_noise: 短期カルマンフィルターのプロセスノイズ
            short_kalman_n_states: 短期カルマンフィルターの状態数
            short_smooth_er: 短期効率比をスムージングするかどうか
            short_er_alma_period: 短期効率比スムージング用ALMAの期間
            short_er_alma_offset: 短期効率比スムージング用ALMAのオフセット
            short_er_alma_sigma: 短期効率比スムージング用ALMAのシグマ
            short_self_adaptive: 短期効率比をセルフアダプティブにするかどうか
            
            # 長期サイクルERパラメータ
            long_detector_type: 長期サイクル検出器タイプ ('hody', 'phac', 'dudi', etc.)
            long_lp_period: 長期サイクル用ローパスフィルター期間
            long_hp_period: 長期サイクル用ハイパスフィルター期間
            long_cycle_part: 長期サイクル部分比率
            long_cycle_max: 長期サイクルの最大期間
            long_cycle_min: 長期サイクルの最小期間
            long_max_output: 長期サイクルの最大出力値
            long_min_output: 長期サイクルの最小出力値
            long_use_kalman_filter: 長期サイクルでカルマンフィルターを使用するか
            long_kalman_measurement_noise: 長期カルマンフィルターの測定ノイズ
            long_kalman_process_noise: 長期カルマンフィルターのプロセスノイズ
            long_kalman_n_states: 長期カルマンフィルターの状態数
            long_smooth_er: 長期効率比をスムージングするかどうか
            long_er_alma_period: 長期効率比スムージング用ALMAの期間
            long_er_alma_offset: 長期効率比スムージング用ALMAのオフセット
            long_er_alma_sigma: 長期効率比スムージング用ALMAのシグマ
            long_self_adaptive: 長期効率比をセルフアダプティブにするかどうか
        """
        super().__init__("ZAdaptiveMACrossover")
        
        # パラメータの設定
        self._parameters = {
            # 短期ZMAパラメータ
            'short_fast_period': short_fast_period,
            'short_slow_period': short_slow_period,
            # 長期ZMAパラメータ
            'long_fast_period': long_fast_period,
            'long_slow_period': long_slow_period,
            # 共通パラメータ
            'src_type': src_type,
            # 短期サイクルERパラメータ
            'short_detector_type': short_detector_type,
            'short_lp_period': short_lp_period,
            'short_hp_period': short_hp_period,
            'short_cycle_part': short_cycle_part,
            'short_cycle_max': short_cycle_max,
            'short_cycle_min': short_cycle_min,
            'short_max_output': short_max_output,
            'short_min_output': short_min_output,
            'short_use_kalman_filter': short_use_kalman_filter,
            'short_kalman_measurement_noise': short_kalman_measurement_noise,
            'short_kalman_process_noise': short_kalman_process_noise,
            'short_kalman_n_states': short_kalman_n_states,
            'short_smooth_er': short_smooth_er,
            'short_er_alma_period': short_er_alma_period,
            'short_er_alma_offset': short_er_alma_offset,
            'short_er_alma_sigma': short_er_alma_sigma,
            'short_self_adaptive': short_self_adaptive,
            # 長期サイクルERパラメータ
            'long_detector_type': long_detector_type,
            'long_lp_period': long_lp_period,
            'long_hp_period': long_hp_period,
            'long_cycle_part': long_cycle_part,
            'long_cycle_max': long_cycle_max,
            'long_cycle_min': long_cycle_min,
            'long_max_output': long_max_output,
            'long_min_output': long_min_output,
            'long_use_kalman_filter': long_use_kalman_filter,
            'long_kalman_measurement_noise': long_kalman_measurement_noise,
            'long_kalman_process_noise': long_kalman_process_noise,
            'long_kalman_n_states': long_kalman_n_states,
            'long_smooth_er': long_smooth_er,
            'long_er_alma_period': long_er_alma_period,
            'long_er_alma_offset': long_er_alma_offset,
            'long_er_alma_sigma': long_er_alma_sigma,
            'long_self_adaptive': long_self_adaptive
        }
        
        # シグナル生成器の初期化
        self.signal_generator = ZAdaptiveMACrossoverSignalGenerator(
            # 短期ZMAパラメータ
            short_fast_period=short_fast_period,
            short_slow_period=short_slow_period,
            # 長期ZMAパラメータ
            long_fast_period=long_fast_period,
            long_slow_period=long_slow_period,
            # 共通パラメータ
            src_type=src_type,
            # 短期サイクルERパラメータ
            short_detector_type=short_detector_type,
            short_lp_period=short_lp_period,
            short_hp_period=short_hp_period,
            short_cycle_part=short_cycle_part,
            short_cycle_max=short_cycle_max,
            short_cycle_min=short_cycle_min,
            short_max_output=short_max_output,
            short_min_output=short_min_output,
            short_use_kalman_filter=short_use_kalman_filter,
            short_kalman_measurement_noise=short_kalman_measurement_noise,
            short_kalman_process_noise=short_kalman_process_noise,
            short_kalman_n_states=short_kalman_n_states,
            short_smooth_er=short_smooth_er,
            short_er_alma_period=short_er_alma_period,
            short_er_alma_offset=short_er_alma_offset,
            short_er_alma_sigma=short_er_alma_sigma,
            short_self_adaptive=short_self_adaptive,
            # 長期サイクルERパラメータ
            long_detector_type=long_detector_type,
            long_lp_period=long_lp_period,
            long_hp_period=long_hp_period,
            long_cycle_part=long_cycle_part,
            long_cycle_max=long_cycle_max,
            long_cycle_min=long_cycle_min,
            long_max_output=long_max_output,
            long_min_output=long_min_output,
            long_use_kalman_filter=long_use_kalman_filter,
            long_kalman_measurement_noise=long_kalman_measurement_noise,
            long_kalman_process_noise=long_kalman_process_noise,
            long_kalman_n_states=long_kalman_n_states,
            long_smooth_er=long_smooth_er,
            long_er_alma_period=long_er_alma_period,
            long_er_alma_offset=long_er_alma_offset,
            long_er_alma_sigma=long_er_alma_sigma,
            long_self_adaptive=long_self_adaptive
        )
    
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
    
    def get_short_ma(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        短期ZMAの値を取得
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: 短期ZMAの値
        """
        try:
            return self.signal_generator.get_short_ma(data)
        except Exception as e:
            self.logger.error(f"短期ZMA取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_long_ma(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        長期ZMAの値を取得
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: 長期ZMAの値
        """
        try:
            return self.signal_generator.get_long_ma(data)
        except Exception as e:
            self.logger.error(f"長期ZMA取得中にエラー: {str(e)}")
            return np.array([])
    
    @classmethod
    def create_optimization_params(cls, trial: optuna.Trial) -> Dict[str, Any]:
        """
        最適化パラメータを生成
        
        Args:
            trial: Optunaのトライアル
            
        Returns:
            Dict[str, Any]: 最適化パラメータ
        """
        params = {
            # 短期ZMAパラメータ
            'short_fast_period': trial.suggest_int('short_fast_period', 1, 5),
            'short_slow_period': trial.suggest_int('short_slow_period', 8, 34),
            
            # 長期ZMAパラメータ
            'long_fast_period': trial.suggest_int('long_fast_period', 1, 5),
            'long_slow_period': trial.suggest_int('long_slow_period', 21, 89),
            
            # 共通パラメータ
            'src_type': trial.suggest_categorical('src_type', ['close', 'hlc3', 'hl2', 'ohlc4']),
            
            # 短期サイクルERパラメータ
            'short_detector_type': trial.suggest_categorical('short_detector_type', 
                                                         ['dudi', 'hody', 'phac', 'dudi_e', 'hody_e', 'phac_e']),
            'short_cycle_part': trial.suggest_float('short_cycle_part', 0.2, 0.9, step=0.1),
            'short_lp_period': trial.suggest_int('short_lp_period', 3, 21),
            'short_hp_period': trial.suggest_int('short_hp_period', 34, 144),
            'short_cycle_max': trial.suggest_int('short_cycle_max', 34, 144),
            'short_cycle_min': trial.suggest_int('short_cycle_min', 3, 13),
            'short_max_output': trial.suggest_int('short_max_output', 21, 89),
            'short_min_output': trial.suggest_int('short_min_output', 3, 13),
            'short_smooth_er': trial.suggest_categorical('short_smooth_er', [True, False]),
            
            # 長期サイクルERパラメータ
            'long_detector_type': trial.suggest_categorical('long_detector_type', 
                                                        ['dudi', 'hody', 'phac', 'dudi_e', 'hody_e', 'phac_e']),
            'long_cycle_part': trial.suggest_float('long_cycle_part', 0.2, 0.9, step=0.1),
            'long_lp_period': trial.suggest_int('long_lp_period', 3, 21),
            'long_hp_period': trial.suggest_int('long_hp_period', 55, 233),
            'long_cycle_max': trial.suggest_int('long_cycle_max', 89, 233),
            'long_cycle_min': trial.suggest_int('long_cycle_min', 5, 21),
            'long_max_output': trial.suggest_int('long_max_output', 34, 144),
            'long_min_output': trial.suggest_int('long_min_output', 5, 21),
            'long_smooth_er': trial.suggest_categorical('long_smooth_er', [True, False]),
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
            # 短期ZMAパラメータ
            'short_fast_period': int(params['short_fast_period']),
            'short_slow_period': int(params['short_slow_period']),
            
            # 長期ZMAパラメータ
            'long_fast_period': int(params['long_fast_period']),
            'long_slow_period': int(params['long_slow_period']),
            
            # 共通パラメータ
            'src_type': params['src_type'],
            
            # 短期サイクルERパラメータ
            'short_detector_type': params['short_detector_type'],
            'short_lp_period': int(params['short_lp_period']),
            'short_hp_period': int(params['short_hp_period']),
            'short_cycle_part': float(params['short_cycle_part']),
            'short_cycle_max': int(params['short_cycle_max']),
            'short_cycle_min': int(params['short_cycle_min']),
            'short_max_output': int(params['short_max_output']),
            'short_min_output': int(params['short_min_output']),
            'short_use_kalman_filter': False,
            'short_kalman_measurement_noise': 1.0,
            'short_kalman_process_noise': 0.01,
            'short_kalman_n_states': 5,
            'short_smooth_er': bool(params['short_smooth_er']),
            'short_er_alma_period': 5,
            'short_er_alma_offset': 0.85,
            'short_er_alma_sigma': 6,
            'short_self_adaptive': False,
            
            # 長期サイクルERパラメータ
            'long_detector_type': params['long_detector_type'],
            'long_lp_period': int(params['long_lp_period']),
            'long_hp_period': int(params['long_hp_period']),
            'long_cycle_part': float(params['long_cycle_part']),
            'long_cycle_max': int(params['long_cycle_max']),
            'long_cycle_min': int(params['long_cycle_min']),
            'long_max_output': int(params['long_max_output']),
            'long_min_output': int(params['long_min_output']),
            'long_use_kalman_filter': False,
            'long_kalman_measurement_noise': 1.0,
            'long_kalman_process_noise': 0.01,
            'long_kalman_n_states': 5,
            'long_smooth_er': bool(params['long_smooth_er']),
            'long_er_alma_period': 5,
            'long_er_alma_offset': 0.85,
            'long_er_alma_sigma': 6,
            'long_self_adaptive': False,
        }
        return strategy_params 