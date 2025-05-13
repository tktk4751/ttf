#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import XTrendSignalGenerator # Import the new signal generator

class XTrendSimpleStrategy(BaseStrategy):
    """
    XTrendインジケーターに基づくシンプルな戦略

    エントリー条件:
    - ロング: XTrend方向が-1から1に転換
    - ショート: XTrend方向が1から-1に転換

    エグジット条件:
    - ロング: XTrend方向が1から-1に転換
    - ショート: XTrend方向が-1から1に転換
    """

    def __init__(
        self,
        # Pass all XTrend parameters here, matching XTrendSignalGenerator
        price_source: str = 'close',
        measurement_noise: float = 3.0,
        process_noise: float = 0.01,
        kalman_n_states: int = 5,
        catr_detector_type: str = 'hody',
        catr_cycle_part: float = 0.5,
        catr_lp_period: int = 5,
        catr_hp_period: int = 55,
        catr_max_cycle: int = 55,
        catr_min_cycle: int = 5,
        catr_max_output: int = 34,
        catr_min_output: int = 5,
        catr_smoother_type: str = 'alma',
        cer_detector_type: str = 'phac_e',
        cer_lp_period: int = 5,
        cer_hp_period: int = 55,
        cer_cycle_part: float = 0.382,
        cer_max_cycle: int = 62,
        cer_min_cycle: int = 5,
        cer_max_output: int = 34,
        cer_min_output: int = 5,
        max_max_multiplier: float = 7.0,
        min_max_multiplier: float = 3.0,
        max_min_multiplier: float = 1.5,
        min_min_multiplier: float = 0.5,
        warmup_periods: Optional[int] = None
    ):
        """初期化"""
        super().__init__("XTrendSimple")

        # Store parameters, excluding 'self' and '__class__'
        self._parameters = {k: v for k, v in locals().items() if k != 'self' and k != '__class__'}

        # Initialize the signal generator with the filtered parameters
        self.signal_generator = XTrendSignalGenerator(**self._parameters)

    def generate_entry(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナルを生成する
        """
        try:
            return self.signal_generator.get_entry_signals(data)
        except Exception as e:
            self.logger.error(f"エントリーシグナル生成中にエラー: {str(e)}")
            return np.zeros(len(data), dtype=np.int8)

    def generate_exit(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """
        エグジットシグナルを生成する
        """
        try:
            return self.signal_generator.get_exit_signals(data, position, index)
        except Exception as e:
            self.logger.error(f"エグジットシグナル生成中にエラー: {str(e)}")
            return False

    @classmethod
    def create_optimization_params(cls, trial: optuna.Trial) -> Dict[str, Any]:
        """
        最適化パラメータを生成
        Note: This should ideally match the parameter space of XTrend and its sub-components.
        Copying the structure from ZCSimpleStrategy for now, but adjust ranges as needed for XTrend.
        """
        params = {
            # XTrend Parameters
            'price_source': trial.suggest_categorical('price_source', ['close', 'hlc3', 'hl2', 'ohlc4']),
            'measurement_noise': trial.suggest_float('measurement_noise', 0.1, 10.0, log=True),
            'process_noise': trial.suggest_float('process_noise', 0.001, 0.1, log=True),
            'kalman_n_states': trial.suggest_int('kalman_n_states', 2, 10),

            # CATR Parameters
            'catr_detector_type': trial.suggest_categorical('catr_detector_type', ['dudi', 'hody', 'phac', 'dudi_e', 'hody_e', 'phac_e']),
            'catr_cycle_part': trial.suggest_float('catr_cycle_part', 0.2, 0.9, step=0.1),
            'catr_lp_period': trial.suggest_int('catr_lp_period', 3, 21),
            'catr_hp_period': trial.suggest_int('catr_hp_period', 34, 144), # Adjusted range
            'catr_max_cycle': trial.suggest_int('catr_max_cycle', 34, 144), # Adjusted range
            'catr_min_cycle': trial.suggest_int('catr_min_cycle', 3, 21),   # Adjusted range
            'catr_max_output': trial.suggest_int('catr_max_output', 21, 89), # Adjusted range
            'catr_min_output': trial.suggest_int('catr_min_output', 3, 13),  # Adjusted range
            'catr_smoother_type': trial.suggest_categorical('catr_smoother_type', ['alma', 'hyper']),

            # CER Parameters
            'cer_detector_type': trial.suggest_categorical('cer_detector_type', ['dudi', 'hody', 'phac', 'dudi_e', 'hody_e', 'phac_e']),
            'cer_lp_period': trial.suggest_int('cer_lp_period', 3, 21),
            'cer_hp_period': trial.suggest_int('cer_hp_period', 34, 144), # Adjusted range
            'cer_cycle_part': trial.suggest_float('cer_cycle_part', 0.2, 0.9, step=0.1),
            'cer_max_cycle': trial.suggest_int('cer_max_cycle', 34, 144), # Adjusted range
            'cer_min_cycle': trial.suggest_int('cer_min_cycle', 3, 21),   # Adjusted range
            'cer_max_output': trial.suggest_int('cer_max_output', 21, 89), # Adjusted range
            'cer_min_output': trial.suggest_int('cer_min_output', 3, 13),  # Adjusted range

            # Dynamic Multiplier Parameters
            'max_max_multiplier': trial.suggest_float('max_max_multiplier', 5.0, 12.0, step=0.5), # Wider range?
            'min_max_multiplier': trial.suggest_float('min_max_multiplier', 2.0, 7.0, step=0.5),  # Wider range?
            'max_min_multiplier': trial.suggest_float('max_min_multiplier', 1.0, 3.0, step=0.1),  # Wider range?
            'min_min_multiplier': trial.suggest_float('min_min_multiplier', 0.1, 1.5, step=0.1),  # Wider range?

            # Warmup periods might not be directly optimizable, but derived
        }
        # Ensure min < max constraints if necessary (Optuna usually handles this if defined correctly)
        if params['min_max_multiplier'] > params['max_max_multiplier']:
            params['min_max_multiplier'] = params['max_max_multiplier']
        if params['min_min_multiplier'] > params['max_min_multiplier']:
            params['min_min_multiplier'] = params['max_min_multiplier']

        return params

    @classmethod
    def convert_params_to_strategy_format(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        最適化パラメータを戦略パラメータに変換 (型変換)
        """
        strategy_params = params.copy() # Start with a copy
        # Ensure correct types (Optuna might return floats for ints sometimes)
        int_keys = [
            'kalman_n_states', 'catr_lp_period', 'catr_hp_period', 'catr_max_cycle',
            'catr_min_cycle', 'catr_max_output', 'catr_min_output', 'cer_lp_period',
            'cer_hp_period', 'cer_max_cycle', 'cer_min_cycle', 'cer_max_output',
            'cer_min_output'
        ]
        float_keys = [
            'measurement_noise', 'process_noise', 'catr_cycle_part', 'cer_cycle_part',
            'max_max_multiplier', 'min_max_multiplier', 'max_min_multiplier',
            'min_min_multiplier'
        ]

        for key in int_keys:
            if key in strategy_params:
                strategy_params[key] = int(strategy_params[key])
        for key in float_keys:
             if key in strategy_params:
                 strategy_params[key] = float(strategy_params[key])

        # Remove warmup_periods if it was added during optimization, as it's calculated internally
        strategy_params.pop('warmup_periods', None)

        return strategy_params
