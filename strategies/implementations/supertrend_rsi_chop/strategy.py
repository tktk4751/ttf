#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import SupertrendRsiChopSignalGenerator

class SupertrendRsiChopStrategy(BaseStrategy):
    """スーパートレンド、RSI、Chopを組み合わせた戦略"""
    
    def __init__(
        self,
        supertrend_params: Dict[str, Any] = None,
        rsi_entry_params: Dict[str, Any] = None,
        rsi_exit_params: Dict[str, Any] = None,
        chop_params: Dict[str, Any] = None
    ):
        """
        コンストラクタ
        
        Args:
            supertrend_params: スーパートレンドのパラメータ
            rsi_entry_params: RSIエントリーのパラメータ
            rsi_exit_params: RSIエグジットのパラメータ
            chop_params: チョピネスインデックスのパラメータ
        """
        super().__init__("SupertrendRsiChop")
        
        # デフォルトパラメータ
        self._parameters = {
            'supertrend': supertrend_params or {
                'period': 10,
                'multiplier': 3.0
            },
            'rsi_entry': rsi_entry_params or {
                'period': 2,
                'solid': {
                    'rsi_long_entry': 20,
                    'rsi_short_entry': 80
                }
            },
            'rsi_exit': rsi_exit_params or {
                'period': 14,
                'solid': {
                    'rsi_long_exit_solid': 70,
                    'rsi_short_exit_solid': 30
                }
            },
            'chop': chop_params or {
                'period': 14,
                'solid': {
                    'chop_solid': 50
                }
            }
        }
        
        # シグナル生成器の初期化
        self.signal_generator = SupertrendRsiChopSignalGenerator(
            supertrend_params=self._parameters['supertrend'],
            rsi_entry_params=self._parameters['rsi_entry'],
            rsi_exit_params=self._parameters['rsi_exit'],
            chop_params=self._parameters['chop']
        )
    
    def generate_entry(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エントリーシグナルを生成する"""
        return self.signal_generator.get_entry_signals(data)
    
    def generate_exit(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """エグジットシグナルを生成する"""
        return self.signal_generator.get_exit_signals(data, position, index)
    
    @classmethod
    def create_optimization_params(cls, trial: optuna.Trial) -> Dict[str, Any]:
        """最適化パラメータを生成する"""
        params = {
            'supertrend_period': trial.suggest_int('supertrend_period', 5, 30),
            'supertrend_multiplier': trial.suggest_float('supertrend_multiplier', 1.0, 5.0),
            'rsi_entry_period': trial.suggest_int('rsi_entry_period', 2, 10),
            'rsi_long_entry': trial.suggest_int('rsi_long_entry', 10, 40),
            'rsi_short_entry': trial.suggest_int('rsi_short_entry', 60, 90),
            'rsi_exit_period': trial.suggest_int('rsi_exit_period', 5, 20),
            'rsi_long_exit': trial.suggest_int('rsi_long_exit', 60, 90),
            'rsi_short_exit': trial.suggest_int('rsi_short_exit', 10, 40),
            'chop_period': trial.suggest_int('chop_period', 5, 30),
            'chop_solid': trial.suggest_int('chop_solid', 30, 70)
        }
        return params
    
    @classmethod
    def convert_params_to_strategy_format(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """最適化パラメータを戦略パラメータに変換する"""
        return {
            'supertrend_params': {
                'period': params['supertrend_period'],
                'multiplier': params['supertrend_multiplier']
            },
            'rsi_entry_params': {
                'period': params['rsi_entry_period'],
                'solid': {
                    'rsi_long_entry': params['rsi_long_entry'],
                    'rsi_short_entry': params['rsi_short_entry']
                }
            },
            'rsi_exit_params': {
                'period': params['rsi_exit_period'],
                'solid': {
                    'rsi_long_exit_solid': params['rsi_long_exit'],
                    'rsi_short_exit_solid': params['rsi_short_exit']
                }
            },
            'chop_params': {
                'period': params['chop_period'],
                'solid': {
                    'chop_solid': params['chop_solid']
                }
            }
        } 