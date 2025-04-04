#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import SupertrendADXLongSignalGenerator

class SupertrendADXLongStrategy(BaseStrategy):
    """スーパートレンド+ADXフィルターの買い専用戦略"""
    
    def __init__(
        self,
        supertrend_period: int = 10,
        supertrend_multiplier: float = 3.0,
        adx_period: int = 14,
        adx_threshold: float = 30.0,
    ):
        """
        初期化
        
        Args:
            supertrend_period: スーパートレンドの期間
            supertrend_multiplier: スーパートレンドの乗数
            adx_period: ADXの期間
            adx_threshold: ADXのしきい値
        """
        super().__init__("SupertrendADXLong")
        
        # パラメータの保存
        self._parameters = {
            'supertrend_period': supertrend_period,
            'supertrend_multiplier': supertrend_multiplier,
            'adx_period': adx_period,
            'adx_threshold': adx_threshold,
        }
        
        # シグナル生成器の初期化
        self._signal_generator = SupertrendADXLongSignalGenerator(
            supertrend_period=supertrend_period,
            supertrend_multiplier=supertrend_multiplier,
            adx_period=adx_period,
            adx_threshold=adx_threshold,
        )
    
    def generate_entry(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エントリーシグナルを生成"""
        return self._signal_generator.get_entry_signals(data)
    
    def generate_exit(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """エグジットシグナルを生成"""
        return self._signal_generator.get_exit_signals(data, position, index)
    
    @classmethod
    def create_optimization_params(cls, trial: optuna.Trial) -> Dict[str, Any]:
        """最適化パラメータを生成"""
        return {
            'supertrend_period': trial.suggest_int('supertrend_period', 3, 100),
            'supertrend_multiplier': trial.suggest_float('supertrend_multiplier', 2.0, 7.0, step=0.5),
            'adx_period': trial.suggest_int('adx_period', 3, 34),
            'adx_threshold': 30,
        }
    
    @classmethod
    def create_grid_params(cls) -> Dict[str, list]:
        """グリッドサーチ用のパラメータを生成"""
        return {
            'supertrend_period': list(range(3, 101, 10)),  # 3から100まで10刻み
            'supertrend_multiplier': [round(x, 1) for x in np.arange(2.0, 7.5, 0.5)],  # 2.0から7.0まで0.5刻み
            'adx_period': list(range(3, 35, 3)),  # 3から34まで3刻み
            'adx_threshold': [30],  # 固定値
        }
    
    @classmethod
    def convert_params_to_strategy_format(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """最適化パラメータを戦略パラメータに変換"""
        return {
            'supertrend_period': int(params['supertrend_period']),
            'supertrend_multiplier': float(params['supertrend_multiplier']),
            'adx_period': int(params['adx_period']),
            'adx_threshold': 30,
        } 