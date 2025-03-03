#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import SupertrendDonchianLongSignalGenerator

class SupertrendDonchianLongStrategy(BaseStrategy):
    """スーパートレンド+ドンチャンの買い専用戦略"""
    
    def __init__(
        self,
        supertrend_period: int = 10,
        supertrend_multiplier: float = 3.0,
        donchian_period: int = 20,
    ):
        """
        初期化
        
        Args:
            supertrend_period: スーパートレンドの期間
            supertrend_multiplier: スーパートレンドの乗数
            donchian_period: ドンチャンの期間
        """
        super().__init__("SupertrendDonchianLong")
        
        # パラメータの保存
        self._parameters = {
            'supertrend_period': supertrend_period,
            'supertrend_multiplier': supertrend_multiplier,
            'donchian_period': donchian_period,
        }
        
        # シグナル生成器の初期化
        self._signal_generator = SupertrendDonchianLongSignalGenerator(
            supertrend_period=supertrend_period,
            supertrend_multiplier=supertrend_multiplier,
            donchian_period=donchian_period,
        )
    
    def generate_entry(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エントリーシグナルを生成"""
        return self._signal_generator.get_entry_signals(data)
    
    def generate_exit(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """エグジットシグナルを生成"""
        return self._signal_generator.get_exit_signals(data, position, index)
    
    def get_entry_price(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> float:
        """エントリー価格を取得"""
        if index == -1:
            index = len(data) - 1
        
        if isinstance(data, pd.DataFrame):
            return float(data.iloc[index]['close'])
        return float(data[index, 3])  # close price
    
    @classmethod
    def create_optimization_params(cls, trial: optuna.Trial) -> Dict[str, Any]:
        """最適化パラメータを生成"""
        return {
            'supertrend_period': trial.suggest_int('supertrend_period', 3, 120, step=1),
            'supertrend_multiplier': trial.suggest_float('supertrend_multiplier', 1.0, 7.0, step=0.5),
            'donchian_period': trial.suggest_int('donchian_period', 3, 80, step=1),
        }
    
    @classmethod
    def convert_params_to_strategy_format(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """最適化パラメータを戦略パラメータに変換"""
        return {
            'supertrend_period': int(params['supertrend_period']),
            'supertrend_multiplier': float(params['supertrend_multiplier']),
            'donchian_period': int(params['donchian_period']),
        } 