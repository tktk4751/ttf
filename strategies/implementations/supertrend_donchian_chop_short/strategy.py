#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import SupertrendDonchianChopShortSignalGenerator

class SupertrendDonchianChopShortStrategy(BaseStrategy):
    """スーパートレンド+ドンチャン+CHOPのショート戦略"""
    
    def __init__(
        self,
        supertrend_period: int = 10,
        supertrend_multiplier: float = 3.0,
        donchian_period: int = 20,
        chop_period: int = 14,
        chop_threshold: float = 50.0,
        rsi_period: int = 14,
        rsi_lower: float = 30.0,
    ):
        """
        初期化
        
        Args:
            supertrend_period: スーパートレンドの期間
            supertrend_multiplier: スーパートレンドの乗数
            donchian_period: ドンチャンの期間
            chop_period: CHOPの期間
            chop_threshold: CHOPのしきい値
            rsi_period: RSIの期間
            rsi_lower: RSIの下限しきい値
        """
        super().__init__("SupertrendDonchianChopShort")
        
        # パラメータの保存
        self._parameters = {
            'supertrend_period': supertrend_period,
            'supertrend_multiplier': supertrend_multiplier,
            'donchian_period': donchian_period,
            'chop_period': chop_period,
            'chop_threshold': chop_threshold,
            'rsi_period': rsi_period,
            'rsi_lower': rsi_lower,
        }
        
        # シグナル生成器の初期化
        self._signal_generator = SupertrendDonchianChopShortSignalGenerator(
            supertrend_period=supertrend_period,
            supertrend_multiplier=supertrend_multiplier,
            donchian_period=donchian_period,
            chop_period=chop_period,
            chop_threshold=chop_threshold,
            rsi_period=rsi_period,
            rsi_lower=rsi_lower,
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
            'supertrend_period': trial.suggest_int('supertrend_period', 5, 30),
            'supertrend_multiplier': trial.suggest_float('supertrend_multiplier', 1.0, 5.0),
            'donchian_period': trial.suggest_int('donchian_period', 10, 100),
            'chop_period': trial.suggest_int('chop_period', 5, 30),
            'chop_threshold': trial.suggest_float('chop_threshold', 40.0, 60.0),
            'rsi_period': trial.suggest_int('rsi_period', 5, 30),
            'rsi_lower': trial.suggest_float('rsi_lower', 15.0, 35.0),
        }
    
    @classmethod
    def convert_params_to_strategy_format(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """最適化パラメータを戦略パラメータに変換"""
        return {
            'supertrend_period': int(params['supertrend_period']),
            'supertrend_multiplier': float(params['supertrend_multiplier']),
            'donchian_period': int(params['donchian_period']),
            'chop_period': int(params['chop_period']),
            'chop_threshold': float(params['chop_threshold']),
            'rsi_period': int(params['rsi_period']),
            'rsi_lower': float(params['rsi_lower']),
        } 