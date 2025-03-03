#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import KAMACirculationChopDonchianLongSignalGenerator


class KAMACirculationChopDonchianLongStrategy(BaseStrategy):
    """
    KAMAの大循環、チョピネスフィルター、ドンチャンブレイクアウト戦略（買い専用）
    
    エントリー条件:
    - KAMAの大循環が買いシグナル(1)
    - チョピネスフィルターがトレンド相場(1)
    - ドンチャンブレイクアウトが売りシグナル(-1)
    
    エグジット条件:
    - KAMAの大循環が売りシグナル(-1)
    """
    
    def __init__(
        self,
        kama_short_period: int = 21,
        kama_middle_period: int = 89,
        kama_long_period: int = 233,
        chop_period: int = 14,
        chop_solid: float = 50.0,
        donchian_period: int = 20,
    ):
        """初期化"""
        super().__init__("KAMACirculationChopDonchianLong")
        
        # パラメータの設定
        self._parameters = {
            'kama_short_period': kama_short_period,
            'kama_middle_period': kama_middle_period,
            'kama_long_period': kama_long_period,
            'chop_period': chop_period,
            'chop_solid': chop_solid,
            'donchian_period': donchian_period,
        }
        
        # シグナル生成器の初期化
        self.signal_generator = KAMACirculationChopDonchianLongSignalGenerator(
            kama_short_period=kama_short_period,
            kama_middle_period=kama_middle_period,
            kama_long_period=kama_long_period,
            chop_period=chop_period,
            chop_solid=chop_solid,
            donchian_period=donchian_period,
        )
    
    def generate_entry(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エントリーシグナルを生成する"""
        return self.signal_generator.get_entry_signals(data)
    
    def generate_exit(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """エグジットシグナルを生成する"""
        return self.signal_generator.get_exit_signals(data, position, index)
    
    @classmethod
    def create_optimization_params(cls, trial: optuna.Trial) -> Dict[str, Any]:
        """最適化パラメータを生成"""
        params = {
            'kama_short_period': trial.suggest_int('kama_short_period', 2, 50),
            'kama_middle_period': trial.suggest_int('kama_middle_period', 55, 120),
            'kama_long_period': trial.suggest_int('kama_long_period', 125, 400),
            'chop_period': 55,
            'chop_solid': 50,
            'donchian_period': trial.suggest_int('donchian_period', 5, 100, step=5),
        }
        return params
    
    @classmethod
    def convert_params_to_strategy_format(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """最適化パラメータを戦略パラメータに変換"""
        return {
            'kama_short_period': int(params['kama_short_period']),
            'kama_middle_period': int(params['kama_middle_period']),
            'kama_long_period': int(params['kama_long_period']),
            'chop_period': 55,
            'chop_solid': 50,
            'donchian_period': int(params['donchian_period']),
        } 