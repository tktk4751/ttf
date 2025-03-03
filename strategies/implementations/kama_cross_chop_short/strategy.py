#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import KAMACrossChopShortSignalGenerator


class KAMACrossChopShortStrategy(BaseStrategy):
    """
    KAMAクロスオーバーとチョピネスフィルター戦略（売り専用）
    
    エントリー条件:
    - KAMAクロスオーバーが売りシグナル(-1)
    - チョピネスフィルターがトレンド相場(1)
    
    エグジット条件:
    - KAMAクロスオーバーが買いシグナル(1)
    """
    
    def __init__(
        self,
        kama_short_period: int = 9,
        kama_long_period: int = 21,
        kama_period: int = 10,
        kama_fastest: int = 2,
        kama_slowest: int = 30,
        chop_period: int = 14,
        chop_solid: float = 50.0,
    ):
        """初期化"""
        super().__init__("KAMACrossChopShort")
        
        # パラメータの設定
        self._parameters = {
            'kama_short_period': kama_short_period,
            'kama_long_period': kama_long_period,
            'kama_period': kama_period,
            'kama_fastest': kama_fastest,
            'kama_slowest': kama_slowest,
            'chop_period': chop_period,
            'chop_solid': chop_solid,
        }
        
        # シグナル生成器の初期化
        self.signal_generator = KAMACrossChopShortSignalGenerator(
            kama_short_period=kama_short_period,
            kama_long_period=kama_long_period,
            kama_period=kama_period,
            kama_fastest=kama_fastest,
            kama_slowest=kama_slowest,
            chop_period=chop_period,
            chop_solid=chop_solid,
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
            'kama_short_period': trial.suggest_int('kama_short_period', 5, 20),
            'kama_long_period': trial.suggest_int('kama_long_period', 20, 50),
            'kama_period': trial.suggest_int('kama_period', 5, 20),
            'kama_fastest': 2,
            'kama_slowest': trial.suggest_int('kama_slowest', 20, 50),
            'chop_period': trial.suggest_int('chop_period', 10, 30),
            'chop_solid': trial.suggest_float('chop_solid', 40.0, 60.0, step=1.0),
        }
        return params
    
    @classmethod
    def convert_params_to_strategy_format(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """最適化パラメータを戦略パラメータに変換"""
        return {
            'kama_short_period': int(params['kama_short_period']),
            'kama_long_period': int(params['kama_long_period']),
            'kama_period': int(params['kama_period']),
            'kama_fastest': 2,
            'kama_slowest': int(params['kama_slowest']),
            'chop_period': int(params['chop_period']),
            'chop_solid': float(params['chop_solid']),
        } 