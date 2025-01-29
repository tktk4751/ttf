#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import SqueezeChopShortSignalGenerator

class SqueezeChopShortStrategy(BaseStrategy):
    """スクイーズモメンタム+チョピネスフィルターの売り専用戦略"""
    
    def __init__(
        self,
        bb_length: int = 20,
        bb_mult: float = 2.0,
        kc_length: int = 20,
        kc_mult: float = 1.5,
        chop_period: int = 14,
        chop_threshold: float = 50.0,
    ):
        """
        初期化
        
        Args:
            bb_length: Bollinger Bandsの期間
            bb_mult: Bollinger Bandsの乗数
            kc_length: Keltner Channelsの期間
            kc_mult: Keltner Channelsの乗数
            chop_period: チョピネスインデックスの期間
            chop_threshold: チョピネスインデックスのしきい値
        """
        super().__init__("SqueezeChopShort")
        
        # パラメータの保存
        self._parameters = {
            'bb_length': bb_length,
            'bb_mult': bb_mult,
            'kc_length': kc_length,
            'kc_mult': kc_mult,
            'chop_period': chop_period,
            'chop_threshold': chop_threshold,
        }
        
        # シグナル生成器の初期化
        self._signal_generator = SqueezeChopShortSignalGenerator(
            bb_length=bb_length,
            bb_mult=bb_mult,
            kc_length=kc_length,
            kc_mult=kc_mult,
            chop_period=chop_period,
            chop_threshold=chop_threshold,
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
            'bb_length': trial.suggest_int('bb_length', 3, 100),
            'bb_mult': trial.suggest_float('bb_mult', 1.0, 3.0, step=0.5),
            'kc_length': trial.suggest_int('kc_length', 3, 100),
            'kc_mult': trial.suggest_float('kc_mult', 1.0, 3.0, step=0.5),
            'chop_period': trial.suggest_int('chop_period', 3, 100),
            'chop_threshold': 50,
        }
    
    @classmethod
    def convert_params_to_strategy_format(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """最適化パラメータを戦略パラメータに変換"""
        return {
            'bb_length': int(params['bb_length']),
            'bb_mult': float(params['bb_mult']),
            'kc_length': int(params['kc_length']),
            'kc_mult': float(params['kc_mult']),
            'chop_period': int(params['chop_period']),
            'chop_threshold': 50,
        } 