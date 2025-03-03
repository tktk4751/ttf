#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any
import numpy as np
import pandas as pd

from ...base.strategy import BaseStrategy
from .signal_generator import Keltner2LongSignalGenerator


class Keltner2LongStrategy(BaseStrategy):
    """
    ケルトナーロング戦略クラス
    
    エントリー条件:
    - エントリー用ケルトナーチャネルのアッパーブレイクアウトで買いシグナル
    
    エグジット条件:
    - エグジット用ケルトナーチャネルの売りシグナル
    """
    
    def __init__(
        self,
        entry_period: int = 20,
        entry_atr_period: int = 10,
        entry_multiplier: float = 2.0,
        exit_period: int = 20,
        exit_atr_period: int = 10,
        exit_multiplier: float = 2.0,
    ):
        """初期化"""
        super().__init__("Keltner2LongStrategy")
        
        # パラメータの保存
        self._parameters = {
            'entry_period': entry_period,
            'entry_atr_period': entry_atr_period,
            'entry_multiplier': entry_multiplier,
            'exit_period': exit_period,
            'exit_atr_period': exit_atr_period,
            'exit_multiplier': exit_multiplier,
        }
        
        # シグナル生成器の初期化
        self.signal_generator = Keltner2LongSignalGenerator(
            entry_period=entry_period,
            entry_atr_period=entry_atr_period,
            entry_multiplier=entry_multiplier,
            exit_period=exit_period,
            exit_atr_period=exit_atr_period,
            exit_multiplier=exit_multiplier,
        )
    
    def generate_entry(self, data: pd.DataFrame) -> np.ndarray:
        """エントリーシグナル生成"""
        return self.signal_generator.get_entry_signals(data)
    
    def generate_exit(self, data: pd.DataFrame, position: int, index: int = -1) -> bool:
        """エグジットシグナル生成"""
        return self.signal_generator.get_exit_signals(data, position, index)
    
    @classmethod
    def create_optimization_params(cls, trial) -> Dict[str, Any]:
        """最適化パラメータの作成"""
        return {
            'entry_period': trial.suggest_int('entry_period', 5, 200),
            'entry_atr_period': trial.suggest_int('entry_atr_period', 5, 120),
            'entry_multiplier': trial.suggest_float('entry_multiplier', 1.0, 5.0, step=0.5),
            'exit_period': trial.suggest_int('exit_period', 5, 200),
            'exit_atr_period': trial.suggest_int('exit_atr_period', 5, 120),
            'exit_multiplier': trial.suggest_float('exit_multiplier',  1.0, 5.0, step=0.5),
        }
    
    @classmethod
    def convert_params_to_strategy_format(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """パラメータを戦略形式に変換"""
        return {
            'entry_period': int(params['entry_period']),
            'entry_atr_period': int(params['entry_atr_period']),
            'entry_multiplier': float(params['entry_multiplier']),
            'exit_period': int(params['exit_period']),
            'exit_atr_period': int(params['exit_atr_period']),
            'exit_multiplier': float(params['exit_multiplier']),
        } 