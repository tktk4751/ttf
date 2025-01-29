#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import MFIDivROCShortSignalGenerator

class MFIDivROCShortStrategy(BaseStrategy):
    """MFIダイバージェンス+ROCの売り専用戦略"""
    
    def __init__(
        self,
        div_lookback: int = 30,
        roc_period: int = 21,
        entry_mfi_period: int = 14,
        exit_mfi_period: int = 14,
        entry_lookback: int = 13,
    ):
        """
        初期化
        
        Args:
            div_lookback: ダイバージェンス検出のルックバック期間
            roc_period: ROCの期間
            entry_mfi_period: エントリー用MFIの期間
            exit_mfi_period: エグジット用MFIの期間
            entry_lookback: エントリー判定のルックバック期間
        """
        super().__init__("MFIDivROCShort")
        
        # パラメータの保存
        self._parameters = {
            'div_lookback': div_lookback,
            'roc_period': roc_period,
            'entry_mfi_period': entry_mfi_period,
            'exit_mfi_period': exit_mfi_period,
            'entry_lookback': entry_lookback,
        }
        
        # シグナル生成器の初期化
        self._signal_generator = MFIDivROCShortSignalGenerator(
            div_lookback=div_lookback,
            roc_period=roc_period,
            entry_mfi_period=entry_mfi_period,
            exit_mfi_period=exit_mfi_period,
            entry_lookback=entry_lookback,
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
            'div_lookback': trial.suggest_int('div_lookback', 20, 100, step=10),
            'roc_period': trial.suggest_int('roc_period', 10, 300, step=5),
            'entry_mfi_period': trial.suggest_int('entry_mfi_period', 10, 34),
            'exit_mfi_period': trial.suggest_int('exit_mfi_period', 10, 34),
            'entry_lookback': 13,
        }
    
    @classmethod
    def convert_params_to_strategy_format(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """最適化パラメータを戦略パラメータに変換"""
        return {
            'div_lookback': int(params['div_lookback']),
            'roc_period': int(params['roc_period']),
            'entry_mfi_period': int(params['entry_mfi_period']),
            'exit_mfi_period': int(params['exit_mfi_period']),
            'entry_lookback': 13,
        } 