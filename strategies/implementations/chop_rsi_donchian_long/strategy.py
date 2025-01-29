#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import ChopRSIDonchianLongSignalGenerator

class ChopRSIDonchianLongStrategy(BaseStrategy):
    """CHOP+RSI+ドンチャンの買い専用戦略"""
    
    def __init__(
        self,
        chop_period: int = 14,
        chop_threshold: float = 50.0,
        rsi_period: int = 14,
        donchian_period: int = 20,
    ):
        """
        初期化
        
        Args:
            chop_period: CHOPの期間
            chop_threshold: CHOPの閾値
            rsi_period: RSIの期間
            donchian_period: ドンチャンの期間
        """
        super().__init__("ChopRSIDonchianLong")
        
        # パラメータの保存
        self._parameters = {
            'chop_period': chop_period,
            'chop_threshold': chop_threshold,
            'rsi_period': rsi_period,
            'donchian_period': donchian_period,
        }
        
        # シグナル生成器の初期化
        self._signal_generator = ChopRSIDonchianLongSignalGenerator(
            chop_period=chop_period,
            chop_threshold=chop_threshold,
            rsi_period=rsi_period,
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
            'chop_period': trial.suggest_int('chop_period', 3, 120, step=1),
            'chop_threshold': 50,
            'rsi_period': trial.suggest_int('rsi_period', 5, 30, step=1),
            'donchian_period': trial.suggest_int('donchian_period', 5, 200, step=5),
        }
    
    @classmethod
    def convert_params_to_strategy_format(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """最適化パラメータを戦略パラメータに変換"""
        return {
            'chop_period': int(params['chop_period']),
            'chop_threshold': 50,
            'rsi_period': int(params['rsi_period']),
            'donchian_period': int(params['donchian_period']),
        } 