#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import ADXRSIDonchianShortSignalGenerator

class ADXRSIDonchianShortStrategy(BaseStrategy):
    """ADX+RSI+ドンチャンの売り専用戦略"""
    
    def __init__(
        self,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        rsi_period: int = 14,
        rsi_upper: float = 70.0,
        rsi_lower: float = 30.0,
        donchian_period: int = 20,
    ):
        """
        初期化
        
        Args:
            adx_period: ADXの期間
            adx_threshold: ADXの閾値
            rsi_period: RSIの期間
            rsi_upper: RSIの上限値
            rsi_lower: RSIの下限値
            donchian_period: ドンチャンの期間
        """
        super().__init__("ADXRSIDonchianShort")
        
        # パラメータの保存
        self._parameters = {
            'adx_period': adx_period,
            'adx_threshold': adx_threshold,
            'rsi_period': rsi_period,
            'rsi_upper': rsi_upper,
            'rsi_lower': rsi_lower,
            'donchian_period': donchian_period,
        }
        
        # シグナル生成器の初期化
        self._signal_generator = ADXRSIDonchianShortSignalGenerator(
            adx_period=adx_period,
            adx_threshold=adx_threshold,
            rsi_period=rsi_period,
            rsi_upper=rsi_upper,
            rsi_lower=rsi_lower,
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
            'adx_period': trial.suggest_int('adx_period', 5, 30, step=1),
            'adx_threshold': trial.suggest_float('adx_threshold', 15.0, 35.0, step=1.0),
            'rsi_period': trial.suggest_int('rsi_period', 5, 30, step=1),
            'rsi_upper': trial.suggest_float('rsi_upper', 60.0, 85.0, step=1.0),
            'rsi_lower': trial.suggest_float('rsi_lower', 15.0, 40.0, step=1.0),
            'donchian_period': trial.suggest_int('donchian_period', 5, 100, step=1),
        }
    
    @classmethod
    def convert_params_to_strategy_format(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """最適化パラメータを戦略パラメータに変換"""
        return {
            'adx_period': int(params['adx_period']),
            'adx_threshold': float(params['adx_threshold']),
            'rsi_period': int(params['rsi_period']),
            'rsi_upper': float(params['rsi_upper']),
            'rsi_lower': float(params['rsi_lower']),
            'donchian_period': int(params['donchian_period']),
        } 