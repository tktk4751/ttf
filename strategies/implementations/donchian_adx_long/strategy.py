#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import DonchianADXLongSignalGenerator

class DonchianADXLongStrategy(BaseStrategy):
    """ドンチャン+ADX+ALMAの買い専用戦略"""
    
    def __init__(
        self,
        donchian_period: int = 20,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        alma_period: int = 200,
    ):
        """
        初期化
        
        Args:
            donchian_period: ドンチャンチャネルの期間
            adx_period: ADXの期間
            adx_threshold: ADXの閾値
            alma_period: ALMAの期間
        """
        super().__init__("DonchianADXLong")
        
        # パラメータの保存
        self._parameters = {
            'donchian_period': donchian_period,
            'adx_period': adx_period,
            'adx_threshold': adx_threshold,
            'alma_period': alma_period,
        }
        
        # シグナル生成器の初期化
        self._signal_generator = DonchianADXLongSignalGenerator(
            donchian_period=donchian_period,
            adx_period=adx_period,
            adx_threshold=adx_threshold,
            alma_period=alma_period,
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
            'donchian_period': trial.suggest_int('donchian_period', 2, 250, step=2),
            'adx_period': trial.suggest_int('adx_period', 3, 34, step=1),
            'adx_threshold': 30,
            'alma_period': trial.suggest_int('alma_period', 20, 600, step=5),
        }
    
    @classmethod
    def convert_params_to_strategy_format(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """最適化パラメータを戦略パラメータに変換"""
        return {
            'donchian_period': int(params['donchian_period']),
            'adx_period': int(params['adx_period']),
            'adx_threshold': 30,
            'alma_period': int(params['alma_period']),
        } 