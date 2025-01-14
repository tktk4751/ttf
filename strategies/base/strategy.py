#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd
import optuna

from ..interfaces.strategy import IStrategy
from ..interfaces.optimization import IOptimization

class BaseStrategy(IStrategy, IOptimization):
    """戦略の基底クラス"""
    
    def __init__(self, name: str):
        """
        コンストラクタ
        
        Args:
            name: 戦略の名前
        """
        self.name = name
        self._parameters: Dict[str, Any] = {}
    
    def get_parameters(self) -> Dict[str, Any]:
        """現在のパラメータを取得する"""
        return self._parameters.copy()
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """パラメータを設定する"""
        self._parameters.update(params)
    
    def generate_entry(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エントリーシグナルを生成する"""
        raise NotImplementedError
    
    def generate_exit(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """エグジットシグナルを生成する"""
        raise NotImplementedError
    
    @classmethod
    def create_optimization_params(cls, trial: optuna.Trial) -> Dict[str, Any]:
        """最適化パラメータを生成する"""
        raise NotImplementedError
    
    @classmethod
    def convert_params_to_strategy_format(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """最適化パラメータを戦略パラメータに変換する"""
        raise NotImplementedError 