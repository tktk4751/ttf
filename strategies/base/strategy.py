#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd
import optuna
import logging
from abc import ABC, abstractmethod

from ..interfaces.strategy import IStrategy
from ..interfaces.optimization import IOptimization

class BaseStrategy(IStrategy, IOptimization, ABC):
    """戦略の基底クラス"""
    
    def __init__(self, name: str):
        """
        コンストラクタ
        
        Args:
            name: 戦略の名前
        """
        self.name = name
        self._parameters: Dict[str, Any] = {}
        
        # ロガーの設定
        self.logger = logging.getLogger(f"strategy.{name}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
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
    
    def get_entry_price(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> float:
        """
        エントリー価格を取得する
        
        Args:
            data: 価格データ
            position: ポジション方向 (1: ロング, -1: ショート)
            index: データのインデックス（デフォルト: -1 = 最新のデータ）
            
        Returns:
            float: エントリー価格（デフォルトでは現在の終値）
        """
        if isinstance(data, pd.DataFrame):
            return data['close'].iloc[index]
        return data[index, 3]  # close価格のインデックスは3
    
    def get_stop_price(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> float:
        """
        ストップロス価格を取得する
        
        Args:
            data: 価格データ
            position: ポジション方向 (1: ロング, -1: ショート)
            index: データのインデックス（デフォルト: -1 = 最新のデータ）
            
        Returns:
            float: ストップロス価格
        """
        raise NotImplementedError

    def get_exit_price(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> float:
        """
        エグジット価格を取得する
        
        Args:
            data: 価格データ
            position: ポジション方向 (1: ロング, -1: ショート)
            index: データのインデックス（デフォルト: -1 = 最新のデータ）
            
        Returns:
            float: エグジット価格（デフォルトでは現在の終値）
        """
        if isinstance(data, pd.DataFrame):
            return data['close'].iloc[index]
        return data[index, 3]  # close価格のインデックスは3 