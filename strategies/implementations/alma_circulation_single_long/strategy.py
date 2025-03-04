#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import ALMACirculationSingleLongSignalGenerator


class ALMACirculationSingleLongStrategy(BaseStrategy):
    """
    ALMAサーキュレーション戦略（単一チャネル・買い専用）
    
    エントリー条件:
    - ALMAサーキュレーションが上昇相場を示している
    - チョピネスフィルターがレンジ相場を示している
    
    エグジット条件:
    - ALMAサーキュレーションが下降相場を示している
    """
    
    def __init__(
        self,
        short_period: int = 9,
        middle_period: int = 21,
        long_period: int = 55,
        sigma: float = 6.0,
        offset: float = 0.85,
        chop_period: int = 14,
        chop_solid: float = 50.0,
    ):
        """
        初期化
        
        Args:
            short_period: 短期ALMAの期間
            middle_period: 中期ALMAの期間
            long_period: 長期ALMAの期間
            sigma: ガウス分布の標準偏差
            offset: 重みの中心位置（0-1）
            chop_period: チョピネスインデックスの期間
            chop_solid: チョピネスインデックスのしきい値
        """
        super().__init__("ALMACirculationSingleLong")
        
        # パラメータの設定
        self._parameters = {
            'short_period': short_period,
            'middle_period': middle_period,
            'long_period': long_period,
            'sigma': sigma,
            'offset': offset,
            'chop_period': chop_period,
            'chop_solid': chop_solid,
        }
        
        # シグナル生成器の初期化
        self.signal_generator = ALMACirculationSingleLongSignalGenerator(
            short_period=short_period,
            middle_period=middle_period,
            long_period=long_period,
            sigma=sigma,
            offset=offset,
            chop_period=chop_period,
            chop_solid=chop_solid,
        )
    
    def generate_entry(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナルを生成する
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: エントリーシグナル
        """
        return self.signal_generator.get_entry_signals(data)
    
    def generate_exit(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """
        エグジットシグナルを生成する
        
        Args:
            data: 価格データ
            position: 現在のポジション（1: ロング、-1: ショート）
            index: データのインデックス（デフォルト: -1）
            
        Returns:
            bool: エグジットすべきかどうか
        """
        return self.signal_generator.get_exit_signals(data, position, index)
    
    @classmethod
    def create_optimization_params(cls, trial: optuna.Trial) -> Dict[str, Any]:
        """
        最適化パラメータを生成
        
        Args:
            trial: Optunaのトライアル
            
        Returns:
            Dict[str, Any]: 最適化パラメータ
        """
        params = {
            'short_period': trial.suggest_int('short_period', 5, 34),
            'middle_period': trial.suggest_int('middle_period', 35, 250),
            'long_period': trial.suggest_int('long_period', 130, 700),
            'sigma': 6.0,
            'offset': 0.85,
            'chop_period': 55,
            'chop_solid': 50,
        }
        return params
    
    @classmethod
    def convert_params_to_strategy_format(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        最適化パラメータを戦略パラメータに変換
        
        Args:
            params: 最適化パラメータ
            
        Returns:
            Dict[str, Any]: 戦略パラメータ
        """
        return {
            'short_period': int(params['short_period']),
            'middle_period': int(params['middle_period']),
            'long_period': int(params['long_period']),
            'sigma': 6,
            'offset': 0.85,
            'chop_period': 55,
            'chop_solid': 50,
        } 