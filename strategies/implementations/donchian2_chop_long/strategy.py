#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import Donchian2ChopLongSignalGenerator


class Donchian2ChopLongStrategy(BaseStrategy):
    """
    2つのドンチャン+チョピネスフィルター戦略（買い専用）
    
    エントリー条件:
    - エントリー用ドンチャンチャネルのアッパーブレイクアウトで買いシグナル
    - チョピネスインデックスがトレンド相場を示している
    
    エグジット条件:
    - エグジット用ドンチャンチャネルの売りシグナル
    """
    
    def __init__(
        self,
        entry_period: int = 20,
        exit_period: int = 20,
        chop_period: int = 14,
        chop_threshold: float = 50.0,
    ):
        """
        初期化
        
        Args:
            entry_period: エントリー用ドンチャンチャネルの期間
            exit_period: エグジット用ドンチャンチャネルの期間
            chop_period: チョピネスインデックスの期間
            chop_threshold: チョピネスインデックスのしきい値
        """
        super().__init__("Donchian2ChopLong")
        
        # パラメータの設定
        self._parameters = {
            'entry_period': entry_period,
            'exit_period': exit_period,
            'chop_period': chop_period,
            'chop_threshold': chop_threshold,
        }
        
        # シグナル生成器の初期化
        self.signal_generator = Donchian2ChopLongSignalGenerator(
            entry_period=entry_period,
            exit_period=exit_period,
            chop_period=chop_period,
            chop_threshold=chop_threshold,
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
            'entry_period': trial.suggest_int('entry_period', 5, 400),
            'exit_period': trial.suggest_int('exit_period', 5, 400),
            'chop_period': trial.suggest_int('chop_period', 5, 150),
            'chop_threshold': 50,
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
            'entry_period': int(params['entry_period']),
            'exit_period': int(params['exit_period']),
            'chop_period': int(params['chop_period']),
            'chop_threshold': 50,
        } 