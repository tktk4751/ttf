#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import Keltner2ADXLongSignalGenerator


class Keltner2ADXLongStrategy(BaseStrategy):
    """
    2つのケルトナーチャネル+ADXを使用したロング戦略
    
    エントリー条件:
    - エントリー用ケルトナーチャネルのアッパーブレイクアウトで買いシグナル
    - ADXがトレンド相場を示している
    
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
        adx_period: int = 14,
        adx_threshold: float = 30.0,
    ):
        """初期化"""
        super().__init__("Keltner2ADXLongStrategy")
        
        # パラメータの設定
        self._parameters = {
            'entry_period': entry_period,
            'entry_atr_period': entry_atr_period,
            'entry_multiplier': entry_multiplier,
            'exit_period': exit_period,
            'exit_atr_period': exit_atr_period,
            'exit_multiplier': exit_multiplier,
            'adx_period': adx_period,
            'adx_threshold': adx_threshold,
        }
        
        # シグナル生成器の初期化
        self.signal_generator = Keltner2ADXLongSignalGenerator(
            entry_period=entry_period,
            entry_atr_period=entry_atr_period,
            entry_multiplier=entry_multiplier,
            exit_period=exit_period,
            exit_atr_period=exit_atr_period,
            exit_multiplier=exit_multiplier,
            adx_period=adx_period,
            adx_threshold=adx_threshold
        )
    
    def generate_entry(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナルを生成する
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: エントリーシグナル
        """
        signals = self.signal_generator.get_entry_signals(data)
        return signals
    
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
        """最適化パラメータの作成"""
        return {
            'entry_period': trial.suggest_int('entry_period', 10, 200),
            'entry_atr_period': trial.suggest_int('entry_atr_period', 5, 120),
            'entry_multiplier': trial.suggest_float('entry_multiplier', 1.0, 5.0, step=0.1),
            'exit_period': trial.suggest_int('exit_period', 10, 200),
            'exit_atr_period': trial.suggest_int('exit_atr_period', 5, 120),
            'exit_multiplier': trial.suggest_float('exit_multiplier', 1.0, 5.0, step=0.1),
            'adx_period': trial.suggest_int('adx_period', 3, 34),
            'adx_threshold': trial.suggest_float('adx_threshold', 20.0, 50.0, step=5.0)
        }
    
    @classmethod
    def convert_params_to_strategy_format(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """パラメータをストラテジーのフォーマットに変換"""
        return {
            'entry_period': int(params['entry_period']),
            'entry_atr_period': int(params['entry_atr_period']),
            'entry_multiplier': float(params['entry_multiplier']),
            'exit_period': int(params['exit_period']),
            'exit_atr_period': int(params['exit_atr_period']),
            'exit_multiplier': float(params['exit_multiplier']),
            'adx_period': int(params['adx_period']),
            'adx_threshold': float(params['adx_threshold'])
        } 