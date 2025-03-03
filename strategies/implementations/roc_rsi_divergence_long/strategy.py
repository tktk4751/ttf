#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import ROCDivergenceLongSignalGenerator


class ROCDivergenceLongStrategy(BaseStrategy):
    """
    ROCダイバージェンス戦略（買い専用）
    
    エントリー条件:
    - ROCダイバージェンスが買いシグナル(1)
    - ADXフィルターがトレンド相場(1)（期間13、閾値35）
    
    エグジット条件:
    - ボリンジャーバンドブレイクアウトが買いエグジット(1)
    - ROCダイバージェンスが売りシグナル(-1)
    """
    
    def __init__(
        self,
        roc_period: int = 12,
        roc_lookback: int = 30,
        bb_period: int = 21,
        bb_num_std: float = 3.0,
    ):
        """初期化"""
        super().__init__("ROCDivergenceLong")
        
        # パラメータの設定
        self._parameters = {
            'roc_period': roc_period,
            'roc_lookback': roc_lookback,
            'bb_period': bb_period,
            'bb_num_std': bb_num_std,
        }
        
        # シグナル生成器の初期化
        self.signal_generator = ROCDivergenceLongSignalGenerator(
            roc_period=roc_period,
            roc_lookback=roc_lookback,
            bb_period=bb_period,
            bb_num_std=bb_num_std,
        )
    
    def generate_entry(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エントリーシグナルを生成する"""
        return self.signal_generator.get_entry_signals(data)
    
    def generate_exit(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """エグジットシグナルを生成する"""
        return self.signal_generator.get_exit_signals(data, position, index)
    
    @classmethod
    def create_optimization_params(cls, trial: optuna.Trial) -> Dict[str, Any]:
        """最適化パラメータを生成"""
        params = {
            'roc_period': trial.suggest_int('roc_period', 8, 233),
            'roc_lookback': trial.suggest_int('roc_lookback', 20, 200, step=10),
            'bb_period': trial.suggest_int('bb_period', 15, 30),
            'bb_num_std': trial.suggest_float('bb_num_std', -2.0, 3.0, step=0.5),
        }
        return params
    
    @classmethod
    def convert_params_to_strategy_format(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """最適化パラメータを戦略パラメータに変換"""
        return {
            'roc_period': int(params['roc_period']),
            'roc_lookback': int(params['roc_lookback']),
            'bb_period': int(params['bb_period']),
            'bb_num_std': float(params['bb_num_std']),
        } 