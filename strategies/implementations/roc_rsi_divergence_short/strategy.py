#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import ROCDivergenceShortSignalGenerator


class ROCDivergenceShortStrategy(BaseStrategy):
    """
    ROCダイバージェンス戦略（売り専用）
    
    エントリー条件:
    - ROCダイバージェンスが売りシグナル(-1)
    - ADXフィルターがトレンド相場(1)（期間13、閾値35）
    
    エグジット条件:
    - ボリンジャーバンドブレイクアウトが売りエグジット(-1)
    - ROCダイバージェンスが買いシグナル(1)
    """
    
    def __init__(
        self,
        roc_period: int = 12,
        roc_lookback: int = 30,
        bb_period: int = 21,
        bb_num_std: float = 3.0,
    ):
        """初期化"""
        super().__init__("ROCDivergenceShort")
        
        # パラメータの設定
        self._parameters = {
            'roc_period': roc_period,
            'roc_lookback': roc_lookback,
            'bb_period': bb_period,
            'bb_num_std': bb_num_std,
        }
        
        # シグナル生成器の初期化
        self.signal_generator = ROCDivergenceShortSignalGenerator(
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
            'roc_period': trial.suggest_int('roc_period', 8, 20),
            'roc_lookback': trial.suggest_int('roc_lookback', 20, 40),
            'bb_period': trial.suggest_int('bb_period', 15, 30),
            'bb_num_std': trial.suggest_float('bb_num_std', 2.0, 4.0, step=0.1),
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