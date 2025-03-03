#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import MACDDivergenceLongSignalGenerator


class MACDDivergenceLongStrategy(BaseStrategy):
    """
    MACDダイバージェンス戦略（買い専用）
    
    エントリー条件:
    - MACDダイバージェンスが買いシグナル(1)
    - ADXフィルターがトレンド相場(1)（期間13、閾値35）
    
    エグジット条件:
    - ボリンジャーバンドブレイクアウトが買いエグジット(1)
    - MACDダイバージェンスが売りシグナル(-1)
    """
    
    def __init__(
        self,
        macd_fast_period: int = 12,
        macd_slow_period: int = 26,
        macd_signal_period: int = 9,
        macd_lookback: int = 30,
        bb_period: int = 21,
        bb_num_std: float = 3.0,
    ):
        """初期化"""
        super().__init__("MACDDivergenceLong")
        
        # パラメータの設定
        self._parameters = {
            'macd_fast_period': macd_fast_period,
            'macd_slow_period': macd_slow_period,
            'macd_signal_period': macd_signal_period,
            'macd_lookback': macd_lookback,
            'bb_period': bb_period,
            'bb_num_std': bb_num_std,
        }
        
        # シグナル生成器の初期化
        self.signal_generator = MACDDivergenceLongSignalGenerator(
            macd_fast_period=macd_fast_period,
            macd_slow_period=macd_slow_period,
            macd_signal_period=macd_signal_period,
            macd_lookback=macd_lookback,
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
            'macd_fast_period': trial.suggest_int('macd_fast_period', 3, 20),
            'macd_slow_period': trial.suggest_int('macd_slow_period', 21, 62),
            'macd_signal_period': 8,
            'macd_lookback': trial.suggest_int('macd_lookback', 20, 200, step=10),
            'bb_period': trial.suggest_int('bb_period', 15, 55),
            'bb_num_std': trial.suggest_float('bb_num_std', 1.0, 3.0, step=0.5),
        }
        return params
    
    @classmethod
    def convert_params_to_strategy_format(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """最適化パラメータを戦略パラメータに変換"""
        return {
            'macd_fast_period': int(params['macd_fast_period']),
            'macd_slow_period': int(params['macd_slow_period']),
            'macd_signal_period': 8,
            'macd_lookback': int(params['macd_lookback']),
            'bb_period': int(params['bb_period']),
            'bb_num_std': float(params['bb_num_std']),
        } 