#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import Supertrend2ChopShortSignalGenerator


class Supertrend2ChopShortStrategy(BaseStrategy):
    """
    スーパートレンド+チョピネスフィルター戦略（売り専用）
    
    エントリー条件:
    - エントリー用スーパートレンドが下降トレンド
    - チョピネスインデックスがトレンド相場を示している
    
    エグジット条件:
    - エグジット用スーパートレンドが上昇トレンド
    """
    
    def __init__(
        self,
        entry_period: int = 10,
        entry_multiplier: float = 3.0,
        exit_period: int = 10,
        exit_multiplier: float = 3.0,
        chop_period: int = 14,
        chop_threshold: float = 50.0,
    ):
        """
        初期化
        
        Args:
            entry_period: エントリー用スーパートレンドの期間
            entry_multiplier: エントリー用スーパートレンドの乗数
            exit_period: エグジット用スーパートレンドの期間
            exit_multiplier: エグジット用スーパートレンドの乗数
            chop_period: チョピネスインデックスの期間
            chop_threshold: チョピネスインデックスの閾値
        """
        super().__init__("Supertrend2ChopShort")
        
        # パラメータの設定
        self._parameters = {
            'entry_period': entry_period,
            'entry_multiplier': entry_multiplier,
            'exit_period': exit_period,
            'exit_multiplier': exit_multiplier,
            'chop_period': chop_period,
            'chop_threshold': chop_threshold
        }
        
        # シグナル生成器の初期化
        self.signal_generator = Supertrend2ChopShortSignalGenerator(
            entry_period=entry_period,
            entry_multiplier=entry_multiplier,
            exit_period=exit_period,
            exit_multiplier=exit_multiplier,
            chop_period=chop_period,
            chop_threshold=chop_threshold
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
            'entry_period': trial.suggest_int('entry_period', 3, 120,),
            'entry_multiplier': trial.suggest_float('entry_multiplier', 1.0, 7.0, step=0.1),
            'exit_period': trial.suggest_int('exit_period', 3, 120,),
            'exit_multiplier': trial.suggest_float('exit_multiplier', 1.0, 7.0, step=0.1),
            'chop_period': trial.suggest_int('chop_period',  3, 120,),
            'chop_threshold': 50.0
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
            'entry_multiplier': float(params['entry_multiplier']),
            'exit_period': int(params['exit_period']),
            'exit_multiplier': float(params['exit_multiplier']),
            'chop_period': int(params['chop_period']),
            'chop_threshold': 50.0
        } 