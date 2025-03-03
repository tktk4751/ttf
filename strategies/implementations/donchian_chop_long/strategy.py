#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import DonchianChopLongSignalGenerator


class DonchianChopLongStrategy(BaseStrategy):
    """
    ドンチャンチャネル+チョピネスフィルター戦略（買い専用）
    
    エントリー条件:
    - ドンチャンチャネルのブレイクアウトで買いシグナル
    - チョピネスインデックスがトレンド相場を示している
    
    エグジット条件:
    - ドンチャンチャネルの売りシグナル
    - RSIのエグジットシグナル
    """
    
    def __init__(
        self,
        donchian_period: int = 20,
        chop_period: int = 14,
        chop_threshold: float = 50.0,
        # rsi_period: int = 14,
        # rsi_exit_threshold: float = 90.0
    ):
        """
        初期化
        
        Args:
            donchian_period: ドンチャンチャネルの期間
            chop_period: チョピネスインデックスの期間
            chop_threshold: チョピネスインデックスの閾値
            rsi_period: RSIの期間
            rsi_exit_threshold: RSIのエグジット閾値
        """
        super().__init__("DonchianChopLong")
        
        # パラメータの設定
        self._parameters = {
            'donchian_period': donchian_period,
            'chop_period': chop_period,
            'chop_threshold': chop_threshold,
            # 'rsi_period': rsi_period,
            # 'rsi_exit_threshold': rsi_exit_threshold
        }
        
        # シグナル生成器の初期化
        self.signal_generator = DonchianChopLongSignalGenerator(
            donchian_period=donchian_period,
            chop_period=chop_period,
            chop_threshold=chop_threshold,
            # rsi_period=rsi_period,
            # rsi_exit_threshold=rsi_exit_threshold
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
            'donchian_period': trial.suggest_int('donchian_period', 10, 300, step=5),
            'chop_period': trial.suggest_int('chop_period', 5, 100, step=1),
            'chop_threshold': 50,
            # 'rsi_period': trial.suggest_int('rsi_period', 8, 34, step=1),
            # 'rsi_exit_threshold': 86
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
            'donchian_period': int(params['donchian_period']),
            'chop_period': int(params['chop_period']),
            'chop_threshold': 50,
            # 'rsi_period': int(params['rsi_period']),
            # 'rsi_exit_threshold': 86
        } 