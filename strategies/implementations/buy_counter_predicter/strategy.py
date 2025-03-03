#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import BuyCounterPredicterSignalGenerator


class BuyCounterPredicter(BaseStrategy):
    """
    複数のシグナルをポイント制で評価する逆張り買い戦略
    
    エントリー条件:
    - 各シグナルのポイントの合計がエントリーしきい値を超えた場合に買いシグナル
    
    エグジット条件:
    - 各シグナルのポイントの合計がエグジットしきい値を超えた場合に売りシグナル
    """
    
    def __init__(
        self,
        entry_threshold: float = 10.0,
        exit_threshold: float = 10.0,
    ):
        """
        初期化
        
        Args:
            entry_threshold: エントリーのしきい値
            exit_threshold: エグジットのしきい値
        """
        super().__init__("BuyCounterPredicter")
        
        # パラメータの設定
        self._parameters = {
            'entry_threshold': entry_threshold,
            'exit_threshold': exit_threshold,
        }
        
        # シグナル生成器の初期化
        self.signal_generator = BuyCounterPredicterSignalGenerator(
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
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
            'entry_threshold': trial.suggest_float('entry_threshold', 4.0, 20.0, step=2.0),
            'exit_threshold': trial.suggest_float('exit_threshold', 2.0, 20.0, step=2.0),
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
            'entry_threshold': float(params['entry_threshold']),
            'exit_threshold': float(params['exit_threshold']),
        } 