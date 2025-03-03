#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import KAMAKeltnerShortSignalGenerator


class KAMAKeltnerShortStrategy(BaseStrategy):
    """
    KAMAケルトナーチャネル戦略（売り専用）
    
    エントリー条件:
    - エントリー用KAMAケルトナーチャネルのロワーブレイクアウトで売りシグナル
    
    エグジット条件:
    - エグジット用KAMAケルトナーチャネルの買いシグナル
    """
    
    def __init__(
        self,
        entry_kama_period: int = 10,
        entry_kama_fast: int = 2,
        entry_kama_slow: int = 30,
        entry_atr_period: int = 10,
        entry_multiplier: float = 2.0,
        exit_kama_period: int = 10,
        exit_kama_fast: int = 2,
        exit_kama_slow: int = 30,
        exit_atr_period: int = 10,
        exit_multiplier: float = 2.0,
    ):
        """
        初期化
        
        Args:
            entry_kama_period: エントリー用KAMAの効率比の計算期間
            entry_kama_fast: エントリー用KAMAの速い移動平均の期間
            entry_kama_slow: エントリー用KAMAの遅い移動平均の期間
            entry_atr_period: エントリー用ATRの期間
            entry_multiplier: エントリー用ATRの乗数
            exit_kama_period: エグジット用KAMAの効率比の計算期間
            exit_kama_fast: エグジット用KAMAの速い移動平均の期間
            exit_kama_slow: エグジット用KAMAの遅い移動平均の期間
            exit_atr_period: エグジット用ATRの期間
            exit_multiplier: エグジット用ATRの乗数
        """
        super().__init__("KAMAKeltnerShort")
        
        # パラメータの設定
        self._parameters = {
            'entry_kama_period': entry_kama_period,
            'entry_kama_fast': entry_kama_fast,
            'entry_kama_slow': entry_kama_slow,
            'entry_atr_period': entry_atr_period,
            'entry_multiplier': entry_multiplier,
            'exit_kama_period': exit_kama_period,
            'exit_kama_fast': exit_kama_fast,
            'exit_kama_slow': exit_kama_slow,
            'exit_atr_period': exit_atr_period,
            'exit_multiplier': exit_multiplier,
        }
        
        # シグナル生成器の初期化
        self.signal_generator = KAMAKeltnerShortSignalGenerator(
            entry_kama_period=entry_kama_period,
            entry_kama_fast=entry_kama_fast,
            entry_kama_slow=entry_kama_slow,
            entry_atr_period=entry_atr_period,
            entry_multiplier=entry_multiplier,
            exit_kama_period=exit_kama_period,
            exit_kama_fast=exit_kama_fast,
            exit_kama_slow=exit_kama_slow,
            exit_atr_period=exit_atr_period,
            exit_multiplier=exit_multiplier,
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
            'entry_kama_period': trial.suggest_int('entry_kama_period', 5, 300),
            'entry_kama_fast': trial.suggest_int('entry_kama_fast', 2, 20),
            'entry_kama_slow': trial.suggest_int('entry_kama_slow', 21, 200),
            'entry_atr_period': trial.suggest_int('entry_atr_period', 5, 150),
            'entry_multiplier': trial.suggest_float('entry_multiplier', 0.5, 5.0, step=0.1),
            'exit_kama_period': trial.suggest_int('exit_kama_period', 5, 300),
            'exit_kama_fast': trial.suggest_int('exit_kama_fast', 2, 20),
            'exit_kama_slow': trial.suggest_int('exit_kama_slow', 21, 200),
            'exit_atr_period': trial.suggest_int('exit_atr_period', 5, 150),
            'exit_multiplier': trial.suggest_float('exit_multiplier', 0.5, 5.0, step=0.1),
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
            'entry_kama_period': int(params['entry_kama_period']),
            'entry_kama_fast': int(params['entry_kama_fast']),
            'entry_kama_slow': int(params['entry_kama_slow']),
            'entry_atr_period': int(params['entry_atr_period']),
            'entry_multiplier': float(params['entry_multiplier']),
            'exit_kama_period': int(params['exit_kama_period']),
            'exit_kama_fast': int(params['exit_kama_fast']),
            'exit_kama_slow': int(params['exit_kama_slow']),
            'exit_atr_period': int(params['exit_atr_period']),
            'exit_multiplier': float(params['exit_multiplier']),
        } 