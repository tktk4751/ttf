#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import KAMAKeltnerSingleLongSignalGenerator


class KAMAKeltnerSingleLongStrategy(BaseStrategy):
    """
    KAMAケルトナーチャネル戦略（単一チャネル・買い専用）
    
    エントリー条件:
    - KAMAケルトナーチャネルのアッパーブレイクアウトで買いシグナル
    
    エグジット条件:
    - KAMAケルトナーチャネルの売りシグナル
    """
    
    def __init__(
        self,
        kama_period: int = 243,
        kama_fast: int = 5,
        kama_slow: int = 111,
        atr_period: int = 150,
        upper_multiplier: float = 2.2,
        lower_multiplier: float = 2.2,
    ):
        """
        初期化
        
        Args:
            kama_period: KAMAの効率比の計算期間
            kama_fast: KAMAの速い移動平均の期間
            kama_slow: KAMAの遅い移動平均の期間
            atr_period: ATRの期間
            upper_multiplier: アッパーバンドのATR乗数
            lower_multiplier: ロワーバンドのATR乗数
        """
        super().__init__("KAMAKeltnerSingleLong")
        
        # パラメータの設定
        self._parameters = {
            'kama_period': kama_period,
            'kama_fast': kama_fast,
            'kama_slow': kama_slow,
            'atr_period': atr_period,
            'upper_multiplier': upper_multiplier,
            'lower_multiplier': lower_multiplier,
        }
        
        # シグナル生成器の初期化
        self.signal_generator = KAMAKeltnerSingleLongSignalGenerator(
            kama_period=kama_period,
            kama_fast=kama_fast,
            kama_slow=kama_slow,
            atr_period=atr_period,
            upper_multiplier=upper_multiplier,
            lower_multiplier=lower_multiplier,
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
            'kama_period': trial.suggest_int('kama_period', 20, 300),
            'kama_fast': 2,
            'kama_slow': trial.suggest_int('kama_slow', 20, 200),
            'atr_period': trial.suggest_int('atr_period', 5, 150),
            'upper_multiplier': trial.suggest_float('upper_multiplier', 0.5, 5.0, step=0.1),
            'lower_multiplier': trial.suggest_float('lower_multiplier', 0.5, 3.0, step=0.1),
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
            'kama_period': int(params['kama_period']),
            'kama_fast': 2,
            'kama_slow': int(params['kama_slow']),
            'atr_period': int(params['atr_period']),
            'upper_multiplier': float(params['upper_multiplier']),
            'lower_multiplier': float(params['lower_multiplier']),
        } 