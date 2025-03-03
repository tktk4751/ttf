#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import SqueezeDonchianShortSignalGenerator


class SqueezeDonchianShortStrategy(BaseStrategy):
    """
    スクイーズ+ドンチャンブレイクアウト戦略（売り専用）
    
    エントリー条件:
    - スクイーズオン状態
    - ドンチャンチャネルのロワーブレイクアウトで売りシグナル
    
    エグジット条件:
    - ドンチャンチャネルの買いシグナル
    """
    
    def __init__(
        self,
        donchian_period: int = 20,
        bb_length: int = 20,
        bb_mult: float = 2.0,
        kc_length: int = 20,
        kc_mult: float = 1.5,
    ):
        """
        初期化
        
        Args:
            donchian_period: ドンチャンチャネルの期間
            bb_length: Bollinger Bands の期間
            bb_mult: Bollinger Bands の乗数
            kc_length: Keltner Channels の期間
            kc_mult: Keltner Channels の乗数
        """
        super().__init__("SqueezeDonchianShort")
        
        # パラメータの設定
        self._parameters = {
            'donchian_period': donchian_period,
            'bb_length': bb_length,
            'bb_mult': bb_mult,
            'kc_length': kc_length,
            'kc_mult': kc_mult
        }
        
        # シグナル生成器の初期化
        self.signal_generator = SqueezeDonchianShortSignalGenerator(
            donchian_period=donchian_period,
            bb_length=bb_length,
            bb_mult=bb_mult,
            kc_length=kc_length,
            kc_mult=kc_mult
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
            'donchian_period': trial.suggest_int('donchian_period', 3, 120,),
            'bb_length': trial.suggest_int('bb_length', 3, 120,),
            'bb_mult': trial.suggest_float('bb_mult', 1.0, 4.0, step=0.1),
            'kc_length': trial.suggest_int('kc_length', 3, 120,),
            'kc_mult': trial.suggest_float('kc_mult', 1.0, 4.0, step=0.1)
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
            'bb_length': int(params['bb_length']),
            'bb_mult': float(params['bb_mult']),
            'kc_length': int(params['kc_length']),
            'kc_mult': float(params['kc_mult'])
        } 