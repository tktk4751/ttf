#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import ALMATrendFollowingSignalGenerator


class ALMATrendFollowingStrategy(BaseStrategy):
    """
    ALMAトレンドフォロー戦略
    
    エントリー条件:
    買い:
        - ALMAが終値より下にある（上昇トレンド）
        - ADXが閾値以上（トレンド相場）
        - ROCが正（上昇モメンタム）
    売り:
        - ALMAが終値より上にある（下降トレンド）
        - ADXが閾値以上（トレンド相場）
        - ROCが負（下降モメンタム）
        
    エグジット条件:
    買いポジション:
        - ATRの倍数分下落
        - もしくはALMAが終値より上に移動（トレンド転換）
    売りポジション:
        - ATRの倍数分上昇
        - もしくはALMAが終値より下に移動（トレンド転換）
    """
    
    def __init__(
        self,
        alma_period: int = 200,
        adx_period: int = 14,
        adx_threshold: float = 30,
        roc_period: int = 21,
        atr_period: int = 14,
        atr_multiplier: float = 5
    ):
        """
        初期化
        
        Args:
            alma_period: ALMA期間
            adx_period: ADX期間
            adx_threshold: ADXの閾値
            roc_period: ROC期間
            atr_period: ATR期間
            atr_multiplier: ATRの乗数
        """
        super().__init__("ALMATrendFollowing")
        
        # パラメータの設定
        self._parameters = {
            'alma_period': alma_period,
            'adx_period': adx_period,
            'adx_threshold': adx_threshold,
            'roc_period': roc_period,
            'atr_period': atr_period,
            'atr_multiplier': atr_multiplier
        }
        
        # シグナル生成器の初期化
        self.signal_generator = ALMATrendFollowingSignalGenerator(
            alma_period=alma_period,
            adx_period=adx_period,
            adx_threshold=adx_threshold,
            roc_period=roc_period,
            atr_period=atr_period,
            atr_multiplier=atr_multiplier
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
            'alma_period': trial.suggest_int('alma_period', 13, 600, step=1),
            'adx_period': trial.suggest_int('adx_period', 3, 35, step=1),
            'adx_threshold': 30,
            'roc_period': trial.suggest_int('roc_period', 5, 100, step=1),
            'atr_period': 14,
            'atr_multiplier': 5
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
            'alma_period': int(params['alma_period']),
            'adx_period': int(params['adx_period']),
            'adx_threshold': 30,
            'roc_period': int(params['roc_period']),
            'atr_period': 14,
            'atr_multiplier': 5
        } 