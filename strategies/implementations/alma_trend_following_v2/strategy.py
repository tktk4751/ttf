#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import ALMATrendFollowingV2SignalGenerator


class ALMATrendFollowingV2Strategy(BaseStrategy):
    """
    ALMAトレンドフォロー戦略V2
    
    エントリー条件:
    買い:
        - 短期ALMAが長期ALMAを上回る（ゴールデンクロス）
        - ADXが閾値以上（トレンド相場）
        - ROCが正（上昇モメンタム）
    売り:
        - 短期ALMAが長期ALMAを下回る（デッドクロス）
        - ADXが閾値以上（トレンド相場）
        - ROCが負（下降モメンタム）
        
    エグジット条件:
    買いポジション:
        - 短期ALMAが長期ALMAを下回る（デッドクロス）
    売りポジション:
        - 短期ALMAが長期ALMAを上回る（ゴールデンクロス）
    """
    
    def __init__(
        self,
        alma_short_period: int = 19,
        alma_long_period: int = 593,
        adx_period: int = 14,
        adx_threshold: float = 30,
        roc_period: int = 40,
    ):
        """
        初期化
        
        Args:
            alma_short_period: 短期ALMA期間
            alma_long_period: 長期ALMA期間
            adx_period: ADX期間
            adx_threshold: ADXの閾値
            roc_period: ROC期間
        """
        super().__init__("ALMATrendFollowingV2")
        
        # パラメータの設定
        self._parameters = {
            'alma_short_period': alma_short_period,
            'alma_long_period': alma_long_period,
            'adx_period': adx_period,
            'adx_threshold': adx_threshold,
            'roc_period': roc_period,
        }
        
        # シグナル生成器の初期化
        self.signal_generator = ALMATrendFollowingV2SignalGenerator(
            alma_short_period=alma_short_period,
            alma_long_period=alma_long_period,
            adx_period=adx_period,
            adx_threshold=adx_threshold,
            roc_period=roc_period,
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
            'alma_short_period': trial.suggest_int('alma_short_period', 3, 120, step=1),
            'alma_long_period': trial.suggest_int('alma_long_period', 21, 700, step=1),
            'adx_period': trial.suggest_int('adx_period', 3, 35, step=1),
            'adx_threshold': 25,
            'roc_period': trial.suggest_int('roc_period', 5, 100, step=1),
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
            'alma_short_period': int(params['alma_short_period']),
            'alma_long_period': int(params['alma_long_period']),
            'adx_period': int(params['adx_period']),
            'adx_threshold': 25,
            'roc_period': int(params['roc_period']),
        } 