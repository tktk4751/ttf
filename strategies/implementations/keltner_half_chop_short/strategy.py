#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import KeltnerHalfChopShortSignalGenerator


class KeltnerHalfChopShortStrategy(BaseStrategy):
    """
    ケルトナーチャネル+チョピネスフィルター戦略（売り専用）
    
    エントリー条件:
    - ケルトナーチャネルのロワーブレイクアウトで売りシグナル
    - チョピネスインデックスがトレンド相場を示している
    
    エグジット条件:
    - ケルトナーチャネルのアッパーハーフラインを上回る
    """
    
    def __init__(
        self,
        keltner_period: int = 20,
        atr_period: int = 10,
        multiplier: float = 2.0,
        chop_period: int = 14,
        chop_threshold: float = 50.0,
    ):
        """
        初期化
        
        Args:
            keltner_period: ケルトナーチャネルの期間
            atr_period: ATRの期間
            multiplier: ATRの乗数
            chop_period: チョピネスインデックスの期間
            chop_threshold: チョピネスインデックスの閾値
        """
        super().__init__("KeltnerHalfChopShort")
        
        # パラメータの設定
        self._parameters = {
            'keltner_period': keltner_period,
            'atr_period': atr_period,
            'multiplier': multiplier,
            'chop_period': chop_period,
            'chop_threshold': chop_threshold
        }
        
        # シグナル生成器の初期化
        self.signal_generator = KeltnerHalfChopShortSignalGenerator(
            keltner_period=keltner_period,
            atr_period=atr_period,
            multiplier=multiplier,
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
            'keltner_period': trial.suggest_int('keltner_period', 3, 120,),
            'atr_period': trial.suggest_int('atr_period', 3, 120,),
            'multiplier': trial.suggest_float('multiplier', 1.0, 7.0, step=0.5),
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
            'keltner_period': int(params['keltner_period']),
            'atr_period': int(params['atr_period']),
            'multiplier': float(params['multiplier']),
            'chop_period': int(params['chop_period']),
            'chop_threshold': 50.0
        } 