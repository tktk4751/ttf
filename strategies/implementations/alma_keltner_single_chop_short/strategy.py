#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import ALMAKeltnerSingleChopShortSignalGenerator


class ALMAKeltnerSingleChopShortStrategy(BaseStrategy):
    """
    ALMAケルトナーチャネル+チョピネスフィルター戦略（単一チャネル・売り専用）
    
    エントリー条件:
    - ALMAケルトナーチャネルのロワーブレイクアウトで売りシグナル
    - チョピネスインデックスがトレンド相場を示している
    
    エグジット条件:
    - ALMAケルトナーチャネルの買いシグナル
    """
    
    def __init__(
        self,
        alma_period: int = 9,
        alma_offset: float = 0.85,
        alma_sigma: float = 6,
        atr_period: int = 150,
        upper_multiplier: float = 2.2,
        lower_multiplier: float = 2.2,
        chop_period: int = 27,
        chop_threshold: float = 50.0,
    ):
        """
        初期化
        
        Args:
            alma_period: ALMAの期間
            alma_offset: ALMAのオフセット
            alma_sigma: ALMAのシグマ
            atr_period: ATRの期間
            upper_multiplier: アッパーバンドのATR乗数
            lower_multiplier: ロワーバンドのATR乗数
            chop_period: チョピネスインデックスの期間
            chop_threshold: チョピネスインデックスのしきい値
        """
        super().__init__("ALMAKeltnerSingleChopShort")
        
        # パラメータの設定
        self._parameters = {
            'alma_period': alma_period,
            'alma_offset': alma_offset,
            'alma_sigma': alma_sigma,
            'atr_period': atr_period,
            'upper_multiplier': upper_multiplier,
            'lower_multiplier': lower_multiplier,
            'chop_period': chop_period,
            'chop_threshold': chop_threshold,
        }
        
        # シグナル生成器の初期化
        self.signal_generator = ALMAKeltnerSingleChopShortSignalGenerator(
            alma_period=alma_period,
            alma_offset=alma_offset,
            alma_sigma=alma_sigma,
            atr_period=atr_period,
            upper_multiplier=upper_multiplier,
            lower_multiplier=lower_multiplier,
            chop_period=chop_period,
            chop_threshold=chop_threshold,
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
            'alma_period': trial.suggest_int('alma_period', 5, 100),
            'alma_offset': 0.85,
            'alma_sigma': 6,
            'atr_period': trial.suggest_int('atr_period', 5, 150),
            'upper_multiplier': trial.suggest_float('upper_multiplier', 0.5, 3.0, step=0.1),
            'lower_multiplier': trial.suggest_float('lower_multiplier', 0.5, 5.0, step=0.1),
            'chop_period': trial.suggest_int('chop_period', 5, 150),
            'chop_threshold': 50.0,
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
            'alma_offset': 0.85,
            'alma_sigma': 6,
            'atr_period': int(params['atr_period']),
            'upper_multiplier': float(params['upper_multiplier']),
            'lower_multiplier': float(params['lower_multiplier']),
            'chop_period': int(params['chop_period']),
            'chop_threshold': 50.0,
        } 