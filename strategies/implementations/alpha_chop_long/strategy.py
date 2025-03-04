#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import AlphaChopLongSignalGenerator


class AlphaChopLongStrategy(BaseStrategy):
    """
    Alphaチャネル+チョピネスフィルター戦略（ロング専用）
    
    エントリー条件:
    - Alphaチャネルのブレイクアウトで買いシグナル（ERに基づいて動的に調整）
    - チョピネスインデックスが非トレンド相場を示している
    
    エグジット条件:
    - Alphaチャネルの売りシグナル
    """
    
    def __init__(
        self,
        kama_period: int = 175,
        kama_fast: int = 3,
        kama_slow: int = 144,
        atr_period: int = 65,
        max_multiplier: float = 4.8,
        min_multiplier: float = 2.9,
        chop_period: int = 55,
        chop_threshold: float = 50.0,
    ):
        """
        初期化
        
        Args:
            kama_period: KAMAの効率比の計算期間
            kama_fast: KAMAの速い移動平均の期間
            kama_slow: KAMAの遅い移動平均の期間
            atr_period: ATRの期間
            max_multiplier: ATR乗数の最大値
            min_multiplier: ATR乗数の最小値
            chop_period: チョピネスインデックスの期間
            chop_threshold: チョピネスインデックスのしきい値
        """
        super().__init__("AlphaChopLong")
        
        # パラメータの設定
        self._parameters = {
            'kama_period': kama_period,
            'kama_fast': kama_fast,
            'kama_slow': kama_slow,
            'atr_period': atr_period,
            'max_multiplier': max_multiplier,
            'min_multiplier': min_multiplier,
            'chop_period': chop_period,
            'chop_threshold': chop_threshold,
        }
        
        # シグナル生成器の初期化
        self.signal_generator = AlphaChopLongSignalGenerator(
            kama_period=kama_period,
            kama_fast=kama_fast,
            kama_slow=kama_slow,
            atr_period=atr_period,
            max_multiplier=max_multiplier,
            min_multiplier=min_multiplier,
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
            'kama_period': trial.suggest_int('kama_period', 5, 300),
            'kama_fast': 2,
            'kama_slow': 30,
            'atr_period': trial.suggest_int('atr_period', 3, 150),
            'max_multiplier': 3,
            'min_multiplier': 1,
            'chop_period': 55,
            'chop_threshold': 50,
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
            'kama_slow': 30,
            'atr_period': int(params['atr_period']),
            'max_multiplier': 3,
            'min_multiplier': 1,
            'chop_period': 55,
            'chop_threshold': 50,
        } 