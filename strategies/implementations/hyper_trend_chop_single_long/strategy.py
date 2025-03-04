#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import HyperTrendChopSingleLongSignalGenerator


class HyperTrendChopSingleLongStrategy(BaseStrategy):
    """
    HyperTrend+チョピネスフィルター戦略（ロング専用）
    
    エントリー条件:
    [ロング]
    - HyperTrendが上昇トレンドを示している
    - チョピネスインデックスがトレンド相場を示している
    
    エグジット条件:
    [ロング]
    - HyperTrendが下降トレンドに転換
    """
    
    def __init__(
        self,
        er_period: int = 10,
        max_percentile_length: int = 55,
        min_percentile_length: int = 14,
        max_atr_period: int = 120,
        min_atr_period: int = 5,
        max_multiplier: float = 3.0,
        min_multiplier: float = 1.0,
        chop_period: int = 14,
        chop_threshold: float = 50.0,
    ):
        """
        初期化
        
        Args:
            er_period: 効率比の計算期間
            max_percentile_length: パーセンタイル計算の最大期間
            min_percentile_length: パーセンタイル計算の最小期間
            max_atr_period: ATR期間の最大値
            min_atr_period: ATR期間の最小値
            max_multiplier: ATR乗数の最大値
            min_multiplier: ATR乗数の最小値
            chop_period: チョピネスインデックスの期間
            chop_threshold: チョピネスインデックスのしきい値
        """
        super().__init__("HyperTrendChopSingleLong")
        
        # パラメータの設定
        self._parameters = {
            'er_period': er_period,
            'max_percentile_length': max_percentile_length,
            'min_percentile_length': min_percentile_length,
            'max_atr_period': max_atr_period,
            'min_atr_period': min_atr_period,
            'max_multiplier': max_multiplier,
            'min_multiplier': min_multiplier,
            'chop_period': chop_period,
            'chop_threshold': chop_threshold,
        }
        
        # シグナル生成器の初期化
        self.signal_generator = HyperTrendChopSingleLongSignalGenerator(
            er_period=er_period,
            max_percentile_length=max_percentile_length,
            min_percentile_length=min_percentile_length,
            max_atr_period=max_atr_period,
            min_atr_period=min_atr_period,
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
            'er_period': trial.suggest_int('er_period', 5, 300),
            'max_percentile_length': trial.suggest_int('max_percentile_length', 27, 300),
            'min_percentile_length': trial.suggest_int('min_percentile_length', 5, 27),
            'max_atr_period': 130,
            'min_atr_period': 5,
            'max_multiplier': 3,
            'min_multiplier': 0.5,
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
            'er_period': int(params['er_period']),
            'max_percentile_length': int(params['max_percentile_length']),
            'min_percentile_length': int(params['min_percentile_length']),
            'max_atr_period': 130,
            'min_atr_period': 5,
            'max_multiplier': 3,
            'min_multiplier': 1,
            'chop_period': 55,
            'chop_threshold': 50,
        } 