#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import HyperTrendV2SignalGenerator


class HyperTrendV2Strategy(BaseStrategy):
    """
    HyperTrend V2戦略 (ガーディアンエンジェルフィルターなし)
    
    エントリー条件:
    [ロング]
    - HyperTrendが上昇トレンドを示している
    
    [ショート]
    - HyperTrendが下降トレンドを示している
    
    エグジット条件:
    [ロング]
    - HyperTrendが下降トレンドに転換
    
    [ショート]
    - HyperTrendが上昇トレンドに転換
    """
    
    def __init__(
        self,
        period: int = 21,
        max_percentile_length: int = 250,
        min_percentile_length: int = 13,
        max_atr_period: int = 130,
        min_atr_period: int = 5,
        max_multiplier: float = 3,
        min_multiplier: float = 0.5
    ):
        """
        初期化
        
        Args:
            period: HyperTrendの効率比の計算期間
            max_percentile_length: パーセンタイル計算の最大期間
            min_percentile_length: パーセンタイル計算の最小期間
            max_atr_period: ATR期間の最大値
            min_atr_period: ATR期間の最小値
            max_multiplier: ATR乗数の最大値
            min_multiplier: ATR乗数の最小値
        """
        super().__init__("HyperTrendV2")
        
        # パラメータの設定
        self._parameters = {
            'period': period,
            'max_percentile_length': max_percentile_length,
            'min_percentile_length': min_percentile_length,
            'max_atr_period': max_atr_period,
            'min_atr_period': min_atr_period,
            'max_multiplier': max_multiplier,
            'min_multiplier': min_multiplier
        }
        
        # シグナル生成器の初期化
        self.signal_generator = HyperTrendV2SignalGenerator(
            period=period,
            max_percentile_length=max_percentile_length,
            min_percentile_length=min_percentile_length,
            max_atr_period=max_atr_period,
            min_atr_period=min_atr_period,
            max_multiplier=max_multiplier,
            min_multiplier=min_multiplier
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
        return {
            'period': trial.suggest_int('period', 5, 300)
        }
    
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
            'period': int(params['period']),
            'max_percentile_length': 100,
            'min_percentile_length': 13,
            'max_atr_period': 90,
            'min_atr_period': 5,
            'max_multiplier': 3,
            'min_multiplier': 0.5
        } 