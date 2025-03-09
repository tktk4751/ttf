#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import TrendAlphaV3SignalGenerator


class TrendAlphaV3Strategy(BaseStrategy):
    """
    TrendAlpha+アダプティブチョピネスフィルター戦略
    
    エントリー条件:
    - ロング: TrendAlphaのブレイクアウトで買いシグナル + アダプティブチョピネスがトレンド相場
    - ショート: TrendAlphaのブレイクアウトで売りシグナル + アダプティブチョピネスがトレンド相場
    
    エグジット条件:
    - ロング: TrendAlphaの売りシグナル
    - ショート: TrendAlphaの買いシグナル
    """
    
    def __init__(
        self,
        period: int = 20,
        max_kama_slow: int = 100,
        min_kama_slow: int = 25,
        max_kama_fast: int = 20,
        min_kama_fast: int = 2,
        max_atr_period: int = 120,
        min_atr_period: int = 5,
        max_multiplier: float = 3.5,
        min_multiplier: float = 0.3,
        chop_mfast: int = 13,
        chop_mslow: int = 120,
        max_threshold: float = 0.6,
        min_threshold: float = 0.4
    ):
        """
        初期化
        
        Args:
            period: KAMAの効率比の計算期間とチョピネスの基本期間
            max_kama_slow: KAMAの遅い移動平均の最大期間
            min_kama_slow: KAMAの遅い移動平均の最小期間
            max_kama_fast: KAMAの速い移動平均の最大期間
            min_kama_fast: KAMAの速い移動平均の最小期間
            max_atr_period: ATR期間の最大値
            min_atr_period: ATR期間の最小値
            max_multiplier: ATR乗数の最大値
            min_multiplier: ATR乗数の最小値
            chop_mfast: チョピネスの最小期間
            chop_mslow: チョピネスの最大期間
            chop_max_threshold: チョピネスのしきい値の最大値（デフォルト: 61.8）
            chop_min_threshold: チョピネスのしきい値の最小値（デフォルト: 38.2）
        """
        super().__init__("TrendAlphaV3")
        
        # パラメータの設定
        self._parameters = {
            'period': period,
            'max_kama_slow': max_kama_slow,
            'min_kama_slow': min_kama_slow,
            'max_kama_fast': max_kama_fast,
            'min_kama_fast': min_kama_fast,
            'max_atr_period': max_atr_period,
            'min_atr_period': min_atr_period,
            'max_multiplier': max_multiplier,
            'min_multiplier': min_multiplier,
            'chop_mfast': chop_mfast,
            'chop_mslow': chop_mslow,
            'max_threshold': max_threshold,
            'min_threshold': min_threshold
        }
        
        # シグナル生成器の初期化
        self.signal_generator = TrendAlphaV3SignalGenerator(
            kama_period=period,
            max_kama_slow=max_kama_slow,
            min_kama_slow=min_kama_slow,
            max_kama_fast=max_kama_fast,
            min_kama_fast=min_kama_fast,
            max_atr_period=max_atr_period,
            min_atr_period=min_atr_period,
            max_multiplier=max_multiplier,
            min_multiplier=min_multiplier,
            chop_mfast=chop_mfast,
            chop_mslow=chop_mslow,
            max_threshold=max_threshold,
            min_threshold=min_threshold
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
            'period': trial.suggest_int('period', 5, 300),
            'max_kama_slow': 100,
            'min_kama_slow': 25,
            'max_kama_fast': 20,
            'min_kama_fast': 2,
            'max_atr_period': 120,
            'min_atr_period': 5,
            'max_multiplier': 3.5,
            'min_multiplier': 0.3,
            'chop_mfast': 13,
            'chop_mslow': 120,
            'max_threshold': 0.6,
            'min_threshold': 0.4
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
            'period': int(params['period']),
            'max_kama_slow': 89,
            'min_kama_slow': 30,
            'max_kama_fast': 15,
            'min_kama_fast': 2,
            'max_atr_period': 120,
            'min_atr_period': 13,
            'max_multiplier': 3,
            'min_multiplier': 0.5,
            'chop_mfast': 5,
            'chop_mslow': 120,
            'max_threshold': 0.6,
            'min_threshold': 0.4
        } 