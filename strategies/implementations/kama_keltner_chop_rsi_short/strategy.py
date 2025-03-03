#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import KAMAKeltnerChopRSIShortSignalGenerator


class KAMAKeltnerChopRSIShortStrategy(BaseStrategy):
    """
    KAMAケルトナーチャネル+チョピネスフィルター+RSIエグジット戦略（売り専用）
    
    エントリー条件:
    - KAMAケルトナーチャネルのロワーブレイクアウトで売りシグナル
    - チョピネスインデックスがトレンド相場を示している
    
    エグジット条件:
    - KAMAケルトナーチャネルの買いシグナル
    - RSIエグジットシグナルが-1
    """
    
    def __init__(
        self,
        kama_period: int = 10,
        kama_fast: int = 3,
        kama_slow: int = 144,
        atr_period: int = 10,
        upper_multiplier: float = 2.0,
        lower_multiplier: float = 2.0,
        chop_period: int = 14,
        chop_threshold: float = 50.0,
        rsi_period: int = 14,
        rsi_solid: Dict[str, Any] = None,
    ):
        """
        初期化
        
        Args:
            kama_period: KAMAの効率比の計算期間
            kama_fast: KAMAの速い移動平均の期間
            kama_slow: KAMAの遅い移動平均の期間
            atr_period: ATRの期間
            upper_multiplier: 上側バンドのATR乗数
            lower_multiplier: 下側バンドのATR乗数
            chop_period: チョピネスインデックスの期間
            chop_threshold: チョピネスインデックスのしきい値
            rsi_period: RSIの期間
            rsi_solid: RSIのパラメータ辞書
                - rsi_long_exit_solid: ロングエグジットのRSIしきい値
                - rsi_short_exit_solid: ショートエグジットのRSIしきい値
        """
        super().__init__("KAMAKeltnerChopRSIShort")
        
        # パラメータの設定
        self._parameters = {
            'kama_period': kama_period,
            'kama_fast': kama_fast,
            'kama_slow': kama_slow,
            'atr_period': atr_period,
            'upper_multiplier': upper_multiplier,
            'lower_multiplier': lower_multiplier,
            'chop_period': chop_period,
            'chop_threshold': chop_threshold,
            'rsi_period': rsi_period,
            'rsi_solid': rsi_solid or {
                'rsi_long_exit_solid': 20,
                'rsi_short_exit_solid': 80
            },
        }
        
        # シグナル生成器の初期化
        self.signal_generator = KAMAKeltnerChopRSIShortSignalGenerator(
            kama_period=kama_period,
            kama_fast=kama_fast,
            kama_slow=kama_slow,
            atr_period=atr_period,
            upper_multiplier=upper_multiplier,
            lower_multiplier=lower_multiplier,
            chop_period=chop_period,
            chop_threshold=chop_threshold,
            rsi_period=rsi_period,
            rsi_solid=rsi_solid,
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
            'kama_period': trial.suggest_int('kama_period', 20, 500, step=5),
            'kama_fast': 3,
            'kama_slow': 144,
            'atr_period': trial.suggest_int('atr_period', 5, 150, step=5),
            'upper_multiplier': trial.suggest_float('upper_multiplier', 0.0, 3.0, step=0.1),
            'lower_multiplier': trial.suggest_float('lower_multiplier', 0.0, 5.0, step=0.1),
            'chop_period': 55,
            'chop_threshold': 50.0,
            'rsi_period': trial.suggest_int('rsi_period', 5, 21),
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
            'kama_fast': 3,
            'kama_slow': 144,
            'atr_period': int(params['atr_period']),
            'upper_multiplier': float(params['upper_multiplier']),
            'lower_multiplier': float(params['lower_multiplier']),
            'chop_period': 55,
            'chop_threshold': 50.0,
            'rsi_period': int(params['rsi_period']),
        } 