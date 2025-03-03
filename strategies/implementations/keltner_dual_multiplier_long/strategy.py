#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import KeltnerDualMultiplierLongSignalGenerator


class KeltnerDualMultiplierLongStrategy(BaseStrategy):
    """
    ケルトナーチャネル+チョピネスフィルター戦略（売買別ATR乗数・買い専用）
    
    エントリー条件:
    - アッパーブレイクアウトで買いシグナル（upper_multiplierを使用）
    - チョピネスインデックスがトレンド相場を示している
    
    エグジット条件:
    - 終値がロワーバンドを下回る
    """
    
    def __init__(
        self,
        ema_period: int = 20,
        atr_period: int = 10,
        upper_multiplier: float = 2.2,
        lower_multiplier: float = 2.2,
        chop_period: int = 14,
        chop_threshold: float = 50.0,
    ):
        """
        初期化
        
        Args:
            ema_period: EMAの期間
            atr_period: ATRの期間
            upper_multiplier: アッパーバンドのATR乗数
            lower_multiplier: ロワーバンドのATR乗数
            chop_period: チョピネスインデックスの期間
            chop_threshold: チョピネスインデックスの閾値
        """
        super().__init__("KeltnerDualMultiplierLong")
        
        # パラメータの設定
        self._parameters = {
            'ema_period': ema_period,
            'atr_period': atr_period,
            'upper_multiplier': upper_multiplier,
            'lower_multiplier': lower_multiplier,
            'chop_period': chop_period,
            'chop_threshold': chop_threshold,
        }
        
        # シグナル生成器の初期化
        self.signal_generator = KeltnerDualMultiplierLongSignalGenerator(
            ema_period=ema_period,
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
            'ema_period': trial.suggest_int('ema_period', 5, 300),
            'atr_period': trial.suggest_int('atr_period', 5, 130),
            'upper_multiplier': trial.suggest_float('upper_multiplier', 0.5, 5.0, step=0.1),
            'lower_multiplier': trial.suggest_float('lower_multiplier', 0.5, 3.0, step=0.1),
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
            'ema_period': int(params['ema_period']),
            'atr_period': int(params['atr_period']),
            'upper_multiplier': float(params['upper_multiplier']),
            'lower_multiplier': float(params['lower_multiplier']),
            'chop_period': int(params['chop_period']),
            'chop_threshold': 50.0,
        } 