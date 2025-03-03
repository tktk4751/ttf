#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import GVIDYAKeltnerSingleLongSignalGenerator


class GVIDYAKeltnerSingleLongStrategy(BaseStrategy):
    """
    G-VIDYAケルトナーチャネル戦略（単一チャネル・買い専用）
    
    エントリー条件:
    - G-VIDYAケルトナーチャネルのアッパーブレイクアウトで買いシグナル
    - チョピネスフィルターがレンジ相場を示している
    
    エグジット条件:
    - G-VIDYAケルトナーチャネルの売りシグナル
    """
    
    def __init__(
        self,
        vidya_period: int = 46,
        sd_period: int = 28,
        gaussian_length: int = 4,
        gaussian_sigma: float = 2.0,
        atr_period: int = 14,
        upper_multiplier: float = 1.3,
        lower_multiplier: float = 1.3,
        chop_period: int = 14,
        chop_solid: float = 50.0,
    ):
        """
        初期化
        
        Args:
            vidya_period: VIDYA期間
            sd_period: 標準偏差の計算期間
            gaussian_length: ガウシアンフィルターの長さ
            gaussian_sigma: ガウシアンフィルターのシグマ
            atr_period: ATRの期間
            upper_multiplier: アッパーバンドのATR乗数
            lower_multiplier: ロワーバンドのATR乗数
            chop_period: チョピネスインデックスの期間
            chop_solid: チョピネスインデックスのしきい値
        """
        super().__init__("GVIDYAKeltnerSingleLong")
        
        # パラメータの設定
        self._parameters = {
            'vidya_period': vidya_period,
            'sd_period': sd_period,
            'gaussian_length': gaussian_length,
            'gaussian_sigma': gaussian_sigma,
            'atr_period': atr_period,
            'upper_multiplier': upper_multiplier,
            'lower_multiplier': lower_multiplier,
            'chop_period': chop_period,
            'chop_solid': chop_solid,
        }
        
        # シグナル生成器の初期化
        self.signal_generator = GVIDYAKeltnerSingleLongSignalGenerator(
            vidya_period=vidya_period,
            sd_period=sd_period,
            gaussian_length=gaussian_length,
            gaussian_sigma=gaussian_sigma,
            atr_period=atr_period,
            upper_multiplier=upper_multiplier,
            lower_multiplier=lower_multiplier,
            chop_period=chop_period,
            chop_solid=chop_solid,
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
            'vidya_period': trial.suggest_int('vidya_period', 5, 300),
            'sd_period': trial.suggest_int('sd_period', 3, 150),
            'gaussian_length': trial.suggest_int('gaussian_length', 2, 10),
            'gaussian_sigma': trial.suggest_float('gaussian_sigma', 0.5, 5.0, step=0.5),
            'atr_period': trial.suggest_int('atr_period', 5, 150),
            'upper_multiplier': trial.suggest_float('upper_multiplier', 0.5, 3.0, step=0.1),
            'lower_multiplier': trial.suggest_float('lower_multiplier', 0.5, 3.0, step=0.1),
            'chop_period': 55,
            'chop_solid': 50,
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
            'vidya_period': int(params['vidya_period']),
            'sd_period': int(params['sd_period']),
            'gaussian_length': int(params['gaussian_length']),
            'gaussian_sigma': float(params['gaussian_sigma']),
            'atr_period': int(params['atr_period']),
            'upper_multiplier': float(params['upper_multiplier']),
            'lower_multiplier': float(params['lower_multiplier']),
            'chop_period': 55,
            'chop_solid': 50,
        } 