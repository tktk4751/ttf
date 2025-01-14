#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd

from ...base_signal import BaseSignal
from ...interfaces.exit import IExitSignal
from indicators.stochastic_rsi import StochasticRSI

class StochasticRSIExitSignal(BaseSignal, IExitSignal):
    """
    ストキャスティクスRSIを使用したエグジットシグナル
    - ロングエグジット: %K(t-1) >= exit_solid かつ %K(t) <= exit_solid
    - ショートエグジット: %K(t-1) <= exit_solid かつ %K(t) >= exit_solid
    """
    
    def __init__(
        self,
        rsi_period: int = 14,
        stoch_period: int = 14,
        k_period: int = 3,
        d_period: int = 3,
        solid: Dict[str, Any] = None
    ):
        """
        コンストラクタ
        
        Args:
            rsi_period: RSIの期間
            stoch_period: ストキャスティクスの期間
            k_period: %Kの期間
            d_period: %Dの期間
            solid: パラメータ辞書
                - stoch_rsi_long_exit_solid: ロングエグジットのストキャスティクスRSIしきい値
                - stoch_rsi_short_exit_solid: ショートエグジットのストキャスティクスRSIしきい値
        """
        params = {
            'rsi_period': rsi_period,
            'stoch_period': stoch_period,
            'k_period': k_period,
            'd_period': d_period,
            'solid': solid or {
                'stoch_rsi_long_exit_solid': 85,
                'stoch_rsi_short_exit_solid': 15
            }
        }
        super().__init__(f"StochasticRSIExit({rsi_period}, {stoch_period})", params)
        self._stoch_rsi = StochasticRSI(rsi_period, stoch_period, k_period, d_period)
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロングエグジット, -1: ショートエグジット, 0: シグナルなし)
        """
        stoch_rsi_values = self._stoch_rsi.calculate(data)
        k_values = pd.Series(stoch_rsi_values.k)
        
        # シグナルの初期化
        signals = np.zeros(len(k_values))
        
        # エグジットシグナルの生成
        solid = self._params['solid']
        
        # ロングポジションのエグジット条件
        long_exit = (k_values.shift(1) >= solid['stoch_rsi_long_exit_solid']) & \
                   (k_values <= solid['stoch_rsi_long_exit_solid'])
        
        # ショートポジションのエグジット条件
        short_exit = (k_values.shift(1) <= solid['stoch_rsi_short_exit_solid']) & \
                    (k_values >= solid['stoch_rsi_short_exit_solid'])
        
        # シグナルの設定
        signals = np.where(long_exit, 1, signals)
        signals = np.where(short_exit, -1, signals)
        
        return signals 