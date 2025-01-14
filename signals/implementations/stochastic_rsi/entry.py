#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.stochastic_rsi import StochasticRSI

class StochasticRSIEntrySignal(BaseSignal, IEntrySignal):
    """
    ストキャスティクスRSIを使用したエントリーシグナル
    - %K <= stoch_rsi_long_entry: ロングエントリー (1)
    - %K >= stoch_rsi_short_entry: ショートエントリー (-1)
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
                - stoch_rsi_long_entry: ロングエントリーのストキャスティクスRSIしきい値
                - stoch_rsi_short_entry: ショートエントリーのストキャスティクスRSIしきい値
        """
        params = {
            'rsi_period': rsi_period,
            'stoch_period': stoch_period,
            'k_period': k_period,
            'd_period': d_period,
            'solid': solid or {
                'stoch_rsi_long_entry': 20,
                'stoch_rsi_short_entry': 80
            }
        }
        super().__init__(f"StochasticRSIEntry({rsi_period}, {stoch_period})", params)
        self._stoch_rsi = StochasticRSI(rsi_period, stoch_period, k_period, d_period)
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
        """
        stoch_rsi_values = self._stoch_rsi.calculate(data)
        k_values = stoch_rsi_values.k
        
        # シグナルの初期化
        signals = np.zeros(len(k_values))
        
        # エントリーシグナルの生成
        solid = self._params['solid']
        signals = np.where(k_values <= solid['stoch_rsi_long_entry'], 1, signals)  # ロングエントリー
        signals = np.where(k_values >= solid['stoch_rsi_short_entry'], -1, signals)  # ショートエントリー
        
        return signals 