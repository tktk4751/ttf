#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.stochastic import Stochastic

class StochasticEntrySignal(BaseSignal, IEntrySignal):
    """
    ストキャスティクスを使用したエントリーシグナル
    - %K <= stoch_long_entry: ロングエントリー (1)
    - %K >= stoch_short_entry: ショートエントリー (-1)
    """
    
    def __init__(
        self,
        k_period: int = 14,
        d_period: int = 3,
        smooth_period: int = 3,
        solid: Dict[str, Any] = None
    ):
        """
        コンストラクタ
        
        Args:
            k_period: %Kの期間
            d_period: %Dの期間
            smooth_period: スムージング期間
            solid: パラメータ辞書
                - stoch_long_entry: ロングエントリーのストキャスティクスしきい値
                - stoch_short_entry: ショートエントリーのストキャスティクスしきい値
        """
        params = {
            'k_period': k_period,
            'd_period': d_period,
            'smooth_period': smooth_period,
            'solid': solid or {
                'stoch_long_entry': 20,
                'stoch_short_entry': 80
            }
        }
        super().__init__(f"StochasticEntry({k_period}, {d_period})", params)
        self._stoch = Stochastic(k_period, d_period, smooth_period)
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
        """
        stoch_values = self._stoch.calculate(data)
        k_values = stoch_values.k
        
        # シグナルの初期化
        signals = np.zeros(len(k_values))
        
        # エントリーシグナルの生成
        solid = self._params['solid']
        signals = np.where(k_values <= solid['stoch_long_entry'], 1, signals)  # ロングエントリー
        signals = np.where(k_values >= solid['stoch_short_entry'], -1, signals)  # ショートエントリー
        
        return signals 