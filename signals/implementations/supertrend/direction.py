#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd

from ...base_signal import BaseSignal
from ...interfaces.direction import IDirectionSignal
from indicators.supertrend import Supertrend

class SupertrendDirectionSignal(BaseSignal, IDirectionSignal):
    """
    Supertrendを使用した方向性シグナル
    - Supertrend > 価格: ショート方向 (-1)
    - Supertrend < 価格: ロング方向 (1)
    """
    
    def __init__(self, period: int = 10, multiplier: float = 3.0):
        """
        コンストラクタ
        
        Args:
            period: Supertrendの期間
            multiplier: ATRの乗数
        """
        params = {
            'period': period,
            'multiplier': multiplier
        }
        super().__init__(f"SupertrendDirection({period}, {multiplier})", params)
        self._supertrend = Supertrend(period, multiplier)
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロング方向, -1: ショート方向)
        """
        # Supertrendの計算
        supertrend_result = self._supertrend.calculate(data)
        
        # トレンド方向をそのままシグナルとして使用
        # trend属性は1（上昇トレンド）または-1（下降トレンド）を返す
        return supertrend_result.trend 