#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union

import numpy as np
import pandas as pd

from .signal import Signal
from indicators.supertrend import Supertrend

class SupertrendDirectionSignal(Signal):
    """
    スーパートレンドを使用した方向性シグナル
    トレンドの方向に基づいて、買い(1)または売り(-1)のシグナルを生成する
    """
    
    def __init__(self, period: int = 10, multiplier: float = 3.0):
        """
        コンストラクタ
        
        Args:
            period: ATRの期間
            multiplier: ATRの乗数
        """
        super().__init__(f"SupertrendDirection({period}, {multiplier})")
        self.period = period
        self.multiplier = multiplier
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: 買い, -1: 売り)
        """
        supertrend = Supertrend(self.period, self.multiplier)
        result = supertrend.calculate(data)
        
        return result.trend  # トレンド方向をそのままシグナルとして使用