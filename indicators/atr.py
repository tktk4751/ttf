#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from typing import Union

from .indicator import Indicator

class ATR(Indicator):
    """
    ATR (Average True Range) インジケーター
    価格のボラティリティを測定する
    """
    
    def __init__(self, period: int = 14):
        """
        コンストラクタ
        
        Args:
            period: 期間
        """
        super().__init__(f"ATR({period})")
        self.period = period
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        ATRを計算する
        
        Args:
            data: 価格データ
        
        Returns:
            ATR値の配列
        """
        df = pd.DataFrame(data)
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range (TR)の計算
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATRの計算 (Wilder's Smoothing)
        atr = tr.ewm(alpha=1/self.period, adjust=False).mean()
        
        self._values = atr.values
        return atr.values 