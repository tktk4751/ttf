#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union

import numpy as np
import pandas as pd

from .indicator import Indicator


class ChoppinessIndex(Indicator):
    """
    チョピネスインデックス（Choppiness Index）
    0-100の範囲で市場のトレンド/レンジ状態を示す
    - 61.8以上: レンジ相場
    - 38.2以下: トレンド相場
    """
    
    def __init__(self, period: int = 14):
        """
        コンストラクタ
        
        Args:
            period: 期間
        """
        super().__init__(f"CHOP({period})")
        self.period = period
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        チョピネスインデックスを計算する
        
        Args:
            data: 価格データ
        
        Returns:
            チョピネスインデックス値の配列
        """
        df = pd.DataFrame(data)
        high = df['high']
        low = df['low']
        
        # True Rangeを計算
        tr = pd.DataFrame()
        tr['h-l'] = high - low
        tr['h-pc'] = abs(high - high.shift(1))
        tr['l-pc'] = abs(low - low.shift(1))
        tr = tr.max(axis=1)
        
        # ATRの合計とレンジの合計を計算
        atr_sum = tr.rolling(window=self.period).sum()
        range_sum = (high.rolling(window=self.period).max() - 
                    low.rolling(window=self.period).min())
        
        # チョピネスインデックスを計算
        self._values = 100 * np.log10(atr_sum / range_sum) / np.log10(self.period)
        
        return self._values.to_numpy() 