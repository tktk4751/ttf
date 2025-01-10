#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd

from .indicator import Indicator
from .atr import ATR

@dataclass
class KeltnerChannelResult:
    """ケルトナーチャネルの計算結果"""
    middle: np.ndarray  # 中心線（EMA）
    upper: np.ndarray   # 上限線
    lower: np.ndarray   # 下限線

class KeltnerChannel(Indicator):
    """
    ケルトナーチャネル インジケーター
    EMATとATRを使用して、価格のボラティリティに基づくチャネルを形成する
    """
    
    def __init__(self, period: int = 20, atr_period: int = 10, multiplier: float = 2.0):
        """
        コンストラクタ
        
        Args:
            period: EMAの期間
            atr_period: ATRの期間
            multiplier: ATRの乗数
        """
        super().__init__(f"Keltner({period}, {multiplier})")
        self.period = period
        self.atr_period = atr_period
        self.multiplier = multiplier
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> KeltnerChannelResult:
        """
        ケルトナーチャネルを計算する
        
        Args:
            data: 価格データ
        
        Returns:
            ケルトナーチャネルの計算結果
        """
        df = pd.DataFrame(data)
        close = df['close']
        
        # 中心線（EMA）の計算
        middle = close.ewm(span=self.period, adjust=False).mean()
        
        # ATRの計算
        atr = ATR(self.atr_period).calculate(data)
        
        # 上限線と下限線の計算
        upper = middle + (self.multiplier * atr)
        lower = middle - (self.multiplier * atr)
        
        self._values = middle.values  # 基底クラスの要件を満たすため
        
        return KeltnerChannelResult(
            middle=middle.values,
            upper=upper,
            lower=lower
        ) 