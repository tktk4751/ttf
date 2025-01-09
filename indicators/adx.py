#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd

from .indicator import Indicator

@dataclass
class ADXResult:
    """ADXの計算結果"""
    adx: np.ndarray    # ADX値
    plus_di: np.ndarray  # +DI値
    minus_di: np.ndarray  # -DI値

class ADX(Indicator):
    """
    ADX (Average Directional Index) インジケーター
    トレンドの強さを測定する
    - 25以上: 強いトレンド
    - 20以下: トレンドなし
    """
    
    def __init__(self, period: int = 14):
        """
        コンストラクタ
        
        Args:
            period: 期間
        """
        super().__init__(f"ADX({period})")
        self.period = period
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> ADXResult:
        """
        ADXを計算する
        
        Args:
            data: 価格データ
        
        Returns:
            ADXの計算結果
        """
        df = pd.DataFrame(data)
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range (TR)の計算 (Pine Scriptの ta.tr に相当)
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # +DM, -DMの計算 (Pine Scriptの dirmov(len) に相当)
        up = high.diff()  # ta.change(high)
        down = -low.diff() # -ta.change(low)
        
        plusDM = pd.Series(np.where((up > down) & (up > 0), up, 0), index=df.index)
        minusDM = pd.Series(np.where((down > up) & (down > 0), down, 0), index=df.index)
        
        # TR, +DM, -DM の平滑化 (Pine Scriptの ta.rma に相当)
        # ta.rma(source, length) は source.rolling(length).mean() ではなく、
        # source.ewm(alpha=1/length, adjust=False).mean() と等価
        truerange = tr.ewm(alpha=1/self.period, adjust=False).mean()
        plus = (100 * plusDM.ewm(alpha=1/self.period, adjust=False).mean() / truerange).fillna(0)
        minus = (100 * minusDM.ewm(alpha=1/self.period, adjust=False).mean() / truerange).fillna(0)

        
        # DXの計算 (Pine Scriptの adx(dilen, adxlen) に相当)
        # sumが0の場合は1で割るように修正
        sum_ = plus + minus
        dx = 100 * (abs(plus - minus) / np.where(sum_ == 0, 1, sum_))
        
        # ADXの計算（DXの平滑化）
        adx = dx.ewm(alpha=1/self.period, adjust=False).mean()
        
        self._values = adx  # 基底クラスの要件を満たすため
        
        return ADXResult(
            adx=adx.to_numpy(),
            plus_di=plus.to_numpy(),
            minus_di=minus.to_numpy()
        )