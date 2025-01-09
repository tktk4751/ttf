#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd

from .indicator import Indicator
from .rsi import RSI


@dataclass
class StochasticRSIResult:
    """ストキャスティクスRSIの計算結果"""
    k: np.ndarray  # %K値
    d: np.ndarray  # %D値（%Kの移動平均）


class StochasticRSI(Indicator):
    """
    ストキャスティクスRSIインジケーター
    RSIにストキャスティクスを適用したもの
    - 80以上: オーバーボート（買われすぎ）
    - 20以下: オーバーソールド（売られすぎ）
    """
    
    def __init__(
        self,
        rsi_period: int = 14,
        stoch_period: int = 14,
        k_period: int = 3,
        d_period: int = 3
    ):
        """
        コンストラクタ
        
        Args:
            rsi_period: RSIの期間
            stoch_period: ストキャスティクスの期間
            k_period: %K期間
            d_period: %D期間
        """
        super().__init__(f"StochRSI({rsi_period},{stoch_period})")
        self.rsi_period = rsi_period
        self.stoch_period = stoch_period
        self.k_period = k_period
        self.d_period = d_period
        self.rsi = RSI(period=rsi_period)
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> StochasticRSIResult:
        """
        ストキャスティクスRSIを計算する
        
        Args:
            data: 価格データ
        
        Returns:
            ストキャスティクスRSIの計算結果
        """
        # RSIを計算
        rsi_values = self.rsi.calculate(data)
        rsi_series = pd.Series(rsi_values)
        
        # RSIの最高値・最安値を計算
        rsi_low = rsi_series.rolling(window=self.stoch_period).min()
        rsi_high = rsi_series.rolling(window=self.stoch_period).max()
        
        # ストキャスティクスRSIを計算
        k = 100 * (rsi_series - rsi_low) / (rsi_high - rsi_low)
        
        # %Kをスムージング
        if self.k_period > 1:
            k = k.rolling(window=self.k_period).mean()
        
        # %Dを計算
        d = k.rolling(window=self.d_period).mean()
        
        self._values = k  # 基底クラスの要件を満たすため
        
        return StochasticRSIResult(
            k=k.to_numpy(),
            d=d.to_numpy()
        ) 