#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd

from .indicator import Indicator


@dataclass
class StochasticResult:
    """ストキャスティクスの計算結果"""
    k: np.ndarray  # %K値
    d: np.ndarray  # %D値（%Kの移動平均）


class Stochastic(Indicator):
    """
    ストキャスティクスインジケーター
    オーバーボート/オーバーソールドを判断する
    - 80以上: オーバーボート（買われすぎ）
    - 20以下: オーバーソールド（売られすぎ）
    """
    
    def __init__(
        self,
        k_period: int = 14,
        d_period: int = 3,
        smooth_k: int = 3
    ):
        """
        コンストラクタ
        
        Args:
            k_period: %K期間
            d_period: %D期間（%Kの移動平均期間）
            smooth_k: %Kのスムージング期間
        """
        super().__init__(f"STOCH({k_period},{d_period})")
        self.k_period = k_period
        self.d_period = d_period
        self.smooth_k = smooth_k
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> StochasticResult:
        """
        ストキャスティクスを計算する
        
        Args:
            data: 価格データ
        
        Returns:
            ストキャスティクスの計算結果
        """
        df = pd.DataFrame(data)
        high = df['high']
        low = df['low']
        close = df['close']
        
        # 期間内の最高値・最安値を計算
        low_min = low.rolling(window=self.k_period).min()
        high_max = high.rolling(window=self.k_period).max()
        
        # %Kを計算
        k = 100 * (close - low_min) / (high_max - low_min)
        
        # %Kをスムージング
        if self.smooth_k > 1:
            k = k.rolling(window=self.smooth_k).mean()
        
        # %Dを計算（%Kの移動平均）
        d = k.rolling(window=self.d_period).mean()
        
        self._values = k  # 基底クラスの要件を満たすため
        
        return StochasticResult(
            k=k.to_numpy(),
            d=d.to_numpy()
        ) 