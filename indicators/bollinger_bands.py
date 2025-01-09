#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple

import numpy as np
import pandas as pd

from .indicator import Indicator
from .moving_average import MovingAverage


@dataclass
class BollingerBandsResult:
    """ボリンジャーバンドの計算結果"""
    upper: np.ndarray  # アッパーバンド
    middle: np.ndarray  # ミドルバンド（SMA）
    lower: np.ndarray  # ロワーバンド
    bandwidth: np.ndarray  # バンド幅 (%)
    percent_b: np.ndarray  # %B


class BollingerBands(Indicator):
    """
    ボリンジャーバンドインディケーター
    - ミドルバンド: N期間の単純移動平均
    - アッパーバンド: ミドルバンド + (N期間の標準偏差 × K)
    - ロワーバンド: ミドルバンド - (N期間の標準偏差 × K)
    """
    
    def __init__(self, period: int = 20, num_std: float = 2.0):
        """
        コンストラクタ
        
        Args:
            period: 期間
            num_std: 標準偏差の乗数
        """
        super().__init__(f"BB{period}")
        self.period = period
        self.num_std = num_std
        self.sma = MovingAverage(period, "sma")
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> BollingerBandsResult:
        """
        ボリンジャーバンドを計算する
        
        Args:
            data: 価格データ
        
        Returns:
            ボリンジャーバンドの計算結果
        """
        prices = self._validate_data(data)
        self._validate_period(self.period, len(prices))
        
        # 移動平均（ミドルバンド）を計算
        middle = self.sma.calculate(prices)
        
        # 移動標準偏差を計算
        rolling_std = self._calculate_rolling_std(prices)
        
        # アッパーバンドとロワーバンドを計算
        upper = middle + (rolling_std * self.num_std)
        lower = middle - (rolling_std * self.num_std)
        
        # バンド幅を計算 (%)
        bandwidth = (upper - lower) / middle * 100
        
        # %Bを計算
        percent_b = (prices - lower) / (upper - lower)
        
        self._values = middle  # 基底クラスの要件を満たすため
        
        return BollingerBandsResult(
            upper=upper,
            middle=middle,
            lower=lower,
            bandwidth=bandwidth,
            percent_b=percent_b
        )
    
    def _calculate_rolling_std(self, prices: np.ndarray) -> np.ndarray:
        """
        移動標準偏差を計算する
        
        Args:
            prices: 価格データの配列
        
        Returns:
            移動標準偏差の配列
        """
        result = np.full_like(prices, np.nan, dtype=np.float64)
        
        for i in range(self.period-1, len(prices)):
            window = prices[i-self.period+1:i+1]
            result[i] = np.std(window, ddof=1)  # ddof=1 for sample standard deviation
        
        return result
