#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Literal, Union

import numpy as np
import pandas as pd

from .indicator import Indicator


class MovingAverage(Indicator):
    """
    移動平均インディケーター
    - 単純移動平均（SMA）
    - 指数移動平均（EMA）
    をサポート
    """
    
    def __init__(
        self,
        period: int,
        ma_type: Literal["sma", "ema"] = "sma"
    ):
        """
        コンストラクタ
        
        Args:
            period: 期間
            ma_type: 移動平均の種類 ("sma" or "ema")
        """
        super().__init__(f"{ma_type.upper()}{period}")
        self.period = period
        self.ma_type = ma_type
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        移動平均を計算する
        
        Args:
            data: 価格データ
        
        Returns:
            移動平均値の配列
        """
        prices = self._validate_data(data)
        self._validate_period(self.period, len(prices))
        
        if self.ma_type == "sma":
            self._values = self._calculate_sma(prices)
        elif self.ma_type == "ema":
            self._values = self._calculate_ema(prices)
        else:
            raise ValueError(f"サポートされていない移動平均タイプです: {self.ma_type}")
        
        return self._values
    
    def _calculate_sma(self, prices: np.ndarray) -> np.ndarray:
        """
        単純移動平均を計算する
        
        Args:
            prices: 価格データの配列
        
        Returns:
            SMAの配列
        """
        # 累積和を使用して効率的に計算
        cumsum = np.cumsum(np.insert(prices, 0, 0))
        sma = (cumsum[self.period:] - cumsum[:-self.period]) / self.period
        
        # 期間未満のデータには NaN を設定
        result = np.full_like(prices, np.nan, dtype=np.float64)
        result[self.period-1:] = sma
        
        return result
    
    def _calculate_ema(self, prices: np.ndarray) -> np.ndarray:
        """
        指数移動平均を計算する
        
        Args:
            prices: 価格データの配列
        
        Returns:
            EMAの配列
        """
        result = np.full_like(prices, np.nan, dtype=np.float64)
        
        # 最初のSMAを計算
        result[self.period-1] = np.mean(prices[:self.period])
        
        # 乗数を計算 (一般的な 2/(period+1) を使用)
        multiplier = 2 / (self.period + 1)
        
        # EMAを計算
        for i in range(self.period, len(prices)):
            result[i] = (prices[i] - result[i-1]) * multiplier + result[i-1]
        
        return result
