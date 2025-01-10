#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from typing import Union

from .indicator import Indicator

class ALMA(Indicator):
    """
    ALMA (Arnaud Legoux Moving Average) インジケーター
    ノイズを低減しながら、価格変動に素早く反応する移動平均線
    """
    
    def __init__(self, period: int = 9, offset: float = 0.85, sigma: float = 6):
        """
        コンストラクタ
        
        Args:
            period: 期間
            offset: オフセット (0-1)。1に近いほど最新のデータを重視
            sigma: シグマ。大きいほど重みの差が大きくなる
        """
        super().__init__(f"ALMA({period})")
        self.period = period
        self.offset = offset
        self.sigma = sigma
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        ALMAを計算する
        
        Args:
            data: 価格データ
        
        Returns:
            ALMA値の配列
        """
        df = pd.DataFrame(data)
        close = df['close'].values
        
        # ウィンドウサイズが価格データより大きい場合は調整
        window_size = min(self.period, len(close))
        
        # ウェイトの計算
        m = self.offset * (window_size - 1)
        s = window_size / self.sigma
        weights = np.array([
            np.exp(-((i - m) ** 2) / (2 * s * s))
            for i in range(window_size)
        ])
        weights = weights / weights.sum()
        
        # ALMAの計算
        result = np.zeros_like(close)
        for i in range(window_size - 1, len(close)):
            result[i] = (close[i - window_size + 1:i + 1] * weights).sum()
        
        self._values = result
        return result 