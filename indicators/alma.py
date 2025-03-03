#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union
import numpy as np
import pandas as pd
from numba import jit

from .indicator import Indicator


@jit(nopython=True)
def calculate_alma(close: np.ndarray, period: int, offset: float, sigma: float) -> np.ndarray:
    """
    ALMAを計算する（高速化版）
    
    Args:
        close: 終値の配列
        period: 期間
        offset: オフセット (0-1)。1に近いほど最新のデータを重視
        sigma: シグマ。大きいほど重みの差が大きくなる
    
    Returns:
        ALMA値の配列
    """
    length = len(close)
    result = np.full(length, np.nan)
    
    # ウィンドウサイズが価格データより大きい場合は調整
    window_size = min(period, length)
    
    # ウェイトの計算
    m = offset * (window_size - 1)
    s = window_size / sigma
    weights = np.zeros(window_size)
    weights_sum = 0.0
    
    for i in range(window_size):
        weight = np.exp(-((i - m) ** 2) / (2 * s * s))
        weights[i] = weight
        weights_sum += weight
    
    # 重みの正規化
    weights = weights / weights_sum
    
    # ALMAの計算
    for i in range(window_size - 1, length):
        result[i] = 0.0
        for j in range(window_size):
            result[i] += close[i - window_size + 1 + j] * weights[j]
    
    return result


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
        # データの検証と変換
        if isinstance(data, pd.DataFrame):
            if 'close' not in data.columns:
                raise ValueError("DataFrameには'close'カラムが必要です")
            close = data['close'].values
        else:
            close = data[:, 3]  # close
        
        # データ長の検証
        data_length = len(close)
        self._validate_period(self.period, data_length)
        
        # ALMAの計算（高速化版）
        self._values = calculate_alma(close, self.period, self.offset, self.sigma)
        return self._values 