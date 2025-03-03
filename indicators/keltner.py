#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple

import numpy as np
import pandas as pd
from numba import jit

from .indicator import Indicator
from .atr import ATR

@dataclass
class KeltnerChannelResult:
    """ケルトナーチャネルの計算結果"""
    middle: np.ndarray  # 中心線（EMA）
    upper: np.ndarray   # 上限線
    lower: np.ndarray   # 下限線
    half_upper: np.ndarray  # 中間上限線
    half_lower: np.ndarray  # 中間下限線

@jit(nopython=True)
def calculate_keltner(close: np.ndarray, atr: np.ndarray, period: int, multiplier: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    ケルトナーチャネルを計算する（高速化版）
    
    Args:
        close: 終値の配列
        atr: ATRの配列
        period: EMAの期間
        multiplier: ATRの乗数
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            中心線、上限線、下限線、中間上限線、中間下限線の配列
    """
    length = len(close)
    
    # EMAの計算
    alpha = 2.0 / (period + 1)
    middle = np.zeros(length, dtype=np.float64)
    middle[0] = close[0]
    
    for i in range(1, length):
        middle[i] = alpha * close[i] + (1 - alpha) * middle[i-1]
    
    # バンドの計算
    upper = middle + (multiplier * atr)
    lower = middle - (multiplier * atr)
    half_upper = middle + (multiplier * 0.5 * atr)
    half_lower = middle - (multiplier * 0.5 * atr)
    
    return middle, upper, lower, half_upper, half_lower

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
        self.atr = ATR(atr_period)  # ATRインスタンスを公開
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> KeltnerChannelResult:
        """
        ケルトナーチャネルを計算する
        
        Args:
            data: 価格データ
        
        Returns:
            ケルトナーチャネルの計算結果
        """
        try:
            # データの準備
            if isinstance(data, pd.DataFrame):
                close = data['close'].to_numpy()
            else:
                close = data[:, 3]  # close
            
            # ATRの計算
            atr = self.atr.calculate(data)
            if atr is None:
                return None
            
            # ケルトナーチャネルの計算（高速化版）
            middle, upper, lower, half_upper, half_lower = calculate_keltner(
                close, atr, self.period, self.multiplier
            )
            
            self._values = middle  # 基底クラスの要件を満たすため
            
            return KeltnerChannelResult(
                middle=middle,
                upper=upper,
                lower=lower,
                half_upper=half_upper,
                half_lower=half_lower
            )
            
        except Exception:
            return None 