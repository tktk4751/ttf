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
class DonchianATRResult:
    """ドンチャンATRの計算結果"""
    middle: np.ndarray  # 中央線（ドンチャンチャネルの中央値）
    upper: np.ndarray   # 上限線（中央線 + ATR * upper_multiplier）
    lower: np.ndarray   # 下限線（中央線 - ATR * lower_multiplier）
    half_upper: np.ndarray  # 中間上限線（中央線 + ATR * upper_multiplier * 0.5）
    half_lower: np.ndarray  # 中間下限線（中央線 - ATR * lower_multiplier * 0.5）


@jit(nopython=True)
def calculate_donchian_middle(high: np.ndarray, low: np.ndarray, period: int) -> np.ndarray:
    """
    ドンチャンチャネルの中央値を計算する（高速化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        period: 期間
    
    Returns:
        np.ndarray: 中央線の配列
    """
    length = len(high)
    middle = np.full(length, np.nan)
    
    for i in range(period-1, length):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        middle[i] = (highest + lowest) / 2
    
    return middle


@jit(nopython=True)
def calculate_donchian_atr_bands(
    middle: np.ndarray,
    atr: np.ndarray,
    upper_multiplier: float,
    lower_multiplier: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    ドンチャンATRのバンドを計算する（高速化版）
    
    Args:
        middle: 中央値の配列
        atr: ATRの配列
        upper_multiplier: アッパーバンドのATR乗数
        lower_multiplier: ロワーバンドのATR乗数
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            上限線、下限線、中間上限線、中間下限線の配列
    """
    upper = middle + (upper_multiplier * atr)
    lower = middle - (lower_multiplier * atr)
    half_upper = middle + (upper_multiplier * 0.5 * atr)
    half_lower = middle - (lower_multiplier * 0.5 * atr)
    
    return upper, lower, half_upper, half_lower


class DonchianATR(Indicator):
    """
    ドンチャンATR インジケーター
    
    ドンチャンチャネルの中央値を基準に、ATRの倍数でバンドを形成する
    - アッパーバンド: 中央値 + upper_multiplier * ATR
    - ロワーバンド: 中央値 - lower_multiplier * ATR
    """
    
    def __init__(
        self,
        period: int = 20,
        atr_period: int = 10,
        upper_multiplier: float = 2.0,
        lower_multiplier: float = 2.0
    ):
        """
        コンストラクタ
        
        Args:
            period: ドンチャンチャネルの期間
            atr_period: ATRの期間
            upper_multiplier: アッパーバンドのATR乗数
            lower_multiplier: ロワーバンドのATR乗数
        """
        super().__init__(f"DonchianATR({period}, {atr_period}, {upper_multiplier}, {lower_multiplier})")
        self.period = period
        self.atr_period = atr_period
        self.upper_multiplier = upper_multiplier
        self.lower_multiplier = lower_multiplier
        self.atr = ATR(atr_period)
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> DonchianATRResult:
        """
        ドンチャンATRを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'high'と'low'カラムが必要
        
        Returns:
            ドンチャンATRの計算結果
        """
        try:
            # データの検証と変換
            if isinstance(data, pd.DataFrame):
                if 'high' not in data.columns or 'low' not in data.columns:
                    raise ValueError("DataFrameには'high'と'low'カラムが必要です")
                high = data['high'].values
                low = data['low'].values
            else:
                high = data[:, 1]  # high
                low = data[:, 2]   # low
            
            # データ長の検証
            data_length = len(high)
            self._validate_period(self.period, data_length)
            
            # ATRの計算
            atr = self.atr.calculate(data)
            if atr is None:
                return None
            
            # ドンチャンチャネルの中央値を計算
            middle = calculate_donchian_middle(high, low, self.period)
            
            # バンドの計算
            upper, lower, half_upper, half_lower = calculate_donchian_atr_bands(
                middle, atr, self.upper_multiplier, self.lower_multiplier
            )
            
            self._values = middle  # 基底クラスの要件を満たすため
            
            return DonchianATRResult(
                middle=middle,
                upper=upper,
                lower=lower,
                half_upper=half_upper,
                half_lower=half_lower
            )
            
        except Exception:
            return None 