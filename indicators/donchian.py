#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple
import numpy as np
import pandas as pd
from numba import jit

from .indicator import Indicator


@dataclass
class DonchianChannelResult:
    """ドンチャンチャネルの計算結果"""
    upper: np.ndarray  # 上限（n期間の最高値）
    lower: np.ndarray  # 下限（n期間の最安値）
    middle: np.ndarray  # 中央線（(上限 + 下限) / 2）


@jit(nopython=True)
def calculate_donchian(high: np.ndarray, low: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ドンチャンチャネルを計算する（高速化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        period: 期間
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 上限、下限、中央線の配列
    """
    length = len(high)
    
    # 結果配列の初期化
    upper = np.full(length, np.nan)
    lower = np.full(length, np.nan)
    middle = np.full(length, np.nan)
    
    # 各時点でのperiod期間の最高値、最安値を計算
    for i in range(period-1, length):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
        middle[i] = (upper[i] + lower[i]) / 2
    
    return upper, lower, middle


class DonchianChannel(Indicator):
    """
    ドンチャンチャネルインディケーター（高速化版）
    
    指定期間の最高値、最安値、およびその中央値を計算する
    - 上限: n期間の最高値
    - 下限: n期間の最安値
    - 中央線: (上限 + 下限) / 2
    """
    
    def __init__(self, period: int = 20):
        """
        コンストラクタ
        
        Args:
            period: 期間（デフォルト: 20）
        """
        super().__init__(f"DonchianChannel({period})")
        self.period = period
        self._result = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        ドンチャンチャネルを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'high'と'low'カラムが必要
        
        Returns:
            中央線の値を返す
        """
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
        
        # ドンチャンチャネルの計算（高速化版）
        upper, lower, middle = calculate_donchian(high, low, self.period)
        
        self._result = DonchianChannelResult(
            upper=upper,
            lower=lower,
            middle=middle
        )
        
        self._values = middle  # 基底クラスの要件を満たすため
        return middle
    
    def get_bands(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        上限、下限、中央線の値を取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (上限, 下限, 中央線)の値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.upper, self._result.lower, self._result.middle 