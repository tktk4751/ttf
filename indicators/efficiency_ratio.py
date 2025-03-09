#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union
import numpy as np
import pandas as pd
from numba import jit

from .indicator import Indicator


@jit(nopython=True)
def calculate_efficiency_ratio(change: np.ndarray, volatility: np.ndarray) -> np.ndarray:
    """
    効率比（Efficiency Ratio）を計算する（高速化版）
    
    Args:
        change: 価格変化（終値の差分）の配列
        volatility: ボラティリティ（価格変化の絶対値の合計）の配列
    
    Returns:
        効率比の配列
    """
    return np.abs(change) / (volatility + 1e-10)  # ゼロ除算を防ぐ


@jit(nopython=True)
def calculate_efficiency_ratio_for_period(close: np.ndarray, period: int) -> np.ndarray:
    """
    指定された期間の効率比（ER）を計算する（高速化版）
    
    Args:
        close: 終値の配列
        period: 計算期間
    
    Returns:
        効率比の配列（0-1の範囲）
        - 1に近いほど効率的な価格変動（強いトレンド）
        - 0に近いほど非効率な価格変動（レンジ・ノイズ）
    """
    length = len(close)
    er = np.zeros(length)
    
    for i in range(period, length):
        change = close[i] - close[i-period]
        volatility = np.sum(np.abs(np.diff(close[i-period:i+1])))
        er[i] = calculate_efficiency_ratio(
            np.array([change]),
            np.array([volatility])
        )[0]
    
    return er


class EfficiencyRatio(Indicator):
    """
    効率比（Efficiency Ratio）インジケーター
    
    価格変動の効率性を測定する指標
    - 1に近いほど効率的な価格変動（強いトレンド）
    - 0に近いほど非効率な価格変動（レンジ・ノイズ）
    
    使用方法：
    - 0.618以上: 効率的な価格変動（強いトレンド）
    - 0.382以下: 非効率な価格変動（レンジ・ノイズ）
    """
    
    def __init__(self, period: int = 10):
        """
        コンストラクタ
        
        Args:
            period: 計算期間（デフォルト: 10）
        """
        super().__init__(f"ER({period})")
        self.period = period
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        効率比を計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'close'カラムが必要
        
        Returns:
            効率比の配列（0-1の範囲）
        """
        # データの検証と変換
        if isinstance(data, pd.DataFrame):
            if 'close' not in data.columns:
                raise ValueError("DataFrameには'close'カラムが必要です")
            close = data['close'].values
        else:
            if data.ndim == 2:
                close = data[:, 3]  # close
            else:
                close = data  # 1次元配列として扱う
        
        # データ長の検証
        data_length = len(close)
        self._validate_period(self.period, data_length)
        
        # 効率比の計算（高速化版）
        self._values = calculate_efficiency_ratio_for_period(close, self.period)
        
        return self._values 