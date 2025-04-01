#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numba import jit
from typing import Union
import pandas as pd


@jit(nopython=True)
def calculate_hyper_smoother_numba(data: np.ndarray, length: int) -> np.ndarray:
    """
    ハイパースムーサーアルゴリズムを使用して時系列データを平滑化します
    
    Args:
        data: 平滑化する時系列データの配列
        length: 平滑化の期間
        
    Returns:
        平滑化されたデータの配列
    """
    size = len(data)
    smoothed = np.zeros(size)
    
    # 初期値の設定
    f28 = np.zeros(size)
    f30 = np.zeros(size)
    vC = np.zeros(size)
    
    f38 = np.zeros(size)
    f40 = np.zeros(size)
    v10 = np.zeros(size)
    
    f48 = np.zeros(size)
    f50 = np.zeros(size)
    v14 = np.zeros(size)
    
    # パラメータの計算
    f18 = 3.0 / (length + 2.0)
    f20 = 1.0 - f18
    
    for i in range(1, size):
        # フィルタリング（1段階目）
        f28[i] = f20 * f28[i-1] + f18 * data[i]
        f30[i] = f18 * f28[i] + f20 * f30[i-1]
        vC[i] = f28[i] * 1.5 - f30[i] * 0.5
        
        # フィルタリング（2段階目）
        f38[i] = f20 * f38[i-1] + f18 * vC[i]
        f40[i] = f18 * f38[i] + f20 * f40[i-1]
        v10[i] = f38[i] * 1.5 - f40[i] * 0.5
        
        # フィルタリング（3段階目）
        f48[i] = f20 * f48[i-1] + f18 * v10[i]
        f50[i] = f18 * f48[i] + f20 * f50[i-1]
        v14[i] = f48[i] * 1.5 - f50[i] * 0.5
        
        # 最終的な平滑化値
        smoothed[i] = v14[i]
    
    return smoothed


def hyper_smoother(data: Union[pd.Series, np.ndarray], length: int = 14) -> np.ndarray:
    """
    ハイパースムーサーアルゴリズムを使用して時系列データを平滑化します。
    RSXインディケーターで使用されている3段階フィルタリングロジックを実装しています。
    
    Args:
        data: 平滑化する時系列データ（pandas.SeriesまたはNumPy配列）
        length: 平滑化の期間（デフォルト: 14）
        
    Returns:
        平滑化されたデータの配列
    
    Examples:
        >>> import numpy as np
        >>> from indicators.hyper_smoother import hyper_smoother
        >>> # ランダムデータの平滑化
        >>> data = np.random.randn(100).cumsum()
        >>> smoothed = hyper_smoother(data, length=10)
    """
    # データの検証と変換
    if isinstance(data, pd.Series):
        values = data.values
    else:
        values = data
    
    # データ長の検証
    if len(values) < length:
        raise ValueError(f"データ長（{len(values)}）は期間（{length}）より大きくなければなりません")
    
    # ハイパースムーサーの計算（Numba高速化版）
    return calculate_hyper_smoother_numba(values, length) 