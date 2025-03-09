#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd
from numba import jit

from ...base_signal import BaseSignal
from ...interfaces.filter import IFilterSignal
from indicators.adaptive_choppiness import AdaptiveChoppinessIndex


@jit(nopython=True)
def generate_signals_numba(
    chop_values: np.ndarray,
    threshold_values: np.ndarray
) -> np.ndarray:
    """
    シグナルを生成する（高速化版）
    
    Args:
        chop_values: チョピネス値の配列
        threshold_values: 動的しきい値の配列
    
    Returns:
        シグナルの配列 (1: トレンド相場, -1: レンジ相場)
    """
    length = len(chop_values)
    signals = np.ones(length)  # デフォルトはトレンド相場
    
    for i in range(length):
        if np.isnan(chop_values[i]) or np.isnan(threshold_values[i]):
            signals[i] = np.nan
        elif chop_values[i] < threshold_values[i]:
            signals[i] = -1  # レンジ相場
    
    return signals


class AdaptiveChopFilterSignal(BaseSignal, IFilterSignal):
    """
    アダプティブ・チョピネスインデックスを使用したフィルターシグナル
    - CHOP >= threshold: トレンド相場 (1)
    - CHOP < threshold: レンジ相場 (-1)
    
    特徴：
    - Numbaによる高速化
    - NaN値の適切な処理
    - 型安全な実装
    - 動的なしきい値の調整（ERが高いほど高く、低いほど低く）
    """
    
    def __init__(
        self,
        period: int = 14,
        mfast: int = 2,
        mslow: int = 30,
        max_threshold: float = 0.6,
        min_threshold: float = 0.4
    ):
        """
        コンストラクタ
        
        Args:
            period: アダプティブ・チョピネスインデックスの基本期間（デフォルト: 14）
            mfast: 最小期間（デフォルト: 2）
            mslow: 最大期間（デフォルト: 30）
            max_threshold: しきい値の最大値（デフォルト: 0.6）
            min_threshold: しきい値の最小値（デフォルト: 0.4）
        """
        params = {
            'period': period,
            'mfast': mfast,
            'mslow': mslow,
            'max_threshold': max_threshold,
            'min_threshold': min_threshold
        }
        super().__init__(f"AdaptiveChopFilter({period}, {mfast}, {mslow}, {max_threshold}, {min_threshold})", params)
        self._chop = AdaptiveChoppinessIndex(period, mfast, mslow, max_threshold, min_threshold)
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'high'と'low'と'close'カラムが必要
        
        Returns:
            シグナルの配列 (1: トレンド相場, -1: レンジ相場)
        """
        # チョピネス値と動的しきい値を計算
        chop_values = self._chop.calculate(data)
        threshold_values = self._chop.get_threshold()
        
        # シグナルの生成（高速化版）
        signals = generate_signals_numba(chop_values, threshold_values)
        
        return signals 