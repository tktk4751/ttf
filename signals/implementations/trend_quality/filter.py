#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd
from numba import jit

from ...base_signal import BaseSignal
from ...interfaces.filter import IFilterSignal
from indicators.trend_quality import TrendQuality

@jit(nopython=True)
def calculate_trend_quality_signals(trend_quality_values: np.ndarray, threshold: float) -> np.ndarray:
    """
    トレンドクオリティのシグナルを計算する（高速化版）
    
    Args:
        trend_quality_values: トレンドクオリティの値
        threshold: しきい値
    
    Returns:
        シグナルの配列 (1: 強いトレンド, -1: 弱いトレンド/レンジ)
    """
    return np.where(trend_quality_values >= threshold, 1, -1)

class TrendQualityFilterSignal(BaseSignal, IFilterSignal):
    """
    トレンドクオリティを使用したフィルターシグナル
    - TQ >= threshold: 強いトレンド (1)
    - TQ < threshold: 弱いトレンド/レンジ (-1)
    """
    
    def __init__(
        self,
        er_period: int = 10,
        max_period: int = 30,
        min_period: int = 10,
        solid: Dict[str, Any] = None
    ):
        """
        コンストラクタ
        
        Args:
            er_period: 効率比の計算期間（デフォルト: 10）
            max_period: インジケーター期間の最大値（デフォルト: 30）
            min_period: インジケーター期間の最小値（デフォルト: 10）
            solid: パラメータ辞書
                - trend_quality_threshold: トレンドクオリティのしきい値
        """
        params = {
            'er_period': er_period,
            'max_period': max_period,
            'min_period': min_period,
            'solid': solid or {
                'trend_quality_threshold': 0.5  # デフォルトのしきい値
            }
        }
        super().__init__(f"TrendQualityFilter({er_period}, {max_period}, {min_period})", params)
        self._trend_quality = TrendQuality(
            er_period=er_period,
            max_period=max_period,
            min_period=min_period
        )
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: 強いトレンド, -1: 弱いトレンド/レンジ)
        """
        trend_quality_values = self._trend_quality.calculate(data)
        
        # シグナルの生成（高速化版）
        solid = self._params['solid']
        signals = calculate_trend_quality_signals(
            trend_quality_values,
            solid['trend_quality_threshold']
        )
        
        return signals 