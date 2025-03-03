#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple
import numpy as np
import pandas as pd
from numba import jit

from .indicator import Indicator


@dataclass
class SpanModelResult:
    """スパンモデルの計算結果"""
    conversion_line: np.ndarray    # 転換線
    base_line: np.ndarray         # 基準線
    span_a: np.ndarray           # スパンA
    span_b: np.ndarray           # スパンB
    lagging_span: np.ndarray     # 遅行スパン
    trend: np.ndarray            # トレンド方向（1=上昇トレンド、-1=下降トレンド、0=中立）


@jit(nopython=True)
def calculate_span_model(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    conversion_period: int,
    base_period: int,
    span_b_period: int,
    displacement: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    スパンモデルを計算する（高速化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
        conversion_period: 転換線期間
        base_period: 基準線期間
        span_b_period: スパンB期間
        displacement: 先行スパン移動期間
    
    Returns:
        Tuple[np.ndarray, ...]: 転換線、基準線、スパンA、スパンB、遅行スパン、トレンド方向の配列
    """
    size = len(close)
    
    # 結果配列の初期化
    conversion_line = np.zeros(size)
    base_line = np.zeros(size)
    span_a = np.zeros(size)
    span_b = np.zeros(size)
    lagging_span = np.zeros(size)
    trend = np.zeros(size)
    
    # 転換線と基準線の計算
    for i in range(max(conversion_period, base_period) - 1, size):
        if i >= conversion_period - 1:
            high_val = np.max(high[i-conversion_period+1:i+1])
            low_val = np.min(low[i-conversion_period+1:i+1])
            conversion_line[i] = (high_val + low_val) / 2
            
        if i >= base_period - 1:
            high_val = np.max(high[i-base_period+1:i+1])
            low_val = np.min(low[i-base_period+1:i+1])
            base_line[i] = (high_val + low_val) / 2
    
    # スパンAとスパンBの計算
    for i in range(size):
        if i >= max(conversion_period, base_period) - 1:
            span_a[i] = (conversion_line[i] + base_line[i]) / 2
    
    # スパンBの計算
    for i in range(span_b_period - 1, size):
        high_val = np.max(high[i-span_b_period+1:i+1])
        low_val = np.min(low[i-span_b_period+1:i+1])
        span_b[i] = (high_val + low_val) / 2
    
    # 遅行スパンの計算
    for i in range(size):
        if i - displacement >= 0:
            lagging_span[i] = close[i-displacement]
    
    # トレンド判定
    for i in range(max(span_b_period, displacement) - 1, size):
        if close[i] > max(span_a[i], span_b[i]):
            trend[i] = 1
        elif close[i] < min(span_a[i], span_b[i]):
            trend[i] = -1
        else:
            trend[i] = 0
    
    return conversion_line, base_line, span_a, span_b, lagging_span, trend


class SpanModel(Indicator):
    """
    一目均衡表スパンモデルインジケーター（高速化版）
    転換線、基準線、スパンA、スパンB、遅行スパンを使用してトレンドを判断する
    """
    
    def __init__(
        self,
        conversion_period: int = 9,
        base_period: int = 26,
        span_b_period: int = 52,
        displacement: int = 26
    ):
        """
        コンストラクタ
        
        Args:
            conversion_period: 転換線期間
            base_period: 基準線期間
            span_b_period: スパンB期間
            displacement: 先行スパン移動期間
        """
        super().__init__(f"SpanModel({conversion_period}, {base_period}, {span_b_period}, {displacement})")
        self.conversion_period = conversion_period
        self.base_period = base_period
        self.span_b_period = span_b_period
        self.displacement = displacement
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> SpanModelResult:
        """
        スパンモデルを計算する
        
        Args:
            data: 価格データ
        
        Returns:
            スパンモデルの計算結果
        """
        # データの準備
        if isinstance(data, pd.DataFrame):
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
        else:
            high = data[:, 1]  # high
            low = data[:, 2]   # low
            close = data[:, 3] # close
        
        # スパンモデルの計算（高速化版）
        conversion_line, base_line, span_a, span_b, lagging_span, trend = calculate_span_model(
            high, low, close,
            self.conversion_period,
            self.base_period,
            self.span_b_period,
            self.displacement
        )
        
        self._values = trend  # 基底クラスの要件を満たすため
        
        return SpanModelResult(
            conversion_line=conversion_line,
            base_line=base_line,
            span_a=span_a,
            span_b=span_b,
            lagging_span=lagging_span,
            trend=trend
        ) 