#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.span_model import SpanModel

class SpanModelEntrySignal(BaseSignal, IEntrySignal):
    """
    スパンモデルを使用したエントリーシグナル
    - トレンドが-1から1に変化: ロングエントリー (1)
    - トレンドが1から-1に変化: ショートエントリー (-1)
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
        params = {
            'conversion_period': conversion_period,
            'base_period': base_period,
            'span_b_period': span_b_period,
            'displacement': displacement
        }
        super().__init__(
            f"SpanModelEntry({conversion_period}, {base_period}, {span_b_period}, {displacement})",
            params
        )
        self._span_model = SpanModel(
            conversion_period,
            base_period,
            span_b_period,
            displacement
        )
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロングエントリー, -1: ショートエントリー, 0: シグナルなし)
        """
        # スパンモデルの計算
        span_result = self._span_model.calculate(data)
        trend = span_result.trend
        
        # シグナルの初期化
        signals = np.zeros(len(data))
        
        # 2番目の要素からシグナルを計算
        for i in range(1, len(data)):
            # ロングエントリー（トレンドが-1から1に変化）
            if trend[i-1] == -1 and trend[i] == 1:
                signals[i] = 1
            # ショートエントリー（トレンドが1から-1に変化）
            elif trend[i-1] == 1 and trend[i] == -1:
                signals[i] = -1
        
        return signals 