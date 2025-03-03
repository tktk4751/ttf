#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd

from ...base_signal import BaseSignal
from ...interfaces.direction import IDirectionSignal
from indicators.span_model import SpanModel

class SpanModelDirectionSignal(BaseSignal, IDirectionSignal):
    """
    スパンモデルを使用した方向性シグナル
    - 価格がスパンA,Bの上: ロング方向 (1)
    - 価格がスパンA,Bの下: ショート方向 (-1)
    - その他: ニュートラル (0)
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
            f"SpanModelDirection({conversion_period}, {base_period}, {span_b_period}, {displacement})",
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
            シグナルの配列 (1: ロング方向, -1: ショート方向, 0: ニュートラル)
        """
        # スパンモデルの計算
        span_result = self._span_model.calculate(data)
        
        # トレンド方向をそのままシグナルとして使用
        return span_result.trend 