#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd

from ...base_signal import BaseSignal
from ...interfaces.filter import IFilterSignal
from indicators.choppiness import ChoppinessIndex

class ChopFilterSignal(BaseSignal, IFilterSignal):
    """
    チョピネスインデックスを使用したフィルターシグナル
    - CHOP >= solid: レンジ相場 (-1)
    - CHOP < solid: トレンド相場 (1)
    """
    
    def __init__(self, period: int = 14, solid: Dict[str, Any] = None):
        """
        コンストラクタ
        
        Args:
            period: チョピネスインデックスの期間
            solid: パラメータ辞書
                - chop_solid: チョピネスインデックスのしきい値
        """
        params = {
            'period': period,
            'solid': solid or {
                'chop_solid': 50  # デフォルトのしきい値
            }
        }
        super().__init__(f"ChopFilter({period})", params)
        self._chop = ChoppinessIndex(period)
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: トレンド相場, -1: レンジ相場)
        """
        chop_values = self._chop.calculate(data)
        
        # シグナルの生成
        solid = self._params['solid']
        signals = np.where(chop_values >= solid['chop_solid'], -1, 1)
        
        return signals 