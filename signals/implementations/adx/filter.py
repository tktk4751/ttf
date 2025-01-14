#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd

from ...base_signal import BaseSignal
from ...interfaces.filter import IFilterSignal
from indicators.adx import ADX

class ADXFilterSignal(BaseSignal, IFilterSignal):
    """
    ADXを使用したフィルターシグナル
    - ADX >= solid: トレンド相場 (1)
    - ADX < solid: レンジ相場 (-1)
    """
    
    def __init__(self, period: int = 14, solid: Dict[str, Any] = None):
        """
        コンストラクタ
        
        Args:
            period: ADXの期間
            solid: パラメータ辞書
                - adx_solid: ADXのしきい値
        """
        params = {
            'period': period,
            'solid': solid or {
                'adx_solid': 40  # デフォルトのしきい値
            }
        }
        super().__init__(f"ADXFilter({period})", params)
        self._adx = ADX(period)
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: トレンド相場, -1: レンジ相場)
        """
        # ADXの計算
        adx_values = self._adx.calculate(data).adx
        
        # シグナルの生成
        solid = self._params['solid']
        signals = np.where(adx_values >= solid['adx_solid'], 1, -1)
        
        return signals 