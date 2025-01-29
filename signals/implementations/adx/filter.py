#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd

from ...base_signal import BaseSignal
from indicators.adx import ADX

class ADXFilterSignal(BaseSignal):
    """
    ADXフィルターシグナル
    
    ADXが閾値を上回っているときにトレンド相場と判断し、1を返す
    """
    
    def __init__(self, period: int = 14, threshold: float = 25.0):
        """
        初期化
        
        Args:
            period: ADXの期間
            threshold: ADXの閾値
        """
        super().__init__("ADXFilter", {
            'period': period,
            'threshold': threshold
        })
        self._adx = ADX(period)
        self.threshold = threshold
        
        # キャッシュ用の変数
        self._data_len = 0
        self._signals = None
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: フィルターシグナル (1: トレンド相場, 0: レンジ相場)
        """
        current_len = len(data)
        
        # データ長が変わった場合のみ再計算
        if self._signals is None or current_len != self._data_len:
            # ADX値の計算
            adx_result = self._adx.calculate(data)
            
            # シグナルの生成
            self._signals = np.where(adx_result.adx > self.threshold, 1, 0)
            self._data_len = current_len
        
        return self._signals 