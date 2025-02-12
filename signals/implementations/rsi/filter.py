#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd

from ...base_signal import BaseSignal
from indicators.rsi import RSI

class RSIRangeFilterSignal(BaseSignal):
    """
    RSIレンジフィルターシグナル
    
    RSIが40以上60未満のときにレンジ相場と判断し、1を返す
    それ以外のときはトレンド相場と判断し、-1を返す
    """
    
    def __init__(self, period: int = 14):
        """
        初期化
        
        Args:
            period: RSIの期間
            lower_threshold: RSIの下限閾値
            upper_threshold: RSIの上限閾値
        """
        super().__init__("RSIRangeFilter", {
            'period': period,
            'lower_threshold': 40,
            'upper_threshold': 60
        })
        self._rsi = RSI(period)
        self.lower_threshold = 40
        self.upper_threshold = 60
        
        # キャッシュ用の変数
        self._data_len = 0
        self._signals = None
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: フィルターシグナル (1: レンジ相場, -1: トレンド相場)
        """
        current_len = len(data)
        
        # データ長が変わった場合のみ再計算
        if self._signals is None or current_len != self._data_len:
            # RSI値の計算
            rsi_values = self._rsi.calculate(data)
            
            # シグナルの生成
            self._signals = np.where(
                (rsi_values >= self.lower_threshold) & (rsi_values < self.upper_threshold),
                1,
                -1
            )
            self._data_len = current_len
        
        return self._signals 