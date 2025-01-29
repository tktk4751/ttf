#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd

from signals.base_signal import BaseSignal
from signals.signal_interfaces import IEntrySignal
from indicators.roc import ROC


class ROCEntrySignal(BaseSignal, IEntrySignal):
    """ROCエントリーシグナル
    
    n期間ROCが0以上であれば1（エントリー）、
    0以下であれば-1（エントリーなし）を出力する
    """
    
    def __init__(self, period: int = 21):
        """
        初期化
        
        Args:
            period: ROC期間（デフォルト: 21）
        """
        super().__init__("ROCEntrySignal", {"period": period})
        self.roc = ROC(period=period)
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成
        
        Args:
            data: 価格データ
            
        Returns:
            1（エントリー）または-1（エントリーなし）の配列
        """
        # ROCの計算
        roc_values = self.roc.calculate(data)
        
        # シグナルの生成
        signals = np.where(roc_values >= 0, 1, -1)
        
        return signals 