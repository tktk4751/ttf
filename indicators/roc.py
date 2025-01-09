#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union

import numpy as np
import pandas as pd

from .indicator import Indicator


class ROC(Indicator):
    """
    ROC (Rate of Change) インジケーター
    価格の変化率を測定する
    - プラス: 上昇モメンタム
    - マイナス: 下降モメンタム
    - ゼロライン: トレンド転換の可能性
    """
    
    def __init__(self, period: int = 12):
        """
        コンストラクタ
        
        Args:
            period: 期間
        """
        super().__init__(f"ROC({period})")
        self.period = period
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        ROCを計算する
        
        Args:
            data: 価格データ
        
        Returns:
            ROC値の配列
        """
        df = pd.DataFrame(data)
        close = df['close']
        
        # ROCの計算: ((現在値 - n期前の値) / n期前の値) * 100
        self._values = ((close - close.shift(self.period)) / 
                       close.shift(self.period) * 100)
        
        return self._values.to_numpy() 