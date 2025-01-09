#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union

import numpy as np
import pandas as pd

from .indicator import Indicator


class MFI(Indicator):
    """
    MFI (Money Flow Index) インジケーター
    価格と出来高を組み合わせたモメンタム指標
    - 80以上: 買われすぎ
    - 20以下: 売られすぎ
    - 中心線: 50
    """
    
    def __init__(self, period: int = 14):
        """
        コンストラクタ
        
        Args:
            period: 期間
        """
        super().__init__(f"MFI({period})")
        self.period = period
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        MFIを計算する
        
        Args:
            data: 価格データ
        
        Returns:
            MFI値の配列
        """
        df = pd.DataFrame(data)
        
        # 典型的価格の計算
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        
        # Raw Money Flow の計算
        raw_money_flow = typical_price * df['volume']
        
        # Money Flow の方向を判定
        money_flow_positive = pd.Series(0.0, index=df.index)
        money_flow_negative = pd.Series(0.0, index=df.index)
        
        # 前日比較で上昇/下降を判定
        direction = typical_price.diff()
        
        # 上昇時のMoney Flow
        money_flow_positive[direction > 0] = raw_money_flow[direction > 0]
        
        # 下降時のMoney Flow
        money_flow_negative[direction < 0] = raw_money_flow[direction < 0]
        
        # 期間内の上昇/下降Money Flowの合計を計算
        positive_sum = money_flow_positive.rolling(window=self.period).sum()
        negative_sum = money_flow_negative.rolling(window=self.period).sum()
        
        # Money Flow Ratio の計算
        money_flow_ratio = positive_sum / negative_sum
        
        # MFIの計算: (Money Flow Ratio / (1 + Money Flow Ratio)) * 100
        self._values = 100 - (100 / (1 + money_flow_ratio))
        
        return self._values.to_numpy() 