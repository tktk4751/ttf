#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple

import numpy as np
import pandas as pd

from .indicator import Indicator
from .atr import ATR


@dataclass
class SupertrendResult:
    """スーパートレンドの計算結果"""
    upper_band: np.ndarray  # 上側のバンド価格
    lower_band: np.ndarray  # 下側のバンド価格
    trend: np.ndarray      # トレンド方向（1=上昇トレンド、-1=下降トレンド）


class Supertrend(Indicator):
    """
    スーパートレンドインジケーター
    ATRを使用して、トレンドの方向と強さを判断する
    """
    
    def __init__(self, period: int = 10, multiplier: float = 3.0):
        """
        コンストラクタ
        
        Args:
            period: ATR期間
            multiplier: ATRの乗数
        """
        super().__init__(f"Supertrend({period}, {multiplier})")
        self.period = period
        self.multiplier = multiplier
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> SupertrendResult:
        """
        スーパートレンドを計算する
        
        Args:
            data: 価格データ
        
        Returns:
            スーパートレンドの計算結果
        """
        df = pd.DataFrame(data)
        high = df['high']
        low = df['low']
        close = df['close']

        atr = ATR(self.period).calculate(data)
        
        # 基準となるバンドの計算
        hl_avg = (high + low) / 2
        final_upper_band = hl_avg + self.multiplier * atr
        final_lower_band = hl_avg - self.multiplier * atr
        
        # トレンド方向の配列を初期化（1: 上昇トレンド、-1: 下降トレンド）
        trend = np.zeros(len(close))
        
        # 最初の値を設定
        trend[0] = 1 if close.iloc[0] > final_upper_band.iloc[0] else -1
        
        # バンドとトレンドの計算
        for i in range(1, len(close)):
            curr_close = close.iloc[i]
            
            # トレンド方向の判定
            if curr_close > final_upper_band.iloc[i-1]:
                trend[i] = 1
            elif curr_close < final_lower_band.iloc[i-1]:
                trend[i] = -1
            else:
                trend[i] = trend[i-1]
                
                # バンドの調整
                if trend[i] == 1:
                    if final_lower_band.iloc[i] < final_lower_band.iloc[i-1]:
                        final_lower_band.iloc[i] = final_lower_band.iloc[i-1]
                else:
                    if final_upper_band.iloc[i] > final_upper_band.iloc[i-1]:
                        final_upper_band.iloc[i] = final_upper_band.iloc[i-1]
        
        # トレンドに基づいてバンドを調整
        upper_band = final_upper_band.copy()
        lower_band = final_lower_band.copy()
        
        for i in range(len(close)):
            if trend[i] == 1:
                upper_band.iloc[i] = np.nan
            else:
                lower_band.iloc[i] = np.nan
        
        self._values = trend  # 基底クラスの要件を満たすため
        
        return SupertrendResult(
            upper_band=upper_band.to_numpy(),
            lower_band=lower_band.to_numpy(),
            trend=trend
        ) 