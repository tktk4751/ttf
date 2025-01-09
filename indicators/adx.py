#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd

from .indicator import Indicator


@dataclass
class ADXResult:
    """ADXの計算結果"""
    adx: np.ndarray    # ADX値
    plus_di: np.ndarray  # +DI値
    minus_di: np.ndarray  # -DI値


class ADX(Indicator):
    """
    ADX (Average Directional Index) インジケーター
    トレンドの強さを測定する
    - 25以上: 強いトレンド
    - 20以下: トレンドなし
    """
    
    def __init__(self, period: int = 14):
        """
        コンストラクタ
        
        Args:
            period: 期間
        """
        super().__init__(f"ADX({period})")
        self.period = period
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> ADXResult:
        """
        ADXを計算する
        
        Args:
            data: 価格データ
        
        Returns:
            ADXの計算結果
        """
        df = pd.DataFrame(data)
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range (TR)の計算
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # +DM, -DMの計算
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0
        )
        minus_dm = np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0
        )
        
        # 平滑化
        tr_smooth = self._smooth(tr)
        plus_dm_smooth = self._smooth(pd.Series(plus_dm))
        minus_dm_smooth = self._smooth(pd.Series(minus_dm))
        
        # +DI, -DIの計算
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        
        # DXの計算
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # ADXの計算（DXの平滑化）
        adx = dx.rolling(window=self.period).mean()
        
        self._values = adx  # 基底クラスの要件を満たすため
        
        return ADXResult(
            adx=adx.to_numpy(),
            plus_di=plus_di.to_numpy(),
            minus_di=minus_di.to_numpy()
        )
    
    def _smooth(self, series: pd.Series) -> pd.Series:
        """
        Wilderの平滑化を適用
        
        Args:
            series: 平滑化する系列
        
        Returns:
            平滑化された系列
        """
        result = series.copy()
        result.iloc[self.period-1] = series.iloc[:self.period].mean()
        
        for i in range(self.period, len(series)):
            result.iloc[i] = (
                result.iloc[i-1] * (self.period - 1) + series.iloc[i]
            ) / self.period
        
        return result 