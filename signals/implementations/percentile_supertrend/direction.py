#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd

from ...base_signal import BaseSignal
from ...interfaces.direction import IDirectionSignal
from indicators.percentile_supertrend import PercentileSupertrend


class PercentileSupertrendDirectionSignal(BaseSignal, IDirectionSignal):
    """
    25-75パーセンタイルスーパートレンドを使用した方向性シグナル
    
    方向性の判断：
    - トレンドが上昇（1）：ロング方向
    - トレンドが下降（-1）：ショート方向
    """
    
    def __init__(
        self,
        subject: int = 14,
        multiplier: float = 1.0,
        percentile_length: int = 27
    ):
        """
        コンストラクタ
        
        Args:
            subject: ATR期間（デフォルト: 14）
            multiplier: ATRの乗数（デフォルト: 1.0）
            percentile_length: パーセンタイル計算期間（デフォルト: 27）
        """
        params = {
            'subject': subject,
            'multiplier': multiplier,
            'percentile_length': percentile_length
        }
        super().__init__(
            f"PercentileSupertrendDirection({subject}, {multiplier}, {percentile_length})",
            params
        )
        self._percentile_supertrend = PercentileSupertrend(
            subject=subject,
            multiplier=multiplier,
            percentile_length=percentile_length
        )
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロング方向, -1: ショート方向)
        """
        # 25-75パーセンタイルスーパートレンドの計算
        result = self._percentile_supertrend.calculate(data)
        
        # トレンド方向をそのままシグナルとして使用
        return result.trend
    
    def get_lower_band(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        25-75パーセンタイルスーパートレンドの下限バンドを取得する
        
        Args:
            data: 価格データ（OHLCV）
        
        Returns:
            np.ndarray: 下限バンドの値の配列
        """
        # 25-75パーセンタイルスーパートレンドの計算
        result = self._percentile_supertrend.calculate(data)
        return result.lower_band 