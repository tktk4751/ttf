#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd

from ...base_signal import BaseSignal
from ...interfaces.direction import IDirectionSignal
from indicators.hyper_trend import HyperTrend


class HyperTrendDirectionSignal(BaseSignal, IDirectionSignal):
    """
    HyperTrendのトレンド方向シグナル
    
    トレンド方向：
    - アップトレンド: ロング (1)
    - ダウントレンド: ショート (-1)
    """
    
    def __init__(
        self,
        er_period: int = 10,
        max_percentile_length: int = 55,
        min_percentile_length: int = 14,
        max_atr_period: int = 120,
        min_atr_period: int = 5,
        max_multiplier: float = 3.0,
        min_multiplier: float = 1.0
    ):
        """
        コンストラクタ
        
        Args:
            er_period: 効率比の計算期間（デフォルト: 10）
            max_percentile_length: パーセンタイル計算の最大期間（デフォルト: 55）
            min_percentile_length: パーセンタイル計算の最小期間（デフォルト: 14）
            max_atr_period: ATR期間の最大値（デフォルト: 120）
            min_atr_period: ATR期間の最小値（デフォルト: 5）
            max_multiplier: ATR乗数の最大値（デフォルト: 3.0）
            min_multiplier: ATR乗数の最小値（デフォルト: 1.0）
        """
        params = {
            'er_period': er_period,
            'max_percentile_length': max_percentile_length,
            'min_percentile_length': min_percentile_length,
            'max_atr_period': max_atr_period,
            'min_atr_period': min_atr_period,
            'max_multiplier': max_multiplier,
            'min_multiplier': min_multiplier
        }
        super().__init__(
            f"HyperTrendDirection({er_period}, {max_percentile_length}, {min_percentile_length}, "
            f"{max_atr_period}, {min_atr_period}, {max_multiplier}, {min_multiplier})",
            params
        )
        self._hyper_trend = HyperTrend(
            er_period=er_period,
            max_percentile_length=max_percentile_length,
            min_percentile_length=min_percentile_length,
            max_atr_period=max_atr_period,
            min_atr_period=min_atr_period,
            max_multiplier=max_multiplier,
            min_multiplier=min_multiplier
        )
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロング方向, -1: ショート方向)
        """
        # HyperTrendの計算
        result = self._hyper_trend.calculate(data)
        
        # トレンド方向をそのままシグナルとして使用
        return result.trend
    
    def get_lower_band(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        HyperTrendの下限バンドを取得する
        
        Args:
            data: 価格データ（OHLCV）
        
        Returns:
            np.ndarray: 下限バンドの値の配列
        """
        # HyperTrendの計算
        result = self._hyper_trend.calculate(data)
        return result.lower_band 