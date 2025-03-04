#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd
from numba import jit

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.percentile_supertrend import PercentileSupertrend


@jit(nopython=True)
def calculate_breakout_signals(
    close: np.ndarray,
    trend: np.ndarray,
    percentile_length: int
) -> np.ndarray:
    """
    ブレイクアウトシグナルを計算する（高速化版）
    
    Args:
        close: 終値の配列
        trend: トレンド方向の配列
        percentile_length: 初期化期間
    
    Returns:
        シグナルの配列
    """
    length = len(close)
    signals = np.zeros(length, dtype=np.int8)
    
    # 最初の期間はシグナルなし
    signals[:percentile_length] = 0
    
    # トレンド転換時のみシグナルを生成
    for i in range(percentile_length, length):
        if trend[i] != trend[i-1]:
            signals[i] = trend[i]
    
    return signals


class PercentileSupertrendBreakoutEntrySignal(BaseSignal, IEntrySignal):
    """
    25-75パーセンタイルスーパートレンドのブレイクアウトによるエントリーシグナル
    
    エントリー条件：
    - トレンドが上昇から下降に転換：ショートエントリー
    - トレンドが下降から上昇に転換：ロングエントリー
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
        super().__init__(
            f"PercentileSupertrendBreakout({subject}, {multiplier}, {percentile_length})"
        )
        
        # パラメータの設定
        self._params = {
            'subject': subject,
            'multiplier': multiplier,
            'percentile_length': percentile_length
        }
        
        # インジケーターの初期化
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
            シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
        """
        # 25-75パーセンタイルスーパートレンドの計算
        result = self._percentile_supertrend.calculate(data)
        
        # 終値の取得
        close = data['close'].values if isinstance(data, pd.DataFrame) else data[:, 3]
        
        # ブレイクアウトシグナルの計算（高速化版）
        return calculate_breakout_signals(
            close,
            result.trend,
            self._params['percentile_length']
        ) 