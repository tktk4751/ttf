#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union
import numpy as np
import pandas as pd
from numba import jit

from .indicator import Indicator


@jit(nopython=True)
def calculate_cumulative_er(
    src: np.ndarray,
    period: int
) -> np.ndarray:
    """
    累積的な効率比（Cumulative Efficiency Ratio）を計算する（高速化版）
    
    Args:
        src: ソース価格の配列
        period: 計算期間
    
    Returns:
        累積的な効率比の配列（0-1の範囲）
        - 1に近いほど効率的な価格変動（強いトレンド）
        - 0に近いほど非効率な価格変動（レンジ・ノイズ）
    """
    length = len(src)
    er = np.zeros(length)
    diff = np.zeros(length)
    signal = np.zeros(length)
    noise = np.zeros(length)
    
    # 差分の計算（ノイズの基礎）
    for i in range(1, length):
        diff[i] = abs(src[i] - src[i-1])
    
    # シグナルの計算（方向性のある動き）
    for i in range(period, length):
        signal[i] = abs(src[i] - src[i-period])
    
    # 累積的なノイズの計算
    for i in range(period, length):
        noise_sum = 0.0
        for j in range(period):
            noise_sum += diff[i-j]
        noise[i] = noise[i-1] + diff[i] - diff[i-period] if i > period else noise_sum
    
    # 効率比の計算
    for i in range(period, length):
        if noise[i] != 0:
            er[i] = min(signal[i] / noise[i], 1.0)  # 1.0を超えないように制限
        else:
            er[i] = 0.0  # ノイズがゼロの場合
    
    return er


class CumulativeER(Indicator):
    """
    累積的な効率比（Cumulative Efficiency Ratio）インジケーター
    
    価格変動の効率性を累積的に測定する指標
    - 1に近いほど効率的な価格変動（強いトレンド）
    - 0に近いほど非効率な価格変動（レンジ・ノイズ）
    
    特徴：
    - 累積的なノイズ計算により、市場の微細な変化を検出
    - 連続的な市場状態の変化を追跡
    - アダプティブATRで使用されている手法をベースに改良
    
    使用方法：
    - 0.618以上: 強いトレンド（効率的な価格変動）
    - 0.382-0.618: 中間的な状態
    - 0.382以下: レンジ・ノイズ（非効率な価格変動）
    """
    
    def __init__(self, period: int = 14):
        """
        コンストラクタ
        
        Args:
            period: 計算期間（デフォルト: 14）
        """
        super().__init__(f"CumulativeER({period})")
        self.period = period
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        累積的な効率比を計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'high'と'low'カラムが必要
        
        Returns:
            累積的な効率比の配列（0-1の範囲）
        """
        try:
            # データの検証と変換
            if isinstance(data, pd.DataFrame):
                if not all(col in data.columns for col in ['high', 'low']):
                    raise ValueError("DataFrameには'high'と'low'カラムが必要です")
                high = data['high'].values
                low = data['low'].values
                # HL2の計算
                src = (high + low) / 2
            else:
                if data.ndim != 2 or data.shape[1] < 4:
                    raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")
                high = data[:, 1]  # high
                low = data[:, 2]   # low
                src = (high + low) / 2
            
            # データ長の検証
            data_length = len(src)
            self._validate_period(self.period, data_length)
            
            # 累積的な効率比の計算（高速化版）
            self._values = calculate_cumulative_er(src, self.period)
            
            return self._values
            
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            return None 