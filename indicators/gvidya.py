#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union
import numpy as np
import pandas as pd
from numba import jit

from .indicator import Indicator


@jit(nopython=True)
def calculate_gaussian_filter(src: np.ndarray, length: int, sigma: float) -> np.ndarray:
    """
    ガウシアンフィルターを計算する（高速化版）
    
    Args:
        src: 入力データの配列
        length: フィルターの長さ
        sigma: ガウシアン分布の標準偏差
    
    Returns:
        np.ndarray: フィルター適用後の配列
    """
    result = np.zeros_like(src)
    weights = np.zeros(length)
    
    # ガウシアンウェイトの計算
    for i in range(length):
        x = i - (length - 1) / 2
        weights[i] = np.exp(-0.5 * (x / sigma) ** 2)
    
    weights_sum = np.sum(weights)
    
    # フィルターの適用
    for i in range(len(src)):
        if i < length - 1:
            result[i] = src[i]
            continue
            
        weighted_sum = 0.0
        for j in range(length):
            weighted_sum += src[i-j] * weights[j]
        
        result[i] = weighted_sum / weights_sum
    
    return result


@jit(nopython=True)
def calculate_vidya(src: np.ndarray, sd_long: np.ndarray, sd_short: np.ndarray, length: int) -> np.ndarray:
    """
    VIDYAを計算する（高速化版）
    
    Args:
        src: 入力データの配列
        sd_long: 長期標準偏差
        sd_short: 短期標準偏差
        length: VIDYAの期間
    
    Returns:
        np.ndarray: VIDYA値の配列
    """
    result = np.zeros_like(src)
    sc = 2.0 / (length + 1)
    
    # 最初の値を設定
    result[0] = src[0]
    
    # VIDYAの計算
    for i in range(1, len(src)):
        if sd_short[i] != 0:
            sd_ratio = sd_long[i] / sd_short[i]
        else:
            sd_ratio = 0
            
        alpha = sd_ratio * sc
        result[i] = alpha * src[i] + (1 - alpha) * result[i-1]
    
    return result


class GVIDYA(Indicator):
    """
    G-VIDYA (Gaussian Variable Index Dynamic Average) インジケーター
    
    ガウシアンフィルターを適用した後、VIDYAを計算する
    """
    
    def __init__(
        self,
        vidya_period: int = 46,
        sd_period: int = 28,
        gaussian_length: int = 4,
        gaussian_sigma: float = 2.0
    ):
        """
        コンストラクタ
        
        Args:
            vidya_period: VIDYA期間（デフォルト: 46）
            sd_period: 標準偏差の計算期間（デフォルト: 28）
            gaussian_length: ガウシアンフィルターの長さ（デフォルト: 4）
            gaussian_sigma: ガウシアンフィルターのシグマ（デフォルト: 2.0）
        """
        super().__init__(f"GVIDYA({vidya_period}, {sd_period}, {gaussian_length}, {gaussian_sigma})")
        self.vidya_period = vidya_period
        self.sd_period = sd_period
        self.gaussian_length = gaussian_length
        self.gaussian_sigma = gaussian_sigma
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        G-VIDYAを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'close'カラムが必要
        
        Returns:
            np.ndarray: G-VIDYA値の配列
        """
        try:
            # データの検証と変換
            if isinstance(data, pd.DataFrame):
                if 'close' not in data.columns:
                    raise ValueError("DataFrameには'close'カラムが必要です")
                close = data['close'].values
            else:
                close = data
            
            # ガウシアンフィルターの適用
            gaussian_filtered = calculate_gaussian_filter(
                close,
                self.gaussian_length,
                self.gaussian_sigma
            )
            
            # 標準偏差の計算
            sd_long = np.zeros_like(close)
            sd_short = np.zeros_like(close)
            
            for i in range(len(close)):
                if i >= self.vidya_period:
                    sd_long[i] = np.std(gaussian_filtered[i-self.vidya_period+1:i+1])
                if i >= self.sd_period:
                    sd_short[i] = np.std(gaussian_filtered[i-self.sd_period+1:i+1])
            
            # VIDYAの計算
            self._values = calculate_vidya(
                gaussian_filtered,
                sd_long,
                sd_short,
                self.vidya_period
            )
            
            return self._values
            
        except Exception:
            return None 