#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd
from numba import jit

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.gvidya import GVIDYA


@jit(nopython=True)
def calculate_crossover_signals(short_gvidya: np.ndarray, long_gvidya: np.ndarray) -> np.ndarray:
    """クロスオーバーシグナルを計算する（高速化版）"""
    length = len(short_gvidya)
    signals = np.zeros(length, dtype=np.int8)
    
    # 最初の要素はクロスの判定ができないのでシグナルなし
    signals[0] = 0
    
    # クロスオーバーの検出
    for i in range(1, length):
        # ゴールデンクロス（短期が長期を上抜け）
        if short_gvidya[i-1] <= long_gvidya[i-1] and short_gvidya[i] > long_gvidya[i]:
            signals[i] = 1
        # デッドクロス（短期が長期を下抜け）
        elif short_gvidya[i-1] >= long_gvidya[i-1] and short_gvidya[i] < long_gvidya[i]:
            signals[i] = -1
    
    return signals


class GVIDYACrossoverEntrySignal(BaseSignal, IEntrySignal):
    """
    G-VIDYAクロスオーバーを使用したエントリーシグナル
    - 短期G-VIDYA > 長期G-VIDYA: ロングエントリー (1)
    - 短期G-VIDYA < 長期G-VIDYA: ショートエントリー (-1)
    """
    
    def __init__(
        self,
        short_vidya_period: int = 9,
        long_vidya_period: int = 21,
        short_sd_period: int = 28,
        long_sd_period: int = 28,
        gaussian_length: int = 4,
        gaussian_sigma: float = 2.0
    ):
        """
        コンストラクタ
        
        Args:
            short_vidya_period: 短期G-VIDYAのVIDYA期間
            long_vidya_period: 長期G-VIDYAのVIDYA期間
            short_sd_period: 短期G-VIDYAの標準偏差の計算期間
            long_sd_period: 長期G-VIDYAの標準偏差の計算期間
            gaussian_length: ガウシアンフィルターの長さ
            gaussian_sigma: ガウシアンフィルターのシグマ
        """
        params = {
            'short_vidya_period': short_vidya_period,
            'long_vidya_period': long_vidya_period,
            'short_sd_period': short_sd_period,
            'long_sd_period': long_sd_period,
            'gaussian_length': gaussian_length,
            'gaussian_sigma': gaussian_sigma
        }
        super().__init__(f"GVIDYACrossover({short_vidya_period}, {long_vidya_period})", params)
        
        # G-VIDYAインジケーターの初期化
        self._short_gvidya = GVIDYA(
            vidya_period=short_vidya_period,
            sd_period=short_sd_period,
            gaussian_length=gaussian_length,
            gaussian_sigma=gaussian_sigma
        )
        self._long_gvidya = GVIDYA(
            vidya_period=long_vidya_period,
            sd_period=long_sd_period,
            gaussian_length=gaussian_length,
            gaussian_sigma=gaussian_sigma
        )
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
        """
        # G-VIDYAの計算
        short_gvidya = self._short_gvidya.calculate(data)
        long_gvidya = self._long_gvidya.calculate(data)
        
        if short_gvidya is None or long_gvidya is None:
            return np.zeros(len(data))
        
        # クロスオーバーシグナルの計算（高速化版）
        return calculate_crossover_signals(short_gvidya, long_gvidya) 