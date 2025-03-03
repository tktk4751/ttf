#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd
from numba import jit

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.gvidya_keltner import GVIDYAKeltnerChannel


@jit(nopython=True)
def calculate_breakout_signals(close: np.ndarray, upper: np.ndarray, lower: np.ndarray, period: int) -> np.ndarray:
    """
    ブレイクアウトシグナルを計算する（高速化版）
    
    Args:
        close: 終値の配列
        upper: アッパーバンドの配列
        lower: ロワーバンドの配列
        period: 期間
    
    Returns:
        シグナルの配列
    """
    length = len(close)
    signals = np.zeros(length, dtype=np.int8)
    
    # 最初の期間はシグナルなし
    signals[:period] = 0
    
    # ブレイクアウトの判定
    for i in range(period, length):
        # ロングエントリー: 終値がアッパーバンドを上回る
        if close[i] > upper[i-1]:
            signals[i] = 1
        # ショートエントリー: 終値がロワーバンドを下回る
        elif close[i] < lower[i-1]:
            signals[i] = -1
    
    return signals


class GVIDYAKeltnerBreakoutEntrySignal(BaseSignal, IEntrySignal):
    """
    G-VIDYAケルトナーチャネルのブレイクアウトによるエントリーシグナル
    
    - 現在の終値が前回のアッパーバンドを上回った場合: ロングエントリー (1)
    - 現在の終値が前回のロワーバンドを下回った場合: ショートエントリー (-1)
    """
    
    def __init__(
        self,
        vidya_period: int = 46,
        sd_period: int = 28,
        gaussian_length: int = 4,
        gaussian_sigma: float = 2.0,
        atr_period: int = 14,
        upper_multiplier: float = 1.3,
        lower_multiplier: float = 1.3
    ):
        """
        コンストラクタ
        
        Args:
            vidya_period: VIDYA期間（デフォルト: 46）
            sd_period: 標準偏差の計算期間（デフォルト: 28）
            gaussian_length: ガウシアンフィルターの長さ（デフォルト: 4）
            gaussian_sigma: ガウシアンフィルターのシグマ（デフォルト: 2.0）
            atr_period: ATRの期間（デフォルト: 14）
            upper_multiplier: アッパーバンドのATR乗数（デフォルト: 1.3）
            lower_multiplier: ロワーバンドのATR乗数（デフォルト: 1.3）
        """
        params = {
            'vidya_period': vidya_period,
            'sd_period': sd_period,
            'gaussian_length': gaussian_length,
            'gaussian_sigma': gaussian_sigma,
            'atr_period': atr_period,
            'upper_multiplier': upper_multiplier,
            'lower_multiplier': lower_multiplier
        }
        super().__init__(
            f"GVIDYAKeltnerBreakout({vidya_period}, {sd_period}, {gaussian_length}, {gaussian_sigma}, {atr_period}, {upper_multiplier}, {lower_multiplier})",
            params
        )
        self._gvidya_keltner = GVIDYAKeltnerChannel(
            vidya_period=vidya_period,
            sd_period=sd_period,
            gaussian_length=gaussian_length,
            gaussian_sigma=gaussian_sigma,
            atr_period=atr_period,
            upper_multiplier=upper_multiplier,
            lower_multiplier=lower_multiplier
        )
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
        """
        # G-VIDYAケルトナーチャネルの計算
        result = self._gvidya_keltner.calculate(data)
        if result is None:
            return np.zeros(len(data))
        
        # 終値の取得
        close = data['close'].values if isinstance(data, pd.DataFrame) else data[:, 3]
        
        # バンドの取得
        _, upper, lower, _, _ = self._gvidya_keltner.get_bands()
        
        # ブレイクアウトシグナルの計算（高速化版）
        return calculate_breakout_signals(
            close,
            upper,
            lower,
            self._params['vidya_period']
        ) 