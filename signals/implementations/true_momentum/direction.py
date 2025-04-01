#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd
from numba import jit

from ...base_signal import BaseSignal
from ...interfaces.direction import IDirectionSignal
from indicators.true_momentum import TrueMomentum


@jit(nopython=True)
def calculate_direction_signals(momentum: np.ndarray, period: int) -> np.ndarray:
    """
    トゥルーモメンタムの方向シグナルを計算（高速化版）
    
    Args:
        momentum: モメンタム値の配列
        period: 初期期間（シグナルを生成しない期間）
    
    Returns:
        シグナルの配列 (1: ロング方向, -1: ショート方向, 0: シグナルなし)
    """
    length = len(momentum)
    signals = np.zeros(length, dtype=np.int8)
    
    # シグナルの生成
    for i in range(period, length):
        if momentum[i] > 0:
            signals[i] = 1  # ロング方向
        elif momentum[i] < 0:
            signals[i] = -1  # ショート方向
    
    return signals


class TrueMomentumDirectionSignal(BaseSignal, IDirectionSignal):
    """
    トゥルーモメンタムによる方向シグナル
    
    - モメンタムが正: ロング方向 (1)
    - モメンタムが負: ショート方向 (-1)
    """
    
    def __init__(
        self,
        period: int = 20,
        max_std_mult: float = 2.0,
        min_std_mult: float = 1.0,
        max_kama_slow: int = 55,
        min_kama_slow: int = 30,
        max_kama_fast: int = 13,
        min_kama_fast: int = 2,
        max_atr_period: int = 120,
        min_atr_period: int = 13,
        max_atr_mult: float = 3.0,
        min_atr_mult: float = 1.0,
        max_momentum_period: int = 100,
        min_momentum_period: int = 20
    ):
        """
        コンストラクタ
        
        Args:
            period: 計算期間（KAMA、ボリンジャーバンド、トレンドアルファに共通）（デフォルト: 20）
            max_std_mult: 標準偏差乗数の最大値（デフォルト: 2.0）
            min_std_mult: 標準偏差乗数の最小値（デフォルト: 1.0）
            max_kama_slow: KAMAの遅い移動平均の最大期間（デフォルト: 55）
            min_kama_slow: KAMAの遅い移動平均の最小期間（デフォルト: 30）
            max_kama_fast: KAMAの速い移動平均の最大期間（デフォルト: 13）
            min_kama_fast: KAMAの速い移動平均の最小期間（デフォルト: 2）
            max_atr_period: ATR期間の最大値（デフォルト: 120）
            min_atr_period: ATR期間の最小値（デフォルト: 13）
            max_atr_mult: ATR乗数の最大値（デフォルト: 3.0）
            min_atr_mult: ATR乗数の最小値（デフォルト: 1.0）
            max_momentum_period: モメンタム計算の最大期間（デフォルト: 100）
            min_momentum_period: モメンタム計算の最小期間（デフォルト: 20）
        """
        params = {
            'period': period,
            'max_std_mult': max_std_mult,
            'min_std_mult': min_std_mult,
            'max_kama_slow': max_kama_slow,
            'min_kama_slow': min_kama_slow,
            'max_kama_fast': max_kama_fast,
            'min_kama_fast': min_kama_fast,
            'max_atr_period': max_atr_period,
            'min_atr_period': min_atr_period,
            'max_atr_mult': max_atr_mult,
            'min_atr_mult': min_atr_mult,
            'max_momentum_period': max_momentum_period,
            'min_momentum_period': min_momentum_period
        }
        super().__init__(f"TrueMomentumDirection({period})", params)
        
        self._indicator = TrueMomentum(
            period=period,
            max_std_mult=max_std_mult,
            min_std_mult=min_std_mult,
            max_kama_slow=max_kama_slow,
            min_kama_slow=min_kama_slow,
            max_kama_fast=max_kama_fast,
            min_kama_fast=min_kama_fast,
            max_atr_period=max_atr_period,
            min_atr_period=min_atr_period,
            max_atr_mult=max_atr_mult,
            min_atr_mult=min_atr_mult,
            max_momentum_period=max_momentum_period,
            min_momentum_period=min_momentum_period
        )
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロング方向, -1: ショート方向)
        """
        try:
            # トゥルーモメンタムの計算
            momentum = self._indicator.calculate(data)
            
            # シグナルの計算（高速化）
            period = self._params['period']
            
            # Numba最適化関数を呼び出し
            return calculate_direction_signals(momentum, period)
            
        except Exception as e:
            print(f"方向シグナル生成中にエラーが発生しました: {e}")
            # エラーが発生した場合は0埋めのシグナルを返す
            return np.zeros(len(data)) 