#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple
import numpy as np
import pandas as pd
from numba import jit

from .indicator import Indicator
from .atr import ATR
from .kama import KaufmanAdaptiveMA, calculate_efficiency_ratio


@dataclass
class AlphaChannelResult:
    """Alphaチャネルの計算結果"""
    middle: np.ndarray  # 中心線（KAMA）
    upper: np.ndarray   # 上限線（KAMA + dynamic_multiplier * ATR）
    lower: np.ndarray   # 下限線（KAMA - dynamic_multiplier * ATR）
    half_upper: np.ndarray  # 中間上限線（KAMA + dynamic_multiplier * 0.5 * ATR）
    half_lower: np.ndarray  # 中間下限線（KAMA - lower_multiplier * 0.5 * ATR）
    atr: np.ndarray     # ATRの値
    er: np.ndarray      # Efficiency Ratio


@jit(nopython=True)
def calculate_dynamic_multiplier(er: np.ndarray, max_multiplier: float, min_multiplier: float) -> np.ndarray:
    """
    効率比に基づいて動的な乗数を計算する（高速化版）
    
    Args:
        er: 効率比の配列
        max_multiplier: 最大乗数
        min_multiplier: 最小乗数
    
    Returns:
        動的な乗数の配列
    """
    # ERが高い（トレンドが強い）ほど乗数は小さくなり、
    # ERが低い（トレンドが弱い/ボラティリティが高い）ほど乗数は大きくなる
    return min_multiplier + (1.0 - er) * (max_multiplier - min_multiplier)


@jit(nopython=True)
def calculate_alpha_channel(
    kama: np.ndarray,
    atr: np.ndarray,
    er: np.ndarray,
    max_multiplier: float,
    min_multiplier: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Alphaチャネルのバンドを計算する（高速化版）
    
    Args:
        kama: KAMAの配列
        atr: ATRの配列
        er: 効率比の配列
        max_multiplier: 最大乗数
        min_multiplier: 最小乗数
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            上限線、下限線、中間上限線、中間下限線の配列
    """
    # 動的な乗数を計算
    multiplier = calculate_dynamic_multiplier(er, max_multiplier, min_multiplier)
    
    # バンドを計算
    upper = kama + (multiplier * atr)
    lower = kama - (multiplier * atr)
    half_upper = kama + (multiplier * 0.5 * atr)
    half_lower = kama - (multiplier * 0.5 * atr)
    
    return upper, lower, half_upper, half_lower


class AlphaChannel(Indicator):
    """
    Alphaチャネル インジケーター
    
    KAMAを中心線として使用し、効率比（ER）に基づいて動的に調整されるATRの倍数でバンドを形成する
    - ERが高い（トレンドが強い）時：バンドは狭くなる（乗数が小さくなる）
    - ERが低い（トレンドが弱い）時：バンドは広くなる（乗数が大きくなる）
    
    - 中心線: KAMA
    - 上限線: KAMA + dynamic_multiplier * ATR
    - 下限線: KAMA - dynamic_multiplier * ATR
    - 中間上限線: KAMA + dynamic_multiplier * 0.5 * ATR
    - 中間下限線: KAMA - dynamic_multiplier * 0.5 * ATR
    """
    
    def __init__(
        self,
        kama_period: int = 10,
        kama_fast: int = 2,
        kama_slow: int = 30,
        atr_period: int = 10,
        max_multiplier: float = 3.0,
        min_multiplier: float = 1.0
    ):
        """
        コンストラクタ
        
        Args:
            kama_period: KAMAの効率比の計算期間（デフォルト: 10）
            kama_fast: KAMAの速い移動平均の期間（デフォルト: 2）
            kama_slow: KAMAの遅い移動平均の期間（デフォルト: 30）
            atr_period: ATRの期間（デフォルト: 10）
            max_multiplier: ATR乗数の最大値（デフォルト: 3.0）
            min_multiplier: ATR乗数の最小値（デフォルト: 1.0）
        """
        super().__init__(f"AlphaChannel({kama_period}, {kama_fast}, {kama_slow}, {atr_period}, {max_multiplier}, {min_multiplier})")
        self.kama = KaufmanAdaptiveMA(kama_period, kama_fast, kama_slow)
        self.atr = ATR(atr_period)
        self.kama_period = kama_period
        self.max_multiplier = max_multiplier
        self.min_multiplier = min_multiplier
        self._result = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Alphaチャネルを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'high'と'low'と'close'カラムが必要
        
        Returns:
            中心線（KAMA）の値を返す
        """
        try:
            # データの検証と変換
            if isinstance(data, pd.DataFrame):
                if not all(col in data.columns for col in ['high', 'low', 'close']):
                    raise ValueError("DataFrameには'high'と'low'と'close'カラムが必要です")
                close = data['close'].values
            else:
                if data.ndim == 2:
                    close = data[:, 3]  # close
                else:
                    close = data
            
            # KAMAとATRの計算
            kama_values = self.kama.calculate(data)
            atr_values = self.atr.calculate(data)
            
            if atr_values is None:
                return None
            
            # 効率比（ER）の計算
            length = len(close)
            er = np.full(length, np.nan)
            
            for i in range(self.kama_period, length):
                # 価格変化とボラティリティの計算
                change = close[i] - close[i-self.kama_period]
                volatility = np.sum(np.abs(np.diff(close[i-self.kama_period:i+1])))
                # 効率比の計算（KAMAクラスの関数を使用）
                er[i] = calculate_efficiency_ratio(
                    np.array([change]),
                    np.array([volatility])
                )[0]
            
            # バンドの計算（高速化版）
            upper, lower, half_upper, half_lower = calculate_alpha_channel(
                kama_values,
                atr_values,
                er,
                self.max_multiplier,
                self.min_multiplier
            )
            
            self._result = AlphaChannelResult(
                middle=kama_values,
                upper=upper,
                lower=lower,
                half_upper=half_upper,
                half_lower=half_lower,
                atr=atr_values,
                er=er
            )
            
            self._values = kama_values  # 基底クラスの要件を満たすため
            return kama_values
            
        except Exception:
            return None
    
    def get_bands(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        すべてのバンドの値を取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
                (中心線, 上限線, 下限線, 中間上限線, 中間下限線)の値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return (
            self._result.middle,
            self._result.upper,
            self._result.lower,
            self._result.half_upper,
            self._result.half_lower
        )
    
    def get_atr(self) -> np.ndarray:
        """
        ATRの値を取得する
        
        Returns:
            np.ndarray: ATRの値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.atr
    
    def get_efficiency_ratio(self) -> np.ndarray:
        """
        効率比（ER）の値を取得する
        
        Returns:
            np.ndarray: 効率比の値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.er 