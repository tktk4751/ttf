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
class TrendAlphaResult:
    """TrendAlphaの計算結果"""
    middle: np.ndarray  # 中心線（KAMA）
    upper: np.ndarray   # 上限線（KAMA + dynamic_multiplier * ATR）
    lower: np.ndarray   # 下限線（KAMA - dynamic_multiplier * ATR）
    half_upper: np.ndarray  # 中間上限線（KAMA + dynamic_multiplier * 0.5 * ATR）
    half_lower: np.ndarray  # 中間下限線（KAMA - lower_multiplier * 0.5 * ATR）
    atr: np.ndarray     # ATRの値
    er: np.ndarray      # Efficiency Ratio
    dynamic_period: np.ndarray  # 動的ATR期間


@jit(nopython=True)
def calculate_tr(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    True Rangeを計算する（高速化版）
    """
    tr = np.zeros_like(high)
    tr[0] = high[0] - low[0]
    
    for i in range(1, len(high)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    return tr

@jit(nopython=True)
def calculate_dynamic_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                         periods: np.ndarray, max_period: int) -> np.ndarray:
    """
    動的なATRを計算する（高速化版）
    """
    length = len(high)
    tr = calculate_tr(high, low, close)
    atr = np.full(length, np.nan)
    
    for i in range(max_period, length):
        if np.isnan(periods[i]):
            continue
        period = int(periods[i])
        if period < 1:
            continue
        atr[i] = np.mean(tr[i-period+1:i+1])
    
    return atr

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
    return min_multiplier + (1.0 - er) * (max_multiplier - min_multiplier)


@jit(nopython=True)
def calculate_dynamic_period(er: np.ndarray, max_period: int, min_period: int) -> np.ndarray:
    """
    効率比に基づいて動的なATR期間を計算する（高速化版）
    
    Args:
        er: 効率比の配列
        max_period: 最大期間
        min_period: 最小期間
    
    Returns:
        動的な期間の配列
    """
    # ERが高い（トレンドが強い）ほど期間は短く、
    # ERが低い（トレンドが弱い）ほど期間は長くなる
    periods = min_period + (1.0 - er) * (max_period - min_period)
    return np.round(periods).astype(np.int32)


@jit(nopython=True)
def calculate_dynamic_kama_periods(er: np.ndarray, max_slow: int, min_slow: int, max_fast: int, min_fast: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    効率比に基づいて動的なKAMAのfast/slow期間を計算する（高速化版）
    
    Args:
        er: 効率比の配列
        max_slow: 遅い移動平均の最大期間
        min_slow: 遅い移動平均の最小期間
        max_fast: 速い移動平均の最大期間
        min_fast: 速い移動平均の最小期間
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: 動的なfast期間とslow期間の配列
    """
    # ERが高い（トレンドが強い）ほど期間は短く、
    # ERが低い（トレンドが弱い）ほど期間は長くなる
    fast_periods = min_fast + (1.0 - er) * (max_fast - min_fast)
    slow_periods = min_slow + (1.0 - er) * (max_slow - min_slow)
    
    return np.round(fast_periods).astype(np.int32), np.round(slow_periods).astype(np.int32)


class TrendAlpha(Indicator):
    """
    TrendAlpha インジケーター
    
    KAMAを中心線として使用し、効率比（ER）に基づいて動的に調整される
    ATRの倍数とATRの期間でバンドを形成する
    - ERが高い（トレンドが強い）時：
        - バンドは狭くなる（乗数が小さくなる）
        - ATR期間は短くなる（より敏感に反応）
        - KAMAのfast/slow期間は短くなる（より敏感に反応）
    - ERが低い（トレンドが弱い）時：
        - バンドは広くなる（乗数が大きくなる）
        - ATR期間は長くなる（ノイズを軽減）
        - KAMAのfast/slow期間は長くなる（ノイズを軽減）
    """
    
    def __init__(
        self,
        kama_period: int = 10,
        max_kama_slow: int = 55,
        min_kama_slow: int = 30,
        max_kama_fast: int = 13,
        min_kama_fast: int = 2,
        max_atr_period: int = 120,
        min_atr_period: int = 5,
        max_multiplier: float = 3.0,
        min_multiplier: float = 1.0
    ):
        """
        コンストラクタ
        
        Args:
            kama_period: KAMAの効率比の計算期間（デフォルト: 10）
            max_kama_slow: KAMAの遅い移動平均の最大期間（デフォルト: 55）
            min_kama_slow: KAMAの遅い移動平均の最小期間（デフォルト: 30）
            max_kama_fast: KAMAの速い移動平均の最大期間（デフォルト: 13）
            min_kama_fast: KAMAの速い移動平均の最小期間（デフォルト: 2）
            max_atr_period: ATR期間の最大値（デフォルト: 120）
            min_atr_period: ATR期間の最小値（デフォルト: 5）
            max_multiplier: ATR乗数の最大値（デフォルト: 3.0）
            min_multiplier: ATR乗数の最小値（デフォルト: 1.0）
        """
        super().__init__(
            f"TrendAlpha({kama_period}, {max_kama_slow}, {min_kama_slow}, "
            f"{max_kama_fast}, {min_kama_fast}, {max_atr_period}, {min_atr_period}, "
            f"{max_multiplier}, {min_multiplier})"
        )
        self.kama_period = kama_period
        self.max_kama_slow = max_kama_slow
        self.min_kama_slow = min_kama_slow
        self.max_kama_fast = max_kama_fast
        self.min_kama_fast = min_kama_fast
        self.max_atr_period = max_atr_period
        self.min_atr_period = min_atr_period
        self.max_multiplier = max_multiplier
        self.min_multiplier = min_multiplier
        self._result = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        TrendAlphaを計算する
        
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
                high = data['high'].values
                low = data['low'].values
                close = data['close'].values
            else:
                if data.ndim == 2:
                    high = data[:, 1]  # high
                    low = data[:, 2]   # low
                    close = data[:, 3] # close
                else:
                    close = data
                    high = close
                    low = close
            
            # 効率比（ER）の計算
            length = len(close)
            er = np.full(length, np.nan)
            
            for i in range(self.kama_period, length):
                change = close[i] - close[i-self.kama_period]
                volatility = np.sum(np.abs(np.diff(close[i-self.kama_period:i+1])))
                er[i] = calculate_efficiency_ratio(
                    np.array([change]),
                    np.array([volatility])
                )[0]
            
            # 動的なKAMAのfast/slow期間の計算
            fast_periods, slow_periods = calculate_dynamic_kama_periods(
                er,
                self.max_kama_slow,
                self.min_kama_slow,
                self.max_kama_fast,
                self.min_kama_fast
            )
            
            # 動的なATR期間の計算
            dynamic_period = calculate_dynamic_period(
                er,
                self.max_atr_period,
                self.min_atr_period
            )
            
            # 動的期間でのATR計算（高速化版）
            atr_values = calculate_dynamic_atr(high, low, close, dynamic_period, self.max_atr_period)
            
            # 動的な乗数の計算
            multiplier = calculate_dynamic_multiplier(er, self.max_multiplier, self.min_multiplier)
            
            # KAMAの計算（動的なfast/slow期間を使用）
            kama_values = np.zeros_like(close)
            kama_values[0] = close[0]
            
            for i in range(1, length):
                if np.isnan(er[i]):
                    kama_values[i] = kama_values[i-1]
                    continue
                
                change = close[i] - close[i-self.kama_period]
                volatility = np.sum(np.abs(np.diff(close[i-self.kama_period:i+1])))
                current_er = abs(change) / (volatility + 1e-10)
                
                fast_constant = 2.0 / (fast_periods[i] + 1)
                slow_constant = 2.0 / (slow_periods[i] + 1)
                
                smoothing_constant = (current_er * (fast_constant - slow_constant) + slow_constant) ** 2
                kama_values[i] = kama_values[i-1] + smoothing_constant * (close[i] - kama_values[i-1])
            
            # バンドの計算
            upper = kama_values + (multiplier * atr_values)
            lower = kama_values - (multiplier * atr_values)
            half_upper = kama_values + (multiplier * 0.5 * atr_values)
            half_lower = kama_values - (multiplier * 0.5 * atr_values)
            
            self._result = TrendAlphaResult(
                middle=kama_values,
                upper=upper,
                lower=lower,
                half_upper=half_upper,
                half_lower=half_lower,
                atr=atr_values,
                er=er,
                dynamic_period=dynamic_period
            )
            
            self._values = kama_values
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
    
    def get_dynamic_period(self) -> np.ndarray:
        """
        動的ATR期間の値を取得する
        
        Returns:
            np.ndarray: 動的ATR期間の値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.dynamic_period 