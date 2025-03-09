#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple
import numpy as np
import pandas as pd
from numba import jit

from .indicator import Indicator
from .adaptive_atr import calculate_adaptive_atr
from .kama import calculate_efficiency_ratio_for_period


@dataclass
class AlphaTrendResult:
    """AlphaTrendの計算結果"""
    middle: np.ndarray  # 中心線（KAMA）
    upper: np.ndarray   # 上限線（KAMA + dynamic_multiplier * Adaptive ATR）
    lower: np.ndarray   # 下限線（KAMA - dynamic_multiplier * Adaptive ATR）
    half_upper: np.ndarray  # 中間上限線（KAMA + dynamic_multiplier * 0.5 * Adaptive ATR）
    half_lower: np.ndarray  # 中間下限線（KAMA - dynamic_multiplier * 0.5 * Adaptive ATR）
    atr: np.ndarray     # アダプティブATRの値
    er: np.ndarray      # Efficiency Ratio
    dynamic_period: np.ndarray  # 動的ATR期間


@jit(nopython=True)
def calculate_dynamic_periods(er: np.ndarray, max_slow: int, min_slow: int, max_fast: int, min_fast: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    効率比に基づいて動的なKAMAのfast/slow期間を計算する（高速化版）
    """
    fast_periods = min_fast + (1.0 - er) * (max_fast - min_fast)
    slow_periods = min_slow + (1.0 - er) * (max_slow - min_slow)
    return np.round(fast_periods).astype(np.int32), np.round(slow_periods).astype(np.int32)


@jit(nopython=True)
def calculate_dynamic_multiplier(er: np.ndarray, max_multiplier: float, min_multiplier: float) -> np.ndarray:
    """
    効率比に基づいて動的な乗数を計算する（高速化版）
    """
    return min_multiplier + (1.0 - er) * (max_multiplier - min_multiplier)


class AlphaTrend(Indicator):
    """
    AlphaTrend インジケーター
    
    トレンドアルファをベースに、アダプティブATRを組み込んだ改良版
    KAMAを中心線として使用し、効率比（ER）に基づいて動的に調整される
    アダプティブATRの倍数とATRの期間でバンドを形成する
    
    特徴：
    - ERが高い（トレンドが強い）時：
        - バンドは狭くなる（乗数が小さくなる）
        - ATR期間は短くなる（より敏感に反応）
        - KAMAのfast/slow期間は短くなる（より敏感に反応）
    - ERが低い（トレンドが弱い）時：
        - バンドは広くなる（乗数が大きくなる）
        - ATR期間は長くなる（ノイズを軽減）
        - KAMAのfast/slow期間は長くなる（ノイズを軽減）
    
    改良点：
    - アダプティブATRの採用による適応性の向上
    - グローバルERとローカルERの組み合わせによる期間調整の改善
    - より洗練された動的パラメーター調整
    """
    
    def __init__(
        self,
        period: int = 10,
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
            period: KAMAの効率比の計算期間（デフォルト: 10）
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
            f"AlphaTrend({period}, {max_kama_slow}, {min_kama_slow}, "
            f"{max_kama_fast}, {min_kama_fast}, {max_atr_period}, {min_atr_period}, "
            f"{max_multiplier}, {min_multiplier})"
        )
        self.period = period
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
        AlphaTrendを計算する
        
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
            er = calculate_efficiency_ratio_for_period(close, self.period)
            
            # 動的なKAMAのfast/slow期間の計算
            fast_periods, slow_periods = calculate_dynamic_periods(
                er,
                self.max_kama_slow,
                self.min_kama_slow,
                self.max_kama_fast,
                self.min_kama_fast
            )
            
            # アダプティブATRの計算
            atr_values = calculate_adaptive_atr(
                high, low, close,
                self.period,
                self.min_atr_period,
                self.max_atr_period
            )
            
            # 動的な乗数の計算
            multiplier = calculate_dynamic_multiplier(er, self.max_multiplier, self.min_multiplier)
            
            # KAMAの計算（動的なfast/slow期間を使用）
            kama_values = np.zeros_like(close)
            kama_values[0] = close[0]
            
            for i in range(1, len(close)):
                if np.isnan(er[i]) or i < self.period:
                    kama_values[i] = kama_values[i-1]
                    continue
                
                try:
                    change = float(close[i] - close[i-self.period])
                    volatility = float(np.sum(np.abs(np.diff(close[i-self.period:i+1]))))
                    current_er = abs(change) / (volatility + 1e-10)
                    
                    fast_constant = 2.0 / (float(fast_periods[i]) + 1.0)
                    slow_constant = 2.0 / (float(slow_periods[i]) + 1.0)
                    
                    smoothing_constant = np.clip((current_er * (fast_constant - slow_constant) + slow_constant) ** 2, 0.0, 1.0)
                    kama_values[i] = float(kama_values[i-1]) + float(smoothing_constant * (close[i] - kama_values[i-1]))
                except (ValueError, OverflowError, ZeroDivisionError):
                    kama_values[i] = kama_values[i-1]
            
            # バンドの計算
            upper = kama_values + (multiplier * atr_values)
            lower = kama_values - (multiplier * atr_values)
            half_upper = kama_values + (multiplier * 0.5 * atr_values)
            half_lower = kama_values - (multiplier * 0.5 * atr_values)
            
            self._result = AlphaTrendResult(
                middle=kama_values,
                upper=upper,
                lower=lower,
                half_upper=half_upper,
                half_lower=half_lower,
                atr=atr_values,
                er=er,
                dynamic_period=np.full_like(er, self.period)  # 動的期間は使用しないため固定値を返す
            )
            
            self._values = kama_values
            return kama_values
            
        except Exception as e:
            print(f"エラーが発生しました: {e}")
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
        アダプティブATRの値を取得する
        
        Returns:
            np.ndarray: アダプティブATRの値
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