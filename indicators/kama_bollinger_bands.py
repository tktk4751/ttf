#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple

import numpy as np
import pandas as pd
from numba import jit

from .indicator import Indicator
from .kama import KaufmanAdaptiveMA
from .bollinger_bands import BollingerBandsResult
from .efficiency_ratio import calculate_efficiency_ratio_for_period


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


@jit(nopython=True)
def calculate_dynamic_std_multiplier(er: np.ndarray, max_mult: float, min_mult: float) -> np.ndarray:
    """
    効率比に基づいて動的な標準偏差乗数を計算する（高速化版）
    
    Args:
        er: 効率比の配列
        max_mult: 最大標準偏差乗数（デフォルト: 2.0）
        min_mult: 最小標準偏差乗数（デフォルト: 1.0）
    
    Returns:
        動的な標準偏差乗数の配列
    """
    # ERが高い（トレンドが強い）ほど乗数は小さく、
    # ERが低い（トレンドが弱い）ほど乗数は大きくなる
    return min_mult + (1.0 - er) * (max_mult - min_mult)


@jit(nopython=True)
def calculate_rolling_std(prices: np.ndarray, period: int) -> np.ndarray:
    """
    移動標準偏差を計算する（Numba最適化版）
    
    Args:
        prices: 価格データの配列
        period: 期間
    
    Returns:
        移動標準偏差の配列
    """
    length = len(prices)
    result = np.full(length, np.nan)
    
    for i in range(period-1, length):
        window = prices[i-period+1:i+1]
        # ddof=1と同等の挙動（サンプル標準偏差）
        mean = 0.0
        for j in range(len(window)):
            mean += window[j]
        mean /= len(window)
        
        var = 0.0
        for j in range(len(window)):
            var += (window[j] - mean) ** 2
        var /= (len(window) - 1)
        
        result[i] = np.sqrt(var)
    
    return result


@jit(nopython=True)
def calculate_dynamic_kama(close: np.ndarray, er: np.ndarray, 
                          period: int, 
                          fast_periods: np.ndarray, 
                          slow_periods: np.ndarray) -> np.ndarray:
    """
    動的なfast/slow期間を使用してKAMAを計算する（高速化版）
    
    Args:
        close: 終値の配列
        er: 効率比の配列
        period: 効率比の計算期間
        fast_periods: 速い移動平均の期間の配列
        slow_periods: 遅い移動平均の期間の配列
    
    Returns:
        KAMAの配列
    """
    length = len(close)
    kama_values = np.zeros_like(close)
    kama_values[0] = close[0]
    
    for i in range(1, length):
        if np.isnan(er[i]) or i < period:
            kama_values[i] = kama_values[i-1]
            continue
        
        try:
            change = close[i] - close[i-period]
            volatility = np.sum(np.abs(np.diff(close[max(0, i-period):i+1])))
            current_er = abs(change) / (volatility + 1e-10)
            
            fast_constant = 2.0 / (fast_periods[i] + 1.0)
            slow_constant = 2.0 / (slow_periods[i] + 1.0)
            
            # Numbaコンパイルのために手動でclipの代わりに条件分岐を使用
            sc = (current_er * (fast_constant - slow_constant) + slow_constant) ** 2
            if sc < 0.0:
                smoothing_constant = 0.0
            elif sc > 1.0:
                smoothing_constant = 1.0
            else:
                smoothing_constant = sc
                
            kama_values[i] = kama_values[i-1] + smoothing_constant * (close[i] - kama_values[i-1])
        except:
            kama_values[i] = kama_values[i-1]
    
    return kama_values


@jit(nopython=True)
def calculate_kama_bollinger_bands(
    kama_values: np.ndarray, 
    prices: np.ndarray, 
    period: int, 
    std_mult: np.ndarray
) -> tuple:
    """
    KAMAベースのボリンジャーバンドを計算する（Numba最適化版）
    
    Args:
        kama_values: KAMA値の配列
        prices: 価格データの配列
        period: 期間
        std_mult: 標準偏差の乗数の配列（動的）
    
    Returns:
        (upper, middle, lower, bandwidth, percent_b)のタプル
    """
    # ミドルバンドはKAMA
    middle = kama_values
    
    # 価格のKAMAからの偏差の標準偏差を計算
    rolling_std = calculate_rolling_std(prices, period)
    
    # アッパーバンドとロワーバンドを計算（動的な乗数を使用）
    upper = np.full_like(middle, np.nan)
    lower = np.full_like(middle, np.nan)
    for i in range(len(middle)):
        if np.isnan(middle[i]) or np.isnan(rolling_std[i]) or np.isnan(std_mult[i]):
            continue
        upper[i] = middle[i] + (rolling_std[i] * std_mult[i])
        lower[i] = middle[i] - (rolling_std[i] * std_mult[i])
    
    # バンド幅を計算 (%)
    bandwidth = np.full_like(middle, np.nan)
    mask = ~np.isnan(middle) & (middle != 0)
    bandwidth[mask] = (upper[mask] - lower[mask]) / middle[mask] * 100
    
    # %Bを計算
    percent_b = np.full_like(middle, np.nan)
    mask = ~np.isnan(upper) & ~np.isnan(lower) & (upper != lower)
    percent_b[mask] = (prices[mask] - lower[mask]) / (upper[mask] - lower[mask])
    
    return upper, middle, lower, bandwidth, percent_b


class DynamicKAMABollingerBands(Indicator):
    """
    動的KAMAベースのボリンジャーバンドインディケーター
    
    効率比（ER）に基づいて、以下のパラメータを動的に調整:
    - KAMAのfast/slow期間: 
        - ERが高い（トレンドが強い）時：期間は短くなる（より敏感に反応）
        - ERが低い（トレンドが弱い）時：期間は長くなる（ノイズを軽減）
    - 標準偏差の乗数:
        - ERが高い（トレンドが強い）時：乗数は小さくなる（バンドが狭くなる）
        - ERが低い（トレンドが弱い）時：乗数は大きくなる（バンドが広くなる）
    
    通常のボリンジャーバンドのSMAの代わりに動的KAMAを使用
    - ミドルバンド: 動的KAMAの値
    - アッパーバンド: ミドルバンド + (価格のKAMAからの偏差のN期間の標準偏差 × 動的乗数)
    - ロワーバンド: ミドルバンド - (価格のKAMAからの偏差のN期間の標準偏差 × 動的乗数)
    """
    
    def __init__(self, 
                 period: int = 20, 
                 max_std_mult: float = 2.0,
                 min_std_mult: float = 1.0,
                 max_kama_slow: int = 55,
                 min_kama_slow: int = 30,
                 max_kama_fast: int = 13,
                 min_kama_fast: int = 2):
        """
        コンストラクタ
        
        Args:
            period: 計算期間（KAMAとボリンジャーバンドの両方に使用）（デフォルト: 20）
            max_std_mult: 標準偏差乗数の最大値（デフォルト: 2.0）
            min_std_mult: 標準偏差乗数の最小値（デフォルト: 1.0）
            max_kama_slow: KAMAの遅い移動平均の最大期間（デフォルト: 55）
            min_kama_slow: KAMAの遅い移動平均の最小期間（デフォルト: 30）
            max_kama_fast: KAMAの速い移動平均の最大期間（デフォルト: 13）
            min_kama_fast: KAMAの速い移動平均の最小期間（デフォルト: 2）
        """
        super().__init__(f"DynamicKAMA-BB({period}, {max_std_mult}-{min_std_mult}, {max_kama_slow}-{min_kama_slow}, {max_kama_fast}-{min_kama_fast})")
        self.period = period
        self.max_std_mult = max_std_mult
        self.min_std_mult = min_std_mult
        self.max_kama_slow = max_kama_slow
        self.min_kama_slow = min_kama_slow
        self.max_kama_fast = max_kama_fast
        self.min_kama_fast = min_kama_fast
        self._result = None
        self._er = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> BollingerBandsResult:
        """
        動的KAMAベースのボリンジャーバンドを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'close'カラムが必要
        
        Returns:
            ボリンジャーバンドの計算結果
        """
        # データの検証と変換
        if isinstance(data, pd.DataFrame):
            if 'close' not in data.columns:
                raise ValueError("DataFrameには'close'カラムが必要です")
            prices = data['close'].values
        else:
            if data.ndim == 2:
                prices = data[:, 3]  # close
            else:
                prices = data  # 1次元配列として扱う
        
        # データ長の検証
        data_length = len(prices)
        self._validate_period(self.period, data_length)
        
        # 効率比（ER）の計算
        er = calculate_efficiency_ratio_for_period(prices, self.period)
        self._er = er  # 後で使用できるように保存
        
        # 動的なKAMAのfast/slow期間の計算
        fast_periods, slow_periods = calculate_dynamic_kama_periods(
            er,
            self.max_kama_slow,
            self.min_kama_slow,
            self.max_kama_fast,
            self.min_kama_fast
        )
        
        # 動的な標準偏差乗数の計算
        std_mult = calculate_dynamic_std_multiplier(
            er,
            self.max_std_mult,
            self.min_std_mult
        )
        
        # 動的KAMAの計算
        kama_values = calculate_dynamic_kama(
            prices,
            er,
            self.period,
            fast_periods,
            slow_periods
        )
        
        # KAMAベースのボリンジャーバンドを計算（Numba最適化版）
        upper, middle, lower, bandwidth, percent_b = calculate_kama_bollinger_bands(
            kama_values, 
            prices, 
            self.period, 
            std_mult
        )
        
        self._values = middle  # 基底クラスの要件を満たすため
        self._result = BollingerBandsResult(
            upper=upper,
            middle=middle,
            lower=lower,
            bandwidth=bandwidth,
            percent_b=percent_b
        )
        
        return self._result
    
    def get_efficiency_ratio(self) -> np.ndarray:
        """
        効率比（ER）の値を取得
        
        Returns:
            効率比の配列
        """
        if self._er is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._er


# 元のKAMABollingerBandsクラスも維持（後方互換性のため）
class KAMABollingerBands(Indicator):
    """
    KAMAベースのボリンジャーバンドインディケーター
    
    通常のボリンジャーバンドのSMAの代わりにKAMA（カウフマン適応移動平均線）を使用
    - ミドルバンド: KAMAの値
    - アッパーバンド: ミドルバンド + (価格のKAMAからの偏差のN期間の標準偏差 × K)
    - ロワーバンド: ミドルバンド - (価格のKAMAからの偏差のN期間の標準偏差 × K)
    """
    
    def __init__(self, 
                 period: int = 20, 
                 num_std: float = 2.0, 
                 kama_period: int = 10, 
                 kama_fast: int = 2, 
                 kama_slow: int = 30):
        """
        コンストラクタ
        
        Args:
            period: ボリンジャーバンドの期間（デフォルト: 20）
            num_std: 標準偏差の乗数（デフォルト: 2.0）
            kama_period: KAMAの効率比計算期間（デフォルト: 10）
            kama_fast: KAMAの速い移動平均の期間（デフォルト: 2）
            kama_slow: KAMAの遅い移動平均の期間（デフォルト: 30）
        """
        super().__init__(f"KAMA-BB({period}, {num_std}, {kama_period}, {kama_fast}, {kama_slow})")
        self.period = period
        self.num_std = num_std
        self.kama = KaufmanAdaptiveMA(kama_period, kama_fast, kama_slow)
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> BollingerBandsResult:
        """
        KAMAベースのボリンジャーバンドを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'close'カラムが必要
        
        Returns:
            ボリンジャーバンドの計算結果
        """
        # データの検証と変換
        if isinstance(data, pd.DataFrame):
            if 'close' not in data.columns:
                raise ValueError("DataFrameには'close'カラムが必要です")
            prices = data['close'].values
        else:
            if data.ndim == 2:
                prices = data[:, 3]  # close
            else:
                prices = data  # 1次元配列として扱う
        
        # データ長の検証
        data_length = len(prices)
        self._validate_period(self.period, data_length)
        
        # KAMA値を計算（ミドルバンドとして使用）
        kama_values = self.kama.calculate(data)
        
        # 固定乗数の配列を作成
        std_mult = np.full_like(prices, self.num_std)
        
        # KAMAベースのボリンジャーバンドを計算（Numba最適化版）
        upper, middle, lower, bandwidth, percent_b = calculate_kama_bollinger_bands(
            kama_values, 
            prices, 
            self.period, 
            std_mult
        )
        
        self._values = middle  # 基底クラスの要件を満たすため
        
        return BollingerBandsResult(
            upper=upper,
            middle=middle,
            lower=lower,
            bandwidth=bandwidth,
            percent_b=percent_b
        ) 