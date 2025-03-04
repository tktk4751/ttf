#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple
import numpy as np
import pandas as pd
from numba import jit

from .indicator import Indicator
from .kama import calculate_efficiency_ratio


@dataclass
class GuardianAngelResult:
    """ガーディアンエンジェルの計算結果"""
    chop: np.ndarray      # チョピネス値
    er: np.ndarray        # 効率比
    dynamic_period: np.ndarray  # 動的期間
    threshold: np.ndarray  # 動的しきい値


@jit(nopython=True)
def calculate_dynamic_period(er: np.ndarray, max_period: int, min_period: int) -> np.ndarray:
    """
    効率比に基づいて動的な期間を計算する（高速化版）
    
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
def calculate_dynamic_threshold(
    er: np.ndarray,
    max_threshold: float,
    min_threshold: float
) -> np.ndarray:
    """
    効率比に基づいて動的なしきい値を計算する（高速化版）
    
    Args:
        er: 効率比の配列
        max_threshold: しきい値の最大値
        min_threshold: しきい値の最小値
    
    Returns:
        np.ndarray: 動的なしきい値の配列
    """
    # ERが高い（トレンドが強い）ほどしきい値は低く、
    # ERが低い（トレンドが弱い）ほどしきい値は高くなる
    return min_threshold + (1.0 - er) * (max_threshold - min_threshold)


class GuardianAngel(Indicator):
    """
    ガーディアンエンジェル インジケーター
    
    チョピネスインデックスを効率比（ER）に基づいて動的に調整する
    - ERが高い（トレンドが強い）時：
        - 期間は短くなる（より敏感に反応）
        - しきい値は低くなる（トレンド判定が容易に）
    - ERが低い（トレンドが弱い）時：
        - 期間は長くなる（ノイズを軽減）
        - しきい値は高くなる（レンジ判定が容易に）
    """
    
    def __init__(
        self,
        er_period: int = 10,
        max_period: int = 30,
        min_period: int = 10,
        max_threshold: float = 61.8,
        min_threshold: float = 38.2
    ):
        """
        コンストラクタ
        
        Args:
            er_period: 効率比の計算期間（デフォルト: 10）
            max_period: チョピネス期間の最大値（デフォルト: 30）
            min_period: チョピネス期間の最小値（デフォルト: 10）
            max_threshold: しきい値の最大値（デフォルト: 61.8）
            min_threshold: しきい値の最小値（デフォルト: 38.2）
        """
        super().__init__(
            f"GuardianAngel({er_period}, {max_period}, {min_period}, "
            f"{max_threshold}, {min_threshold})"
        )
        self.er_period = er_period
        self.max_period = max_period
        self.min_period = min_period
        self.max_threshold = max_threshold
        self.min_threshold = min_threshold
        self._result = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        ガーディアンエンジェルを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'high'と'low'と'close'カラムが必要
        
        Returns:
            動的チョピネス値の配列
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
            
            for i in range(self.er_period, length):
                change = close[i] - close[i-self.er_period]
                volatility = np.sum(np.abs(np.diff(close[i-self.er_period:i+1])))
                er[i] = calculate_efficiency_ratio(
                    np.array([change]),
                    np.array([volatility])
                )[0]
            
            # 動的な期間の計算
            dynamic_period = calculate_dynamic_period(
                er,
                self.max_period,
                self.min_period
            )
            
            # 動的なしきい値の計算
            threshold = calculate_dynamic_threshold(
                er,
                self.max_threshold,
                self.min_threshold
            )
            
            # チョピネス値の計算（動的期間を使用）
            chop = np.full(length, np.nan)
            
            # True Rangeを計算
            tr = np.zeros(length)
            tr[0] = high[0] - low[0]
            for i in range(1, length):
                hl = high[i] - low[i]
                hc = abs(high[i] - close[i-1])
                lc = abs(low[i] - close[i-1])
                tr[i] = max(hl, hc, lc)
            
            # 動的期間でチョピネス値を計算
            for i in range(self.max_period, length):
                if np.isnan(dynamic_period[i]):
                    continue
                    
                period = int(dynamic_period[i])
                if period < 1:
                    continue
                
                atr_sum = np.sum(tr[i-period+1:i+1])
                price_range = max(high[i-period+1:i+1]) - min(low[i-period+1:i+1])
                
                if price_range > 0:
                    chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
            
            self._result = GuardianAngelResult(
                chop=chop,
                er=er,
                dynamic_period=dynamic_period,
                threshold=threshold
            )
            
            self._values = chop
            return chop
            
        except Exception:
            return None
    
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
        動的期間の値を取得する
        
        Returns:
            np.ndarray: 動的期間の値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.dynamic_period
    
    def get_threshold(self) -> np.ndarray:
        """
        動的しきい値を取得する
        
        Returns:
            np.ndarray: しきい値の値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.threshold
    
    def get_state(self) -> np.ndarray:
        """
        チョピネス値の状態を取得する
        
        Returns:
            np.ndarray: 状態値の配列
                1: チョピネス値がしきい値を下回る（トレンド相場）
                -1: チョピネス値がしきい値を上回る（レンジ相場）
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        
        state = np.ones_like(self._result.chop)  # デフォルトはトレンド相場
        state[self._result.chop >= self._result.threshold] = -1  # レンジ相場
        
        return state 