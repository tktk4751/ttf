#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union
import numpy as np
import pandas as pd
from numba import jit

from .indicator import Indicator


@dataclass
class AdaptiveChoppinessResult:
    """アダプティブチョピネスインデックスの計算結果"""
    chop: np.ndarray      # チョピネス値
    er: np.ndarray        # 効率比
    dynamic_period: np.ndarray  # 動的期間
    threshold: np.ndarray  # 動的しきい値


@jit(nopython=True)
def calculate_adaptive_period(
    src: np.ndarray,
    period: int,
    mfast: int,
    mslow: int
) -> np.ndarray:
    """
    アダプティブな期間を計算する（高速化版）
    
    Args:
        src: ソース価格の配列（通常はHL2）
        period: 基本期間
        mfast: 最小期間
        mslow: 最大期間
    
    Returns:
        適応的な期間の配列
    """
    length = len(src)
    diff = np.zeros(length)
    signal = np.zeros(length)
    noise = np.zeros(length)
    adaptive_period = np.full(length, period)
    
    # パラメーターの計算
    mper = max(period, 1)
    mper_diff = mslow - mfast
    
    # 差分の計算
    for i in range(1, length):
        diff[i] = abs(src[i] - src[i-1])
    
    # シグナルの計算
    for i in range(mper, length):
        signal[i] = abs(src[i] - src[i-mper])
    
    # ノイズの計算
    for i in range(mper, length):
        noise_sum = 0.0
        for j in range(mper):
            noise_sum += diff[i-j]
        noise[i] = noise[i-1] + diff[i] - diff[i-mper] if i > mper else noise_sum
    
    # 適応的な期間の計算（ゼロ除算対策）
    for i in range(mper, length):
        if noise[i] > 1e-10:  # 非常に小さい値でも除算を防ぐ
            er = min(signal[i] / noise[i], 1.0)
            adaptive_period[i] = max(int(er * mper_diff + mfast), 1)  # 最小期間を保証
    
    return adaptive_period


@jit(nopython=True)
def calculate_dynamic_threshold(er: np.ndarray, max_threshold: float, min_threshold: float) -> np.ndarray:
    """
    効率比に基づいて動的なしきい値を計算する（高速化版）
    
    Args:
        er: 効率比の配列
        max_threshold: しきい値の最大値
        min_threshold: しきい値の最小値
    
    Returns:
        動的なしきい値の配列
    """
    # ERが高い（トレンドが強い）ほどしきい値は高く、
    # ERが低い（トレンドが弱い）ほどしきい値は低くなる
    return min_threshold + er * (max_threshold - min_threshold)


@jit(nopython=True)
def calculate_adaptive_choppiness(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
    mfast: int,
    mslow: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    アダプティブ・チョピネスインデックスを計算する（高速化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
        period: 基本期間
        mfast: 最小期間
        mslow: 最大期間
    
    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: 
            - 正規化されたチョピネス値（0-1の範囲）
            - 効率比
            - 動的期間
    """
    length = len(high)
    tr = np.zeros(length)
    chop = np.zeros(length)
    er = np.zeros(length)
    src = (high + low) / 2
    
    # True Rangeの計算
    tr[0] = high[0] - low[0]
    for i in range(1, length):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i-1])
        tr3 = abs(low[i] - close[i-1])
        tr[i] = max(tr1, tr2, tr3)
    
    # 適応的な期間の計算
    adaptive_period = calculate_adaptive_period(src, period, mfast, mslow)
    
    # チョピネスインデックスと効率比の計算
    for i in range(period, length):
        current_period = int(adaptive_period[i])
        if current_period < 1:
            current_period = 1
            
        if i >= current_period:
            try:
                # ATRの計算（指数移動平均を使用）
                alpha = 2.0 / (current_period + 1.0)
                atr = tr[i]
                for j in range(1, current_period):
                    atr = (1 - alpha) * atr + alpha * tr[i-j]
                atr_sum = atr * current_period
                
                # レンジの計算
                high_max = high[i]
                low_min = low[i]
                for j in range(1, current_period):
                    high_max = max(high_max, high[i-j])
                    low_min = min(low_min, low[i-j])
                range_sum = high_max - low_min
                
                # チョピネスインデックスの計算と正規化（ゼロ除算対策）
                if range_sum > 1e-10 and current_period > 1:  # 非常に小さい値でも除算を防ぐ
                    # チョピネス値を0-1の範囲に正規化し、トレンド指数として反転
                    chop_value = np.log10(atr_sum / range_sum) / np.log10(float(current_period))
                    chop_value = min(max(chop_value, 0.0), 1.0)  # 0-1の範囲に制限
                    chop[i] = 1.0 - chop_value  # トレンド指数として反転
                    
                    # 効率比の計算
                    change = abs(close[i] - close[i-current_period])
                    volatility = 0.0
                    for j in range(1, current_period + 1):
                        volatility += abs(close[i-j+1] - close[i-j])
                    if volatility > 1e-10:
                        er[i] = min(change / volatility, 1.0)
                    
            except:
                # エラーが発生した場合は前の値を維持
                chop[i] = chop[i-1] if i > 0 else 0.0
                er[i] = er[i-1] if i > 0 else 0.0
    
    return chop, er, adaptive_period


class AdaptiveChoppinessIndex(Indicator):
    """
    アダプティブ・チョピネスインデックス（正規化版）
    
    市場の状態に応じて期間を動的に調整する正規化されたトレンド指数
    - 0.618以上: 強いトレンド相場
    - 0.382以下: 強いレンジ相場
    
    特徴：
    - アダプティブATRの適応メカニズムを使用
    - 市場の状態に応じて期間を自動調整
    - より早い市場状態の変化の検出が可能
    - 0-1の範囲で正規化（1に近いほどトレンド、0に近いほどレンジ）
    - 動的なしきい値の調整（ERが高いほど高く、低いほど低く）
    """
    
    def __init__(
        self,
        period: int = 14,
        mfast: int = 2,
        mslow: int = 30,
        max_threshold: float = 0.6,
        min_threshold: float = 0.4
    ):
        """
        コンストラクタ
        
        Args:
            period: 基本期間（デフォルト: 14）
            mfast: 最小期間（デフォルト: 2）
            mslow: 最大期間（デフォルト: 30）
            max_threshold: しきい値の最大値（デフォルト: 0.6）
            min_threshold: しきい値の最小値（デフォルト: 0.4）
        """
        super().__init__(f"AdaptiveCHOP({period}, {mfast}, {mslow}, {max_threshold}, {min_threshold})")
        self.period = period
        self.mfast = mfast
        self.mslow = mslow
        self.max_threshold = max_threshold
        self.min_threshold = min_threshold
        self._result = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        アダプティブ・チョピネスインデックスを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'high'と'low'と'close'カラムが必要
        
        Returns:
            アダプティブ・チョピネスインデックス値の配列
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
                if data.ndim != 2 or data.shape[1] < 4:
                    raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")
                high = data[:, 1]  # high
                low = data[:, 2]   # low
                close = data[:, 3] # close
            
            # データ長の検証
            data_length = len(high)
            self._validate_period(self.period, data_length)
            
            # アダプティブ・チョピネスインデックスの計算（高速化版）
            chop, er, dynamic_period = calculate_adaptive_choppiness(
                high, low, close, self.period, self.mfast, self.mslow
            )
            
            # 動的なしきい値の計算
            threshold = calculate_dynamic_threshold(er, self.max_threshold, self.min_threshold)
            
            self._result = AdaptiveChoppinessResult(
                chop=chop,
                er=er,
                dynamic_period=dynamic_period,
                threshold=threshold
            )
            
            self._values = chop
            return chop
            
        except Exception as e:
            print(f"エラーが発生しました: {e}")
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
                1: チョピネス値がしきい値以上（トレンド相場）
                -1: チョピネス値がしきい値未満（レンジ相場）
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        
        state = np.ones_like(self._result.chop)  # デフォルトはトレンド相場
        state[self._result.chop < self._result.threshold] = -1  # レンジ相場
        
        return state 