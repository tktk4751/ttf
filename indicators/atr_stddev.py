#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from typing import Union
from numba import jit

from .indicator import Indicator


@jit(nopython=True)
def calculate_true_range(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> np.ndarray:
    """
    True Range (TR)を計算する（高速化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
    
    Returns:
        True Range (TR)の配列
    """
    length = len(high)
    tr = np.zeros(length)
    
    # 最初の要素は単純なレンジ
    tr[0] = high[0] - low[0]
    
    # 2番目以降の要素はTRを計算
    for i in range(1, length):
        tr1 = high[i] - low[i]  # 当日の高値 - 当日の安値
        tr2 = abs(high[i] - close[i-1])  # |当日の高値 - 前日の終値|
        tr3 = abs(low[i] - close[i-1])  # |当日の安値 - 前日の終値|
        tr[i] = max(tr1, tr2, tr3)
    
    return tr


@jit(nopython=True)
def calculate_atr(tr: np.ndarray, atr_period: int) -> np.ndarray:
    """
    ATR (Average True Range)を計算する（高速化版）
    Wilder's Smoothingを使用
    
    Args:
        tr: True Range (TR)の配列
        atr_period: ATR計算期間
    
    Returns:
        ATR値の配列
    """
    length = len(tr)
    atr = np.full(length, np.nan)
    
    if length < atr_period:
        return atr
    
    # 最初のATRは単純移動平均で計算
    atr[atr_period-1] = np.mean(tr[:atr_period])
    
    # 2番目以降はWilder's Smoothingで計算
    # ATR(t) = ((period-1) * ATR(t-1) + TR(t)) / period
    for i in range(atr_period, length):
        atr[i] = ((atr_period - 1) * atr[i-1] + tr[i]) / atr_period
    
    return atr


@jit(nopython=True)
def calculate_rolling_stddev(values: np.ndarray, period: int) -> np.ndarray:
    """
    ローリング標準偏差を計算する（高速化版）
    
    Args:
        values: 値の配列
        period: 標準偏差計算期間
    
    Returns:
        標準偏差の配列
    """
    length = len(values)
    stddev = np.full(length, np.nan)
    
    if length < period:
        return stddev
    
    for i in range(period - 1, length):
        # ウィンドウ内の値を取得
        window = values[i - period + 1:i + 1]
        
        # NaNが含まれていないかチェック
        valid_values = window[~np.isnan(window)]
        
        if len(valid_values) >= 2:  # 標準偏差には最低2つの値が必要
            # 標準偏差を計算（ddof=0: 母標準偏差）
            mean_val = np.mean(valid_values)
            variance = np.mean((valid_values - mean_val) ** 2)
            stddev[i] = np.sqrt(variance)
        elif len(valid_values) == 1:
            stddev[i] = 0.0  # 値が1つだけの場合は0
    
    return stddev


class ATRStdDev(Indicator):
    """
    ATR標準偏差 (ATR Standard Deviation) インジケーター
    
    ATRの標準偏差を計算することで、ボラティリティの変動性を測定する
    - 高い値：ボラティリティが大きく変動している
    - 低い値：ボラティリティが安定している
    
    Numbaによる高速化を実装：
    - True Range (TR)の計算を最適化
    - ATRの計算を最適化
    - ローリング標準偏差の計算を最適化
    """
    
    def __init__(self, atr_period: int = 14, stddev_period: int = 14):
        """
        コンストラクタ
        
        Args:
            atr_period: ATR計算期間（デフォルト: 14）
            stddev_period: 標準偏差計算期間（デフォルト: 14）
        """
        super().__init__(f"ATRStdDev(atr={atr_period},std={stddev_period})")
        self.atr_period = atr_period
        self.stddev_period = stddev_period
        
        self._atr_values = None
        self._stddev_values = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        ATRの標準偏差を計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'high'と'low'と'close'カラムが必要
        
        Returns:
            ATR標準偏差の配列
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
            min_required_length = self.atr_period + self.stddev_period - 1
            
            if data_length < min_required_length:
                self.logger.warning(f"データ長({data_length})が必要な最小長({min_required_length})より短いです")
                return np.full(data_length, np.nan)
            
            # NumPy配列に変換し、float64に統一
            high = np.asarray(high, dtype=np.float64)
            low = np.asarray(low, dtype=np.float64)
            close = np.asarray(close, dtype=np.float64)
            
            # C-contiguous配列にする
            if not high.flags['C_CONTIGUOUS']:
                high = np.ascontiguousarray(high)
            if not low.flags['C_CONTIGUOUS']:
                low = np.ascontiguousarray(low)
            if not close.flags['C_CONTIGUOUS']:
                close = np.ascontiguousarray(close)
            
            # True Range (TR)の計算（高速化版）
            tr = calculate_true_range(high, low, close)
            
            # ATRの計算（高速化版）
            atr_values = calculate_atr(tr, self.atr_period)
            
            # ATRの標準偏差を計算（高速化版）
            atr_stddev = calculate_rolling_stddev(atr_values, self.stddev_period)
            
            # 計算結果を保存
            self._atr_values = atr_values
            self._stddev_values = atr_stddev
            self._values = atr_stddev
            
            return atr_stddev
            
        except Exception as e:
            self.logger.error(f"ATRStdDev計算中にエラー: {e}")
            data_length = len(data) if hasattr(data, '__len__') else 0
            return np.full(data_length, np.nan)
    
    def get_atr_values(self) -> np.ndarray:
        """
        内部で計算されたATR値を取得する
        
        Returns:
            ATR値の配列
        """
        if self._atr_values is not None:
            return self._atr_values.copy()
        return np.array([])
    
    def get_stddev_values(self) -> np.ndarray:
        """
        ATR標準偏差値を取得する（calculateの結果と同じ）
        
        Returns:
            ATR標準偏差の配列
        """
        if self._stddev_values is not None:
            return self._stddev_values.copy()
        return np.array([])
    
    def get_current_atr(self) -> float:
        """
        現在のATR値を取得する
        
        Returns:
            現在のATR値
        """
        if self._atr_values is not None and len(self._atr_values) > 0:
            return self._atr_values[-1] if not np.isnan(self._atr_values[-1]) else np.nan
        return np.nan
    
    def get_current_stddev(self) -> float:
        """
        現在のATR標準偏差値を取得する
        
        Returns:
            現在のATR標準偏差値
        """
        if self._stddev_values is not None and len(self._stddev_values) > 0:
            return self._stddev_values[-1] if not np.isnan(self._stddev_values[-1]) else np.nan
        return np.nan
    
    def reset(self) -> None:
        """インジケータの状態をリセットする"""
        super().reset()
        self._atr_values = None
        self._stddev_values = None
        self.logger.debug(f"インジケータ '{self.name}' がリセットされました。") 