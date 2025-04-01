#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple
import numpy as np
import pandas as pd
from numba import jit

from .indicator import Indicator
from .efficiency_ratio import calculate_efficiency_ratio_for_period


@dataclass
class AlphaDonchianResult:
    """AlphaDonchianの計算結果"""
    upper_band: np.ndarray    # 上側バンド（動的期間の最高値）
    lower_band: np.ndarray    # 下側バンド（動的期間の最安値）
    middle_band: np.ndarray   # 中央バンド（パーセンタイルから計算）
    smooth_upper: np.ndarray  # 75パーセンタイルの平滑化値
    smooth_lower: np.ndarray  # 25パーセンタイルの平滑化値
    er: np.ndarray           # 効率比
    dynamic_period: np.ndarray  # 動的期間


@jit(nopython=True)
def calculate_dynamic_period(er: np.ndarray, max_period: int, min_period: int) -> np.ndarray:
    """
    効率比に基づいて動的なドンチャン期間を計算する（高速化版）
    
    Args:
        er: 効率比の配列
        max_period: 最大期間（ERが低い時）
        min_period: 最小期間（ERが高い時）
    
    Returns:
        動的な期間の配列
    """
    # ERが高い（トレンドが強い）ほど期間は短く、
    # ERが低い（トレンドが弱い）ほど期間は長くなる
    periods = min_period + (1.0 - er) * (max_period - min_period)
    return np.round(periods).astype(np.int32)


@jit(nopython=True)
def calculate_dynamic_percentiles(
    high: np.ndarray,
    low: np.ndarray,
    dynamic_length: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    動的なパーセンタイルを計算する（高速化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        dynamic_length: 動的な期間配列
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: 25パーセンタイル（下限）と75パーセンタイル（上限）の配列
    """
    length = len(high)
    smooth_lower = np.zeros_like(high)
    smooth_upper = np.zeros_like(high)
    
    for i in range(length):
        current_length = min(i + 1, int(dynamic_length[i]))
        if current_length < 2:
            current_length = 2  # 最低2データポイント必要
            
        start_idx = max(0, i - current_length + 1)
        high_window = high[start_idx:i+1]
        low_window = low[start_idx:i+1]
        
        # パーティションによる高速化（完全ソートよりも速い）
        k25 = max(0, min(len(low_window) - 1, int(np.ceil(25/100.0 * len(low_window))) - 1))
        k75 = max(0, min(len(high_window) - 1, int(np.ceil(75/100.0 * len(high_window))) - 1))
        
        sorted_high = np.sort(high_window)
        sorted_low = np.sort(low_window)
        
        smooth_lower[i] = sorted_low[k25]
        smooth_upper[i] = sorted_high[k75]
    
    return smooth_lower, smooth_upper


@jit(nopython=True)
def calculate_alpha_donchian(
    high: np.ndarray,
    low: np.ndarray,
    dynamic_period: np.ndarray,
    smooth_upper: np.ndarray,
    smooth_lower: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    アルファドンチャンを計算する（高速化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        dynamic_period: 動的期間の配列
        smooth_upper: 75パーセンタイルの平滑化値
        smooth_lower: 25パーセンタイルの平滑化値
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 上バンド、下バンド、中央バンドの配列
    """
    length = len(high)
    upper_band = np.zeros_like(high)
    lower_band = np.zeros_like(high)
    middle_band = np.zeros_like(high)
    
    for i in range(length):
        current_period = min(i + 1, int(dynamic_period[i]))
        if current_period < 2:
            current_period = 2  # 最低2データポイント必要
            
        start_idx = max(0, i - current_period + 1)
        
        # 動的期間での最高値・最安値を計算
        upper_band[i] = np.max(high[start_idx:i+1])
        lower_band[i] = np.min(low[start_idx:i+1])
        
        # 中央バンドはパーセンタイルベースで計算
        middle_band[i] = (smooth_upper[i] + smooth_lower[i]) / 2.0
    
    return upper_band, lower_band, middle_band


class AlphaDonchian(Indicator):
    """
    アルファドンチャンチャネル - 効率比に基づく動的なドンチャンチャネル
    
    特徴:
    - 効率比（ER）に基づく動的期間の調整
    - 市場環境に適応して最適な期間を自動調整
    - パーセンタイルベースの平滑化
    
    注意:
    - ERが高い（トレンドが強い）場合は短い期間を使用
    - ERが低い（レンジ相場）場合は長い期間を使用
    """
    
    def __init__(
        self,
        er_period: int = 21,
        max_donchian_period: int = 55,
        min_donchian_period: int = 13
    ):
        """
        初期化
        
        Args:
            er_period: 効率比の計算期間
            max_donchian_period: ドンチャン期間の最大値（レンジ相場用）
            min_donchian_period: ドンチャン期間の最小値（トレンド相場用）
        """
        self.er_period = er_period
        self.max_donchian_period = max_donchian_period
        self.min_donchian_period = min_donchian_period
        
        # 結果の保存用
        self._result = None
        self._data_len = 0
        self._last_close_value = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        アルファドンチャンを計算する
        
        Args:
            data: ローソク足データ。pd.DataFrameまたはnp.ndarray
                 np.ndarrayの場合は、[open, high, low, close]の順
                 
        Returns:
            AlphaDonchianResultの計算結果
        """
        # 最適化されたキャッシュ機構 - データ長とデータの最終値で判断
        try:
            current_len = len(data)
            
            # データの取得を先に行う
            if isinstance(data, pd.DataFrame):
                close_value = data['close'].iloc[-1] if not data.empty else None
                high = data['high'].values
                low = data['low'].values
                close = data['close'].values
            else:
                close_value = data[-1, 3] if len(data) > 0 else None
                high = data[:, 1]  # high
                low = data[:, 2]   # low
                close = data[:, 3]  # close
                
            # キャッシュチェック - 同じデータなら再計算しない
            if (self._result is not None and current_len == self._data_len and 
                close_value == self._last_close_value):
                return self._result
            
            # 初期化
            length = len(close)
            
            # 効率比の計算
            er = calculate_efficiency_ratio_for_period(close, self.er_period)
            
            # 動的な期間の計算
            dynamic_period = calculate_dynamic_period(
                er,
                self.max_donchian_period,
                self.min_donchian_period
            )
            
            # パーセンタイルの計算
            smooth_upper, smooth_lower = calculate_dynamic_percentiles(high, low, dynamic_period)
            
            # アルファドンチャンの計算
            upper_band, lower_band, middle_band = calculate_alpha_donchian(
                high, low, dynamic_period, smooth_upper, smooth_lower
            )
            
            # 結果の保存
            self._result = AlphaDonchianResult(
                upper_band=upper_band,
                lower_band=lower_band,
                middle_band=middle_band,
                smooth_upper=smooth_upper,
                smooth_lower=smooth_lower,
                er=er,
                dynamic_period=dynamic_period
            )
            
            # キャッシュの更新
            self._data_len = current_len
            self._last_close_value = close_value
            
            return self._result
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"AlphaDonchian計算中にエラー: {str(e)}")
            return None
    
    def get_bands(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        アルファドンチャンのバンド値を取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (上限バンド, 下限バンド, 中央バンド)の値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.upper_band, self._result.lower_band, self._result.middle_band
    
    def get_percentiles(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        25パーセンタイルと75パーセンタイルの値を取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (25パーセンタイル, 75パーセンタイル)の値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.smooth_lower, self._result.smooth_upper
    
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
            np.ndarray: 動的ドンチャン期間の値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.dynamic_period
    
    def get_alpha_atr(self) -> np.ndarray:
        """
        AlphaATR値を取得する
        
        Returns:
            AlphaATR値の配列（削除されたため空の配列を返す）
        """
        return np.array([]) 