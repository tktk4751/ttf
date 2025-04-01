#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple
import numpy as np
import pandas as pd
from numba import jit

from .indicator import Indicator
from .alpha_ma import AlphaMA
from .efficiency_ratio import calculate_efficiency_ratio_for_period


@dataclass
class AlphaBollingerBandsResult:
    """AlphaBollingerBandsの計算結果"""
    middle: np.ndarray        # 中心線（AlphaMA）
    upper: np.ndarray         # 上限バンド
    lower: np.ndarray         # 下限バンド
    er: np.ndarray            # Efficiency Ratio
    dynamic_multiplier: np.ndarray  # 動的標準偏差乗数
    std_dev: np.ndarray       # 標準偏差


@jit(nopython=True)
def calculate_dynamic_multiplier(er: np.ndarray, max_mult: float, min_mult: float) -> np.ndarray:
    """
    効率比に基づいて動的な標準偏差乗数を計算する（高速化版）
    
    Args:
        er: 効率比の配列
        max_mult: 最大乗数
        min_mult: 最小乗数
    
    Returns:
        動的な乗数の配列
    """
    # ERが高い（トレンドが強い）ほど乗数は小さく、
    # ERが低い（トレンドが弱い）ほど乗数は大きくなる
    multipliers = max_mult - er * (max_mult - min_mult)
    return multipliers


@jit(nopython=True)
def calculate_rolling_std_dev(close: np.ndarray, period: int) -> np.ndarray:
    """
    ローリング標準偏差を計算する（高速化版）
    
    Args:
        close: 終値の配列
        period: 計算期間
    
    Returns:
        ローリング標準偏差の配列
    """
    length = len(close)
    std_dev = np.zeros(length)
    
    # 最初のperiod-1要素は標準偏差を計算しない
    for i in range(period-1):
        std_dev[i] = np.nan
    
    # 各ウィンドウごとに標準偏差を計算
    for i in range(period-1, length):
        # 現在のウィンドウを取得
        window = close[i-period+1:i+1]
        
        # 平均を計算
        window_sum = 0.0
        for j in range(period):
            window_sum += window[j]
        window_mean = window_sum / period
        
        # 二乗差の合計を計算
        sum_sq_diff = 0.0
        for j in range(period):
            diff = window[j] - window_mean
            sum_sq_diff += diff * diff
        
        # 標準偏差を計算
        std_dev[i] = np.sqrt(sum_sq_diff / period)
    
    return std_dev


@jit(nopython=True)
def calculate_alpha_bollinger_bands(
    alpha_ma: np.ndarray,
    std_dev: np.ndarray,
    dynamic_multiplier: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    アルファボリンジャーバンドを計算する（高速化版）
    
    Args:
        alpha_ma: AlphaMA値の配列
        std_dev: 標準偏差の配列
        dynamic_multiplier: 動的乗数の配列
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            (中心線, 上限バンド, 下限バンド)の配列
    """
    length = len(alpha_ma)
    middle = alpha_ma.copy()
    upper = np.zeros(length)
    lower = np.zeros(length)
    
    for i in range(length):
        if np.isnan(alpha_ma[i]) or np.isnan(std_dev[i]) or np.isnan(dynamic_multiplier[i]):
            upper[i] = np.nan
            lower[i] = np.nan
        else:
            band_width = std_dev[i] * dynamic_multiplier[i]
            upper[i] = middle[i] + band_width
            lower[i] = middle[i] - band_width
    
    return middle, upper, lower


class AlphaBollingerBands(Indicator):
    """
    アルファボリンジャーバンド（Alpha Bollinger Bands）インジケーター
    
    特徴:
    - 中心線にAlphaMA（動的適応型移動平均）を使用
    - 標準偏差の乗数が効率比（ER）に基づいて動的に調整
    
    市場状態に応じた最適な挙動:
    - トレンド強い（ER高い）:
      - 狭いバンド幅（小さい乗数）でトレンドをタイトに追従
    - トレンド弱い（ER低い）:
      - 広いバンド幅（大きい乗数）でレンジ相場の振れ幅を捉える
    """
    
    def __init__(
        self,
        er_period: int = 21,
        max_kama_period: int = 55,
        min_kama_period: int = 8,
        std_dev_period: int = 20,
        max_multiplier: float = 2.5,
        min_multiplier: float = 1.0
    ):
        """
        コンストラクタ
        
        Args:
            er_period: 効率比の計算期間（デフォルト: 21）
            max_kama_period: AlphaMAのKAMA最大期間（デフォルト: 55）
            min_kama_period: AlphaMAのKAMA最小期間（デフォルト: 8）
            std_dev_period: 標準偏差計算期間（デフォルト: 20）
            max_multiplier: 標準偏差乗数の最大値（デフォルト: 2.5）
            min_multiplier: 標準偏差乗数の最小値（デフォルト: 1.0）
        """
        super().__init__(
            f"AlphaBB({er_period}, {std_dev_period}, {max_multiplier}, {min_multiplier})"
        )
        self.er_period = er_period
        self.max_kama_period = max_kama_period
        self.min_kama_period = min_kama_period
        self.std_dev_period = std_dev_period
        self.max_multiplier = max_multiplier
        self.min_multiplier = min_multiplier
        
        # AlphaMAのインスタンス化
        self.alpha_ma = AlphaMA(
            er_period=er_period,
            max_kama_period=max_kama_period,
            min_kama_period=min_kama_period
        )
        
        self._result = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        アルファボリンジャーバンドを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'close'カラムが必要
        
        Returns:
            中心線の値（AlphaMA）
        """
        try:
            # データの検証と変換
            if isinstance(data, pd.DataFrame):
                if 'close' not in data.columns:
                    raise ValueError("DataFrameには'close'カラムが必要です")
                close = data['close'].values
            else:
                if data.ndim == 2 and data.shape[1] >= 4:
                    close = data[:, 3]  # close
                else:
                    raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")
            
            # データ長の検証
            data_length = len(close)
            self._validate_period(self.er_period, data_length)
            self._validate_period(self.std_dev_period, data_length)
            
            # AlphaMAの計算
            alpha_ma_values = self.alpha_ma.calculate(data)
            
            # 効率比（ER）の取得（AlphaMAから）
            er = self.alpha_ma.get_efficiency_ratio()
            
            # 標準偏差の計算
            std_dev = calculate_rolling_std_dev(close, self.std_dev_period)
            
            # 動的標準偏差乗数の計算
            dynamic_multiplier = calculate_dynamic_multiplier(
                er,
                self.max_multiplier,
                self.min_multiplier
            )
            
            # アルファボリンジャーバンドの計算
            middle, upper, lower = calculate_alpha_bollinger_bands(
                alpha_ma_values,
                std_dev,
                dynamic_multiplier
            )
            
            # 結果の保存
            self._result = AlphaBollingerBandsResult(
                middle=middle,
                upper=upper,
                lower=lower,
                er=er,
                dynamic_multiplier=dynamic_multiplier,
                std_dev=std_dev
            )
            
            # 中心線を値として保存
            self._values = middle
            return middle
            
        except Exception as e:
            self.logger.error(f"AlphaBollingerBands計算中にエラー: {str(e)}")
            return None
    
    def get_bands(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        ボリンジャーバンドのバンド値を取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                (中心線, 上限バンド, 下限バンド)の値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.middle, self._result.upper, self._result.lower
    
    def get_efficiency_ratio(self) -> np.ndarray:
        """
        効率比（ER）の値を取得する
        
        Returns:
            np.ndarray: 効率比の値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.er
    
    def get_dynamic_multiplier(self) -> np.ndarray:
        """
        動的標準偏差乗数の値を取得する
        
        Returns:
            np.ndarray: 動的標準偏差乗数の値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.dynamic_multiplier
    
    def get_standard_deviation(self) -> np.ndarray:
        """
        標準偏差の値を取得する
        
        Returns:
            np.ndarray: 標準偏差の値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.std_dev 