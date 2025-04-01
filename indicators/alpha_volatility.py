#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Optional
import numpy as np
import pandas as pd
from numba import jit

from .indicator import Indicator
from .efficiency_ratio import calculate_efficiency_ratio_for_period
from .hyper_smoother import hyper_smoother, calculate_hyper_smoother_numba


@dataclass
class AlphaVolatilityResult:
    """AlphaVolatilityの計算結果"""
    values: np.ndarray         # ボラティリティの値（%ベース）
    absolute_values: np.ndarray # ボラティリティの値（金額ベース）
    er: np.ndarray             # 効率比
    dynamic_period: np.ndarray  # 動的期間


@jit(nopython=True)
def calculate_dynamic_period(er: np.ndarray, max_period: int, min_period: int) -> np.ndarray:
    """
    効率比に基づいて動的な標準偏差計算期間を計算する（高速化版）
    
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
def calculate_rolling_returns(prices: np.ndarray) -> np.ndarray:
    """
    価格データから収益率を計算する（高速化版）
    
    Args:
        prices: 価格データの配列
    
    Returns:
        収益率の配列
    """
    length = len(prices)
    returns = np.zeros(length)
    
    # 最初の要素は収益率を計算できない
    returns[0] = 0.0
    
    # 2番目以降の要素は収益率を計算
    for i in range(1, length):
        if prices[i-1] != 0:
            returns[i] = (prices[i] / prices[i-1]) - 1.0
    
    return returns


@jit(nopython=True)
def calculate_alpha_volatility(
    prices: np.ndarray,
    er: np.ndarray,
    dynamic_period: np.ndarray,
    max_period: int,
    use_dynamic_smoothing: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    アルファボラティリティを計算する（高速化版）
    
    Args:
        prices: 価格データの配列
        er: 効率比の配列
        dynamic_period: 動的期間の配列
        max_period: 最大期間（計算開始位置用）
        use_dynamic_smoothing: 動的期間を平滑化にも使用するかどうか
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: アルファボラティリティ値と標準偏差の配列
    """
    length = len(prices)
    returns = calculate_rolling_returns(prices)
    std_dev = np.zeros(length)
    
    # 各時点で動的な期間を使用して標準偏差を計算
    for i in range(max_period, length):
        # その時点での動的な期間を取得
        curr_period = int(dynamic_period[i])
        if curr_period < 2:  # 標準偏差計算には最低2つのデータポイントが必要
            curr_period = 2
        
        # 取得する窓のサイズを定義（最低でも2要素以上確保）
        start_idx = max(0, i - curr_period + 1)
        window = returns[start_idx:i+1]
        
        # 標準偏差の計算
        if len(window) > 1:
            # 平均を計算
            window_sum = 0.0
            for j in range(len(window)):
                window_sum += window[j]
            window_mean = window_sum / len(window)
            
            # 二乗差の合計を計算
            sum_sq_diff = 0.0
            for j in range(len(window)):
                diff = window[j] - window_mean
                sum_sq_diff += diff * diff
            
            # 標準偏差を計算（不偏標準偏差）
            std_dev[i] = np.sqrt(sum_sq_diff / (len(window) - 1))
    
    return std_dev, returns


class AlphaVolatility(Indicator):
    """
    アルファボラティリティ（Alpha Volatility）インジケーター（シンプル化版）
    
    特徴:
    - 効率比（ER）に基づいて期間を動的に調整
    - ハイパースムーサーアルゴリズムで平滑化
    - トレンドの強さに応じた適応性
    - 金額ベースと%ベースの標準偏差を直接出力
    
    使用方法:
    - ボラティリティに基づいたポジションサイジング
    - 相場状況の判断
    - リスク管理の調整
    """
    
    def __init__(
        self,
        er_period: int = 21,
        max_vol_period: int = 89,
        min_vol_period: int = 13,
        smoothing_period: int = 14
    ):
        """
        コンストラクタ
        
        Args:
            er_period: 効率比の計算期間（デフォルト: 21）
            max_vol_period: ボラティリティ期間の最大値（デフォルト: 89）
            min_vol_period: ボラティリティ期間の最小値（デフォルト: 13）
            smoothing_period: ハイパースムーサー期間（デフォルト: 14）
        """
        super().__init__(
            f"AlphaVolatility({er_period}, {max_vol_period}, {min_vol_period}, {smoothing_period})"
        )
        self.er_period = er_period
        self.max_vol_period = max_vol_period
        self.min_vol_period = min_vol_period
        self.smoothing_period = smoothing_period
        self._result = None
        self._prices = None  # 計算に使用した価格データを保存
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        アルファボラティリティを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'close'カラムが必要
        
        Returns:
            ボラティリティの値（%ベース）
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
                    close = data  # 1次元配列として扱う
            
            # 計算に使用した価格データを保存
            self._prices = close
            
            # データ長の検証
            data_length = len(close)
            self._validate_period(self.er_period, data_length)
            self._validate_period(self.max_vol_period, data_length)
            
            # 効率比（ER）の計算
            er = calculate_efficiency_ratio_for_period(close, self.er_period)
            
            # 動的なボラティリティ期間の計算
            dynamic_period = calculate_dynamic_period(
                er,
                self.max_vol_period,
                self.min_vol_period
            )
            
            # アルファボラティリティ計算用の標準偏差と収益率を計算
            std_dev, returns = calculate_alpha_volatility(
                close,
                er,
                dynamic_period,
                self.max_vol_period,
                False  # 動的平滑化は使用しない
            )
            
            # ハイパースムーサーを使用して標準偏差を平滑化
            std_dev_smoothed = hyper_smoother(std_dev, self.smoothing_period)
            
            # 金額ベースの標準偏差を計算
            absolute_std_dev = np.zeros_like(std_dev_smoothed)
            for i in range(len(std_dev_smoothed)):
                if not np.isnan(std_dev_smoothed[i]):
                    absolute_std_dev[i] = std_dev_smoothed[i] * close[i]
            
            # 結果の保存
            self._result = AlphaVolatilityResult(
                values=std_dev_smoothed,  # %ベースの標準偏差
                absolute_values=absolute_std_dev,  # 金額ベースの標準偏差
                er=er,
                dynamic_period=dynamic_period
            )
            
            self._values = std_dev_smoothed  # 標準インジケーターインターフェース用
            return std_dev_smoothed
            
        except Exception as e:
            self.logger.error(f"AlphaVolatility計算中にエラー: {str(e)}")
            return np.array([])
    
    def get_percent_volatility(self) -> np.ndarray:
        """
        %ベースのボラティリティを取得する
        
        Returns:
            np.ndarray: %ベースのボラティリティ値（100倍されたパーセンテージ値）
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.values * 100  # 100倍して返す
    
    def get_absolute_volatility(self) -> np.ndarray:
        """
        金額ベースのボラティリティを取得する
        
        Returns:
            np.ndarray: 金額ベースのボラティリティ値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.absolute_values
    
    def get_volatility_multiple(self, multiplier: float = 1.0) -> np.ndarray:
        """
        %ベースのボラティリティの倍数を取得する
        
        Args:
            multiplier: ボラティリティの倍数（デフォルト: 1.0）
            
        Returns:
            np.ndarray: %ベースのボラティリティ × 倍数
        """
        vol = self.get_percent_volatility()
        return vol * multiplier
    
    def get_absolute_volatility_multiple(self, multiplier: float = 1.0) -> np.ndarray:
        """
        金額ベースのボラティリティの倍数を取得する
        
        Args:
            multiplier: ボラティリティの倍数（デフォルト: 1.0）
            
        Returns:
            np.ndarray: 金額ベースのボラティリティ × 倍数
        """
        abs_vol = self.get_absolute_volatility()
        return abs_vol * multiplier
    
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
        動的ボラティリティ期間の値を取得する
        
        Returns:
            np.ndarray: 動的ボラティリティ期間の値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.dynamic_period 