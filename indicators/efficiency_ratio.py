#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union
import numpy as np
import pandas as pd
from numba import jit

from .indicator import Indicator


@jit(nopython=True)
def calculate_efficiency_ratio(change: np.ndarray, volatility: np.ndarray) -> np.ndarray:
    """
    効率比（Efficiency Ratio）を計算する（高速化版）
    
    Args:
        change: 価格変化（終値の差分）の配列
        volatility: ボラティリティ（価格変化の絶対値の合計）の配列
    
    Returns:
        効率比の配列
    """
    return np.abs(change) / (volatility + 1e-10)  # ゼロ除算を防ぐ


@jit(nopython=True)
def calculate_efficiency_ratio_for_period(prices: np.ndarray, period: int) -> np.ndarray:
    """
    指定された期間の効率比（ER）を計算する（高速化版）
    
    Args:
        prices: 価格の配列（closeやhlc3などの計算済みソース）
        period: 計算期間
    
    Returns:
        効率比の配列（0-1の範囲）
        - 1に近いほど効率的な価格変動（強いトレンド）
        - 0に近いほど非効率な価格変動（レンジ・ノイズ）
    """
    length = len(prices)
    er = np.zeros(length)
    
    for i in range(period, length):
        change = prices[i] - prices[i-period]
        volatility = np.sum(np.abs(np.diff(prices[i-period:i+1])))
        er[i] = calculate_efficiency_ratio(
            np.array([change]),
            np.array([volatility])
        )[0]
    
    return er


class EfficiencyRatio(Indicator):
    """
    効率比（Efficiency Ratio）インジケーター
    
    価格変動の効率性を測定する指標
    - 1に近いほど効率的な価格変動（強いトレンド）
    - 0に近いほど非効率な価格変動（レンジ・ノイズ）
    
    使用方法：
    - 0.618以上: 効率的な価格変動（強いトレンド）
    - 0.382以下: 非効率な価格変動（レンジ・ノイズ）
    """
    
    def __init__(self, period: int = 10, src_type: str = 'close'):
        """
        コンストラクタ
        
        Args:
            period: 計算期間（デフォルト: 10）
            src_type: ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4')
                - 'close': 終値（デフォルト）
                - 'hlc3': (高値 + 安値 + 終値) / 3
                - 'hl2': (高値 + 安値) / 2
                - 'ohlc4': (始値 + 高値 + 安値 + 終値) / 4
        """
        super().__init__(f"ER({period}, {src_type})")
        self.period = period
        self.src_type = src_type.lower()
        if self.src_type not in self.SRC_TYPES:
            raise ValueError(f"無効なソースタイプです: {src_type}。有効なオプション: {', '.join(self.SRC_TYPES)}")
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        効率比を計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、選択したソースタイプに必要なカラムが必要
        
        Returns:
            効率比の配列（0-1の範囲）
        """
        try:
            # 指定されたソースタイプの価格データを取得
            prices = self.calculate_source_values(data, self.src_type)
            
            # データ長の検証
            data_length = len(prices)
            self._validate_period(self.period, data_length)
            
            # 効率比の計算（高速化版）
            self._values = calculate_efficiency_ratio_for_period(prices, self.period)
            
            return self._values
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"EfficiencyRatio計算中にエラー: {error_msg}\n{stack_trace}")
            return np.array([]) 