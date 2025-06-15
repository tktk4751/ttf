#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Optional, NamedTuple
import numpy as np
import pandas as pd
from numba import jit

try:
    from .indicator import Indicator
    from .price_source import PriceSource
except ImportError:
    # Fallback for potential execution context issues
    print("Warning: Could not import from relative path. Assuming base classes are available.")
    class Indicator:
        def __init__(self, name): self.name = name; self.logger = self._get_logger()
        def reset(self): pass
        def _get_logger(self): import logging; return logging.getLogger(self.__class__.__name__)
    class PriceSource:
        @staticmethod
        def calculate_source(data, src_type):
            if isinstance(data, pd.DataFrame):
                if src_type == 'close': return data['close'].values
                elif src_type == 'hlc3': return ((data['high'] + data['low'] + data['close']) / 3).values
                else: return data['close'].values
            else:
                return data[:, 3] if data.ndim > 1 and data.shape[1] > 3 else data


class EMAResult(NamedTuple):
    """EMA計算結果"""
    values: np.ndarray


@jit(nopython=True, cache=True)
def calculate_ema_numba(prices: np.ndarray, period: int) -> np.ndarray:
    """
    EMA (Exponential Moving Average) を計算する（Numba JIT）
    
    Args:
        prices: 価格の配列
        period: 期間
    
    Returns:
        EMA値の配列
    """
    length = len(prices)
    result = np.full(length, np.nan)
    
    if period <= 0 or length == 0:
        return result
    
    # EMAの平滑化係数
    alpha = 2.0 / (period + 1.0)
    
    # 最初の値を設定（最初の有効な値を使用）
    first_valid_idx = -1
    for i in range(length):
        if not np.isnan(prices[i]):
            result[i] = prices[i]
            first_valid_idx = i
            break
    
    if first_valid_idx == -1:
        return result  # 全てNaNの場合
    
    # EMAの計算
    for i in range(first_valid_idx + 1, length):
        if not np.isnan(prices[i]):
            if not np.isnan(result[i-1]):
                result[i] = alpha * prices[i] + (1.0 - alpha) * result[i-1]
            else:
                result[i] = prices[i]
        # prices[i]がNaNの場合、result[i]はNaNのまま
    
    return result


class EMA(Indicator):
    """
    EMA (Exponential Moving Average) インジケーター
    
    シンプルな指数移動平均の実装
    """
    
    def __init__(self, 
                 period: int = 20, 
                 src_type: str = 'close'):
        """
        コンストラクタ
        
        Args:
            period: 期間
            src_type: 価格ソース
        """
        super().__init__(f"EMA(p={period},src={src_type})")
        
        self.period = period
        self.src_type = src_type
        
        self.price_source_extractor = PriceSource()
        self._result: Optional[EMAResult] = None

    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> EMAResult:
        """
        EMAを計算する
        
        Args:
            data: 価格データ
        
        Returns:
            EMAResult: EMA値を含む結果
        """
        try:
            # PriceSourceを使ってソース価格を取得
            src_prices = PriceSource.calculate_source(data, self.src_type)

            # データ長の検証
            data_length = len(src_prices)
            if data_length == 0:
                self.logger.warning("価格データが空です。空の配列を返します。")
                empty_result = EMAResult(values=np.array([]))
                self._result = empty_result
                return empty_result

            # EMAの計算
            ema_values = calculate_ema_numba(src_prices.astype(np.float64), self.period)

            result = EMAResult(values=ema_values)
            self._result = result
            return result

        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"{self.name} 計算中にエラー: {error_msg}\n{stack_trace}")
            data_len = len(data) if hasattr(data, '__len__') else 0
            self._result = None
            error_result = EMAResult(values=np.full(data_len, np.nan))
            return error_result

    def get_values(self) -> Optional[np.ndarray]:
        """EMA値のみを取得する"""
        if self._result is not None:
            return self._result.values.copy()
        return None

    def reset(self) -> None:
        """インジケータの状態をリセットする"""
        super().reset()
        self._result = None 