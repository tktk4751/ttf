#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Optional
import numpy as np
import pandas as pd
from numba import njit
import traceback

from .indicator import Indicator
from .price_source import PriceSource


@njit(fastmath=True, cache=True)
def calculate_zscore_numba(
    values: np.ndarray,
    period: int,
    min_periods: int = 1
) -> np.ndarray:
    """
    Z-Scoreを計算する（Numba最適化版）
    
    Args:
        values: 値の配列
        period: 計算期間
        min_periods: 最小期間
        
    Returns:
        Z-Scoreの配列
    """
    length = len(values)
    zscore = np.full(length, np.nan)
    
    for i in range(length):
        start_idx = max(0, i - period + 1)
        window_size = i - start_idx + 1
        
        if window_size >= min_periods:
            # ウィンドウデータを取得
            window_values = values[start_idx:i+1]
            
            # NaN値を除外
            valid_values = window_values[~np.isnan(window_values)]
            
            if len(valid_values) >= min_periods:
                # 平均と標準偏差を計算
                mean_val = np.mean(valid_values)
                std_val = np.std(valid_values)
                
                # Z-Scoreを計算
                if std_val > 0:
                    zscore[i] = (values[i] - mean_val) / std_val
                else:
                    zscore[i] = 0.0
    
    return zscore


class ZScore(Indicator):
    """
    Z-Score（標準化）インジケーター
    
    指定された期間内での値の標準偏差を使用して、
    現在の値がその期間の平均からどれだけ離れているかを測定します。
    
    Z-Score = (現在値 - 平均値) / 標準偏差
    
    特徴:
    - 値の相対的な位置を標準化
    - 異常値検出に有効
    - 平均回帰戦略の基礎
    """
    
    def __init__(
        self,
        period: int = 20,
        src_type: str = 'close',
        min_periods: int = 1
    ):
        """
        コンストラクタ
        
        Args:
            period: 計算期間
            src_type: 価格ソースタイプ
            min_periods: 最小期間
        """
        super().__init__(f"ZScore(period={period}, src={src_type})")
        
        self.period = period
        self.src_type = src_type.lower()
        self.min_periods = min_periods
        
        # パラメータ検証
        if self.period <= 0:
            raise ValueError("periodは0より大きい必要があります")
        if self.min_periods <= 0:
            raise ValueError("min_periodsは0より大きい必要があります")
        if self.min_periods > self.period:
            raise ValueError("min_periodsはperiod以下である必要があります")
        
        # 価格ソースの検証
        available_sources = PriceSource.get_available_sources()
        if self.src_type not in available_sources:
            raise ValueError(f"無効なソースタイプです: {self.src_type}")
        
        # 結果キャッシュ
        self._result_cache = {}
        self._max_cache_size = 20
        self._cache_keys = []
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算"""
        try:
            if isinstance(data, pd.DataFrame):
                length = len(data)
                first_val = float(data.iloc[0].get(self.src_type, data.iloc[0, -1])) if length > 0 else 0.0
                last_val = float(data.iloc[-1].get(self.src_type, data.iloc[-1, -1])) if length > 0 else 0.0
            else:
                length = len(data)
                if length > 0:
                    if data.ndim > 1:
                        first_val = float(data[0, -1])
                        last_val = float(data[-1, -1])
                    else:
                        first_val = float(data[0])
                        last_val = float(data[-1])
                else:
                    first_val = last_val = 0.0
            
            params_sig = f"{self.period}_{self.src_type}_{self.min_periods}"
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(params_sig)}"
            
        except Exception:
            return f"{id(data)}_{self.period}_{self.src_type}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Z-Scoreを計算
        
        Args:
            data: 価格データ
            
        Returns:
            Z-Scoreの配列
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            
            if data_hash in self._result_cache:
                if data_hash in self._cache_keys:
                    self._cache_keys.remove(data_hash)
                self._cache_keys.append(data_hash)
                return self._result_cache[data_hash].copy()
            
            # データ長の確認
            data_length = len(data) if hasattr(data, '__len__') else 0
            if data_length == 0:
                return np.array([])
            
            # 価格ソースの取得
            src_values = PriceSource.calculate_source(data, self.src_type)
            
            if src_values is None or len(src_values) == 0:
                self.logger.warning(f"価格ソース '{self.src_type}' の取得に失敗しました")
                return np.full(data_length, np.nan)
            
            # NumPy配列に変換
            src_values = np.asarray(src_values, dtype=np.float64)
            
            # Z-Scoreを計算
            zscore_values = calculate_zscore_numba(
                src_values,
                self.period,
                self.min_periods
            )
            
            # キャッシュ管理
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = zscore_values.copy()
            self._cache_keys.append(data_hash)
            
            self._values = zscore_values
            
            return zscore_values
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"Z-Score計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は空の配列を返す
            return np.array([])
    
    def is_extreme(self, threshold: float = 2.0) -> Optional[np.ndarray]:
        """
        極値判定（絶対値がthreshold以上）
        
        Args:
            threshold: 極値判定の閾値
            
        Returns:
            極値判定の配列（True/False）
        """
        if self._values is None:
            return None
        
        return np.abs(self._values) >= threshold
    
    def is_overbought(self, threshold: float = 2.0) -> Optional[np.ndarray]:
        """
        買われすぎ判定
        
        Args:
            threshold: 買われすぎ判定の閾値
            
        Returns:
            買われすぎ判定の配列（True/False）
        """
        if self._values is None:
            return None
        
        return self._values >= threshold
    
    def is_oversold(self, threshold: float = -2.0) -> Optional[np.ndarray]:
        """
        売られすぎ判定
        
        Args:
            threshold: 売られすぎ判定の閾値
            
        Returns:
            売られすぎ判定の配列（True/False）
        """
        if self._values is None:
            return None
        
        return self._values <= threshold
    
    def reset(self) -> None:
        """インジケーターをリセット"""
        super().reset()
        self._result_cache = {}
        self._cache_keys = []