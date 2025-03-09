#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union
import numpy as np
import pandas as pd
from dataclasses import dataclass
from numba import jit

from .indicator import Indicator
from .normalized_chop import NormalizedChop
from .efficiency_ratio import EfficiencyRatio
from .normalized_adx import NormalizedADX
from .kama import calculate_efficiency_ratio


@dataclass
class TrendQualityResult:
    """トレンドクオリティの計算結果"""
    values: np.ndarray    # トレンドクオリティ値
    er: np.ndarray        # 効率比
    dynamic_period: np.ndarray  # 動的期間


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


class TrendQuality(Indicator):
    """
    トレンドクオリティ（Trend Quality）インジケーター
    
    NormalizedChop、EfficiencyRatio、NormalizedADXを組み合わせた
    総合的なトレンド評価指標
    
    - 0.7以上：非常に強いトレンド
    - 0.5-0.7：中程度のトレンド
    - 0.3-0.5：弱いトレンド/レンジ転換の可能性
    - 0.3以下：明確なレンジ相場
    
    重み付け：
    - EfficiencyRatio: 2
    - NormalizedChop: 2
    - NormalizedADX: 1
    
    注意：
    - ADXは固定期間13で計算されます
    - 他のインジケーターは効率比（ER）に基づいて動的に期間が調整されます
    """
    
    def __init__(
        self,
        er_period: int = 10,
        max_period: int = 30,
        min_period: int = 10
    ):
        """
        コンストラクタ
        
        Args:
            er_period: 効率比の計算期間（デフォルト: 10）
            max_period: インジケーター期間の最大値（デフォルト: 30）
            min_period: インジケーター期間の最小値（デフォルト: 10）
        """
        super().__init__(f"TQ({er_period}, {max_period}, {min_period})")
        self.er_period = er_period
        self.max_period = max_period
        self.min_period = min_period
        self._result = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        トレンドクオリティを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'high'と'low'と'close'カラムが必要
        
        Returns:
            トレンドクオリティの値（0-1の範囲）
            - 1に近いほど質の高いトレンド
            - 0に近いほどレンジ相場
        """
        try:
            # データの検証と変換
            if isinstance(data, pd.DataFrame):
                if not all(col in data.columns for col in ['high', 'low', 'close']):
                    raise ValueError("DataFrameには'high'と'low'と'close'カラムが必要です")
                close = data['close'].values
            else:
                if data.ndim == 2:
                    close = data[:, 3]  # close
                else:
                    close = data
            
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
            
            # 各インジケーターのインスタンス化と計算
            nc_values = np.full(length, np.nan)
            er_values = np.full(length, np.nan)
            adx_values = np.full(length, np.nan)
            
            # ADXは固定期間13で計算
            adx = NormalizedADX(13)
            adx_values = adx.calculate(data)
            
            # 動的期間でインジケーターを計算
            for i in range(self.max_period, length):
                if np.isnan(dynamic_period[i]):
                    continue
                    
                period = int(dynamic_period[i])
                if period < 1:
                    continue
                
                # 現在の期間でインジケーターを計算
                nc = NormalizedChop(period)
                er_ind = EfficiencyRatio(period)
                
                # 部分データを使用して計算
                data_slice = data.iloc[i-period+1:i+1] if isinstance(data, pd.DataFrame) else data[i-period+1:i+1]
                nc_values[i] = nc.calculate(data_slice)[-1]
                er_values[i] = er_ind.calculate(data_slice)[-1]
            
            # 重み付け平均の計算
            # (2 * ER + 2 * NC + ADX) / 5
            self._values = (
                2 * er_values +
                2 * nc_values +
                adx_values
            ) / 5.0
            
            self._result = TrendQualityResult(
                values=self._values,
                er=er,
                dynamic_period=dynamic_period
            )
            
            return self._values
            
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