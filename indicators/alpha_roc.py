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
class AlphaROCResult:
    """AlphaROCの計算結果"""
    roc: np.ndarray           # ROC値
    er: np.ndarray            # 効率比
    dynamic_period: np.ndarray  # 動的ROC期間


@jit(nopython=True)
def calculate_dynamic_roc_period(
    er: np.ndarray,
    max_period: int,
    min_period: int
) -> np.ndarray:
    """
    効率比（ER）に基づいて動的なROC期間を計算する
    
    Args:
        er: 効率比の配列
        max_period: 最大期間
        min_period: 最小期間
    
    Returns:
        動的なROC期間の配列
    """
    # 配列の初期化
    dynamic_period = np.zeros_like(er)
    
    # ERに基づいて期間を調整
    # ER = 1（完全なトレンド）の場合は最小期間
    # ER = 0（完全なノイズ）の場合は最大期間
    for i in range(len(er)):
        if np.isnan(er[i]):
            dynamic_period[i] = max_period
        else:
            # 線形補間: period = max_period - er * (max_period - min_period)
            dynamic_period[i] = max_period - er[i] * (max_period - min_period)
            # 整数に丸める
            dynamic_period[i] = round(dynamic_period[i])
    
    return dynamic_period


@jit(nopython=True)
def calculate_alpha_roc(
    close: np.ndarray,
    dynamic_period: np.ndarray
) -> np.ndarray:
    """
    動的期間を使用してROC（Rate of Change）を計算する
    
    Args:
        close: 終値の配列
        dynamic_period: 動的なROC期間の配列
    
    Returns:
        ROC値の配列
    """
    # 結果を格納する配列
    roc = np.zeros_like(close)
    
    # 各時点でのROCを計算
    for i in range(len(close)):
        period = int(dynamic_period[i])
        if i < period:
            roc[i] = np.nan
        else:
            # ROC = ((現在値 - n期前の値) / n期前の値) * 100
            roc[i] = ((close[i] - close[i - period]) / close[i - period]) * 100
    
    return roc


class AlphaROC(Indicator):
    """
    AlphaROCインジケーター
    
    効率比（ER）に基づいて期間を動的に調整するROC（Rate of Change）インジケーター。
    
    特徴:
    - トレンドが強い時：短い期間で素早く反応
    - レンジ相場時：長い期間でノイズを除去
    """
    
    def __init__(
        self,
        er_period: int = 21,
        max_roc_period: int = 50,
        min_roc_period: int = 5
    ):
        """
        コンストラクタ
        
        Args:
            er_period: 効率比の計算期間（デフォルト: 21）
            max_roc_period: ROC期間の最大値（デフォルト: 50）
            min_roc_period: ROC期間の最小値（デフォルト: 5）
        """
        super().__init__(
            f"AlphaROC({er_period}, {max_roc_period}, {min_roc_period})"
        )
        self.er_period = er_period
        self.max_roc_period = max_roc_period
        self.min_roc_period = min_roc_period
        self._result = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> AlphaROCResult:
        """
        AlphaROCを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'close'カラムが必要
        
        Returns:
            AlphaROCの計算結果
        """
        try:
            # データの検証と変換
            if isinstance(data, pd.DataFrame):
                if 'close' not in data.columns:
                    raise ValueError("DataFrameには'close'カラムが必要です")
                close = data['close'].values
            else:
                if data.ndim == 2:
                    close = data[:, 3]  # close
                else:
                    close = data  # 1次元配列として扱う
            
            # データ長の検証
            data_length = len(close)
            self._validate_period(self.er_period, data_length)
            
            # 効率比（ER）の計算
            er = calculate_efficiency_ratio_for_period(close, self.er_period)
            
            # 動的なROC期間の計算
            dynamic_period = calculate_dynamic_roc_period(
                er,
                self.max_roc_period,
                self.min_roc_period
            )
            
            # AlphaROCの計算
            roc = calculate_alpha_roc(close, dynamic_period)
            
            # 結果の保存
            self._result = AlphaROCResult(
                roc=roc,
                er=er,
                dynamic_period=dynamic_period
            )
            
            self._values = roc  # 基底クラスの要件を満たすため
            
            return self._result
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"AlphaROC計算中にエラー: {error_msg}\n{stack_trace}")
            # エラー時に初期化済みの配列を返す
            if 'close' in locals():
                self._values = np.zeros_like(close)
                return AlphaROCResult(
                    roc=np.zeros_like(close),
                    er=np.zeros_like(close),
                    dynamic_period=np.zeros_like(close)
                )
            return AlphaROCResult(
                roc=np.array([]),
                er=np.array([]),
                dynamic_period=np.array([])
            )
    
    def get_efficiency_ratio(self) -> np.ndarray:
        """
        効率比（ER）の値を取得する
        
        Returns:
            np.ndarray: 効率比の値
        """
        if self._result is None:
            # 結果がない場合は空の配列を返す
            return np.array([])
        return self._result.er
    
    def get_dynamic_period(self) -> np.ndarray:
        """
        動的なROC期間の値を取得する
        
        Returns:
            np.ndarray: 動的なROC期間の値
        """
        if self._result is None:
            # 結果がない場合は空の配列を返す
            return np.array([])
        return self._result.dynamic_period 